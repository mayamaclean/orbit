//! Phase-E acceptance (assertion 2): a `RunningThread` (the hart executing
//! a thread) and a `ManagerThread` (the manager peeking that same thread
//! from another hart) coexist over one `*mut Thread` with **no aliasing
//! UB** under miri's Stacked- and Tree-Borrows models.
//!
//! This is the central soundness claim of the capability design: the caps
//! hold *raw* pointers and only ever materialize short-lived references for
//! a single field access, so a hart writing a thread's frame and the
//! manager OR-ing a wake reason into the same thread never form a
//! conflicting `&mut`/`&`. It mirrors production exactly:
//! `set_wake_reason_where` walks every thread — including ones currently
//! Running on another hart — calling `view()` (atomic reads) and
//! `note_wake()` (atomic fetch_or), while the running hart commits its
//! frame. They touch disjoint / atomic fields only; `claim_parked` refuses
//! the Running thread (bug 2), so the manager never reaches its frame.
//!
//! Run under: `./test miri` (Stacked Borrows), `./test miri-tree` (Tree
//! Borrows), and `./test miri-hammer` (many-seeds + dense preemption).

mod common;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::thread;

use common::{leak_thread, with_guard};
use process::{ManagerThread, RunningThread, ThreadState};

// miri is an interpreter — keep iteration counts tiny so the many-seeds
// sweep finishes in seconds; native runs hammer harder.
#[cfg(miri)]
const ITERS: usize = 20;
#[cfg(not(miri))]
const ITERS: usize = 5000;

const REASON: u64 = 0b100;

/// Raw `*mut Thread` is not `Send`; ship it across threads in a wrapper
/// whose `Send` we vouch for (the allocation outlives both threads — it's
/// leaked, joined before the test returns).
#[derive(Clone, Copy)]
struct ThreadPtr(*mut process::Thread);
unsafe impl Send for ThreadPtr {}

// Regression guard for the Phase-E finding: the hart-side frame writers used
// to form `&mut Thread`, whose whole-struct retag raced the manager's
// concurrent atomic reads of `state`/`wake_override`. `RunningThread`'s
// writers now project to the `frame` field / use narrow atomic access, so
// the caps coexist cleanly under both borrow models.
#[test]
fn running_and_manager_caps_coexist() {
    let ptr = ThreadPtr(leak_thread(ThreadState::Running));
    let barrier = Arc::new(Barrier::new(2));

    // "Hart" H: owns the thread's execution. Commits frame registers (a
    // separate TrapFrame allocation) and the resume pc (atomic) in a loop.
    let hart = {
        let b = barrier.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p; // capture the whole (Send) ThreadPtr, not p.0
            // SAFETY: this thread is the exclusive owner of `p` as its
            // current-running thread — the production trap-entry seam.
            let mut rt = unsafe { RunningThread::from_ptr(p.0) };
            b.wait();
            for i in 0..ITERS {
                rt.set_frame_reg(10, i);
                rt.set_pc(0x1000 + i);
            }
        })
    };

    // "Hart" M: the manager. Peeks the same thread (atomic state +
    // wake_override reads) and OR's a wake reason — never touching its
    // frame, exactly as `set_wake_reason_where` does for a thread that is
    // live on another hart.
    let manager = {
        let b = barrier.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p; // capture the whole (Send) ThreadPtr, not p.0
            with_guard(|g| {
                // SAFETY: live registry thread, guard held — the manager seam.
                let mgr = unsafe { ManagerThread::from_raw(p.0, g) };
                b.wait();
                for _ in 0..ITERS {
                    let v = mgr.view();
                    let _ = v.state();
                    let _ = v.has_pending_wake();
                    mgr.note_wake(REASON);
                }
            });
        })
    };

    hart.join().unwrap();
    manager.join().unwrap();

    // Post-conditions: the wake reason landed, the manager never moved the
    // thread off Running, and a claim is correctly refused (bug 2).
    unsafe {
        assert_eq!(
            (*ptr.0).state_load(Ordering::Acquire),
            ThreadState::Running as usize,
            "manager must not transition a Running thread",
        );
        assert_ne!(
            (*ptr.0).wake_override.load(Ordering::Acquire) & REASON,
            0,
            "note_wake reason must be observable",
        );
    }
    with_guard(|g| {
        let mgr = unsafe { ManagerThread::from_raw(ptr.0, g) };
        assert!(
            mgr.claim_parked().is_none(),
            "Running thread must not be claimable (bug 2)",
        );
    });
}
