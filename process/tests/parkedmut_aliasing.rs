//! Phase-E acceptance (assertion 2 — the `ParkedMut` half): the manager
//! resuming a parked thread and the parker's own lock-free post-park
//! re-check coexist over one `*mut Thread` with **no aliasing UB** under
//! miri's Stacked- and Tree-Borrows models.
//!
//! This is the `ParkedMut` counterpart to [`cap_aliasing`]. The manager's
//! resume verbs (`ParkedMut::{resume_published, write_rets, promote_wake}`)
//! must field-project off the raw pointer — **never** materialize
//! `&mut *self.ptr` — because the parker's own re-check
//! (`RunningThread::try_claim_own_pending`) concurrently `compare_exchange`s
//! `pending_state` on the SAME allocation with **no scheduler lock held**.
//! A whole-struct `&mut Thread` retag (Unique) coexisting with that atomic
//! access is the Phase-E UB. The `take_pending_results` CAS is the
//! at-most-once gate that decides *which* side resumes — but it does not
//! prevent the *memory* race: a `&mut` retag asserts whole-allocation
//! exclusivity at formation, before the CAS runs.
//!
//! Mirrors production: `set_wake_reason_where` (manager, under `SchedGuard`)
//! runs `claim_parked → resume_published` on a Blocking+SIGNALED thread
//! while `exit_thread_with_state`'s post-park re-check
//! (`try_wake_pending_inline → try_claim_own_pending`) runs on the parker's
//! own hart with no guard.
//!
//! Run under: `./test miri` (Stacked Borrows), `./test miri-tree` (Tree
//! Borrows), and `./test miri-hammer` (many-seeds + dense preemption — this
//! is the mode that reliably surfaces the interleaving).

mod common;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::thread;

use common::{leak_thread, with_guard};
use process::{ManagerThread, RunningThread, ThreadState};

#[cfg(miri)]
const ITERS: usize = 20;
#[cfg(not(miri))]
const ITERS: usize = 5000;

/// Raw `*mut Thread` is not `Send`; ship it across threads in a wrapper
/// whose `Send` we vouch for (the allocation is leaked and outlives both
/// threads, which are joined before the test returns).
#[derive(Clone, Copy)]
struct ThreadPtr(*mut process::Thread);
unsafe impl Send for ThreadPtr {}

#[test]
fn parkedmut_resume_and_parker_recheck_coexist() {
    let ptr = ThreadPtr(leak_thread(ThreadState::Blocking));
    // Two barriers bracket each round so it models kernel behavior exactly: a
    // thread parks ONCE per blocking syscall and is resumed ONCE — whichever
    // of {manager drain, parker's own re-check} wins the take-CAS. The
    // at-most-once CAS lets only one side marshal the (non-atomic) frame for
    // a given publish; the thread then "runs" (the round boundary) before it
    // could re-park. Re-publishing without that boundary would manufacture
    // two winners marshaling concurrently — a data race that cannot occur in
    // the kernel (the thread must run between parks). What we DO want
    // concurrent is the resume itself: manager `resume_published` (which must
    // not form `&mut Thread`) vs the parker's lock-free `take_pending_results`
    // CAS on the same allocation.
    let round_start = Arc::new(Barrier::new(2));
    let round_end = Arc::new(Barrier::new(2));

    // "Hart" M: the manager. Stages one park + publish (alone), then — once
    // the round opens — claims + resumes under the scheduler guard, racing
    // the parker for the single published result.
    let manager = {
        let rs = round_start.clone();
        let re = round_end.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p; // capture the whole (Send) ThreadPtr
            for _ in 0..ITERS {
                // Stage the round single-threaded (parker is parked at the
                // round barrier): re-park Blocking + publish one result.
                with_guard(|g| {
                    // SAFETY: feature-gated fixture setter.
                    unsafe { (*p.0).transition_to_unchecked(ThreadState::Blocking) };
                    // SAFETY: live registry thread, guard held.
                    let mgr = unsafe { ManagerThread::from_raw(p.0, g) };
                    mgr.publish_results(&[1, 2]);
                });
                rs.wait(); // open the round — both race now
                with_guard(|g| {
                    let mgr = unsafe { ManagerThread::from_raw(p.0, g) };
                    if let Some(parked) = mgr.claim_parked() {
                        let _ = parked.resume_published();
                    }
                });
                re.wait(); // round closes; thread has "run" — safe to re-park
            }
        })
    };

    // "Hart" P: the parker's own-hart post-park re-check — lock-free, racing
    // the manager's resume for the same publish. `try_claim_own_pending`
    // `compare_exchange`s `pending_state` and (on a win) field-projects the
    // frame + flips Ready. This is the concurrent foreign accessor that a
    // manager-side `&mut Thread` retag would race.
    let parker = {
        let rs = round_start.clone();
        let re = round_end.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p;
            for _ in 0..ITERS {
                rs.wait(); // wait for the round to be staged
                // SAFETY: models the parker on its own hart; sole minter of a
                // RunningThread for this ptr on this "hart".
                let mut rt = unsafe { RunningThread::from_ptr(p.0) };
                let _ = rt.try_claim_own_pending();
                re.wait();
            }
        })
    };

    manager.join().unwrap();
    parker.join().unwrap();

    // Exactly one side resumed each round → Ready (or Blocking if the round's
    // CAS somehow found nothing); either is a valid, untorn state.
    let st = unsafe { (*ptr.0).state_load(Ordering::Acquire) };
    assert!(
        st == ThreadState::Blocking as usize || st == ThreadState::Ready as usize,
        "unexpected end state {st}",
    );
}
