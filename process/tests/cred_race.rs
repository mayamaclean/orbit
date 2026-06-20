//! Cred / wake_time propagation race acceptance.
//!
//! The manager field-writes a thread's credential snapshot (setuid/setgid/
//! pledge propagation) — and the own-hart parker stamps its `wake_time` —
//! while that same thread is reachable lock-free from another "hart":
//!   * the perm-gate / `getuid` fast path reads `permissions`/`uid`/… , and
//!   * the stuck-thread watchdog reads `wake_time`,
//! both through a field-projecting [`ThreadView`] (never a whole-struct
//! `&Thread`).
//!
//! With those fields atomic and the view field-projecting, this is
//! race-free under both borrow models *and* miri's data-race detector.
//! Before the atomic conversion it was a data race on `uid`/`permissions`/
//! `wake_time` and a Stacked/Tree-Borrows violation (a shared whole-struct
//! `&Thread` froze those fields while a foreign hart wrote them). This is
//! the acceptance test for that fix.
//!
//! Run under: `./test miri` (Stacked Borrows), `./test miri-tree` (Tree
//! Borrows), `./test miri-hammer` (many-seeds + dense preemption).

mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use common::{leak_thread, with_guard};
use orbit_abi::perms::Permissions;
use process::{ManagerThread, RunningThread, ThreadState, ThreadView};

// miri is an interpreter — keep iteration counts tiny so the many-seeds
// sweep finishes in seconds; native runs hammer harder.
#[cfg(miri)]
const ITERS: usize = 20;
#[cfg(not(miri))]
const ITERS: usize = 5000;

/// Raw `*mut Thread` is not `Send`; ship it across threads in a wrapper
/// whose `Send` we vouch for (the allocation is leaked, joined before the
/// test returns).
#[derive(Clone, Copy)]
struct ThreadPtr(*mut process::Thread);
unsafe impl Send for ThreadPtr {}

/// Two whole identities the manager alternates between. Each field is its
/// own atomic, so a reader always observes a legally-stored value per
/// field — the assertion below rejects any out-of-set (torn/garbage) read.
const ID_A: u32 = 1000;
const ID_B: u32 = 2000;

#[test]
fn cred_propagation_and_lockfree_read_coexist() {
    let ptr = ThreadPtr(leak_thread(ThreadState::Running));
    let barrier = Arc::new(Barrier::new(2));

    // "Hart" M: the manager, propagating setuid/setgid/pledge to a sibling
    // that is Running on another hart — under the guard, via the cap's
    // field-projected atomic `Release` stores.
    let manager = {
        let b = barrier.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p;
            with_guard(|g| {
                // SAFETY: live registry thread, guard held — the manager seam.
                let mgr = unsafe { ManagerThread::from_raw(p.0, g) };
                b.wait();
                for i in 0..ITERS {
                    let id = if i % 2 == 0 { ID_A } else { ID_B };
                    mgr.set_uid_triplet(id, id, id);
                    mgr.set_gid_triplet(id, id, id);
                    let mut perms = Permissions::ZERO;
                    perms.perms = if i % 2 == 0 { 0xFF } else { 0x0F };
                    perms.role = id;
                    mgr.set_permissions(perms);
                }
            });
        })
    };

    // "Hart" H: the Running thread reading its own cred snapshot lock-free,
    // exactly as `perm_gate_check` and the `getuid` family do — `Acquire`
    // atomic loads through a field-projecting view (no `&Thread` freeze).
    let reader = {
        let b = barrier.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p;
            // SAFETY: the view never forms a whole-struct `&Thread`; each
            // accessor is an atomic load, sound against the manager writes.
            let v = unsafe { ThreadView::from_ptr(p.0 as *const _) };
            b.wait();
            for _ in 0..ITERS {
                // The exact reads the gate / getuid path perform.
                let _ = v.uid();
                let _ = v.euid();
                let _ = v.gid();
                let _ = v.egid();
                let snap = v.permissions_snapshot();
                let _ = snap.allows(1); // exercise the gate's read path
                // Each field is atomic, so every read is a value some store
                // actually wrote — never garbage / a torn half-update.
                assert!(
                    matches!(v.uid(), 0 | ID_A | ID_B),
                    "uid read out of set",
                );
                assert!(
                    matches!(snap.role, 0 | ID_A | ID_B),
                    "role read out of set: {}",
                    snap.role,
                );
            }
        })
    };

    manager.join().unwrap();
    reader.join().unwrap();

    // Last writer wins with a whole identity.
    let v = unsafe { ThreadView::from_ptr(ptr.0 as *const _) };
    assert!(matches!(v.uid(), ID_A | ID_B));
}

/// The `wake_time` watchdog race: the own-hart parker stamps `wake_time`
/// (now an atomic, via [`RunningThread::set_wake_time`]) while the manager's
/// stuck-thread watchdog reads it cross-hart via [`ThreadView::wake_time`].
/// The watchdog's census dump reads *every* thread's `wake_time`, including
/// ones Running and re-parking on another hart — so the field must be
/// atomic for that read to be defined. This is the acceptance for the
/// `wake_time` atomic conversion.
#[test]
fn wake_time_stamp_and_watchdog_read_coexist() {
    let ptr = ThreadPtr(leak_thread(ThreadState::Running));
    let barrier = Arc::new(Barrier::new(2));

    let parker = {
        let b = barrier.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p;
            // SAFETY: own-hart exclusive over its current thread.
            let mut rt = unsafe { RunningThread::from_ptr(p.0) };
            b.wait();
            for i in 0..ITERS {
                rt.set_wake_time(i);
            }
        })
    };

    let watchdog = {
        let b = barrier.clone();
        let p = ptr;
        thread::spawn(move || {
            let p = p;
            // SAFETY: read-only view; atomic load, no `&Thread` freeze.
            let v = unsafe { ThreadView::from_ptr(p.0 as *const _) };
            b.wait();
            for _ in 0..ITERS {
                let _ = v.wake_time();
            }
        })
    };

    parker.join().unwrap();
    watchdog.join().unwrap();
}
