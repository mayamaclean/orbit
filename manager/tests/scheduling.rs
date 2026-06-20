//! Phase-E coverage for the extracted `Manager` scheduling state: the
//! per-hart ready-inbox → ready-queue → dispatch round-trip, and the
//! registry accessors that the Phase-D extraction added. Runs under the
//! miri sweep (the `manager` crate is the Phase-E target).

mod common;

use common::leak_ready_thread;
use manager::{Manager, READY_INBOXES, ReadyNotice};
use process::ThreadHandle;

/// A thread published onto a per-hart `READY_INBOXES` slot (the lock-free
/// path non-manager harts use) is folded into the ready queue by
/// `drain_ready_inboxes` and then handed out by `get_runnable_thread`.
#[test]
fn ready_inbox_feeds_dispatch() {
    let ptr = leak_ready_thread(1);
    // SAFETY: `ptr` is a freshly-leaked Ready thread with a valid frame —
    // the fresh-creation `Runnable` contract.
    let runnable = unsafe { process::Runnable::from_raw(ptr) };
    READY_INBOXES[0]
        .push(ReadyNotice::from_runnable(runnable))
        .expect("inbox has room");

    let mut mgr = Manager::new();
    mgr.drain_ready_inboxes();

    assert_eq!(
        mgr.get_runnable_thread(u64::MAX),
        Some(ptr),
        "drained inbox entry must be dispatchable",
    );
    assert_eq!(
        mgr.get_runnable_thread(u64::MAX),
        None,
        "queue is now empty",
    );
}

/// `register` / `thread` / `tid_in_use` / `unregister` — the accessors the
/// STAY-in-kmain paths (creation, reap, resume, snapshot) reach the
/// relocated registry through.
#[test]
fn registry_accessors_round_trip() {
    let mut mgr = Manager::new();
    let ptr = leak_ready_thread(7);

    assert!(!mgr.tid_in_use(7));
    mgr.register(7, unsafe { ThreadHandle::from_raw(ptr) });

    assert!(mgr.tid_in_use(7));
    assert_eq!(mgr.thread(7).map(|h| h.peek().tid()), Some(7));

    let handle = mgr.unregister(7).expect("was registered");
    assert!(!mgr.tid_in_use(7));
    assert!(mgr.thread(7).is_none());
    // The thread allocation is leaked + miri-registered; don't drop the
    // handle (which would free it and dangle the registered root).
    core::mem::forget(handle);
}
