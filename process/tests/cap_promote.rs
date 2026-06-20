//! Phase-E acceptance: the claim/promote capability truth table.
//!
//! Pins the decision matrix `set_wake_reason_where` relies on, at the
//! capability primitives that enforce it (`ManagerThread::claim_parked`,
//! `ParkedMut::{promote_wake,resume_published,write_rets}`):
//!
//! | state     | published rets | cap outcome                              |
//! |-----------|----------------|-----------------------------------------|
//! | Suspended | none           | `promote_wake` → Ready, no frame write  |
//! | Suspended | yes            | `resume_published` → Ready + rets        |
//! | Blocking  | none           | `resume_published` → **None**, stays parked |
//! | Blocking  | yes            | `resume_published` → Ready + rets        |
//! | Running   | —              | `claim_parked` → **None** (bug 2)        |
//! | Assigned  | —              | `claim_parked` → **None** (bug 2)        |
//! | Ready     | —              | `claim_parked` → **None**                |
//! | Exited    | —              | `claim_parked` → **None**                |
//!
//! The `(Blocking, none) → None` row is the fs_read-6001 regression: a
//! parked thread with nothing published must NOT be flipped Ready with a
//! stale frame — `resume_published` returns no `Runnable`, so by bug-4
//! construction it can't be enqueued.

mod common;

use std::sync::atomic::Ordering;

use common::{leak_thread, with_guard};
use process::{ManagerThread, RunningThread, ThreadState};

fn state_of(ptr: *mut process::Thread) -> usize {
    unsafe { (*ptr).state_load(Ordering::Acquire) }
}
fn last_wake_reason(ptr: *mut process::Thread) -> u64 {
    unsafe { (*ptr).last_wake_reason.load(Ordering::Acquire) }
}

// ── bug-2 gate: claim_parked only succeeds for parked states ──────────

#[test]
fn claim_parked_gating() {
    for (st, claimable) in [
        (ThreadState::Suspended, true),
        (ThreadState::Blocking, true),
        (ThreadState::Running, false),
        (ThreadState::Assigned, false),
        (ThreadState::Ready, false),
        (ThreadState::Exited, false),
    ] {
        let ptr = leak_thread(st);
        with_guard(|g| {
            let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
            assert_eq!(
                mgr.claim_parked().is_some(),
                claimable,
                "claim_parked for {st:?} should be {claimable}",
            );
        });
    }
}

// ── (Suspended, none) → promote_wake: Ready, no rets, override consumed ─

#[test]
fn suspended_no_rets_promotes() {
    const REASON: u64 = 0b10;
    let ptr = leak_thread(ThreadState::Suspended);
    with_guard(|g| {
        let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
        mgr.note_wake(REASON);
        let parked = mgr.claim_parked().expect("Suspended is claimable");
        let runnable = parked.promote_wake();
        assert_eq!(runnable.tid(), 1);
    });
    assert_eq!(state_of(ptr), ThreadState::Ready as usize, "→ Ready");
    assert_eq!(
        last_wake_reason(ptr),
        REASON,
        "override drained into reason"
    );
}

// ── (Suspended, yes) / (Blocking, yes) → resume_published: Ready + rets ─

#[test]
fn published_rets_resume_to_ready() {
    for st in [ThreadState::Suspended, ThreadState::Blocking] {
        let ptr = leak_thread(st);
        let runnable_minted = with_guard(|g| {
            let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
            mgr.publish_results(&[7, 9]);
            let parked = mgr.claim_parked().expect("parked is claimable");
            let r = parked.resume_published();
            assert!(r.is_some(), "{st:?} with published rets must resume");
            r.is_some()
        });
        assert!(runnable_minted);
        assert_eq!(state_of(ptr), ThreadState::Ready as usize, "{st:?} → Ready");
        // rets marshaled into a0/a1 — inspected via the manager cap (the
        // thread is now a quiescent `Ready` registry thread, so the gated
        // frame read returns `Some`).
        with_guard(|g| {
            let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
            assert_eq!(mgr.frame_reg(10), Some(7));
            assert_eq!(mgr.frame_reg(11), Some(9));
        });
    }
}

// ── (Blocking, none) → resume_published == None: the 6001 regression ──

#[test]
fn blocking_no_rets_does_not_resume() {
    let ptr = leak_thread(ThreadState::Blocking);
    with_guard(|g| {
        let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
        // Claiming is allowed, but with nothing published the take-CAS
        // finds no results → no Runnable is minted (bug-4), and the
        // thread is left parked (NOT flipped to a dispatchable Ready with
        // a stale frame).
        let parked = mgr.claim_parked().expect("Blocking is claimable");
        assert!(
            parked.resume_published().is_none(),
            "no published rets ⇒ no Runnable",
        );
    });
    assert_eq!(
        state_of(ptr),
        ThreadState::Blocking as usize,
        "thread stays parked — no spurious Ready",
    );
}

// ── write_rets: the direct-resume path marshals + Ready ───────────────

#[test]
fn write_rets_marshals_and_readies() {
    let ptr = leak_thread(ThreadState::Blocking);
    with_guard(|g| {
        let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
        let parked = mgr.claim_parked().expect("claimable");
        let _runnable = parked.write_rets(&[42]);
    });
    assert_eq!(state_of(ptr), ThreadState::Ready as usize);
    with_guard(|g| {
        let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
        assert_eq!(mgr.frame_reg(10), Some(42));
    });
}

// ── into_runnable self-check (#7): mints a token only for a Ready thread ─
//
// `into_runnable` is the own-hart cooperative-yield mint. Unlike
// `claim_ready` it historically trusted the caller to have already set
// Ready. A future second caller minting a `Runnable` for a parked thread
// would smuggle a non-dispatchable thread onto the ready queue (bug-4). The
// cap must self-verify the from-state is Ready and panic otherwise.

#[test]
fn into_runnable_on_ready_ok() {
    let ptr = leak_thread(ThreadState::Ready);
    // SAFETY: single-threaded test; sole accessor of the leaked thread.
    let rt = unsafe { RunningThread::from_ptr(ptr) };
    let r = rt.into_runnable();
    assert_eq!(r.tid(), 1);
}

#[test]
#[should_panic(expected = "into_runnable")]
fn into_runnable_on_non_ready_panics() {
    let ptr = leak_thread(ThreadState::Blocking);
    // SAFETY: single-threaded test; sole accessor of the leaked thread.
    let rt = unsafe { RunningThread::from_ptr(ptr) };
    // Blocking is not Ready → minting a dispatch token here is a bug-4
    // violation and must panic.
    let _ = rt.into_runnable();
}
