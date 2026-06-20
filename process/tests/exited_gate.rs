//! `ExitedThread` + `ManagerThread::frame_reg` gating acceptance.
//!
//! The sealed reaper reads (`fault_info`, frame regs) must be reachable
//! only when the thread is provably **not running**, so they can't race the
//! owning hart's frame writes:
//!   * [`ManagerThread::claim_exited`] mints an [`process::ExitedThread`]
//!     only for an `Exited` thread (construction-enforced for the reaper);
//!   * [`ManagerThread::frame_reg`] returns `None` for a `Running`/
//!     `Assigned` thread (runtime-gated for general manager inspection).

mod common;

use common::{leak_thread, with_guard};
use process::{ManagerThread, ThreadState};

#[test]
fn claim_exited_refuses_live_states() {
    for s in [
        ThreadState::Running,
        ThreadState::Assigned,
        ThreadState::Ready,
        ThreadState::Blocking,
        ThreadState::Suspended,
    ] {
        let ptr = leak_thread(s);
        with_guard(|g| {
            // SAFETY: live registry thread, guard held.
            let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
            assert!(
                mgr.claim_exited().is_none(),
                "{s:?} must not be claim_exited-able",
            );
        });
    }
}

#[test]
fn claim_exited_allows_exited_and_reads_sealed_fields() {
    let ptr = leak_thread(ThreadState::Exited);
    with_guard(|g| {
        // SAFETY: live registry thread, guard held.
        let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
        let ex = mgr.claim_exited().expect("Exited is claimable");
        // A clean-exit fixture records no fault.
        assert!(ex.fault_info().is_none());
        // Sealed frame readable through the exit token (zeroed fixture frame).
        assert_eq!(ex.frame_reg(11), 0);
    });
}

#[test]
fn manager_frame_reg_gated_on_not_running() {
    // Running / Assigned ⇒ None (reading would race the owning hart).
    for s in [ThreadState::Running, ThreadState::Assigned] {
        let ptr = leak_thread(s);
        with_guard(|g| {
            // SAFETY: live registry thread, guard held.
            let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
            assert!(mgr.frame_reg(10).is_none(), "{s:?} frame must be gated off");
        });
    }
    // Quiescent states ⇒ Some (safe to inspect).
    for s in [
        ThreadState::Ready,
        ThreadState::Blocking,
        ThreadState::Suspended,
        ThreadState::Exited,
    ] {
        let ptr = leak_thread(s);
        with_guard(|g| {
            // SAFETY: live registry thread, guard held.
            let mgr = unsafe { ManagerThread::from_raw(ptr, g) };
            assert!(mgr.frame_reg(10).is_some(), "{s:?} frame must be readable");
        });
    }
}
