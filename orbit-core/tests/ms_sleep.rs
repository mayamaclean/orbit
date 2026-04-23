mod common;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_core::{SyscallOutcome, syscall};

use common::{FakeHw, make_thread};

#[test]
fn sleeps_for_requested_ms() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw { now_ticks: 1_000_000, ticks_per_ms: 10_000, ..Default::default() };

    let outcome = syscall::ms_sleep(&mut t, 50, &hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Suspended, ret: Some(0) }
    );
    assert_eq!(t.wake_time, 1_000_000 + 50 * 10_000);
}

#[test]
fn zero_ms_still_yields() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw { now_ticks: 42, ..Default::default() };

    let outcome = syscall::ms_sleep(&mut t, 0, &hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Suspended, ret: Some(0) }
    );
    assert_eq!(t.wake_time, 42);
}

#[test]
fn rejects_ms_at_cap() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.wake_time = 0xDEAD; // must stay untouched on reject path
    let hw = FakeHw::default();

    let outcome = syscall::ms_sleep(&mut t, syscall::MAX_SLEEP_MS, &hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: -2 });
    assert_eq!(t.wake_time, 0xDEAD);
}

#[test]
fn rejects_ms_above_cap() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw::default();

    let outcome = syscall::ms_sleep(&mut t, syscall::MAX_SLEEP_MS + 1, &hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: -2 });
}

#[test]
fn wake_time_wraps_on_overflow() {
    // Matches the production `wrapping_add` / `wrapping_mul` semantics —
    // a near-max `now` plus a small sleep should wrap cleanly.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw { now_ticks: u64::MAX - 1000, ticks_per_ms: 10_000, ..Default::default() };

    let outcome = syscall::ms_sleep(&mut t, 1, &hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Suspended, ret: Some(0) }
    );
    let expected = (u64::MAX - 1000).wrapping_add(10_000) as usize;
    assert_eq!(t.wake_time, expected);
}
