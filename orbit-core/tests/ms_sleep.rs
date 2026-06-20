mod common;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{EINVAL, Errno};
use orbit_core::{ShimAction, SyscallOutcome, apply_syscall_outcome, syscall};

use common::{FakeHw, make_frame, make_thread};

#[test]
fn sleeps_for_requested_ms() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw {
        now_ticks: 1_000_000,
        ticks_per_ms: 10_000,
        ..Default::default()
    };

    // The body computes the deadline and carries it in the outcome;
    // `apply` is what stamps `thread.wake_time` (phase-C closure).
    let deadline = 1_000_000 + 50 * 10_000;
    let outcome = syscall::ms_sleep(common::view(&t), 50, &hw);
    assert_eq!(outcome, SyscallOutcome::SleepUntil { deadline });

    // End-to-end: applying the outcome stamps wake_time on the thread.
    let mut r = unsafe { process::RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();
    let action = apply_syscall_outcome(outcome, &mut r, &mut f, 0x2_2000_0400);
    assert_eq!(action, ShimAction::Yield(ThreadState::Suspended));
    assert_eq!(common::view(&t).wake_time(), deadline);
}

#[test]
fn zero_ms_still_yields() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw {
        now_ticks: 42,
        ..Default::default()
    };

    let outcome = syscall::ms_sleep(common::view(&t), 0, &hw);

    assert_eq!(outcome, SyscallOutcome::SleepUntil { deadline: 42 });
}

#[test]
fn rejects_ms_at_cap() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    common::running(&mut t).set_wake_time(0xDEAD); // must stay untouched on reject path
    let hw = FakeHw::default();

    let outcome = syscall::ms_sleep(common::view(&t), syscall::MAX_SLEEP_MS, &hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret()
        }
    );
    assert_eq!(common::view(&t).wake_time(), 0xDEAD);
}

#[test]
fn rejects_ms_above_cap() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw::default();

    let outcome = syscall::ms_sleep(common::view(&t), syscall::MAX_SLEEP_MS + 1, &hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret()
        }
    );
}

#[test]
fn wake_time_wraps_on_overflow() {
    // Matches the production `wrapping_add` / `wrapping_mul` semantics —
    // a near-max `now` plus a small sleep should wrap cleanly.
    let t = make_thread(ThreadState::Running, SPP::User);
    let hw = FakeHw {
        now_ticks: u64::MAX - 1000,
        ticks_per_ms: 10_000,
        ..Default::default()
    };

    let outcome = syscall::ms_sleep(common::view(&t), 1, &hw);

    let expected = (u64::MAX - 1000).wrapping_add(10_000) as usize;
    assert_eq!(outcome, SyscallOutcome::SleepUntil { deadline: expected });
}
