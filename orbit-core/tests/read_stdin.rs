//! Pure-handler tests for `read_stdin`. Exercises every branch of
//! the try → recheck → park decision tree against the FakeHw stdin
//! harness so the integration with [`process::ProcessStdin`] (which
//! the kmain Hardware impl wraps) doesn't have to round-trip through
//! QEMU to validate the policy.

mod common;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{EAGAIN, EBUSY, EFAULT, EINVAL, Errno};
use orbit_core::syscall::READ_STDIN_NONBLOCK;
use orbit_core::{PAGE_SIZE, ShimAction, SyscallOutcome, apply_syscall_outcome, syscall};

use common::{FakeHw, make_frame, make_thread};

const UVA: u64 = 0x2_0000_1000;
const PID: u16 = 1;

fn frame_with(len: usize, flags: usize) -> device::TrapFrame {
    let mut f = make_frame();
    f.regs[11] = UVA as usize;
    f.regs[12] = len;
    f.regs[13] = flags;
    f
}

fn ready(ret: isize) -> SyscallOutcome {
    SyscallOutcome::Yield {
        state: ThreadState::Ready,
        ret: Some(ret),
    }
}

#[test]
fn bytes_available_returns_count_synchronously() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(16, 0);
    let mut hw = FakeHw::default();
    hw.stdin_ready.insert(PID, vec![b"hello".to_vec()]);

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(5));
    assert!(t.handle.is_none(), "no parking when bytes are available");
    assert_eq!(hw.stdin_drain_writes.len(), 1);
    assert_eq!(hw.stdin_drain_writes[0], (PID, UVA, b"hello".to_vec()));
    assert!(hw.stdin_parked.is_empty(), "no park on synchronous read");
}

#[test]
fn nonblock_empty_returns_eagain() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(16, READ_STDIN_NONBLOCK);
    let mut hw = FakeHw::default();
    // No stdin_ready entry → drain returns 0.

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EAGAIN).to_ret()));
    assert!(t.handle.is_none());
    assert!(hw.stdin_parked.is_empty(), "NONBLOCK never parks");
}

#[test]
fn block_empty_parks_then_yields_retry() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(16, 0);
    let mut hw = FakeHw::default();
    // Both drain calls (try + recheck) return 0.

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert!(matches!(
        outcome,
        SyscallOutcome::YieldRetry {
            state: ThreadState::Blocking
        }
    ));
    assert!(t.handle.is_some(), "thread parked on a handle");
    assert_eq!(hw.stdin_parked.len(), 1, "park called exactly once");
    assert_eq!(hw.stdin_parked[0].0, PID);
    assert!(
        hw.stdin_unparked.is_empty(),
        "no cancel on truly-empty path"
    );
}

#[test]
fn block_empty_recheck_drain_cancels_park() {
    // The race-window case: try_drain returns 0, park succeeds, then
    // a producer pushes a byte → recheck drain returns count, handler
    // cancels its park and returns synchronously.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(16, 0);
    let mut hw = FakeHw::default();
    // First drain: empty (no entries). Second drain: hands back 4 bytes.
    // Encoding: stdin_ready returns one entry per call, so we stage
    // an empty-then-data sequence by inserting two entries: the
    // first empty Vec is consumed by the first drain (returning 0),
    // the second by the recheck.
    hw.stdin_ready
        .insert(PID, vec![Vec::new(), b"x42!".to_vec()]);

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(4));
    assert!(t.handle.is_none(), "park cancelled before yield");
    assert_eq!(hw.stdin_parked.len(), 0, "unpark removed the entry");
    assert_eq!(hw.stdin_unparked, vec![PID]);
    // Recheck is what produced the byte count.
    assert_eq!(hw.stdin_drain_writes.last().unwrap().2, b"x42!".to_vec());
}

#[test]
fn park_failure_returns_ebusy() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(16, 0);
    let mut hw = FakeHw::default();
    hw.stdin_park_ok = false;

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EBUSY).to_ret()));
    assert!(t.handle.is_none());
    assert!(hw.stdin_parked.is_empty());
}

#[test]
fn bad_user_va_returns_efault() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(16, 0);
    let mut hw = FakeHw {
        translates: false,
        ..Default::default()
    };

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EFAULT).to_ret()));
    assert!(hw.stdin_drain_writes.is_empty(), "no drain on EFAULT");
}

#[test]
fn len_zero_returns_einval() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(0, 0);
    let mut hw = FakeHw::default();

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EINVAL).to_ret()));
}

#[test]
fn len_above_page_returns_einval() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let frame = frame_with(PAGE_SIZE + 1, 0);
    let mut hw = FakeHw::default();

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EINVAL).to_ret()));
}

#[test]
fn rejects_kernel_vaddr() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let mut frame = frame_with(16, 0);
    frame.regs[11] = 0xFFFF_FFC0_0000_0000;
    let mut hw = FakeHw::default();

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EFAULT).to_ret()));
    assert!(
        hw.stdin_drain_writes.is_empty(),
        "no drain on out-of-range va"
    );
    assert!(hw.stdin_parked.is_empty(), "no park on out-of-range va");
}

#[test]
fn rejects_null_guard_vaddr() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    let mut frame = frame_with(16, 0);
    frame.regs[11] = 0x0;
    let mut hw = FakeHw::default();

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EFAULT).to_ret()));
    assert!(hw.stdin_drain_writes.is_empty());
}

#[test]
fn yield_retry_keeps_pc_so_resume_re_executes_ecall() {
    // End-to-end with apply_syscall_outcome: the YieldRetry shape
    // produced by read_stdin must keep pc at the ecall (so the
    // resumed thread re-enters the syscall handler) and a-regs
    // unchanged (so the re-execute sees the same buffer pointer
    // and length).
    const ECALL_EPC: usize = 0x2_2000_0400;
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = PID;
    use std::sync::atomic::Ordering;
    t.pc.store(ECALL_EPC, Ordering::Release);
    let mut frame = frame_with(16, 0);
    // Spy: stash a synthetic syscall number into a0 so we can
    // confirm the snapshot preserves it.
    frame.regs[10] = 0xDEAD_BEEF;
    let mut hw = FakeHw::default();

    let outcome = syscall::read_stdin(&mut t, &frame, &mut hw);
    let action = apply_syscall_outcome(outcome, &mut t, &mut frame, ECALL_EPC);

    assert_eq!(action, ShimAction::Yield(ThreadState::Blocking));
    assert_eq!(
        t.pc.load(Ordering::Acquire),
        ECALL_EPC,
        "park-and-retry must keep pc at the ecall"
    );
    assert_eq!(
        t.frame.regs[10], 0xDEAD_BEEF,
        "syscall number snapshot preserved"
    );
    assert_eq!(t.frame.regs[11], UVA as usize, "buf ptr snapshot preserved");
    assert_eq!(t.frame.regs[12], 16, "len snapshot preserved");
}
