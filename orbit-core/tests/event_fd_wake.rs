//! Handler-level smoke for the EventFd + wake_tid syscalls (A4 / A6).
//!
//! Verifies the syscall-stub side of the doorbell chain:
//! - `wake_tid_req` short-circuits on `target_tid == 0` (the
//!   fast-path skip writers use when they read a zero hint from the
//!   shared region).
//! - `wake_tid_req` for a non-zero target yields `Blocking` and pushes
//!   a `PendingWork::WakeTid` with the right pid/target_tid.
//! - `eventfd_req` rejects unknown flag bits, vaddrs outside the
//!   shared range, and unaligned vaddrs at the boundary.
//! - `eventfd_req` for valid args yields `Blocking` and pushes a
//!   `PendingWork::EventFdCreate` carrying the validated args.
//!
//! The manager-side wake mechanics (`set_wake_reason_where` →
//! `wake_override |= NET_IO` → CAS Suspended → Ready) live in kmain
//! and aren't exercised here — they're shared with `WakeEvent::Pid`
//! (existing) and already covered by `read_stdin.rs`.

mod common;

use device::TrapFrame;
use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{EAGAIN, EINVAL, Errno};
use orbit_abi::event_fd::{EFD_CLOEXEC, EFD_NONBLOCK, EFD_SEMAPHORE, EVENTFD_REGION_SIZE};
use orbit_abi::layout::UPROC_SHARED_BASE;
use orbit_core::{PendingWork, SyscallOutcome, WakeTidReq, syscall};

use common::{FakeHw, make_thread};

// ---- wake_tid -------------------------------------------------------

#[test]
fn wake_tid_zero_target_returns_zero_without_pending_work() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.tid = 7;
    t.pid = 3;
    let mut hw = FakeHw::default();
    let mut frame = TrapFrame::empty();
    // a0 = syscall, a1 = target_tid. We set a1=0 (the sentinel).
    frame.regs[10] = orbit_abi::syscall::WAKE_TID;
    frame.regs[11] = 0;

    let outcome = syscall::wake_tid_req(common::view(&t), &frame, &mut hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: 0 });
    assert!(
        hw.pending_work.is_empty(),
        "target_tid=0 must skip the manager queue"
    );
}

#[test]
fn wake_tid_nonzero_target_pushes_pending_work_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.tid = 7;
    t.pid = 3;
    let mut hw = FakeHw::default();
    let mut frame = TrapFrame::empty();
    frame.regs[10] = orbit_abi::syscall::WAKE_TID;
    frame.regs[11] = 42; // target_tid

    let outcome = syscall::wake_tid_req(common::view(&t), &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::ParkForPublish
    );
    assert_eq!(hw.pending_work.len(), 1);
    match &hw.pending_work[0] {
        PendingWork::WakeTid { req, pid, tid } => {
            assert_eq!(*req, WakeTidReq { target_tid: 42 });
            assert_eq!(*pid, 3);
            assert_eq!(*tid, 7);
        }
        other => panic!("expected PendingWork::WakeTid, got {other:?}"),
    }
}

#[test]
fn wake_tid_returns_eagain_when_work_ring_full() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.tid = 7;
    t.pid = 3;
    let mut hw = FakeHw::default();
    hw.pending_work_ok = false;
    let mut frame = TrapFrame::empty();
    frame.regs[10] = orbit_abi::syscall::WAKE_TID;
    frame.regs[11] = 42;

    let outcome = syscall::wake_tid_req(common::view(&t), &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret()
        }
    );
}

// ---- eventfd --------------------------------------------------------

fn good_vaddr() -> u64 {
    // Page-aligned, inside UPROC_SHARED_BASE..UPROC_SHARED_END.
    UPROC_SHARED_BASE
}

fn run_eventfd(initval: u64, flags: u32, vaddr: u64, hw: &mut FakeHw) -> SyscallOutcome {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.tid = 11;
    t.pid = 5;
    let mut frame = TrapFrame::empty();
    frame.regs[10] = orbit_abi::syscall::EVENTFD;
    frame.regs[11] = vaddr as usize;
    frame.regs[12] = initval as usize;
    frame.regs[13] = flags as usize;
    syscall::eventfd_req(common::view(&t), &frame, hw)
}

#[test]
fn eventfd_rejects_unknown_flags() {
    let mut hw = FakeHw::default();
    let outcome = run_eventfd(0, 0xFFFF_0000, good_vaddr(), &mut hw);
    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret()
        }
    );
    assert!(hw.pending_work.is_empty());
}

#[test]
fn eventfd_rejects_vaddr_outside_shared_range() {
    let mut hw = FakeHw::default();
    // priv range vaddr
    let priv_vaddr = orbit_abi::layout::UPROC_PRIV_BASE;
    let outcome = run_eventfd(0, 0, priv_vaddr, &mut hw);
    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret()
        }
    );
    assert!(hw.pending_work.is_empty());
}

#[test]
fn eventfd_rejects_unaligned_vaddr() {
    let mut hw = FakeHw::default();
    let outcome = run_eventfd(0, 0, good_vaddr() + 0x123, &mut hw);
    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret()
        }
    );
}

#[test]
fn eventfd_valid_args_yield_blocking_and_push_pending_work() {
    let mut hw = FakeHw::default();
    let outcome = run_eventfd(
        42,
        EFD_NONBLOCK | EFD_SEMAPHORE | EFD_CLOEXEC,
        good_vaddr(),
        &mut hw,
    );

    assert_eq!(
        outcome,
        SyscallOutcome::ParkForPublish
    );
    assert_eq!(hw.pending_work.len(), 1);
    match &hw.pending_work[0] {
        PendingWork::EventFdCreate { req, pid, tid, .. } => {
            assert_eq!(*pid, 5);
            assert_eq!(*tid, 11);
            assert_eq!(req.initval, 42);
            assert_eq!(req.flags, EFD_NONBLOCK | EFD_SEMAPHORE | EFD_CLOEXEC);
            assert_eq!(req.vaddr_hint.raw(), good_vaddr());
        }
        other => panic!("expected PendingWork::EventFdCreate, got {other:?}"),
    }
}

#[test]
fn eventfd_full_work_ring_returns_eagain() {
    let mut hw = FakeHw::default();
    hw.pending_work_ok = false;
    let outcome = run_eventfd(0, 0, good_vaddr(), &mut hw);
    assert_eq!(
        outcome,
        SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret()
        }
    );
}

#[test]
fn eventfd_region_size_invariants_hold() {
    // ABI sanity: the region is exactly one page and the EventFd header
    // fits in one cache line. The kernel's run_eventfd_create_req
    // allocates against `EVENTFD_REGION_SIZE`; if it ever drifts from
    // PAGE_SIZE the manager's sfence_vma broadcast would need a sweep
    // wider than one page.
    assert_eq!(EVENTFD_REGION_SIZE, orbit_core::PAGE_SIZE);
    assert_eq!(core::mem::size_of::<orbit_abi::event_fd::EventFd>(), 64);
}
