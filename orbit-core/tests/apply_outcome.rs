//! Validates the `SyscallOutcome → ShimAction` contract that kmain's
//! dispatch shim relies on. These tests exist specifically because the
//! contract isn't enforced by the type system alone — forgetting to
//! bump `thread.pc` on the `Return` path silently manifests as the
//! thread re-executing its ecall forever (the kind of bug that passed
//! the happy-path demo but broke the moment umode hit a synchronous
//! error return).

mod common;

use std::sync::atomic::Ordering;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_core::{ShimAction, SyscallOutcome, apply_syscall_outcome};

use common::{make_frame, make_thread};

const ECALL_EPC: usize = 0x2_2000_0400;

// ---- Return ----

#[test]
fn return_bumps_pc_past_ecall() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pc.store(ECALL_EPC, Ordering::Release);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::Return { ret: 0 },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Resume);
    assert_eq!(t.pc.load(Ordering::Acquire), ECALL_EPC + 4);
}

#[test]
fn return_writes_ret_into_frame_reg10() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: -2 },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    // The thread resumes from its snapshotted frame, not the local one,
    // so thread.frame.regs[10] is what the user code will see.
    assert_eq!(t.frame.regs[10], (-2isize) as usize);
}

#[test]
fn return_snapshots_full_frame_into_thread_frame() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut f = make_frame();
    f.regs[11] = 0xAAAA;
    f.regs[12] = 0xBBBB;
    f.regs[15] = 0xCCCC;

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: 7 },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(t.frame.regs[10], 7);
    assert_eq!(t.frame.regs[11], 0xAAAA);
    assert_eq!(t.frame.regs[12], 0xBBBB);
    assert_eq!(t.frame.regs[15], 0xCCCC);
}

#[test]
fn return_action_is_resume_not_yield() {
    // The kmain shim uses this to decide whether to fall through to
    // check_context_and_switch (Resume) or invoke
    // exit_thread_with_state (Yield). Getting this wrong sends the
    // thread to the wrong post-syscall path.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::Return { ret: 0 },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    match action {
        ShimAction::Resume => {}
        ShimAction::Yield(_) => panic!("Return must not produce Yield"),
    }
}

// ---- Yield with Some(ret) ----

#[test]
fn yield_some_ret_writes_ret_before_snapshot() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::Yield { state: ThreadState::Suspended, ret: Some(0) },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Yield(ThreadState::Suspended));
    assert_eq!(t.frame.regs[10], 0);
    assert_eq!(t.pc.load(Ordering::Acquire), ECALL_EPC + 4);
}

#[test]
fn yield_ready_state_propagates_to_action() {
    // serial_print yields Ready after committing the result — exercises
    // the "syscall completed, re-enter scheduler" path.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(-5) },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Yield(ThreadState::Ready));
    assert_eq!(t.frame.regs[10], (-5isize) as usize);
}

// ---- Yield with None (manager-completed syscalls) ----

#[test]
fn yield_none_ret_preserves_frame_reg10() {
    // mmap / nc_create / close yield with ret=None because the manager
    // fills the return value at unblock time. apply_syscall_outcome
    // must leave regs[10] exactly as it was (typically the syscall
    // number, stale but soon overwritten).
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut f = make_frame();
    f.regs[10] = 4096; // MMAP syscall number

    let _ = apply_syscall_outcome(
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    // frame.regs[10] stays the stale syscall number (manager overwrites
    // thread.frame.regs[10] directly when unblocking).
    assert_eq!(f.regs[10], 4096);
    assert_eq!(t.frame.regs[10], 4096);
}

#[test]
fn yield_blocking_still_bumps_pc() {
    // Block-reason syscalls (mmap_req etc) need pc to advance even
    // though the thread isn't running anything right now — when the
    // manager unblocks them, execution resumes at epc+4, not the
    // original ecall.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pc.store(ECALL_EPC, Ordering::Release);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(t.pc.load(Ordering::Acquire), ECALL_EPC + 4);
}

// ---- Regression pin: the Return-arm bug ----

#[test]
fn return_does_NOT_leave_pc_unchanged() {
    // This test fails with the historical bug where the Return branch
    // set frame.regs[10] but skipped both the snapshot and the pc
    // bump. Documented here so a future refactor can't silently
    // reintroduce that behavior.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pc.store(ECALL_EPC, Ordering::Release);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: -2 },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_ne!(
        t.pc.load(Ordering::Acquire),
        ECALL_EPC,
        "Return must bump pc past the ecall — without this, \
         the thread re-executes the same ecall on resume"
    );
}

#[test]
fn return_does_NOT_leave_thread_frame_stale() {
    // Second half of the bug: without the snapshot, thread.frame still
    // holds whatever was written before the trap, so the user code
    // sees a stale (non-ret) value in a0 on resume.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.frame.regs[10] = 0xDEAD_BEEF;
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: -2 },
        &mut t,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(t.frame.regs[10], (-2isize) as usize);
    assert_ne!(t.frame.regs[10], 0xDEAD_BEEF);
}
