//! Validates the `SyscallOutcome → ShimAction` contract that kmain's
//! dispatch shim relies on. These tests exist specifically because the
//! contract isn't enforced by the type system alone — forgetting to
//! bump `thread.pc` on the `Return` path silently manifests as the
//! thread re-executing its ecall forever (the kind of bug that passed
//! the happy-path demo but broke the moment umode hit a synchronous
//! error return).

mod common;


use process::{RunningThread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::{ShimAction, SyscallOutcome, apply_syscall_outcome};

use common::{make_frame, make_thread};

const ECALL_EPC: usize = 0x2_2000_0400;

// ---- Return ----

#[test]
fn return_bumps_pc_past_ecall() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(ECALL_EPC);
    let mut f = make_frame();

    let action =
        apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    assert_eq!(action, ShimAction::Resume);
    assert_eq!(r.view().pc(), ECALL_EPC + 4);
}

#[test]
fn return_writes_ret_into_frame_reg10() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: -2 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    // The thread resumes from its snapshotted frame, not the local one,
    // so thread.frame.regs[10] is what the user code will see.
    assert_eq!(r.frame_reg(10), (-2isize) as usize);
}

#[test]
fn return_snapshots_full_frame_into_thread_frame() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();
    f.regs[11] = 0xAAAA;
    f.regs[12] = 0xBBBB;
    f.regs[15] = 0xCCCC;

    let _ = apply_syscall_outcome(SyscallOutcome::Return { ret: 7 }, &mut r, &mut f, ECALL_EPC);

    assert_eq!(r.frame_reg(10), 7);
    assert_eq!(r.frame_reg(11), 0xAAAA);
    assert_eq!(r.frame_reg(12), 0xBBBB);
    assert_eq!(r.frame_reg(15), 0xCCCC);
}

#[test]
fn return_action_is_resume_not_yield() {
    // The kmain shim uses this to decide whether to fall through to
    // check_context_and_switch (Resume) or invoke
    // exit_thread_with_state (Yield). Getting this wrong sends the
    // thread to the wrong post-syscall path.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();

    let action =
        apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    match action {
        ShimAction::Resume => {}
        ShimAction::Yield(_) => panic!("Return must not produce Yield"),
    }
}

// ---- Yield with Some(ret) ----

#[test]
fn yield_some_ret_writes_ret_before_snapshot() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::SleepUntil { deadline: 0 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Yield(ThreadState::Suspended));
    assert_eq!(r.frame_reg(10), 0);
    assert_eq!(r.view().pc(), ECALL_EPC + 4);
}

#[test]
fn yield_ready_state_propagates_to_action() {
    // serial_print yields Ready after committing the result — exercises
    // the "syscall completed, re-enter scheduler" path.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::DoneReschedule { ret: -5 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Yield(ThreadState::Ready));
    assert_eq!(r.frame_reg(10), (-5isize) as usize);
}

// ---- Yield with None (manager-completed syscalls) ----

#[test]
fn yield_none_ret_preserves_frame_reg10() {
    // mmap / nc_create / close yield with ret=None because the manager
    // fills the return value at unblock time. apply_syscall_outcome
    // must leave regs[10] exactly as it was (typically the syscall
    // number, stale but soon overwritten).
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();
    f.regs[10] = 4096; // MMAP syscall number

    let _ = apply_syscall_outcome(
        SyscallOutcome::ParkForPublish,
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    // frame.regs[10] stays the stale syscall number (manager overwrites
    // thread.frame.regs[10] directly when unblocking).
    assert_eq!(f.regs[10], 4096);
    assert_eq!(r.frame_reg(10), 4096);
}

#[test]
fn yield_blocking_still_bumps_pc() {
    // Block-reason syscalls (mmap_req etc) need pc to advance even
    // though the thread isn't running anything right now — when the
    // manager unblocks them, execution resumes at epc+4, not the
    // original ecall.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(ECALL_EPC);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::ParkForPublish,
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(r.view().pc(), ECALL_EPC + 4);
}

// ---- Regression pin: the Return-arm bug ----

#[test]
fn return_must_bump_pc_past_ecall() {
    // This test fails with the historical bug where the Return branch
    // set frame.regs[10] but skipped both the snapshot and the pc
    // bump. Documented here so a future refactor can't silently
    // reintroduce that behavior.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(ECALL_EPC);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: -2 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_ne!(
        r.view().pc(),
        ECALL_EPC,
        "Return must bump pc past the ecall — without this, \
         the thread re-executes the same ecall on resume"
    );
}

#[test]
fn return_must_overwrite_thread_frame_with_ret() {
    // Second half of the bug: without the snapshot, thread.frame still
    // holds whatever was written before the trap, so the user code
    // sees a stale (non-ret) value in a0 on resume.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_frame_reg(10, 0xDEAD_BEEF);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(
        SyscallOutcome::Return { ret: -2 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(r.frame_reg(10), (-2isize) as usize);
    assert_ne!(r.frame_reg(10), 0xDEAD_BEEF);
}

// ---- YieldRetry ----

#[test]
fn retry_on_doorbell_keeps_pc_at_ecall_and_yields_suspended() {
    // The whole point of a retry park is that the resumed thread re-runs
    // the ecall instead of stepping past it. Bumping pc here would
    // turn read_stdin's park-and-retry into a single-shot read that
    // returns garbage on wake. Phase-C closure: a retry park is *always*
    // Suspended (the only state whose wake doesn't require a published
    // completion slot) — there is no longer a Blocking-retry to express.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(ECALL_EPC);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::RetryOnDoorbell,
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Yield(ThreadState::Suspended));
    assert_eq!(r.view().pc(), ECALL_EPC);
}

#[test]
fn yield_retry_preserves_a_regs_for_re_execute() {
    // Args land in a1..a4 (frame.regs[11..15]) for the user's ecall.
    // Resume must restore them as-of the trap so the re-executed
    // syscall handler sees identical inputs.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let mut f = make_frame();
    f.regs[10] = 0x42; // syscall number
    f.regs[11] = 0xAAAA; // arg0
    f.regs[12] = 0xBBBB; // arg1
    f.regs[13] = 0xCCCC; // arg2
    f.regs[14] = 0xDDDD; // arg3

    let _ = apply_syscall_outcome(
        SyscallOutcome::RetryOnDoorbell,
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(r.frame_reg(10), 0x42);
    assert_eq!(r.frame_reg(11), 0xAAAA);
    assert_eq!(r.frame_reg(12), 0xBBBB);
    assert_eq!(r.frame_reg(13), 0xCCCC);
    assert_eq!(r.frame_reg(14), 0xDDDD);
}

// ---- Mode/state gate ----
//
// These pin down the defense-in-depth gate that prevents a U-mode-ecall
// commit from landing on the wrong thread when `hart.current` was
// retargeted between trap entry and `apply_syscall_outcome`. The QEMU
// repro: orbit-loader's `ch_yield(4100)` ecall lands on cpu2, but
// `cpu2.current` has been swapped to knet (a Supervisor kthread). Without
// the gate, `apply_syscall_outcome` would stamp `epc + 4 = 0x22000339c`
// (a user VA) into `knet.pc`, and the next dispatch sret-s to a user
// address in S-mode → cause=12 panic. Mirrors the gate philosophy of
// `trap::update_trap_frame`.

const KTHREAD_PC: usize = 0xFFFF_FFC0_0001_0000;

#[test]
fn return_refuses_to_commit_to_kthread() {
    let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(KTHREAD_PC);
    let orig_reg11 = r.frame_reg(11);
    let mut f = make_frame();
    f.regs[11] = 0xDEAD;

    let action =
        apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    // No-op fallback so the trap dispatcher still unwinds, but no writes
    // to thread.pc/thread.frame.
    assert_eq!(action, ShimAction::Resume);
    assert_eq!(r.view().pc(), KTHREAD_PC);
    assert_eq!(r.frame_reg(11), orig_reg11);
}

#[test]
fn yield_refuses_to_commit_to_kthread() {
    let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(KTHREAD_PC);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::SleepUntil { deadline: 0 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    // Crucially NOT Yield(Suspended) — that would null kmain's
    // hart.current and then long-jump via exit_thread_with_state. We
    // want the dispatcher to return normally so cleanup happens
    // elsewhere.
    assert_eq!(action, ShimAction::Resume);
    assert_eq!(r.view().pc(), KTHREAD_PC);
    assert_eq!(r.view().state(), ThreadState::Running as usize);
}

#[test]
fn yield_retry_refuses_to_commit_to_kthread() {
    let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(KTHREAD_PC);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::RetryOnDoorbell,
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Resume);
    assert_eq!(r.view().pc(), KTHREAD_PC);
}

#[test]
fn return2_refuses_to_commit_to_kthread() {
    let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(KTHREAD_PC);
    let mut f = make_frame();

    let action = apply_syscall_outcome(
        SyscallOutcome::Return2 { ret0: 1, ret1: 2 },
        &mut r,
        &mut f,
        ECALL_EPC,
    );

    assert_eq!(action, ShimAction::Resume);
    assert_eq!(r.view().pc(), KTHREAD_PC);
    // Neither ret0 nor ret1 stamped into the kthread frame.
    assert_ne!(r.frame_reg(10), 1);
    assert_ne!(r.frame_reg(11), 2);
}

#[test]
fn assigned_user_thread_refuses_to_commit() {
    // `Assigned` means the manager just installed the thread but
    // dispatch hasn't started — same logic as the trap-frame gate.
    // Important specifically because `Assigned` was the state we saw
    // most often in the QEMU mode-mismatch warnings.
    let mut t = make_thread(ThreadState::Assigned, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let original_pc = 0x2200_0000;
    r.set_pc(original_pc);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    assert_eq!(r.view().pc(), original_pc);
}

#[test]
fn ready_user_thread_refuses_to_commit() {
    let mut t = make_thread(ThreadState::Ready, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    let original_pc = 0x2200_0000;
    r.set_pc(original_pc);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    assert_eq!(r.view().pc(), original_pc);
}

#[test]
fn suspended_user_thread_commits_normally() {
    // Sanity check that the gate isn't over-tightened — a user thread in
    // {Running, Suspended, Blocking} is the legitimate trap-saving set,
    // matching `update_trap_frame`. A `Suspended` user thread can
    // legitimately be `current` mid-syscall (e.g., the manager just
    // transitioned it during ms_sleep before `exit_thread_with_state`
    // ran).
    let mut t = make_thread(ThreadState::Suspended, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(ECALL_EPC);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    assert_eq!(r.view().pc(), ECALL_EPC + 4);
}

#[test]
fn blocking_user_thread_commits_normally() {
    let mut t = make_thread(ThreadState::Blocking, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(ECALL_EPC);
    let mut f = make_frame();

    let _ = apply_syscall_outcome(SyscallOutcome::Return { ret: 0 }, &mut r, &mut f, ECALL_EPC);

    assert_eq!(r.view().pc(), ECALL_EPC + 4);
}
