//! Trap-dispatch helpers that aren't syscall bodies.
//!
//! These run on every trap entry (sync or async) before the dispatch arm
//! decides what to do. Pure logic; the shim handles the null-current and
//! `enter_hart_context` plumbing around them.

use device::TrapFrame;
use process::{RunningThread, ThreadState};
use riscv::register::sstatus::SPP;

/// Mirror of the kmain `update_thread_and_trap_frame` body.
///
/// Always writes `frame.asid` so post-trap kernel work on this hart can rely
/// on it. Then, *only if* the trap actually described the thread's own
/// execution (mode gate), snapshots the frame into `thread.frame` and
/// advances `thread.pc = epc`.
///
/// The mode gate exists because an async S-mode interrupt can fire while
/// the kernel is mid-context-switch for a user thread (SIE left on inside
/// `enter_hart_context_asm`). In that case `epc` points into kernel
/// `.text` and saving it as `thread.pc` would break `sret` on resume —
/// see [docs/trap-mode-guard.md](../../docs/trap-mode-guard.md).
pub fn update_trap_frame(
    running: &mut RunningThread,
    epc: usize,
    frame: &mut TrapFrame,
    from_user: bool,
) {
    let view = running.view();
    frame.asid = view.pid() as usize;

    let trap_was_in_thread = (view.mode() == SPP::User) == from_user;
    if !trap_was_in_thread {
        return;
    }

    let state = view.state();
    if state == ThreadState::Running as usize
        || state == ThreadState::Suspended as usize
        || state == ThreadState::Blocking as usize
    {
        // Own-hart frame/pc snapshot — the sealed write flows through the
        // capability, which is sound because the hart owns this thread.
        running.save_trap_frame(frame, epc);
    }
}
