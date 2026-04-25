//! Pure-logic half of the Orbit kernel.
//!
//! Syscall bodies, scheduler policy, and the `k_net` step function live here,
//! parameterized over a [`Hardware`] effect trait. kmain supplies a concrete
//! RISC-V impl; tests supply an in-memory fake. See
//! [docs/testable-kernel.md](../../docs/testable-kernel.md).

#![no_std]

extern crate alloc;

use process::ThreadState;

pub mod manager;
pub mod net;
pub mod pending_work;
pub mod sched;
pub mod syscall;
pub mod trap;

pub use pending_work::{
    CloseHandleReq, CreateProcessReq, MemMapReq, NetChannelCreationReq, PendingWork,
};

/// Page size assumed by pure logic when bounding user-memory ranges. Must
/// match the walker's leaf granularity on the live target (Sv48 4 KiB).
pub const PAGE_SIZE: usize = 4096;

/// Narrow effect surface the pure logic uses to reach hardware. Grows as
/// migrations pull more handlers in. Keep it narrow — this is not an HAL.
pub trait Hardware {
    /// Free-running tick counter. RISC-V `time` CSR on hardware.
    fn now_ticks(&self) -> u64;

    /// Tick rate of [`Hardware::now_ticks`]. Used to convert ms deadlines to
    /// absolute tick values.
    fn ticks_per_ms(&self) -> u64;

    /// True iff `user_va` resolves to a mapped page under the root table at
    /// `root_table_pa` (`thread.root_table_addr()`). Only the starting VA
    /// is checked — callers bound `len` at the [`PAGE_SIZE`] level so the
    /// range can't straddle an unchecked second page.
    fn user_va_translates(&self, root_table_pa: u64, user_va: u64) -> bool;

    /// Copy `dst.len()` bytes from user space starting at `user_va` into
    /// `dst`. Impl toggles SUM around the read. Caller must have validated
    /// the range with [`Hardware::user_va_translates`] first.
    fn copy_from_user(&mut self, user_va: u64, dst: &mut [u8]);

    /// Write user-originated text to the kernel serial console, prefixed
    /// with the standard `{now_ticks}t USER[{pid}.{tid}]: ` tag so user
    /// output lines up visually with kernel tracing logs. Impl uses
    /// `core::fmt` via the serial driver; no buffering needed in pure
    /// code. Returns Err on UART failure.
    fn serial_write_user(&mut self, pid: u16, tid: u32, text: &str) -> Result<(), ()>;

    /// Append `bytes` to `pid`'s framebuffer scrollback. Real impl
    /// pushes a `Cmd` onto `k_gpu`'s thingbuf ring; the compositor
    /// thread eventually appends to `scrollbacks[Process(pid)]` and
    /// repaints if that source is active. Returns `Err(())` if the
    /// ring is full or the gpu package isn't initialized — in which
    /// case the syscall returns `-7` (EAGAIN-analog).
    fn console_write_user(&mut self, pid: u16, bytes: &[u8]) -> Result<(), ()>;

    /// Send an inter-processor interrupt to `hart_id`. Real impl writes
    /// the hart's ACLINT SSWI MSIP; tests record the call.
    fn wake_hart(&mut self, hart_id: u32);

    /// Enqueue `work` onto the manager's work ring. Real impl pushes
    /// onto a `thingbuf::StaticThingBuf` and returns `Err(work)` if the
    /// ring is full (caller maps to `-EAGAIN`); tests record the push
    /// for assertion.
    fn push_pending_work(&mut self, work: PendingWork) -> Result<(), PendingWork>;
}

/// What a pure syscall handler tells the shim to do after it returns.
///
/// The pure handler only mutates in-memory state and reports the intended
/// outcome; [`apply_syscall_outcome`] translates that into the concrete
/// frame / pc / state mutations a shim needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallOutcome {
    /// Commit side effects (snapshot frame, bump pc) and yield the current
    /// thread into `state` via the asm switch. If `ret` is `Some`, the shim
    /// writes it into `regs[10]` before the snapshot so the resumed thread
    /// sees that value; `None` means "leave the frame alone" for
    /// manager-completed syscalls (mmap, nc_create, close) whose return
    /// value is written into `thread.frame.regs[10]` at unblock time.
    Yield { state: ThreadState, ret: Option<isize> },

    /// Write `ret` into `regs[10]`, commit the frame snapshot + pc bump
    /// (so the thread resumes past the ecall with `ret` visible), and
    /// return to the trap dispatcher without yielding. Used for
    /// synchronous error returns from handlers that don't block.
    Return { ret: isize },
}

/// What the shim should do after [`apply_syscall_outcome`] commits the
/// thread/frame state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShimAction {
    /// The thread's state is already runnable; the shim returns
    /// normally from the trap dispatcher (kmain's s_trap then calls
    /// `check_context_and_switch`).
    Resume,
    /// The thread must yield into `state` via the context-switch asm.
    /// kmain's shim invokes `exit_thread_with_state(state)` which
    /// doesn't return.
    Yield(ThreadState),
}

/// Translate a [`SyscallOutcome`] into thread-state mutations + a
/// [`ShimAction`]. Shared between kmain's real shim and host tests so
/// the two can't drift — a bug in here is caught at
/// `cargo test`, not only when QEMU boots and a thread loops on an
/// ecall forever.
///
/// Both variants commit the frame snapshot into `thread.frame` and
/// advance `thread.pc` to `epc + 4`. The variants differ only in what
/// the shim does next: `Return` falls back through the trap dispatcher,
/// `Yield` triggers the context-switch asm.
pub fn apply_syscall_outcome(
    outcome: SyscallOutcome,
    thread: &mut process::Thread,
    frame: &mut device::TrapFrame,
    epc: usize,
) -> ShimAction {
    use core::sync::atomic::Ordering;

    match outcome {
        SyscallOutcome::Return { ret } => {
            frame.regs[10] = ret as usize;
            *thread.frame = *frame;
            thread.pc.store(epc + 4, Ordering::Release);
            ShimAction::Resume
        }
        SyscallOutcome::Yield { state, ret } => {
            if let Some(r) = ret {
                frame.regs[10] = r as usize;
            }
            *thread.frame = *frame;
            thread.pc.store(epc + 4, Ordering::Release);
            ShimAction::Yield(state)
        }
    }
}
