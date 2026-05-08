//! Pure-logic half of the Orbit kernel.
//!
//! Syscall bodies, scheduler policy, and the `k_net` step function live here,
//! parameterized over a [`Hardware`] effect trait. kmain supplies a concrete
//! RISC-V impl; tests supply an in-memory fake. See
//! [docs/testable-kernel.md](../../docs/testable-kernel.md).

#![no_std]

extern crate alloc;

use mmu::sv48::PhysAddr;
use orbit_abi::layout::UserVa;
use process::ThreadState;

pub mod accounting;
pub mod denial_ring;
pub mod manager;
pub mod net;
pub mod pending_work;
pub mod ready_queue;
pub mod sched;
pub mod sleep_heap;
pub mod syscall;
pub mod tlb_shootdown;
pub mod trap;

/// Role registry, transition gates, and witness types.
///
/// Moved into `orbit-abi` so the [`process`] crate can name
/// [`ChildPerms`](orbit_abi::roles::ChildPerms) on the
/// `Process::install_child` signature without taking an
/// orbit-core dependency (orbit-core depends on process, not the
/// reverse). Re-exported here for ergonomics — call sites that
/// already import `orbit_core::roles::*` keep working.
///
/// Gated on the `kernel-policy` feature in orbit-abi; orbit-core
/// enables it unconditionally.
pub use orbit_abi::roles;

pub use pending_work::{
    CloseHandleReq, CreateProcessExReq, CreateProcessReq, CreateProcessV2Req, CreateThreadReq,
    FbSurfaceCreateReq, FbSurfaceDestroyReq, FsOpenReq, FsReadReq, FsReaddirReq, FsStatReq,
    FutexWaitReq, FutexWakeReq, MAX_FS_PATH_LEN, MemMapReq, NetChannelCreationReq, PendingWork,
    PledgeReq, SpawnContext, WaitPidReq,
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
    fn user_va_translates(&self, root_table_pa: PhysAddr, user_va: UserVa) -> bool;

    /// Copy `dst.len()` bytes from user space starting at `user_va` into
    /// `dst`. Impl toggles SUM around the read. Caller must have validated
    /// the range with [`Hardware::user_va_translates`] first.
    fn copy_from_user(&mut self, user_va: UserVa, dst: &mut [u8]);

    /// Write user-originated text to the kernel serial console, prefixed
    /// with the standard `{now_ticks}t USER[{pid}.{tid}]: ` tag so user
    /// output lines up visually with kernel tracing logs. Impl uses
    /// `core::fmt` via the serial driver; no buffering needed in pure
    /// code. Returns Err on UART failure.
    fn serial_write_user(&mut self, pid: u16, tid: u32, text: &str) -> Result<(), ()>;

    /// Append `bytes` to the framebuffer scrollback for the pane
    /// `dest_pid`. Real impl pushes a `Cmd` onto `k_gpu`'s thingbuf
    /// ring; the compositor thread eventually appends to
    /// `scrollbacks[Process(dest_pid)]` and repaints if that source
    /// is active. Returns `Err(())` if the ring is full or the gpu
    /// package isn't initialized — in which case the syscall returns
    /// `-7` (EAGAIN-analog).
    ///
    /// `dest_pid` is the *destination* pane, which the syscall
    /// resolves from the calling thread's `stdout_redirect` snapshot
    /// — for unredirected processes it equals the producer's pid; for
    /// children spawned with `CREATE_PROCESS_V2 stdout_capture=1` it
    /// equals the parent's pid.
    fn console_write_user(&mut self, dest_pid: u16, bytes: &[u8]) -> Result<(), ()>;

    /// Send an inter-processor interrupt to `hart_id`. Real impl writes
    /// the hart's ACLINT SSWI MSIP; tests record the call.
    fn wake_hart(&mut self, hart_id: usize);

    /// Enqueue `work` onto the manager's work ring. Real impl pushes
    /// onto a `thingbuf::StaticThingBuf` and returns `Err(work)` if the
    /// ring is full (caller maps to `-EAGAIN`); tests record the push
    /// for assertion.
    fn push_pending_work(&mut self, work: PendingWork) -> Result<(), PendingWork>;

    /// Drain up to `max_len` bytes from `pid`'s stdin ring directly
    /// into the user buffer at `user_va`, performing the SUM-gated
    /// copy on the user's satp. Returns the count actually drained
    /// (0 if the ring is empty or `pid` isn't registered).
    fn read_stdin_drain(&mut self, pid: u16, user_va: UserVa, max_len: usize) -> usize;

    /// Park `handle` on `pid`'s stdin slot. Returns `false` if a
    /// reader was already parked (caller emits EBUSY) or `pid` isn't
    /// registered. The handle is moved in; on success the impl
    /// retains the Arc, which `unpark_stdin_reader` (or
    /// `input::dispatch`'s push-and-signal) later reclaims.
    fn park_stdin_reader(&mut self, pid: u16, handle: process::CompletionHandle) -> bool;

    /// Cancel a park on `pid`'s stdin slot. Used by the read_stdin
    /// re-check path when bytes arrive between try_drain and park.
    /// Returns `true` if there was a parked reader to cancel; the
    /// impl drops the handle.
    fn unpark_stdin_reader(&mut self, pid: u16) -> bool;

    /// Drain up to `max_count` `KeyEvent`s from `pid`'s structured
    /// event ring directly into the user buffer at `user_va`. Returns
    /// the count drained (0 if empty or `pid` not registered). Same
    /// shape as [`Hardware::read_stdin_drain`] but for the structured
    /// counterpart.
    fn read_key_events_drain(&mut self, pid: u16, user_va: UserVa, max_count: usize) -> usize;

    /// Stamp `tid` as the parked reader on `pid`'s key-event ring.
    /// See [`process::key_events::ParkOutcome`] for the three result
    /// states. Returns `Busy` if `pid` isn't registered (treats
    /// "no ring" the same as "another tid is parked" — both should
    /// fail the syscall with EBUSY rather than silently parking
    /// against a missing ring).
    fn set_key_event_parker(&mut self, pid: u16, tid: u32) -> process::key_events::ParkOutcome;

    /// Clear `pid`'s key-event parker if it currently holds `tid`.
    /// Used by the read-side re-check race after stamping our tid:
    /// if events arrived during the park-vs-push window we cancel
    /// the park rather than yield Suspended.
    fn clear_key_event_parker_if(&mut self, pid: u16, tid: u32) -> bool;
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
    Yield {
        state: ThreadState,
        ret: Option<isize>,
    },

    /// Write `ret` into `regs[10]`, commit the frame snapshot + pc bump
    /// (so the thread resumes past the ecall with `ret` visible), and
    /// return to the trap dispatcher without yielding. Used for
    /// synchronous error returns from handlers that don't block.
    Return { ret: isize },

    /// Like `Return`, but writes a pair into `regs[10]` and `regs[11]`.
    /// For synchronous syscalls returning two `isize`s — modelled on
    /// Windows's `GetProcessAffinityMask` shape (current + allowed in
    /// one trap). The async two-return path (`create_netch`) goes via
    /// `Yield + signal_n`, not this variant.
    Return2 { ret0: isize, ret1: isize },

    /// Snapshot frame, *retain* pc at the ecall, yield into `state`.
    /// On wake the thread re-executes the ecall with its original
    /// args — used for park-and-retry primitives like `read_stdin`,
    /// where the signaler doesn't compute a return value (it just
    /// wakes the reader to retry). a-regs preserved from the trap
    /// snapshot, so the syscall handler re-enters with identical
    /// inputs.
    YieldRetry { state: ThreadState },
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
/// `Return` and `Yield` snapshot the frame and advance pc to `epc+4`.
/// `YieldRetry` snapshots the frame but leaves pc at `epc` so the
/// resumed thread re-executes the ecall.
///
/// **Mode/state gate**: cause=8 traps are by definition U-mode ecalls,
/// so `thread` should be a User thread in {Running, Suspended, Blocking}.
/// If a kthread or an Assigned/Ready/Exited thread is passed in, the
/// commit is skipped (Resume returned, no writes). Mirrors the gate in
/// [`trap::update_trap_frame`] — if the hart's `current` was retargeted
/// between trap entry and the apply call, we must not stamp a user epc
/// onto the wrong thread (the QEMU repro is knet ending up with
/// `pc=0x22000339c` from an orbit-loader `nc_yield`, then sret-ing into
/// user text in S-mode).
pub fn apply_syscall_outcome(
    outcome: SyscallOutcome,
    thread: &mut process::Thread,
    frame: &mut device::TrapFrame,
    epc: usize,
) -> ShimAction {
    use core::sync::atomic::Ordering;

    if !commit_allowed(thread) {
        // Caller-side panics in kmain's dispatch_syscall already catch
        // mode mismatches; this is the no-op fallback so a test or any
        // future caller that doesn't pre-check can't corrupt the wrong
        // thread. Fall through to Resume so the trap dispatcher unwinds
        // cleanly — the *user* thread that actually ecall'd is on
        // another hart and will retry on its next park cycle.
        return ShimAction::Resume;
    }

    match outcome {
        SyscallOutcome::Return { ret } => {
            frame.regs[10] = ret as usize;
            *thread.frame = *frame;
            thread.pc.store(epc + 4, Ordering::Release);
            ShimAction::Resume
        }
        SyscallOutcome::Return2 { ret0, ret1 } => {
            frame.regs[10] = ret0 as usize;
            frame.regs[11] = ret1 as usize;
            *thread.frame = *frame;
            thread.pc.store(epc + 4, Ordering::Release);
            ShimAction::Resume
        }
        SyscallOutcome::YieldRetry { state } => {
            // No reg writes — handler relies on a-reg snapshot
            // matching trap entry so the re-execute sees the
            // original args. pc stays at epc; the resumed thread
            // re-enters the ecall and the handler re-runs.
            *thread.frame = *frame;
            thread.pc.store(epc, Ordering::Release);
            ShimAction::Yield(state)
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

/// True iff `thread` is in a state where committing a U-mode-ecall
/// outcome is safe. Mirrors the trap-frame save gate in
/// [`trap::update_trap_frame`]:
///   * mode must be User — cause=8 is by definition a U-ecall, so the
///     thread that produced the trap is a User thread. A Supervisor
///     thread reaching here means `hart.current` was retargeted mid-trap.
///   * state must be Running / Suspended / Blocking — a freshly-`Assigned`
///     thread hasn't actually run yet, and Ready/Exited shouldn't be
///     observed as `current` in a syscall path.
fn commit_allowed(thread: &process::Thread) -> bool {
    use core::sync::atomic::Ordering;
    use riscv::register::sstatus::SPP;

    if thread.mode != SPP::User {
        return false;
    }
    let state = thread.state.load(Ordering::Acquire);
    state == process::ThreadState::Running as usize
        || state == process::ThreadState::Suspended as usize
        || state == process::ThreadState::Blocking as usize
}
