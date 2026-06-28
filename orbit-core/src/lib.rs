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

/// Upper bound on hart count for static per-hart array sizing
/// (TLB-shootdown rings, the manager's `READY_INBOXES`). The real CPU
/// count is a runtime value (`SysInfo::cpu_count`, ≤ this). Homed here —
/// the lowest crate both kmain (`shootdown`) and the `manager` crate
/// (`READY_INBOXES`) can reach — so per-hart arrays size in either.
pub const MAX_HARTS: usize = 8;

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
    ChInspectReq, ChdirReq, CloseHandleReq, CreateProcessExReq, CreateProcessReq,
    CreateProcessV2Req, CreateThreadReq, EventFdCreateReq, FbSurfaceCreateReq, FbSurfaceDestroyReq,
    FsFstatReq, FsOpenReq, FsReadReq, FsReaddirReq, FsSeekReq, FsStatReq, FutexWaitReq,
    FutexWakeReq, GetCwdReq, GetGroupsReq, GetLoginReq, MAX_FS_PATH_LEN, MAX_LOGIN_NAME, MemMapReq,
    NetChannelCreationReq, PendingWork, PledgeReq, SetGidReq, SetGroupsReq, SetLoginReq, SetUidReq,
    SpawnContext, WaitPidReq, WakeTidReq,
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
    /// `-EAGAIN`.
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

    /// Park `tid` on `pid`'s stdin slot. Returns `false` if a reader
    /// was already parked (caller emits EBUSY) or `pid` isn't
    /// registered. Producer's `push_byte` swaps the slot back to
    /// "empty" and returns the parked tid; the trap-context caller
    /// (kmain's `input::dispatch`) then issues a
    /// `WAKE_QUEUE.push(WakeEvent::InputTid(tid))` so the manager
    /// resumes the parker.
    fn park_stdin_reader(&mut self, pid: u16, tid: u32) -> bool;

    /// Cancel a park on `pid`'s stdin slot. Used by the read_stdin
    /// re-check path when bytes arrive between try_drain and park.
    /// Returns `true` if there was a parked reader to cancel.
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
/// **Park-shape closure (phase C).** The old open `state: ThreadState`
/// field on `Yield`/`YieldRetry` let a body pick a park state that
/// mismatched its wake mechanism — the read-stdin regression was a
/// doorbell parker that chose `Blocking` (whose only wake is
/// publish-then-push) instead of `Suspended`. Each variant below fixes
/// the `(park state, pc/ret commit shape, wake_time policy)` triple as
/// one atomic choice, so that class of bug is unrepresentable: there is
/// no "Blocking + doorbell" variant to construct. `apply` is the sole
/// place those three are realized, and it stamps `wake_time` itself so a
/// body can't set a deadline inconsistent with its park state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallOutcome {
    /// Write `ret` into `regs[10]`, commit the frame snapshot + pc bump
    /// (so the thread resumes past the ecall with `ret` visible), and
    /// return to the trap dispatcher without yielding. Used for
    /// synchronous error returns from handlers that don't block.
    Return { ret: isize },

    /// Like `Return`, but writes a pair into `regs[10]` and `regs[11]`.
    /// For synchronous syscalls returning two `isize`s — modelled on
    /// Windows's `GetProcessAffinityMask` shape (current + allowed in
    /// one trap). The async two-return path (`create_netch`) goes via
    /// a park + `signal_n`, not this variant.
    Return2 { ret0: isize, ret1: isize },

    /// Syscall completed: write `ret`, advance pc, and yield `Ready`
    /// (back through the scheduler so siblings get a turn). The
    /// [`syscall::ready`]-shaped fast path — serial_print / console_write
    /// / most non-blocking syscalls.
    DoneReschedule { ret: isize },

    /// Sleep until the absolute `deadline` tick. Resume returns 0 with pc
    /// advanced past the ecall; `apply` stamps `wake_time = deadline`.
    /// Park state `Suspended`. `ms_sleep`.
    SleepUntil { deadline: usize },

    /// Park `Blocking` awaiting a manager publish-then-push: the body has
    /// already queued a `PendingWork`; the manager publishes return
    /// values into the on-thread completion slot and the wake drain
    /// marshals them on resume (pc advanced, `regs[10]` filled at unblock
    /// time). No `wake_time`. Every converted-from-sync `*_req`.
    ParkForPublish,

    /// Doorbell park, no deadline: park `Suspended` and *retain* pc at the
    /// ecall so the thread re-executes (a-regs preserved) on wake; the
    /// only wake path is `wake_override` (an input doorbell). `apply`
    /// stamps `wake_time = usize::MAX` so the sleep heap never fires.
    /// `read_stdin`, `read_key_event` (indefinite).
    RetryOnDoorbell,

    /// Doorbell park with a deadline: like [`Self::RetryOnDoorbell`] but
    /// `apply` stamps `wake_time = deadline`, so a timer wake re-executes
    /// the ecall (which observes the elapsed timeout and returns).
    /// `read_key_event` (timeout).
    RetryUntilDeadline { deadline: usize },
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
/// `pc=0x22000339c` from an orbit-loader `ch_yield`, then sret-ing into
/// user text in S-mode).
pub fn apply_syscall_outcome(
    outcome: SyscallOutcome,
    running: &mut process::RunningThread,
    frame: &mut device::TrapFrame,
    epc: usize,
) -> ShimAction {
    if !commit_allowed(running.view()) {
        // Caller-side panics in kmain's dispatch_syscall already catch
        // mode mismatches; this is the no-op fallback so a test or any
        // future caller that doesn't pre-check can't corrupt the wrong
        // thread. Fall through to Resume so the trap dispatcher unwinds
        // cleanly — the *user* thread that actually ecall'd is on
        // another hart and will retry on its next park cycle.
        return ShimAction::Resume;
    }

    // The frame/pc writes are sealed to the own-hart capability — the
    // hart owns the thread it's committing for, so the writes are
    // uncontended (the bug-2 invariant, now by construction). `apply` is
    // also the sole place each park variant's `(state, commit, wake_time)`
    // triple is realized (the phase-C park-shape closure).
    use process::ThreadState::{Blocking, Ready, Suspended};
    match outcome {
        SyscallOutcome::Return { ret } => {
            running.commit_return(ret, epc, frame);
            ShimAction::Resume
        }
        SyscallOutcome::Return2 { ret0, ret1 } => {
            running.commit_return2(ret0, ret1, epc, frame);
            ShimAction::Resume
        }
        SyscallOutcome::DoneReschedule { ret } => {
            running.commit_yield(Some(ret), epc, frame);
            ShimAction::Yield(Ready)
        }
        SyscallOutcome::SleepUntil { deadline } => {
            running.commit_yield(Some(0), epc, frame);
            running.set_wake_time(deadline);
            ShimAction::Yield(Suspended)
        }
        SyscallOutcome::ParkForPublish => {
            // No reg write — the manager fills `regs[10]` from the
            // completion slot at unblock time. pc advances so the resumed
            // thread continues past the ecall.
            running.commit_yield(None, epc, frame);
            ShimAction::Yield(Blocking)
        }
        SyscallOutcome::RetryOnDoorbell => {
            // No reg writes — the a-reg snapshot matches trap entry so the
            // re-executed ecall sees identical args. pc stays at epc.
            running.commit_yield_retry(epc, frame);
            running.set_wake_time(usize::MAX);
            ShimAction::Yield(Suspended)
        }
        SyscallOutcome::RetryUntilDeadline { deadline } => {
            running.commit_yield_retry(epc, frame);
            running.set_wake_time(deadline);
            ShimAction::Yield(Suspended)
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
fn commit_allowed(thread: process::ThreadView<'_>) -> bool {
    use riscv::register::sstatus::SPP;

    if thread.mode() != SPP::User {
        return false;
    }
    let state = thread.state();
    state == process::ThreadState::Running as usize
        || state == process::ThreadState::Suspended as usize
        || state == process::ThreadState::Blocking as usize
}
