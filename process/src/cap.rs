//! Thread access capabilities.
//!
//! After the Phase-B seal, the resume-payload fields of [`Thread`]
//! (`frame`, `pc`, `state`, `fault_info`, `pending_*`) are `pub(crate)`
//! — unreachable from kmain/orbit-core directly. All access flows
//! through the typed handles here, so the bug-2/bug-4 invariants are
//! enforced by construction rather than by per-site discipline:
//!
//! - **[`RunningThread`]** — the hart's exclusive access to the thread
//!   it is currently running (domain B). Minted by `HartContext` from
//!   the hart's `current` pointer. May freely write its own frame (it
//!   is about to `sret`).
//! - **[`ManagerThread`]** — manager-side access to a *registry* thread,
//!   minted by [`crate::ThreadHandle::as_manager`] (needs a
//!   [`SchedGuard`]). Its only mutating route to the frame is
//!   [`ManagerThread::claim_parked`] → [`ParkedMut`], which returns
//!   `None` when the thread is Running/Assigned — so the manager can
//!   never scribble a live frame (**bug 2**).
//! - **[`ParkedMut`]** — proof the thread is parked + claimed; the only
//!   producer of [`Runnable`].
//! - **[`Runnable`]** — proof a thread's frame was marshaled under a
//!   won claim; the only key to `ReadyQueue::push` / `push_ready_notice`,
//!   so "make dispatchable" is welded to "I marshaled it" (**bug 4**).
//! - **[`ThreadView`]** — read-only snapshot (atomics + `Copy` fields);
//!   never exposes `&frame` (reading it would race the owner's writes).

use core::marker::PhantomData;
use core::sync::atomic::Ordering;

use device::TrapFrame;
use mmu::sv48::PhysAddr;
use orbit_abi::perms::Permissions;
use riscv::register::satp::Satp;
use riscv::register::sstatus::SPP;

use crate::{FaultInfo, SchedGuard, Thread, ThreadState};

/// Read-only snapshot of a [`Thread`]. Exposes atomics + `Copy` scalar
/// fields only — **never** the frame: a shared read of the non-atomic
/// `frame` would race the owning hart's frame writes.
///
/// Every accessor field-projects off the raw `*const Thread`; the view
/// **never** forms a whole-struct `&Thread`. That matters because a
/// shared `&Thread` would freeze the non-atomic credential fields, which
/// the manager field-writes on a possibly-Running sibling
/// ([`ManagerThread::set_uid_triplet`] / `set_permissions`) — so a
/// whole-struct retag would race that write. With field projection a
/// `ThreadView` is genuinely safe to hold while another hart runs the
/// thread, including while creds propagate to it (cred reads are
/// `Acquire`, paired with the manager's `Release` stores).
pub struct ThreadView<'a> {
    ptr: *const Thread,
    _life: PhantomData<&'a Thread>,
}

impl<'a> ThreadView<'a> {
    /// # Safety
    /// `ptr` must point at a live `Thread` for `'a`.
    pub(crate) unsafe fn new(ptr: *const Thread) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }

    /// Public read-only seam: mint a view directly from a raw `*const
    /// Thread`. The kmain hart helpers use it to snapshot this hart's (or a
    /// foreign hart's) `current` thread without hand-rolling
    /// `as_ref_unchecked()`. Read-only by construction — exposes atomics +
    /// `Copy` scalars only, never `&frame`, so it cannot race the owner's
    /// frame writes even for a thread live on another hart.
    ///
    /// # Safety
    /// `ptr` must point at a live `Thread` for `'a` (a registry thread, or a
    /// hart's `current`).
    #[inline]
    pub unsafe fn from_ptr(ptr: *const Thread) -> Self {
        // SAFETY: forwarded to the constructor invariant.
        unsafe { Self::new(ptr) }
    }

    // Every accessor **field-projects off the raw `*const Thread`**
    // (`(*self.ptr).field`) rather than going through a single
    // `&*self.ptr`. A `&Thread` would retag the *whole* struct and freeze
    // the non-atomic credential fields (`uid`/…/`permissions` snapshot),
    // which `ManagerThread::set_uid_triplet` / `set_permissions`
    // field-write on a possibly-Running sibling from another hart — a
    // shared whole-struct read-retag racing that write is UB. Projecting
    // to the exact field retags only that field's bytes (atomics are
    // `UnsafeCell`, so even those don't freeze), so a `ThreadView` is
    // sound to hold while another hart runs the thread *and* propagates
    // creds to it. There is intentionally no whole-`&Thread` accessor.

    pub fn state(&self) -> usize {
        unsafe { (*self.ptr).state.load(Ordering::Acquire) }
    }
    pub fn pid(&self) -> u16 {
        unsafe { (*self.ptr).pid }
    }
    pub fn tid(&self) -> u32 {
        unsafe { (*self.ptr).tid }
    }
    /// Per-thread user-stack slot index, if this thread owns one.
    pub fn slot(&self) -> Option<u16> {
        unsafe { (*self.ptr).slot }
    }
    pub fn cpu_ticks_total(&self) -> u64 {
        unsafe { (*self.ptr).cpu_ticks_total.load(Ordering::Relaxed) }
    }
    pub fn context_switches(&self) -> u64 {
        unsafe { (*self.ptr).context_switches.load(Ordering::Relaxed) }
    }
    pub fn syscall_count(&self) -> u64 {
        unsafe { (*self.ptr).syscall_count.load(Ordering::Relaxed) }
    }
    pub fn syscall_ticks(&self) -> u64 {
        unsafe { (*self.ptr).syscall_ticks.load(Ordering::Relaxed) }
    }
    /// `true` when the completion slot holds published, unconsumed
    /// return values (`pending_state == SIGNALED`).
    pub fn pending_signaled(&self) -> bool {
        unsafe { (*self.ptr).pending_state.load(Ordering::Acquire) == crate::PENDING_STATE_SIGNALED }
    }
    /// `true` when a wake reason has been OR'd into `wake_override` but
    /// not yet consumed. The sleep-heap drain reads this to decide
    /// whether to eager-promote a Suspended sleeper rather than file it
    /// back onto the heap.
    pub fn has_pending_wake(&self) -> bool {
        unsafe { (*self.ptr).wake_override.load(Ordering::Acquire) != 0 }
    }
    /// Privilege mode the thread runs in (`User` / `Supervisor`). Immutable
    /// for the thread's lifetime — safe to snapshot even for a thread live
    /// on another hart.
    pub fn mode(&self) -> SPP {
        unsafe { (*self.ptr).mode }
    }
    /// The thread's scheduled wake tick (`usize::MAX` = indefinite park).
    /// Diagnostic read for the stuck-thread watchdog, which scans threads
    /// cross-hart (including Running ones in its census dump) — hence the
    /// field is atomic and this is a field-projected `Relaxed` load. See
    /// [`Thread::wake_time`] for why ordering rides the `state` handshake
    /// rather than this load.
    pub fn wake_time(&self) -> usize {
        unsafe { (*self.ptr).wake_time.load(Ordering::Relaxed) }
    }
    /// Address-space root (`satp`). Immutable for the thread's lifetime.
    pub fn satp(&self) -> Satp {
        unsafe { (*self.ptr).satp }
    }
    /// Physical address of the root page table (derived from `satp`).
    pub fn root_table_addr(&self) -> PhysAddr {
        PhysAddr::from(self.satp())
    }
    /// Snapshot of the owning process's `stdout_redirect`. Immutable for
    /// the thread's lifetime.
    pub fn stdout_redirect(&self) -> Option<u16> {
        unsafe { (*self.ptr).stdout_redirect }
    }
    /// Immutable upper bound on which harts this thread may run on.
    pub fn allowed_affinity(&self) -> u64 {
        unsafe { (*self.ptr).allowed_affinity }
    }
    /// Current per-hart eligibility mask (atomic; the user may narrow it).
    pub fn affinity(&self) -> u64 {
        unsafe { (*self.ptr).affinity.load(Ordering::Relaxed) }
    }
    /// Resume program counter (the sealed `pc` atom). Diagnostic read for
    /// the dispatch / trap-mode-mismatch loggers, which previously formed a
    /// bare `&Thread` over a foreign hart's thread.
    pub fn pc(&self) -> usize {
        unsafe { (*self.ptr).pc.load(Ordering::Acquire) }
    }
    /// Bitmask consumed at the most recent `Suspended → Ready` transition.
    /// Diagnostic read (the fault / mode-mismatch loggers).
    pub fn last_wake_reason(&self) -> u64 {
        unsafe { (*self.ptr).last_wake_reason.load(Ordering::Acquire) }
    }

    // ─── credential / permission snapshot (Acquire) ─────────────────
    // The lock-free read side of the setuid/setgid/pledge propagation
    // writes (`ManagerThread::set_*`). Acquire pairs with those Release
    // stores so a reader either sees the whole new identity or the whole
    // old one — never a torn value.

    /// Raw effective syscall-class permission mask.
    pub fn perms_raw(&self) -> u64 {
        unsafe { (*self.ptr).perms.load(Ordering::Acquire) }
    }
    /// Role-id of the permission snapshot.
    pub fn role(&self) -> u32 {
        unsafe { (*self.ptr).perm_role.load(Ordering::Acquire) }
    }
    /// Reconstruct a [`Permissions`] from the thread snapshot for the
    /// perm-gate's `allows()` check + denial-event fields. Only `perms`
    /// and `role` are snapshotted on the thread (the gate's whole read
    /// set); `allowed_perms` / `_reserved` are zeroed here — they live on
    /// the owning process and are read under the lock, never off a thread.
    pub fn permissions_snapshot(&self) -> Permissions {
        Permissions {
            perms: self.perms_raw(),
            allowed_perms: 0,
            role: self.role(),
            _pad: 0,
            _reserved: [0; 2],
        }
    }
    pub fn uid(&self) -> u32 {
        unsafe { (*self.ptr).uid.load(Ordering::Acquire) }
    }
    pub fn euid(&self) -> u32 {
        unsafe { (*self.ptr).euid.load(Ordering::Acquire) }
    }
    pub fn suid(&self) -> u32 {
        unsafe { (*self.ptr).suid.load(Ordering::Acquire) }
    }
    pub fn gid(&self) -> u32 {
        unsafe { (*self.ptr).gid.load(Ordering::Acquire) }
    }
    pub fn egid(&self) -> u32 {
        unsafe { (*self.ptr).egid.load(Ordering::Acquire) }
    }
    pub fn sgid(&self) -> u32 {
        unsafe { (*self.ptr).sgid.load(Ordering::Acquire) }
    }
}

/// Proof that a thread's frame was marshaled under a won claim and the
/// thread is in `Ready`. The **only** key to `ReadyQueue::push` /
/// `push_ready_notice`. Non-`Clone`, constructor private to this crate —
/// unforgeable outside `process`, so "make dispatchable" cannot be
/// performed without "I marshaled the frame under a claim" (**bug 4**).
pub struct Runnable {
    ptr: *mut Thread,
}

impl Runnable {
    /// Reconstitute an enqueue token from a raw `*mut Thread`. The two
    /// legitimate sources, both of which uphold the bug-4 contract
    /// ("the frame is valid and this enqueue is sanctioned"):
    ///
    /// - **Inbox-drain reconstitution.** [`Self::into_raw`] stored a
    ///   previously-minted token as a bare ptr in a per-hart
    ///   `READY_INBOXES` slot; the manager drains it back and re-`push`es
    ///   it onto `ReadyQueue`. Round-tripping the same token.
    /// - **Fresh creation.** A just-built thread sits in `Ready` with a
    ///   fully-initialized frame (no claim to win — there was never a
    ///   park). The creating path mints the token to surface it to the
    ///   scheduler.
    ///
    /// # Safety
    /// `ptr` names a live registry `Thread` whose frame is valid and
    /// which is genuinely dispatchable (one of the two cases above).
    #[inline]
    pub unsafe fn from_raw(ptr: *mut Thread) -> Self {
        Self { ptr }
    }

    #[inline]
    pub fn tid(&self) -> u32 {
        // SAFETY: the ptr came from a live registry Thread under a guard.
        unsafe { (*self.ptr).tid }
    }

    /// Hart-affinity mask of the queued thread. The one field
    /// [`ReadyQueue::pop_for`] must read to filter its `VecDeque<Runnable>`
    /// by `hart_mask` without consuming the token — keeps the affinity
    /// scan working now that the queue stores `Runnable`s rather than bare
    /// `*mut Thread`. Reads the `pub` atomic through the ptr.
    #[inline]
    pub fn affinity(&self) -> u64 {
        // SAFETY: live registry Thread (see `from_raw`).
        unsafe { (*self.ptr).affinity.load(Ordering::Relaxed) }
    }

    /// Does this token target `thread`? Lets the ready queue scrub a
    /// reaped thread's pending entry by pointer identity without consuming
    /// the token ([`ReadyQueue::remove_thread`]).
    #[inline]
    pub fn points_to(&self, thread: *const Thread) -> bool {
        self.ptr as *const Thread == thread
    }

    /// Bridge to the raw `*mut Thread` the dispatch seam needs (the popped
    /// thread's ptr is stored into the target hart's `current` slot).
    /// Consuming `self` at the pop boundary keeps the "only a `Runnable`
    /// is dispatchable" guarantee alive through the queue; the raw pointer
    /// is only ever re-read by the manager under the lock.
    #[inline]
    pub fn into_raw(self) -> *mut Thread {
        self.ptr
    }
}

/// The hart's exclusive capability over the thread it is currently
/// running. Minted by `HartContext::running()`. Because the running hart
/// owns this thread's execution, frame/pc writes here are uncontended.
pub struct RunningThread<'a> {
    ptr: *mut Thread,
    _life: PhantomData<&'a mut Thread>,
}

impl<'a> RunningThread<'a> {
    /// # Safety
    /// The caller must be the hart that owns `ptr` as its `current`
    /// thread (it is executing it), so no other hart holds a reference.
    /// This is the cross-crate trap-entry seam — the one `pub unsafe`
    /// capability constructor.
    pub unsafe fn from_ptr(ptr: *mut Thread) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }

    /// Read-only snapshot of the running thread. Field-projected (atomics +
    /// `Copy` scalars) — there is intentionally **no** whole-struct
    /// `&Thread` accessor on the cap: reads go through this view, frame/
    /// state writes through the `commit_*` / `resume_*` verbs.
    #[inline]
    pub fn view(&self) -> ThreadView<'_> {
        // SAFETY: own-hart exclusive; read-only snapshot.
        unsafe { ThreadView::new(self.ptr) }
    }

    /// Commit a single-value syscall return: write `ret` into the live
    /// frame's `a0`, snapshot the frame into the thread, advance `pc`.
    ///
    /// Writes are field-projected off the raw pointer — `*(*self.ptr).frame`
    /// retags only the separate `TrapFrame` allocation, and `pc` is a narrow
    /// atomic store. Forming `&mut Thread` here would retag the *whole*
    /// struct, and that write-retag races a manager peeking this Running
    /// thread's atomics from another hart (Phase-E miri finding). Same
    /// reason for every frame writer below.
    pub fn commit_return(&mut self, ret: isize, epc: usize, frame: &mut TrapFrame) {
        frame.regs[10] = ret as usize;
        unsafe {
            *(*self.ptr).frame = *frame;
            (*self.ptr).pc.store(epc + 4, Ordering::Release);
        }
    }

    /// Two-value return (`a0`, `a1`).
    pub fn commit_return2(&mut self, ret0: isize, ret1: isize, epc: usize, frame: &mut TrapFrame) {
        frame.regs[10] = ret0 as usize;
        frame.regs[11] = ret1 as usize;
        unsafe {
            *(*self.ptr).frame = *frame;
            (*self.ptr).pc.store(epc + 4, Ordering::Release);
        }
    }

    /// Yield: optionally write `a0`, snapshot the frame, advance `pc`
    /// (the syscall completed; the thread parks but resumes past it).
    pub fn commit_yield(&mut self, ret: Option<isize>, epc: usize, frame: &mut TrapFrame) {
        if let Some(r) = ret {
            frame.regs[10] = r as usize;
        }
        unsafe {
            *(*self.ptr).frame = *frame;
            (*self.ptr).pc.store(epc + 4, Ordering::Release);
        }
    }

    /// Yield-retry: snapshot the frame but leave `pc` at `epc` so the
    /// thread re-executes the `ecall` on resume (no reg writes — the
    /// a-reg args must be preserved).
    pub fn commit_yield_retry(&mut self, epc: usize, frame: &mut TrapFrame) {
        unsafe {
            *(*self.ptr).frame = *frame;
            (*self.ptr).pc.store(epc, Ordering::Release);
        }
    }

    /// Stamp the thread's scheduled wake tick. Single-writer (the parking
    /// thread is the sole writer of its own `wake_time`); set by `apply`
    /// when committing a deadline-bearing park (`SleepUntil` / the doorbell
    /// retries) so the wake-time choice travels atomically with the park
    /// state rather than being set separately in the body.
    pub fn set_wake_time(&mut self, wake_time: usize) {
        // Field-projected atomic store — no `&mut Thread` retag (see
        // `commit_return`). `Relaxed`: ordering for the sleep-heap path
        // rides the `state` Release that follows; atomicity is for the
        // watchdog's lock-free cross-hart read (see `Thread::wake_time`).
        unsafe { (*self.ptr).wake_time.store(wake_time, Ordering::Relaxed) };
    }

    /// Trap-entry snapshot: copy the live trap frame into the thread and
    /// stamp the resume `pc`. Own-hart (the hart that trapped saving its
    /// own current thread), so the frame write is uncontended.
    pub fn save_trap_frame(&mut self, frame: &TrapFrame, pc: usize) {
        unsafe {
            *(*self.ptr).frame = *frame;
            (*self.ptr).pc.store(pc, Ordering::Release);
        }
    }

    /// Record a fault on the way to killing the thread.
    pub fn set_fault(&mut self, info: FaultInfo) {
        unsafe { (*self.ptr).fault_info = Some(info) };
    }

    /// Read a saved trap-frame register. Own-hart exclusive, so reading
    /// the frame is uncontended. (Granular companion to the `commit_*`
    /// verbs — used by trap-contract tests to assert post-commit state.)
    #[inline]
    pub fn frame_reg(&self, i: usize) -> usize {
        let t = unsafe { &*self.ptr };
        t.frame.regs[i]
    }

    /// Write a saved trap-frame register. Own-hart exclusive; the frame
    /// write is uncontended (the hart is about to `sret`). Used to stage
    /// fixtures in trap-contract tests; production marshaling goes
    /// through the `commit_*` / `resume_*` verbs.
    #[inline]
    pub fn set_frame_reg(&mut self, i: usize, v: usize) {
        // Project through the `frame` reference (separate allocation) — no
        // `&mut Thread` retag (see `commit_return`).
        unsafe { (*self.ptr).frame.regs[i] = v };
    }

    /// Set the resume `pc` directly. Own-hart exclusive. Production stamps
    /// `pc` through `save_trap_frame` / the `commit_*` verbs; this is the
    /// granular setter for test fixtures.
    #[inline]
    pub fn set_pc(&mut self, pc: usize) {
        let t = unsafe { &*self.ptr };
        t.pc.store(pc, Ordering::Release);
    }

    /// Reset/stamp the preemption tick counter. Field-projected (no
    /// `&mut Thread` retag): `ticks` is a `pub` non-atomic byte owned by
    /// the own-hart scheduler. Lets `kthread_park` zero ticks at park time
    /// without materializing a whole-struct `&mut` (the Phase-E retag).
    #[inline]
    pub fn set_ticks(&mut self, ticks: u8) {
        unsafe { (*self.ptr).ticks = ticks };
    }

    /// Set the thread's hart-affinity mask (own-hart; the `set_affinity`
    /// syscall, where a thread re-pins itself). Field-projected atomic
    /// `Release` store — `affinity` is a `pub` atom the scheduler reads
    /// cross-hart via [`Runnable::affinity`]; the store needs only `&self`
    /// but must not form a whole-struct `&Thread` (cred-freeze).
    #[inline]
    pub fn set_affinity_mask(&self, mask: u64) {
        unsafe {
            (*self.ptr).affinity.store(mask, Ordering::Release);
        }
    }

    /// Per-thread syscall accounting bump (own-hart). Field-projects the
    /// `pub` accounting atoms off the raw ptr — never `&Thread`, which
    /// would freeze the cred fields a sibling may be propagating to this
    /// still-Running thread. `&self` because the counters are atomics
    /// (interior-mutable) and foreign-hart readable (`query_stats`), but
    /// only this owning hart ever writes them.
    #[inline]
    pub fn account_syscall(&self, elapsed: u64) {
        unsafe {
            (*self.ptr).syscall_count.fetch_add(1, Ordering::Relaxed);
            (*self.ptr).syscall_ticks.fetch_add(elapsed, Ordering::Relaxed);
        }
    }

    /// Take the parked thread's completion handle, if any. Projects to the
    /// `handle` field (a `pub` `Option`) — `&mut` of that sub-place retags
    /// only the field, never the whole `Thread`. The signaler that won the
    /// waiter-swap claim owns this thread exclusively, so the take is
    /// uncontended. Used by `wake_blocked_inline`.
    #[inline]
    pub fn take_handle(&mut self) -> Option<crate::CompletionHandle> {
        unsafe { (*self.ptr).handle.take() }
    }

    /// Yield the enqueue token for a thread the caller has already set
    /// to `Ready` (the preemption / yield-to-scheduler path — no new
    /// return values, the existing frame is resumed as-is). Own-hart, so
    /// no claim race.
    pub fn into_runnable(self) -> Runnable {
        // Bug-4 self-check: only a `Ready` thread is dispatchable. The sole
        // production caller sets `Ready` via the own-hart
        // `transition_to(Ready)` cooperative yield immediately before this;
        // a future caller minting a token for a parked/running thread would
        // smuggle a non-dispatchable thread onto the ready queue. Panic
        // (always-on, release included) rather than enqueue it.
        let st = unsafe { (*self.ptr).state.load(Ordering::Acquire) };
        assert!(
            st == ThreadState::Ready as usize,
            "into_runnable on non-Ready thread (state {st}, tid {})",
            unsafe { (*self.ptr).tid },
        );
        Runnable { ptr: self.ptr }
    }

    /// Marshal `vals` into `a0..`, transition `Ready`, and yield the
    /// enqueue token. For exclusive-access wake paths that carry their
    /// own return values (the `CompletionHandle` signaler, which won the
    /// waiter-swap claim before minting this cap).
    pub fn resume_with(&mut self, vals: &[isize]) -> Runnable {
        // 4 = the syscall return registers a0..a3 (`frame.regs[10..14]`);
        // more would spill past the return convention. >4 is a caller
        // bug — loud in test/miri, clamped in release.
        debug_assert!(vals.len() <= 4, "resume_with: >4 return values");
        let n = vals.len().min(4);
        unsafe {
            for (i, &v) in vals.iter().enumerate().take(n) {
                (*self.ptr).frame.regs[10 + i] = v as usize;
            }
            // parked(Blocking)→Ready via the checked door (the generic
            // `transition_to` forbids parked→Ready; bug-4 gate).
            (*self.ptr).promote_ready_from_parked();
        }
        Runnable { ptr: self.ptr }
    }

    /// Claim this thread's own published pending results at park-commit
    /// time (the parker's post-park re-check). Returns `Some(Runnable)`
    /// if this hart won the take-CAS (marshaled the rets, → Ready), or
    /// `None` if the manager's drain won it (it owns the resume). Same
    /// at-most-once gate as the manager path.
    pub fn try_claim_own_pending(&mut self) -> Option<Runnable> {
        let mut rets = [0i64; 4];
        // `take_pending_results` is `&self` (atomic take-CAS) — a read
        // retag, not a `&mut Thread`. Frame/state writes are projected.
        let n = unsafe { (*self.ptr).take_pending_results(&mut rets) }?;
        unsafe {
            for i in 0..n {
                (*self.ptr).frame.regs[10 + i] = rets[i] as usize;
            }
            // parked(Blocking)→Ready via the checked door (bug-4 gate).
            (*self.ptr).promote_ready_from_parked();
        }
        Some(Runnable { ptr: self.ptr })
    }
}

/// Manager-side capability over a registry thread, minted by
/// [`crate::ThreadHandle::as_manager`] (requires a [`SchedGuard`], whose
/// lifetime `'g` bounds this handle to the critical section).
pub struct ManagerThread<'g> {
    ptr: *mut Thread,
    _guard: PhantomData<&'g SchedGuard>,
}

impl<'g> ManagerThread<'g> {
    /// # Safety
    /// `ptr` is a live registry Thread and the caller holds the
    /// `SchedGuard` bounding `'g`. Private — minted only via
    /// `ThreadHandle::as_manager`.
    pub(crate) unsafe fn new(ptr: *mut Thread) -> Self {
        Self {
            ptr,
            _guard: PhantomData,
        }
    }

    /// Mint a manager capability from a raw registry pointer the caller
    /// is holding directly (rather than via a [`crate::ThreadHandle`]) —
    /// the sleep heap and other manager-owned structures store bare
    /// `*mut Thread`. The `&SchedGuard` is the lock-held proof and bounds
    /// `'g`, exactly as in [`crate::ThreadHandle::as_manager`].
    ///
    /// # Safety
    /// `ptr` names a live registry `Thread` (freed only by the manager's
    /// own cleanup, in the same critical section), and the caller holds
    /// the guard bounding `'g`.
    pub unsafe fn from_raw(ptr: *mut Thread, _guard: &'g SchedGuard) -> Self {
        Self {
            ptr,
            _guard: PhantomData,
        }
    }

    #[inline]
    pub fn view(&self) -> ThreadView<'_> {
        // SAFETY: registry ptr, live under the guard.
        unsafe { ThreadView::new(self.ptr) }
    }

    /// Read a saved trap-frame register of a registry thread — **`None`
    /// when the thread is `Running`/`Assigned`**, because reading a live
    /// frame would race the owning hart's writes (the read-side mirror of
    /// the bug-2 frame-write gate). For manager-side inspection of a
    /// *quiescent* thread's frame: a just-resumed `Ready` thread, a parked
    /// waiter, or exit diagnostics. Field-projected off the frame
    /// reference — no whole-struct `&Thread` retag.
    ///
    /// The reaper, which has a stronger static proof (it holds an
    /// [`ExitedThread`] from [`Self::claim_exited`]), reads frame regs
    /// unconditionally via [`ExitedThread::frame_reg`] instead.
    pub fn frame_reg(&self, i: usize) -> Option<usize> {
        let st = unsafe { (*self.ptr).state.load(Ordering::Acquire) };
        if st == ThreadState::Running as usize || st == ThreadState::Assigned as usize {
            return None;
        }
        Some(unsafe { (*self.ptr).frame.regs[i] })
    }

    /// OR a wake reason into `wake_override` (always safe — atomic hint;
    /// does not make the thread dispatchable on its own).
    pub fn note_wake(&self, reason: u64) {
        let t = unsafe { &*self.ptr };
        t.wake_override.fetch_or(reason, Ordering::Release);
    }

    /// Claim the thread for resume **iff** it is parked
    /// (`Blocking`/`Suspended`). Returns `None` when it is
    /// `Running`/`Assigned` — the manager must not touch a live thread's
    /// frame (**bug 2**). Consumes `self`: a claim is exclusive.
    pub fn claim_parked(self) -> Option<ParkedMut<'g>> {
        let st = unsafe { (*self.ptr).state.load(Ordering::Acquire) };
        if st == ThreadState::Blocking as usize || st == ThreadState::Suspended as usize {
            Some(ParkedMut {
                ptr: self.ptr,
                _guard: PhantomData,
            })
        }
        else {
            None
        }
    }

    /// Publish return values into the thread's completion slot (does not
    /// touch the frame or make it dispatchable — the parker's re-check
    /// or a later wake marshals + enqueues). Manager-only; the guard is
    /// the proof.
    pub fn publish_results(&self, vals: &[isize]) {
        let t = unsafe { &*self.ptr };
        t.publish_results(vals);
    }

    /// Mint an enqueue token for a thread that is **already** `Ready`
    /// with a valid frame — a freshly-created thread, or one whose frame
    /// was set elsewhere. No marshaling (there's nothing to marshal);
    /// just the proof needed to call `ReadyQueue::push`. `None` if the
    /// thread isn't `Ready` (caller shouldn't be enqueuing it).
    pub fn claim_ready(self) -> Option<Runnable> {
        let st = unsafe { (*self.ptr).state.load(Ordering::Acquire) };
        if st == ThreadState::Ready as usize {
            Some(Runnable { ptr: self.ptr })
        }
        else {
            None
        }
    }

    /// Replace the thread's permission snapshot (pledge propagation).
    ///
    /// **Atomic `Release` stores.** Pledge/setuid propagate to *sibling*
    /// threads, which may be Running on another hart and reading the
    /// snapshot lock-free in the perm-gate ([`ThreadView::perms_raw`]).
    /// Atomic stores (vs a non-atomic field write under a whole-struct
    /// retag) are what make that concurrent read race-free; `Release`
    /// pairs with the reader's `Acquire`. Only `perms` + `role` are
    /// snapshotted on the thread (see [`Thread::perms`]).
    pub fn set_permissions(&self, perms: Permissions) {
        unsafe {
            (*self.ptr).perms.store(perms.perms, Ordering::Release);
            (*self.ptr).perm_role.store(perms.role, Ordering::Release);
        }
    }

    /// Stamp the uid triplet (setuid propagation across sibling threads).
    /// Atomic `Release` stores — see [`Self::set_permissions`].
    pub fn set_uid_triplet(&self, uid: u32, euid: u32, suid: u32) {
        unsafe {
            (*self.ptr).uid.store(uid, Ordering::Release);
            (*self.ptr).euid.store(euid, Ordering::Release);
            (*self.ptr).suid.store(suid, Ordering::Release);
        }
    }

    /// Stamp the gid triplet (setgid propagation). Atomic `Release` stores.
    pub fn set_gid_triplet(&self, gid: u32, egid: u32, sgid: u32) {
        unsafe {
            (*self.ptr).gid.store(gid, Ordering::Release);
            (*self.ptr).egid.store(egid, Ordering::Release);
            (*self.ptr).sgid.store(sgid, Ordering::Release);
        }
    }

    /// Claim the thread for reaping **iff** it is `Exited`. Returns `None`
    /// for any live state. The sealed-field reaper reads
    /// ([`ExitedThread::fault_info`] / [`ExitedThread::frame_reg`]) are
    /// reachable only through the returned token, so the "no hart runs this
    /// thread" obligation those reads depend on is enforced by
    /// construction — not by a `state == Exited` check the caller must
    /// remember to perform. Consumes `self`: an exit claim is exclusive.
    pub fn claim_exited(self) -> Option<ExitedThread<'g>> {
        let st = unsafe { (*self.ptr).state.load(Ordering::Acquire) };
        if st == ThreadState::Exited as usize {
            Some(ExitedThread {
                ptr: self.ptr,
                _guard: PhantomData,
            })
        }
        else {
            None
        }
    }

    /// Mark the thread `Exited` (the `exit_group` kill path). Unlike the
    /// resume verbs this is *not* claim-gated: the target may be
    /// `Running` on another hart — the store is the kill signal that the
    /// running hart observes on its next `check_context_and_switch`. It
    /// only touches the `state` atom (never the frame, never an enqueue),
    /// so neither bug-2 nor bug-4 is in scope.
    pub fn mark_exited(&self) {
        let t = unsafe { &*self.ptr };
        t.state
            .store(ThreadState::Exited as usize, Ordering::Release);
    }
}

/// A thread the manager has confirmed `Exited`. Minted only by the
/// state-gated [`ManagerThread::claim_exited`], so its sealed-field reads
/// (`fault_info` / `frame_reg`) carry a construction-enforced proof that
/// no hart runs the thread — the reaper can read the resume payload for
/// diagnostics without a separate `state == Exited` check or a
/// whole-struct `&Thread` retag over a possibly-live thread.
pub struct ExitedThread<'g> {
    ptr: *mut Thread,
    _guard: PhantomData<&'g SchedGuard>,
}

impl<'g> ExitedThread<'g> {
    /// Read-only snapshot (e.g. `tid`/`pid`/`slot` for the reap log).
    #[inline]
    pub fn view(&self) -> ThreadView<'_> {
        // SAFETY: registry ptr, live under the guard; Exited-gated.
        unsafe { ThreadView::new(self.ptr) }
    }

    /// Recorded fault, if the thread died on one (`None` ⇒ clean exit).
    /// Field-projected read of a sealed field — sound because the claim
    /// proved the thread is `Exited`, so no hart writes `fault_info`.
    #[inline]
    pub fn fault_info(&self) -> Option<FaultInfo> {
        unsafe { (*self.ptr).fault_info }
    }

    /// Read a saved trap-frame register (`a1` = exit status, `sp`, `ra`
    /// for the reaper's diagnostics). Field-projected off the frame
    /// reference; see [`Self::fault_info`] for the not-running proof.
    #[inline]
    pub fn frame_reg(&self, i: usize) -> usize {
        unsafe { (*self.ptr).frame.regs[i] }
    }
}

/// A parked, claimed thread. The **only** producer of [`Runnable`]: the
/// `write_*` verbs marshal the frame (a sealed-field write, legitimate
/// here because `claim_parked` proved the thread is not Running) and
/// transition it to `Ready`, returning the enqueue token.
pub struct ParkedMut<'g> {
    ptr: *mut Thread,
    _guard: PhantomData<&'g SchedGuard>,
}

impl<'g> ParkedMut<'g> {
    /// Read access (e.g. to inspect pending state before marshaling).
    #[inline]
    pub fn view(&self) -> ThreadView<'_> {
        unsafe { ThreadView::new(self.ptr) }
    }

    /// Marshal `vals` into `a0..` , reset the completion slot, → `Ready`,
    /// and yield the enqueue token. Used by the direct-resume path
    /// (`resume_thread_with_values`).
    pub fn write_rets(self, vals: &[isize]) -> Runnable {
        // 4 = a0..a3 return registers; >4 is a caller bug (see resume_with).
        debug_assert!(vals.len() <= 4, "write_rets: >4 return values");
        let n = vals.len().min(4);
        // Field-project off the raw ptr — NEVER `let t = &mut *self.ptr`. A
        // whole-struct `&mut Thread` retag races the parker's concurrent
        // lock-free `try_claim_own_pending` take-CAS on the same allocation
        // (Phase-E). `frame` is a separate `&'static mut TrapFrame`, so the
        // marshal retags only it; the rest are atomics / `&self` calls.
        unsafe {
            for (i, &v) in vals.iter().enumerate().take(n) {
                (*self.ptr).frame.regs[10 + i] = v as usize;
            }
            (*self.ptr).reset_pending();
            (*self.ptr).promote_ready_from_parked();
        }
        Runnable { ptr: self.ptr }
    }

    /// The `set_wake_reason_where` resume path. Atomically claims the
    /// **published** pending results (the take-CAS — the at-most-once
    /// gate), marshals them, records the consumed `wake_override` into
    /// `last_wake_reason`, → `Ready`, and yields the token. `None` when
    /// the take-CAS lost (the parker's own re-check won — it owns the
    /// resume). This couples the enqueue token to winning the *same* CAS
    /// that owns the frame write (**bug 4**).
    pub fn resume_published(self) -> Option<Runnable> {
        let mut rets = [0i64; 4];
        // Field-project — NEVER `&mut *self.ptr` (see `write_rets`). The
        // take-CAS arbitrates which side resumes; it does NOT make a
        // whole-struct retag safe, since the retag is asserted at formation
        // before the CAS and races the parker's concurrent atomic access.
        let n = unsafe { (*self.ptr).take_pending_results(&mut rets) }?;
        unsafe {
            for i in 0..n {
                (*self.ptr).frame.regs[10 + i] = rets[i] as usize;
            }
            let pending = (*self.ptr).wake_override.swap(0, Ordering::AcqRel);
            (*self.ptr).last_wake_reason.store(pending, Ordering::Release);
            (*self.ptr).promote_ready_from_parked();
        }
        Some(Runnable { ptr: self.ptr })
    }

    /// Promote with no rets: record the consumed `wake_override` into
    /// `last_wake_reason`, → `Ready`, yield the token. For wake paths
    /// that re-run rather than return a value (no marshaling).
    pub fn promote_wake(self) -> Runnable {
        // Field-project — NEVER `&mut *self.ptr` (see `write_rets`).
        unsafe {
            let pending = (*self.ptr).wake_override.swap(0, Ordering::AcqRel);
            (*self.ptr).last_wake_reason.store(pending, Ordering::Release);
            (*self.ptr).promote_ready_from_parked();
        }
        Runnable { ptr: self.ptr }
    }
}
