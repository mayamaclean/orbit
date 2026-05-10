#![no_std]

extern crate alloc;

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicI64, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use alloc::collections::{btree_map::BTreeMap, btree_set::BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use device::{Stack, TrapFrame};
use mmu::sv48::PhysAddr;
use orbit_abi::perms::{Permissions, PermsRequest};
use orbit_abi::roles::ChildPerms;
use riscv::register::{satp::Satp, sstatus::SPP};
use smoltcp::iface::SocketHandle;

pub mod completion;
pub mod key_events;
pub mod spsc;
pub mod stdin;
pub use completion::{AckCounter, CompletionHandle};
pub use key_events::ProcessKeyEvents;
pub use spsc::SpscQueue;
pub use stdin::ProcessStdin;

/// Reason-flag bits for [`Thread::wake_override`]. Producers `fetch_or`
/// a flag to mark "wake this thread now and report this cause." The
/// scheduler atomically `swap(0)` the union of pending bits at dispatch
/// time and forwards them to userspace so a woken thread can tell why
/// it was scheduled.
///
/// Layout is intentionally a bitmask (not an enum) so multiple wake
/// reasons that arrive between dispatches collapse into a single wake
/// with all the causes the thread needs to know about.
pub mod wake_reason {
    /// Manager-driven generic tickle. Today: WAKE_QUEUE drains for
    /// "you should re-check your park condition." Future: housekeeping
    /// signals like "your process group changed."
    pub const TICKLE: u64 = 1 << 0;
    /// Network I/O is ready: the kernel staged a fresh rx slice for
    /// this thread's NetCh, or drained tx and there's room. Set by
    /// `update_tcp` when its [`UpdateOutcome`] reports user-visible
    /// ring progress.
    pub const NET_IO: u64 = 1 << 1;
    /// External device interrupt the thread had asked to wait on
    /// (future use — e.g. file descriptors backed by virtio events).
    pub const DEVICE_IO: u64 = 1 << 2;
    /// POSIX-style signal delivery (future use).
    pub const SIGNAL: u64 = 1 << 3;
    /// Structured key event arrived in the thread's process's
    /// `ProcessKeyEvents` ring. Set by `kernel::input::dispatch` after
    /// each `push_event`, targeting the tid that called
    /// `read_key_event` and parked. Suspended thread is eagerly
    /// promoted to Ready so the syscall re-runs and drains.
    pub const INPUT_IO: u64 = 1 << 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum ThreadState {
    Ready = 0,
    Blocking = 1,
    Assigned = 2,
    Running = 3,
    Exited = 4,
    Suspended = 5,
}

#[derive(Debug, Clone, Copy)]
pub struct FaultInfo {
    pub cause: usize,
    pub epc: usize,
    pub stval: usize,
}

/// Per-process thread index allocator. A thread's `slot` keys all of its
/// per-thread user mappings (stack, guard, trap frame, TLS) so teardown is a
/// single pass over [`Process::maps`].
#[derive(Debug, Clone, Copy)]
pub struct SlotAlloc {
    bits: [u64; Self::WORDS],
}

impl SlotAlloc {
    pub const CAPACITY: u16 = 256;
    const WORDS: usize = (Self::CAPACITY as usize) / 64;

    pub const fn new() -> Self {
        Self {
            bits: [0; Self::WORDS],
        }
    }

    pub fn alloc(&mut self) -> Option<u16> {
        for (i, word) in self.bits.iter_mut().enumerate() {
            if *word != u64::MAX {
                let bit = word.trailing_ones() as u16;
                *word |= 1u64 << bit;
                return Some(i as u16 * 64 + bit);
            }
        }
        None
    }

    pub fn free(&mut self, slot: u16) {
        let word = (slot / 64) as usize;
        let bit = slot % 64;
        self.bits[word] &= !(1u64 << bit);
    }

    pub fn is_allocated(&self, slot: u16) -> bool {
        let word = (slot / 64) as usize;
        let bit = slot % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    pub fn len(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }
}

/// Kernel frame pool marker. Type-level tag used by [`Frame<P>`] to track
/// which allocator a physical page came from, and whether it's reachable
/// from kernel code via KDMAP. The trait has no methods — it exists just
/// to gate which [`Frame<P>`] conversions are legal.
pub trait Pool: Copy + fmt::Debug + 'static {
    /// For runtime diagnostics where a concrete pool is needed (logs,
    /// debug prints). Not used for control flow.
    fn name() -> &'static str;
}

/// Kernel-accessible pool. Pages have a KDMAP alias under every satp and
/// can be dereferenced from supervisor code. Use for kernel-owned
/// allocations (trap frames, rings) and for shared user memory the
/// kernel must dereference after creation (NetChannel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shared;
/// User-private pool. Mapped only under the owning user satp. Kernel
/// writes at setup time have to go through a temporary window
/// (`UserPageWindow`) — there is no `to_kdmap` conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserOnly;
/// Page-table pool. Shares KDMAP-visibility with [`Shared`] but is
/// distinct so that returning a table to `kernel_pages` (or vice versa)
/// is a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Table;

impl Pool for Shared {
    fn name() -> &'static str {
        "Shared"
    }
}
impl Pool for UserOnly {
    fn name() -> &'static str {
        "UserOnly"
    }
}
impl Pool for Table {
    fn name() -> &'static str {
        "Table"
    }
}

/// A physical address tagged with the pool it was drawn from. `Frame<P>`
/// is the tagged counterpart of [`mmu::sv48::PhysAddr`]: the walker
/// consumes the raw `PhysAddr`, but the rest of the kernel works in
/// terms of `Frame<P>` so wrong-pool mix-ups are caught at compile time.
///
/// Construction is `pub` to keep call sites readable — the pool
/// commitment happens where a caller decides what pool the PA belongs to
/// (typically the allocator wrapper). Treat `new` as a promise that
/// `pa` came from the `P` pool.
#[repr(transparent)]
pub struct Frame<P: Pool> {
    pa: PhysAddr,
    _p: PhantomData<P>,
}

impl<P: Pool> Frame<P> {
    pub const fn new(pa: PhysAddr) -> Self {
        Self {
            pa,
            _p: PhantomData,
        }
    }
    pub fn raw(&self) -> PhysAddr {
        self.pa
    }
    pub fn get_raw(&self) -> u64 {
        self.pa.get_raw()
    }
}

impl<P: Pool> fmt::Debug for Frame<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame<{}>(pa=0x{:016X})", P::name(), self.pa.get_raw())
    }
}

/// Physical backing for a [`UserMapping`]. Absent for pure vaddr
/// reservations like guard pages. The variant tag (`Shared` / `User`)
/// is the pool the backing was drawn from — free paths match on this
/// to dispatch to the right typed allocator.
#[derive(Debug)]
pub enum PhysBacking {
    Shared {
        frame: Frame<Shared>,
        layout: Layout,
    },
    User {
        frame: Frame<UserOnly>,
        layout: Layout,
    },
}

impl PhysBacking {
    pub fn pa(&self) -> PhysAddr {
        match self {
            Self::Shared { frame, .. } => frame.raw(),
            Self::User { frame, .. } => frame.raw(),
        }
    }
    pub fn layout(&self) -> Layout {
        match self {
            Self::Shared { layout, .. } => *layout,
            Self::User { layout, .. } => *layout,
        }
    }
    pub fn pool_name(&self) -> &'static str {
        match self {
            Self::Shared { .. } => Shared::name(),
            Self::User { .. } => UserOnly::name(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MappingKind {
    /// Loaded from a PT_LOAD segment at process creation.
    Elf,
    /// Anonymous user mmap.
    Anon,
    /// Thread stack (readable/writeable region above its guard).
    Stack { slot: u16 },
    /// Reserved vaddr range with no backing — a fault here signals an overflow
    /// into an adjacent stack/TLS/etc.
    Guard { slot: u16 },
    /// Per-thread trap frame, kernel-owned and mapped read-only into user.
    TrapFrame { slot: u16 },
    /// Per-thread TLS block; the TCB sits at the low end.
    Tls { slot: u16 },
    /// Kernel-allocated NetChannel region shared with the net thread.
    NetCh { fd: u32 },
}

impl MappingKind {
    pub fn slot(&self) -> Option<u16> {
        match *self {
            Self::Stack { slot }
            | Self::Guard { slot }
            | Self::TrapFrame { slot }
            | Self::Tls { slot } => Some(slot),
            _ => None,
        }
    }
}

/// A contiguous [`vaddr`, `vaddr + len`) region in a process's address space.
/// Keyed by `vaddr` in [`Process::maps`]; ranges never overlap.
#[derive(Debug)]
pub struct UserMapping {
    pub vaddr: u64,
    pub len: u64,
    pub perms: u64,
    pub backing: Option<PhysBacking>,
    pub kind: MappingKind,
}

impl UserMapping {
    pub fn end(&self) -> u64 {
        self.vaddr + self.len
    }

    pub fn contains(&self, v: u64) -> bool {
        self.vaddr <= v && v < self.end()
    }
}

#[derive(Debug)]
#[repr(C, align(64))]
pub struct Thread {
    pub pc: AtomicUsize,
    pub state: AtomicUsize,
    /// Thread's own scheduled park time. The thread itself is the sole
    /// writer (set on park; the kernel scheduler reads but never
    /// writes here). Non-atomic because of that single-writer
    /// invariant — concurrent reads of an aligned `usize` on RV64
    /// are safe under the current ABI, and the field's value is
    /// only consulted in the `Suspended → Ready` transition where
    /// the `state` field's release-store fences any prior write.
    pub wake_time: usize,
    /// Pending wake reasons as a bitmask of [`wake_reason`] flags.
    ///
    /// `0` = no pending wake. Any non-zero bit-pattern is "wake this
    /// thread now, regardless of `wake_time`" — and the bits encode
    /// *why*. Producers (PLIC IRQ → WAKE_QUEUE drain, ring-staging,
    /// syscall paths) `fetch_or` their reason bit. The scheduler
    /// `swap(0)` to atomically consume the union of pending reasons
    /// when it transitions the thread `Suspended → Ready`, and stashes
    /// the consumed bitmask in [`last_wake_reason`] for later query.
    ///
    /// The split keeps the parking-thread → manager race off the
    /// critical path: the parking thread writes `wake_time` only,
    /// producers write `wake_override` only, the two writers touch
    /// disjoint fields and never overwrite each other's signals.
    pub wake_override: AtomicU64,
    /// Bitmask of [`wake_reason`] flags consumed at the most recent
    /// `Suspended → Ready` transition. The scheduler writes this when
    /// it `swap(0)`s [`wake_override`]; userspace can query via a
    /// future syscall to learn what woke it (I/O, signal, timer-cap,
    /// etc.). Kept off the trap frame so we don't clobber user a-regs.
    pub last_wake_reason: AtomicU64,
    /// Generation counter for park instances. `fetch_add(1, Release)`-ed
    /// on every transition into `Suspended`. The sleep-heap entry
    /// captures the post-increment value at push; on pop the heap
    /// compares the captured seq against the live `sleep_seq` and
    /// treats a mismatch as stale (the thread re-parked since this
    /// entry was pushed). See [orbit-core/src/sleep_heap.rs] for the
    /// staleness contract.
    pub sleep_seq: AtomicU64,
    pub frame: &'static mut TrapFrame,
    pub stack: &'static mut Stack,
    /// Owning handle on the kernel-thread stack and trap-frame allocations.
    /// `Some` only for kernel threads (pid==0); user threads track their
    /// stack/trap-frame via `Process.maps` `PhysBacking` entries instead.
    /// `dealloc_thread` `take()`s these and hands them to
    /// `kernel_pages.free` directly — no PA round-tripping.
    pub kernel_stack: Option<Frame<Shared>>,
    pub kernel_trap_frame: Option<Frame<Shared>>,
    pub satp: Satp,
    pub mode: SPP,
    /// Set by the trap handler when this thread is killed by a fault.
    /// `None` means the thread exited cleanly (e.g. via the exit syscall).
    pub fault_info: Option<FaultInfo>,
    /// Wait/signal handle this thread is parked on while
    /// `state == Blocking`. The manager scans for signaled handles each
    /// scheduler pass; on a hit it writes `result()` / `extra()` into
    /// `frame.regs[10]` / `frame.regs[11]`, clears the slot, and marks
    /// the thread `Ready`.
    pub handle: Option<CompletionHandle>,
    /// On-thread completion-result slots used by manager-resolved
    /// blocking syscalls — the no-Arc alternative to [`Self::handle`].
    /// The manager drains a `PendingWork` keyed on this thread's tid,
    /// writes return values via [`Self::publish_results`], then pushes
    /// `WakeEvent::Tid(tid)` onto `WAKE_QUEUE`. The wake drain marshals
    /// `pending_rets[..pending_ret_count]` into `frame.regs[10..]`
    /// before promoting Suspended → Ready.
    ///
    /// Only the manager (under MANAGER_LOCK) writes these. The parker
    /// reads them at most once on the post-publish re-check or via the
    /// drain path; both happen *after* the manager's Release-store of
    /// `pending_state = SIGNALED`, which Acquire-paired loads observe.
    /// Trap-context signalers (read_stdin's `push_byte`, future IRQ
    /// paths) keep using [`Self::handle`] instead — the swap-of-waiter
    /// protocol is what makes that case safe without the lock.
    pub pending_rets: [AtomicI64; 4],
    /// Snapshot of the owning process's [`Permissions`] at the time
    /// the thread was created. Refreshed when the process pledges
    /// (manager-side: walks every thread of the process under
    /// MANAGER_LOCK and rewrites this field).
    ///
    /// The dispatch site reads this snapshot without locking, calls
    /// `permissions.allows(sysno)`, and on `false` records to the
    /// kernel-wide `DenialRing` + bumps the owning process's
    /// `perm_denials` counter, then short-circuits the syscall with
    /// `-EPERM`.
    pub permissions: Permissions,
    /// Immutable upper bound on which harts this thread can ever run on.
    /// Bit `i` set ⇔ hart `i` permitted. Set at construction; `set_affinity`
    /// rejects any mask that escapes this bound (Windows-style: a parent
    /// can fence a child's reach without the child being able to expand).
    pub allowed_affinity: u64,
    /// Current per-hart eligibility mask. Initialized to `allowed_affinity`;
    /// the user may narrow it via the `set_affinity` syscall, but
    /// `affinity & !allowed_affinity` is always zero. Atomic so the
    /// scheduler reads it without locking the process table.
    pub affinity: AtomicU64,
    // ─── per-thread accounting (read by query_stats from any hart) ──
    /// Cumulative user-mode CPU time in `time` CSR ticks. Credited at
    /// each User → ¬User hart-bucket transition by the owning hart;
    /// foreign-hart reads (stats snapshot) are racy but tear-safe on
    /// RV64 via the atomic. Phase 2.
    pub cpu_ticks_total: AtomicU64,
    /// Times this thread transitioned into Running. Incremented by the
    /// scheduler on dispatch; foreign-hart reads as above.
    pub context_switches: AtomicU64,
    /// Syscalls dispatched against this thread.
    pub syscall_count: AtomicU64,
    /// Cumulative kernel service ticks across this thread's syscalls
    /// (excludes time spent parked).
    pub syscall_ticks: AtomicU64,
    /// Snapshot of the owning process's POSIX credential triplet. Read
    /// without locking by the `getuid`/`geteuid`/`getgid`/`getegid`
    /// fast paths; refreshed alongside [`Self::permissions`] when the
    /// process mutates its credentials (today: only at construction;
    /// future: by `setuid`/`setgid` propagating across all threads of
    /// the process under MANAGER_LOCK, same shape as pledge).
    ///
    /// Supplementary groups and login name are *not* snapshotted onto
    /// the thread — they're variable-size and rarely-read; the
    /// `getgroups`/`getlogin` syscalls go through the manager-side
    /// [`Process`] lookup instead.
    pub uid: u32,
    pub euid: u32,
    pub suid: u32,
    pub gid: u32,
    pub egid: u32,
    pub sgid: u32,
    pub tid: u32,
    pub pid: u16,
    /// Per-process slot index. `None` for kernel threads.
    pub slot: Option<u16>,
    /// Snapshot of [`Process::stdout_redirect`] taken at thread
    /// construction. The `console_write` syscall reads this without
    /// locking the process table; immutable for the thread's
    /// lifetime because the owning `Process`'s redirect is set at
    /// spawn and never mutated. `None` ⇒ writes go to the thread's
    /// own pid pane (today's behavior); `Some(target)` ⇒ writes
    /// route to `Source::Process(target)` instead.
    pub stdout_redirect: Option<u16>,
    /// State byte for the on-thread completion path. `PENDING_STATE_NONE`
    /// (0) is the initial value and the post-consume reset; any
    /// non-zero value (today only `PENDING_STATE_SIGNALED` = 1) means
    /// `pending_rets[..pending_ret_count]` carries valid return data.
    /// Ordering: the manager's Release-store of SIGNALED publishes the
    /// rets writes; the wake drain's Acquire-load synchronizes against
    /// it before reading rets.
    pub pending_state: AtomicU8,
    /// Number of valid slots in `pending_rets` (0..=4). Manager writes
    /// before the SIGNALED store; readers Acquire-load via [`Self::pending_state`]
    /// and then trust this count.
    pub pending_ret_count: AtomicU8,
    pub ticks: u8,
}

/// Initial / consumed state for [`Thread::pending_state`].
pub const PENDING_STATE_NONE: u8 = 0;
/// Manager has published return values into [`Thread::pending_rets`]
/// and the parker (or the drain path) should marshal them into
/// `frame.regs[10..]` on resume.
pub const PENDING_STATE_SIGNALED: u8 = 1;

impl Thread {
    pub fn root_table_addr(&self) -> PhysAddr {
        PhysAddr::from(self.satp)
    }

    /// Reset the on-thread completion slot to its post-consume state.
    /// The manager calls this on every dispatch so a successor syscall
    /// never observes stale rets from a prior call. Cheap: two
    /// `Relaxed` stores. Per-thread single-writer at the dispatch path,
    /// so no atomicity drama with concurrent reads — there are none in
    /// flight at this point (the thread is `Running` on this hart, no
    /// sibling can be reading its rets).
    #[inline]
    pub fn reset_pending(&self) {
        self.pending_state
            .store(PENDING_STATE_NONE, Ordering::Relaxed);
        self.pending_ret_count.store(0, Ordering::Relaxed);
    }

    /// Manager-side: publish up to 4 return values to this thread.
    /// `vals.len()` is clamped to [`pending_rets`]'s width. The store
    /// order is rets → count → state, all `Relaxed` until the final
    /// state store, which is `Release` — that single ordering point
    /// publishes every prior write to any Acquire-paired reader of
    /// `pending_state`.
    ///
    /// Caller-required invariant: only the manager (under MANAGER_LOCK)
    /// invokes this. Thread state must be `Blocking` or `Suspended` so
    /// the parker isn't observing `frame.regs` on another hart at the
    /// same time. The wake drain or the parker's post-publish re-check
    /// is responsible for actually moving the rets into `frame.regs`.
    ///
    /// [`pending_rets`]: Self::pending_rets
    pub fn publish_results(&self, vals: &[isize]) {
        let n = vals.len().min(self.pending_rets.len());
        for (i, &v) in vals.iter().enumerate().take(n) {
            self.pending_rets[i].store(v as i64, Ordering::Relaxed);
        }
        self.pending_ret_count.store(n as u8, Ordering::Relaxed);
        self.pending_state
            .store(PENDING_STATE_SIGNALED, Ordering::Release);
    }

    /// Reader-side: atomically claim the SIGNALED state and return
    /// the published return values. CAS-claim shape (SIGNALED →
    /// NONE) is the at-most-once gate: when the parker (post-park
    /// re-check) and the manager (`set_wake_reason_where` drain) both
    /// race to wake a thread, exactly one wins this CAS — the other
    /// gets `None` and bails out. Without this, both paths would
    /// marshal regs + transition state + push the thread onto a ready
    /// queue, and `assign_threads` would dispatch the same thread to
    /// two harts simultaneously (double-dispatch corruption).
    ///
    /// The successful CAS uses AcqRel: the Acquire half synchronizes
    /// with the manager's Release store in [`publish_results`] so the
    /// subsequent `pending_rets` loads observe a coherent snapshot;
    /// the Release half (paired with Acquire failure-ordering on the
    /// loser side) ensures any reader that observes NONE also sees
    /// the winner's later writes (frame regs, state).
    pub fn take_pending_results(&self, out: &mut [i64; 4]) -> Option<usize> {
        if self
            .pending_state
            .compare_exchange(
                PENDING_STATE_SIGNALED,
                PENDING_STATE_NONE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return None;
        }
        let n = self.pending_ret_count.load(Ordering::Relaxed) as usize;
        let n = n.min(out.len());
        for i in 0..n {
            out[i] = self.pending_rets[i].load(Ordering::Relaxed);
        }
        Some(n)
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct PThread(pub *mut Thread);

#[derive(Copy, Clone, Debug)]
pub enum ProcessState {
    Running,
    Waiting,
    Broken,
}

/// Maximum supplementary groups per process — POSIX `NGROUPS_MAX`. Set
/// to 16 to match OpenBSD's default; `setgroups` will reject longer
/// lists with `EINVAL`. Bump cautiously — the value is part of the
/// observable ABI (`getgroups` callers size their buffers off this) and
/// changing it forces every consumer to re-check.
pub const NGROUPS_MAX: usize = 16;

#[derive(Debug)]
pub struct Process {
    pub pid: u16,
    /// Pid of the spawning process. `0` for the boot process (no
    /// parent). §13a.2's `wait_pid` checks this against the caller's
    /// pid to gate exit-status visibility.
    pub parent_pid: u16,
    /// Last exit status observed for this process — written when the
    /// thread that empties `threads` reports its `exit(code)` value.
    /// Read by `dealloc_process` to signal `exit_waiter` (if any).
    /// Multi-threaded processes are last-writer-wins by default; the
    /// `exit_finalized` flag (set by exit-group) suppresses subsequent
    /// overwrites so the caller's status is preserved even after
    /// rayon-style worker reaps.
    pub exit_code: i32,
    /// Set by `EXIT` (sysno 0) when it kicks off process-wide
    /// teardown. While true, `cleanup_threads_and_processes` leaves
    /// `exit_code` alone — the value the exit-caller passed wins,
    /// regardless of the order in which sibling threads are reaped.
    pub exit_finalized: bool,
    /// Single-waiter slot for §13a.2 `wait_pid`. v1 contract: at most
    /// one parent thread parks here at a time; a second `wait_pid`
    /// call returns EBUSY. Multi-waiter wants a `Vec<u32>` and lands
    /// with futex (§13a.5).
    ///
    /// Stores the parker's tid. `dealloc_process` resolves it via
    /// `Orbit::publish_pending_for_tid(tid, &[0, exit_code])` — the
    /// on-thread completion path's two-register publish. Stale tids
    /// (parker exited mid-wait) are silently dropped by the resume
    /// helper.
    pub exit_waiter: Option<u32>,
    /// Already-exited children whose parent (this process) hasn't
    /// called `wait_pid` yet. Keyed by child pid → child's exit code.
    /// Drained when the parent waits, or freed wholesale when the
    /// parent itself exits. Closes the wait_pid race when the child
    /// exits before the parent has a chance to park.
    pub dead_children: BTreeMap<u16, i32>,
    /// `true` when this process was spawned with the
    /// [`CreateProcessV2Args::DETACH`] flag. The exit path for a
    /// detached process skips both the `exit_waiter` notify and the
    /// parent-side `dead_children` insert, so a long-lived parent
    /// (orbit-loader) doesn't accumulate per-spawn exit-code entries
    /// across thousands of fire-and-forget children. Once the process
    /// is reaped its identity is forgotten; a parent-side `wait_pid`
    /// will see `ECHILD` and not block.
    pub detached: bool,
    /// §13a.3 argv blob backing — `Some` when the process was
    /// spawned via `CREATE_PROCESS_EX` with non-empty argv. The
    /// kernel maps this single page R+U+S at
    /// `USER_ARGV_BASE` in the process PT; `dealloc_process` returns
    /// the frame to `kernel_pages`.
    pub argv_blob: Option<PhysBacking>,
    /// §13e envp blob backing — `Some` when the process was spawned
    /// via `CREATE_PROCESS_EX` with a non-zero `envp_vaddr`. Wire
    /// format is identical to `argv_blob`; the kernel maps this
    /// single page R+U+S at `USER_ENVP_BASE`. `dealloc_process`
    /// returns the frame to `kernel_pages`.
    pub envp_blob: Option<PhysBacking>,
    pub state: ProcessState,
    pub threads: BTreeSet<u32>,
    pub thread_count: u16,
    pub satp: Satp,
    pub heap_pages: Vec<PhysBacking>,
    pub sockets: BTreeSet<SocketHandle>,

    /// VMAs for this process, keyed by vaddr. Invariant: no two entries overlap.
    pub maps: BTreeMap<u64, UserMapping>,
    /// First-fit arena cursor. Kernel sets this to the low end of the mmap
    /// arena during process creation.
    pub mmap_cursor: u64,
    /// Per-process thread slot allocator.
    pub thread_slots: SlotAlloc,

    /// Static TLS template snapshotted from the binary's `PT_TLS` at
    /// ELF-load time. `None` means the binary has no TLS (or an empty
    /// PT_TLS, which the linker still emits) and per-thread create
    /// skips the TLS allocation. When `Some`:
    /// - `tls_template` holds the first `tls_filesz = template.len()`
    ///   bytes of the TLS image (the `.tdata` initial values). Per-thread
    ///   TLS pages are populated by copying these bytes in; the trailing
    ///   `tls_memsz - tls_filesz` bytes are implicitly zero.
    /// - `tls_memsz` is `p_memsz` from PT_TLS (rounded up to PAGE_SIZE
    ///   when allocating per-thread pages).
    /// - `tls_align` is `p_align` (RISC-V variant-I: typically 8 or 16;
    ///   subsumed by the page-aligned per-thread allocation).
    pub tls_template: Option<Vec<u8>>,
    pub tls_memsz: usize,
    pub tls_align: usize,

    /// Authoritative process permissions. Mutated by the manager
    /// under MANAGER_LOCK on `pledge` (narrowing-only) and at child
    /// install time during `create_process_v2`. Each thread of the
    /// process holds a snapshot in [`Thread::permissions`] which the
    /// dispatch-site permission gate reads without locking; pledge
    /// propagates by walking every live thread.
    ///
    /// Default at construct time is [`Permissions::ZERO`] — fail
    /// closed, NOROLE with empty masks. Construction sites that need
    /// real perms (`create_new_process` stamps BOOTSTRAP-shaped
    /// `Permissions::ALL` so legacy `CREATE_PROCESS` /
    /// `CREATE_PROCESS_EX` callers keep working; the
    /// role-resolved `create_process_v2` path installs the
    /// witness-derived mask) must call
    /// [`Process::install_permissions`] explicitly. A new spawn
    /// path that forgets to set perms produces an unprivileged
    /// process rather than a fully-trusted one — the safer failure
    /// mode.
    pub permissions: Permissions,
    /// Real uid — POSIX `getuid()`. Identity-only, not used for
    /// authentication or authorization (roles + permissions own that
    /// surface). Carried for Unix-compat observability (`getuid`,
    /// `ps`-style diagnostics) and for fs ownership metadata once
    /// writes land. The real/effective/saved triplet matches POSIX
    /// `_POSIX_SAVED_IDS` so a future `setuid(2)` can implement the
    /// standard saved-set rules without growing this struct.
    pub uid: u32,
    /// Effective uid — POSIX `geteuid()`. The id used for FS access
    /// checks (`vaccess` against inode `st_uid`) once enforcement
    /// lands. v1: equals `uid` for every process; only diverges when
    /// `setuid`/`seteuid` ship.
    pub euid: u32,
    /// Saved-set uid — POSIX `getresuid()`'s third slot. Stamped at
    /// spawn from the parent's `euid` (POSIX exec semantics carried
    /// over to orbit's spawn-only model); future `seteuid` may swap
    /// the effective uid back to this value to reclaim privilege a
    /// process voluntarily dropped.
    pub suid: u32,
    /// Real gid — POSIX `getgid()`. Same identity-only caveats as
    /// [`uid`](Self::uid); paired with `egid`/`sgid` for the standard
    /// triplet.
    pub gid: u32,
    /// Effective gid — POSIX `getegid()`. FS-access counterpart to
    /// `euid`; checked against inode `st_gid` after the uid arm of
    /// `vaccess`.
    pub egid: u32,
    /// Saved-set gid — POSIX `getresgid()`'s third slot. Mirror of
    /// `suid` for gids.
    pub sgid: u32,
    /// Supplementary group memberships — POSIX `getgroups()`. Capped
    /// at [`NGROUPS_MAX`] entries (matches OpenBSD); `setgroups`
    /// rejects longer lists with `EINVAL`. Empty by default; populated
    /// at spawn time by a `LOGIN`-role caller (future) reading
    /// `/etc/group` and stamping the child via the spawn syscall.
    pub groups: Vec<u32>,
    /// Session login name — POSIX `getlogin()` / `setlogin()`. Pure
    /// accounting / observability surface; auth never consults it
    /// (mirrors OpenBSD's documented "advisory" classification).
    /// `None` means "no login name installed" — the read syscall
    /// reports `ENOENT`. Set once at spawn by the future `login`
    /// binary; mutable in-process via the future `setlogin(2)` (gated
    /// on `euid == 0`).
    pub login_name: Option<String>,
    /// Count of times the dispatch-site bitmask gate has EPERMed a
    /// syscall from this process. Surfaced via `query_stats`'s
    /// `perm_denials` field. Incremented by the manager-side
    /// `drain_denial_events` pass after consuming a `PermDeny`
    /// event off the lock-free producer queue. Atomic so
    /// foreign-hart reads (e.g. from the stats snapshot path) are
    /// tear-safe.
    pub perm_denials: AtomicU64,
    /// Count of times `create_process_v2`'s role-transition gate
    /// has EPERMed a spawn from this process. Surfaced via
    /// `query_stats`'s `role_denials`. Incremented inline by the
    /// manager-side `create_process_v2` handler before it returns
    /// `-EPERM`.
    pub role_denials: AtomicU64,
    /// Per-process current working directory. Always an absolute
    /// UTF-8 path (rooted at `/`) — relative path syscalls
    /// (`fs_open`/`fs_stat`/`fs_readdir`) prefix this before
    /// resolution. Mutated by `chdir`; copied from the parent at
    /// spawn time, or overridden by `CreateProcessV2Args.cwd_*` if
    /// the spawn caller passed a non-empty buffer. Init process
    /// boots with `/`.
    pub cwd: String,
    /// `Some(parent_pid)` ⇒ this process's `console_write` syscalls
    /// route their bytes to `Source::Process(parent_pid)` in the
    /// display compositor instead of this process's own pane. Set at
    /// spawn time when `CreateProcessV2Args.stdout_capture == 1`;
    /// never mutated. Each [`Thread`] of this process holds a
    /// snapshot in [`Thread::stdout_redirect`] so the syscall hot
    /// path is lock-free.
    ///
    /// If the redirect target has exited by the time the child
    /// writes, `k_gpu::push_chunk` quietly drops the bytes — same
    /// failure shape as a full ring. The kernel does not validate
    /// the target is alive at write time.
    pub stdout_redirect: Option<u16>,
}

impl Process {
    pub fn new(pid: u16, parent_pid: u16, satp: Satp) -> Self {
        Self {
            pid,
            parent_pid,
            exit_code: 0,
            exit_finalized: false,
            exit_waiter: None,
            dead_children: BTreeMap::new(),
            detached: false,
            argv_blob: None,
            envp_blob: None,
            state: ProcessState::Running,
            threads: BTreeSet::new(),
            thread_count: 0,
            satp,
            heap_pages: Vec::new(),
            sockets: BTreeSet::new(),
            maps: BTreeMap::new(),
            mmap_cursor: 0,
            thread_slots: SlotAlloc::new(),
            tls_template: None,
            tls_memsz: 0,
            tls_align: 0,
            // Fail closed: NOROLE + no perms. Spawn-path callers
            // (`create_new_process`, future role-resolved spawns)
            // override via `install_permissions` so a missed
            // assignment produces an unprivileged process rather
            // than a fully-trusted one.
            permissions: Permissions::ZERO,
            uid: 0,
            euid: 0,
            suid: 0,
            gid: 0,
            egid: 0,
            sgid: 0,
            groups: Vec::new(),
            login_name: None,
            perm_denials: AtomicU64::new(0),
            role_denials: AtomicU64::new(0),
            cwd: "/".to_string(),
            stdout_redirect: None,
        }
    }

    /// Pledge-narrow this process's permissions in place. Caller must
    /// also propagate the new value to every live [`Thread`] of this
    /// process (each thread caches a snapshot for the lock-free
    /// dispatch path). The two-step caller responsibility lives at
    /// the manager: only it has the thread registry needed to walk
    /// siblings. See `run_pledge_req` in kmain.
    pub fn pledge(&mut self, request: PermsRequest) {
        self.permissions = self.permissions.pledge(request);
    }

    /// Install perms on a freshly-spawned child via the witness
    /// path. Only [`ChildPerms`] can be constructed by
    /// [`derive_child_perms`](orbit_abi::roles::derive_child_perms),
    /// which itself requires a [`TransitionAllowed`] from
    /// [`check_transition`](orbit_abi::roles::check_transition) — so
    /// reaching this method type-enforces "both gates ran." The
    /// `create_process_v2` handler is the only caller.
    ///
    /// [`ChildPerms`]: orbit_abi::roles::ChildPerms
    /// [`TransitionAllowed`]: orbit_abi::roles::TransitionAllowed
    pub fn install_child(&mut self, c: ChildPerms) {
        self.permissions = c.into_permissions();
    }

    /// Migration backstop. Stamps `Permissions` directly without
    /// requiring a witness — used by the legacy
    /// `CREATE_PROCESS` / `CREATE_PROCESS_EX` spawn paths to install
    /// the BOOTSTRAP-shaped default. Wider than [`install_child`];
    /// reviewers police new call sites. Deletes when the legacy
    /// syscalls retire.
    pub fn install_permissions(&mut self, p: Permissions) {
        self.permissions = p;
    }

    /// Find the mapping (if any) whose range contains `vaddr`.
    pub fn find_mapping(&self, vaddr: u64) -> Option<&UserMapping> {
        self.maps
            .range(..=vaddr)
            .next_back()
            .map(|(_, m)| m)
            .filter(|m| m.contains(vaddr))
    }

    /// First-fit scan of the arena. Returns the lowest vaddr `>= mmap_cursor`
    /// where `[v, v+len)` fits without overlapping an existing mapping and
    /// stays below `top`. Does not mutate state; caller is expected to insert
    /// the resulting mapping and advance `mmap_cursor` itself.
    pub fn pick_user_vaddr(&self, len: u64, align: u64, top: u64) -> Option<u64> {
        let mut candidate = round_up(self.mmap_cursor, align);
        for (_, m) in self.maps.range(candidate..) {
            if candidate + len <= m.vaddr {
                return Some(candidate);
            }
            candidate = round_up(m.end(), align);
        }
        if candidate + len <= top {
            Some(candidate)
        }
        else {
            None
        }
    }

    /// Check that `[vaddr, vaddr+len)` overlaps no existing mapping. Used by
    /// MAP_FIXED-style requests where the caller chose the address.
    pub fn validate_free_range(&self, vaddr: u64, len: u64) -> Result<(), OverlapErr> {
        if let Some((_, prev)) = self.maps.range(..=vaddr).next_back() {
            if prev.end() > vaddr {
                return Err(OverlapErr);
            }
        }
        if let Some((_, next)) = self.maps.range(vaddr..).next() {
            if vaddr + len > next.vaddr {
                return Err(OverlapErr);
            }
        }
        Ok(())
    }

    pub fn insert_mapping(&mut self, m: UserMapping) {
        self.maps.insert(m.vaddr, m);
    }

    /// Iterate mappings owned by a specific thread slot. Used by teardown.
    pub fn mappings_for_slot(&self, slot: u16) -> impl Iterator<Item = &UserMapping> {
        self.maps
            .values()
            .filter(move |m| m.kind.slot() == Some(slot))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OverlapErr;

const fn round_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod slot_alloc_tests {
    use super::SlotAlloc;

    #[test]
    fn new_is_empty() {
        let sa = SlotAlloc::new();
        assert_eq!(sa.len(), 0);
        for i in 0..SlotAlloc::CAPACITY {
            assert!(!sa.is_allocated(i));
        }
    }

    #[test]
    fn alloc_returns_0_then_1_then_2() {
        let mut sa = SlotAlloc::new();
        assert_eq!(sa.alloc(), Some(0));
        assert_eq!(sa.alloc(), Some(1));
        assert_eq!(sa.alloc(), Some(2));
        assert_eq!(sa.len(), 3);
        assert!(sa.is_allocated(0));
        assert!(sa.is_allocated(1));
        assert!(sa.is_allocated(2));
        assert!(!sa.is_allocated(3));
    }

    #[test]
    fn free_releases_slot_and_next_alloc_reuses() {
        let mut sa = SlotAlloc::new();
        let _ = sa.alloc(); // 0
        let _ = sa.alloc(); // 1
        sa.free(0);
        assert!(!sa.is_allocated(0));
        assert_eq!(sa.len(), 1);
        // trailing_ones-based alloc finds the first hole → 0 again.
        assert_eq!(sa.alloc(), Some(0));
    }

    #[test]
    fn fills_to_capacity_and_then_returns_none() {
        let mut sa = SlotAlloc::new();
        for i in 0..SlotAlloc::CAPACITY {
            assert_eq!(sa.alloc(), Some(i));
        }
        assert_eq!(sa.len(), SlotAlloc::CAPACITY as u32);
        assert!(sa.alloc().is_none());
    }

    #[test]
    fn free_non_allocated_slot_is_noop() {
        let mut sa = SlotAlloc::new();
        sa.free(42);
        assert_eq!(sa.len(), 0);
        // Subsequent alloc still starts at 0.
        assert_eq!(sa.alloc(), Some(0));
    }

    #[test]
    fn free_in_middle_then_allocate_fills_the_hole() {
        let mut sa = SlotAlloc::new();
        for _ in 0..10 {
            sa.alloc();
        }
        sa.free(5);
        assert_eq!(sa.len(), 9);
        assert!(!sa.is_allocated(5));
        // trailing_ones finds bit 5 first.
        assert_eq!(sa.alloc(), Some(5));
    }

    #[test]
    fn len_matches_is_allocated_count() {
        let mut sa = SlotAlloc::new();
        let taken: [u16; 5] = [0, 1, 2, 3, 4];
        for _ in &taken {
            sa.alloc();
        }
        sa.free(1);
        sa.free(3);
        let counted = (0..SlotAlloc::CAPACITY)
            .filter(|i| sa.is_allocated(*i))
            .count() as u32;
        assert_eq!(sa.len(), counted);
    }

    #[test]
    fn word_boundary_alloc_sequence() {
        // Capacity is 256 = 4 * 64; allocs cross word boundaries at 64, 128,
        // 192. Confirm trailing_ones math works across them.
        let mut sa = SlotAlloc::new();
        for expected in 0..200u16 {
            assert_eq!(sa.alloc(), Some(expected));
        }
        assert!(sa.is_allocated(63));
        assert!(sa.is_allocated(64));
        assert!(sa.is_allocated(127));
        assert!(sa.is_allocated(128));
        assert!(sa.is_allocated(199));
        assert!(!sa.is_allocated(200));
    }

    #[test]
    fn free_and_realloc_across_word_boundary() {
        // Allocate past a word boundary so bit 128 exists, then free
        // holes in two different words and confirm trailing_ones picks
        // them up in the expected order (word 1 hole before word 2).
        let mut sa = SlotAlloc::new();
        for _ in 0..=128 {
            sa.alloc();
        }
        sa.free(65); // word 1, bit 1
        sa.free(128); // word 2, bit 0
        assert_eq!(sa.alloc(), Some(65), "word 1 hole filled first");
        assert_eq!(sa.alloc(), Some(128), "then word 2");
    }
}
