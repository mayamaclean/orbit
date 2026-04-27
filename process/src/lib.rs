#![no_std]

extern crate alloc;

use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, AtomicUsize};

use alloc::collections::{btree_map::BTreeMap, btree_set::BTreeSet};
use alloc::vec::Vec;
use device::{Stack, TrapFrame};
use mmu::sv48::PhysAddr;
use riscv::register::{satp::Satp, sstatus::SPP};
use smoltcp::iface::SocketHandle;

pub mod completion;
pub mod spsc;
pub mod stdin;
pub use completion::{AckCounter, CompletionHandle};
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum ThreadState {
    Ready     = 0,
    Blocking  = 1,
    Assigned  = 2,
    Running   = 3,
    Exited    = 4,
    Suspended = 5
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
        Self { bits: [0; Self::WORDS] }
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
        let bit  = slot % 64;
        self.bits[word] &= !(1u64 << bit);
    }

    pub fn is_allocated(&self, slot: u16) -> bool {
        let word = (slot / 64) as usize;
        let bit  = slot % 64;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct Shared;
/// User-private pool. Mapped only under the owning user satp. Kernel
/// writes at setup time have to go through a temporary window
/// (`UserPageWindow`) — there is no `to_kdmap` conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct UserOnly;
/// Page-table pool. Shares KDMAP-visibility with [`Shared`] but is
/// distinct so that returning a table to `kernel_pages` (or vice versa)
/// is a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct Table;

impl Pool for Shared   { fn name() -> &'static str { "Shared" } }
impl Pool for UserOnly { fn name() -> &'static str { "UserOnly" } }
impl Pool for Table    { fn name() -> &'static str { "Table" } }

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
        Self { pa, _p: PhantomData }
    }
    pub fn raw(&self) -> PhysAddr { self.pa }
    pub fn get_raw(&self) -> u64 { self.pa.get_raw() }
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
    Shared { frame: Frame<Shared>,   layout: Layout },
    User   { frame: Frame<UserOnly>, layout: Layout },
}

impl PhysBacking {
    pub fn pa(&self) -> PhysAddr {
        match self {
            Self::Shared { frame, .. } => frame.raw(),
            Self::User   { frame, .. } => frame.raw(),
        }
    }
    pub fn layout(&self) -> Layout {
        match self {
            Self::Shared { layout, .. } => *layout,
            Self::User   { layout, .. } => *layout,
        }
    }
    pub fn pool_name(&self) -> &'static str {
        match self {
            Self::Shared { .. } => Shared::name(),
            Self::User   { .. } => UserOnly::name(),
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
    Stack     { slot: u16 },
    /// Reserved vaddr range with no backing — a fault here signals an overflow
    /// into an adjacent stack/TLS/etc.
    Guard     { slot: u16 },
    /// Per-thread trap frame, kernel-owned and mapped read-only into user.
    TrapFrame { slot: u16 },
    /// Per-thread TLS block; the TCB sits at the low end.
    Tls       { slot: u16 },
    /// Kernel-allocated NetChannel region shared with the net thread.
    NetCh     { fd: u32 },
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
    pub vaddr:   u64,
    pub len:     u64,
    pub perms:   u64,
    pub backing: Option<PhysBacking>,
    pub kind:    MappingKind,
}

impl UserMapping {
    pub fn end(&self) -> u64 { self.vaddr + self.len }

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
    pub frame: &'static mut TrapFrame,
    pub stack: &'static mut Stack,
    pub satp: Satp,
    pub mode: SPP,
    /// Wait/signal handle this thread is parked on while
    /// `state == Blocking`. The manager scans for signaled handles each
    /// scheduler pass; on a hit it writes `result()` / `extra()` into
    /// `frame.regs[10]` / `frame.regs[11]`, clears the slot, and marks
    /// the thread `Ready`.
    pub handle: Option<CompletionHandle>,
    pub tid: u32,
    pub pid: u16,
    pub ticks: u8,
    /// Per-process slot index. `None` for kernel threads.
    pub slot: Option<u16>,
    /// Set by the trap handler when this thread is killed by a fault.
    /// `None` means the thread exited cleanly (e.g. via the exit syscall).
    pub fault_info: Option<FaultInfo>,
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
    /// (excludes time spent parked on a `ThreadBlockReason`).
    pub syscall_ticks: AtomicU64,
}

impl Thread {
    pub fn root_table_addr(&self) -> usize {
        self.satp.ppn() * 4096
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct PThread(pub *mut Thread);

#[derive(Copy, Clone, Debug)]
pub enum ProcessState {
    Running,
    Waiting,
    Broken
}

#[derive(Debug)]
pub struct Process {
    pub pid: u16,
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
}

impl Process {
    pub fn new(pid: u16, satp: Satp) -> Self {
        Self {
            pid,
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
        }
    }

    /// Find the mapping (if any) whose range contains `vaddr`.
    pub fn find_mapping(&self, vaddr: u64) -> Option<&UserMapping> {
        self.maps.range(..=vaddr)
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
            if candidate + len <= m.vaddr { return Some(candidate); }
            candidate = round_up(m.end(), align);
        }
        if candidate + len <= top { Some(candidate) } else { None }
    }

    /// Check that `[vaddr, vaddr+len)` overlaps no existing mapping. Used by
    /// MAP_FIXED-style requests where the caller chose the address.
    pub fn validate_free_range(&self, vaddr: u64, len: u64) -> Result<(), OverlapErr> {
        if let Some((_, prev)) = self.maps.range(..=vaddr).next_back() {
            if prev.end() > vaddr { return Err(OverlapErr); }
        }
        if let Some((_, next)) = self.maps.range(vaddr..).next() {
            if vaddr + len > next.vaddr { return Err(OverlapErr); }
        }
        Ok(())
    }

    pub fn insert_mapping(&mut self, m: UserMapping) {
        self.maps.insert(m.vaddr, m);
    }

    /// Iterate mappings owned by a specific thread slot. Used by teardown.
    pub fn mappings_for_slot(&self, slot: u16) -> impl Iterator<Item = &UserMapping> {
        self.maps.values().filter(move |m| m.kind.slot() == Some(slot))
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
