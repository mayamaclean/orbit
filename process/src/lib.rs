#![no_std]

extern crate alloc;

use core::alloc::Layout;
use core::sync::atomic::{AtomicUsize};

use alloc::collections::{btree_map::BTreeMap, btree_set::BTreeSet};
use alloc::vec::Vec;
use device::{Stack, TrapFrame};
use riscv::register::{satp::Satp, sstatus::SPP};
use smoltcp::iface::SocketHandle;

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
pub struct MemMapReq {
    pub vaddr: usize,
    pub size: usize,
    pub page_permissions: u64,
    pub share_with_kernel: bool
}

#[derive(Debug, Clone, Copy)]
pub struct NetChannelRegistrationReq {
    pub nc_vaddr: usize,
    pub nc_type: usize
}

#[derive(Debug, Clone, Copy)]
pub enum ThreadBlockReason {
    NotBlocking,
    MemMap(MemMapReq),
    NetChannelRegistration(NetChannelRegistrationReq)
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

/// Which kernel frame pool a [`PhysBacking`] was drawn from. Drives
/// teardown routing and, eventually, whether the page is KDMAP-visible
/// from supervisor code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    /// Kernel-accessible via KDMAP in every satp. Use for kernel-owned
    /// allocations (trap frames, rings) and for shared user memory the
    /// kernel must dereference after creation (NetChannel).
    Shared,
    /// Mapped only into the owning user satp. Kernel writes at setup
    /// time have to go through a temporary window.
    UserOnly,
}

/// Physical backing for a [`UserMapping`]. Absent for pure vaddr reservations
/// like guard pages.
#[derive(Debug, Clone, Copy)]
pub struct PhysBacking {
    pub paddr:  u64,
    pub layout: Layout,
    pub pool:   Pool,
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
#[derive(Debug, Clone, Copy)]
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
    pub wake_time: usize,
    pub frame: &'static mut TrapFrame,
    pub stack: &'static mut Stack,
    pub satp: Satp,
    pub mode: SPP,
    pub block_reason: ThreadBlockReason,
    pub tid: u32,
    pub pid: u16,
    pub ticks: u8,
    /// Per-process slot index. `None` for kernel threads.
    pub slot: Option<u16>,
    /// Set by the trap handler when this thread is killed by a fault.
    /// `None` means the thread exited cleanly (e.g. via the exit syscall).
    pub fault_info: Option<FaultInfo>,
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
