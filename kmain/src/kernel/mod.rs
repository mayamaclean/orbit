use core::alloc::Layout;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::time::Duration;

use alloc::collections::btree_map::BTreeMap;
use alloc::{boxed::Box, vec::Vec};

use device::{HartContext, Stack, TrapFrame};
use dtoolkit::fdt::FdtNode;
use dtoolkit::{Node, fdt::Fdt};
use elf::endian::LittleEndian;
use mem::{round_u64_down, round_u64_up};
use mmu::mmap::{PageAlloc, map_address_range, unmap, unmap_page};
use mmu::sv48::{PageTable, PhysAddr, VirtAddr};
// `PAGE_SIZE` (usize) intentionally shadows the u64 re-export from
// `orbit_abi::layout::*` below — kmain consumes the usize form internally.
#[allow(hidden_glob_reexports)]
use mmu::{KB, MB, MappingConfig, PAGE_SIZE, PagePermissions, SupervisorTag};
use net_channel::NetChannel;
use process::{
    Frame, MappingKind, PThread, PhysBacking, Process, Shared, Thread, ThreadState, UserMapping,
    UserOnly,
};

use orbit_abi::errno::{
    EAGAIN, EBADF, EFAULT, EINVAL, EIO, ENOEXEC, ENOMEM, ENOTDIR, EPERM, ESRCH, Errno,
};
use orbit_core::ready_queue::ReadyQueue;
use orbit_core::sleep_heap::SleepHeap;
use orbit_core::{
    CloseHandleReq, CreateProcessExReq, CreateProcessReq, CreateThreadReq, FsOpenReq, FsReadReq,
    FsReaddirReq, FsStatReq, FutexWaitReq, FutexWakeReq, MAX_FS_PATH_LEN, MemMapReq,
    NetChannelCreationReq, PendingWork, WaitPidReq,
};
use thingbuf::StaticThingBuf;

use crate::kernel::fs::{FsErr, WorkNotification};
use crate::kernel::handle::{Handle, OpenFile, ProcessHandles};
use crate::kernel::memmap::FrameToKdmap;
use crate::kernel::shared_user_ptr::SharedUserPtr;
use riscv::register::satp::{Mode, Satp};
use riscv::register::sstatus::SPP;
use smoltcp::iface::{Config, Interface, SocketHandle};
use smoltcp::wire::EthernetAddress;
use tracing::{debug, error, info, trace, warn};

use crate::drivers::e1000::{
    E1000, E1000Pbuf, RX_RING_BUFS_BYTES, RX_RING_BYTES, RX_RING_LEN, RxDesc, TX_RING_BUFS_BYTES,
    TX_RING_BYTES, TX_RING_LEN, TxDesc,
};

use crate::kernel::context::get_hart_context;
use crate::kernel::pci::PciDevice;
use crate::{NetPackage, ProcessComponents, SocketReq};

pub mod accounting;
pub mod context;
pub mod fs;
pub mod handle;
pub mod input;
pub mod memmap;
pub mod orbital_elf;
pub mod pci;
pub mod pending_frees;
pub mod shared_frame;
pub mod shared_user_ptr;
pub mod shootdown;
pub mod stdin;
pub mod user_page;

pub use memmap::KernelLayout;

// TODO: page unmapping

// kmain always embeds orbit-loader as the initial user program. The
// loader fs_opens an init-binary path off the tarfs disk image and
// `create_process`es it; what binary is selected is controlled by the
// boot argv built in `k_smpstart`, not by recompiling kmain. The
// previously-conditional `smoke` / `hello-std` swaps for embedding
// `umode` / `hello-std` directly are gone — those binaries live on
// disk under `/bin/smoke` and `/bin/hello-std` (see
// tools/build-disk.sh) and the loader picks them via its argv.
pub const UMODE_TEST_ELF: &'static [u8] =
    include_bytes!("../../../orbit-loader/target/riscv64gc-unknown-none-elf/release/orbit-loader");

// User address-space layout lives in the canonical orbit_abi::layout module.
// Re-exported so existing `kernel::USER_TEXT_BASE`-style call sites keep
// working.
pub use orbit_abi::layout::*;

/// MPSC ring of `PendingWork` entries pushed by blocking-syscall paths
/// on any hart and drained by whichever hart next holds `MANAGER_LOCK`.
/// Cap chosen at ~8x current hart count so a steady-state burst of
/// concurrent blocking syscalls doesn't EAGAIN until something is
/// genuinely wedged. Default slot is `PendingWork::Empty`.
pub static MANAGER_WORK: StaticThingBuf<PendingWork, 32> = StaticThingBuf::new();

/// Targeted "tickle a parked thread" events. Producers: PLIC IRQ
/// handlers (e.g. e1000 RX → wake k_net), `update_tcp` (slice staged
/// → wake the NetCh's owner), syscall paths that publish state a
/// peer might be sleep-polling on. Consumer: the manager drains this
/// alongside `MANAGER_WORK` and bumps the matching thread's
/// `wake_time` to 0 so the next scheduler scan dispatches it.
///
/// This is *not* the cross-hart IPI mechanism (that's `write_sswi`).
/// It's a "the runnable predicate just became true; please re-check"
/// signal — the scheduler still does the actual dispatch.
///
/// Default slot is `WakeEvent::None` (the Default impl returns it).
pub static WAKE_QUEUE: StaticThingBuf<WakeEvent, 64> = StaticThingBuf::new();

/// Lock-free MPSC ring of denial events produced by the dispatch-
/// site gate. Producers: any hart's `s_trap` cause=8 arm on syscall
/// denial. Consumer: the manager drains this alongside
/// `MANAGER_WORK` and folds each event into the kernel-wide
/// [`Orbit::denial_ring`] + the owning process's `perm_denials` /
/// `role_denials` counter.
///
/// Lock-free is the load-bearing property: the trap path must not
/// spin on `MANAGER_LOCK` to log a denial. Push-on-full drops the
/// event and bumps [`DENIAL_EVENTS_DROPPED`] — best-effort retention,
/// matching the ring's "what was denied recently" semantics rather
/// than a "every denial since boot" guarantee.
///
/// Default slot is `None` — `Option<DenialEvent>::default()` returns
/// `None`, satisfying thingbuf's `T: Default` requirement without
/// adding a kernel-internal sentinel variant to the wire-shape
/// `DenialEvent` enum.
pub static DENIAL_EVENT_QUEUE: StaticThingBuf<Option<orbit_abi::denial::DenialEvent>, 64> =
    StaticThingBuf::new();

/// Count of denial events dropped due to a full [`DENIAL_EVENT_QUEUE`].
/// Surfaces queue-pressure issues for diagnostics. Atomic so any
/// hart's gate can bump it without coordination.
pub static DENIAL_EVENTS_DROPPED: AtomicU64 = AtomicU64::new(0);

/// Targeted wake-up event. See [`WAKE_QUEUE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeEvent {
    /// Sentinel default — pushed nowhere, drained as a no-op. The
    /// thingbuf API requires `Default` to mean "empty slot."
    None,
    /// Wake every kernel thread (pid=0). Today that's k_net (and
    /// possibly k_gpu); a finer-grained variant can come later.
    Net,
    /// Wake every thread of the given user pid. Coarse but cheap —
    /// each thread re-checks its own wait predicate on wake and
    /// re-parks if not actually ready, so over-waking is harmless.
    Pid(u16),
    /// Wake a specific thread by tid. Used for future per-session
    /// owner wake-ups where we know exactly which thread is parked
    /// on a given NetCh.
    Tid(u32),
}

impl Default for WakeEvent {
    fn default() -> Self {
        WakeEvent::None
    }
}

/// One park notification queued by a parking hart for the manager to
/// fold into [`Orbit::sleeping`]. The parking hart writes the
/// `Suspended` state and `fetch_add(1)`-s `sleep_seq` first, then
/// pushes this notice. The manager later drains the inbox under
/// `MANAGER_LOCK` and re-issues each entry into the heap.
///
/// `thread == null` is the [`Default`] sentinel that fills empty
/// thingbuf slots; the drain skips these without touching the heap.
#[derive(Clone, Copy)]
pub struct SleepNotice {
    pub wake_time: u64,
    pub sleep_seq: u64,
    pub thread: *mut Thread,
}

impl Default for SleepNotice {
    fn default() -> Self {
        Self {
            wake_time: 0,
            sleep_seq: 0,
            thread: core::ptr::null_mut(),
        }
    }
}

// SAFETY: `*mut Thread` here points into the kernel thread registry.
// The registry frees a Thread only from `cleanup_threads_and_processes`,
// which runs on the manager hart between `WAKE_QUEUE`/`SLEEP_INBOX`
// drain and `assign_threads` — so a notice in the inbox always names
// a live allocation when the manager pops it. Cross-hart movement of
// the raw pointer is the whole point of this inbox; the SafetyDoc
// captures that ordering.
unsafe impl Send for SleepNotice {}
unsafe impl Sync for SleepNotice {}

/// MPSC ring of [`SleepNotice`] entries pushed by parking harts and
/// drained into [`Orbit::sleeping`] by the manager. Same shape as
/// [`WAKE_QUEUE`]; cap chosen to absorb burst parks across all harts
/// without EAGAIN — at 4 harts and one park per syscall, 64 covers
/// well over a manager tick of activity.
pub static SLEEP_INBOX: StaticThingBuf<SleepNotice, 64> = StaticThingBuf::new();

/// Per-hart "thread just became Ready" notification, queued by
/// non-manager paths (e.g. `exit_thread_with_state(Ready)` on a
/// preempted hart). The manager drains every per-hart inbox into
/// `Orbit::ready` at the head of each `assign_threads` pass.
///
/// `thread == null` is the [`Default`] sentinel; the drain skips it.
#[derive(Clone, Copy)]
pub struct ReadyNotice {
    pub thread: *mut Thread,
}

impl Default for ReadyNotice {
    fn default() -> Self {
        Self {
            thread: core::ptr::null_mut(),
        }
    }
}

// SAFETY: same registry-lifetime argument as `SleepNotice` — the
// pointed-to Thread is freed only from the manager's
// `cleanup_threads_and_processes`, which runs in the same critical
// section as the inbox drain. No use-after-free window.
unsafe impl Send for ReadyNotice {}
unsafe impl Sync for ReadyNotice {}

/// Per-hart inbox of newly-Ready threads. Indexed by hart id. SPSC
/// from a single hart's perspective (it pushes; manager pops), but
/// the static array as a whole holds one entry per hart — manager
/// drains all of them.
///
/// Cap of 32 per hart is well above the working set: a hart can have
/// at most one `current` thread plus a handful of in-flight unblocked
/// threads waiting to be drained.
pub static READY_INBOXES: [StaticThingBuf<ReadyNotice, 32>; shootdown::MAX_HARTS] =
    [const { StaticThingBuf::new() }; shootdown::MAX_HARTS];

/// Wake hook called from `process::completion::signal_n` when a
/// signal claims a parked waiter. Reads the handle's freshly-stored
/// rets out of `t.handle`, marshals them into the saved frame,
/// clears the handle slot, marks the thread Ready, and pushes onto
/// the current hart's READY_INBOXES.
///
/// Runs on the signaling hart (any hart). The thread isn't
/// "current" on any hart at this point — the parker already set
/// state=Blocking and cleared its own current — so writing
/// `t.frame.regs` doesn't race with a dispatch.
pub fn wake_blocked_inline(thread_ptr: *mut Thread) {
    if thread_ptr.is_null() {
        return;
    }
    // SAFETY: signaler claimed the waiter via take_waiter; the
    // parker's set_waiter Release-ordered the prior `t.handle =
    // Some(...)` write so reading it here is safe.
    let t = unsafe { (thread_ptr as *mut Thread).as_mut_unchecked() };
    let handle = match t.handle.take() {
        Some(h) => h,
        None => {
            error!("wake_blocked_inline: tid={} has no handle", t.tid);
            return;
        }
    };
    let n = handle.ret_count();
    for i in 0..n {
        t.frame.regs[10 + i] = handle.ret(i) as usize;
    }
    drop(handle);
    t.state
        .store(ThreadState::Ready as usize, Ordering::Release);
    if push_ready_notice(thread_ptr).is_err() {
        error!(
            "READY_INBOX full on blocked-wake: tid={} — thread \
             marked Ready but not queued; will need a fallback path",
            t.tid,
        );
    }
}

/// Install [`wake_blocked_inline`] as the `process::completion`
/// wake hook. Called once at boot by `rust_main` so signal_n can
/// fire the kmain wake path without process needing to depend on
/// kmain.
pub fn install_completion_wake_hook() {
    process::completion::set_wake_hook(wake_blocked_inline);
}

/// Push `thread` onto the calling hart's `READY_INBOXES` slot. Used
/// by non-manager paths to publish a Ready transition without
/// touching `Orbit::ready` (which is manager-only).
///
/// Must be called from a hart context (`sscratch` points at a valid
/// `HartContext`). Returns `Err` if the inbox is full — caller is
/// responsible for logging; the dropped notice means the thread is
/// `Ready` but not queued, and currently nothing rescues it (no
/// fallback scan exists post-Phase C). At cap=32 per hart this should
/// not realistically fire.
pub fn push_ready_notice(thread: *mut Thread) -> Result<(), ()> {
    let hart_id = unsafe {
        (riscv::register::sscratch::read() as *const HartContext)
            .as_ref_unchecked()
            .hart_id as usize
    };
    if hart_id >= shootdown::MAX_HARTS {
        error!("push_ready_notice: hart_id={} >= MAX_HARTS", hart_id);
        return Err(());
    }
    READY_INBOXES[hart_id]
        .push(ReadyNotice { thread })
        .map_err(|_| ())
}

/// PLIC IRQ handler for e1000 RX/TX events. Wired up in
/// [`Orbit::setup_igb`] to whichever PLIC source the QEMU virt
/// PCI swizzle assigns to the device's slot.
///
/// Runs in trap context with SIE=0. Two responsibilities:
///  1. Ack the device's IRQ line by reading ICR. Without this, the
///     INTx line stays asserted and PLIC re-claims us in a tight
///     loop in `plic::dispatch`.
///  2. Push a `WakeEvent::Net` so the manager wakes k_net at the
///     next scheduler scan
/// Drops the wake event silently if `WAKE_QUEUE` is full. Cap is 64;
/// at e1000 burst rates (~1 IRQ per 1446 B at 1 Gbps = 86k IRQ/s)
/// the manager will drain faster than we fill, but a temporarily-
/// stalled manager would just lose redundant wakes — k_net's 10 ms
/// heartbeat is the safety net.
pub fn e1000_plic_handler(src: u32) {
    let icr = crate::drivers::e1000::ack_irq_static();
    let pushed = WAKE_QUEUE.push(WakeEvent::Net).is_ok();
    let hart_id = unsafe {
        (riscv::register::sscratch::read() as *const HartContext)
            .as_ref_unchecked()
            .hart_id
    };
    trace!(
        "e1000 IRQ: cpu{} src={} icr={:#010x} wake_pushed={}",
        hart_id, src, icr, pushed,
    );
}

pub struct Orbit {
    dtb_addr: usize,
    _serial_addr: usize,
    pub cpu_count: usize,
    satp: Satp,
    layout: KernelLayout,

    current_process_id: u16,
    current_thread_id: u32,

    processes: BTreeMap<u16, Process>,
    threads: BTreeMap<u32, PThread>,

    table_pages: memmap::TablePages,
    kernel_pages: memmap::KernelPages,
    user_pages: memmap::UserPages,

    net_pkg: NetPackage,
    /// TID of the k_net kernel thread, set by `setup_igb` once it
    /// spawns. `None` until then, and during the boot window before
    /// e1000 PLIC IRQs can fire — `WakeEvent::Net` consumers fall
    /// back to a coarse "wake all kernel threads" scan in that
    /// window. Once latched, `WakeEvent::Net` targets exactly this
    /// tid so unrelated kernel threads (k_gpu) aren't woken
    /// spuriously by every netch tickle.
    net_thread_tid: Option<u32>,
    orphaned_sockets: Vec<SocketHandle>,

    /// Min-heap of `(wake_time, sleep_seq, *mut Thread)` for Suspended
    /// sleepers. Manager-only; populated each pass by draining
    /// `SLEEP_INBOX`. Replaces the per-pass O(N_threads) Suspended
    /// walk in `get_runnable_thread` with O(woken) at dispatch time.
    /// See [orbit-core/src/sleep_heap.rs].
    sleeping: SleepHeap,

    /// FIFO of runnable threads. Manager-only. Populated by:
    ///   * `drain_ready_inboxes` (per-hart inboxes — non-manager
    ///     Ready transitions: preempted threads, signal_n's wake
    ///     hook for unblocked threads).
    ///   * `drain_sleeps` (sleep-heap promotion).
    ///   * `set_wake_reason_where` (eager Suspended → Ready).
    ///   * thread creation paths.
    /// Drained by `get_runnable_thread` via `pop_for(hart_mask)`.
    ready: ReadyQueue,

    /// Per-process handle tables. The manager's strong refs on
    /// `SharedUserPtr`-backed resources live here, keyed by the u32 Fd
    /// assigned at creation. k_net gets separate clones via
    /// `SocketReq`. On process exit the table is walked to revoke
    /// every Shared mapping before the manager drops its Arcs.
    process_handles: BTreeMap<u16, ProcessHandles>,

    /// §13a.5 — futex wait queues keyed on the *physical* page+offset
    /// of `uaddr`. Two threads in different processes that mapped the
    /// same shared frame end up under the same key, so a single
    /// `futex_wake` reaches them both. Manager-only; mutated under
    /// `MANAGER_LOCK`. v1 has no timeout scan — `timeout_ns` is
    /// captured but ignored (waiters block until woken or until
    /// their owning process exits).
    futex_waiters: BTreeMap<u64, Vec<FutexWaiter>>,

    /// Kernel-wide denial event ring. Receives `DenialEvent::PermDeny`
    /// from the manager-side `drain_denial_events` pass (which folds
    /// events off [`DENIAL_EVENT_QUEUE`]) and `DenialEvent::RoleDeny`
    /// from `create_process_v2`'s role-transition gate (manager-side
    /// inline push). Both pushers hold `MANAGER_LOCK`. Snapshotted
    /// by `query_denial_log` under the same lock. Bounded at
    /// `DENIAL_RING_CAPACITY` (50).
    denial_ring: orbit_core::denial_ring::DenialRing,
}

/// One slot on a futex wait queue. Captured at `futex_wait` request
/// time; consumed by `futex_wake` (signal `0`) or by `dealloc_process`
/// when the calling thread's process exits before a wake arrives
/// (signal `-ESRCH`, which the unblock path turns back into a
/// detectable errno on resume).
#[derive(Debug)]
pub struct FutexWaiter {
    pub handle: process::CompletionHandle,
    /// Pid of the parking thread. Used at `dealloc_process` time to
    /// find and signal-and-drop waiters whose owner is going away,
    /// so a futex queue keyed on a still-shared frame doesn't keep
    /// pointing at a freed `CompletionHandle`'s consumer.
    pub pid: u16,
    /// Reserved: absolute tick deadline for `-ETIMEDOUT`. `0` = no
    /// timeout. v1 always parks `0` regardless of the user-supplied
    /// `timeout_ns` (the timeout-scan path lands when std::sync needs
    /// it).
    pub deadline_ticks: u64,
}

impl Orbit {
    const THREAD_STACK_LAYOUT: Layout =
        unsafe { Layout::from_size_align_unchecked(2 * MB as usize, 2 * MB as usize) };

    const THREAD_TRAP_FRAME_LAYOUT: Layout =
        unsafe { Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE) };

    const TABLE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE) };

    /// Physical address we program into the e1000's BAR0; the device decodes
    /// MMIO accesses to this PA on the bus. The kernel reaches the same
    /// region through a high-half KMMIO alias allocated at setup_igb.
    pub const IGB_BAR_PA: u64 = 0x4000_0000;

    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Snapshot per-process and kernel-wide accounting for `pid`.
    /// Phase 1 covers memory only — time-related fields (cpu_ticks,
    /// syscall_ticks, hart_*_ticks, context_switches, syscalls) read
    /// as 0 until the per-hart bucket state machine lands.
    ///
    /// Returns `None` if `pid` doesn't name a live process. The caller
    /// must hold `MANAGER_LOCK` (or accept slightly stale reads) — we
    /// walk `Process::heap_pages` and `Process::maps`, both of which
    /// the manager mutates under that lock.
    pub fn snapshot_process_stats(&self, pid: u16) -> Option<orbit_abi::stats::ProcessStats> {
        let proc = self.processes.get(&pid)?;

        let heap_bytes: u64 = proc
            .heap_pages
            .iter()
            .map(|b| b.layout().size() as u64)
            .sum();

        // Resident = sum of mapped (backing != None) VMA lengths.
        // Guard reservations and bare-VA holes are excluded.
        let resident_bytes: u64 = proc
            .maps
            .values()
            .filter(|m| m.backing.is_some())
            .map(|m| m.len)
            .sum();

        // Per-thread accumulator sums. `process.threads` holds tids;
        // each maps via `self.threads` to a `PThread` (raw ptr to a
        // Box-leaked Thread). Foreign-hart reads are racy but
        // tear-safe via the per-field atomics.
        let mut cpu_ticks: u64 = 0;
        let mut context_switches: u64 = 0;
        let mut syscalls: u64 = 0;
        let mut syscall_ticks: u64 = 0;
        for tid in &proc.threads {
            if let Some(pt) = self.threads.get(tid) {
                let t: &Thread = unsafe { (pt.0 as *const Thread).as_ref_unchecked() };
                cpu_ticks = cpu_ticks.wrapping_add(t.cpu_ticks_total.load(Ordering::Relaxed));
                context_switches =
                    context_switches.wrapping_add(t.context_switches.load(Ordering::Relaxed));
                syscalls = syscalls.wrapping_add(t.syscall_count.load(Ordering::Relaxed));
                syscall_ticks = syscall_ticks.wrapping_add(t.syscall_ticks.load(Ordering::Relaxed));
            }
        }

        // System-wide hart-bucket sums (every hart contributes).
        use crate::kernel::accounting::sum_hart_counter;
        let hart_user_ticks = sum_hart_counter(|h| h.user_ticks.load(Ordering::Relaxed));
        let hart_kernel_ticks = sum_hart_counter(|h| h.kernel_ticks.load(Ordering::Relaxed));
        let hart_scheduler_ticks = sum_hart_counter(|h| h.scheduler_ticks.load(Ordering::Relaxed));
        let hart_idle_ticks = sum_hart_counter(|h| h.idle_ticks.load(Ordering::Relaxed));

        Some(orbit_abi::stats::ProcessStats {
            size: core::mem::size_of::<orbit_abi::stats::ProcessStats>() as u32,
            _reserved: 0,
            pid: proc.pid,
            thread_count: proc.thread_count,
            _pad0: 0,
            cpu_ticks,
            context_switches,
            syscalls,
            resident_bytes,
            heap_bytes,
            kernel_kpages_bytes: self.kernel_pages.allocated_bytes() as u64,
            kernel_user_pages_bytes: self.user_pages.allocated_bytes() as u64,
            kernel_ktables_bytes: self.table_pages.allocated_bytes() as u64,
            // KHEAP usage requires intercepting `#[global_allocator]`
            // — orthogonal to time accounting, deferred.
            kernel_heap_bytes: 0,
            syscall_ticks,
            hart_user_ticks,
            hart_kernel_ticks,
            hart_scheduler_ticks,
            hart_idle_ticks,
            perm_denials: proc.perm_denials.load(Ordering::Relaxed),
            role_denials: proc.role_denials.load(Ordering::Relaxed),
        })
    }

    pub fn runnable_thread_count(&self) -> usize {
        self.threads
            .iter()
            .filter(|(_, t)| unsafe {
                let thread = (t.0 as *const Thread).as_ref_unchecked();
                thread.state.load(Ordering::Acquire) == ThreadState::Ready as usize
            })
            .count()
    }

    pub const fn new(
        dtb_addr: usize,
        _serial_addr: usize,
        cpu_count: usize,
        layout: KernelLayout,
        table_pages: memmap::TablePages,
        kernel_pages: memmap::KernelPages,
        user_pages: memmap::UserPages,
        satp: Satp,
    ) -> Self {
        Self {
            dtb_addr,
            _serial_addr,
            table_pages,
            kernel_pages,
            user_pages,
            cpu_count,
            satp,
            layout,
            current_process_id: 0,
            current_thread_id: 0,
            processes: BTreeMap::new(),
            threads: BTreeMap::new(),
            net_thread_tid: None,
            net_pkg: NetPackage {
                phy: None,
                iface: None,
                socket_reqs: alloc::vec::Vec::new(),
                socket_associations: heapless::spsc::Queue::new(),
                socket_deletions: heapless::spsc::Queue::new(),
            },
            orphaned_sockets: Vec::new(),
            sleeping: SleepHeap::new(),
            ready: ReadyQueue::new(),
            process_handles: BTreeMap::new(),
            futex_waiters: BTreeMap::new(),
            denial_ring: orbit_core::denial_ring::DenialRing::new(),
        }
    }

    /// Push a `DenialEvent` onto the kernel-wide ring. Caller must
    /// already hold `MANAGER_LOCK` — the ring is not internally
    /// synchronised. Used by `drain_denial_events` (folding events
    /// off [`DENIAL_EVENT_QUEUE`]) and by `create_process_v2`'s
    /// role-transition gate when it logs a `RoleDeny` inline.
    pub fn push_denial_event(&mut self, event: orbit_abi::denial::DenialEvent) {
        use orbit_abi::denial::DenialSink;
        self.denial_ring.push(event);
    }

    /// Snapshot the denial ring into `buf` in chronological order
    /// (oldest first). Returns the number of events written. Caller
    /// must hold `MANAGER_LOCK` since the ring is mutated by the
    /// manager-side handlers without per-ring synchronisation.
    pub fn denial_ring_snapshot(&self, buf: &mut [orbit_abi::denial::DenialEvent]) -> usize {
        self.denial_ring.snapshot(buf)
    }

    /// Allocate a kthread stack. Kernel-accessible (Shared pool) so the
    /// kernel can write through KDMAP during setup.
    fn allocate_thread_stack(&mut self) -> Result<(Frame<Shared>, memmap::KdmapVa), ()> {
        self.kernel_pages
            .alloc_kdmap(Self::THREAD_STACK_LAYOUT)
            .ok_or_else(|| {
                error!("failed to allocate new thread stack");
            })
    }

    /// Allocate a user thread stack. `user_pages` has no KDMAP alias in
    /// the kernel satp — setup-time zeroing goes through `UserPageWindow`.
    fn allocate_user_thread_stack(
        &mut self,
        stack_size: u64,
    ) -> Result<(Frame<UserOnly>, Layout), ()> {
        let layout = Layout::from_size_align(stack_size as usize, UPROC_STACK_GRAIN as usize)
            .map_err(|e| {
                error!("bad user stack layout for size={stack_size}: {e:?}");
            })?;
        let frame = self.user_pages.alloc_pa(layout).ok_or_else(|| {
            error!("failed to allocate user thread stack size={stack_size}");
        })?;

        // Zero before the PTE install exposes the stack to user code — the
        // page may have been returned by a previous process.
        unsafe {
            let mut w = user_page::UserPageWindow::map(frame.get_raw(), layout.size());
            w.as_mut_slice().fill(0);
        }

        Ok((frame, layout))
    }

    /// Allocate a trap-frame page (Shared pool, kernel-writable via KDMAP).
    fn allocate_trap_frame(&mut self) -> Result<(Frame<Shared>, memmap::KdmapVa), ()> {
        self.kernel_pages
            .alloc_kdmap(Self::THREAD_TRAP_FRAME_LAYOUT)
            .ok_or_else(|| {
                error!("failed to allocate new trap frame");
            })
    }

    /// Allocate a fresh page table from `table_pages` and return a
    /// `RootTable` view on it. The page is zeroed before handoff.
    fn create_new_page_table(
        &mut self,
    ) -> Result<(Frame<process::Table>, mmu::mmap::RootTable<'static>), ()> {
        let (frame, kva) = self.table_pages.alloc(Self::TABLE_LAYOUT).ok_or_else(|| {
            error!("failed to allocate new page table");
        })?;
        unsafe {
            core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, PAGE_SIZE);
            let table = kva.as_ptr::<PageTable>().as_ref_unchecked();
            Ok((frame, memmap::kernel_root(table)))
        }
    }

    /// Mask covering `[0, cpu_count)`. Used as the default `allowed_affinity`
    /// for every newly-spawned thread; restricted callers (kthreads pinned
    /// to a single hart, future capability-style child processes) override
    /// at construction.
    pub fn all_harts_mask(&self) -> u64 {
        if self.cpu_count >= 64 {
            u64::MAX
        }
        else {
            (1u64 << self.cpu_count) - 1
        }
    }

    pub fn create_kernel_thread(
        &mut self,
        entrypoint: usize,
        a0: Option<usize>,
    ) -> Result<u32, ()> {
        let (stack_frame, stack_kva) = self.allocate_thread_stack()?;

        let (trap_frame_frame, trap_frame_kva) = match self.allocate_trap_frame() {
            Ok(p) => p,
            Err(_) => {
                self.kernel_pages
                    .free(stack_frame, Self::THREAD_STACK_LAYOUT);
                error!("failed to alloc trap_frame for kthread");
                return Err(());
            }
        };

        let pid = 0;
        let tid = self.next_tid();

        let (frame, stack) = unsafe {
            let f = trap_frame_kva.as_mut_ptr::<TrapFrame>();
            core::ptr::write_bytes(f as *mut u8, 0, PAGE_SIZE);

            let s = stack_kva.as_mut_ptr::<Stack>();
            core::ptr::write_bytes(s as *mut u8, 0, 2 * MB as usize);

            (f.as_mut_unchecked(), s.as_mut_unchecked())
        };

        frame.regs[1] = entrypoint;
        frame.regs[2] = stack_kva.raw() as usize + Self::THREAD_STACK_LAYOUT.size();
        frame.regs[10] = a0.unwrap_or(0);
        frame.asid = 0;

        let all_harts = self.all_harts_mask();
        let kthread = Thread {
            pc: AtomicUsize::new(entrypoint),
            satp: self.satp,
            mode: SPP::Supervisor,
            tid,
            pid,
            ticks: 0,
            frame,
            stack,
            kernel_stack: Some(stack_frame),
            kernel_trap_frame: Some(trap_frame_frame),
            state: AtomicUsize::new(ThreadState::Ready as usize),
            wake_time: 0,
            wake_override: AtomicU64::new(0),
            last_wake_reason: AtomicU64::new(0),
            sleep_seq: AtomicU64::new(0),
            handle: None,
            slot: None,
            fault_info: None,
            allowed_affinity: all_harts,
            affinity: AtomicU64::new(all_harts),
            cpu_ticks_total: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            syscall_count: AtomicU64::new(0),
            syscall_ticks: AtomicU64::new(0),
            // Kernel threads run in S-mode and never reach the
            // dispatch-site permission gate (cause=8 is U-mode by
            // construction). Stamp ZERO so a future bug that did
            // route a kthread through the gate fails closed rather
            // than silently running with full caps.
            permissions: orbit_abi::perms::Permissions::ZERO,
        };

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(kthread);
        let tptr = Box::into_raw(t);
        info!("created kthread@{:016X?}", tptr);

        self.threads.insert(tid, PThread(tptr));
        // Constructor sets state=Ready; surface to the scheduler by
        // pushing onto self.ready directly (we're in manager context).
        self.ready.push(tptr);

        Ok(tid)
    }

    fn run_mmap_req(&mut self, req: MemMapReq, pid: u16, root_pa: PhysAddr) -> isize {
        info!("handling mmap req {req:08X?}");

        let Some(orbit_core::manager::MappingGeometry { align, levels }) =
            orbit_core::manager::select_mapping_geometry(req.vaddr.raw() as usize, req.size)
        else {
            error!("failed to select alignment for mmap req: {req:?}");
            return Errno::new(EINVAL).to_ret();
        };

        let size = req.size;

        let layout = match Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(e) => {
                error!("failed to create alignment for mmap req: {e:?}");
                return Errno::new(EINVAL).to_ret();
            }
        };

        // Shared mmaps stay in kernel_pages so the kernel (net thread,
        // deferred handlers) can deref through KDMAP long after setup.
        // Private mmaps go to user_pages; kernel has no long-lived alias.
        // The two branches produce different typed frames — normalize to
        // (backing_pa_raw, PhysBacking) at the end.
        let (backing_pa, backing) = if req.share_with_kernel {
            let Some(frame) = self.kernel_pages.alloc_pa(layout)
            else {
                error!("failed to alloc shared pages for mmap req: {req:?}");
                return Errno::new(ENOMEM).to_ret();
            };

            // Zero via KDMAP alias.
            unsafe {
                let kva = frame.to_kdmap();
                core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, layout.size());
            }
            (frame.raw(), PhysBacking::Shared { frame, layout })
        }
        else {
            let Some(frame) = self.user_pages.alloc_pa(layout)
            else {
                error!("failed to alloc user pages for mmap req: {req:?}");
                return Errno::new(ENOMEM).to_ret();
            };

            // Zero via a transient kernel window — no KDMAP alias exists.
            unsafe {
                let mut w = user_page::UserPageWindow::map(frame.get_raw(), layout.size());
                w.as_mut_slice().fill(0);
            }
            (frame.raw(), PhysBacking::User { frame, layout })
        };

        let supervisor_tag = if req.share_with_kernel {
            SupervisorTag::SharedRevocable
        }
        else {
            SupervisorTag::None
        };

        let config = MappingConfig {
            permissions: (req.page_permissions & 0xE) | PagePermissions::U,
            levels,
            page_size: align as u64,
            vaddr: VirtAddr::new(req.vaddr.raw()),
            paddr: backing_pa,
            log: false,
            supervisor_tag,
        };

        let vend = VirtAddr::new(req.vaddr.raw() + req.size as u64);
        let pend = PhysAddr::new(backing_pa.get_raw() + req.size as u64);

        unsafe {
            let root_table = memmap::kernel_root_from_pa(root_pa);

            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

            if let Err(_) = map_address_range(&root_table, &mut pages, &config, vend, pend) {
                error!("failed to map pages for mmap req: {req:?}");
                self.free_backing(backing);
                return Errno::new(ENOMEM).to_ret();
            }
        }

        let owning_process = match self.processes.get_mut(&pid) {
            Some(proc) => proc,
            None => {
                error!("failed to add pages to process metadata (no pid): {req:?}");
                self.free_backing(backing);
                return Errno::new(ESRCH).to_ret();
            }
        };

        owning_process.heap_pages.push(backing);

        core::sync::atomic::fence(Ordering::SeqCst);

        // Local single-VA fence handles the manager hart. Cross-hart
        // broadcast (whole-TLB sentinel via len=0) covers every other
        // hart that may have cached a negative entry for this newly-
        // mapped range.
        riscv::asm::sfence_vma(pid as usize, req.vaddr.raw() as usize);
        crate::kernel::shootdown::broadcast(0, 0);

        info!("fulfilled {req:?}:\n\tpa={backing_pa:016X?} {layout:08X?}");

        0
    }

    /// Dispatch a single typed free based on the backing's pool variant.
    fn free_backing(&mut self, backing: PhysBacking) {
        match backing {
            PhysBacking::Shared { frame, layout } => self.kernel_pages.free(frame, layout),
            PhysBacking::User { frame, layout } => self.user_pages.free(frame, layout),
        }
    }

    /// Run an enqueued NetChannel creation. Returns `(vaddr, fd)` on
    /// success — the manager forwards both via `signal_pair`. Negative
    /// `vaddr` on the error path; `fd` is unused in that case.
    fn run_nc_create_req(
        &mut self,
        req: NetChannelCreationReq,
        pid: u16,
        root_pa: PhysAddr,
    ) -> (isize, isize) {
        info!("handling nc creation req: {req:08X?}");

        let Some(region_size) = NetChannel::normalize_region_size(req.region_size)
        else {
            warn!("nc create: bad region_size {}", req.region_size);
            return (Errno::new(EINVAL).to_ret(), 0);
        };

        if req.nc_vaddr.raw() % PAGE_SIZE as u64 != 0 {
            warn!("nc create: unaligned user vaddr 0x{:X}", req.nc_vaddr);
            return (Errno::new(EINVAL).to_ret(), 0);
        }

        let layout = match Layout::from_size_align(region_size, PAGE_SIZE) {
            Ok(l) => l,
            Err(e) => {
                warn!("nc create: bad layout {e:?}");
                return (Errno::new(EINVAL).to_ret(), 0);
            }
        };

        // NetChannel lives in kpages (Shared pool) so the kernel can drive
        // smoltcp through the KDMAP alias after creation.
        let Some((frame, kva)) = self.kernel_pages.alloc_kdmap(layout)
        else {
            warn!("nc create: alloc failed for {} bytes", region_size);
            return (Errno::new(ENOMEM).to_ret(), 0);
        };

        // Zero then init before the user PTE exists — user never observes a
        // half-initialized NetChannel, and previous tenant bytes can't leak.
        unsafe {
            core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, region_size);
            NetChannel::init(kva.as_mut_ptr::<u8>(), region_size);
        }

        let config = MappingConfig {
            permissions: (PagePermissions::R as u64)
                | (PagePermissions::W as u64)
                | (PagePermissions::U as u64),
            levels: 4,
            page_size: PAGE_SIZE as u64,
            vaddr: VirtAddr::new(req.nc_vaddr.raw()),
            paddr: frame.raw(),
            log: false,
            supervisor_tag: SupervisorTag::SharedRevocable,
        };

        let vend = VirtAddr::new(req.nc_vaddr.raw() + region_size as u64);
        let pend = PhysAddr::new(frame.get_raw() + region_size as u64);

        unsafe {
            let root_table = memmap::kernel_root_from_pa(root_pa);
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

            if map_address_range(&root_table, &mut pages, &config, vend, pend).is_err() {
                warn!("nc create: map failed {req:?}");
                self.kernel_pages.free(frame, layout);
                return (Errno::new(ENOMEM).to_ret(), 0);
            }
        }

        if !self.processes.contains_key(&pid) {
            warn!("nc create: no owning process {req:?}");
            self.kernel_pages.free(frame, layout);
            return (Errno::new(ESRCH).to_ret(), 0);
        }

        // Frame ownership moves into the SharedUserPtr's Arc — not into
        // `proc.heap_pages`, which would double-free on teardown. The
        // Arc's last drop pushes to `pending_frees`; the manager returns
        // it to `kernel_pages` during cleanup.
        let shared: SharedUserPtr<NetChannel> =
            SharedUserPtr::new(frame, layout, req.nc_vaddr, region_size, pid);

        // Register the manager's strong ref and grab the Fd. Return it
        // to the user in a1 alongside the VA in a0 — avoids taking a
        // user out-pointer, which would have to resolve through KDMAP
        // (Shared-pool only) or a transient UserPageWindow, neither of
        // which is worth the machinery for 4 bytes.
        let fd = self
            .process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new)
            .insert(Handle::NetChannel(shared.clone()));

        core::sync::atomic::fence(Ordering::SeqCst);

        // Local whole-asid + cross-hart whole-TLB broadcast — same
        // shape as run_mmap_req's post-install fence.
        riscv::asm::sfence_vma(pid as usize, 0);
        crate::kernel::shootdown::broadcast(0, 0);

        let socket_req = SocketReq {
            netchan: shared,
            nc_type: req.nc_type,
            pid,
            ctx: net_channel::ChannelCtx::new(req.bind),
        };

        if let Some(np) = self
            .net_pkg
            .socket_reqs
            .get_mut(get_hart_context().hart_id as usize)
        {
            if let Err(e) = np.enqueue(socket_req) {
                warn!("nc create: failed to queue socket req {e:?}");
                return (Errno::new(EAGAIN).to_ret(), 0);
            }
        }

        info!(
            "nc created user_va=0x{:08X} kva=0x{:016X} region={} fd={}",
            req.nc_vaddr,
            kva.raw(),
            region_size,
            fd
        );
        (req.nc_vaddr.raw() as isize, fd as isize)
    }

    fn run_close_req(&mut self, req: CloseHandleReq, pid: u16, root_pa: PhysAddr) -> isize {
        info!("handling close req: {req:?}");

        // Look up the handle, revoke if Shared, then drop the Arc.
        // k_net may still hold a clone; the backing lives until it's
        // dropped too. Post-revoke, any user access to the old VA
        // faults, and `try_as_ref` returns None for future kernel
        // observers — close is safe to race against an in-flight
        // update_tcp on another hart.
        let Some(ph) = self.process_handles.get_mut(&pid)
        else {
            return Errno::new(EBADF).to_ret();
        };
        let Some(handle) = ph.remove(req.fd)
        else {
            return Errno::new(EBADF).to_ret();
        };

        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

        match handle {
            Handle::NetChannel(sup) => {
                if let Err(e) = sup.revoke(&root_table) {
                    warn!(
                        "close_handle: revoke failed for fd={} sup={sup:?}: {e:?}",
                        req.fd
                    );
                    return Errno::new(EIO).to_ret();
                }
                // sup drops here — Arc release. Same shape as before
                // the by-value match.
            }
            Handle::File(of) => {
                // The scratch SharedFrame drops with `of` here.
                // If a DMA is in flight (loading=true), the
                // CopyDescriptor in the virtio-blk slot table holds
                // another clone; the page survives until the
                // manager finishes the post-DMA copy. Otherwise the
                // last clone drops here and the page goes to
                // pending_frees.
                drop(of);
            }
        }
        0
    }

    /// Copy `len` bytes of a user path string into a kernel-side
    /// buffer. Caller has already enforced `len <= MAX_FS_PATH_LEN`
    /// at the syscall boundary so this stays bounded. Returns the
    /// path as a `&str` borrowed from `out`, or an errno on failure.
    fn copy_user_path<'a>(
        &mut self,
        root_pa: PhysAddr,
        vaddr: u64,
        len: usize,
        out: &'a mut [u8; MAX_FS_PATH_LEN],
    ) -> Result<&'a str, isize> {
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let mut copied = 0;
        while copied < len {
            let cursor = vaddr + copied as u64;
            let page_base = cursor & !(PAGE_SIZE as u64 - 1);
            let page_off = (cursor - page_base) as usize;
            let take = core::cmp::min(PAGE_SIZE - page_off, len - copied);
            let pa = unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) }
                .ok_or(Errno::new(EFAULT).to_ret())?;
            unsafe {
                let mut w = user_page::UserPageWindow::map(pa as u64, PAGE_SIZE);
                let page = w.as_mut_slice();
                out[copied..copied + take].copy_from_slice(&page[page_off..page_off + take]);
            }
            copied += take;
        }
        core::str::from_utf8(&out[..len]).map_err(|_| Errno::new(EINVAL).to_ret())
    }

    fn run_fs_open_req(&mut self, req: FsOpenReq, pid: u16, root_pa: PhysAddr) -> isize {
        let Some(fs) = crate::kernel::fs::mounted()
        else {
            warn!("fs_open: no mounted filesystem");
            return Errno::new(EIO).to_ret();
        };
        let mut path_buf = [0u8; MAX_FS_PATH_LEN];
        let path =
            match self.copy_user_path(root_pa, req.path_vaddr.raw(), req.path_len, &mut path_buf) {
                Ok(p) => p,
                Err(e) => return e,
            };
        let inode = match fs.open(path) {
            Ok(i) => i,
            Err(FsErr::NotFound) => return Errno::new(orbit_abi::errno::ENOENT).to_ret(),
            Err(_) => return Errno::new(EIO).to_ret(),
        };
        // Allocate the per-fd sector cache for regular files. Stat
        // tells us the kind. For directories the scratch is `None` —
        // dir reads go through `fs_readdir`, which has its own
        // page-aligned out-buffer in user space and doesn't need the
        // bounce.
        let stat = match fs.stat(inode) {
            Ok(s) => s,
            Err(_) => return Errno::new(EIO).to_ret(),
        };
        let is_reg = (stat.st_mode & orbit_abi::fs::S_IFMT) == orbit_abi::fs::S_IFREG;
        let scratch = if is_reg {
            let layout = match Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
                Ok(l) => l,
                Err(_) => return Errno::new(EIO).to_ret(),
            };
            let frame = match crate::kernel::shared_frame::SharedFrame::alloc(
                &mut self.kernel_pages,
                layout,
            ) {
                Some(f) => f,
                None => {
                    error!("fs_open: kernel_pages exhausted; cannot allocate scratch");
                    return Errno::new(orbit_abi::errno::ENOMEM).to_ret();
                }
            };
            Some(crate::kernel::handle::ScratchSector {
                frame,
                cached_sector: u64::MAX,
                valid_bytes: 0,
                loading: false,
            })
        } else {
            None
        };
        // Lazy-create the handle table — same pattern create_netch
        // uses, since a process that opens a file before ever creating
        // a NetChannel won't have an entry yet.
        let fd = self
            .process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new)
            .insert(Handle::File(OpenFile {
                fs,
                inode,
                offset: 0,
                dir_cursor: 0,
                scratch
            }));
        info!("fs_open: pid={pid} path={path} → fd={fd} ino={inode}");
        fd as isize
    }

    /// Owns the `handle` it's passed: either consumes it into a
    /// `WorkNotification::Bounce` (async — IRQ + manager FsReadCopy
    /// signal it) or signals it inline with the appropriate value.
    /// Caller must not signal again.
    fn run_fs_read_req(
        &mut self,
        req: FsReadReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: process::CompletionHandle,
    ) {
        const SECTOR: u64 = 512;

        // Look up the file handle and snapshot what we need.
        let Some(ph) = self.process_handles.get_mut(&pid)
        else {
            handle.signal(Errno::new(EBADF).to_ret());
            return;
        };
        let Some(handle_ref) = ph.get_mut(req.fd)
        else {
            handle.signal(Errno::new(EBADF).to_ret());
            return;
        };
        let Handle::File(of) = handle_ref
        else {
            handle.signal(Errno::new(EBADF).to_ret());
            return;
        };

        let fs = of.fs;
        let inode = of.inode;
        let prev_off = of.offset;

        let file_size = match fs.size(inode) {
            Ok(s) => s,
            Err(_) => {
                handle.signal(Errno::new(EIO).to_ret());
                return;
            }
        };
        if prev_off >= file_size {
            // EOF — sync signal 0; don't touch the device.
            handle.signal(0);
            return;
        }

        // Buffer constraints already enforced at the syscall layer
        // (one-page span, len > 0, user_range_ok). Resolve the user
        // PA once.
        let buf_va = req.buf_vaddr.raw();
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = buf_va & !(PAGE_SIZE as u64 - 1);
        let user_page_off = (buf_va - page_base) as u32;
        let user_page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => {
                    handle.signal(Errno::new(EFAULT).to_ret());
                    return;
                }
            };

        // Compute the relevant slice of the current sector. `intra`
        // is the byte offset within the sector that we want to start
        // reading from; `visible_len` is the most we can return in
        // this call (clipped at the sector boundary and at EOF).
        let target_sector = prev_off / SECTOR;
        let intra = (prev_off & (SECTOR - 1)) as u32;
        let sector_remaining = SECTOR - intra as u64;
        let file_remaining = file_size - prev_off;
        let visible_len = core::cmp::min(
            req.len as u64,
            core::cmp::min(sector_remaining, file_remaining),
        ) as u32;

        // Take the scratch by &mut so we can flip `loading` in
        // place. Regular files always have a scratch (allocated at
        // fs_open); directories don't, but hitting fs_read on a
        // directory should surface as EISDIR through the FS, not
        // panic here.
        let scratch = match &mut of.scratch {
            Some(s) => s,
            None => {
                handle.signal(Errno::new(orbit_abi::errno::EISDIR).to_ret());
                return;
            }
        };

        // EAGAIN any concurrent reader while a DMA is in flight on
        // this fd — we don't have a per-fd waiter queue in v1, and
        // we must not submit a second DMA into the same scratch
        // page. The `loading` check has to come *before* the
        // cache-hit check: while loading is true `cached_sector` may
        // still hold the previous valid sector, but the scratch
        // page is being overwritten, so cache hits would return torn
        // data.
        if scratch.loading {
            handle.signal(Errno::new(orbit_abi::errno::EAGAIN).to_ret());
            return;
        }

        // Cache hit: the scratch already holds `target_sector`.
        // Copy the slice into the user buffer through a
        // UserPageWindow (we hold MANAGER_LOCK here, so single-slot
        // serialisation is fine), advance offset, signal
        // synchronously.
        if scratch.cached_sector == target_sector {
            let src_kva = scratch.frame.kva().raw() + intra as u64;
            unsafe {
                let mut w = user_page::UserPageWindow::map(user_page_pa, PAGE_SIZE);
                let dst = w.as_mut_slice();
                let src = core::slice::from_raw_parts(src_kva as *const u8, visible_len as usize);
                dst[user_page_off as usize..user_page_off as usize + visible_len as usize]
                    .copy_from_slice(src);
            }
            of.offset = prev_off + visible_len as u64;
            handle.signal(visible_len as isize);
            return;
        }

        // Cache miss: arm the loading state, build the bounce
        // descriptor (cloning the SharedFrame so the in-flight
        // descriptor keeps the scratch page alive across
        // close_handle / process teardown), submit, park.
        // cached_sector / valid_bytes stay at whatever they were
        // (empty or the previous valid sector) — they get
        // published from run_fs_read_copy on success. Only
        // `loading` and `of.offset` move here, so the
        // submit-failure rollback below has minimal state to undo.
        let scratch_clone = scratch.frame.clone();
        let scratch_pa = scratch_clone.pa();
        let sector_off = target_sector * SECTOR;
        let valid_bytes = core::cmp::min(SECTOR, file_size - sector_off) as u32;
        let notif = WorkNotification::Bounce(crate::kernel::fs::CopyDescriptor {
            handle: handle.clone(),
            pid,
            fd: req.fd,
            target_sector,
            valid_bytes,
            scratch: scratch_clone,
            user_page_pa: PhysAddr::new(user_page_pa),
            user_page_off,
            intra,
            len: visible_len,
        });

        scratch.loading = true;
        of.offset = prev_off + visible_len as u64;

        match unsafe {
            fs.read_async(inode, sector_off, SECTOR as u32, scratch_pa.get_raw(), notif)
        } {
            Ok(()) => {
                // The notification (and its handle clone) is in the
                // virtio-blk slot table; the IRQ + FsReadCopy will
                // signal once the chain completes. Our retained
                // `handle` here drops with this scope, releasing
                // the manager-side Arc ref.
            }
            Err(e) => {
                // Submit failed — no FsReadCopy will ever fire to
                // clear `loading`, so undo the speculative state
                // here and signal the retained handle directly.
                if let Some(ph) = self.process_handles.get_mut(&pid)
                    && let Some(Handle::File(of)) = ph.get_mut(req.fd)
                {
                    of.offset = prev_off;
                    if let Some(s) = of.scratch.as_mut() {
                        s.loading = false;
                        // cached_sector / valid_bytes stay
                        // unchanged — the DMA never went out, so
                        // whatever was valid before still is.
                    }
                }
                let errno = match e {
                    FsErr::NotRegular => orbit_abi::errno::EISDIR,
                    FsErr::NotADirectory => ENOTDIR,
                    FsErr::BadInode => EBADF,
                    FsErr::BadRange => EINVAL,
                    FsErr::IoError => EIO,
                    FsErr::NotFound => orbit_abi::errno::ENOENT,
                };
                handle.signal(Errno::new(errno).to_ret());
            }
        }
    }

    /// Manager-side post-DMA completion for `fs_read`.
    ///
    /// `notif_ptr` is the same `Box<WorkNotification>` that
    /// `submit_blk_read` stashed in the virtio-blk slot table; the
    /// IRQ swapped it out of `IN_FLIGHT` and forwarded the pointer
    /// through `PendingWork::FsReadCopy` for us to unbox here.
    /// Always the `Bounce` variant — Direct is handled inline by
    /// the IRQ.
    ///
    /// Three cases:
    /// - `status != 0` (device error): clear `loading` on the fd
    ///   (if still alive), invalidate the cache to avoid serving
    ///   the partially-written sector, signal `-EIO`.
    /// - Process exited mid-flight: skip the user-page copy
    ///   (`user_page_pa` may have been reallocated to another
    ///   tenant). Signal anyway — the parked thread is gone, so
    ///   it's a no-op, but it releases the manager's clone.
    /// - Otherwise: copy scratch→user, publish cache state on the
    ///   fd if still alive (close_handle while DMA was in flight
    ///   leaves us with no fd to update, but the user buffer is
    ///   still mapped because the process is alive — the read
    ///   completes as a courtesy).
    fn run_fs_read_copy(&mut self, notif_ptr: usize, status: u8) {
        // SAFETY: `notif_ptr` was produced by `Box::into_raw` in
        // submit_blk_read for exactly this work item; the IRQ
        // forwarded it through PendingWork unchanged. This is the
        // unique consumer.
        let notif = unsafe {
            *Box::from_raw(notif_ptr as *mut crate::kernel::fs::WorkNotification)
        };
        let desc = match notif {
            crate::kernel::fs::WorkNotification::Bounce(d) => d,
            crate::kernel::fs::WorkNotification::Direct { .. } => {
                // Direct never reaches the manager — the IRQ
                // signals it inline. If we get here something
                // smuggled the wrong variant through PendingWork.
                error!("run_fs_read_copy: received WorkNotification::Direct — invariant violated");
                return;
            }
        };

        let process_alive = self.processes.contains_key(&desc.pid);
        let scratch_state = self
            .process_handles
            .get_mut(&desc.pid)
            .and_then(|ph| ph.get_mut(desc.fd))
            .and_then(|h| match h {
                Handle::File(of) => of.scratch.as_mut(),
                _ => None,
            });

        if status != 0 {
            // Block device reported failure. Clear `loading` so the
            // fd isn't permanently stuck at EAGAIN; conservatively
            // invalidate the cache (a partial-sector DMA may have
            // landed before the failure, so anything previously
            // valid is now suspect).
            if let Some(s) = scratch_state {
                s.loading = false;
                s.cached_sector = u64::MAX;
                s.valid_bytes = 0;
            }
            desc.handle.signal(-(orbit_abi::errno::EIO as isize));
            return;
        }

        if !process_alive {
            // Process exited mid-flight. The DMA landed in scratch
            // (still alive via the SharedFrame clone in `desc`), but
            // `user_page_pa` belonged to a tenant that is no longer
            // there — writing to it would clobber whoever owns those
            // physical pages now. Signal and return; desc drops →
            // SharedFrame clone drops → scratch returns to
            // pending_frees if this was the last clone.
            desc.handle.signal(-(orbit_abi::errno::EIO as isize));
            return;
        }

        // Success path. Copy scratch→user first, then publish cache
        // state. UserPageWindow is single-slot but we're under
        // MANAGER_LOCK, so no contention with other manager-side
        // copies.
        let scratch_kva = desc.scratch.kva().raw();
        unsafe {
            let mut w = user_page::UserPageWindow::map(desc.user_page_pa.get_raw(), PAGE_SIZE);
            let dst = w.as_mut_slice();
            let src = core::slice::from_raw_parts(
                (scratch_kva + desc.intra as u64) as *const u8,
                desc.len as usize,
            );
            dst[desc.user_page_off as usize..desc.user_page_off as usize + desc.len as usize]
                .copy_from_slice(src);
        }

        if let Some(s) = scratch_state {
            s.cached_sector = desc.target_sector;
            s.valid_bytes = desc.valid_bytes;
            s.loading = false;
        }
        // If scratch_state was None, the fd was closed while the
        // DMA was in flight. The read still completed (we wrote
        // into the user buffer); the cache update is moot because
        // there's no fd to consult it. desc drops here → final
        // SharedFrame clone drops → pending_frees.

        desc.handle.signal(desc.len as isize);
    }

    fn run_fs_stat_req(&mut self, req: FsStatReq, pid: u16, root_pa: PhysAddr) -> isize {
        let Some(fs) = crate::kernel::fs::mounted()
        else {
            return Errno::new(EIO).to_ret();
        };
        let mut path_buf = [0u8; MAX_FS_PATH_LEN];
        let path =
            match self.copy_user_path(root_pa, req.path_vaddr.raw(), req.path_len, &mut path_buf) {
                Ok(p) => p,
                Err(e) => return e,
            };
        let inode = match fs.open(path) {
            Ok(i) => i,
            Err(FsErr::NotFound) => return Errno::new(orbit_abi::errno::ENOENT).to_ret(),
            Err(_) => return Errno::new(EIO).to_ret(),
        };
        let stat = match fs.stat(inode) {
            Ok(s) => s,
            Err(_) => return Errno::new(EIO).to_ret(),
        };

        // Copy out the Stat struct. Fits inside one page (128 B), so
        // a single UserPageWindow does it. Cross-page case: same
        // single-buffer constraint as fs_read.
        let stat_bytes = unsafe {
            core::slice::from_raw_parts(
                &stat as *const _ as *const u8,
                core::mem::size_of::<orbit_abi::fs::Stat>(),
            )
        };
        let stat_va = req.stat_vaddr.raw();
        if (stat_va & (PAGE_SIZE as u64 - 1)) + stat_bytes.len() as u64 > PAGE_SIZE as u64 {
            return Errno::new(EINVAL).to_ret();
        }
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = stat_va & !(PAGE_SIZE as u64 - 1);
        let page_off = (stat_va - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            page[page_off..page_off + stat_bytes.len()].copy_from_slice(stat_bytes);
        }
        info!(
            "fs_stat: pid={pid} path={path} ino={inode} size={}",
            stat.st_size
        );
        0
    }

    /// `fs_readdir` handler — packs a chunk of directory entries into
    /// the user buffer, advances the fd's `dir_cursor`, returns
    /// bytes-written. `0` means end-of-directory.
    ///
    /// Same single-page constraint as `fs_stat`: the user buffer must
    /// fit inside one 4 KiB page so a single `UserPageWindow` covers
    /// the copy-out. The pure-syscall layer caps `len` at `PAGE_SIZE`
    /// before we get here.
    fn run_fs_readdir_req(&mut self, req: FsReaddirReq, pid: u16, root_pa: PhysAddr) -> isize {
        // Look up the file handle and snapshot what we need so we can
        // drop the &mut on `process_handles` before the page-window
        // map (which doesn't borrow the handle table, but keeping the
        // borrow scope tight is consistent with run_fs_read_req).
        let Some(ph) = self.process_handles.get_mut(&pid)
        else {
            return Errno::new(EBADF).to_ret();
        };
        let Some(handle_ref) = ph.get_mut(req.fd)
        else {
            return Errno::new(EBADF).to_ret();
        };
        let Handle::File(of) = handle_ref
        else {
            return Errno::new(EBADF).to_ret();
        };
        let fs = of.fs;
        let inode = of.inode;
        let cursor = of.dir_cursor;

        // Single-page constraint: buffer must fit inside one 4 KiB page.
        let buf_va = req.buf_vaddr.raw();
        if (buf_va & (PAGE_SIZE as u64 - 1)) + req.len as u64 > PAGE_SIZE as u64 {
            return Errno::new(EINVAL).to_ret();
        }
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = buf_va & !(PAGE_SIZE as u64 - 1);
        let page_off = (buf_va - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };

        // Pack into the user page directly. UserPageWindow gives us a
        // kernel-mapped alias for the user's frame; we slice the
        // range corresponding to `buf_va..buf_va+len` and hand it to
        // the FS.
        let (written, next_cursor) = unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            let dst = &mut page[page_off..page_off + req.len];
            match fs.readdir(inode, cursor, dst) {
                Ok(pair) => pair,
                Err(e) => {
                    let errno = match e {
                        FsErr::NotADirectory => ENOTDIR,
                        FsErr::BadInode => EBADF,
                        FsErr::BadRange => EINVAL,
                        FsErr::IoError => EIO,
                        FsErr::NotFound => orbit_abi::errno::ENOENT,
                        FsErr::NotRegular => ENOTDIR,
                    };
                    return Errno::new(errno).to_ret();
                }
            }
        };

        // Commit the cursor advance now that the FS reported success.
        if let Some(ph) = self.process_handles.get_mut(&pid)
            && let Some(Handle::File(of)) = ph.get_mut(req.fd)
        {
            of.dir_cursor = next_cursor;
        }
        info!(
            "fs_readdir: pid={pid} fd={} ino={inode} cursor={cursor}->{next_cursor} bytes={written}",
            req.fd
        );
        written as isize
    }

    /// §13a.3 — `create_process_ex`. Same shape as
    /// `run_create_process_req` plus the argv blob copy + map step.
    /// The blob is one page at most (cap enforced at the syscall
    /// boundary); copy it out via a single page walk, then after the
    /// child Process is spawned, allocate a fresh kernel_pages page,
    /// fix up the offset slots into absolute pointers (since the
    /// child's mapping is at the constant `USER_ARGV_BASE`), and
    /// install the page R+U+S in the child PT.
    fn run_create_process_ex_req(
        &mut self,
        req: CreateProcessExReq,
        parent_pid: u16,
        root_pa: PhysAddr,
    ) -> isize {
        const MAX_ELF_BYTES: usize = 4 * 1024 * 1024;
        if req.elf_len == 0 || req.elf_len > MAX_ELF_BYTES {
            return Errno::new(EINVAL).to_ret();
        }
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

        // Copy the ELF (same loop as run_create_process_req).
        let mut blob: Vec<u8> = Vec::with_capacity(req.elf_len);
        let mut copied = 0u64;
        let elf_len = req.elf_len as u64;
        while copied < elf_len {
            let cursor = req.elf_vaddr.raw() + copied;
            let page_base = cursor & !(PAGE_SIZE as u64 - 1);
            let page_off = (cursor - page_base) as usize;
            let take = core::cmp::min(PAGE_SIZE - page_off, (elf_len - copied) as usize);
            let pa = match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) }
            {
                Some(p) => p as u64,
                None => {
                    error!(
                        "create_process_ex: elf user va 0x{:X} does not translate",
                        page_base
                    );
                    return Errno::new(EFAULT).to_ret();
                }
            };
            unsafe {
                let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                let page = w.as_mut_slice();
                blob.extend_from_slice(&page[page_off..page_off + take]);
            }
            copied += take as u64;
        }

        // Copy argv blob (single page at most).
        let argv_bytes: Option<Vec<u8>> = if req.argv_len > 0 {
            let mut buf = Vec::with_capacity(req.argv_len);
            let mut argv_copied = 0u64;
            let argv_len = req.argv_len as u64;
            while argv_copied < argv_len {
                let cursor = req.argv_vaddr.raw() + argv_copied;
                let page_base = cursor & !(PAGE_SIZE as u64 - 1);
                let page_off = (cursor - page_base) as usize;
                let take = core::cmp::min(PAGE_SIZE - page_off, (argv_len - argv_copied) as usize);
                let pa =
                    match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) }
                    {
                        Some(p) => p as u64,
                        None => {
                            error!(
                                "create_process_ex: argv va 0x{:X} does not translate",
                                page_base
                            );
                            return Errno::new(EFAULT).to_ret();
                        }
                    };
                unsafe {
                    let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                    let page = w.as_mut_slice();
                    buf.extend_from_slice(&page[page_off..page_off + take]);
                }
                argv_copied += take as u64;
            }
            Some(buf)
        }
        else {
            None
        };

        // Copy envp blob (always one page; syscall layer already
        // bound-checked alignment and range when envp_vaddr != 0).
        let envp_bytes: Option<Vec<u8>> = if req.envp_vaddr.raw() != 0 {
            let pa = match unsafe {
                mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(req.envp_vaddr.raw()))
            } {
                Some(p) => p as u64,
                None => {
                    error!(
                        "create_process_ex: envp va 0x{:X} does not translate",
                        req.envp_vaddr,
                    );
                    return Errno::new(EFAULT).to_ret();
                }
            };
            let mut buf = Vec::with_capacity(PAGE_SIZE);
            unsafe {
                let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                buf.extend_from_slice(w.as_mut_slice());
            }
            Some(buf)
        }
        else {
            None
        };

        // Affinity validation, identical to run_create_process_req.
        let all_harts = self.all_harts_mask();
        let allowed = if req.allowed_affinity == 0 {
            all_harts
        }
        else {
            req.allowed_affinity
        };
        let affinity = if req.affinity == 0 {
            allowed
        }
        else {
            req.affinity
        };
        if allowed & !all_harts != 0 || affinity & !allowed != 0 || affinity == 0 {
            error!("create_process_ex: affinity validation failed");
            return Errno::new(EINVAL).to_ret();
        }

        let proc_components = ProcessComponents {
            elf_blob: &blob,
            stack_size: UPROC_STACK_DEFAULT,
            allowed_affinity: allowed,
            affinity,
            parent_pid,
            argv_bytes: argv_bytes.as_deref(),
            envp_bytes: envp_bytes.as_deref(),
            perms: None,
        };

        let pid = match self.create_new_process(proc_components) {
            Ok(pid) => pid,
            Err(()) => {
                error!("create_process_ex: create_new_process failed");
                return Errno::new(ENOEXEC).to_ret();
            }
        };

        info!(
            "create_process_ex: spawned pid={pid} parent={parent_pid} argv_len={} envp={}",
            req.argv_len,
            if envp_bytes.is_some() { "yes" } else { "no" },
        );
        pid as isize
    }

    /// `pledge(*const PermsRequest)`. Manager-side: copies the
    /// request struct from user memory, applies the narrowing to
    /// `Process.permissions`, and propagates the new value to every
    /// live thread of the process so the lock-free dispatch-site
    /// gate sees the narrower mask.
    ///
    /// Returns `0` on success; `-EFAULT` if the request VA doesn't
    /// translate; `-ESRCH` if the process record vanished mid-flight
    /// (defensive — can't happen on the live path since the caller
    /// is one of the process's threads).
    fn run_pledge_req(&mut self, req: orbit_core::PledgeReq, pid: u16, root_pa: PhysAddr) -> isize {
        use orbit_abi::perms::PermsRequest;

        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

        // Resolve the request VA to a PA and read the 16-byte struct
        // out via a single UserPageWindow. The struct is u64-aligned
        // so a misaligned read can't straddle the page boundary
        // unless `req_vaddr` itself was bad — the syscall layer
        // already enforced 8-byte alignment.
        let page_base = req.req_vaddr.raw() & !(PAGE_SIZE as u64 - 1);
        let page_off = (req.req_vaddr.raw() - page_base) as usize;
        let pa = match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
            Some(p) => p as u64,
            None => {
                error!("pledge: req va 0x{:X} does not translate", req.req_vaddr);
                return Errno::new(EFAULT).to_ret();
            }
        };

        let request: PermsRequest = unsafe {
            let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            let req_ptr = page.as_ptr().add(page_off) as *const PermsRequest;
            core::ptr::read_unaligned(req_ptr)
        };

        // Apply the pledge to the authoritative copy on Process,
        // snapshot the result, then walk every live thread of the
        // process to refresh its `Thread.permissions` cache. The
        // two-step is needed because we can't hold &mut Process
        // while iterating `self.threads`.
        let new_perms = {
            let proc = match self.processes.get_mut(&pid) {
                Some(p) => p,
                None => {
                    error!("pledge: pid={pid} vanished");
                    return Errno::new(ESRCH).to_ret();
                }
            };
            proc.pledge(request);
            proc.permissions
        };

        let tids: alloc::vec::Vec<u32> = self
            .processes
            .get(&pid)
            .map(|p| p.threads.iter().copied().collect())
            .unwrap_or_default();
        for tid in tids {
            if let Some(pt) = self.threads.get(&tid) {
                let t = unsafe { (pt.0 as *mut Thread).as_mut_unchecked() };
                t.permissions = new_perms;
            }
        }

        0
    }

    /// `create_process_v2(*const CreateProcessV2Args)`. Role-aware
    /// spawn: validate the role transition; on success derive the
    /// child's perms through the witness path and proceed with the
    /// ELF copy + spawn. On failure record an audit event into the
    /// kernel-wide `DenialRing`, bump the parent's `role_denials`
    /// counter, and return `-EPERM` — no fall-through, no child.
    fn run_create_process_v2_req(
        &mut self,
        req: orbit_core::CreateProcessV2Req,
        parent_pid: u16,
        root_pa: PhysAddr,
    ) -> isize {
        use orbit_abi::denial::{DenialEvent, DenialSink};
        use orbit_abi::perms::CreateProcessV2Args;
        use orbit_core::roles::{check_transition, deny_reason_code, derive_child_perms};

        const MAX_ELF_BYTES: usize = 4 * 1024 * 1024;

        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

        // Copy the args struct (one user page; the 8-byte alignment
        // check at the syscall boundary guarantees no straddle).
        let args_page_base = req.args_vaddr.raw() & !(PAGE_SIZE as u64 - 1);
        let args_page_off = (req.args_vaddr.raw() - args_page_base) as usize;
        let args_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(args_page_base)) } {
                Some(p) => p as u64,
                None => {
                    error!(
                        "create_process_v2: args va 0x{:X} does not translate",
                        req.args_vaddr
                    );
                    return Errno::new(EFAULT).to_ret();
                }
            };
        let args: CreateProcessV2Args = unsafe {
            let mut w = user_page::UserPageWindow::map(args_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            let p = page.as_ptr().add(args_page_off) as *const CreateProcessV2Args;
            core::ptr::read_unaligned(p)
        };

        if args.elf_len == 0 || args.elf_len > MAX_ELF_BYTES {
            return Errno::new(EINVAL).to_ret();
        }

        if !user_range_ok(args.elf_vaddr as u64, args.elf_len as u64) {
            return Errno::new(EFAULT).to_ret();
        }

        // Snapshot the parent's permissions for the gate. `Permissions`
        // is `Copy`, so this releases the borrow on `processes`
        // before we touch `denial_ring` (which also lives on `self`).
        let parent_perms = match self.processes.get(&parent_pid) {
            Some(p) => p.permissions,
            None => {
                error!("create_process_v2: parent pid={parent_pid} vanished");
                return Errno::new(ESRCH).to_ret();
            }
        };

        // Role-transition gate. Ok: derive the child's perms via the
        // witness path; the resulting `ChildPerms` flows directly
        // into `Process::install_child` below — no detour through
        // raw `Permissions`. Err: record a `RoleDeny` audit event,
        // bump the parent's counter, return -EPERM.
        let request = args.request();
        let child_perms = match check_transition(parent_perms.role, args.target_role) {
            Ok(transition) => derive_child_perms(&parent_perms, transition, request),
            Err(spawn_deny) => {
                // The calling tid isn't carried in PendingWork — for
                // audit logging the parent pid is the actionable
                // identity. Stamp 0 as a sentinel; readers use `pid`
                // for "which process tried this."
                let now_ticks = riscv::register::time::read64();
                self.denial_ring.push(DenialEvent::RoleDeny {
                    time_ticks: now_ticks,
                    _reserved: 0,
                    tid: 0,
                    source_role: parent_perms.role,
                    target_role: args.target_role,
                    deny_reason: deny_reason_code(spawn_deny),
                    pid: parent_pid,
                });
                if let Some(proc) = self.processes.get(&parent_pid) {
                    proc.role_denials.fetch_add(1, Ordering::Relaxed);
                }
                return Errno::new(EPERM).to_ret();
            }
        };

        // Copy the ELF (same loop as run_create_process_req).
        let mut blob: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(args.elf_len);
        let mut copied = 0usize;
        while copied < args.elf_len {
            let cursor = args.elf_vaddr + copied;
            let page_base = cursor & !(PAGE_SIZE - 1);
            let page_off = cursor - page_base;
            let take = core::cmp::min(PAGE_SIZE - page_off, args.elf_len - copied);
            let pa = match unsafe {
                mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base as u64))
            } {
                Some(p) => p as u64,
                None => {
                    error!(
                        "create_process_v2: elf va 0x{:X} does not translate",
                        page_base
                    );
                    return Errno::new(EFAULT).to_ret();
                }
            };
            unsafe {
                let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                let page = w.as_mut_slice();
                blob.extend_from_slice(&page[page_off..page_off + take]);
            }
            copied += take;
        }

        // Affinity validation, identical to run_create_process_req.
        let all_harts = self.all_harts_mask();
        let allowed = if args.allowed_affinity == 0 {
            all_harts
        }
        else {
            args.allowed_affinity
        };
        let affinity = if args.affinity == 0 {
            allowed
        }
        else {
            args.affinity
        };
        if allowed & !all_harts != 0 || affinity & !allowed != 0 || affinity == 0 {
            error!("create_process_v2: affinity validation failed");
            return Errno::new(EINVAL).to_ret();
        }

        let proc_components = ProcessComponents {
            elf_blob: &blob,
            stack_size: UPROC_STACK_DEFAULT,
            allowed_affinity: allowed,
            affinity,
            parent_pid,
            argv_bytes: None,
            envp_bytes: None,
            perms: None,
        };

        let child_pid = match self.create_new_process(proc_components) {
            Ok(p) => p,
            Err(()) => {
                error!("create_process_v2: create_new_process failed");
                return Errno::new(ENOEXEC).to_ret();
            }
        };

        // Install the witness-derived perms on the child via the
        // type-enforced path. `create_new_process` stamps BOOTSTRAP-
        // shaped ALL by default for legacy callers; v2 overrides
        // that with the role-resolved value here. Then walk the
        // child's threads (just the initial one at this point) and
        // refresh each `Thread.permissions` snapshot — that copy
        // pulls a plain `Permissions` out of the witness via
        // `permissions()`, which doesn't consume the `ChildPerms`.
        if let Some(proc) = self.processes.get_mut(&child_pid) {
            proc.install_child(child_perms);
        }
        let perms_snapshot = child_perms.permissions();
        let tids: alloc::vec::Vec<u32> = self
            .processes
            .get(&child_pid)
            .map(|p| p.threads.iter().copied().collect())
            .unwrap_or_default();
        for tid in tids {
            if let Some(pt) = self.threads.get(&tid) {
                let t = unsafe { (pt.0 as *mut Thread).as_mut_unchecked() };
                t.permissions = perms_snapshot;
            }
        }

        info!(
            "create_process_v2: spawned pid={child_pid} parent={parent_pid} role={} perms={:#x}/{:#x}",
            args.target_role, perms_snapshot.perms, perms_snapshot.allowed_perms,
        );
        child_pid as isize
    }

    /// Allocate one kernel_pages page, copy `blob` into it with the
    /// offset → absolute-pointer fixup, and map at `USER_ARGV_BASE`
    /// in the child process's PT (R+U+S, no W/X). Stash the backing
    /// on `Process.argv_blob` for later cleanup.
    fn install_argv_blob(&mut self, pid: u16, blob: &[u8]) -> Result<(), ()> {
        use orbit_abi::layout::USER_ARGV_BASE;
        let backing = self.install_argv_envp_page(pid, blob, USER_ARGV_BASE, "argv")?;
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.argv_blob = Some(backing);
        }
        Ok(())
    }

    /// `install_argv_blob`'s envp twin — same wire format
    /// (`orbit_abi::envp` re-exports `argv`'s types) so the install
    /// helper handles both. Maps the rewritten page at
    /// `USER_ENVP_BASE` and stashes the backing on
    /// `Process.envp_blob`.
    fn install_envp_blob(&mut self, pid: u16, blob: &[u8]) -> Result<(), ()> {
        use orbit_abi::layout::USER_ENVP_BASE;
        let backing = self.install_argv_envp_page(pid, blob, USER_ENVP_BASE, "envp")?;
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.envp_blob = Some(backing);
        }
        Ok(())
    }

    /// Shared body of `install_argv_blob` / `install_envp_blob`. The
    /// argv and envp blobs share the wire format
    /// (`[ArgvHeader][offsets][strings]`); this helper allocates the
    /// kernel page, validates argc, fixes up offsets to
    /// `target_va + offset`, and maps R+U at `target_va` in the
    /// child's PT. Returns the `PhysBacking` so the caller can stash
    /// it on the right `Process` slot for dealloc-time cleanup.
    fn install_argv_envp_page(
        &mut self,
        pid: u16,
        blob: &[u8],
        target_va: u64,
        tag: &'static str,
    ) -> Result<process::PhysBacking, ()> {
        use orbit_abi::argv::{ARGV_OFFSETS_OFFSET, ArgvHeader};

        if blob.len() > PAGE_SIZE {
            error!("install_{tag}_blob: blob {} > page", blob.len());
            return Err(());
        }
        if blob.len() < core::mem::size_of::<ArgvHeader>() {
            error!("install_{tag}_blob: blob too small");
            return Err(());
        }

        // Sanity-check argc against what the blob can hold.
        let argc = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        let strings_off = ARGV_OFFSETS_OFFSET + argc * core::mem::size_of::<u64>();
        if strings_off > blob.len() {
            error!(
                "install_{tag}_blob: argc={argc} overflows blob len={}",
                blob.len()
            );
            return Err(());
        }

        let layout = match Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
            Ok(l) => l,
            Err(_) => return Err(()),
        };
        let (frame, kva) = self.kernel_pages.alloc_kdmap(layout).ok_or(())?;
        let backing = process::PhysBacking::Shared { frame, layout };

        // Zero the page first so any unused tail reads as zeros.
        unsafe {
            core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, PAGE_SIZE);
        }
        // Copy header + offsets + strings verbatim, then walk the
        // offset slots and rewrite each as target_va + offset.
        unsafe {
            let dst = kva.as_mut_ptr::<u8>();
            core::ptr::copy_nonoverlapping(blob.as_ptr(), dst, blob.len());

            let slots = dst.add(ARGV_OFFSETS_OFFSET) as *mut u64;
            for i in 0..argc {
                let off = slots.add(i).read();
                if off >= blob.len() as u64 {
                    error!("install_{tag}_blob: entry {i} offset {off} >= blob len");
                    self.free_backing(backing);
                    return Err(());
                }
                slots.add(i).write(target_va.wrapping_add(off));
            }
        }

        // Map the page R+U into the child's PT at target_va.
        let proc = self.processes.get(&pid).ok_or(())?;
        let proc_root_pa = PhysAddr::from(proc.satp);
        let proc_root_table = unsafe { memmap::kernel_root_from_pa(proc_root_pa) };
        let blob_pa = backing.pa();

        let config = MappingConfig {
            permissions: PagePermissions::R | PagePermissions::U,
            levels: 4,
            page_size: PAGE_SIZE as u64,
            vaddr: VirtAddr::new(target_va),
            paddr: blob_pa,
            log: false,
            // No SharedRevocable tag — the page is freed via
            // dealloc_process when the process exits, not via
            // SharedUserPtr::revoke. The tag is purely a kernel-side
            // policy bit.
            supervisor_tag: SupervisorTag::None,
        };
        let vend = VirtAddr::new(target_va + PAGE_SIZE as u64);
        let pend = PhysAddr::new(blob_pa.get_raw() + PAGE_SIZE as u64);
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        if let Err(_) =
            unsafe { map_address_range(&proc_root_table, &mut pages, &config, vend, pend) }
        {
            error!("install_{tag}_blob: map_address_range failed");
            self.free_backing(backing);
            return Err(());
        }

        riscv::asm::sfence_vma(pid as usize, target_va as usize);
        crate::kernel::shootdown::broadcast(0, 0);
        Ok(backing)
    }

    /// Owns the signaling end-to-end. Sync errors signal
    /// `(errno, 0)` here; async success installs the handle on the
    /// target's `exit_waiter` slot and `dealloc_process` later signals
    /// `(0, exit_code)`. The pair shape (a0 = success/errno, a1 =
    /// exit_code) keeps the negative-as-errno convention orthogonal
    /// to negative exit codes — see `orbit-abi/src/user.rs::wait_pid`.
    fn run_wait_pid_req(
        &mut self,
        req: WaitPidReq,
        caller_pid: u16,
        handle: process::CompletionHandle,
    ) {
        // First check the caller's `dead_children` — covers the race
        // where the target exited before this wait_pid syscall ran.
        // dealloc_process stashed (target_pid → exit_code) on the
        // parent's process struct; drain it here for sync return.
        if let Some(parent) = self.processes.get_mut(&caller_pid)
            && let Some(code) = parent.dead_children.remove(&req.target_pid)
        {
            handle.signal_pair(0, code as isize);
            return;
        }

        let Some(target) = self.processes.get_mut(&req.target_pid)
        else {
            // Never existed (or exited and the parent's already gone
            // / wasn't tracked) — POSIX surfaces this as ECHILD.
            handle.signal_pair(Errno::new(orbit_abi::errno::ECHILD).to_ret(), 0);
            return;
        };
        if target.parent_pid != caller_pid {
            handle.signal_pair(Errno::new(EPERM).to_ret(), 0);
            return;
        }
        if target.exit_waiter.is_some() {
            // Single-waiter v1 — multi-waiter wants a Vec and lands
            // with futex (§13a.5).
            handle.signal_pair(Errno::new(orbit_abi::errno::EBUSY).to_ret(), 0);
            return;
        }
        // Install the parent's handle on the target. dealloc_process
        // will take + signal it with the child's exit code.
        target.exit_waiter = Some(handle);
        info!(
            "wait_pid: pid={caller_pid} parked on target={} exit",
            req.target_pid
        );
    }

    /// §13a.5 — futex wait. Owns the signaling: sync errors signal
    /// here; the async park installs the waiter on the per-PA queue
    /// and a later `futex_wake` (or process teardown) signals the
    /// handle.
    ///
    /// The compare-then-park is atomic against any concurrent
    /// `futex_wake` because both run on the manager hart under
    /// `MANAGER_LOCK`. A wake that drains the queue runs to
    /// completion before this wait arm sees it.
    fn run_futex_wait_req(
        &mut self,
        req: FutexWaitReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: process::CompletionHandle,
    ) {
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(req.uaddr.raw())) } {
                Some(p) => p as u64,
                None => {
                    handle.signal(Errno::new(EFAULT).to_ret());
                    return;
                }
            };
        // Read `*uaddr` through a transient KSCRATCH window. user_pages
        // has no KDMAP alias under the kernel satp (kernel only KDMAPs
        // its own pools), so a direct deref of `phys_to_kdmap(pa)`
        // would land on an unmapped VA. UserPageWindow installs a
        // leaf PTE at KSCRATCH for the page containing `pa`, lets us
        // read the word, and tears down on drop. We hold
        // `MANAGER_LOCK`, which is the single-slot serializer
        // UserPageWindow assumes.
        let page_pa = pa & !(PAGE_SIZE as u64 - 1);
        let page_off = (pa - page_pa) as usize;
        let observed = unsafe {
            let mut win = crate::kernel::user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            core::ptr::read_volatile(win.as_mut_ptr().add(page_off) as *const u32)
        };
        if observed != req.expected {
            handle.signal(Errno::new(EAGAIN).to_ret());
            return;
        }
        // Park: install the waiter on the per-PA queue. v1 ignores
        // `timeout_ns` — the field is reserved; the wait blocks
        // until a `futex_wake` (or `dealloc_process`) drains it.
        let waiter = FutexWaiter {
            handle,
            pid,
            deadline_ticks: 0,
        };
        self.futex_waiters.entry(pa).or_default().push(waiter);
        trace!("futex_wait: pid={pid} pa={pa:#x} expected={}", req.expected);
    }

    /// §13a.5 — futex wake. Drains up to `req.n` waiters from
    /// `futex_waiters[pa]`, signals each with `0`, and signals the
    /// caller's handle with the count (or a negative errno on
    /// translation failure).
    fn run_futex_wake_req(
        &mut self,
        req: FutexWakeReq,
        _pid: u16,
        root_pa: PhysAddr,
        handle: process::CompletionHandle,
    ) {
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(req.uaddr.raw())) } {
                Some(p) => p as u64,
                None => {
                    handle.signal(Errno::new(EFAULT).to_ret());
                    return;
                }
            };
        let n_woken = match self.futex_waiters.get_mut(&pa) {
            Some(waiters) => {
                let take = core::cmp::min(req.n as usize, waiters.len());
                // Drain from the front so wake order matches park
                // order (FIFO). Since waiters are pushed at the tail
                // in `run_futex_wait_req`, the oldest is at index 0.
                let drained: Vec<FutexWaiter> = waiters.drain(..take).collect();
                if waiters.is_empty() {
                    self.futex_waiters.remove(&pa);
                }
                for w in drained {
                    w.handle.signal(0);
                }
                take as isize
            }
            None => 0,
        };
        handle.signal(n_woken);
        trace!("futex_wake: pa={pa:#x} requested={} woke={n_woken}", req.n);
    }

    fn run_create_thread_req(
        &mut self,
        req: CreateThreadReq,
        pid: u16,
        parent_allowed: u64,
    ) -> isize {
        info!("handling create_thread req: {req:?} pid={pid} parent_allowed={parent_allowed:#x}");

        let all_harts = self.all_harts_mask();
        // Resolve sentinels exactly like create_process: 0 → "default."
        // Default for `allowed_affinity` is the parent's cap (so children
        // inherit the family reach); default for `affinity` follows the
        // resolved `allowed_affinity`.
        let allowed = if req.allowed_affinity == 0 {
            parent_allowed
        }
        else {
            req.allowed_affinity
        };
        let affinity = if req.affinity == 0 {
            allowed
        }
        else {
            req.affinity
        };

        // Capability-style check: a thread can't claim reach the parent
        // doesn't have. Bits-beyond-cpu_count surfaces here too because
        // parent_allowed is itself a subset of all_harts.
        if allowed & !parent_allowed != 0 {
            error!(
                "create_thread: requested allowed={allowed:#x} escapes parent={parent_allowed:#x}"
            );
            return Errno::new(EPERM).to_ret();
        }
        if affinity & !allowed != 0 || affinity == 0 || allowed & !all_harts != 0 {
            error!("create_thread: affinity={affinity:#x} allowed={allowed:#x} all={all_harts:#x}");
            return Errno::new(EINVAL).to_ret();
        }

        if !self.processes.contains_key(&pid) {
            error!("create_thread: pid{pid} vanished");
            return Errno::new(ESRCH).to_ret();
        }

        // Pre-allocation check: reading the captured tid out of the
        // newly-inserted Thread requires a fresh registry lookup, since
        // add_new_thread_to_process boxes the Thread internally and only
        // returns Result<(), ()>. Snapshot the next tid by inspecting
        // the current max + 1 — close enough for diagnostics; the real
        // tid is read off the registry below on success.
        match self.add_new_thread_to_process(
            pid,
            req.entry.raw() as usize,
            UPROC_STACK_DEFAULT,
            allowed,
            affinity,
            req.arg,
        ) {
            Ok(()) => {
                // Find the most-recently-inserted thread for this pid:
                // the slot allocator is monotonic per process, so the
                // highest tid in proc.threads is ours.
                let proc = self.processes.get(&pid).expect("pid present, just checked");
                let new_tid = match proc.threads.iter().next_back() {
                    Some(t) => *t,
                    None => {
                        error!("create_thread: pid{pid} has no threads after insert");
                        return Errno::new(EAGAIN).to_ret();
                    }
                };
                info!(
                    "create_thread: spawned tid={new_tid} in pid={pid} \
                    allowed={allowed:#x} affinity={affinity:#x}"
                );
                new_tid as isize
            }
            Err(()) => {
                error!("create_thread: add_new_thread_to_process failed");
                Errno::new(ENOMEM).to_ret()
            }
        }
    }

    fn run_create_process_req(
        &mut self,
        req: CreateProcessReq,
        parent_pid: u16,
        root_pa: PhysAddr,
    ) -> isize {
        info!("handling create_process req: {req:?}");

        // Dev-loop safety cap. Well above any realistic test ELF but small
        // enough that a bogus `elf_len` can't drive the kernel into a giant
        // allocation. Bump when we actually need to.
        const MAX_ELF_BYTES: usize = 4 * 1024 * 1024;

        if req.elf_len == 0 || req.elf_len > MAX_ELF_BYTES {
            return Errno::new(EINVAL).to_ret();
        }

        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

        // Copy the ELF out page-by-page. UserPageWindow is single-slot, so
        // we materialize each page in turn and release it before the next.
        // User pages come from a single contiguous mmap in practice, but
        // don't assume — translate each page independently.
        let mut blob: Vec<u8> = Vec::with_capacity(req.elf_len);
        let mut copied = 0u64;
        let elf_len = req.elf_len as u64;
        while copied < elf_len {
            let cursor = req.elf_vaddr.raw() + copied;
            let page_base = cursor & !(PAGE_SIZE as u64 - 1);
            let page_off = (cursor - page_base) as usize;
            let take = core::cmp::min(PAGE_SIZE - page_off, (elf_len - copied) as usize);

            let pa = match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) }
            {
                Some(p) => p as u64,
                None => {
                    error!(
                        "create_process: user va 0x{:X} does not translate",
                        page_base
                    );
                    return Errno::new(EFAULT).to_ret();
                }
            };

            unsafe {
                let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                let page = w.as_mut_slice();
                blob.extend_from_slice(&page[page_off..page_off + take]);
            }

            copied += take as u64;
        }

        // Sentinel 0 → "default" (all harts). Otherwise validate that
        // the requested affinity is a subset of the requested allowed
        // mask, and that both fit within the actual cpu_count. Bits
        // beyond cpu_count mean the caller is naming harts that don't
        // exist — reject as EINVAL rather than silently masking, so the
        // caller learns rather than getting a different mask than they
        // asked for.
        let all_harts = self.all_harts_mask();
        let allowed = if req.allowed_affinity == 0 {
            all_harts
        }
        else {
            req.allowed_affinity
        };
        let affinity = if req.affinity == 0 {
            allowed
        }
        else {
            req.affinity
        };
        if allowed & !all_harts != 0 || affinity & !allowed != 0 || affinity == 0 {
            error!(
                "create_process: affinity validation failed: \
                allowed={allowed:#x} affinity={affinity:#x} all={all_harts:#x}"
            );
            return Errno::new(EINVAL).to_ret();
        }

        let proc_components = ProcessComponents {
            elf_blob: &blob,
            stack_size: UPROC_STACK_DEFAULT,
            allowed_affinity: allowed,
            affinity,
            parent_pid,
            argv_bytes: None,
            envp_bytes: None,
            perms: None,
        };

        match self.create_new_process(proc_components) {
            Ok(pid) => {
                info!(
                    "create_process: spawned pid={pid} parent={parent_pid} from {} bytes \
                    allowed_affinity={allowed:#x} affinity={affinity:#x}",
                    blob.len()
                );
                pid as isize
            }
            Err(()) => {
                error!("create_process: create_new_process failed");
                Errno::new(ENOEXEC).to_ret()
            }
        }
    }

    /// Drain `WAKE_QUEUE`. Each event names a thread (or set of
    /// threads) plus a [`wake_reason`] bitmask explaining the cause.
    /// We OR the bitmask into the matching thread's `wake_override` —
    /// the scheduler's next `Suspended → Ready` scan will observe the
    /// non-zero override and dispatch the thread, atomically consuming
    /// the bits and stashing them in `last_wake_reason` for query.
    ///
    /// Producer/consumer split: the parking thread writes `wake_time`,
    /// producers `fetch_or` into `wake_override`, the scheduler
    /// `swap(0)` into `last_wake_reason`. No two writers ever touch
    /// the same field — the parking-thread → manager race that would
    /// otherwise overwrite a wake signal can't happen.
    ///
    /// Coarse over-waking is harmless: each thread re-checks its own
    /// wait predicate on wake (e.g. `read_some` retries `recv_tcp`)
    /// and re-parks if not actually ready. So `Pid` waking every
    /// thread of a process is fine even when only one is parked on
    /// the NetCh — the others go right back to sleep.
    pub(crate) fn drain_wakes(&mut self) {
        while let Some(mut slot) = WAKE_QUEUE.pop_ref() {
            let event = core::mem::take(&mut *slot);
            drop(slot);
            match event {
                WakeEvent::None => {}
                WakeEvent::Net => {
                    // Target k_net specifically once `setup_igb` has
                    // latched its tid. Before then (boot window), fall
                    // back to a coarse pid=0 scan — by the time
                    // anything pushes `WakeEvent::Net` for real (PLIC
                    // IRQ, user nc_yield) the latch has fired, so the
                    // fallback is just a safety net for self-pushes
                    // during k_net's own bringup.
                    match self.net_thread_tid {
                        Some(tid) => self
                            .set_wake_reason_where(process::wake_reason::TICKLE, |t| t.tid == tid),
                        None => {
                            self.set_wake_reason_where(process::wake_reason::TICKLE, |t| t.pid == 0)
                        }
                    }
                }
                WakeEvent::Pid(pid) => {
                    self.set_wake_reason_where(process::wake_reason::NET_IO, |t| t.pid == pid);
                }
                WakeEvent::Tid(tid) => {
                    self.set_wake_reason_where(process::wake_reason::NET_IO, |t| t.tid == tid);
                }
            }
        }
    }

    /// `fetch_or(reason)` into `wake_override` on every thread matching
    /// `pred`. Helper for [`drain_wakes`]; `pred` runs against a
    /// `&Thread` from the global table.
    ///
    /// **Eager Suspended promotion**: when the matched thread is
    /// currently `Suspended`, we don't just OR the reason bit and
    /// wait for `drain_sleeps` to notice — we immediately consume the
    /// override into `last_wake_reason` and flip the thread to
    /// `Ready`. The corresponding sleep-heap entry becomes stale and
    /// gets reaped on the next `drain_woken` (state mismatch). This
    /// closes the latency gap between "tickle arrived" and "thread
    /// dispatched": same-pass dispatch instead of waiting for the
    /// next manager pass to walk the heap.
    fn set_wake_reason_where(&mut self, reason: u64, mut pred: impl FnMut(&Thread) -> bool) {
        for (_, p) in self.threads.iter() {
            // SAFETY: `PThread.0` is a raw ptr the registry owns; it
            // stays valid as long as the entry's in `self.threads`.
            let thread = unsafe { (p.0 as *mut Thread).as_mut_unchecked() };
            if !pred(thread) {
                continue;
            }
            thread.wake_override.fetch_or(reason, Ordering::Release);
            // Eager promotion. CAS state Suspended → Ready; if state
            // is anything else (already Ready, Running, etc.) leave
            // it alone. The wake_override OR above means a thread
            // that hadn't yet committed its park (Running on its way
            // to Suspended) will see the override on its next
            // dispatch via the sleep-heap path.
            if thread
                .state
                .compare_exchange(
                    ThreadState::Suspended as usize,
                    ThreadState::Ready as usize,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let pending = thread.wake_override.swap(0, Ordering::AcqRel);
                thread.last_wake_reason.store(pending, Ordering::Release);
                // Just promoted Suspended → Ready; queue it so
                // get_runnable_thread picks it up this same pass.
                // The sleep-heap entry becomes stale (state mismatch)
                // and is reaped on the next drain_woken.
                self.ready.push(p.0);
            }
        }
    }

    /// Drain [`DENIAL_EVENT_QUEUE`]. Producers (the dispatch-site
    /// permission gate on any hart) push lock-free; this consumer
    /// runs each manager pass and folds events into the kernel-wide
    /// [`Self::denial_ring`] + bumps the owning process's
    /// per-event-kind counter.
    ///
    /// Best-effort: a process that has already exited between the
    /// gate push and the manager drain has no `Process` record to
    /// update — the event still lands in the ring (the syscall
    /// number / pid / tid are still useful diagnostics) but the
    /// counter bump is a no-op. Same shape for an unknown pid
    /// (defensive against a future bug; can't happen on the live
    /// path).
    pub(crate) fn drain_denial_events(&mut self) {
        while let Some(mut slot) = DENIAL_EVENT_QUEUE.pop_ref() {
            let entry = core::mem::take(&mut *slot);
            drop(slot);
            let Some(event) = entry
            else {
                continue;
            };

            // Match each event variant once: stash the pid for the
            // counter bump (different counter per variant) and push
            // the event into the ring via the DenialSink trait.
            use orbit_abi::denial::{DenialEvent, DenialSink};
            let (pid, is_perm_deny) = match event {
                DenialEvent::PermDeny { pid, .. } => (pid, true),
                DenialEvent::RoleDeny { pid, .. } => (pid, false),
            };
            self.denial_ring.push(event);
            if let Some(proc) = self.processes.get(&pid) {
                let counter = if is_perm_deny {
                    &proc.perm_denials
                }
                else {
                    &proc.role_denials
                };
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Drain `MANAGER_WORK`. Each entry is a syscall handler bundled
    /// with its [`CompletionHandle`]; we run the handler, signal the
    /// handle with the result, and let the next scheduler scan resume
    /// the parked thread off `thread.handle.is_signaled()`.
    pub(crate) fn drain_pending_work(&mut self) {
        while let Some(mut slot) = MANAGER_WORK.pop_ref() {
            let work = core::mem::take(&mut *slot);
            drop(slot);
            match work {
                PendingWork::Empty => {}
                PendingWork::MemMap {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_mmap_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::NetChannelCreation {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let (r, e) = self.run_nc_create_req(req, pid, root_pa);
                    handle.signal_pair(r, e);
                }
                PendingWork::CloseHandle {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_close_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::CreateProcess {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_create_process_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::CreateThread {
                    req,
                    pid,
                    parent_allowed,
                    handle,
                } => {
                    let result = self.run_create_thread_req(req, pid, parent_allowed);
                    handle.signal(result);
                }
                PendingWork::FsOpen {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_fs_open_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::FsRead {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    // run_fs_read_req owns the handle: it either
                    // consumes it into a WorkNotification::Bounce
                    // (async — the IRQ + FsReadCopy signal it) or
                    // signals it inline (sync — cache hit, EOF, or
                    // error). No return value to dispatch on.
                    self.run_fs_read_req(req, pid, root_pa, handle);
                }
                PendingWork::FsStat {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_fs_stat_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::FsReaddir {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_fs_readdir_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::WaitPid { req, pid, handle } => {
                    // run_wait_pid_req owns the signaling — sync
                    // errors signal (errno, 0); the async success
                    // path installs the handle on the target's
                    // exit_waiter slot and dealloc_process signals
                    // (0, exit_code) when the child exits.
                    self.run_wait_pid_req(req, pid, handle);
                }
                PendingWork::CreateProcessEx {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_create_process_ex_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::FutexWait {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    // run_futex_wait_req owns the signaling — sync
                    // EAGAIN/EFAULT signal here; the async park
                    // installs the handle on `futex_waiters[pa]` and
                    // a later `futex_wake` signals it with `0`.
                    self.run_futex_wait_req(req, pid, root_pa, handle);
                }
                PendingWork::FutexWake {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    self.run_futex_wake_req(req, pid, root_pa, handle);
                }
                PendingWork::Pledge {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_pledge_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::CreateProcessV2 {
                    req,
                    pid,
                    root_pa,
                    handle,
                } => {
                    let result = self.run_create_process_v2_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::FsReadCopy { notif_ptr, status } => {
                    self.run_fs_read_copy(notif_ptr, status);
                }
            }
        }
    }

    fn get_runnable_thread(&mut self, hart_mask: u64) -> Option<PThread> {
        // O(1) common case: the queue head matches the hart's
        // affinity. Misses fall through to a head-scan in `pop_for`,
        // bounded by ready-queue depth. All Ready transitions
        // (preemption, sleep-heap wake, eager promote, blocking
        // signal, thread creation) push onto self.ready before this
        // method runs — see assign_threads's prelude.
        self.ready.pop_for(hart_mask).map(PThread)
    }

    fn dealloc_thread(&mut self, mut thread: Box<Thread>) {
        match (thread.slot, thread.pid) {
            (None, 0) => {
                // Kernel thread. Its stack and trap frame were allocated
                // directly from kernel_pages with fixed layouts and aren't
                // recorded in any proc.maps, so free them here. Move the
                // owning `Frame<Shared>` out of the Thread and hand it to
                // the typed allocator.
                if let Some(frame) = thread.kernel_stack.take() {
                    self.kernel_pages.free(frame, Self::THREAD_STACK_LAYOUT);
                }
                if let Some(frame) = thread.kernel_trap_frame.take() {
                    self.kernel_pages
                        .free(frame, Self::THREAD_TRAP_FRAME_LAYOUT);
                }
            }
            (Some(slot), 0) => error!(
                "dealloc_thread: tid{} is a kernel thread but carries slot{}",
                thread.tid, slot
            ),
            (None, pid) => error!(
                "dealloc_thread: tid{} user thread in pid{} is missing its slot",
                thread.tid, pid
            ),
            (Some(slot), pid) => match self.processes.get_mut(&pid) {
                Some(proc) => {
                    let root_table =
                        unsafe { memmap::kernel_root_from_pa(PhysAddr::from(proc.satp)) };

                    // Two passes: gather the vaddrs matching this slot
                    // (u64 is Copy so the collect doesn't tangle with
                    // proc's borrow), then pull each UserMapping out of
                    // proc.maps by `remove` — that transfers ownership
                    // of its `backing: Option<PhysBacking>`, which we
                    // can hand to `free_backing`. Single copy avoided
                    // because `PhysBacking` (and therefore UserMapping)
                    // is no longer Copy.
                    let vaddrs: Vec<u64> = proc.mappings_for_slot(slot).map(|m| m.vaddr).collect();

                    for v in &vaddrs {
                        let proc = self
                            .processes
                            .get_mut(&pid)
                            .expect("proc vanished mid-teardown");
                        let Some(m) = proc.maps.remove(v)
                        else {
                            continue;
                        };

                        match m.kind {
                            MappingKind::Stack { .. } => {
                                // Stack is a range of 2 MiB megapages; flush
                                // each page's TLB entry as we tear it down so
                                // nothing survives for slots 2..N.
                                for v in
                                    (m.vaddr..m.vaddr + m.len).step_by(UPROC_STACK_GRAIN as usize)
                                {
                                    unsafe {
                                        let _ = unmap_page(&root_table, VirtAddr::new(v), 3);
                                        riscv::asm::sfence_vma(pid as usize, v as usize);
                                        crate::kernel::shootdown::broadcast(0, 0);
                                    }
                                }
                            }
                            MappingKind::TrapFrame { .. } => unsafe {
                                let _ = unmap_page(&root_table, VirtAddr::new(m.vaddr), 4);
                                riscv::asm::sfence_vma(pid as usize, m.vaddr as usize);
                                crate::kernel::shootdown::broadcast(0, 0);
                            },
                            MappingKind::Guard { .. } => {
                                // No leaf backs the guard; only the proc.maps
                                // entry needs clearing below.
                            }
                            MappingKind::Tls { .. } => {
                                // One 2 MiB megapage at level 3 — same
                                // shape as the Stack arm. Backing freed
                                // by the tail of this loop.
                                unsafe {
                                    let _ = unmap_page(&root_table, VirtAddr::new(m.vaddr), 3);
                                    riscv::asm::sfence_vma(pid as usize, m.vaddr as usize);
                                }
                                crate::kernel::shootdown::broadcast(0, 0);
                            }
                            // mappings_for_slot filters on MappingKind::slot(),
                            // which only returns Some for the arms above.
                            MappingKind::Elf | MappingKind::Anon | MappingKind::NetCh { .. } => {
                                unreachable!("mappings_for_slot yielded non-slot kind {:?}", m.kind)
                            }
                        }

                        if let Some(b) = m.backing {
                            self.free_backing(b);
                        }
                    }

                    let proc = self
                        .processes
                        .get_mut(&pid)
                        .expect("proc vanished mid-teardown");
                    proc.thread_slots.free(slot);
                }
                None => error!(
                    "dealloc_thread: tid{} references missing pid{}",
                    thread.tid, pid
                ),
            },
        }
    }

    fn dealloc_process(&mut self, mut process: Process) {
        let process_root_table_pa = PhysAddr::from(process.satp);

        // §13a.2 — three exit paths:
        //  1. Parent already parked on `wait_pid` → signal the waiter
        //     directly with `(0, exit_code)`. Wake hook copies into
        //     a-regs on resume.
        //  2. Parent is alive but hasn't called `wait_pid` yet →
        //     stash the exit code in the parent's `dead_children`
        //     map. A later `wait_pid` drains it and returns sync.
        //     Closes the race where the child exits faster than the
        //     parent can park.
        //  3. No parent (boot init) or parent already gone → drop
        //     the exit code on the floor.
        if let Some(handle) = process.exit_waiter.take() {
            handle.signal_pair(0, process.exit_code as isize);
        }
        else if process.parent_pid != 0
            && let Some(parent) = self.processes.get_mut(&process.parent_pid)
        {
            parent.dead_children.insert(process.pid, process.exit_code);
        }

        // §13a.3 / §13e — return the argv / envp blob pages to kernel_pages.
        if let Some(backing) = process.argv_blob.take() {
            self.free_backing(backing);
        }
        if let Some(backing) = process.envp_blob.take() {
            self.free_backing(backing);
        }

        // Release the scrollback source so k_gpu advances `active`
        // off this pid on the next drain. Paired with the
        // `push_insert_source` in `create_new_process`.
        let _ = crate::drivers::k_gpu::push_remove_source(
            crate::drivers::display::Source::Process(process.pid),
        );

        // Tear down the per-process stdin slot. If a reader is parked
        // on it, `unregister` signals the handle so the manager-scan
        // unblocks the thread; the resumed thread re-enters
        // `read_stdin` and gets ENOENT for the gone pid (in practice
        // this only fires if a thread parks an instant before the
        // owning process exits — rare).
        crate::kernel::stdin::unregister(process.pid);

        while let Some(socket_handle) = process.sockets.pop_last() {
            if let Err(e) = self.net_pkg.socket_deletions.enqueue(socket_handle) {
                error!(
                    "failed to queue socket for deletion while deallocating pid{} ({e:?})",
                    process.pid
                );
                self.orphaned_sockets.push(socket_handle);
            }
        }

        // Revoke every Shared user mapping for this pid *before* tearing
        // down the manager's Arcs and the PT itself. Revoke walks the
        // user PT and clears each tagged leaf — so once this loop
        // completes, the user VA is unreachable even though k_net might
        // still hold an nc clone for one more poll. Two invariants fall
        // out:
        //   1. revoked == true ⇒ user PTEs are already gone (post-
        //      condition, not plan), so k_net observers using
        //      try_as_ref() can bail safely.
        //   2. Must happen before `unmap` below, which frees the
        //      intermediate PT pages the revoker walks.
        let root_table = unsafe { memmap::kernel_root_from_pa(process_root_table_pa) };
        // NetChannels need an explicit revoke walk before drop;
        // File handles' scratch SharedFrames drop on their own (in-
        // flight DMA descriptors, if any, retain a clone so the page
        // survives until the post-DMA copy completes — see the
        // close-mid-flight fix in run_fs_read_copy).
        if let Some(ph) = self.process_handles.remove(&process.pid) {
            for (_fd, handle) in ph.into_iter() {
                match handle {
                    Handle::NetChannel(sup) => {
                        if let Err(e) = sup.revoke(&root_table) {
                            warn!(
                                "dealloc_process: revoke failed for pid{} sup={sup:?}: {e:?}",
                                process.pid,
                            );
                        }
                        drop(sup);
                    }
                    Handle::File(of) => {
                        drop(of);
                    }
                }
            }
        }

        while let Some(b) = process.heap_pages.pop() {
            info!(
                "dealloc heap page pa@{:016X} {:08X?} pool={}",
                b.pa().get_raw(),
                b.layout(),
                b.pool_name()
            );
            self.free_backing(b);
        }

        // Drain any per-thread mappings still resident in proc.maps. In
        // the normal teardown path, dealloc_thread already pulled each
        // slot's Stack / TrapFrame / TLS entries out before this point,
        // so the loop is a no-op. The partial-build Err arms in
        // create_new_process route through here without dealloc_thread
        // ever running, and would leak the user_pages stack/TLS frames
        // and the kernel_pages trap frame otherwise.
        while let Some((_va, m)) = process.maps.pop_first() {
            if let Some(b) = m.backing {
                self.free_backing(b);
            }
        }

        // Leak-localization instrumentation for the ktables-growth bug.
        // Dump the pool size at three boundaries — into dealloc, after the
        // recursive unmap walk, and after the root free — so the per-pid
        // delta is recoverable from the log. `unmap_freed` should account
        // for every intermediate this pid materialized in user-half slots
        // plus its private KTEXT/KDMAP/KSCRATCH chains; `root_freed`
        // should be exactly `Self::TABLE_LAYOUT.size()`. A persistent net
        // increase across two smoke runs localizes the leak to whatever
        // sits between create_new_process entry and dealloc_process exit
        // for this pid.
        let ktables_before = self.table_pages.allocated_bytes();
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe {
            // Detach the shared kernel high-half L2 first — `unmap` is
            // recursive and would otherwise descend into and free the
            // shared subtree, corrupting every other satp's kernel
            // surface (KTEXT / KDMAP / KMMIO / KSCRATCH all live there).
            memmap::detach_shared_kernel_l2(&root_table);
            unmap(&root_table, &mut pages);
            let ktables_after_unmap = self.table_pages.allocated_bytes();
            // table_pages now returns typed frames — the walker's
            // `free_page` takes a raw PA directly; the root was allocated
            // from this pool so we reconstruct a `Frame<Table>` here.
            self.table_pages.free(
                Frame::<process::Table>::new(process_root_table_pa),
                Self::TABLE_LAYOUT,
            );
            let ktables_after = self.table_pages.allocated_bytes();
            debug!(
                "dealloc pid{}: ktables before={}B unmap_freed={}B root_freed={}B after={}B",
                process.pid,
                ktables_before,
                ktables_before.saturating_sub(ktables_after_unmap),
                ktables_after_unmap.saturating_sub(ktables_after),
                ktables_after,
            );

            // Whole-ASID flush before `next_pid` can hand this u16 to a
            // fresh process. The dealloc_thread loop sfenced stack/trap
            // leaves, but ELF / anon / NetCh mappings were only zapped by
            // `unmap` above. Cross-hart broadcast (whole-TLB sentinel)
            // catches every hart that ever ran this pid's threads.
            riscv::asm::sfence_vma(process.pid as usize, 0);
            crate::kernel::shootdown::broadcast(0, 0);
        }
    }

    pub fn cleanup_threads_and_processes(&mut self) {
        let mut tids_to_remove = Vec::new();
        let mut pids_to_remove = Vec::new();
        for (_tid, p) in self.threads.iter() {
            let t = unsafe { p.0.as_ref_unchecked() };

            {
                let proc = match self.processes.get_mut(&t.pid) {
                    Some(p) => p,
                    None => continue,
                };

                let thread_alive = t.state.load(Ordering::Acquire) != ThreadState::Exited as usize;

                if !thread_alive {
                    let _ = proc.threads.remove(&t.tid);
                    tids_to_remove.push(t.tid);

                    // Read through the existing &Thread — do NOT take a Box
                    // here. The second pass (dealloc_thread + free) runs
                    // after this loop, and needs this Thread's fields still
                    // readable; taking a Box would drop-free it at scope end
                    // and leave a use-after-free in the next pass.
                    match t.fault_info {
                        Some(f) => {
                            let label = match proc.find_mapping(f.stval as u64).map(|m| m.kind) {
                                Some(MappingKind::Guard { .. }) => "stack overflow",
                                Some(_) => "permission/range violation",
                                None => "bad access",
                            };
                            warn!(
                                "tid{} killed: {} cause={} epc={:#x} stval={:#x}",
                                t.tid, label, f.cause, f.epc, f.stval
                            );
                            // Faulted threads carry no clean exit
                            // value; surface as -1 to wait_pid waiters.
                            // POSIX would use WIFSIGNALED here; a
                            // distinguished negative is good enough
                            // for v1.
                            proc.exit_code = -1;
                        }
                        None => {
                            let status = t.frame.regs[11] as isize;
                            info!("tid{} dead, removing status={status}", t.tid);
                            proc.exit_code = status as i32;
                        }
                    }
                }

                if !proc.threads.is_empty() || t.pid == 0 {
                    continue;
                }
            }

            info!("pid{} dead, removing", t.pid);

            pids_to_remove.push(t.pid);
        }

        for tid in tids_to_remove {
            let p = self.threads.remove(&tid).unwrap();

            // Take ownership of the heap-leaked Thread and hand it to
            // dealloc_thread, which moves its kernel-thread `Frame<Shared>`
            // backings into `kernel_pages.free` and lets the Box drop at
            // the end of the call to release the Thread allocation.
            self.dealloc_thread(unsafe { Box::from_raw(p.0) });
        }

        for pid in pids_to_remove {
            let proc = self.processes.remove(&pid).unwrap();

            self.dealloc_process(proc);
        }

        // Drain SharedUserPtr Drops that landed since the last pass.
        // Each queued item is a `Frame<Shared>` whose last Arc just
        // dropped on some hart — return it to `kernel_pages` here,
        // under the Orbit lock, not in Drop context.
        let kpages = &mut self.kernel_pages;
        pending_frees::drain(|frame, layout| {
            info!(
                "dealloc shared ptr backing pa@{:016X} {:08X?}",
                frame.get_raw(),
                layout
            );
            kpages.free(frame, layout);
        });
    }

    /// Drain `SLEEP_INBOX` into the heap, then promote any sleepers
    /// whose deadline has passed to `Ready`. Called from
    /// `assign_threads` so the registry walk that follows already sees
    /// the freshly-promoted threads as Ready and dispatches them like
    /// any other runnable thread.
    pub(crate) fn drain_sleeps(&mut self) {
        while let Some(mut slot) = SLEEP_INBOX.pop_ref() {
            let notice = core::mem::take(&mut *slot);
            drop(slot);
            if notice.thread.is_null() {
                continue;
            }
            // Race repair: if `set_wake_reason_where` ran while this
            // thread was mid-park (state=Running on its way to
            // Suspended), the eager-promote CAS failed but the
            // wake_override bit is set. Now that state has committed
            // to Suspended, check the bit before filing the entry —
            // if non-zero, eagerly promote here instead of letting
            // the thread wait for its deadline.
            let t = unsafe { (notice.thread as *mut Thread).as_mut_unchecked() };
            if t.wake_override.load(Ordering::Acquire) != 0 {
                if t.state
                    .compare_exchange(
                        ThreadState::Suspended as usize,
                        ThreadState::Ready as usize,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    let pending = t.wake_override.swap(0, Ordering::AcqRel);
                    t.last_wake_reason.store(pending, Ordering::Release);
                    self.ready.push(notice.thread);
                    // Skip the heap push — entry would be stale
                    // immediately anyway (state=Ready).
                    continue;
                }
                // CAS failed: state was already Ready (a concurrent
                // promotion won). The thread is queued; nothing to
                // do here — also skip the heap push for the same
                // staleness reason.
                continue;
            }
            self.sleeping
                .push(notice.thread, notice.wake_time, notice.sleep_seq);
        }

        let now = riscv::register::time::read64();
        let ready = &mut self.ready;
        self.sleeping.drain_woken(now, |thread_ptr| {
            // SAFETY: heap entries name live registry threads — see
            // SLEEP_INBOX safety doc. We're under MANAGER_LOCK; no
            // other writer touches state/wake_override here.
            let t = unsafe { (thread_ptr as *mut Thread).as_mut_unchecked() };
            // Mirror the (now-deleted) Suspended arm in
            // `get_runnable_thread`: consume any pending wake_override
            // bits into last_wake_reason so userspace can later query
            // why it woke (timer-only wakes leave the bitmask 0).
            let pending = t.wake_override.swap(0, Ordering::AcqRel);
            t.last_wake_reason.store(pending, Ordering::Release);
            t.state
                .store(ThreadState::Ready as usize, Ordering::Release);
            ready.push(thread_ptr);
        });
    }

    /// Drain every per-hart `READY_INBOXES` slot into `self.ready`.
    /// Producers use these inboxes to publish Ready transitions
    /// without touching `self.ready` directly (which is manager-only).
    pub(crate) fn drain_ready_inboxes(&mut self) {
        for inbox in READY_INBOXES.iter() {
            while let Some(mut slot) = inbox.pop_ref() {
                let notice = core::mem::take(&mut *slot);
                drop(slot);
                if notice.thread.is_null() {
                    continue;
                }
                self.ready.push(notice.thread);
            }
        }
    }

    /// Cycles until the earliest sleep-heap deadline, capped at the
    /// safety-net `cap` (so the manager still runs periodically and
    /// observes any new SLEEP_INBOX entries pushed after this read).
    /// Returns `cap` when the heap is empty or the earliest entry is
    /// further out than `cap`. Used by `k_hart_loop` to size the WFI
    /// timer so a near-term sleeper wakes on its own deadline rather
    /// than waiting for the next heartbeat.
    ///
    /// Manager-only: callers must hold `MANAGER_LOCK` (the heap is
    /// not synchronized for concurrent peeks).
    pub fn next_sleep_in_cycles(&self, now: u64, cap: u64) -> u64 {
        match self.sleeping.next_wake() {
            Some(t) if t > now => (t - now).min(cap),
            Some(_) => 0,
            None => cap,
        }
    }

    pub fn assign_threads(&mut self, context: &'static HartContext) {
        use orbit_core::sched::HartView;

        // Order matters: drain_sleeps may push freshly-woken sleepers
        // onto self.ready (so they get the same dispatch this pass),
        // then drain_ready_inboxes folds in non-manager Ready
        // transitions (preempted threads from other harts, and
        // unblocked threads pushed by signal_n's wake hook). After
        // this prelude, self.ready holds every runnable thread and
        // get_runnable_thread is purely a queue pop.
        self.drain_sleeps();
        self.drain_ready_inboxes();

        // `sscratch` on this hart points at its own HartContext inside the
        // contiguous array allocated at boot; subtract the hart id to get
        // the array base, then index for each remote. Built lazily so no
        // per-tick allocation happens.
        let hart_root = unsafe {
            (riscv::register::sscratch::read() as *const HartContext).sub(context.hart_id as usize)
        };

        let self_hart_id = context.hart_id as usize;
        let cpu_count = self.cpu_count;

        let self_view = HartView {
            hart_id: context.hart_id as usize,
            current: &context.current,
        };

        let remotes = (0..cpu_count)
            .filter(move |&i| i != self_hart_id)
            .map(move |i| {
                let hc = unsafe { hart_root.add(i).as_ref_unchecked() };
                HartView {
                    hart_id: hc.hart_id as usize,
                    current: &hc.current,
                }
            });

        let mut hw = crate::hw::RiscvHardware;
        orbit_core::sched::assign_threads(&self_view, remotes, self, &mut hw);
    }

    pub fn print_threads(&self) {
        for (_, t) in self.threads.iter() {
            let thread = unsafe { (t.0 as *const Thread).as_ref_unchecked() };

            info!(
                "tid{}: state{}",
                thread.tid,
                thread.state.load(Ordering::Acquire)
            );
        }
    }

    /// Kernel root table as a `RootTable` with the correct PA→VA bias for
    /// tables allocated from `table_pages`. Use this wherever walker/mapper
    /// helpers need to follow intermediate PPNs.
    fn root(&self) -> mmu::mmap::RootTable<'static> {
        unsafe { memmap::kernel_root_from_pa(PhysAddr::from(self.satp)) }
    }

    fn setup_igb(&mut self, device: &PciDevice) {
        device.print_info();

        let ort = self.root();

        let bar_kva = unsafe {
            let bar_size = device.get_bar_size(0) as u64;
            if bar_size > (2 * MB) {
                error!("bar2big");
                return;
            }

            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

            info!("mapping {}KB BAR0", bar_size / KB);

            // BAR0's PA stays at IGB_BAR_PA (we still program that into the
            // device's BAR register so the device decodes it on the bus);
            // kernel-side accesses go through a high-half KMMIO alias.
            let kva = match memmap::install_kmmio_alias(
                &ort,
                &mut pages,
                Self::IGB_BAR_PA..(Self::IGB_BAR_PA + bar_size),
            ) {
                Ok(v) => v,
                Err(_) => {
                    error!("failed to map bar");
                    return;
                }
            };

            device.write_bar(0, Self::IGB_BAR_PA as u32);

            riscv::register::satp::write(self.satp);
            riscv::asm::sfence_vma(0, 0);

            kva
        };

        unsafe {
            let (_, tx_ring_kva) = self
                .kernel_pages
                .alloc_kdmap(Layout::from_size_align_unchecked(TX_RING_BYTES, PAGE_SIZE))
                .expect("no e1000 tx ring");
            let tx_ring = tx_ring_kva
                .as_mut_ptr::<[TxDesc; TX_RING_LEN]>()
                .as_mut_unchecked();

            let (_, rx_ring_kva) = self
                .kernel_pages
                .alloc_kdmap(Layout::from_size_align_unchecked(RX_RING_BYTES, PAGE_SIZE))
                .expect("no e1000 rx ring");
            let rx_ring = rx_ring_kva
                .as_mut_ptr::<[RxDesc; RX_RING_LEN]>()
                .as_mut_unchecked();

            let (_, tx_bufs_kva) = self
                .kernel_pages
                .alloc_kdmap(Layout::from_size_align_unchecked(
                    TX_RING_BUFS_BYTES,
                    PAGE_SIZE,
                ))
                .expect("no e1000 tx bufs");
            let tx_bufs = tx_bufs_kva
                .as_mut_ptr::<[E1000Pbuf; TX_RING_LEN]>()
                .as_mut_unchecked();

            let (_, rx_bufs_kva) = self
                .kernel_pages
                .alloc_kdmap(Layout::from_size_align_unchecked(
                    RX_RING_BUFS_BYTES,
                    PAGE_SIZE,
                ))
                .expect("no e1000 rx bufs");
            let rx_bufs = rx_bufs_kva
                .as_mut_ptr::<[E1000Pbuf; RX_RING_LEN]>()
                .as_mut_unchecked();

            let mut e1000 = E1000::new(bar_kva as *mut u32, tx_ring, tx_bufs, rx_ring, rx_bufs);
            let mac = e1000.read_mac().unwrap();
            if let Err(_) = e1000.init_hw(mac) {
                // free everything ig
                error!("failed to init e1000");
            }

            let mut config = Config::new(EthernetAddress(mac).into());
            config.random_seed = 4;

            let iface = Interface::new(
                config,
                &mut e1000,
                smoltcp::time::Instant::from_micros(riscv::register::time::read() as i64 / 10),
            );

            let socket_reqs = (0..self.cpu_count)
                .map(|_| heapless::spsc::Queue::<crate::SocketReq, 8>::new())
                .collect();

            self.net_pkg.iface = Some(iface);
            self.net_pkg.phy = Some(e1000);
            self.net_pkg.socket_reqs = socket_reqs;

            // Publish a stable pointer to the e1000 so the PLIC handler
            // can ack ICR from trap context. The Some(E1000) lives
            // inside `self.net_pkg.phy`, which lives inside the heap-
            // allocated Orbit — pointer is stable for the kernel's
            // lifetime.
            if let Some(phy_ref) = self.net_pkg.phy.as_mut() {
                let raw = phy_ref as *mut E1000;
                crate::drivers::e1000::E1000_DEVICE.store(raw, Ordering::Release);
            }

            // Wire e1000 INTx → PLIC → push WakeEvent::Net so k_net
            // wakes the moment a packet lands instead of waiting up to
            // 10 ms for the heartbeat. QEMU virt swizzles PCI INTA on
            // slot N to PLIC source `32 + (N % 4)` (see the `pci@..`
            // node's `interrupt-map` in the DTS). Most e1000s sit on
            // pin INTA, so we use pin=1 and just compute by slot.
            let slot = (device.address >> 15) & 0x1F;
            let plic_irq = 32 + (slot as u32 % 4);
            if let Err(()) = crate::drivers::plic::plic_register(
                plic_irq,
                e1000_plic_handler,
                self.cpu_count - 1,
            ) {
                error!("e1000: plic_register failed for irq {}", plic_irq);
            }
            else {
                info!("e1000: PLIC IRQ {} → wake k_net", plic_irq);
            }

            let entrypoint = crate::k_net as *const () as usize;
            let a0 = (&mut self.net_pkg) as *mut NetPackage;
            match self.create_kernel_thread(entrypoint, Some(a0 as usize)) {
                Ok(tid) => {
                    info!("created knet thread tid={tid}");
                    // Latch the tid so `WakeEvent::Net` can target this
                    // thread specifically. Without this latch the wake
                    // would fan out to every kernel thread (k_gpu, etc.)
                    // — harmless to correctness but it pulls k_gpu out
                    // of its 50 ms park on every netch tickle, wastes
                    // CPU and worse can interfere with display refresh
                    // pacing.
                    self.net_thread_tid = Some(tid);
                }
                Err(_) => {
                    error!("failed to create knet thread");
                }
            }
        }
    }

    pub fn get_pci_info<'n>(&mut self, node: FdtNode<'n>) {
        let reg = match node.reg() {
            Ok(Some(mut r)) => match r.nth(0) {
                Some(re) => re,
                None => return,
            },
            _ => return,
        };

        info!("reg={reg:?}");

        let base = match reg.address::<u64>() {
            Ok(b) => b as usize,
            Err(_) => return,
        };

        let size = match reg.size::<u64>() {
            Ok(b) => b as usize,
            Err(_) => return,
        };

        info!("pci@{:08X}..{:08X}", base, base + size);

        // PCI config space lives at a high-half KMMIO alias instead of
        // identity-mapped at its PA — keeps the kernel root free of low-half
        // entries that would shadow user VA space.
        let pci_cfg_va = unsafe {
            let ort = self.root();
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
            let va = match memmap::install_kmmio_alias(
                &ort,
                &mut pages,
                (base as u64)..((base + size) as u64),
            ) {
                Ok(v) => v,
                Err(_) => {
                    error!("failed to map pci config space");
                    return;
                }
            };
            riscv::asm::sfence_vma(0, 0);
            va
        };

        let matches = pci::scan_pci(pci_cfg_va as usize, &[(0x8086, 0x100E)]);
        if matches.is_empty() {
            return;
        }

        self.setup_igb(&matches[0]);
    }

    pub fn get_environment_info(&mut self) {
        // Access the DTB through its KDMAP alias — map_kernel_self installs
        // it at `phys_to_kdmap(dtb_phys)` and no longer identity-maps the
        // dtb guard.
        let dtb_kva = memmap::phys_to_kdmap(PhysAddr::new(self.dtb_addr as u64));
        let fdt = unsafe { Fdt::from_raw_unchecked(dtb_kva.as_ptr()) };
        let root = fdt.root();

        // Two-phase walk: setup_plic must run before any device that
        // wants to register a PLIC handler (e1000, virtio-input, …).
        // The DTB child order isn't guaranteed, so collect PCI nodes
        // during the traversal and defer them — same pattern virtio
        // already uses below.
        let mut pci_nodes: Vec<_> = Vec::new();
        let mut nodes: Vec<_> = root.children().collect();
        while let Some(node) = nodes.pop() {
            let name = node.name();
            if name.starts_with("pci") {
                pci_nodes.push(node);
                continue;
            }
            if name.starts_with("plic") {
                self.setup_plic(&fdt);
                continue;
            }

            for child in node.children() {
                nodes.push(child);
            }
        }

        // PLIC is installed; now devices can register IRQ handlers.
        for node in pci_nodes {
            self.get_pci_info(node);
        }
        self.discover_virtio(&fdt);
        self.setup_virtio_gpu();
        self.setup_virtio_input();
        self.setup_virtio_blk();
    }

    fn setup_plic(&mut self, fdt: &Fdt<'_>) {
        let ort = self.root();
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        if unsafe { crate::drivers::plic::install(fdt, &ort, &mut pages) }.is_err() {
            error!("plic install failed");
        }
    }

    fn discover_virtio(&mut self, fdt: &Fdt<'_>) {
        let ort = self.root();
        crate::drivers::virtio_probe::discover(fdt, &ort, &mut self.table_pages);
    }

    fn setup_virtio_gpu(&mut self) {
        let Some(outcome) =
            crate::drivers::virtio_gpu_dev::setup_virtio_gpu(&mut self.kernel_pages)
        else {
            return;
        };

        // Build the Display + GpuPackage, hand ownership to k_gpu.
        let fb = unsafe {
            crate::drivers::fb::FrameBuffer::new(outcome.fb_kva, outcome.width, outcome.height)
        };
        let pkg = crate::drivers::k_gpu::GpuPackage {
            display: crate::drivers::display::Display::new(fb),
            fb_resource_id: outcome.resource_id,
        };
        crate::drivers::k_gpu::install_package(pkg);

        let entrypoint = crate::drivers::k_gpu::k_gpu as *const () as usize;
        if self.create_kernel_thread(entrypoint, None).is_err() {
            error!("virtio-gpu: failed to spawn k_gpu thread");
        }
    }

    fn setup_virtio_input(&mut self) {
        crate::drivers::virtio_input_dev::setup_virtio_input(&mut self.kernel_pages);
    }

    fn setup_virtio_blk(&mut self) {
        crate::drivers::virtio_blk_dev::setup_virtio_blk(&mut self.kernel_pages);
    }

    /// `stack_pa` is the physical base of the user stack. User PT leaves
    /// take PAs directly.
    fn map_stack(
        &mut self,
        root_table: &mmu::mmap::RootTable<'_>,
        stack_pa: PhysAddr,
        stackv: u64,
        stack_size: u64,
    ) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe {
            map_address_range(
                root_table,
                &mut pages,
                &MappingConfig {
                    permissions: PagePermissions::U | PagePermissions::R | PagePermissions::W,
                    levels: 3,
                    page_size: UPROC_STACK_GRAIN,
                    vaddr: VirtAddr::new(stackv),
                    paddr: stack_pa,
                    log: false,
                    supervisor_tag: SupervisorTag::None,
                },
                VirtAddr::new(stackv + stack_size),
                PhysAddr::new(stack_pa.get_raw() + stack_size),
            )
        }
    }

    fn map_trap_frame(
        &mut self,
        root_table: &mmu::mmap::RootTable<'_>,
        trap_frame_pa: PhysAddr,
        user_vaddr: u64,
    ) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe {
            map_address_range(
                root_table,
                &mut pages,
                &MappingConfig {
                    permissions: PagePermissions::R.into(),
                    levels: 4,
                    page_size: PAGE_SIZE as u64,
                    vaddr: VirtAddr::new(user_vaddr),
                    paddr: trap_frame_pa,
                    log: false,
                    supervisor_tag: SupervisorTag::None,
                },
                VirtAddr::new(user_vaddr + PAGE_SIZE as u64),
                PhysAddr::new(trap_frame_pa.get_raw() + PAGE_SIZE as u64),
            )
        }
    }

    pub fn add_new_thread_to_process(
        &mut self,
        pid: u16,
        entrypoint: usize,
        stack_size: u64,
        allowed_affinity: u64,
        affinity: u64,
        arg: usize,
    ) -> Result<(), ()> {
        if !self.processes.contains_key(&pid) {
            return Err(());
        }

        let slot = self
            .processes
            .get_mut(&pid)
            .unwrap()
            .thread_slots
            .alloc()
            .ok_or(())?;

        let root_table = unsafe {
            let addr = PhysAddr::from(self.processes.get(&pid).unwrap().satp);
            memmap::kernel_root_from_pa(addr)
        };

        let thread = match self.create_new_thread(
            pid,
            &root_table,
            entrypoint,
            slot,
            stack_size,
            allowed_affinity,
            affinity,
            arg,
        ) {
            Ok(t) => t,
            Err(e) => {
                self.processes
                    .get_mut(&pid)
                    .unwrap()
                    .thread_slots
                    .free(slot);
                return Err(e);
            }
        };

        let tid = thread.tid;
        let rpt = thread.root_table_addr();

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(thread);
        let tptr = Box::into_raw(t);
        info!("created uthread@{tptr:016X?},pid={pid},tid={tid},table={rpt:016X?}");

        let owning_process = self.processes.get_mut(&pid).unwrap();

        if !owning_process.threads.insert(tid) {
            // Reclaim the Box we leaked at `Box::into_raw(t)` so the
            // Thread allocation isn't lost on this rollback path.
            self.dealloc_thread(unsafe { Box::from_raw(tptr) });
            return Err(());
        }

        owning_process.thread_count = owning_process.thread_count.saturating_add(1);

        self.threads.insert(tid, PThread(tptr));
        // Constructor sets state=Ready; queue for the scheduler.
        self.ready.push(tptr);

        Ok(())
    }

    /// Build a fresh user thread for `pid`. Snapshots the current
    /// `Process.permissions` into [`Thread::permissions`] so the
    /// dispatch-site permission gate can read it without locking. If
    /// the process pledges later, the manager re-walks all live
    /// threads and rewrites this field.
    pub fn create_new_thread(
        &mut self,
        pid: u16,
        root_table: &mmu::mmap::RootTable<'_>,
        entrypoint: usize,
        slot: u16,
        stack_size: u64,
        allowed_affinity: u64,
        affinity: u64,
        arg: usize,
    ) -> Result<Thread, ()> {
        if !validate_user_stack_size(stack_size) {
            error!("invalid user stack size {stack_size}");
            return Err(());
        }

        let (stack_frame, stack_layout) = self.allocate_user_thread_stack(stack_size)?;

        let (tf_frame, trap_frame_kva) = match self.allocate_trap_frame() {
            Ok(v) => v,
            Err(_) => {
                self.user_pages.free(stack_frame, stack_layout);
                return Err(());
            }
        };

        // Snapshot PAs now — we need them for the map_* calls below and
        // also for the &mut Stack build, but the frames themselves move
        // into `PhysBacking` at the end. `&self` readers on Frame keep
        // the originals intact across these lines.
        let stack_pa = stack_frame.raw();
        let tf_pa = tf_frame.raw();

        let stack_vaddr = user_stack_vaddr(slot, stack_size);
        let guard_vaddr = user_stack_guard_vaddr(slot);
        let guard_size = user_stack_guard_size(stack_size);
        let trap_frame_vaddr = user_trap_frame_vaddr(slot);

        // Root table PA: derive directly from the borrowed `root_table`
        // handle. We only need the PPN to stamp into the new thread's
        // satp — the page belongs to the caller (the Process), and any
        // intermediates we materialize below land in it.
        let root_kva = memmap::KdmapVa::new(root_table.table as *const _ as u64);
        let root_pa = root_kva.to_phys();
        let root_ppn = root_pa.get_raw() as usize / PAGE_SIZE;

        if let Err(_) = self.map_stack(root_table, stack_pa, stack_vaddr, stack_size) {
            self.user_pages.free(stack_frame, stack_layout);
            self.kernel_pages
                .free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
            error!("failed to map stack");
            return Err(());
        }

        if let Err(_) = self.map_trap_frame(root_table, tf_pa, trap_frame_vaddr) {
            self.user_pages.free(stack_frame, stack_layout);
            self.kernel_pages
                .free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
            error!("failed to map trap frame");
            return Err(());
        }

        // Per-thread TLS — only when the binary's PT_TLS had memsz > 0.
        // Snapshot the template + sizes out of the Process now so we
        // can drop the borrow before touching the allocators.
        //
        // Allocation matches the stack convention: one 2-MiB-aligned
        // megapage covering the full UPROC_TLS_MAX reservation,
        // installed as a single L1 leaf. Trades up to ~2 MiB of
        // physical-per-thread for one PTE instead of up to 512 (and
        // a single-shot teardown). For umode's typical TLS (a few
        // bytes of `#[thread_local]`) the waste is real but bounded
        // and the code stays uniform with the stack mapping.
        let (tls_template, tls_memsz) = match self.processes.get(&pid) {
            Some(p) if p.tls_memsz > 0 => (p.tls_template.clone(), p.tls_memsz),
            _ => (None, 0),
        };
        let tls_vaddr = user_tls_vaddr(slot);
        let tls_backing: Option<(Frame<UserOnly>, Layout)> = if tls_memsz > 0 {
            let layout =
                match Layout::from_size_align(UPROC_TLS_MAX as usize, UPROC_STACK_GRAIN as usize) {
                    Ok(l) => l,
                    Err(e) => {
                        self.user_pages.free(stack_frame, stack_layout);
                        self.kernel_pages
                            .free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
                        error!("bad TLS layout: {e:?}");
                        return Err(());
                    }
                };
            let frame = match self.user_pages.alloc_pa(layout) {
                Some(f) => f,
                None => {
                    self.user_pages.free(stack_frame, stack_layout);
                    self.kernel_pages
                        .free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
                    error!("failed to alloc TLS megapage");
                    return Err(());
                }
            };
            // Zero the whole megapage (page may have been returned by
            // a previous process), then overwrite the leading filesz
            // bytes with the .tdata template — the trailing memsz -
            // filesz bytes are .tbss (already zero) and the megapage
            // tail above memsz is unused but kept zero for hygiene.
            unsafe {
                let mut w = user_page::UserPageWindow::map(frame.get_raw(), layout.size());
                let buf = w.as_mut_slice();
                buf.fill(0);
                if let Some(template) = tls_template.as_ref() {
                    let copy_len = core::cmp::min(template.len(), buf.len());
                    buf[..copy_len].copy_from_slice(&template[..copy_len]);
                }
            }
            // One L1 leaf at user_tls_vaddr(slot). R|W|U;
            // SupervisorTag::None — TLS isn't shared, doesn't get
            // revoked. levels=3 + page_size=UPROC_STACK_GRAIN matches
            // map_stack's shape so unmap symmetry is one call.
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
            let cfg = MappingConfig {
                permissions: (PagePermissions::U | PagePermissions::R | PagePermissions::W) as u64,
                levels: 3,
                page_size: UPROC_STACK_GRAIN,
                vaddr: VirtAddr::new(tls_vaddr),
                paddr: frame.raw(),
                log: false,
                supervisor_tag: SupervisorTag::None,
            };
            let map_result = unsafe {
                map_address_range(
                    root_table,
                    &mut pages,
                    &cfg,
                    VirtAddr::new(tls_vaddr + layout.size() as u64),
                    PhysAddr::new(frame.get_raw() + layout.size() as u64),
                )
            };
            if map_result.is_err() {
                self.user_pages.free(frame, layout);
                self.user_pages.free(stack_frame, stack_layout);
                self.kernel_pages
                    .free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
                error!("failed to map TLS into process");
                return Err(());
            }
            Some((frame, layout))
        }
        else {
            None
        };

        if let Some(proc) = self.processes.get_mut(&pid) {
            // Reserved vaddr range at slot top. No leaves — a fault inside
            // here is a stack overflow from slot N+1 (whose stack low end
            // is exactly this slot's top), which the page-fault path
            // turns into a thread kill once it consults proc.maps.
            // Slot 0's guard sits at slot top too; nothing overflows
            // into it (slot 0's own overflow falls below UPROC_STACK_BASE
            // into the unmapped span there) but the entry is still
            // recorded for layout uniformity.
            proc.insert_mapping(UserMapping {
                vaddr: guard_vaddr,
                len: guard_size,
                perms: 0,
                backing: None,
                kind: MappingKind::Guard { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr: stack_vaddr,
                len: stack_size,
                perms: (PagePermissions::U | PagePermissions::R | PagePermissions::W) as u64,
                backing: Some(PhysBacking::User {
                    frame: stack_frame,
                    layout: stack_layout,
                }),
                kind: MappingKind::Stack { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr: trap_frame_vaddr,
                len: PAGE_SIZE as u64,
                perms: PagePermissions::R as u64,
                backing: Some(PhysBacking::Shared {
                    frame: tf_frame,
                    layout: Self::THREAD_TRAP_FRAME_LAYOUT,
                }),
                kind: MappingKind::TrapFrame { slot },
            });
            if let Some((frame, layout)) = tls_backing {
                proc.insert_mapping(UserMapping {
                    vaddr: tls_vaddr,
                    len: layout.size() as u64,
                    perms: (PagePermissions::U | PagePermissions::R | PagePermissions::W) as u64,
                    backing: Some(PhysBacking::User { frame, layout }),
                    kind: MappingKind::Tls { slot },
                });
            }
        }

        let tid = self.next_tid();

        let (frame, stack) = unsafe {
            let f = trap_frame_kva.as_mut_ptr::<TrapFrame>();
            core::ptr::write_bytes(f as *mut u8, 0, PAGE_SIZE);

            // Stack was zeroed inside allocate_user_thread_stack via
            // UserPageWindow — `stack_pa` is a user_pages physical address
            // with no KDMAP alias under the kernel satp, so writing
            // through it here would fault. The &mut Stack reference below
            // is built for `Thread.stack` but never derefed kernel-side;
            // user code reaches the same backing via the user-VA mapping.
            let s = stack_pa.get_raw() as *mut Stack;

            (f.as_mut_unchecked(), s.as_mut_unchecked())
        };

        let mut satp = Satp::from_bits(0);
        satp.set_asid(pid as usize);
        satp.set_mode(riscv::register::satp::Mode::Sv48);
        satp.set_ppn(root_ppn);

        frame.regs[1] = entrypoint;
        frame.regs[2] = (stack_vaddr + stack_size - 16) as usize;
        // tp = x4 = regs[4]. (regs[3] is gp — RISC-V's global pointer,
        // not the thread pointer.) Variant-I model: tp points at the
        // start of the static TLS block. Set unconditionally to
        // user_tls_vaddr(slot); if the binary has no TLS the
        // reservation stays unmapped and any access faults clean.
        frame.regs[4] = tls_vaddr as usize;
        // a0 = x10 = regs[10]: opaque thread arg the spawn syscall
        // hands the new thread. `std::thread::spawn` boxes its
        // closure state and passes the boxed pointer here so the
        // C-ABI entry trampoline can read it as its first argument.
        frame.regs[10] = arg;
        frame.asid = pid as usize;

        info!(
            "ventry={:016X?},vsp=0x{:016X?},vtp=0x{:016X?},rpt_pa={:016X?}",
            entrypoint,
            frame.regs[2],
            frame.regs[4],
            root_pa.get_raw(),
        );

        // Snapshot the owning process's permissions for the new
        // thread's dispatch-gate read path. `processes.get(&pid)` was
        // the source of truth when create_new_thread was invoked
        // (caller holds MANAGER_LOCK across thread creation); fall
        // back to `Permissions::ZERO` only if the process record has
        // already been removed (impossible on the live spawn path,
        // defensive against future refactors). Fail-closed default
        // matches `Process::new`'s ZERO baseline.
        let perms_snapshot = self
            .processes
            .get(&pid)
            .map(|p| p.permissions)
            .unwrap_or(orbit_abi::perms::Permissions::ZERO);

        Ok(Thread {
            pc: AtomicUsize::new(entrypoint),
            satp,
            mode: SPP::User,
            tid,
            pid,
            ticks: 0,
            frame: frame,
            stack,
            // User threads track stack/trap-frame ownership via
            // `Process.maps` `PhysBacking` entries — these fields are
            // kthread-only.
            kernel_stack: None,
            kernel_trap_frame: None,
            state: AtomicUsize::new(ThreadState::Ready as usize),
            wake_time: 0,
            wake_override: AtomicU64::new(0),
            last_wake_reason: AtomicU64::new(0),
            sleep_seq: AtomicU64::new(0),
            handle: None,
            slot: Some(slot),
            fault_info: None,
            allowed_affinity,
            affinity: AtomicU64::new(affinity),
            cpu_ticks_total: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            syscall_count: AtomicU64::new(0),
            syscall_ticks: AtomicU64::new(0),
            permissions: perms_snapshot,
        })
    }

    /// Build a fresh process from `elf_blob`. If `argv_bytes` /
    /// `envp_bytes` is `Some(blob)`, the packed blob is installed at
    /// `USER_ARGV_BASE` / `USER_ENVP_BASE` before the process becomes
    /// runnable; install failure for either is non-fatal — the
    /// process still spawns and the child sees `0` for the
    /// corresponding slot in [`orbit_abi::user::argv_envp`]. Mirrors
    /// the warn-but-continue policy in `run_create_process_ex_req`.
    pub fn create_new_process(&mut self, proc_components: ProcessComponents) -> Result<u16, ()> {
        // Leak-localization: pair this with the ktables snapshots in
        // dealloc_process. `create_pid{N}: ktables consumed=B` reports the
        // total table_pages footprint installed at process-construction
        // time; a matching `dealloc pid{N}: ... root_freed=B after=B`
        // shows what was reclaimed. The two should bracket each other on
        // a leak-free run.
        let ktables_at_create = self.table_pages.allocated_bytes();
        let (root_pa, root_table) = self.create_new_page_table()?;
        let mut elf = self.load_elf(&root_table, proc_components.elf_blob)?;
        let pid = self.next_pid();

        let mut proc_satp = Satp::from_bits(0);
        proc_satp.set_ppn(root_pa.get_raw() as usize / PAGE_SIZE);
        proc_satp.set_mode(Mode::Sv48);
        proc_satp.set_asid(pid as usize);

        let mut proc = Process::new(pid, proc_components.parent_pid, proc_satp);
        // Migration default: `Process::new()` defaults to ZERO perms
        // (fail closed). Legacy CREATE_PROCESS / CREATE_PROCESS_EX
        // callers stamp BOOTSTRAP-shaped Permissions::ALL here so
        // they keep working without role-aware spawn arguments.
        // CREATE_PROCESS_V2 resolves perms through the role registry
        // and calls `install_permissions` itself with the
        // witness-derived value, overwriting this default.
        proc.install_permissions(orbit_abi::perms::Permissions::ALL);
        let slot = proc.thread_slots.alloc().ok_or(())?;

        // ELF segment backings are tracked on the process so dealloc_process
        // returns them to user_pages on teardown — previously dropped on the
        // floor here.
        proc.heap_pages.append(&mut elf.segments);

        // Stash the PT_TLS template (if any) so per-thread create can
        // copy-init the TLS block without re-walking the user PT.
        if let Some(t) = elf.tls.take() {
            proc.tls_template = Some(t.template);
            proc.tls_memsz = t.memsz;
            proc.tls_align = t.align;
        }

        // Insert the Process before creating the thread so create_new_thread
        // can record per-thread UserMappings (TrapFrame, eventually Stack/TLS)
        // into proc.maps via self.processes.get_mut.
        self.processes.insert(pid, proc);

        // Initial process thread: arg=0. There's no parent closure to
        // pass through — the binary's `_start` ignores a0. (argv is
        // installed at a fixed VA via `install_argv_blob`, not via
        // this register.)
        let thread = match self.create_new_thread(
            pid,
            &root_table,
            elf.entrypoint,
            slot,
            proc_components.stack_size,
            proc_components.allowed_affinity,
            proc_components.affinity,
            0,
        ) {
            Ok(t) => t,
            Err(e) => {
                // Process was inserted before create_new_thread, with
                // ELF segments tracked in heap_pages and ELF leaves +
                // intermediates installed in the user satp. Hand it to
                // dealloc_process for the full sweep — recursive unmap
                // walks the user-half tree, drains heap_pages backings,
                // frees the root. Zero parent_pid first so we don't
                // pollute the parent's dead_children with a phantom
                // (pid, 0) entry that nobody will wait_pid on.
                if let Some(mut proc) = self.processes.remove(&pid) {
                    proc.parent_pid = 0;
                    self.dealloc_process(proc);
                }
                return Err(e);
            }
        };
        let tid = thread.tid;

        if let Err(_) = self.map_kernel_into(&root_table) {
            error!("failed to map kernel into process");
            // Same shape as the create_new_thread Err arm above, but
            // proc.maps now also holds slot-0's Stack/TrapFrame/TLS
            // entries from the just-completed create_new_thread.
            // dealloc_process's proc.maps drain frees those backings;
            // recursive unmap covers their intermediates.
            if let Some(mut proc) = self.processes.remove(&pid) {
                proc.parent_pid = 0;
                self.dealloc_process(proc);
            }
            return Err(());
        }

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(thread);
        let tptr = Box::into_raw(t);
        info!(
            "created uprocess@{tptr:016X?},pid={pid},tid={tid},table_pa={:016X?}",
            root_pa.get_raw()
        );

        let proc = self.processes.get_mut(&pid).expect("just inserted");

        proc.threads.insert(tid);
        proc.thread_count = 1;

        self.threads.insert(tid, PThread(tptr));
        // Constructor sets state=Ready; queue for the scheduler.
        self.ready.push(tptr);

        // Register a scrollback source with k_gpu so the process's
        // console_write output lands somewhere. If the ring is full
        // the cmd is dropped — the user_prints path via UART still
        // works as a fallback. k_gpu picks up the InsertSource cmd
        // in its drain loop.
        let _ = crate::drivers::k_gpu::push_insert_source(
            crate::drivers::display::Source::Process(pid),
        );

        // Register a per-process stdin slot so input::dispatch has a
        // place to deliver keystrokes once the process becomes the
        // active source. Removed by `dealloc_process` on teardown.
        crate::kernel::stdin::register(pid);

        // Install argv / envp blobs if the caller provided them.
        // Same warn-but-continue policy as `run_create_process_ex_req`:
        // the process is alive and runnable; a blob-install failure
        // just means the child observes argc=0 / envc=0.
        if let Some(blob) = proc_components.argv_bytes {
            if self.install_argv_blob(pid, blob).is_err() {
                warn!(
                    "create_new_process: argv install failed for pid={pid}, child will see no args",
                );
            }
        }
        if let Some(blob) = proc_components.envp_bytes {
            if self.install_envp_blob(pid, blob).is_err() {
                warn!(
                    "create_new_process: envp install failed for pid={pid}, child will see no env",
                );
            }
        }

        debug!(
            "create pid{}: ktables consumed={}B (entry={}B now={}B)",
            pid,
            self.table_pages
                .allocated_bytes()
                .saturating_sub(ktables_at_create),
            ktables_at_create,
            self.table_pages.allocated_bytes(),
        );

        Ok(pid)
    }

    fn free_backings(&mut self, backings: Vec<PhysBacking>) {
        for b in backings {
            self.free_backing(b);
        }
    }

    pub fn load_elf(
        &mut self,
        root_table: &mmu::mmap::RootTable<'_>,
        elf_blob: &[u8],
    ) -> Result<orbital_elf::ElfInfo, ()> {
        let elf = match elf::ElfBytes::<LittleEndian>::minimal_parse(elf_blob) {
            Ok(e) => e,
            Err(e) => {
                error!("failed to parse umode elf: {e:?}");
                return Err(());
            }
        };

        let mut segment_allocations = Vec::new();

        let segments = match elf.segments() {
            Some(seg) => seg,
            None => {
                error!("load_elf fed bad bytes");
                return Err(())
            }
        };

        for segment in segments.iter() {
            let load_segment = segment.p_type == elf::abi::PT_LOAD;
            if !load_segment {
                continue;
            }

            if segment.p_vaddr < USER_TEXT_BASE {
                error!(
                    "illegal elf p_vaddr 0x{:X} (below USER_TEXT_BASE 0x{:X})",
                    segment.p_vaddr, USER_TEXT_BASE
                );
                return Err(());
            }

            if segment.p_memsz == 0 {
                continue;
            }

            info!("loading {segment:08x?}");

            let segment_data = match elf.segment_data(&segment) {
                Ok(seg) => seg,
                Err(e) => {
                    error!("error parsing loadable segment data: {e:?}");
                    return Err(());
                }
            };

            unsafe {
                // Size the backing by memsz, not filesz: pure-BSS segments
                // (filesz=0, memsz>0, as emitted once user ELFs grow any
                // uninitialized statics) need the memsz-sized allocation
                // even though there's nothing to copy in from the file.
                let seg_mem_size = core::cmp::max(segment_data.len(), segment.p_memsz as usize);
                let layout = Layout::from_size_align_unchecked(seg_mem_size, PAGE_SIZE);
                let seg_pa = match self.user_pages.alloc_pa(layout) {
                    Some(p) => p,
                    None => {
                        self.free_backings(segment_allocations);
                        error!("failed to alloc segment");
                        return Err(());
                    }
                };
                let paddr_start = seg_pa.get_raw();

                // Segment bytes are copied in + the bss tail is zeroed through
                // UserPageWindow so step 8's KDMAP-alias removal doesn't
                // regress this: the kernel can't deref user_pages PAs directly
                // post-split, only through the window's install/invalidate.
                {
                    let mut w = user_page::UserPageWindow::map(paddr_start, layout.size());
                    let buf = w.as_mut_slice();
                    let file_len = segment_data.len();
                    buf[..file_len].copy_from_slice(segment_data);
                    if segment.p_memsz > segment.p_filesz {
                        let tail = &mut buf[segment.p_filesz as usize..];
                        tail.fill(0);
                    }
                }

                segment_allocations.push(PhysBacking::User {
                    frame: seg_pa,
                    layout,
                });

                let vaddr_start = round_u64_down(segment.p_vaddr, PAGE_SIZE as u64);

                let segment_aligned_len = round_u64_up(seg_mem_size as u64, PAGE_SIZE as u64);

                let paddr_end = paddr_start + segment_aligned_len;
                let vaddr_end = vaddr_start + segment_aligned_len;

                let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

                let mut permissions = PagePermissions::U as u64;
                if (segment.p_flags & 0x1) > 0 {
                    permissions |= PagePermissions::X as u64;
                }
                if (segment.p_flags & 0x2) > 0 {
                    permissions |= PagePermissions::W as u64;
                }
                if (segment.p_flags & 0x4) > 0 {
                    permissions |= PagePermissions::R as u64;
                }

                let config = MappingConfig {
                    permissions,
                    levels: 4,
                    page_size: PAGE_SIZE as u64,
                    vaddr: VirtAddr::new(vaddr_start),
                    paddr: PhysAddr::new(paddr_start),
                    log: false,
                    supervisor_tag: SupervisorTag::None,
                };

                let map = map_address_range(
                    root_table,
                    &mut pages,
                    &config,
                    VirtAddr::new(vaddr_end),
                    PhysAddr::new(paddr_end),
                );

                if map.is_err() {
                    self.free_backings(segment_allocations);
                    error!("failed to map segment into process");
                    return Err(());
                }
            }
        }
        // PT_TLS — captured AFTER PT_LOAD because the TLS template's
        // initial bytes (.tdata) live inside the same file image we
        // just walked. Snapshot from `elf.segment_data` (kernel-side
        // file bytes), not via the user satp — keeps the snapshot
        // independent of the user mapping's permissions and saves a
        // PT walk per thread create. Only one PT_TLS allowed per ELF.
        let mut tls: Option<orbital_elf::TlsTemplate> = None;
        for segment in segments.iter() {
            if segment.p_type != elf::abi::PT_TLS {
                continue;
            }
            if segment.p_memsz == 0 {
                // Empty PT_TLS — emitted by the linker even when the
                // binary has no `#[thread_local]`. Treat as "no TLS"
                // so thread create skips the allocation.
                continue;
            }
            if segment.p_memsz > UPROC_TLS_MAX {
                error!(
                    "elf PT_TLS p_memsz=0x{:X} exceeds UPROC_TLS_MAX=0x{:X}",
                    segment.p_memsz, UPROC_TLS_MAX,
                );
                self.free_backings(segment_allocations);
                return Err(());
            }
            if tls.is_some() {
                error!("elf has more than one PT_TLS segment");
                self.free_backings(segment_allocations);
                return Err(());
            }
            let template_bytes = match elf.segment_data(&segment) {
                Ok(s) => s,
                Err(e) => {
                    error!("error reading PT_TLS segment data: {e:?}");
                    self.free_backings(segment_allocations);
                    return Err(());
                }
            };
            // template_bytes.len() == p_filesz (the file image of the
            // segment). The trailing `p_memsz - p_filesz` bytes are
            // implicit zeros and never stored.
            tls = Some(orbital_elf::TlsTemplate {
                template: template_bytes.to_vec(),
                memsz: segment.p_memsz as usize,
                align: segment.p_align as usize,
            });
            info!(
                "elf PT_TLS: filesz=0x{:X} memsz=0x{:X} align=0x{:X}",
                segment.p_filesz, segment.p_memsz, segment.p_align,
            );
        }

        Ok(orbital_elf::ElfInfo {
            entrypoint: elf.ehdr.e_entry as usize,
            segments: segment_allocations,
            tls,
        })
    }

    fn map_kernel_into(&mut self, root_table: &mmu::mmap::RootTable<'_>) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe {
            memmap::map_kernel_shared(
                root_table,
                &mut pages,
                &self.layout,
                /*is_kernel_root=*/ false,
            )
        }
    }

    fn next_tid(&mut self) -> u32 {
        let mut next = self.current_thread_id.wrapping_add(1);
        loop {
            let matches = self.threads.iter().filter(|(t, _)| next == **t).count();

            if matches == 0 {
                break;
            }
            next = next.wrapping_add(1);
        }

        self.current_thread_id = next;

        next
    }

    /// §13a.3 — does the named process have an argv blob installed?
    /// Backs the argv half of the `argv_envp` syscall return pair
    /// (returns `USER_ARGV_BASE` if true, `0` otherwise).
    pub fn process_has_argv(&self, pid: u16) -> bool {
        self.processes
            .get(&pid)
            .map(|p| p.argv_blob.is_some())
            .unwrap_or(false)
    }

    /// §13e — does the named process have an envp blob installed?
    /// Backs the envp half of the `argv_envp` syscall return pair
    /// (returns `USER_ENVP_BASE` if true, `0` otherwise).
    pub fn process_has_envp(&self, pid: u16) -> bool {
        self.processes
            .get(&pid)
            .map(|p| p.envp_blob.is_some())
            .unwrap_or(false)
    }

    fn next_pid(&mut self) -> u16 {
        let mut next = self.current_process_id.wrapping_add(1);
        loop {
            let matches = self
                .processes
                .iter()
                .filter(|(pid, _)| **pid == next)
                .count();

            if matches == 0 {
                break;
            }
            next = next.wrapping_add(1);

            if next == 0 {
                next = 1;
            }
        }

        self.current_process_id = next;

        next
    }

    pub fn check_net(&mut self) {
        while let Some((pid, socket_handle)) = self.net_pkg.socket_associations.dequeue() {
            if let Some(process) = self.processes.get_mut(&(pid as u16)) {
                process.sockets.insert(socket_handle);

                info!("associated socket {socket_handle:?} with pid{pid}");
            }
            else {
                if let Err(e) = self.net_pkg.socket_deletions.enqueue(socket_handle) {
                    error!("failed to queue socket for deletion: {e:?}");
                }
            }
        }
    }
}

impl orbit_core::sched::Scheduler for Orbit {
    fn next_runnable(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        // PThread wraps a raw ptr sourced from the thread registry (Box
        // allocations); returning it directly keeps provenance rooted
        // at that allocation — no `&mut` reborrow whose tag would be
        // popped on return (which would dangle the ptr stored in the
        // target hart's `current` slot).
        self.get_runnable_thread(hart_mask).map(|pt| pt.0)
    }
}

pub fn ksleep(duration: Duration) {
    let context = get_hart_context();
    let current_thread =
        unsafe { (context.current.load(Ordering::Acquire) as *mut Thread).as_mut_unchecked() };

    const TICKS_PER_MS: usize = 10_000;
    current_thread.wake_time = riscv::register::time::read()
        .wrapping_add((duration.as_millis() as usize).wrapping_mul(TICKS_PER_MS));
}
