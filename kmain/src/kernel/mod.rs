use core::alloc::Layout;
use core::sync::atomic::{AtomicI64, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use core::time::Duration;

use alloc::collections::btree_map::BTreeMap;
use alloc::{boxed::Box, vec::Vec};

use device::{HartContext, Stack, TrapFrame};
use dtoolkit::fdt::FdtNode;
use dtoolkit::{Node, fdt::Fdt};
use elf::endian::LittleEndian;
use mem::{round_u64_down, round_u64_up};
use mmu::mmap::{PageAlloc, map_address_range, unmap, unmap_page, unmap_range};
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
    EAGAIN, EBADF, EFAULT, EINVAL, EIO, EMFILE, ENOEXEC, ENOMEM, ENOTDIR, EPERM, ESRCH, Errno,
};
use orbit_core::ready_queue::ReadyQueue;
use orbit_core::sleep_heap::SleepHeap;
use orbit_core::{
    CloseHandleReq, CreateProcessExReq, CreateProcessReq, CreateThreadReq, EventFdCreateReq,
    FbSurfaceCreateReq, FbSurfaceDestroyReq, FsOpenReq, FsReadReq, FsReaddirReq, FsStatReq,
    FutexWaitReq, FutexWakeReq, MAX_FS_PATH_LEN, MemMapReq, NetChannelCreationReq, PendingWork,
    WaitPidReq, WakeTidReq,
};
use thingbuf::StaticThingBuf;

use crate::kernel::fs::FsErr;
use crate::kernel::handle::{EventFdSlot, Handle, OpenFile, ProcessHandles};
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
pub mod key_events;
pub mod memmap;
pub mod orbital_elf;
pub mod page_cache;
pub mod pci;
pub mod pending_frees;
pub mod shared_frame;
pub mod shared_user_ptr;
pub mod shootdown;
pub mod stdin;
pub mod surface;
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
///
/// Cap bumped 64 → 128 alongside the on-thread completion migration:
/// every manager-resolved blocking syscall now pushes a `Tid` event
/// per signal (instead of routing through `CompletionHandle`'s wake
/// hook), so a manager pass that resolves a burst of pending work
/// pushes one event per resolved item. Telemetry — `wake_queue_peak`
/// and `wake_queue_drops` exported via `query_stats` — reports
/// whether 128 is sufficient for the live workload.
pub static WAKE_QUEUE: StaticThingBuf<WakeEvent, 128> = StaticThingBuf::new();

/// High-water mark of `WAKE_QUEUE.len()` observed at any push. Sampled
/// after each successful push via [`wake_queue_push`] (`fetch_max`),
/// so the value is monotonic for the kernel's lifetime. Surfaces queue
/// pressure: a peak approaching cap (currently 64) is the cue to bump
/// the cap, since drops are silent in most callers. Read by
/// `query_stats` for the userspace stats command.
pub static WAKE_QUEUE_PEAK: AtomicU64 = AtomicU64::new(0);

/// Count of `WAKE_QUEUE.push()` attempts that EAGAIN'd because the
/// ring was full. Each drop is a missed wake — semantically harmless
/// in the cases that exist today (e1000 IRQ has a 10 ms heartbeat
/// fallback; net pushes coalesce), but a non-zero counter is the
/// signal that the cap is undersized for the workload. Bumped from
/// any hart via the [`wake_queue_push`] helper.
pub static WAKE_QUEUE_DROPS: AtomicU64 = AtomicU64::new(0);

/// Push a [`WakeEvent`] onto [`WAKE_QUEUE`] and update telemetry.
/// Returns `Err(ev)` if the queue is full — caller decides whether to
/// log the drop, retry, or coalesce. The drop counter is bumped here
/// regardless so a global "are we losing wakes?" answer is always
/// available without the call sites needing to coordinate.
///
/// Trap-context-safe: two atomic ops (push + counter) on success,
/// one on failure. No allocations, no locks.
pub fn wake_queue_push(ev: WakeEvent) -> Result<(), WakeEvent> {
    match WAKE_QUEUE.push(ev) {
        Ok(()) => {
            // `len()` after the push gives the post-push depth. Racy
            // across pushers (a concurrent pop can shrink it before
            // we sample), but `fetch_max` is monotonic so under-
            // sampling can never inflate the peak — only miss it,
            // which is fine for a high-water diagnostic.
            let depth = WAKE_QUEUE.len() as u64;
            let _ = WAKE_QUEUE_PEAK.fetch_max(depth, Ordering::Relaxed);
            Ok(())
        }
        Err(e) => {
            WAKE_QUEUE_DROPS.fetch_add(1, Ordering::Relaxed);
            Err(e.into_inner())
        }
    }
}

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
    /// Wake the thread parked on a process's `ProcessKeyEvents` ring.
    /// Carries the tid latched at park time. Pushed by
    /// `kernel::input::dispatch` from trap context after a
    /// `push_event`; the manager's `drain_wakes` runs
    /// `set_wake_reason_where(INPUT_IO, |t| t.tid == tid)` which
    /// eagerly promotes the Suspended thread to Ready.
    InputTid(u32),
    /// Wake the k_gpu compositor thread. Pushed by every producer
    /// that adds a command to `CONSOLE_RING` (`push_chunk`,
    /// `push_cycle_active`, `push_insert_source`,
    /// `push_present_surface`). Without this, k_gpu sleeps out its
    /// 50 ms park before draining new commands — visible to the user
    /// as up-to-50 ms latency on pane cycles, redraws, and
    /// console echoes, plus dropped commands once `CONSOLE_RING`
    /// fills before the next park-expiry. Targets `gpu_thread_tid`
    /// (latched in `setup_virtio_gpu`); falls back to "wake all
    /// pid=0 kthreads" during the boot window.
    Gpu,
    /// Wake the k_serial UART-drain thread. Mirror of `Gpu` for the
    /// serial side — `ktrace::emit` (and any future serial producer)
    /// pushes onto `SERIAL_RING`, the manager folds those into one
    /// nudge per pass via `nudge_serial_if_pending`. Targets
    /// `serial_thread_tid` (latched in `setup_serial_kthread`);
    /// falls back to a coarse pid=0 scan during the boot window
    /// before the latch.
    Serial,
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

/// On-thread completion analog of [`wake_blocked_inline`]. If the
/// thread has a SIGNALED `pending_state`, marshal `pending_rets` into
/// `frame.regs[10..]`, clear the signal, transition Ready, and push
/// the per-hart `READY_INBOX` notice. Returns `true` on a signal hit,
/// `false` if no rets were pending (the cheap path: one Acquire load).
///
/// Same call-site contract as `wake_blocked_inline`: invoked from the
/// parker's post-publish re-check in `apply_syscall_outcome`. Takes
/// `*mut Thread` (rather than `&Thread`) so the brief `&mut Thread`
/// reborrow needed for the `frame.regs` write doesn't alias the outer
/// `&Thread` binding the caller holds for atomic ops.
pub fn try_wake_pending_inline(thread_ptr: *mut Thread) -> bool {
    if thread_ptr.is_null() {
        return false;
    }
    // SAFETY: caller (the parker on its own hart) just Release-stored
    // `state == Blocking` and `current == null`, so no other hart can
    // be running this thread or hold a competing reference. The
    // `&mut Thread` here is the only live access for the duration of
    // this call.
    // Probe through `&Thread` first — `take_pending_results` is a CAS
    // claim, so we don't materialize a `&mut Thread` until we've won
    // exclusive logical ownership. If we lose to the manager-side
    // drain, no `&mut` ever existed and the parker just returns to
    // the kernel loop with `state == Blocking` (the manager's drain
    // already transitioned it to Ready).
    let t = unsafe { (thread_ptr as *const Thread).as_ref_unchecked() };
    let mut rets = [0i64; 4];
    let Some(n) = t.take_pending_results(&mut rets)
    else {
        return false;
    };
    let tid = t.tid;
    // `t_ref: &Thread` is Copy and its borrow ends at the last
    // use above; the &mut reborrow below doesn't alias.
    //
    // SAFETY: the take CAS just succeeded. The competing path
    // (`set_wake_reason_where`'s eager-promote arm) will see NONE
    // and bail without forming a `&mut` of its own. We have
    // exclusive logical ownership of `frame.regs` and `state` for
    // the duration of this call.
    let t = unsafe { (thread_ptr as *mut Thread).as_mut_unchecked() };
    for i in 0..n {
        t.frame.regs[10 + i] = rets[i] as usize;
    }
    t.state
        .store(ThreadState::Ready as usize, Ordering::Release);
    if push_ready_notice(thread_ptr).is_err() {
        error!(
            "READY_INBOX full on post-publish re-check: tid={} — \
             thread marked Ready but not queued",
            tid,
        );
    }
    true
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
    let pushed = wake_queue_push(WakeEvent::Net).is_ok();
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
    /// TID of the k_gpu compositor thread. Latched in
    /// `setup_virtio_gpu` after the kthread is created; consumed by
    /// `WakeEvent::Gpu` to nudge k_gpu out of its 50 ms park as soon
    /// as a producer (`push_chunk`, `push_cycle_active`,
    /// `push_insert_source`, `push_present_surface`) lands a command
    /// on `CONSOLE_RING`. `None` during the boot window before
    /// virtio-gpu init completes; `WakeEvent::Gpu` falls back to
    /// "wake all pid=0 kthreads" in that case.
    gpu_thread_tid: Option<u32>,
    /// TID of the k_serial UART-drain thread. Latched in
    /// `setup_serial_kthread`; consumed by `WakeEvent::Serial` to
    /// pull k_serial out of its 50 ms park as soon as a producer
    /// (`ktrace::emit`) lands a chunk on `SERIAL_RING`. `None` during
    /// the boot window before the kthread spawns; the wake falls back
    /// to a coarse pid=0 scan in that case.
    serial_thread_tid: Option<u32>,
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

    /// Block-keyed page cache. `None` until [`setup_page_cache`]
    /// runs at boot. Mediates every fs read: lookup → Ready (sync
    /// copy) / Loading (register waiter) / Absent (start a fill).
    /// Mutated only under MANAGER_LOCK.
    pub page_cache: Option<crate::kernel::page_cache::PageCache>,

    /// Per-tid in-flight fs_read state. A multi-page read may
    /// fan out across several cache slots and several waiters; each
    /// completion decrements `bytes_pending` and the last one
    /// resumes the parked thread with `bytes_done` (or `-EIO` on
    /// any per-page failure). Empty for synchronous all-hits reads
    /// — the manager just calls `resume_thread_with_value` inline.
    fs_reads_in_progress: BTreeMap<u32, FsReadInProgress>,

    /// Per-tid in-flight path-mode spawn state machines. Created
    /// by the path-mode arm of `run_create_process_v2_req` once
    /// it's done validation + blob alloc + page-0 read; advanced
    /// by `run_cache_fill` for each completing kernel waiter;
    /// destroyed when the last page lands (install runs, handle
    /// signaled). Replaces the old k_io kthread.
    spawns_in_progress: BTreeMap<u32, SpawnInProgress>,

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
/// (drop-without-signal — the matching `Thread.handle` drops in
/// `dealloc_thread`, so no consumer is left pointing at gone state).
#[derive(Debug)]
pub struct FutexWaiter {
    /// Tid of the parking thread. `run_futex_wake_req` resumes each
    /// woken waiter via `Orbit::publish_pending_for_tid(tid, &[0])`
    /// — the on-thread completion analog of the prior
    /// `handle.signal(0)`. Stale tids (parker exited before wake)
    /// are silently dropped by the resume helper.
    pub tid: u32,
    /// Pid of the parking thread. Read by `dealloc_process` to drop
    /// every waiter whose owner is going away — without that sweep,
    /// `futex_waiters` retains entries past the death of the parking
    /// thread, and a later `futex_wake` on the same PA would resolve
    /// the stale tid (now potentially recycled into a new thread).
    /// FutexWaiter.pid was reserved for exactly this sweep.
    pub pid: u16,
    /// Reserved: absolute tick deadline for `-ETIMEDOUT`. `0` = no
    /// timeout. v1 always parks `0` regardless of the user-supplied
    /// `timeout_ns` (the timeout-scan path lands when std::sync needs
    /// it).
    pub deadline_ticks: u64,
}

/// Frame-pool size for the page cache. 64 frames × `PAGE_SIZE` =
/// 256 KiB of cached file pages — comfortably exceeds the
/// `MAX_FS_READ_LEN` (64 KiB → up to 16 cache pages) so a single
/// large read can't exhaust the pool, and big enough to absorb
/// tarfs metadata + a couple of open binaries.
pub const PAGE_CACHE_CAPACITY: usize = 64;

/// Per-tid in-flight state for a `fs_read` call that fanned out
/// across at least one cache miss / coalesced waiter. Created in
/// [`Orbit::run_fs_read_req`] when any waiter gets registered;
/// destroyed by the per-waiter completion arm of
/// [`Orbit::run_cache_fill`] once `bytes_pending` reaches zero
/// (which then resumes the parked tid with `bytes_done` or `-EIO`).
///
/// All-Ready-hits reads never construct one — the manager copies
/// inline and resumes immediately.
#[derive(Debug)]
pub struct FsReadInProgress {
    /// Sum of `len` across the still-outstanding waiters. Decreases
    /// to zero as each cache fill completes.
    pub bytes_pending: u32,
    /// Bytes successfully copied into the user buffer so far.
    /// Frozen on first failure; if `failed` is set, the eventual
    /// resume returns `-EIO` regardless of this.
    pub bytes_done: u32,
    /// Sticky flag: any per-page failure makes the entire read
    /// resolve as `-EIO`. Strict POSIX-prefix semantics (return the
    /// contiguous successful prefix, ignore subsequent failures) is
    /// a v2 improvement.
    pub failed: bool,
}

/// Per-tid state machine for a path-mode `create_process_v2` spawn.
/// The manager initiates a read of page 0 inline at v2-dispatch
/// time, parks the caller via the carried `handle`, and returns. As
/// each per-page `CacheFill` lands the manager checks
/// `spawns_in_progress[tid]`; if found and more pages remain, it
/// issues the next page; if all pages have landed it runs
/// [`Orbit::install_spawn`] and signals the handle with the new
/// pid (or errno).
///
/// The state machine replaces the old k_io kthread loop, which
/// parked synchronously on per-sector handles. Splitting the work
/// across cache events keeps the manager non-blocking — each event
/// handler runs short.
pub struct SpawnInProgress {
    /// Validated context captured at v2 dispatch time. Used to
    /// drive `install_spawn` once the blob is fully populated.
    pub ctx: orbit_core::SpawnContext,
    /// Resolved inode of the spawn-source path. Used by
    /// `lba_for_page` for each page read.
    pub inode: crate::kernel::fs::Inode,
    /// Growable buffer the cache fills write directly into. Sized
    /// to the file's reported length at start.
    pub blob: alloc::vec::Vec<u8>,
    /// File size in bytes (`= blob.len()` once filled).
    pub total_size: u64,
    /// Number of pages that have completed the cache → blob copy
    /// step. Increments per `CacheFill`. The next page to issue is
    /// `pages_done` (when issuing matches `pages_done == issued`,
    /// which is true here since each issue waits for completion
    /// before issuing the next — a future optimization could
    /// pipeline issues, at which point `pages_issued` becomes a
    /// separate counter).
    pub pages_done: u64,
    /// `ceil(total_size / PAGE_SIZE)` — total page reads we'll do.
    pub total_pages: u64,
    // Caller's parker tid is the BTreeMap key (`spawns_in_progress`),
    // not stored here. `advance_spawn` / `issue_next_spawn_page` use
    // their `tid` argument directly when calling
    // `Orbit::publish_pending_for_tid` to resume the parker.
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
            kernel_heap_bytes: crate::tracked_heap::KHEAP.allocated_bytes() as u64,
            syscall_ticks,
            hart_user_ticks,
            hart_kernel_ticks,
            hart_scheduler_ticks,
            hart_idle_ticks,
            perm_denials: proc.perm_denials.load(Ordering::Relaxed),
            role_denials: proc.role_denials.load(Ordering::Relaxed),
            wake_queue_peak: WAKE_QUEUE_PEAK.load(Ordering::Relaxed),
            wake_queue_drops: WAKE_QUEUE_DROPS.load(Ordering::Relaxed),
            wake_queue_capacity: WAKE_QUEUE.capacity() as u64,
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
            gpu_thread_tid: None,
            serial_thread_tid: None,
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
            page_cache: None,
            fs_reads_in_progress: BTreeMap::new(),
            spawns_in_progress: BTreeMap::new(),
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
            pending_rets: [
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
            ],
            pending_state: AtomicU8::new(0),
            pending_ret_count: AtomicU8::new(0),
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
            // Kthread credentials: uid 0 across the triplet. Kthreads
            // never invoke the U-mode getuid/getgid syscalls, so this
            // is observability-only — but the all-zeroes default
            // matches `Process::new` and stays consistent with the
            // "kthreads run as root-equivalent" mental model.
            uid: 0,
            euid: 0,
            suid: 0,
            gid: 0,
            egid: 0,
            sgid: 0,
            // Kernel threads don't `console_write` via the U-mode
            // syscall path; trace output goes through `Source::Kernel`
            // directly. `None` keeps the field consistent with the
            // U-mode contract (no redirect ⇒ writes go to own pane).
            stdout_redirect: None,
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

    /// `fb_surface_create(w, h, format)` — allocate a `kernel_pages`-
    /// backed pixel surface, map it user-writable in the calling
    /// process's shared range, and register a per-process surface table
    /// entry. Returns `(handle_id, user_va)` on success; the caller of
    /// the manager wraps these as `signal_pair` args.
    ///
    /// Errnos surface via the negative isize convention; both return
    /// slots carry the errno on failure (caller looks at the first).
    fn run_fb_surface_create_req(
        &mut self,
        req: FbSurfaceCreateReq,
        pid: u16,
        root_pa: PhysAddr,
    ) -> (isize, isize) {
        use orbit_abi::fb::FbFormat;

        // Format already validated at the syscall boundary; defensive
        // re-check here lets the manager assume well-formed input.
        let Some(format) = FbFormat::from_u32(req.format_raw)
        else {
            return (Errno::new(EINVAL).to_ret(), 0);
        };
        if req.width == 0 || req.height == 0 {
            return (Errno::new(EINVAL).to_ret(), 0);
        }

        let bpp = format.bytes_per_pixel() as usize;
        let Some(pixel_bytes) = (req.width as usize)
            .checked_mul(req.height as usize)
            .and_then(|n| n.checked_mul(bpp))
        else {
            return (Errno::new(EINVAL).to_ret(), 0);
        };

        // Round size up to PAGE_SIZE so the mapping is page-aligned at
        // both ends. v1 uses 4 KiB pages always — large surfaces still
        // map at page granularity. Megapage promotion is a follow-up.
        let size_bytes = (pixel_bytes + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        let layout = match Layout::from_size_align(size_bytes, PAGE_SIZE) {
            Ok(l) => l,
            Err(_) => return (Errno::new(EINVAL).to_ret(), 0),
        };

        // Allocate from kernel_pages so the compositor's per-frame blit
        // can run through the KDMAP alias without going back through the
        // user PT.
        let Some((frame, kva)) = self.kernel_pages.alloc_kdmap(layout)
        else {
            warn!(
                "fb_surface_create: kernel_pages alloc failed for {} bytes",
                size_bytes
            );
            return (Errno::new(ENOMEM).to_ret(), 0);
        };

        // Zero the surface bytes before the user PTE exists — same
        // hygiene as run_nc_create_req. Previous tenant pixels can't
        // leak across processes.
        unsafe {
            core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, size_bytes);
        }

        // Look up the per-process surface registry; lazily register
        // (the create_new_process hook should have done this already
        // but defending against it lets a runtime-spawned source still
        // work).
        let surfaces = match crate::kernel::surface::get(pid) {
            Some(s) => s,
            None => {
                crate::kernel::surface::register(pid);
                match crate::kernel::surface::get(pid) {
                    Some(s) => s,
                    None => {
                        self.kernel_pages.free(frame, layout);
                        return (Errno::new(ESRCH).to_ret(), 0);
                    }
                }
            }
        };

        // Reserve a fresh shared-range VA. Cursor only ever increases;
        // destroyed surfaces don't recycle their VA. 62 TiB shared
        // range absorbs millions of creates per process before
        // exhaustion.
        let user_va = surfaces.alloc_va(size_bytes as u64);
        if user_va + size_bytes as u64 > orbit_abi::layout::UPROC_SHARED_END {
            self.kernel_pages.free(frame, layout);
            warn!("fb_surface_create: pid={pid} exhausted shared VA range");
            return (Errno::new(ENOMEM).to_ret(), 0);
        }

        let config = MappingConfig {
            permissions: (PagePermissions::R as u64)
                | (PagePermissions::W as u64)
                | (PagePermissions::U as u64),
            levels: 4,
            page_size: PAGE_SIZE as u64,
            vaddr: VirtAddr::new(user_va),
            paddr: frame.raw(),
            log: false,
            // SupervisorTag::None — surfaces are managed via the
            // explicit fb_surface_destroy / dealloc_process paths,
            // not the SharedUserPtr revoke walk.
            supervisor_tag: SupervisorTag::None,
        };

        let vend = VirtAddr::new(user_va + size_bytes as u64);
        let pend = PhysAddr::new(frame.get_raw() + size_bytes as u64);

        unsafe {
            let root_table = memmap::kernel_root_from_pa(root_pa);
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
            if map_address_range(&root_table, &mut pages, &config, vend, pend).is_err() {
                warn!("fb_surface_create: map_address_range failed for pid={pid} va=0x{user_va:X}");
                self.kernel_pages.free(frame, layout);
                return (Errno::new(ENOMEM).to_ret(), 0);
            }
        }

        if !self.processes.contains_key(&pid) {
            // Process disappeared between the cursor bump and the map.
            // Roll back the mapping (best effort) and free the frame.
            unsafe {
                let root_table = memmap::kernel_root_from_pa(root_pa);
                let _ = unmap_range(&root_table, user_va..user_va + size_bytes as u64);
            }
            self.kernel_pages.free(frame, layout);
            return (Errno::new(ESRCH).to_ret(), 0);
        }

        let id = surfaces.alloc_id();
        let entry = crate::kernel::surface::SurfaceEntry {
            user_va,
            kdmap_kva: kva.raw(),
            width: req.width,
            height: req.height,
            format,
            size_bytes,
            backing: PhysBacking::Shared { frame, layout },
        };
        surfaces.insert(id, entry);

        core::sync::atomic::fence(Ordering::SeqCst);
        riscv::asm::sfence_vma(pid as usize, 0);
        crate::kernel::shootdown::broadcast(0, 0);

        info!(
            "fb_surface_create: pid={pid} id={id} {}x{} user_va=0x{:X} kva=0x{:016X} size={}",
            req.width,
            req.height,
            user_va,
            kva.raw(),
            size_bytes
        );
        (id as isize, user_va as isize)
    }

    /// `fb_surface_destroy(handle)` — remove the surface from the
    /// per-process table, unmap its user VA range, and return the
    /// backing frame to `kernel_pages`.
    ///
    /// Note: there is a brief window between `Cmd::RemoveSource`
    /// being queued and k_gpu draining it where the Display still
    /// references the soon-to-be-freed kdmap KVA. For surface
    /// destroy specifically (not process teardown) we rely on the
    /// caller having stopped issuing `fb_present` on this handle —
    /// any lingering damage from prior presents is rendered with
    /// stale-but-still-mapped bytes one last time before the unmap
    /// lands. Process-exit teardown has the same race; v1 accepts
    /// both as best-effort.
    fn run_fb_surface_destroy_req(
        &mut self,
        req: FbSurfaceDestroyReq,
        pid: u16,
        root_pa: PhysAddr,
    ) -> isize {
        let Some(surfaces) = crate::kernel::surface::get(pid)
        else {
            return Errno::new(ESRCH).to_ret();
        };

        let Some(entry) = surfaces.remove(req.handle)
        else {
            return Errno::new(EBADF).to_ret();
        };

        // Clear leaf PTEs across the surface's user VA range.
        // unmap_range expects 4 KiB-aligned bounds — surface VA was
        // chosen page-aligned at create time and size_bytes was rounded
        // up to PAGE_SIZE, so the bounds are clean.
        unsafe {
            let root_table = memmap::kernel_root_from_pa(root_pa);
            if let Err(()) = unmap_range(
                &root_table,
                entry.user_va..entry.user_va + entry.size_bytes as u64,
            ) {
                warn!(
                    "fb_surface_destroy: unmap_range failed pid={pid} handle={} va=0x{:X}",
                    req.handle, entry.user_va
                );
                // Continue anyway — leaving stale PTEs would be worse
                // than letting the free path proceed; the user PT is
                // about to be freed at next dealloc anyway.
            }
        }

        core::sync::atomic::fence(Ordering::SeqCst);
        riscv::asm::sfence_vma(pid as usize, 0);
        crate::kernel::shootdown::broadcast(0, 0);

        // Return the backing to kernel_pages.
        self.free_backing(entry.backing);

        info!(
            "fb_surface_destroy: pid={pid} handle={} user_va=0x{:X}",
            req.handle, entry.user_va
        );
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

        // Existence check *before* the mapping: the manager holds
        // MANAGER_LOCK for this whole handler, so a process that's alive
        // here stays alive through the install below. Mapping first would
        // leave a recycled frame mapped R/W/U in a dead process's PT on
        // this early return (SharedInner::drop only enqueues the frame,
        // it never revokes the PTEs).
        if !self.processes.contains_key(&pid) {
            warn!("nc create: no owning process {req:?}");
            self.kernel_pages.free(frame, layout);
            return (Errno::new(ESRCH).to_ret(), 0);
        }

        unsafe {
            let root_table = memmap::kernel_root_from_pa(root_pa);
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

            if map_address_range(&root_table, &mut pages, &config, vend, pend).is_err() {
                warn!("nc create: map failed {req:?}");
                self.kernel_pages.free(frame, layout);
                return (Errno::new(ENOMEM).to_ret(), 0);
            }
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
        let Some(fd) = self
            .process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new)
            .insert(Handle::NetChannel(shared.clone()))
        else {
            warn!("nc create: pid{pid} handle table exhausted (fd > i32::MAX)");
            // The PTEs are already installed; the dropped NetChannel handle
            // only enqueues the frame to pending_frees, so without this
            // revoke the recycled frame would stay mapped R/W/U in the user
            // PT. `shared` is still live here — `.insert` took a clone.
            let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
            if let Err(e) = shared.revoke(&root_table) {
                warn!("nc create: revoke after EMFILE failed: {e:?}");
            }
            return (Errno::new(EMFILE).to_ret(), 0);
        };

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
        trace!("handling close req: {req:?}");

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
            Handle::Stdin | Handle::Stdout | Handle::Stderr => {
                // Stdio handles are zero-sized markers — the actual
                // sinks (console scrollback, key-event ring) live
                // outside the handle table and don't need per-slot
                // teardown. Closing 0/1/2 just removes the slot.
            }
            Handle::EventFd(slot) => {
                // Wake-on-close cleanup is wired against the
                // kernel-side parked-tid shadow but no kernel path
                // currently parks a reader (no `read(fd)` dispatch
                // exists for EventFd yet), so `kernel_parked_tid` is
                // always zero here today and the wake branch is a
                // dormant scaffold for the eventual POSIX read(fd)
                // path. Leaving the load + check in place so the
                // path lights up the moment the read-syscall arm
                // starts stamping the shadow.
                let parked = slot
                    .kernel_parked_tid
                    .load(core::sync::atomic::Ordering::Acquire);
                if parked != 0 {
                    let _ = wake_queue_push(WakeEvent::Tid(parked));
                }
                if let Err(e) = slot.region.revoke(&root_table) {
                    warn!(
                        "close_handle: eventfd revoke failed for fd={}: {e:?}",
                        req.fd
                    );
                    return Errno::new(EIO).to_ret();
                }
                drop(slot);
            }
        }
        0
    }

    /// `eventfd(vaddr_hint, initval, flags)` — manager path. Mirrors
    /// `run_nc_create_req` shape: validate, allocate one
    /// `kernel_pages` frame, initialize the [`EventFd`](orbit_abi::event_fd::EventFd)
    /// header in-place via KDMAP, map the page user-RW SharedRevocable
    /// at `vaddr_hint`, build a `SharedUserPtr<EventFdRegion>`, and
    /// install a `Handle::EventFd` slot with the `cloexec`/`nonblock`
    /// bits derived from `flags`.
    fn run_eventfd_create_req(
        &mut self,
        req: EventFdCreateReq,
        pid: u16,
        root_pa: PhysAddr,
    ) -> (isize, isize) {
        use orbit_abi::event_fd::{
            EFD_CLOEXEC, EFD_NONBLOCK, EVENTFD_REGION_SIZE, EventFd as EventFdRegion,
        };

        info!("handling eventfd req: {req:08X?}");

        let region_size = EVENTFD_REGION_SIZE;
        let layout = match Layout::from_size_align(region_size, PAGE_SIZE) {
            Ok(l) => l,
            Err(e) => {
                warn!("eventfd: bad layout {e:?}");
                return (Errno::new(EINVAL).to_ret(), 0);
            }
        };

        let Some((frame, kva)) = self.kernel_pages.alloc_kdmap(layout)
        else {
            warn!("eventfd: alloc failed for {} bytes", region_size);
            return (Errno::new(ENOMEM).to_ret(), 0);
        };

        // Zero then init the header — user observes a fully-init region
        // before the PTE ever lands.
        unsafe {
            core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, region_size);
            EventFdRegion::init(kva.as_mut_ptr::<u8>(), req.initval, req.flags);
        }

        let config = MappingConfig {
            permissions: (PagePermissions::R as u64)
                | (PagePermissions::W as u64)
                | (PagePermissions::U as u64),
            levels: 4,
            page_size: PAGE_SIZE as u64,
            vaddr: VirtAddr::new(req.vaddr_hint.raw()),
            paddr: frame.raw(),
            log: false,
            supervisor_tag: SupervisorTag::SharedRevocable,
        };
        let vend = VirtAddr::new(req.vaddr_hint.raw() + region_size as u64);
        let pend = PhysAddr::new(frame.get_raw() + region_size as u64);

        // Existence check *before* the mapping: the manager holds
        // MANAGER_LOCK for this whole handler, so a process that's alive
        // here stays alive through the install below. Mapping first would
        // leave a recycled frame mapped R/W/U in a dead process's PT on
        // this early return (SharedInner::drop only enqueues the frame,
        // it never revokes the PTEs).
        if !self.processes.contains_key(&pid) {
            warn!("eventfd: no owning process pid{pid}");
            self.kernel_pages.free(frame, layout);
            return (Errno::new(ESRCH).to_ret(), 0);
        }

        unsafe {
            let root_table = memmap::kernel_root_from_pa(root_pa);
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

            if map_address_range(&root_table, &mut pages, &config, vend, pend).is_err() {
                warn!("eventfd: map failed {req:?}");
                self.kernel_pages.free(frame, layout);
                return (Errno::new(ENOMEM).to_ret(), 0);
            }
        }

        let region: SharedUserPtr<EventFdRegion> =
            SharedUserPtr::new(frame, layout, req.vaddr_hint, region_size, pid);

        let efd_slot = EventFdSlot {
            region: region.clone(),
            kernel_parked_tid: core::sync::atomic::AtomicU32::new(0),
        };

        let cloexec = req.flags & EFD_CLOEXEC != 0;
        let nonblock = req.flags & EFD_NONBLOCK != 0;

        let Some(fd) = self
            .process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new)
            .insert_with_flags(Handle::EventFd(efd_slot), cloexec, nonblock)
        else {
            warn!("eventfd: pid{pid} handle table exhausted (fd > i32::MAX)");
            // The PTEs are already installed; the dropped EventFdSlot only
            // enqueues the frame to pending_frees, so without this revoke
            // the recycled frame would stay mapped R/W/U in the user PT.
            // `region` is a retained clone — `efd_slot`'s went into the
            // dropped handle.
            let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
            if let Err(e) = region.revoke(&root_table) {
                warn!("eventfd: revoke after EMFILE failed: {e:?}");
            }
            return (Errno::new(EMFILE).to_ret(), 0);
        };

        core::sync::atomic::fence(Ordering::SeqCst);
        riscv::asm::sfence_vma(pid as usize, 0);
        crate::kernel::shootdown::broadcast(0, 0);

        info!(
            "eventfd created user_va=0x{:08X} kva=0x{:016X} fd={fd}",
            req.vaddr_hint.raw(),
            kva.raw(),
        );
        (req.vaddr_hint.raw() as isize, fd as isize)
    }

    /// `wake_tid(target_tid)` — manager path. Validates that
    /// `target_tid` belongs to `caller_pid` by walking the per-process
    /// thread set, then pushes `WakeEvent::Tid(target_tid)`.
    ///
    /// Returns:
    /// - `0` on success.
    /// - `-ESRCH` if `target_tid` isn't a live thread anywhere.
    /// - `-EPERM` if `target_tid` exists but belongs to a different pid.
    /// - `-EAGAIN` if the wake queue is full (transient).
    fn run_wake_tid_req(&mut self, req: WakeTidReq, caller_pid: u16) -> isize {
        let Some(pt) = self.threads.get(&req.target_tid)
        else {
            return Errno::new(ESRCH).to_ret();
        };
        let target_pid = unsafe { (pt.0 as *const process::Thread).as_ref_unchecked().pid };
        if target_pid != caller_pid {
            return Errno::new(EPERM).to_ret();
        }
        if wake_queue_push(WakeEvent::Tid(req.target_tid)).is_err() {
            return Errno::new(EAGAIN).to_ret();
        }
        0
    }

    /// Resolve a user-supplied fs path against `pid`'s cwd. Absolute
    /// inputs are returned as-is; relative inputs become `<cwd>/<path>`
    /// (with a single `/` between them). Allocates because the prefix
    /// is dynamic. Falls back to `/<path>` if the process record has
    /// vanished — a defensive shape; the caller's pid is always alive
    /// in practice since the syscall is on its own thread.
    fn resolve_fs_path(&self, pid: u16, path: &str) -> alloc::string::String {
        use alloc::string::String;
        if path.starts_with('/') {
            return alloc::borrow::ToOwned::to_owned(path);
        }
        // Normalize a relative `path` against the caller's cwd:
        // strip any leading `./` so "./foo" resolves to "<cwd>/foo",
        // and treat a bare "." as the cwd itself. Without this,
        // `eza` (which defaults to listing ".") gets handed "/."
        // which the tarfs walker doesn't recognize.
        let mut tail = path;
        loop {
            tail = match tail {
                "." => "",
                t if t.starts_with("./") => &t[2..],
                _ => break,
            };
        }
        let mut out = String::new();
        if let Some(p) = self.processes.get(&pid) {
            out.push_str(&p.cwd);
        }
        else {
            out.push('/');
        }
        if !tail.is_empty() {
            if !out.ends_with('/') {
                out.push('/');
            }
            out.push_str(tail);
        }
        out
    }

    /// POSIX `vaccess`-style check against the calling process's
    /// effective credentials. Returns the raw negative isize errno on
    /// the deny path so handlers can `return e;` directly. The pure
    /// rule logic + unit tests live in
    /// [`orbit_abi::fs::vaccess`]; this method is a thin adapter that
    /// reads the credential triple off `Process` and forwards.
    ///
    /// `ESRCH` if `pid` isn't in the process table — same shape as
    /// the other `run_*_req` helpers.
    fn vaccess_pid(&self, pid: u16, st: &orbit_abi::fs::Stat, want: u32) -> Result<(), isize> {
        let proc = match self.processes.get(&pid) {
            Some(p) => p,
            None => return Err(Errno::new(ESRCH).to_ret()),
        };
        orbit_abi::fs::vaccess(proc.euid, proc.egid, &proc.groups, st, want).map_err(|e| e.to_ret())
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
        let raw =
            match self.copy_user_path(root_pa, req.path_vaddr.raw(), req.path_len, &mut path_buf) {
                Ok(p) => p,
                Err(e) => return e,
            };
        let resolved = self.resolve_fs_path(pid, raw);
        let path = resolved.as_str();
        let inode = match fs.open(path) {
            Ok(i) => i,
            Err(FsErr::NotFound) => return Errno::new(orbit_abi::errno::ENOENT).to_ret(),
            Err(_) => return Errno::new(EIO).to_ret(),
        };
        // Snapshot kind for the read/readdir gate. The page cache
        // owns all DMA scratch now, so per-fd state is just
        // `(offset, dir_cursor, is_regular)`.
        let stat = match fs.stat(inode) {
            Ok(s) => s,
            Err(_) => return Errno::new(EIO).to_ret(),
        };
        // POSIX mode-bit access check. Today fs_open is read-only
        // (`OPEN_RDONLY = 0`), so we always need R. When O_WRONLY /
        // O_RDWR land, derive `want` from `req.flags` instead.
        // Path-walk traversal checks (X on each parent dir) are TBD
        // — tarfs's `open(path)` is a single-pass internal walk that
        // doesn't surface intermediate inodes. v1 just checks the
        // final inode; that's already a meaningful improvement over
        // "any process can read anything that role-FS_RO permits."
        if let Err(e) = self.vaccess_pid(pid, &stat, orbit_abi::fs::ACCESS_R_OK) {
            debug!(
                "fs_open: vaccess EACCES pid={pid} path={path} mode={:#o}",
                stat.st_mode
            );
            return e;
        }
        let is_regular = (stat.st_mode & orbit_abi::fs::S_IFMT) == orbit_abi::fs::S_IFREG;
        // Lazy-create the handle table — same pattern create_netch
        // uses, since a process that opens a file before ever creating
        // a NetChannel won't have an entry yet.
        let Some(fd) = self
            .process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new)
            .insert(Handle::File(OpenFile {
                fs,
                inode,
                offset: 0,
                dir_cursor: 0,
                is_regular,
            }))
        else {
            warn!("fs_open: pid{pid} handle table exhausted (fd > i32::MAX)");
            return Errno::new(EMFILE).to_ret();
        };
        debug!("fs_open: pid={pid} path={path} → fd={fd} ino={inode}");
        fd as isize
    }

    /// Cache-driven `fs_read`. Walks the requested range page-by-
    /// page, dispatching each user-buffer slice via the page cache:
    /// Ready hits copy out synchronously; Loading slots register a
    /// waiter and coalesce; Absent slots allocate + submit a DMA
    /// (`submit_blk_read_cached`).
    ///
    /// All-Ready (or all-EOF) reads resume `tid` inline with the
    /// total bytes copied. Reads that registered any waiter insert
    /// an [`FsReadInProgress`] entry keyed by tid; the matching
    /// completions in [`run_cache_fill`] decrement `bytes_pending`,
    /// and the last completion resumes the parked thread.
    ///
    /// Per-page failures during the walk (lba lookup, virt_to_phys,
    /// PoolExhausted, submit) truncate the read at the offset
    /// reached so far. If at least one byte landed by then the read
    /// completes with that byte count; if zero, it resolves to
    /// `-EIO` (or the more specific errno for sync-error cases like
    /// EBADF / EISDIR / EFAULT).
    fn run_fs_read_req(&mut self, req: FsReadReq, pid: u16, root_pa: PhysAddr, tid: u32) {
        const PAGE: u64 = PAGE_SIZE as u64;
        use crate::kernel::page_cache::{CacheKey, SlotState, Waiter};

        // Snapshot fd-side state up front, drop the handle-table
        // borrow before touching `self.page_cache`. `is_regular`
        // rides on the legacy per-fd scratch sentinel (Some =
        // regular, None = directory) — the field stays for now
        // since we'll retire it wholesale in the cleanup pass.
        let (fs, inode, prev_off, is_regular) = {
            let Some(ph) = self.process_handles.get_mut(&pid)
            else {
                self.resume_thread_with_value(tid, Errno::new(EBADF).to_ret());
                return;
            };
            let Some(handle_ref) = ph.get_mut(req.fd)
            else {
                self.resume_thread_with_value(tid, Errno::new(EBADF).to_ret());
                return;
            };
            let Handle::File(of) = handle_ref
            else {
                self.resume_thread_with_value(tid, Errno::new(EBADF).to_ret());
                return;
            };
            (of.fs, of.inode, of.offset, of.is_regular)
        };

        if !is_regular {
            self.resume_thread_with_value(tid, Errno::new(orbit_abi::errno::EISDIR).to_ret());
            return;
        }
        if self.page_cache.is_none() {
            self.resume_thread_with_value(tid, Errno::new(EAGAIN).to_ret());
            return;
        }

        let file_size = match fs.size(inode) {
            Ok(s) => s,
            Err(_) => {
                self.resume_thread_with_value(tid, Errno::new(EIO).to_ret());
                return;
            }
        };
        if prev_off >= file_size {
            // EOF — sync resume with 0; don't touch the device.
            self.resume_thread_with_value(tid, 0);
            return;
        }

        let cap = core::cmp::min(req.len as u64, file_size - prev_off) as u32;
        let buf_va = req.buf_vaddr.raw();
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let dev_id = fs.dev_id();

        // Bytes written into the user buffer synchronously
        // (Ready-cache hits + EOF short-fills).
        let mut bytes_done: u32 = 0;
        // Bytes registered as outstanding waiters; resolved
        // asynchronously via run_cache_fill.
        let mut bytes_pending: u32 = 0;
        // Running total `bytes_done + bytes_pending`. Sole offset
        // beyond `buf_va` for per-slice user-VA arithmetic — the
        // bug we fixed was double-adding `cache_offset` on top.
        let mut total_committed: u32 = 0;
        // Sticky: any per-page validation/submit failure flips this
        // and breaks the walk; partial-success bytes still complete.
        let mut walk_failed = false;
        // Set when a Ready slot's `valid_bytes` is shorter than its
        // slice (truncated file). Sync path; logically "EOF mid-walk".
        let mut hit_eof = false;

        // File-offset cursor (advances per outer iter) and its
        // upper bound (`prev_off + cap`).
        let end_off = prev_off + cap as u64;
        let mut cur_off = prev_off;

        'pages: while cur_off < end_off {
            // File-relative page index for this cache slice.
            let target_page = cur_off / PAGE;
            // Byte offset *into* `target_page` where this read
            // starts. Nonzero only on the first outer iter when the
            // caller's `prev_off` was misaligned; subsequent iters
            // are page-aligned.
            let cache_intra = (cur_off & (PAGE - 1)) as u32;
            // Bytes still wanted by this fs_read call.
            let in_call_remaining = end_off - cur_off;
            // Bytes from `cache_intra` to the end of this cache page.
            let in_page_remaining = PAGE - cache_intra as u64;
            // Bytes this outer iter will process (sum of inner-loop
            // slice_lens for this iter equals this value).
            let cache_slice_len = core::cmp::min(in_call_remaining, in_page_remaining) as u32;

            let lba = match fs.lba_for_page(inode, target_page) {
                Ok(l) => l,
                Err(_) => {
                    walk_failed = true;
                    break 'pages;
                }
            };
            let key = CacheKey { dev: dev_id, lba };
            // File offset of `target_page`'s first byte; combined
            // with `file_size` it gives the slot's `valid_bytes`.
            let page_off_bytes = target_page * PAGE;
            // File-valid byte count for this whole cache page,
            // clamped at EOF. Stored on the slot so the completion
            // path can clamp without re-deriving from inode size.
            let valid_bytes = core::cmp::min(PAGE, file_size - page_off_bytes) as u32;

            // Inner loop subdivides `cache_slice_len` across
            // however many user pages the destination span touches.
            // A single cache slice might land in 1 (aligned) or 2
            // (straddling) user pages. `cache_offset` is the local
            // progress counter; user-VA arithmetic uses
            // `total_committed` only — see the bug note above.
            let mut cache_offset: u32 = 0;
            while cache_offset < cache_slice_len {
                // User VA of the next byte to write.
                let user_dst_va = buf_va + total_committed as u64;
                // 4 KiB-aligned VA of the user page containing
                // `user_dst_va`, plus the byte offset within it.
                let user_page_va = user_dst_va & !(PAGE - 1);
                let user_page_off = (user_dst_va - user_page_va) as u32;
                let Some(user_page_pa) =
                    (unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(user_page_va)) })
                else {
                    walk_failed = true;
                    break 'pages;
                };
                let user_page_pa = PhysAddr::new(user_page_pa as u64);

                // Bytes left in the current user page (`PAGE -
                // user_page_off`) and the current cache slice
                // (`cache_slice_len - cache_offset`); copy the
                // smaller of the two this iteration.
                let user_remaining = PAGE as u32 - user_page_off;
                let cache_remaining = cache_slice_len - cache_offset;
                let slice_len = core::cmp::min(user_remaining, cache_remaining);
                // Byte offset *into the cache page* this iteration
                // reads from. Becomes the waiter's `intra` field.
                let waiter_intra = cache_intra + cache_offset;

                // Lookup. Snapshot `(kva, valid_bytes)` for hits so
                // the borrow on cache drops before we mutate it.
                #[derive(Copy, Clone)]
                enum Action {
                    Hit { src_kva: u64, valid: u32 },
                    Loading,
                    Absent,
                }
                let action = {
                    let cache = self.page_cache.as_ref().unwrap();
                    match cache.lookup(key) {
                        Some(SlotState::Ready { frame, valid_bytes }) => Action::Hit {
                            src_kva: frame.kva().raw(),
                            valid: *valid_bytes,
                        },
                        Some(SlotState::Loading { .. }) => Action::Loading,
                        None => Action::Absent,
                    }
                };

                match action {
                    Action::Hit { src_kva, valid } => {
                        let copy_len =
                            core::cmp::min(slice_len, valid.saturating_sub(waiter_intra));
                        if copy_len > 0 {
                            unsafe {
                                let mut w = user_page::UserPageWindow::map(
                                    user_page_pa.get_raw(),
                                    PAGE_SIZE,
                                );
                                let dst = w.as_mut_slice();
                                let src = core::slice::from_raw_parts(
                                    (src_kva + waiter_intra as u64) as *const u8,
                                    copy_len as usize,
                                );
                                dst[user_page_off as usize
                                    ..user_page_off as usize + copy_len as usize]
                                    .copy_from_slice(src);
                            }
                        }
                        let cache = self.page_cache.as_mut().unwrap();
                        cache.record_hit();
                        cache.touch_lru(key);
                        bytes_done += copy_len;
                        total_committed += copy_len;
                        if copy_len < slice_len {
                            // EOF inside a Ready slot — file is
                            // shorter than we thought (only really
                            // happens if a writable FS truncated;
                            // tarfs is immutable). Stop walking.
                            hit_eof = true;
                            break 'pages;
                        }
                    }
                    Action::Loading => {
                        let waiter = Waiter::User {
                            tid,
                            pid,
                            intra: waiter_intra,
                            user_page_pa,
                            user_page_off,
                            len: slice_len,
                        };
                        let cache = self.page_cache.as_mut().unwrap();
                        if cache.register_waiter(key, waiter).is_err() {
                            walk_failed = true;
                            break 'pages;
                        }
                        bytes_pending += slice_len;
                        total_committed += slice_len;
                    }
                    Action::Absent => {
                        let waiter = Waiter::User {
                            tid,
                            pid,
                            intra: waiter_intra,
                            user_page_pa,
                            user_page_off,
                            len: slice_len,
                        };
                        let begin =
                            self.page_cache
                                .as_mut()
                                .unwrap()
                                .begin_load(key, valid_bytes, waiter);
                        let dma_pa = match begin {
                            Ok(pa) => pa,
                            Err(_) => {
                                walk_failed = true;
                                break 'pages;
                            }
                        };
                        let packed = crate::kernel::page_cache::pack(key);
                        match unsafe {
                            crate::drivers::virtio_blk_dev::submit_blk_read_cached(
                                lba,
                                dma_pa.get_raw(),
                                PAGE as u32,
                                packed,
                            )
                        } {
                            Ok(_head) => {
                                bytes_pending += slice_len;
                                total_committed += slice_len;
                            }
                            Err(_) => {
                                // Tear down the slot we just made.
                                // complete_slot drains the lone
                                // waiter we registered; we discard
                                // it (no FsReadInProgress yet, so
                                // nothing to update).
                                let _ = self.page_cache.as_mut().unwrap().complete_slot(key, 1);
                                walk_failed = true;
                                break 'pages;
                            }
                        }
                    }
                }

                cache_offset += slice_len;
            }

            cur_off += cache_slice_len as u64;
        }

        // Decision time.
        if bytes_pending == 0 {
            // Synchronous: every slice was a Ready hit (or EOF).
            // Advance fd offset by what we actually copied and
            // resume the thread with that count (or `-EIO` only if
            // the walk failed before producing any bytes).
            if let Some(ph) = self.process_handles.get_mut(&pid)
                && let Some(Handle::File(of)) = ph.get_mut(req.fd)
            {
                of.offset = prev_off + bytes_done as u64;
            }
            let result = if walk_failed && bytes_done == 0 {
                Errno::new(EIO).to_ret()
            }
            else {
                bytes_done as isize
            };
            self.resume_thread_with_value(tid, result);
            return;
        }

        // Async path: speculatively advance the offset past every
        // committed byte. Insert FsReadInProgress so the
        // CacheFill arms can decrement bytes_pending and resume.
        let _ = hit_eof; // EOF inside a Ready slot already broke walking; nothing async to do.
        if let Some(ph) = self.process_handles.get_mut(&pid)
            && let Some(Handle::File(of)) = ph.get_mut(req.fd)
        {
            of.offset = prev_off + total_committed as u64;
        }
        self.fs_reads_in_progress.insert(
            tid,
            FsReadInProgress {
                bytes_pending,
                bytes_done,
                failed: walk_failed,
            },
        );
    }

    fn run_fs_stat_req(&mut self, req: FsStatReq, pid: u16, root_pa: PhysAddr) -> isize {
        let Some(fs) = crate::kernel::fs::mounted()
        else {
            return Errno::new(EIO).to_ret();
        };
        let mut path_buf = [0u8; MAX_FS_PATH_LEN];
        let raw =
            match self.copy_user_path(root_pa, req.path_vaddr.raw(), req.path_len, &mut path_buf) {
                Ok(p) => p,
                Err(e) => return e,
            };
        let resolved = self.resolve_fs_path(pid, raw);
        let path = resolved.as_str();
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
        debug!(
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
        trace!(
            "fs_readdir: pid={pid} fd={} ino={inode} cursor={cursor}->{next_cursor} bytes={written}",
            req.fd
        );
        written as isize
    }

    /// Manager arm for [`PendingWork::QueryDenials`]. Drains any
    /// PermDeny / RoleDeny events the dispatch gate has pushed since
    /// the last manager pass (so a caller racing the manager observes
    /// its own gate-induced denial), snapshots the kernel-wide ring,
    /// and writes up to `buf_len` bytes — capped at `DENIAL_RING_CAPACITY *
    /// size_of::<DenialEvent>()` and to whole events — into the user
    /// buffer via a single `UserPageWindow` against `root_pa`. Returns
    /// `bytes_written` or a negative errno.
    ///
    /// Single-page constraint enforced by `handle_query_denial_log`
    /// (matches `fs_stat` / `fs_readdir`); `50 events × 48 bytes` =
    /// 2400 B fits in any page-aligned 4 KiB slot.
    fn run_query_denials(
        &mut self,
        buf_vaddr: orbit_abi::layout::UserVa,
        buf_len: usize,
        _pid: u16,
        root_pa: PhysAddr,
    ) -> isize {
        use orbit_abi::denial::{DENIAL_RING_CAPACITY, DenialEvent};

        self.drain_denial_events();

        let mut tmp = [DenialEvent::PermDeny {
            required_class: 0,
            perms: 0,
            time_ticks: 0,
            tid: 0,
            sysno: 0,
            source_role: 0,
            pid: 0,
        }; DENIAL_RING_CAPACITY];
        let n = self.denial_ring_snapshot(&mut tmp);

        let event_size = core::mem::size_of::<DenialEvent>();
        let max_events = buf_len / event_size;
        let to_emit = core::cmp::min(n, max_events);
        let to_write = to_emit * event_size;
        if to_write == 0 {
            return 0;
        }

        let buf_va = buf_vaddr.raw();
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = buf_va & !(PAGE_SIZE as u64 - 1);
        let page_off = (buf_va - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            let src = core::slice::from_raw_parts(tmp.as_ptr() as *const u8, to_write);
            page[page_off..page_off + to_write].copy_from_slice(src);
        }
        to_write as isize
    }

    /// Manager arm for [`PendingWork::QueryStats`]. Snapshots
    /// per-process accounting for `target_pid` (today: the caller's
    /// own pid) under `MANAGER_LOCK` and copies the resulting
    /// [`orbit_abi::stats::ProcessStats`] into the user buffer via
    /// a single `UserPageWindow`. Returns `bytes_written`, `-ESRCH`
    /// if the pid is no longer live, or `-EFAULT` on PT lookup
    /// failure.
    fn run_query_stats(
        &mut self,
        target_pid: u16,
        buf_vaddr: orbit_abi::layout::UserVa,
        buf_len: usize,
        root_pa: PhysAddr,
    ) -> isize {
        use orbit_abi::errno::ESRCH;
        use orbit_abi::stats::ProcessStats;

        let stats = match self.snapshot_process_stats(target_pid) {
            Some(s) => s,
            None => return Errno::new(ESRCH).to_ret(),
        };

        let native = core::mem::size_of::<ProcessStats>();
        let to_write = core::cmp::min(native, buf_len);
        if to_write == 0 {
            return 0;
        }

        let buf_va = buf_vaddr.raw();
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = buf_va & !(PAGE_SIZE as u64 - 1);
        let page_off = (buf_va - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            let src =
                core::slice::from_raw_parts(&stats as *const ProcessStats as *const u8, to_write);
            page[page_off..page_off + to_write].copy_from_slice(src);
        }
        to_write as isize
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
            cwd: None,
            stdout_redirect: None,
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
        caller_tid: u32,
    ) {
        use orbit_abi::denial::{DenialEvent, DenialSink};
        use orbit_abi::perms::CreateProcessV2Args;
        use orbit_core::roles::{check_transition, deny_reason_code, derive_child_perms};

        // Macro to bail with an errno: resume the parker on the on-
        // thread completion path and return. Replaces the previous
        // `handle.signal(...)` pattern now that we use the no-Arc
        // resume helper.
        macro_rules! bail {
            ($errno:expr) => {{
                self.publish_pending_for_tid(caller_tid, &[Errno::new($errno).to_ret()]);
                return;
            }};
        }

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
                    bail!(EFAULT);
                }
            };
        let args: CreateProcessV2Args = unsafe {
            let mut w = user_page::UserPageWindow::map(args_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            let p = page.as_ptr().add(args_page_off) as *const CreateProcessV2Args;
            core::ptr::read_unaligned(p)
        };

        // ELF range validation only applies to bytes mode — path mode
        // gets its bytes from k_io reading off disk, so elf_vaddr /
        // elf_len are ignored there. Bytes mode is also LOADER-only;
        // both checks happen below at the spawn-source branch after
        // the role-transition gate has fired.

        // Reject unknown stdout_capture values up front. `_pad2` is
        // also expected zero so a future expansion of the slot doesn't
        // silently accept legacy callers' uninitialized stack bytes.
        if args.stdout_capture > 1 || args._pad2 != 0 {
            bail!(EINVAL);
        }

        // Snapshot the parent's permissions for the gate, plus the
        // credential triplet + login_name + groups for the inherit
        // path. Cloning `groups`/`login_name` upfront releases the
        // borrow on `processes` before we touch `denial_ring` (which
        // also lives on `self`).
        let (
            parent_perms,
            parent_uid,
            parent_euid,
            parent_suid,
            parent_gid,
            parent_egid,
            parent_sgid,
            parent_login,
            parent_groups,
        ) = match self.processes.get(&parent_pid) {
            Some(p) => (
                p.permissions,
                p.uid,
                p.euid,
                p.suid,
                p.gid,
                p.egid,
                p.sgid,
                p.login_name.clone(),
                p.groups.clone(),
            ),
            None => {
                error!("create_process_v2: parent pid={parent_pid} vanished");
                bail!(ESRCH);
            }
        };

        // Identity-stamping gate. Sentinel `-1` on either uid/gid
        // means "inherit"; any non-inherit value (uid/gid, login
        // override, or groups override) requires the parent to be
        // running with `role::LOADER`. Other roles get -EPERM here
        // before any further work. The check is deliberately strict
        // (LOADER-only, not "any role with full caps") because uid is
        // identity, not authorization — we don't want a BOOTSTRAP-
        // shaped rescue process to be able to forge identity on a
        // child it spawns.
        let stamps_identity = args.setuid_uid != orbit_abi::perms::CreateProcessV2Args::INHERIT_ID
            || args.setuid_gid != orbit_abi::perms::CreateProcessV2Args::INHERIT_ID
            || args.setlogin_vaddr != 0
            || args.groups_vaddr != 0;
        if stamps_identity && parent_perms.role != orbit_abi::perms::role::LOADER {
            error!(
                "create_process_v2: identity stamping requires LOADER role (parent role={})",
                parent_perms.role
            );
            bail!(EPERM);
        }

        // Validate the uid/gid sentinels: either INHERIT_ID (-1) or a
        // non-negative value that fits in u32. Anything else is
        // EINVAL — catches accidental sign-extension bugs in callers.
        let setuid_override: Option<u32> = match args.setuid_uid {
            orbit_abi::perms::CreateProcessV2Args::INHERIT_ID => None,
            v if (0..=u32::MAX as i64).contains(&v) => Some(v as u32),
            _ => bail!(EINVAL),
        };
        let setgid_override: Option<u32> = match args.setuid_gid {
            orbit_abi::perms::CreateProcessV2Args::INHERIT_ID => None,
            v if (0..=u32::MAX as i64).contains(&v) => Some(v as u32),
            _ => bail!(EINVAL),
        };

        // Optional login-name override. MAXLOGNAME = 32 matches
        // OpenBSD's `_POSIX_LOGIN_NAME_MAX`. Validation mirrors the
        // cwd path: bound-check, single-page, copy, UTF-8 check. The
        // copy reuses `copy_user_path`, which insists on a
        // `MAX_FS_PATH_LEN`-sized scratch — we cap the meaningful
        // payload at MAX_LOGIN_NAME but use the larger fixed buffer.
        const MAX_LOGIN_NAME: usize = 32;
        let mut login_buf = [0u8; MAX_FS_PATH_LEN];
        let login_override: Option<&str> = if args.setlogin_vaddr != 0 && args.setlogin_len != 0 {
            if args.setlogin_len > MAX_LOGIN_NAME {
                bail!(orbit_abi::errno::ENAMETOOLONG);
            }
            if !user_range_ok(args.setlogin_vaddr as u64, args.setlogin_len as u64) {
                bail!(EFAULT);
            }
            let s = match self.copy_user_path(
                root_pa,
                args.setlogin_vaddr as u64,
                args.setlogin_len,
                &mut login_buf,
            ) {
                Ok(s) => s,
                Err(e) => {
                    self.publish_pending_for_tid(caller_tid, &[e]);
                    return;
                }
            };
            Some(s)
        }
        else {
            None
        };

        // Optional supplementary-groups override. Single page, packed
        // u32 array. groups_count == 0 means "install an empty list"
        // (distinct from groups_vaddr == 0 which means "inherit").
        let mut groups_override: Option<alloc::vec::Vec<u32>> = None;
        if args.groups_vaddr != 0 {
            if args.groups_count > process::NGROUPS_MAX {
                bail!(EINVAL);
            }
            let bytes = args
                .groups_count
                .checked_mul(core::mem::size_of::<u32>())
                .and_then(|n| u64::try_from(n).ok())
                .ok_or(())
                .map(|n| n)
                .unwrap_or(0);
            if bytes != 0 {
                if !user_range_ok(args.groups_vaddr as u64, bytes) {
                    bail!(EFAULT);
                }
                if (args.groups_vaddr as u64 & (PAGE_SIZE as u64 - 1)) + bytes > PAGE_SIZE as u64 {
                    bail!(EINVAL);
                }
                let page_base = (args.groups_vaddr as u64) & !(PAGE_SIZE as u64 - 1);
                let page_off = (args.groups_vaddr as u64 - page_base) as usize;
                let pa =
                    match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) }
                    {
                        Some(p) => p as u64,
                        None => bail!(EFAULT),
                    };
                let mut buf: alloc::vec::Vec<u32> =
                    alloc::vec::Vec::with_capacity(args.groups_count);
                unsafe {
                    let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                    let page = w.as_mut_slice();
                    let src = &page[page_off..page_off + bytes as usize];
                    for chunk in src.chunks_exact(4) {
                        let mut le = [0u8; 4];
                        le.copy_from_slice(chunk);
                        buf.push(u32::from_le_bytes(le));
                    }
                }
                groups_override = Some(buf);
            }
            else {
                groups_override = Some(alloc::vec::Vec::new());
            }
        }

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
                bail!(EPERM);
            }
        };

        // Spawn-source branch. Path mode (`spawn_path_vaddr != 0`)
        // resolves the inode, runs the X+R access check, allocates
        // the destination blob, and inserts a `SpawnInProgress`
        // entry keyed by `caller_tid`. It then issues the first
        // page's cache read and returns; subsequent pages are
        // driven by `advance_spawn` from each completing
        // `CacheFill` event. The final page completion runs
        // `install_spawn` and signals `handle` with the new pid
        // (or errno).
        //
        // Bytes mode requires LOADER role (no FS file means no X
        // bit to check, so we restrict the delivery surface to the
        // privileged spawner) and runs the install inline below.
        if args.spawn_path_vaddr != 0 {
            use orbit_abi::fs::{ACCESS_R_OK, ACCESS_X_OK, S_IFMT, S_IFREG};

            if args.spawn_path_len == 0 || args.spawn_path_len > MAX_FS_PATH_LEN {
                bail!(orbit_abi::errno::ENAMETOOLONG);
            }
            if !user_range_ok(args.spawn_path_vaddr as u64, args.spawn_path_len as u64) {
                bail!(EFAULT);
            }
            let mut path_buf = [0u8; MAX_FS_PATH_LEN];
            let raw_path = match self.copy_user_path(
                root_pa,
                args.spawn_path_vaddr as u64,
                args.spawn_path_len,
                &mut path_buf,
            ) {
                Ok(s) => s,
                Err(e) => {
                    self.publish_pending_for_tid(caller_tid, &[e]);
                    return;
                }
            };
            let resolved = self.resolve_fs_path(parent_pid, raw_path);
            let path = resolved.as_str();

            // Resolve + validate the spawn target inline. Anything
            // that fails here resumes the parker synchronously and
            // never builds a SpawnInProgress entry.
            let Some(fs) = crate::kernel::fs::mounted()
            else {
                bail!(EIO);
            };
            let inode = match fs.open(path) {
                Ok(i) => i,
                Err(FsErr::NotFound) => bail!(orbit_abi::errno::ENOENT),
                Err(_) => bail!(EIO),
            };
            let stat = match fs.stat(inode) {
                Ok(s) => s,
                Err(_) => bail!(EIO),
            };
            if (stat.st_mode & S_IFMT) != S_IFREG {
                bail!(orbit_abi::errno::ENOEXEC);
            }
            if let Err(e) = orbit_abi::fs::vaccess(
                parent_euid,
                parent_egid,
                &parent_groups,
                &stat,
                ACCESS_X_OK | ACCESS_R_OK,
            ) {
                debug!(
                    "create_process_v2 path: vaccess EACCES path={path} mode={:#o}",
                    stat.st_mode
                );
                self.publish_pending_for_tid(caller_tid, &[e.to_ret()]);
                return;
            }
            let total_size = stat.st_size as u64;
            if total_size == 0 || (total_size as usize) > MAX_ELF_BYTES {
                bail!(EINVAL);
            }

            // Allocate the destination blob, sized exactly. Cache
            // fills will memcpy directly into `blob[page_idx *
            // PAGE..]`, so we extend to full length up front.
            let mut blob = alloc::vec::Vec::<u8>::with_capacity(total_size as usize);
            blob.resize(total_size as usize, 0);
            let total_pages = (total_size + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64;

            let owned_login = login_override.map(alloc::string::String::from);
            let ctx = orbit_core::SpawnContext {
                args,
                parent_pid,
                root_pa,
                child_perms,
                setuid_override,
                setgid_override,
                login_override: owned_login,
                groups_override,
                parent_uid,
                parent_euid,
                parent_suid,
                parent_gid,
                parent_egid,
                parent_sgid,
                parent_login,
                parent_groups,
            };

            let progress = SpawnInProgress {
                ctx,
                inode,
                blob,
                total_size,
                pages_done: 0,
                total_pages,
            };
            self.spawns_in_progress.insert(caller_tid, progress);

            // Kick off the state machine. `issue_next_spawn_page`
            // may resolve synchronously (cache hits) and chain
            // through to `install_spawn` immediately if every page
            // is already cached; otherwise it submits the next
            // missing page's DMA and returns, with the rest driven
            // by `CacheFill` → `advance_spawn`.
            self.issue_next_spawn_page(caller_tid);
            return;
        }

        // Bytes mode — caller asserts they trust these bytes; only
        // LOADER-roled callers are allowed (the X-bit story degrades
        // for paths the kernel never sees, so we restrict the
        // delivery surface to the single privileged spawner).
        if parent_perms.role != orbit_abi::perms::role::LOADER {
            error!(
                "create_process_v2: bytes-mode spawn requires LOADER role (parent role={}); use spawn_path instead",
                parent_perms.role
            );
            bail!(EPERM);
        }
        if args.elf_len == 0 || args.elf_len > MAX_ELF_BYTES {
            bail!(EINVAL);
        }
        if !user_range_ok(args.elf_vaddr as u64, args.elf_len as u64) {
            bail!(EFAULT);
        }

        // Copy the ELF (same loop as run_create_process_req). Bytes-
        // mode path: source bytes from caller's user memory and
        // converge on the canonical `install_spawn` helper below
        // (same one path-mode hits via the SpawnReady bounce).
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
                    bail!(EFAULT);
                }
            };
            unsafe {
                let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                let page = w.as_mut_slice();
                blob.extend_from_slice(&page[page_off..page_off + take]);
            }
            copied += take;
        }

        let result = self.install_spawn(
            &blob,
            &args,
            parent_pid,
            root_pa,
            child_perms,
            setuid_override,
            setgid_override,
            login_override,
            groups_override,
            parent_uid,
            parent_euid,
            parent_suid,
            parent_gid,
            parent_egid,
            parent_sgid,
            parent_login,
            parent_groups,
        );
        self.publish_pending_for_tid(caller_tid, &[result]);
    }

    /// Post-blob spawn-install: validate affinity, resolve cwd/argv/envp
    /// against the parent's user memory, allocate the child Process via
    /// `create_new_process`, install the witness-derived perms via
    /// `install_child`, then walk the child's threads to refresh
    /// per-thread perms + credential snapshots.
    ///
    /// Both spawn front-doors (bytes-mode in the manager and path-mode
    /// via `k_spawn` bouncing back as `PendingWork::SpawnReady`) call
    /// this helper with their resolved blob. The split lets the path
    /// front-door do its FS reads without holding the manager up while
    /// keeping a single canonical install codepath — one place to fix
    /// any spawn-related regression.
    ///
    /// Caller responsibilities:
    /// - `args` validated (struct copy + `stdout_capture <= 1` + `_pad2 == 0`)
    /// - parent credential snapshot supplied (parent's `Process` may
    ///   exit between the snapshot and this call; the snapshot is
    ///   self-contained for the credential-resolve step)
    /// - `child_perms` already derived via the role-transition gate
    /// - identity-stamping gate already enforced (any non-inherit
    ///   `setuid_override` / `setgid_override` / `login_override` /
    ///   `groups_override` was vetted as LOADER-only at the gate)
    ///
    /// Returns the new pid on success or a negative errno on failure.
    /// Caller signals the original CompletionHandle with this value.
    fn install_spawn(
        &mut self,
        blob: &[u8],
        args: &orbit_abi::perms::CreateProcessV2Args,
        parent_pid: u16,
        root_pa: PhysAddr,
        child_perms: orbit_abi::roles::ChildPerms,
        setuid_override: Option<u32>,
        setgid_override: Option<u32>,
        login_override: Option<&str>,
        groups_override: Option<alloc::vec::Vec<u32>>,
        parent_uid: u32,
        parent_euid: u32,
        parent_suid: u32,
        parent_gid: u32,
        parent_egid: u32,
        parent_sgid: u32,
        parent_login: Option<alloc::string::String>,
        parent_groups: alloc::vec::Vec<u32>,
    ) -> isize {
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

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

        // Optional cwd override: `cwd_vaddr == 0 || cwd_len == 0` means
        // "inherit parent's cwd verbatim". Otherwise copy the bytes in,
        // validate UTF-8 + absolute + dir-exists, and pass to
        // create_new_process. Validation here mirrors run_chdir so a
        // child can't be spawned into a cwd a chdir(...) would reject.
        let mut cwd_buf = [0u8; MAX_FS_PATH_LEN];
        let cwd_override: Option<&str> = if args.cwd_vaddr != 0 && args.cwd_len != 0 {
            if args.cwd_len > MAX_FS_PATH_LEN {
                return Errno::new(orbit_abi::errno::ENAMETOOLONG).to_ret();
            }
            if !user_range_ok(args.cwd_vaddr as u64, args.cwd_len as u64) {
                return Errno::new(EFAULT).to_ret();
            }
            let s = match self.copy_user_path(
                root_pa,
                args.cwd_vaddr as u64,
                args.cwd_len,
                &mut cwd_buf,
            ) {
                Ok(s) => s,
                Err(e) => return e,
            };
            if !s.starts_with('/') {
                return Errno::new(EINVAL).to_ret();
            }
            let Some(fs) = crate::kernel::fs::mounted()
            else {
                return Errno::new(EIO).to_ret();
            };
            let inode = match fs.open(s) {
                Ok(i) => i,
                Err(FsErr::NotFound) => {
                    return Errno::new(orbit_abi::errno::ENOENT).to_ret();
                }
                Err(_) => return Errno::new(EIO).to_ret(),
            };
            let st = match fs.stat(inode) {
                Ok(s) => s,
                Err(_) => return Errno::new(EIO).to_ret(),
            };
            if (st.st_mode & orbit_abi::fs::S_IFMT) != orbit_abi::fs::S_IFDIR {
                return Errno::new(ENOTDIR).to_ret();
            }
            Some(s)
        }
        else {
            None
        };

        // Optional argv blob: same shape as the legacy EX path —
        // single-page wire-format blob from `orbit_abi::argv::pack`.
        // `argv_vaddr == 0 || argv_len == 0` means "no argv."
        let argv_bytes: Option<Vec<u8>> = if args.argv_vaddr != 0 && args.argv_len != 0 {
            if args.argv_len > PAGE_SIZE {
                return Errno::new(EINVAL).to_ret();
            }
            if !user_range_ok(args.argv_vaddr as u64, args.argv_len as u64) {
                return Errno::new(EFAULT).to_ret();
            }
            let mut buf = Vec::with_capacity(args.argv_len);
            let mut argv_copied = 0usize;
            while argv_copied < args.argv_len {
                let cursor = args.argv_vaddr + argv_copied;
                let page_base = cursor & !(PAGE_SIZE - 1);
                let page_off = cursor - page_base;
                let take = core::cmp::min(PAGE_SIZE - page_off, args.argv_len - argv_copied);
                let pa = match unsafe {
                    mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base as u64))
                } {
                    Some(p) => p as u64,
                    None => return Errno::new(EFAULT).to_ret(),
                };
                unsafe {
                    let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                    let page = w.as_mut_slice();
                    buf.extend_from_slice(&page[page_off..page_off + take]);
                }
                argv_copied += take;
            }
            Some(buf)
        }
        else {
            None
        };

        // Optional envp blob: always one page, must be page-aligned.
        // `envp_vaddr == 0` means "no envp."
        let envp_bytes: Option<Vec<u8>> = if args.envp_vaddr != 0 {
            if (args.envp_vaddr as u64) & (PAGE_SIZE as u64 - 1) != 0 {
                return Errno::new(EINVAL).to_ret();
            }
            if !user_range_ok(args.envp_vaddr as u64, PAGE_SIZE as u64) {
                return Errno::new(EFAULT).to_ret();
            }
            let pa = match unsafe {
                mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(args.envp_vaddr as u64))
            } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
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

        // `stdout_capture == 1` ⇒ child writes route to the parent's
        // pane. Validated above; only 0/1 reach this point. The
        // parent_pid is unconditional here because the V2 syscall is
        // only callable from a U-mode process (parent_pid != 0).
        let stdout_redirect = if args.stdout_capture == 1 {
            Some(parent_pid)
        }
        else {
            None
        };

        let proc_components = ProcessComponents {
            elf_blob: blob,
            stack_size: UPROC_STACK_DEFAULT,
            allowed_affinity: allowed,
            affinity,
            parent_pid,
            argv_bytes: argv_bytes.as_deref(),
            envp_bytes: envp_bytes.as_deref(),
            perms: None,
            cwd: cwd_override,
            stdout_redirect,
        };

        let child_pid = match self.create_new_process(proc_components) {
            Ok(p) => p,
            Err(()) => {
                error!("create_process_v2: create_new_process failed");
                return Errno::new(ENOEXEC).to_ret();
            }
        };

        // Resolve final credentials. Inherit copies parent's matching
        // slots verbatim (POSIX fork semantic); stamp installs the
        // overridden value on all three slots of the triplet (POSIX
        // fresh-login semantic — what `login(1)` does after auth).
        let (child_uid, child_euid, child_suid) = match setuid_override {
            Some(uid) => (uid, uid, uid),
            None => (parent_uid, parent_euid, parent_suid),
        };
        let (child_gid, child_egid, child_sgid) = match setgid_override {
            Some(gid) => (gid, gid, gid),
            None => (parent_gid, parent_egid, parent_sgid),
        };
        let child_login: Option<alloc::string::String> = match login_override {
            Some(s) => Some(alloc::string::String::from(s)),
            None => parent_login,
        };
        let child_groups: alloc::vec::Vec<u32> = match groups_override {
            Some(g) => g,
            None => parent_groups,
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
            // Same-locked-section credential stamp. Ordering doesn't
            // matter relative to install_child — both are pure field
            // writes — but bundling them under one `get_mut` keeps
            // the borrow life narrow.
            proc.uid = child_uid;
            proc.euid = child_euid;
            proc.suid = child_suid;
            proc.gid = child_gid;
            proc.egid = child_egid;
            proc.sgid = child_sgid;
            proc.login_name = child_login;
            proc.groups = child_groups;
            // Detach: parent skips both exit_waiter signal and
            // dead_children stash on this child's exit. Used by
            // fire-and-forget spawners (orbit-loader) so a long-lived
            // parent doesn't accumulate per-spawn exit-code entries.
            proc.detached = (args.flags & orbit_abi::perms::CreateProcessV2Args::DETACH) != 0;
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
                // Refresh the per-thread credential snapshot so the
                // getuid/geteuid/getgid/getegid fast paths read the
                // freshly stamped values. login_name + groups stay
                // on Process — getlogin/getgroups go through the
                // manager-side lookup, no thread snapshot to refresh.
                t.uid = child_uid;
                t.euid = child_euid;
                t.suid = child_suid;
                t.gid = child_gid;
                t.egid = child_egid;
                t.sgid = child_sgid;
            }
        }

        info!(
            "create_process_v2: spawned pid={child_pid} parent={parent_pid} role={} perms={:#x}/{:#x} uid={}:{} gid={}:{}",
            args.target_role,
            perms_snapshot.perms,
            perms_snapshot.allowed_perms,
            child_uid,
            child_euid,
            child_gid,
            child_egid,
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
    fn run_wait_pid_req(&mut self, req: WaitPidReq, caller_pid: u16, caller_tid: u32) {
        // First check the caller's `dead_children` — covers the race
        // where the target exited before this wait_pid syscall ran.
        // dealloc_process stashed (target_pid → exit_code) on the
        // parent's process struct; drain it here for sync return.
        if let Some(parent) = self.processes.get_mut(&caller_pid)
            && let Some(code) = parent.dead_children.remove(&req.target_pid)
        {
            self.publish_pending_for_tid(caller_tid, &[0, code as isize]);
            return;
        }

        let Some(target) = self.processes.get_mut(&req.target_pid)
        else {
            // Never existed (or exited and the parent's already gone
            // / wasn't tracked) — POSIX surfaces this as ECHILD.
            self.publish_pending_for_tid(
                caller_tid,
                &[Errno::new(orbit_abi::errno::ECHILD).to_ret(), 0],
            );
            return;
        };
        if target.parent_pid != caller_pid {
            self.publish_pending_for_tid(caller_tid, &[Errno::new(EPERM).to_ret(), 0]);
            return;
        }
        if target.exit_waiter.is_some() {
            // Single-waiter v1 — multi-waiter wants a Vec and lands
            // with futex (§13a.5).
            self.publish_pending_for_tid(
                caller_tid,
                &[Errno::new(orbit_abi::errno::EBUSY).to_ret(), 0],
            );
            return;
        }
        // Install the parent's tid on the target. dealloc_process
        // will take + resume via `publish_pending_for_tid` with the
        // child's exit code.
        target.exit_waiter = Some(caller_tid);
        info!(
            "wait_pid: pid={caller_pid} tid={caller_tid} parked on target={} exit",
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
    fn run_futex_wait_req(&mut self, req: FutexWaitReq, pid: u16, root_pa: PhysAddr, tid: u32) {
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(req.uaddr.raw())) } {
                Some(p) => p as u64,
                None => {
                    self.publish_pending_for_tid(tid, &[Errno::new(EFAULT).to_ret()]);
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
            // Diagnostic for the eza-stress heap-corruption hypothesis:
            // log every EAGAIN with full context so we can correlate
            // with user-side mutex state when something goes wrong.
            trace!(
                "futex_wait: tid={tid} pid={pid} pa={pa:#x} expected={} observed={} \
                 action=EAGAIN",
                req.expected, observed,
            );
            self.publish_pending_for_tid(tid, &[Errno::new(EAGAIN).to_ret()]);
            return;
        }
        // Park: install the waiter on the per-PA queue. v1 ignores
        // `timeout_ns` — the field is reserved; the wait blocks
        // until a `futex_wake` (or `dealloc_process`) drains it.
        let waiter = FutexWaiter {
            tid,
            pid,
            deadline_ticks: 0,
        };
        let n_after = {
            let q = self.futex_waiters.entry(pa).or_default();
            q.push(waiter);
            q.len()
        };
        trace!(
            "futex_wait: tid={tid} pid={pid} pa={pa:#x} expected={} observed={} \
             action=install waiters_after={n_after}",
            req.expected, observed,
        );
    }

    /// §13a.5 — futex wake. Drains up to `req.n` waiters from
    /// `futex_waiters[pa]`, resumes each via
    /// `publish_pending_for_tid(w.tid, &[0])`, and returns the count
    /// of waiters woken (or a negative errno on translation failure).
    /// The caller arm in `drain_pending_work` then resumes the
    /// wake-caller's parked thread via the same helper with the count.
    fn run_futex_wake_req(
        &mut self,
        req: FutexWakeReq,
        caller_pid: u16,
        root_pa: PhysAddr,
    ) -> isize {
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(req.uaddr.raw())) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        // Read *uaddr too — same diagnostic shape as futex_wait. If
        // the user's mutex protocol is broken (e.g. unlock writes
        // didn't reach the kernel before the wake), comparing the
        // observed word against what the unlocker thought it wrote
        // would catch it.
        let page_pa = pa & !(PAGE_SIZE as u64 - 1);
        let page_off = (pa - page_pa) as usize;
        let observed = unsafe {
            let mut win = crate::kernel::user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            core::ptr::read_volatile(win.as_mut_ptr().add(page_off) as *const u32)
        };
        let mut drained_tids: alloc::vec::Vec<u32> = alloc::vec::Vec::new();
        let n_woken = match self.futex_waiters.get_mut(&pa) {
            Some(waiters) => {
                let take = core::cmp::min(req.n as usize, waiters.len());
                // Drain from the front so wake order matches park
                // order (FIFO). Since waiters are pushed at the tail
                // in `run_futex_wait_req`, the oldest is at index 0.
                let drained: alloc::vec::Vec<FutexWaiter> = waiters.drain(..take).collect();
                if waiters.is_empty() {
                    self.futex_waiters.remove(&pa);
                }
                for w in drained {
                    drained_tids.push(w.tid);
                    self.publish_pending_for_tid(w.tid, &[0]);
                }
                take as isize
            }
            None => 0,
        };
        trace!(
            "futex_wake: caller_pid={caller_pid} pa={pa:#x} observed={observed} \
             requested={} drained={drained_tids:?} n_woken={n_woken}",
            req.n,
        );
        n_woken
    }

    fn run_create_thread_req(
        &mut self,
        req: CreateThreadReq,
        pid: u16,
        parent_allowed: u64,
    ) -> isize {
        trace!("handling create_thread req: {req:?} pid={pid} parent_allowed={parent_allowed:#x}");

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
                trace!(
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
            cwd: None,
            stdout_redirect: None,
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
                    // IRQ, user ch_yield) the latch has fired, so the
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
                WakeEvent::InputTid(tid) => {
                    self.set_wake_reason_where(process::wake_reason::INPUT_IO, |t| t.tid == tid);
                }
                WakeEvent::Gpu => {
                    // Mirror of the `WakeEvent::Net` branch: target
                    // k_gpu specifically once `setup_virtio_gpu` has
                    // latched its tid; before then (boot window),
                    // fall back to a coarse pid=0 scan. By the time
                    // any producer pushes `WakeEvent::Gpu` for real
                    // (a console_write / pane-cycle / surface-present
                    // path), virtio-gpu init has run and the tid is
                    // latched, so the fallback is just a safety net.
                    match self.gpu_thread_tid {
                        Some(tid) => self
                            .set_wake_reason_where(process::wake_reason::TICKLE, |t| t.tid == tid),
                        None => {
                            self.set_wake_reason_where(process::wake_reason::TICKLE, |t| t.pid == 0)
                        }
                    }
                }
                WakeEvent::Serial => {
                    // Same shape as `WakeEvent::Gpu` — target
                    // k_serial once `setup_serial_kthread` has latched
                    // its tid; coarse pid=0 fallback during the boot
                    // window before the latch.
                    match self.serial_thread_tid {
                        Some(tid) => self
                            .set_wake_reason_where(process::wake_reason::TICKLE, |t| t.tid == tid),
                        None => {
                            self.set_wake_reason_where(process::wake_reason::TICKLE, |t| t.pid == 0)
                        }
                    }
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
            // Eager-promote both Suspended (sleep-heap parkers,
            // e.g. read_key_event) and Blocking (manager-resolved
            // syscall parkers, e.g. migrated mmap). Pre-migration the
            // Blocking branch was unreachable from the wake-event
            // path: blocking syscalls used `CompletionHandle`, which
            // signaled inline via the wake hook, transitioning
            // Blocking → Ready synchronously. Migrated syscalls drop
            // the handle and rely on `WakeEvent::Tid` instead, so
            // this CAS is the canonical wake.
            //
            // `fetch_update` lets us accept either Blocking or
            // Suspended as the prior state without two CAS round-
            // trips. Other states (Ready / Running / Assigned /
            // Exited) bail out — the wake_override OR above already
            // recorded the reason for the next dispatch to consume.
            let promoted = thread
                .state
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                    if s == ThreadState::Blocking as usize || s == ThreadState::Suspended as usize {
                        Some(ThreadState::Ready as usize)
                    }
                    else {
                        None
                    }
                })
                .is_ok();
            if promoted {
                // On-thread completion path: if a manager-resolved
                // syscall published return values via
                // `Thread::publish_results` before pushing this
                // wake event, marshal them into `frame.regs[10..]`
                // and clear the SIGNALED state. We hold MANAGER_LOCK
                // and the prior state was Blocking-or-Suspended
                // (no hart is running it), so writing `frame.regs`
                // here doesn't race a dispatch. No-op for wakes that
                // don't carry rets (coarse pid/net wakes,
                // input-ring retries) — `take_pending_results`
                // returns `None` when `pending_state == NONE`.
                // `take_pending_results` is a CAS-claim: only one of
                // {this drain, the parker's post-publish re-check}
                // wins. If we lose, the parker already marshaled rets
                // and transitioned state itself — leave its work
                // alone. If we win, the rets snapshot is exclusively
                // ours and writing `frame.regs` here is uncontended.
                let mut rets = [0i64; 4];
                if let Some(n) = thread.take_pending_results(&mut rets) {
                    for i in 0..n {
                        thread.frame.regs[10 + i] = rets[i] as usize;
                    }
                }
                let pending = thread.wake_override.swap(0, Ordering::AcqRel);
                thread.last_wake_reason.store(pending, Ordering::Release);
                // Just promoted Blocking/Suspended → Ready; queue it
                // so get_runnable_thread picks it up this same pass.
                // Any sleep-heap entry becomes stale (state mismatch)
                // and is reaped on the next drain_woken.
                self.ready.push(p.0);
            }
            else {
                //trace!(
                //    "[set_wake_reason] thread #{} not in Blocking/Suspended",
                //    thread.tid,
                //);
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
                    tid,
                } => {
                    let result = self.run_mmap_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::NetChannelCreation {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let (r, e) = self.run_nc_create_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[r, e]);
                }
                PendingWork::CloseHandle {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_close_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::CreateProcess {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_create_process_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::CreateThread {
                    req,
                    pid,
                    parent_allowed,
                    tid,
                } => {
                    let result = self.run_create_thread_req(req, pid, parent_allowed);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::FsOpen {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_fs_open_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::FsRead {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    // Cache-driven: either resumes `tid` inline (all
                    // pages were Ready hits, EOF, or a sync error) or
                    // registers waiters that the eventual CacheFill
                    // arm resumes after the per-page DMAs land. No
                    // CompletionHandle in the loop.
                    self.run_fs_read_req(req, pid, root_pa, tid);
                }
                PendingWork::FsStat {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_fs_stat_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::FsReaddir {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_fs_readdir_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::WaitPid { req, pid, tid } => {
                    // run_wait_pid_req owns the resume — sync errors
                    // publish (errno, 0); the async success path
                    // installs the tid on the target's exit_waiter
                    // slot and dealloc_process publishes (0, exit_code)
                    // when the child exits.
                    self.run_wait_pid_req(req, pid, tid);
                }
                PendingWork::CreateProcessEx {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_create_process_ex_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::FutexWait {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    // run_futex_wait_req owns the resume — sync
                    // EAGAIN/EFAULT publish here; the async park
                    // installs the tid on `futex_waiters[pa]` and a
                    // later `futex_wake` resumes via
                    // `publish_pending_for_tid(tid, &[0])`.
                    self.run_futex_wait_req(req, pid, root_pa, tid);
                }
                PendingWork::FutexWake {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_futex_wake_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::Pledge {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_pledge_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::CreateProcessV2 {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    // Handler owns the resume — bytes mode publishes
                    // inline with the install result; path mode
                    // initiates a `SpawnInProgress` state machine
                    // and publishes later via `advance_spawn` /
                    // `issue_next_spawn_page` when the install
                    // completes. Either way, no manager-side
                    // publish here.
                    self.run_create_process_v2_req(req, pid, root_pa, tid);
                }
                PendingWork::FbSurfaceCreate {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let (r, e) = self.run_fb_surface_create_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[r, e]);
                }
                PendingWork::FbSurfaceDestroy {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_fb_surface_destroy_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::EventFdCreate {
                    req,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let (vaddr, fd) = self.run_eventfd_create_req(req, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[vaddr, fd]);
                }
                PendingWork::WakeTid { req, pid, tid } => {
                    let result = self.run_wake_tid_req(req, pid);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::CacheFill { packed_key, status } => {
                    self.run_cache_fill(packed_key, status);
                }
                PendingWork::ExitGroup {
                    pid,
                    leader_tid,
                    exit_code,
                } => {
                    self.request_exit_group(pid, leader_tid, exit_code);
                }
                PendingWork::QueryDenials {
                    buf_vaddr,
                    buf_len,
                    pid,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_query_denials(buf_vaddr, buf_len, pid, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
                PendingWork::QueryStats {
                    target_pid,
                    buf_vaddr,
                    buf_len,
                    root_pa,
                    tid,
                } => {
                    let result = self.run_query_stats(target_pid, buf_vaddr, buf_len, root_pa);
                    self.publish_pending_for_tid(tid, &[result]);
                }
            }
        }
    }

    /// Stub manager arm for [`PendingWork::CacheFill`]. The IRQ
    /// handler posts these whenever a chain submitted via the
    /// cached path completes. Once the cache is wired into
    /// `run_fs_read_req`, this method will drain the slot's waiter
    /// list, copy bytes per waiter, and resume the waiting tids.
    /// IRQ → manager handler for a completed cache fill. Drains
    /// the slot's waiter list (transitioning the slot to Ready on
    /// success or removing it on failure), iterates each waiter
    /// performing the appropriate copy (UserPageWindow for User
    /// waiters, direct memcpy for Kernel waiters), and decrements
    /// per-tid `FsReadInProgress` accounting; the last decrement
    /// per tid resumes the parked thread with the byte count or
    /// `-EIO`.
    fn run_cache_fill(&mut self, packed_key: u64, status: u8) {
        use crate::kernel::page_cache::Waiter;

        let key = match crate::kernel::page_cache::unpack(packed_key) {
            Some(k) => k,
            None => {
                error!("run_cache_fill: empty packed_key {packed_key:#x}");
                return;
            }
        };

        // Snapshot the slot frame's KVA before draining (we need
        // it for source bytes during the per-waiter copy loop).
        // complete_slot consumes the frame on success-publish or
        // recycles it on failure; the SharedFrame's KVA stays
        // valid for the entire lifetime of either branch since
        // we're under MANAGER_LOCK and no eviction can race us.
        let (waiters, src_kva) = {
            let cache = match self.page_cache.as_mut() {
                Some(c) => c,
                None => {
                    error!("run_cache_fill: cache not initialized");
                    return;
                }
            };
            // Snapshot the source KVA before complete_slot —
            // success keeps the frame in the slot (still
            // accessible), failure recycles it (we still hold a
            // pointer through the post-borrow snapshot below; the
            // pool keeps it alive as long as we're under the lock).
            let src_kva = match cache.lookup(key) {
                Some(s) => s.frame().kva().raw(),
                None => {
                    warn!("run_cache_fill: completion for absent key {key:?}");
                    return;
                }
            };
            let waiters = cache.complete_slot(key, status);
            (waiters, src_kva)
        };

        // Per-waiter dispatch. Skip-if-process-dead for User
        // waiters; tid resume happens via run_fs_read_complete_one.
        for waiter in waiters {
            match waiter {
                Waiter::User {
                    tid,
                    pid,
                    intra,
                    user_page_pa,
                    user_page_off,
                    len,
                } => {
                    let process_alive = self.processes.contains_key(&pid);
                    if status == 0 && process_alive {
                        unsafe {
                            let mut w =
                                user_page::UserPageWindow::map(user_page_pa.get_raw(), PAGE_SIZE);
                            let dst = w.as_mut_slice();
                            let src = core::slice::from_raw_parts(
                                (src_kva + intra as u64) as *const u8,
                                len as usize,
                            );
                            dst[user_page_off as usize..user_page_off as usize + len as usize]
                                .copy_from_slice(src);
                        }
                    }
                    self.complete_fs_read_waiter(tid, status, len);
                }
                Waiter::Kernel {
                    tid,
                    intra,
                    dst_kva,
                    len,
                } => {
                    if status == 0 {
                        unsafe {
                            let src = core::slice::from_raw_parts(
                                (src_kva + intra as u64) as *const u8,
                                len as usize,
                            );
                            let dst =
                                core::slice::from_raw_parts_mut(dst_kva as *mut u8, len as usize);
                            dst.copy_from_slice(src);
                        }
                    }
                    // Dispatch: spawn waiters drive the
                    // per-page state machine; standalone
                    // kernel-buffer reads (none today; future
                    // demand-paged exec, etc.) just resume the
                    // tid with `len` or `-EIO`.
                    if self.spawns_in_progress.contains_key(&tid) {
                        self.advance_spawn(tid, status);
                    }
                    else {
                        let val = if status == 0 {
                            len as isize
                        }
                        else {
                            Errno::new(EIO).to_ret()
                        };
                        self.resume_thread_with_value(tid, val);
                    }
                }
            }
        }
    }

    /// Per-User-waiter completion arm. Decrements the tid's
    /// `FsReadInProgress` counters and resumes the parked thread
    /// when the last waiter for this tid lands.
    ///
    /// `len` is the count of bytes the manager attempted to copy
    /// (regardless of success — `bytes_pending` tracks issued
    /// work, `bytes_done` tracks successful work). On any per-page
    /// failure the entire read resolves to `-EIO` (sticky `failed`
    /// flag); strict POSIX-prefix semantics is a v2 improvement.
    fn complete_fs_read_waiter(&mut self, tid: u32, status: u8, len: u32) {
        let Some(prog) = self.fs_reads_in_progress.get_mut(&tid)
        else {
            // No in-progress entry — fs_read_req resolved
            // synchronously, or this tid never owned an entry.
            // Bug if it fires; log and bail.
            warn!("complete_fs_read_waiter: no FsReadInProgress for tid={tid}");
            return;
        };
        prog.bytes_pending = prog.bytes_pending.saturating_sub(len);
        if status == 0 {
            if !prog.failed {
                prog.bytes_done = prog.bytes_done.saturating_add(len);
            }
        }
        else {
            prog.failed = true;
        }
        if prog.bytes_pending == 0 {
            let bytes_done = prog.bytes_done;
            let failed = prog.failed;
            let _ = self.fs_reads_in_progress.remove(&tid);
            let result = if failed && bytes_done == 0 {
                Errno::new(EIO).to_ret()
            }
            else if failed {
                // Sticky-failure with some successful bytes:
                // return `-EIO` per the v1 "any failure → EIO"
                // policy. POSIX-prefix semantics later.
                Errno::new(EIO).to_ret()
            }
            else {
                bytes_done as isize
            };
            self.resume_thread_with_value(tid, result);
        }
    }

    /// Manager-side analog of [`wake_blocked_inline`] for the
    /// no-CompletionHandle path: write `value` directly into the
    /// parked thread's `regs[10]`, mark it Ready, and queue it onto
    /// the calling hart's READY_INBOX so the next scheduler pass
    /// dispatches it.
    ///
    /// Used by the page-cache completion machinery
    /// ([`run_cache_fill`] and forthcoming `advance_spawn` /
    /// `complete_fs_read_waiter`): no `CompletionHandle` exists for
    /// these waits, so the manager owns both "what value to return"
    /// and "where to wake."
    ///
    /// Tid lookups against `self.threads` are O(log N) BTree —
    /// fine; manager runs hold MANAGER_LOCK so the registry is
    /// stable. Returns silently if the thread has already exited
    /// (its in-progress entry was the only thing keeping the wait
    /// alive; nothing to wake).
    pub fn resume_thread_with_value(&mut self, tid: u32, value: isize) {
        self.resume_thread_with_values(tid, &[value]);
    }

    /// Publish completion results for a thread parked on a manager-
    /// resolved blocking syscall, then push `WakeEvent::Tid` so any
    /// hart's `drain_wakes` (typically the same pass as ours) covers
    /// the Suspended-parker case.
    ///
    /// Why this and not [`Self::resume_thread_with_values`]: the
    /// blocking-syscall parker runs `exit_thread_with_state(Blocking)`
    /// which Release-stores `state = Blocking` *after* the manager has
    /// (potentially) already drained the work item and processed the
    /// resume. If we wrote `state = Ready` directly, that
    /// unconditional `store(Blocking)` from the parker would clobber
    /// us and the wake would vanish. Publishing into the on-thread
    /// completion slot side-steps the race: the parker's post-publish
    /// re-check in `apply_syscall_outcome` checks
    /// `take_pending_results` *after* its own `state.store(Blocking)`
    /// and self-promotes to Ready if a signal landed. Mirrors the
    /// `CompletionHandle::set_waiter` / `is_signaled` shape, just
    /// with a per-thread atom instead of an Arc.
    ///
    /// `WakeEvent::Tid` push covers the Suspended branch: future
    /// migrations (read_stdin, read_key_event in Phase 6) park in
    /// `Suspended` rather than `Blocking`, and the eager
    /// `Suspended → Ready` CAS in `set_wake_reason_where` handles
    /// those — already extended in Phase 1.5 to marshal
    /// `pending_rets` before promoting.
    pub fn publish_pending_for_tid(&self, tid: u32, vals: &[isize]) {
        let Some(pt) = self.threads.get(&tid)
        else {
            // Thread exited mid-flight. No-op.
            return;
        };
        // SAFETY: registry-owned raw ptr; thread is alive while the
        // entry is in `self.threads` (we hold MANAGER_LOCK on the
        // calling path).
        let thread = unsafe { (pt.0 as *const Thread).as_ref_unchecked() };
        thread.publish_results(vals);
        let _ = wake_queue_push(WakeEvent::Tid(tid));
    }

    /// Multi-register variant of [`Self::resume_thread_with_value`]:
    /// write up to 4 return values into `frame.regs[10..10+vals.len()]`,
    /// mark the parked thread Ready, and queue the wake. `vals` is
    /// clamped to 4; excess slots are silently dropped (mirrors
    /// [`process::CompletionHandle::signal_n`]). An empty `vals` writes
    /// no rets — the thread resumes with its trap-entry register
    /// snapshot intact, the same pattern `read_stdin` uses to mean
    /// "wake and retry."
    ///
    /// Same call-site contract as the single-arg version: hold
    /// MANAGER_LOCK, target should be Blocking or Suspended (warn but
    /// proceed otherwise — turns ABA-induced misuse into a loud noop).
    pub fn resume_thread_with_values(&mut self, tid: u32, vals: &[isize]) {
        let Some(pt) = self.threads.get(&tid)
        else {
            // Thread exited mid-flight. No-op.
            return;
        };
        // SAFETY: PThread.0 is the registry-owned raw ptr; the
        // thread is alive as long as `self.threads` holds the entry.
        // We hold MANAGER_LOCK so no concurrent mutator races us on
        // `frame`/`state`.
        let t = unsafe { (pt.0 as *mut Thread).as_mut_unchecked() };
        let prior = t.state.load(Ordering::Acquire);
        if prior != ThreadState::Blocking as usize && prior != ThreadState::Suspended as usize {
            warn!(
                "resume_thread_with_values: tid={tid} not Blocking/Suspended (state={prior}); writing regs anyway"
            );
        }
        let n = vals.len().min(4);
        for (i, &v) in vals.iter().enumerate().take(n) {
            t.frame.regs[10 + i] = v as usize;
        }
        // Reset the on-thread completion slot so a subsequent syscall
        // on this thread doesn't observe stale SIGNALED state. The
        // canonical writer for `pending_state` is `publish_results`;
        // resuming via the manager-direct path bypasses that, so we
        // clear here to keep the invariant ("SIGNALED ⇒ rets are
        // valid and unread") intact.
        t.reset_pending();
        t.state
            .store(ThreadState::Ready as usize, Ordering::Release);
        if push_ready_notice(pt.0).is_err() {
            error!(
                "resume_thread_with_values: READY_INBOX full for tid={tid} — \
                 marked Ready but not queued",
            );
        }
    }

    /// Drive the path-mode spawn state machine by one step:
    /// dispatch successive page reads against the page cache,
    /// taking whatever path each page needs (Ready → memcpy
    /// inline + advance; Loading → register a kernel waiter and
    /// return; Absent → allocate slot + submit DMA + return).
    /// When `pages_done == total_pages` the blob is fully
    /// populated; this helper takes ownership, runs
    /// [`install_spawn`], and signals the spawn handle with the
    /// new pid (or errno on install failure).
    ///
    /// Called twice in the spawn lifecycle:
    /// 1. From `run_create_process_v2_req` immediately after
    ///    inserting `SpawnInProgress`, to fire the first read.
    /// 2. From `advance_spawn` after each `CacheFill` (status==OK)
    ///    completes a kernel waiter for this tid.
    ///
    /// Cache hits chain inline — a fully-cached spawn target
    /// completes synchronously without ever yielding to the
    /// scheduler.
    fn issue_next_spawn_page(&mut self, tid: u32) {
        use crate::kernel::page_cache::{CacheKey, SlotState, Waiter};
        const PAGE: u64 = PAGE_SIZE as u64;

        loop {
            let Some(spawn) = self.spawns_in_progress.get_mut(&tid)
            else {
                error!("issue_next_spawn_page: no SpawnInProgress for tid={tid}");
                return;
            };

            if spawn.pages_done >= spawn.total_pages {
                // All pages landed — finalize. Pull `spawn` out so
                // we can run `install_spawn` without holding a
                // borrow on `self.spawns_in_progress`.
                let spawn = self
                    .spawns_in_progress
                    .remove(&tid)
                    .expect("just confirmed present");
                let SpawnInProgress { ctx, blob, .. } = spawn;
                let result = self.run_spawn_ready(ctx, &blob);
                self.publish_pending_for_tid(tid, &[result]);
                return;
            }

            // Compute the next page's metadata.
            let page_idx = spawn.pages_done;
            let page_off = page_idx * PAGE;
            let take = core::cmp::min(PAGE, spawn.total_size - page_off) as u32;
            let dst_kva = unsafe { spawn.blob.as_mut_ptr().add(page_off as usize) as usize };
            let inode = spawn.inode;

            let Some(fs) = crate::kernel::fs::mounted()
            else {
                self.spawns_in_progress.remove(&tid);
                self.publish_pending_for_tid(tid, &[Errno::new(EIO).to_ret()]);
                return;
            };
            let lba = match fs.lba_for_page(inode, page_idx) {
                Ok(l) => l,
                Err(_) => {
                    self.spawns_in_progress.remove(&tid);
                    self.publish_pending_for_tid(tid, &[Errno::new(EIO).to_ret()]);
                    return;
                }
            };
            let key = CacheKey {
                dev: fs.dev_id(),
                lba,
            };
            // Cache valid_bytes is the file-valid count for this
            // page — same shape as the fs_read path.
            let valid_bytes = take;

            // Look up the cache. Snapshot src_kva on Hit so the
            // immutable borrow drops before we mutate the cache.
            #[derive(Copy, Clone)]
            enum Action {
                Hit { src_kva: u64 },
                Loading,
                Absent,
            }
            let action = {
                let cache = self.page_cache.as_ref().unwrap();
                match cache.lookup(key) {
                    Some(SlotState::Ready { frame, .. }) => Action::Hit {
                        src_kva: frame.kva().raw(),
                    },
                    Some(SlotState::Loading { .. }) => Action::Loading,
                    None => Action::Absent,
                }
            };

            match action {
                Action::Hit { src_kva } => {
                    // Synchronous copy directly into the blob.
                    unsafe {
                        let src = core::slice::from_raw_parts(src_kva as *const u8, take as usize);
                        let dst =
                            core::slice::from_raw_parts_mut(dst_kva as *mut u8, take as usize);
                        dst.copy_from_slice(src);
                    }
                    let cache = self.page_cache.as_mut().unwrap();
                    cache.record_hit();
                    cache.touch_lru(key);
                    self.spawns_in_progress.get_mut(&tid).unwrap().pages_done += 1;
                    // Continue loop — next iter handles next page
                    // or finalizes.
                }
                Action::Loading => {
                    let waiter = Waiter::Kernel {
                        tid,
                        intra: 0,
                        dst_kva,
                        len: take,
                    };
                    let cache = self.page_cache.as_mut().unwrap();
                    if cache.register_waiter(key, waiter).is_err() {
                        self.spawns_in_progress.remove(&tid);
                        self.publish_pending_for_tid(tid, &[Errno::new(EIO).to_ret()]);
                        return;
                    }
                    // Parked; advance_spawn will resume on CacheFill.
                    return;
                }
                Action::Absent => {
                    let waiter = Waiter::Kernel {
                        tid,
                        intra: 0,
                        dst_kva,
                        len: take,
                    };
                    let begin =
                        self.page_cache
                            .as_mut()
                            .unwrap()
                            .begin_load(key, valid_bytes, waiter);
                    let dma_pa = match begin {
                        Ok(pa) => pa,
                        Err(_) => {
                            self.spawns_in_progress.remove(&tid);
                            self.publish_pending_for_tid(tid, &[Errno::new(EAGAIN).to_ret()]);
                            return;
                        }
                    };
                    let packed = crate::kernel::page_cache::pack(key);
                    match unsafe {
                        crate::drivers::virtio_blk_dev::submit_blk_read_cached(
                            lba,
                            dma_pa.get_raw(),
                            PAGE as u32,
                            packed,
                        )
                    } {
                        Ok(_head) => {
                            // DMA submitted; CacheFill →
                            // advance_spawn drives the rest.
                            return;
                        }
                        Err(_) => {
                            let _ = self.page_cache.as_mut().unwrap().complete_slot(key, 1);
                            self.spawns_in_progress.remove(&tid);
                            self.publish_pending_for_tid(tid, &[Errno::new(EIO).to_ret()]);
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Per-page completion arm for path-mode spawn. Called from
    /// [`run_cache_fill`] when a `Waiter::Kernel` for `tid`
    /// belongs to a `SpawnInProgress` entry. The actual byte copy
    /// already happened in `run_cache_fill`; we just bump
    /// `pages_done` and chain to `issue_next_spawn_page` (which
    /// either issues the next read or finalizes).
    ///
    /// On per-page IO failure, drops the in-progress entry and
    /// signals the carried handle with `-EIO`.
    fn advance_spawn(&mut self, tid: u32, status: u8) {
        if status != 0 {
            if self.spawns_in_progress.remove(&tid).is_none() {
                error!("advance_spawn: no SpawnInProgress for tid={tid}");
                return;
            }
            self.publish_pending_for_tid(tid, &[Errno::new(EIO).to_ret()]);
            return;
        }
        let Some(spawn) = self.spawns_in_progress.get_mut(&tid)
        else {
            error!("advance_spawn: no SpawnInProgress for tid={tid}");
            return;
        };
        spawn.pages_done += 1;
        self.issue_next_spawn_page(tid);
    }

    /// Manager arm for [`PendingWork::SpawnReady`]. k_spawn has already
    /// loaded the bytes; the manager now runs the canonical install
    /// pipeline (same one bytes-mode calls inline).
    fn run_spawn_ready(&mut self, ctx: orbit_core::SpawnContext, blob: &[u8]) -> isize {
        let orbit_core::SpawnContext {
            args,
            parent_pid,
            root_pa,
            child_perms,
            setuid_override,
            setgid_override,
            login_override,
            groups_override,
            parent_uid,
            parent_euid,
            parent_suid,
            parent_gid,
            parent_egid,
            parent_sgid,
            parent_login,
            parent_groups,
        } = ctx;
        self.install_spawn(
            blob,
            &args,
            parent_pid,
            root_pa,
            child_perms,
            setuid_override,
            setgid_override,
            login_override.as_deref(),
            groups_override,
            parent_uid,
            parent_euid,
            parent_suid,
            parent_gid,
            parent_egid,
            parent_sgid,
            parent_login,
            parent_groups,
        )
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

    /// POSIX `_exit(2)` / `exit_group(2)` semantics for sysno 0:
    /// terminate every thread of `pid`, finalize the exit code, and
    /// IPI any hart still running a doomed sibling so it traps and
    /// drops the thread on the next `check_context_and_switch` pass.
    ///
    /// The calling thread is the one whose `tid == leader_tid`; this
    /// method does **not** touch the caller's state (the dispatch
    /// site falls through to `exit_thread_with_state(Exited)` after
    /// this returns). Siblings get state Released to `Exited` here so
    /// the manager's next `cleanup_threads_and_processes` reaps them.
    ///
    /// Caller must hold `MANAGER_LOCK` — we mutate `proc.exit_code`,
    /// `proc.exit_finalized`, and walk `self.threads` directly.
    pub fn request_exit_group(&mut self, pid: u16, leader_tid: u32, exit_code: i32) {
        let proc = match self.processes.get_mut(&pid) {
            Some(p) => p,
            None => return,
        };
        proc.exit_code = exit_code;
        proc.exit_finalized = true;

        // Collect sibling tids; can't borrow `self.threads` mutably
        // while iterating `proc.threads`. The set is small (≤ NGROUPS-
        // ish in practice; rayon's pool is `cpu_count`).
        let siblings: alloc::vec::Vec<u32> = proc
            .threads
            .iter()
            .copied()
            .filter(|tid| *tid != leader_tid)
            .collect();

        for tid in &siblings {
            if let Some(pt) = self.threads.get(tid) {
                let t = unsafe { pt.0.as_ref_unchecked() };
                t.state
                    .store(ThreadState::Exited as usize, Ordering::Release);
            }
        }

        // Forward-progress IPI: any hart currently running a doomed
        // sibling needs to trap so its `check_context_and_switch` arm
        // notices `state == Exited` and bails to k_idle. Without this
        // we'd wait for the next timer tick on that hart — bounded
        // (sub-ms with Sstc) but not immediate.
        crate::kernel::accounting::for_each_hart_context(|h| {
            let cur = h.current.load(Ordering::Acquire);
            if cur.is_null() {
                return;
            }
            let cur_t = unsafe { (cur as *const Thread).as_ref_unchecked() };
            if cur_t.pid == pid && cur_t.tid != leader_tid {
                crate::supervisor_wake_hart(h.hart_id as usize);
            }
        });
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

        // Drop every `futex_waiters` entry whose owner is this dying
        // process. Their `CompletionHandle` clones drop with the entry;
        // the matching `Thread.handle` drops in `dealloc_thread` after
        // we return. Without this sweep a later `futex_wake` on the
        // same PA (e.g. a child process re-using the same physical
        // page in its private heap) would walk the stale entry and
        // dereference the dead Thread via the wake hook —
        // `wake_blocked_inline` reads `t.handle.take().ret_count()` on
        // freed memory and faults inside `CompletionHandle::ret_count`.
        // FutexWaiter.pid was reserved for exactly this sweep.
        let dead_pid = process.pid;
        self.futex_waiters.retain(|_pa, waiters| {
            waiters.retain(|w| w.pid != dead_pid);
            !waiters.is_empty()
        });

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
        //  4. Detached child (`Process.detached`) → behave like (3)
        //     unconditionally. The parent opted out of observing this
        //     child's exit at spawn time via `CREATE_PROCESS_V2`'s
        //     `DETACH` flag, so we skip both the exit_waiter notify
        //     and the dead_children insert. Without this opt-out, a
        //     long-lived parent like orbit-loader accumulates one
        //     dead_children entry per network payload spawn — under
        //     stress (eza_stress.py) the BTreeMap insert path was the
        //     trigger for the jump-through-bad-fnptr fault we chased.
        if process.detached {
            // Belt-and-suspenders: clear exit_waiter so a stale value
            // doesn't get signaled later by some other path.
            process.exit_waiter = None;
        }
        else if let Some(waiter_tid) = process.exit_waiter.take() {
            self.publish_pending_for_tid(waiter_tid, &[0, process.exit_code as isize]);
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

        // Same teardown shape for the structured-event ring; signals
        // any parked `read_key_event` reader so the blocked thread
        // unblocks and falls through to ENOENT on retry.
        crate::kernel::key_events::unregister(process.pid);

        // Tear down the per-process surface table. Each remaining
        // entry's backing frame goes back to `kernel_pages`. The user
        // PT is freed wholesale by the `unmap` call later in this
        // function, so per-entry `unmap_range` is unnecessary; we
        // only need to recover the physical pages.
        //
        // Race with k_gpu: the `push_remove_source` above will be
        // drained by k_gpu and clear any active SurfaceState for this
        // pid before any further repaint runs. Between the push and
        // the drain, k_gpu may briefly hold a kdmap_kva that we're
        // about to free below. v1 accepts this — the worst case is
        // one frame of stale pixels before the source is removed.
        if let Some(surfaces) = crate::kernel::surface::unregister(process.pid) {
            for (_id, entry) in surfaces.drain_all() {
                self.free_backing(entry.backing);
            }
        }

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
                    Handle::Stdin | Handle::Stdout | Handle::Stderr => {
                        // No backing to release — sinks are global.
                    }
                    Handle::EventFd(slot) => {
                        if let Err(e) = slot.region.revoke(&root_table) {
                            warn!(
                                "dealloc_process: eventfd revoke failed for pid{}: {e:?}",
                                process.pid,
                            );
                        }
                        drop(slot);
                    }
                }
            }
        }

        while let Some(b) = process.heap_pages.pop() {
            debug!(
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
                            // Diagnostic context for the eza-lahR
                            // cross-thread-stale-pointer hypothesis:
                            // log own slot, sp + ra at fault, the slot
                            // index that *would* contain stval if it
                            // points into the per-thread stack region,
                            // and the live sibling tids/slots in the
                            // same process at the moment of fault.
                            let own_slot = t.slot;
                            let sp_at_fault = t.frame.regs[2];
                            let ra_at_fault = t.frame.regs[1];
                            let stval_slot = if (f.stval as u64) >= UPROC_STACK_BASE
                                && (f.stval as u64) < UPROC_STACK_BASE + 256 * UPROC_STACK_STRIDE
                            {
                                Some(
                                    ((f.stval as u64 - UPROC_STACK_BASE) / UPROC_STACK_STRIDE)
                                        as u16,
                                )
                            }
                            else {
                                None
                            };
                            let mut siblings: alloc::string::String = alloc::string::String::new();
                            {
                                use core::fmt::Write as _;
                                let _ = write!(&mut siblings, "[");
                                let mut first = true;
                                for &sib_tid in proc.threads.iter() {
                                    if sib_tid == t.tid {
                                        continue;
                                    }
                                    let sib_slot = self.threads.get(&sib_tid).and_then(|p| {
                                        unsafe { (p.0 as *const Thread).as_ref() }
                                            .and_then(|t| t.slot)
                                    });
                                    if !first {
                                        let _ = write!(&mut siblings, ",");
                                    }
                                    first = false;
                                    match sib_slot {
                                        Some(s) => {
                                            let _ =
                                                write!(&mut siblings, "tid{}=slot{}", sib_tid, s);
                                        }
                                        None => {
                                            let _ = write!(&mut siblings, "tid{}=?", sib_tid);
                                        }
                                    }
                                }
                                let _ = write!(&mut siblings, "]");
                            }
                            warn!(
                                "tid{} killed: {} cause={} epc={:#x} stval={:#x} \
                                 own_slot={:?} sp={:#x} ra={:#x} stval_slot={:?} siblings={}",
                                t.tid,
                                label,
                                f.cause,
                                f.epc,
                                f.stval,
                                own_slot,
                                sp_at_fault,
                                ra_at_fault,
                                stval_slot,
                                siblings,
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
                            debug!("tid{} dead, removing status={status}", t.tid);
                            // exit-group: the EXIT-caller already
                            // stamped `exit_code` with the value it
                            // passed; sibling threads reaped after it
                            // (rayon workers parked in futex_wait, etc.)
                            // would otherwise clobber it with whatever
                            // their stale `regs[11]` happens to be.
                            if !proc.exit_finalized {
                                proc.exit_code = status as i32;
                            }
                        }
                    }
                }

                if !proc.threads.is_empty() || t.pid == 0 {
                    continue;
                }
            }

            debug!("pid{} dead, removing", t.pid);

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
                core::cmp::min(1, self.cpu_count - 1),
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

    /// Allocate the page cache's frame pool. Sized at
    /// [`PAGE_CACHE_CAPACITY`] frames (256 KiB at the default 64).
    /// On allocation failure logs and leaves `self.page_cache =
    /// None`; fs_read paths will surface `-EAGAIN` if invoked
    /// before this completes (impossible in practice once the boot
    /// sequence wires this in).
    pub fn setup_page_cache(&mut self) {
        match crate::kernel::page_cache::PageCache::with_capacity(
            &mut self.kernel_pages,
            PAGE_CACHE_CAPACITY,
        ) {
            Some(cache) => {
                info!(
                    "page_cache: allocated {} frames ({} KiB)",
                    cache.capacity(),
                    cache.capacity() * PAGE_SIZE / 1024
                );
                self.page_cache = Some(cache);
            }
            None => {
                error!("page_cache: failed to allocate frame pool — fs_read will EAGAIN");
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
        // Spawn k_serial first so device-bringup tracing routes
        // through the lock-free ring instead of contending on
        // `serial::SERIAL`'s spinlock once multiple harts come up.
        // Producers that fire before this point fall back to the
        // spinlock path via `k_serial::is_ready()`.
        self.setup_serial_kthread();

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
        // Page cache runs after virtio-blk + tarfs are up: it
        // allocates a frame pool from kernel_pages, and the
        // FS-backed lookups need a mounted filesystem to drive
        // the cache keys.
        self.setup_page_cache();
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

    /// Spawn the k_serial UART-drain kthread. Once running, every
    /// `ktrace::emit` call routes through `SERIAL_RING` instead of
    /// taking the UART spinlock, so multi-hart trace bursts no longer
    /// serialize on `serial::SERIAL`'s lock.
    ///
    /// Safe to call once during boot; called from `get_environment_info`
    /// before device init so device-bringup logging benefits from the
    /// queue. Pre-spawn `emit` calls fall back to the spinlock path
    /// via `k_serial::is_ready()` returning `false`.
    pub fn setup_serial_kthread(&mut self) {
        let entrypoint = crate::drivers::k_serial::k_serial as *const () as usize;
        match self.create_kernel_thread(entrypoint, None) {
            Ok(tid) => {
                info!("created k_serial thread tid={tid}");
                self.serial_thread_tid = Some(tid);
            }
            Err(_) => {
                error!("failed to spawn k_serial thread");
            }
        }
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
        match self.create_kernel_thread(entrypoint, None) {
            Ok(tid) => {
                info!("created kgpu thread tid={tid}");
                // Latch the tid so `WakeEvent::Gpu` targets this
                // thread specifically. Without the latch the wake
                // would fan out to every kernel thread (k_net, etc.)
                // — pulls them out of their parks unnecessarily and
                // muddles per-thread wake-reason accounting.
                self.gpu_thread_tid = Some(tid);
            }
            Err(_) => {
                error!("virtio-gpu: failed to spawn k_gpu thread");
            }
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
        debug!("created uthread@{tptr:016X?},pid={pid},tid={tid},table={rpt:016X?}");

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

        debug!(
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
        //
        // Same single-snapshot contract for `stdout_redirect` — read
        // once here, fall back to `None` if the record is gone (no
        // redirect ⇒ writes to own pane, identical to legacy spawns).
        // Snapshot uids/gids alongside permissions so the
        // getuid/getgid fast paths can read thread-local fields
        // without re-acquiring MANAGER_LOCK. groups + login_name stay
        // on Process (variable-size, rarely read).
        let (perms_snapshot, stdout_redirect_snapshot, cred_snapshot) = self
            .processes
            .get(&pid)
            .map(|p| {
                (
                    p.permissions,
                    p.stdout_redirect,
                    (p.uid, p.euid, p.suid, p.gid, p.egid, p.sgid),
                )
            })
            .unwrap_or((
                orbit_abi::perms::Permissions::ZERO,
                None,
                (0, 0, 0, 0, 0, 0),
            ));
        let (cred_uid, cred_euid, cred_suid, cred_gid, cred_egid, cred_sgid) = cred_snapshot;

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
            pending_rets: [
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
            ],
            pending_state: AtomicU8::new(0),
            pending_ret_count: AtomicU8::new(0),
            slot: Some(slot),
            fault_info: None,
            allowed_affinity,
            affinity: AtomicU64::new(affinity),
            cpu_ticks_total: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            syscall_count: AtomicU64::new(0),
            syscall_ticks: AtomicU64::new(0),
            permissions: perms_snapshot,
            uid: cred_uid,
            euid: cred_euid,
            suid: cred_suid,
            gid: cred_gid,
            egid: cred_egid,
            sgid: cred_sgid,
            stdout_redirect: stdout_redirect_snapshot,
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
        // (fail closed). Honor an explicit `proc_components.perms`
        // override (used by the boot path to install LOADER directly
        // on orbit-loader); otherwise fall back to BOOTSTRAP-shaped
        // `Permissions::ALL` so legacy CREATE_PROCESS /
        // CREATE_PROCESS_EX callers keep working without role-aware
        // spawn arguments. CREATE_PROCESS_V2 still calls
        // `install_permissions` (via `install_child`) itself with
        // the witness-derived value, overwriting whichever default
        // landed here.
        proc.install_permissions(
            proc_components
                .perms
                .unwrap_or(orbit_abi::perms::Permissions::ALL),
        );
        // cwd inheritance: explicit override > parent's cwd > "/" default
        // (`Process::new` already seeded cwd = "/" for the boot case where
        // there's no parent). The override is the caller-side
        // `Command::current_dir(...)` shape.
        if let Some(p) = proc_components.cwd {
            proc.cwd.clear();
            proc.cwd.push_str(p);
        }
        else if proc_components.parent_pid != 0 {
            if let Some(parent) = self.processes.get(&proc_components.parent_pid) {
                proc.cwd.clear();
                proc.cwd.push_str(&parent.cwd);
            }
        }
        // Install the stdout redirect target before the first thread
        // is constructed — `create_new_thread` snapshots this onto the
        // thread's lock-free read slot. Setting it later would leave
        // the initial thread with a stale `None`.
        proc.stdout_redirect = proc_components.stdout_redirect;
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

        // Seed slots 0 / 1 / 2 with the Stdin / Stdout / Stderr stdio
        // markers so every new process has the POSIX-shaped fds
        // available before its first `read(0)` / `write(1)` /
        // `write(2)`. Pre-seeding (rather than lazy at first I/O) is
        // what makes the fd numbers stable: `dup(0)` from the entry
        // point now lands at slot 3 regardless of whether anything's
        // touched stdio yet. Inheritance from the parent's stdio
        // configuration lands later via `CreateProcessV2Args.inherit_fds`.
        let ph = self
            .process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new);
        if ph.is_empty() {
            ph.seed_stdio();
        }

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
        //
        // Skip the registration entirely for `stdout_redirect`-ed
        // children: their `console_write` bytes route to the parent's
        // pane, so this pane would only ever be empty. Without this
        // skip every captured spawn would leave a `pid N (exited)`
        // tombstone behind — `display::remove_source` keeps the
        // scrollback around in case the user wants to flip back to
        // the (in our case nonexistent) listing. The matching
        // `push_remove_source` in `dealloc_process` is a no-op when
        // the source was never inserted.
        if proc_components.stdout_redirect.is_none() {
            let _ = crate::drivers::k_gpu::push_insert_source(
                crate::drivers::display::Source::Process(pid),
            );
        }

        // Register a per-process stdin slot so input::dispatch has a
        // place to deliver keystrokes once the process becomes the
        // active source. Removed by `dealloc_process` on teardown.
        crate::kernel::stdin::register(pid);

        // Register the structured key-event ring next to stdin —
        // input::dispatch fans the same key out to both, the byte
        // path stays for shells while ratatui-shaped consumers read
        // KeyEvents via `read_key_event`.
        crate::kernel::key_events::register(pid);

        // Register a per-process surface table so `fb_surface_create`
        // has a slot to insert into without taking the slow path of
        // lazily registering under MANAGER_LOCK. Drained by
        // `dealloc_process`.
        crate::kernel::surface::register(pid);

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
                return Err(());
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

            trace!("loading {segment:08x?}");

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
            trace!(
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

    /// `chdir(path)` body. Synchronous — orbital's tarfs lookups are
    /// in-memory so the manager-round-trip pattern other fs syscalls
    /// use isn't needed here. Validates absolute UTF-8, confirms the
    /// target resolves to an existing directory in the active fs, then
    /// mutates `Process.cwd` in place. Returns 0 or `-errno`.
    pub fn run_chdir(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        path_vaddr: u64,
        path_len: usize,
    ) -> isize {
        if path_len == 0 {
            return Errno::new(EINVAL).to_ret();
        }
        if path_len > MAX_FS_PATH_LEN {
            return Errno::new(orbit_abi::errno::ENAMETOOLONG).to_ret();
        }
        if !orbit_abi::layout::user_range_ok(path_vaddr, path_len as u64) {
            return Errno::new(EFAULT).to_ret();
        }
        let mut path_buf = [0u8; MAX_FS_PATH_LEN];
        let path = match self.copy_user_path(root_pa, path_vaddr, path_len, &mut path_buf) {
            Ok(p) => p,
            Err(e) => return e,
        };
        // v1 only handles absolute targets — `chdir("..")` would need
        // a normalized path-walk that resolves dot/dot-dot against the
        // current cwd, which we don't have yet. Userland that wants
        // relative chdir does the join itself and chdirs absolute.
        if !path.starts_with('/') {
            return Errno::new(EINVAL).to_ret();
        }
        let Some(fs) = crate::kernel::fs::mounted()
        else {
            return Errno::new(EIO).to_ret();
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
        if (stat.st_mode & orbit_abi::fs::S_IFMT) != orbit_abi::fs::S_IFDIR {
            return Errno::new(ENOTDIR).to_ret();
        }
        match self.processes.get_mut(&pid) {
            Some(p) => {
                p.cwd.clear();
                p.cwd.push_str(path);
                info!("chdir: pid={pid} cwd={path}");
                0
            }
            None => Errno::new(ESRCH).to_ret(),
        }
    }

    /// `fs_fstat(fd, &mut Stat)` body. Sync — looks up the calling
    /// process's `OpenFile` for `fd`, runs `Filesystem::stat` on its
    /// inode, copies the result into the user buffer. Single-page
    /// constraint mirrors `fs_stat`. Returns 0 or `-errno`.
    pub fn run_fs_fstat(&mut self, pid: u16, root_pa: PhysAddr, fd: u32, stat_vaddr: u64) -> isize {
        let stat_size = core::mem::size_of::<orbit_abi::fs::Stat>() as u64;
        if !orbit_abi::layout::user_range_ok(stat_vaddr, stat_size) {
            return Errno::new(EFAULT).to_ret();
        }
        if (stat_vaddr & (PAGE_SIZE as u64 - 1)) + stat_size > PAGE_SIZE as u64 {
            return Errno::new(EINVAL).to_ret();
        }
        // Snapshot the inode + fs ref under the handle borrow, then
        // drop the borrow before doing the fs lookup + user-copy.
        let (fs, inode) = {
            let Some(ph) = self.process_handles.get(&pid)
            else {
                return Errno::new(EBADF).to_ret();
            };
            let Some(handle_ref) = ph.get(fd)
            else {
                return Errno::new(EBADF).to_ret();
            };
            let Handle::File(of) = handle_ref
            else {
                return Errno::new(EBADF).to_ret();
            };
            (of.fs, of.inode)
        };
        let stat = match fs.stat(inode) {
            Ok(s) => s,
            Err(_) => return Errno::new(EIO).to_ret(),
        };
        let stat_bytes = unsafe {
            core::slice::from_raw_parts(&stat as *const _ as *const u8, stat_size as usize)
        };
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = stat_vaddr & !(PAGE_SIZE as u64 - 1);
        let page_off = (stat_vaddr - page_base) as usize;
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
        0
    }

    /// `ch_inspect(fd, *mut ChInfo) → 0 | -errno`. Sync. Resolves the
    /// fd in the calling process's handle table, populates a
    /// `ChInfo` based on the slot's variant (NetChannel / EventFd /
    /// File / Stdin / Stdout / Stderr), and copies it into the user
    /// buffer via `UserPageWindow`.
    pub fn run_ch_inspect_req(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        fd: u32,
        info_vaddr: u64,
    ) -> isize {
        use orbit_abi::handle::{ChInfo, HandleKind};

        let info_size = core::mem::size_of::<ChInfo>() as u64;
        if !orbit_abi::layout::user_range_ok(info_vaddr, info_size) {
            return Errno::new(EFAULT).to_ret();
        }
        if (info_vaddr & (PAGE_SIZE as u64 - 1)) + info_size > PAGE_SIZE as u64 {
            return Errno::new(EINVAL).to_ret();
        }

        let mut info = ChInfo::default();

        // Snapshot kind + region details under the handle borrow; we
        // drop the borrow before touching user memory so the page
        // window mapping doesn't observe a process_handles mutation
        // mid-copy.
        {
            let Some(ph) = self.process_handles.get(&pid)
            else {
                return Errno::new(EBADF).to_ret();
            };
            let Some(handle_ref) = ph.get(fd)
            else {
                return Errno::new(EBADF).to_ret();
            };
            match handle_ref {
                Handle::NetChannel(sup) => {
                    info.kind = HandleKind::NetChannel as u8;
                    info.region_va = sup.user_va().raw();
                    info.region_size = sup.len() as u32;
                    // Snapshot the kernel-published peer / state via
                    // the shared header — cheap atomic loads, no
                    // syscall on the caller's side. The shared region
                    // is mapped via KDMAP so `try_as_ref` is the safe
                    // accessor; on a revoked channel we report the
                    // zeroed state which is what userspace would
                    // observe anyway after `close_handle`.
                    if let Some(nc) = sup.try_as_ref() {
                        let cur = nc.current();
                        info.peer_addr = cur.peer_addr.load(Ordering::Acquire);
                        info.peer_port = cur.peer_port.load(Ordering::Acquire);
                        info.state = cur.state.load(Ordering::Acquire);
                    }
                }
                Handle::File(_) => {
                    info.kind = HandleKind::File as u8;
                    // Region fields stay zero — fs reads bounce
                    // through per-fd scratch; userspace doesn't peek
                    // directly at any shared region.
                }
                Handle::Stdin => info.kind = HandleKind::Stdin as u8,
                Handle::Stdout => info.kind = HandleKind::Stdout as u8,
                Handle::Stderr => info.kind = HandleKind::Stderr as u8,
                Handle::EventFd(slot) => {
                    info.kind = HandleKind::EventFd as u8;
                    info.region_va = slot.region.user_va().raw();
                    info.region_size = slot.region.len() as u32;
                    // Read the flags field directly off the shared
                    // header — it's plain-write (no atomic) and we
                    // only need the create-time snapshot.
                    if let Some(efd) = slot.region.try_as_ref() {
                        info.flags = efd.flags;
                    }
                }
            }
        }

        let info_bytes = unsafe {
            core::slice::from_raw_parts(&info as *const _ as *const u8, info_size as usize)
        };
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = info_vaddr & !(PAGE_SIZE as u64 - 1);
        let page_off = (info_vaddr - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            page[page_off..page_off + info_bytes.len()].copy_from_slice(info_bytes);
        }
        0
    }

    /// `fs_seek(fd, offset, whence)` body. Sync — touches only the
    /// per-fd `OpenFile.offset`, no fs / DMA. Returns the new
    /// absolute offset on success or `-errno` on failure.
    pub fn run_fs_seek(&mut self, pid: u16, fd: u32, offset: i64, whence: u32) -> isize {
        let Some(ph) = self.process_handles.get_mut(&pid)
        else {
            return Errno::new(EBADF).to_ret();
        };
        let Some(handle_ref) = ph.get_mut(fd)
        else {
            return Errno::new(EBADF).to_ret();
        };
        let Handle::File(of) = handle_ref
        else {
            return Errno::new(EBADF).to_ret();
        };
        // Directories use the opaque `dir_cursor` tracked by
        // `fs_readdir`; seeking on a dir fd is a separate POSIX
        // operation (`seekdir`) we don't model. Reject so callers
        // get a clear error instead of silent breakage.
        if !of.is_regular {
            return Errno::new(EBADF).to_ret();
        }
        let new_offset: i64 = match whence {
            x if x == orbit_abi::fs::SEEK_SET => offset,
            x if x == orbit_abi::fs::SEEK_CUR => match (of.offset as i64).checked_add(offset) {
                Some(n) => n,
                None => return Errno::new(EINVAL).to_ret(),
            },
            x if x == orbit_abi::fs::SEEK_END => {
                let size = match of.fs.size(of.inode) {
                    Ok(s) => s,
                    Err(_) => return Errno::new(EIO).to_ret(),
                };
                match (size as i64).checked_add(offset) {
                    Some(n) => n,
                    None => return Errno::new(EINVAL).to_ret(),
                }
            }
            _ => return Errno::new(EINVAL).to_ret(),
        };
        if new_offset < 0 {
            return Errno::new(EINVAL).to_ret();
        }
        of.offset = new_offset as u64;
        new_offset as isize
    }

    /// `getcwd(buf)` body. Snapshots `Process.cwd`, copies the bytes
    /// (no NUL terminator) into the user buffer via a `UserPageWindow`,
    /// returns the byte count written. The buffer must lie within a
    /// single 4 KiB page — same constraint as the other fs copy-out
    /// syscalls.
    pub fn run_getcwd(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        buf_vaddr: u64,
        buf_len: usize,
    ) -> isize {
        if buf_len == 0 || buf_len > PAGE_SIZE {
            return Errno::new(EINVAL).to_ret();
        }
        if !orbit_abi::layout::user_range_ok(buf_vaddr, buf_len as u64) {
            return Errno::new(EFAULT).to_ret();
        }
        if (buf_vaddr & (PAGE_SIZE as u64 - 1)) + buf_len as u64 > PAGE_SIZE as u64 {
            return Errno::new(EINVAL).to_ret();
        }
        // Snapshot into a stack buffer so we can drop the borrow on
        // `self.processes` before calling into UserPageWindow (which
        // is independent kernel-managed mapping but keeping the
        // process borrow narrow is the conservative shape).
        let mut snap = [0u8; MAX_FS_PATH_LEN];
        let cwd_len = match self.processes.get(&pid) {
            Some(p) => {
                let bytes = p.cwd.as_bytes();
                if bytes.len() > snap.len() {
                    // Defensive: chdir caps at MAX_FS_PATH_LEN, so a
                    // longer cwd shouldn't be reachable, but if a future
                    // path lifts that cap we want EIO not a panic.
                    return Errno::new(EIO).to_ret();
                }
                snap[..bytes.len()].copy_from_slice(bytes);
                bytes.len()
            }
            None => return Errno::new(ESRCH).to_ret(),
        };
        if cwd_len > buf_len {
            return Errno::new(orbit_abi::errno::ERANGE).to_ret();
        }
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = buf_vaddr & !(PAGE_SIZE as u64 - 1);
        let page_off = (buf_vaddr - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            page[page_off..page_off + cwd_len].copy_from_slice(&snap[..cwd_len]);
        }
        cwd_len as isize
    }

    /// `getgroups(buf, count)` body. Snapshots the calling process's
    /// supplementary group list, copies up to `count` entries (one
    /// `u32` per slot) into the user buffer via a `UserPageWindow`.
    /// Returns the number of entries written. POSIX special case:
    /// `count == 0` returns the current group count without writing.
    /// Buffer must lie within a single 4 KiB page — same constraint as
    /// the other fs copy-out syscalls.
    pub fn run_getgroups(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        buf_vaddr: u64,
        count: usize,
    ) -> isize {
        // Snapshot the group list under the process borrow, then drop
        // it before touching UserPageWindow.
        let mut snap = [0u32; process::NGROUPS_MAX];
        let group_count = match self.processes.get(&pid) {
            Some(p) => {
                if p.groups.len() > snap.len() {
                    // Defensive: setgroups (when it lands) will cap
                    // input at NGROUPS_MAX; a longer list shouldn't
                    // be reachable, but EIO not panic if it is.
                    return Errno::new(EIO).to_ret();
                }
                for (i, &g) in p.groups.iter().enumerate() {
                    snap[i] = g;
                }
                p.groups.len()
            }
            None => return Errno::new(ESRCH).to_ret(),
        };

        // POSIX: count == 0 returns the count without writing. The
        // user buffer is not consulted in this case (callers pass
        // null when sizing).
        if count == 0 {
            return group_count as isize;
        }

        if count > process::NGROUPS_MAX {
            return Errno::new(EINVAL).to_ret();
        }
        if count < group_count {
            return Errno::new(orbit_abi::errno::ERANGE).to_ret();
        }

        let bytes = group_count
            .checked_mul(core::mem::size_of::<u32>())
            .and_then(|n| u64::try_from(n).ok())
            .unwrap_or(0);
        // Zero-byte writes are a no-op (group_count == 0 with
        // count > 0); skip the page checks because there's nothing
        // to map.
        if bytes != 0 {
            if !orbit_abi::layout::user_range_ok(buf_vaddr, bytes) {
                return Errno::new(EFAULT).to_ret();
            }
            if (buf_vaddr & (PAGE_SIZE as u64 - 1)) + bytes > PAGE_SIZE as u64 {
                return Errno::new(EINVAL).to_ret();
            }
            let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
            let page_base = buf_vaddr & !(PAGE_SIZE as u64 - 1);
            let page_off = (buf_vaddr - page_base) as usize;
            let page_pa =
                match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                    Some(p) => p as u64,
                    None => return Errno::new(EFAULT).to_ret(),
                };
            unsafe {
                let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
                let page = w.as_mut_slice();
                let dst = &mut page[page_off..page_off + bytes as usize];
                for (i, slot) in dst.chunks_exact_mut(4).enumerate() {
                    slot.copy_from_slice(&snap[i].to_le_bytes());
                }
            }
        }
        group_count as isize
    }

    /// `setuid(uid)` body. POSIX rules:
    ///   - euid == 0: stamp `uid` on all three triplet slots — the
    ///     privilege-drop path. Once dropped, `setuid` from non-root
    ///     can only toggle euid back to ruid or suid, never widen.
    ///   - euid != 0: set only euid, IFF `uid ∈ {ruid, suid}` (POSIX
    ///     privilege-toggle rule used by setuid-bit binaries today).
    ///     Anything else returns `EPERM`.
    ///
    /// On success walks the calling process's thread set and refreshes
    /// each `Thread.uid/euid/suid` snapshot so subsequent
    /// `getuid`/`geteuid` reads from sibling threads observe the new
    /// identity.
    pub fn run_setuid(&mut self, pid: u16, uid: u32) -> isize {
        let new_triplet: (u32, u32, u32) = {
            let proc = match self.processes.get_mut(&pid) {
                Some(p) => p,
                None => return Errno::new(ESRCH).to_ret(),
            };
            if proc.euid == 0 {
                proc.uid = uid;
                proc.euid = uid;
                proc.suid = uid;
            }
            else if uid == proc.uid || uid == proc.suid {
                proc.euid = uid;
            }
            else {
                return Errno::new(EPERM).to_ret();
            }
            (proc.uid, proc.euid, proc.suid)
        };
        // Walk threads to refresh per-thread snapshot. Same pattern as
        // run_pledge_req: snapshot tids first, then walk to drop the
        // borrow on `processes`.
        for tid in self.processes.get(&pid).unwrap().threads.iter().copied() {
            if let Some(pt) = self.threads.get(&tid) {
                let t = unsafe { (pt.0 as *mut Thread).as_mut_unchecked() };
                t.uid = new_triplet.0;
                t.euid = new_triplet.1;
                t.suid = new_triplet.2;
            }
        }
        0
    }

    /// `setgid(gid)` body. POSIX gid mirror of [`run_setuid`]: the
    /// privilege-test slot is `euid == 0` (matches POSIX — gid
    /// privilege still keys off uid==0, not gid==0).
    pub fn run_setgid(&mut self, pid: u16, gid: u32) -> isize {
        let new_triplet: (u32, u32, u32) = {
            let proc = match self.processes.get_mut(&pid) {
                Some(p) => p,
                None => return Errno::new(ESRCH).to_ret(),
            };
            if proc.euid == 0 {
                proc.gid = gid;
                proc.egid = gid;
                proc.sgid = gid;
            }
            else if gid == proc.gid || gid == proc.sgid {
                proc.egid = gid;
            }
            else {
                return Errno::new(EPERM).to_ret();
            }
            (proc.gid, proc.egid, proc.sgid)
        };
        let tids: alloc::vec::Vec<u32> = self
            .processes
            .get(&pid)
            .map(|p| p.threads.iter().copied().collect())
            .unwrap_or_default();
        for tid in tids {
            if let Some(pt) = self.threads.get(&tid) {
                let t = unsafe { (pt.0 as *mut Thread).as_mut_unchecked() };
                t.gid = new_triplet.0;
                t.egid = new_triplet.1;
                t.sgid = new_triplet.2;
            }
        }
        0
    }

    /// `setgroups(buf, count)` body. Replace the caller's supplementary
    /// group list with `count` u32s read from `buf_vaddr`. Requires
    /// `euid == 0`. `count == 0` is legal (empties the list — same
    /// shape as `setgroups(0, NULL)` on POSIX). Buffer must lie
    /// within a single page.
    ///
    /// Groups are stored on Process only — no per-thread snapshot to
    /// refresh, since `getgroups` already goes through the manager-side
    /// lookup.
    pub fn run_setgroups(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        buf_vaddr: u64,
        count: usize,
    ) -> isize {
        if count > process::NGROUPS_MAX {
            return Errno::new(EINVAL).to_ret();
        }
        // Snapshot euid first (read-only), short-circuit before any
        // user-buffer work if the caller isn't root.
        let euid = match self.processes.get(&pid) {
            Some(p) => p.euid,
            None => return Errno::new(ESRCH).to_ret(),
        };
        if euid != 0 {
            return Errno::new(EPERM).to_ret();
        }

        // Read the group list out of the user buffer. count == 0 is a
        // valid "empty list" request — no buffer access needed.
        let mut new_groups: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(count);
        if count > 0 {
            let bytes = (count * core::mem::size_of::<u32>()) as u64;
            if !orbit_abi::layout::user_range_ok(buf_vaddr, bytes) {
                return Errno::new(EFAULT).to_ret();
            }
            if (buf_vaddr & (PAGE_SIZE as u64 - 1)) + bytes > PAGE_SIZE as u64 {
                return Errno::new(EINVAL).to_ret();
            }
            let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
            let page_base = buf_vaddr & !(PAGE_SIZE as u64 - 1);
            let page_off = (buf_vaddr - page_base) as usize;
            let pa = match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) }
            {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
            unsafe {
                let mut w = user_page::UserPageWindow::map(pa, PAGE_SIZE);
                let page = w.as_mut_slice();
                let src = &page[page_off..page_off + bytes as usize];
                for chunk in src.chunks_exact(4) {
                    let mut le = [0u8; 4];
                    le.copy_from_slice(chunk);
                    new_groups.push(u32::from_le_bytes(le));
                }
            }
        }

        // Install. Process-only mutation; no thread snapshot to refresh.
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.groups = new_groups;
        }
        0
    }

    /// `setlogin(name_ptr, name_len)` body. Stamp the calling
    /// process's session login name. Requires `euid == 0`. Capped at
    /// 32 bytes (POSIX `_POSIX_LOGIN_NAME_MAX`, matches OpenBSD).
    /// Validates UTF-8 to keep the field compatible with the
    /// `getlogin` copy-out path.
    pub fn run_setlogin(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        name_vaddr: u64,
        name_len: usize,
    ) -> isize {
        const MAX_LOGIN_NAME: usize = 32;
        if name_len == 0 || name_len > MAX_LOGIN_NAME {
            return Errno::new(orbit_abi::errno::ENAMETOOLONG).to_ret();
        }
        let euid = match self.processes.get(&pid) {
            Some(p) => p.euid,
            None => return Errno::new(ESRCH).to_ret(),
        };
        if euid != 0 {
            return Errno::new(EPERM).to_ret();
        }
        if !orbit_abi::layout::user_range_ok(name_vaddr, name_len as u64) {
            return Errno::new(EFAULT).to_ret();
        }

        // Reuse copy_user_path's MAX_FS_PATH_LEN-sized scratch shape;
        // the helper does the page-walk + UTF-8 validation for us.
        let mut name_buf = [0u8; MAX_FS_PATH_LEN];
        let name = match self.copy_user_path(root_pa, name_vaddr, name_len, &mut name_buf) {
            Ok(s) => s,
            Err(e) => return e,
        };
        let name_owned = alloc::string::String::from(name);
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.login_name = Some(name_owned);
        }
        0
    }

    /// `getlogin(buf)` body. Snapshots `Process.login_name`, copies
    /// the bytes (no NUL terminator) into the user buffer via a
    /// `UserPageWindow`. Returns the byte count written, or `ENOENT`
    /// if no login name has been installed (initial state). Buffer
    /// must lie within a single 4 KiB page.
    pub fn run_getlogin(
        &mut self,
        pid: u16,
        root_pa: PhysAddr,
        buf_vaddr: u64,
        buf_len: usize,
    ) -> isize {
        if buf_len == 0 || buf_len > PAGE_SIZE {
            return Errno::new(EINVAL).to_ret();
        }
        if !orbit_abi::layout::user_range_ok(buf_vaddr, buf_len as u64) {
            return Errno::new(EFAULT).to_ret();
        }
        if (buf_vaddr & (PAGE_SIZE as u64 - 1)) + buf_len as u64 > PAGE_SIZE as u64 {
            return Errno::new(EINVAL).to_ret();
        }
        // setlogin (when it lands) will cap names at MAXLOGNAME; until
        // then no path produces a long name, but use the same cwd-shape
        // stack snapshot to keep the borrow narrow.
        const MAX_LOGIN_NAME: usize = 256;
        let mut snap = [0u8; MAX_LOGIN_NAME];
        let name_len = match self.processes.get(&pid) {
            Some(p) => match &p.login_name {
                Some(name) => {
                    let bytes = name.as_bytes();
                    if bytes.len() > snap.len() {
                        return Errno::new(EIO).to_ret();
                    }
                    snap[..bytes.len()].copy_from_slice(bytes);
                    bytes.len()
                }
                None => return Errno::new(orbit_abi::errno::ENOENT).to_ret(),
            },
            None => return Errno::new(ESRCH).to_ret(),
        };
        if name_len > buf_len {
            return Errno::new(orbit_abi::errno::ERANGE).to_ret();
        }
        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };
        let page_base = buf_vaddr & !(PAGE_SIZE as u64 - 1);
        let page_off = (buf_vaddr - page_base) as usize;
        let page_pa =
            match unsafe { mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base)) } {
                Some(p) => p as u64,
                None => return Errno::new(EFAULT).to_ret(),
            };
        unsafe {
            let mut w = user_page::UserPageWindow::map(page_pa, PAGE_SIZE);
            let page = w.as_mut_slice();
            page[page_off..page_off + name_len].copy_from_slice(&snap[..name_len]);
        }
        name_len as isize
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

    /// End-of-manager-pass nudge for k_gpu. Called from `k_manage`'s
    /// loop after `drain_wakes` + `check_net` and before
    /// `assign_threads`, so any k_gpu-readying CAS lands in `self.ready`
    /// in time for this same pass to dispatch.
    ///
    /// Why batch here instead of nudging from each `CONSOLE_RING`
    /// producer? Per-producer wakes worked for low-rate paths
    /// (cycle / insert / remove) but flooded `WAKE_QUEUE` once high-rate
    /// paths (`push_chunk` from console_write spam, `push_present` from
    /// orbit-top-std at 4 Hz) joined in — pinned k_gpu Ready
    /// continuously and starved k_net + manager. Concrete repro:
    /// launching orbit-top-std mid-eza-stress wedged the whole system
    /// silently. Batching at end-of-pass collapses every push that
    /// landed during the pass — manager-side handler push, trap-context
    /// `push_chunk` from a concurrent user-hart syscall — into a
    /// single Ready transition.
    ///
    /// `set_wake_reason_where`'s CAS is a no-op when k_gpu is already
    /// Ready / Running / Assigned; it only flips Suspended/Blocking →
    /// Ready. So the call is cheap and idempotent — calling it on an
    /// already-running k_gpu just OR's TICKLE into wake_override and
    /// returns.
    pub fn nudge_gpu_if_pending(&mut self) {
        if crate::drivers::k_gpu::CONSOLE_RING.is_empty() {
            return;
        }
        let Some(tid) = self.gpu_thread_tid
        else {
            return;
        };
        self.set_wake_reason_where(process::wake_reason::TICKLE, |t| t.tid == tid);
    }

    /// Manager-end batched nudge for k_serial. Same shape and rationale
    /// as `nudge_gpu_if_pending`: every `SERIAL_RING` push that landed
    /// during this manager pass (trace from any hart in any context)
    /// gets folded into one TICKLE → Ready transition for k_serial.
    /// Without this, k_serial would sleep out its 50 ms park before
    /// draining trace lines, and a sustained burst would fill the ring
    /// and force producers onto the spinlock fallback — which was the
    /// whole problem k_serial exists to solve.
    pub fn nudge_serial_if_pending(&mut self) {
        if crate::drivers::k_serial::SERIAL_RING.is_empty() {
            return;
        }
        let Some(tid) = self.serial_thread_tid
        else {
            return;
        };
        self.set_wake_reason_where(process::wake_reason::TICKLE, |t| t.tid == tid);
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
