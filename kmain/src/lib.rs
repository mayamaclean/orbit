#![no_std]

extern crate alloc;

use crate::kernel::{shared_user_ptr::SharedUserPtr, shootdown::CPU_COUNT};
use alloc::{collections::btree_map::BTreeMap, vec::Vec};
use core::{
    arch::asm,
    ptr::null_mut,
    sync::atomic::{AtomicBool, Ordering},
};
use device::{HartContext, TrapFrame};
use net_channel::NetChannel;
use orbit_abi::{layout::UserVa, perms::Permissions};
use process::{Thread, ThreadState};
use smoltcp::{
    iface::{PollResult, SocketHandle, SocketSet},
    socket::{dhcpv4, tcp::CongestionControl},
    storage::RingBuffer,
};
use tracing::trace;

use crate::{
    drivers::e1000::E1000,
    kernel::context::{
        enter_hart_context, exit_thread_with_state, get_hart_context, hart_has_thread,
    },
};

pub mod channel;
pub mod drivers;
pub mod hw;
pub mod kernel;
pub mod ktrace;
pub mod tracked_heap;

/// Scope guard that enables `sstatus.SUM` for the duration of its lifetime
/// so the supervisor can touch `U=1` pages. Syscall handlers build one,
/// access the user buffer through the user VA (the trap vector stays on
/// the user's satp), and drop it before returning.
///
/// Outside a `UserAccess` scope any kernel access that lands on a user VA
/// faults, which is the whole point — a stray user-pointer deref is a bug,
/// and SUM = 0 turns it into a diagnosable trap instead of a silent read
/// through whatever happened to be mapped at that address.
pub struct UserAccess {
    _private: (),
}

impl UserAccess {
    #[inline]
    pub fn enter() -> Self {
        unsafe {
            riscv::register::sstatus::set_sum();
        }
        Self { _private: () }
    }

    /// Borrow a read-only byte slice at a user VA. Caller must have
    /// verified (via PT walk) that the range is mapped and user-readable.
    /// Lifetime ties the slice to this guard so it can't outlive SUM.
    #[inline]
    pub unsafe fn slice<'s>(&'s self, vaddr: UserVa, len: usize) -> &'s [u8] {
        unsafe { core::slice::from_raw_parts(vaddr.raw() as *const u8, len) }
    }

    /// Borrow a writable byte slice at a user VA. Caller must have
    /// verified the range is mapped user-writable. Lifetime ties the
    /// slice to this guard so it can't outlive SUM.
    #[inline]
    pub unsafe fn slice_mut<'s>(&'s self, vaddr: UserVa, len: usize) -> &'s mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(vaddr.raw() as *mut u8, len) }
    }

    /// Read a value of type `T` from a user VA. Caller must have verified
    /// the source is mapped and the read is size/alignment-safe.
    #[inline]
    pub unsafe fn read_volatile<T>(&self, vaddr: UserVa) -> T {
        unsafe { core::ptr::read_volatile(vaddr.raw() as *const T) }
    }
}

impl Drop for UserAccess {
    fn drop(&mut self) {
        unsafe {
            riscv::register::sstatus::clear_sum();
        }
    }
}

pub struct ProcessComponents<'c> {
    pub elf_blob: &'c [u8],
    pub stack_size: u64,
    pub allowed_affinity: u64,
    pub affinity: u64,
    pub parent_pid: u16,
    pub argv_bytes: Option<&'c [u8]>,
    pub envp_bytes: Option<&'c [u8]>,
    pub perms: Option<Permissions>,
    /// Initial cwd for the child. `None` = inherit verbatim from
    /// `parent_pid`'s cwd (or `"/"` for boot, where parent_pid is 0).
    /// `Some(p)` overrides — caller is `Command::current_dir(...)` or
    /// equivalent. Must be absolute UTF-8; the manager validates the
    /// dir exists in the active fs before installing.
    pub cwd: Option<&'c str>,
    /// `Some(parent_pid)` ⇒ install [`process::Process::stdout_redirect`]
    /// on the new process so its `console_write` bytes route to the
    /// parent's pane. Reachable only via `CREATE_PROCESS_V2` with
    /// `stdout_capture == 1`; legacy `CREATE_PROCESS` /
    /// `CREATE_PROCESS_EX` always pass `None`.
    pub stdout_redirect: Option<u16>,
}

pub fn write_sswi(hart: usize, val: u32) {
    unsafe {
        let base = crate::kernel::memmap::kmmio_sswi() as *mut u32;
        base.add(hart).write_volatile(val);
    }
}

/// Kick a hart out of WFI by writing its CLINT MSIP slot. Used on boot to pull
/// harts 1..N out of bl's `kinit_hart` spin. Runs S-mode under the kernel
/// satp, so it writes via the KMMIO alias of CLINT (bl's M-mode path still
/// uses the raw PA via `device::wake_hart`).
pub fn kick_machine_hart(hart: usize) {
    unsafe {
        let base = crate::kernel::memmap::kmmio_clint() as *mut u32;
        base.add(hart).write_volatile(1);
    }
}

pub fn kick_machine_harts(hart_count: usize) {
    for hart in 0..hart_count {
        kick_machine_hart(hart);
    }
}

pub fn supervisor_wake_hart(hart: usize) {
    //trace!("hart{} sending wake ipi to hart{hart}", get_hart_context().hart_id);
    write_sswi(hart, 1);
}

pub fn supervisor_clear_ipi(hart: usize) {
    write_sswi(hart, 0);
}

pub fn wait_until(target_time: u64) {
    unsafe {
        riscv::register::sstatus::clear_sie();

        // stimecmp
        asm!(
            "csrw 0x14D, {}",
            in(reg) target_time
        );
        riscv::register::sie::set_stimer();

        while riscv::register::time::read64() < target_time {
            riscv::asm::wfi();
        }
    }
}

pub fn wait_for(target: u64) {
    wait_until(riscv::register::time::read64().wrapping_add(target));
}

pub static MANAGER_LOCK: AtomicBool = AtomicBool::new(false);

pub fn try_acquire_manager() -> bool {
    MANAGER_LOCK
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
        .is_ok()
}

pub fn release_manager() {
    MANAGER_LOCK.store(false, Ordering::Release);
}

/// Dispatch-site permission gate. Called by `s_trap`'s cause=8 arm
/// (the U-mode ecall path) before the syscall dispatch match runs.
/// Returns `true` if the syscall is allowed under the calling
/// thread's snapshotted [`Permissions`]; on `false` the gate has
/// queued a [`DenialEvent::PermDeny`] audit event for the manager
/// to fold into the kernel-wide ring, and the caller is expected
/// to short-circuit the dispatch with `-EPERM`.
///
/// Lock-free hot path. The thread's `Permissions` is read without
/// synchronisation (manager-side pledge propagation walks every
/// live thread under MANAGER_LOCK; we accept the brief window where
/// a sibling sees a not-yet-narrowed snapshot). Denials push onto
/// [`crate::kernel::DENIAL_EVENT_QUEUE`] — a `try_push` so the trap
/// path never spins on a full queue; a full queue drops the event
/// and bumps [`crate::kernel::DENIAL_EVENTS_DROPPED`] for diagnostics.
///
/// [`DenialEvent::PermDeny`]: orbit_abi::denial::DenialEvent::PermDeny
/// [`Permissions`]: orbit_abi::perms::Permissions
pub fn perm_gate_check(thread: &Thread, syscall: usize) -> bool {
    let perms = thread.permissions;
    if perms.allows(syscall) {
        return true;
    }

    // Build the would-be-denial event. `now_ticks` reads the time
    // CSR directly — same domain as `query_stats` so log readers can
    // align timelines without a conversion.
    let now_ticks = riscv::register::time::read64();
    let event = orbit_abi::denial::DenialEvent::PermDeny {
        required_class: orbit_abi::perms::Permissions::class_for(syscall).raw(),
        perms: perms.perms,
        time_ticks: now_ticks,
        tid: thread.tid,
        sysno: syscall as u32,
        source_role: perms.role,
        pid: thread.pid,
    };

    tracing::warn!("[{}.{}] {event:?}", thread.pid, thread.tid);

    if crate::kernel::DENIAL_EVENT_QUEUE.push(Some(event)).is_err() {
        crate::kernel::DENIAL_EVENTS_DROPPED.fetch_add(1, Ordering::Relaxed);
    }

    false
}

/// Short-circuit the syscall dispatch with `-EPERM`. Used by
/// `s_trap`'s cause=8 arm when [`perm_gate_check`] returns `false`:
/// the audit event is already queued, and this commits the EPERM
/// return value into the frame + advances pc past the ecall via
/// the same `dispatch_syscall` shim that the regular handlers use.
pub fn enforce_eperm(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |_t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: -(orbit_abi::errno::EPERM as isize),
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn k_hart_loop() -> ! {
    let hart_context =
        unsafe { (riscv::register::sscratch::read() as *const HartContext).as_ref_unchecked() };

    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };

    loop {
        // Always start a loop iteration with sstatus.SIE clear. After
        // WFI returns from a trap, `sret` restores SIE from SPIE —
        // typically back ON — so without this clear the dispatch path
        // below would run with async traps live, re-opening the same
        // race the `arm_hart_timer` (no-SIE) variant was meant to
        // close. This single clear at the top covers both the dispatch
        // window and the manager critical section.
        unsafe {
            riscv::register::sstatus::clear_sie();
        }

        if hart_has_thread(hart_context) {
            // Arm the timer without enabling SIE: `enter_hart_context`
            // is a one-way trip via sret, and the new thread's SIE
            // gets restored from SPIE there. Leaving SIE off across
            // the kernel-side window keeps the dispatch path
            // uninterruptible by async traps.
            arm_hart_timer(1_000_000);
            unsafe {
                enter_hart_context(hart_context);
            }
        }

        // Disable sstatus.SIE around the acquire + critical section. If a
        // trap fired mid-section the handler would long-jump via kptr back
        // to k_hart_loop without releasing MANAGER_LOCK, deadlocking all
        // harts. (Already off from the loop-top clear above; redundant
        // store kept for clarity / defense in depth.)
        unsafe {
            riscv::register::sstatus::clear_sie();
        }
        // Default WFI duration when we don't know the next sleep
        // deadline (lock contention path, or no sleepers). The
        // manager runs at least this often as a safety net for any
        // SLEEP_INBOX entry that landed between our read and the WFI.
        const WFI_HEARTBEAT_CYCLES: u64 = 100_000;
        let mut wfi_cycles: u64 = WFI_HEARTBEAT_CYCLES;
        if try_acquire_manager() {
            // Bucket hook 4: enter scheduler critical section.
            crate::kernel::accounting::switch_bucket(
                hart_context,
                crate::kernel::accounting::HartBucket::Scheduler,
            );

            orbit.cleanup_threads_and_processes();
            orbit.drain_pending_work();
            // Drain denial events queued by the dispatch-site gate —
            // best-effort audit logging into the kernel-wide ring +
            // per-process counters. Cheap (lock-free MPSC pop), runs
            // before assign_threads so a thread that ecall'd a denied
            // syscall has its counter visible to a same-pass
            // query_stats from another hart.
            orbit.drain_denial_events();
            // Drain wake events *before* assign_threads so any thread
            // whose wake_time was just bumped to 0 is observed Ready
            // by the next scheduler scan in this same critical section.
            orbit.drain_wakes();
            orbit.check_net();
            // End-of-pass batched nudge: every CONSOLE_RING push that
            // landed during this pass (manager-side handlers + concurrent
            // trap-context `push_chunk` from user-hart syscalls) gets a
            // single Ready transition for k_gpu instead of one wake per
            // push. Runs *before* `assign_threads` so the readied k_gpu
            // is dispatched this same pass — no extra manager-pass of
            // latency. See `Orbit::nudge_gpu_if_pending` for rationale.
            orbit.nudge_gpu_if_pending();
            // Same shape as the gpu nudge — fold every SERIAL_RING
            // push that landed during this pass (trace from any hart
            // in any context) into one Ready transition for k_serial.
            orbit.nudge_serial_if_pending();
            orbit.assign_threads(hart_context);

            // Read the next sleep deadline while still holding the
            // lock. Out of the lock-guarded section the heap can be
            // mutated by the next manager pass; the value we cache
            // here is the snapshot at end-of-pass.
            let now = riscv::register::time::read64();
            wfi_cycles = orbit.next_sleep_in_cycles(now, WFI_HEARTBEAT_CYCLES);

            // Bucket hook 5: leave scheduler critical section. Switch
            // *before* release so the next acquirer's snapshot doesn't
            // race a still-Scheduler bucket on this hart.
            crate::kernel::accounting::switch_bucket(
                hart_context,
                crate::kernel::accounting::HartBucket::Kernel,
            );
            release_manager();

            if hart_has_thread(hart_context) {
                arm_hart_timer(1_000_000);
                unsafe {
                    enter_hart_context(hart_context);
                }
            }
        }

        // For the WFI path we *do* want SIE on so async traps fire and
        // wake us — WFI returns on any pending interrupt regardless of
        // SIE, but with SIE off the trap handler doesn't run and the
        // pending IRQ would just stay set until something else picked
        // it up (like an sret to user-mode, where SIE is irrelevant).
        // Keeping SIE on here means PLIC handlers / SSWI dispatchers
        // run promptly while the hart is otherwise idle.
        unsafe {
            riscv::register::sie::set_ssoft();
            // Sized from sleep-heap peek when we held the lock above:
            // wakes at the earliest deadline rather than waiting the
            // full heartbeat. Capped at WFI_HEARTBEAT_CYCLES so
            // SLEEP_INBOX entries pushed after our snapshot still get
            // observed within one heartbeat.
            setup_hart_timer(wfi_cycles);

            // Bucket hook 3: drop into idle. Bracketed automatically
            // on wake by the s_trap entry hook (→ Kernel).
            crate::kernel::accounting::switch_bucket(
                hart_context,
                crate::kernel::accounting::HartBucket::Idle,
            );
            riscv::asm::wfi();
        }
    }
}

#[derive(Debug)]
pub struct SocketReq {
    /// Refcounted handle on the NetChannel. Cloned from the registry when
    /// the manager enqueues the request; k_net drops its clone when the
    /// socket goes through `socket_deletions`.
    netchan: SharedUserPtr<NetChannel>,
    nc_type: usize,
    pid: u16,
    /// Reconciler state: latched binding (sticky from create-time),
    /// current phase, ring-ack flags, retain backoff. Threaded into
    /// `NetChannel::update_tcp` each poll; the netchannel impl never
    /// reads from shared memory for the binding params, only for the
    /// per-session `engaged` flag.
    ctx: net_channel::ChannelCtx,
}

impl orbit_core::net::RevocableConn for SocketReq {
    fn is_revoked(&self) -> bool {
        self.netchan.is_revoked()
    }
}

#[repr(align(64))]
pub struct NetPackage {
    phy: Option<E1000>,
    iface: Option<smoltcp::iface::Interface>,
    socket_reqs: Vec<heapless::spsc::Queue<SocketReq, 8>>,
    socket_associations: heapless::spsc::Queue<(usize, SocketHandle), 8>,
    socket_deletions: heapless::spsc::Queue<SocketHandle, 8>,
}

fn set_ipv4_addr(iface: &mut smoltcp::iface::Interface, cidr: smoltcp::wire::Ipv4Cidr) {
    iface.update_ip_addrs(|addrs| {
        addrs.clear();
        addrs.push(smoltcp::wire::IpCidr::Ipv4(cidr)).unwrap();
    });
}

fn handle_dhcp_event(
    mut iface: smoltcp::iface::Interface,
    event: dhcpv4::Event,
) -> smoltcp::iface::Interface {
    use tracing::{info, warn};

    match event {
        dhcpv4::Event::Configured(config) => {
            info!("net: DHCP config acquired!");

            info!("net: IP address: {}", config.address);
            set_ipv4_addr(&mut iface, config.address);

            if let Some(router) = config.router {
                info!("net: Default gateway: {}", router);
                iface.routes_mut().add_default_ipv4_route(router).unwrap();
            }
            else {
                warn!("net: Default gateway: None");
                iface.routes_mut().remove_default_ipv4_route();
            }

            for (i, s) in config.dns_servers.iter().enumerate() {
                info!("net: DNS server {}: {}", i, s);
            }
        }
        dhcpv4::Event::Deconfigured => {
            warn!("net: DHCP lost config!");
            iface.update_ip_addrs(|addrs| addrs.clear());
            iface.routes_mut().remove_default_ipv4_route();
        }
    }
    iface
}

#[unsafe(no_mangle)]
pub extern "C" fn k_net(device: *mut NetPackage) {
    use tracing::{error, info};

    unsafe {
        riscv::register::sstatus::clear_sie();
    }

    info!("net: pkg@{device:016X?}");

    let net_package = unsafe { device.as_mut_unchecked() };

    let NetPackage {
        phy,
        iface,
        socket_reqs,
        socket_associations,
        socket_deletions,
    } = net_package;
    let mut phy = match phy.take() {
        Some(p) => p,
        None => {
            error!("net: no phy");
            unsafe { exit_thread_with_state(ThreadState::Exited) };
        }
    };

    let mut iface = match iface.take() {
        Some(i) => i,
        None => {
            error!("net: no iface");
            unsafe { exit_thread_with_state(ThreadState::Exited) };
        }
    };

    let mut sockets = SocketSet::new(Vec::new());

    let dhcp_sock = dhcpv4::Socket::new();
    let dhcp_handle = sockets.add(dhcp_sock);

    let mut user_conns: BTreeMap<smoltcp::iface::SocketHandle, SocketReq> = BTreeMap::new();
    let mut user_revocations: Vec<SocketHandle> = Vec::new();

    loop {
        unsafe {
            riscv::register::sstatus::clear_sie();
            riscv::register::sstatus::set_sum();
        }

        let mut now = riscv::register::time::read();
        let mut timestamp = smoltcp::time::Instant::from_micros(now as i64 / 10);

        // Always run iface.poll on every wake. The previous
        // `now >= next_poll || phy.read_interrupt_status() > 0` gate
        // was an optimization to skip iface.poll when nothing's
        // pending — but with the e1000 PLIC handler now acking ICR
        // *before* k_net runs, `read_interrupt_status` would return 0
        // even on an IRQ-driven wake, and we'd wrongly skip the poll
        // we were just woken up to do. iface.poll returns None
        // immediately when there's nothing to do, so unconditional
        // polling is cheap; the timer-elapsed/IRQ-pending checks only
        // saved a function call.
        unsafe {
            core::arch::asm!("fence iorw, iorw");
        }

        for _ in 0..8 {
            if iface.poll(timestamp, &mut phy, &mut sockets) == PollResult::None {
                break;
            }

            now = riscv::register::time::read();
            timestamp = smoltcp::time::Instant::from_micros(now as i64 / 10);

            if let Some(event) = sockets.get_mut::<dhcpv4::Socket>(dhcp_handle).poll() {
                iface = handle_dhcp_event(iface, event);
            }
        }

        unsafe {
            core::arch::asm!("fence iorw, iorw");
        }

        orbit_core::net::drain_socket_deletions(
            &mut user_conns,
            || socket_deletions.dequeue(),
            |h| {
                sockets.remove(h);
            },
        );

        orbit_core::net::prune_revoked_conns(&mut user_conns, &mut user_revocations, |h| {
            sockets.remove(h);
        });

        // ChannelCtx wants microseconds matching the iface clock — the
        // iface is fed `Instant::from_micros(now / 10)` so the same
        // unit applies here. Keep the shift in one place.
        let now_us = (now / 10) as u64;

        let mut do_second_poll = false;
        for (sock_handle, req) in user_conns.iter_mut() {
            if let Some(nc) = req.netchan.try_as_ref() {
                if req.nc_type == 0 {
                    let socket = sockets.get_mut::<smoltcp::socket::tcp::Socket>(*sock_handle);
                    let pre_recv_queue = socket.recv_queue();
                    let (iface_back, outcome) = nc.update_tcp(iface, socket, &mut req.ctx, now_us);
                    iface = iface_back;

                    // Re-poll if any data is in flight OR if update_tcp advanced
                    // smoltcp's read pointer (drained any increment). The drain
                    // case is the one that bites: socket.recv() inside update_tcp
                    // frees window space internally, but smoltcp won't emit the
                    // window-update segment until iface.poll runs again. Without
                    // this, drain-but-no-new-data cycles wait the full ~100ms
                    // heartbeat for the window-update to go on the wire, gating
                    // every round-trip on the heartbeat instead of the wake chain.
                    if socket.recv_queue() > 0
                        || socket.send_queue() > 0
                        || socket.recv_queue() != pre_recv_queue
                    {
                        do_second_poll = true;
                    }

                    // If the user-visible state of the channel just
                    // changed (state byte moved, fresh rx slice staged,
                    // tx slice freed), push a WakeEvent so the manager
                    // wakes the owner thread now instead of letting it
                    // miss the change and spin on its own sleep cadence.
                    // `Pid` is coarse — wakes every thread of the
                    // process — but cheap and correct (each thread re-
                    // checks its park condition and re-sleeps if not
                    // actually ready).
                    if outcome.should_wake_user() {
                        let _ =
                            crate::kernel::wake_queue_push(crate::kernel::WakeEvent::Pid(req.pid));

                        do_second_poll = true;
                    }
                }
            }
            else {
                user_revocations.push(*sock_handle);
            }
        }

        orbit_core::net::prune_revoked_conns(&mut user_conns, &mut user_revocations, |h| {
            sockets.remove(h);
        });

        // Re-poll after update_tcp. update_tcp's `socket.recv()` calls
        // advance the rx pointer (freeing window space) and `socket.send()`
        // calls queue tx data — both leave segments smoltcp wants to
        // emit *now* (window updates, outgoing data, piggybacked ACKs).
        // Without this second pass, those segments wait for the next
        // wake-up event — typically the keep-alive timer. With a 5 s
        // keep-alive, sustained transfers throttled to that cadence and
        // bursts arrived ~once per keep-alive period.
        if do_second_poll {
            unsafe {
                core::arch::asm!("fence iorw, iorw");
            }
            for _ in 0..8 {
                if iface.poll(timestamp, &mut phy, &mut sockets) == PollResult::None {
                    break;
                }

                now = riscv::register::time::read();
                timestamp = smoltcp::time::Instant::from_micros(now as i64 / 10);

                if let Some(event) = sockets.get_mut::<dhcpv4::Socket>(dhcp_handle).poll() {
                    iface = handle_dhcp_event(iface, event);
                }
            }
            unsafe {
                core::arch::asm!("fence iorw, iorw");
            }
        }

        let default_wake = now + 1_000_000;
        let wake_time = iface
            .poll_at(timestamp, &mut sockets)
            .map(|i| i.total_micros() as usize * 10)
            .unwrap_or(default_wake);

        for q in socket_reqs.iter_mut() {
            while let Some(mut req) = q.dequeue() {
                info!("net: processing req {req:?}");

                if req.nc_type == 0 {
                    let req_pid = req.pid;
                    let (txr, rxr) = req.netchan.as_ref().rings();

                    info!(
                        "net: tcp socket ring lens: rx={},tx={}",
                        rxr.len(),
                        txr.len()
                    );

                    // Lockstep-offsets mode: the kernel publishes
                    // `(offset, len)` slices to userspace via
                    // `get_next_tx`/`get_next_rx`, and userspace
                    // writes directly into the shared storage at
                    // those offsets. smoltcp's default
                    // empty-ring `read_at = 0` snap-back would
                    // rebase the write pointer away from where
                    // userspace just wrote, causing stale-byte
                    // retransmits on the wire. See
                    // `RingBuffer::set_lockstep_offsets` for the
                    // full rationale.
                    let mut tx_buffer = RingBuffer::new(txr);
                    tx_buffer.set_lockstep_offsets(true);
                    let mut rx_buffer = RingBuffer::new(rxr);
                    rx_buffer.set_lockstep_offsets(true);

                    let mut sock = smoltcp::socket::tcp::Socket::new(rx_buffer, tx_buffer);
                    // Keep-alive: smoltcp emits an ACK with current
                    // window after this interval of socket-level
                    // silence. Doubles as the window-update piggyback
                    // we need when peer has stalled mid-burst (drained
                    // rx buffer, no new data → smoltcp's window_to_update
                    // heuristic doesn't fire on its own, peer parks on
                    // RTO backoff). 100 ms is short enough to nudge peer
                    // before its RTO doubles to noticeable territory,
                    // long enough to not flood normal traffic with
                    // probes (steady-state ACKs reset the timer anyway).
                    sock.set_keep_alive(Some(smoltcp::time::Duration::from_millis(5000)));
                    sock.set_ack_delay(None);
                    sock.set_nagle_enabled(true);
                    sock.set_congestion_control(CongestionControl::Cubic);

                    let handle = sockets.add(sock);

                    info!("net: created tcp socket: {handle:?}");

                    // Arm the smoltcp socket *now*, in the same iteration
                    // we created it, so the very next `iface.poll` sees
                    // a Listening (or SynSent) socket — not a freshly-
                    // CLOSED one. Without this, any SYN that lands
                    // between this iteration's poll and the next
                    // iteration's update_tcp call gets RST'd. That's
                    // the "first few attempts at connecting fail" race
                    // — closed-window of one iface poll.
                    if let Some(nc) = req.netchan.try_as_ref() {
                        let socket = sockets.get_mut::<smoltcp::socket::tcp::Socket>(handle);
                        let (iface_back, outcome) =
                            nc.update_tcp(iface, socket, &mut req.ctx, now_us);
                        iface = iface_back;
                        // First-poll arm typically writes state=1
                        // (Listening / SynSent) — wake the owner that
                        // may already be parked on `next_session`.
                        if outcome.should_wake_user() {
                            let _ = crate::kernel::WAKE_QUEUE
                                .push(crate::kernel::WakeEvent::Pid(req.pid));
                        }
                    }

                    user_conns.insert(handle, req);

                    if let Err(assoc) = socket_associations.enqueue((req_pid as usize, handle)) {
                        error!("net: was unable to inform manager of socket association {assoc:?}");
                    }
                }
            }
        }

        // Drop SUM before parking — the kthread doesn't need user
        // access while suspended, and leaving SUM set across the
        // park widens the window in which a stray kernel deref of a
        // user VA goes silently through instead of faulting.
        unsafe {
            riscv::register::sstatus::clear_sum();
        }

        // Park until either `wake_time` ticks elapse or a producer
        // (e1000 PLIC handler, update_tcp ring-progress, nc_yield
        // syscall) ORs a wake reason into our wake_override. Resumes
        // at the next iteration of this loop with all locals intact.
        //
        // Previously this was a hand-rolled `cscratch2 = 1; ebreak`
        // that left state=Running and rode the s_trap cause=3 path
        // → check_context_and_switch (Running → Ready) → busy-loop —
        // a workaround for the double-dispatch race that
        // kthread_park's stack-switch-then-publish ordering closes
        // properly. The wake_time field is now honored, so the
        // kthread actually sleeps between events.
        crate::kernel::context::kthread_park(
            ThreadState::Suspended,
            core::cmp::min(default_wake, wake_time),
        );
    }
}

pub fn setup_hart_timer(cycles: u64) {
    unsafe {
        riscv::register::sie::set_stimer();
        riscv::register::sstatus::set_sie();

        let t = riscv::register::time::read64().wrapping_add(cycles);
        // write stimecmp
        asm!(
            "csrw 0x14D, {}",
            in(reg) t
        );
    }
}

/// Like [`setup_hart_timer`] but leaves `sstatus.SIE` alone — for use
/// on the dispatch path where the caller wants the timer armed but not
/// async traps delivered until `sret` restores the new thread's
/// `SPIE → SIE`. Pre-this, k_hart_loop's `setup_hart_timer` enabled
/// SIE and then `load_thread_into_hart_context_and_jump`'s first line
/// cleared it; the window between exposed every dispatch to a
/// timer/SSWI race that the trap-handler mode-gate caught but
/// log-spammed.
pub fn arm_hart_timer(cycles: u64) {
    unsafe {
        riscv::register::sie::set_stimer();

        let t = riscv::register::time::read64().wrapping_add(cycles);
        asm!(
            "csrw 0x14D, {}",
            in(reg) t
        );
    }
}

#[unsafe(no_mangle)]
pub fn check_context_and_switch() -> ! {
    let c = get_hart_context();
    let t = c.current.load(Ordering::Acquire);

    if t != null_mut() {
        let thread: &Thread = unsafe { (t as *mut Thread).as_ref_unchecked() };
        let thread_state = thread.state.load(Ordering::Acquire);
        if thread_state == ThreadState::Running as usize {
            unsafe {
                exit_thread_with_state(ThreadState::Ready);
            }
        }
        else if thread_state == ThreadState::Exited as usize {
            unsafe {
                exit_thread_with_state(ThreadState::Exited);
            }
        }
        else if thread_state == ThreadState::Suspended as usize {
            tracing::info!(
                "DBG check_ctx_switch null-cur(Suspended): hart={} tid={}",
                c.hart_id,
                thread.tid,
            );
            c.current.store(null_mut(), Ordering::Release);
        }
        else if thread_state == ThreadState::Blocking as usize {
            tracing::info!(
                "DBG check_ctx_switch null-cur(Blocking): hart={} tid={}",
                c.hart_id,
                thread.tid,
            );
            c.current.store(null_mut(), Ordering::Release);
        }
    }

    unsafe {
        enter_hart_context(c);
    }
}

pub fn update_thread_and_trap_frame(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
    from_user: bool,
) {
    let cptr = hart_context.current.load(Ordering::Acquire);
    if cptr == null_mut() {
        return;
    }
    let thread: &mut Thread = unsafe { (cptr as *mut Thread).as_mut_unchecked() };

    // Watchdog: if the trap's from_user disagrees with the current
    // thread's mode, the hart's `current` was retargeted between when
    // the trap fired and when we got here. The mode-gate in
    // `update_trap_frame` skips the save in that case, so we don't
    // corrupt thread.pc.
    //
    // An `Assigned` thread is the benign case: `assign_threads`
    // transitions Ready → Assigned and writes the ptr into a hart's
    // `current` slot before the dispatch runs. If that hart was
    // still in S-mode (e.g., WFI inside k_hart_loop), the SSWI/timer
    // that delivers the wake naturally fires with from_user=false.
    // No actual bug — suppress to keep the watchdog signal
    // meaningful. Any other state with a mismatch is a real
    // scheduling anomaly worth logging.
    let thread_in_user_mode = thread.mode == riscv::register::sstatus::SPP::User;
    if thread_in_user_mode != from_user
        && thread.state.load(Ordering::Acquire) != ThreadState::Assigned as usize
    {
        tracing::warn!(
            "trap mode mismatch on cpu{}: tid={} mode={:?} state={} from_user={} epc={:#x} — \
             scheduler retargeted current mid-trap?",
            hart_context.hart_id,
            thread.tid,
            thread.mode,
            thread.state.load(Ordering::Acquire),
            from_user,
            epc,
        );
    }

    orbit_core::trap::update_trap_frame(thread, epc, frame, from_user);
}

/// `exit(code)` (sysno 0) — POSIX `_exit(2)` / `exit_group(2)` shape.
///
/// Terminates the *whole calling process*, not just the calling
/// thread. Marks every sibling thread `Exited` and IPIs any hart
/// still running one so it traps and bails to k_idle. The manager's
/// next `cleanup_threads_and_processes` pass reaps the process.
///
/// Why exit-group instead of thread-exit: every consumer of sysno 0
/// (orbit-rt's `_start`, std-on-orbit's `_start`) calls it as the
/// process's final act after `main` returns. A binary that uses any
/// thread pool (rayon, the std test harness, `Command`'s reaper)
/// leaves daemon threads parked in `futex_wait` after main exits;
/// thread-exit only takes the leader, so the process dangles.
/// Promote thread-exit semantics to a separate syscall (e.g.
/// `THREAD_EXIT` paired with `pthread_exit`) when a real consumer
/// needs it.
///
/// Noreturn: ends with `exit_thread_with_state(Exited)` for the
/// calling thread, which jumps to k_idle.
#[unsafe(no_mangle)]
pub unsafe fn handle_exit(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
    from_user: bool,
) -> ! {
    update_thread_and_trap_frame(epc, hart_context, frame, from_user);

    let exit_code = frame.regs[11] as i32;
    let cur = hart_context.current.load(Ordering::Acquire);
    if !cur.is_null() {
        let t = unsafe { (cur as *const Thread).as_ref_unchecked() };
        let pid = t.pid;
        let leader_tid = t.tid;
        let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
        // Brief manager spin — exit is not a hot path. Holding the
        // lock here keeps `assign_threads` from picking up a sibling
        // we haven't marked Exited yet.
        while !try_acquire_manager() {
            core::hint::spin_loop();
        }
        orbit.request_exit_group(pid, leader_tid, exit_code);
        release_manager();
    }

    unsafe {
        kernel::context::exit_thread_with_state(ThreadState::Exited);
    }
}

pub fn handle_serial_print(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::serial_print(t, f, &mut crate::hw::RiscvHardware)
    });
}

pub fn handle_console_write(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::console_write(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// Shared dispatch shim for syscall handlers whose body lives in
/// `orbit_core::syscall`. Resolves the current thread, invokes `body`,
/// then delegates frame/pc commit to
/// [`orbit_core::apply_syscall_outcome`] so kmain and the host tests
/// share one implementation of the outcome contract.
fn dispatch_syscall<F>(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
    body: F,
) where
    F: FnOnce(&mut Thread, &mut TrapFrame) -> orbit_core::SyscallOutcome,
{
    // Snapshot the syscall number + start tick *before* `body` runs,
    // since the handler clobbers `regs[10]` with its return value.
    // These are consumed by `record_syscall` below and feed the
    // global per-syscall histogram. The bracket lives here rather
    // than at the trap-loop call site so that blocking syscalls
    // (which long-jump out of `apply_syscall_outcome` via
    // `exit_thread_with_state`) get recorded too — see the
    // `ShimAction::Yield` arm below.
    let syscall_no = frame.regs[10];
    let syscall_start_ticks = riscv::register::time::read64();

    unsafe {
        let current = hart_context.current.load(Ordering::Acquire);
        if current == null_mut() {
            frame.regs[10] = (-1 as isize) as usize;
            return;
        }
        let thread = (current as *mut Thread).as_mut_unchecked();

        // U-mode-only path: cause=8 traps are by definition from U-mode,
        // so `current` should be a User thread. If it's a kthread,
        // dispatch_syscall would happily overwrite its pc with epc+4 in
        // apply_syscall_outcome and we'd later sret to a user VA in
        // S-mode — exactly the cause=12 panic we hit before. Refuse to
        // commit, dump every adjacent hart's `current` so we can
        // correlate (the actual user thread is presumably assigned
        // somewhere else right now), and fall through. The
        // `apply_syscall_outcome` gate is also a no-op on a kthread, so
        // even if we did reach it nothing would corrupt — we just want
        // the diagnostic out.
        if thread.mode != riscv::register::sstatus::SPP::User {
            // Walk the HartContext array so we can print every hart's
            // current ptr alongside the offending one. This is the
            // diagnostic that pins down where the racy User thread
            // actually landed.
            let hart_root = {
                (riscv::register::sscratch::read() as *const HartContext)
                    .sub(hart_context.hart_id as usize)
            };
            tracing::error!(
                "dispatch_syscall mode mismatch — cpu{} epc={:#x} a0={:#x} \
                 cur=tid={} pid={} mode={:?} state={} thread.pc={:#x} \
                 last_wake_reason={:#x}",
                hart_context.hart_id,
                epc,
                frame.regs[10],
                thread.tid,
                thread.pid,
                thread.mode,
                thread.state.load(Ordering::Acquire),
                thread.pc.load(Ordering::Acquire),
                thread.last_wake_reason.load(Ordering::Acquire),
            );
            // Hart count isn't readily available here (Orbit owns it
            // via cscratch); 4 covers the QEMU virt config we ship.
            for i in 0..(CPU_COUNT.load(Ordering::Relaxed)) {
                let hc = hart_root.add(i).as_ref_unchecked();
                let cur = hc.current.load(Ordering::Acquire) as *mut Thread;
                if cur.is_null() {
                    tracing::error!("  cpu{}: cur=<null>", i);
                }
                else {
                    let t = cur.as_ref_unchecked();
                    tracing::error!(
                        "  cpu{}: cur=tid={} pid={} mode={:?} state={} pc={:#x}",
                        i,
                        t.tid,
                        t.pid,
                        t.mode,
                        t.state.load(Ordering::Acquire),
                        t.pc.load(Ordering::Acquire),
                    );
                }
            }
            // Halt now: returning from here lets the trap path call
            // check_context_and_switch on the wrong thread, which
            // transitions knet Running → Ready and the next manager
            // pass re-dispatches it from its corrupted frame. We saw
            // that cascade produce `net: pkg@<garbage>` / `net: no phy`
            // in arm_hart_timer_*.log. Halting preserves the dump
            // above as the last forensic state.
            panic!(
                "dispatch_syscall on cpu{}: cur tid={} mode={:?} (expected User) — \
                 a kthread is current during a U-mode ecall (epc={:#x})",
                hart_context.hart_id, thread.tid, thread.mode, epc,
            );
        }

        let outcome = body(thread, frame);
        let action = orbit_core::apply_syscall_outcome(outcome, thread, frame, epc);
        // Bracket close: record on *both* arms before `Yield` long-
        // jumps. Service-time semantics still hold — `apply_syscall_
        // outcome` has finished its work (frame committed, state
        // transition decided) by this point, so `now() - start_ticks`
        // is the time the kernel spent before either resuming the
        // caller or parking it.
        crate::kernel::accounting::record_syscall(syscall_no, thread, syscall_start_ticks);
        match action {
            orbit_core::ShimAction::Resume => {}
            orbit_core::ShimAction::Yield(state) => exit_thread_with_state(state),
        }
    }
}

pub fn handle_ms_sleep(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::ms_sleep(t, f.regs[11], &crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_mmap_req(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::mmap_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_read_stdin(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::read_stdin(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_read_key_event(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::read_key_event(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_close_req(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::close_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_nc_create_req(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::nc_create_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `nc_yield(timeout_ms)` — push a kernel `WakeEvent::Net` so k_net
/// processes whatever the caller just queued in a NetCh ring, then
/// optionally park the caller for up to `timeout_ms` (capped via
/// `ms_sleep`'s existing one-hour ceiling). The park returns early
/// when the manager bumps this thread's `wake_override` (e.g. via
/// `update_tcp`'s `outcome.ring_progress` writing
/// `WakeEvent::Pid(self.pid)` after staging a fresh slice). With
/// phases 2 + 3 in place, this is the syscall that closes the
/// "user-side `sleep_ms(10)` floor" — request/response round trips
/// drop from ~20 ms (one timer tick on each side) to scheduler
/// dispatch latency.
///
/// `timeout_ms == 0`: pure notification, no park; returns
/// immediately. Useful as a "fire and forget" wake from a path that
/// doesn't itself want to sleep.
#[unsafe(no_mangle)]
pub fn handle_nc_yield(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let _ = crate::kernel::wake_queue_push(crate::kernel::WakeEvent::Net);
        let timeout_ms = f.regs[11];
        if timeout_ms == 0 {
            orbit_core::SyscallOutcome::Return { ret: 0 }
        }
        else {
            orbit_core::syscall::ms_sleep(t, timeout_ms, &crate::hw::RiscvHardware)
        }
    });
}

#[unsafe(no_mangle)]
pub fn handle_create_process_req(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::create_process_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `pledge(*const PermsRequest)`. Async manager round-trip: the
/// syscall layer parks the caller, the manager copies the request
/// struct from user memory, applies the narrowing to
/// `Process.permissions`, and propagates to every live thread's
/// snapshot before signaling. Wired into `s_trap` at syscall
/// number `9`.
#[unsafe(no_mangle)]
pub fn handle_pledge(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::pledge_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `create_process_v2(*const CreateProcessV2Args)`. Role-aware
/// spawn: the manager runs the role-transition gate; on failure it
/// logs a `DenialEvent::RoleDeny` audit event, bumps the parent's
/// `role_denials` counter, and returns `-EPERM`. On success the
/// witness-derived perms are installed on the freshly-spawned
/// child. Wired into `s_trap` at syscall number `4105`.
#[unsafe(no_mangle)]
pub fn handle_create_process_v2(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::create_process_v2_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `query_denial_log(buf_ptr, buf_len) → bytes | -errno`. Synchronous
/// read of the kernel-wide denial event ring into a user buffer.
/// Acquires `MANAGER_LOCK` briefly for the snapshot — events are
/// otherwise mutated by `drain_denial_events` under the same lock.
/// Wired into `s_trap` at syscall number `4106`.
///
/// Lives in kmain (not orbit-core) for the same reason as
/// `query_stats`: the ring is owned by `Orbit`, which `Hardware`
/// deliberately doesn't expose.
#[unsafe(no_mangle)]
pub fn handle_query_denial_log(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    use orbit_abi::denial::{DENIAL_RING_CAPACITY, DenialEvent};
    use orbit_abi::errno::{EFAULT, EINVAL};
    use orbit_abi::layout::user_range_ok;

    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };

    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let Ok(buf_va) = UserVa::new(f.regs[11] as u64)
        else {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        };
        let buf_len = f.regs[12];

        let event_size = core::mem::size_of::<DenialEvent>();
        if buf_len < event_size {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EINVAL as isize),
            };
        }
        if !user_range_ok(buf_va.raw(), buf_len as u64) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }
        use orbit_core::Hardware;
        if !crate::hw::RiscvHardware.user_va_translates(t.root_table_addr(), buf_va) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }

        // Snapshot under the manager lock — short critical section,
        // bounded by DENIAL_RING_CAPACITY events. Drain the producer
        // queue first so any PermDeny the dispatch gate has pushed
        // since the last manager pass lands in the ring before we
        // snapshot. Without this drain, a caller racing the manager
        // can read the ring before its own gate-induced PermDeny
        // gets folded in — the lock-free queue lets the gate-side
        // push race the read. Doesn't eliminate the race (a new
        // event from another hart between drain and snapshot still
        // beats us) but makes "any denial I caused is visible" hold
        // for the calling thread.
        let mut tmp = [DenialEvent::PermDeny {
            required_class: 0,
            perms: 0,
            time_ticks: 0,
            tid: 0,
            sysno: 0,
            source_role: 0,
            pid: 0,
        }; DENIAL_RING_CAPACITY];
        while !try_acquire_manager() {
            core::hint::spin_loop();
        }
        orbit.drain_denial_events();
        let n = orbit.denial_ring_snapshot(&mut tmp);
        release_manager();

        let max_events = buf_len / event_size;
        let to_emit = core::cmp::min(n, max_events);
        let to_write = to_emit * event_size;

        let guard = UserAccess::enter();
        unsafe {
            let dst = guard.slice_mut(buf_va, to_write);
            let src = core::slice::from_raw_parts(tmp.as_ptr() as *const u8, to_write);
            dst.copy_from_slice(src);
        }
        drop(guard);

        orbit_core::SyscallOutcome::Return {
            ret: to_write as isize,
        }
    });
}

/// §13a.3 — `create_process_ex(elf, argv_blob)`. Same async shape as
/// `create_process` but carries an argv blob the kernel maps into
/// the new process at `USER_ARGV_BASE`.
#[unsafe(no_mangle)]
pub fn handle_create_process_ex(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::create_process_ex_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// §13a.3 / §13e — `argv_envp() → (argv_va, envp_va)`. Synchronous
/// read off the caller's `Process.argv_blob` / `envp_blob` slots; a
/// `0` in either slot means "not installed" — orbit-rt treats those
/// as empty `argv` / `envp`.
#[unsafe(no_mangle)]
pub fn handle_argv_envp(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        let argv_va = if orbit.process_has_argv(t.pid) {
            orbit_abi::layout::USER_ARGV_BASE as isize
        }
        else {
            0
        };
        let envp_va = if orbit.process_has_envp(t.pid) {
            orbit_abi::layout::USER_ENVP_BASE as isize
        }
        else {
            0
        };
        orbit_core::SyscallOutcome::Return2 {
            ret0: argv_va,
            ret1: envp_va,
        }
    });
}

#[unsafe(no_mangle)]
pub fn handle_create_thread(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::create_thread(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_fs_open(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::fs_open_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_fs_read(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::fs_read_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_fs_stat(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::fs_stat_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_fs_readdir(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::fs_readdir_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

#[unsafe(no_mangle)]
pub fn handle_set_affinity(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::set_affinity(t, f)
    });
}

#[unsafe(no_mangle)]
pub fn handle_get_affinity(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::syscall::get_affinity(t)
    });
}

/// `wait_pid(target_pid) → exit_code | -errno`. Async via the
/// manager — the caller parks on a `CompletionHandle` that gets
/// signaled either by `run_wait_pid_req` (sync error path) or by
/// `dealloc_process` when the target exits.
#[unsafe(no_mangle)]
pub fn handle_wait_pid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::wait_pid_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// §13a.5 — `futex_wait(uaddr, expected, timeout_ns) → 0 | -EAGAIN
/// | -ETIMEDOUT | -E*`. Park on the per-PA queue; wake by
/// `futex_wake` or sync error.
#[unsafe(no_mangle)]
pub fn handle_futex_wait(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::futex_wait_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// §13a.5 — `futex_wake(uaddr, n) → n_woken | -E*`. Drain up to `n`
/// waiters from the per-PA queue and signal each with `0`.
#[unsafe(no_mangle)]
pub fn handle_futex_wake(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::futex_wake_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `fs_fstat(fd, &mut Stat) → 0 | -errno`. Sync — looks up the
/// process's `OpenFile`, runs `Filesystem::stat`, copies into the
/// user buffer.
#[unsafe(no_mangle)]
pub fn handle_fs_fstat(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let fd = f.regs[11] as u32;
        let stat_va = f.regs[12] as u64;
        let ret = orbit.run_fs_fstat(t.pid, t.root_table_addr(), fd, stat_va);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `fs_seek(fd, offset, whence) → new_offset | -errno`. Sync — only
/// touches the per-fd `OpenFile.offset`, no DMA / manager work.
#[unsafe(no_mangle)]
pub fn handle_fs_seek(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let fd = f.regs[11] as u32;
        let offset = f.regs[12] as i64;
        let whence = f.regs[13] as u32;
        let ret = orbit.run_fs_seek(t.pid, fd, offset, whence);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `chdir(path_ptr, path_len) → 0 | -errno`. Sync handler — mutates
/// the calling process's cwd in place after the kernel-side fs lookup
/// confirms the target dir exists. Body lives on `Orbit` so it can
/// reach into `self.processes`.
#[unsafe(no_mangle)]
pub fn handle_chdir(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let ret = orbit.run_chdir(t.pid, t.root_table_addr(), f.regs[11] as u64, f.regs[12]);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `getcwd(buf_ptr, buf_len) → bytes | -errno`. Sync handler — copies
/// the calling process's cwd into the user buffer. Caller passes a
/// page-resident buffer at least `cwd.len()` bytes long; ERANGE if the
/// buffer is too short.
#[unsafe(no_mangle)]
pub fn handle_getcwd(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let ret = orbit.run_getcwd(t.pid, t.root_table_addr(), f.regs[11] as u64, f.regs[12]);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `getpid() → u16` — pid of the calling process. Stable for the
/// process's lifetime; reads `thread.pid` directly. No manager
/// round-trip, no blocking — same shape as `get_hart_id`.
#[unsafe(no_mangle)]
pub fn handle_getpid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: t.pid as isize,
        }
    });
}

/// `gettid() → u32` — tid of the calling thread. System-wide unique
/// (not per-process); reads `thread.tid` directly. Trivially safe.
#[unsafe(no_mangle)]
pub fn handle_gettid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: t.tid as isize,
        }
    });
}

/// `getuid() → uid` — POSIX `getuid(2)`. Reads the per-thread
/// credential snapshot installed by `create_new_thread` from the
/// owning process's `Process.uid`. Same fast-path shape as
/// `handle_getpid` — no manager round-trip, no MANAGER_LOCK.
#[unsafe(no_mangle)]
pub fn handle_getuid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: t.uid as isize,
        }
    });
}

/// `geteuid() → euid` — POSIX `geteuid(2)`. Reads `Thread.euid`.
#[unsafe(no_mangle)]
pub fn handle_geteuid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: t.euid as isize,
        }
    });
}

/// `getgid() → gid` — POSIX `getgid(2)`. Reads `Thread.gid`.
#[unsafe(no_mangle)]
pub fn handle_getgid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: t.gid as isize,
        }
    });
}

/// `getegid() → egid` — POSIX `getegid(2)`. Reads `Thread.egid`.
#[unsafe(no_mangle)]
pub fn handle_getegid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: t.egid as isize,
        }
    });
}

/// `getgroups(buf_ptr, count) → count | -errno` — POSIX
/// `getgroups(2)`. `count` is in `u32` slots, not bytes. POSIX
/// special case: `count == 0` returns the current group count
/// without writing.
#[unsafe(no_mangle)]
pub fn handle_getgroups(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let ret = orbit.run_getgroups(t.pid, t.root_table_addr(), f.regs[11] as u64, f.regs[12]);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `getlogin(buf_ptr, buf_len) → bytes | -errno` — POSIX
/// `getlogin_r(3)`. Copies the calling process's session login name
/// (no NUL terminator) into the user buffer.
#[unsafe(no_mangle)]
pub fn handle_getlogin(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let ret = orbit.run_getlogin(t.pid, t.root_table_addr(), f.regs[11] as u64, f.regs[12]);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `setuid(uid) → 0 | -errno` — POSIX `setuid(2)`. Sync handler:
/// mutates the calling process's uid triplet under POSIX rules and
/// refreshes per-thread credential snapshots so subsequent
/// `getuid`/`geteuid` from sibling threads see the new identity.
#[unsafe(no_mangle)]
pub fn handle_setuid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let uid = f.regs[11] as u32;
        let ret = orbit.run_setuid(t.pid, uid);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `setgid(gid) → 0 | -errno` — POSIX `setgid(2)`. Sync gid mirror
/// of [`handle_setuid`].
#[unsafe(no_mangle)]
pub fn handle_setgid(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let gid = f.regs[11] as u32;
        let ret = orbit.run_setgid(t.pid, gid);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `setgroups(buf_ptr, count) → 0 | -errno` — POSIX `setgroups(2)`.
/// Replace the caller's supplementary group list. Requires
/// `euid == 0`.
#[unsafe(no_mangle)]
pub fn handle_setgroups(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let ret = orbit.run_setgroups(t.pid, t.root_table_addr(), f.regs[11] as u64, f.regs[12]);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `setlogin(name_ptr, name_len) → 0 | -errno` — POSIX `setlogin(2)`.
/// Stamp the calling process's session login name. Requires
/// `euid == 0`.
#[unsafe(no_mangle)]
pub fn handle_setlogin(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let ret = orbit.run_setlogin(t.pid, t.root_table_addr(), f.regs[11] as u64, f.regs[12]);
        orbit_core::SyscallOutcome::Return { ret }
    });
}

/// `get_hart_id()` — return the hart id this syscall is running on.
/// Lives in kmain (not orbit-core) because the source of truth is the
/// per-hart `HartContext`, which `Hardware` deliberately doesn't expose
/// — letting the orbit-core handlers depend on it would couple them to
/// machine-mode device state. Trivially safe: no thread mutation, no
/// blocking, no manager work.
#[unsafe(no_mangle)]
pub fn handle_get_hart_id(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let hart_id = hart_context.hart_id;
    dispatch_syscall(epc, hart_context, frame, |_t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: hart_id as isize,
        }
    });
}

/// `get_micros()` — absolute monotonic microseconds since boot.
///
/// Backed by the RISC-V `time` CSR (10 MHz on QEMU virt; 10 ticks/μs).
/// The CSR is unprivileged on M-mode but gated behind `scounteren.TM`
/// for U-mode — keeping the read in the kernel sidesteps that gate
/// and gives userspace a stable abstraction over whatever clock
/// source future platforms use.
#[unsafe(no_mangle)]
pub fn handle_get_micros(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let now_ticks = riscv::register::time::read64();
    // 10 MHz → divide by 10 for microseconds. The division is by a
    // const so the compiler turns it into a multiply-shift.
    let micros = now_ticks / 10;
    dispatch_syscall(epc, hart_context, frame, |_t, _f| {
        orbit_core::SyscallOutcome::Return {
            ret: micros as isize,
        }
    });
}

/// `get_realtime()` — wall-clock seconds + nanoseconds since the UNIX
/// epoch. Two-register return: secs in `a0`, nsec in `a1`.
///
/// Backed by the Goldfish RTC at PA `0x101000` on QEMU virt
/// (driver: [`crate::drivers::goldfish_rtc`]). The device returns
/// nanoseconds since the epoch as a 64-bit value; we split into
/// `(secs, nsec)` here so the user side can build a `(secs, nsec)`
/// `SystemTime` without a follow-up divmod. Synchronous; no manager
/// round-trip.
#[unsafe(no_mangle)]
pub fn handle_get_realtime(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    let nanos = crate::drivers::goldfish_rtc::now_nanos();
    let secs = (nanos / 1_000_000_000) as isize;
    let nsec = (nanos % 1_000_000_000) as isize;
    dispatch_syscall(epc, hart_context, frame, |_t, _f| {
        orbit_core::SyscallOutcome::Return2 {
            ret0: secs,
            ret1: nsec,
        }
    });
}

/// `query_stats(buf_ptr, buf_len)` — copy a [`ProcessStats`] snapshot
/// of the calling process into the user buffer. Returns bytes
/// written, or a negative errno.
///
/// Lives in kmain for the same reason as `get_hart_id`: the data
/// source is the kernel-side [`Orbit`] struct (process table, frame
/// allocator stats), which `Hardware` deliberately doesn't surface.
/// Synchronous — no manager round-trip; we acquire `MANAGER_LOCK`
/// inline for the duration of the snapshot walk.
///
/// [`ProcessStats`]: orbit_abi::stats::ProcessStats
#[unsafe(no_mangle)]
pub fn handle_query_stats(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    use orbit_abi::errno::{EFAULT, EINVAL, ESRCH};
    use orbit_abi::layout::user_range_ok;
    use orbit_abi::stats::{ProcessStats, STATS_MIN_LEN};

    let orbit = unsafe { (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked() };

    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let Ok(buf_va) = UserVa::new(f.regs[11] as u64)
        else {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        };
        let buf_len = f.regs[12];

        if buf_len < STATS_MIN_LEN {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EINVAL as isize),
            };
        }
        // Saved-feedback rule: bound-check user VAs at the syscall
        // boundary before the kernel acts on them.
        if !user_range_ok(buf_va.raw(), buf_len as u64) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }
        // ProcessStats fits in a page; walk only the start (matches
        // serial_print/console_write convention). user_range_ok
        // already excluded the kernel half and overflow.
        use orbit_core::Hardware;
        if !crate::hw::RiscvHardware.user_va_translates(t.root_table_addr(), buf_va) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }

        // Spin on MANAGER_LOCK so the heap_pages / maps walk doesn't
        // race the manager mutating them. Brief — manager passes are
        // <1 ms.
        while !try_acquire_manager() {
            core::hint::spin_loop();
        }
        let snapshot = orbit.snapshot_process_stats(t.pid);
        release_manager();

        let stats = match snapshot {
            Some(s) => s,
            None => {
                return orbit_core::SyscallOutcome::Return {
                    ret: -(ESRCH as isize),
                };
            }
        };

        let native = core::mem::size_of::<ProcessStats>();
        let to_write = core::cmp::min(native, buf_len);

        // SUM gate the write. `user_va_translates` confirmed the start
        // page; if `buf_len` straddles a page boundary into an
        // unmapped follow-on the store faults — same convention as
        // serial_print's PAGE_SIZE-bounded copy.
        let guard = UserAccess::enter();
        unsafe {
            let dst = guard.slice_mut(buf_va, to_write);
            let src =
                core::slice::from_raw_parts(&stats as *const ProcessStats as *const u8, to_write);
            dst.copy_from_slice(src);
        }
        drop(guard);

        orbit_core::SyscallOutcome::Return {
            ret: to_write as isize,
        }
    });
}

/// `query_syscall_stats(buf_ptr, buf_len)` — copy the system-wide
/// per-syscall latency table into the user buffer. Layout matches
/// [`SyscallStatsHeader`] + N × [`SyscallEntry`] where N is the
/// kernel's `Sysno::COUNT`. Returns bytes written, or a negative errno.
///
/// [`SyscallStatsHeader`]: orbit_abi::syscall_stats::SyscallStatsHeader
/// [`SyscallEntry`]: orbit_abi::syscall_stats::SyscallEntry
#[unsafe(no_mangle)]
pub fn handle_query_syscall_stats(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    use core::sync::atomic::Ordering;
    use orbit_abi::Sysno;
    use orbit_abi::errno::{EFAULT, EINVAL};
    use orbit_abi::layout::user_range_ok;
    use orbit_abi::syscall_stats::{SYSCALL_STATS_MIN_LEN, SyscallEntry, SyscallStatsHeader};

    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let Ok(buf_va) = UserVa::new(f.regs[11] as u64)
        else {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        };
        let buf_len = f.regs[12];

        if buf_len < SYSCALL_STATS_MIN_LEN {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EINVAL as isize),
            };
        }
        if !user_range_ok(buf_va.raw(), buf_len as u64) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }
        use orbit_core::Hardware;
        if !crate::hw::RiscvHardware.user_va_translates(t.root_table_addr(), buf_va) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }

        // Native payload size = header + COUNT entries. Caller's
        // buffer may be smaller (older userland with smaller COUNT)
        // or larger (newer); we honor whichever is smaller and rely
        // on the header.count field to tell the reader how many
        // entries are valid.
        let hdr_size = core::mem::size_of::<SyscallStatsHeader>();
        let entry_size = core::mem::size_of::<SyscallEntry>();
        let native = hdr_size + Sysno::COUNT * entry_size;
        let to_write = core::cmp::min(native, buf_len);
        // How many full entries fit after the header? (Truncate the
        // tail rather than write a partial entry.)
        let entries_capacity = if to_write >= hdr_size {
            (to_write - hdr_size) / entry_size
        }
        else {
            0
        };
        let entries_to_write = core::cmp::min(entries_capacity, Sysno::COUNT);

        let guard = UserAccess::enter();
        unsafe {
            // Header.
            let hdr = SyscallStatsHeader {
                size: (hdr_size + entries_to_write * entry_size) as u32,
                count: entries_to_write as u32,
            };
            let hdr_dst = guard.slice_mut(buf_va, hdr_size);
            hdr_dst.copy_from_slice(core::slice::from_raw_parts(
                &hdr as *const _ as *const u8,
                hdr_size,
            ));
            // Entries: read each slot and write it directly. Using a
            // stack scratch keeps the SUM window tight.
            for i in 0..entries_to_write {
                let slot = &crate::kernel::accounting::SYSCALL_STATS[i];
                let entry = SyscallEntry {
                    count: slot.count.load(Ordering::Relaxed),
                    total_ticks: slot.total_ticks.load(Ordering::Relaxed),
                    max_ticks: slot.max_ticks.load(Ordering::Relaxed),
                };
                let dst = guard.slice_mut(
                    buf_va.wrapping_add((hdr_size + i * entry_size) as u64),
                    entry_size,
                );
                dst.copy_from_slice(core::slice::from_raw_parts(
                    &entry as *const _ as *const u8,
                    entry_size,
                ));
            }
        }
        drop(guard);

        let written = hdr_size + entries_to_write * entry_size;
        orbit_core::SyscallOutcome::Return {
            ret: written as isize,
        }
    });
}

/// `fb_query(&mut FbInfo) -> 0 | -errno` — sync. Read the active
/// framebuffer dims off `k_gpu`'s installed package and copy a `FbInfo`
/// payload into the user buffer. Returns `EAGAIN` until k_gpu has
/// installed its package at boot — every legitimate caller hits this
/// long after init, so the race window is theoretical.
#[unsafe(no_mangle)]
pub fn handle_fb_query(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    use orbit_abi::errno::{EAGAIN, EFAULT, EINVAL};
    use orbit_abi::fb::{FbFormat, FbInfo};
    use orbit_abi::layout::user_range_ok;

    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let Ok(buf_va) = UserVa::new(f.regs[11] as u64)
        else {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        };
        let buf_len = core::mem::size_of::<FbInfo>();

        if !user_range_ok(buf_va.raw(), buf_len as u64) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }
        // FbInfo is 16 bytes — fits in one page comfortably; reject
        // straddling so the SUM-gated copy below can be a single
        // bounded write.
        if (buf_va.raw() as usize & (mmu::PAGE_SIZE - 1)) + buf_len > mmu::PAGE_SIZE {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EINVAL as isize),
            };
        }
        // Verify the start page is mapped under the caller's satp.
        // The straddle check above + this single resolve cover the
        // whole copy.
        use orbit_core::Hardware;
        if !crate::hw::RiscvHardware.user_va_translates(t.root_table_addr(), buf_va) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EFAULT as isize),
            };
        }

        let (w, h) = match crate::drivers::k_gpu::fb_size() {
            Some(d) => d,
            None => {
                return orbit_core::SyscallOutcome::Return {
                    ret: -(EAGAIN as isize),
                };
            }
        };
        let info = FbInfo {
            width: w,
            height: h,
            format: FbFormat::Bgra8888 as u32,
            flags: 0,
        };

        let guard = UserAccess::enter();
        unsafe {
            let dst = guard.slice_mut(buf_va, buf_len);
            let src = core::slice::from_raw_parts(&info as *const FbInfo as *const u8, buf_len);
            dst.copy_from_slice(src);
        }
        drop(guard);

        orbit_core::SyscallOutcome::Return { ret: 0 }
    });
}

/// `fb_surface_create(w, h, format) -> (handle, user_va) | -errno` —
/// async, manager-handled. Pure forwarder: orbit-core builds the
/// `PendingWork::FbSurfaceCreate` and parks the caller; the manager
/// runs the alloc + map + register and signals back via `signal_pair`.
#[unsafe(no_mangle)]
pub fn handle_fb_surface_create(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::fb_surface_create_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `fb_surface_destroy(handle) -> 0 | -errno` — async, manager-handled.
/// Same forwarder shape as create.
#[unsafe(no_mangle)]
pub fn handle_fb_surface_destroy(
    epc: usize,
    hart_context: &'static HartContext,
    frame: &mut TrapFrame,
) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::fb_surface_destroy_req(t, f, &mut crate::hw::RiscvHardware)
    });
}

/// `fb_present(handle, x, y, w, h) -> 0 | -errno` — sync. Look up the
/// caller's surface table entry, validate the rect, and push a
/// `Cmd::PresentSurface` carrying the snapshot onto k_gpu's ring.
/// `k_gpu` does the per-row blit + virtio-gpu transfer on its next
/// drain.
#[unsafe(no_mangle)]
pub fn handle_fb_present(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    use crate::drivers::display::Source;
    use crate::drivers::k_gpu::{PresentArgs, push_present};
    use orbit_abi::errno::{EAGAIN, EBADF, EINVAL};

    dispatch_syscall(epc, hart_context, frame, |t, f| {
        let handle = f.regs[11] as u32;
        let rect_x = f.regs[12] as u32;
        let rect_y = f.regs[13] as u32;
        let rect_w = f.regs[14] as u32;
        let rect_h = f.regs[15] as u32;

        if rect_w == 0 || rect_h == 0 {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EINVAL as isize),
            };
        }

        let Some(surfaces) = crate::kernel::surface::get(t.pid)
        else {
            // No surface table for this pid → certainly no handle.
            return orbit_core::SyscallOutcome::Return {
                ret: -(EBADF as isize),
            };
        };

        let Some(snapshot) = surfaces.snapshot(handle)
        else {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EBADF as isize),
            };
        };

        // Reject rects that extend past the surface dims. Saturating
        // adds catch the overflow case — a malicious caller passing
        // u32::MAX in `w` would otherwise wrap.
        let x_end = rect_x.saturating_add(rect_w);
        let y_end = rect_y.saturating_add(rect_h);
        if x_end > snapshot.width || y_end > snapshot.height {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EINVAL as isize),
            };
        }

        let args = PresentArgs {
            kdmap_kva: snapshot.kdmap_kva,
            width: snapshot.width,
            height: snapshot.height,
            rect_x,
            rect_y,
            rect_w,
            rect_h,
            format_raw: snapshot.format as u32,
        };

        if !push_present(Source::Process(t.pid), args) {
            return orbit_core::SyscallOutcome::Return {
                ret: -(EAGAIN as isize),
            };
        }

        orbit_core::SyscallOutcome::Return { ret: 0 }
    });
}

pub struct SerialWriter {
    buf: [u8; crate::drivers::k_serial::CHUNK_BYTES],
    len: usize,
}

impl SerialWriter {
    pub const fn new() -> Self {
        Self {
            buf: [0u8; crate::drivers::k_serial::CHUNK_BYTES],
            len: 0,
        }
    }
    pub fn flush(&mut self) {
        if self.len == 0 {
            return;
        }
        crate::drivers::k_serial::push_chunk(&self.buf[..self.len]);
        self.len = 0;
    }
}

impl core::fmt::Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len >= self.buf.len() {
                self.flush();
            }
            self.buf[self.len] = b;
            self.len += 1;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! serialln {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let mut w = $crate::SerialWriter::new();
        let _ = writeln!(w, $($arg)*);
        w.flush();
    }};
}
