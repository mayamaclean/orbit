#![no_std]

extern crate alloc;

use core::{arch::asm, ptr::null_mut, sync::atomic::{AtomicBool, Ordering}};
use alloc::{collections::btree_map::BTreeMap, vec::Vec};
use device::{HartContext, TrapFrame};
use net_channel::NetChannel;
use crate::kernel::shared_user_ptr::SharedUserPtr;
use process::{Thread, ThreadState};
use smoltcp::{iface::{PollResult, SocketHandle, SocketSet}, socket::dhcpv4, storage::RingBuffer};

use crate::{drivers::e1000::E1000, kernel::context::{enter_hart_context, exit_thread_with_state, get_hart_context, hart_has_thread}};

pub mod channel;
pub mod drivers;
pub mod hw;
pub mod kernel;
pub mod ktrace;

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
        unsafe { riscv::register::sstatus::set_sum(); }
        Self { _private: () }
    }

    /// Borrow a read-only byte slice at a user VA. Caller must have
    /// verified (via PT walk) that the range is mapped and user-readable.
    /// Lifetime ties the slice to this guard so it can't outlive SUM.
    #[inline]
    pub unsafe fn slice<'s>(&'s self, vaddr: u64, len: usize) -> &'s [u8] {
        unsafe { core::slice::from_raw_parts(vaddr as *const u8, len) }
    }

    /// Borrow a writable byte slice at a user VA. Caller must have
    /// verified the range is mapped user-writable. Lifetime ties the
    /// slice to this guard so it can't outlive SUM.
    #[inline]
    pub unsafe fn slice_mut<'s>(&'s self, vaddr: u64, len: usize) -> &'s mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(vaddr as *mut u8, len) }
    }

    /// Read a value of type `T` from a user VA. Caller must have verified
    /// the source is mapped and the read is size/alignment-safe.
    #[inline]
    pub unsafe fn read_volatile<T>(&self, vaddr: u64) -> T {
        unsafe { core::ptr::read_volatile(vaddr as *const T) }
    }
}

impl Drop for UserAccess {
    fn drop(&mut self) {
        unsafe { riscv::register::sstatus::clear_sum(); }
    }
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
    //serial::println!("hart{} sending wake ipi to hart{hart}", get_hart_context().hart_id);
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
    wait_until(
        riscv::register::time::read64()
            .wrapping_add(target)
    );
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

#[unsafe(no_mangle)]
pub extern "C" fn k_hart_loop() -> ! {
    let hart_context = unsafe {
        (riscv::register::sscratch::read() as *const HartContext).as_ref_unchecked()
    };

    let orbit = unsafe {
        (hart_context.cscratch as *mut kernel::Orbit).as_mut_unchecked()
    };

    loop {
        if hart_has_thread(hart_context) {
            setup_hart_timer(1_000_000);
            unsafe { enter_hart_context(hart_context); }
        }

        // Disable sstatus.SIE around the acquire + critical section. If a
        // trap fired mid-section the handler would long-jump via kptr back
        // to k_hart_loop without releasing MANAGER_LOCK, deadlocking all
        // harts. setup_hart_timer restores SIE on the way out.
        unsafe { riscv::register::sstatus::clear_sie(); }
        if try_acquire_manager() {
            orbit.cleanup_threads_and_processes();
            orbit.drain_pending_work();
            orbit.check_net();
            orbit.assign_threads(hart_context);
            release_manager();

            if hart_has_thread(hart_context) {
                setup_hart_timer(1_000_000);
                unsafe { enter_hart_context(hart_context); }
            }
        }

        unsafe {
            riscv::register::sie::set_ssoft();
            setup_hart_timer(100_000);
            riscv::asm::wfi();
        }
    }
}


#[derive(Debug, Clone)]
pub struct SocketReq {
    /// Refcounted handle on the NetChannel. Cloned from the registry when
    /// the manager enqueues the request; k_net drops its clone when the
    /// socket goes through `socket_deletions`.
    netchan: SharedUserPtr<NetChannel>,
    nc_type: usize,
    pid: u16,
    /// "A slice is enqueued on rx.slices and we haven't yet drained the
    /// matching increment." Set when `update_tcp` enqueues an rx slice,
    /// cleared when it drains an increment. Gates re-enqueue so we don't
    /// race with the user's dequeue→f→increment sequence and deposit a
    /// duplicate slice pointing at bytes smoltcp hasn't been told are
    /// consumed yet.
    pending_rx_ack: bool,
    /// Same invariant on the tx side: set on enqueue of a tx slice,
    /// cleared when we drain the user's send-ack increment.
    pending_tx_ack: bool,
    /// Last `desired_state.state` value the kernel issued a connect /
    /// listen for. Keeps `update_tcp` level-triggered: if the user's
    /// intent hasn't changed since we acted on it, we don't re-call
    /// `socket.connect` / `socket.listen` just because smoltcp
    /// transitioned back to CLOSED (e.g. RST from a peer with no
    /// listener). `0` means "no intent pending" — matches the reset
    /// path, so the existing idle sentinel doubles as the reset
    /// marker.
    issued_desired: i32,
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
    socket_deletions: heapless::spsc::Queue<SocketHandle, 8>
}

fn set_ipv4_addr(iface: &mut smoltcp::iface::Interface, cidr: smoltcp::wire::Ipv4Cidr) {
    iface.update_ip_addrs(|addrs| {
        addrs.clear();
        addrs.push(smoltcp::wire::IpCidr::Ipv4(cidr)).unwrap();
    });
}

fn handle_dhcp_event(mut iface: smoltcp::iface::Interface, event: dhcpv4::Event) -> smoltcp::iface::Interface {
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
    use tracing::{info, error};

    unsafe {
        riscv::register::sstatus::clear_sie();
    }

    info!("net: pkg@{device:016X?}");

    let net_package = unsafe { device.as_mut_unchecked() };

    let NetPackage { phy, iface , socket_reqs, socket_associations, socket_deletions } = net_package;
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

    let mut next_poll = 0;
    loop {
        unsafe {
            riscv::register::sstatus::clear_sie();
            riscv::register::sstatus::set_sum();
        }

        let mut now = riscv::register::time::read();
        let mut timestamp = smoltcp::time::Instant::from_micros(
            now as i64 / 10
        );

        if now >= next_poll || phy.read_interrupt_status() > 0 {
            unsafe {
                core::arch::asm!("fence iorw, iorw");
            }
            
            while iface.poll(timestamp, &mut phy, &mut sockets) != PollResult::None {
                now = riscv::register::time::read();
                timestamp = smoltcp::time::Instant::from_micros(
                    now as i64 / 10
                );

                if let Some(event) = sockets.get_mut::<dhcpv4::Socket>(dhcp_handle).poll() {
                    iface = handle_dhcp_event(iface, event);
                }
            }

            unsafe {
                core::arch::asm!("fence iorw, iorw");
            }
        }

        orbit_core::net::drain_socket_deletions(
            &mut user_conns,
            || socket_deletions.dequeue(),
            |h| { sockets.remove(h); },
        );

        orbit_core::net::prune_revoked_conns(
            &mut user_conns,
            &mut user_revocations,
            |h| { sockets.remove(h); },
        );

        for (sock_handle, req) in user_conns.iter_mut() {
            if let Some(nc) = req.netchan.try_as_ref() {
                if req.nc_type == 0 {
                    let socket = sockets.get_mut::<smoltcp::socket::tcp::Socket>(*sock_handle);
                    iface = nc.update_tcp(
                        iface,
                        socket,
                        &mut req.pending_rx_ack,
                        &mut req.pending_tx_ack,
                        &mut req.issued_desired,
                    );
                }
            }
            else {
                user_revocations.push(*sock_handle);
            }
        }

        orbit_core::net::prune_revoked_conns(
            &mut user_conns,
            &mut user_revocations,
            |h| { sockets.remove(h); },
        );
        
        let default_wake = now + 100_000;
        let wake_time = iface.poll_at(timestamp, &mut sockets)
            .map(|i| i.total_micros() as usize * 10)
            .unwrap_or(default_wake);

        next_poll = wake_time;

        for q in socket_reqs.iter_mut() {
            while let Some(req) = q.dequeue() {
                info!("net: processing req {req:?}");

                if req.nc_type == 0 {
                    let req_pid = req.pid;
                    let (txr, rxr) = req.netchan.as_ref().rings();

                    info!("net: tcp socket ring lens: rx={},tx={}", rxr.len(), txr.len());

                    let tx_buffer = RingBuffer::new(txr);
                    let rx_buffer = RingBuffer::new(rxr);

                    let sock = smoltcp::socket::tcp::Socket::new(rx_buffer, tx_buffer);
                    let handle = sockets.add(sock);

                    info!("net: created tcp socket: {handle:?}");
                    user_conns.insert(handle, req);

                    next_poll = 0;

                    if let Err(assoc) = socket_associations.enqueue((req_pid as usize, handle)) {
                        error!("net: was unable to inform manager of socket association {assoc:?}");
                    }
                }
            }
        }

        unsafe {
            let hart_context = {
                (riscv::register::sscratch::read() as *mut HartContext).as_mut_unchecked()
            };

            let this_thread = {
                let p = hart_context.current.load(Ordering::Acquire) as *mut Thread;
                //serial::println!("net thread on cpu{} t={p:016X?}", hart_context.hart_id);
                
                p.as_mut_unchecked()
            };

            hart_context.cscratch2 = 1;
            this_thread.ticks = 0;
            this_thread.wake_time = core::cmp::min(default_wake, wake_time);
            this_thread.state.store(ThreadState::Suspended as usize, Ordering::Release);

            riscv::register::sstatus::clear_sum();
            riscv::register::sstatus::set_sie();

            asm!("ebreak", "nop");

            // TODO: store registers and stuff into a trap frame and switch threads
            // -OR- *modiify supervisor ebreak into syscall type thing
            // -OR- *try to pass some supervisor ecalls from machine mode back into supervisor mode (ssoft?) 
        };
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

#[unsafe(no_mangle)]
pub fn check_context_and_switch() -> ! {
    let c = get_hart_context();
    let t = c.current.load(Ordering::Acquire);

    if t != null_mut() {
        let thread: &Thread = unsafe { (t as *mut Thread).as_ref_unchecked() };
        let thread_state = thread.state.load(Ordering::Acquire);
        if thread_state == ThreadState::Running as usize {
            unsafe { exit_thread_with_state(ThreadState::Ready); }
        }
        else if thread_state == ThreadState::Exited as usize {
            unsafe { exit_thread_with_state(ThreadState::Exited); }
        }
        else if thread_state == ThreadState::Suspended as usize {
            //serial::println!("hart{} returning suspended thread{}", c.hart_id, thread.tid);
            c.current.store(null_mut(), Ordering::Release);
        }
        else if thread_state == ThreadState::Blocking as usize {
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
    if cptr == null_mut() { return; }
    let thread: &mut Thread = unsafe { (cptr as *mut Thread).as_mut_unchecked() };
    orbit_core::trap::update_trap_frame(thread, epc, frame, from_user);
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
    unsafe {
        let current = hart_context.current.load(Ordering::Acquire);
        if current == null_mut() {
            frame.regs[10] = (-1 as isize) as usize;
            return;
        }
        let thread = (current as *mut Thread).as_mut_unchecked();

        let outcome = body(thread, frame);
        match orbit_core::apply_syscall_outcome(outcome, thread, frame, epc) {
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

#[unsafe(no_mangle)]
pub fn handle_create_process_req(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    dispatch_syscall(epc, hart_context, frame, |t, f| {
        orbit_core::syscall::create_process_req(t, f, &mut crate::hw::RiscvHardware)
    });
}
