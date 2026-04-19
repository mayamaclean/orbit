#![no_std]

extern crate alloc;

use core::{arch::asm, ptr::null_mut, sync::atomic::{AtomicBool, Ordering}};
use alloc::{collections::btree_map::BTreeMap, vec::Vec};
use device::{HartContext, TrapFrame};
use mmu::{PAGE_SIZE, sv48::PageTable};
use net_channel::NetChannel;
use process::{MemMapReq, NetChannelRegistrationReq, Thread, ThreadBlockReason, ThreadState};
use riscv::register::sstatus::SPP;
use smoltcp::{iface::{PollResult, SocketHandle, SocketSet, SocketStorage}, socket::dhcpv4, storage::{PacketBuffer, RingBuffer}};

use crate::{drivers::e1000::E1000, kernel::context::{enter_hart_context, exit_thread_with_state, get_hart_context, hart_has_thread}};

pub mod channel;
pub mod drivers;
pub mod kernel;
pub mod ktrace;

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


#[derive(Debug, Clone, Copy)]
pub struct SocketReq {
    netchan: NetChannel,
    nc_type: usize,
    pid: u16
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
    use tracing::{info, warn, error};

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

        while let Some(handle) = socket_deletions.dequeue() {
            let _ = user_conns.remove(&handle);
            sockets.remove(handle);
        }

        for (sock_handle, req) in user_conns.iter_mut() {
            let nc = &mut req.netchan;

            if req.nc_type == 0 {
                let socket = sockets.get_mut::<smoltcp::socket::tcp::Socket>(*sock_handle);
                unsafe {
                    iface = nc.update_tcp(iface, socket);
                }   
            }
        }
        
        let default_wake = now + 100_000;
        let mut wake_time = iface.poll_at(timestamp, &mut sockets)
            .map(|i| i.total_micros() as usize * 10)
            .unwrap_or(default_wake);

        next_poll = wake_time;

        for q in socket_reqs.iter_mut() {
            while let Some(mut req) = q.dequeue() {
                info!("net: processing req {req:016X?}");

                let mut nc = &mut req.netchan;

                if req.nc_type == 0 {
                    let (txr, rxr) = nc.rings();

                    info!("net: tcp socket ring lens: rx={},tx={}", rxr.len(), txr.len());

                    let tx_buffer = RingBuffer::new(txr);
                    let rx_buffer = RingBuffer::new(rxr);

                    let sock = smoltcp::socket::tcp::Socket::new(rx_buffer, tx_buffer);
                    let handle = sockets.add(sock);

                    info!("net: created tcp socket: {handle:?}");
                    user_conns.insert(handle, req);

                    next_poll = 0;

                    if let Err(assoc) = socket_associations.enqueue((req.pid as usize, handle)) {
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
                let p = hart_context.current.load(Ordering::Relaxed) as *mut Thread;
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
    unsafe {
        let thread: &Thread = (cptr as *const Thread).as_ref_unchecked();

        // Always safe: asid restore is for this hart's post-trap kernel work.
        frame.asid = thread.pid as usize;

        // Only snapshot thread state if the trap actually describes the
        // thread's own execution. Mismatch means an S-mode interrupt fired
        // while the kernel was mid-context-switch for a user thread (SIE
        // left on inside enter_hart_context_asm) — EPC points into kernel
        // .text, and saving it as thread.pc would break sret on resume.
        let trap_was_in_thread = (thread.mode == SPP::User) == from_user;
        if !trap_was_in_thread { return; }

        let thread_state = thread.state.load(Ordering::Acquire);
        if thread_state == ThreadState::Running as usize
            || thread_state == ThreadState::Suspended as usize
            || thread_state == ThreadState::Blocking as usize
        {
            let frame_ptr = thread.frame as *const TrapFrame as *mut TrapFrame;
            core::ptr::copy_nonoverlapping(frame as *const _, frame_ptr, 1);
            thread.pc.store(epc, Ordering::Release);
        }
    }
}

pub fn handle_serial_print(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    unsafe {
        let current = hart_context.current.load(Ordering::Acquire);
        if current == null_mut() {
            frame.regs[10] = (-1 as isize) as usize;
            return
        }

        let thread = (current as *const Thread)
            .as_ref_unchecked();

        let root_table = crate::kernel::memmap::kernel_root_from_pa(thread.root_table_addr() as u64);

        // Walk the user PT by hand and access the buffer through its KDMAP
        // alias. No SUM gate applies here — the trap vector already switched
        // to the kernel satp, so user VAs aren't in the active address space.
        let mut ret = 0isize;

        let arg1 = frame.regs[12];
        if arg1 > PAGE_SIZE {
            ret = -3;
        }
        else {
            if let Some(kva) = crate::kernel::memmap::user_va_to_kdmap(&root_table, frame.regs[11] as u64) {
                let slice = core::slice::from_raw_parts(kva as *const u8, arg1);
                if let Ok(s) = core::str::from_utf8(slice) {
                    match serial::SERIAL.print_str(s) {
                        Ok(_) => (),
                        Err(_) => {ret = -5}
                    }
                }
                else {
                    ret = -4;
                }
            }
            else {
                ret = -2;
            }
        }

        frame.regs[10] = ret as usize;

        // restore asid after potential switch from user mode
        frame.asid = thread.pid as usize;

        let frame_ptr = thread.frame as *const TrapFrame as usize as *mut TrapFrame;
        core::ptr::copy_nonoverlapping(
            frame as *const _,
            frame_ptr,
            1);

        thread.pc.store(epc + 4, Ordering::Release);

        exit_thread_with_state(ThreadState::Ready)
    }
}

pub fn handle_ms_sleep(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    unsafe {
        let current = hart_context.current.load(Ordering::Acquire);
        if current == null_mut() {
            frame.regs[10] = (-1 as isize) as usize;
            return
        }

        if frame.regs[11] >= (60 * 60 * 1000) {
            frame.regs[10] = (-2 as isize) as usize;
            return
        }

        let thread = (current as *mut Thread)
            .as_mut_unchecked();

        const TICKS_PER_MS: usize = 10_000;
        let wake_time = riscv::register::time::read()
            .wrapping_add(frame.regs[11].wrapping_mul(TICKS_PER_MS));

        thread.wake_time = wake_time;

        frame.regs[10] = 0;

        let frame_ptr = thread.frame as *const TrapFrame as usize as *mut TrapFrame;
        core::ptr::copy_nonoverlapping(
            frame as *const _,
            frame_ptr,
            1);

        thread.pc.store(epc + 4, Ordering::Release);
        
        exit_thread_with_state(ThreadState::Suspended);
    }
}

#[unsafe(no_mangle)]
pub fn handle_mmap_req(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    unsafe {
        let current = hart_context.current.load(Ordering::Acquire);
        if current == null_mut() {
            frame.regs[10] = (-1 as isize) as usize;
            return
        }

        let thread = (current as *mut Thread)
            .as_mut_unchecked();

        let mmap_req = MemMapReq {
            vaddr: frame.regs[11],
            size: frame.regs[12],
            page_permissions: frame.regs[13] as u64,
            share_with_kernel: frame.regs[14] > 0
        };

        thread.block_reason = ThreadBlockReason::MemMap(mmap_req);
        
        let frame_ptr = thread.frame as *const TrapFrame as usize as *mut TrapFrame;
        core::ptr::copy_nonoverlapping(
            frame as *const _,
            frame_ptr,
            1);

        thread.pc.store(epc + 4, Ordering::Release);
        
        exit_thread_with_state(ThreadState::Blocking);
    }
}

#[unsafe(no_mangle)]
pub fn handle_nc_registration_req(epc: usize, hart_context: &'static HartContext, frame: &mut TrapFrame) {
    unsafe {
        let current = hart_context.current.load(Ordering::Acquire);
        if current == null_mut() {
            frame.regs[10] = (-1 as isize) as usize;
            return
        }

        let thread = (current as *mut Thread)
            .as_mut_unchecked();

        let nc_req = NetChannelRegistrationReq {
            nc_vaddr: frame.regs[11],
            nc_type: frame.regs[12]
        };

        thread.block_reason = ThreadBlockReason::NetChannelRegistration(nc_req);
        
        let frame_ptr = thread.frame as *const TrapFrame as usize as *mut TrapFrame;
        core::ptr::copy_nonoverlapping(
            frame as *const _,
            frame_ptr,
            1);

        thread.pc.store(epc + 4, Ordering::Release);
        
        exit_thread_with_state(ThreadState::Blocking);
    }
}
