use core::alloc::Layout;
use core::ptr::{NonNull, null_mut};
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;

use alloc::collections::{btree_map::BTreeMap};
use alloc::{boxed::Box, vec::Vec};

use device::{HartContext, Stack, TrapFrame};
use dtoolkit::fdt::FdtNode;
use dtoolkit::{Node, fdt::{Fdt}};
use elf::endian::LittleEndian;
use heapless::spsc::Queue;
use mem::frame::FrameAllocator;
use mem::{round_u64_down, round_u64_up, round_usize_up};
use mmu::mmap::{PageAlloc, id_map_range, map_address_range, unmap, unmap_page, virt_to_phys};
use mmu::sv48::{PageTable, PhysAddr, VirtAddr};
use mmu::{KB, MB, MappingConfig, PAGE_SIZE, PagePermissions};
use net_channel::{NetChannel, NetChannelQueue, NetChannelState};
use process::{MappingKind, MemMapReq, NetChannelRegistrationReq, PThread, PhysBacking, Process, Thread, ThreadBlockReason, ThreadState, UserMapping};
use riscv::register::satp::{Mode, Satp};
use riscv::register::sstatus::SPP;
use serial::println;
use smoltcp::iface::{Config, Interface, SocketHandle};
use smoltcp::wire::{EthernetAddress};
use tracing::{error, info, warn};

use crate::drivers::e1000::{E1000, E1000Pbuf, RX_RING_BUFS_BYTES, RX_RING_BYTES, RX_RING_LEN, RxDesc, TX_RING_BUFS_BYTES, TX_RING_BYTES, TX_RING_LEN, TxDesc};
use crate::kernel::context::get_hart_context;
use crate::kernel::pci::PciDevice;
use crate::{NetPackage, SocketReq, supervisor_wake_hart};

pub mod context;
pub mod memmap;
pub mod orbital_elf;
pub mod pci;

pub use memmap::KernelLayout;

// TODO: page unmapping

pub const UMODE_TEST_ELF: &'static [u8] = include_bytes!("../../../umode/target/riscv64gc-unknown-none-elf/release/umode");

// Per-thread stack region layout. Each slot gets a 4 MiB chunk: a 2 MiB
// guard (reserved vaddrs, no leaves) followed by a 2 MiB stack mapped as a
// single superpage. Stacks grow down, so the guard sits below the stack to
// catch overflow as a page fault. 256 slots * 4 MiB = 1 GiB; the region
// ends exactly at USER_TRAP_FRAME_BASE below.
pub const UPROC_STACK_BASE:   u64 = 0x2000_0000;
pub const UPROC_STACK_STRIDE: u64 = 4 * MB;
pub const UPROC_STACK_SIZE:   u64 = 2 * MB;
pub const UPROC_STACK_GUARD:  u64 = 2 * MB;

pub const fn user_stack_guard_vaddr(slot: u16) -> u64 {
    UPROC_STACK_BASE + (slot as u64) * UPROC_STACK_STRIDE
}

pub const fn user_stack_vaddr(slot: u16) -> u64 {
    user_stack_guard_vaddr(slot) + UPROC_STACK_GUARD
}

// Kernel-private region inside every user pagetable that holds per-thread
// TrapFrame mappings (no U bit — only S-mode reads them, in
// enter_hart_context_asm, after the satp switch). One page per slot, indexed
// by the thread's per-process slot, so siblings in the same address space
// don't collide. Sized to cover SlotAlloc::CAPACITY. Lives above the stack
// region (UPROC_STACK_BASE + slot * UPROC_STACK_STRIDE) and below
// USER_TEXT_BASE so it can't alias either.
pub const USER_TRAP_FRAME_BASE:   u64 = 0x6000_0000;
pub const USER_TRAP_FRAME_STRIDE: u64 = PAGE_SIZE as u64;

pub const fn user_trap_frame_vaddr(slot: u16) -> u64 {
    USER_TRAP_FRAME_BASE + (slot as u64) * USER_TRAP_FRAME_STRIDE
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

    table_pages: FrameAllocator<33>,
    kernel_pages: FrameAllocator<33>,

    net_pkg: NetPackage,
    orphaned_sockets: Vec<SocketHandle>
}

impl Orbit {
    const THREAD_STACK_LAYOUT: Layout = unsafe {
        Layout::from_size_align_unchecked(2 * MB as usize, 2 * MB as usize)
    };

    const THREAD_TRAP_FRAME_LAYOUT: Layout = unsafe {
        Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE)
    };

    const TABLE_LAYOUT: Layout = unsafe {
        Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE)
    };

    pub const IGB_ADDR: u64 = 0x4000_0000;

    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    pub fn runnable_thread_count(&self) -> usize {
        self.threads.iter()
            .filter(|(_, t)| unsafe {
                let thread = (t.0 as *const Thread).as_ref_unchecked();
                if thread.state.load(Ordering::Acquire) == ThreadState::Ready as usize {
                    return true
                }
                else if thread.state.load(Ordering::Acquire) == ThreadState::Suspended as usize {
                    if riscv::register::time::read() >= thread.wake_time {
                        thread.state.store(ThreadState::Ready as usize, Ordering::Release);
                    }
                    return true
                }
                false
            })
            .count()
    }

    pub const fn new(
        dtb_addr: usize,
        _serial_addr: usize,
        cpu_count: usize,
        layout: KernelLayout,
        table_pages: FrameAllocator<33>,
        kernel_pages: FrameAllocator<33>,
        satp: Satp)
        -> Self
    {
        Self {
            dtb_addr,
            _serial_addr,
            table_pages,
            kernel_pages,
            cpu_count,
            satp,
            layout,
            current_process_id: 0,
            current_thread_id: 0,
            processes: BTreeMap::new(),
            threads: BTreeMap::new(),
            net_pkg: NetPackage {
                phy: None,
                iface: None,
                socket_reqs: alloc::vec::Vec::new(),
                socket_associations: heapless::spsc::Queue::new(),
                socket_deletions: heapless::spsc::Queue::new()
            },
            orphaned_sockets: Vec::new()
        }
    }

    fn allocate_thread_stack(&mut self) -> Result<usize, ()> {
        self.kernel_pages.alloc_aligned(Self::THREAD_STACK_LAYOUT)
            .ok_or(())
            .map_err(|_| {
                serial::println!("failed to allocate new thread stack"); })
    }

    fn allocate_trap_frame(&mut self) -> Result<usize, ()> {
        self.kernel_pages.alloc_aligned(Self::THREAD_TRAP_FRAME_LAYOUT)
            .ok_or(())
            .map_err(|_| {
                serial::println!("failed to allocate new trap frame"); })
    }

    fn create_new_page_table(&mut self) -> Result<&'static PageTable, ()> {
        let addr = self.table_pages.alloc_aligned(Self::TABLE_LAYOUT) 
            .ok_or(())
            .map_err(|_| {
                serial::println!("failed to allocate new page table"); })?;

        unsafe {
            Ok((addr as *const PageTable).as_ref_unchecked())
        }
    }

    pub fn create_kernel_thread(&mut self, entrypoint: usize, a0: Option<usize>) -> Result<(), ()> {
        if self.current_process_id == u16::MAX {
            serial::println!("too many processes running to spawn another");
            return Err(())
        }
        
        let stackp = self.allocate_thread_stack()?;

        let trap_frame = match self.allocate_trap_frame() {
            Ok(p) => p,
            Err(_) => {
                self.kernel_pages.dealloc_aligned(stackp, Self::THREAD_STACK_LAYOUT);
                serial::println!("failed to alloc trap_frame for kthread");
                return Err(())
            }
        };

        let pid = 0;
        let tid = self.next_tid();

        let (frame, stack) = unsafe {
            let f = trap_frame as *mut TrapFrame;
            core::ptr::write_bytes(f as *mut u8, 0, PAGE_SIZE);

            let s = stackp as *mut Stack;
            core::ptr::write_bytes(s as *mut u8, 0, 2 * MB as usize);

            (
                f.as_mut_unchecked(),
                s.as_mut_unchecked()
            )
        };

        frame.regs[1] = entrypoint;
        frame.regs[2] = stackp + Self::THREAD_STACK_LAYOUT.size();
        frame.regs[10] = a0.unwrap_or(0);
        frame.asid = 0;

        let kthread = Thread {
            pc: AtomicUsize::new(entrypoint),
            satp: self.satp,
            mode: SPP::Supervisor,
            tid, pid,
            ticks: 0,
            frame,
            stack,
            state: AtomicUsize::new(ThreadState::Ready as usize),
            wake_time: 0,
            block_reason: ThreadBlockReason::NotBlocking,
            slot: None,
            fault_info: None,
        };

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(kthread);
        let tptr = Box::into_raw(t);
        serial::println!("created kthread@{:016X?}", tptr);

        self.threads.insert(tid, PThread(tptr));

        Ok(())
    }

    fn handle_mmap_req<'t>(&mut self, thread: &'t mut Thread, req: MemMapReq) {
        serial::println!("handling mmap req {req:08X?}");
        
        let large_align = 2 * MB as usize;
        let (align, levels) = if (req.size % large_align) == 0 && (req.vaddr % large_align) == 0 {
            (large_align, 3)
        }
        else if (req.size % PAGE_SIZE) == 0 && (req.vaddr % PAGE_SIZE) == 0 {
            (PAGE_SIZE, 4)
        }
        else {
            serial::println!("failed to select alignment for mmap req: {req:?}");
            thread.frame.regs[10] = -1isize as usize;

            return
        };

        let size = req.size;

        let layout = match Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(e) => {
                serial::println!("failed to create alignment for mmap req: {e:?}");
                thread.frame.regs[10] = -2isize as usize;

                return
            }
        };

        let paddr = match self.kernel_pages.alloc_aligned(layout) {
            Some(p) => p,
            None => {
                serial::println!("failed to alloc pages for mmap req: {req:?}");
                thread.frame.regs[10] = -3isize as usize;

                return
            }
        };

        let supervisor_tag = if req.share_with_kernel {
            Some(0x1)
        }
        else { None };

        let config = MappingConfig {
            permissions: (req.page_permissions & 0xE) | PagePermissions::U,
            levels,
            page_size: align as u64,
            vaddr: VirtAddr::new(req.vaddr as u64),
            paddr: PhysAddr::new(paddr as u64),
            log: true,
            supervisor_tag
        };

        let vend = VirtAddr::new((req.vaddr + req.size) as u64);
        let pend = PhysAddr::new((paddr + req.size) as u64);

        unsafe {
            let root_table = {
                (thread.root_table_addr() as *const PageTable)
                    .as_ref_unchecked()
            };

            let mut pages = PageAlloc::FA(&mut self.table_pages);

            if let Err(_) = map_address_range(root_table, &mut pages, &config, vend, pend) {
                serial::println!("failed to map pages for mmap req: {req:?}");
                thread.frame.regs[10] = -4isize as usize;

                self.kernel_pages.dealloc_aligned(paddr, layout);

                return
            }
        }

        let owning_process = match self.processes.get_mut(&thread.pid) {
            Some(proc) => proc,
            None => {
                serial::println!("failed to add pages to process metadata (no pid): {req:?}");
                thread.frame.regs[10] = -5isize as usize;

                self.kernel_pages.dealloc_aligned(paddr, layout);

                return
            }
        };

        owning_process.heap_pages.push((paddr, layout));

        core::sync::atomic::fence(Ordering::SeqCst);
        
        unsafe {
            riscv::asm::sfence_vma(thread.pid as usize, 0);
            riscv::asm::sfence_vma(0, 0);
        }

        serial::println!("fulfilled {req:?}:\n\t0x{paddr:016X} {layout:08X?}");

        thread.frame.regs[10] = 0;
    }

    fn translate_nc_addrs(rpt: &PageTable, nc: &mut NetChannel) -> bool {
        unsafe {
            {
                let v = nc.desired_state.as_ptr() as u64;
                let p = match virt_to_phys(rpt, VirtAddr::new(v)) {
                    None => return false,
                    Some(a) => a
                };

                info!("translated v0x{v:08X?}->p0x{p:08X?}");

                nc.desired_state = NonNull::new_unchecked(p as *mut NetChannelState);
            }

            {
                let v = nc.current_state.as_ptr() as u64;
                let p = match virt_to_phys(rpt, VirtAddr::new(v)) {
                    None => return false,
                    Some(a) => a
                };

                info!("translated v0x{v:08X?}->p0x{p:08X?}");

                nc.current_state = NonNull::new_unchecked(p as *mut NetChannelState);
            }

            {
                let v = nc.tx.as_ptr() as u64;
                let p = match virt_to_phys(rpt, VirtAddr::new(v)) {
                    None => return false,
                    Some(a) => a
                };

                info!("translated v0x{v:08X?}->p0x{p:08X?}");

                nc.tx = NonNull::new_unchecked(p as *mut NetChannelQueue);
            }

            {
                let v = nc.rx.as_ptr() as u64;
                let p = match virt_to_phys(rpt, VirtAddr::new(v)) {
                    None => return false,
                    Some(a) => a
                };

                info!("translated v0x{v:08X?}->p0x{p:08X?}");

                nc.rx = NonNull::new_unchecked(p as *mut NetChannelQueue);
            }
        }
        true
    }

    fn handle_nc_register_req<'t>(&mut self, thread: &'t mut Thread, req: NetChannelRegistrationReq) {
        info!("handling nc registration req: {req:08X?}");

        let (nc_paddr, rpt) = unsafe {
            let rpt = (thread.root_table_addr() as *const PageTable).as_ref_unchecked();
            (virt_to_phys(rpt, VirtAddr::new(req.nc_vaddr as u64)), rpt)
        };

        let nc_paddr = match nc_paddr {
            Some(p) => p,
            None => {
                thread.frame.regs[10] = -1isize as usize;
                return
            }
        };

        info!("nc@p0x{nc_paddr:08X?}");

        let mut nc = unsafe {
            (nc_paddr as *const NetChannel)
                .read_volatile()
        };

        info!("nc={nc:?}");

        if !Self::translate_nc_addrs(rpt, &mut nc) {
            warn!("failed to translate socket req {req:?}");
            thread.frame.regs[10] = -2isize as usize;
            return
        }

        let socket_req = SocketReq {
            netchan: nc,
            nc_type: req.nc_type,
            pid: thread.pid
        };

        if let Some(np) = self.net_pkg.socket_reqs.get_mut(get_hart_context().hart_id as usize) {
            if let Err(e) = np.enqueue(socket_req) {
                warn!("failed to queue socket req {socket_req:?}");
                thread.frame.regs[10] = -3isize as usize;
                return
            }
            else {
                info!("queued socket req");
                thread.frame.regs[10] = 0;
                return
            }
        }
    }

    fn handle_block_reason<'t>(&mut self, thread: &'t mut Thread, reason: ThreadBlockReason) {
        match reason {
            ThreadBlockReason::MemMap(req) => self.handle_mmap_req(thread, req),
            ThreadBlockReason::NetChannelRegistration(req) => self.handle_nc_register_req(thread, req),
            _ => {}
        }
    }
    
    fn get_runnable_thread(&mut self) -> Option<PThread> {
        for (_tid, p) in self.threads.iter() {
            let thread: &mut Thread = unsafe {
                p.0.as_mut_unchecked()
            };

            let state = thread.state.load(Ordering::Acquire);
            if state == ThreadState::Ready as usize {
                return Some(PThread(p.0))
            }
            else if state == ThreadState::Running as usize {
                continue
            }
            else if state == ThreadState::Assigned as usize {
                continue
            }
            else if state == ThreadState::Exited as usize {
                continue
            }
            else if state == ThreadState::Suspended as usize {
                let now = riscv::register::time::read();

                if now < thread.wake_time {
                    continue
                }

                thread.state.store(ThreadState::Ready as usize, Ordering::Release);

                return Some(PThread(p.0))
            }
            else if state == ThreadState::Blocking as usize {
                let reason = thread.block_reason;
                let pt = PThread(p.0);

                self.handle_block_reason(thread, reason);

                info!("unblocked thread{}", thread.tid);

                thread.block_reason = ThreadBlockReason::NotBlocking;
                thread.state.store(ThreadState::Ready as usize, Ordering::Release);

                return Some(pt)
            }
        }
        None
    }

    fn dealloc_thread(&mut self, thread: &'static Thread) {
        match (thread.slot, thread.pid) {
            (None, 0) => { /* kernel thread — no per-process slot expected */ }
            (Some(slot), 0) => error!(
                "dealloc_thread: tid{} is a kernel thread but carries slot{}",
                thread.tid, slot),
            (None, pid) => error!(
                "dealloc_thread: tid{} user thread in pid{} is missing its slot",
                thread.tid, pid),
            (Some(slot), pid) => match self.processes.get_mut(&pid) {
                Some(proc) => {
                    let tf_vaddr    = user_trap_frame_vaddr(slot);
                    let stack_vaddr = user_stack_vaddr(slot);
                    let guard_vaddr = user_stack_guard_vaddr(slot);
                    let root_table  = unsafe {
                        ((proc.satp.ppn() * PAGE_SIZE) as *const PageTable)
                            .as_ref_unchecked()
                    };
                    // Drop the per-thread leaves so the slot's vaddrs are
                    // reusable. Failure here just means the entry was already
                    // gone (e.g. process teardown raced ahead). The guard has
                    // no leaf — only its proc.maps entry needs clearing.
                    unsafe {
                        let _ = unmap_page(root_table, VirtAddr::new(tf_vaddr), 4);
                        let _ = unmap_page(root_table, VirtAddr::new(stack_vaddr), 3);
                        riscv::asm::sfence_vma(pid as usize, tf_vaddr as usize);
                        riscv::asm::sfence_vma(pid as usize, stack_vaddr as usize);
                    }
                    let _ = proc.maps.remove(&tf_vaddr);
                    let _ = proc.maps.remove(&stack_vaddr);
                    let _ = proc.maps.remove(&guard_vaddr);
                    proc.thread_slots.free(slot);
                }
                None => error!(
                    "dealloc_thread: tid{} references missing pid{}",
                    thread.tid, pid),
            }
        }

        let tstack = thread.stack as *const _ as usize;
        self.kernel_pages.dealloc_aligned(tstack, Self::THREAD_STACK_LAYOUT);

        let trap_frame = thread.frame as *const _ as usize;
        self.kernel_pages.dealloc_aligned(trap_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
    }

    fn dealloc_process(&mut self, mut process: Process) {
        let process_root_table_addr = process.satp.ppn() * PAGE_SIZE;

        while let Some(socket_handle) = process.sockets.pop_last() {
            if let Err(e) = self.net_pkg.socket_deletions.enqueue(socket_handle) {
                error!("failed to queue socket for deletion while deallocating pid{}", process.pid);
                self.orphaned_sockets.push(socket_handle);
            }
        }

        let root_table = unsafe {
            (process_root_table_addr as *const PageTable)
                .as_ref_unchecked()
        };

        while let Some((paddr, layout)) = process.heap_pages.pop() {
            serial::println!("dealloc heap page@{paddr:0016X} {layout:08X?}");
            self.kernel_pages.dealloc_aligned(paddr, layout);
        }

        let mut pages = PageAlloc::FA(&mut self.table_pages);
        unsafe { 
            unmap(root_table, &mut pages);
            self.table_pages.dealloc_aligned(process_root_table_addr, Self::TABLE_LAYOUT);
        }
    }

    pub fn cleanup_threads_and_processes(&mut self) {
        let mut tids_to_remove = Vec::new();
        let mut pids_to_remove = Vec::new();
        for (_tid, p) in self.threads.iter() {
            let t = unsafe {
                p.0.as_ref_unchecked()
            };

            {
                let proc = match self.processes.get_mut(&t.pid) {
                    Some(p) => p,
                    None => continue
                };

                let thread_alive = t.state.load(Ordering::Acquire)
                    != ThreadState::Exited as usize;

                if !thread_alive {
                    let _ = proc.threads.remove(&t.tid);
                    tids_to_remove.push(t.tid);
                    let boxed = unsafe { Box::from_raw(p.0) };

                    match boxed.fault_info {
                        Some(f) => {
                            let label = match proc.find_mapping(f.stval as u64).map(|m| m.kind) {
                                Some(MappingKind::Guard { .. }) => "stack overflow",
                                Some(_)                         => "permission/range violation",
                                None                            => "bad access",
                            };
                            serial::println!(
                                "tid{} killed: {} cause={} epc={:#x} stval={:#x}",
                                t.tid, label, f.cause, f.epc, f.stval);
                        }
                        None => {
                            let status = boxed.frame.regs[11] as isize;
                            serial::println!("tid{} dead, removing status={status}", t.tid);
                        }
                    }
                }

                if !proc.threads.is_empty() || t.pid == 0 {
                    continue
                }
            }

            serial::println!("pid{} dead, removing", t.pid);

            pids_to_remove.push(t.pid);
        }

        for tid in tids_to_remove {
            let p = self.threads.remove(&tid)
                .unwrap();

            let thread = unsafe {
                p.0.as_ref_unchecked()
            };

            self.dealloc_thread(thread);
        }

        for pid in pids_to_remove {
            let proc = self.processes.remove(&pid)
                .unwrap();

            self.dealloc_process(proc);
        }
    }
    
    pub fn assign_threads(&mut self, context: &'static HartContext) {
        let hart_root = unsafe {
            (riscv::register::sscratch::read() as *const HartContext)
                .sub(context.hart_id as usize)
        };

        for hart in 0..self.cpu_count {
            if hart == context.hart_id as usize {
                continue
            }

            let hart_context = unsafe {
                hart_root.add(hart).as_ref_unchecked()
            };

            if hart_context.current.load(Ordering::Acquire) != null_mut() {
                //serial::println!("skipping CPU{hart} (busy)");
                continue
            }

            if let Some(t) = self.get_runnable_thread() {
                let thread = unsafe {
                    t.0.as_mut_unchecked()
                };

                //info!("assigning thread{} state{} to CPU{hart}", thread.tid, thread.state.load(Ordering::Acquire));

                thread.ticks = thread.ticks.wrapping_add(1);
                thread.state.store(ThreadState::Assigned as usize, Ordering::Release);

                hart_context.current.store(t.0 as usize as *mut (), Ordering::Release);

                supervisor_wake_hart(hart);
            }
        }

        if let Some(t) = self.get_runnable_thread() {
            let thread = unsafe {
                t.0.as_mut_unchecked()
            };

            //info!("assigning thread{} state{} to CPU{}", thread.tid, thread.state.load(Ordering::Acquire), context.hart_id);

            thread.ticks = thread.ticks.wrapping_add(1);
            thread.state.store(ThreadState::Assigned as usize, Ordering::Release);

            context.current.store(t.0 as usize as *mut (), Ordering::Release);
        }
    }

    pub fn print_threads(&self) {
        for (_, t) in self.threads.iter() {
            let thread = unsafe {
                (t.0 as *const Thread).as_ref_unchecked()
            };

            serial::println!("tid{}: state{}", thread.tid, thread.state.load(Ordering::Acquire));
        }
    }

    fn get_root_table(&self) -> &'static PageTable {
        unsafe { ((self.satp.ppn() * PAGE_SIZE) as *const PageTable).as_ref_unchecked() }
    }
    
    fn setup_igb(&mut self, device: &PciDevice) {
        device.print_info();

        let map: MappingConfig = MappingConfig {
            permissions: PagePermissions::R | PagePermissions::W | PagePermissions::G,
            levels: 4,
            page_size: 4096,
            vaddr: VirtAddr::new(Self::IGB_ADDR),
            paddr: PhysAddr::new(Self::IGB_ADDR),
            log: false,
            supervisor_tag: None
        };
        
        let ort = self.get_root_table();

        unsafe {
            let bar_size = device.get_bar_size(0) as u64;
            if bar_size > (2 * MB) {
                serial::println!("bar2big");
                return
            }

            let mut pages = PageAlloc::FA(&mut self.table_pages);

            serial::println!("mapping {}KB BAR0", bar_size / KB);
            
            match id_map_range(ort, &mut pages, map, Self::IGB_ADDR..(Self::IGB_ADDR + bar_size)) {
                Err(_) => {
                    serial::println!("failed to map bar");
                    return
                },
                Ok(id) => {
                    serial::println!("{id:?}");
                }
            }

            device.write_bar(0, Self::IGB_ADDR as u32);

            riscv::register::satp::write(self.satp);
            riscv::asm::sfence_vma(0, 0);

            let tx_ring = (self.kernel_pages.alloc_aligned(
                Layout::from_size_align_unchecked(TX_RING_BYTES, PAGE_SIZE))
                .expect("no e1000 tx ring") as *mut [TxDesc; TX_RING_LEN])
                .as_mut_unchecked();

            let rx_ring = (self.kernel_pages.alloc_aligned(
                Layout::from_size_align_unchecked(RX_RING_BYTES, PAGE_SIZE))
                .expect("no e1000 rx ring") as *mut [RxDesc; RX_RING_LEN])
                .as_mut_unchecked();

            let tx_bufs = (self.kernel_pages.alloc_aligned(
                Layout::from_size_align_unchecked(TX_RING_BUFS_BYTES, PAGE_SIZE))
                .expect("no e1000 tx bufs") as *mut [E1000Pbuf; TX_RING_LEN])
                .as_mut_unchecked();

            let rx_bufs = (self.kernel_pages.alloc_aligned(
                Layout::from_size_align_unchecked(RX_RING_BUFS_BYTES, PAGE_SIZE))
                .expect("no e1000 rx bufs") as *mut [E1000Pbuf; RX_RING_LEN])
                .as_mut_unchecked();

            let mut e1000 = E1000::new(Self::IGB_ADDR as *mut u32, tx_ring, tx_bufs, rx_ring, rx_bufs);
            let mac = e1000.read_mac().unwrap();
            if let Err(_) = e1000.init_hw(mac) {
                // free everything ig
                serial::println!("failed to init e1000");
            }

            let mut config = Config::new(EthernetAddress(mac).into());
            config.random_seed = 4;

            let iface = Interface::new(config, &mut e1000, smoltcp::time::Instant::from_micros(
                riscv::register::time::read() as i64 / 10
            ));

            let socket_reqs = (0..self.cpu_count)
                .map(|_| heapless::spsc::Queue::<crate::SocketReq, 8>::new())
                .collect();

            self.net_pkg.iface = Some(iface);
            self.net_pkg.phy = Some(e1000);
            self.net_pkg.socket_reqs = socket_reqs;

            let entrypoint = crate::k_net as *const () as usize;
            let a0 = (&mut self.net_pkg) as *mut NetPackage;
            if let Err(_) = self.create_kernel_thread(entrypoint, Some(a0 as usize)) {
                serial::println!("failed to create knet thread");
            }
            else {
                serial::println!("created knet thread");
            }

            /*
            // Create sockets
            let mut dhcp_socket = dhcpv4::Socket::new();
            dhcp_socket.set_max_lease_duration(Some(smoltcp::time::Duration::from_secs(600)));

            let mut socket_shit = [SocketStorage::EMPTY; 16];

            let mut sockets = SocketSet::new(&mut socket_shit[..]);
            let dhcp_handle = sockets.add(dhcp_socket);

            fn set_ipv4_addr(iface: &mut Interface, cidr: Ipv4Cidr) {
                iface.update_ip_addrs(|addrs| {
                    addrs.clear();
                    addrs.push(IpCidr::Ipv4(cidr)).unwrap();
                });
            }

            let tx_sockb = SocketBuffer::new(vec![0u8; 2048]);
            let rx_sockb = SocketBuffer::new(vec![0u8; 2048]);
            let tcp = smoltcp::socket::tcp::Socket::new(rx_sockb, tx_sockb);

            let tcp_handle = sockets.add(tcp);
            let mystery: &mut TcpSocket = sockets.get_mut(tcp_handle);

            /*
                1) allocate 4096b page for tx/rx bufs
                2) map into kernel + maybe user memory
                3) pass socket ingredients to k_net e.g. type + buffers

                1) check for socket ingredient messages and create new sockets
                2) disable interrupts
                3) activate sum bit
                3) poll iface and socketset
                4) data goes into shared memory
                5) kernel updates atomic ring buffer thing to inform thread of
                   progress on its connection
                6) 
            */

            loop {
                let timestamp = smoltcp::time::Instant::from_micros(
                    riscv::register::time::read() as i64 / 10
                );

                if iface.poll(timestamp, &mut e1000, &mut sockets) {
                    let event = sockets.get_mut::<dhcpv4::Socket>(dhcp_handle).poll();
                    match event {
                        None => {
                            //serial::println!("no event");
                        }
                        Some(dhcpv4::Event::Configured(config)) => {
                            serial::println!("DHCP config acquired!");

                            serial::println!("IP address:      {}", config.address);
                            set_ipv4_addr(&mut iface, config.address);

                            if let Some(router) = config.router {
                                serial::println!("Default gateway: {}", router);
                                iface.routes_mut().add_default_ipv4_route(router).unwrap();
                            } else {
                                serial::println!("Default gateway: None");
                                iface.routes_mut().remove_default_ipv4_route();
                            }

                            for (i, s) in config.dns_servers.iter().enumerate() {
                                serial::println!("DNS server {}:    {}", i, s);
                            }
                        }
                        Some(dhcpv4::Event::Deconfigured) => {
                            serial::println!("DHCP lost config!");
                            iface.update_ip_addrs(|addrs| addrs.clear());
                            iface.routes_mut().remove_default_ipv4_route();
                        }
                    }
                }

                

                riscv::asm::delay(10_000_000);
            }
            */
        }
    }
    
    pub fn get_pci_info<'n>(&mut self, node: FdtNode<'n>) {
        let reg = match node.reg() {
            Ok(Some(mut r)) => {
                match r.nth(0) {
                    Some(re) => re,
                    None => return
                }
            },
            _ => return
        };

        serial::println!("reg={reg:?}");

        let base = match reg.address::<u64>() {
            Ok(b) => b as usize,
            Err(_) => return
        };

        let size = match reg.size::<u64>() {
            Ok(b) => b as usize,
            Err(_) => return
        };

        serial::println!("pci@{:08X}..{:08X}", base, base+size);

        let matches = pci::scan_pci(base, &[(0x8086, 0x100E)]);
        if matches.is_empty() {
            return
        }

        self.setup_igb(&matches[0]);
    }
    
    pub fn get_environment_info(&mut self) {
        let fdt = unsafe { Fdt::from_raw_unchecked(self.dtb_addr as *const _) };
        let root = fdt.root();

        let mut nodes: Vec<_> = root.children().collect();
        while let Some(node) = nodes.pop() {
            let name = node.name();
            if name.starts_with("pci") {
                let mut no_virt = Satp::from_bits(0);
                no_virt.set_mode(Mode::Bare);
                unsafe { riscv::register::satp::write(no_virt); }

                self.get_pci_info(node);

                unsafe { riscv::register::satp::write(self.satp); }

                continue
            }

            /*
            println!("\nexamining {}", name);
            for prop in node.properties() {
                println!("\t{prop:?}");
            }
            */

            for child in node.children() {
                nodes.push(child);
            }
        }
    }

    fn map_stack(&mut self, root_table: &'static PageTable, stackp: u64, stackv: u64) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(&mut self.table_pages);
        unsafe {
            map_address_range(
                root_table,
                &mut pages,
                &MappingConfig {
                    permissions: PagePermissions::U | PagePermissions::R | PagePermissions::W,
                    levels: 3, page_size: UPROC_STACK_SIZE,
                    vaddr: VirtAddr::new(stackv),
                    paddr: PhysAddr::new(stackp),
                    log: false,
                    supervisor_tag: None
                },
                VirtAddr::new(stackv + UPROC_STACK_SIZE),
                PhysAddr::new(stackp + UPROC_STACK_SIZE))
        }
    }

    fn map_trap_frame(&mut self, root_table: &'static PageTable, trap_frame: usize, user_vaddr: u64) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(&mut self.table_pages);
        unsafe {
            map_address_range(
                root_table,
                &mut pages,
                &MappingConfig {
                    permissions: PagePermissions::R.into(),
                    levels: 4, page_size: PAGE_SIZE as u64,
                    vaddr: VirtAddr::new(user_vaddr),
                    paddr: PhysAddr::new(trap_frame as u64),
                    log: false,
                    supervisor_tag: None
                },
                VirtAddr::new(user_vaddr + PAGE_SIZE as u64),
                PhysAddr::new((trap_frame + PAGE_SIZE) as u64))
        }
    }
    
    pub fn add_new_thread_to_process(&mut self, pid: u16, entrypoint: usize) -> Result<(), ()> {
        if !self.processes.contains_key(&pid) {
            return Err(())
        }

        let slot = self.processes.get_mut(&pid)
            .unwrap()
            .thread_slots
            .alloc()
            .ok_or(())?;

        let root_table = unsafe {
            let addr = self.processes.get(&pid).unwrap().satp.ppn() * PAGE_SIZE;
            (addr as *const PageTable)
                .as_ref_unchecked()

        };

        let thread = match self.create_new_thread(pid, root_table, entrypoint, slot) {
            Ok(t) => t,
            Err(e) => {
                self.processes.get_mut(&pid).unwrap().thread_slots.free(slot);
                return Err(e);
            }
        };
        
        let tid = thread.tid;
        let rpt = thread.root_table_addr();

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(thread);
        let tptr = Box::into_raw(t);
        serial::println!("created uthread@{tptr:016X?},pid={pid},tid={tid},table={rpt:016X?}");

        let owning_process = self.processes.get_mut(&pid)
            .unwrap();
        
        if !owning_process.threads.insert(tid) {
            self.dealloc_thread(unsafe {tptr.as_ref_unchecked()});
            return Err(())
        }

        owning_process.thread_count = owning_process
            .thread_count
            .saturating_add(1);

        self.threads.insert(tid, PThread(tptr));

        Ok(())
    }
    
    pub fn create_new_thread(&mut self, pid: u16, root_table: &'static PageTable, entrypoint: usize, slot: u16) -> Result<Thread, ()> {
        let mut page_list = Vec::new();

        let stackp = self.allocate_thread_stack()?;
        page_list.push((stackp, Self::THREAD_STACK_LAYOUT));

        let trap_frame = self.allocate_trap_frame()
            .map_err(|_| {
                self.free_kernel_pages(&page_list[..]);
            })?;

        page_list.push((trap_frame, Self::THREAD_TRAP_FRAME_LAYOUT));

        let stack_vaddr      = user_stack_vaddr(slot);
        let guard_vaddr      = user_stack_guard_vaddr(slot);
        let trap_frame_vaddr = user_trap_frame_vaddr(slot);

        if let Err(_) = self.map_stack(root_table, stackp as u64, stack_vaddr) {
            self.free_kernel_pages(&page_list[..]);
            self.table_pages.dealloc_aligned(root_table as *const _ as usize, Self::TABLE_LAYOUT);

            serial::println!("failed to map stack");

            return Err(())
        }

        if let Err(_) = self.map_trap_frame(root_table, trap_frame, trap_frame_vaddr) {
            self.free_kernel_pages(&page_list[..]);
            self.table_pages.dealloc_aligned(root_table as *const _ as usize, Self::TABLE_LAYOUT);

            serial::println!("failed to map trap frame");

            return Err(())
        }

        if let Some(proc) = self.processes.get_mut(&pid) {
            // Reserved vaddr range below the stack. No leaves — a fault inside
            // here is a stack overflow, which the page-fault path will turn
            // into a thread kill once it consults proc.maps.
            proc.insert_mapping(UserMapping {
                vaddr:   guard_vaddr,
                len:     UPROC_STACK_GUARD,
                perms:   0,
                backing: None,
                kind:    MappingKind::Guard { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr:   stack_vaddr,
                len:     UPROC_STACK_SIZE,
                perms:   (PagePermissions::U | PagePermissions::R | PagePermissions::W) as u64,
                backing: Some(PhysBacking {
                    paddr:  stackp as u64,
                    layout: Self::THREAD_STACK_LAYOUT,
                }),
                kind:    MappingKind::Stack { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr:   trap_frame_vaddr,
                len:     PAGE_SIZE as u64,
                perms:   PagePermissions::R as u64,
                backing: Some(PhysBacking {
                    paddr:  trap_frame as u64,
                    layout: Self::THREAD_TRAP_FRAME_LAYOUT,
                }),
                kind:    MappingKind::TrapFrame { slot },
            });
        }

        let tid = self.next_tid();

        let (frame, stack) = unsafe {
            let f = trap_frame as *mut TrapFrame;
            core::ptr::write_bytes(f as *mut u8, 0, PAGE_SIZE);

            let s = stackp as *mut Stack;
            core::ptr::write_bytes(s as *mut u8, 0, 2 * MB as usize);

            (
                f.as_mut_unchecked(),
                s.as_mut_unchecked()
            )
        };

        let mut satp = Satp::from_bits(0);
        satp.set_asid(pid as usize);
        satp.set_mode(riscv::register::satp::Mode::Sv48);
        satp.set_ppn(root_table as *const _ as usize / PAGE_SIZE);

        frame.regs[1] = entrypoint;
        frame.regs[2] = (stack_vaddr + UPROC_STACK_SIZE - 16) as usize;
        frame.asid = pid as usize;

        serial::println!("ventry={:016X?},vsp=0x{:016X?},rpt={:016X?}", entrypoint, frame.regs[2], root_table as *const PageTable);

        Ok(Thread {
            pc: AtomicUsize::new(entrypoint),
            satp,
            mode: SPP::User,
            tid, pid,
            ticks: 0,
            frame: frame,
            stack,
            state: AtomicUsize::new(ThreadState::Ready as usize),
            wake_time: 0,
            block_reason: ThreadBlockReason::NotBlocking,
            slot: Some(slot),
            fault_info: None,
        })
    }
    
    pub fn create_new_process(&mut self, elf_blob: &[u8]) -> Result<(), ()> {
        let root_table = self.create_new_page_table()?;
        let elf = self.load_elf(root_table, elf_blob)?;
        let pid = self.next_pid();

        let mut proc_satp = Satp::from_bits(0);
        proc_satp.set_ppn((root_table as *const PageTable as usize) / PAGE_SIZE);
        proc_satp.set_mode(Mode::Sv48);
        proc_satp.set_asid(pid as usize);

        let mut proc = Process::new(pid, proc_satp);
        let slot = proc.thread_slots.alloc().ok_or(())?;

        // Insert the Process before creating the thread so create_new_thread
        // can record per-thread UserMappings (TrapFrame, eventually Stack/TLS)
        // into proc.maps via self.processes.get_mut.
        self.processes.insert(pid, proc);

        let thread = match self.create_new_thread(pid, root_table, elf.entrypoint, slot) {
            Ok(t) => t,
            Err(e) => {
                let _ = self.processes.remove(&pid);
                return Err(e);
            }
        };
        let tid = thread.tid;
        let rpt = root_table as *const PageTable;

        if let Err(_) = self.map_kernel_into(root_table) {
            let _ = self.processes.remove(&pid);
            self.table_pages.dealloc_aligned(root_table as *const _ as usize, Self::TABLE_LAYOUT);

            serial::println!("failed to map kernel into process");

            return Err(())
        }

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(thread);
        let tptr = Box::into_raw(t);
        serial::println!("created uprocess@{tptr:016X?},pid={pid},tid={tid},table={rpt:016X?}");

        let proc = self.processes.get_mut(&pid)
            .expect("just inserted");
        proc.threads.insert(tid);
        proc.thread_count = 1;

        self.threads.insert(tid, PThread(tptr));

        Ok(())
    }

    fn free_kernel_pages(&mut self, pages: &[(usize, Layout)]) {
        for page in pages {
            self.kernel_pages.dealloc_aligned(page.0, page.1);
        }
    }
    
    pub fn load_elf(&mut self, root_table: &'static PageTable, elf_blob: &[u8]) -> Result<orbital_elf::ElfInfo, ()> {
        let elf = match elf::ElfBytes::<LittleEndian>::minimal_parse(elf_blob) {
            Ok(e) => e,
            Err(e) => { serial::println!("failed to parse umode elf: {e:?}"); return Err(()) }
        };

        let mut segment_allocations = Vec::new();

        let segments = elf.segments().unwrap();
        for segment in segments.iter() {
            let load_segment = segment.p_type == elf::abi::PT_LOAD;
            if !load_segment {
                continue
            }

            if segment.p_vaddr < (0x80000000 + (256 * MB)) {
                serial::println!("illegal elf p_vaddr");
                return Err(())
            }

            if segment.p_memsz == 0 {
                continue
            }

            serial::println!("loading {segment:08x?}");

            let segment_data = match elf.segment_data(&segment) {
                Ok(seg) => seg,
                Err(e) => {
                    serial::println!("error parsing loadable segment data: {e:?}");
                    return Err(())
                }
            };

            unsafe {
                let layout = Layout::from_size_align_unchecked(segment_data.len(), PAGE_SIZE);
                let phys = match self.kernel_pages.alloc_aligned(layout) {
                    Some(p) => p,
                    None => {
                        self.free_kernel_pages(&segment_allocations[..]);
                        serial::println!("failed to alloc segment");
                        return Err(())
                    },
                };

                segment_allocations.push((phys, layout));

                core::ptr::copy_nonoverlapping(segment_data.as_ptr(), phys as *mut u8, segment_data.len());

                if segment.p_memsz > segment.p_filesz {
                    core::ptr::write_bytes(
                        (phys + segment.p_filesz as usize) as *mut u8,
                        0,
                        (segment.p_memsz - segment.p_filesz) as usize
                    );
                }

                let paddr_start = phys as u64;
                let vaddr_start = round_u64_down(segment.p_vaddr, PAGE_SIZE as u64);

                let segment_aligned_len = round_u64_up(segment_data.len() as u64, PAGE_SIZE as u64);

                let paddr_end = paddr_start + segment_aligned_len;
                let vaddr_end = vaddr_start + segment_aligned_len;

                let mut pages = PageAlloc::FA(&mut self.table_pages);

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
                    log: true,
                    supervisor_tag: None
                };

                let map = map_address_range(
                    root_table,
                    &mut pages,
                    &config,
                    VirtAddr::new(vaddr_end),
                    PhysAddr::new(paddr_end));

                if map.is_err() {
                    self.free_kernel_pages(&segment_allocations);
                    serial::println!("failed to map segment into process");
                    return Err(())
                }
            }
        }
        Ok(orbital_elf::ElfInfo {
            entrypoint: elf.ehdr.e_entry as usize,
            segments: segment_allocations
        })
    }

    fn map_kernel_into(&mut self, root_table: &'static PageTable) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(&mut self.table_pages);
        unsafe { memmap::map_kernel_shared(root_table, &mut pages, &self.layout) }
    }

    fn next_tid(&mut self) -> u32 {
        let mut next = self.current_thread_id.wrapping_add(1);
        loop {
            let matches = self.threads.iter()
                .filter(|(t, _)| next == **t)
                .count();

            if matches == 0 {
                break
            }
            next = next.wrapping_add(1);
        }

        self.current_thread_id = next;

        next
    }

    fn next_pid(&mut self) -> u16 {
        let mut next = self.current_process_id.wrapping_add(1);
        loop {
            let matches = self.processes.iter()
                .filter(|(pid, _)| **pid == next)
                .count();

            if matches == 0 {
                break
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

pub fn ksleep(duration: Duration) {
    let context = get_hart_context();
    let current_thread = unsafe {
        (context.current.load(Ordering::Acquire)
            as *mut Thread).as_mut_unchecked() };
    
    const TICKS_PER_MS: usize = 10_000;
    current_thread.wake_time = riscv::register::time::read()
        .wrapping_add((duration.as_millis() as usize).wrapping_mul(TICKS_PER_MS));
}
