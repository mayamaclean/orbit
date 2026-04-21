use core::alloc::Layout;
use core::ptr::null_mut;
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
use mmu::mmap::{PageAlloc, id_map_range, map_address_range, unmap, unmap_page};
use mmu::sv48::{PageTable, PhysAddr, VirtAddr};
use mmu::{KB, MB, MappingConfig, PAGE_SIZE, PagePermissions};
use net_channel::NetChannel;
use process::{MappingKind, MemMapReq, NetChannelRegistrationReq, PThread, PhysBacking, Pool, Process, Thread, ThreadBlockReason, ThreadState, UserMapping};
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
pub mod user_page;

pub use memmap::KernelLayout;

// TODO: page unmapping

pub const UMODE_TEST_ELF: &'static [u8] = include_bytes!("../../../umode/target/riscv64gc-unknown-none-elf/release/umode");

// User address-space layout lives in the canonical orbit_abi::layout module.
// Re-exported so existing `kernel::USER_TEXT_BASE`-style call sites keep
// working.
pub use orbit_abi::layout::*;

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
    // User-private backings (stacks, ELF, anon mmaps). Wired now, routed
    // to in later steps of the pool-split milestone.
    user_pages: FrameAllocator<33>,

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

    /// Physical address we program into the e1000's BAR0; the device decodes
    /// MMIO accesses to this PA on the bus. The kernel reaches the same
    /// region through a high-half KMMIO alias allocated at setup_igb.
    pub const IGB_BAR_PA: u64 = 0x4000_0000;

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
        user_pages: FrameAllocator<33>,
        satp: Satp)
        -> Self
    {
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

    fn allocate_user_thread_stack(&mut self, stack_size: u64) -> Result<(usize, Layout), ()> {
        let layout = Layout::from_size_align(stack_size as usize, UPROC_STACK_GRAIN as usize)
            .map_err(|e| {
                serial::println!("bad user stack layout for size={stack_size}: {e:?}");
            })?;
        let kva = self.user_pages.alloc_aligned(layout)
            .ok_or(())
            .map_err(|_| {
                serial::println!("failed to allocate user thread stack size={stack_size}");
            })?;

        // Zero before the PTE install exposes the stack to user code — the
        // page may have been returned by a previous process.
        unsafe {
            let paddr = memmap::virt_to_phys_dmap(kva as u64);
            let mut w = user_page::UserPageWindow::map(paddr, layout.size());
            w.as_mut_slice().fill(0);
        }

        Ok((kva, layout))
    }

    fn allocate_trap_frame(&mut self) -> Result<usize, ()> {
        self.kernel_pages.alloc_aligned(Self::THREAD_TRAP_FRAME_LAYOUT)
            .ok_or(())
            .map_err(|_| {
                serial::println!("failed to allocate new trap frame"); })
    }

    fn create_new_page_table(&mut self) -> Result<mmu::mmap::RootTable<'static>, ()> {
        let addr = self.table_pages.alloc_aligned(Self::TABLE_LAYOUT)
            .ok_or(())
            .map_err(|_| {
                serial::println!("failed to allocate new page table"); })?;

        unsafe {
            let table = (addr as *const PageTable).as_ref_unchecked();
            Ok(memmap::kernel_root(table))
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

        // Shared mmaps stay in kernel_pages so the kernel (net thread,
        // deferred handlers) can deref through KDMAP long after setup.
        // Private mmaps go to user_pages; kernel has no long-lived alias.
        let (pool, alloc_result) = if req.share_with_kernel {
            (Pool::Shared, self.kernel_pages.alloc_aligned(layout))
        } else {
            (Pool::UserOnly, self.user_pages.alloc_aligned(layout))
        };
        let paddr = match alloc_result {
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

        // `paddr` here is the allocator handle — a KDMAP VA while user_pages
        // still has a KDMAP alias; PTE paddrs are physical, so convert.
        let backing_phys = memmap::virt_to_phys_dmap(paddr as u64);

        // Zero before the PTE install exposes the page to user code — a
        // previous owner's contents would otherwise be visible to the new
        // process. Shared pages can be written through KDMAP directly; for
        // user-only pages we go through UserPageWindow so step 8's KDMAP
        // removal doesn't regress this.
        unsafe {
            match pool {
                Pool::Shared => {
                    let kva = memmap::phys_to_virt(backing_phys);
                    core::ptr::write_bytes(kva as *mut u8, 0, layout.size());
                }
                Pool::UserOnly => {
                    let mut w = user_page::UserPageWindow::map(backing_phys, layout.size());
                    w.as_mut_slice().fill(0);
                }
            }
        }

        let config = MappingConfig {
            permissions: (req.page_permissions & 0xE) | PagePermissions::U,
            levels,
            page_size: align as u64,
            vaddr: VirtAddr::new(req.vaddr as u64),
            paddr: PhysAddr::new(backing_phys),
            log: true,
            supervisor_tag
        };

        let vend = VirtAddr::new((req.vaddr + req.size) as u64);
        let pend = PhysAddr::new(backing_phys + req.size as u64);

        unsafe {
            let root_table = memmap::kernel_root_from_pa(thread.root_table_addr() as u64);

            let mut pages = PageAlloc::FA(&mut self.table_pages);

            if let Err(_) = map_address_range(&root_table, &mut pages, &config, vend, pend) {
                serial::println!("failed to map pages for mmap req: {req:?}");
                thread.frame.regs[10] = -4isize as usize;

                match pool {
                    Pool::Shared   => self.kernel_pages.dealloc_aligned(paddr, layout),
                    Pool::UserOnly => self.user_pages.dealloc_aligned(paddr, layout),
                }

                return
            }
        }

        let owning_process = match self.processes.get_mut(&thread.pid) {
            Some(proc) => proc,
            None => {
                serial::println!("failed to add pages to process metadata (no pid): {req:?}");
                thread.frame.regs[10] = -5isize as usize;

                match pool {
                    Pool::Shared   => self.kernel_pages.dealloc_aligned(paddr, layout),
                    Pool::UserOnly => self.user_pages.dealloc_aligned(paddr, layout),
                }

                return
            }
        };

        owning_process.heap_pages.push(PhysBacking {
            paddr: backing_phys,
            layout,
            pool,
        });

        core::sync::atomic::fence(Ordering::SeqCst);
        
        unsafe {
            riscv::asm::sfence_vma(thread.pid as usize, 0);
            riscv::asm::sfence_vma(0, 0);
        }

        serial::println!("fulfilled {req:?}:\n\t0x{paddr:016X} {layout:08X?}");

        thread.frame.regs[10] = 0;
    }

    fn handle_nc_register_req<'t>(&mut self, thread: &'t mut Thread, req: NetChannelRegistrationReq) {
        info!("handling nc registration req: {req:08X?}");

        let nc_kva = unsafe {
            let rpt = memmap::kernel_root_from_pa(thread.root_table_addr() as u64);
            memmap::user_va_to_kdmap(&rpt, req.nc_vaddr as u64)
        };

        // Deferred work: the manager runs under the kernel satp, not the
        // user's, so SUM can't bridge user VAs here. Walk the user PT and
        // stash the KDMAP alias of the NetChannel header — accessors are
        // self-anchored and resolve sub-regions from that base.
        let nc_kva = match nc_kva {
            Some(p) => p,
            None => {
                thread.frame.regs[10] = -1isize as usize;
                return
            }
        };

        info!("nc@kva0x{nc_kva:08X?}");

        let socket_req = SocketReq {
            netchan: nc_kva as *mut NetChannel,
            nc_type: req.nc_type,
            pid: thread.pid
        };

        if let Some(np) = self.net_pkg.socket_reqs.get_mut(get_hart_context().hart_id as usize) {
            if let Err(e) = np.enqueue(socket_req) {
                warn!("failed to queue socket req {e:?}");
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
            (None, 0) => {
                // Kernel thread. Its stack and trap frame were allocated
                // directly from kernel_pages with fixed layouts and aren't
                // recorded in any proc.maps, so free them here.
                let tstack = thread.stack as *const _ as usize;
                self.kernel_pages.dealloc_aligned(tstack, Self::THREAD_STACK_LAYOUT);

                let trap_frame = thread.frame as *const _ as usize;
                self.kernel_pages.dealloc_aligned(trap_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
            }
            (Some(slot), 0) => error!(
                "dealloc_thread: tid{} is a kernel thread but carries slot{}",
                thread.tid, slot),
            (None, pid) => error!(
                "dealloc_thread: tid{} user thread in pid{} is missing its slot",
                thread.tid, pid),
            (Some(slot), pid) => match self.processes.get_mut(&pid) {
                Some(proc) => {
                    let root_table = unsafe {
                        memmap::kernel_root_from_pa((proc.satp.ppn() * PAGE_SIZE) as u64)
                    };

                    // Collect first so we can release the immutable borrow on
                    // proc before tearing down leaves / freeing backings.
                    let mappings: Vec<UserMapping> = proc.mappings_for_slot(slot).copied().collect();

                    for m in &mappings {
                        match m.kind {
                            MappingKind::Stack { .. } => {
                                // Stack is a range of 2 MiB megapages; flush
                                // each page's TLB entry as we tear it down so
                                // nothing survives for slots 2..N.
                                for v in (m.vaddr..m.vaddr + m.len).step_by(UPROC_STACK_GRAIN as usize) {
                                    unsafe {
                                        let _ = unmap_page(&root_table, VirtAddr::new(v), 3);
                                        riscv::asm::sfence_vma(pid as usize, v as usize);
                                    }
                                }
                            }
                            MappingKind::TrapFrame { .. } => {
                                unsafe {
                                    let _ = unmap_page(&root_table, VirtAddr::new(m.vaddr), 4);
                                    riscv::asm::sfence_vma(pid as usize, m.vaddr as usize);
                                }
                            }
                            MappingKind::Guard { .. } => {
                                // No leaf backs the guard; only the proc.maps
                                // entry needs clearing below.
                            }
                            MappingKind::Tls { .. } => {
                                // TODO: unmap TLS leaves once TLS is wired up.
                                // Backing (if any) is still freed by the tail
                                // of this loop, but leaves would leak.
                            }
                            // mappings_for_slot filters on MappingKind::slot(),
                            // which only returns Some for the arms above.
                            MappingKind::Elf
                            | MappingKind::Anon
                            | MappingKind::NetCh { .. } => unreachable!(
                                "mappings_for_slot yielded non-slot kind {:?}", m.kind),
                        }

                        if let Some(b) = m.backing {
                            // PhysBacking.paddr is physical; the allocator's
                            // handles are KDMAP VAs post-migration.
                            let kva = memmap::phys_to_virt(b.paddr) as usize;
                            match b.pool {
                                Pool::Shared => self.kernel_pages.dealloc_aligned(kva, b.layout),
                                // user_pages allocator lands in step 3; no
                                // UserOnly backings exist yet.
                                Pool::UserOnly => self.kernel_pages.dealloc_aligned(kva, b.layout),
                            }
                        }

                        let proc = self.processes.get_mut(&pid).expect("proc vanished mid-teardown");
                        let _ = proc.maps.remove(&m.vaddr);
                    }

                    let proc = self.processes.get_mut(&pid).expect("proc vanished mid-teardown");
                    proc.thread_slots.free(slot);
                }
                None => error!(
                    "dealloc_thread: tid{} references missing pid{}",
                    thread.tid, pid),
            }
        }
    }

    fn dealloc_process(&mut self, mut process: Process) {
        let process_root_table_pa = (process.satp.ppn() * PAGE_SIZE) as u64;

        while let Some(socket_handle) = process.sockets.pop_last() {
            if let Err(e) = self.net_pkg.socket_deletions.enqueue(socket_handle) {
                error!("failed to queue socket for deletion while deallocating pid{}", process.pid);
                self.orphaned_sockets.push(socket_handle);
            }
        }

        let root_table = unsafe { memmap::kernel_root_from_pa(process_root_table_pa) };

        while let Some(b) = process.heap_pages.pop() {
            // PhysBacking.paddr is physical; allocator handles are KDMAP VAs.
            let kva = memmap::phys_to_virt(b.paddr) as usize;
            serial::println!("dealloc heap page pa@{:016X} kva@{:016X} {:08X?} pool={:?}",
                b.paddr, kva, b.layout, b.pool);
            match b.pool {
                Pool::Shared   => self.kernel_pages.dealloc_aligned(kva, b.layout),
                Pool::UserOnly => self.user_pages.dealloc_aligned(kva, b.layout),
            }
        }

        let mut pages = PageAlloc::FA(&mut self.table_pages);
        unsafe {
            unmap(&root_table, &mut pages);
            // table_pages hands out KDMAP VAs, so dealloc takes the KDMAP VA
            // of the root — convert the PA we pulled from satp.
            let root_kva = memmap::phys_to_virt(process_root_table_pa) as usize;
            self.table_pages.dealloc_aligned(root_kva, Self::TABLE_LAYOUT);
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

                    // Read through the existing &Thread — do NOT take a Box
                    // here. The second pass (dealloc_thread + free) runs
                    // after this loop, and needs this Thread's fields still
                    // readable; taking a Box would drop-free it at scope end
                    // and leave a use-after-free in the next pass.
                    match t.fault_info {
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
                            let status = t.frame.regs[11] as isize;
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

            // Now that no kernel state references this Thread, take
            // ownership and drop — frees the heap allocation exactly once.
            drop(unsafe { Box::from_raw(p.0) });
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

    /// Kernel root table as a `RootTable` with the correct PA→VA bias for
    /// tables allocated from `table_pages`. Use this wherever walker/mapper
    /// helpers need to follow intermediate PPNs.
    fn root(&self) -> mmu::mmap::RootTable<'static> {
        unsafe { memmap::kernel_root_from_pa((self.satp.ppn() * PAGE_SIZE) as u64) }
    }
    
    fn setup_igb(&mut self, device: &PciDevice) {
        device.print_info();

        let ort = self.root();

        let bar_kva = unsafe {
            let bar_size = device.get_bar_size(0) as u64;
            if bar_size > (2 * MB) {
                serial::println!("bar2big");
                return
            }

            let mut pages = PageAlloc::FA(&mut self.table_pages);

            serial::println!("mapping {}KB BAR0", bar_size / KB);

            // BAR0's PA stays at IGB_BAR_PA (we still program that into the
            // device's BAR register so the device decodes it on the bus);
            // kernel-side accesses go through a high-half KMMIO alias.
            let kva = match memmap::install_kmmio_alias(
                &ort, &mut pages, Self::IGB_BAR_PA..(Self::IGB_BAR_PA + bar_size)
            ) {
                Ok(v) => v,
                Err(_) => { serial::println!("failed to map bar"); return }
            };

            device.write_bar(0, Self::IGB_BAR_PA as u32);

            riscv::register::satp::write(self.satp);
            riscv::asm::sfence_vma(0, 0);

            kva
        };

        unsafe {
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

            let mut e1000 = E1000::new(bar_kva as *mut u32, tx_ring, tx_bufs, rx_ring, rx_bufs);
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

        // PCI config space lives at a high-half KMMIO alias instead of
        // identity-mapped at its PA — keeps the kernel root free of low-half
        // entries that would shadow user VA space.
        let pci_cfg_va = unsafe {
            let ort = self.root();
            let mut pages = PageAlloc::FA(&mut self.table_pages);
            let va = match memmap::install_kmmio_alias(
                &ort, &mut pages, (base as u64)..((base + size) as u64)
            ) {
                Ok(v) => v,
                Err(_) => {
                    serial::println!("failed to map pci config space");
                    return;
                }
            };
            riscv::asm::sfence_vma(0, 0);
            va
        };

        let matches = pci::scan_pci(pci_cfg_va as usize, &[(0x8086, 0x100E)]);
        if matches.is_empty() {
            return
        }

        self.setup_igb(&matches[0]);
    }
    
    pub fn get_environment_info(&mut self) {
        // Access the DTB through its KDMAP alias — map_kernel_self installs it
        // at phys_to_virt(dtb_phys) and no longer identity-maps the dtb guard.
        let dtb_kva = memmap::phys_to_virt(self.dtb_addr as u64);
        let fdt = unsafe { Fdt::from_raw_unchecked(dtb_kva as *const _) };
        let root = fdt.root();

        let mut nodes: Vec<_> = root.children().collect();
        while let Some(node) = nodes.pop() {
            let name = node.name();
            if name.starts_with("pci") {
                // get_pci_info maps PCI config space itself; no satp gymnastics.
                self.get_pci_info(node);
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

    /// `stack_kva` is the allocator-returned kernel VA (KDMAP). PTE `paddr`
    /// fields need the physical — convert at the boundary.
    fn map_stack(&mut self, root_table: &mmu::mmap::RootTable<'_>, stack_kva: u64, stackv: u64, stack_size: u64) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(&mut self.table_pages);
        let stackp = memmap::virt_to_phys_dmap(stack_kva);
        unsafe {
            map_address_range(
                root_table,
                &mut pages,
                &MappingConfig {
                    permissions: PagePermissions::U | PagePermissions::R | PagePermissions::W,
                    levels: 3, page_size: UPROC_STACK_GRAIN,
                    vaddr: VirtAddr::new(stackv),
                    paddr: PhysAddr::new(stackp),
                    log: false,
                    supervisor_tag: None
                },
                VirtAddr::new(stackv + stack_size),
                PhysAddr::new(stackp + stack_size))
        }
    }

    fn map_trap_frame(&mut self, root_table: &mmu::mmap::RootTable<'_>, trap_frame_kva: usize, user_vaddr: u64) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(&mut self.table_pages);
        let trap_frame = memmap::virt_to_phys_dmap(trap_frame_kva as u64) as usize;
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
    
    pub fn add_new_thread_to_process(&mut self, pid: u16, entrypoint: usize, stack_size: u64) -> Result<(), ()> {
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
            memmap::kernel_root_from_pa(addr as u64)
        };

        let thread = match self.create_new_thread(pid, &root_table, entrypoint, slot, stack_size) {
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
    
    pub fn create_new_thread(&mut self, pid: u16, root_table: &mmu::mmap::RootTable<'_>, entrypoint: usize, slot: u16, stack_size: u64) -> Result<Thread, ()> {
        if !validate_user_stack_size(stack_size) {
            serial::println!("invalid user stack size {stack_size}");
            return Err(())
        }

        let mut page_list: Vec<PhysBacking> = Vec::new();

        let (stackp, stack_layout) = self.allocate_user_thread_stack(stack_size)?;
        page_list.push(PhysBacking {
            paddr: memmap::virt_to_phys_dmap(stackp as u64),
            layout: stack_layout,
            pool: Pool::UserOnly,
        });

        let trap_frame = self.allocate_trap_frame()
            .map_err(|_| {
                self.free_backings(&page_list[..]);
            })?;

        page_list.push(PhysBacking {
            paddr: memmap::virt_to_phys_dmap(trap_frame as u64),
            layout: Self::THREAD_TRAP_FRAME_LAYOUT,
            pool: Pool::Shared,
        });

        let stack_vaddr      = user_stack_vaddr(slot, stack_size);
        let guard_vaddr      = user_stack_guard_vaddr(slot);
        let guard_size       = user_stack_guard_size(stack_size);
        let trap_frame_vaddr = user_trap_frame_vaddr(slot);

        let root_kva = root_table.table as *const _ as usize;
        let root_pa = memmap::virt_to_phys_dmap(root_kva as u64);

        if let Err(_) = self.map_stack(root_table, stackp as u64, stack_vaddr, stack_size) {
            self.free_backings(&page_list[..]);
            self.table_pages.dealloc_aligned(root_kva, Self::TABLE_LAYOUT);

            serial::println!("failed to map stack");

            return Err(())
        }

        if let Err(_) = self.map_trap_frame(root_table, trap_frame, trap_frame_vaddr) {
            self.free_backings(&page_list[..]);
            self.table_pages.dealloc_aligned(root_kva, Self::TABLE_LAYOUT);

            serial::println!("failed to map trap frame");

            return Err(())
        }

        if let Some(proc) = self.processes.get_mut(&pid) {
            // Reserved vaddr range below the stack. No leaves — a fault inside
            // here is a stack overflow, which the page-fault path will turn
            // into a thread kill once it consults proc.maps.
            proc.insert_mapping(UserMapping {
                vaddr:   guard_vaddr,
                len:     guard_size,
                perms:   0,
                backing: None,
                kind:    MappingKind::Guard { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr:   stack_vaddr,
                len:     stack_size,
                perms:   (PagePermissions::U | PagePermissions::R | PagePermissions::W) as u64,
                backing: Some(PhysBacking {
                    // Allocator returns KDMAP VAs; PhysBacking.paddr is
                    // a physical address by contract (used for DMA/PTE
                    // construction during teardown), so convert here.
                    paddr:  memmap::virt_to_phys_dmap(stackp as u64),
                    layout: stack_layout,
                    pool:   Pool::UserOnly,
                }),
                kind:    MappingKind::Stack { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr:   trap_frame_vaddr,
                len:     PAGE_SIZE as u64,
                perms:   PagePermissions::R as u64,
                backing: Some(PhysBacking {
                    paddr:  memmap::virt_to_phys_dmap(trap_frame as u64),
                    layout: Self::THREAD_TRAP_FRAME_LAYOUT,
                    pool:   Pool::Shared,
                }),
                kind:    MappingKind::TrapFrame { slot },
            });
        }

        let tid = self.next_tid();

        let (frame, stack) = unsafe {
            let f = trap_frame as *mut TrapFrame;
            core::ptr::write_bytes(f as *mut u8, 0, PAGE_SIZE);

            // Stack was zeroed inside allocate_user_thread_stack via
            // UserPageWindow — `stackp` is a user_pages KDMAP VA that no
            // longer resolves under any satp, so writing through it here
            // would fault. The &mut Stack reference below is constructed
            // for `Thread.stack` but never dereferenced kernel-side; user
            // code reaches the same backing via the user-VA mapping.
            let s = stackp as *mut Stack;

            (
                f.as_mut_unchecked(),
                s.as_mut_unchecked()
            )
        };

        let mut satp = Satp::from_bits(0);
        satp.set_asid(pid as usize);
        satp.set_mode(riscv::register::satp::Mode::Sv48);
        satp.set_ppn(root_pa as usize / PAGE_SIZE);

        frame.regs[1] = entrypoint;
        frame.regs[2] = (stack_vaddr + stack_size - 16) as usize;
        frame.asid = pid as usize;

        serial::println!("ventry={:016X?},vsp=0x{:016X?},rpt_pa={:016X?}", entrypoint, frame.regs[2], root_pa);

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
    
    pub fn create_new_process(&mut self, elf_blob: &[u8], stack_size: u64) -> Result<(), ()> {
        let root_table = self.create_new_page_table()?;
        let mut elf = self.load_elf(&root_table, elf_blob)?;
        let pid = self.next_pid();

        let root_kva = root_table.table as *const _ as u64;
        let root_pa = memmap::virt_to_phys_dmap(root_kva);

        let mut proc_satp = Satp::from_bits(0);
        proc_satp.set_ppn(root_pa as usize / PAGE_SIZE);
        proc_satp.set_mode(Mode::Sv48);
        proc_satp.set_asid(pid as usize);

        let mut proc = Process::new(pid, proc_satp);
        let slot = proc.thread_slots.alloc().ok_or(())?;

        // ELF segment backings are tracked on the process so dealloc_process
        // returns them to user_pages on teardown — previously dropped on the
        // floor here.
        proc.heap_pages.append(&mut elf.segments);

        // Insert the Process before creating the thread so create_new_thread
        // can record per-thread UserMappings (TrapFrame, eventually Stack/TLS)
        // into proc.maps via self.processes.get_mut.
        self.processes.insert(pid, proc);

        let thread = match self.create_new_thread(pid, &root_table, elf.entrypoint, slot, stack_size) {
            Ok(t) => t,
            Err(e) => {
                let _ = self.processes.remove(&pid);
                return Err(e);
            }
        };
        let tid = thread.tid;

        if let Err(_) = self.map_kernel_into(&root_table) {
            let _ = self.processes.remove(&pid);
            self.table_pages.dealloc_aligned(root_kva as usize, Self::TABLE_LAYOUT);

            serial::println!("failed to map kernel into process");

            return Err(())
        }

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(thread);
        let tptr = Box::into_raw(t);
        serial::println!("created uprocess@{tptr:016X?},pid={pid},tid={tid},table_pa={root_pa:016X?}");

        let proc = self.processes.get_mut(&pid)
            .expect("just inserted");
        proc.threads.insert(tid);
        proc.thread_count = 1;

        self.threads.insert(tid, PThread(tptr));

        Ok(())
    }

    fn free_backings(&mut self, backings: &[PhysBacking]) {
        for b in backings {
            let kva = memmap::phys_to_virt(b.paddr) as usize;
            match b.pool {
                Pool::Shared   => self.kernel_pages.dealloc_aligned(kva, b.layout),
                Pool::UserOnly => self.user_pages.dealloc_aligned(kva, b.layout),
            }
        }
    }
    
    pub fn load_elf(&mut self, root_table: &mmu::mmap::RootTable<'_>, elf_blob: &[u8]) -> Result<orbital_elf::ElfInfo, ()> {
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

            if segment.p_vaddr < USER_TEXT_BASE {
                serial::println!("illegal elf p_vaddr 0x{:X} (below USER_TEXT_BASE 0x{:X})", segment.p_vaddr, USER_TEXT_BASE);
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
                let kva = match self.user_pages.alloc_aligned(layout) {
                    Some(p) => p,
                    None => {
                        self.free_backings(&segment_allocations[..]);
                        serial::println!("failed to alloc segment");
                        return Err(())
                    },
                };
                let paddr_start = memmap::virt_to_phys_dmap(kva as u64);

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

                segment_allocations.push(PhysBacking {
                    paddr: paddr_start,
                    layout,
                    pool: Pool::UserOnly,
                });

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
                    self.free_backings(&segment_allocations);
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

    fn map_kernel_into(&mut self, root_table: &mmu::mmap::RootTable<'_>) -> Result<(), ()> {
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
