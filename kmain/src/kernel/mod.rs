use core::alloc::Layout;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;

use alloc::collections::{btree_map::BTreeMap};
use alloc::{boxed::Box, vec::Vec};

use device::{HartContext, Stack, TrapFrame};
use dtoolkit::fdt::FdtNode;
use dtoolkit::{Node, fdt::{Fdt}};
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
    Frame, MappingKind, PThread, PhysBacking, Process,
    Shared, Thread, ThreadState,
    UserMapping, UserOnly
};

use orbit_abi::errno::{
    Errno, EAGAIN, EBADF, EFAULT, EINVAL, EIO, ENOEXEC, ENOMEM, ESRCH,
};
use orbit_core::{
    CloseHandleReq, CreateProcessReq, MemMapReq, NetChannelCreationReq, PendingWork,
};
use thingbuf::StaticThingBuf;

use crate::kernel::handle::{Handle, ProcessHandles};
use crate::kernel::memmap::FrameToKdmap;
use crate::kernel::shared_user_ptr::SharedUserPtr;
use riscv::register::satp::{Mode, Satp};
use riscv::register::sstatus::SPP;
use smoltcp::iface::{Config, Interface, SocketHandle};
use smoltcp::wire::{EthernetAddress};
use tracing::{error, info, warn};

use crate::drivers::e1000::{
    E1000, E1000Pbuf, RX_RING_BUFS_BYTES,
    RX_RING_BYTES, RX_RING_LEN, RxDesc,
    TX_RING_BUFS_BYTES, TX_RING_BYTES,
    TX_RING_LEN, TxDesc
};

use crate::kernel::context::get_hart_context;
use crate::kernel::pci::PciDevice;
use crate::{NetPackage, SocketReq};

pub mod context;
pub mod handle;
pub mod input;
pub mod memmap;
pub mod orbital_elf;
pub mod pending_frees;
pub mod shared_user_ptr;
pub mod pci;
pub mod stdin;
pub mod user_page;

pub use memmap::KernelLayout;

// TODO: page unmapping

// Default build embeds orbit-loader as the initial user program —
// listens on TCP :7777 and spawns ELFs via create_process, so umode
// rebuilds don't drag kmain+bl along. The `smoke` Cargo feature swaps
// in umode directly so ./smoke can run the automated self-test without
// a host-side sender (and without the network-ready latency).
#[cfg(not(feature = "smoke"))] 
pub const UMODE_TEST_ELF: &'static [u8] = include_bytes!("../../../orbit-loader/target/riscv64gc-unknown-none-elf/release/orbit-loader");
#[cfg(feature = "smoke")]
pub const UMODE_TEST_ELF: &'static [u8] = include_bytes!("../../../umode/target/riscv64gc-unknown-none-elf/release/umode");

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
    orphaned_sockets: Vec<SocketHandle>,

    /// Per-process handle tables. The manager's strong refs on
    /// `SharedUserPtr`-backed resources live here, keyed by the u32 Fd
    /// assigned at creation. k_net gets separate clones via
    /// `SocketReq`. On process exit the table is walked to revoke
    /// every Shared mapping before the manager drops its Arcs.
    process_handles: BTreeMap<u16, ProcessHandles>,
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
        table_pages: memmap::TablePages,
        kernel_pages: memmap::KernelPages,
        user_pages: memmap::UserPages,
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
            orphaned_sockets: Vec::new(),
            process_handles: BTreeMap::new(),
        }
    }

    /// Allocate a kthread stack. Kernel-accessible (Shared pool) so the
    /// kernel can write through KDMAP during setup.
    fn allocate_thread_stack(&mut self) -> Result<(Frame<Shared>, memmap::KdmapVa), ()> {
        self.kernel_pages.alloc_kdmap(Self::THREAD_STACK_LAYOUT)
            .ok_or_else(|| { error!("failed to allocate new thread stack"); })
    }

    /// Allocate a user thread stack. `user_pages` has no KDMAP alias in
    /// the kernel satp — setup-time zeroing goes through `UserPageWindow`.
    fn allocate_user_thread_stack(&mut self, stack_size: u64) -> Result<(Frame<UserOnly>, Layout), ()> {
        let layout = Layout::from_size_align(stack_size as usize, UPROC_STACK_GRAIN as usize)
            .map_err(|e| {
                error!("bad user stack layout for size={stack_size}: {e:?}");
            })?;
        let frame = self.user_pages.alloc_pa(layout)
            .ok_or_else(|| {
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
        self.kernel_pages.alloc_kdmap(Self::THREAD_TRAP_FRAME_LAYOUT)
            .ok_or_else(|| { error!("failed to allocate new trap frame"); })
    }

    /// Allocate a fresh page table from `table_pages` and return a
    /// `RootTable` view on it. The page is zeroed before handoff.
    fn create_new_page_table(&mut self) -> Result<(Frame<process::Table>, mmu::mmap::RootTable<'static>), ()> {
        let (frame, kva) = self.table_pages.alloc(Self::TABLE_LAYOUT)
            .ok_or_else(|| { error!("failed to allocate new page table"); })?;
        unsafe {
            core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, PAGE_SIZE);
            let table = kva.as_ptr::<PageTable>().as_ref_unchecked();
            Ok((frame, memmap::kernel_root(table)))
        }
    }

    pub fn create_kernel_thread(&mut self, entrypoint: usize, a0: Option<usize>) -> Result<(), ()> {
        if self.current_process_id == u16::MAX {
            error!("too many processes running to spawn another");
            return Err(())
        }
        
        let (stack_frame, stack_kva) = self.allocate_thread_stack()?;

        let (_trap_frame_frame, trap_frame_kva) = match self.allocate_trap_frame() {
            Ok(p) => p,
            Err(_) => {
                self.kernel_pages.free(stack_frame, Self::THREAD_STACK_LAYOUT);
                error!("failed to alloc trap_frame for kthread");
                return Err(())
            }
        };

        let pid = 0;
        let tid = self.next_tid();

        let (frame, stack) = unsafe {
            let f = trap_frame_kva.as_mut_ptr::<TrapFrame>();
            core::ptr::write_bytes(f as *mut u8, 0, PAGE_SIZE);

            let s = stack_kva.as_mut_ptr::<Stack>();
            core::ptr::write_bytes(s as *mut u8, 0, 2 * MB as usize);

            (
                f.as_mut_unchecked(),
                s.as_mut_unchecked()
            )
        };

        frame.regs[1] = entrypoint;
        frame.regs[2] = stack_kva.raw() as usize + Self::THREAD_STACK_LAYOUT.size();
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
            handle: None,
            slot: None,
            fault_info: None,
        };

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(kthread);
        let tptr = Box::into_raw(t);
        info!("created kthread@{:016X?}", tptr);

        self.threads.insert(tid, PThread(tptr));

        Ok(())
    }

    fn run_mmap_req(&mut self, req: MemMapReq, pid: u16, root_pa: u64) -> isize {
        info!("handling mmap req {req:08X?}");

        let Some(orbit_core::manager::MappingGeometry { align, levels }) =
            orbit_core::manager::select_mapping_geometry(req.vaddr, req.size)
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
        let (backing_pa_raw, backing) = if req.share_with_kernel {
            let Some(frame) = self.kernel_pages.alloc_pa(layout) else {
                error!("failed to alloc shared pages for mmap req: {req:?}");
                return Errno::new(ENOMEM).to_ret();
            };
            // Zero via KDMAP alias.
            unsafe {
                let kva = frame.to_kdmap();
                core::ptr::write_bytes(kva.as_mut_ptr::<u8>(), 0, layout.size());
            }
            (frame.get_raw(), PhysBacking::Shared { frame, layout })
        } else {
            let Some(frame) = self.user_pages.alloc_pa(layout) else {
                error!("failed to alloc user pages for mmap req: {req:?}");
                return Errno::new(ENOMEM).to_ret();
            };
            // Zero via a transient kernel window — no KDMAP alias exists.
            unsafe {
                let mut w = user_page::UserPageWindow::map(frame.get_raw(), layout.size());
                w.as_mut_slice().fill(0);
            }
            (frame.get_raw(), PhysBacking::User { frame, layout })
        };

        let supervisor_tag = if req.share_with_kernel {
            SupervisorTag::SharedRevocable
        } else {
            SupervisorTag::None
        };

        let config = MappingConfig {
            permissions: (req.page_permissions & 0xE) | PagePermissions::U,
            levels,
            page_size: align as u64,
            vaddr: VirtAddr::new(req.vaddr as u64),
            paddr: PhysAddr::new(backing_pa_raw),
            log: false,
            supervisor_tag
        };

        let vend = VirtAddr::new((req.vaddr + req.size) as u64);
        let pend = PhysAddr::new(backing_pa_raw + req.size as u64);

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

        riscv::asm::sfence_vma(pid as usize, 0);
        riscv::asm::sfence_vma(0, 0);

        info!("fulfilled {req:?}:\n\tpa=0x{backing_pa_raw:016X} {layout:08X?}");

        0
    }

    /// Dispatch a single typed free based on the backing's pool variant.
    fn free_backing(&mut self, backing: PhysBacking) {
        match backing {
            PhysBacking::Shared { frame, layout } => self.kernel_pages.free(frame, layout),
            PhysBacking::User   { frame, layout } => self.user_pages.free(frame, layout),
        }
    }

    /// Run an enqueued NetChannel creation. Returns `(vaddr, fd)` on
    /// success — the manager forwards both via `signal_pair`. Negative
    /// `vaddr` on the error path; `fd` is unused in that case.
    fn run_nc_create_req(&mut self, req: NetChannelCreationReq, pid: u16, root_pa: u64) -> (isize, isize) {
        info!("handling nc creation req: {req:08X?}");

        let Some(region_size) = NetChannel::normalize_region_size(req.region_size) else {
            warn!("nc create: bad region_size {}", req.region_size);
            return (Errno::new(EINVAL).to_ret(), 0);
        };

        if req.nc_vaddr % PAGE_SIZE != 0 {
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
        let Some((frame, kva)) = self.kernel_pages.alloc_kdmap(layout) else {
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
            vaddr: VirtAddr::new(req.nc_vaddr as u64),
            paddr: frame.raw(),
            log: false,
            supervisor_tag: SupervisorTag::SharedRevocable,
        };

        let vend = VirtAddr::new((req.nc_vaddr + region_size) as u64);
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
        let shared: SharedUserPtr<NetChannel> = SharedUserPtr::new(
            frame, layout, req.nc_vaddr as u64, region_size, pid);

        // Register the manager's strong ref and grab the Fd. Return it
        // to the user in a1 alongside the VA in a0 — avoids taking a
        // user out-pointer, which would have to resolve through KDMAP
        // (Shared-pool only) or a transient UserPageWindow, neither of
        // which is worth the machinery for 4 bytes.
        let fd = self.process_handles
            .entry(pid)
            .or_insert_with(ProcessHandles::new)
            .insert(Handle::NetChannel(shared.clone()));

        core::sync::atomic::fence(Ordering::SeqCst);

        riscv::asm::sfence_vma(pid as usize, 0);

        let socket_req = SocketReq {
            netchan: shared,
            nc_type: req.nc_type,
            pid,
            pending_rx_ack: false,
            pending_tx_ack: false,
            issued_desired: 0,
        };

        if let Some(np) = self.net_pkg.socket_reqs.get_mut(get_hart_context().hart_id as usize) {
            if let Err(e) = np.enqueue(socket_req) {
                warn!("nc create: failed to queue socket req {e:?}");
                return (Errno::new(EAGAIN).to_ret(), 0);
            }
        }

        info!("nc created user_va=0x{:08X} kva=0x{:016X} region={} fd={}",
            req.nc_vaddr, kva.raw(), region_size, fd);
        (req.nc_vaddr as isize, fd as isize)
    }

    fn run_close_req(&mut self, req: CloseHandleReq, pid: u16, root_pa: u64) -> isize {
        info!("handling close req: {req:?}");

        // Look up the handle, revoke if Shared, then drop the Arc.
        // k_net may still hold a clone; the backing lives until it's
        // dropped too. Post-revoke, any user access to the old VA
        // faults, and `try_as_ref` returns None for future kernel
        // observers — close is safe to race against an in-flight
        // update_tcp on another hart.
        let Some(ph) = self.process_handles.get_mut(&pid) else {
            return Errno::new(EBADF).to_ret();
        };
        let Some(handle) = ph.remove(req.fd) else {
            return Errno::new(EBADF).to_ret();
        };

        let root_table = unsafe { memmap::kernel_root_from_pa(root_pa) };

        match &handle {
            Handle::NetChannel(sup) => {
                if let Err(e) = sup.revoke(&root_table) {
                    warn!("close_handle: revoke failed for fd={} sup={sup:?}: {e:?}",
                        req.fd);
                    return Errno::new(EIO).to_ret();
                }
            }
        }

        // `handle` drops here, releasing the manager's Arc. If k_net
        // still holds a clone the backing survives until its next
        // drop.
        drop(handle);
        0
    }

    fn run_create_process_req(&mut self, req: CreateProcessReq, root_pa: u64) -> isize {
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
        let mut copied = 0usize;
        while copied < req.elf_len {
            let cursor = req.elf_vaddr + copied;
            let page_base = cursor & !(PAGE_SIZE - 1);
            let page_off = cursor - page_base;
            let take = core::cmp::min(PAGE_SIZE - page_off, req.elf_len - copied);

            let pa = match unsafe {
                mmu::mmap::virt_to_phys(&root_table, VirtAddr::new(page_base as u64))
            } {
                Some(p) => p as u64,
                None => {
                    error!("create_process: user va 0x{:X} does not translate", page_base);
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

        match self.create_new_process(&blob, UPROC_STACK_DEFAULT) {
            Ok(pid) => {
                info!("create_process: spawned pid={pid} from {} bytes", blob.len());
                pid as isize
            }
            Err(()) => {
                error!("create_process: create_new_process failed");
                Errno::new(ENOEXEC).to_ret()
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
                PendingWork::MemMap { req, pid, root_pa, handle } => {
                    let result = self.run_mmap_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::NetChannelCreation { req, pid, root_pa, handle } => {
                    let (r, e) = self.run_nc_create_req(req, pid, root_pa);
                    handle.signal_pair(r, e);
                }
                PendingWork::CloseHandle { req, pid, root_pa, handle } => {
                    let result = self.run_close_req(req, pid, root_pa);
                    handle.signal(result);
                }
                PendingWork::CreateProcess { req, root_pa, handle, .. } => {
                    let result = self.run_create_process_req(req, root_pa);
                    handle.signal(result);
                }
            }
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
                // Thread parked on a CompletionHandle; resume it once
                // the handle is signaled, writing exactly the
                // a-registers the handler claimed via signal_n(N).
                // Slots a-regs the handler did not claim retain their
                // trap-entry snapshot — preserves caller-saved regs
                // that user code (e.g. orbit-loader) may depend on
                // surviving across the ecall in practice.
                let Some(handle) = thread.handle.as_ref() else {
                    error!("thread{} Blocking with no handle", thread.tid);
                    continue;
                };
                if !handle.is_signaled() {
                    continue;
                }
                let n = handle.ret_count();
                for i in 0..n {
                    thread.frame.regs[10 + i] = handle.ret(i) as usize;
                }
                let logged = handle.ret(0);
                thread.handle = None;
                info!("unblocked thread{} (handle, n={}, a0={})",
                    thread.tid, n, logged);
                thread.state.store(ThreadState::Ready as usize, Ordering::Release);
                return Some(PThread(p.0));
            }
        }
        None
    }

    fn dealloc_thread(&mut self, thread: &'static Thread) {
        match (thread.slot, thread.pid) {
            (None, 0) => {
                // Kernel thread. Its stack and trap frame were allocated
                // directly from kernel_pages with fixed layouts and aren't
                // recorded in any proc.maps, so free them here. The
                // Thread references them by KDMAP VA; reverse through
                // KdmapVa → PhysAddr → Frame<Shared> at the boundary.
                let tstack_kva = memmap::KdmapVa::new(thread.stack as *const _ as u64);
                self.kernel_pages.free(Frame::<Shared>::new(tstack_kva.to_phys()), Self::THREAD_STACK_LAYOUT);

                let trap_frame_kva = memmap::KdmapVa::new(thread.frame as *const _ as u64);
                self.kernel_pages.free(Frame::<Shared>::new(trap_frame_kva.to_phys()), Self::THREAD_TRAP_FRAME_LAYOUT);
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

                    // Two passes: gather the vaddrs matching this slot
                    // (u64 is Copy so the collect doesn't tangle with
                    // proc's borrow), then pull each UserMapping out of
                    // proc.maps by `remove` — that transfers ownership
                    // of its `backing: Option<PhysBacking>`, which we
                    // can hand to `free_backing`. Single copy avoided
                    // because `PhysBacking` (and therefore UserMapping)
                    // is no longer Copy.
                    let vaddrs: Vec<u64> = proc.mappings_for_slot(slot)
                        .map(|m| m.vaddr)
                        .collect();

                    for v in &vaddrs {
                        let proc = self.processes.get_mut(&pid)
                            .expect("proc vanished mid-teardown");
                        let Some(m) = proc.maps.remove(v) else { continue };

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
                            self.free_backing(b);
                        }
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
                error!("failed to queue socket for deletion while deallocating pid{} ({e:?})", process.pid);
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
        if let Some(ph) = self.process_handles.get(&process.pid) {
            for (_fd, handle) in ph.iter() {
                match handle {
                    Handle::NetChannel(sup) => {
                        if let Err(e) = sup.revoke(&root_table) {
                            warn!(
                                "dealloc_process: revoke failed for pid{} sup={sup:?}: {e:?}",
                                process.pid,
                            );
                        }
                    }
                }
            }
        }
        // Drop the manager's Arcs. k_net still holds its own clones via
        // `user_conns`; those drop later when `socket_deletions`
        // removes them. When *both* sides have released, the
        // SharedInner Drop fires and pushes the backing onto
        // `pending_frees`.
        let _ = self.process_handles.remove(&process.pid);

        while let Some(b) = process.heap_pages.pop() {
            info!("dealloc heap page pa@{:016X} {:08X?} pool={}",
                b.pa().get_raw(), b.layout(), b.pool_name());
            self.free_backing(b);
        }

        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe {
            // Detach the shared KMMIO L2 first — `unmap` is recursive and
            // would otherwise descend into and free the shared subtree,
            // corrupting every other satp's KMMIO surface.
            memmap::detach_shared_kmmio_l2(&root_table);
            unmap(&root_table, &mut pages);
            // table_pages now returns typed frames — the walker's
            // `free_page` takes a raw PA directly; the root was allocated
            // from this pool so we reconstruct a `Frame<Table>` here.
            self.table_pages.free(
                Frame::<process::Table>::new(PhysAddr::new(process_root_table_pa)),
                Self::TABLE_LAYOUT);
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
                            warn!(
                                "tid{} killed: {} cause={} epc={:#x} stval={:#x}",
                                t.tid, label, f.cause, f.epc, f.stval);
                        }
                        None => {
                            let status = t.frame.regs[11] as isize;
                            info!("tid{} dead, removing status={status}", t.tid);
                        }
                    }
                }

                if !proc.threads.is_empty() || t.pid == 0 {
                    continue
                }
            }

            info!("pid{} dead, removing", t.pid);

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

        // Drain SharedUserPtr Drops that landed since the last pass.
        // Each queued item is a `Frame<Shared>` whose last Arc just
        // dropped on some hart — return it to `kernel_pages` here,
        // under the Orbit lock, not in Drop context.
        let kpages = &mut self.kernel_pages;
        pending_frees::drain(|frame, layout| {
            info!("dealloc shared ptr backing pa@{:016X} {:08X?}",
                frame.get_raw(), layout);
            kpages.free(frame, layout);
        });
    }
    
    pub fn assign_threads(&mut self, context: &'static HartContext) {
        use orbit_core::sched::HartView;

        // `sscratch` on this hart points at its own HartContext inside the
        // contiguous array allocated at boot; subtract the hart id to get
        // the array base, then index for each remote. Built lazily so no
        // per-tick allocation happens.
        let hart_root = unsafe {
            (riscv::register::sscratch::read() as *const HartContext)
                .sub(context.hart_id as usize)
        };
        let self_hart_id = context.hart_id as usize;
        let cpu_count = self.cpu_count;

        let self_view = HartView {
            hart_id: context.hart_id as u32,
            current: &context.current,
        };
        let remotes = (0..cpu_count).filter(move |&i| i != self_hart_id).map(move |i| {
            let hc = unsafe { hart_root.add(i).as_ref_unchecked() };
            HartView {
                hart_id: hc.hart_id as u32,
                current: &hc.current,
            }
        });

        let mut hw = crate::hw::RiscvHardware;
        orbit_core::sched::assign_threads(&self_view, remotes, self, &mut hw);
    }

    pub fn print_threads(&self) {
        for (_, t) in self.threads.iter() {
            let thread = unsafe {
                (t.0 as *const Thread).as_ref_unchecked()
            };

            info!("tid{}: state{}", thread.tid, thread.state.load(Ordering::Acquire));
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
                error!("bar2big");
                return
            }

            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());

            info!("mapping {}KB BAR0", bar_size / KB);

            // BAR0's PA stays at IGB_BAR_PA (we still program that into the
            // device's BAR register so the device decodes it on the bus);
            // kernel-side accesses go through a high-half KMMIO alias.
            let kva = match memmap::install_kmmio_alias(
                &ort, &mut pages, Self::IGB_BAR_PA..(Self::IGB_BAR_PA + bar_size)
            ) {
                Ok(v) => v,
                Err(_) => { error!("failed to map bar"); return }
            };

            device.write_bar(0, Self::IGB_BAR_PA as u32);

            riscv::register::satp::write(self.satp);
            riscv::asm::sfence_vma(0, 0);

            kva
        };

        unsafe {
            let (_, tx_ring_kva) = self.kernel_pages.alloc_kdmap(
                Layout::from_size_align_unchecked(TX_RING_BYTES, PAGE_SIZE))
                .expect("no e1000 tx ring");
            let tx_ring = tx_ring_kva.as_mut_ptr::<[TxDesc; TX_RING_LEN]>().as_mut_unchecked();

            let (_, rx_ring_kva) = self.kernel_pages.alloc_kdmap(
                Layout::from_size_align_unchecked(RX_RING_BYTES, PAGE_SIZE))
                .expect("no e1000 rx ring");
            let rx_ring = rx_ring_kva.as_mut_ptr::<[RxDesc; RX_RING_LEN]>().as_mut_unchecked();

            let (_, tx_bufs_kva) = self.kernel_pages.alloc_kdmap(
                Layout::from_size_align_unchecked(TX_RING_BUFS_BYTES, PAGE_SIZE))
                .expect("no e1000 tx bufs");
            let tx_bufs = tx_bufs_kva.as_mut_ptr::<[E1000Pbuf; TX_RING_LEN]>().as_mut_unchecked();

            let (_, rx_bufs_kva) = self.kernel_pages.alloc_kdmap(
                Layout::from_size_align_unchecked(RX_RING_BUFS_BYTES, PAGE_SIZE))
                .expect("no e1000 rx bufs");
            let rx_bufs = rx_bufs_kva.as_mut_ptr::<[E1000Pbuf; RX_RING_LEN]>().as_mut_unchecked();

            let mut e1000 = E1000::new(bar_kva as *mut u32, tx_ring, tx_bufs, rx_ring, rx_bufs);
            let mac = e1000.read_mac().unwrap();
            if let Err(_) = e1000.init_hw(mac) {
                // free everything ig
                error!("failed to init e1000");
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
                error!("failed to create knet thread");
            }
            else {
                info!("created knet thread");
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

        info!("reg={reg:?}");

        let base = match reg.address::<u64>() {
            Ok(b) => b as usize,
            Err(_) => return
        };

        let size = match reg.size::<u64>() {
            Ok(b) => b as usize,
            Err(_) => return
        };

        info!("pci@{:08X}..{:08X}", base, base+size);

        // PCI config space lives at a high-half KMMIO alias instead of
        // identity-mapped at its PA — keeps the kernel root free of low-half
        // entries that would shadow user VA space.
        let pci_cfg_va = unsafe {
            let ort = self.root();
            let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
            let va = match memmap::install_kmmio_alias(
                &ort, &mut pages, (base as u64)..((base + size) as u64)
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
            return
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

        let mut nodes: Vec<_> = root.children().collect();
        while let Some(node) = nodes.pop() {
            let name = node.name();
            if name.starts_with("pci") {
                // get_pci_info maps PCI config space itself; no satp gymnastics.
                self.get_pci_info(node);
                continue
            }
            if name.starts_with("plic") {
                self.setup_plic(&fdt);
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

        // Setup virtio devices last so PLIC is already installed by
        // the plic node match above — input registers an IRQ handler
        // and a future gpu IRQ wake will too.
        self.discover_virtio(&fdt);
        self.setup_virtio_gpu();
        self.setup_virtio_input();
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
        let Some(outcome) = crate::drivers::virtio_gpu_dev::setup_virtio_gpu(
            &mut self.kernel_pages,
        ) else {
            return;
        };

        // Build the Display + GpuPackage, hand ownership to k_gpu.
        let fb = unsafe {
            crate::drivers::fb::FrameBuffer::new(
                outcome.fb_kva,
                outcome.width,
                outcome.height,
            )
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

    /// `stack_pa` is the physical base of the user stack. User PT leaves
    /// take PAs directly.
    fn map_stack(&mut self, root_table: &mmu::mmap::RootTable<'_>, stack_pa: u64, stackv: u64, stack_size: u64) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe {
            map_address_range(
                root_table,
                &mut pages,
                &MappingConfig {
                    permissions: PagePermissions::U | PagePermissions::R | PagePermissions::W,
                    levels: 3, page_size: UPROC_STACK_GRAIN,
                    vaddr: VirtAddr::new(stackv),
                    paddr: PhysAddr::new(stack_pa),
                    log: false,
                    supervisor_tag: SupervisorTag::None
                },
                VirtAddr::new(stackv + stack_size),
                PhysAddr::new(stack_pa + stack_size))
        }
    }

    fn map_trap_frame(&mut self, root_table: &mmu::mmap::RootTable<'_>, trap_frame_pa: u64, user_vaddr: u64) -> Result<(), ()> {
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        let trap_frame = trap_frame_pa as usize;
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
                    supervisor_tag: SupervisorTag::None
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
        info!("created uthread@{tptr:016X?},pid={pid},tid={tid},table={rpt:016X?}");

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
            error!("invalid user stack size {stack_size}");
            return Err(())
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
        let stack_pa = stack_frame.get_raw();
        let tf_pa = tf_frame.get_raw();

        let stack_vaddr      = user_stack_vaddr(slot, stack_size);
        let guard_vaddr      = user_stack_guard_vaddr(slot);
        let guard_size       = user_stack_guard_size(stack_size);
        let trap_frame_vaddr = user_trap_frame_vaddr(slot);

        // Root table PA is what satp wants. The `RootTable` handle stores
        // a `&PageTable` at its KDMAP alias — reverse that to a `Frame<Table>`.
        let root_kva = memmap::KdmapVa::new(root_table.table as *const _ as u64);
        let root_frame = Frame::<process::Table>::new(root_kva.to_phys());
        let root_ppn = root_frame.get_raw() as usize / PAGE_SIZE;

        if let Err(_) = self.map_stack(root_table, stack_pa, stack_vaddr, stack_size) {
            self.user_pages.free(stack_frame, stack_layout);
            self.kernel_pages.free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
            self.table_pages.free(root_frame, Self::TABLE_LAYOUT);

            error!("failed to map stack");

            return Err(())
        }

        if let Err(_) = self.map_trap_frame(root_table, tf_pa, trap_frame_vaddr) {
            self.user_pages.free(stack_frame, stack_layout);
            self.kernel_pages.free(tf_frame, Self::THREAD_TRAP_FRAME_LAYOUT);
            self.table_pages.free(root_frame, Self::TABLE_LAYOUT);

            error!("failed to map trap frame");

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
                backing: Some(PhysBacking::User { frame: stack_frame, layout: stack_layout }),
                kind:    MappingKind::Stack { slot },
            });
            proc.insert_mapping(UserMapping {
                vaddr:   trap_frame_vaddr,
                len:     PAGE_SIZE as u64,
                perms:   PagePermissions::R as u64,
                backing: Some(PhysBacking::Shared { frame: tf_frame, layout: Self::THREAD_TRAP_FRAME_LAYOUT }),
                kind:    MappingKind::TrapFrame { slot },
            });
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
            let s = stack_pa as *mut Stack;

            (
                f.as_mut_unchecked(),
                s.as_mut_unchecked()
            )
        };

        let mut satp = Satp::from_bits(0);
        satp.set_asid(pid as usize);
        satp.set_mode(riscv::register::satp::Mode::Sv48);
        satp.set_ppn(root_ppn);

        frame.regs[1] = entrypoint;
        frame.regs[2] = (stack_vaddr + stack_size - 16) as usize;
        frame.asid = pid as usize;

        info!("ventry={:016X?},vsp=0x{:016X?},rpt_pa={:016X?}", entrypoint, frame.regs[2], root_frame.get_raw());

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
            handle: None,
            slot: Some(slot),
            fault_info: None,
        })
    }
    
    pub fn create_new_process(&mut self, elf_blob: &[u8], stack_size: u64) -> Result<u16, ()> {
        let (root_pa, root_table) = self.create_new_page_table()?;
        let mut elf = self.load_elf(&root_table, elf_blob)?;
        let pid = self.next_pid();

        let mut proc_satp = Satp::from_bits(0);
        proc_satp.set_ppn(root_pa.get_raw() as usize / PAGE_SIZE);
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
            self.table_pages.free(root_pa, Self::TABLE_LAYOUT);

            error!("failed to map kernel into process");

            return Err(())
        }

        // TODO: figure out why pin<box<thread>> doesnt work
        // or move this to a pool
        let t = Box::new(thread);
        let tptr = Box::into_raw(t);
        info!("created uprocess@{tptr:016X?},pid={pid},tid={tid},table_pa={:016X?}", root_pa.get_raw());

        let proc = self.processes.get_mut(&pid)
            .expect("just inserted");
        proc.threads.insert(tid);
        proc.thread_count = 1;

        self.threads.insert(tid, PThread(tptr));

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

        Ok(pid)
    }

    fn free_backings(&mut self, backings: Vec<PhysBacking>) {
        for b in backings {
            self.free_backing(b);
        }
    }
    
    pub fn load_elf(&mut self, root_table: &mmu::mmap::RootTable<'_>, elf_blob: &[u8]) -> Result<orbital_elf::ElfInfo, ()> {
        let elf = match elf::ElfBytes::<LittleEndian>::minimal_parse(elf_blob) {
            Ok(e) => e,
            Err(e) => { error!("failed to parse umode elf: {e:?}"); return Err(()) }
        };

        let mut segment_allocations = Vec::new();

        let segments = elf.segments().unwrap();
        for segment in segments.iter() {
            let load_segment = segment.p_type == elf::abi::PT_LOAD;
            if !load_segment {
                continue
            }

            if segment.p_vaddr < USER_TEXT_BASE {
                error!("illegal elf p_vaddr 0x{:X} (below USER_TEXT_BASE 0x{:X})", segment.p_vaddr, USER_TEXT_BASE);
                return Err(())
            }

            if segment.p_memsz == 0 {
                continue
            }

            info!("loading {segment:08x?}");

            let segment_data = match elf.segment_data(&segment) {
                Ok(seg) => seg,
                Err(e) => {
                    error!("error parsing loadable segment data: {e:?}");
                    return Err(())
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
                        return Err(())
                    },
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

                segment_allocations.push(PhysBacking::User { frame: seg_pa, layout });

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
                    supervisor_tag: SupervisorTag::None
                };

                let map = map_address_range(
                    root_table,
                    &mut pages,
                    &config,
                    VirtAddr::new(vaddr_end),
                    PhysAddr::new(paddr_end));

                if map.is_err() {
                    self.free_backings(segment_allocations);
                    error!("failed to map segment into process");
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
        let mut pages = PageAlloc::FA(self.table_pages.frames_mut());
        unsafe { memmap::map_kernel_shared(root_table, &mut pages, &self.layout, /*is_kernel_root=*/ false) }
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

impl orbit_core::sched::Scheduler for Orbit {
    fn next_runnable(&mut self) -> Option<*mut Thread> {
        // PThread wraps a raw ptr sourced from the thread registry (Box
        // allocations); returning it directly keeps provenance rooted
        // at that allocation — no `&mut` reborrow whose tag would be
        // popped on return (which would dangle the ptr stored in the
        // target hart's `current` slot).
        self.get_runnable_thread().map(|pt| pt.0)
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
