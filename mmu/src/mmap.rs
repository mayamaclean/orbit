use core::alloc::Layout;
use core::ops::Range;

#[cfg(feature = "alloc")]
use mem::frame::FrameAllocator;
use serial::println;

use crate::{GB, MB, MappingConfig, PAGE_SIZE, sv48::{PageTable, PhysAddr, VirtAddr}};

pub struct PageTableVec {
    start_addr: usize,
    max_pages: usize,
    current_page: usize
}

impl PageTableVec {
    pub const fn new(start_addr: usize, max_size: usize) -> Self {
        Self {
            start_addr, current_page: 0, max_pages: max_size / crate::PAGE_SIZE
        }
    }

    /// returns Result<page_addr, ()>
    pub unsafe fn allocate_page_table(&mut self) -> Result<&'static PageTable, ()> {
        if self.current_page >= self.max_pages {
            return Err(())
        }

        let current_page_addr = self.start_addr + (self.current_page * crate::PAGE_SIZE);
        self.current_page += 1;

        let table = unsafe {(current_page_addr as *const PageTable).as_ref_unchecked()};
        table.entries.iter().for_each(|e| e.set_raw(0));

        Ok(table)
    }

    pub fn table_count(&self) -> (usize, usize) {
        (self.current_page, self.max_pages)
    }

    pub fn current_tables_size(&self) -> usize {
        self.current_page * PAGE_SIZE
    }
}

/// A root page-table reference paired with the offset to add to any PPN-derived
/// PA to produce the supervisor VA of that intermediate table. Walkers read
/// PTEs (which always store physical PPNs) and follow them by applying the
/// bias; PTE writers invert it to recover the PA from an allocator-returned
/// VA. bl and kmain's early trampoline allocate tables from identity-mapped
/// RAM (bias=0); post-init kmain allocates them from the KDMAP window
/// (bias=kdmap_base - ram_phys_base).
#[derive(Copy, Clone)]
pub struct RootTable<'a> {
    pub table: &'a PageTable,
    pub pa_to_va_bias: u64,
}

impl<'a> RootTable<'a> {
    pub const fn new(table: &'a PageTable, pa_to_va_bias: u64) -> Self {
        Self { table, pa_to_va_bias }
    }

    pub const fn identity(table: &'a PageTable) -> Self {
        Self { table, pa_to_va_bias: 0 }
    }

    #[inline]
    pub fn va_from_pa(&self, pa: u64) -> u64 {
        pa.wrapping_add(self.pa_to_va_bias)
    }

    #[inline]
    pub fn pa_from_va(&self, va: u64) -> u64 {
        va.wrapping_sub(self.pa_to_va_bias)
    }
}

pub enum PageAlloc<'a> {
    PTV(&'a mut PageTableVec),
    #[cfg(feature = "alloc")]
    FA(&'a mut FrameAllocator)
}

impl<'a> PageAlloc<'a> {
    pub const PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE) };

    pub fn allocate_page_table(&mut self) -> Result<&'static PageTable, ()> {
        match self {
            Self::PTV(v) => unsafe {v.allocate_page_table()},
            #[cfg(feature = "alloc")]
            Self::FA(f) => unsafe {
                let ptr = f.alloc_aligned(Self::PAGE_LAYOUT);

                if ptr.is_none() {
                    return Err(())
                }

                let table = (ptr.unwrap() as *const PageTable).as_ref_unchecked();
                table.entries.iter().for_each(|e| e.set_raw(0));

                Ok(table)
            }
        }
    }

    pub fn free_page(&mut self, phys_addr: usize) -> Result<(), ()> {
        match self {
            Self::PTV(_v) => Err(()),
            #[cfg(feature = "alloc")]
            Self::FA(f) => {
                f.dealloc_aligned(phys_addr, Self::PAGE_LAYOUT);
                Ok(())
            }
        }
    }
}

/// Sv48 table depth. Counted so that 3 = root (its PTEs are indexed by
/// vpn3 and point at L2 tables) and 0 = leaf table (its PTEs are
/// 4 KiB-leaf PTEs). For superpages the leaf PTE is installed at
/// the *table* whose depth matches `4 - config.levels`: 4 KiB leaves go
/// in the depth-0 table, 2 MiB superpages in the depth-1 table, 1 GiB in
/// depth-2.
pub type Level = u8;

/// Descend `root` toward the table at `target_level` without allocating
/// or modifying anything. Returns `None` if any PTE above the target is
/// invalid (`V=0`) or a leaf (superpage present above target level).
///
/// # Safety
/// - `root.table` must reference a live Sv48 table.
/// - PTE PPNs along the walk must correspond to live page tables
///   reachable via `root.pa_to_va_bias`.
pub unsafe fn walk_to_table<'a>(
    root: &'a RootTable<'_>,
    vaddr: VirtAddr,
    target_level: Level,
) -> Option<&'a PageTable> {
    let mut table = root.table;
    for lvl in ((target_level + 1)..=3).rev() {
        let idx = vaddr.vpn_n(lvl as usize) as usize;
        let pte = &table.entries[idx];
        if !pte.is_valid() || pte.is_leaf() {
            return None;
        }
        let next_pa = pte.ppn() * PAGE_SIZE as u64;
        let next_va = root.va_from_pa(next_pa) as *const PageTable;
        table = unsafe { next_va.as_ref_unchecked() };
    }
    Some(table)
}

/// Descend `root` toward the table at `target_level`, materializing any
/// missing intermediate tables from `pages`. Errors if an intermediate
/// PTE is a leaf (superpage conflict with a deeper target) or if
/// allocation fails.
///
/// # Safety
/// - Same invariants as [`walk_to_table`].
/// - `pages` must hand back page-table frames that the caller owns for
///   the lifetime of the resulting mappings.
pub unsafe fn walk_to_table_materialize<'a, 'p>(
    root: &'a RootTable<'_>,
    pages: &mut PageAlloc<'p>,
    vaddr: VirtAddr,
    target_level: Level,
    log: bool,
) -> Result<&'a PageTable, ()> {
    let mut table = root.table;
    for lvl in ((target_level + 1)..=3).rev() {
        let idx = vaddr.vpn_n(lvl as usize) as usize;
        let pte = &table.entries[idx];

        if pte.is_leaf() {
            if log { println!("\twalk_materialize: unexpected leaf at level {lvl}"); }
            return Err(());
        }

        table = if !pte.is_valid() {
            let new_table = pages.allocate_page_table()?;
            let new_table_va = new_table as *const _ as u64;
            let new_table_pa = root.pa_from_va(new_table_va);
            let ppn = new_table_pa / PAGE_SIZE as u64;
            pte.set_raw(crate::sv48::PageTableEntry::pack_table(ppn));
            if log {
                println!("\ttable@0x{:08X}[vpn{lvl}={idx}]={:08x}",
                    table as *const _ as usize, new_table_pa);
            }
            new_table
        } else {
            let next_pa = pte.ppn() * PAGE_SIZE as u64;
            if log {
                println!("\ttable@0x{:08X}[vpn{lvl}={idx}]={:08x}",
                    table as *const _ as usize, next_pa);
            }
            let next_va = root.va_from_pa(next_pa) as *const PageTable;
            unsafe { next_va.as_ref_unchecked() }
        };
    }
    Ok(table)
}

pub unsafe fn map_address_page<'a>(root_table: &RootTable<'_>, pages: &mut PageAlloc<'a>, config: &MappingConfig) -> Result<(), ()> {
    if (config.paddr.get_raw() % config.page_size) != 0 || (config.vaddr.get_raw() % config.page_size) != 0 {
        if config.log { println!("misaligned map call: {config:?}"); }
        return Err(())
    }

    if config.log { println!("\n{:08x?}", &config); }

    let target_level = (4 - config.levels) as Level;
    let table = unsafe {
        walk_to_table_materialize(root_table, pages, config.vaddr, target_level, config.log)?
    };

    let idx = config.vaddr.vpn_n(target_level as usize) as usize;
    let pte = &table.entries[idx];

    if pte.is_valid() {
        if config.log { println!("leaf pte already exists"); }
        return Err(())
    }

    let ppn = config.paddr.get_raw() / PAGE_SIZE as u64;
    pte.set_raw(crate::sv48::PageTableEntry::pack_leaf(ppn, config.permissions));

    if config.log {
        println!("\tleaf in table@0x{:08X}[vpn{}={idx}]=0x{:08x}",
            table as *const _ as usize, config.levels, pte.get_raw());
    }

    Ok(())
}

/// Materialize page-table intermediates for `[va_start, va_end)` down to (but
/// not including) the leaf level. Leaf PTEs are left `V=0`, so the VA range
/// walks-to-fault until a caller installs leaves on demand. Used to reserve
/// scratch VA windows whose leaves are written and cleared transiently.
pub unsafe fn reserve_va_range<'a>(
    root_table: &RootTable<'_>,
    pages: &mut PageAlloc<'a>,
    va_start: u64,
    va_end: u64,
) -> Result<(), ()> {
    if (va_start % PAGE_SIZE as u64) != 0 || (va_end % PAGE_SIZE as u64) != 0 {
        return Err(())
    }

    // Advance by 2 MiB per iteration — the span of one L0 table, which the
    // walk ensures exists on the way down.
    const L0_SPAN: u64 = 512 * PAGE_SIZE as u64;
    let mut va = va_start;
    while va < va_end {
        unsafe {
            walk_to_table_materialize(root_table, pages, VirtAddr::new(va), 0, false)?;
        }
        va = va.saturating_add(L0_SPAN);
    }
    Ok(())
}

/// Install a 4 KiB leaf PTE for `vaddr` under `root_table`, pointing at
/// `paddr` with `perms`, or clear the leaf entirely when `paddr` is None.
/// Returns `Err(())` if the walk hits an invalid or leaf PTE above L0 —
/// callers must pre-reserve intermediates for the VA range (e.g. via
/// `reserve_va_range`). The caller owns the post-write `sfence.vma`.
pub unsafe fn write_leaf_pte(
    root_table: &RootTable<'_>,
    vaddr: VirtAddr,
    paddr: Option<PhysAddr>,
    perms: u64,
) -> Result<(), ()> {
    let table = unsafe { walk_to_table(root_table, vaddr, 0) }.ok_or(())?;
    let leaf_idx = vaddr.vpn_n(0) as usize;
    match paddr {
        Some(pa) => {
            let ppn = pa.get_raw() / PAGE_SIZE as u64;
            table.entries[leaf_idx].set_raw(
                crate::sv48::PageTableEntry::pack_leaf(ppn, perms),
            );
        }
        None => table.entries[leaf_idx].set_raw(0),
    }
    Ok(())
}

pub unsafe fn map_address_range<'a>(root_table: &RootTable<'_>, pages: &mut PageAlloc<'a>, config: &MappingConfig, vend: VirtAddr, pend: PhysAddr) -> Result<(), ()> {
    let pstart = config.paddr.get_raw();
    let pend = pend.get_raw();
    let plen = pend - pstart;

    let vstart = config.vaddr.get_raw();
    let vend = vend.get_raw();

    if config.log { println!("map range: p0x{pstart:016X?}..p0x{pend:016X?}, v0x{vstart:016X?}..v0x{vend:016X?}"); }

    let vlen = vend - vstart;

    if vlen != plen {
        if config.log { println!("virtual and physical address ranges are differnt lengths"); }
        return Err(())
    }

    if (pstart % config.page_size) != 0 || (vstart % config.page_size) != 0 {
        if config.log { println!("virtual or physical address was not aligned to requested page size"); }
        return Err(())
    }

    let pages_needed = plen / config.page_size;
    let mut range_config = *config;

    for _ in 0..pages_needed {
        unsafe { map_address_page(root_table, pages, &range_config)?; }
        range_config.paddr = PhysAddr::new(range_config.paddr.get_raw() + config.page_size);
        range_config.vaddr = VirtAddr::new(range_config.vaddr.get_raw() + config.page_size);
    }
    Ok(())
}

/// One step of a full page-table walk — the PTE read at a given level plus
/// its index in its containing table. Read-only snapshot; the returned
/// values reflect the PTE as observed at walk time.
#[derive(Debug, Clone, Copy)]
pub struct WalkStep {
    pub level: u8,
    pub pte_idx: u16,
    pub pte_raw: u64,
}

impl WalkStep {
    pub const EMPTY: Self = Self { level: 0, pte_idx: 0, pte_raw: 0 };
}

/// Top-down Sv48 walk of `root` for `vaddr`, recording every PTE visited.
/// Stops at the first invalid or leaf PTE. Returns the number of steps
/// written into `out` (1..=4); `out[n - 1]` is the terminal PTE.
pub unsafe fn walk_pte_chain(
    root_table: &RootTable<'_>,
    vaddr: VirtAddr,
    out: &mut [WalkStep; 4],
) -> usize {
    let mut table = root_table.table;
    let mut n = 0;
    for level in (0..=3).rev() {
        let idx = vaddr.vpn_n(level) as usize;
        let pte = &table.entries[idx];
        let raw = pte.get_raw();
        out[n] = WalkStep {
            level: level as u8,
            pte_idx: idx as u16,
            pte_raw: raw,
        };
        n += 1;
        let valid = (raw & 1) != 0;
        let leaf = (raw & 0xE) != 0;
        if !valid || leaf {
            return n;
        }
        let next_pa = (pte.get_ppn() as u64) << 2;
        let next_va = root_table.va_from_pa(next_pa);
        table = unsafe { (next_va as *const PageTable).as_ref_unchecked() };
    }
    n
}

#[unsafe(no_mangle)]
pub unsafe fn virt_to_phys(root_table: &RootTable<'_>, vaddr: VirtAddr) -> Option<usize> {
    let mut chain = [WalkStep::EMPTY; 4];
    let n = unsafe { walk_pte_chain(root_table, vaddr, &mut chain) };
    let terminal = chain[n - 1];
    let valid = (terminal.pte_raw & 1) != 0;
    let leaf = (terminal.pte_raw & 0xE) != 0;
    if !valid || !leaf { return None; }
    // Shift PPN from in-place (bit 10) down and back up to PA (bit 12):
    // net shift by 2 is equivalent to `(pte_raw >> 10) * PAGE_SIZE`.
    let phys_base = (terminal.pte_raw & !crate::sv48::PageTableEntry::STATUS_BITS_MASK) << 2;
    // 4 KiB-leaf assumption: superpage leaves would need a bigger
    // offset mask. Orbit has no user superpages today.
    Some(phys_base as usize + vaddr.page_offset() as usize)
}

/// Walk to the leaf PTE for `vaddr` at `levels` granularity and clear it.
/// Returns `Err(())` if the walk hits an invalid PTE or encounters a leaf at
/// an unexpected level — the latter prevents accidentally clobbering a
/// superpage that covers more than the caller intends to unmap.
pub unsafe fn unmap_page(root_table: &RootTable<'_>, vaddr: VirtAddr, levels: usize) -> Result<(), ()> {
    let target_level = (4 - levels) as Level;
    let table = unsafe { walk_to_table(root_table, vaddr, target_level) }.ok_or(())?;
    let idx = vaddr.vpn_n(target_level as usize) as usize;
    let pte = &table.entries[idx];
    if !pte.is_valid() || !pte.is_leaf() {
        return Err(())
    }
    pte.set_raw(0);
    Ok(())
}

/// Clear leaf PTEs covering every 4 KiB page in `range`. Both endpoints must
/// be page-aligned; the caller is responsible for `sfence.vma` afterwards.
pub unsafe fn unmap_range(root_table: &RootTable<'_>, range: Range<u64>) -> Result<(), ()> {
    if (range.start % PAGE_SIZE as u64) != 0 || (range.end % PAGE_SIZE as u64) != 0 {
        return Err(())
    }
    let mut addr = range.start;
    while addr < range.end {
        unsafe { unmap_page(root_table, VirtAddr::new(addr), 4)? }
        addr += PAGE_SIZE as u64;
    }
    Ok(())
}

/// Free every intermediate (non-leaf) page-table owned by this satp, except
/// the root itself — the caller frees that. Leaf PTEs aren't invalidated
/// and leaf-backed physical pages aren't freed here: some leaves (kernel
/// .text, pools, MMIO) are shared by PPN across every satp, and the
/// user-owned leaves are freed separately from their respective pools via
/// `PhysBacking` teardown before this call.
///
/// Every intermediate reached via the recursion IS per-satp — both for the
/// user half and the kernel-shared half, since `map_kernel_shared` allocates
/// fresh L1/L2/L3 tables from `table_pages` for each new root. So walking
/// all 512 root entries is correct: we free the per-process intermediates,
/// and the shared leaves hang off PPNs we never deref.
pub unsafe fn unmap<'a>(root_table: &RootTable<'_>, pages: &mut PageAlloc<'a>) {
    unsafe { unmap_subtree(root_table, pages) }
}

unsafe fn unmap_subtree<'a>(table: &RootTable<'_>, pages: &mut PageAlloc<'a>) {
    for entry in table.table.entries.iter() {
        if !entry.is_valid() || entry.is_leaf() {
            continue
        }

        let next_pa = (entry.get_ppn() as u64) << 2;
        let next_va = table.va_from_pa(next_pa);
        let child_table = unsafe {
            (next_va as *const PageTable).as_ref_unchecked()
        };
        let child = RootTable::new(child_table, table.pa_to_va_bias);

        unsafe { unmap_subtree(&child, pages) }

        let _ = pages.free_page(next_va as usize);
    }
}

/// Like [`map_address_page`] but overwrites any existing leaf PTE at the
/// target slot rather than failing. Used by the bulk-mapping helpers
/// ([`map_va_range`], [`id_map_range`]) which iterate over fresh VA
/// ranges owned by the caller and don't expect prior contents.
pub unsafe fn map_page<'a>(root_table: &RootTable<'_>, pages: &mut PageAlloc<'a>, config: &MappingConfig) -> Result<(), ()> {
    if (config.paddr.get_raw() % config.page_size) != 0 || (config.vaddr.get_raw() % config.page_size) != 0 {
        if config.log { println!("misaligned map call: {config:?}"); }
        return Err(())
    }

    if config.log { println!("\n{:08x?}", &config); }

    let target_level = (4 - config.levels) as Level;
    let table = unsafe {
        walk_to_table_materialize(root_table, pages, config.vaddr, target_level, config.log)?
    };

    let idx = config.vaddr.vpn_n(target_level as usize) as usize;
    let pte = &table.entries[idx];
    let ppn = config.paddr.get_raw() / PAGE_SIZE as u64;
    pte.set_raw(crate::sv48::PageTableEntry::pack_leaf(ppn, config.permissions));

    if config.log {
        println!("\tleaf in table@0x{:08X}[vpn{}={idx}]=0x{:08x}",
            table as *const _ as usize, target_level, pte.get_raw());
    }

    Ok(())
}

#[derive(Debug)]
pub struct IdMapReport {
    gigapages_mapped: u64,
    megapages_mapped: u64,
    regularpages_mapped: u64
}

impl Default for IdMapReport {
    fn default() -> Self {
        IdMapReport { gigapages_mapped: 0, megapages_mapped: 0, regularpages_mapped: 0 }
    }
}

/// Map a physical range to a distinct virtual start, auto-selecting gigapage
/// / megapage / 4 KiB leaves by the alignment and remaining length — the same
/// tiering as `id_map_range` but without the VA == PA assumption. `permissions`,
/// `log`, and `supervisor_tag` are read from `base_config`; its `vaddr`/`paddr`/
/// `levels`/`page_size` fields are overwritten per step.
pub unsafe fn map_va_range<'a>(
    root_table: &RootTable<'_>,
    pages: &mut PageAlloc<'a>,
    base_config: MappingConfig,
    va_start: u64,
    pa_range: Range<u64>,
) -> Result<IdMapReport, ()> {
    let mut report = IdMapReport::default();

    if pa_range.end < pa_range.start {
        if base_config.log { println!("bad map range: {pa_range:016X?}"); }
        return Err(());
    }

    let mut cur_pa = pa_range.start;
    let mut cur_va = va_start;
    let mut rem = pa_range.end - pa_range.start;

    while rem >= PAGE_SIZE as u64 {
        let mut levels = 4;
        let mut page_size = PAGE_SIZE as u64;

        if rem >= GB && (cur_pa % GB) == 0 && (cur_va % GB) == 0 {
            levels = 2;
            page_size = GB;
        } else if rem >= (2 * MB) && (cur_pa % (2 * MB)) == 0 && (cur_va % (2 * MB)) == 0 {
            levels = 3;
            page_size = 2 * MB;
        }

        let config = MappingConfig {
            paddr: PhysAddr::new(cur_pa),
            vaddr: VirtAddr::new(cur_va),
            levels,
            page_size,
            permissions: base_config.permissions,
            log: base_config.log,
            supervisor_tag: base_config.supervisor_tag,
        };

        if unsafe { map_page(root_table, pages, &config).is_err() } {
            if base_config.log { println!("failed to map v0x{cur_va:016X}..p0x{cur_pa:016X}"); }
            return Err(());
        }

        match levels {
            4 => report.regularpages_mapped += 1,
            3 => report.megapages_mapped += 1,
            2 => report.gigapages_mapped += 1,
            _ => unreachable!(),
        }

        cur_pa += page_size;
        cur_va += page_size;
        rem -= page_size;
    }
    Ok(report)
}

/// base config for permissions
pub unsafe fn id_map_range<'a>(root_table: &RootTable<'_>, pages: &mut PageAlloc<'a>, base_config: MappingConfig, range: Range<u64>) -> Result<IdMapReport, ()> {
    let mut id_report = IdMapReport::default();

    if range.end < range.start {
        if base_config.log { println!("bad mmap range: {range:016X?} {base_config:?}"); }
        return Err(())
    }

    if base_config.log { println!("starting id map: {base_config:016?}"); }

    let mut cur_page_addr = range.start;
    let mut rem_ram = range.end - range.start;
    while rem_ram >= 4096 {
        let mut levels = 4;
        let mut page_size = PAGE_SIZE as u64;

        if rem_ram >= GB && (cur_page_addr % GB) == 0 {
            levels = 2;
            page_size = GB as u64;
            
        }
        else if rem_ram >= (2 * MB) && (cur_page_addr % (2 * MB)) == 0{
            levels = 3;
            page_size = (2 * MB) as u64;
        }

        let paddr = PhysAddr::new(cur_page_addr as u64);
        let vaddr = VirtAddr::new(cur_page_addr as u64);

        let config = MappingConfig {
            paddr, vaddr, levels, page_size,
            permissions: base_config.permissions,
            log: base_config.log,
            supervisor_tag: base_config.supervisor_tag
        };

        if unsafe { map_page(root_table, pages, &config).is_err() } {
            if base_config.log { println!("failed to id map 0x{:016x?}", cur_page_addr); }
            return Err(())
        }
        else {
            match levels {
                4 => id_report.regularpages_mapped += 1,
                3 => id_report.megapages_mapped += 1,
                2 => id_report.gigapages_mapped += 1,
                _ => unreachable!()
            }
        }

        //println!("mapped page, {rem_ram}B remaining");

        cur_page_addr += page_size;
        rem_ram -= page_size;
    }
    Ok(id_report)
}
