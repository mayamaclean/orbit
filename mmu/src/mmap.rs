use core::alloc::Layout;
use core::{ops::Range, sync::atomic::Ordering};

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

pub unsafe fn map_address_page<'a>(root_table: &PageTable, pages: &mut PageAlloc<'a>, config: &MappingConfig) -> Result<(), ()> {
    if (config.paddr.get_raw() % config.page_size) != 0 || (config.vaddr.get_raw() % config.page_size) != 0 {
        if config.log { println!("misaligned map call: {config:?}"); }
        return Err(())
    }

    if config.log { println!("\n{:08x?}", &config); }

    let mut current_table = root_table;
    for level in 0..(config.levels - 1) {
        let lidx = 4 - level - 1;
        let idx = config.vaddr.vpn_n(lidx);
        let pte = &current_table.entries[idx as usize];
        
        current_table = if !pte.is_valid() {
            let new_page_table = pages.allocate_page_table()?;
            let new_table_addr = new_page_table as *const _ as u64;

            // set ppn of root entry for secondary table + valid bit
            pte.set_raw(((new_table_addr / crate::PAGE_SIZE as u64) << 10) | 1);

            if config.log { println!("\ttable@0x{:08X}[vpn{lidx}={idx}]={:08x}", current_table as *const _ as usize, new_table_addr); }

            new_page_table
        }
        else {
            let raw = (pte.ppn() * crate::PAGE_SIZE as u64) as *const PageTable;

            if config.log { println!("\ttable@0x{:08X}[vpn{lidx}={idx}]={:08x}", current_table as *const _ as usize, raw as *const _ as usize); }

            unsafe {
                raw.as_ref_unchecked()
            }
        };
    }

    // current table should now be at the table containing our leaves
    let idx = config.vaddr.vpn_n(4 - config.levels);
    let pte = &current_table.entries[idx as usize];

    if pte.is_valid() {
        if config.log { println!("leaf pte already exists"); }
        return Err(())
    }

    pte.set_raw(0);
    pte.set_ppn(config.paddr.get_raw() / PAGE_SIZE as u64);
    pte.set_raw(pte.get_raw() | config.permissions as u64);
    pte.set_accessed(true);
    pte.set_dirty(true);
    pte.set_valid(true);

    if config.log { println!("\tleaf in table@0x{:08X}[vpn{}={idx}]=0x{:08x}", current_table as *const _ as usize, config.levels, pte.get_raw()); }

    Ok(())
}

pub unsafe fn map_address_range<'a>(root_table: &PageTable, pages: &mut PageAlloc<'a>, config: &MappingConfig, vend: VirtAddr, pend: PhysAddr) -> Result<(), ()> {
    let pstart = config.paddr.get_raw();
    let pend = pend.get_raw();
    let plen = pend - pstart;

    let vstart = config.vaddr.get_raw();
    let vend = vend.get_raw();

    serial::println!("map range: p0x{pstart:016X?}..p0x{pend:016X?}, v0x{vstart:016X?}..v0x{vend:016X?}");

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
    let range_config = config.copy();

    for _ in 0..pages_needed {
        unsafe { map_address_page(root_table, pages, &range_config)?; }
        range_config.paddr.a.fetch_add(config.page_size, Ordering::AcqRel);
        range_config.vaddr.a.fetch_add(config.page_size, Ordering::AcqRel);
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe fn virt_to_phys(root_table: &PageTable, vaddr: VirtAddr) -> Option<usize> {
    //serial::println!("translating {vaddr:016X?} with table@{:016X?}", root_table as *const PageTable);

    let mut current_table = root_table;
    for l in (0..=3).rev() {
        let index = vaddr.vpn_n(l) as usize;
        let entry = &current_table.entries[index];
        let phys = (entry.get_ppn() as usize) << 2;

        /*
        serial::println!("translating l{} t0x{:016X?}[{}]=0x{:016X?},phys=0x{:016X?}",
            l,
            current_table as *const PageTable,
            index,
            entry.get_raw(),
            phys);
        */

        if entry.is_leaf() {
            let page_offset = vaddr.page_offset() as usize;
            return Some(phys + page_offset)
        }

        current_table = unsafe {
            (phys as *const PageTable)
                .as_ref_unchecked()
        };

        //serial::println!("current_table@{:016X?}", current_table as *const PageTable);
    }
    None
}

/// Walk to the leaf PTE for `vaddr` at `levels` granularity and clear it.
/// Returns `Err(())` if the walk hits an invalid PTE or encounters a leaf at
/// an unexpected level — the latter prevents accidentally clobbering a
/// superpage that covers more than the caller intends to unmap.
pub unsafe fn unmap_page(root_table: &PageTable, vaddr: VirtAddr, levels: usize) -> Result<(), ()> {
    let target_level = 4 - levels;
    let mut current_table = root_table;
    for lidx in ((target_level + 1)..=3).rev() {
        let idx = vaddr.vpn_n(lidx) as usize;
        let pte = &current_table.entries[idx];
        if !pte.is_valid() || pte.is_leaf() {
            return Err(())
        }
        let next = (pte.ppn() * PAGE_SIZE as u64) as *const PageTable;
        current_table = unsafe { next.as_ref_unchecked() };
    }
    let idx = vaddr.vpn_n(target_level) as usize;
    let pte = &current_table.entries[idx];
    if !pte.is_valid() || !pte.is_leaf() {
        return Err(())
    }
    pte.set_raw(0);
    Ok(())
}

/// Clear leaf PTEs covering every 4 KiB page in `range`. Both endpoints must
/// be page-aligned; the caller is responsible for `sfence.vma` afterwards.
pub unsafe fn unmap_range(root_table: &PageTable, range: Range<u64>) -> Result<(), ()> {
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

pub unsafe fn unmap<'a>(root_table: &PageTable, pages: &mut PageAlloc<'a>) {
    for (_, entry) in root_table.entries.iter().enumerate() {
        if !entry.is_valid() {
            continue
        }

        if !entry.is_leaf() {
            let next_table_addr = (entry.get_ppn() as usize) << 2;
            let table = unsafe {
                (next_table_addr as *const PageTable)
                    .as_ref_unchecked()
            };

            unsafe { unmap(table, pages); }

            serial::println!("freeing t{:016X?}", root_table as *const _);
            let _ = pages.free_page(root_table as *const _ as usize);
        }
    }
}

pub unsafe fn map_page<'a>(root_table: &PageTable, pages: &mut PageAlloc<'a>, config: &MappingConfig) -> Result<(), ()> {
    if (config.paddr.get_raw() % config.page_size) != 0 || (config.vaddr.get_raw() % config.page_size) != 0 {
        if config.log { println!("misaligned map call: {config:?}"); }
        return Err(())
    }

    if config.log { println!("\n{:08x?}", &config); }

    let target_level = 4 - config.levels;

    let mut current_table = root_table;
    for lidx in ((target_level+ 1)..=3).rev() {
        let idx = config.vaddr.vpn_n(lidx);
        let pte = &current_table.entries[idx as usize];

        if pte.is_leaf() {
            if config.log { println!("\tfound leaf pte where we didnt expect one"); }
            return Err(())
        }
        
        current_table = if !pte.is_valid() {
            let new_page_table = pages.allocate_page_table()?;
            let new_table_addr = new_page_table as *const _ as u64;

            // set ppn of root entry for secondary table + valid bit
            pte.set_raw(((new_table_addr / crate::PAGE_SIZE as u64) << 10) | 1);

            if config.log { println!("\ttable@0x{:08X}[vpn{lidx}={idx}]={:08x}", current_table as *const _ as usize, new_table_addr); }

            new_page_table
        }
        else {
            let raw = (pte.ppn() * crate::PAGE_SIZE as u64) as *const PageTable;

            if config.log { println!("\ttable@0x{:08X}[vpn{lidx}={idx}]={:08x}", current_table as *const _ as usize, raw as *const _ as usize); }

            unsafe {
                raw.as_ref_unchecked()
            }
        };
    }

    // current table should now be at the table containing our leaves
    let idx = config.vaddr.vpn_n(target_level);
    let pte = &current_table.entries[idx as usize];

    pte.set_raw(0);
    pte.set_ppn(config.paddr.get_raw() / PAGE_SIZE as u64);
    pte.set_raw(pte.get_raw() | config.permissions as u64);
    pte.set_accessed(true);
    pte.set_dirty(true);
    pte.set_valid(true);

    if config.log { println!("\tleaf in table@0x{:08X}[vpn{}={idx}]=0x{:08x}", current_table as *const _ as usize, target_level, pte.get_raw()); }

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

/// base config for permissions
pub unsafe fn id_map_range<'a>(root_table: &PageTable, pages: &mut PageAlloc<'a>, base_config: MappingConfig, range: Range<u64>) -> Result<IdMapReport, ()> {
    let mut id_report = IdMapReport::default();

    if range.end < range.start {
        println!("bad mmap range: {range:016X?} {base_config:?}");
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
