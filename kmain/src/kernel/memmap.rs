use core::ops::Range;

use mem::round_u64_up;
use mmu::mmap::{PageAlloc, id_map_range, unmap_range};
use mmu::sv48::{PageTable, PhysAddr, VirtAddr};
use mmu::{MappingConfig, PAGE_SIZE, PagePermissions};

pub const KRX: u64 =
    PagePermissions::R as u64 | PagePermissions::X as u64 | PagePermissions::G as u64;
pub const KRW: u64 =
    PagePermissions::R as u64 | PagePermissions::W as u64 | PagePermissions::G as u64;
pub const KRO: u64 = PagePermissions::R as u64 | PagePermissions::G as u64;

pub const CLINT_MSIP_BASE: u64 = 0x0200_0000;
pub const ACLINT_SSWI_BASE: u64 = 0x02F0_0000;

unsafe extern "C" {
    unsafe static _text_start: u8;
    unsafe static _text_end: u8;
    unsafe static _rodata_start: u8;
    unsafe static _rodata_end: u8;
    unsafe static _data_start: u8;
    unsafe static _data_end: u8;
    unsafe static _bss_start: u8;
    unsafe static _bss_end: u8;
    unsafe static _got_start: u8;
    unsafe static _got_end: u8;

    unsafe static _reladyn_start: u8;
    unsafe static _reladyn_end: u8;
    unsafe static _gnuhash_start: u8;
    unsafe static _gnuhash_end: u8;
    unsafe static _dynsym_start: u8;
    unsafe static _dynsym_end: u8;
    unsafe static _hash_start: u8;
    unsafe static _hash_end: u8;
    unsafe static _dynstr_start: u8;
    unsafe static _dynstr_end: u8;
    unsafe static _ehframe_start: u8;
    unsafe static _ehframe_end: u8;
    unsafe static _DYNAMIC: u8;
    unsafe static _DYNAMIC_END: u8;
}

#[derive(Debug, Clone)]
pub struct KernelLayout {
    pub kheap: Range<u64>,
    pub kpages: Range<u64>,
    pub ktables: Range<u64>,
    pub dtb: Range<u64>,
    pub serial: u64,
}

struct Region {
    range: Range<u64>,
    perms: u64,
    name: &'static str,
}

fn section_range(start: u64, end: u64) -> Range<u64> {
    start..round_u64_up(end, PAGE_SIZE as u64)
}

unsafe fn kernel_elf_regions() -> [Region; 5] {
    unsafe {
        [
            Region {
                range: section_range(&_text_start as *const _ as u64, &_text_end as *const _ as u64),
                perms: KRX,
                name: ".text",
            },
            Region {
                range: section_range(&_rodata_start as *const _ as u64, &_rodata_end as *const _ as u64),
                perms: KRO,
                name: ".rodata",
            },
            Region {
                range: section_range(&_data_start as *const _ as u64, &_data_end as *const _ as u64),
                perms: KRW,
                name: ".data",
            },
            Region {
                range: section_range(&_bss_start as *const _ as u64, &_bss_end as *const _ as u64),
                perms: KRW,
                name: ".bss",
            },
            Region {
                range: section_range(&_got_start as *const _ as u64, &_got_end as *const _ as u64),
                perms: KRO,
                name: ".got",
            },
        ]
    }
}

fn mmio_regions(layout: &KernelLayout) -> [Region; 3] {
    [
        Region {
            range: layout.serial..layout.serial + PAGE_SIZE as u64,
            perms: KRW,
            name: "serial",
        },
        Region {
            range: CLINT_MSIP_BASE..CLINT_MSIP_BASE + PAGE_SIZE as u64,
            perms: KRW,
            name: "clint.msip",
        },
        Region {
            range: ACLINT_SSWI_BASE..ACLINT_SSWI_BASE + PAGE_SIZE as u64,
            perms: KRW,
            name: "aclint.sswi",
        },
    ]
}

fn pool_regions(layout: &KernelLayout) -> [Region; 2] {
    [
        Region {
            range: layout.kheap.clone(),
            perms: KRW,
            name: "kheap",
        },
        Region {
            range: layout.kpages.clone(),
            perms: KRW,
            name: "kpages",
        },
    ]
}

unsafe fn map_region(rt: &PageTable, pa: &mut PageAlloc, r: &Region) -> Result<(), ()> {
    let cfg = MappingConfig {
        permissions: r.perms,
        levels: 4,
        page_size: PAGE_SIZE as u64,
        vaddr: VirtAddr::new(0),
        paddr: PhysAddr::new(0),
        log: false,
        supervisor_tag: None,
    };
    match unsafe { id_map_range(rt, pa, cfg, r.range.clone()) } {
        Ok(_) => Ok(()),
        Err(_) => {
            serial::println!("memmap: failed mapping {} {:016X?}", r.name, r.range);
            Err(())
        }
    }
}

/// Map the minimal kernel surface a user process needs so that S-mode trap
/// handling executes correctly under its satp. Kernel code executing under
/// a user satp dereferences GOT on essentially every call, reads/writes the
/// heap (tracing/log), touches hart contexts + trap frames + thread stacks
/// (all in kpages), and prints to serial. It also pokes CLINT MSIPs and the
/// ACLINT SSWI to clear/send IPIs from the scheduler.
pub unsafe fn map_kernel_shared(
    rt: &PageTable,
    pa: &mut PageAlloc,
    layout: &KernelLayout,
) -> Result<(), ()> {
    unsafe {
        for r in kernel_elf_regions().iter() {
            map_region(rt, pa, r)?;
        }
        for r in pool_regions(layout).iter() {
            map_region(rt, pa, r)?;
        }
        for r in mmio_regions(layout).iter() {
            map_region(rt, pa, r)?;
        }
    }
    Ok(())
}

unsafe fn boot_only_elf_regions() -> [Region; 7] {
    unsafe {
        [
            Region {
                range: section_range(&_reladyn_start as *const _ as u64, &_reladyn_end as *const _ as u64),
                perms: KRO,
                name: ".rela.dyn",
            },
            Region {
                range: section_range(&_gnuhash_start as *const _ as u64, &_gnuhash_end as *const _ as u64),
                perms: KRO,
                name: ".gnu.hash",
            },
            Region {
                range: section_range(&_dynsym_start as *const _ as u64, &_dynsym_end as *const _ as u64),
                perms: KRO,
                name: ".dynsym",
            },
            Region {
                range: section_range(&_hash_start as *const _ as u64, &_hash_end as *const _ as u64),
                perms: KRO,
                name: ".hash",
            },
            Region {
                range: section_range(&_dynstr_start as *const _ as u64, &_dynstr_end as *const _ as u64),
                perms: KRO,
                name: ".dynstr",
            },
            Region {
                range: section_range(&_ehframe_start as *const _ as u64, &_ehframe_end as *const _ as u64),
                perms: KRO,
                name: ".eh_frame",
            },
            Region {
                range: section_range(&_DYNAMIC as *const _ as u64, &_DYNAMIC_END as *const _ as u64),
                perms: KRO,
                name: ".dynamic",
            },
        ]
    }
}

fn self_only_pool_regions(layout: &KernelLayout) -> [Region; 2] {
    [
        Region {
            range: layout.ktables.clone(),
            perms: KRW,
            name: "ktables",
        },
        Region {
            range: layout.dtb.clone(),
            perms: KRO,
            name: "dtb",
        },
    ]
}

/// Map every kernel region the S-mode kernel needs under its own satp. This
/// is a superset of `map_kernel_shared`: on top of the process-visible surface
/// it adds the ktables page-table pool (so the kernel can walk/modify satp
/// tables) and the DTB (parsed once during boot), along with dynamic-link
/// sections consumed during self-relocation.
pub unsafe fn map_kernel_self(
    rt: &PageTable,
    pa: &mut PageAlloc,
    layout: &KernelLayout,
) -> Result<(), ()> {
    unsafe {
        map_kernel_shared(rt, pa, layout)?;
        for r in boot_only_elf_regions().iter() {
            map_region(rt, pa, r)?;
        }
        for r in self_only_pool_regions(layout).iter() {
            map_region(rt, pa, r)?;
        }
    }
    Ok(())
}

/// Drop the boot-only ELF mappings (`.rela.dyn`, `.dynamic`, `.dynsym`,
/// `.dynstr`, `.hash`, `.gnu.hash`, `.eh_frame`) from `rt`. These are only
/// consumed by self-relocation, which runs before paging is enabled — by the
/// time the kernel is running under `satp`, nothing reads them. Caller is
/// responsible for `sfence.vma` if `rt` is the active root.
pub unsafe fn unmap_boot_only_regions(rt: &PageTable) -> Result<(), ()> {
    unsafe {
        for r in boot_only_elf_regions().iter() {
            if let Err(_) = unmap_range(rt, r.range.clone()) {
                serial::println!("memmap: failed unmapping {} {:016X?}", r.name, r.range);
                return Err(())
            }
        }
    }
    Ok(())
}
