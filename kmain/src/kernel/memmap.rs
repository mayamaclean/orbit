use core::ops::Range;
use core::sync::atomic::{AtomicU64, Ordering};

use mem::round_u64_up;
use mmu::mmap::{PageAlloc, RootTable, id_map_range, map_va_range, unmap_range, virt_to_phys};
use mmu::sv48::{PageTable, PhysAddr, VirtAddr};
use mmu::{MappingConfig, PAGE_SIZE, PagePermissions};

pub const KRX: u64 =
    PagePermissions::R as u64 | PagePermissions::X as u64 | PagePermissions::G as u64;
pub const KRW: u64 =
    PagePermissions::R as u64 | PagePermissions::W as u64 | PagePermissions::G as u64;
pub const KRO: u64 = PagePermissions::R as u64 | PagePermissions::G as u64;

pub const CLINT_MSIP_BASE: u64 = 0x0200_0000;
pub const ACLINT_SSWI_BASE: u64 = 0x02F0_0000;

// Kernel link base — must match `. = 0x1000;` in kmain/memory.x. The kernel
// stays linked at low-half; relocation slide at runtime is `ktext_base -
// LINK_BASE`, applied by the post-trampoline walker.
pub const LINK_BASE: u64 = 0x1000;

// Nominal high-half base values. Fixed for non-KASLR; a future randomizer
// picks different values at boot and feeds them into `init_layout`. The
// three windows are 16 GiB apart — well over the range any of them will
// occupy — so they can't collide.
pub const KTEXT_NOMINAL: u64 = 0xFFFF_FFC0_0000_0000;
pub const KDMAP_NOMINAL: u64 = 0xFFFF_FFD0_0000_0000;
pub const KMMIO_NOMINAL: u64 = 0xFFFF_FFE0_0000_0000;

// KMMIO window slot assignments. Three devices for now — UART, CLINT MSIP,
// ACLINT SSWI — each getting one 4 KiB page.
#[inline] pub fn kmmio_uart()  -> u64 { kmmio_base() }
#[inline] pub fn kmmio_clint() -> u64 { kmmio_base() + PAGE_SIZE as u64 }
#[inline] pub fn kmmio_sswi()  -> u64 { kmmio_base() + 2 * PAGE_SIZE as u64 }

// Runtime-parameterized kernel address-space layout. Set once by `init_layout`
// during early `rust_main` on hart 0, before any other hart is woken. Reads
// from other harts are safe with Relaxed because the hart-wake IPI is the
// synchronizing event.
static RAM_PHYS_BASE:    AtomicU64 = AtomicU64::new(0);
static KTEXT_BASE:       AtomicU64 = AtomicU64::new(0);
static KDMAP_BASE:       AtomicU64 = AtomicU64::new(0);
static KMMIO_BASE:       AtomicU64 = AtomicU64::new(0);
// Physical address of `_text_start` — the base the kernel ELF was loaded at.
// Post-trampoline `&_text_start as u64` returns the high-half VA, so helpers
// that need the physical (PT construction, DMA setup) read this instead.
static KERNEL_PHYS_BASE: AtomicU64 = AtomicU64::new(0);

pub fn init_layout(ram_phys: u64, ktext: u64, kdmap: u64, kmmio: u64, kernel_phys: u64) {
    RAM_PHYS_BASE.store(ram_phys, Ordering::Relaxed);
    KTEXT_BASE.store(ktext, Ordering::Relaxed);
    KDMAP_BASE.store(kdmap, Ordering::Relaxed);
    KMMIO_BASE.store(kmmio, Ordering::Relaxed);
    KERNEL_PHYS_BASE.store(kernel_phys, Ordering::Relaxed);
}

#[inline] pub fn ram_phys_base()    -> u64 { RAM_PHYS_BASE.load(Ordering::Relaxed) }
#[inline] pub fn ktext_base()       -> u64 { KTEXT_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kdmap_base()       -> u64 { KDMAP_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kmmio_base()       -> u64 { KMMIO_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kernel_phys_base() -> u64 { KERNEL_PHYS_BASE.load(Ordering::Relaxed) }

/// Translate a physical address in the RAM region to its direct-map VA.
#[inline]
pub fn phys_to_virt(pa: u64) -> u64 {
    kdmap_base() + (pa - ram_phys_base())
}

/// Translate a direct-map VA back to its physical address.
#[inline]
pub fn virt_to_phys_dmap(va: u64) -> u64 {
    ram_phys_base() + (va - kdmap_base())
}

/// Resolve a user VA under `root_table` to the kernel's KDMAP alias of the
/// same physical backing. Lets syscall handlers dereference user buffers
/// without SUM and without identity-mapping kpages — the KDMAP VA is a
/// supervisor mapping over the same RAM page.
///
/// # Safety
/// Walks `root_table`'s PTE tree, which must remain valid for the duration
/// of the call.
#[inline]
pub unsafe fn user_va_to_kdmap(root_table: &RootTable<'_>, user_va: u64) -> Option<u64> {
    unsafe { virt_to_phys(root_table, VirtAddr::new(user_va)) }
        .map(|pa| phys_to_virt(pa as u64))
}

/// PA → KDMAP VA offset shared by every kernel-allocated page table. Lets
/// the kernel wrap a raw root-table PA (from `satp.ppn()` or similar) in a
/// `RootTable` without threading the bias through every caller.
#[inline]
pub fn kernel_table_bias() -> u64 {
    kdmap_base().wrapping_sub(ram_phys_base())
}

/// Construct a `RootTable` from a PA that points into `ktables`. Derefs the
/// KDMAP alias of the table.
///
/// # Safety
/// `pa` must be the physical address of a page in the kernel's ktables pool,
/// and its KDMAP alias must be currently mapped under the active satp.
#[inline]
pub unsafe fn kernel_root_from_pa<'a>(pa: u64) -> RootTable<'a> {
    let bias = kernel_table_bias();
    let table = unsafe { (pa.wrapping_add(bias) as *const PageTable).as_ref_unchecked() };
    RootTable::new(table, bias)
}

/// Construct a `RootTable` over an already-dereferenced kernel-owned table.
#[inline]
pub fn kernel_root<'a>(table: &'a PageTable) -> RootTable<'a> {
    RootTable::new(table, kernel_table_bias())
}

/// Translate a linked kernel VA (as seen in symbols like `_text_start`) to
/// the high-half VA the kernel will execute from. Under the identity layout
/// used in Phase 1 this is the identity function.
#[inline]
pub fn linked_to_high_half(linked_va: u64) -> u64 {
    ktext_base() + (linked_va - LINK_BASE)
}

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

// Kernel physical pool sizes. Each pool is 128 MiB; DTB sits in a 2 MiB
// guard at the top of RAM. Layout grows downward from the DTB guard:
//   [kpages][kheap][ktables][dtb guard]  with `mem_end = ram_end - DTB_GUARD`.
pub const KTABLES_SIZE:   u64 = 128 * mmu::MB;
pub const KHEAP_SIZE:     u64 = 128 * mmu::MB;
pub const KPAGES_SIZE:    u64 = 128 * mmu::MB;
pub const DTB_GUARD_SIZE: u64 = 2   * mmu::MB;

#[derive(Debug, Clone)]
pub struct KernelLayout {
    pub kheap: Range<u64>,
    pub kpages: Range<u64>,
    pub ktables: Range<u64>,
    pub dtb: Range<u64>,
    pub serial: u64,
}

impl KernelLayout {
    /// Carve the physical pools out of the top of RAM and stash MMIO bases.
    /// `dtb_addr` is the DTB's physical location (passed in from M-mode);
    /// the DTB region is the 2 MiB guard itself.
    pub fn new(ram_base: u64, ram_size: u64, dtb_addr: u64, serial_addr: u64) -> Self {
        let mem_end = ram_base + ram_size - DTB_GUARD_SIZE;
        let ktables_start = mem_end - KTABLES_SIZE;
        let kheap_start = ktables_start - KHEAP_SIZE;
        let kpages_start = kheap_start - KPAGES_SIZE;
        Self {
            kheap: kheap_start..(kheap_start + KHEAP_SIZE),
            kpages: kpages_start..(kpages_start + KPAGES_SIZE),
            ktables: ktables_start..(ktables_start + KTABLES_SIZE),
            dtb: dtb_addr..(dtb_addr + DTB_GUARD_SIZE),
            serial: serial_addr,
        }
    }
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

unsafe fn map_region(rt: &RootTable<'_>, pa: &mut PageAlloc, r: &Region) -> Result<(), ()> {
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

/// Map `pa_range` at an explicit virtual start. Picks gigapage / megapage /
/// 4 KiB leaves by alignment so pool-scale ranges don't explode into
/// thousands of PTEs.
unsafe fn map_region_va(
    rt: &RootTable<'_>,
    pa_alloc: &mut PageAlloc,
    va_start: u64,
    pa_range: Range<u64>,
    perms: u64,
    name: &str,
) -> Result<(), ()> {
    let cfg = MappingConfig {
        permissions: perms,
        levels: 4,
        page_size: PAGE_SIZE as u64,
        vaddr: VirtAddr::new(0),
        paddr: PhysAddr::new(0),
        log: false,
        supervisor_tag: None,
    };
    match unsafe { map_va_range(rt, pa_alloc, cfg, va_start, pa_range.clone()) } {
        Ok(_) => Ok(()),
        Err(()) => {
            serial::println!(
                "memmap: failed high-half {} v0x{:016X} p{:016X?}",
                name, va_start, pa_range
            );
            Err(())
        }
    }
}

/// Install high-half aliases for the ELF, pool, and MMIO regions. Pool/MMIO
/// ranges are already physical (they come from `KernelLayout`); ELF ranges
/// come from linker symbols, which resolve to runtime VAs — under identity
/// that's the physical address, post-trampoline it's the high-half VA. Either
/// way, `offset = r.range.start - image_va_base` is the offset within the
/// image, and `kernel_phys_base() + offset` is the matching PA.
unsafe fn map_kernel_high_half(
    rt: &RootTable<'_>,
    pa: &mut PageAlloc,
    layout: &KernelLayout,
) -> Result<(), ()> {
    unsafe {
        let image_va_base = &_text_start as *const _ as u64;
        let image_pa_base = kernel_phys_base();
        for r in kernel_elf_regions().iter() {
            let offset = r.range.start - image_va_base;
            let len = r.range.end - r.range.start;
            let va_start = ktext_base() + offset;
            let pa_start = image_pa_base + offset;
            map_region_va(rt, pa, va_start, pa_start..pa_start + len, r.perms, r.name)?;
        }
        for r in pool_regions(layout).iter() {
            let va_start = phys_to_virt(r.range.start);
            map_region_va(rt, pa, va_start, r.range.clone(), r.perms, r.name)?;
        }
        map_region_va(rt, pa, kmmio_uart(),
            layout.serial..layout.serial + PAGE_SIZE as u64, KRW, "serial.hh")?;
        map_region_va(rt, pa, kmmio_clint(),
            CLINT_MSIP_BASE..CLINT_MSIP_BASE + PAGE_SIZE as u64, KRW, "clint.msip.hh")?;
        map_region_va(rt, pa, kmmio_sswi(),
            ACLINT_SSWI_BASE..ACLINT_SSWI_BASE + PAGE_SIZE as u64, KRW, "aclint.sswi.hh")?;
    }
    Ok(())
}

/// High-half aliases for the boot-only regions that `map_kernel_self`
/// overlays on top of `map_kernel_shared`. Kept separate so the future
/// `unmap_boot_only_regions` can walk matching pairs.
unsafe fn map_kernel_self_high_half(
    rt: &RootTable<'_>,
    pa: &mut PageAlloc,
    layout: &KernelLayout,
) -> Result<(), ()> {
    unsafe {
        let image_va_base = &_text_start as *const _ as u64;
        let image_pa_base = kernel_phys_base();
        for r in boot_only_elf_regions().iter() {
            let offset = r.range.start - image_va_base;
            let len = r.range.end - r.range.start;
            let va_start = ktext_base() + offset;
            let pa_start = image_pa_base + offset;
            map_region_va(rt, pa, va_start, pa_start..pa_start + len, r.perms, r.name)?;
        }
        for r in self_only_pool_regions(layout).iter() {
            // Pool ranges are physical. ktables is in RAM so phys_to_virt
            // works; dtb sits in the reserved top-of-RAM region which is
            // still within the kdmap span.
            let va_start = phys_to_virt(r.range.start);
            map_region_va(rt, pa, va_start, r.range.clone(), r.perms, r.name)?;
        }
    }
    Ok(())
}

/// Map the minimal kernel surface a user process needs so that S-mode trap
/// handling executes correctly under its satp. Kernel code executing under
/// a user satp dereferences GOT on essentially every call, reads/writes the
/// heap (tracing/log), touches hart contexts + trap frames + thread stacks
/// (all in kpages), and prints to serial. It also pokes CLINT MSIPs and the
/// ACLINT SSWI to clear/send IPIs from the scheduler.
pub unsafe fn map_kernel_shared(
    rt: &RootTable<'_>,
    pa: &mut PageAlloc,
    layout: &KernelLayout,
) -> Result<(), ()> {
    unsafe {
        // Kernel ELF, pools, and MMIO all live at high-half VAs. KDMAP covers
        // RAM (so pool_regions doesn't need an identity alias); KMMIO covers
        // device pages.
        map_kernel_high_half(rt, pa, layout)?;
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
    rt: &RootTable<'_>,
    pa: &mut PageAlloc,
    layout: &KernelLayout,
) -> Result<(), ()> {
    unsafe {
        map_kernel_shared(rt, pa, layout)?;
        // ktables and dtb only need their KDMAP aliases now — the kernel
        // walks tables through `RootTable`'s PA→VA bias, and the dtb is
        // parsed once via `phys_to_virt`. No identity leg.
        map_kernel_self_high_half(rt, pa, layout)?;
    }
    Ok(())
}

/// Drop the boot-only ELF mappings (`.rela.dyn`, `.dynamic`, `.dynsym`,
/// `.dynstr`, `.hash`, `.gnu.hash`, `.eh_frame`) from `rt`. These are only
/// consumed by self-relocation, which runs before paging is enabled — by the
/// time the kernel is running under `satp`, nothing reads them. Caller is
/// responsible for `sfence.vma` if `rt` is the active root.
pub unsafe fn unmap_boot_only_regions(rt: &RootTable<'_>) -> Result<(), ()> {
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
