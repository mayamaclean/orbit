use core::alloc::Layout;
use core::ops::Range;
use core::sync::atomic::{AtomicU64, Ordering};

use mem::frame::FrameAllocator;
use mem::round_u64_up;
use mmu::mmap::{PageAlloc, RootTable, id_map_range, map_va_range, reserve_va_range, unmap_range, virt_to_phys};
use mmu::sv48::{PageTable, PhysAddr, VirtAddr};
use mmu::{MappingConfig, PAGE_SIZE, PagePermissions, SupervisorTag};
use process::{Frame, Shared, Table, UserOnly};

// =========================================================================
// Address-kind newtypes
//
// `PhysAddr` / `VirtAddr` live in `mmu`. The kinds *here* are orbit's
// kernel-specific VA flavors: they tag which window a VA lives in, which
// in turn dictates which conversion / dereference paths are legal.
//
// - `KdmapVa` — kernel-side direct-map alias of `Shared`-pool RAM. The
//   kernel can `*p` a `KdmapVa` as long as the referenced PA is in a pool
//   that's KDMAP-mapped under the active satp. Post-pool-split, that's
//   `kpages` / `ktables` / kheap; `user_pages` is NOT, so
//   `phys_to_kdmap(user_pa)` is arithmetically valid but produces a VA
//   the kernel can't deref — touch user_pages only via `UserPageWindow`.
// - `UserVa` — a VA in user space. Only resolvable under the owning
//   process's satp; kernel-side conversion requires walking that PT.
// =========================================================================

/// Kernel direct-map alias of physical RAM. Produced by
/// [`phys_to_kdmap`]; reversible via [`KdmapVa::to_phys`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct KdmapVa(u64);

impl KdmapVa {
    pub const fn new(raw: u64) -> Self { Self(raw) }
    pub const fn raw(self) -> u64 { self.0 }
    pub fn to_virt(self) -> VirtAddr { VirtAddr::new(self.0) }
    pub fn to_phys(self) -> PhysAddr {
        PhysAddr::new(self.0.wrapping_sub(kdmap_base().wrapping_sub(ram_phys_base())))
    }
    pub fn as_mut_ptr<T>(self) -> *mut T { self.0 as *mut T }
    pub fn as_ptr<T>(self) -> *const T { self.0 as *const T }
}

/// A VA that lives in user address space. Only resolvable under the
/// owner's satp; kernel-side use goes through [`user_va_to_kdmap`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct UserVa(u64);

impl UserVa {
    pub const fn new(raw: u64) -> Self { Self(raw) }
    pub const fn raw(self) -> u64 { self.0 }
    pub fn to_virt(self) -> VirtAddr { VirtAddr::new(self.0) }
}

/// Arithmetic conversion from a physical address to its KDMAP alias.
/// Raw `PhysAddr` input — the caller asserts that the PA is in a pool
/// that's KDMAP-mapped under the active satp. Prefer
/// [`FrameToKdmap::to_kdmap`] on a `Frame<Shared>` / `Frame<Table>`,
/// which makes this promise a compile-time property.
#[inline]
pub fn phys_to_kdmap(pa: PhysAddr) -> KdmapVa {
    KdmapVa::new(pa.get_raw().wrapping_add(kdmap_base().wrapping_sub(ram_phys_base())))
}

/// Extension trait adding `to_kdmap()` to frames drawn from pools with
/// a kernel-side KDMAP alias. Explicitly unimplemented for
/// `Frame<UserOnly>`: the kernel has no KDMAP for user_pages, so
/// attempting that conversion must be a compile error.
pub trait FrameToKdmap {
    fn to_kdmap(&self) -> KdmapVa;
}

impl FrameToKdmap for Frame<Shared> {
    #[inline]
    fn to_kdmap(&self) -> KdmapVa { phys_to_kdmap(self.raw()) }
}

impl FrameToKdmap for Frame<Table> {
    #[inline]
    fn to_kdmap(&self) -> KdmapVa { phys_to_kdmap(self.raw()) }
}

// =========================================================================
// Frame-pool wrappers
//
// `FrameAllocator` is kept bias-agnostic — it just tracks `usize`-valued
// ranges. Orbit feeds it physical addresses and these wrappers expose
// typed `PhysAddr` / `KdmapVa` return values so the caller's intent is
// visible at the boundary.
// =========================================================================

const PAGE_LAYOUT: Layout = unsafe {
    Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE)
};

/// Pool of pages that back Sv48 intermediate tables. Every alloc
/// materializes a new table, which the kernel always needs to zero and
/// stamp — so returning `(PhysAddr, KdmapVa)` saves each caller a
/// separate conversion.
pub struct TablePages {
    inner: FrameAllocator<33>,
}

impl TablePages {
    pub const fn new() -> Self { Self { inner: FrameAllocator::new() } }

    pub fn add_pa_range(&mut self, pa_range: Range<u64>) {
        self.inner.add_frame(pa_range.start as usize, pa_range.end as usize);
    }

    pub fn alloc(&mut self, layout: Layout) -> Option<(Frame<Table>, KdmapVa)> {
        let pa = PhysAddr::new(self.inner.alloc_aligned(layout)? as u64);
        let frame = Frame::<Table>::new(pa);
        let kva = frame.to_kdmap();
        Some((frame, kva))
    }

    pub fn free(&mut self, frame: Frame<Table>, layout: Layout) {
        self.inner.dealloc_aligned(frame.get_raw() as usize, layout);
    }

    /// Raw inner allocator, for passing to the mmu walker via
    /// `PageAlloc::FA`. The walker consumes raw PAs (see
    /// `mmu::mmap::PageAlloc`).
    pub fn frames_mut(&mut self) -> &mut FrameAllocator<33> {
        &mut self.inner
    }

}

/// Pool of pages that are kernel-accessible (KDMAP alias under every
/// satp). Allocations tag as `Frame<Shared>`. Callers that need to deref
/// use `alloc_kdmap`; callers that just want a PA (DMA, PTE install) use
/// `alloc_pa`.
pub struct KernelPages {
    inner: FrameAllocator<33>,
}

impl KernelPages {
    pub const fn new() -> Self { Self { inner: FrameAllocator::new() } }

    pub fn add_pa_range(&mut self, pa_range: Range<u64>) {
        self.inner.add_frame(pa_range.start as usize, pa_range.end as usize);
    }

    pub fn alloc_pa(&mut self, layout: Layout) -> Option<Frame<Shared>> {
        self.inner.alloc_aligned(layout).map(|pa| Frame::<Shared>::new(PhysAddr::new(pa as u64)))
    }

    pub fn alloc_kdmap(&mut self, layout: Layout) -> Option<(Frame<Shared>, KdmapVa)> {
        let frame = self.alloc_pa(layout)?;
        let kva = frame.to_kdmap();
        Some((frame, kva))
    }

    pub fn free(&mut self, frame: Frame<Shared>, layout: Layout) {
        self.inner.dealloc_aligned(frame.get_raw() as usize, layout);
    }

    pub fn frames_mut(&mut self) -> &mut FrameAllocator<33> {
        &mut self.inner
    }

}

/// Pool of pages that are user-only. No KDMAP alias under the kernel
/// satp, so there is deliberately no `alloc_kdmap` — touching a backing
/// from kernel code goes through `UserPageWindow`. Attempting
/// `to_kdmap()` on a `Frame<UserOnly>` is a compile error.
pub struct UserPages {
    inner: FrameAllocator<33>,
}

impl UserPages {
    pub const fn new() -> Self { Self { inner: FrameAllocator::new() } }

    pub fn add_pa_range(&mut self, pa_range: Range<u64>) {
        self.inner.add_frame(pa_range.start as usize, pa_range.end as usize);
    }

    pub fn alloc_pa(&mut self, layout: Layout) -> Option<Frame<UserOnly>> {
        self.inner.alloc_aligned(layout).map(|pa| Frame::<UserOnly>::new(PhysAddr::new(pa as u64)))
    }

    pub fn free(&mut self, frame: Frame<UserOnly>, layout: Layout) {
        self.inner.dealloc_aligned(frame.get_raw() as usize, layout);
    }

}

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
// four windows are 16 GiB apart — well over the range any of them will
// occupy — so they can't collide.
pub const KTEXT_NOMINAL:    u64 = 0xFFFF_FFC0_0000_0000;
pub const KDMAP_NOMINAL:    u64 = 0xFFFF_FFD0_0000_0000;
pub const KMMIO_NOMINAL:    u64 = 0xFFFF_FFE0_0000_0000;
pub const KSCRATCH_NOMINAL: u64 = 0xFFFF_FFF0_0000_0000;

// Transient per-window view into a user_pages backing for setup-time writes.
// Single slot, serialized by the Orbit lock; size bounded to cover the
// largest allocation we window in one shot (UPROC_STACK_MAX, 30 MiB, rounded
// up to a megapage multiple).
pub const KSCRATCH_SIZE: u64 = 32 * mmu::MB;

// KMMIO window slot assignments. Three single-page fixed slots at the
// bottom of the window — UART, CLINT MSIP, ACLINT SSWI — and an arena
// past them for dynamically-discovered MMIO (PCI config, e1000 BAR, ...).
#[inline] pub fn kmmio_uart()  -> u64 { kmmio_base() }
#[inline] pub fn kmmio_clint() -> u64 { kmmio_base() + PAGE_SIZE as u64 }
#[inline] pub fn kmmio_sswi()  -> u64 { kmmio_base() + 2 * PAGE_SIZE as u64 }

/// Offset within the KMMIO window past the fixed-slot pages where the
/// dynamic arena begins. Megapage-aligned so larger regions get natural
/// superpage alignment.
const KMMIO_ARENA_OFFSET: u64 = 2 * mmu::MB;
static KMMIO_ARENA_NEXT: AtomicU64 = AtomicU64::new(KMMIO_ARENA_OFFSET);

/// Reserve `size` bytes of KMMIO VA space (rounded up to `PAGE_SIZE`) and
/// return the base. Pure VA bookkeeping — caller is responsible for
/// installing the leaf PTEs that map the returned VA to the device's PA.
pub fn kmmio_alloc(size: u64) -> u64 {
    let aligned = mem::round_u64_up(size, PAGE_SIZE as u64);
    let offset = KMMIO_ARENA_NEXT.fetch_add(aligned, Ordering::AcqRel);
    kmmio_base() + offset
}

// Runtime-parameterized kernel address-space layout. Set once by `init_layout`
// during early `rust_main` on hart 0, before any other hart is woken. Reads
// from other harts are safe with Relaxed because the hart-wake IPI is the
// synchronizing event.
static RAM_PHYS_BASE:    AtomicU64 = AtomicU64::new(0);
static KTEXT_BASE:       AtomicU64 = AtomicU64::new(0);
static KDMAP_BASE:       AtomicU64 = AtomicU64::new(0);
static KMMIO_BASE:       AtomicU64 = AtomicU64::new(0);
static KSCRATCH_BASE:    AtomicU64 = AtomicU64::new(0);
// Physical address of `_text_start` — the base the kernel ELF was loaded at.
// Post-trampoline `&_text_start as u64` returns the high-half VA, so helpers
// that need the physical (PT construction, DMA setup) read this instead.
static KERNEL_PHYS_BASE: AtomicU64 = AtomicU64::new(0);

pub fn init_layout(ram_phys: u64, ktext: u64, kdmap: u64, kmmio: u64, kscratch: u64, kernel_phys: u64) {
    RAM_PHYS_BASE.store(ram_phys, Ordering::Relaxed);
    KTEXT_BASE.store(ktext, Ordering::Relaxed);
    KDMAP_BASE.store(kdmap, Ordering::Relaxed);
    KMMIO_BASE.store(kmmio, Ordering::Relaxed);
    KSCRATCH_BASE.store(kscratch, Ordering::Relaxed);
    KERNEL_PHYS_BASE.store(kernel_phys, Ordering::Relaxed);
}

#[inline] pub fn ram_phys_base()    -> u64 { RAM_PHYS_BASE.load(Ordering::Relaxed) }
#[inline] pub fn ktext_base()       -> u64 { KTEXT_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kdmap_base()       -> u64 { KDMAP_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kmmio_base()       -> u64 { KMMIO_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kscratch_base()    -> u64 { KSCRATCH_BASE.load(Ordering::Relaxed) }
#[inline] pub fn kernel_phys_base() -> u64 { KERNEL_PHYS_BASE.load(Ordering::Relaxed) }

/// Resolve a user VA under `root_table` to the kernel's KDMAP alias of
/// the same physical backing. Lets syscall handlers dereference user
/// buffers without SUM and without identity-mapping kpages — the KDMAP
/// VA is a supervisor mapping over the same RAM page.
///
/// # Safety
/// Walks `root_table`'s PTE tree, which must remain valid for the
/// duration of the call.
#[inline]
pub unsafe fn user_va_to_kdmap(root_table: &RootTable<'_>, user_va: UserVa) -> Option<KdmapVa> {
    unsafe { virt_to_phys(root_table, user_va.to_virt()) }
        .map(|pa| phys_to_kdmap(PhysAddr::new(pa as u64)))
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

// Kernel physical pool sizes. DTB sits in a 2 MiB guard at the top of RAM.
// Layout grows downward from the DTB guard:
//   [user_pages][kpages][kheap][ktables][dtb guard]
//     with `mem_end = ram_end - DTB_GUARD`.
//
// `user_pages` is the home of user-private allocations (stacks, ELF
// backings, anon mmaps) once pool-split routing lands (roadmap milestone 3).
// Reserved and tracked from day one; wiring an allocator and actually
// drawing from it is later steps in the same milestone.
pub const KTABLES_SIZE:    u64 = 128 * mmu::MB;
pub const KHEAP_SIZE:      u64 = 128 * mmu::MB;
pub const KPAGES_SIZE:     u64 = 128 * mmu::MB;
pub const USER_PAGES_SIZE: u64 = 128 * mmu::MB;
pub const DTB_GUARD_SIZE:  u64 = 2   * mmu::MB;

#[derive(Debug, Clone)]
pub struct KernelLayout {
    pub kheap: Range<u64>,
    pub kpages: Range<u64>,
    pub ktables: Range<u64>,
    pub user_pages: Range<u64>,
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
        let user_pages_start = kpages_start - USER_PAGES_SIZE;
        Self {
            kheap: kheap_start..(kheap_start + KHEAP_SIZE),
            kpages: kpages_start..(kpages_start + KPAGES_SIZE),
            ktables: ktables_start..(ktables_start + KTABLES_SIZE),
            user_pages: user_pages_start..(user_pages_start + USER_PAGES_SIZE),
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

fn pool_regions(layout: &KernelLayout) -> [Region; 3] {
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
        // ktables has to be reachable under every satp: syscall handlers can
        // walk the user's PT (virt_to_phys, user_va_to_kdmap) while running
        // under the user satp, and the walker follows child PPNs through the
        // KDMAP alias of the table pool.
        Region {
            range: layout.ktables.clone(),
            perms: KRW,
            name: "ktables",
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
        supervisor_tag: SupervisorTag::None,
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
/// Reserve a KMMIO arena slot covering `pa_range` and install RW kernel
/// leaves at the resulting VA in `rt`. Returns the VA so the caller can
/// hand it to a driver as `*mut u32` etc. Caller owns the post-install
/// `sfence.vma`.
pub unsafe fn install_kmmio_alias(
    rt: &RootTable<'_>,
    pa_alloc: &mut PageAlloc,
    pa_range: Range<u64>,
) -> Result<u64, ()> {
    let len = pa_range.end.wrapping_sub(pa_range.start);
    let va_start = kmmio_alloc(len);
    unsafe { map_region_va(rt, pa_alloc, va_start, pa_range, KRW, "kmmio.dyn")? };
    Ok(va_start)
}

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
        supervisor_tag: SupervisorTag::None,
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
            let va_start = phys_to_kdmap(PhysAddr::new(r.range.start)).raw();
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
            let va_start = phys_to_kdmap(PhysAddr::new(r.range.start)).raw();
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

        // Pre-materialize KSCRATCH intermediates (down to L0) but leave
        // leaves V=0. UserPageWindow opens a transient leaf PTE here to
        // access a user_pages backing from the kernel, and invalidates it
        // on drop. Installed in every satp so the window works regardless
        // of which satp is live when setup-time writes happen.
        reserve_va_range(rt, pa, kscratch_base(), kscratch_base() + KSCRATCH_SIZE)?;
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

fn self_only_pool_regions(layout: &KernelLayout) -> [Region; 1] {
    [
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
