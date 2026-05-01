//! Manager-side completion policy — pure decisions that the scheduler
//! thread (k_manage) makes while fulfilling user-blocked requests. The
//! bulk of the manager handlers is allocator / page-table plumbing that
//! stays in kmain; this module is for the decisions that are cleanly
//! separable from that state.

use crate::PAGE_SIZE;

/// 2 MiB megapage boundary. Matches `mmu`'s Sv48 level-1 page size. Kept
/// local so orbit-core doesn't pull in the whole mmu crate for a single
/// constant.
pub const MEGAPAGE_SIZE: usize = 2 * 1024 * 1024;

/// Geometry chosen for a single mmap request. `levels` is the walker
/// depth for `MappingConfig`: 3 for megapages (stop at L1), 4 for 4 KiB
/// pages (walk to leaf).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappingGeometry {
    pub align: usize,
    pub levels: usize,
}

/// Pick a page geometry for an mmap. Megapages are preferred when both
/// `vaddr` and `size` are 2 MiB-aligned (fewer PTEs, cheaper TLB); falls
/// back to 4 KiB pages when both are merely page-aligned; returns `None`
/// when neither alignment holds.
///
/// Misaligned inputs are a user error — the shim returns `-1` to the
/// caller and the manager doesn't touch allocators.
pub fn select_mapping_geometry(vaddr: usize, size: usize) -> Option<MappingGeometry> {
    if vaddr % MEGAPAGE_SIZE == 0 && size % MEGAPAGE_SIZE == 0 {
        Some(MappingGeometry {
            align: MEGAPAGE_SIZE,
            levels: 3_usize,
        })
    }
    else if vaddr % PAGE_SIZE == 0 && size % PAGE_SIZE == 0 {
        Some(MappingGeometry {
            align: PAGE_SIZE,
            levels: 4_usize,
        })
    }
    else {
        None
    }
}
