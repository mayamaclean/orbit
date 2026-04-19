//! User-mode address-space layout (Sv48, low→high).
//!
//! The 2–4 GiB band is reserved for the kernel's shared identity map
//! (kernel text, kheap, kpages, MMIO), so user layout skips over it.
//!
//!   0..USER_NULL_GUARD_END                         null guard
//!   UPROC_STACK_BASE..+8 GiB                       256 * 32 MiB stack slots
//!   USER_TEXT_BASE..                               user ELF image + mmap arena
//!   USER_TRAP_FRAME_BASE..                         256 * 4 KiB TrapFrames (S-only)

pub const PAGE_SIZE:  u64 = 4096;
pub const LARGE_PAGE: u64 = 2 * 1024 * 1024;

/// Matches Linux-style mmap_min_addr; catches NULL-region derefs as faults.
pub const USER_NULL_GUARD_END: u64 = 0x1_0000;

/// Per-thread stack region. 256 slots * 32 MiB stride = 8 GiB total.
/// Each slot holds a stack sized per-thread (2..=30 MiB, multiples of 2 MiB),
/// anchored at the high end of the slot; the remainder is an unmapped guard.
pub const UPROC_STACK_BASE:    u64 = 0x1_0000_0000;
pub const UPROC_STACK_STRIDE:  u64 = 32 * LARGE_PAGE;
pub const UPROC_STACK_GRAIN:   u64 = LARGE_PAGE;
pub const UPROC_STACK_MIN:     u64 = UPROC_STACK_GRAIN;
/// One grain reserved for the guard at the low end of each slot.
pub const UPROC_STACK_MAX:     u64 = UPROC_STACK_STRIDE - UPROC_STACK_GRAIN;
pub const UPROC_STACK_DEFAULT: u64 = UPROC_STACK_GRAIN;

pub const fn user_stack_slot_base(slot: u16) -> u64 {
    UPROC_STACK_BASE + (slot as u64) * UPROC_STACK_STRIDE
}

pub const fn user_stack_slot_top(slot: u16) -> u64 {
    user_stack_slot_base(slot) + UPROC_STACK_STRIDE
}

/// Low end of the writable stack; stack grows down from `user_stack_slot_top`.
pub const fn user_stack_vaddr(slot: u16, stack_size: u64) -> u64 {
    user_stack_slot_top(slot) - stack_size
}

pub const fn user_stack_guard_vaddr(slot: u16) -> u64 {
    user_stack_slot_base(slot)
}

pub const fn user_stack_guard_size(stack_size: u64) -> u64 {
    UPROC_STACK_STRIDE - stack_size
}

pub const fn validate_user_stack_size(size: u64) -> bool {
    size >= UPROC_STACK_MIN
        && size <= UPROC_STACK_MAX
        && size % UPROC_STACK_GRAIN == 0
}

/// User ELF image and mmap arena sit above the stack region.
pub const USER_TEXT_BASE: u64 = 0x3_4000_0000;

/// Kernel-private per-thread TrapFrame region (no U bit). One page per slot.
pub const USER_TRAP_FRAME_BASE:   u64 = 0x100_0000_0000;
pub const USER_TRAP_FRAME_STRIDE: u64 = PAGE_SIZE;

pub const fn user_trap_frame_vaddr(slot: u16) -> u64 {
    USER_TRAP_FRAME_BASE + (slot as u64) * USER_TRAP_FRAME_STRIDE
}
