//! User-mode address-space layout.
//!
//! Sv48 user vaddrs occupy bits [0..47]. The kernel enforces that all
//! dynamically-allocated user mappings fall inside [`USER_MMAP_BASE`,
//! [`USER_MMAP_TOP`]). The ELF image loads below the mmap arena; the stack
//! lives above it and grows down from [`USER_STACK_TOP`].

pub const USER_TEXT_BASE: u64 = 0x0000_0000_9000_0000;

pub const USER_MMAP_BASE: u64 = 0x0000_0000_A000_0000;
pub const USER_MMAP_TOP:  u64 = 0x0000_003F_0000_0000;

pub const USER_STACK_TOP: u64 = 0x0000_003F_FFFF_F000;
pub const USER_STACK_MAX: u64 = 8 * 1024 * 1024;

pub const PAGE_SIZE:      u64 = 4096;
pub const LARGE_PAGE:     u64 = 2 * 1024 * 1024;
