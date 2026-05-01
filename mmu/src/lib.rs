#![no_std]

use core::ops::{BitAnd, BitOr};

use crate::sv48::{PhysAddr, VirtAddr};

pub mod mmap;
pub mod sv48;

pub const PAGE_SIZE: usize = 4096;

pub const KB: u64 = 1024;
pub const MB: u64 = KB * 1024;
pub const GB: u64 = MB * 1024;

#[repr(u64)]
pub enum PagePermissions {
    R = 0x2,
    W = 0x4,
    X = 0x8,
    U = 0x10,
    G = 0x20,
}

impl Into<u64> for PagePermissions {
    fn into(self) -> u64 {
        self as u64
    }
}

impl BitOr for PagePermissions {
    type Output = u64;

    fn bitor(self, rhs: Self) -> Self::Output {
        self as u64 | rhs as u64
    }
}

impl BitAnd for PagePermissions {
    type Output = u64;

    fn bitand(self, rhs: Self) -> Self::Output {
        self as u64 & rhs as u64
    }
}

impl BitOr<u64> for PagePermissions {
    type Output = u64;

    fn bitor(self, rhs: u64) -> Self::Output {
        self as u64 | rhs
    }
}

impl BitAnd<u64> for PagePermissions {
    type Output = u64;

    fn bitand(self, rhs: u64) -> Self::Output {
        self as u64 & rhs
    }
}

impl BitOr<PagePermissions> for u64 {
    type Output = u64;

    fn bitor(self, rhs: PagePermissions) -> Self::Output {
        self | rhs as u64
    }
}

impl BitAnd<PagePermissions> for u64 {
    type Output = u64;

    fn bitand(self, rhs: PagePermissions) -> Self::Output {
        self & rhs as u64
    }
}

/// Value stored in PTE[8:9] (the two RSW "reserved for supervisor
/// software" bits). The hardware walker ignores these entirely — they're
/// exclusively for kernel policy. Variant names reflect orbit's policy;
/// the u8 discriminant is what actually lands in the PTE.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisorTag {
    /// PTE[8:9] = 0b00. Default — no tag attached.
    None = 0b00,
    /// PTE[8:9] = 0b01. User mapping backed by a `Shared`-pool frame,
    /// revocable via `SharedUserPtr::revoke()`. The revoker walks the
    /// user PT, matches this tag, and clears V before the backing is
    /// freed — so a SharedUserPtr teardown can't race into a UAF.
    SharedRevocable = 0b01,
}

#[derive(Debug, Clone, Copy)]
pub struct MappingConfig {
    pub permissions: u64,
    pub levels: usize,
    pub page_size: u64,
    pub vaddr: VirtAddr,
    pub paddr: PhysAddr,
    pub log: bool,
    pub supervisor_tag: SupervisorTag,
}
