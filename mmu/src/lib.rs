#![no_std]

use core::{ops::{BitAnd, BitOr}};

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
    G = 0x20
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

#[derive(Debug)]
pub struct MappingConfig {
    pub permissions: u64,
    pub levels: usize,
    pub page_size: u64,
    pub vaddr: VirtAddr,
    pub paddr: PhysAddr,
    pub log: bool,
    pub supervisor_tag: Option<u8>
}

impl MappingConfig {
    pub fn copy(&self) -> Self {
        Self {
            permissions: self.permissions,
            levels: self.levels,
            page_size: self.page_size,
            vaddr: self.vaddr.copy(),
            paddr: self.paddr.copy(),
            log: self.log,
            supervisor_tag: self.supervisor_tag
        }
    }
}
