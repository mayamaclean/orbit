use core::alloc::Layout;

use alloc::vec::Vec;

#[derive(Debug)]
pub struct ElfInfo {
    pub entrypoint: usize,
    pub segments: Vec<(usize, Layout)>
}
