use alloc::vec::Vec;

use process::PhysBacking;

#[derive(Debug)]
pub struct ElfInfo {
    pub entrypoint: usize,
    pub segments: Vec<PhysBacking>,
}
