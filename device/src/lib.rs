#![no_std]

use core::{fmt::Debug, sync::atomic::{AtomicPtr, AtomicUsize}};
use dtoolkit::{Node, Property, fdt::{Fdt, FdtNode}};
use riscv::register::satp::Satp;

pub const TRAP_STACK_SIZE: usize = 2 * 1024 * 1024;

#[repr(C, align(16))]
pub struct Stack {
    pub stack_data: [u8; TRAP_STACK_SIZE]
}

impl Debug for Stack {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Stack")
    }
}

#[repr(C, align(128))]
pub struct HartContext {
    pub frame: TrapFrame,
    pub tsp: usize,             // 528
    pub kptr: AtomicPtr<()>,    // 536
    pub current: AtomicPtr<()>, // 544
    pub cscratch: u64,          // 552
    pub cscratch2: u64,         // 560
    pub satp: Satp,             // 568
    pub hart_id: u64,           // 576
    pub s_trap_addr: u64,       // 584
    pub trap_stack: Stack,      // 592
    pub k_stack: Stack,
    /// PLIC S-mode context index for this hart. Populated after PLIC
    /// install; sentinel `u32::MAX` means "no PLIC context assigned".
    /// Appended past the load-bearing offsets read by `asm/trap.S`.
    pub plic_s_context: u32,
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct TrapFrame {
	pub regs:  [usize; 32], // 0 - 255
	pub fregs: [usize; 32], // 256 - 511
    pub asid: usize, // 512-520
    pub scratch: usize // 520-528
}

impl TrapFrame {
    pub const fn empty() -> Self {
        Self {
            regs: [0usize; 32],
            fregs: [0usize; 32],
            asid: 0,
            scratch: 0
        }
    }
}

#[repr(C, align(16))]
pub struct SysInfo {
    pub dtb_addr: AtomicUsize,
    pub serial: AtomicUsize
}

pub fn handle_dtb_node<'a>(n: FdtNode<'a>) -> Result<FdtNode<'a>, ()> {
    if let Some(compat) = n.property("compatible") {
        if let Ok(compat_str) = compat.as_str() {
            if compat_str.contains("ns16550a") {
                return Ok(n)
            }
        }
    }

    for child in n.children() {
        if let Ok(node) = handle_dtb_node(child) {
            return Ok(node)
        }
    }
    Err(())
}

pub unsafe fn find_serial_port(dtb_addr: *const u8) -> Result<usize, ()> {
    let fdt = unsafe { Fdt::from_raw_unchecked(dtb_addr) };
    let root = fdt.root();
    
    let node = handle_dtb_node(root)?;
    let n_name = node.name();
    let addr_index = n_name.find("@")
        .ok_or(())? + 1;

    let addr = usize::from_str_radix(&n_name[addr_index..], 16)
        .map_err(|_| ())?;

    Ok(addr)
}

fn find_memory_node<'a>(n: FdtNode<'a>) -> Option<FdtNode<'a>> {
    if let Some(dt) = n.property("device_type") {
        if let Ok(s) = dt.as_str() {
            if s == "memory" {
                return Some(n)
            }
        }
    }
    for child in n.children() {
        if let Some(m) = find_memory_node(child) {
            return Some(m)
        }
    }
    None
}

pub unsafe fn find_ram(dtb_addr: *const u8) -> Result<(u64, u64), ()> {
    let fdt = unsafe { Fdt::from_raw_unchecked(dtb_addr) };
    let node = find_memory_node(fdt.root()).ok_or(())?;
    let mut regs = node.reg().map_err(|_| ())?.ok_or(())?;
    let reg = regs.next().ok_or(())?;
    let base = reg.address::<u64>().map_err(|_| ())?;
    let size = reg.size::<u64>().map_err(|_| ())?;
    Ok((base, size))
}

/// assumes this is only called from hart 0
pub unsafe fn wake_harts(hart_count: usize) {
    unsafe {
        for hart in 0..hart_count {
            wake_hart(hart);
        }
    }
}

pub unsafe fn wake_hart(hart: usize) {
    let base_addr = 0x02000000u32;
    unsafe {
        (base_addr as *mut u32).add(hart).write_volatile(1);
    }
}

pub unsafe fn clear_hart_int(hart: usize) {
    let base_addr = 0x02000000u32;
    unsafe {
        (base_addr as *mut u32).add(hart).write_volatile(0);
    }
}