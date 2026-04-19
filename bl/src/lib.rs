#![no_std]

use core::arch::asm;
use core::sync::{atomic::{AtomicU64, AtomicUsize, Ordering}};

use device::{SysInfo, TrapFrame};
use mmu::{GB, MB};
use serial::println;

use elf::{endian::LittleEndian};
use riscv::register::mstatus::SPP;

unsafe extern "C" {
    unsafe static KERNEL_STACK_END: u64;
}

pub const KERNEL_ELF: &'static [u8] = include_bytes!("../../kmain/target/riscv64gc-unknown-none-elf/release/orbit");
pub const KERNEL_ELF_LEN: usize = KERNEL_ELF.len();

pub const TRAP_FRAME_ADDR: usize = 0x80800000;
pub const TRAP_FRAMES: *mut TrapFrame = 0x80800000 as *mut _;
pub const ID_MAP_TABLES: usize = TRAP_FRAME_ADDR- (2 * MB as usize);

pub static ID_MAP_ADDR: AtomicU64 = AtomicU64::new(0);
pub static SYSINFO: SysInfo = SysInfo {
    dtb_addr: AtomicUsize::new(0),
    serial: AtomicUsize::new(0)
};
pub static HART_ROOT: AtomicUsize = AtomicUsize::new(0);

pub fn setup_interrupts() {
    use riscv::register::{mstatus, mie, mcounteren, mideleg, medeleg};

    unsafe {
        //riscv::register::mstatus::set_mie();
        mstatus::set_sie();

        mie::set_msoft();
        //mie::set_mtimer();
        mie::set_stimer();
        mie::set_ssoft();

        mcounteren::set_tm();

        asm!(
            "csrw menvcfg, t0",
            in("t0") 0x8000000000000000u64
        );

        mideleg::set_stimer();
        mideleg::set_ssoft();
        mideleg::set_sext();

        medeleg::set_breakpoint();
        medeleg::set_user_env_call();
        
        medeleg::set_instruction_fault();
        medeleg::set_instruction_page_fault();
        medeleg::set_load_page_fault();
        medeleg::set_store_page_fault();
    }
}

fn supervisor_write_hart_swi(hart: usize, val: u32) {
    const BASE: usize = 0x2f00000;
    unsafe {
        (BASE as *mut u32).add(hart).write_volatile(val);
    }
}

pub fn supervisor_wake_hart(hart: usize) {
    supervisor_write_hart_swi(hart, 1);
}

pub fn supervisor_clear_hart_swi(hart: usize) {
    supervisor_write_hart_swi(hart, 0);
}

pub extern "C" fn kmain_enter(serial_addr: usize, dtb_addr: usize) {
    // load kernel code into ram
    // reset stack and sret to kernel code

    let kernel_elf_addr = &KERNEL_ELF as *const _ as u64;
    println!("kernel elf @ 0x{:08X}", kernel_elf_addr);

    let elf = match elf::ElfBytes::<LittleEndian>::minimal_parse(&KERNEL_ELF[..]) {
        Ok(e) => e,
        Err(e) => { println!("failed to parse kernel elf: {e:?}"); return }
    };

    const VBASE: u64 = 0x8000_0000 + (64 * MB);

    let segments = elf.segments().unwrap();
    for segment in segments.iter() {
        let load_segment = segment.p_type == elf::abi::PT_LOAD;
        if !load_segment {
            continue
        }

        let vaddr = VBASE + segment.p_vaddr;
        println!("loading {}KB 0x{vaddr:08X}={segment:08x?}",
            mem::round_u64_up(segment.p_memsz, 4096) / 1024);

        let segment_data = match elf.segment_data(&segment) {
            Ok(seg) => seg,
            Err(e) => {
                println!("error parsing loadable segment data: {e:?}");
                return
            }
        };

        unsafe { core::ptr::copy_nonoverlapping(segment_data.as_ptr(), vaddr as *mut u8, segment_data.len()); }

        if segment.p_memsz > segment.p_filesz {
            unsafe {
                core::ptr::write_bytes(
                    (vaddr + segment.p_filesz) as *mut u8,
                    0,
                    (segment.p_memsz - segment.p_filesz) as usize
                );
            }
        }
    }

    println!("finished loading segments?");

    unsafe {
        let entrypoint = (elf.ehdr.e_entry + VBASE) as usize;

        println!("mret to 0x{entrypoint:016X}");

        riscv::register::sepc::write(entrypoint);
        riscv::register::mstatus::set_spp(SPP::Supervisor);

        setup_interrupts();
        
        let sysinfo_ptr = &SYSINFO as *const _ as usize;
        let hartid = riscv::register::mhartid::read();

        SYSINFO.dtb_addr.store(dtb_addr, Ordering::Relaxed);
        SYSINFO.serial.store(serial_addr, Ordering::Relaxed);

        asm!(
            "csrw pmpaddr0, {apmp}",
            "csrw pmpcfg0, {acfg}",
            "li t0, 0x02004000",          // mtimecmp address
            "li t1, 0x0200bff8",    // mtime address
            "ld t2, 0(t1)",               // Load current 64-bit mtime
            "li t3, 100000000",             // Example interval (1 million cycles)
            "add t2, t2, t3",             // t2 = mtime + interval
            "sd t2, 0(t0)",
            //"li t0, 0x222",        // This sets bits 1, 5, and 9
            //"csrs mideleg, t0",    // Ensure bit 5 (0x20) is definitely set2
            "mv sp, {s}",
            "fence w, rw", // Ensure ELF writes are visible to all harts
            "fence.i",    // Synchronize I-cache with D-cache
            "sfence.vma", // Flush the MMU TLB
            "sret",
            apmp = in(reg) !0,
            acfg = in(reg) 0xf | 0x80,
            s = in(reg) KERNEL_STACK_END,
            in("a0") hartid,
            in("a1") sysinfo_ptr,
            options(noreturn),
        );
    }
}
