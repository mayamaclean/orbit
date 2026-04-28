#![no_std]

use core::arch::asm;
use core::sync::{atomic::{AtomicUsize, Ordering}};

use device::{SysInfo, TrapFrame, wake_hart};
use mmu::{MB};
use serial::println;

use elf::{endian::LittleEndian};
use riscv::register::mstatus::SPP;

unsafe extern "C" {
    unsafe static KERNEL_STACK_END: u64;
    unsafe static RODATA_START: u64;
    unsafe static RODATA_END: u64;
}

pub const KERNEL_ELF: &'static [u8] = include_bytes!("../../kmain/target/riscv64gc-unknown-none-elf/release/orbit");
pub const KERNEL_ELF_LEN: usize = KERNEL_ELF.len();

pub const TRAP_FRAME_ADDR: usize = 0x80800000;
pub const TRAP_FRAMES: *mut TrapFrame = 0x80800000 as *mut _;

pub static SYSINFO: SysInfo = SysInfo {
    dtb_addr: AtomicUsize::new(0),
    serial: AtomicUsize::new(0)
};

// PA of kmain's `_start`, computed by hart 0 once it has copied the kernel
// ELF into RAM. Secondary harts in `kinit_hart` spin on this; once set,
// they sret straight to S-mode kmain (bare satp), where the per-hart
// bringup now lives.
pub static KMAIN_ENTRY: AtomicUsize = AtomicUsize::new(0);

pub fn setup_interrupts() {
    use riscv::register::{mstatus, mie, mcounteren, mideleg, medeleg};

    unsafe {
        //riscv::register::mstatus::set_mie();
        mstatus::set_sie();

        // Sstc: enable supervisor counter+timer extensions in menvcfg
        // (bit 63 = STCE). Done before stimecmp write below so the
        // CSR exists for us to write.
        asm!(
            "csrw menvcfg, t0",
            in("t0") 0x8000000000000000u64
        );

        // Park the S-mode timer comparator at +∞ before enabling
        // mie.STIE. With Sstc on, mip.STIP follows mtime > stimecmp;
        // stimecmp resets to 0, so mtime > 0 leaves STIP perpetually
        // pending. Flipping mie.STIE without parking stimecmp first
        // fires an STI trap in M-mode immediately (mideleg can't help
        // because mideleg only delegates traps from S/U), and m_trap
        // has no cause-5 arm — mret-stuck-in-m_trap-loop. kmain's
        // s_trap arms its own stimecmp when it actually wants timer
        // ticks.
        asm!(
            "csrw 0x14d, {0}",   // stimecmp = MAX
            in(reg) usize::MAX,
        );

        mie::set_msoft();
        //mie::set_mtimer();
        mie::set_stimer();
        mie::set_ssoft();

        mcounteren::set_tm();

        mideleg::set_stimer();
        mideleg::set_ssoft();
        mideleg::set_sext();

        medeleg::set_breakpoint();
        medeleg::set_supervisor_env_call();
        medeleg::set_user_env_call();
        
        medeleg::set_instruction_fault();
        medeleg::set_instruction_page_fault();
        medeleg::set_load_page_fault();
        medeleg::set_store_page_fault();
    }
}

/// Configure PMP to (a) hide bl's M-mode region from S-mode and (b) lock
/// `.rodata` read-only against M-mode itself. The latter turns a stack
/// overflow from above into a store-access fault at the rodata boundary,
/// since the stack region in [memory.x](../memory.x) sits immediately
/// above `.rodata`.
///
/// Layout (PMP entries 0..4 in priority order, lowest wins):
///
///   0: TOR [0, 0x80000000)              RWX, L=0  — MMIO. S allowed, M bypasses.
///   1: TOR [0x80000000, _rodata_start)  ---, L=0  — bl text/bss/data. S denied. M bypasses.
///   2: TOR [_rodata_start, _rodata_end) R--, L=1  — rodata RO for *both* modes (stack guard).
///   3: TOR [_rodata_end, 0x84000000)    ---, L=0  — bl stack + id_tables + trap frames + slack. S denied. M bypasses.
///   4: TOR [0x84000000, !0)             RWX, L=0  — kmain region + everything past it. S allowed.
///
/// The asymmetry on rodata (M+S both R, neither W) is a PMP limitation:
/// no single entry can give "S denied + M read-only", so we pick the
/// half that catches a real bug class. S can read rodata bytes, but
/// they're either things kmain has its own copy of (KERNEL_ELF) or
/// non-sensitive (boot strings, mem.S linker constants). Per-hart
/// CSR; called once on each hart's M-mode init (kmain_enter for hart 0,
/// kinit_hart for secondaries) before sret.
pub unsafe fn setup_pmp() {
    // pmpcfg byte layout (one byte per entry):
    //   bits 0..2 = R/W/X
    //   bits 3..4 = A (0=Off, 1=TOR, 2=NA4, 3=NAPOT)
    //   bits 5..6 = reserved
    //   bit  7    = L
    const TOR: u64 = 1 << 3;
    const R:   u64 = 1 << 0;
    const W:   u64 = 1 << 1;
    const X:   u64 = 1 << 2;
    const L:   u64 = 1 << 7;
    let entry0 = TOR | R | W | X;       // 0x0F
    let entry1 = TOR;                   // 0x08
    let entry2 = TOR | R | L;           // 0x89
    let entry3 = TOR;                   // 0x08
    let entry4 = TOR | R | W | X;       // 0x0F
    let pmpcfg0 = entry0
        | (entry1 << 8)
        | (entry2 << 16)
        | (entry3 << 24)
        | (entry4 << 32);

    // Upper bound for "below kmain" — bl's text/data, stack, id_map_tables
    // and the M-mode trap frames at 0x80800000 all live below this.
    // kmain's load base is `0x80000000 + 64 MiB = 0x84000000`.
    const KMAIN_LOAD_BASE: u64 = 0x8400_0000;

    let pa0 = 0x8000_0000_u64 >> 2;
    let pa1 = unsafe { RODATA_START } >> 2;
    let pa2 = unsafe { RODATA_END } >> 2;
    let pa3 = KMAIN_LOAD_BASE >> 2;
    let pa4 = !0_u64 >> 2;

    unsafe {
        asm!(
            "csrw pmpaddr0, {p0}",
            "csrw pmpaddr1, {p1}",
            "csrw pmpaddr2, {p2}",
            "csrw pmpaddr3, {p3}",
            "csrw pmpaddr4, {p4}",
            "csrw pmpcfg0,  {cfg}",
            p0 = in(reg) pa0,
            p1 = in(reg) pa1,
            p2 = in(reg) pa2,
            p3 = in(reg) pa3,
            p4 = in(reg) pa4,
            cfg = in(reg) pmpcfg0,
        );
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
        // Note: PMP is already armed by `kinit` at the top — running it
        // again here would be a no-op for entries 0-1/3-4 (idempotent
        // CSR writes) but the L=1 lock on entry 2 makes that entry's
        // pmpcfg byte and pmpaddr2 immutable, so a re-write would
        // either silently no-op or fault depending on hardware.

        let hartid = riscv::register::mhartid::read();

        SYSINFO.dtb_addr.store(dtb_addr, Ordering::Relaxed);
        SYSINFO.serial.store(serial_addr, Ordering::Relaxed);

        // Publish kmain entry + wake the secondary harts. They've been
        // sitting in M-mode WFI inside `kinit_hart`; CLINT MSIP unblocks
        // the WFI, the M-mode trap clears it, and they resume the
        // KMAIN_ENTRY poll, see the new value, and sret to S-mode kmain
        // with bare satp (kmain's `_start` is at the load PA, identity).
        KMAIN_ENTRY.store(entrypoint, Ordering::Release);
        for hart in 0..4 {
            if hart != hartid {
                wake_hart(hart);
            }
        }

        asm!(
            "li t0, 0x02004000",          // mtimecmp address
            "li t1, 0x0200bff8",    // mtime address
            "ld t2, 0(t1)",               // Load current 64-bit mtime
            "li t3, 100000000",             // Example interval (1 million cycles)
            "add t2, t2, t3",             // t2 = mtime + interval
            "sd t2, 0(t0)",
            "mv sp, {s}",
            "fence w, rw", // Ensure ELF writes are visible to all harts
            "fence.i",    // Synchronize I-cache with D-cache
            "sfence.vma", // Flush the MMU TLB
            "sret",
            s = in(reg) KERNEL_STACK_END,
            in("a0") hartid,
            in("a1") dtb_addr,
            in("a2") serial_addr,
            options(noreturn),
        );
    }
}
