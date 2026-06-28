#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, Ordering};

use device::TrapFrame;
use riscv::register::mstatus::SPP;
use riscv::register::mtvec::Mtvec;

use bl::setup_interrupts;
use device::*;
use serial::{init_serial, println};

global_asm!(
    ".attribute arch, \"rv64gc\"",
    include_str!("../../asm/boot.S")
);
global_asm!(
    ".attribute arch, \"rv64gc\"",
    include_str!("../../asm/mem.S")
);
global_asm!(
    ".attribute arch, \"rv64gc\"",
    include_str!("../../asm/trap.S")
);

unsafe extern "C" {
    unsafe static KERNEL_STACK_END: u64;
    unsafe fn m_trap_vector();

    unsafe static BSS_START: u64;
    unsafe static BSS_END: u64;

    unsafe static KERNEL_STACK_START: u64;
}

// Set by the first hart to take an unhandled M-mode trap. Other harts'
// cause-3 arm checks this after clear_hart_int — a wake from a faulted
// hart redirects them into the same WFI halt instead of returning to
// whatever they were doing (S-mode kmain, kinit_hart's KMAIN_ENTRY
// poll, etc.). One bad hart stops the world.
static M_FAULTED: AtomicBool = AtomicBool::new(false);

fn broadcast_halt(self_hart: usize) {
    M_FAULTED.store(true, Ordering::Release);
    for h in 0..4 {
        if h != self_hart {
            unsafe {
                wake_hart(h);
            }
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn m_trap(
    epc: usize,
    tval: usize,
    cause: usize,
    hart: usize,
    _status: usize,
    _frame: &mut TrapFrame,
    _code: usize,
    _sarg: usize,
) -> usize {
    let is_async = { if cause >> 63 & 1 == 1 { true } else { false } };
    // The cause contains the type of trap (sync, async) as well as the cause
    // number. So, here we narrow down just the cause number.
    let cause_num = cause & 0xfff;
    let mut return_pc = epc;
    if is_async {
        // Asynchronous trap. Only cause 3 (machine software) should ever
        // reach here in normal operation: bl uses CLINT MSIPs to kick
        // secondaries out of `kinit_hart`'s WFI poll on KMAIN_ENTRY, and
        // m_trap clears them on the receiving hart.
        //
        // Causes 1/5/9 (supervisor software/timer/external) are
        // delegated to S-mode by `mideleg.SSIE/STIE/SEIE`, so they fire
        // at stvec, not here. Causes 7/11 (machine timer/external)
        // require `mie.MTIE`/`mie.MEIE` to fire and bl never sets
        // either. Anything that lands in the catch-all is a bug — halt
        // the hart so the failure is loud.
        match cause_num {
            3 => unsafe {
                clear_hart_int(hart);
                // A wake from a faulted hart redirects us into the
                // halt loop instead of returning. Checked after the
                // MSIP clear so the pending bit doesn't immediately
                // re-trap us.
                if M_FAULTED.load(Ordering::Acquire) {
                    loop {
                        riscv::asm::wfi();
                    }
                }
            },
            _ => {
                println!(
                    "Unhandled M-mode async trap CPU#{} -> {}\n",
                    hart, cause_num
                );
                broadcast_halt(hart);
                loop {
                    riscv::asm::wfi();
                }
            }
        }
    }
    else {
        // Synchronous trap. medeleg now delegates breakpoints, ecalls
        // (U+S), instruction-access/page-fault, load-page-fault and
        // store-page-fault to S-mode, so kmain's s_trap handles all of
        // those. The only synchronous traps that should reach M-mode
        // are bugs: PMP/access faults from M-mode itself, illegal
        // instructions, ecalls bl issues against itself (it doesn't),
        // or anything bl::setup_interrupts forgot to delegate. Halt
        // loudly with the cause and tval so the failure is debuggable.
        match cause_num {
            9 => {
                // Ecall-from-S is delegated by medeleg.set_supervisor_env_call,
                // but if it ever leaks into M-mode just skip past it
                // instead of looping the hart.
                return_pc += 4;
            }
            _ => {
                println!(
                    "Unhandled M-mode sync trap CPU#{} -> cause={} epc=0x{:08x} stval=0x{:08x}\n",
                    hart, cause_num, epc, tval,
                );
                broadcast_halt(hart);
                loop {
                    riscv::asm::wfi();
                }
            }
        }
    };
    return_pc
}

#[unsafe(no_mangle)]
extern "C" fn kinit(hartid: usize, dtb_addr: usize) {
    // boot.S only routes hart 0 here (secondaries go to `kinit_hart`),
    // so no `if hartid == 0` gate is needed. M-mode satp stays bare:
    // kmain's `_start` builds its own trampoline page tables via
    // `early_paging_setup`, so bl no longer needs to hand-set an id-map.
    unsafe {
        let frame_offset = bl::TRAP_FRAMES.add(hartid);
        riscv::register::mscratch::write(frame_offset as usize);
        riscv::register::mtvec::write(Mtvec::new(
            m_trap_vector as *const () as usize,
            riscv::register::mtvec::TrapMode::Direct,
        ));
        // PMP first thing — locks rodata RO so any subsequent M-mode
        // stack overflow traps as a store-access fault instead of
        // silently chewing through bl. Also denies S-mode access to
        // the bl region; protection is in effect for the entire kinit
        // path, kmain_enter, and the eventual sret to S-mode.
        bl::setup_pmp();

        let dtb_addr = dtb_addr as *const u8;

        let addr = find_serial_port(dtb_addr).unwrap();
        init_serial(addr);

        println!("dtb @ {dtb_addr:016X?}");

        let (ram_start, ram_size) = find_ram(dtb_addr).unwrap();
        let mb = (ram_size as f64) / 1024. / 1024.;
        println!(
            "0x{:016x?}..0x{:016x?} ({:.02}MiB)",
            ram_start,
            ram_start + ram_size,
            mb
        );

        println!("BSS=0x{BSS_START:016X?}..0x{BSS_END:016X?}");
        println!("BLSTACK=0x{KERNEL_STACK_START:016X?}..{KERNEL_STACK_END:016X?}");
        println!(
            "KELF=0x{:016X?}..{:016X?}",
            bl::KERNEL_ELF.as_ptr() as usize,
            bl::KERNEL_ELF.as_ptr() as usize + bl::KERNEL_ELF_LEN
        );

        bl::kmain_enter(addr, dtb_addr as usize);
    }
}

#[panic_handler]
fn panic_time(_: &PanicInfo) -> ! {
    loop {
        riscv::asm::wfi();
    }
}

#[unsafe(no_mangle)]
extern "C" fn kinit_hart() {
    let hartid = riscv::register::mhartid::read();
    if hartid == 0 {
        loop {
            riscv::asm::wfi();
        }
    }

    // PMP first — same rationale as `kinit`: lock rodata RO and deny
    // S-mode access to bl before anything else. mscratch was set in the
    // common _start prologue and mtvec in boot.S part3, both before the
    // mret to here, so traps from PMP violations resolve correctly.
    unsafe {
        bl::setup_pmp();
    }

    // Wait for hart 0 (still in bl) to copy the kernel ELF into RAM and
    // publish its `_start` PA. CLINT MSIP from `kmain_enter`'s wake_hart
    // unblocks the WFI; the M-mode trap clears the pending bit, the loop
    // re-runs, sees KMAIN_ENTRY, and we sret straight to S-mode kmain.
    let entry = loop {
        let v = bl::KMAIN_ENTRY.load(Ordering::Acquire);
        if v != 0 {
            break v;
        }
        riscv::asm::wfi();
    };

    unsafe {
        // Per-hart M-mode delegation + Sstc enable. Used to live further
        // down inside the cooking block; pulled up so the sret path is
        // a thin tail.
        setup_interrupts();

        riscv::register::sepc::write(entry);
        riscv::register::mstatus::set_spp(SPP::Supervisor);
        // Bare satp: kmain's `_start` runs at the load PA. The S-mode
        // entry path (`_start_secondary` → trampoline satp →
        // `secondary_rust_setup`) drives the satp transitions itself.
        riscv::register::satp::write(riscv::register::satp::Satp::from_bits(0));

        let dtb_ptr = bl::SYSINFO.dtb_addr.load(Ordering::Acquire);
        let serial_ptr = bl::SYSINFO.serial.load(Ordering::Acquire);

        asm!(
            "fence w, rw",
            "fence.i",
            "sfence.vma",
            "sret",
            in("a0") hartid,
            in("a1") dtb_ptr,
            in("a2") serial_ptr,
            options(noreturn),
        );
    }
}
