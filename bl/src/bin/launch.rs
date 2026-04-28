#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::panic::PanicInfo;
use core::sync::atomic::{Ordering};

use mmu::mmap::{PageTableVec, RootTable, id_map_range, map_address_page};
use mmu::sv48::{PhysAddr, VirtAddr};
use device::{TrapFrame};
use riscv::register::mtvec::Mtvec;
use riscv::register::satp::Mode;
use riscv::register::mstatus::SPP;

use bl::{setup_interrupts};
use device::*;
use mmu::*;
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

#[unsafe(no_mangle)]
extern "C" fn m_trap(epc: usize,
                    tval: usize,
                    cause: usize,
                    hart: usize,
                    _status: usize,
                    _frame: &mut TrapFrame,
                    _code: usize, _sarg: usize)
                     -> usize
{
	let is_async = {
		if cause >> 63 & 1 == 1 {
			true
		}
		else {
			false
		}
	};
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
			3 => unsafe { clear_hart_int(hart); },
			_ => {
				println!("Unhandled M-mode async trap CPU#{} -> {}\n", hart, cause_num);
				loop { riscv::asm::wfi(); }
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
			},
			_ => {
				println!(
					"Unhandled M-mode sync trap CPU#{} -> cause={} epc=0x{:08x} stval=0x{:08x}\n",
					hart, cause_num, epc, tval,
				);
				loop { riscv::asm::wfi(); }
			}
		}
	};
	return_pc
}

#[unsafe(no_mangle)]
extern "C" fn kinit(hartid: usize, dtb_addr: usize) {
    unsafe {
        let frame_offset = bl::TRAP_FRAMES.add(hartid);
        riscv::register::mscratch::write(frame_offset as usize);
        riscv::register::mtvec::write(Mtvec::new(m_trap_vector as *const () as usize, riscv::register::mtvec::TrapMode::Direct));
    }

    if hartid == 0 {
        // load dtb and stuff
        /*
        a1 = 0x0000000087e00000
        a2 = 0x0000000000001028
         */
        unsafe {
            let dtb_addr = dtb_addr as *const u8;

            let addr = find_serial_port(dtb_addr).unwrap();
            init_serial(addr);

            println!("dtb @ {dtb_addr:016X?}");

            let test_va = VirtAddr::new(0xb8001000);
            println!("test_va=vpn0={},vpn1={},vpn2={},vpn3={}",
                test_va.vpn0(), test_va.vpn1(), test_va.vpn2(), test_va.vpn3()
            );
            
            let (ram_start, ram_size) = find_ram(dtb_addr).unwrap();

            let mb = (ram_size as f64) / 1024. / 1024.;
            println!("0x{:016x?}..0x{:016x?} ({:.02}MiB)", ram_start, ram_start+ram_size, mb);

            println!("BSS=0x{BSS_START:016X?}..0x{BSS_END:016X?}");
            println!("BLSTACK=0x{KERNEL_STACK_START:016X?}..{KERNEL_STACK_END:016X?}");
            println!("KELF=0x{:016X?}..{:016X?}", bl::KERNEL_ELF.as_ptr() as usize, bl::KERNEL_ELF.as_ptr() as usize + bl::KERNEL_ELF_LEN);

            const MAX_ID_TABLES: usize = 32;
            let page_table_start = bl::id_map_tables();
            let mut ptv = PageTableVec::new(page_table_start, 4096 * MAX_ID_TABLES);
            let mut pages = mmap::PageAlloc::PTV(&mut ptv);
            let root_pa = pages.allocate_page_table().unwrap();
            // PTV is identity-mapped in bl, so PA == VA. Zero before use —
            // PageAlloc no longer zeros internally.
            core::ptr::write_bytes(root_pa as *mut u8, 0, 4096);
            let root_ref = (root_pa as *const mmu::sv48::PageTable).as_ref_unchecked();
            let root_table = RootTable::identity(root_ref);

            println!("made page table pool @ {:016x}, root table @ {:016x}", page_table_start, root_pa as usize);

            let base_id_map_config = MappingConfig {
                permissions: PagePermissions::R | PagePermissions::W | PagePermissions::X,
                levels: 0, page_size: 0, vaddr: VirtAddr::new(0), paddr: PhysAddr::new(0),
                log: false,
                supervisor_tag: SupervisorTag::None
            };
            let id_mapping = id_map_range(&root_table, &mut pages, base_id_map_config, ram_start..(ram_start + ram_size));

            let serial_perms = MappingConfig {
                permissions: PagePermissions::R | PagePermissions::W,
                levels: 4,
                page_size: 4096,
                vaddr: VirtAddr::new(addr as u64),
                paddr: PhysAddr::new(addr as u64),
                log: false,
                supervisor_tag: SupervisorTag::None
            };
            map_address_page(&root_table, &mut pages, &serial_perms).unwrap();

            println!("{id_mapping:?}");

            riscv::asm::sfence_vma_all();
            let _ = riscv::register::satp::try_set(Mode::Sv48, 0, root_ref as *const _ as usize / PAGE_SIZE);
            riscv::asm::sfence_vma_all();

            println!("PTABLES=0x{:016X}..0x{:016X}", page_table_start, page_table_start + ptv.current_tables_size());

            bl::kmain_enter(addr, dtb_addr as usize);
        }
    }
}

#[panic_handler]
fn panic_time(_: &PanicInfo) -> ! {
    loop{riscv::asm::wfi();}
}

#[unsafe(no_mangle)]
extern "C" fn kinit_hart() {
    let hartid = riscv::register::mhartid::read();
    if hartid == 0 {
        loop { riscv::asm::wfi(); }
    }

    // Wait for hart 0 (still in bl) to copy the kernel ELF into RAM and
    // publish its `_start` PA. CLINT MSIP from `kmain_enter`'s wake_hart
    // unblocks the WFI; the M-mode trap clears the pending bit, the loop
    // re-runs, sees KMAIN_ENTRY, and we sret straight to S-mode kmain.
    let entry = loop {
        let v = bl::KMAIN_ENTRY.load(Ordering::Acquire);
        if v != 0 { break v; }
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
            "csrw pmpaddr0, {apmp}",
            "csrw pmpcfg0, {acfg}",
            "fence w, rw",
            "fence.i",
            "sfence.vma",
            "sret",
            apmp = in(reg) !0,
            acfg = in(reg) 0xf | 0x80,
            in("a0") hartid,
            in("a1") dtb_ptr,
            in("a2") serial_ptr,
            options(noreturn),
        );
    }
}