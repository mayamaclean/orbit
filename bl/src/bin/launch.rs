#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::panic::PanicInfo;
use core::sync::atomic::{Ordering};

use mmu::mmap::{PageTableVec, id_map_range, map_address_page};
use mmu::sv48::{PhysAddr, VirtAddr};
use device::{TrapFrame};
use riscv::register::mtvec::Mtvec;
use riscv::register::satp::Mode;
use riscv::register::mstatus::SPP;
use riscv::register::stvec::{Stvec, TrapMode as STrapMode};

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
                    code: usize, sarg: usize)
                     -> usize
{
    if cause == 0 {
        loop {
            unsafe {
                riscv::asm::wfi();
            }
        }
    }

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
		// Asynchronous trap
		match cause_num {
            1 => {
                // supervisor software
                unsafe {
                    riscv::register::mie::clear_ssoft();
                }
            },
			3 => {
				// Machine software
				//println!("Machine software interrupt CPU#{} 0x{return_pc:016X?}", hart);
                unsafe { 
                    clear_hart_int(hart);
                }
			},
            5 => (),
            7 => unsafe {
                let mtimecmp = (0x0200_4000 as *mut u64).add(hart);
                mtimecmp.write_volatile(u64::MAX);
                riscv::register::mie::clear_mtimer();
                mtimecmp.write_volatile(u64::MAX);

                //println!("machine trap timer CPU#{hart}");
            }
			11 => {
				// Machine external (interrupt from Platform Interrupt Controller (PLIC))
				println!("Machine external interrupt CPU#{}", hart);
			},
			_ => {
				println!("Unhandled async trap CPU#{} -> {}\n", hart, cause_num);
                loop{riscv::asm::wfi();}
			}
		}
	}
	else {
		// Synchronous trap
		match cause_num {
            1 => {
                println!("Unhandled sync trap CPU#{} -> 0x{:08x}: 0x{:08x}", hart, epc, tval);
                //return_pc += 4;
                loop{riscv::asm::wfi();}
            },
			2 => {
				// Illegal instruction
				println!("Illegal instruction CPU#{} -> 0x{:08x}: 0x{:08x}\n", hart, epc, tval);
                loop{riscv::asm::wfi();}
			},
            5 => {
				// Load access fault
				println!("Load access fault CPU#{} -> 0x{:08x}: 0x{:08x}\n", hart, epc, tval);
                loop{riscv::asm::wfi();}
			},
			8 => {
				// Environment (system) call from User mode
				println!("E-call from User mode! CPU#{} -> 0x{:08x}", hart, epc);
                loop{riscv::asm::wfi();}
				return_pc += 4;
			},
			9 => {
				// Environment (system) call from Supervisor mode
				
                //println!("E-call({code}, {sarg:016X}) from Supervisor mode! CPU#{} -> 0x{:08x}", hart, epc);

				return_pc += 4;

                match code {
                    1 => bl::HART_ROOT.store(sarg, Ordering::Release),
                    2 => (),
                    3 => unsafe {
                        wake_hart(sarg);
                    }
                    _ => ()
                }
			},
			11 => {
				// Environment (system) call from Machine mode
				println!("E-call from Machine mode! CPU#{} -> 0x{:08x}\n", hart, epc);
                return_pc += 4;
			},
			// Page faults
			12 => {
				// 2 fault
				println!("Instruction page fault CPU#{} -> 0x{:08x}: 0x{:08x}", hart, epc, tval);
				//return_pc += 4;
                loop{riscv::asm::wfi();}
			},
			13 => {
				// Load page fault
				println!("Load page fault CPU#{} -> 0x{:08x}: 0x{:08x}", hart, epc, tval);
				//return_pc += 4;
                loop{riscv::asm::wfi();}
			},
			15 => {
				// Store page fault
				println!("Store page fault CPU#{} -> 0x{:08x}: 0x{:08x}", hart, epc, tval);
				//return_pc += 4;
                loop{riscv::asm::wfi();}
			},
			_ => {
                println!("Unhandled sync trap CPU#{} -> 0x{:08x}: 0x{:08x}", hart, epc, tval);
                //return_pc += 4;
                loop{riscv::asm::wfi();}
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
            let page_table_start= bl::ID_MAP_TABLES as usize;
            let mut ptv = PageTableVec::new(page_table_start, 4096 * MAX_ID_TABLES);
            let mut pages = mmap::PageAlloc::PTV(&mut ptv);
            let root_table = pages.allocate_page_table().unwrap();

            println!("made page table pool @ {:016x}, root table @ {:016x}", page_table_start, root_table as *const _ as usize);

            let base_id_map_config = MappingConfig {
                permissions: PagePermissions::R | PagePermissions::W | PagePermissions::X,
                levels: 0, page_size: 0, vaddr: VirtAddr::new(0), paddr: PhysAddr::new(0),
                log: false,
                supervisor_tag: None
            };
            let id_mapping = id_map_range(root_table, &mut pages, base_id_map_config, ram_start..(ram_start + ram_size));

            let serial_perms = MappingConfig { 
                permissions: PagePermissions::R | PagePermissions::W,
                levels: 4,
                page_size: 4096,
                vaddr: VirtAddr::new(addr as u64),
                paddr: PhysAddr::new(addr as u64),
                log: false,
                supervisor_tag: None
            };
            map_address_page(root_table, &mut pages, &serial_perms).unwrap();

            println!("{id_mapping:?}");

            riscv::asm::sfence_vma_all();
            let _ = riscv::register::satp::try_set(Mode::Sv48, 0, root_table as *const _ as usize / PAGE_SIZE);
            riscv::asm::sfence_vma_all();

            bl::ID_MAP_ADDR.store(root_table as *const _ as u64, Ordering::Release);

            println!("PTABLES=0x{:016X}..0x{:016X}", bl::ID_MAP_TABLES, bl::ID_MAP_TABLES + ptv.current_tables_size());

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

    loop {
        riscv::asm::wfi();
        let hart_root = bl::HART_ROOT.load(Ordering::Acquire);
        if hart_root > 0 {
            //serial::println!("hart{hartid} waking up");

            unsafe {
                let hart_context = (hart_root as *const HartContext)
                    .add(hartid);

                riscv::register::sscratch::write(hart_context as usize);

                let hart_context = hart_context.as_ref_unchecked();
                let target = hart_context.kptr.load(Ordering::Acquire) as usize;
                let sp = hart_context.k_stack.stack_data.as_ptr() as usize + TRAP_STACK_SIZE - 16;
                
                riscv::register::stvec::write(Stvec::new(hart_context.s_trap_addr as usize, STrapMode::Direct));

                riscv::register::sepc::write(target);
                riscv::register::mstatus::set_spp(SPP::Supervisor);

                setup_interrupts();

                // 1. Sync data across cores
                core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
                // 2. Invalidate local I-Cache
                riscv::asm::fence_i();

                riscv::asm::sfence_vma_all();
                let _ = riscv::register::satp::write(hart_context.satp);
                riscv::asm::sfence_vma_all();

                let mtimecmp_addr = (0x02004000 as *mut u64).add(hartid);

                asm!(
                    "csrw pmpaddr0, {apmp}",
                    "csrw pmpcfg0, {acfg}",
                    "mv t0, {mt}",          // mtimecmp address for
                    "li t1, 0x0200bff8",          // mtime address
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
                    mt = in(reg) mtimecmp_addr,
                    s = in(reg) sp,
                    options(noreturn),
                );
            }
        }
    }
}