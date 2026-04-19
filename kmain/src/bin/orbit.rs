#![no_std]
#![no_main]

extern crate alloc;

use core::arch::{asm, global_asm, naked_asm};
use core::ptr::null_mut;
use core::sync::atomic::{Ordering};
use core::{alloc::Layout, panic::PanicInfo};

use device::{HartContext, SysInfo, TRAP_STACK_SIZE, find_ram, wake_harts};
use kmain::ktrace::OrbitSubscriber;
use kmain::{check_context_and_switch, supervisor_clear_ipi};
use kmain::kernel::Orbit;
use kmain::kernel::context::{enter_hart_context, fault_thread};
use kmain::kernel::memmap::{map_kernel_self, unmap_boot_only_regions};
use mmu::{GB, MB};
use mmu::mmap::PageAlloc;
use mmu::{PAGE_SIZE, sv48::PageTable};
use process::{FaultInfo, ThreadState};
use riscv::register::satp::Satp;
use riscv::{register::{satp::Mode, stvec::{Stvec, TrapMode}}};

use linked_list_allocator::LockedHeap;

use mem::{frame::FrameAllocator, round_u64_up};
use serial::println;

use tracing::{Level};

use device::TrapFrame;

global_asm!(
    ".attribute arch, \"rv64gc\"",
    include_str!("../../asm/trap.S"),
);


#[global_allocator]
static KHEAP: LockedHeap = LockedHeap::empty();

unsafe extern "C" {
    unsafe fn s_trap_vector();
}

fn setup_interrupts() {
    unsafe {
        riscv::register::sstatus::set_sie();
        riscv::register::sie::set_stimer();
        riscv::register::sie::set_ssoft();
        riscv::register::sie::set_sext();
    }
}

#[unsafe(no_mangle)]
extern "C" fn s_trap(
    epc: usize,
    tval: usize,
    cause: usize,
    status: usize,
    frame: &mut TrapFrame,
    _code: usize, _sarg: usize)
    -> usize
{
    let hart_context = unsafe {
        (riscv::register::sscratch::read() as *mut HartContext).as_mut_unchecked()
    };

    let cause_num = cause & 0xfff;
	let mut return_pc = epc;
    let is_async = {
		if cause >> 63 & 1 == 1 {
			true
		}
		else {
			false
		}
	};
    // sstatus.SPP (bit 8): 0 = trap from U, 1 = trap from S.
    let from_user = (status >> 8) & 1 == 0;

    if is_async {
        match cause_num {
            1 => {
                unsafe {
                    supervisor_clear_ipi(hart_context.hart_id as usize);
                    riscv::register::sip::clear_ssoft();

                    kmain::update_thread_and_trap_frame(epc, hart_context, frame, from_user);

                    check_context_and_switch();
                }
            },
            5 | 7 => {
                unsafe {
                    // write stimecmp
                    const DISABLE: usize = usize::MAX;
                    asm!(
                        "csrw 0x14D, {}",
                        in(reg) DISABLE
                    );
                    //riscv::write_csr_as_usize!(0x14D, DISABLE);

                    riscv::register::sie::clear_stimer();

                    kmain::update_thread_and_trap_frame(epc, hart_context, frame, from_user);

                    check_context_and_switch();
                }
            }
            c => {
                println!("unhandled kint {c}");
            }
        }
    }
    else {
        match cause_num {
            // Instruction access fault.
            1 | 12 | 13 | 15 => {
                if !from_user {
                    panic!("S-mode fault on cpu{}: cause={} epc={:#x} stval={:#x}",
                        hart_context.hart_id, cause_num, epc, tval);
                }
                unsafe { fault_thread(FaultInfo { cause: cause_num, epc, stval: tval }); }
            }
            // supervisor ebreak
            3 => {
                match hart_context.cscratch2 {
                    1 => {
                        //serial::println!("smode ebreak call");

                        hart_context.cscratch2 = 0;
                        kmain::update_thread_and_trap_frame(epc, hart_context, frame, from_user);
                        check_context_and_switch();
                    },
                    _ => ()
                }

                return_pc += 4;
            }
            8 => {
                let syscall = frame.regs[10];
                match syscall {
                    // exit
                    0 => {
                        unsafe {
                            kmain::update_thread_and_trap_frame(epc, hart_context, frame, from_user);
                            kmain::kernel::context::exit_thread_with_state(ThreadState::Exited);
                        }
                    },
                    1 => {
                        serial::println!("orbit handling u mode ecall({syscall})");
                        kmain::handle_serial_print(epc, hart_context, frame);
                    }
                    2 => {
                        kmain::handle_ms_sleep(epc, hart_context, frame);
                    }
                    4096 => {
                        serial::println!("orbit handling u mode ecall({syscall})");
                        kmain::handle_mmap_req(epc, hart_context, frame);
                    }
                    4097 => {
                        serial::println!("orbit handling u mode ecall({syscall})");
                        kmain::handle_nc_registration_req(epc, hart_context, frame);
                    }
                    _ => {
                        serial::println!("orbit handling u mode ecall({syscall})");
                        kmain::update_thread_and_trap_frame(epc + 4, hart_context, frame, from_user);
                    }
                }
                check_context_and_switch();
            }
            _ => {
                if !from_user {
                    panic!("S-mode unhandled sync trap on cpu{}: cause={} epc={:#x} stval={:#x}",
                        hart_context.hart_id, cause_num, epc, tval);
                }
                unsafe { fault_thread(FaultInfo { cause: cause_num, epc, stval: tval }); }
            }
        }
    }
    return_pc
}

#[unsafe(no_mangle)]
extern "C" fn k_harthello() {
    //println!("hey there");

    let hart_context = unsafe {
        (riscv::register::sscratch::read() as *const HartContext).as_ref_unchecked()
    };

    unsafe {
        //println!("hart_context @ {:016X?} hartid={} kptr={:016X?}", hart_context as *const _, hart_context.hart_id, hart_context.kptr.load(Ordering::Relaxed));

        hart_context.kptr.store(kmain::k_hart_loop as *mut (), Ordering::Relaxed);

        let s_trap_addr = { s_trap_vector as *const () as usize };
        riscv::register::stvec::write(Stvec::new(s_trap_addr, TrapMode::Direct));

        setup_interrupts();

        enter_hart_context(hart_context);
    }
}

// only gets called by hart 0
#[unsafe(no_mangle)]
pub extern "C" fn k_smpstart() {
    let hart_context = unsafe {
        (riscv::register::sscratch::read() as *const HartContext).as_ref_unchecked()
    };

    let orbit = unsafe {
        (hart_context.cscratch as *mut kmain::kernel::Orbit).as_mut_unchecked()
    };

    orbit.create_new_process(kmain::kernel::UMODE_TEST_ELF, kmain::kernel::UPROC_STACK_DEFAULT)
        .expect("no test uprocess");

    orbit.get_environment_info();

    unsafe {
        //signal bios to set hart root
        asm!(
            "li a6, 1",
            "mv a7, {0}",
            "ecall",
            in(reg) hart_context as *const _ as usize
        );

        wake_harts(4);
    }

    k_harthello();
}

/*
#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        "la sp, {stack_end}",
        "j {rust_main}",
        stack_end = sym _stack_end,
        rust_main = sym rust_main
    );
}
*/
#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // t0 = runtime &_start (auipc is the first instruction here)
        "auipc t0, 0",

        // t1 = linked &_start. Must be absolute — `la` would be PC-relative
        // and give the runtime address again, collapsing the slide to 0.
        // 0x1000 comes from memory.x ORIGIN + .text ALIGN(4096); this relies
        // on _start living in .text.init so it lands at _text_start.
        "li t1, 0x1000",

        // t2 = slide (unused here, but kept for future use)
        "sub t2, t0, t1",

        // lla (load local address) forces the PC-relative auipc+addi form.
        // Plain `la` may expand to auipc+ld through .got, which requires
        // relocations that haven't been applied yet.
        "lla t3, _DYNAMIC",

        "mv a2, t1",   // reloc_base (linked)
        "mv a3, t0",   // load_addr  (runtime)
        "mv a4, t3",   // dynamic_section (runtime)

        "jal {relocate_rust}",
        relocate_rust = sym relocate_rust
    );
}

#[repr(C)]
pub struct Elf64Dyn {
    pub tag: u64,
    pub val: u64,
}

#[repr(C)]
pub struct Elf64Rela {
    pub offset: u64,
    pub info: u64,
    pub addend: i64,
}

// RISC-V specific constants
const R_RISCV_RELATIVE: u64 = 3; // Relocation type for relative addressing
const DT_NULL: u64 = 0;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_RELAENT: u64 = 9;

// This function MUST be #[inline(never)] and only use local variables
#[unsafe(no_mangle)]
#[inline(never)]
unsafe extern "C" fn relocate_rust(hartid: usize, sysinfo: usize, reloc_base: u64, load_addr: u64, dynamic_section: *const Elf64Dyn) -> ! {
    // The "slide" is the difference between where we are and where we linked
    // Note: For KASLR, this is usually `load_addr - reloc_base`.
    // Ensure your arithmetic handles wrapping if you are moving across high/low memory.
    let slide = load_addr.wrapping_sub(reloc_base);

    let mut rela_base: *const Elf64Rela = core::ptr::null();
    let mut rela_size = 0;
    let mut rela_ent = 0;

    let mut current = dynamic_section;
    while (*current).tag != DT_NULL {
        match (*current).tag {
            DT_RELA => {
                // The value in DT_RELA is a virtual address (compile-time).
                // We must apply the slide to find it in physical memory.
                rela_base = ((*current).val.wrapping_add(slide)) as *const Elf64Rela;
            }
            DT_RELASZ => rela_size = (*current).val,
            DT_RELAENT => rela_ent = (*current).val,
            _ => {}
        }
        current = current.add(1);
    }

    if !rela_base.is_null() && rela_ent > 0 {
        let count = rela_size / rela_ent;
        for i in 0..count {
            let entry = &*rela_base.add(i as usize);
            
            // The relocation type is in the lower 32 bits of info
            if (entry.info & 0xFFFFFFFF) == R_RISCV_RELATIVE {
                // Address to fix up: linked address (offset) + slide
                let target_addr = (entry.offset.wrapping_add(slide)) as *mut u64;
                
                // New value: original addend + slide
                *target_addr = (entry.addend as u64).wrapping_add(slide);
            }
        }
    }

    // Now it's safe to jump to the rest of the kernel!
    rust_main(hartid, sysinfo);
}

#[unsafe(no_mangle)]
extern "C" fn rust_main(_hartid: usize, sysinfo: usize) -> ! {
    unsafe {
        // 1. Sync data across cores
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        // 2. Invalidate local I-Cache
        riscv::asm::fence_i();
        // 3. Invalidate local TLB
        riscv::asm::sfence_vma_all();

        riscv::register::sstatus::clear_sie();
        riscv::register::sie::clear_stimer();
        riscv::register::sie::clear_ssoft();

        let sysinfo = (sysinfo as *const SysInfo)
            .as_ref_unchecked();

        let dtb_addr = sysinfo.dtb_addr.load(Ordering::Acquire);
        let serial_addr = sysinfo.serial.load(Ordering::Acquire);

        serial::init_serial(serial_addr as usize);

        println!("boot! dtb @ {dtb_addr:016X?}");
        
        let (ram_base, ram_size) = find_ram(dtb_addr as *const u8)
            .expect("failed to find RAM node in DTB");
        // reserve a 2 MiB guard at the top of RAM for the DTB
        let mem_end: u64 = ram_base + ram_size - (2 * MB);

        // page tables
        const KTABLES_SIZE: u64 = 128 * MB;
        let ktables_start: u64 = mem_end - KTABLES_SIZE;

        // general purpose alloc heap
        const KHEAP_SIZE: u64 = 128 * MB;
        let kheap_start: u64 = ktables_start - KHEAP_SIZE;

        // pages for stacks, etc. well-aligned allocations
        const KPAGES_SIZE: u64 = 128 * MB;
        let kpages_start: u64 = kheap_start - KPAGES_SIZE;

        core::ptr::write_bytes(ktables_start as *mut u8, 0, KTABLES_SIZE as usize);

        KHEAP.make_guard_unchecked()
            .init(kheap_start as *mut u8, KHEAP_SIZE as usize);

        static LOGGER: kmain::ktrace::OrbitLogger = kmain::ktrace::OrbitLogger;

        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
        tracing::subscriber::set_global_default(OrbitSubscriber::new(Level::TRACE))
            .expect("no tracing");

        let mut kernel_tables = FrameAllocator::<33>::new();
        kernel_tables.add_frame(ktables_start as usize, mem_end as usize);

        let orbit_root_table = (kernel_tables.alloc_aligned(Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE))
            .unwrap() as *const PageTable).as_ref_unchecked();

        println!("ort=0x{:016X?}", orbit_root_table as *const _ as usize);

        let layout = kmain::kernel::KernelLayout {
            kheap: kheap_start..(kheap_start + KHEAP_SIZE),
            kpages: kpages_start..(kpages_start + KPAGES_SIZE),
            ktables: ktables_start..(ktables_start + KTABLES_SIZE),
            dtb: (dtb_addr as u64)..(dtb_addr as u64 + (2 * MB)),
            serial: serial_addr as u64,
        };

        {
            let mut pages = PageAlloc::FA(&mut kernel_tables);
            map_kernel_self(orbit_root_table, &mut pages, &layout)
                .expect("failed to map kernel self-view");
        }

        let mut kpages = FrameAllocator::<33>::new();
        kpages.add_frame(kpages_start as usize, (kpages_start + KPAGES_SIZE) as usize);

        let cpu_count = 4;
        let context_size = cpu_count * core::mem::size_of::<HartContext>();
        let hart_contexts = {
            kpages.alloc_aligned(Layout::from_size_align_unchecked(context_size, 4096))
                .expect("failed to alloc hart contexts")
                as *mut HartContext
        };

        let mut satp = Satp::from_bits(0);
        satp.set_asid(0);
        satp.set_mode(Mode::Sv48);
        satp.set_ppn(orbit_root_table as *const _ as usize / PAGE_SIZE);

        let orbit = {
            let orbit_ptr = (kpages.alloc_aligned(
                Layout::from_size_align_unchecked(
                    round_u64_up(core::mem::size_of::<Orbit>() as u64, 4096) as usize,
                    4096))
                    .expect("failed to alloc space for kernel state")
                    as *mut Orbit)
                    .as_mut_unchecked();

            *orbit_ptr = Orbit::new(dtb_addr as usize, serial_addr as usize, cpu_count, layout, kernel_tables, kpages, satp.clone());

            orbit_ptr
        };

        println!("allocated orbit state @ {:016X?}", &raw const *orbit as usize);

        let hart_root = hart_contexts as usize;
        riscv::register::sscratch::write(hart_root);

        println!("allocated hart contexts @ {hart_root:016X?}");

        let s_trap_addr = { s_trap_vector as *const () as usize };
        riscv::register::stvec::write(Stvec::new(s_trap_addr, TrapMode::Direct));

        for hart in 0..cpu_count {
            let ptr = hart_contexts.add(hart);
            let hart_context = ptr.as_mut_unchecked();
            let target = k_harthello as *mut ();

            hart_context.kptr.store(target, Ordering::Relaxed);
            hart_context.current.store(null_mut(), Ordering::Release);
            hart_context.hart_id = hart as u64;
            hart_context.satp = satp;
            hart_context.s_trap_addr = s_trap_addr as u64;
            hart_context.cscratch2 = 0;
            hart_context.cscratch = orbit as *mut _ as u64;
            hart_context.tsp =
                &hart_context.trap_stack.stack_data[hart_context.trap_stack.stack_data.len() - 16]
                as *const _ as usize;

            println!("setting hart context @ {ptr:016X?} to kidle hart{hart}");
        }

        let this_sp = hart_contexts.as_ref_unchecked().k_stack.stack_data.as_ptr() as usize + TRAP_STACK_SIZE - 16;
        let this_pc = k_smpstart as *const ();

        riscv::register::sepc::write(this_pc as usize);
        riscv::register::sstatus::set_spp(riscv::register::sstatus::SPP::Supervisor);

        unmap_boot_only_regions(orbit_root_table)
            .expect("failed to unmap boot-only regions");

        println!("jump sp={this_sp:016X} pc={this_pc:016X?}");

        asm!(
            "fence.i",
            "fence w, w",
            "sfence.vma",
            "csrw satp, {p}",     // 2. Enable the new MMU map
            "sfence.vma",         // 3. Flush TLB so new map takes effect
            "mv sp, {s}",         // 1. Switch to the new hart-specific stack
            "sret",               // 4. Jump to sepc (Kernel Idle)
            s = in(reg) this_sp,
            p = in(reg) satp.bits(),
            options(noreturn)
        );
    }
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    println!("{p:?}");
    loop{riscv::asm::wfi();}
}

#[unsafe(no_mangle)]
pub extern "C" fn _start_rust() {}

#[unsafe(no_mangle)]
pub extern "C" fn _start_trap_rust() {}
