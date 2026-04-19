#![no_std]
#![no_main]

extern crate alloc;

use core::arch::{asm, global_asm, naked_asm};
use core::ptr::null_mut;
use core::sync::atomic::{Ordering};
use core::{alloc::Layout, panic::PanicInfo};

use device::{HartContext, SysInfo, TRAP_STACK_SIZE, find_ram};
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
    // Re-point serial at its KMMIO VA. rust_main init'd it at the raw PA under
    // the early trampoline satp; now that orbit_root_table is active, KMMIO
    // aliases exist and the eventual goal is to drop identity MMIO from the
    // kernel satp. Must happen before any println.
    unsafe {
        serial::init_serial(kmain::kernel::memmap::kmmio_uart() as usize);
    }

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
        // bl dereferences HART_ROOT in M-mode with no paging, so it must be a
        // physical address. hart_context is a KDMAP VA post-higher-half, so
        // translate at the boundary.
        let hart_root_phys =
            kmain::kernel::memmap::virt_to_phys_dmap(hart_context as *const _ as u64) as usize;

        // Tell bl how to turn a RAM PA into its KDMAP alias. bl uses this to
        // hand secondary harts a sscratch/sp that resolves under the kernel
        // satp (pool identity is no longer mapped).
        let kdmap_bias = kmain::kernel::memmap::kdmap_base()
            .wrapping_sub(kmain::kernel::memmap::ram_phys_base()) as usize;

        asm!(
            "li a6, 4",
            "mv a7, {0}",
            "ecall",
            in(reg) kdmap_bias
        );

        //signal bios to set hart root
        asm!(
            "li a6, 1",
            "mv a7, {0}",
            "ecall",
            in(reg) hart_root_phys
        );

        kmain::kick_machine_harts(4);
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
// Early paging tables, used only by the trampoline. 8 pages (32 KiB):
// room for root + L2(identity) + L2/L1/L0(high-half) with slack. Zero-
// initialized by bl's write_bytes over the PT_LOAD memsz-filesz gap — .bss
// lands in that gap.
#[repr(C, align(4096))]
struct EarlyPt([u64; 512 * 8]);

#[unsafe(no_mangle)]
#[used]
static mut EARLY_PT: EarlyPt = EarlyPt([0; 512 * 8]);

const EARLY_PT_SIZE: usize = core::mem::size_of::<EarlyPt>();

// Upper 32 bits of KTEXT_NOMINAL (0xFFFF_FFC0_0000_0000). The asm loads
// this as a signed 32-bit immediate and sllis by 32 to reconstruct the
// full constant — a 64-bit `li` isn't portable in LLVM's assembler. When
// KASLR lands, the trampoline returns a runtime ktext_base instead of
// the asm baking this constant in.
const KTEXT_NOMINAL_HI32: u64 = kmain::kernel::memmap::KTEXT_NOMINAL >> 32;

#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // bl enters with a0=hartid, a1=sysinfo. Preserve them across the call
        // to early_paging_setup via callee-saved s-registers.
        "auipc t0, 0",              // t0 = physical VA of _start (load_addr)
        "mv s0, t0",                // s0 = load_addr (callee-saved)
        "mv s1, a0",                // s1 = hartid
        "mv s2, a1",                // s2 = sysinfo

        // early_paging_setup(pt_base, pt_size, load_addr) -> satp. Args
        // computed PC-relative / as immediates; no GOT, no relocated globals.
        "lla a0, {early_pt}",
        "li  a1, {early_pt_size}",
        "mv  a2, t0",
        "call {early_paging_setup}",

        // Install the early satp. PC is still at physical; the early PT
        // identity-maps RAM so instruction fetch across this boundary works.
        "csrw satp, a0",
        "sfence.vma",

        // Compute high-half VA of post_trampoline_entry. The high-half
        // mapping is `VA = KTEXT_NOMINAL + X`, `PA = load_addr + X`, so a
        // symbol at runtime PA `lla(post_tramp)` lives at high-half VA
        // `KTEXT_NOMINAL + (lla(post_tramp) - load_addr)`.
        "lla t1, {post_tramp}",     // t1 = physical VA of post_tramp
        "sub t2, t1, s0",           // t2 = X = post_tramp_phys - load_addr
        "li  t3, {ktext_hi32}",     // t3 (sign-extended) = 0xFFFF_FFFF_FFFF_FFC0
        "slli t3, t3, 32",          // t3 = 0xFFFF_FFC0_0000_0000 (KTEXT_NOMINAL)
        "add t4, t3, t2",           // t4 = high-half VA of post_tramp

        // Args for post_trampoline_entry(hartid, sysinfo, ktext_base, load_addr).
        "mv a0, s1",
        "mv a1, s2",
        "mv a2, t3",                // ktext_base
        "mv a3, s0",                // load_addr

        "jr t4",
        early_pt = sym EARLY_PT,
        early_pt_size = const EARLY_PT_SIZE,
        early_paging_setup = sym early_paging_setup,
        post_tramp = sym post_trampoline_entry,
        ktext_hi32 = const KTEXT_NOMINAL_HI32,
    );
}

/// Build the early page table and return its satp value. Runs pre-jump at
/// physical PC, so it must not touch any static — every input comes through
/// parameters. Builds the table via `PageTableVec` (bump allocator over
/// `pt_base`) and the existing mmu helpers so the PTE format stays in one
/// place.
///
/// Entries installed:
///   identity gigapage  [0, 1 GiB)                  — MMIO (UART, CLINT, ACLINT, e1000)
///   identity gigapages [2, 4 GiB)                  — all of RAM
///   high-half          KTEXT_NOMINAL..+2 MiB       — aliases `[load_addr, load_addr+2MB)`
///                                                    so symbol S at linked LINK_BASE+N is
///                                                    accessible at KTEXT_NOMINAL + N.
///
/// The identity half keeps the subsequent asm executing after `csrw satp`.
/// rust_main later installs the full final satp.
///
/// On any failure the function spins — pre-relocation, panicking would try
/// to format through relocated globals and crash harder.
#[unsafe(no_mangle)]
#[inline(never)]
unsafe extern "C" fn early_paging_setup(pt_base: *mut u8, pt_size: usize, load_addr: u64) -> u64 {
    use mmu::{MappingConfig, PAGE_SIZE, PagePermissions};
    use mmu::mmap::{PageAlloc, PageTableVec, RootTable, id_map_range, map_va_range};
    use mmu::sv48::{PhysAddr, VirtAddr};

    let mut ptv = PageTableVec::new(pt_base as usize, pt_size);
    let Ok(root_ref) = (unsafe { ptv.allocate_page_table() }) else {
        loop { unsafe { riscv::asm::wfi(); } }
    };
    // Early trampoline tables live in identity-mapped RAM (bias = 0).
    let root = RootTable::identity(root_ref);
    let mut pages = PageAlloc::PTV(&mut ptv);

    let perms = PagePermissions::R as u64
              | PagePermissions::W as u64
              | PagePermissions::X as u64
              | PagePermissions::G as u64;

    let cfg = MappingConfig {
        permissions: perms,
        levels: 4,
        page_size: PAGE_SIZE as u64,
        vaddr: VirtAddr::new(0),
        paddr: PhysAddr::new(0),
        log: false,
        supervisor_tag: None,
    };

    // Identity [0, 1 GiB) — low-half MMIO range
    if unsafe { id_map_range(&root, &mut pages, cfg.copy(), 0..(1u64 << 30)) }.is_err() {
        loop { unsafe { riscv::asm::wfi(); } }
    }
    // Identity [2, 4 GiB) — all of RAM (kernel image, kheap, kpages, ktables, dtb)
    if unsafe { id_map_range(&root, &mut pages, cfg.copy(), (2u64 << 30)..(4u64 << 30)) }.is_err() {
        loop { unsafe { riscv::asm::wfi(); } }
    }

    // High-half kernel image: 2 MiB at KTEXT_NOMINAL -> load_addr. One 2 MiB
    // window is enough — the kernel is < 1 MiB. VA and PA increment together
    // so symbol at linked LINK_BASE+X lands at VA KTEXT_NOMINAL + X and PA
    // load_addr + X, which matches the convention the final satp uses.
    let ktext = kmain::kernel::memmap::KTEXT_NOMINAL;
    let len = 2u64 * 1024 * 1024;
    if unsafe { map_va_range(&root, &mut pages, cfg.copy(), ktext, load_addr..(load_addr + len)) }.is_err() {
        loop { unsafe { riscv::asm::wfi(); } }
    }

    // KDMAP: 2 GiB of RAM at KDMAP_NOMINAL → [2 GiB, 4 GiB). Both ends are
    // 1 GiB-aligned so map_va_range emits two gigapages. Needed here (not just
    // in the final satp) so rust_main can initialize KHEAP/kpages through
    // their KDMAP VAs before the final satp is installed.
    let kdmap = kmain::kernel::memmap::KDMAP_NOMINAL;
    if unsafe { map_va_range(&root, &mut pages, cfg.copy(), kdmap, (2u64 << 30)..(4u64 << 30)) }.is_err() {
        loop { unsafe { riscv::asm::wfi(); } }
    }

    // satp: Sv48 (mode=9), asid=0, ppn = root / 4096. Early tables are
    // identity, so the table VA is the PA.
    let root_ppn = (root.table as *const _ as u64) / (PAGE_SIZE as u64);
    (9u64 << 60) | root_ppn
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

const R_RISCV_RELATIVE: u64 = 3;
const DT_NULL:    u64 = 0;
const DT_RELA:    u64 = 7;
const DT_RELASZ:  u64 = 8;
const DT_RELAENT: u64 = 9;

/// First Rust code to run after the trampoline. PC is now at high-half. The
/// relocation walker has NOT run yet — any access to a relocated global would
/// UB. Keep this function to: fetch `_DYNAMIC` via PC-relative lla, apply
/// relocations with slide = ktext_base - LINK_BASE, then tail-call rust_main.
#[unsafe(no_mangle)]
#[inline(never)]
unsafe extern "C" fn post_trampoline_entry(
    hartid: usize,
    sysinfo: usize,
    ktext_base: u64,
    load_addr: u64,
) -> ! {
    unsafe {
        let dynamic_section: *const Elf64Dyn;
        core::arch::asm!(
            "lla {out}, _DYNAMIC",
            out = out(reg) dynamic_section,
            options(nomem, nostack, preserves_flags),
        );

        let slide = ktext_base.wrapping_sub(kmain::kernel::memmap::LINK_BASE);
        apply_relocations(slide, dynamic_section);

        rust_main(hartid, sysinfo, load_addr);
    }
}

#[inline(never)]
unsafe fn apply_relocations(slide: u64, dynamic_section: *const Elf64Dyn) {
    unsafe {
        let mut rela_base: *const Elf64Rela = core::ptr::null();
        let mut rela_size = 0u64;
        let mut rela_ent  = 0u64;

        let mut current = dynamic_section;
        while (*current).tag != DT_NULL {
            match (*current).tag {
                DT_RELA    => rela_base = ((*current).val.wrapping_add(slide)) as *const Elf64Rela,
                DT_RELASZ  => rela_size = (*current).val,
                DT_RELAENT => rela_ent  = (*current).val,
                _ => {}
            }
            current = current.add(1);
        }

        if rela_base.is_null() || rela_ent == 0 {
            return;
        }

        let count = rela_size / rela_ent;
        for i in 0..count {
            let entry = &*rela_base.add(i as usize);
            if (entry.info & 0xFFFFFFFF) == R_RISCV_RELATIVE {
                let target_addr = (entry.offset.wrapping_add(slide)) as *mut u64;
                *target_addr = (entry.addend as u64).wrapping_add(slide);
            }
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn rust_main(_hartid: usize, sysinfo: usize, load_addr: u64) -> ! {
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

        // Publish the full layout including kernel_phys_base (`load_addr` from
        // the trampoline). map_kernel_shared/self reads kernel_phys_base to
        // compute physical addresses for ELF regions now that rust_main itself
        // runs at high-half (so `&_text_start as u64` no longer equals the PA).
        kmain::kernel::memmap::init_layout(
            ram_base,
            kmain::kernel::memmap::KTEXT_NOMINAL,
            kmain::kernel::memmap::KDMAP_NOMINAL,
            kmain::kernel::memmap::KMMIO_NOMINAL,
            load_addr,
        );

        let layout = kmain::kernel::memmap::KernelLayout::new(
            ram_base, ram_size, dtb_addr as u64, serial_addr as u64,
        );

        // Zero the page-table pool via identity (valid under the early PT).
        core::ptr::write_bytes(layout.ktables.start as *mut u8, 0, layout.ktables.end.saturating_sub(layout.ktables.start) as usize);

        // Initialize KHEAP through its KDMAP VA. Allocator-returned pointers
        // are KDMAP VAs from here on — they stay valid after identity pools
        // are eventually dropped.
        KHEAP.make_guard_unchecked()
            .init(
                kmain::kernel::memmap::phys_to_virt(layout.kheap.start) as *mut u8,
                kmain::kernel::memmap::KHEAP_SIZE as usize,
            );

        static LOGGER: kmain::ktrace::OrbitLogger = kmain::ktrace::OrbitLogger;

        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
        tracing::subscriber::set_global_default(OrbitSubscriber::new(Level::TRACE))
            .expect("no tracing");

        let mut kernel_tables = FrameAllocator::<33>::new();
        kernel_tables.add_frame_with_va_base(
            layout.ktables.start as usize,
            layout.ktables.end   as usize,
            kmain::kernel::memmap::phys_to_virt(layout.ktables.start) as usize,
        );

        // Allocator hands back KDMAP VAs; that's the supervisor-visible
        // address we deref through. RootTable carries the matching bias so
        // walkers can convert PPNs (always physical) back to KDMAP VAs.
        let orbit_root_ref = (kernel_tables.alloc_aligned(Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE))
            .unwrap() as *const PageTable).as_ref_unchecked();
        let orbit_root_table = kmain::kernel::memmap::kernel_root(orbit_root_ref);

        println!("ort=0x{:016X?}", orbit_root_ref as *const _ as usize);

        {
            let mut pages = PageAlloc::FA(&mut kernel_tables);
            map_kernel_self(&orbit_root_table, &mut pages, &layout)
                .expect("failed to map kernel self-view");
        }

        let mut kpages = FrameAllocator::<33>::new();
        kpages.add_frame_with_va_base(
            layout.kpages.start as usize,
            layout.kpages.end   as usize,
            kmain::kernel::memmap::phys_to_virt(layout.kpages.start) as usize,
        );

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
        // satp takes the physical PPN; orbit_root_ref is a KDMAP VA, so
        // translate back to PA.
        let orbit_root_pa =
            kmain::kernel::memmap::virt_to_phys_dmap(orbit_root_ref as *const _ as u64) as usize;
        satp.set_ppn(orbit_root_pa / PAGE_SIZE);

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

        unmap_boot_only_regions(&orbit_root_table)
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
