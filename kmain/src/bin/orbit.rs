#![no_std]
#![no_main]

extern crate alloc;

use core::arch::{asm, global_asm, naked_asm};
use core::ptr::null_mut;
use core::sync::atomic::{Ordering};
use core::{alloc::Layout, panic::PanicInfo};

use device::{HartContext, TRAP_STACK_SIZE, find_ram};
use kmain::ktrace::OrbitSubscriber;
use kmain::{check_context_and_switch, supervisor_clear_ipi};
use kmain::kernel::Orbit;
use kmain::kernel::context::{enter_hart_context, fault_thread};
use kmain::kernel::memmap::{map_kernel_self, unmap_boot_only_regions};
use mmu::mmap::PageAlloc;
use mmu::{PAGE_SIZE, sv48::PageTable};
use process::{FaultInfo, Thread, ThreadState};
use riscv::register::satp::Satp;
use riscv::{register::{satp::Mode, stvec::{Stvec, TrapMode}}};

use linked_list_allocator::LockedHeap;

use mem::{round_u64_up};
use serial::println;

use tracing::{Level, debug, error, info};

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
        // Program every sie bit *before* flipping sstatus.SIE on. With
        // the previous order the first `set_sie()` could be preempted
        // by a pending STIP, and the timer arm's context-switch to
        // `k_hart_loop` made the remaining `set_*()` calls unreachable
        // — sie.SEXT stayed 0 forever, so S-mode external interrupts
        // never fired.
        riscv::register::sie::set_sext();
        riscv::register::sie::set_ssoft();
        riscv::register::sie::set_stimer();
        riscv::register::sstatus::set_sie();
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

    // Bucket hook 1: trap entry. Whatever bucket the hart was in
    // (User on a syscall/timer from user-mode, Idle if a wfi just woke
    // up, Kernel for nested S-mode traps) gets credited; future ticks
    // through this trap land in Kernel until we sret back or drop into
    // the manager / wfi.
    kmain::kernel::accounting::switch_bucket(
        hart_context,
        kmain::kernel::accounting::HartBucket::Kernel,
    );

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

                    // SSWI is overloaded: it carries scheduler wake-ups
                    // AND TLB-shootdown drain requests. Drain the
                    // shootdown ring before the context-switch check
                    // so any pending invalidations land before the
                    // resumed thread can reissue a stale-TLB load.
                    kmain::kernel::shootdown::drain_local();

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
            // Supervisor external interrupt from the PLIC. Drain all
            // pending sources on this hart's S-context and return to the
            // preempted thread — no context switch needed.
            9 => {
                kmain::drivers::plic::dispatch(hart_context.plic_s_context);
            }
            c => {
                error!("unhandled kint {c}");
            }
        }
    }
    else {
        match cause_num {
            // Instruction access fault.
            1 | 12 | 13 | 15 => {
                if !from_user {
                    let kptr = hart_context.kptr.load(Ordering::Relaxed) as usize;
                    let satp = riscv::register::satp::read().bits();
                    let sp = frame.regs[2];
                    let ra = frame.regs[1];
                    let cur = hart_context.current.load(Ordering::Acquire) as *const Thread;
                    if !cur.is_null() {
                        let t: &Thread = unsafe { cur.as_ref_unchecked() };
                        let pc = t.pc.load(Ordering::Acquire);
                        let state = t.state.load(Ordering::Acquire);
                        let wake_reason = t.last_wake_reason.load(Ordering::Acquire);
                        panic!(
                            "S-mode fault on cpu{}: cause={} epc={:#x} stval={:#x} \
                             ra={:#x} sp={:#x} satp={:#x} kptr={:#x} \
                             tid={} pid={} mode={:?} thread.pc={:#x} state={} wake_reason={:#x}",
                            hart_context.hart_id, cause_num, epc, tval,
                            ra, sp, satp, kptr,
                            t.tid, t.pid, t.mode, pc, state, wake_reason,
                        );
                    } else {
                        panic!(
                            "S-mode fault on cpu{}: cause={} epc={:#x} stval={:#x} \
                             ra={:#x} sp={:#x} satp={:#x} kptr={:#x} (no current thread)",
                            hart_context.hart_id, cause_num, epc, tval,
                            ra, sp, satp, kptr,
                        );
                    }
                }
                unsafe { fault_thread(FaultInfo { cause: cause_num, epc, stval: tval }); }
            }
            // supervisor ebreak
            3 => {
                match hart_context.cscratch2 {
                    1 => {
                        // kthread self-yield (knet, k_gpu). Save the
                        // trap frame so the next dispatch resumes past
                        // the ebreak, then park via
                        // `exit_thread_with_state(Suspended)`. The new
                        // ordering inside `exit_thread_with_state` is
                        // load-bearing: it nulls `hart_context.current`
                        // *before* writing `state=Suspended`, so any
                        // hart's manager that observes the new state
                        // (Acquire) is guaranteed to see this hart's
                        // current=null too. Without that ordering, a
                        // remote manager's `assign_threads` self_view
                        // path picked up a still-running kthread and
                        // caused double-dispatch (knet running on two
                        // harts simultaneously, smoltcp ring corruption).
                        hart_context.cscratch2 = 0;
                        kmain::update_thread_and_trap_frame(epc, hart_context, frame, from_user);
                        unsafe { kmain::kernel::context::exit_thread_with_state(ThreadState::Suspended); }
                    },
                    _ => ()
                }

                return_pc += 4;
            }
            8 => {
                let syscall = frame.regs[10];
                // Bracket the dispatch for per-syscall + per-thread
                // service-time accounting. Only handlers that *return*
                // (non-blocking, non-exit) reach `record_syscall`
                // below; blocking/exit paths long-jump out via
                // `exit_thread_with_state` and are excluded — that's
                // the right semantic since "service time" stops when
                // the handler hands control back to the trap path.
                let syscall_start_ticks = riscv::register::time::read64();
                match syscall {
                    // exit
                    0 => {
                        unsafe {
                            kmain::update_thread_and_trap_frame(epc, hart_context, frame, from_user);
                            kmain::kernel::context::exit_thread_with_state(ThreadState::Exited);
                        }
                    },
                    1 => {
                        //debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_serial_print(epc, hart_context, frame);
                    }
                    2 => {
                        kmain::handle_ms_sleep(epc, hart_context, frame);
                    }
                    3 => {
                        kmain::handle_console_write(epc, hart_context, frame);
                    }
                    4 => {
                        kmain::handle_read_stdin(epc, hart_context, frame);
                    }
                    5 => {
                        kmain::handle_set_affinity(epc, hart_context, frame);
                    }
                    6 => {
                        kmain::handle_get_affinity(epc, hart_context, frame);
                    }
                    7 => {
                        kmain::handle_get_hart_id(epc, hart_context, frame);
                    }
                    8 => {
                        kmain::handle_get_micros(epc, hart_context, frame);
                    }
                    4096 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_mmap_req(epc, hart_context, frame);
                    }
                    4097 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_nc_create_req(epc, hart_context, frame);
                    }
                    4098 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_close_req(epc, hart_context, frame);
                    }
                    4099 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_create_process_req(epc, hart_context, frame);
                    }
                    4100 => {
                        kmain::handle_nc_yield(epc, hart_context, frame);
                    }
                    4101 => {
                        kmain::handle_query_stats(epc, hart_context, frame);
                    }
                    4102 => {
                        kmain::handle_query_syscall_stats(epc, hart_context, frame);
                    }
                    4103 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_create_process_ex(epc, hart_context, frame);
                    }
                    4104 => {
                        kmain::handle_argv_envp(epc, hart_context, frame);
                    }
                    5000 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_create_thread(epc, hart_context, frame);
                    }
                    5001 => {
                        kmain::handle_getpid(epc, hart_context, frame);
                    }
                    5002 => {
                        kmain::handle_gettid(epc, hart_context, frame);
                    }
                    5003 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_wait_pid(epc, hart_context, frame);
                    }
                    6000 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_fs_open(epc, hart_context, frame);
                    }
                    6001 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_fs_read(epc, hart_context, frame);
                    }
                    6002 => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::handle_fs_stat(epc, hart_context, frame);
                    }
                    _ => {
                        debug!("orbit handling u mode ecall({syscall})");
                        kmain::update_thread_and_trap_frame(epc + 4, hart_context, frame, from_user);
                    }
                }
                // Close the bracket: handlers that returned hit this;
                // long-jumping ones (exit, blocking SyscallOutcome)
                // skipped past it. `current` may be null if the
                // syscall path nulled it (shouldn't happen on
                // non-yielding paths, but check defensively).
                let cur = hart_context.current.load(Ordering::Acquire);
                if !cur.is_null() {
                    let t = unsafe { (cur as *const Thread).as_ref_unchecked() };
                    kmain::kernel::accounting::record_syscall(
                        syscall, t, syscall_start_ticks,
                    );
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
        info!("hart_context @ {:016X?} hartid={} kptr={:016X?}",
            hart_context as *const _,
            hart_context.hart_id,
            hart_context.kptr.load(Ordering::Relaxed));

        hart_context.kptr.store(kmain::k_hart_loop as *mut (), Ordering::Relaxed);

        let s_trap_addr = { s_trap_vector as *const () as usize };
        riscv::register::stvec::write(Stvec::new(s_trap_addr, TrapMode::Direct));

        setup_interrupts();

        // Bucket hook 6: hart bringup. Stamp `bucket_enter_tick`
        // with a real `now()` so the first `switch_bucket` computes
        // a sane elapsed instead of crediting ~all of system uptime
        // to whichever bucket fires first. Boot prologue between
        // power-on and this point is unaccounted (deliberate).
        kmain::kernel::accounting::init_hart_bucket(
            hart_context,
            kmain::kernel::accounting::HartBucket::Kernel,
        );

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

    orbit.get_environment_info();

    // Populate each hart_context's PLIC S-mode context index from the
    // PlicInfo stashed by `drivers::plic::install`. Done here (before
    // the SECONDARY_GO publish below) so secondary harts see correct
    // values once they come online.
    if let Some(info) = kmain::drivers::plic::info() {
        unsafe {
            let my_hc = hart_context as *const HartContext as *mut HartContext;
            let base = my_hc.sub(hart_context.hart_id as usize);
            for i in 0..orbit.cpu_count {
                let hc = base.add(i).as_mut_unchecked();
                hc.plic_s_context = info
                    .s_contexts
                    .get(i)
                    .copied()
                    .flatten()
                    .unwrap_or(u32::MAX);
            }
        }

        let _ = kmain::drivers::plic::install_uart_rx_cycle();
    }

    let boot_affinity = orbit.all_harts_mask();
    orbit.create_new_process(kmain::kernel::UMODE_TEST_ELF, kmain::kernel::UPROC_STACK_DEFAULT, boot_affinity, boot_affinity, 0)
        .expect("no test uprocess");

    // Release the secondary-hart S-mode spin in `secondary_rust_setup`.
    // Release-store; secondaries Acquire and observe the publishes
    // from `rust_main` (KSATP / HART_CTX_PA / KDMAP_BIAS_BOOT).
    for hart in 1..orbit.cpu_count.min(MAX_HARTS) {
        SECONDARY_GO[hart].store(true, Ordering::Release);
    }

    (0..orbit.cpu_count).for_each(|hart| kmain::supervisor_wake_hart(hart));

    info!("kicked harts");

    k_harthello();
}

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

// =========================================================================
// Secondary-hart entry path
//
// bl srets every hart into kmain's `_start` (a0 = hartid). Hart 0 takes the
// existing trampoline path; hart != 0 dispatches to `_start_secondary`,
// which sits on a tiny per-hart bootstrap stack and waits for hart 0 to
// finish self-relocation. Once relocations are done it adopts the
// trampoline satp (identity + high-half), jumps to high-VA, and from there
// runs `secondary_rust_setup` in normal Rust — all globals are valid post-
// relocation under trampoline satp because both PA and high-VA are mapped.
// `secondary_rust_setup` then waits for the per-hart `SECONDARY_GO`
// signal, reads its `HartContext` and the kernel satp, and switches to
// the final kernel satp + `k_harthello`.
//
// This used to live in M-mode (bl::kinit_hart): bl polled `HART_ROOT`,
// translated PA → KDMAP via `KDMAP_BIAS`, and cooked sret state per hart.
// Moving it to S-mode shrinks the M-mode surface to "launch + trap".
// =========================================================================

const MAX_HARTS: usize = 4;
// 16 KiB per hart — covers early_paging_setup, post_trampoline_entry, and
// most of rust_main before hart 0 switches to its real kernel stack. Used
// by hart 0 *and* secondaries as their pre-kernel-context stack, so that
// no hart's S-mode sp aliases bl's `KERNEL_STACK_END` (which m_trap_vector
// loads into sp on every M-mode trap — sharing that with hart 0's S-mode
// path corrupts saved-ra slots and produces wild jumps into m_trap_vector
// code that fault as "csrw mepc illegal" in S-mode).
const BOOT_STACK_SHIFT: u32 = 14;
const BOOT_STACK_SIZE: usize = 1 << BOOT_STACK_SHIFT;

#[repr(C, align(16))]
struct BootStacks([u8; BOOT_STACK_SIZE * MAX_HARTS]);

#[unsafe(no_mangle)]
#[used]
static mut BOOT_STACKS: BootStacks =
    BootStacks([0; BOOT_STACK_SIZE * MAX_HARTS]);

// Set by hart 0 in `post_trampoline_entry` *after* `apply_relocations`.
// While zero, secondary harts spin in `_start_secondary` under bare satp —
// only PC-relative accesses are safe, since post-relocation Rust globals
// resolve to high-VA which bare satp does not map.
#[unsafe(no_mangle)]
#[used]
static RELOCS_DONE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

// Trampoline satp value (snapshot of `satp` while `post_trampoline_entry`
// runs). Has both identity and high-half mappings, so secondaries can
// adopt it from bare satp and then jump to high-VA in one step.
#[unsafe(no_mangle)]
#[used]
static TRAMP_SATP: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

// High-half VA of `secondary_rust_setup`. Captured via `lla` in
// `post_trampoline_entry` (which already runs at high-VA), so secondaries
// can `jr` to it under the trampoline satp.
#[unsafe(no_mangle)]
#[used]
static SECONDARY_RUST_SETUP_VA: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

// Final kernel satp (KTEXT + KDMAP + KMMIO, no identity). Published at
// the end of `rust_main` before hart 0 srets to `k_smpstart`.
#[unsafe(no_mangle)]
#[used]
static KSATP: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

// Physical address of the per-hart `HartContext` array. Secondaries
// index by hartid * size_of::<HartContext>().
#[unsafe(no_mangle)]
#[used]
static HART_CTX_PA: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

// kdmap_base() - ram_phys_base(). Add to a RAM PA to get its KDMAP VA.
// Published once `init_layout` has run.
#[unsafe(no_mangle)]
#[used]
static KDMAP_BIAS_BOOT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

// Per-hart "go" flag set by `k_smpstart` after the kernel state is ready.
// Secondaries WFI-spin on this in `secondary_rust_setup` (under trampoline
// satp, so WFI/Rust accesses are all safe).
#[unsafe(no_mangle)]
#[used]
static SECONDARY_GO: [core::sync::atomic::AtomicBool; MAX_HARTS] =
    [const { core::sync::atomic::AtomicBool::new(false) }; MAX_HARTS];

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
        // bl enters with a0=hartid, a1=dtb, a2=serial. Preserve them across the
        // call to early_paging_setup via callee-saved s-registers.
        //
        // auipc MUST be the first instruction of _start: its result is
        // used as `load_addr` everywhere downstream (raw = image_end_pa -
        // load_addr; pa_range.start = load_addr). The 4 KiB-page path in
        // map_va_range rejects a misaligned paddr, so even a 2-byte
        // compressed instruction in front of auipc shifts load_addr off
        // alignment and the bootstrap mapping fails. Dispatch on hartid
        // *after* this snapshot — secondaries don't read s0/s1/s2.
        "auipc t0, 0",              // t0 = physical VA of _start (load_addr)
        "mv s0, t0",                // s0 = load_addr (callee-saved)
        "mv s1, a0",                // s1 = hartid
        "mv s2, a1",                // s2 = dtb
        "mv s3, a2",                // s3 = serial

        // sp from per-hart boot stack. bl's `mv sp, KERNEL_STACK_END`
        // before sret hands every hart the SAME pointer (0x80080000),
        // which m_trap_vector also loads on every M-mode trap. With
        // shared sp between hart 0's S-mode path and concurrent M-mode
        // trap handlers on harts 1..3, hart 0's saved-ra gets clobbered
        // with the m_trap_vector return address and the next ret jumps
        // there in S-mode, faulting on `csrw mepc, a0`. Per-hart slot
        // here decouples them.
        "lla   t1, {boot_stacks}",
        "addi  t2, a0, 1",
        "slli  t2, t2, {boot_stack_shift}",
        "add   sp, t1, t2",

        // bl enters every hart here. Secondaries take a separate S-mode
        // path that waits for hart 0's bringup — bl is out of the
        // per-hart cooking business entirely.
        "bnez a0, 9f",

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

        // Args for post_trampoline_entry(hartid, dtb, serial, ktext_base, load_addr).
        "mv a0, s1",
        "mv a1, s2",
        "mv a2, s3",
        "mv a3, t3",                // ktext_base
        "mv a4, s0",                // load_addr

        "jr t4",

        // Secondary fall-through: tail-call into `_start_secondary`.
        // Materialized as `lla + jr` so the assembler doesn't have to
        // reach `_start_secondary` within ±1 MiB.
    "9:",
        "lla t0, {start_secondary}",
        "jr  t0",

        early_pt = sym EARLY_PT,
        early_pt_size = const EARLY_PT_SIZE,
        early_paging_setup = sym early_paging_setup,
        post_tramp = sym post_trampoline_entry,
        ktext_hi32 = const KTEXT_NOMINAL_HI32,
        start_secondary = sym _start_secondary,
        boot_stacks = sym BOOT_STACKS,
        boot_stack_shift = const BOOT_STACK_SHIFT,
    );
}

/// Secondary-hart entry. Hart != 0 lands here from `_start`'s dispatch
/// with a0 = hartid, bare satp. Spins (under bare satp, PC-relative only)
/// until hart 0 publishes `RELOCS_DONE` + `TRAMP_SATP` +
/// `SECONDARY_RUST_SETUP_VA`, then adopts the trampoline satp and jumps
/// to high-VA Rust.
#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn _start_secondary() -> ! {
    naked_asm!(
        // sp is already set by `_start` (per-hart slot in BOOT_STACKS)
        // before we get here.

        // Busy-spin on RELOCS_DONE. We can't WFI here without an stvec,
        // and the wait is short (only until hart 0 finishes
        // `apply_relocations` in post_trampoline_entry). The fence after
        // the load makes the subsequent reads observe the publishes that
        // happened-before the Release store on hart 0.
        "lla   t0, {relocs_done}",
    "1:",
        "lw    t1, 0(t0)",
        "beqz  t1, 1b",
        "fence r, r",

        // Read trampoline satp + high-VA target. PC-rel under bare satp
        // gives PAs of the statics; the *values* loaded are just
        // integers (not pointers needing relocation), so they're correct.
        "lla   t0, {tramp_satp}",
        "ld    t1, 0(t0)",
        "lla   t0, {sec_setup_va}",
        "ld    t2, 0(t0)",

        // Switch to trampoline satp. PC stays valid because trampoline
        // identity-maps RAM. After this, both PA and high-VA resolve, so
        // jumping to the high-VA secondary_rust_setup is safe.
        "sfence.vma",
        "csrw  satp, t1",
        "sfence.vma",

        // Tail-call to high-VA secondary_rust_setup(hartid). a0 is
        // already hartid; convention preserved.
        "jr    t2",

        relocs_done = sym RELOCS_DONE,
        tramp_satp = sym TRAMP_SATP,
        sec_setup_va = sym SECONDARY_RUST_SETUP_VA,
    );
}

/// Runs at high-VA under the trampoline satp. Both PA (identity) and
/// high-VA are mapped here, so post-relocation Rust globals resolve
/// correctly. Waits for the per-hart `SECONDARY_GO`, then reads the
/// `HartContext`, switches to the final kernel satp, and jumps to
/// `k_harthello`.
#[unsafe(no_mangle)]
unsafe extern "C" fn secondary_rust_setup(hartid: usize) -> ! {
    // bl::setup_interrupts left mie.MSIE set on this hart, and bl's
    // m_trap_vector now save/restores mie around `call m_trap`, so
    // MSIPs from `kick_machine_harts` correctly wake WFI.
    while !SECONDARY_GO[hartid].load(Ordering::Acquire) {
        riscv::asm::wfi();
    }

    let ksatp = KSATP.load(Ordering::Acquire);
    let hart_ctx_base_pa = HART_CTX_PA.load(Ordering::Acquire) as usize;
    let kdmap_bias = KDMAP_BIAS_BOOT.load(Ordering::Acquire) as usize;

    let stride = core::mem::size_of::<HartContext>();
    let hart_ctx_pa = hart_ctx_base_pa + hartid * stride;
    let hart_ctx_kva = hart_ctx_pa.wrapping_add(kdmap_bias);

    // Trampoline satp identity-maps RAM, so reading the HartContext
    // through its PA is safe. Equivalently could use the KDMAP VA — same
    // backing memory either way.
    let hart_ctx = unsafe { &*(hart_ctx_pa as *const HartContext) };
    let kptr = hart_ctx.kptr.load(Ordering::Acquire) as usize;
    let stvec_addr = hart_ctx.s_trap_addr as usize;
    let sp_pa = hart_ctx.k_stack.stack_data.as_ptr() as usize
        + device::TRAP_STACK_SIZE - 16;
    let sp_kva = sp_pa.wrapping_add(kdmap_bias);

    // No stack access between csrw satp and `mv sp` — inline asm doesn't
    // spill within a block, so sp can stay pointing at the bootstrap
    // stack PA (no longer mapped under the final satp) until we move it
    // to the KDMAP VA of the kernel stack.
    unsafe {
        asm!(
            "csrw sscratch, {sscratch}",
            "csrw stvec, {stvec}",
            "sfence.vma",
            "csrw satp, {ksatp}",
            "sfence.vma",
            "mv   sp, {sp}",
            "fence.i",
            "jr   {kptr}",
            sscratch = in(reg) hart_ctx_kva,
            stvec = in(reg) stvec_addr,
            ksatp = in(reg) ksatp,
            sp = in(reg) sp_kva,
            kptr = in(reg) kptr,
            options(noreturn),
        );
    }
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
    use mmu::{MappingConfig, PAGE_SIZE, PagePermissions, SupervisorTag};
    use mmu::mmap::{PageAlloc, PageTableVec, RootTable, id_map_range, map_va_range};
    use mmu::sv48::{PhysAddr, VirtAddr};

    let mut ptv = PageTableVec::new(pt_base as usize, pt_size);
    let Ok(root_pa) = (unsafe { ptv.allocate_page_table() }) else {
        loop { riscv::asm::wfi(); }
    };
    // Early trampoline tables live in identity-mapped RAM (bias = 0), so
    // PA == VA. Zero the freshly-allocated root before exposing it as a
    // page table — PageAlloc no longer zeros.
    let root_ref = unsafe {
        core::ptr::write_bytes(root_pa as *mut u8, 0, PAGE_SIZE);
        (root_pa as *const PageTable).as_ref_unchecked()
    };
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
        supervisor_tag: SupervisorTag::None,
    };

    // Identity [0, 1 GiB) — low-half MMIO range
    if unsafe { id_map_range(&root, &mut pages, cfg, 0..(1u64 << 30)) }.is_err() {
        loop { riscv::asm::wfi(); }
    }
    // Identity [2, 4 GiB) — all of RAM (kernel image, kheap, kpages, ktables, dtb)
    if unsafe { id_map_range(&root, &mut pages, cfg, (2u64 << 30)..(4u64 << 30)) }.is_err() {
        loop { riscv::asm::wfi(); }
    }

    // High-half kernel image: cover [load_addr, _DYNAMIC_END) at
    // KTEXT_NOMINAL -> load_addr, rounded up to the next 2 MiB so
    // `map_va_range` emits megapages. The bootstrap window has to span
    // the whole loaded image (.text + .rodata + .data + .bss + .dynamic)
    // because rust_main reads relocated statics and applies relocations
    // before map_kernel_self installs the permission-correct final
    // mappings. Permissions stay RWX|G to match the identity range
    // above; W^X comes in with the post-bootstrap satp.
    // Sizing is dynamic so the mapping grows automatically as the
    // embedded user-mode loader (and the console it embeds) get bigger.
    let ktext = kmain::kernel::memmap::KTEXT_NOMINAL;
    let image_end_pa: u64;
    unsafe {
        core::arch::asm!(
            "lla {out}, _DYNAMIC_END",
            out = out(reg) image_end_pa,
            options(nomem, nostack, preserves_flags),
        );
    }
    const MEGAPAGE: u64 = 2 * 1024 * 1024;
    let raw = image_end_pa.wrapping_sub(load_addr);
    let len = (raw + (MEGAPAGE - 1)) & !(MEGAPAGE - 1);
    if unsafe { map_va_range(&root, &mut pages, cfg, ktext, load_addr..(load_addr + len)) }.is_err() {
        loop { riscv::asm::wfi(); }
    }

    // KDMAP: 2 GiB of RAM at KDMAP_NOMINAL → [2 GiB, 4 GiB). Both ends are
    // 1 GiB-aligned so map_va_range emits two gigapages. Needed here (not just
    // in the final satp) so rust_main can initialize KHEAP/kpages through
    // their KDMAP VAs before the final satp is installed.
    let kdmap = kmain::kernel::memmap::KDMAP_NOMINAL;
    if unsafe { map_va_range(&root, &mut pages, cfg, kdmap, (2u64 << 30)..(4u64 << 30)) }.is_err() {
        loop { riscv::asm::wfi(); }
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
    dtb: usize,
    serial: usize,
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

        // Publish the trampoline satp + the high-VA of secondary_rust_setup
        // so the secondaries' asm prologue can adopt them. PC is already
        // high-VA here (we jumped via the trampoline at the end of `_start`),
        // so `lla` gives the high-VA of the symbol. `csrr satp` gives the
        // trampoline satp bits.
        let tramp_satp: u64;
        core::arch::asm!(
            "csrr {0}, satp",
            out(reg) tramp_satp,
            options(nomem, nostack, preserves_flags),
        );
        let sec_setup_va: u64;
        core::arch::asm!(
            "lla {0}, {sym}",
            out(reg) sec_setup_va,
            sym = sym secondary_rust_setup,
            options(nomem, nostack, preserves_flags),
        );
        TRAMP_SATP.store(tramp_satp, Ordering::Release);
        SECONDARY_RUST_SETUP_VA.store(sec_setup_va, Ordering::Release);
        // Release-store last; secondaries Acquire on this and then read
        // the two values above.
        RELOCS_DONE.store(1, Ordering::Release);

        rust_main(hartid, dtb, serial, load_addr);
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
extern "C" fn rust_main(_hartid: usize, dtb: usize, serial: usize, load_addr: u64) -> ! {
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

        let dtb_addr = dtb;
        let serial_addr = serial;

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
            kmain::kernel::memmap::KSCRATCH_NOMINAL,
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
                kmain::kernel::memmap::phys_to_kdmap(mmu::sv48::PhysAddr::new(layout.kheap.start))
                    .as_mut_ptr::<u8>(),
                kmain::kernel::memmap::KHEAP_SIZE as usize,
            );

        static LOGGER: kmain::ktrace::OrbitLogger = kmain::ktrace::OrbitLogger;

        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Info);
        tracing::subscriber::set_global_default(OrbitSubscriber::new(Level::INFO))
            .expect("no tracing");

        let mut kernel_tables = kmain::kernel::memmap::TablePages::new();
        kernel_tables.add_pa_range(layout.ktables.clone());

        // Allocator hands back `(PhysAddr, KdmapVa)`; we deref through the
        // KDMAP alias. `RootTable` carries the PA→KDMAP bias so walkers
        // convert child-PPNs (always physical) back to supervisor-visible
        // VAs.
        let (orbit_root_pa, orbit_root_kva) = kernel_tables
            .alloc(Layout::from_size_align_unchecked(PAGE_SIZE, PAGE_SIZE))
            .expect("failed to alloc orbit root table");
        // Fresh frame — zero before exposing as a page table.
        core::ptr::write_bytes(orbit_root_kva.as_mut_ptr::<u8>(), 0, PAGE_SIZE);
        let orbit_root_ref = orbit_root_kva.as_ptr::<PageTable>().as_ref_unchecked();
        let orbit_root_table = kmain::kernel::memmap::kernel_root(orbit_root_ref);

        info!("ort=0x{:016X?}", orbit_root_ref as *const _ as usize);

        {
            let mut pages = PageAlloc::FA(kernel_tables.frames_mut());
            map_kernel_self(&orbit_root_table, &mut pages, &layout)
                .expect("failed to map kernel self-view");
        }

        let mut kpages = kmain::kernel::memmap::KernelPages::new();
        kpages.add_pa_range(layout.kpages.clone());

        // User-private pool. No KDMAP alias in the kernel satp — setup-time
        // writes go through `UserPageWindow` at a KSCRATCH-reserved VA.
        let mut user_pages = kmain::kernel::memmap::UserPages::new();
        user_pages.add_pa_range(layout.user_pages.clone());

        let cpu_count = 4;
        let context_size = cpu_count * core::mem::size_of::<HartContext>();
        let (hart_contexts_pa, hart_contexts_kva) = kpages
            .alloc_kdmap(Layout::from_size_align_unchecked(context_size, 4096))
            .expect("failed to alloc hart contexts");
        let hart_contexts = hart_contexts_kva.as_mut_ptr::<HartContext>();

        let mut satp = Satp::from_bits(0);
        satp.set_asid(0);
        satp.set_mode(Mode::Sv48);
        // satp takes the physical PPN — `TablePages::alloc` already
        // handed us `orbit_root_pa` directly, no translation needed.
        satp.set_ppn((orbit_root_pa.get_raw() / PAGE_SIZE as u64) as usize);

        let orbit = {
            let (_orbit_pa, orbit_kva) = kpages
                .alloc_kdmap(Layout::from_size_align_unchecked(
                    round_u64_up(core::mem::size_of::<Orbit>() as u64, 4096) as usize,
                    4096))
                .expect("failed to alloc space for kernel state");
            let orbit_ptr = orbit_kva.as_mut_ptr::<Orbit>().as_mut_unchecked();

            *orbit_ptr = Orbit::new(dtb_addr as usize, serial_addr as usize, cpu_count, layout, kernel_tables, kpages, user_pages, satp.clone());

            orbit_ptr
        };

        // Per-hart SPSC rings for deferred frees. Sized to `cpu_count`,
        // installed before any SharedUserPtr can exist (which means
        // before the first umode syscall, which is long after this
        // point).
        kmain::kernel::pending_frees::init(cpu_count);

        // Same window: shootdown ring fan-out target count. Must run
        // before any user PTE modification can fire `broadcast`. The
        // statics themselves are already pre-initialized.
        kmain::kernel::shootdown::init(cpu_count);

        // Wire process::completion's wake hook to kmain's
        // wake_blocked_inline. Once installed, signal_n on a
        // CompletionHandle whose parker has set_waiter'd will
        // immediately marshal the rets, mark the thread Ready,
        // and queue it on the signaling hart's READY_INBOX —
        // no manager scan required.
        kmain::kernel::install_completion_wake_hook();

        info!("allocated orbit state @ {:016X?}", &raw const *orbit as usize);

        let hart_root = hart_contexts as usize;
        riscv::register::sscratch::write(hart_root);

        info!("allocated hart contexts @ {hart_root:016X?}");

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
            // Sentinel until plic install populates the real S-context in
            // k_smpstart. Any cause-9 that lands here before then will be
            // rejected by `plic::dispatch`.
            hart_context.plic_s_context = u32::MAX;

            // Bucket accounting starts in the Kernel bucket. The
            // bucket_enter_tick is set to zero here and re-stamped to
            // `now()` in `k_harthello` just before control reaches
            // user-mode/idle for the first time, so the boot prologue
            // doesn't get charged to anything.
            hart_context.current_bucket.store(
                kmain::kernel::accounting::HartBucket::Kernel as u8,
                Ordering::Relaxed,
            );
            hart_context.bucket_enter_tick.store(0, Ordering::Relaxed);
            hart_context.user_ticks.store(0, Ordering::Relaxed);
            hart_context.kernel_ticks.store(0, Ordering::Relaxed);
            hart_context.scheduler_ticks.store(0, Ordering::Relaxed);
            hart_context.idle_ticks.store(0, Ordering::Relaxed);

            info!("setting hart context @ {ptr:016X?} to kidle hart{hart}");
        }

        let this_sp = hart_contexts.as_ref_unchecked().k_stack.stack_data.as_ptr() as usize + TRAP_STACK_SIZE - 16;
        let this_pc = k_smpstart as *const ();

        riscv::register::sepc::write(this_pc as usize);
        riscv::register::sstatus::set_spp(riscv::register::sstatus::SPP::Supervisor);

        // Publish boot state for secondary harts. They're spinning under
        // the trampoline satp on `SECONDARY_GO[hartid]`, which
        // `k_smpstart` flips after the kernel-side bringup has finished.
        // Releases here happen-before that flip (same hart), so any
        // Acquirer that observes GO=true also sees these values.
        let kdmap_bias = kmain::kernel::memmap::kdmap_base()
            .wrapping_sub(kmain::kernel::memmap::ram_phys_base());
        KDMAP_BIAS_BOOT.store(kdmap_bias, Ordering::Release);
        HART_CTX_PA.store(hart_contexts_pa.get_raw(), Ordering::Release);
        KSATP.store(satp.bits() as u64, Ordering::Release);

        unmap_boot_only_regions(&orbit_root_table)
            .expect("failed to unmap boot-only regions");

        info!("jump sp={this_sp:016X} pc={this_pc:016X?}");

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
