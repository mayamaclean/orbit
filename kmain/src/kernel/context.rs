use core::ptr::null_mut;
use core::{arch::asm};
use core::sync::atomic::Ordering;

use device::{HartContext, TRAP_STACK_SIZE};
use process::{FaultInfo, Thread, ThreadState};
use riscv::register::sstatus::SPP;
use tracing::{error};

use crate::kernel::user_trap_frame_vaddr;

/// Lower bound for any kernel-text VA. The kernel ELF is linked at low
/// VA `0x1000` and post-trampoline relocates to the high half (KTEXT
/// nominal); anything below `USER_VA_END` (and above the null guard)
/// belongs to user space. Used purely as a sanity check in the
/// dispatch path — `thread.pc` for a kernel thread should never land
/// in user range.
const USER_VA_END: usize = 0x0080_0000_0000;

unsafe fn jump(context: &'static HartContext, target: usize) -> ! {
    unsafe {
        let this_sp = context.k_stack.stack_data.as_ptr() as usize + TRAP_STACK_SIZE - 16;

        riscv::register::sepc::write(target);
        riscv::register::sstatus::set_spp(riscv::register::sstatus::SPP::Supervisor);

        if context.satp.bits() != riscv::register::satp::read().bits() {
            riscv::asm::sfence_vma(context.satp.asid(), 0);
            riscv::register::satp::write(context.satp);
            riscv::asm::sfence_vma(context.satp.asid(), 0);
        }

        asm!(
            "mv sp, {s}",         // 1. Switch to the new hart-specific stack
            "sret",               // 4. Jump to sepc (Kernel Idle)
            s = in(reg) this_sp,
            options(noreturn)
        );
    }
}


unsafe extern "C" {
    unsafe fn enter_hart_context_asm(user_frame_vaddr: usize, satp: usize, asid: usize) -> !;
    unsafe fn enter_hart_kcontext_asm(trap_frame: *const ()) -> !;
}

pub unsafe fn load_thread_into_hart_context_and_jump(context: &'static HartContext, thread: &'static Thread) -> ! {
    unsafe {
        // No S-mode interrupts across the register-load → sret window. Without
        // this, a stimer/SSWI landing inside enter_hart_context_asm would
        // trap with EPC in kernel text, and update_thread_and_trap_frame
        // would (before the mode gate) save that as thread.pc. sret restores
        // SIE from SPIE; U-mode preemption is unaffected because S-interrupts
        // fire in U regardless of sstatus.SIE.
        riscv::register::sstatus::clear_sie();

        riscv::register::sstatus::set_spp(thread.mode);
        let pc = thread.pc.load(Ordering::Acquire);

        // Sanity check: a kernel thread's pc must never be a user VA.
        // Kernel text lives in the high half (KTEXT_NOMINAL =
        // 0xFFFF_FFC0_0000_0000); user text caps at USER_VA_END
        // (0x0080_0000_0000). If we let a Supervisor-mode dispatch
        // with a user-range pc through, sret would trap on the first
        // instruction fetch (cause=12) and the panic_handler reports
        // "S-mode fault" — far from the actual corruption point.
        // Catch it here with the thread metadata instead.
        if thread.mode == SPP::Supervisor && pc < USER_VA_END {
            error!(
                "dispatch: kernel thread tid={} pid={} pc={:#x} state={} \
                 last_wake_reason={:#x} — pc looks like a user VA, refusing to sret",
                thread.tid,
                thread.pid,
                pc,
                thread.state.load(Ordering::Acquire),
                thread.last_wake_reason.load(Ordering::Acquire),
            );
            panic!(
                "kernel thread tid={} dispatched with pc={:#x} (user range)",
                thread.tid, pc,
            );
        }

        // Trace every dispatch so we can correlate the (tid, pc) pair
        // immediately before any cause=12 fault. Verbose but the
        // signal-to-noise is fine while debugging — a few hundred
        // lines per second under steady-state, comparable to existing
        // e1000 status logs.
        /*
        info!(
            "dispatch: tid={} pid={} mode={:?} pc={:#x}",
            thread.tid, thread.pid, thread.mode, pc,
        );
        */

        riscv::register::sepc::write(pc);

        //serial::println!("cpu{} marking thread{} as running", context.hart_id, thread.tid);
        thread.state.store(ThreadState::Running as usize, Ordering::Release);
        // Per-thread context-switch tally: every Ready→Running dispatch
        // (user or kernel thread) is one switch from this thread's
        // perspective. Foreign-hart reads via `query_stats` go through
        // the same atomic.
        thread.context_switches.fetch_add(1, Ordering::Relaxed);

        if thread.mode == SPP::User {
            // Bucket hook 2: sret to user. Switch right before the
            // asm so trap-prologue ticks remain in Kernel and the
            // first cycle in user code starts the User bucket.
            // Kernel-thread (Supervisor) sret stays in Kernel — these
            // threads are kernel code that just happens to run in a
            // dedicated context.
            crate::kernel::accounting::switch_bucket(
                context,
                crate::kernel::accounting::HartBucket::User,
            );

            // The kernel-vaddr frame ptr is unreachable after the satp switch,
            // so we hand the asm the user-side mapping at slot's vaddr instead.
            let slot = thread.slot
                .expect("user thread missing slot");
            let user_frame_vaddr = user_trap_frame_vaddr(slot) as usize;
            enter_hart_context_asm(user_frame_vaddr, thread.satp.bits(), thread.satp.asid());
        }
        else {
            let frame_ptr = thread.frame as *const _ as *const ();
            enter_hart_kcontext_asm(frame_ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe fn enter_hart_context(context: &'static HartContext) -> ! {
    let thread_addr = context.current.load(Ordering::Acquire);
    if thread_addr != core::ptr::null_mut() {
        unsafe {
            load_thread_into_hart_context_and_jump(context, (thread_addr as *const Thread).as_ref_unchecked());
        }
    }

    let kidle = context.kptr.load(Ordering::Acquire);
    unsafe { jump(context, kidle as usize); }
}

/// Stash a `FaultInfo` on the current thread and exit it. Called from the trap
/// handler when a user thread can't continue (page fault, bad ecall, etc.).
/// Manager-side cleanup reads `fault_info` to classify and log.
pub unsafe fn fault_thread(info: FaultInfo) -> ! {
    unsafe {
        let context = (riscv::register::sscratch::read()
            as *const HartContext).as_ref_unchecked();
        let cur = context.current.load(Ordering::Acquire);
        if !cur.is_null() {
            (cur as *mut Thread).as_mut_unchecked().fault_info = Some(info);
        }
        exit_thread_with_state(ThreadState::Exited)
    }
}

/// caller must actually be in a thread context!
///
/// **Order matters**: `current` is nulled *before* `state` is stored.
///
/// The manager (running on another hart) can observe the new state via
/// an Acquire load. Release/Acquire on the same atomic establishes a
/// happens-before for *all prior writes* by the storing hart — so a
/// reader that sees `state == Suspended` is guaranteed to also see
/// `current == null` on this hart.
///
/// If we did it the other way around (state first, then cur=null), the
/// manager could observe `state=Suspended` while `current` still
/// pointed at the thread on this hart. Its `assign_threads` self_view
/// path (no `is_busy()` gate) would then assign the thread to itself
/// while it's still claimed here — a double-dispatch race that
/// manifested as knet running on two harts at once and corrupting
/// smoltcp + e1000 ring state. See bl/arm_hart_timer_*.log for the
/// repro.
pub unsafe fn exit_thread_with_state(state: ThreadState) -> ! {
    unsafe {
        let context = (riscv::register::sscratch::read()
            as *const HartContext).as_ref_unchecked();

        let thread_addr = context.current.load(Ordering::Acquire);
        if thread_addr != null_mut() {
            let thread = (thread_addr as *const Thread).as_ref_unchecked();
            // Null `current` first so the new `state` Release-store
            // carries the cur=null write with it. See doc comment.
            context.current.store(null_mut(), Ordering::Release);
            thread.state.store(state as usize, Ordering::Release);
        }

        let target = context.kptr.load(Ordering::Acquire);
        jump(context, target as usize)
    }
}

pub fn hart_has_thread(context: &'static HartContext) -> bool {
    context.current.load(Ordering::Acquire) != null_mut()
}

pub fn get_hart_context() -> &'static HartContext {
    unsafe {
        let addr = riscv::register::sscratch::read();
        (addr as *const HartContext).as_ref_unchecked()
    }
}

#[unsafe(no_mangle)]
pub fn enter_kernel_busywork() -> ! {
    let context = get_hart_context();
    let target = context.kptr.load(Ordering::Acquire);
    unsafe { jump(context, target as usize) }
}