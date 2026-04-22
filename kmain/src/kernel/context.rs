use core::ptr::null_mut;
use core::{arch::asm};
use core::sync::atomic::Ordering;

use device::{HartContext, TRAP_STACK_SIZE};
use process::{FaultInfo, Thread, ThreadState};
use riscv::register::sstatus::SPP;

use crate::kernel::user_trap_frame_vaddr;

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

pub unsafe fn load_thread_into_hart_context_and_jump(_context: &'static HartContext, thread: &'static Thread) -> ! {
    unsafe {
        // No S-mode interrupts across the register-load → sret window. Without
        // this, a stimer/SSWI landing inside enter_hart_context_asm would
        // trap with EPC in kernel text, and update_thread_and_trap_frame
        // would (before the mode gate) save that as thread.pc. sret restores
        // SIE from SPIE; U-mode preemption is unaffected because S-interrupts
        // fire in U regardless of sstatus.SIE.
        riscv::register::sstatus::clear_sie();

        riscv::register::sstatus::set_spp(thread.mode);
        riscv::register::sepc::write(thread.pc.load(Ordering::Acquire));

        //serial::println!("cpu{} marking thread{} as running", context.hart_id, thread.tid);
        thread.state.store(ThreadState::Running as usize, Ordering::Release);

        if thread.mode == SPP::User {
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
pub unsafe fn exit_thread_with_state(state: ThreadState) -> ! {
    unsafe {
        let context = (riscv::register::sscratch::read()
            as *const HartContext).as_ref_unchecked();

        let thread_addr = context.current.load(Ordering::Acquire);
        if thread_addr != null_mut() {
            let thread = (thread_addr as *const Thread).as_ref_unchecked();
            thread.state.store(state as usize, Ordering::Release);
            context.current.store(null_mut(), Ordering::Release);
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