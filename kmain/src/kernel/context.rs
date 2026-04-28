use core::ptr::null_mut;
use core::{arch::asm};
use core::sync::atomic::Ordering;

use device::{HartContext, TRAP_STACK_SIZE, TrapFrame};
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

    /// Setjmp-style save for a kernel thread that wants to park
    /// without going through the trap vector. See `trap.S` for the
    /// register save list and resume semantics. Returns 0 on the
    /// first call; on later redispatch, sret restores GPRs from the
    /// frame and the synthetic return value is 1.
    unsafe fn kthread_save_resume_point(frame: *mut TrapFrame, pc_cell: *mut usize) -> usize;

    /// Noreturn hand-off used by `kthread_park` after a successful
    /// `kthread_save_resume_point`. Switches sp onto this hart's
    /// idle stack, nulls the current-thread slot at
    /// `current_slot_ptr` (Release), publishes `state` into
    /// `*state_ptr` (Release), and srets to `kptr` (k_hart_loop).
    ///
    /// `current_slot_ptr` is the *address* of `HartContext.current`'s
    /// storage cell — typed as `*mut *mut ()` to match
    /// `AtomicPtr<()>::as_ptr`'s return and stay FFI-safe (the asm
    /// only zero-stores through it). See `trap.S` for the ordering
    /// rationale.
    unsafe fn kthread_handoff_to_kidle(
        state_ptr: *mut usize,
        state_val: usize,
        current_slot_ptr: *mut *mut (),
        kstack_top: usize,
        kptr: usize,
    ) -> !;
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
        trace!(
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

/// Park the currently-dispatched kernel thread on this hart and hand
/// control back to `k_hart_loop`. Resumes at the call site as if this
/// were an ordinary function return — locals on the kthread's own
/// stack survive the round trip.
///
/// Replaces the per-call ad-hoc sequence (set wake_time + ticks +
/// state, fire `ebreak`, rely on the cause=3 + `cscratch2` sidechannel
/// in `s_trap` to switch) that lived inline in `k_net` and `k_gpu`.
/// Compared to that path:
///   * No trap-vector round trip — direct asm save + sret to k_hart_loop.
///   * No `cscratch2` overload of the ebreak cause; ordinary debugger
///     ebreaks regain their unambiguous meaning if they ever appear.
///   * State publish happens *after* sp has switched off the kthread's
///     stack, closing the "remote scheduler dispatches us while this
///     hart is still using the same stack" race.
///
/// Caller invariants:
///   * Must be running as a kernel thread (`thread.mode == Supervisor`).
///   * Must hold no spinlocks — handing off to k_hart_loop deadlocks
///     anything else trying to acquire them.
pub fn kthread_park(state: ThreadState, wake_time: usize) {
    // Async traps off across the save → publish window. A timer/SSWI
    // landing between the save and the handoff would route through
    // s_trap → check_context_and_switch and park us via the legacy
    // path, but with thread.pc still pointing at the original
    // (pre-save) sepc — resume would jump back into the middle of
    // this function instead of the saved resume label, observe a
    // partially-published state, and corrupt the loop. Same
    // rationale as load_thread_into_hart_context_and_jump's clear.
    unsafe { riscv::register::sstatus::clear_sie(); }

    let context = get_hart_context();
    let cur = context.current.load(Ordering::Acquire);
    if cur.is_null() {
        // No current thread — nothing to park. Bail to k_hart_loop
        // directly; matches exit_thread_with_state's null-current path.
        unsafe {
            let target = context.kptr.load(Ordering::Acquire);
            jump(context, target as usize);
        }
    }
    let thread = unsafe { (cur as *mut Thread).as_mut_unchecked() };

    if thread.mode != SPP::Supervisor {
        panic!(
            "kthread_park: tid={} is not a kernel thread (mode={:?}); \
             user threads must yield via the syscall path",
            thread.tid, thread.mode,
        );
    }

    thread.ticks = 0;
    thread.wake_time = wake_time;

    // Setjmp-style save. Returns 0 on the first call (we proceed to
    // hand off); returns 1 when the scheduler later redispatches
    // this thread (sret has restored ra/sp/s-regs, just resume the
    // caller). The asm snapshots the live values of s-regs at the
    // call point, so locals derived after this point but stored in
    // s-regs would be lost across the round trip — we don't keep
    // any: the only post-save Rust work is reading immutable
    // context fields and calling the noreturn handoff asm.
    let frame_ptr = thread.frame as *mut TrapFrame;
    let pc_ptr = thread.pc.as_ptr();
    if unsafe { kthread_save_resume_point(frame_ptr, pc_ptr) } != 0 {
        return;
    }

    // For a Suspended park, register the new park instance with the
    // sleep heap before the noreturn handoff. fetch_add(Release) is
    // ordered with the state.store(Release) the handoff asm performs
    // after the sp switch — manager observing state=Suspended is
    // guaranteed to see the matching seq. The push happens before
    // the handoff because we lose control after; the manager won't
    // observe the inbox entry as live until state actually hits
    // Suspended (which is only published after the sp switch
    // inside `kthread_handoff_to_kidle`).
    if state == ThreadState::Suspended {
        let seq = thread.sleep_seq.fetch_add(1, Ordering::Release)
            .wrapping_add(1);
        let notice = crate::kernel::SleepNotice {
            wake_time: wake_time as u64,
            sleep_seq: seq,
            thread: cur as *mut Thread,
        };
        if crate::kernel::SLEEP_INBOX.push(notice).is_err() {
            error!(
                "SLEEP_INBOX full on kthread_park: tid={} wake_time={}",
                thread.tid, wake_time,
            );
        }
    }

    // First-time path. Frame is self-consistent; hand off to
    // k_hart_loop. The handoff switches sp onto k_stack first,
    // *then* publishes current=null (Release) and state (Release)
    // — see trap.S for the ordering rationale.
    unsafe {
        let kstack_top = context.k_stack.stack_data.as_ptr() as usize
            + TRAP_STACK_SIZE
            - 16;
        let kptr = context.kptr.load(Ordering::Acquire) as usize;
        kthread_handoff_to_kidle(
            thread.state.as_ptr(),
            state as usize,
            context.current.as_ptr(),
            kstack_top,
            kptr,
        );
    }
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
            // For a Suspended park, bump sleep_seq *before* publishing
            // state so a manager observing state=Suspended via the
            // Release store below also observes the new seq. Captured
            // here for the SLEEP_INBOX push after the state release.
            let mut sleep_notice: Option<crate::kernel::SleepNotice> = None;
            if state == ThreadState::Suspended {
                let seq = thread.sleep_seq.fetch_add(1, Ordering::Release)
                    .wrapping_add(1);
                sleep_notice = Some(crate::kernel::SleepNotice {
                    wake_time: thread.wake_time as u64,
                    sleep_seq: seq,
                    thread: thread_addr as *mut Thread,
                });
            }
            // Null `current` first so the new `state` Release-store
            // carries the cur=null write with it. See doc comment.
            context.current.store(null_mut(), Ordering::Release);
            thread.state.store(state as usize, Ordering::Release);
            if let Some(notice) = sleep_notice {
                // Inbox-overflow: log and proceed. The thread still
                // parks correctly (state is set), but won't be woken
                // by the sleep-heap path until something else (e.g.
                // a WAKE_QUEUE event) flips it Ready. At cap=64 this
                // should not realistically fire; a hit means the
                // manager is starved and the diagnostic is the
                // important output.
                if crate::kernel::SLEEP_INBOX.push(notice).is_err() {
                    error!(
                        "SLEEP_INBOX full on park: tid={} wake_time={}",
                        thread.tid, notice.wake_time,
                    );
                }
            }
            // Ready transition (preemption / yield-to-scheduler):
            // publish through the per-hart inbox so the manager
            // folds it into self.ready next pass. Push happens
            // after state.store(Ready) so the manager observing
            // the inbox entry also observes the consistent state.
            if state == ThreadState::Ready {
                if crate::kernel::push_ready_notice(thread_addr as *mut Thread).is_err() {
                    error!(
                        "READY_INBOX full on yield: tid={} — thread will sit \
                         until another path requeues it",
                        thread.tid,
                    );
                }
            }
            // Blocking transition: wire the back-ref between the
            // handle and this thread *after* the state Release so a
            // concurrent signaler that observes state=Blocking also
            // observes the waiter slot. set_waiter must run here
            // (not in the syscall handler) because apply_syscall_outcome
            // already committed the frame snapshot above us — running
            // the hook earlier could have its frame marshaling
            // clobbered by the snapshot copy.
            //
            // After publishing the waiter, re-check is_signaled:
            // catches the race where the signaler raced our
            // set_waiter and returned null from take_waiter (no wake
            // fired). If we got back our own ptr, un-park inline.
            if state == ThreadState::Blocking {
                if let Some(handle) = thread.handle.as_ref() {
                    handle.set_waiter(thread_addr as *mut Thread);
                    if handle.is_signaled() {
                        let claimed = handle.take_waiter();
                        if !claimed.is_null() {
                            // We won the race. Marshal handle rets
                            // into the saved frame, take the handle
                            // out of the thread, mark Ready, queue.
                            // Same shape as the wake hook (kmain
                            // has the registered fn).
                            crate::kernel::wake_blocked_inline(
                                thread_addr as *mut Thread,
                            );
                        }
                    }
                }
            }
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