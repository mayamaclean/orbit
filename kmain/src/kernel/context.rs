use core::arch::asm;
use core::ptr::null_mut;
use core::sync::atomic::Ordering;

use device::{HartContext, TRAP_STACK_SIZE};
use process::{FaultInfo, Thread, ThreadState};
use riscv::register::sstatus::SPP;
use tracing::error;

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

/// Sentinel placed in `a7` by `kthread_park` before its `ecall`. The
/// `s_trap` cause=9 (S-mode ecall) arm verifies this value before
/// treating the trap as a park request — defense against accidental
/// future S-mode ecall consumers landing in the same dispatch arm.
/// Cause=9 is currently unused by anything else in kmain, so any
/// non-zero value works; this one is just a memorable magic number.
pub const KPARK_ECALL_NR: usize = 0xC0DE_0001;

pub unsafe fn load_thread_into_hart_context_and_jump(
    context: &'static HartContext,
    thread: &'static Thread,
) -> ! {
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
        thread
            .state
            .store(ThreadState::Running as usize, Ordering::Release);
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
            let slot = thread.slot.expect("user thread missing slot");
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
            load_thread_into_hart_context_and_jump(
                context,
                (thread_addr as *const Thread).as_ref_unchecked(),
            );
        }
    }

    let kidle = context.kptr.load(Ordering::Acquire);
    unsafe {
        jump(context, kidle as usize);
    }
}

/// Park the currently-dispatched kernel thread on this hart and hand
/// control back to `k_hart_loop`. Resumes at the call site as if this
/// were an ordinary function return — locals on the kthread's own
/// stack survive the round trip.
///
/// Implementation: stage `(state, wake_time)` into the thread struct,
/// then `ecall` from S-mode (cause=9, delegated to S-mode by bl's
/// `medeleg`). The `s_trap` cause=9 arm snapshots the trap frame at
/// `epc + 4` and calls [`exit_thread_with_state`], which publishes
/// `current = null` (Release) and `state` (Release) and pushes the
/// matching SLEEP_INBOX entry for `Suspended`. Resume is the canonical
/// dispatch path — `enter_hart_kcontext_asm` restores all 32 GPRs from
/// the saved frame and srets to `epc + 4`, landing one instruction
/// past the ecall.
///
/// Caller invariants:
///   * Must be running as a kernel thread (`thread.mode == Supervisor`).
///   * Must hold no spinlocks — handing off to k_hart_loop deadlocks
///     anything else trying to acquire them.
///   * `state` must be `Suspended`. `Blocking` is rejected — kthreads
///     wake on a `CompletionHandle` by polling `is_signaled()` between
///     `Suspended` parks; the wake-hook dance only makes sense for
///     user threads on the syscall-return path.
pub fn kthread_park(state: ThreadState, wake_time: usize) {
    // Async traps off across the entire function. Without this, a
    // timer/SSWI landing between `get_hart_context()` and any later
    // use of the cached `context`/`cur` register values would route
    // through s_trap → check_context_and_switch → exit_thread_with_state(Ready),
    // null this hart's `current`, and let the manager redispatch us
    // on a *different* hart. On resume, `enter_hart_kcontext_asm`
    // restores the trap-time frame — including the register that
    // held `context` — so post-resume reads of `context.current`
    // touch the *previous* hart's HartContext, observe the null we
    // left behind, and fire a spurious "no current thread" panic.
    // Same shape as `load_thread_into_hart_context_and_jump`'s clear,
    // and exactly the smell I dropped when removing the setjmp dance.
    unsafe {
        riscv::register::sstatus::clear_sie();
    }

    let context = get_hart_context();
    let cur = context.current.load(Ordering::Acquire);
    if cur.is_null() {
        panic!(
            "kthread_park with no current thread on hart{} — caller \
             must be running as a kernel thread",
            context.hart_id,
        );
    }
    let thread = unsafe { (cur as *mut Thread).as_mut_unchecked() };

    if thread.mode != SPP::Supervisor {
        panic!(
            "kthread_park: tid={} is not a kernel thread (mode={:?}); \
             user threads must yield via the syscall path",
            thread.tid, thread.mode,
        );
    }

    let tstate = thread.state.load(Ordering::Acquire);
    if tstate != ThreadState::Running as usize {
        let tid = thread.tid;
        let tlast_wake_reason = thread.last_wake_reason.load(Ordering::Acquire);
        let twake_override = thread.wake_override.load(Ordering::Acquire);
        error!("[ktpark] t{tid} s{tstate} lwr{tlast_wake_reason} two{twake_override}");
    }

    if state == ThreadState::Blocking {
        panic!(
            "kthread_park(Blocking) on tid={} — kthreads must use Suspended \
             + a WakeEvent push from the IRQ handler, then poll \
             `handle.is_signaled()` for the result.",
            thread.tid
        );
    }

    // Stage state-machine inputs the s_trap arm reads via the saved
    // frame regs. `wake_time` lives on the thread struct so
    // `exit_thread_with_state(Suspended)` finds it for the SLEEP_INBOX
    // push; an async preemption between this store and the ecall is
    // benign — we'd be redispatched, resume at the ecall, and park
    // with the (already-consistent) value.
    thread.ticks = 0;
    if state == ThreadState::Suspended {
        thread.wake_time = wake_time;
    }

    // ecall → cause=9 → s_trap_vector saves the trap frame in the
    // canonical way → s_trap dispatches on a7 == KPARK_ECALL_NR and
    // calls exit_thread_with_state(state). That noreturn path
    // publishes current=null + state and srets to k_hart_loop. On
    // later redispatch, enter_hart_kcontext_asm restores the saved
    // frame and srets to (epc + 4), landing one instruction past
    // this ecall.
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") state as usize,
            in("a7") KPARK_ECALL_NR,
            // a-regs the ecall doesn't pin are clobbered by convention;
            // s-regs and ra/sp survive via the trap-frame save/restore.
            lateout("a1") _, lateout("a2") _, lateout("a3") _,
            lateout("a4") _, lateout("a5") _, lateout("a6") _,
            options(nostack),
        );
    }
}

/// Stash a `FaultInfo` on the current thread and exit it. Called from the trap
/// handler when a user thread can't continue (page fault, bad ecall, etc.).
/// Manager-side cleanup reads `fault_info` to classify and log.
pub unsafe fn fault_thread(info: FaultInfo) -> ! {
    unsafe {
        let context = (riscv::register::sscratch::read() as *const HartContext).as_ref_unchecked();
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
        let context = (riscv::register::sscratch::read() as *const HartContext).as_ref_unchecked();

        let thread_addr = context.current.load(Ordering::Acquire);
        if thread_addr != null_mut() {
            let thread = (thread_addr as *const Thread).as_ref_unchecked();
            // For a Suspended park, bump sleep_seq *before* publishing
            // state so a manager observing state=Suspended via the
            // Release store below also observes the new seq. Captured
            // here for the SLEEP_INBOX push after the state release.
            let mut sleep_notice: Option<crate::kernel::SleepNotice> = None;
            if state == ThreadState::Suspended {
                let seq = thread
                    .sleep_seq
                    .fetch_add(1, Ordering::Release)
                    .wrapping_add(1);
                sleep_notice = Some(crate::kernel::SleepNotice {
                    wake_time: thread.wake_time as u64,
                    sleep_seq: seq,
                    thread: thread_addr as *mut Thread,
                });
            }

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
                            crate::kernel::wake_blocked_inline(thread_addr as *mut Thread);
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
