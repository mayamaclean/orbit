# trap.S
# Trap handler and global context
# Steve Operating System
# Stephen Marz
# 24 February 2019
.option norvc
.altmacro
.set NUM_GP_REGS, 32  # Number of registers per context
.set REG_SIZE, 8   # Register size (in bytes)

# Use macros for saving and restoring multiple registers
.macro save_gp i, basereg=t6
	sd	x\i, ((\i)*REG_SIZE)(\basereg)
.endm
.macro load_gp i, basereg=t6
	ld	x\i, ((\i)*REG_SIZE)(\basereg)
.endm
.macro save_fp i, basereg=t6
	fsd	f\i, ((NUM_GP_REGS+(\i))*REG_SIZE)(\basereg)
.endm
.macro load_fp i, basereg=t6
	fld	f\i, ((NUM_GP_REGS+(\i))*REG_SIZE)(\basereg)
.endm

.section .text
.global s_trap_vector
# This must be aligned by 4 since the last two bits
# of the mtvec register do not contribute to the address
# of this vector.
.align 4
s_trap_vector:
	csrrw	t6, sscratch, t6

	.set 	i, 0
	.rept	31
		save_gp	%i
		.set	i, i+1
	.endr

	# Save the actual t6 register, which we swapped into
	# sscratch
	mv		t5, t6
	csrr	t6, sscratch
	save_gp 31, t5

	# Restore the kernel trap frame into sscratch
	csrw	sscratch, t5

	csrr	t1, sstatus
	srli	t0, t1, 13
	andi	t0, t0, 3
	li		t3, 3
	bne		t0, t3, 1f
	# Save floating point registers
	.set 	i, 0
	.rept	32
		save_fp	%i, t5
		.set	i, i+1
	.endr
1:
	# Stay on whichever satp triggered the trap. The kernel's shared surface
	# (KTEXT / KDMAP / KMMIO / ktables) is installed in every satp, so kernel
	# code runs correctly; user memory only becomes reachable with SUM set,
	# which syscall handlers toggle via the UserAccess guard.
	csrr t0, sscratch
	ld sp, 528(t0)

	csrr	a0, sepc
	csrr	a1, stval
	csrr	a2, scause
	csrr	a3, sstatus
	csrr	a4, sscratch
	call	s_trap

	# When we get here, we've returned from m_trap, restore registers
	# and return.
	# s_trap will return the return address via a0.

	csrw	sepc, a0
	# Now load the trap frame back into t6
	csrr	t6, sscratch

	csrr	t1, sstatus
	srli	t0, t1, 13
	andi	t0, t0, 3
	li		t3, 3
	bne		t0, t3, 1f
	.set	i, 0
	.rept	32
		load_fp %i
		.set i, i+1
	.endr
1:
	# Restore all GP registers
	.set	i, 1
	.rept	31
		load_gp %i
		.set	i, i+1
	.endr

	# Since we ran this loop 31 times starting with i = 1,
	# the last one loaded t6 back to its original value.
	sret

.global enter_hart_kcontext_asm
# Resume a supervisor-mode thread from its saved frame.
# Caller must ensure sscratch already points at this hart's HartContext -
# the trap vector depends on it. No satp switch happens, so a0 stays valid
# as the frame base throughout (contrast enter_hart_context_asm, where the
# satp switch kills the kernel mapping of the frame and the caller must
# instead pass the user vaddr).
enter_hart_kcontext_asm:
	mv	t6, a0

	# Only touch FP regs when sstatus.FS == Dirty, matching s_trap_vector.
	csrr	t1, sstatus
	srli	t0, t1, 13
	andi	t0, t0, 3
	li	t3, 3
	bne	t0, t3, 2f
	.set	i, 0
	.rept	32
		load_fp %i
		.set i, i+1
	.endr
2:
	.set	i, 1
	.rept	31
		load_gp %i, t6
		.set	i, i+1
	.endr

	sret

.global kthread_save_resume_point
# Setjmp-style save for an in-flight kernel thread that wants to park
# without going through the trap vector.
#
# Args:
#   a0 = *mut TrapFrame  (thread.frame; regs[i] is at offset i*REG_SIZE)
#   a1 = *mut usize      (&thread.pc, the AtomicUsize storage cell)
#
# Returns:
#   a0 = 0 on the first call (saved successfully; caller must complete
#        the park hand-off).
#   a0 = 1 when the thread is later redispatched via
#        enter_hart_kcontext_asm — sret restores all 32 GPRs from the
#        frame (including ra/sp/s0-s11/a0), lands at the resume label
#        below, and the trailing `ret` returns to the original caller
#        of kthread_save_resume_point with a0 = 1.
#
# Saves only the regs the SysV / Rust calling convention requires the
# callee to preserve across this call: ra (x1), sp (x2), gp (x3),
# tp (x4), s0..s11 (x8, x9, x18..x27). Caller-saved regs are dead at
# the call boundary so their slots in the frame are left as-is —
# enter_hart_kcontext_asm will load garbage into them on resume, but
# the caller already treats them as clobbered.
#
# FP state is intentionally not saved/restored: kernel threads in this
# codebase don't dirty FS (matches the FS=Dirty gate in
# enter_hart_kcontext_asm). If a future kthread starts using FP, both
# this path and the existing trap-driven park path need an FP save.
kthread_save_resume_point:
    sd ra,   1*REG_SIZE(a0)
    sd sp,   2*REG_SIZE(a0)
    sd gp,   3*REG_SIZE(a0)
    sd tp,   4*REG_SIZE(a0)
    sd s0,   8*REG_SIZE(a0)
    sd s1,   9*REG_SIZE(a0)
    sd s2,  18*REG_SIZE(a0)
    sd s3,  19*REG_SIZE(a0)
    sd s4,  20*REG_SIZE(a0)
    sd s5,  21*REG_SIZE(a0)
    sd s6,  22*REG_SIZE(a0)
    sd s7,  23*REG_SIZE(a0)
    sd s8,  24*REG_SIZE(a0)
    sd s9,  25*REG_SIZE(a0)
    sd s10, 26*REG_SIZE(a0)
    sd s11, 27*REG_SIZE(a0)

    # Pre-stage the resume-time a0 value as 1 so sret-into-resume
    # delivers the "I was woken" signal in a0. Live a0 below is set
    # to 0 for the first-time return.
    li t0, 1
    sd t0, 10*REG_SIZE(a0)

    # Resume PC = label below. Stored into thread.pc so
    # load_thread_into_hart_context_and_jump's sepc::write picks it up
    # when the scheduler eventually redispatches us.
    lla t0, .Lkthread_resume
    sd  t0, 0(a1)

    # First-time return: a0 = 0, jr ra to caller.
    li a0, 0
    ret

.Lkthread_resume:
    # Reentry from enter_hart_kcontext_asm. All 32 GPRs already
    # restored from the frame: a0 = 1 (pre-staged above), ra/sp/
    # s-regs match values saved on the first call. Just return.
    ret

.global kthread_handoff_to_kidle
# Noreturn hand-off used by kthread_park after a successful save.
# Switches sp off the kthread's stack onto this hart's idle stack
# *before* publishing state = Suspended, so a remote dispatch
# (which restores our saved sp) cannot collide with our continued
# use of the same stack while we finish the hand-off.
#
# Args:
#   a0 = *mut usize        (&thread.state — AtomicUsize storage)
#   a1 = usize             (state value to publish, e.g. Suspended)
#   a2 = *mut *mut Thread  (&hart_context.current — AtomicPtr storage)
#   a3 = usize             (top-of-stack address for this hart's k_stack)
#   a4 = usize             (kptr — k_hart_loop entrypoint)
#
# Preconditions:
#   * sstatus.SIE = 0 (caller cleared it; we keep async traps off
#     across the publish window).
#   * sscratch already points at this hart's HartContext (unchanged
#     since we never returned from S-mode).
#   * The kernel satp is already loaded (kthread runs under it).
#
# After sret to a4, control resumes in k_hart_loop on the idle stack;
# k_hart_loop sees current = null, falls through to the manager /
# WFI path. The parked thread is dispatched on the next scheduler
# pass that finds a runnable hart.
kthread_handoff_to_kidle:
    # Step 1: switch off the kthread stack first. After this point a
    # remote hart could legally restore our saved sp and reuse the
    # vacated stack region.
    mv sp, a3

    # Step 2: clear this hart's `current` slot first. Mirrors the
    # ordering in exit_thread_with_state — the subsequent state
    # Release-store carries the cur=null write with it, so a remote
    # Acquire-load that sees state==Suspended is guaranteed to also
    # see current==null on this hart. The reverse order opens a
    # double-dispatch race against the manager's self_view path.
    fence rw, w
    sd zero, 0(a2)

    # Step 3: publish state with Release ordering. The fence rw, w
    # plus ordinary store matches AtomicUsize::store(_, Release) on
    # RV64.
    fence rw, w
    sd a1, 0(a0)

    # Step 4: sret into k_hart_loop. SPP must be Supervisor so we
    # stay in S-mode; SPIE inherits whatever sstatus.SIE was at the
    # csrs (0, since the caller cleared it), so SIE will be 0 on
    # entry to k_hart_loop. k_hart_loop's loop-top `clear_sie()`
    # plus its own SIE management cover the rest.
    csrw sepc, a4
    li   t0, 1 << 8        # sstatus.SPP
    csrs sstatus, t0
    sret

.global enter_hart_context_asm
# Resume a user-mode thread.
#   a0 = user vaddr of this thread's TrapFrame (mapped R-only in the user
#        pagetable by map_trap_frame at USER_TRAP_FRAME_BASE + slot * 4096)
#   a1 = thread satp (user pagetable + asid + Sv48)
#   a2 = asid (for sfence.vma)
#
# Precondition: sscratch already holds this hart's HartContext, and the kernel
# (text, data, GOT, kpages, MMIO) is identity-mapped into the user pagetable
# via map_kernel_shared, so this asm keeps executing across the satp switch.
# The trap vector restores sscratch on every entry, so we don't need to
# round-trip it through the frame here.
enter_hart_context_asm:
	# Switch to the user address space. The kernel is mapped into it, so
	# the next instruction still fetches from the same physical .text.
	csrw	satp, a1
	sfence.vma	x0, a2

	# Hold the frame base in t6 — it gets clobbered last by load_gp 31
	# (x31 == t6), which leaves it holding the user's saved t6, exactly
	# what we want for sret.
	mv	t6, a0

	# Only touch FP regs when sstatus.FS == Dirty, matching s_trap_vector.
	csrr	t1, sstatus
	srli	t0, t1, 13
	andi	t0, t0, 3
	li	t3, 3
	bne	t0, t3, 2f
	.set	i, 0
	.rept	32
		load_fp %i
		.set i, i+1
	.endr
2:
	.set	i, 1
	.rept	31
		load_gp %i
		.set	i, i+1
	.endr

	sret
