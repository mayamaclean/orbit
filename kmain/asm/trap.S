# Derived from Stephen Marz's osblog (The Adventures of OS):
# https://github.com/sgmarz/osblog — MIT, Copyright (c) 2019 Stephen Marz
# (original header below). Heavily modified for orbit; full license text
# in THIRD_PARTY_NOTICES.md at the repo root.
#
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

	# When we get here, we've returned from s_trap, restore registers
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

.global enter_hart_context_asm
# Resume a user-mode thread.
#   a0 = user vaddr of this thread's TrapFrame (mapped R-only in the user
#        pagetable by map_trap_frame at USER_TRAP_FRAME_BASE + slot * 4096)
#   a1 = thread satp (user pagetable + asid + Sv48)
#   a2 = asid (for sfence.vma)
#
# Precondition: sscratch already holds this hart's HartContext, and the kernel
# (text, data, GOT, kpages, MMIO) is mapped into the user pagetable via the
# shared high-half L2 (map_kernel_shared), so this asm keeps executing across
# the satp switch.
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
