# trap.S
# Trap handler and global context
# Steve Operating System
# Stephen Marz
# 24 February 2019
.option norvc
.altmacro
.set NUM_GP_REGS, 32  # Number of registers per context
.set REG_SIZE, 8   # Register size (in bytes)

# TrapFrame field offsets (mirrored from device::TrapFrame).
# regs[32] @ 0, fregs[32] @ 256, asid @ 512, scratch @ 520.
# We piggyback the M-mode trap handler on the asid slot to stash the
# saved mepc; scratch is currently unused but reserved for future
# per-trap state.
.set TRAPFRAME_MEPC_SAVE, 512

# Per-hart M-mode trap stack slot. The 512 KiB stack region in memory.x
# is split into one slot per hart so concurrent traps don't share sp.
# Slot size = 128 KiB → shift 17. Hart_n's sp = _stack_end - n*128 KiB,
# so hart 0 has the top slot; hart MAX-1 has the bottom slot adjacent
# to .rodata (overflow there hits PMP-locked rodata once PR5 lands).
.set HART_TRAP_STACK_SHIFT, 17

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
.global m_trap_vector
# This must be aligned by 4 since the last two bits
# of the mtvec register do not contribute to the address
# of this vector.
.align 4
m_trap_vector:
	# All registers are volatile here, we need to save them
	# before we do anything.
	csrrw	t6, mscratch, t6
	# csrrw will atomically swap t6 into mscratch and the old
	# value of mscratch into t6. This is nice because we just
	# switched values and didn't destroy anything -- all atomically!
	# in cpu.rs we have a structure of:
	#  32 gp regs		0
	#  32 fp regs		256
	# We use t6 as the temporary register because it is the very
	# bottom register (x31)
	.set 	i, 0
	.rept	31
		save_gp	%i
		.set	i, i+1
	.endr

	# Save the actual t6 register, which we swapped into
	# mscratch
	mv		t5, t6
	csrr	t6, mscratch
	save_gp 31, t5

	# Restore the kernel trap frame into mscratch
	csrw	mscratch, t5

	csrr	t1, mstatus
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
	# Note: hardware already sets mstatus.MIE = 0 on trap entry, which
	# blocks nested M-mode interrupts for the duration of m_trap_vector.
	# We deliberately do NOT touch `mie` here: m_trap's per-cause arms
	# need to clear specific bits (e.g. mie.SSIE on cause-1, mie.MTIE on
	# cause-7) to prevent the same interrupt re-firing the instant we
	# mret. Clobbering mie around the call would undo those clears.

	csrr	a0, mepc
	sd		a0, TRAPFRAME_MEPC_SAVE(t5)
	csrr	a1, mtval
	csrr	a2, mcause
	csrr	a3, mhartid
	csrr	a4, mstatus
	csrr	a5, mscratch

	# Per-hart M-mode trap stack. KERNEL_STACK_END holds `_stack_end`,
	# the top of bl's stack region. Each hart gets a 128 KiB slot
	# (HART_TRAP_STACK_SHIFT) carved from the top down — hart 0 at
	# _stack_end, hart 1 at _stack_end - 128 KiB, etc. Without this
	# every hart loaded the same sp and concurrent M-mode traps raced
	# on the shared stack frames inside Rust's m_trap.
	la		t0, KERNEL_STACK_END
	ld		t0, 0(t0)
	csrr	t1, mhartid
	slli	t1, t1, HART_TRAP_STACK_SHIFT
	sub		sp, t0, t1
	call	m_trap

	# When we get here, we've returned from m_trap, restore registers
	# and return.
	# m_trap will return the return address via a0.

	csrw	mepc, a0
	# Now load the trap frame back into t6
	csrr	t6, mscratch

	csrr	t1, mstatus
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
	mret
