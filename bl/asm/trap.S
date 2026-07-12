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

# TrapFrame field offsets (mirrored from device::TrapFrame).
# regs[32] @ 0, fregs[32] @ 256, asid @ 512, scratch @ 520.
# We piggyback the M-mode trap handler on the asid slot to stash the
# saved mepc; scratch is currently unused but reserved for future
# per-trap state.
.set TRAPFRAME_MEPC_SAVE, 512

# Per-hart M-mode trap stack slot. The 512 KiB stack region in memory.x
# is split into two halves:
#   top 256 KiB  — boot stacks  (set up by boot.S part2/part3)
#   bottom 256 KiB — M-mode trap stacks (this layout)
# Each hart gets a 64 KiB slot in each half; trap and boot slots for
# the *same* hart are in different halves so an MSI taken inside
# kinit_hart's WFI doesn't have m_trap's register-save area land on
# top of kinit_hart's saved hartid (which used to corrupt the post-
# mret a0 to 0 and made the secondary fall through to the primary
# boot path in kmain `_start`).
#
# Hart N's trap sp = _stack_end - 0x40000 - N * 64 KiB.
# 64 KiB → shift 16; the explicit 0x40000 sub places the trap region
# below the boot region.
.set HART_TRAP_STACK_SHIFT, 16
.set HART_TRAP_STACK_BASE_OFFSET, 0x40000

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
	# We deliberately do NOT touch `mie` here: m_trap's only async arm
	# (cause-3) clears the CLINT MSIP to stop the same interrupt
	# re-firing the instant we mret. Clobbering mie around the call could
	# undo state the handler relies on.

	csrr	a0, mepc
	sd		a0, TRAPFRAME_MEPC_SAVE(t5)
	csrr	a1, mtval
	csrr	a2, mcause
	csrr	a3, mhartid
	csrr	a4, mstatus
	csrr	a5, mscratch

	# Per-hart M-mode trap stack. Trap stacks occupy the bottom half
	# of bl's 512 KiB stack region; boot stacks live in the top half
	# (see boot.S part3). For hart N:
	#   sp = _stack_end - HART_TRAP_STACK_BASE_OFFSET - N * 64 KiB
	# The base offset places the trap region 256 KiB below stack_end
	# so it never overlaps any hart's boot stack — fixing the prior
	# corruption where m_trap's saved s1 landed on kinit_hart's saved
	# hartid and made the post-mret a0=0.
	la		t0, KERNEL_STACK_END
	ld		t0, 0(t0)
	li		t2, HART_TRAP_STACK_BASE_OFFSET
	sub		t0, t0, t2                        # t0 = _stack_end - 256 KiB
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
