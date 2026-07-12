//! Local-hart TLB maintenance wrappers.
//!
//! `SFENCE.VMA` picks its flush scope from whether each operand register
//! *is* `x0` (architectural register 0), not from the runtime value the
//! register holds. The crate's [`riscv::asm::sfence_vma`] always
//! materializes its arguments into general-purpose registers, so passing
//! a literal `0` produces a register that *holds* zero but is not `x0` —
//! which the ISA reads as a concrete VA/ASID of 0 and narrows the flush
//! to a single leaf, not the broad form the caller intended. Encoding the
//! `all-VAs` or `all-ASIDs` scope requires `x0` spelled out in the
//! instruction, which only hand-written asm can do.
//!
//! The four `SFENCE.VMA` scopes and how to reach each:
//!
//! | rs1 (VA) | rs2 (ASID) | scope                       | helper |
//! |----------|------------|-----------------------------|--------|
//! | `x0`     | `x0`       | all VAs, all ASIDs          | [`riscv::asm::sfence_vma_all`] |
//! | `x0`     | reg        | all VAs, one ASID           | [`flush_asid`] |
//! | reg      | `x0`       | one VA, all ASIDs           | [`flush_page_all_asid`] |
//! | reg      | reg        | one VA, one ASID            | [`riscv::asm::sfence_vma`]`(asid, va)` |
//!
//! The `reg`/`reg` quadrant only behaves as a single-VA/single-ASID flush
//! when both operands are genuinely non-zero — a nonzero VA can never land
//! in `x0`, so [`riscv::asm::sfence_vma`] is correct there. The two
//! mixed-`x0` rows are what this module fills in.

use core::arch::asm;

/// `sfence.vma x0, {asid}` — order all reads and writes to every leaf in
/// the address space identified by `asid`, on the local hart. Accesses to
/// global mappings are not ordered (per the privileged ISA).
#[inline(always)]
pub fn flush_asid(asid: usize) {
    // SAFETY: SFENCE.VMA is a fence, not a memory access, and is always
    // legal in S-mode. `x0` in rs1 selects the "all virtual addresses"
    // scope; `asid` in rs2 restricts it to that one address space.
    unsafe {
        asm!("sfence.vma x0, {asid}", asid = in(reg) asid, options(nostack));
    }
}

/// `sfence.vma {va}, x0` — order reads and writes to the leaf PTE for `va`
/// across every address space, on the local hart. Use when the ASID isn't
/// available (e.g. a shootdown ring entry that carries only the VA).
#[inline(always)]
pub fn flush_page_all_asid(va: usize) {
    // SAFETY: as above. `va` in rs1 selects a single page; `x0` in rs2
    // selects the "all ASIDs" scope.
    unsafe {
        asm!("sfence.vma {va}, x0", va = in(reg) va, options(nostack));
    }
}
