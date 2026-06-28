//! mmap ABI.
//!
//! Syscall signature:
//!
//! ```text
//! a0 = MMAP
//! a1 = vaddr                (required; must be in the priv/shared range — 0 is rejected)
//! a2 = len                  (bytes, will be rounded up to a page)
//! a3 = perms                (PTE-style R/W/X bits — see `prot`)
//! a4 = share_with_kernel    (0 = private, 1 = shared)
//! -> a0 = vaddr on success, -errno on failure
//! ```
//!
//! `perms` uses the same bit positions as Sv48 PTEs (`mmu::PagePermissions`):
//! R=0x2, W=0x4, X=0x8. The kernel masks with `& 0xE` (drops everything
//! outside R|W|X) and ORs in U; user never sets U directly. X is rejected
//! when `share_with_kernel=1` to preserve W^X across the kernel's KDMAP
//! alias.

pub mod prot {
    pub const R: u64 = 1 << 1;
    pub const W: u64 = 1 << 2;
    pub const X: u64 = 1 << 3;

    pub const MASK: u64 = R | W | X;
}

pub mod flags {
    /// Reserved. `mmap` has no flags argument today and always maps at
    /// exactly the requested vaddr; there is no overlap check, so this
    /// flag is not yet wired.
    pub const FIXED: u64 = 1 << 0;

    /// Reserve a guard at the lower end; the mapping may grow down to fill it.
    /// Intended for stacks — not yet implemented kernel-side.
    pub const GROWSDOWN: u64 = 1 << 1;

    pub const MASK: u64 = FIXED | GROWSDOWN;
}
