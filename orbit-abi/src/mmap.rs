//! mmap ABI.
//!
//! Syscall signature:
//!
//! ```text
//! a0 = MMAP
//! a1 = hint_vaddr   (0 if kernel should pick)
//! a2 = len          (bytes, will be rounded up to a page)
//! a3 = prot         (Prot bits)
//! a4 = flags        (Flags bits)
//! -> a0 = vaddr on success, -errno on failure
//! ```
//!
//! Kernel rejects mappings that set both W and X, or that set any bit outside
//! [`Prot::MASK`]. U is always added by the kernel; user never sets it.

pub mod prot {
    pub const R: u64 = 1 << 0;
    pub const W: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;

    pub const MASK: u64 = R | W | X;
}

pub mod flags {
    /// Require the kernel to use exactly `hint_vaddr`. Fails with `EEXIST` if
    /// the region overlaps an existing mapping, or `EINVAL` if out of range.
    pub const FIXED: u64 = 1 << 0;

    /// Reserve a guard at the lower end; the mapping may grow down to fill it.
    /// Intended for stacks — not yet implemented kernel-side.
    pub const GROWSDOWN: u64 = 1 << 1;

    pub const MASK: u64 = FIXED | GROWSDOWN;
}
