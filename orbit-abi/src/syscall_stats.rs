//! System-wide per-syscall latency table.
//!
//! Syscall signature (`sys_query_syscall_stats`, syscall number
//! [`QUERY_SYSCALL_STATS`](crate::syscall::QUERY_SYSCALL_STATS)):
//!
//! ```text
//! a0 = QUERY_SYSCALL_STATS  (4102)
//! a1 = buf_ptr              (writable user VA)
//! a2 = buf_len              (>= SYSCALL_STATS_MIN_LEN)
//! -> a0 = bytes written on success, -errno on failure
//! ```
//!
//! Layout: a fixed [`SyscallStatsHeader`] followed by `header.count`
//! [`SyscallEntry`] records, indexed by [`Sysno::ordinal`]. `count` is
//! the kernel's [`Sysno::COUNT`] at build time.
//!
//! Forward-compat: a user built against an older `Sysno::COUNT` reads
//! the prefix it knows and ignores trailing entries; a user built
//! against a newer COUNT sees `header.count < local COUNT` and treats
//! the missing slots as zero. Ordinals themselves are append-only —
//! see [`Sysno::ordinal`](crate::Sysno::ordinal).
//!
//! The table is system-wide (not per-process) — totals across every
//! caller. Per-process attribution can be added later by summing on
//! the kernel side at query time.
//!
//! Errors:
//! - `EFAULT` — buf range is outside the caller's mappable space.
//! - `EINVAL` — `buf_len < SYSCALL_STATS_MIN_LEN`.

use crate::Sysno;

/// Minimum caller buffer: just the header. Anything smaller can't
/// carry the version handshake.
pub const SYSCALL_STATS_MIN_LEN: usize = core::mem::size_of::<SyscallStatsHeader>();

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SyscallStatsHeader {
    /// Total bytes the kernel populated (header + entries).
    pub size: u32,
    /// Number of [`SyscallEntry`] records that follow.
    pub count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SyscallEntry {
    /// Times the kernel dispatched this syscall (system-wide).
    pub count: u64,
    /// Cumulative service ticks (excludes parked-thread wait time).
    pub total_ticks: u64,
}

/// Wire-format size for the kernel's current [`Sysno::COUNT`]. Userland
/// can size its receive buffer to this; if its local COUNT is smaller
/// the kernel writes more than this, the prefix is still valid.
pub const fn payload_size() -> usize {
    core::mem::size_of::<SyscallStatsHeader>()
        + Sysno::COUNT * core::mem::size_of::<SyscallEntry>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_8_bytes() {
        assert_eq!(core::mem::size_of::<SyscallStatsHeader>(), 8);
    }

    #[test]
    fn entry_is_16_bytes() {
        assert_eq!(core::mem::size_of::<SyscallEntry>(), 16);
    }

    #[test]
    fn payload_size_matches_count() {
        // 8-byte header + 16 bytes per ordinal slot.
        assert_eq!(payload_size(), 8 + Sysno::COUNT * 16);
    }
}
