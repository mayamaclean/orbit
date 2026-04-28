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

    #[test]
    fn header_layout_size_first_then_count() {
        // The kernel writes `size` and `count` as little-endian u32s
        // back-to-back; the order is part of the wire ABI. Userland
        // reads `size` to validate the payload byte length, then
        // `count` to know how many entries follow.
        let h = SyscallStatsHeader::default();
        let base = &h as *const _ as usize;
        assert_eq!(&h.size as *const _ as usize - base, 0);
        assert_eq!(&h.count as *const _ as usize - base, 4);
    }

    #[test]
    fn entry_layout_count_first_then_total_ticks() {
        let e = SyscallEntry::default();
        let base = &e as *const _ as usize;
        assert_eq!(&e.count as *const _ as usize - base, 0);
        assert_eq!(&e.total_ticks as *const _ as usize - base, 8);
    }

    #[test]
    fn truncated_buffer_yields_partial_entries() {
        // Kernel writes 16 entries (256 + 8 = 264 bytes). User has a
        // 100-byte buffer: kernel must clamp to floor((100-8)/16) = 5
        // full entries + 8-byte header = 88 bytes written, header
        // declares count=5. Trailing 12 bytes of user buffer untouched.
        let kernel_total = SYSCALL_STATS_MIN_LEN + Sysno::COUNT * core::mem::size_of::<SyscallEntry>();
        assert_eq!(kernel_total, 264);

        let user_buf_len: usize = 100;
        let entries_capacity =
            (user_buf_len - SYSCALL_STATS_MIN_LEN) / core::mem::size_of::<SyscallEntry>();
        assert_eq!(entries_capacity, 5);

        let written = SYSCALL_STATS_MIN_LEN + entries_capacity * core::mem::size_of::<SyscallEntry>();
        assert_eq!(written, 88);
        assert!(written <= user_buf_len);
    }

    #[test]
    fn newer_kernel_count_smaller_when_user_buffer_is_too_small() {
        // The header `count` reflects what the kernel actually wrote,
        // not what its native COUNT is. Userland iterates exactly
        // `header.count` entries — never assumes more.
        let user_buf_len = SYSCALL_STATS_MIN_LEN + 3 * core::mem::size_of::<SyscallEntry>();
        let entries_capacity =
            (user_buf_len - SYSCALL_STATS_MIN_LEN) / core::mem::size_of::<SyscallEntry>();
        let header = SyscallStatsHeader {
            count: entries_capacity as u32,
            size: (SYSCALL_STATS_MIN_LEN + entries_capacity * core::mem::size_of::<SyscallEntry>())
                as u32,
        };
        assert_eq!(header.count, 3);
        assert_eq!(header.size as usize, user_buf_len);
        // A reader iterating min(header.count, local_count) gets 3,
        // which is correct: the missing 13 ordinals are absent.
    }

    #[test]
    fn min_len_buffer_writes_header_only() {
        // A user buffer exactly SYSCALL_STATS_MIN_LEN can carry the
        // header but zero entries. count=0 is a valid response — the
        // user learns nothing about syscalls but the version
        // handshake works.
        let entries_capacity = (SYSCALL_STATS_MIN_LEN - SYSCALL_STATS_MIN_LEN)
            / core::mem::size_of::<SyscallEntry>();
        assert_eq!(entries_capacity, 0);

        let written = SYSCALL_STATS_MIN_LEN + entries_capacity * core::mem::size_of::<SyscallEntry>();
        assert_eq!(written, SYSCALL_STATS_MIN_LEN);
    }
}
