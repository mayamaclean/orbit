//! Process and kernel statistics ABI.
//!
//! Syscall signature (`sys_query_stats`, syscall number
//! [`QUERY_STATS`](crate::syscall::QUERY_STATS)):
//!
//! ```text
//! a0 = QUERY_STATS  (4101)
//! a1 = buf_ptr      (writable user VA, must satisfy user_range_ok)
//! a2 = buf_len      (bytes; >= STATS_MIN_LEN)
//! -> a0 = bytes written on success, -errno on failure
//! ```
//!
//! Forward-compat: the kernel writes `min(buf_len, sizeof::<ProcessStats>())`
//! bytes and stores its native struct size in the leading `size` field.
//! Older userland with a smaller local struct reads a valid prefix; newer
//! userland with a larger local struct sees the kernel's `size` and treats
//! trailing fields as zero. Fields are append-only — never reorder, never
//! repurpose, never shrink.
//!
//! Errors:
//! - `EFAULT` — `buf_ptr` / `buf_len` falls outside the caller's mappable range.
//! - `EINVAL` — `buf_len < STATS_MIN_LEN` (must hold at least the size hdr).

/// Minimum caller buffer: just the `size` u32 + the `_reserved` u32.
/// Anything smaller can't carry the version handshake.
pub const STATS_MIN_LEN: usize = 8;

/// All times are in `time` CSR ticks (10 MHz on qemu-virt → divide by
/// 10_000 for milliseconds).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessStats {
    /// Bytes the kernel actually populated (its native sizeof). Userland
    /// uses this to detect appended fields without trapping again.
    pub size: u32,
    /// Reserved for future flags / version bits. Kernel writes 0.
    pub _reserved: u32,

    // ─── per-process identity ────────────────────────────────────────
    pub pid: u16,
    /// Live thread count (Ready + Running + Suspended + Blocking;
    /// excludes Exited).
    pub thread_count: u16,
    pub _pad0: u32,

    // ─── per-process accounting ──────────────────────────────────────
    /// Cumulative CPU time across every thread of this process.
    pub cpu_ticks: u64,
    /// Times any thread of this process transitioned into Running.
    pub context_switches: u64,
    /// Syscalls dispatched against any thread of this process.
    pub syscalls: u64,

    // ─── per-process memory ──────────────────────────────────────────
    /// Sum of layout sizes for all backed VMAs in `process.maps`
    /// (guard reservations excluded). Approximates RSS.
    pub resident_bytes: u64,
    /// Sum of layout sizes for entries in `process.heap_pages` —
    /// covers user-private + shared mmap pools, regardless of whether
    /// the user has unmapped the VA. Equals the process's footprint
    /// against the user_pages / kpages allocators.
    pub heap_bytes: u64,

    // ─── kernel-wide snapshot (same value for all callers) ───────────
    /// Bytes outstanding from the kpages pool (kernel-shared backing,
    /// e.g. trap frames, thread stacks, NetChannel rings).
    pub kernel_kpages_bytes: u64,
    /// Bytes outstanding from the user_pages pool (user-private backing).
    pub kernel_user_pages_bytes: u64,
    /// Bytes outstanding from the ktables pool (page-table frames).
    pub kernel_ktables_bytes: u64,
    /// Bytes outstanding from KHEAP (linked-list kernel allocator).
    pub kernel_heap_bytes: u64,

    // ─── per-process syscall service time ────────────────────────────
    /// Cumulative kernel service time across all syscalls dispatched
    /// against threads of this process. Excludes time spent parked on
    /// a `ThreadBlockReason` — measures kernel work, not response time.
    pub syscall_ticks: u64,

    // ─── system-wide hart accounting (summed across every hart) ──────
    /// All four buckets are partition-disjoint: at any instant a hart
    /// is in exactly one of {user, kernel, scheduler, idle}, and the
    /// four fields together approximate `HART_COUNT * uptime_ticks`.
    pub hart_user_ticks: u64,
    pub hart_kernel_ticks: u64,
    pub hart_scheduler_ticks: u64,
    pub hart_idle_ticks: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_128_bytes() {
        // Pinning the size keeps reviewers honest about ABI growth.
        // Bump only when appending fields (and update the kernel's
        // matching write path in lockstep).
        assert_eq!(core::mem::size_of::<ProcessStats>(), 128);
    }

    #[test]
    fn size_field_is_at_offset_zero() {
        // Forward-compat depends on the size prefix being the first
        // u32 of the struct.
        let s = ProcessStats::default();
        let base = &s as *const _ as usize;
        let size = &s.size as *const _ as usize;
        assert_eq!(size - base, 0);
    }
}
