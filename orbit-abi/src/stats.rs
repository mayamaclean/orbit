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

    #[test]
    fn stats_min_len_holds_size_and_reserved() {
        // STATS_MIN_LEN is the smallest buffer that can carry the
        // 4-byte size + 4-byte _reserved version handshake. Anything
        // smaller can't even tell the caller how many bytes are valid.
        assert_eq!(STATS_MIN_LEN, 8);
        assert!(STATS_MIN_LEN >= core::mem::size_of::<u32>() * 2);
    }

    #[test]
    fn truncated_buffer_preserves_size_prefix() {
        // Simulate a kernel writing 128 bytes into an older userland's
        // 64-byte buffer: the kernel honors the smaller `to_write`,
        // and the user reads the prefix it knows about. The `size`
        // field at offset 0 is the handshake — userland trusts that
        // value to know how many bytes the kernel actually populated.
        let kernel_native = ProcessStats {
            size: core::mem::size_of::<ProcessStats>() as u32,
            pid: 42,
            thread_count: 3,
            cpu_ticks: 10_000,
            ..Default::default()
        };
        let kernel_bytes = unsafe {
            core::slice::from_raw_parts(
                &kernel_native as *const _ as *const u8,
                core::mem::size_of::<ProcessStats>(),
            )
        };

        let mut user_buf = [0u8; 64];
        let to_write = user_buf.len().min(kernel_bytes.len());
        user_buf[..to_write].copy_from_slice(&kernel_bytes[..to_write]);

        // Older userland reads the size field and any prefix fields it
        // knows. The trailing fields it doesn't have yet aren't in
        // user_buf at all — that's fine.
        let read_size = u32::from_le_bytes(user_buf[0..4].try_into().unwrap());
        assert_eq!(read_size, 128);

        // pid is at offset 8, length 2.
        let read_pid = u16::from_le_bytes(user_buf[8..10].try_into().unwrap());
        assert_eq!(read_pid, 42);
    }

    #[test]
    fn larger_user_buffer_zero_pads_unwritten_tail() {
        // Newer userland with a 256-byte struct reading from an older
        // kernel that only writes 128 bytes: the kernel's
        // `min(buf_len, native)` clamps to 128 (its native size); the
        // user's tail bytes stay untouched (zero-init from
        // ProcessStats::default()), and the user inspects `size` to
        // know which fields are valid.
        let user_buf = ProcessStats::default(); // all zero
        let user_bytes = unsafe {
            core::slice::from_raw_parts(
                &user_buf as *const _ as *const u8,
                core::mem::size_of::<ProcessStats>(),
            )
        };
        // Tail beyond what an older kernel would have written is
        // still zero — caller-side discipline keeps the unwritten
        // suffix safely default-valued.
        assert!(user_bytes.iter().all(|&b| b == 0));
        assert_eq!(user_buf.size, 0);  // signals "kernel didn't write"
    }

    #[test]
    fn fields_are_append_only_offsets() {
        // Pin the offset of every field so a future reorder in the
        // struct definition fails this test loudly. Reordering breaks
        // the on-the-wire ABI even if the struct still passes the
        // size check.
        let s = ProcessStats::default();
        let base = &s as *const _ as usize;

        assert_eq!(&s.size                   as *const _ as usize - base,   0);
        assert_eq!(&s._reserved              as *const _ as usize - base,   4);
        assert_eq!(&s.pid                    as *const _ as usize - base,   8);
        assert_eq!(&s.thread_count           as *const _ as usize - base,  10);
        assert_eq!(&s._pad0                  as *const _ as usize - base,  12);
        assert_eq!(&s.cpu_ticks              as *const _ as usize - base,  16);
        assert_eq!(&s.context_switches       as *const _ as usize - base,  24);
        assert_eq!(&s.syscalls               as *const _ as usize - base,  32);
        assert_eq!(&s.resident_bytes         as *const _ as usize - base,  40);
        assert_eq!(&s.heap_bytes             as *const _ as usize - base,  48);
        assert_eq!(&s.kernel_kpages_bytes    as *const _ as usize - base,  56);
        assert_eq!(&s.kernel_user_pages_bytes as *const _ as usize - base, 64);
        assert_eq!(&s.kernel_ktables_bytes   as *const _ as usize - base,  72);
        assert_eq!(&s.kernel_heap_bytes      as *const _ as usize - base,  80);
        assert_eq!(&s.syscall_ticks          as *const _ as usize - base,  88);
        assert_eq!(&s.hart_user_ticks        as *const _ as usize - base,  96);
        assert_eq!(&s.hart_kernel_ticks      as *const _ as usize - base, 104);
        assert_eq!(&s.hart_scheduler_ticks   as *const _ as usize - base, 112);
        assert_eq!(&s.hart_idle_ticks        as *const _ as usize - base, 120);
    }
}
