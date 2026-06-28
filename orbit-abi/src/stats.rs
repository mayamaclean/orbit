//! Process and kernel statistics ABI.
//!
//! Syscall signature (`handle_query_stats`, syscall number
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
    ///
    /// Includes the per-source framebuffer scrollback (one
    /// `VecDeque<String>` capped at `SCROLLBACK_LINES` lines per
    /// source). Each `console_write`d line that completes with `\n`
    /// donates the `pending` String's grown capacity (≤ `MAX_LINE_LEN`
    /// bytes) into the deque. Until the deque hits its line cap the
    /// per-source contribution grows monotonically — looks like a
    /// leak in stats output, but it plateaus at ~140 KB per source
    /// and releases on `dealloc_process`'s `RemoveSource`. See
    /// `hello/src/main.rs::scrollback_bounding_test` for a
    /// reproduction.
    pub kernel_heap_bytes: u64,

    // ─── per-process syscall service time ────────────────────────────
    /// Cumulative kernel service time across all syscalls dispatched
    /// against threads of this process. Excludes time spent parked
    /// — measures kernel work, not response time.
    pub syscall_ticks: u64,

    // ─── system-wide hart accounting (summed across every hart) ──────
    /// All four buckets are partition-disjoint: at any instant a hart
    /// is in exactly one of {user, kernel, scheduler, idle}, and the
    /// four fields together approximate `HART_COUNT * uptime_ticks`.
    pub hart_user_ticks: u64,
    pub hart_kernel_ticks: u64,
    pub hart_scheduler_ticks: u64,
    pub hart_idle_ticks: u64,

    // ─── per-process denial counters ─────────────────────────────────
    /// Number of times the dispatch-site bitmask gate has EPERMed a
    /// syscall from this process. Monotonic; the smoke-test fast
    /// path ("did the gate fire at all since `before_delta`?")
    /// reads this delta.
    ///
    /// Bookkeeping is kernel-side: the manager-side
    /// `drain_denial_events` pass increments this field as it
    /// folds `PermDeny` events off the lock-free producer queue
    /// into the kernel-wide ring. Pairs with the
    /// [`DenialEvent`](crate::denial::DenialEvent) ring — this
    /// counter answers "how many?", the ring answers "which?".
    pub perm_denials: u64,
    /// Number of times `create_process_v2`'s role-transition gate
    /// has EPERMed a spawn from this process. Same per-process
    /// shape as `perm_denials`. The `create_process_v2` handler
    /// records the audit event and bumps this counter inline
    /// (under MANAGER_LOCK) before returning `-EPERM`.
    pub role_denials: u64,

    // ─── kernel-wide WAKE_QUEUE telemetry ────────────────────────────
    /// High-water mark of `WAKE_QUEUE` depth observed across the
    /// kernel's lifetime. Sampled `fetch_max` after each successful
    /// push, so it never decreases. Combined with `wake_queue_capacity`
    /// gives "how close did we come to dropping?" — a peak approaching
    /// cap is the cue to bump the cap.
    pub wake_queue_peak: u64,
    /// Cumulative count of `WAKE_QUEUE.push()` attempts that EAGAIN'd
    /// because the ring was full. Each drop is a missed wake;
    /// callers today either coalesce naturally (net) or rely on a
    /// heartbeat fallback (k_net 10 ms). Non-zero is a signal that
    /// the cap is undersized for the workload.
    pub wake_queue_drops: u64,
    /// Build-time capacity of `WAKE_QUEUE`. Reported in the snapshot
    /// so userland can compute headroom without depending on a
    /// kernel-side const that might change. Today's value: 128.
    pub wake_queue_capacity: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_168_bytes() {
        // Pinning the size keeps reviewers honest about ABI growth.
        // Bump only when appending fields (and update the kernel's
        // matching write path in lockstep). The trailing three u64s
        // (`wake_queue_peak` + `wake_queue_drops` + `wake_queue_capacity`)
        // appended for migration measurement bring the size from
        // 144 to 168.
        assert_eq!(core::mem::size_of::<ProcessStats>(), 168);
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
        assert_eq!(read_size, core::mem::size_of::<ProcessStats>() as u32);

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
        assert_eq!(user_buf.size, 0); // signals "kernel didn't write"
    }

    #[test]
    fn fields_are_append_only_offsets() {
        // Pin the offset of every field so a future reorder in the
        // struct definition fails this test loudly. Reordering breaks
        // the on-the-wire ABI even if the struct still passes the
        // size check.
        let s = ProcessStats::default();
        let base = &s as *const _ as usize;

        assert_eq!(&s.size as *const _ as usize - base, 0);
        assert_eq!(&s._reserved as *const _ as usize - base, 4);
        assert_eq!(&s.pid as *const _ as usize - base, 8);
        assert_eq!(&s.thread_count as *const _ as usize - base, 10);
        assert_eq!(&s._pad0 as *const _ as usize - base, 12);
        assert_eq!(&s.cpu_ticks as *const _ as usize - base, 16);
        assert_eq!(&s.context_switches as *const _ as usize - base, 24);
        assert_eq!(&s.syscalls as *const _ as usize - base, 32);
        assert_eq!(&s.resident_bytes as *const _ as usize - base, 40);
        assert_eq!(&s.heap_bytes as *const _ as usize - base, 48);
        assert_eq!(&s.kernel_kpages_bytes as *const _ as usize - base, 56);
        assert_eq!(&s.kernel_user_pages_bytes as *const _ as usize - base, 64);
        assert_eq!(&s.kernel_ktables_bytes as *const _ as usize - base, 72);
        assert_eq!(&s.kernel_heap_bytes as *const _ as usize - base, 80);
        assert_eq!(&s.syscall_ticks as *const _ as usize - base, 88);
        assert_eq!(&s.hart_user_ticks as *const _ as usize - base, 96);
        assert_eq!(&s.hart_kernel_ticks as *const _ as usize - base, 104);
        assert_eq!(&s.hart_scheduler_ticks as *const _ as usize - base, 112);
        assert_eq!(&s.hart_idle_ticks as *const _ as usize - base, 120);
        assert_eq!(&s.perm_denials as *const _ as usize - base, 128);
        assert_eq!(&s.role_denials as *const _ as usize - base, 136);
    }
}
