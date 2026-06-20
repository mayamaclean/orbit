//! Per-hart bucket accounting and per-syscall service-time helpers.
//!
//! Time-source agnostic: every entry point takes a `now: u64` (and
//! optionally a `start: u64`) so the same code drives both the live
//! kernel and host tests. Wrappers in
//! [`kmain::kernel::accounting`](../../../../kmain/src/kernel/accounting.rs)
//! supply `riscv::register::time::read64()`; tests script `now`
//! values directly.
//!
//! Concurrency: per-hart fields are written only by the owning hart;
//! foreign-hart reads (stats snapshots) go through `Relaxed` atomic
//! loads and are advisory. Per-thread accumulators are read via the
//! same path. Tear-safe on RV64 where 8-byte loads/stores are
//! naturally atomic.

use core::sync::atomic::{AtomicU64, Ordering};

use device::HartContext;
use process::{RunningThread, Thread};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HartBucket {
    User = 0,
    Kernel = 1,
    Scheduler = 2,
    Idle = 3,
}

impl HartBucket {
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::User,
            1 => Self::Kernel,
            2 => Self::Scheduler,
            _ => Self::Idle,
        }
    }
}

/// Seed `current_bucket` and `bucket_enter_tick` for a hart that's
/// about to start participating in the bucket state machine. Without
/// this, the first [`switch_bucket`] would compute
/// `elapsed = now - 0` and credit ~all of system uptime to the
/// previous bucket. Pre-init wall time isn't charged anywhere — that
/// is the price of a clean baseline.
#[inline]
pub fn init_hart_bucket(hart: &HartContext, bucket: HartBucket, now: u64) {
    hart.current_bucket.store(bucket as u8, Ordering::Relaxed);
    hart.bucket_enter_tick.store(now, Ordering::Relaxed);
}

/// Credit ticks elapsed since the last transition to whichever bucket
/// the hart was in, then start charging future ticks to `new`. If the
/// hart was in [`HartBucket::User`], the same elapsed slice is also
/// added to the current thread's `cpu_ticks_total` (per-thread CPU
/// time and `hart.user_ticks` are two views of the same wall time).
///
/// Owning-hart only writes; calling this from a foreign hart corrupts
/// the bucket. Live call sites are per-hart by construction (`s_trap`
/// on the trapping hart, `k_hart_loop` on the looping hart).
#[inline]
pub fn switch_bucket(hart: &HartContext, new: HartBucket, now: u64) {
    let prev_u8 = hart.current_bucket.swap(new as u8, Ordering::Relaxed);
    let prev_start = hart.bucket_enter_tick.swap(now, Ordering::Relaxed);
    let elapsed = now.wrapping_sub(prev_start);
    let prev = HartBucket::from_u8(prev_u8);

    let counter: &AtomicU64 = match prev {
        HartBucket::User => &hart.user_ticks,
        HartBucket::Kernel => &hart.kernel_ticks,
        HartBucket::Scheduler => &hart.scheduler_ticks,
        HartBucket::Idle => &hart.idle_ticks,
    };
    counter.fetch_add(elapsed, Ordering::Relaxed);

    if prev == HartBucket::User {
        // `current` is the user thread that just trapped (we haven't
        // switched threads yet).
        let cur = hart.current.load(Ordering::Acquire);
        if !cur.is_null() {
            // SAFETY: `current` was set by the scheduler to a live
            // Thread under MANAGER_LOCK; it's only nulled by
            // `exit_thread_with_state` in the same hart, before the
            // owning hart enters S-mode kernel code that would call
            // `switch_bucket`.
            //
            // Field-project the atomic bump off the raw `current` ptr —
            // forming `&Thread` would retag the whole struct and freeze
            // the cred fields a sibling may be propagating to this
            // still-Running thread (the cap-layer aliasing invariant).
            unsafe {
                (*(cur as *const Thread))
                    .cpu_ticks_total
                    .fetch_add(elapsed, Ordering::Relaxed);
            }
        }
    }
}

/// One row of the per-syscall stats table. Three atomics keep the
/// triple tear-safe under concurrent dispatch from multiple harts;
/// `Relaxed` is fine since the count, total, and max aren't required
/// to be observed atomically as a group.
///
/// `max_ticks` tracks the longest single dispatch ever recorded.
/// Combined with `total_ticks / count` (mean) this gives a quick
/// outlier signal without a full histogram — primary consumer is
/// migration A/B comparisons where a regression can hide in tail
/// latency even if the mean looks flat.
#[repr(C)]
pub struct SyscallSlot {
    pub count: AtomicU64,
    pub total_ticks: AtomicU64,
    pub max_ticks: AtomicU64,
}

impl SyscallSlot {
    pub const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ticks: AtomicU64::new(0),
            max_ticks: AtomicU64::new(0),
        }
    }
}

/// Bracket a syscall: bumps the optional global slot and the
/// per-thread `syscall_count` / `syscall_ticks`. Service time only —
/// the caller chooses `start` (snapshotted at trap entry) and `end`
/// (snapshotted right before the dispatch path either returns to
/// user or hands off to the manager).
///
/// `slot = None` skips the global histogram while still updating the
/// thread (used when the syscall number isn't recognized — kmain
/// guards with `Sysno::from_usize` so unknown sysnos don't pollute
/// the dense ordinal table).
#[inline]
pub fn record_syscall(
    slot: Option<&SyscallSlot>,
    running: &RunningThread,
    start: u64,
    end: u64,
) {
    let elapsed = end.wrapping_sub(start);
    if let Some(s) = slot {
        s.count.fetch_add(1, Ordering::Relaxed);
        s.total_ticks.fetch_add(elapsed, Ordering::Relaxed);
        s.max_ticks.fetch_max(elapsed, Ordering::Relaxed);
    }
    // Field-projected per-thread bump (own-hart) — no whole-struct
    // `&Thread` over the cred fields a sibling may be propagating.
    running.account_syscall(elapsed);
}
