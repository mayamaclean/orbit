//! Per-hart wall-time accounting and per-syscall latency histogram.
//!
//! Wall time on each hart partitions into four disjoint buckets:
//! `User`, `Kernel`, `Scheduler`, and `Idle`. Six [hook
//! sites](#hook-sites) call [`switch_bucket`] to credit the elapsed
//! ticks since the previous transition to the bucket the hart is
//! leaving, then start charging future ticks to the new bucket.
//!
//! Per-thread CPU time piggy-backs on the same transitions: when a
//! hart leaves the `User` bucket, the elapsed slice is also added to
//! `current_thread.cpu_ticks_total`.
//!
//! # Hook sites
//!
//! 1. Top of `s_trap` Rust body — `→ Kernel` (was User or Idle).
//! 2. Just before sret back to user — `→ User`.
//! 3. Top of `k_hart_loop` WFI — `→ Idle`. Wake-up is bracketed by
//!    the next `s_trap` `→ Kernel` so no explicit "exit Idle" hook.
//! 4. Successful `try_acquire_manager` — `→ Scheduler`.
//! 5. Just before `release_manager` — `→ Kernel`.
//! 6. Hart bringup (`k_harthello`) — call [`init_hart_bucket`] to
//!    seed `bucket_enter_tick` with `now()`. The first `switch_bucket`
//!    then computes a sane elapsed.
//!
//! # Concurrency
//!
//! `current_bucket` and `bucket_enter_tick` are written only by the
//! owning hart — atomics are used for interior mutability under the
//! `&'static HartContext` shape; ordering is `Relaxed`. The `*_ticks`
//! counters are read by [`crate::handle_query_stats`] from any hart;
//! same `Relaxed` ordering, since stats are advisory and tear-read on
//! u64 is impossible on RV64.
//!
//! # Per-syscall histogram
//!
//! [`SYSCALL_STATS`] is a global `[SyscallSlot; Sysno::COUNT]`
//! indexed by [`orbit_abi::Sysno::ordinal`]. The s_trap dispatch
//! brackets each syscall with [`record_syscall`] (count + service
//! ticks). Per-thread `syscall_count` / `syscall_ticks` get the same
//! bracket so per-process aggregates can be summed at snapshot time.

use core::sync::atomic::{AtomicU64, Ordering};

use device::HartContext;
use orbit_abi::Sysno;
use process::Thread;

use crate::kernel::shootdown::CPU_COUNT;

/// Iterator over every hart's `HartContext`. Computed from the
/// current hart's `sscratch` and `hart_id` — the contexts are a
/// contiguous array allocated at boot, so the base is at
/// `sscratch - hart_id * size_of::<HartContext>()`. Length is
/// [`CPU_COUNT`]; returns an empty iterator if it hasn't been
/// published yet (early boot).
pub fn for_each_hart_context(mut visit: impl FnMut(&HartContext)) {
    let count = CPU_COUNT.load(Ordering::Acquire);
    if count == 0 {
        return;
    }
    let here = riscv::register::sscratch::read() as *const HartContext;
    if here.is_null() {
        return;
    }
    // SAFETY: `here` is this hart's context pointer published at boot;
    // dereferencing for `hart_id` is the same access pattern every
    // syscall handler uses. The array is contiguous in kpages and
    // outlives the kernel.
    let here_id = unsafe { (*here).hart_id } as usize;
    let base = unsafe { here.sub(here_id) };
    for i in 0..count {
        // SAFETY: `base..base + count` is bounded by `CPU_COUNT` and
        // matches the allocation in `bin/orbit.rs::rust_main`.
        let h: &HartContext = unsafe { &*base.add(i) };
        visit(h);
    }
}

/// Sum a per-hart `u64` counter (one of the bucket atomics) across
/// every hart.
pub fn sum_hart_counter(field: impl Fn(&HartContext) -> u64) -> u64 {
    let mut acc: u64 = 0;
    for_each_hart_context(|h| acc = acc.wrapping_add(field(h)));
    acc
}

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
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::User,
            1 => Self::Kernel,
            2 => Self::Scheduler,
            _ => Self::Idle,
        }
    }
}

/// Seed `current_bucket` and `bucket_enter_tick` for a hart that's
/// about to start participating in the bucket state machine. Called
/// once per hart in `k_harthello` before the first scheduler pass.
///
/// Without this, the first `switch_bucket` would compute
/// `elapsed = now - 0` and credit ~all of system uptime to the
/// previous bucket. The boot prologue itself isn't charged anywhere
/// — that's the price of a clean baseline.
#[inline]
pub fn init_hart_bucket(hart: &HartContext, bucket: HartBucket) {
    let now = riscv::register::time::read64();
    hart.current_bucket.store(bucket as u8, Ordering::Relaxed);
    hart.bucket_enter_tick.store(now, Ordering::Relaxed);
}

/// Credit ticks elapsed since the last transition to whichever
/// bucket the hart was in, then start charging future ticks to
/// `new`. If the hart was in `User`, the same elapsed slice is also
/// added to the current thread's `cpu_ticks_total` (since per-thread
/// CPU time and `hart.user_ticks` are two views of the same wall
/// time).
///
/// Owning-hart only writes; calling this from a foreign hart corrupts
/// the bucket. All call sites are per-hart by construction (s_trap on
/// the trapping hart, k_hart_loop on the looping hart).
#[inline]
pub fn switch_bucket(hart: &HartContext, new: HartBucket) {
    let now = riscv::register::time::read64();
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
        // Credit per-thread accumulator. `current` is the user thread
        // that just trapped (we haven't switched threads yet).
        let cur = hart.current.load(Ordering::Acquire);
        if !cur.is_null() {
            // SAFETY: `current` was set by the scheduler to a live
            // Thread under MANAGER_LOCK; it's only nulled by
            // `exit_thread_with_state` in the same hart, before the
            // owning hart enters S-mode kernel code that would call
            // `switch_bucket`.
            let t: &Thread = unsafe { (cur as *const Thread).as_ref_unchecked() };
            t.cpu_ticks_total.fetch_add(elapsed, Ordering::Relaxed);
        }
    }
}

// ─── per-syscall histogram (system-wide) ─────────────────────────────

/// One row of the per-syscall stats table. Two atomics keeps the
/// pair tear-safe under concurrent dispatch from multiple harts;
/// `Relaxed` is fine since the count and ticks aren't required to be
/// observed atomically as a pair.
#[repr(C)]
pub struct SyscallSlot {
    pub count: AtomicU64,
    pub total_ticks: AtomicU64,
}

impl SyscallSlot {
    pub const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ticks: AtomicU64::new(0),
        }
    }
}

/// Indexed by [`Sysno::ordinal`] — append-only, sized by
/// [`Sysno::COUNT`].
pub static SYSCALL_STATS: [SyscallSlot; Sysno::COUNT] = [
    SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(),
    SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(),
    SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(),
    SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(), SyscallSlot::new(),
];

const _: () = assert!(
    Sysno::COUNT == 16,
    "SYSCALL_STATS literal must be resized when Sysno::COUNT changes"
);

/// Bracket a syscall: bumps the system-wide histogram and the
/// per-thread `syscall_count` / `syscall_ticks` accumulators.
/// `start_ticks` is the `now()` snapshot taken right before the
/// dispatch arm in `s_trap` invoked the handler; the call here
/// snapshots `now()` again as the end and credits `end - start`.
///
/// Service time only — if the handler returned `Blocking` and parked
/// the thread, the parked window is excluded because the bracket
/// closes before the manager unparks.
#[inline]
pub fn record_syscall(syscall: usize, thread: &Thread, start_ticks: u64) {
    let now = riscv::register::time::read64();
    let elapsed = now.wrapping_sub(start_ticks);

    if let Some(s) = Sysno::from_usize(syscall) {
        let slot = &SYSCALL_STATS[s.ordinal()];
        slot.count.fetch_add(1, Ordering::Relaxed);
        slot.total_ticks.fetch_add(elapsed, Ordering::Relaxed);
    }

    thread.syscall_count.fetch_add(1, Ordering::Relaxed);
    thread.syscall_ticks.fetch_add(elapsed, Ordering::Relaxed);
}
