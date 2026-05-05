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
//! The state-machine logic lives in
//! [`orbit_core::accounting`] (host-testable, time-source agnostic).
//! kmain wraps it here with `riscv::register::time::read64()` as the
//! `now` source, plus the CSR-only helpers
//! ([`for_each_hart_context`], [`sum_hart_counter`]) that walk every
//! hart's context via `sscratch` arithmetic.
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

use core::sync::atomic::Ordering;

use device::HartContext;
use orbit_abi::Sysno;
use process::Thread;

pub use orbit_core::accounting::{HartBucket, SyscallSlot};

use crate::kernel::shootdown::CPU_COUNT;

/// Indexed by [`Sysno::ordinal`] — append-only, sized by
/// [`Sysno::COUNT`]. The kmain side owns the global histogram so that
/// `record_syscall` can update it without threading a pool reference
/// through the trap path; tests in orbit-core construct their own
/// `SyscallSlot`s directly.
pub static SYSCALL_STATS: [SyscallSlot; Sysno::COUNT] = [
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    // GetUid, GetEuid, GetGid, GetEgid, GetGroups, GetLogin —
    // ordinals 35..=40, appended when POSIX credential read syscalls
    // landed.
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    // SetUid, SetGid, SetGroups, SetLogin — ordinals 41..=44,
    // appended for POSIX credential write syscalls.
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
    SyscallSlot::new(),
];

const _: () = assert!(
    Sysno::COUNT == 45,
    "SYSCALL_STATS literal must be resized when Sysno::COUNT changes"
);

/// Seed `current_bucket` and `bucket_enter_tick` for a hart that's
/// about to start participating in the bucket state machine. Reads
/// `now` from the RISC-V `time` CSR; tests use the orbit-core entry
/// point directly.
#[inline]
pub fn init_hart_bucket(hart: &HartContext, bucket: HartBucket) {
    let now = riscv::register::time::read64();
    orbit_core::accounting::init_hart_bucket(hart, bucket, now);
}

/// Credit ticks elapsed since the last transition; see
/// [`orbit_core::accounting::switch_bucket`] for the full state-machine
/// semantics.
#[inline]
pub fn switch_bucket(hart: &HartContext, new: HartBucket) {
    let now = riscv::register::time::read64();
    orbit_core::accounting::switch_bucket(hart, new, now);
}

/// Bracket a syscall: bumps the global histogram and the per-thread
/// `syscall_count` / `syscall_ticks` accumulators. `start_ticks` is
/// the `now()` snapshot taken right before the dispatch arm in
/// `s_trap` invoked the handler; this call snapshots `now()` again as
/// the end and credits `end - start`.
///
/// Service time only — if the handler returned `Blocking` and parked
/// the thread, the parked window is excluded because the bracket
/// closes before the manager unparks.
#[inline]
pub fn record_syscall(syscall: usize, thread: &Thread, start_ticks: u64) {
    let now = riscv::register::time::read64();
    let slot = Sysno::from_usize(syscall).map(|s| &SYSCALL_STATS[s.ordinal()]);
    orbit_core::accounting::record_syscall(slot, thread, start_ticks, now);
}

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
