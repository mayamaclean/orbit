//! Tests for the time-source-agnostic accounting state machine.
//! kmain wraps these entry points with `riscv::register::time::read64()`
//! as the `now` source; tests script `now` explicitly.

mod common;

use std::sync::atomic::Ordering;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_core::accounting::{
    HartBucket, SyscallSlot, init_hart_bucket, record_syscall, switch_bucket,
};

use common::{make_hart_context, make_thread};

// ───── switch_bucket ───────────────────────────────────────────────

#[test]
fn switch_bucket_credits_previous_bucket() {
    let hart = make_hart_context();
    init_hart_bucket(hart, HartBucket::Kernel, 1000);

    switch_bucket(hart, HartBucket::User, 1500);

    assert_eq!(hart.kernel_ticks.load(Ordering::Relaxed), 500);
    assert_eq!(hart.user_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(hart.scheduler_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(hart.idle_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(
        hart.current_bucket.load(Ordering::Relaxed),
        HartBucket::User as u8
    );
    assert_eq!(hart.bucket_enter_tick.load(Ordering::Relaxed), 1500);
}

#[test]
fn switch_bucket_chain_partitions_correctly() {
    // Drive every bucket through every other; each transition's elapsed
    // must land in exactly the bucket the hart was leaving. Ends with
    // sums that cover the full 0..500 wall-time window.
    let hart = make_hart_context();
    init_hart_bucket(hart, HartBucket::Kernel, 0);

    switch_bucket(hart, HartBucket::User, 100); // K → U: kernel +100
    switch_bucket(hart, HartBucket::Kernel, 250); // U → K: user +150
    switch_bucket(hart, HartBucket::Scheduler, 300); // K → S: kernel +50
    switch_bucket(hart, HartBucket::Kernel, 320); // S → K: scheduler +20
    switch_bucket(hart, HartBucket::Idle, 400); // K → I: kernel +80
    switch_bucket(hart, HartBucket::Kernel, 500); // I → K: idle +100

    assert_eq!(hart.kernel_ticks.load(Ordering::Relaxed), 100 + 50 + 80);
    assert_eq!(hart.user_ticks.load(Ordering::Relaxed), 150);
    assert_eq!(hart.scheduler_ticks.load(Ordering::Relaxed), 20);
    assert_eq!(hart.idle_ticks.load(Ordering::Relaxed), 100);

    let total = hart.kernel_ticks.load(Ordering::Relaxed)
        + hart.user_ticks.load(Ordering::Relaxed)
        + hart.scheduler_ticks.load(Ordering::Relaxed)
        + hart.idle_ticks.load(Ordering::Relaxed);
    // Time spent inside the FINAL bucket (Kernel after the last call)
    // hasn't been credited yet — that happens on the next transition.
    // Pre-final-credit sum equals 500.
    assert_eq!(total, 500);
}

#[test]
fn switch_bucket_to_same_bucket_credits_anyway() {
    // Kernel → Kernel still credits elapsed to kernel and resets the
    // start tick. Treats no-op transitions as "I observed wall time
    // pass" rather than collapsing them. Useful when callers want to
    // re-stamp the baseline without a real bucket change.
    let hart = make_hart_context();
    init_hart_bucket(hart, HartBucket::Kernel, 0);

    switch_bucket(hart, HartBucket::Kernel, 200);
    assert_eq!(hart.kernel_ticks.load(Ordering::Relaxed), 200);
    assert_eq!(hart.bucket_enter_tick.load(Ordering::Relaxed), 200);
}

#[test]
fn user_exit_credits_current_thread() {
    let hart = make_hart_context();
    let thread = Box::leak(Box::new(make_thread(ThreadState::Running, SPP::User)));

    // Simulate the scheduler stamping `current` before the thread runs.
    hart.current
        .store(thread as *const _ as *mut _, Ordering::Release);

    init_hart_bucket(hart, HartBucket::User, 1000);
    switch_bucket(hart, HartBucket::Kernel, 1750);

    // 750 ticks of user time should land in `hart.user_ticks` AND in
    // `thread.cpu_ticks_total` — they're two views of the same wall
    // time, summed for different consumers.
    assert_eq!(hart.user_ticks.load(Ordering::Relaxed), 750);
    assert_eq!(thread.cpu_ticks_total.load(Ordering::Relaxed), 750);
}

#[test]
fn kernel_exit_does_not_credit_thread() {
    let hart = make_hart_context();
    let thread = Box::leak(Box::new(make_thread(ThreadState::Running, SPP::Supervisor)));
    hart.current
        .store(thread as *const _ as *mut _, Ordering::Release);

    init_hart_bucket(hart, HartBucket::Kernel, 0);
    switch_bucket(hart, HartBucket::Idle, 1000);

    assert_eq!(hart.kernel_ticks.load(Ordering::Relaxed), 1000);
    // Per-thread credit only fires on User→X transitions.
    assert_eq!(thread.cpu_ticks_total.load(Ordering::Relaxed), 0);
}

#[test]
fn user_exit_with_null_current_does_not_panic() {
    // Edge case: a hart parks its current thread (Suspended path nulls
    // hart.current first) and then a trap fires. switch_bucket sees
    // null current and must not deref it.
    let hart = make_hart_context();
    init_hart_bucket(hart, HartBucket::User, 0);

    switch_bucket(hart, HartBucket::Kernel, 500);

    assert_eq!(hart.user_ticks.load(Ordering::Relaxed), 500);
    // No thread to credit, no panic.
}

#[test]
fn user_exit_credit_accumulates_across_quanta() {
    // A thread runs in user mode multiple times across its lifetime —
    // each quantum's elapsed must add to cpu_ticks_total, not
    // overwrite it.
    let hart = make_hart_context();
    let thread = Box::leak(Box::new(make_thread(ThreadState::Running, SPP::User)));
    hart.current
        .store(thread as *const _ as *mut _, Ordering::Release);

    init_hart_bucket(hart, HartBucket::User, 0);
    switch_bucket(hart, HartBucket::Kernel, 100); // +100 user
    switch_bucket(hart, HartBucket::User, 200);
    switch_bucket(hart, HartBucket::Kernel, 350); // +150 user
    switch_bucket(hart, HartBucket::User, 400);
    switch_bucket(hart, HartBucket::Idle, 500); // +100 user

    assert_eq!(
        thread.cpu_ticks_total.load(Ordering::Relaxed),
        100 + 150 + 100
    );
    assert_eq!(hart.user_ticks.load(Ordering::Relaxed), 100 + 150 + 100);
}

#[test]
fn init_hart_bucket_does_not_credit() {
    // init is "publish a fresh baseline"; nothing accumulates until
    // the first switch.
    let hart = make_hart_context();
    init_hart_bucket(hart, HartBucket::Kernel, 5000);

    assert_eq!(hart.kernel_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(hart.user_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(hart.scheduler_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(hart.idle_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(hart.bucket_enter_tick.load(Ordering::Relaxed), 5000);
    assert_eq!(
        hart.current_bucket.load(Ordering::Relaxed),
        HartBucket::Kernel as u8
    );
}

#[test]
fn hart_bucket_from_u8_round_trips() {
    for b in [
        HartBucket::User,
        HartBucket::Kernel,
        HartBucket::Scheduler,
        HartBucket::Idle,
    ] {
        assert_eq!(HartBucket::from_u8(b as u8), b);
    }
    // Unknown values fall through to Idle (defensive — only an owning
    // hart writes the field, but the from_u8 must total).
    assert_eq!(HartBucket::from_u8(99), HartBucket::Idle);
}

// ───── record_syscall ──────────────────────────────────────────────

#[test]
fn record_syscall_bumps_thread_and_optional_slot() {
    let thread = make_thread(ThreadState::Running, SPP::User);
    let slot = SyscallSlot::new();

    record_syscall(Some(&slot), &thread, 1000, 1300);

    assert_eq!(slot.count.load(Ordering::Relaxed), 1);
    assert_eq!(slot.total_ticks.load(Ordering::Relaxed), 300);
    assert_eq!(thread.syscall_count.load(Ordering::Relaxed), 1);
    assert_eq!(thread.syscall_ticks.load(Ordering::Relaxed), 300);
}

#[test]
fn record_syscall_no_slot_only_credits_thread() {
    // kmain passes `None` when the syscall number isn't recognized
    // by `Sysno::from_usize` — keeps unknown sysnos out of the dense
    // ordinal histogram while still attributing service time to the
    // calling thread.
    let thread = make_thread(ThreadState::Running, SPP::User);

    record_syscall(None, &thread, 1000, 1300);

    assert_eq!(thread.syscall_count.load(Ordering::Relaxed), 1);
    assert_eq!(thread.syscall_ticks.load(Ordering::Relaxed), 300);
}

#[test]
fn record_syscall_accumulates_across_calls() {
    let thread = make_thread(ThreadState::Running, SPP::User);
    let slot = SyscallSlot::new();

    record_syscall(Some(&slot), &thread, 1000, 1100);
    record_syscall(Some(&slot), &thread, 2000, 2050);
    record_syscall(Some(&slot), &thread, 3000, 3200);

    assert_eq!(slot.count.load(Ordering::Relaxed), 3);
    assert_eq!(slot.total_ticks.load(Ordering::Relaxed), 100 + 50 + 200);
    assert_eq!(thread.syscall_count.load(Ordering::Relaxed), 3);
    assert_eq!(thread.syscall_ticks.load(Ordering::Relaxed), 350);
}

#[test]
fn record_syscall_zero_elapsed_still_increments_count() {
    // start == end (e.g. a free-running counter that's slow on the
    // first read) should still count the call.
    let thread = make_thread(ThreadState::Running, SPP::User);
    let slot = SyscallSlot::new();

    record_syscall(Some(&slot), &thread, 7777, 7777);

    assert_eq!(slot.count.load(Ordering::Relaxed), 1);
    assert_eq!(slot.total_ticks.load(Ordering::Relaxed), 0);
    assert_eq!(thread.syscall_count.load(Ordering::Relaxed), 1);
    assert_eq!(thread.syscall_ticks.load(Ordering::Relaxed), 0);
}

#[test]
fn record_syscall_distinct_threads_distinct_accumulators() {
    // Two threads, one shared slot. Per-thread fields stay segregated
    // even though the histogram aggregates.
    let t1 = make_thread(ThreadState::Running, SPP::User);
    let t2 = make_thread(ThreadState::Running, SPP::User);
    let slot = SyscallSlot::new();

    record_syscall(Some(&slot), &t1, 0, 50);
    record_syscall(Some(&slot), &t2, 100, 130);
    record_syscall(Some(&slot), &t1, 200, 280);

    assert_eq!(slot.count.load(Ordering::Relaxed), 3);
    assert_eq!(slot.total_ticks.load(Ordering::Relaxed), 50 + 30 + 80);

    assert_eq!(t1.syscall_count.load(Ordering::Relaxed), 2);
    assert_eq!(t1.syscall_ticks.load(Ordering::Relaxed), 50 + 80);
    assert_eq!(t2.syscall_count.load(Ordering::Relaxed), 1);
    assert_eq!(t2.syscall_ticks.load(Ordering::Relaxed), 30);
}
