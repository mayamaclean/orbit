//! `sleep_ms` accuracy histogram.
//!
//! For each target duration in [`TARGETS`], call `sleep_ms(target)` in
//! a loop and record `(actual_us - target_us)` — the *overshoot*
//! beyond the requested deadline. Bucket the deltas into a small
//! fixed histogram and print one line per target so a host script can
//! grep + diff across kernel revisions.
//!
//! Pre-Phase D this would show a tail at the manager-loop heartbeat
//! (10 ms): a 1 ms sleep waited up to 10 ms because `setup_hart_timer`
//! armed unconditionally for that long. Post-Phase D the WFI cycles
//! are sized from `SleepHeap::next_wake`, so a 1 ms sleep should
//! wake within a sub-millisecond margin.
//!
//! Negative overshoot ("woke early") shouldn't happen — the kernel
//! only marks a sleeper Ready when `now >= wake_time`. If the
//! `under_target` count is non-zero something is computing the
//! deadline incorrectly.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use core::fmt::Write;
use core::panic::PanicInfo;

use orbit_abi::{logln, user::{exit, get_micros, sleep_ms, ConsoleWriter}};

/// Sleep durations to characterize, in milliseconds. 1ms is the
/// classic "sub-heartbeat" case; 10/100ms are above the heartbeat
/// and should overshoot by a much smaller relative margin.
const TARGETS_MS: &[usize] = &[1, 5, 10, 100];

/// Iterations per target. Smaller for longer sleeps to keep total
/// wallclock bounded — at 100ms × 100 iterations = ~10s per sweep.
fn iterations_for(target_ms: usize) -> usize {
    match target_ms {
        0..=2   => 500,
        3..=20  => 200,
        _       => 50,
    }
}

/// Bucket upper bounds in microseconds. Last bucket catches >= the
/// final value. Tuned to the 10ms heartbeat: a regression to the
/// pre-Phase D behavior would dump 1ms-target samples into the
/// `<10000us` bucket; healthy post-Phase D should land in the
/// first two.
const BUCKET_BOUNDS_US: &[u64] = &[100, 500, 1_000, 2_000, 5_000, 10_000];

/// Number of histogram cells (one per bound + a "≥last" overflow).
const NUM_BUCKETS: usize = 7;

#[derive(Default, Clone, Copy)]
struct Stats {
    n: u64,
    sum_us: u64,
    min_us: u64,
    max_us: u64,
    under_target: u64,
    buckets: [u64; NUM_BUCKETS],
}

impl Stats {
    fn record(&mut self, overshoot_us: i64) {
        self.n += 1;
        if overshoot_us < 0 {
            self.under_target += 1;
            return;
        }
        let v = overshoot_us as u64;
        self.sum_us += v;
        if self.n == 1 || v < self.min_us { self.min_us = v; }
        if v > self.max_us { self.max_us = v; }
        for (i, &bound) in BUCKET_BOUNDS_US.iter().enumerate() {
            if v < bound {
                self.buckets[i] += 1;
                return;
            }
        }
        self.buckets[NUM_BUCKETS - 1] += 1;
    }

    fn mean_us(&self) -> u64 {
        let valid = self.n.saturating_sub(self.under_target);
        if valid == 0 { 0 } else { self.sum_us / valid }
    }
}

fn run_sweep(target_ms: usize) -> Stats {
    let target_us = (target_ms as u64) * 1_000;
    let iters = iterations_for(target_ms);
    let mut s = Stats::default();
    for _ in 0..iters {
        let t0 = get_micros();
        let _ = sleep_ms(target_ms);
        let t1 = get_micros();
        let actual = t1.saturating_sub(t0);
        let overshoot = actual as i64 - target_us as i64;
        s.record(overshoot);
    }
    s
}

fn print_stats(target_ms: usize, s: &Stats) {
    let mut w = ConsoleWriter::new();
    let _ = write!(
        w,
        "SLEEP target_ms={} n={} under={} min_us={} mean_us={} max_us={} buckets=[",
        target_ms, s.n, s.under_target, s.min_us, s.mean_us(), s.max_us,
    );
    // Bucket label format: "<100=N <500=N <1000=N ... >=10000=N".
    for (i, &bound) in BUCKET_BOUNDS_US.iter().enumerate() {
        if i > 0 { let _ = write!(w, " "); }
        let _ = write!(w, "<{}us={}", bound, s.buckets[i]);
    }
    let _ = write!(w, " >={}us={}", BUCKET_BOUNDS_US[BUCKET_BOUNDS_US.len() - 1],
                   s.buckets[NUM_BUCKETS - 1]);
    let _ = writeln!(w, "]");
    w.flush();
}

// orbit-rt's `_start` (§13b) is the canonical entrypoint; downstream
// binaries provide `main` and let orbit-rt do the eager argv resolve
// and exit. Defining `_start` here would collide with the rt one and
// fail to link.
#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    logln!("umode-sleep-bench: starting");

    // Brief settle so any startup churn doesn't skew the first sample.
    let _ = sleep_ms(200);

    for &target_ms in TARGETS_MS {
        let s = run_sweep(target_ms);
        print_stats(target_ms, &s);
    }

    logln!("umode-sleep-bench: done");
    0
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "umode-sleep-bench panic: {p}");
    w.flush();
    exit(isize::MIN);
}
