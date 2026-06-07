//! End-to-end smoke for the EventFd doorbell pattern.
//!
//! Validates the full chain landed in Milestone A4 + the orbit-rt
//! `EventFd` wrapper:
//!
//! 1. `eventfd(2)` allocates a kernel-shared region and maps it into
//!    the calling process at the requested VA.
//! 2. The shared `count` field is visible from a sibling thread (so
//!    pure-memory `signal` works without a kernel round-trip).
//! 3. `wake_tid(target_tid)` actually drags a thread parked in
//!    `ch_yield` back to Ready within scheduler-dispatch latency.
//! 4. `EventFd::signal_waker(tid)` composes (1)+(3) so a selector-
//!    shaped wake fires correctly.
//!
//! ## Layout
//!
//! - Main thread: creates the EventFd, spawns the worker thread,
//!   waits for the worker to publish its tid, then signals.
//! - Worker thread: publishes its tid, enters a poll/park loop. On
//!   wake, checks the EventFd count via shared memory, consumes it,
//!   stamps the observed value into a shared atomic, and exits.
//!
//! `umode-event-fd-smoke: PASS` on success; `: FAIL ...` with the
//! failure reason otherwise. A host script greps for those markers.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use alloc::sync::Arc;
use core::fmt::Write;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use orbit_abi::{
    logln,
    user::{ConsoleWriter, create_thread_with_arg, exit, get_micros, gettid, sleep_ms},
};
use orbit_rt::event_fd::EventFd;

/// Worker publishes its tid here so main knows who to `wake_tid`.
/// Zero before the worker starts; non-zero once published.
static WORKER_TID: AtomicU32 = AtomicU32::new(0);

/// Worker stamps the count it observed (after `try_consume`) here so
/// main can verify the wake actually delivered the signal we sent
/// rather than e.g. a spurious one.
///
/// `u64::MAX` is the "not yet" sentinel — anything else means the
/// worker woke and consumed. Using `u64::MAX` (rather than 0) lets us
/// distinguish "worker hasn't run yet" from "worker observed a zero
/// count and shouldn't have."
static WORKER_OBSERVED: AtomicU64 = AtomicU64::new(u64::MAX);

/// Tracks how many `ch_yield` parks the worker accumulated before the
/// wake landed. With the doorbell working, this should be 1 (one
/// park, then immediate wake). If the worker is spurious-waking or
/// the wake is being missed and recovered via timeout, the count
/// will be higher.
static WORKER_PARK_COUNT: AtomicU32 = AtomicU32::new(0);

/// Wait until `pred()` returns true or `deadline_us` micros have
/// elapsed since `start_us`. Returns `true` if `pred` succeeded
/// before the deadline. Polls with brief sleeps so we don't busy-spin
/// the manager.
fn wait_until(
    start_us: u64,
    deadline_us: u64,
    poll_ms: usize,
    mut pred: impl FnMut() -> bool,
) -> bool {
    loop {
        if pred() {
            return true;
        }
        let now = get_micros();
        if now.saturating_sub(start_us) >= deadline_us {
            return false;
        }
        let _ = sleep_ms(poll_ms);
    }
}

extern "C" fn worker_entry(region_va: usize) -> ! {
    // Publish our tid first so main knows who to wake. Release pairs
    // with main's Acquire load.
    let my_tid = gettid();
    WORKER_TID.store(my_tid, Ordering::Release);

    // `region_va` was packed into the spawn syscall's `arg` slot —
    // kernel-write into our trap frame's a0 before sret. Cast back to
    // the shared header.
    let region = region_va as *const orbit_abi::event_fd::EventFd;
    if region.is_null() {
        // Shouldn't happen — main always passes a non-null pointer.
        exit(2);
    }

    // Poll/park loop. Read the count via shared memory (no syscall);
    // if non-zero, claim it and exit. Otherwise park via ch_yield
    // for up to 5s — the doorbell from main's `wake_tid` should
    // return us early.
    const POLL_BUDGET: u32 = 50;
    const PARK_TIMEOUT_MS: usize = 5_000;

    for _ in 0..POLL_BUDGET {
        let count = unsafe { (*region).count.swap(0, Ordering::AcqRel) };
        if count > 0 {
            WORKER_OBSERVED.store(count, Ordering::Release);
            exit(0);
        }
        WORKER_PARK_COUNT.fetch_add(1, Ordering::Relaxed);
        let _ = orbit_abi::user::ch_yield(PARK_TIMEOUT_MS);
    }

    // Exhausted budget without observing a signal. Leave OBSERVED at
    // the sentinel so main reports FAIL.
    exit(1);
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    logln!("umode-event-fd-smoke: starting (main tid={})", gettid());

    // §1. Allocate the EventFd. `Arc` so Drop is deferred past the
    // worker's lifetime — the worker exits before main does, but
    // we shouldn't rely on that ordering.
    let efd = match EventFd::create(0, 0) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            let mut w = ConsoleWriter::new();
            let _ = writeln!(
                w,
                "umode-event-fd-smoke: FAIL EventFd::create errno={}",
                e.0
            );
            w.flush();
            return 1;
        }
    };

    // §2. Spawn the worker, packing the EventFd region pointer into
    // the spawn `arg` slot — the kernel writes it into the new
    // thread's a0 before sret, so `worker_entry` receives it as its
    // first C-ABI argument. allowed_affinity=0/affinity=0 inherit
    // main's mask — the scheduler can pick any hart.
    let region_va = efd.region_ptr().as_ptr() as usize;
    let worker_tid = match create_thread_with_arg(worker_entry, region_va, 0, 0) {
        Ok(t) => t,
        Err(e) => {
            let mut w = ConsoleWriter::new();
            let _ = writeln!(w, "umode-event-fd-smoke: FAIL create_thread errno={}", e.0);
            w.flush();
            return 1;
        }
    };

    // §4. Wait for the worker to publish its own tid (sanity-check
    // that the spawn actually landed and we're seeing cross-thread
    // memory visibility on the published value).
    let t_start = get_micros();
    let published = wait_until(t_start, 1_000_000, 5, || {
        WORKER_TID.load(Ordering::Acquire) != 0
    });
    if !published {
        logln!("umode-event-fd-smoke: FAIL worker tid never published");
        return 1;
    }
    let published_tid = WORKER_TID.load(Ordering::Acquire);
    if published_tid != worker_tid {
        logln!(
            "umode-event-fd-smoke: FAIL tid mismatch syscall={} self-published={}",
            worker_tid,
            published_tid
        );
        return 1;
    }
    logln!(
        "umode-event-fd-smoke: worker tid={} published in {} us",
        published_tid,
        get_micros().saturating_sub(t_start)
    );

    // §5. Give the worker a slice so it actually falls into
    // `ch_yield`. Without this we'd race the worker's loop and the
    // doorbell might land while the worker is still on the polling
    // pass, defeating the test of the wake_tid path.
    let _ = sleep_ms(50);

    // §6. Fire the doorbell. `signal_waker` bumps count + issues
    // wake_tid(worker_tid).
    let signal_at = get_micros();
    efd.signal_waker(worker_tid);

    // §7. Wait for the worker to consume + exit. Bounded at 2s — the
    // doorbell should resolve in milliseconds; anything close to the
    // bound is a sign wake_tid isn't reaching the parked thread.
    let woken = wait_until(get_micros(), 2_000_000, 5, || {
        WORKER_OBSERVED.load(Ordering::Acquire) != u64::MAX
    });
    if !woken {
        let parks = WORKER_PARK_COUNT.load(Ordering::Acquire);
        logln!("umode-event-fd-smoke: FAIL worker did not wake within 2s (parks={parks})");
        return 1;
    }

    let observed = WORKER_OBSERVED.load(Ordering::Acquire);
    let latency_us = get_micros().saturating_sub(signal_at);
    let parks = WORKER_PARK_COUNT.load(Ordering::Acquire);

    if observed != 1 {
        logln!(
            "umode-event-fd-smoke: FAIL observed count={} (wanted 1)",
            observed
        );
        return 1;
    }

    logln!(
        "umode-event-fd-smoke: PASS wake_latency_us={} worker_parks={}",
        latency_us,
        parks,
    );
    0
}

#[panic_handler]
fn panic(p: &PanicInfo) -> ! {
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "umode-event-fd-smoke panic: {p}");
    w.flush();
    exit(isize::MIN);
}
