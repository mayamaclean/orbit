//! Concurrency stress for [`process::ProcessStdin`].
//!
//! ProcessStdin is lock-free: SPSC byte ring + AtomicU32 parked-tid
//! slot. The interesting hazards live in the park-vs-signal window:
//!
//! - Producer pushes a byte while the consumer is between
//!   `try_drain → 0` and `park`. Without the park-then-recheck
//!   pattern, the byte arrives but no one is parked to wake, so the
//!   reader sleeps forever. With re-check, the reader either sees the
//!   byte on re-drain (and cancels its park) or has parked in time
//!   for the producer's `push_byte` to capture its tid for wakeup.
//!
//! - In the kernel, the producer-side `push_byte` returns
//!   `Option<u32>`; on `Some(tid)` the trap-context caller pushes
//!   `WakeEvent::InputTid(tid)` onto WAKE_QUEUE. The host test
//!   stand-in is "the byte is now in the ring, so the consumer's
//!   next try_drain succeeds" — same correctness story.
//!
//! Test harness mirrors `completion.rs`: spawn producer + consumer
//! threads behind a Barrier, push K bytes, drain K bytes, assert
//! count + ordering. Loop iterations cap is `cfg(miri)`-aware so
//! miri runs in seconds.

use std::sync::{Arc as StdArc, Barrier};
use std::thread;

use process::ProcessStdin;

#[cfg(miri)]
const ITERS: usize = 5;
#[cfg(not(miri))]
const ITERS: usize = 200;

const BYTES_PER_ITER: usize = 32;

/// Synthetic tid for the host tests — any non-zero value works since
/// we don't dispatch through a real Thread registry.
const TID: u32 = 1;
const TID2: u32 = 2;

/// Producer pushes BYTES_PER_ITER bytes; consumer drains until it
/// has all of them. Reader uses the park-then-recheck pattern.
/// Asserts:
/// - No bytes lost (received.len() == BYTES_PER_ITER).
/// - SPSC preserves ordering (received[i] == i).
/// - No deadlock — without a real wake bus the test just polls the
///   ring after parking, which still exercises the
///   park-then-re-check sequence.
#[test]
fn no_byte_loss_under_concurrent_push_drain() {
    for iter in 0..ITERS {
        let stdin = ProcessStdin::new();
        let barrier = StdArc::new(Barrier::new(2));

        let prod = {
            let s = stdin.clone();
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                for i in 0..BYTES_PER_ITER {
                    // push_byte returns Ok(Some(tid)) if a reader
                    // was parked, Ok(None) if not, Err(()) if the
                    // ring was full. Host test ignores the wake
                    // signal (no real wake bus) and bails on full.
                    if s.push_byte(i as u8).is_err() {
                        break;
                    }
                }
            })
        };

        let cons = {
            let s = stdin.clone();
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                let mut received = Vec::new();
                while received.len() < BYTES_PER_ITER {
                    let mut buf = [0u8; 16];
                    let n = s.try_drain(&mut buf);
                    if n > 0 {
                        received.extend_from_slice(&buf[..n]);
                        continue;
                    }
                    // Park-then-recheck: stamp our tid, then re-drain
                    // to close the park-vs-push window.
                    assert!(s.park(TID), "single-reader invariant");
                    let n2 = s.try_drain(&mut buf);
                    if n2 > 0 {
                        let _ = s.unpark();
                        received.extend_from_slice(&buf[..n2]);
                        continue;
                    }

                    // Parked + re-drain saw nothing → wait for the
                    // producer to push. Without a real wake mechanism
                    // here, poll the ring and cancel our park when
                    // bytes arrive.
                    while s.is_empty() {
                        thread::yield_now();
                    }
                    let _ = s.unpark();
                }
                received
            })
        };

        prod.join().unwrap();
        let received = cons.join().unwrap();

        assert_eq!(received.len(), BYTES_PER_ITER, "iter {iter}: lost bytes");
        for (i, &b) in received.iter().enumerate() {
            assert_eq!(b, i as u8, "iter {iter}: SPSC order broken at {i}");
        }
    }
}

/// Park then drop the ProcessStdin without an unpark. Pre-Phase-6
/// this exercised the Drop impl that reclaimed the parked Arc; with
/// the tid scheme there's nothing to reclaim, so the test reduces to
/// "drop doesn't panic," but we keep it as a sanity check that the
/// lifecycle still composes.
#[test]
fn drop_with_tid_parked_does_not_panic() {
    let stdin = ProcessStdin::new();
    assert!(stdin.park(TID), "park ok");
    drop(stdin);
}

/// `unpark` returns the parked tid exactly once. A second unpark
/// sees the cleared slot.
#[test]
fn unpark_takes_once() {
    let stdin = ProcessStdin::new();
    assert!(stdin.park(TID), "park ok");
    let taken = stdin.unpark();
    assert_eq!(taken, Some(TID));
    assert!(
        stdin.unpark().is_none(),
        "second unpark must see empty slot"
    );
}

/// Re-park after an unpark works (lifecycle: park → unpark → park).
#[test]
fn re_park_after_unpark() {
    let stdin = ProcessStdin::new();
    assert!(stdin.park(TID), "first park");
    let _ = stdin.unpark();
    assert!(stdin.park(TID2), "second park");
}

/// Single-reader invariant: a second `park` while one is already
/// parked returns `false` and leaves the original entry intact.
#[test]
fn double_park_returns_false() {
    let stdin = ProcessStdin::new();
    assert!(stdin.park(TID), "first park ok");
    assert!(!stdin.park(TID2), "second park must be rejected");
    // First park's tid stays installed.
    assert_eq!(stdin.unpark(), Some(TID));
}

/// `push_byte` returns the parked tid exactly once: the byte is
/// delivered, the slot is cleared, and the caller (kmain's
/// `input::dispatch`) issues the wake.
#[test]
fn push_byte_returns_parked_tid_once() {
    let stdin = ProcessStdin::new();
    assert!(stdin.park(TID));
    assert_eq!(
        stdin.push_byte(b'a'),
        Ok(Some(TID)),
        "first push wakes parker"
    );
    assert_eq!(
        stdin.push_byte(b'b'),
        Ok(None),
        "second push has no parker (slot cleared on first)"
    );
}

/// Filling the ring past `RING_CAP` bytes returns `Err(())`. Callers
/// (kmain's `input::dispatch`) should bail on any outer byte loop
/// rather than continue pushing, since further bytes would also be
/// dropped until the consumer drains.
#[test]
fn push_byte_full_ring_returns_err() {
    let stdin = ProcessStdin::new();
    for i in 0..process::stdin::RING_CAP {
        assert!(stdin.push_byte((i & 0xFF) as u8).is_ok());
    }
    assert!(stdin.push_byte(0xFF).is_err(), "post-cap push must Err");
}

/// `park(0)` is rejected — `0` is the empty-slot sentinel and would
/// alias `NO_PARKER` in the AtomicU32. Real tids are non-zero in
/// practice (the boot thread starts at 1), so this is a defensive
/// assertion; misbehaving callers that try to park `0` get `false`
/// rather than poisoning the slot.
#[test]
fn park_zero_rejected() {
    let stdin = ProcessStdin::new();
    assert!(!stdin.park(0));
    // Slot stays empty: a real park still works afterward.
    assert!(stdin.park(TID));
}
