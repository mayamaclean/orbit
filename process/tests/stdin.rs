//! Concurrency stress for [`process::ProcessStdin`].
//!
//! ProcessStdin is lock-free: SPSC byte ring + AtomicPtr parked-reader
//! slot. The interesting hazards live in the park-vs-signal window:
//!
//! - Producer pushes a byte while the consumer is between
//!   `try_drain → 0` and `park`. Without the park-then-recheck
//!   pattern, the byte arrives but no one is parked to signal, so
//!   the reader sleeps forever. With re-check, the reader either
//!   sees the byte on re-drain (and cancels its park) or has parked
//!   in time to be signaled.
//!
//! - Two reclaims of the parked Arc would be use-after-free; miri
//!   catches that. Producer's `push_byte` does atomic-swap-then-
//!   reclaim; consumer's `unpark` and `Drop` both do the same.
//!   Whichever wins the swap reclaims; the loser sees null.
//!
//! Test harness mirrors `completion.rs`: spawn producer + consumer
//! threads behind a Barrier, push K bytes, drain K bytes, assert
//! count + ordering. Loop iterations cap is `cfg(miri)`-aware so
//! miri runs in seconds.

use std::sync::{Arc as StdArc, Barrier};
use std::thread;

use process::{CompletionHandle, ProcessStdin};

#[cfg(miri)]
const ITERS: usize = 5;
#[cfg(not(miri))]
const ITERS: usize = 200;

const BYTES_PER_ITER: usize = 32;

/// Producer pushes BYTES_PER_ITER bytes; consumer drains until it
/// has all of them. Reader uses the park-then-recheck pattern.
/// Asserts:
/// - No bytes lost (received.len() == BYTES_PER_ITER).
/// - SPSC preserves ordering (received[i] == i).
/// - No deadlock (test completes within thread::join, no infinite
///   wait on `is_signaled`).
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
                    s.push_byte(i as u8);
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
                    // Park-then-recheck. Either we observe bytes
                    // arrived during the window (cancel park, retry)
                    // or we wait on the handle for the next push.
                    let h = CompletionHandle::new();
                    s.park(h.clone()).expect("single-reader invariant");

                    let n2 = s.try_drain(&mut buf);
                    if n2 > 0 {
                        let _ = s.unpark();
                        received.extend_from_slice(&buf[..n2]);
                        continue;
                    }

                    // Wait for the producer to signal. In the
                    // kernel this is the manager-scan resuming a
                    // Blocking thread; in the host test we just
                    // poll the atomic state.
                    while !h.is_signaled() {
                        thread::yield_now();
                    }
                }
                received
            })
        };

        prod.join().unwrap();
        let received = cons.join().unwrap();

        assert_eq!(
            received.len(),
            BYTES_PER_ITER,
            "iter {iter}: lost bytes"
        );
        for (i, &b) in received.iter().enumerate() {
            assert_eq!(b, i as u8, "iter {iter}: SPSC order broken at {i}");
        }
    }
}

/// Park, then teardown the stdin entirely (Arc::drop) without ever
/// signaling. The Drop impl must reclaim the parked Arc so miri
/// doesn't flag a leak.
#[test]
fn drop_reclaims_parked_arc() {
    let stdin = ProcessStdin::new();
    let h = CompletionHandle::new();
    stdin.park(h).expect("park ok");
    drop(stdin); // ProcessStdin::Drop reclaims the parked Arc.
}

/// `unpark` returns the same handle that was parked, exactly once.
/// A second unpark sees the cleared slot.
#[test]
fn unpark_takes_once() {
    let stdin = ProcessStdin::new();
    let h = CompletionHandle::new();
    stdin.park(h.clone()).expect("park ok");
    let taken = stdin.unpark();
    assert!(taken.is_some());
    assert!(stdin.unpark().is_none(), "second unpark must see null slot");
    // `taken` and `h` are clones of the same Arc — signaling either
    // observes from both.
    taken.unwrap().signal_n(&[42]);
    assert!(h.is_signaled());
    assert_eq!(h.ret(0), 42);
}

/// Re-park after an unpark works (lifecycle: park → unpark → park).
#[test]
fn re_park_after_unpark() {
    let stdin = ProcessStdin::new();
    let h1 = CompletionHandle::new();
    stdin.park(h1).expect("first park");
    let _ = stdin.unpark();
    let h2 = CompletionHandle::new();
    stdin.park(h2).expect("second park");
}

/// Single-reader invariant: a second `park` while one is already
/// parked returns Err(handle) and leaves the original entry intact.
#[test]
fn double_park_errors() {
    let stdin = ProcessStdin::new();
    let h1 = CompletionHandle::new();
    stdin.park(h1.clone()).expect("first park");

    let h2 = CompletionHandle::new();
    let returned = stdin.park(h2.clone()).expect_err("must reject second park");

    // The returned handle is h2 (the rejected one); h1 stays parked.
    assert!(stdin.unpark().is_some(), "h1 should still be parked");
    // Drop returned handle cleanly.
    drop(returned);
    drop(h2);
}
