//! Tests for [`net_channel::SpscQueue`] — the masked-index copy.
//!
//! This is the copy that lives in user-RW shared memory on the
//! NetChannel path: its `#[repr(C)]` layout is user/kernel ABI, and
//! its index loads are defensively masked (`% N`) because either
//! index can be scribbled by a misbehaving user process. Three things
//! get pinned here:
//!
//! 1. Single-threaded ring semantics — capacity `N - 1`, FIFO order
//!    across wraparound, `len`/`is_empty`/`is_full` bookkeeping, the
//!    cooperative `reset_*` helpers.
//! 2. Concurrent producer/consumer correctness — no loss, no
//!    duplication, order preserved. This is the test miri's seed and
//!    preemption fuzzing is aimed at (`./test miri-hammer` runs this
//!    file): a too-weak ordering on the head/tail Acquire/Release
//!    pairs shows up as a stale or torn slot read under some
//!    interleaving.
//! 3. Corrupted-index containment — out-of-range head/tail values
//!    must degrade to garbage *within* the ring, never an
//!    out-of-bounds slot access (which would panic the kernel), and
//!    the cooperative resets must restore a working queue.
//!
//! Iteration counts are `cfg(miri)`-aware (mirrors `process`'s
//! stdin.rs harness) so miri-many-seeds runs finish in seconds.

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use net_channel::SpscQueue;

#[cfg(miri)]
const ITERS: usize = 4;
#[cfg(not(miri))]
const ITERS: usize = 200;

#[cfg(miri)]
const ITEMS: usize = 64;
#[cfg(not(miri))]
const ITEMS: usize = 1024;

/// Deliberately tiny ring so the concurrent test wraps the indices
/// many times per iteration — wraparound under contention is where an
/// off-by-one in the full/empty math would surface.
const N: usize = 8;

fn fresh<T: Copy, const M: usize>() -> SpscQueue<T, M> {
    // Zero-init is a valid starting state (head == tail == 0 means
    // empty); same construction the kernel uses for in-place rings.
    unsafe { core::mem::zeroed() }
}

// ---- single-threaded ring semantics ----

#[test]
fn capacity_is_n_minus_one() {
    let q: SpscQueue<u32, N> = fresh();
    for i in 0..(N - 1) as u32 {
        assert!(unsafe { q.enqueue(i) }.is_ok(), "slot {i} should fit");
    }
    assert!(q.is_full());
    assert_eq!(q.len(), N - 1);
    // The reserved slot: enqueue at capacity hands the value back.
    assert_eq!(unsafe { q.enqueue(99) }, Err(99));
}

#[test]
fn fifo_order_across_wraparound() {
    let q: SpscQueue<u32, N> = fresh();
    // 4 * N items through an N-slot ring => several full wraps.
    let mut next_out = 0u32;
    for i in 0..(4 * N) as u32 {
        assert!(unsafe { q.enqueue(i) }.is_ok());
        if q.len() >= N / 2 {
            while let Some(v) = unsafe { q.dequeue() } {
                assert_eq!(v, next_out, "FIFO order broken");
                next_out += 1;
            }
        }
    }
    while let Some(v) = unsafe { q.dequeue() } {
        assert_eq!(v, next_out);
        next_out += 1;
    }
    assert_eq!(next_out, (4 * N) as u32, "items lost");
    assert!(q.is_empty());
    assert_eq!(q.len(), 0);
}

#[test]
fn dequeue_empty_returns_none() {
    let q: SpscQueue<u32, N> = fresh();
    assert!(q.is_empty());
    assert!(!q.is_full());
    assert_eq!(unsafe { q.dequeue() }, None);
}

#[test]
fn len_tracks_enqueue_dequeue() {
    let q: SpscQueue<u32, N> = fresh();
    for i in 0..3 {
        assert!(unsafe { q.enqueue(i) }.is_ok());
        assert_eq!(q.len(), (i + 1) as usize);
    }
    assert_eq!(unsafe { q.dequeue() }, Some(0));
    assert_eq!(q.len(), 2);
}

#[test]
fn cooperative_reset_restores_empty() {
    let q: SpscQueue<u32, N> = fresh();
    for i in 0..5 {
        let _ = unsafe { q.enqueue(i) };
    }
    let _ = unsafe { q.dequeue() };
    // Both sides reset their own index (NetChannel reuse protocol).
    unsafe {
        q.reset_producer();
        q.reset_consumer();
    }
    assert!(q.is_empty());
    assert_eq!(q.len(), 0);
    // And the queue still works.
    assert!(unsafe { q.enqueue(7) }.is_ok());
    assert_eq!(unsafe { q.dequeue() }, Some(7));
}

// ---- concurrent producer/consumer ----

/// One producer thread, one consumer thread, ITEMS sequence numbers
/// through an 8-slot ring. Asserts no loss, no duplication, and FIFO
/// order — under miri-hammer's seed + preemption fuzzing this is the
/// check that the Relaxed-own-index / Acquire-other-index /
/// Release-publish protocol is actually sufficient.
#[test]
fn concurrent_fifo_no_loss() {
    for iter in 0..ITERS {
        let q: Arc<SpscQueue<u32, N>> = Arc::new(fresh());
        let barrier = Arc::new(Barrier::new(2));

        let prod = {
            let q = Arc::clone(&q);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for i in 0..ITEMS as u32 {
                    // SAFETY: this thread is the sole producer.
                    while unsafe { q.enqueue(i) }.is_err() {
                        thread::yield_now(); // ring full — consumer will drain
                    }
                }
            })
        };

        let cons = {
            let q = Arc::clone(&q);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                let mut got = Vec::with_capacity(ITEMS);
                while got.len() < ITEMS {
                    // SAFETY: this thread is the sole consumer.
                    match unsafe { q.dequeue() } {
                        Some(v) => got.push(v),
                        None => thread::yield_now(),
                    }
                }
                got
            })
        };

        prod.join().unwrap();
        let got = cons.join().unwrap();

        assert_eq!(got.len(), ITEMS, "iter {iter}: lost items");
        for (i, &v) in got.iter().enumerate() {
            assert_eq!(v, i as u32, "iter {iter}: order broken at {i}");
        }
    }
}

// ---- corrupted-index containment ----

/// Reach the private head/tail atomics through the frozen `#[repr(C)]`
/// layout: head at offset 0, tail one usize after. This coupling is
/// deliberate — the layout is user/kernel ABI ("do not reorder
/// fields"), and the corruption scenario this test models is exactly
/// "the other side of the shared mapping wrote through that layout".
fn index_atomics<T: Copy, const M: usize>(q: &SpscQueue<T, M>) -> (&AtomicUsize, &AtomicUsize) {
    let base = q as *const SpscQueue<T, M> as *const AtomicUsize;
    // SAFETY: repr(C) puts head then tail at the struct's start; both
    // are atomics, so shared-reference access is the normal API.
    unsafe { (&*base, &*base.add(1)) }
}

/// A user process can scribble *both* indices of a NetChannel ring.
/// The kernel-side contract is containment, not correctness: every
/// operation keeps indexing inside the ring (no panic, no OOB), and a
/// cooperative reset gets back to a working queue.
#[test]
fn scribbled_indices_are_contained() {
    let q: SpscQueue<u32, N> = fresh();
    for i in 0..4 {
        let _ = unsafe { q.enqueue(i) };
    }

    let (head, tail) = index_atomics(&q);
    for (h, t) in [
        (usize::MAX, usize::MAX - 5),
        (N * 1000 + 3, 1),
        (0, usize::MAX / 2),
    ] {
        head.store(h, Ordering::Release);
        tail.store(t, Ordering::Release);
        // Values out are garbage by contract; what matters is that
        // none of these panic or index out of bounds (miri would flag
        // an OOB slot access here even where a release build might
        // silently read a neighbor).
        let _ = q.len();
        let _ = q.is_empty();
        let _ = q.is_full();
        let _ = unsafe { q.dequeue() };
        let _ = unsafe { q.enqueue(0xDEAD) };
    }

    unsafe {
        q.reset_producer();
        q.reset_consumer();
    }
    assert!(q.is_empty());
    assert!(unsafe { q.enqueue(42) }.is_ok());
    assert_eq!(unsafe { q.dequeue() }, Some(42));
}
