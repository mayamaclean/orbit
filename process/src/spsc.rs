//! Single-producer / single-consumer ring queue.
//!
//! Lock-free via atomic head/tail + `[UnsafeCell<T>; N]` slots. The
//! producer owns `tail`, the consumer owns `head`; they synchronize
//! through Acquire/Release pairs on those atomics. Slots are read /
//! written through `read_volatile` / `write_volatile` to defend
//! against any alias-analysis-driven elision.
//!
//! Capacity is `N - 1` — one slot stays reserved so `head == tail`
//! unambiguously means empty.
//!
//! In-crate consumers:
//! - [`crate::stdin::ProcessStdin`] — per-process stdin byte ring
//!   (input::dispatch producer, read_stdin consumer).
//! - [`crate::key_events::ProcessKeyEvents`] — per-process key-event ring.
//!
//! `net_channel` keeps its own independent identical copy of this type
//! (its `#[repr(C)]` layout is part of the user/kernel ABI for that
//! path; do not reorder fields there).
//!
//! # Safety contract
//!
//! `enqueue` / `dequeue` / the `reset_*` helpers are `unsafe` because
//! they require the caller to be the *sole* producer or consumer.
//! Multi-producer or multi-consumer use is data-race UB.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

#[repr(C)]
pub struct SpscQueue<T: Copy, const N: usize> {
    /// Consumer-owned: index of next slot to dequeue from.
    head: AtomicUsize,
    /// Producer-owned: index of next slot to enqueue into.
    tail: AtomicUsize,
    /// Backing ring storage. Raw `UnsafeCell<T>` so the slots can be
    /// written under `&self` via `.get()` → `*mut T`. Zero-init is a
    /// valid starting state since `head == tail == 0` marks empty and no
    /// slot is observed before being written.
    buffer: [UnsafeCell<T>; N],
}

// Producer and consumer are on different harts / threads; heads/tails
// are atomic, slots are synchronized via release/acquire.
unsafe impl<T: Copy + Send, const N: usize> Sync for SpscQueue<T, N> {}

impl<T: Copy, const N: usize> SpscQueue<T, N> {
    /// # Safety
    /// Caller must be the sole producer on this queue.
    #[inline]
    pub unsafe fn enqueue(&self, val: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next = (tail + 1) % N;
        if next == self.head.load(Ordering::Acquire) {
            return Err(val);
        }
        unsafe {
            self.buffer[tail].get().write_volatile(val);
        }
        self.tail.store(next, Ordering::Release);
        Ok(())
    }

    /// # Safety
    /// Caller must be the sole consumer on this queue.
    #[inline]
    pub unsafe fn dequeue(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        if head == self.tail.load(Ordering::Acquire) {
            return None;
        }
        let val = unsafe { self.buffer[head].get().read_volatile() };
        self.head.store((head + 1) % N, Ordering::Release);
        Some(val)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail + 1) % N == head
    }

    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail + N - head) % N
    }

    /// Zero the producer-owned index. Used during cooperative resets
    /// (e.g. NetChannel reuse): both sides reset their own side so the
    /// next session starts with empty queues. `head` is left alone —
    /// the consumer clears it via [`reset_consumer`].
    ///
    /// # Safety
    /// Caller must be the sole producer on this queue, and must guarantee
    /// the consumer is not actively reading.
    #[inline]
    pub unsafe fn reset_producer(&self) {
        self.tail.store(0, Ordering::Release);
    }

    /// Zero the consumer-owned index. See [`reset_producer`] for the
    /// cooperative-reset safety requirements.
    ///
    /// # Safety
    /// Caller must be the sole consumer on this queue, and must guarantee
    /// the producer is not actively writing.
    #[inline]
    pub unsafe fn reset_consumer(&self) {
        self.head.store(0, Ordering::Release);
    }
}
