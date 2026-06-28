//! Per-process stdin state: lock-free byte ring + parked-reader slot.
//!
//! One [`ProcessStdin`] per pid, reached via an `Arc<ProcessStdin>` in
//! the kmain-side `STDIN_TABLE` registry. Producer is
//! `input::dispatch` (PLIC trap context, exactly one hart at a time
//! per the IRQ claim/complete protocol). Consumer is the
//! `read_stdin` syscall handler on the user's hart.
//!
//! No locks: the byte ring is a [`SpscQueue`] (atomic head/tail), and
//! the parked-reader slot is an [`AtomicU32`] holding the parker's
//! `tid` (`0` means no parker). Pre-Phase-6 this was an
//! `AtomicPtr<CompletionInner>`; the migration to the on-thread
//! completion path eliminates the per-park Arc allocation.
//!
//! # Park-vs-signal race
//!
//! The lock-free design opens a classic park-vs-signal window. Reader
//! must `try_drain → park → re-drain` and undo the park if the
//! re-drain finds bytes; otherwise a producer that pushed between the
//! initial try and the park observed `parked_tid == 0` and won't
//! notify anyone. Producers should call [`push_byte`] which atomically
//! enqueues then takes-and-returns any parked tid in one operation;
//! the caller (kmain's `input::dispatch`) issues
//! `WAKE_QUEUE.push(WakeEvent::InputTid(tid))`.

use core::sync::atomic::{AtomicU32, Ordering};

use alloc::sync::Arc;

use crate::SpscQueue;

/// Bytes the reader can buffer before producer pushes start dropping.
/// One screen of typeahead is plenty for an interactive shell;
/// further buffering past 4 KiB would buy nothing for a human typist
/// and just defer the discard.
pub const RING_CAP: usize = 4096;

/// `SpscQueue` reserves one slot to keep `head == tail` unambiguous,
/// so the backing array needs `RING_CAP + 1`.
const RING_N: usize = RING_CAP + 1;

/// Sentinel for "no parked reader." Tids are non-zero in practice
/// (tids are allocated from 1 by the kernel's `next_tid`),
/// so 0 is safe as the empty-slot value.
const NO_PARKER: u32 = 0;

pub struct ProcessStdin {
    /// SPSC byte ring. Lock-free; producer is `input::dispatch`,
    /// consumer is the `read_stdin` handler.
    ring: SpscQueue<u8, RING_N>,
    /// Tid of the parked reader, or [`NO_PARKER`] (`0`) when no one
    /// is parked. Producer's `push_byte` atomically enqueues a byte
    /// and swaps this slot to `NO_PARKER`, returning the previous
    /// tid (if any) so the caller can issue a `WakeEvent::InputTid`.
    parked_tid: AtomicU32,
}

impl ProcessStdin {
    /// Allocate a fresh, empty stdin slot.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            // Zero-init is a valid SpscQueue starting state
            // (head == tail == 0 means empty).
            ring: unsafe { core::mem::zeroed() },
            parked_tid: AtomicU32::new(NO_PARKER),
        })
    }

    /// Producer side. Enqueue one byte and take any parked reader's
    /// tid in one operation. Returns:
    /// - `Ok(None)` — byte enqueued, no parker.
    /// - `Ok(Some(tid))` — byte enqueued and a parker was woken;
    ///   caller is responsible for issuing the wake (typically
    ///   `WAKE_QUEUE.push(WakeEvent::InputTid(tid))`).
    /// - `Err(())` — ring was full, byte dropped. Caller should bail
    ///   on any outer byte loop rather than pushing more (subsequent
    ///   bytes would also be dropped until the consumer drains).
    ///
    /// Trap-safe: atomic enqueue + atomic swap, no allocations, no
    /// locks.
    pub fn push_byte(&self, b: u8) -> Result<Option<u32>, ()> {
        // SAFETY: caller is the sole producer per the SPSC contract
        // documented on this struct.
        unsafe { self.ring.enqueue(b) }.map_err(|_| ())?;

        let prev = self.parked_tid.swap(NO_PARKER, Ordering::AcqRel);
        Ok(if prev == NO_PARKER { None } else { Some(prev) })
    }

    /// Consumer side. Drain up to `out.len()` bytes. Returns the
    /// count actually drained (may be 0 if the ring is empty).
    pub fn try_drain(&self, out: &mut [u8]) -> usize {
        let mut n = 0;
        while n < out.len() {
            // SAFETY: caller is the sole consumer per the SPSC
            // contract.
            let Some(b) = (unsafe { self.ring.dequeue() })
            else {
                break;
            };
            out[n] = b;
            n += 1;
        }
        n
    }

    /// Park the caller's tid. Returns `false` if a reader was already
    /// parked — the single-reader invariant is the caller's
    /// responsibility, so this signals a logic bug.
    ///
    /// Caller must pass a non-zero tid; `0` would collide with the
    /// `NO_PARKER` sentinel and is rejected via the CAS.
    pub fn park(&self, tid: u32) -> bool {
        if tid == NO_PARKER {
            return false;
        }
        self.parked_tid
            .compare_exchange(NO_PARKER, tid, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Take whatever tid is parked, if any. Used by the reader's
    /// re-check path (cancel a park when bytes arrived during the
    /// window) and by `dealloc_process` to wake a parked reader on
    /// teardown.
    pub fn unpark(&self) -> Option<u32> {
        let prev = self.parked_tid.swap(NO_PARKER, Ordering::AcqRel);
        if prev == NO_PARKER { None } else { Some(prev) }
    }

    /// True if there are no bytes in the ring. Useful for diagnostics
    /// and the read_stdin nonblock-check.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

// No `Drop` impl needed: the parked slot is just a `u32`. Pre-Phase-6
// we reclaimed the parked Arc here so it didn't leak when the owning
// process died with a reader parked; with the tid scheme the dying
// thread's struct gets reaped via `dealloc_thread` independently, and
// stale tids in `parked_tid` never get dereferenced (the lookup in
// `Orbit::publish_pending_for_tid` returns silently on a missing
// entry).
