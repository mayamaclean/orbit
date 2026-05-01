//! Per-process stdin state: lock-free byte ring + parked-reader slot.
//!
//! One [`ProcessStdin`] per pid, reached via an `Arc<ProcessStdin>` in
//! the kmain-side `STDIN_TABLE` registry. Producer is
//! `input::dispatch` (PLIC trap context, exactly one hart at a time
//! per the IRQ claim/complete protocol). Consumer is the
//! `read_stdin` syscall handler on the user's hart.
//!
//! No locks: the byte ring is a [`SpscQueue`] (atomic head/tail), and
//! the parked-reader slot is an [`AtomicPtr<CompletionInner>`] holding
//! `Arc::into_raw` of the parked handle.
//!
//! # Park-vs-signal race
//!
//! The lock-free design opens a classic park-vs-signal window. Reader
//! must `try_drain → park → re-drain` and undo the park if the
//! re-drain finds bytes; otherwise a producer that pushed between the
//! initial try and the park observed `parked == null` and won't
//! signal anyone. Producers should call [`push_byte`] which atomically
//! enqueues then takes-and-signals any parked handle in one operation.

use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use alloc::sync::Arc;

use crate::completion::CompletionInner;
use crate::{CompletionHandle, SpscQueue};

/// Bytes the reader can buffer before producer pushes start dropping.
/// One screen of typeahead is plenty for an interactive shell;
/// further buffering past 4 KiB would buy nothing for a human typist
/// and just defer the discard.
pub const RING_CAP: usize = 4096;

/// `SpscQueue` reserves one slot to keep `head == tail` unambiguous,
/// so the backing array needs `RING_CAP + 1`.
const RING_N: usize = RING_CAP + 1;

pub struct ProcessStdin {
    /// SPSC byte ring. Lock-free; producer is `input::dispatch`,
    /// consumer is the `read_stdin` handler.
    ring: SpscQueue<u8, RING_N>,
    /// Parked reader's `CompletionInner` Arc, encoded as a raw
    /// pointer. `null` = no parked reader. Encoding/decoding via
    /// [`CompletionHandle::into_raw`] / [`CompletionHandle::from_raw`].
    parked: AtomicPtr<CompletionInner>,
}

impl ProcessStdin {
    /// Allocate a fresh, empty stdin slot.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            // Zero-init is a valid SpscQueue starting state
            // (head == tail == 0 means empty).
            ring: unsafe { core::mem::zeroed() },
            parked: AtomicPtr::new(null_mut()),
        })
    }

    /// Producer side. Enqueue one byte (silently dropped on
    /// ring-full), then take-and-signal any parked reader. Safe to
    /// call from trap context: atomic enqueue + atomic swap +
    /// atomic signal — no allocations, no locks.
    pub fn push_byte(&self, b: u8) {
        // SAFETY: caller is the sole producer per the SPSC contract
        // documented on this struct.
        let _ = unsafe { self.ring.enqueue(b) };

        let raw = self.parked.swap(null_mut(), Ordering::AcqRel);
        if !raw.is_null() {
            // SAFETY: we hold the only pointer (we just swapped it
            // out atomically), so this `from_raw` reclaims the Arc
            // exactly once.
            let h = unsafe { CompletionHandle::from_raw(raw) };
            // Wake the reader without touching its a-regs (ret_count
            // = 0 leaves the parked thread's register snapshot
            // intact). Reader retries the syscall on resume.
            h.signal_n(&[]);
        }
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

    /// Park the caller's handle. Returns `Err(handle)` if a reader
    /// was already parked — the single-reader invariant is the
    /// caller's responsibility, so this signals a logic bug.
    pub fn park(&self, handle: CompletionHandle) -> Result<(), CompletionHandle> {
        let raw = handle.into_raw() as *mut CompletionInner;
        let prev =
            self.parked
                .compare_exchange(null_mut(), raw, Ordering::AcqRel, Ordering::Acquire);
        match prev {
            Ok(_) => Ok(()),
            Err(_) => {
                // CAS lost — another reader is parked. Reclaim our
                // own Arc and hand it back to the caller.
                let h = unsafe { CompletionHandle::from_raw(raw) };
                Err(h)
            }
        }
    }

    /// Take whatever handle is parked, if any. Used by the reader's
    /// re-check path (cancel a park when bytes arrived during the
    /// window) and by `dealloc_process` to wake a parked reader on
    /// teardown.
    pub fn unpark(&self) -> Option<CompletionHandle> {
        let raw = self.parked.swap(null_mut(), Ordering::AcqRel);
        if raw.is_null() {
            None
        }
        else {
            // SAFETY: atomic swap gives us sole ownership of the
            // raw pointer.
            Some(unsafe { CompletionHandle::from_raw(raw) })
        }
    }

    /// True if there are no bytes in the ring. Useful for diagnostics
    /// and the read_stdin nonblock-check.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

impl Drop for ProcessStdin {
    fn drop(&mut self) {
        // Reclaim any parked Arc so we don't leak it. The owning
        // process is gone; no one will ever signal this handle.
        let raw = self.parked.swap(null_mut(), Ordering::AcqRel);
        if !raw.is_null() {
            unsafe { drop(CompletionHandle::from_raw(raw)) };
        }
    }
}
