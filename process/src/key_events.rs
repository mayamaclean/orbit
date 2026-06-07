//! Per-process structured key-event ring + parked-reader tid.
//!
//! Sister of [`crate::stdin::ProcessStdin`] but built on a different
//! wake mechanism. Instead of an `AtomicPtr<CompletionInner>` Arc that
//! the producer consumes via `signal_n`, the parker stamps its tid
//! into [`Self::parked_tid`] and the producer pushes a
//! `WakeEvent::InputTid(tid)` into the kernel's `WAKE_QUEUE`. The
//! manager's `drain_wakes` then ORs `wake_reason::INPUT_IO` into the
//! parked thread's `wake_override`, which eagerly promotes the
//! Suspended thread back to Ready.
//!
//! Why this shape and not the handle one:
//!
//! - **Unified with the timeout path.** A reader can park with
//!   `wake_time = now + timeout_ms` and rely on the same wake_override
//!   bit for early wake. The handle path can't combine cleanly with a
//!   sleep_heap entry; this one does.
//! - **Same primitive as `ch_yield`.** That syscall already
//!   demonstrates the "ms_sleep + producer-side wake_override"
//!   pattern. Reusing it avoids accumulating wake mechanisms.
//! - **No allocation on the producer.** The handle path needs an Arc
//!   reclaim on each wake; pushing a `WakeEvent` enum variant is a
//!   thingbuf slot write — pure atomic.
//!
//! Producer ordering:
//!   1. `ring.enqueue(ev)` — events visible to consumer
//!   2. `parked_tid.swap(0)` — atomically claim the wake right
//!   3. `WAKE_QUEUE.push(WakeEvent::InputTid(tid))` — manager wakes
//!
//! Consumer ordering (in the syscall handler):
//!   1. `try_drain` — fast path
//!   2. `set_parker(tid)` (CAS 0 → tid) — claim the slot
//!   3. `try_drain` again — close the park-vs-push race
//!   4. If still empty, set `thread.wake_time` and yield Suspended
//!
//! On wake the syscall re-executes from step 1; producer has already
//! cleared parked_tid in its own step 2, so step 2 here CAS-loops
//! cleanly.

use core::sync::atomic::{AtomicU32, Ordering};

use alloc::sync::Arc;

use orbit_abi::input::KeyEvent;

use crate::SpscQueue;

/// Events the reader can buffer before producer pushes start dropping.
/// 256 events is several seconds of human-rate keystrokes; further
/// buffering past that helps no one (a TUI app would have processed by
/// then or doesn't care about ancient keys).
pub const RING_CAP: usize = 256;

/// `SpscQueue` reserves one slot to keep `head == tail` unambiguous,
/// so the backing array needs `RING_CAP + 1`.
const RING_N: usize = RING_CAP + 1;

/// Sentinel value for "no parked reader" — tid 0 is never allocated
/// (the kmain `next_tid` allocator starts at 1).
const TID_NONE: u32 = 0;

/// Result of a `set_parker` attempt. See `ProcessKeyEvents::set_parker`
/// for the three outcomes' meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParkOutcome {
    /// Slot was empty; our tid is now installed. First-time park.
    Installed,
    /// Slot already held our tid (timer-wake re-entry — no producer
    /// has cleared the slot since the previous park).
    AlreadyOwned,
    /// A different tid is parked. Single-reader contract violation;
    /// caller emits `EBUSY`.
    Busy,
}

pub struct ProcessKeyEvents {
    /// SPSC event ring. Lock-free; producer is `input::dispatch`,
    /// consumer is the `read_key_event` handler.
    ring: SpscQueue<KeyEvent, RING_N>,
    /// Tid of the thread parked on this ring, or `TID_NONE` (0) if
    /// unclaimed. Producer atomically swaps to `TID_NONE` on push;
    /// consumer atomically CASes from `TID_NONE` to its tid on park.
    parked_tid: AtomicU32,
}

impl ProcessKeyEvents {
    /// Allocate a fresh, empty event slot.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            // Zero-init is a valid SpscQueue starting state.
            ring: unsafe { core::mem::zeroed() },
            parked_tid: AtomicU32::new(TID_NONE),
        })
    }

    /// Producer side. Enqueue one event, then atomically claim the
    /// parked-reader's tid (if any). Returns `Some(tid)` when there
    /// was a parker — the caller is then responsible for issuing the
    /// kernel-side wake (`WAKE_QUEUE.push(WakeEvent::InputTid(tid))`).
    /// Returns `None` when no reader was parked, in which case the
    /// event sits in the ring until a future `read_key_event` drains
    /// it synchronously.
    ///
    /// Safe to call from trap context — only atomic ops.
    pub fn push_event(&self, ev: KeyEvent) -> Option<u32> {
        // SAFETY: caller is the sole producer per the SPSC contract.
        let _ = unsafe { self.ring.enqueue(ev) };
        let tid = self.parked_tid.swap(TID_NONE, Ordering::AcqRel);
        if tid == TID_NONE { None } else { Some(tid) }
    }

    /// Consumer side. Drain up to `out.len()` events.
    pub fn try_drain(&self, out: &mut [KeyEvent]) -> usize {
        let mut n = 0;
        while n < out.len() {
            // SAFETY: caller is the sole consumer.
            let Some(ev) = (unsafe { self.ring.dequeue() })
            else {
                break;
            };
            out[n] = ev;
            n += 1;
        }
        n
    }

    /// Consumer side. Stamp the calling thread's tid as the parker.
    ///
    /// The three outcomes encode what the read_key_event handler
    /// needs to know:
    /// - [`ParkOutcome::Installed`] — slot was empty, our tid is now
    ///   in. First-time park; caller may yield Suspended.
    /// - [`ParkOutcome::AlreadyOwned`] — slot already held our tid.
    ///   This is the timer-wake re-entry path: the deadline fired
    ///   without any producer push (which would have swapped the
    ///   slot to zero). A fresh `compare_exchange(0 → tid)` would
    ///   spuriously fail; we surface this distinct state instead so
    ///   the handler can return 0 to userspace (timeout) rather than
    ///   re-park indefinitely.
    /// - [`ParkOutcome::Busy`] — a *different* tid is parked. Real
    ///   single-reader violation; caller maps to `EBUSY`.
    pub fn set_parker(&self, tid: u32) -> ParkOutcome {
        debug_assert!(tid != TID_NONE, "set_parker(0) — TID_NONE is reserved");
        let current = self.parked_tid.load(Ordering::Acquire);
        if current == tid {
            return ParkOutcome::AlreadyOwned;
        }
        match self
            .parked_tid
            .compare_exchange(TID_NONE, tid, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => ParkOutcome::Installed,
            Err(_) => ParkOutcome::Busy,
        }
    }

    /// Consumer side. Clear the parker slot if it currently holds
    /// `tid`. Used by the read-side re-check race: after stamping our
    /// tid we re-drain; if events arrived during the window we cancel
    /// our park instead of yielding. CAS so a producer that already
    /// took our slot doesn't see us spuriously clear it.
    pub fn clear_parker_if(&self, tid: u32) -> bool {
        self.parked_tid
            .compare_exchange(tid, TID_NONE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Take whatever tid is parked, if any. Used by the registry's
    /// `unregister` path to wake a parked reader on process teardown
    /// — the caller then issues the kernel-side wake.
    pub fn take_parker(&self) -> Option<u32> {
        let tid = self.parked_tid.swap(TID_NONE, Ordering::AcqRel);
        if tid == TID_NONE { None } else { Some(tid) }
    }

    /// True if the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}
