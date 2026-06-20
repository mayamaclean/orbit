//! FIFO of runnable threads, popped by the scheduler with hart-affinity
//! filtering.
//!
//! Replaces the per-pass O(N_threads) registry walk in
//! `get_runnable_thread`. The queue holds only `*mut Thread` pointers
//! known to be in `ThreadState::Ready`; producers push when a thread
//! transitions into Ready, and the scheduler pops one per idle hart.
//!
//! ## Why not a strict ring?
//!
//! Affinity. A pinned thread sitting at the head of a strict FIFO would
//! block all unpinned threads behind it from being dispatched on harts
//! the pinned thread can't run on. Instead [`pop_for`] scans from the
//! head for the oldest entry whose `affinity & hart_mask != 0`, leaving
//! later-but-incompatible entries in place. In the common case (most
//! threads have wide affinity) the head matches and the scan is O(1).
//!
//! ## Single-consumer
//!
//! The queue lives in `Orbit` and is mutated only by the manager hart
//! while it holds `MANAGER_LOCK`. Producers from non-manager harts go
//! through per-hart `READY_INBOXES` (in kmain), which the manager
//! drains into this queue at the head of each `assign_threads` pass.
//! That keeps push paths lock-free and the queue itself a plain
//! `VecDeque` — no atomics, no contention concerns inside the type.

use alloc::collections::VecDeque;

use process::{Runnable, Thread};

pub struct ReadyQueue {
    inner: VecDeque<Runnable>,
}

impl ReadyQueue {
    pub const fn new() -> Self {
        Self {
            inner: VecDeque::new(),
        }
    }

    /// Append a runnable thread to the back of the queue. The
    /// [`Runnable`] token is the proof that the thread's frame was
    /// marshaled under a won claim (or is a fresh `Ready` thread) — so
    /// "enqueue for dispatch" is welded to "the frame is valid"
    /// (**bug 4**, by construction). The queue now *stores* the move-only
    /// token (not a bare `*mut Thread`), so the distinct-dispatch-right
    /// invariant is held by the type for as long as the thread is queued:
    /// a non-`Clone` `Runnable` can't be duplicated into two slots, and
    /// the raw ptr is only re-materialized at the dispatch pop
    /// ([`Self::pop_for`] → [`Runnable::into_raw`]).
    pub fn push(&mut self, runnable: Runnable) {
        self.inner.push_back(runnable);
    }

    /// Pop the oldest thread whose `affinity & hart_mask != 0`, leaving
    /// non-matching entries in their slots. Returns the raw `*mut Thread`
    /// for the dispatch seam (stored into the hart's `current`); `None` if
    /// no entry matches.
    ///
    /// Order-preserving: entries skipped due to affinity stay where
    /// they are, so a later call with a different mask still picks
    /// them up in original push order.
    pub fn pop_for(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        // The affinity read goes through `Runnable::affinity` (which
        // dereferences the live registry Thread); the registry only frees
        // from `cleanup_threads_and_processes`, on the manager hart in the
        // same critical section as queue mutation — no use-after-free.
        let idx = self
            .inner
            .iter()
            .position(|r| r.affinity() & hart_mask != 0)?;
        self.inner.remove(idx).map(Runnable::into_raw)
    }

    /// Drop any queued token targeting `thread`. Called from the reap
    /// path so a freed `Thread` leaves no dispatchable entry behind (which
    /// would otherwise be popped and run as a use-after-free). A reaped
    /// thread is `Exited`, so it has no business being dispatched.
    pub fn remove_thread(&mut self, thread: *mut Thread) {
        self.inner.retain(|r| !r.points_to(thread));
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for ReadyQueue {
    fn default() -> Self {
        Self::new()
    }
}
