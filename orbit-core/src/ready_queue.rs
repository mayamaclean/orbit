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
use core::sync::atomic::Ordering;

use process::Thread;

pub struct ReadyQueue {
    inner: VecDeque<*mut Thread>,
}

impl ReadyQueue {
    pub const fn new() -> Self {
        Self { inner: VecDeque::new() }
    }

    /// Append a thread to the back of the queue. Caller has ensured
    /// `state == Ready` and that this exact pointer isn't already
    /// queued (double-queue would let `pop_for` hand the same thread
    /// out twice in one pass — distinct-pointer contract violation).
    pub fn push(&mut self, thread: *mut Thread) {
        self.inner.push_back(thread);
    }

    /// Pop the oldest thread whose `affinity & hart_mask != 0`, leaving
    /// non-matching entries in their slots. Returns `None` if no entry
    /// matches.
    ///
    /// Order-preserving: entries skipped due to affinity stay where
    /// they are, so a later call with a different mask still picks
    /// them up in original push order.
    pub fn pop_for(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        // SAFETY: each pointer was supplied by a producer that owned
        // the Thread allocation through the kernel registry. The
        // registry only frees from `cleanup_threads_and_processes`,
        // which runs on the manager hart in the same critical
        // section as queue mutation — no use-after-free window.
        let idx = self.inner.iter().position(|&t| unsafe {
            (*t).affinity.load(Ordering::Relaxed) & hart_mask != 0
        })?;
        self.inner.remove(idx)
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
