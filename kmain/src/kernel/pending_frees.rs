//! Global pending-frees queue for `Shared`-pool backings whose last
//! `SharedUserPtr` Arc just dropped.
//!
//! Drop can fire from any kernel-thread context — including `k_net` on a
//! hart that does *not* hold the Orbit lock, so calling the frame
//! allocator inline is unsound. Instead, [`SharedInner::drop`] pushes the
//! [`PhysBacking`] here and the manager (under the Orbit lock) drains via
//! [`pop`] in [`Orbit::cleanup_threads_and_processes`].
//!
//! The backing queue is a lock-free [`heapless::mpmc::Q64`]. A hart-local
//! SPSC version is on the roadmap once the per-hart infrastructure lands;
//! a single MPMC is good enough for orbit's current drop rate
//! (single-digit backings per process teardown).

use heapless::mpmc::Queue;
use process::PhysBacking;

static PENDING: Queue<PhysBacking, 64> = Queue::new();

pub fn push(b: PhysBacking) {
    if PENDING.enqueue(b).is_err() {
        panic!("pending_frees: Q64 exhausted; raise capacity or throttle drops");
    }
}

pub fn pop() -> Option<PhysBacking> {
    PENDING.dequeue()
}
