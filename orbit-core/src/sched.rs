//! Scheduler policy: assign runnable threads to idle harts.
//!
//! The pure function only manipulates thread state (ticks, `ThreadState`)
//! and publishes pointers into per-hart `current` slots. It has no
//! awareness of sscratch, CLINT, or CSRs — those are hidden behind the
//! [`Hardware`] and [`HartView`] boundaries.

use core::sync::atomic::{AtomicPtr, Ordering};

use process::Thread;

use crate::Hardware;

/// A per-hart view exposing only what the scheduler needs: the hart id
/// (for IPIs and affinity-mask construction) and a shared-ref to the
/// atomic pointer that names the thread that hart owns.
///
/// Real callers construct one from `&'a HartContext` by borrowing the
/// `current` field. Tests construct one by borrowing a standalone
/// `AtomicPtr<()>`.
#[derive(Clone, Copy)]
pub struct HartView<'a> {
    pub hart_id: usize,
    pub current: &'a AtomicPtr<()>,
}

impl<'a> HartView<'a> {
    /// Single-bit affinity mask naming this hart. The scheduler uses
    /// this when asking for the next thread runnable here.
    #[inline]
    pub fn affinity_bit(&self) -> u64 {
        1u64 << self.hart_id
    }

    /// True if the hart already has a thread assigned.
    pub fn is_busy(&self) -> bool {
        !self.current.load(Ordering::Acquire).is_null()
    }

    /// Publish `thread` as this hart's current thread. Release-orders the
    /// store so the target hart observes the assignment after the prior
    /// `thread.state = Assigned` write.
    ///
    /// Takes `*mut Thread` (not `&mut`) on purpose: the pointer must
    /// survive the scheduler's per-call borrow scope so the remote hart
    /// can safely dereference it later. A `&mut` reborrow would push a
    /// Stacked Borrows tag that gets popped on scope exit, leaving the
    /// stored raw ptr with dangling provenance — miri catches this.
    pub fn assign(&self, thread: *mut Thread) {
        self.current.store(thread as *mut (), Ordering::Release);
    }
}

/// Source of runnable threads. Each call returns a distinct `*mut Thread`
/// (or `None` when the queue is empty). Raw pointer on purpose: the
/// scheduler owns the underlying storage (Vec / Box / queue) and hands
/// out pointers with provenance derived from that storage, not from a
/// transient `&mut` reborrow. See [`HartView::assign`] for the aliasing
/// rationale.
///
/// The `hart_mask` argument lets the dispatcher request a thread that's
/// permitted to run on a specific hart (`mask = 1 << hart_id`) — see
/// the affinity machinery on `process::Thread`. Implementations must
/// only return threads whose `affinity & hart_mask != 0`. Threads that
/// don't fit the mask stay queued; a later call with a different mask
/// may pick them up.
pub trait Scheduler {
    /// # Safety contract
    ///
    /// Repeated calls must return pointers to *distinct* threads (no
    /// aliasing). The caller (assign_threads) dereferences each pointer
    /// to mutate thread fields before publishing it; concurrent returns
    /// of the same pointer would race.
    fn next_runnable(&mut self, hart_mask: u64) -> Option<*mut Thread>;
}

/// Distribute runnable threads across idle harts.
///
/// `remotes` is an iterator so the real caller can synthesize [`HartView`]s
/// on the fly from its `HartContext` array without any per-tick allocation.
/// Tests pass `Vec::into_iter()` or `slice.iter().copied()`.
///
/// Remotes are tried first in iteration order; each assignment marks the
/// thread `Assigned`, bumps its tick counter, publishes it to the hart's
/// `current` slot, and sends an IPI via [`Hardware::wake_hart`].
///
/// The caller's own hart (`self_view`) is tried last and doesn't receive
/// an IPI — it's already running. Matches the pre-migration ordering in
/// kmain: remotes are preferred so the caller only picks up work when
/// there's more to do than remotes to wake.
pub fn assign_threads<'a, H, S, I>(self_view: &HartView, remotes: I, sched: &mut S, hw: &mut H)
where
    H: Hardware,
    S: Scheduler,
    I: IntoIterator<Item = HartView<'a>>,
{
    for hart in remotes {
        if hart.is_busy() {
            continue;
        }
        // Affinity-aware: ask only for threads permitted on this hart.
        // A no-match here doesn't end the loop — another hart on the
        // next iteration may have a wider permitted set, and we don't
        // want a single restrictive thread sitting at the head of the
        // ready queue to starve unrelated work.
        let Some(t) = sched.next_runnable(hart.affinity_bit())
        else {
            continue;
        };
        // SAFETY: Scheduler contract guarantees `t` is a distinct,
        // non-aliased pointer; we hold no other reference to this
        // thread for the duration of the deref.
        unsafe { assign_thread_to(&hart, t) };
        hw.wake_hart(hart.hart_id);
    }

    // Symmetric `is_busy()` guard with the remotes loop above. The
    // kmain caller reaches this with a stale "I'm idle" view from
    // `hart_has_thread()` taken *before* MANAGER_LOCK was acquired;
    // in the window between that check and our acquire, a peer hart
    // running its own `assign_threads` pass under the lock could have
    // assigned a thread to us via its `remotes` arm and written to
    // `self_view.current`. Skipping self-assignment in that case lets
    // k_hart_loop dispatch the peer-supplied thread on its next
    // `hart_has_thread()` poll.
    if !self_view.is_busy() {
        if let Some(t) = sched.next_runnable(self_view.affinity_bit()) {
            // SAFETY: as above.
            unsafe { assign_thread_to(self_view, t) };
        }
    }
}

#[inline]
unsafe fn assign_thread_to(view: &HartView, thread: *mut Thread) {
    // Raw-pointer writes keep the pointer's provenance rooted in the
    // scheduler's owning storage. A `&mut Thread` reborrow here would
    // pop its tag on scope exit and invalidate the raw ptr stored in
    // `view.current` — see `HartView::assign` docs.
    //
    // `try_mark_assigned` is the checked `Ready → Assigned` edge: it
    // transitions only a genuinely-Ready thread, and returns `false`
    // (without storing or panicking) if the thread was killed while
    // queued (`Exited` — the benign kill race). In that case we must NOT
    // publish a dead thread to the hart's `current`, so we bump ticks /
    // assign only on a successful transition. A non-Ready, non-Exited
    // from-state panics inside the verb (a thread that shouldn't be in
    // the ready queue at all).
    let assigned = unsafe {
        (*thread).ticks = (*thread).ticks.wrapping_add(1);
        (*thread).try_mark_assigned()
    };
    if assigned {
        view.assign(thread);
    }
}
