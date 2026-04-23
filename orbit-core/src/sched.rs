//! Scheduler policy: assign runnable threads to idle harts.
//!
//! The pure function only manipulates thread state (ticks, `ThreadState`)
//! and publishes pointers into per-hart `current` slots. It has no
//! awareness of sscratch, CLINT, or CSRs — those are hidden behind the
//! [`Hardware`] and [`HartView`] boundaries.

use core::sync::atomic::{AtomicPtr, Ordering};

use process::{Thread, ThreadState};

use crate::Hardware;

/// A per-hart view exposing only what the scheduler needs: the hart id
/// (for IPIs) and a shared-ref to the atomic pointer that names the
/// thread that hart owns.
///
/// Real callers construct one from `&'a HartContext` by borrowing the
/// `current` field. Tests construct one by borrowing a standalone
/// `AtomicPtr<()>`.
#[derive(Clone, Copy)]
pub struct HartView<'a> {
    pub hart_id: u32,
    pub current: &'a AtomicPtr<()>,
}

impl<'a> HartView<'a> {
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
pub trait Scheduler {
    /// # Safety contract
    ///
    /// Repeated calls must return pointers to *distinct* threads (no
    /// aliasing). The caller (assign_threads) dereferences each pointer
    /// to mutate thread fields before publishing it; concurrent returns
    /// of the same pointer would race.
    fn next_runnable(&mut self) -> Option<*mut Thread>;
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
pub fn assign_threads<'a, H, S, I>(
    self_view: &HartView,
    remotes: I,
    sched: &mut S,
    hw: &mut H,
) where
    H: Hardware,
    S: Scheduler,
    I: IntoIterator<Item = HartView<'a>>,
{
    for hart in remotes {
        if hart.is_busy() {
            continue;
        }
        let Some(t) = sched.next_runnable() else { return };
        // SAFETY: Scheduler contract guarantees `t` is a distinct,
        // non-aliased pointer; we hold no other reference to this
        // thread for the duration of the deref.
        unsafe { assign_thread_to(&hart, t) };
        hw.wake_hart(hart.hart_id);
    }

    if let Some(t) = sched.next_runnable() {
        // SAFETY: as above.
        unsafe { assign_thread_to(self_view, t) };
    }
}

#[inline]
unsafe fn assign_thread_to(view: &HartView, thread: *mut Thread) {
    // Raw-pointer writes keep the pointer's provenance rooted in the
    // scheduler's owning storage. A `&mut Thread` reborrow here would
    // pop its tag on scope exit and invalidate the raw ptr stored in
    // `view.current` — see `HartView::assign` docs.
    unsafe {
        (*thread).ticks = (*thread).ticks.wrapping_add(1);
        (*thread)
            .state
            .store(ThreadState::Assigned as usize, Ordering::Release);
    }
    view.assign(thread);
}
