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
    pub fn assign(&self, thread: &Thread) {
        self.current
            .store(thread as *const Thread as *mut (), Ordering::Release);
    }
}

/// Source of runnable threads. Each call returns a distinct `&mut Thread`
/// (or `None` when the queue is empty). The trait takes `&mut self` so the
/// returned reference's lifetime is bounded by the borrow — callers can't
/// hold two threads at once, which mirrors how the real Orbit scheduler
/// pops its queue.
pub trait Scheduler {
    fn next_runnable(&mut self) -> Option<&mut Thread>;
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
        assign_thread_to(&hart, t);
        hw.wake_hart(hart.hart_id);
    }

    if let Some(t) = sched.next_runnable() {
        assign_thread_to(self_view, t);
    }
}

#[inline]
fn assign_thread_to(view: &HartView, thread: &mut Thread) {
    thread.ticks = thread.ticks.wrapping_add(1);
    thread
        .state
        .store(ThreadState::Assigned as usize, Ordering::Release);
    view.assign(thread);
}
