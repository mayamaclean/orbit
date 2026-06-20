//! Multi-threaded validation of the scheduler's cross-hart atomic
//! handoff. Runs under miri (both Stacked and Tree Borrows) to check
//! that the Release/Acquire pair on `current` and `thread.state`
//! establishes a happens-before chain strong enough to safely read
//! `thread.ticks` (plain `u8`, non-atomic) through the published
//! pointer.
//!
//! This is the core correctness requirement of kmain's whole
//! cross-hart assignment mechanism — if it's wrong, kernel code
//! running on the target hart can observe stale or partial writes
//! from the assigning hart.

mod common;

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::thread;

use process::{Thread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::sched::{HartView, Scheduler, assign_threads};

use common::{FakeHw, make_thread};

/// One-shot scheduler: hands out a single thread on the first
/// `next_runnable`, then reports empty. Owns the thread via a raw
/// pointer into a heap-allocated Box so the pointer's provenance is
/// rooted in the Box's allocation (not in a transient `&mut` to a
/// struct field), preserving validity across the scheduler's return
/// and the remote hart's subsequent deref.
struct OneShotSched {
    thread_ptr: *mut Thread,
    handed_out: bool,
}

impl OneShotSched {
    fn new(t: Thread) -> Self {
        Self {
            thread_ptr: Box::into_raw(Box::new(t)),
            handed_out: false,
        }
    }
}

impl Drop for OneShotSched {
    fn drop(&mut self) {
        // Reclaim the Box so miri doesn't flag a leak.
        unsafe { drop(Box::from_raw(self.thread_ptr)) };
    }
}

impl Scheduler for OneShotSched {
    fn next_runnable(&mut self, _hart_mask: u64) -> Option<*mut Thread> {
        if self.handed_out {
            return None;
        }
        self.handed_out = true;
        Some(self.thread_ptr)
    }
}

#[test]
fn remote_observes_state_and_ticks_via_release_acquire() {
    let remote_slot: AtomicPtr<()> = AtomicPtr::new(null_mut());
    let self_slot: AtomicPtr<()> = AtomicPtr::new(null_mut());
    // Sentinel values the main thread assumes were never written.
    let observed_state = AtomicUsize::new(0xFFFF_FFFF);
    let observed_ticks = AtomicUsize::new(0xFFFF_FFFF);

    let mut thread = make_thread(ThreadState::Ready, SPP::User);
    thread.ticks = 41;
    let mut sched = OneShotSched::new(thread);

    thread::scope(|s| {
        s.spawn(|| {
            // Target hart: spin until the scheduler publishes a thread.
            // The Acquire load synchronizes with the Release store on
            // `current` inside `HartView::assign`, so every write
            // sequenced-before it (including the non-atomic
            // `thread.ticks` bump and the `thread.state = Assigned`
            // store) is visible here without a data race.
            loop {
                let p = remote_slot.load(Ordering::Acquire);
                if !p.is_null() {
                    let t = unsafe { &*(p as *const Thread) };
                    observed_state.store(t.state_load(Ordering::Acquire), Ordering::Release);
                    observed_ticks.store(t.ticks as usize, Ordering::Release);
                    return;
                }
                thread::yield_now();
            }
        });

        // Assigning hart (main): build views + invoke. The iterator
        // variant of assign_threads takes HartView by value, which is
        // Copy, so we can construct inline without retained borrows.
        let self_view = HartView {
            hart_id: 0,
            current: &self_slot,
        };
        let remote_view = HartView {
            hart_id: 1,
            current: &remote_slot,
        };
        let mut hw = FakeHw::default();

        assign_threads(&self_view, [remote_view], &mut sched, &mut hw);
    });

    // After thread::scope returns, the remote thread has joined.
    assert_eq!(
        observed_state.load(Ordering::Acquire),
        ThreadState::Assigned as usize,
        "remote must see ThreadState::Assigned after observing the published ptr"
    );
    assert_eq!(
        observed_ticks.load(Ordering::Acquire),
        42,
        "remote must see ticks == 42 (41 + 1 bump) — the Release store on `current` \
         carries the non-atomic ticks write across threads"
    );
    // And the self slot stayed null — only remote consumed the thread.
    assert!(self_slot.load(Ordering::Acquire).is_null());
}

/// Multiple threads racing on their respective slots, each getting a
/// distinct thread. Exercises the happens-before chain for four
/// independent remotes simultaneously — catches any accidental shared
/// state in the assign loop (e.g. if `hw.wake_hart` held a borrow
/// across iterations).
#[test]
fn multiple_remotes_each_observe_their_own_thread() {
    const N: usize = 4;
    let slots: [AtomicPtr<()>; N] = std::array::from_fn(|_| AtomicPtr::new(null_mut()));
    let self_slot: AtomicPtr<()> = AtomicPtr::new(null_mut());
    let observed_tids: [AtomicUsize; N] = std::array::from_fn(|_| AtomicUsize::new(0));

    // Hand out N threads with distinct tids. Vec::as_mut_ptr roots
    // provenance at the Vec's allocation, not at per-element reborrows.
    struct MultiSched {
        threads: Vec<Thread>,
        next: usize,
    }
    impl Scheduler for MultiSched {
        fn next_runnable(&mut self, _hart_mask: u64) -> Option<*mut Thread> {
            let i = self.next;
            if i >= self.threads.len() {
                return None;
            }
            self.next += 1;
            // SAFETY: i is in-bounds.
            Some(unsafe { self.threads.as_mut_ptr().add(i) })
        }
    }

    let threads: Vec<Thread> = (0..N)
        .map(|i| {
            let mut t = make_thread(ThreadState::Ready, SPP::User);
            t.tid = (i + 100) as u32;
            t
        })
        .collect();
    let mut sched = MultiSched { threads, next: 0 };

    thread::scope(|s| {
        for i in 0..N {
            let slot = &slots[i];
            let obs = &observed_tids[i];
            s.spawn(move || {
                loop {
                    let p = slot.load(Ordering::Acquire);
                    if !p.is_null() {
                        let t = unsafe { &*(p as *const Thread) };
                        obs.store(t.tid as usize, Ordering::Release);
                        return;
                    }
                    thread::yield_now();
                }
            });
        }

        let self_view = HartView {
            hart_id: 0,
            current: &self_slot,
        };
        let remotes: [HartView; N] = std::array::from_fn(|i| HartView {
            hart_id: (i + 1),
            current: &slots[i],
        });
        let mut hw = FakeHw::default();

        assign_threads(&self_view, remotes, &mut sched, &mut hw);
    });

    // Each remote should have observed a DISTINCT tid in 100..104.
    let mut seen: Vec<usize> = observed_tids
        .iter()
        .map(|a| a.load(Ordering::Acquire))
        .collect();
    seen.sort();
    assert_eq!(seen, vec![100, 101, 102, 103]);
}

/// Regression: prior to the `is_busy()` guard on the self-view branch,
/// `assign_threads` would clobber `self_view.current` if a peer hart
/// had raced ahead and assigned a thread to us between kmain's
/// pre-MANAGER_LOCK `hart_has_thread()` check and our own
/// `assign_threads` call. The clobber leaked the peer-assigned thread
/// (stuck in `Assigned` state, never dispatched).
///
/// We simulate the race by pre-populating `self_slot` with a sentinel
/// pointer (representing the peer's assignment) and then invoking
/// `assign_threads` with a single thread available in the scheduler.
/// With the guard, the self_view branch sees `is_busy() == true` and
/// skips both `next_runnable` and the assignment — `self_slot` keeps
/// the peer's pointer and the scheduler's thread stays queued for a
/// later pass.
#[test]
fn self_view_busy_skips_assignment_does_not_clobber() {
    // A non-null sentinel for the peer-assigned `current`. Doesn't
    // need to point at a real Thread for this test — `assign_threads`
    // never derefs it; it only reads it via `is_busy()`.
    let peer_sentinel: usize = 0xDEAD_BEEF;
    let self_slot: AtomicPtr<()> = AtomicPtr::new(core::ptr::without_provenance_mut(peer_sentinel));

    let thread = make_thread(ThreadState::Ready, SPP::User);
    let mut sched = OneShotSched::new(thread);

    let self_view = HartView {
        hart_id: 0,
        current: &self_slot,
    };
    let mut hw = FakeHw::default();

    // Empty `remotes` so the only candidate slot is self_view, which
    // is busy.
    assign_threads(&self_view, std::iter::empty(), &mut sched, &mut hw);

    // Peer's assignment must survive untouched.
    assert_eq!(
        self_slot.load(Ordering::Acquire) as usize,
        peer_sentinel,
        "self_view.current was clobbered despite is_busy() — guard regression"
    );
    // And the scheduler's thread must still be queued: the busy guard
    // skips `next_runnable`, so a later pass can pick it up.
    assert!(
        !sched.handed_out,
        "scheduler should not have been polled when self_view was busy"
    );
}
