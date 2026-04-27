mod common;

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use process::{Thread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::sched::{HartView, Scheduler, assign_threads};

use common::{FakeHw, make_thread};

struct FakeSched {
    threads: Vec<Thread>,
    next: usize,
}

impl FakeSched {
    fn with(n: usize) -> Self {
        let threads = (0..n)
            .map(|i| {
                let mut t = make_thread(ThreadState::Ready, SPP::User);
                t.tid = (i + 1) as u32;
                t
            })
            .collect();
        Self { threads, next: 0 }
    }
    fn none() -> Self {
        Self { threads: Vec::new(), next: 0 }
    }
}

impl Scheduler for FakeSched {
    fn next_runnable(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        // Linear scan from `next` for the first thread compatible with
        // `hart_mask`. Skipping incompatible entries by *not* advancing
        // `next` past them would leave them re-pickable for the next
        // call (potentially with a different mask) — match the kernel
        // impl's "any-thread / first-fit" shape so test expectations
        // match production behavior.
        while self.next < self.threads.len() {
            let idx = self.next;
            self.next += 1;
            let aff = self.threads[idx].affinity.load(std::sync::atomic::Ordering::Relaxed);
            if aff & hart_mask != 0 {
                // SAFETY: `idx` is in-bounds (loop guard).
                return Some(unsafe { self.threads.as_mut_ptr().add(idx) });
            }
        }
        None
    }
}

/// Build `hart_count` AtomicPtr slots plus the `current` pointer of the
/// caller's own hart. Index 0 is self, 1..hart_count are remotes.
fn make_slots(hart_count: usize) -> Vec<AtomicPtr<()>> {
    (0..hart_count).map(|_| AtomicPtr::new(null_mut())).collect()
}

fn views<'a>(slots: &'a [AtomicPtr<()>]) -> (HartView<'a>, Vec<HartView<'a>>) {
    let self_view = HartView { hart_id: 0, current: &slots[0] };
    let remotes = slots[1..]
        .iter()
        .enumerate()
        .map(|(i, slot)| HartView { hart_id: (i + 1) as u32, current: slot })
        .collect();
    (self_view, remotes)
}

fn assigned(slot: &AtomicPtr<()>) -> bool {
    !slot.load(Ordering::Acquire).is_null()
}

#[test]
fn fills_all_idle_harts_remotes_get_ipis_self_does_not() {
    let slots = make_slots(4);
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(4);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // 3 remotes + 1 self = 4 assignments; 3 wakes.
    for slot in &slots {
        assert!(assigned(slot));
    }
    assert_eq!(hw.wakes, vec![1, 2, 3]);

    // Every thread now Assigned with ticks bumped.
    for t in &sched.threads {
        assert_eq!(t.state.load(Ordering::Acquire), ThreadState::Assigned as usize);
        assert_eq!(t.ticks, 1);
    }
}

#[test]
fn busy_remote_is_skipped() {
    let slots = make_slots(4);
    // Mark hart 2 busy with a real heap allocation (miri-friendly under
    // Tree Borrows; an `0xDEAD` int-to-ptr cast triggers a strict-
    // provenance warning even though the pointer is never dereferenced).
    let busy_sentinel = Box::into_raw(Box::new(0u8)) as *mut ();
    slots[2].store(busy_sentinel, Ordering::Release);

    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(3);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Hart 1: assigned. Hart 2: still the busy sentinel (untouched).
    // Hart 3: assigned.
    assert!(assigned(&slots[1]));
    assert_eq!(
        slots[2].load(Ordering::Acquire),
        busy_sentinel,
        "is_busy gate must leave hart 2's slot pointer untouched",
    );
    assert!(assigned(&slots[3]));
    // Self gets the 3rd thread.
    assert!(assigned(&slots[0]));
    // Only harts 1 and 3 received IPIs.
    assert_eq!(hw.wakes, vec![1, 3]);

    unsafe { drop(Box::from_raw(busy_sentinel as *mut u8)) };
}

#[test]
fn fewer_runnable_than_idle_harts_leaves_trailing_idle() {
    let slots = make_slots(4);
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(2);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Remotes come first. First 2 remotes (hart 1, hart 2) get threads;
    // hart 3 and self stay idle.
    assert!(assigned(&slots[1]));
    assert!(assigned(&slots[2]));
    assert!(!assigned(&slots[3]));
    assert!(!assigned(&slots[0]));
    assert_eq!(hw.wakes, vec![1, 2]);
}

#[test]
fn zero_runnable_is_noop() {
    let slots = make_slots(4);
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::none();
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    for slot in &slots {
        assert!(!assigned(slot));
    }
    assert!(hw.wakes.is_empty());
}

#[test]
fn remotes_exhaust_queue_before_self_tries() {
    let slots = make_slots(4); // 1 self + 3 remotes
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(3);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // All 3 threads went to remotes; self got nothing.
    assert!(assigned(&slots[1]));
    assert!(assigned(&slots[2]));
    assert!(assigned(&slots[3]));
    assert!(!assigned(&slots[0]), "self must not get a thread when remotes drain the queue");
    assert_eq!(hw.wakes, vec![1, 2, 3]);
}

#[test]
fn self_overwrites_own_current_unconditionally() {
    // The live kmain code doesn't gate the self-assignment on is_busy —
    // if self already has a current thread, it's clobbered. Preserve that.
    let slots = make_slots(2); // self + 1 remote
    let stale_sentinel = Box::into_raw(Box::new(0u8)) as *mut ();
    slots[0].store(stale_sentinel, Ordering::Release);

    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(2);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Remote got first thread; self clobbered with second.
    assert!(assigned(&slots[1]));
    let self_ptr = slots[0].load(Ordering::Acquire);
    assert!(!self_ptr.is_null());
    assert_ne!(
        self_ptr, stale_sentinel,
        "self's current should have been overwritten",
    );
    assert_eq!(hw.wakes, vec![1]);

    unsafe { drop(Box::from_raw(stale_sentinel as *mut u8)) };
}

#[test]
fn correct_thread_published_to_correct_hart() {
    let slots = make_slots(3); // self + 2 remotes
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(3);
    let mut hw = FakeHw::default();

    // Remember each thread's address for comparison post-call.
    let expected: Vec<*const Thread> = sched.threads.iter().map(|t| t as *const _).collect();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Remote 1 got thread[0], remote 2 got thread[1], self got thread[2].
    assert_eq!(slots[1].load(Ordering::Acquire) as *const Thread, expected[0]);
    assert_eq!(slots[2].load(Ordering::Acquire) as *const Thread, expected[1]);
    assert_eq!(slots[0].load(Ordering::Acquire) as *const Thread, expected[2]);
}

#[test]
fn ticks_wrap_at_u8_max() {
    let slots = make_slots(2);
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(1);
    sched.threads[0].ticks = u8::MAX;
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    assert_eq!(sched.threads[0].ticks, 0, "ticks must wrapping_add, not panic");
}

#[test]
fn no_remotes_still_assigns_to_self() {
    let slots = make_slots(1); // self only
    let (self_view, remote_views) = views(&slots);
    assert!(remote_views.is_empty());
    let mut sched = FakeSched::with(1);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    assert!(assigned(&slots[0]));
    assert!(hw.wakes.is_empty());
}

/// One pass must satisfy several simultaneous invariants per thread:
///   - each assigned thread has `ticks` bumped by EXACTLY 1
///   - each assigned thread is in `ThreadState::Assigned`
///   - every slot points to a distinct thread from our queue
///   - every thread in the queue ends up pointed-at by some slot
///
/// Using distinct initial ticks across threads (so a shared/global
/// increment wouldn't look right) and asserting per-thread.
#[test]
fn distribution_preserves_per_thread_invariants() {
    let slots = make_slots(4);
    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(4);
    // Seed distinct starting ticks: [10, 20, 30, 40]
    for (i, t) in sched.threads.iter_mut().enumerate() {
        t.ticks = 10 * (i as u8 + 1);
    }
    let expected_addrs: Vec<*const Thread> =
        sched.threads.iter().map(|t| t as *const _).collect();
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Per-thread: tick+=1 (NOT shared), state==Assigned.
    let expected_ticks: [u8; 4] = [11, 21, 31, 41];
    for (i, t) in sched.threads.iter().enumerate() {
        assert_eq!(
            t.ticks, expected_ticks[i],
            "thread[{i}] ticks should be {} (was {}), bumped by exactly 1",
            expected_ticks[i], t.ticks
        );
        assert_eq!(
            t.state.load(Ordering::Acquire),
            ThreadState::Assigned as usize,
            "thread[{i}] should be Assigned"
        );
    }

    // Published ptrs are distinct and cover every thread in the queue.
    let published: Vec<*const Thread> = slots
        .iter()
        .map(|s| s.load(Ordering::Acquire) as *const Thread)
        .collect();
    let mut sorted_pub = published.clone();
    sorted_pub.sort();
    sorted_pub.dedup();
    assert_eq!(
        sorted_pub.len(),
        4,
        "four slots must hold four distinct thread pointers"
    );
    for addr in &expected_addrs {
        assert!(
            published.contains(addr),
            "every queued thread must be pointed at by some slot"
        );
    }

    // No Ready thread left over (queue drained).
    assert!(sched.threads.iter().all(|t| t.state.load(Ordering::Acquire)
        != ThreadState::Ready as usize));
}
