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
    fn next_runnable(&mut self) -> Option<&mut Thread> {
        let idx = self.next;
        if idx >= self.threads.len() {
            return None;
        }
        self.next += 1;
        self.threads.get_mut(idx)
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
    // Mark hart 2 busy with a bogus non-null pointer.
    slots[2].store(0xDEAD as *mut (), Ordering::Release);

    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(3);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Hart 1: assigned. Hart 2: still 0xDEAD. Hart 3: assigned.
    assert!(assigned(&slots[1]));
    assert_eq!(slots[2].load(Ordering::Acquire) as usize, 0xDEAD);
    assert!(assigned(&slots[3]));
    // Self gets the 3rd thread.
    assert!(assigned(&slots[0]));
    // Only harts 1 and 3 received IPIs.
    assert_eq!(hw.wakes, vec![1, 3]);
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
    slots[0].store(0xBEEF as *mut (), Ordering::Release);

    let (self_view, remote_views) = views(&slots);
    let mut sched = FakeSched::with(2);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remote_views.iter().copied(), &mut sched, &mut hw);

    // Remote got first thread; self clobbered with second.
    assert!(assigned(&slots[1]));
    let self_ptr = slots[0].load(Ordering::Acquire);
    assert!(!self_ptr.is_null());
    assert_ne!(self_ptr as usize, 0xBEEF, "self's current should have been overwritten");
    assert_eq!(hw.wakes, vec![1]);
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
