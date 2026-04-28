//! Pure-logic tests for [`SleepHeap`].
//!
//! Each test boxes its threads so the `*mut Thread` stays stable for
//! the heap entry's lifetime. The `park` helper mirrors the kmain
//! sequence: bump `sleep_seq`, set `wake_time`, store
//! `state=Suspended` — exactly what `exit_thread_with_state(Suspended)`
//! and `kthread_park` do before pushing to `SLEEP_INBOX`.

use std::sync::atomic::Ordering;

use orbit_core::sleep_heap::SleepHeap;
use process::ThreadState;
use riscv::register::sstatus::SPP;

mod common;

/// Mirrors the parking-hart sequence in
/// `exit_thread_with_state`/`kthread_park`. Returns the post-increment
/// `sleep_seq` value the caller would push into `SLEEP_INBOX`.
fn park(thread: &mut process::Thread, wake_time: usize) -> u64 {
    let seq = thread.sleep_seq.fetch_add(1, Ordering::Release).wrapping_add(1);
    thread.wake_time = wake_time;
    thread.state.store(ThreadState::Suspended as usize, Ordering::Release);
    seq
}

fn make_suspended() -> Box<process::Thread> {
    Box::new(common::make_thread(ThreadState::Suspended, SPP::User))
}

#[test]
fn empty_heap_no_wakes() {
    let mut h = SleepHeap::new();
    assert!(h.is_empty());
    assert_eq!(h.next_wake(), None);
    let mut woken = Vec::new();
    h.drain_woken(1000, |t| woken.push(t));
    assert!(woken.is_empty());
}

#[test]
fn single_entry_deadline_passed() {
    let mut t = make_suspended();
    let tp: *mut _ = &mut *t;
    let seq = park(&mut t, 100);
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);
    assert_eq!(h.next_wake(), Some(100));

    let mut woken = Vec::new();
    h.drain_woken(150, |p| woken.push(p));
    assert_eq!(woken, vec![tp]);
    assert!(h.is_empty());
}

#[test]
fn deadline_in_future_not_drained() {
    let mut t = make_suspended();
    let tp: *mut _ = &mut *t;
    let seq = park(&mut t, 200);
    let mut h = SleepHeap::new();
    h.push(tp, 200, seq);

    let mut woken = Vec::new();
    h.drain_woken(150, |p| woken.push(p));
    assert!(woken.is_empty());
    assert_eq!(h.len(), 1);
    assert_eq!(h.next_wake(), Some(200));
}

#[test]
fn min_heap_order() {
    let mut t1 = make_suspended();
    let mut t2 = make_suspended();
    let mut t3 = make_suspended();
    let tp1: *mut _ = &mut *t1;
    let tp2: *mut _ = &mut *t2;
    let tp3: *mut _ = &mut *t3;
    let s1 = park(&mut t1, 300);
    let s2 = park(&mut t2, 100);
    let s3 = park(&mut t3, 200);

    let mut h = SleepHeap::new();
    // Push out of order; heap must still drain in deadline order.
    h.push(tp1, 300, s1);
    h.push(tp2, 100, s2);
    h.push(tp3, 200, s3);

    assert_eq!(h.next_wake(), Some(100));

    let mut woken = Vec::new();
    h.drain_woken(250, |p| woken.push(p));
    assert_eq!(woken, vec![tp2, tp3]);
    assert_eq!(h.next_wake(), Some(300));
}

#[test]
fn stale_entry_state_changed_to_ready() {
    let mut t = make_suspended();
    let tp: *mut _ = &mut *t;
    let seq = park(&mut t, 100);
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);

    // Eager promotion: state → Ready, no seq change. Mirrors what
    // set_wake_reason_where does in kmain.
    t.state.store(ThreadState::Ready as usize, Ordering::Release);

    let mut woken = Vec::new();
    h.drain_woken(150, |p| woken.push(p));
    assert!(woken.is_empty(), "stale entry must not fire callback");
    assert!(h.is_empty(), "stale entry must be removed");
}

#[test]
fn stale_entry_state_exited() {
    let mut t = make_suspended();
    let tp: *mut _ = &mut *t;
    let seq = park(&mut t, 100);
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);

    t.state.store(ThreadState::Exited as usize, Ordering::Release);

    let mut woken = Vec::new();
    h.drain_woken(150, |p| woken.push(p));
    assert!(woken.is_empty(), "Exited thread must not fire wake");
    assert!(h.is_empty());
}

#[test]
fn stale_entry_seq_mismatch_after_repark() {
    let mut t = make_suspended();
    let tp: *mut _ = &mut *t;
    let seq1 = park(&mut t, 100);
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq1);

    // Eager wake → re-park with new deadline. seq increments.
    t.state.store(ThreadState::Ready as usize, Ordering::Release);
    let seq2 = park(&mut t, 500);
    h.push(tp, 500, seq2);
    assert_eq!(h.len(), 2);
    assert_ne!(seq1, seq2);

    // Drain at time 200 — old entry has wake_time=100 (≤ 200) and
    // state==Suspended (after re-park). Without the seq check this
    // would mis-fire as a deadline-elapsed wake; with it the entry is
    // recognized as stale.
    let mut woken = Vec::new();
    h.drain_woken(200, |p| woken.push(p));
    assert!(woken.is_empty(), "re-park must make T1 entry stale");
    assert_eq!(h.len(), 1, "T2 entry stays in heap");
    assert_eq!(h.next_wake(), Some(500));
}

#[test]
fn transient_state_running_with_matching_seq() {
    // Models the kthread_park push-before-handoff window: inbox push
    // happens after fetch_add(seq) but before the asm publishes
    // state=Suspended. seq matches, state still Running. The heap
    // must NOT pop+drop (the park is real and pending), and must
    // NOT fire (deadline check would be on a half-committed park).
    let mut t = make_suspended();
    let tp: *mut _ = &mut *t;
    // Pre-park sequence: fetch_add seq, set wake_time, but state is
    // still Running (asm handoff hasn't published Suspended yet).
    let seq = t.sleep_seq.fetch_add(1, Ordering::Release).wrapping_add(1);
    t.wake_time = 100;
    t.state.store(ThreadState::Running as usize, Ordering::Release);
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);

    // Time well past deadline. With Running state, heap should leave
    // the entry alone instead of firing.
    let mut woken = Vec::new();
    h.drain_woken(500, |p| woken.push(p));
    assert!(woken.is_empty(), "transient entry must not fire");
    assert_eq!(h.len(), 1, "transient entry must stay in heap");

    // Once state commits to Suspended, next pass fires normally.
    t.state.store(ThreadState::Suspended as usize, Ordering::Release);
    h.drain_woken(500, |p| woken.push(p));
    assert_eq!(woken, vec![tp]);
    assert!(h.is_empty());
}

#[test]
fn mix_of_live_stale_and_pending() {
    let mut t1 = make_suspended();
    let mut t2 = make_suspended();
    let mut t3 = make_suspended();
    let tp1: *mut _ = &mut *t1;
    let tp2: *mut _ = &mut *t2;
    let tp3: *mut _ = &mut *t3;
    let s1 = park(&mut t1, 100);
    let s2 = park(&mut t2, 200);
    let s3 = park(&mut t3, 300);
    let mut h = SleepHeap::new();
    h.push(tp1, 100, s1);
    h.push(tp2, 200, s2);
    h.push(tp3, 300, s3);

    // Eagerly wake t1 (stale). t2 deadline-elapsed at drain time. t3 future.
    t1.state.store(ThreadState::Ready as usize, Ordering::Release);

    let mut woken = Vec::new();
    h.drain_woken(250, |p| woken.push(p));
    assert_eq!(woken, vec![tp2], "t1 stale (popped silently), t2 woken, t3 future");
    assert_eq!(h.len(), 1);
    assert_eq!(h.next_wake(), Some(300));
}

#[test]
fn drain_stops_at_first_future_live_entry() {
    // Confirms drain doesn't keep peeking past a live future entry —
    // important so a heap full of future deadlines doesn't get walked
    // top-to-bottom each pass.
    let mut t1 = make_suspended();
    let mut t2 = make_suspended();
    let tp1: *mut _ = &mut *t1;
    let tp2: *mut _ = &mut *t2;
    let s1 = park(&mut t1, 1000);
    let s2 = park(&mut t2, 2000);
    let mut h = SleepHeap::new();
    h.push(tp1, 1000, s1);
    h.push(tp2, 2000, s2);

    let mut woken = Vec::new();
    h.drain_woken(500, |p| woken.push(p));
    assert!(woken.is_empty());
    assert_eq!(h.len(), 2);
    assert_eq!(h.next_wake(), Some(1000));
}
