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
///
/// Takes `*mut Thread` rather than `&mut Thread` so the write to
/// `wake_time` doesn't reborrow as a sibling `&mut Thread` of the heap
/// entry's raw pointer — under Tree Borrows that would Disable the
/// raw tag, and the eventual `drain_woken` reborrow would fail.
unsafe fn park(thread: *mut process::Thread, wake_time: usize) -> u64 {
    unsafe {
        let seq = (*thread)
            .sleep_seq
            .fetch_add(1, Ordering::Release)
            .wrapping_add(1);
        process::RunningThread::from_ptr(thread).set_wake_time(wake_time);
        (*thread).transition_to_unchecked(ThreadState::Suspended);
        seq
    }
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
    let tp: *mut _ = &raw mut *t;
    let seq = unsafe { park(tp, 100) };
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
    let tp: *mut _ = &raw mut *t;
    let seq = unsafe { park(tp, 200) };
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
    let tp1: *mut _ = &raw mut *t1;
    let tp2: *mut _ = &raw mut *t2;
    let tp3: *mut _ = &raw mut *t3;
    let s1 = unsafe { park(tp1, 300) };
    let s2 = unsafe { park(tp2, 100) };
    let s3 = unsafe { park(tp3, 200) };

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
    let tp: *mut _ = &raw mut *t;
    let seq = unsafe { park(tp, 100) };
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);

    // Eager promotion: state → Ready, no seq change. Mirrors what
    // set_wake_reason_where does in kmain.
    t.transition_to_unchecked(ThreadState::Ready);

    let mut woken = Vec::new();
    h.drain_woken(150, |p| woken.push(p));
    assert!(woken.is_empty(), "stale entry must not fire callback");
    assert!(h.is_empty(), "stale entry must be removed");
}

#[test]
fn stale_entry_state_exited() {
    let mut t = make_suspended();
    let tp: *mut _ = &raw mut *t;
    let seq = unsafe { park(tp, 100) };
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);

    t.transition_to_unchecked(ThreadState::Exited);

    let mut woken = Vec::new();
    h.drain_woken(150, |p| woken.push(p));
    assert!(woken.is_empty(), "Exited thread must not fire wake");
    assert!(h.is_empty());
}

#[test]
fn stale_entry_seq_mismatch_after_repark() {
    let mut t = make_suspended();
    let tp: *mut _ = &raw mut *t;
    let seq1 = unsafe { park(tp, 100) };
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq1);

    // Eager wake → re-park with new deadline. seq increments.
    t.transition_to_unchecked(ThreadState::Ready);
    let seq2 = unsafe { park(tp, 500) };
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
    let tp: *mut _ = &raw mut *t;
    // Pre-park sequence: fetch_add seq, set wake_time, but state is
    // still Running (asm handoff hasn't published Suspended yet).
    let seq = unsafe {
        let s = (*tp)
            .sleep_seq
            .fetch_add(1, Ordering::Release)
            .wrapping_add(1);
        process::RunningThread::from_ptr(tp).set_wake_time(100);
        (*tp).transition_to_unchecked(ThreadState::Running);
        s
    };
    let mut h = SleepHeap::new();
    h.push(tp, 100, seq);

    // Time well past deadline. With Running state, heap should leave
    // the entry alone instead of firing.
    let mut woken = Vec::new();
    h.drain_woken(500, |p| woken.push(p));
    assert!(woken.is_empty(), "transient entry must not fire");
    assert_eq!(h.len(), 1, "transient entry must stay in heap");

    // Once state commits to Suspended, next pass fires normally.
    t.transition_to_unchecked(ThreadState::Suspended);
    h.drain_woken(500, |p| woken.push(p));
    assert_eq!(woken, vec![tp]);
    assert!(h.is_empty());
}

#[test]
fn mix_of_live_stale_and_pending() {
    let mut t1 = make_suspended();
    let mut t2 = make_suspended();
    let mut t3 = make_suspended();
    let tp1: *mut _ = &raw mut *t1;
    let tp2: *mut _ = &raw mut *t2;
    let tp3: *mut _ = &raw mut *t3;
    let s1 = unsafe { park(tp1, 100) };
    let s2 = unsafe { park(tp2, 200) };
    let s3 = unsafe { park(tp3, 300) };
    let mut h = SleepHeap::new();
    h.push(tp1, 100, s1);
    h.push(tp2, 200, s2);
    h.push(tp3, 300, s3);

    // Eagerly wake t1 (stale). t2 deadline-elapsed at drain time. t3 future.
    t1.transition_to_unchecked(ThreadState::Ready);

    let mut woken = Vec::new();
    h.drain_woken(250, |p| woken.push(p));
    assert_eq!(
        woken,
        vec![tp2],
        "t1 stale (popped silently), t2 woken, t3 future"
    );
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
    let tp1: *mut _ = &raw mut *t1;
    let tp2: *mut _ = &raw mut *t2;
    let s1 = unsafe { park(tp1, 1000) };
    let s2 = unsafe { park(tp2, 2000) };
    let mut h = SleepHeap::new();
    h.push(tp1, 1000, s1);
    h.push(tp2, 2000, s2);

    let mut woken = Vec::new();
    h.drain_woken(500, |p| woken.push(p));
    assert!(woken.is_empty());
    assert_eq!(h.len(), 2);
    assert_eq!(h.next_wake(), Some(1000));
}

/// Regression: a stale heap entry whose thread is now `Blocking` (with a
/// matching `sleep_seq`) must NOT freeze the drain. This is the
/// hello-std `yield_now` hang: a thread `sleep_ms(0)`s (Suspended heap
/// entry, earliest deadline), gets eager-promoted out WITHOUT its entry
/// popped, then `futex_wait`s into a `Blocking` park (which does not bump
/// `sleep_seq`). The stale Blocking entry sits at the heap top; if it's
/// classified `Transient` (the old behavior) `drain_woken` `break`s on it
/// forever and every sleeper below never wakes. It must be reaped as
/// `Stale` so deeper Live sleepers still fire.
#[test]
fn blocking_stale_entry_does_not_freeze_drain() {
    let mut blocker = make_suspended();
    let mut sleeper = make_suspended();
    let bp: *mut _ = &raw mut *blocker;
    let sp: *mut _ = &raw mut *sleeper;

    // Blocker parked first with the EARLIEST deadline (heap top).
    let bseq = unsafe { park(bp, 10) };
    let sseq = unsafe { park(sp, 100) };
    let mut h = SleepHeap::new();
    h.push(bp, 10, bseq);
    h.push(sp, 100, sseq);

    // The blocker left its Suspended park for a Blocking (futex) park —
    // seq UNCHANGED (Blocking doesn't bump it), so its heap entry's seq
    // still matches. Old classify => Transient => freeze.
    blocker.transition_to_unchecked(ThreadState::Blocking);

    let mut woken = Vec::new();
    h.drain_woken(200, |p| woken.push(p));

    // The stale Blocking entry is reaped; the genuine sleeper fires.
    assert_eq!(
        woken,
        vec![sp],
        "deeper sleeper must wake despite the stale Blocking entry on top"
    );
    assert!(h.is_empty(), "both entries consumed (one stale, one woken)");
}

#[test]
fn remove_thread_scrubs_entry_and_unblocks_drain() {
    // The reap-path scrub (`dealloc_thread` -> `Manager::forget_thread` ->
    // `SleepHeap::remove_thread`): a reaped thread's entry sits at the heap
    // top in a Transient-shaped (Running) state that would otherwise
    // `break` the drain. Scrubbing it lets the genuine sleeper below wake —
    // and, in the kernel, means a freed allocation is never dereferenced.
    let mut reaped = make_suspended();
    let mut sleeper = make_suspended();
    let rp: *mut _ = &raw mut *reaped;
    let sp: *mut _ = &raw mut *sleeper;

    let rseq = unsafe { park(rp, 10) }; // earliest deadline -> heap top
    let sseq = unsafe { park(sp, 100) };
    let mut h = SleepHeap::new();
    h.push(rp, 10, rseq);
    h.push(sp, 100, sseq);

    // The about-to-be-reaped thread, mid-teardown, in the freeze shape.
    reaped.transition_to_unchecked(ThreadState::Running);

    h.remove_thread(rp);
    assert_eq!(h.len(), 1, "only the reaped thread's entry was removed");

    let mut woken = Vec::new();
    h.drain_woken(200, |p| woken.push(p));
    assert_eq!(
        woken,
        vec![sp],
        "deeper sleeper wakes; the reaped entry was scrubbed, not left to block",
    );
    assert!(h.is_empty());
}

#[test]
fn remove_thread_drops_every_entry_for_that_thread() {
    // A thread can leave several stale entries (re-parked across passes);
    // remove_thread must drop them all, leaving other threads' entries.
    let mut t = make_suspended();
    let tp: *mut _ = &raw mut *t;
    let mut other = make_suspended();
    let op: *mut _ = &raw mut *other;

    let mut h = SleepHeap::new();
    h.push(tp, 10, 1);
    h.push(tp, 20, 2);
    h.push(tp, 30, 3);
    let oseq = unsafe { park(op, 15) };
    h.push(op, 15, oseq);
    assert_eq!(h.len(), 4);

    h.remove_thread(tp);
    assert_eq!(h.len(), 1, "all of tp's entries gone; other's remains");
    assert_eq!(h.next_wake(), Some(15), "other's entry survived");
}

#[test]
fn unknown_state_entry_is_evicted_not_transient() {
    // A matching-seq entry whose target reads a non-enumerated `state`
    // (e.g. a freed+recycled allocation) must be evicted, never treated as
    // Transient — otherwise it sits at the heap top and freezes the drain.
    let mut garbage = make_suspended();
    let mut sleeper = make_suspended();
    let gp: *mut _ = &raw mut *garbage;
    let sp: *mut _ = &raw mut *sleeper;

    let gseq = unsafe { park(gp, 10) }; // top
    let sseq = unsafe { park(sp, 50) };
    let mut h = SleepHeap::new();
    h.push(gp, 10, gseq);
    h.push(sp, 50, sseq);

    // Stamp an out-of-range discriminant via the raw state atom (what a
    // recycled allocation might read). seq still matches the entry.
    garbage.store_state_raw(0xDEAD);

    let mut woken = Vec::new();
    h.drain_woken(200, |p| woken.push(p));
    assert_eq!(
        woken,
        vec![sp],
        "garbage entry evicted as stale; sleeper wakes"
    );
    assert!(h.is_empty());
}
