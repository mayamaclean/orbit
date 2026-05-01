//! Pure-logic tests for [`ReadyQueue`].
//!
//! Each test boxes its threads so the `*mut Thread` stays stable for
//! the queue's lifetime. Affinity is mutated in place via the atomic
//! `Thread::affinity` field — same shape as the live kmain code path.

use std::sync::atomic::Ordering;

use orbit_core::ready_queue::ReadyQueue;
use process::ThreadState;
use riscv::register::sstatus::SPP;

mod common;

fn make_ready() -> Box<process::Thread> {
    Box::new(common::make_thread(ThreadState::Ready, SPP::User))
}

fn set_affinity(t: &mut process::Thread, mask: u64) {
    t.affinity.store(mask, Ordering::Release);
}

#[test]
fn empty_queue_pop_returns_none() {
    let mut q = ReadyQueue::new();
    assert!(q.is_empty());
    assert_eq!(q.pop_for(u64::MAX), None);
}

#[test]
fn fifo_order_with_wide_affinity() {
    let mut t1 = make_ready();
    let mut t2 = make_ready();
    let mut t3 = make_ready();
    let tp1: *mut _ = &mut *t1;
    let tp2: *mut _ = &mut *t2;
    let tp3: *mut _ = &mut *t3;
    let mut q = ReadyQueue::new();
    q.push(tp1);
    q.push(tp2);
    q.push(tp3);
    assert_eq!(q.len(), 3);
    assert_eq!(q.pop_for(u64::MAX), Some(tp1));
    assert_eq!(q.pop_for(u64::MAX), Some(tp2));
    assert_eq!(q.pop_for(u64::MAX), Some(tp3));
    assert_eq!(q.pop_for(u64::MAX), None);
}

#[test]
fn affinity_skip_preserves_order_for_remaining() {
    // Pinned thread at the head; an unpinned thread behind it.
    // pop_for(other_hart) must skip the pinned one and return the
    // unpinned one — and a subsequent pop_for(matching_hart) must
    // still find the pinned one in original position.
    let mut t_pinned = make_ready();
    let mut t_open = make_ready();
    set_affinity(&mut t_pinned, 1 << 3); // hart 3 only
    set_affinity(&mut t_open, u64::MAX);

    let tp_pinned: *mut _ = &mut *t_pinned;
    let tp_open: *mut _ = &mut *t_open;

    let mut q = ReadyQueue::new();
    q.push(tp_pinned);
    q.push(tp_open);

    // hart 0 looks for work — pinned skipped, open returned.
    assert_eq!(q.pop_for(1 << 0), Some(tp_open));
    assert_eq!(q.len(), 1, "pinned still queued");

    // hart 3 looks — gets the pinned one.
    assert_eq!(q.pop_for(1 << 3), Some(tp_pinned));
    assert!(q.is_empty());
}

#[test]
fn no_match_returns_none_without_consuming() {
    let mut t = make_ready();
    set_affinity(&mut t, 1 << 2);
    let tp: *mut _ = &mut *t;
    let mut q = ReadyQueue::new();
    q.push(tp);

    assert_eq!(q.pop_for(1 << 0), None);
    assert_eq!(q.len(), 1, "no matching hart must not consume");
    assert_eq!(q.pop_for(1 << 2), Some(tp));
}

#[test]
fn multiple_pinned_pop_in_push_order_per_mask() {
    // Three pinned threads to hart 1; two pinned to hart 2. A pop_for
    // hart 1 must drain in push order from the hart-1 subset; hart 2
    // similarly.
    let mut t1a = make_ready();
    let mut t1b = make_ready();
    let mut t2a = make_ready();
    let mut t1c = make_ready();
    let mut t2b = make_ready();
    set_affinity(&mut t1a, 1 << 1);
    set_affinity(&mut t1b, 1 << 1);
    set_affinity(&mut t2a, 1 << 2);
    set_affinity(&mut t1c, 1 << 1);
    set_affinity(&mut t2b, 1 << 2);

    let p1a: *mut _ = &mut *t1a;
    let p1b: *mut _ = &mut *t1b;
    let p2a: *mut _ = &mut *t2a;
    let p1c: *mut _ = &mut *t1c;
    let p2b: *mut _ = &mut *t2b;

    let mut q = ReadyQueue::new();
    q.push(p1a);
    q.push(p1b);
    q.push(p2a);
    q.push(p1c);
    q.push(p2b);

    assert_eq!(q.pop_for(1 << 1), Some(p1a));
    assert_eq!(q.pop_for(1 << 2), Some(p2a));
    assert_eq!(q.pop_for(1 << 1), Some(p1b));
    assert_eq!(q.pop_for(1 << 1), Some(p1c));
    assert_eq!(q.pop_for(1 << 2), Some(p2b));
    assert!(q.is_empty());
}

#[test]
fn pop_with_intersecting_mask() {
    // Multi-bit hart_mask (e.g. when scheduler is asking "any of these
    // harts"). Affinity bit just needs to overlap.
    let mut t = make_ready();
    set_affinity(&mut t, 1 << 5);
    let tp: *mut _ = &mut *t;
    let mut q = ReadyQueue::new();
    q.push(tp);

    let mask = (1 << 1) | (1 << 5) | (1 << 7);
    assert_eq!(q.pop_for(mask), Some(tp));
}

#[test]
fn affinity_changes_visible_to_subsequent_pop() {
    // Mirrors what set_affinity does in the live syscall — caller
    // mutates Thread.affinity while the entry is in queue. Next
    // pop_for must see the new mask.
    let mut t = make_ready();
    set_affinity(&mut t, 1 << 0);
    let tp: *mut _ = &mut *t;
    let mut q = ReadyQueue::new();
    q.push(tp);

    // hart 1 doesn't match yet.
    assert_eq!(q.pop_for(1 << 1), None);

    // Re-pin to hart 1.
    t.affinity.store(1 << 1, Ordering::Release);
    assert_eq!(q.pop_for(1 << 1), Some(tp));
}
