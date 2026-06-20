//! `Thread::try_mark_assigned` — the checked `Ready → Assigned` edge the
//! scheduler's `assign_thread_to` now uses (finding #5).
//!
//! Replaces the unconditional `mark_assigned` on the assign path: a thread
//! popped off the ready queue might have been killed while queued (Exited)
//! or, in a bug, be in some other state. The checked verb transitions only
//! a `Ready` thread, refuses an `Exited` one (the benign kill race — never
//! publish a dead thread to a hart's `current`), and panics on anything
//! else (a genuine logic error).

mod common;

use std::sync::atomic::Ordering;

use common::make_thread;
use process::ThreadState;
use riscv::register::sstatus::SPP;

fn state_of(t: &process::Thread) -> usize {
    t.state_load(Ordering::Acquire)
}

#[test]
fn ready_transitions_to_assigned() {
    let t = make_thread(ThreadState::Ready, SPP::User);
    assert!(t.try_mark_assigned(), "Ready -> Assigned returns true");
    assert_eq!(state_of(&t), ThreadState::Assigned as usize);
}

#[test]
fn exited_is_refused_not_assigned() {
    // Killed while queued: do not publish a dead thread, and do not panic.
    let t = make_thread(ThreadState::Exited, SPP::User);
    assert!(!t.try_mark_assigned(), "Exited -> (refused) returns false");
    assert_eq!(state_of(&t), ThreadState::Exited as usize);
}

#[test]
#[should_panic(expected = "illegal Ready->Assigned")]
fn running_panics() {
    // A Running thread should never be in the ready queue; popping one is a
    // logic bug worth surfacing loudly.
    let t = make_thread(ThreadState::Running, SPP::User);
    let _ = t.try_mark_assigned();
}
