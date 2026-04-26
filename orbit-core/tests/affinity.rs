mod common;

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use process::{Thread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{Errno, EINVAL, EPERM};
use orbit_core::sched::{HartView, Scheduler, assign_threads};
use orbit_core::{SyscallOutcome, syscall};

use common::{FakeHw, make_frame, make_thread};

// =====================================================================
// set_affinity / get_affinity
// =====================================================================

#[test]
fn set_affinity_narrows_within_cap() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b1111;
    t.affinity = AtomicU64::new(0b1111);

    let mut f = make_frame();
    f.regs[11] = 0b0010;

    let outcome = syscall::set_affinity(&t, &f);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(0) }
    );
    assert_eq!(t.affinity.load(Ordering::Acquire), 0b0010);
    // Cap is immutable.
    assert_eq!(t.allowed_affinity, 0b1111);
}

#[test]
fn set_affinity_zero_mask_is_einval() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b1111;
    t.affinity = AtomicU64::new(0b1111);

    let mut f = make_frame();
    f.regs[11] = 0;

    let outcome = syscall::set_affinity(&t, &f);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(Errno::new(EINVAL).to_ret()) }
    );
    // Reject path leaves the mask untouched — checked via the cap-immutable
    // semantics of allowed_affinity and the unchanged current value.
    assert_eq!(t.affinity.load(Ordering::Acquire), 0b1111);
}

#[test]
fn set_affinity_outside_allowed_is_eperm() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b0011;
    t.affinity = AtomicU64::new(0b0001);

    let mut f = make_frame();
    f.regs[11] = 0b0100; // bit outside the cap

    let outcome = syscall::set_affinity(&t, &f);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(Errno::new(EPERM).to_ret()) }
    );
    assert_eq!(t.affinity.load(Ordering::Acquire), 0b0001);
}

#[test]
fn set_affinity_partial_overlap_outside_cap_is_eperm() {
    // Even one stray bit outside the cap rejects the whole mask;
    // we don't silently mask the user's request.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b0011;
    t.affinity = AtomicU64::new(0b0001);

    let mut f = make_frame();
    f.regs[11] = 0b0111; // bits 0,1 in cap; bit 2 outside

    let outcome = syscall::set_affinity(&t, &f);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(Errno::new(EPERM).to_ret()) }
    );
}

#[test]
fn set_affinity_to_full_cap_is_ok() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b0011;
    t.affinity = AtomicU64::new(0b0001);

    let mut f = make_frame();
    f.regs[11] = 0b0011;

    let outcome = syscall::set_affinity(&t, &f);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(0) }
    );
    assert_eq!(t.affinity.load(Ordering::Acquire), 0b0011);
}

#[test]
fn get_affinity_returns_current_and_allowed_in_a0_a1() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b1111;
    t.affinity = AtomicU64::new(0b0010);

    let outcome = syscall::get_affinity(&t);

    assert_eq!(
        outcome,
        SyscallOutcome::Return2 { ret0: 0b0010, ret1: 0b1111 }
    );
}

#[test]
fn get_affinity_does_not_mutate() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0b0101;
    t.affinity = AtomicU64::new(0b0100);

    let _ = syscall::get_affinity(&t);

    assert_eq!(t.affinity.load(Ordering::Acquire), 0b0100);
    assert_eq!(t.allowed_affinity, 0b0101);
}

// =====================================================================
// Scheduler dispatch respects affinity
// =====================================================================

/// Vec-backed scheduler that mirrors the kernel impl: each call does a
/// fresh full scan and returns the first not-yet-handed-out thread
/// compatible with the requested `hart_mask`. The kernel uses
/// `ThreadState::Assigned` to filter handed-out threads on subsequent
/// scans; this mock uses a `taken` bitmap. Critical that we *don't*
/// advance a `next` cursor past affinity-incompatible threads — a
/// thread refused for hart 1 must still be visible to hart 2's scan.
struct AffSched {
    threads: Vec<Thread>,
    taken: Vec<bool>,
}

impl AffSched {
    fn new(threads: Vec<Thread>) -> Self {
        let n = threads.len();
        Self { threads, taken: vec![false; n] }
    }
}

impl Scheduler for AffSched {
    fn next_runnable(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        for i in 0..self.threads.len() {
            if self.taken[i] {
                continue;
            }
            let aff = self.threads[i].affinity.load(Ordering::Relaxed);
            if aff & hart_mask != 0 {
                self.taken[i] = true;
                return Some(unsafe { self.threads.as_mut_ptr().add(i) });
            }
        }
        None
    }
}

fn make_slots(n: usize) -> Vec<AtomicPtr<()>> {
    (0..n).map(|_| AtomicPtr::new(null_mut())).collect()
}

fn views<'a>(slots: &'a [AtomicPtr<()>]) -> (HartView<'a>, Vec<HartView<'a>>) {
    let self_view = HartView { hart_id: 0, current: &slots[0] };
    let remotes = slots[1..]
        .iter()
        .enumerate()
        .map(|(i, slot)| HartView {
            hart_id: (i + 1) as u32,
            current: slot,
        })
        .collect();
    (self_view, remotes)
}

#[test]
fn affinity_pinned_thread_only_lands_on_permitted_hart() {
    // 4 harts (0..3); one thread pinned to hart 2 only.
    let mut t = make_thread(ThreadState::Ready, SPP::User);
    t.tid = 42;
    t.allowed_affinity = 0b0100;
    t.affinity = AtomicU64::new(0b0100);

    let mut sched = AffSched::new(vec![t]);
    let slots = make_slots(4);
    let (self_view, remotes) = views(&slots);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remotes, &mut sched, &mut hw);

    // Only hart 2's slot should be populated.
    assert!(slots[0].load(Ordering::Acquire).is_null(), "self (hart 0) untouched");
    assert!(slots[1].load(Ordering::Acquire).is_null(), "hart 1 untouched");
    assert!(!slots[2].load(Ordering::Acquire).is_null(), "hart 2 received the thread");
    assert!(slots[3].load(Ordering::Acquire).is_null(), "hart 3 untouched");

    // And hart 2 specifically got the IPI.
    assert_eq!(hw.wakes, vec![2]);
}

#[test]
fn restrictive_thread_does_not_starve_unrelated_harts() {
    // Two threads: t1 pinned to hart 3, t2 runnable anywhere. With the
    // gate at the right place, hart 1 and hart 2 still pick up t2 even
    // though t1 (queue head) refused them.
    let mut t1 = make_thread(ThreadState::Ready, SPP::User);
    t1.tid = 1;
    t1.allowed_affinity = 0b1000;
    t1.affinity = AtomicU64::new(0b1000);

    let mut t2 = make_thread(ThreadState::Ready, SPP::User);
    t2.tid = 2;
    // any-hart default

    let mut sched = AffSched::new(vec![t1, t2]);
    let slots = make_slots(4);
    let (self_view, remotes) = views(&slots);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remotes, &mut sched, &mut hw);

    // Hart 1 tries first, t1 doesn't fit, so the scheduler advances to t2,
    // which lands on hart 1. Hart 2 then has no compatible thread (t1 is
    // bit-3 only). Hart 3 picks up t1.
    assert!(slots[0].load(Ordering::Acquire).is_null());
    assert!(!slots[1].load(Ordering::Acquire).is_null(), "hart 1 got t2");
    assert!(slots[2].load(Ordering::Acquire).is_null(), "hart 2 has no compatible work");
    assert!(!slots[3].load(Ordering::Acquire).is_null(), "hart 3 got t1");

    // IPI count: harts 1 and 3.
    assert_eq!(hw.wakes.len(), 2);
    assert!(hw.wakes.contains(&1));
    assert!(hw.wakes.contains(&3));
}

#[test]
fn no_compatible_hart_leaves_thread_unassigned() {
    // Only hart 7 is permitted, but the runtime only has harts 0..3.
    // No assignment, no IPI, no panic.
    let mut t = make_thread(ThreadState::Ready, SPP::User);
    t.allowed_affinity = 1u64 << 7;
    t.affinity = AtomicU64::new(1u64 << 7);

    let mut sched = AffSched::new(vec![t]);
    let slots = make_slots(4);
    let (self_view, remotes) = views(&slots);
    let mut hw = FakeHw::default();

    assign_threads(&self_view, remotes, &mut sched, &mut hw);

    for s in &slots {
        assert!(s.load(Ordering::Acquire).is_null());
    }
    assert!(hw.wakes.is_empty());
}
