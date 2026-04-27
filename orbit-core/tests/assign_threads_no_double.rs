//! Pin down the no-double-assignment invariant: once `assign_threads`
//! places a thread into a hart's `current` slot and transitions its
//! state to `Assigned`, no subsequent `assign_threads` pass on a
//! *different* set of hart views may pick the same thread again.
//!
//! This is the host-side reproducer for the QEMU panic where knet
//! ended up as `current` on cpu2 while a User thread was simultaneously
//! current on cpu1 — the symptom of either a stale state read in the
//! scheduler or a TOCTOU between `is_busy()` and `assign_thread_to`.
//! Uses a state-aware scheduler that mirrors `Orbit::get_runnable_thread`'s
//! real filtering (Ready → return; everything else → continue).

mod common;

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use process::{Thread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::sched::{HartView, Scheduler, assign_threads};

use common::{FakeHw, make_thread};

/// Scheduler that walks a fixed thread list and returns the first one
/// in state `Ready` whose affinity admits `hart_mask`. Mirrors the
/// production behavior in `Orbit::get_runnable_thread` closely enough
/// to expose state-transition bugs without dragging the kmain
/// dependencies in.
struct StateAwareSched {
    threads: Vec<*mut Thread>,
}

impl Scheduler for StateAwareSched {
    fn next_runnable(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        for &p in &self.threads {
            let t = unsafe { &*p };
            if t.affinity.load(Ordering::Relaxed) & hart_mask == 0 {
                continue;
            }
            if t.state.load(Ordering::Acquire) == ThreadState::Ready as usize {
                return Some(p);
            }
        }
        None
    }
}

#[test]
fn second_pass_skips_already_assigned_thread() {
    // One Ready user thread, two consecutive assign_threads passes from
    // different self-views. The first publishes it to a remote; the
    // second must not re-publish it anywhere.
    let t1 = Box::into_raw(Box::new(make_thread(ThreadState::Ready, SPP::User)));

    let mut sched = StateAwareSched { threads: vec![t1] };

    let slot_self_a = AtomicPtr::new(null_mut());
    let slot_b = AtomicPtr::new(null_mut());
    let slot_self_c = AtomicPtr::new(null_mut());
    let slot_d = AtomicPtr::new(null_mut());
    let mut hw = FakeHw::default();

    // Pass 1: hart A is the manager. Picks thread for remote B.
    let self_a = HartView { hart_id: 0, current: &slot_self_a };
    let remote_b = HartView { hart_id: 1, current: &slot_b };
    assign_threads(&self_a, [remote_b], &mut sched, &mut hw);

    // Thread should be sitting on slot_b with state=Assigned.
    assert_eq!(slot_b.load(Ordering::Acquire) as *mut Thread, t1);
    let t1_ref = unsafe { &*t1 };
    assert_eq!(
        t1_ref.state.load(Ordering::Acquire),
        ThreadState::Assigned as usize,
    );

    // Pass 2: a different hart C becomes manager (B is now busy).
    // Remote D is idle. Even though next_runnable would return *some*
    // thread if it existed, the only thread is in state=Assigned, so
    // the StateAwareSched correctly returns None and nothing gets
    // double-published.
    let self_c = HartView { hart_id: 2, current: &slot_self_c };
    let remote_d = HartView { hart_id: 3, current: &slot_d };
    assign_threads(&self_c, [remote_d], &mut sched, &mut hw);

    assert!(
        slot_d.load(Ordering::Acquire).is_null(),
        "remote D must not receive an Assigned thread"
    );
    assert!(
        slot_self_c.load(Ordering::Acquire).is_null(),
        "self C must not receive an Assigned thread"
    );

    unsafe { drop(Box::from_raw(t1)) };
}

#[test]
fn busy_remote_protects_against_overwrite() {
    // The real-world race: hart B already has a thread running. The
    // manager on hart A must observe `is_busy() == true` and skip,
    // even if next_runnable would otherwise hand out a fresh thread.
    // Without this gate, hart A's write would clobber hart B's `current`
    // pointer mid-trap on B and the next dispatch on B would commit
    // syscall outcomes onto the wrong thread.
    let busy_thread = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Running, SPP::User);
        t.tid = 50;
        t
    }));
    let ready_thread = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Ready, SPP::User);
        t.tid = 51;
        t
    }));

    let mut sched = StateAwareSched { threads: vec![ready_thread] };

    let slot_self = AtomicPtr::new(null_mut());
    // Hart B is already running busy_thread — its current points there.
    let slot_b = AtomicPtr::new(busy_thread as *mut ());
    let mut hw = FakeHw::default();

    let self_view = HartView { hart_id: 0, current: &slot_self };
    let remote_b = HartView { hart_id: 1, current: &slot_b };
    assign_threads(&self_view, [remote_b], &mut sched, &mut hw);

    // Hart B's current must STILL be busy_thread — no overwrite.
    assert_eq!(
        slot_b.load(Ordering::Acquire) as *mut Thread,
        busy_thread,
        "is_busy() gate must protect a non-null `current` from overwrite",
    );
    // ready_thread was free for the self-view to pick up.
    assert_eq!(
        slot_self.load(Ordering::Acquire) as *mut Thread,
        ready_thread,
    );
    let ready_ref = unsafe { &*ready_thread };
    assert_eq!(
        ready_ref.state.load(Ordering::Acquire),
        ThreadState::Assigned as usize,
    );

    unsafe { drop(Box::from_raw(busy_thread)) };
    unsafe { drop(Box::from_raw(ready_thread)) };
}

#[test]
fn affinity_pin_blocks_unwanted_assignment() {
    // Affinity is the per-thread bound on which harts can run it. A
    // thread pinned to hart 0 must not be assigned to hart 2 even if
    // hart 2 is idle. This is structurally what protects user threads
    // from migrating onto kthreads' assigned harts in mixed scheduling.
    let pinned = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Ready, SPP::User);
        // Hart 0 only.
        t.affinity.store(1u64 << 0, Ordering::Release);
        t.allowed_affinity = 1u64 << 0;
        t
    }));

    let mut sched = StateAwareSched { threads: vec![pinned] };

    let slot_self = AtomicPtr::new(null_mut()); // hart 1 is self
    let slot_remote = AtomicPtr::new(null_mut()); // hart 2 is remote
    let mut hw = FakeHw::default();

    let self_view = HartView { hart_id: 1, current: &slot_self };
    let remote = HartView { hart_id: 2, current: &slot_remote };
    assign_threads(&self_view, [remote], &mut sched, &mut hw);

    assert!(
        slot_self.load(Ordering::Acquire).is_null(),
        "hart-0-pinned thread must not land on hart 1",
    );
    assert!(
        slot_remote.load(Ordering::Acquire).is_null(),
        "hart-0-pinned thread must not land on hart 2",
    );
    let pinned_ref = unsafe { &*pinned };
    assert_eq!(
        pinned_ref.state.load(Ordering::Acquire),
        ThreadState::Ready as usize,
        "rejected thread stays Ready",
    );

    unsafe { drop(Box::from_raw(pinned)) };
}

#[test]
fn no_runnables_leaves_all_slots_null() {
    // Sanity case: if every thread in the registry is Running/Assigned/
    // Suspended, assign_threads must not write *anywhere*. This catches
    // a bug where a buggy scheduler returns a non-Ready thread anyway
    // and the assign loop publishes it.
    let running = Box::into_raw(Box::new(make_thread(ThreadState::Running, SPP::User)));
    let assigned = Box::into_raw(Box::new(make_thread(ThreadState::Assigned, SPP::User)));
    let suspended = Box::into_raw(Box::new(make_thread(ThreadState::Suspended, SPP::User)));

    let mut sched = StateAwareSched { threads: vec![running, assigned, suspended] };

    let slots: [AtomicPtr<()>; 4] = std::array::from_fn(|_| AtomicPtr::new(null_mut()));
    let mut hw = FakeHw::default();

    let self_view = HartView { hart_id: 0, current: &slots[0] };
    let remotes: [HartView; 3] = std::array::from_fn(|i| HartView {
        hart_id: (i + 1) as u32,
        current: &slots[i + 1],
    });
    assign_threads(&self_view, remotes, &mut sched, &mut hw);

    for (i, s) in slots.iter().enumerate() {
        assert!(
            s.load(Ordering::Acquire).is_null(),
            "slot {i} should be null when no threads are Ready",
        );
    }

    unsafe { drop(Box::from_raw(running)) };
    unsafe { drop(Box::from_raw(assigned)) };
    unsafe { drop(Box::from_raw(suspended)) };
}
