//! Reproduces the QEMU race that corrupted knet's `pc`. The host model:
//!
//! 1. A "hart slot" (`AtomicPtr<()>`) initially points at a User thread.
//! 2. A worker reads the slot, then calls `apply_syscall_outcome` on
//!    whatever it observes — emulating kmain's `dispatch_syscall`.
//! 3. A retargeter, possibly racing the worker, swaps the slot's
//!    pointer to a kthread.
//!
//! The invariants we want to hold:
//! - When the worker's read returns the User thread, the outcome
//!   commits as expected.
//! - When the worker's read returns the kthread, the gate in
//!   `apply_syscall_outcome` short-circuits and the kthread's `pc` /
//!   frame remain untouched.
//!
//! Without the gate (the pre-fix shape), the kthread's `pc` would
//! land at `epc + 4` and the next dispatch would `sret` to a user VA
//! in S-mode — exactly the cause=12 panic in `arm_hart_timer14.log`.

mod common;

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread;

use device::TrapFrame;
use process::{RunningThread, Thread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::{ShimAction, SyscallOutcome, apply_syscall_outcome};

use common::make_thread;

const ECALL_EPC: usize = 0x2_2000_3398;
const KTHREAD_PC: usize = 0xFFFF_FFC0_0002_0118;
const USER_PC: usize = 0x2_2000_0554;

/// Drive the dispatch_syscall pattern against whatever pointer is
/// currently in the slot. Returns `Some(action)` if a thread was
/// observed, `None` if the slot was null.
unsafe fn dispatch_against_slot(slot: &AtomicPtr<()>, epc: usize) -> Option<ShimAction> {
    let p = slot.load(Ordering::Acquire);
    if p.is_null() {
        return None;
    }
    // SAFETY: the slot holds a live, boxed Thread for the test's
    // duration; this fixture is the one place we model "the hart's
    // current ptr" so minting the own-hart cap is the faithful analog.
    let mut r = unsafe { RunningThread::from_ptr(p as *mut Thread) };
    // The fixture only asserts pc/state, so a blank trap-entry frame is
    // sufficient (the gate-reject paths don't write it; the commit path
    // snapshots it but no test inspects the snapshot here).
    let mut frame = TrapFrame::empty();
    // A Suspended park (resumes with ret 0, pc advanced). The fixture
    // models a generic U-ecall whose outcome the gate must reject when
    // `current` was retargeted to a kthread.
    Some(apply_syscall_outcome(
        SyscallOutcome::SleepUntil { deadline: 0 },
        &mut r,
        &mut frame,
        epc,
    ))
}

#[test]
fn user_thread_in_slot_commits_normally() {
    let user = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Running, SPP::User);
        unsafe { RunningThread::from_ptr(&mut t) }.set_pc(USER_PC);
        t
    }));

    let slot = AtomicPtr::new(user as *mut ());

    let action = unsafe { dispatch_against_slot(&slot, ECALL_EPC) };

    let user_ref = unsafe { &*user };
    assert_eq!(action, Some(ShimAction::Yield(ThreadState::Suspended)));
    assert_eq!(user_ref.pc_load(Ordering::Acquire), ECALL_EPC + 4);

    unsafe { drop(Box::from_raw(user)) };
}

#[test]
fn slot_retargeted_to_kthread_does_not_corrupt() {
    // Sequential analog of the QEMU race: the trap was supposed to land
    // on a User thread, but by the time the dispatch shim reads `slot`
    // it's been swapped to a kthread. Without the gate this stamps
    // ECALL_EPC+4 into the kthread's pc.
    let kthread = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
        unsafe { RunningThread::from_ptr(&mut t) }.set_pc(KTHREAD_PC);
        t.tid = 1; // matches the QEMU repro: knet was tid=1
        t
    }));
    let _ = kthread; // silence unused mut

    let slot = AtomicPtr::new(kthread as *mut ());

    let action = unsafe { dispatch_against_slot(&slot, ECALL_EPC) };

    // Gate fires → Resume returned (not Yield), pc unchanged.
    assert_eq!(action, Some(ShimAction::Resume));
    let kthread_ref = unsafe { &*kthread };
    assert_eq!(
        kthread_ref.pc_load(Ordering::Acquire),
        KTHREAD_PC,
        "kthread.pc must not be stamped with a user epc",
    );
    assert_eq!(
        kthread_ref.state_load(Ordering::Acquire),
        ThreadState::Running as usize,
        "kthread.state must not transition to Suspended via the spurious Yield",
    );

    unsafe { drop(Box::from_raw(kthread)) };
}

#[test]
fn concurrent_retarget_never_corrupts_kthread() {
    // The race itself: a retargeter thread keeps swapping the slot
    // between User and kthread pointers while a worker repeatedly
    // dispatches whatever it observes. After many iterations, the
    // kthread's pc must never have been corrupted into the user range,
    // regardless of who won which race.
    const ITERS: usize = 200;

    let user = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Running, SPP::User);
        unsafe { RunningThread::from_ptr(&mut t) }.set_pc(USER_PC);
        t
    }));
    let kthread = Box::into_raw(Box::new({
        let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
        unsafe { RunningThread::from_ptr(&mut t) }.set_pc(KTHREAD_PC);
        t
    }));

    let slot = AtomicPtr::new(user as *mut ());

    // Wrap raw pointers in a Send-able newtype with method access.
    // 2021-edition disjoint capture would pull `kthread_p.0` (a raw
    // pointer, not Send) into the closure even though the wrapper is
    // Send; routing through a method forces the closure to capture
    // the whole `SendPtr` value. Pointers come straight from
    // `Box::into_raw` allocations so provenance is preserved for
    // miri's strict-provenance + Tree Borrows checks.
    #[derive(Copy, Clone)]
    struct SendPtr(*mut ());
    unsafe impl Send for SendPtr {}
    impl SendPtr {
        fn raw(self) -> *mut () {
            self.0
        }
    }

    let user_p = SendPtr(user as *mut ());
    let kthread_p = SendPtr(kthread as *mut ());

    thread::scope(|s| {
        let slot_ref = &slot;
        // Retargeter: alternate the slot.
        s.spawn(move || {
            for i in 0..ITERS {
                let target = if i % 2 == 0 {
                    kthread_p.raw()
                }
                else {
                    user_p.raw()
                };
                slot_ref.store(target, Ordering::Release);
                thread::yield_now();
            }
        });

        // Worker: dispatch repeatedly. Each call commits against
        // whichever thread it observes. The User commits update
        // user.pc; the kthread observations are gated.
        s.spawn(move || {
            for _ in 0..ITERS {
                let _ = unsafe { dispatch_against_slot(slot_ref, ECALL_EPC) };
                thread::yield_now();
            }
        });
    });

    // Final state: user.pc may be ECALL_EPC+4 (most-recent commit) or
    // its initial USER_PC (if no User-observation commits ran), but
    // kthread.pc must equal KTHREAD_PC unchanged.
    let kthread_ref = unsafe { &*kthread };
    assert_eq!(
        kthread_ref.pc_load(Ordering::Acquire),
        KTHREAD_PC,
        "kthread.pc must be untouched after {ITERS} contended dispatch passes",
    );
    assert_eq!(
        kthread_ref.state_load(Ordering::Acquire),
        ThreadState::Running as usize,
    );

    unsafe { drop(Box::from_raw(user)) };
    unsafe { drop(Box::from_raw(kthread)) };
}

#[test]
fn null_slot_is_handled_cleanly() {
    // dispatch_syscall's null guard: if `current` is null, return
    // without touching anything. No thread to commit to, no panic.
    let slot: AtomicPtr<()> = AtomicPtr::new(null_mut());
    let action = unsafe { dispatch_against_slot(&slot, ECALL_EPC) };
    assert!(action.is_none());
}
