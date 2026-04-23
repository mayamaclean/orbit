mod common;

use std::sync::atomic::Ordering;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_core::trap;

use common::{make_frame, make_thread};

/// User thread, user trap → snapshot proceeds.
#[test]
fn user_thread_user_trap_snapshots() {
    let t = make_thread(ThreadState::Running, SPP::User);
    t.pc.store(0xDEAD, Ordering::Release);
    t.frame.regs[10] = 0xAA;

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;
    frame.regs[11] = 0xCC;

    trap::update_trap_frame(&t, 0x1000, &mut frame, /* from_user = */ true);

    assert_eq!(frame.asid, t.pid as usize);
    // pc should have been advanced to the new epc.
    assert_eq!(t.pc.load(Ordering::Acquire), 0x1000);
    // frame snapshot should be visible on thread.frame.
    assert_eq!(t.frame.regs[10], 0xBB);
    assert_eq!(t.frame.regs[11], 0xCC);
}

/// User thread, S-mode trap (async interrupt during context switch) →
/// asid still set, but pc and frame must not move. This is the bug the
/// mode gate exists to prevent (see docs/trap-mode-guard.md).
#[test]
fn user_thread_s_mode_trap_does_not_snapshot() {
    let t = make_thread(ThreadState::Running, SPP::User);
    t.pc.store(0xDEAD, Ordering::Release);
    t.frame.regs[10] = 0xAA;

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&t, 0x1000, &mut frame, /* from_user = */ false);

    assert_eq!(frame.asid, t.pid as usize);
    assert_eq!(t.pc.load(Ordering::Acquire), 0xDEAD, "pc must not move");
    assert_eq!(t.frame.regs[10], 0xAA, "frame must not be overwritten");
}

/// Supervisor thread (k_net), S-mode trap → snapshot proceeds.
#[test]
fn supervisor_thread_s_mode_trap_snapshots() {
    let t = make_thread(ThreadState::Running, SPP::Supervisor);
    t.pc.store(0xDEAD, Ordering::Release);

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&t, 0x2000, &mut frame, /* from_user = */ false);

    assert_eq!(t.pc.load(Ordering::Acquire), 0x2000);
    assert_eq!(t.frame.regs[10], 0xBB);
}

/// Supervisor thread, user trap (shouldn't happen in practice, but the
/// mode gate still rejects) → no snapshot.
#[test]
fn supervisor_thread_user_trap_does_not_snapshot() {
    let t = make_thread(ThreadState::Running, SPP::Supervisor);
    t.pc.store(0xDEAD, Ordering::Release);
    t.frame.regs[10] = 0xAA;

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&t, 0x2000, &mut frame, /* from_user = */ true);

    assert_eq!(t.pc.load(Ordering::Acquire), 0xDEAD);
    assert_eq!(t.frame.regs[10], 0xAA);
}

/// asid is written even if the gate rejects — post-trap kernel work on this
/// hart depends on it being set.
#[test]
fn asid_always_updated() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = 42;

    let mut frame = make_frame();
    frame.asid = 0;

    trap::update_trap_frame(&t, 0x1000, &mut frame, /* from_user = */ false);

    assert_eq!(frame.asid, 42);
}

/// Ready/Assigned/Exited states don't snapshot — only Running/Suspended/Blocking do.
#[test]
fn non_runnable_state_skips_snapshot() {
    let t = make_thread(ThreadState::Ready, SPP::User);
    t.pc.store(0xDEAD, Ordering::Release);
    t.frame.regs[10] = 0xAA;

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&t, 0x1000, &mut frame, /* from_user = */ true);

    assert_eq!(t.pc.load(Ordering::Acquire), 0xDEAD);
    assert_eq!(t.frame.regs[10], 0xAA);
}
