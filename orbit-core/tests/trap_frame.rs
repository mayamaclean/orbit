mod common;


use process::{RunningThread, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::trap;

use common::{make_frame, make_thread};

/// User thread, user trap → snapshot proceeds.
#[test]
fn user_thread_user_trap_snapshots() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(0xDEAD);
    r.set_frame_reg(10, 0xAA);

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;
    frame.regs[11] = 0xCC;

    trap::update_trap_frame(&mut r, 0x1000, &mut frame, /* from_user = */ true);

    assert_eq!(frame.asid, r.view().pid() as usize);
    // pc should have been advanced to the new epc.
    assert_eq!(r.view().pc(), 0x1000);
    // frame snapshot should be visible on thread.frame.
    assert_eq!(r.frame_reg(10), 0xBB);
    assert_eq!(r.frame_reg(11), 0xCC);
}

/// User thread, S-mode trap (async interrupt during context switch) →
/// asid still set, but pc and frame must not move. This is the bug the
/// mode gate exists to prevent (see docs/trap-mode-guard.md).
#[test]
fn user_thread_s_mode_trap_does_not_snapshot() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(0xDEAD);
    r.set_frame_reg(10, 0xAA);

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&mut r, 0x1000, &mut frame, /* from_user = */ false);

    assert_eq!(frame.asid, r.view().pid() as usize);
    assert_eq!(r.view().pc(), 0xDEAD, "pc must not move");
    assert_eq!(r.frame_reg(10), 0xAA, "frame must not be overwritten");
}

/// Supervisor thread (k_net), S-mode trap → snapshot proceeds.
#[test]
fn supervisor_thread_s_mode_trap_snapshots() {
    let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(0xDEAD);

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&mut r, 0x2000, &mut frame, /* from_user = */ false);

    assert_eq!(r.view().pc(), 0x2000);
    assert_eq!(r.frame_reg(10), 0xBB);
}

/// Supervisor thread, user trap (shouldn't happen in practice, but the
/// mode gate still rejects) → no snapshot.
#[test]
fn supervisor_thread_user_trap_does_not_snapshot() {
    let mut t = make_thread(ThreadState::Running, SPP::Supervisor);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(0xDEAD);
    r.set_frame_reg(10, 0xAA);

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&mut r, 0x2000, &mut frame, /* from_user = */ true);

    assert_eq!(r.view().pc(), 0xDEAD);
    assert_eq!(r.frame_reg(10), 0xAA);
}

/// asid is written even if the gate rejects — post-trap kernel work on this
/// hart depends on it being set.
#[test]
fn asid_always_updated() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = 42;
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };

    let mut frame = make_frame();
    frame.asid = 0;

    trap::update_trap_frame(&mut r, 0x1000, &mut frame, /* from_user = */ false);

    assert_eq!(frame.asid, 42);
}

/// Ready/Assigned/Exited states don't snapshot — only Running/Suspended/Blocking do.
#[test]
fn non_runnable_state_skips_snapshot() {
    let mut t = make_thread(ThreadState::Ready, SPP::User);
    let mut r = unsafe { RunningThread::from_ptr(&mut t) };
    r.set_pc(0xDEAD);
    r.set_frame_reg(10, 0xAA);

    let mut frame = make_frame();
    frame.regs[10] = 0xBB;

    trap::update_trap_frame(&mut r, 0x1000, &mut frame, /* from_user = */ true);

    assert_eq!(r.view().pc(), 0xDEAD);
    assert_eq!(r.frame_reg(10), 0xAA);
}

/// Exhaustive `ThreadState × mode × from_user` matrix. Asserts the
/// snapshot decision matches the documented rule:
///   gate passes  = (mode == User) == from_user
///   snapshots if = gate && state ∈ {Running, Suspended, Blocking}
/// asid is always written regardless of gate.
#[test]
fn state_mode_from_user_matrix_is_exhaustive() {
    const STATES: &[(ThreadState, bool)] = &[
        (ThreadState::Ready, false),
        (ThreadState::Blocking, true),
        (ThreadState::Assigned, false),
        (ThreadState::Running, true),
        (ThreadState::Exited, false),
        (ThreadState::Suspended, true),
    ];
    const MODES: &[SPP] = &[SPP::User, SPP::Supervisor];
    const FROM_USER: &[bool] = &[true, false];

    for &(state, is_snapshot_state) in STATES {
        for &mode in MODES {
            for &from_user in FROM_USER {
                let mut t = make_thread(state, mode);
                t.pid = 9;
                let mut r = unsafe { RunningThread::from_ptr(&mut t) };
                r.set_pc(0xDEAD);
                r.set_frame_reg(10, 0xAA);

                let mut frame = make_frame();
                frame.regs[10] = 0xBB;
                frame.asid = 0;

                trap::update_trap_frame(&mut r, 0x2000, &mut frame, from_user);

                // asid is always set (even when gate rejects) — post-trap
                // kernel work on this hart depends on it.
                assert_eq!(
                    frame.asid, 9,
                    "asid must be written unconditionally (state={state:?}, mode={mode:?}, from_user={from_user})"
                );

                let gate_passes = (mode == SPP::User) == from_user;
                let expected_snapshot = gate_passes && is_snapshot_state;

                if expected_snapshot {
                    assert_eq!(
                        r.view().pc(),
                        0x2000,
                        "pc should advance (state={state:?}, mode={mode:?}, from_user={from_user})"
                    );
                    assert_eq!(
                        r.frame_reg(10),
                        0xBB,
                        "frame should snapshot (state={state:?}, mode={mode:?}, from_user={from_user})"
                    );
                }
                else {
                    assert_eq!(
                        r.view().pc(),
                        0xDEAD,
                        "pc must NOT move (state={state:?}, mode={mode:?}, from_user={from_user})"
                    );
                    assert_eq!(
                        r.frame_reg(10),
                        0xAA,
                        "frame must NOT be overwritten (state={state:?}, mode={mode:?}, from_user={from_user})"
                    );
                }
            }
        }
    }
}
