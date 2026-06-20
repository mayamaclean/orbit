//! Proves `Thread::transition_to`'s edge validation has teeth — and that
//! the validation is **always compiled** (release kernel included), not the
//! old debug-only `debug_assert!`.
//!
//! The contract:
//!   * Legal edges (the exact production set) pass.
//!   * A genuinely-illegal, non-racy edge (un-gated `parked → Ready`, a
//!     skipped dispatch step) **panics** — louder than a silent corrupt
//!     store, in every build.
//!   * `Exited` is terminal: a redundant `Exited → Exited` (the reaper
//!     re-running) and an attempted `Exited → runnable` (the cross-hart
//!     kill race — a sibling marked Exited between assign/dispatch and this
//!     store) are *refused, not panicked* — the thread stays Exited and is
//!     reaped. Panicking there would crash the kernel on a legitimate
//!     concurrent exit.
//!
//! Not gated on `debug_assertions` anymore: the check now runs in release,
//! so these assertions hold under both `cargo test` and `cargo test
//! --release`. Fixtures stand states up with the feature-gated
//! `transition_to_unchecked`; the assertions exercise the *checked* path.

mod common;

use std::sync::atomic::Ordering;

use common::make_thread;
use process::ThreadState;
use riscv::register::sstatus::SPP;

fn state_of(t: &process::Thread) -> usize {
    t.state_load(Ordering::Acquire)
}

// ── legal edges: the exact production set ──────────────────────────────

#[test]
fn ready_to_assigned_ok() {
    let t = make_thread(ThreadState::Ready, SPP::User);
    t.mark_assigned(); // Ready -> Assigned
    assert_eq!(state_of(&t), ThreadState::Assigned as usize);
}

#[test]
fn assigned_to_running_ok() {
    let t = make_thread(ThreadState::Assigned, SPP::User);
    t.mark_running(); // Assigned -> Running
    assert_eq!(state_of(&t), ThreadState::Running as usize);
}

#[test]
fn running_departs_to_each_park_state_ok() {
    for to in [
        ThreadState::Ready, // own-hart yield
        ThreadState::Blocking,
        ThreadState::Suspended,
        ThreadState::Exited,
    ] {
        let t = make_thread(ThreadState::Running, SPP::User);
        t.transition_to(to);
        assert_eq!(state_of(&t), to as usize);
    }
}

// ── Exited is terminal: refused, never panicked, never resurrected ─────

#[test]
fn exited_to_runnable_is_refused_not_panicked() {
    // The cross-hart kill race: a thread is marked Exited (the un-gated
    // kill store) between this hart's assign/dispatch and a `transition_to`
    // that expected it live. transition_to must refuse the resurrection and
    // leave it Exited — NOT panic (that would crash the kernel on a
    // legitimate concurrent exit) and NOT store Running (the old
    // release-only resurrection-loop bug).
    let t = make_thread(ThreadState::Exited, SPP::User);
    t.transition_to(ThreadState::Running);
    assert_eq!(
        state_of(&t),
        ThreadState::Exited as usize,
        "Exited must stay Exited (no resurrection)",
    );
}

#[test]
fn exited_to_exited_is_noop() {
    // Redundant reap (check_context_and_switch re-running on an already
    // dead thread) must be a quiet no-op, not a panic.
    let t = make_thread(ThreadState::Exited, SPP::User);
    t.transition_to(ThreadState::Exited);
    assert_eq!(state_of(&t), ThreadState::Exited as usize);
}

// ── illegal, non-racy edges: must panic (release included) ─────────────

#[test]
#[should_panic(expected = "illegal Thread state transition")]
fn parked_to_ready_is_rejected() {
    // A Suspended sleeper reaching Ready must mint a Runnable through
    // ParkedMut::promote_wake (promote_ready_from_parked) — never the
    // generic setter (bug-4 gate).
    let t = make_thread(ThreadState::Suspended, SPP::User);
    t.transition_to(ThreadState::Ready);
}

#[test]
#[should_panic(expected = "illegal Thread state transition")]
fn ready_to_running_skips_assigned() {
    // Dispatch must go Ready -> Assigned -> Running; the middle step is
    // not skippable.
    let t = make_thread(ThreadState::Ready, SPP::User);
    t.mark_running();
}

// `try_mark_assigned` (the checked assign edge, #5) is exercised in
// `assign_checked.rs` — kept separate so this file compiles against the
// pre-fix tree for the red/green demonstration.
