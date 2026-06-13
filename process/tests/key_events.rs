//! Concurrency + lifecycle tests for [`process::ProcessKeyEvents`].
//!
//! Sister harness to `stdin.rs` — same hazards, different wake
//! mechanism. ProcessKeyEvents is the per-process structured-event
//! ring behind `read_key_event` (syscall 12): SPSC `KeyEvent` ring +
//! AtomicU32 parked-tid slot. The interesting window is park-vs-push:
//!
//! - Consumer is between `try_drain → 0` and `set_parker`. The
//!   producer's push lands but its `parked_tid.swap(0)` sees no
//!   parker, so no wake is issued. The consumer's post-park re-drain
//!   must observe the event (and cancel its park via
//!   `clear_parker_if`) or the read would sleep through delivered
//!   input.
//! - In the kernel the producer-side `push_event` returning
//!   `Some(tid)` makes `input::dispatch` push
//!   `WakeEvent::InputTid(tid)`. The host stand-in is "the event is
//!   in the ring, so the consumer's next try_drain succeeds" — same
//!   correctness story as stdin.rs.
//!
//! Also pins the three-way `set_parker` outcome contract
//! (`Installed` / `AlreadyOwned` / `Busy`) that the syscall handler's
//! timeout re-entry path depends on, and the silent-drop overflow
//! behavior of `push_event` on a full ring.
//!
//! Run under `./test miri-hammer` for seed/preemption fuzzing of the
//! park-vs-push window.

use std::sync::{Arc as StdArc, Barrier};
use std::thread;

use orbit_abi::input::KeyEvent;
use process::ProcessKeyEvents;
use process::key_events::{ParkOutcome, RING_CAP};

#[cfg(miri)]
const ITERS: usize = 5;
#[cfg(not(miri))]
const ITERS: usize = 200;

/// Events per concurrent iteration. Must stay ≤ RING_CAP: `push_event`
/// silently drops on a full ring, so a producer that outruns the
/// consumer by more than the ring would turn the no-loss assertion
/// into a flake instead of a finding.
#[cfg(miri)]
const EVENTS_PER_ITER: usize = 32;
#[cfg(not(miri))]
const EVENTS_PER_ITER: usize = 200;

const TID: u32 = 1;
const TID2: u32 = 2;

/// KeyEvent with a recognizable payload; only `code` carries the
/// sequence number, the rest mirrors what the kernel writes.
fn ev(code: u32) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: 0,
        kind: 0,
        _reserved: 0,
    }
}

const ZERO_EV: KeyEvent = KeyEvent {
    code: 0,
    modifiers: 0,
    kind: 0,
    _reserved: 0,
};

/// Producer pushes EVENTS_PER_ITER events; consumer drains with the
/// park-then-recheck pattern the read_key_event handler uses.
/// Asserts no events lost and SPSC order preserved.
#[test]
fn no_event_loss_under_concurrent_push_drain() {
    for iter in 0..ITERS {
        let ke = ProcessKeyEvents::new();
        let barrier = StdArc::new(Barrier::new(2));

        let prod = {
            let k = ke.clone();
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                for i in 0..EVENTS_PER_ITER as u32 {
                    // Some(tid) means a parked reader was claimed; the
                    // kernel caller would issue the wake. The host
                    // consumer polls instead, so the value is dropped.
                    let _ = k.push_event(ev(i));
                }
            })
        };

        let cons = {
            let k = ke.clone();
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                let mut got: Vec<u32> = Vec::with_capacity(EVENTS_PER_ITER);
                let mut buf = [ZERO_EV; 16];
                while got.len() < EVENTS_PER_ITER {
                    let n = k.try_drain(&mut buf);
                    if n > 0 {
                        got.extend(buf[..n].iter().map(|e| e.code));
                        continue;
                    }
                    // Park-then-recheck: stamp our tid, then re-drain
                    // to close the park-vs-push window. AlreadyOwned
                    // is legal here — it's the timer-wake re-entry
                    // shape (no producer cleared our slot since the
                    // previous park).
                    match k.set_parker(TID) {
                        ParkOutcome::Installed | ParkOutcome::AlreadyOwned => {}
                        ParkOutcome::Busy => panic!("iter {iter}: single-reader invariant"),
                    }
                    let n2 = k.try_drain(&mut buf);
                    if n2 > 0 {
                        let _ = k.clear_parker_if(TID);
                        got.extend(buf[..n2].iter().map(|e| e.code));
                        continue;
                    }
                    // Parked + re-drain saw nothing → in the kernel
                    // we'd yield Suspended and wait for
                    // WakeEvent::InputTid. Host stand-in: poll the
                    // ring, then cancel the park.
                    while k.is_empty() {
                        thread::yield_now();
                    }
                    let _ = k.clear_parker_if(TID);
                }
                got
            })
        };

        prod.join().unwrap();
        let got = cons.join().unwrap();

        assert_eq!(got.len(), EVENTS_PER_ITER, "iter {iter}: lost events");
        for (i, &code) in got.iter().enumerate() {
            assert_eq!(code, i as u32, "iter {iter}: SPSC order broken at {i}");
        }
    }
}

/// The three set_parker outcomes: empty slot installs, re-entry with
/// the same tid reports AlreadyOwned (timer-wake path — must NOT look
/// like a fresh park or the handler re-parks forever instead of
/// returning timeout), and a different tid is Busy.
#[test]
fn set_parker_three_outcomes() {
    let ke = ProcessKeyEvents::new();
    assert_eq!(ke.set_parker(TID), ParkOutcome::Installed);
    assert_eq!(ke.set_parker(TID), ParkOutcome::AlreadyOwned);
    assert_eq!(ke.set_parker(TID2), ParkOutcome::Busy);
    // Busy must not have clobbered the original parker.
    assert_eq!(ke.take_parker(), Some(TID));
}

/// push_event claims the parked tid exactly once; the second push has
/// no parker to wake (slot was swapped to 0 by the first).
#[test]
fn push_event_returns_parked_tid_once() {
    let ke = ProcessKeyEvents::new();
    assert_eq!(ke.set_parker(TID), ParkOutcome::Installed);
    assert_eq!(ke.push_event(ev(1)), Some(TID), "first push wakes parker");
    assert_eq!(ke.push_event(ev(2)), None, "slot cleared on first push");
}

/// clear_parker_if is a CAS on our own tid: succeeds while we still
/// own the slot, fails (without side effect) once a producer's swap
/// already claimed it — the read handler must not "give back" a wake
/// the producer has spent a WAKE_QUEUE slot on.
#[test]
fn clear_parker_if_only_clears_own_claim() {
    let ke = ProcessKeyEvents::new();
    assert_eq!(ke.set_parker(TID), ParkOutcome::Installed);
    assert!(ke.clear_parker_if(TID), "own claim clears");
    assert!(!ke.clear_parker_if(TID), "already empty");

    assert_eq!(ke.set_parker(TID), ParkOutcome::Installed);
    assert_eq!(ke.push_event(ev(1)), Some(TID));
    assert!(
        !ke.clear_parker_if(TID),
        "producer already consumed the claim"
    );
}

/// take_parker (process-teardown path) takes at most once.
#[test]
fn take_parker_takes_once() {
    let ke = ProcessKeyEvents::new();
    assert_eq!(ke.take_parker(), None);
    assert_eq!(ke.set_parker(TID), ParkOutcome::Installed);
    assert_eq!(ke.take_parker(), Some(TID));
    assert_eq!(ke.take_parker(), None);
}

/// Overflow contract: pushes past RING_CAP are silently dropped, the
/// first RING_CAP events survive intact and in order. (Callers accept
/// drop-newest for key input — see RING_CAP's sizing rationale.)
#[test]
fn overflow_drops_newest_keeps_order() {
    let ke = ProcessKeyEvents::new();
    for i in 0..(RING_CAP + 5) as u32 {
        let _ = ke.push_event(ev(i));
    }
    let mut got: Vec<u32> = Vec::new();
    let mut buf = [ZERO_EV; 32];
    loop {
        let n = ke.try_drain(&mut buf);
        if n == 0 {
            break;
        }
        got.extend(buf[..n].iter().map(|e| e.code));
    }
    assert_eq!(got.len(), RING_CAP, "exactly the ring's capacity survives");
    for (i, &code) in got.iter().enumerate() {
        assert_eq!(code, i as u32, "surviving prefix must be in order");
    }
    assert!(ke.is_empty());
}
