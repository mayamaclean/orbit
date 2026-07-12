mod common;

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use orbit_core::tlb_shootdown::{
    FlushScope, SHOOTDOWN_RING_CAP, ShootdownEntry, ShootdownErr, ShootdownRing,
    drain_shootdown_ring, tlb_shootdown,
};

use common::FakeHw;

// =====================================================================
// Single-hart synchronous mechanics
//
// These tests run the orchestrator and the drain in the same thread to
// validate the protocol shape (slot publish → wake → drain → ack)
// without bringing concurrency in. Concurrent tests come further down.
// =====================================================================

/// Drain helper that just records `(scope, va, len)` tuples into a Vec.
/// Stand-in for `sfence.vma` in tests.
fn record_into(vec: &mut Vec<(FlushScope, u64, u64)>) -> impl FnMut(FlushScope, u64, u64) + '_ {
    |scope, va, len| vec.push((scope, va, len))
}

#[test]
fn zero_targets_short_circuits_without_ack() {
    let mut hw = FakeHw::default();
    let result = tlb_shootdown(
        0,
        core::iter::empty(),
        FlushScope::Asid(7),
        0xdead_0000,
        4096,
        &mut hw,
    );
    assert_eq!(result, Ok(()));
    assert!(hw.wakes.is_empty(), "no IPIs for zero-target shootdown");
}

#[test]
fn single_target_push_drain_ack_roundtrip() {
    static RING: ShootdownRing = ShootdownRing::new();

    // Spawn the orchestrator on a thread so we can drive the drain
    // from the main thread before the orchestrator's wait_zero_spin
    // returns. (The orchestrator blocks on the ack until we pop+ack.)
    let h = thread::spawn(|| {
        let mut hw = FakeHw::default();
        tlb_shootdown(
            1,
            core::iter::once((2, &RING)),
            FlushScope::Asid(9),
            0x4000_0000,
            4096,
            &mut hw,
        )
    });

    // Wait for the entry to appear (push happens before wake on the
    // sender side, but wait_zero_spin would still spin even if we
    // raced past push), then drain.
    let mut got = Vec::new();
    loop {
        let serviced = drain_shootdown_ring(&RING, record_into(&mut got));
        if serviced > 0 {
            break;
        }
        std::hint::spin_loop();
    }

    let result = h.join().unwrap();
    assert_eq!(result, Ok(()));
    assert_eq!(got, vec![(FlushScope::Asid(9), 0x4000_0000u64, 4096u64)]);
}

#[test]
fn multi_target_each_acks_once() {
    static R0: ShootdownRing = ShootdownRing::new();
    static R1: ShootdownRing = ShootdownRing::new();
    static R2: ShootdownRing = ShootdownRing::new();

    let h = thread::spawn(|| {
        let mut hw = FakeHw::default();
        let targets = [(0, &R0), (1, &R1), (2, &R2)];
        let r = tlb_shootdown(3, targets, FlushScope::Asid(5), 0xc0de_0000, 0x1000, &mut hw);
        (r, hw.wakes)
    });

    // Drain each ring once. Until all three drain, the orchestrator is
    // spinning on the ack.
    let mut got: Vec<(FlushScope, u64, u64)> = Vec::new();
    let mut total_serviced = 0u32;
    while total_serviced < 3 {
        for ring in [&R0, &R1, &R2] {
            total_serviced += drain_shootdown_ring(ring, record_into(&mut got));
        }
        if total_serviced < 3 {
            std::hint::spin_loop();
        }
    }

    let (result, wakes) = h.join().unwrap();
    assert_eq!(result, Ok(()));
    // Every target got exactly one entry with the right (va, len).
    for entry in &got {
        assert_eq!(*entry, (FlushScope::Asid(5), 0xc0de_0000u64, 0x1000u64));
    }
    assert_eq!(got.len(), 3);
    // Every target got an IPI.
    let mut wakes_sorted = wakes.clone();
    wakes_sorted.sort();
    assert_eq!(wakes_sorted, vec![0, 1, 2]);
}

#[test]
fn ring_full_returns_failed_count_after_other_acks() {
    static FULL_RING: ShootdownRing = ShootdownRing::new();
    static OPEN_RING: ShootdownRing = ShootdownRing::new();

    // Pre-fill FULL_RING to capacity. push_ref returns Err once full.
    for _ in 0..SHOOTDOWN_RING_CAP {
        let mut slot = FULL_RING.push_ref().expect("filling pre-test");
        *slot = ShootdownEntry::Empty;
    }
    assert!(FULL_RING.push_ref().is_err(), "ring should be saturated");

    // Orchestrator should ack-decrement the failed target immediately
    // and only block on the open one.
    let h = thread::spawn(|| {
        let mut hw = FakeHw::default();
        let targets = [(7, &FULL_RING), (8, &OPEN_RING)];
        tlb_shootdown(2, targets, FlushScope::Asid(3), 0x9000_0000, 0x1000, &mut hw)
    });

    // Drain only the open ring — failed one stays full but the
    // orchestrator already accounted for it via decrement.
    let mut got = Vec::new();
    loop {
        let n = drain_shootdown_ring(&OPEN_RING, record_into(&mut got));
        if n > 0 {
            break;
        }
        std::hint::spin_loop();
    }

    let result = h.join().unwrap();
    assert_eq!(result, Err(ShootdownErr::RingFull { failed: 1 }));
    assert_eq!(got, vec![(FlushScope::Asid(3), 0x9000_0000u64, 0x1000u64)]);

    // Drain the pre-fill so subsequent tests reusing FULL_RING (none
    // today, statics are per-test) start clean.
    while drain_shootdown_ring(&FULL_RING, |_, _, _| {}) > 0 {}
}

// =====================================================================
// Concurrent stress (cheap host run; miri will catch ordering bugs)
// =====================================================================

/// Many senders shoot down the same target concurrently. Validates:
/// - The MPSC ring serializes concurrent push_ref correctly.
/// - Each sender's individual AckCounter is independent.
/// - The drain side eventually services all entries.
#[test]
fn concurrent_senders_one_target() {
    static TARGET: ShootdownRing = ShootdownRing::new();

    // miri runs this loop slowly; native is fine with many more.
    const SENDERS: usize = if cfg!(miri) { 4 } else { 32 };
    const PER_SENDER_REQS: usize = if cfg!(miri) { 2 } else { 8 };

    let barrier = Arc::new(Barrier::new(SENDERS + 1));
    let total_serviced = Arc::new(AtomicU32::new(0));

    let mut sender_handles = Vec::new();
    for sid in 0..SENDERS {
        let bar = barrier.clone();
        let h = thread::spawn(move || {
            bar.wait();
            for i in 0..PER_SENDER_REQS {
                let mut hw = FakeHw::default();
                let va = ((sid << 16) | (i << 4)) as u64;
                // Loop on RingFull — under heavy contention with cap=16
                // some pushes will fail. Recover by re-trying after
                // the drainer has caught up.
                loop {
                    let r = tlb_shootdown(
                        1,
                        core::iter::once((0, &TARGET)),
                        FlushScope::Asid(1),
                        va,
                        4096,
                        &mut hw,
                    );
                    match r {
                        Ok(()) => break,
                        Err(ShootdownErr::RingFull { .. }) => {
                            // Yield + retry. Drainer is advancing.
                            std::thread::yield_now();
                        }
                    }
                }
            }
        });
        sender_handles.push(h);
    }

    // Drainer thread.
    let drainer_total = total_serviced.clone();
    let drainer_bar = barrier.clone();
    let drainer = thread::spawn(move || {
        drainer_bar.wait();
        let target_count = (SENDERS * PER_SENDER_REQS) as u32;
        while drainer_total.load(Ordering::Acquire) < target_count {
            let n = drain_shootdown_ring(&TARGET, |_, _, _| {});
            drainer_total.fetch_add(n, Ordering::AcqRel);
        }
    });

    for h in sender_handles {
        h.join().unwrap();
    }
    drainer.join().unwrap();

    assert_eq!(
        total_serviced.load(Ordering::Acquire),
        (SENDERS * PER_SENDER_REQS) as u32,
    );
}

/// One sender broadcasts to multiple targets, drained concurrently.
/// Mirrors the typical case: hart 0 does a single-PTE invalidation
/// affecting harts 1..N, and the wait_zero_spin observes Acquire-
/// ordered decrements from N independent threads. Catches the
/// counter's release/acquire pairing under miri.
#[test]
fn one_sender_concurrent_drainers() {
    static R0: ShootdownRing = ShootdownRing::new();
    static R1: ShootdownRing = ShootdownRing::new();
    static R2: ShootdownRing = ShootdownRing::new();
    static R3: ShootdownRing = ShootdownRing::new();

    const ITERS: usize = if cfg!(miri) { 5 } else { 50 };

    for i in 0..ITERS {
        let va = (i << 12) as u64;

        // Spawn one drainer per target. Each spins until it sees one
        // entry, services it, exits.
        let drainers: Vec<_> = [&R0, &R1, &R2, &R3]
            .into_iter()
            .map(|ring| {
                thread::spawn(move || {
                    let mut got = None;
                    while got.is_none() {
                        drain_shootdown_ring(ring, |_scope, va, len| {
                            got = Some((va, len));
                        });
                        if got.is_none() {
                            std::hint::spin_loop();
                        }
                    }
                    got.unwrap()
                })
            })
            .collect();

        let mut hw = FakeHw::default();
        let targets = [(0, &R0), (1, &R1), (2, &R2), (3, &R3)];
        let result = tlb_shootdown(4, targets, FlushScope::Asid(4), va, 4096, &mut hw);
        assert_eq!(result, Ok(()));

        for h in drainers {
            assert_eq!(h.join().unwrap(), (va, 4096));
        }
    }
}
