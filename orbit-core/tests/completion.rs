//! Concurrency stress for [`process::CompletionHandle`].
//!
//! `signal_n` claims the slot via CAS and then writes `rets` /
//! `ret_count` under the claim. Earlier shape (load-state-then-write-
//! then-CAS) raced when two signalers observed `PENDING` together:
//! both wrote, then the CAS winner published a half-from-A,
//! half-from-B mix. The fix CAS-claims first, so concurrent signalers
//! either own the slot or bail.
//!
//! Test: many signalers race on the same handle, each with a distinct
//! (length, values) tuple. After all return, the observed
//! `(ret_count, rets[..count])` must match exactly one signaler's
//! input — never a mix.

use std::sync::{Arc as StdArc, Barrier};
use std::thread;

use process::CompletionHandle;

/// Each signaler offers one of these. The `id` is a tag so a torn
/// write is easy to spot in failure output.
const VARIANTS: &[&[isize]] = &[
    &[1, 2, 3],
    &[100, 200],
    &[7],
    &[1_000_000, 2_000_000, 3_000_000, 4_000_000],
];

/// Iteration budget. Under miri the per-iteration cost is enormous
/// (each iter explores a fresh thread interleaving), so 10 is
/// sufficient — miri's value here is exhaustive op-ordering coverage,
/// not iteration count. Native runs do 1000 to maximize the chance of
/// catching a non-miri-detectable timing race.
#[cfg(miri)]
const ITERS: usize = 10;
#[cfg(not(miri))]
const ITERS: usize = 1000;

/// N racing signalers per iteration. On the broken shape, a torn
/// write trips within the first handful of iterations in practice. On
/// the fixed shape (CAS-claim-then-write), all iterations must
/// observe a coherent winner.
#[test]
fn signal_n_winner_takes_all_under_race() {
    for iter in 0..ITERS {
        let h = CompletionHandle::new();
        let barrier = StdArc::new(Barrier::new(VARIANTS.len()));

        let handles: Vec<_> = VARIANTS
            .iter()
            .map(|vals| {
                let h = h.clone();
                let barrier = barrier.clone();
                let vals: &'static [isize] = vals;
                thread::spawn(move || {
                    // All signalers wait at the barrier, then race.
                    // Maximizes the interleaving window.
                    barrier.wait();
                    h.signal_n(vals);
                })
            })
            .collect();

        for jh in handles {
            jh.join().unwrap();
        }

        assert!(h.is_signaled(), "iter {iter}: handle never signaled");

        let n = h.ret_count();
        let observed: Vec<isize> = (0..n).map(|i| h.ret(i)).collect();

        let matched = VARIANTS
            .iter()
            .any(|v| v.len() == n && v.iter().zip(observed.iter()).all(|(a, b)| a == b));

        assert!(
            matched,
            "iter {iter}: torn write — observed {observed:?} matches no input variant"
        );
    }
}

/// First signal wins; later signalers don't mutate state. Pin the
/// idempotence guarantee callers (e.g. `SharedUserPtr::revoke` racing
/// a legitimate completion) rely on.
#[test]
fn signal_n_first_caller_wins() {
    let h = CompletionHandle::new();
    h.signal_n(&[42]);
    h.signal_n(&[99, 100]);
    h.signal_n(&[1, 2, 3, 4]);

    assert!(h.is_signaled());
    assert_eq!(h.ret_count(), 1);
    assert_eq!(h.ret(0), 42);
}

/// `signal` and `signal_pair` go through `signal_n` under the hood;
/// confirm the count + values they publish.
#[test]
fn signal_one_and_pair_match_ret_count() {
    let h1 = CompletionHandle::new();
    h1.signal(7);
    assert_eq!(h1.ret_count(), 1);
    assert_eq!(h1.ret(0), 7);

    let h2 = CompletionHandle::new();
    h2.signal_pair(0x240000000, 3);
    assert_eq!(h2.ret_count(), 2);
    assert_eq!(h2.ret(0), 0x240000000);
    assert_eq!(h2.ret(1), 3);
}

/// Excess values past `MAX_RET_SLOTS` are clamped, not panicked on.
#[test]
fn signal_n_clamps_to_max_ret_slots() {
    let h = CompletionHandle::new();
    h.signal_n(&[1, 2, 3, 4, 5, 6]);
    assert!(h.is_signaled());
    assert_eq!(h.ret_count(), process::completion::MAX_RET_SLOTS);
    for i in 0..process::completion::MAX_RET_SLOTS {
        assert_eq!(h.ret(i), (i + 1) as isize);
    }
}
