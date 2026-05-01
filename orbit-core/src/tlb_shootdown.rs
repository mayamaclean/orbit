//! Cross-hart TLB shootdown protocol.
//!
//! When one hart modifies a page table entry that another hart may
//! have cached in its TLB, the modifying hart broadcasts a shootdown
//! request: *each target hart drains a per-hart ring of `(va, len)`
//! tuples and `sfence.vma`'s the entries on its local TLB*. The sender
//! spins on a refcounted ack counter until every receiver decrements,
//! then proceeds knowing the system has globally observed the
//! invalidation.
//!
//! This module owns the protocol — types, orchestrator, drain helper.
//! Hardware-side bits (the per-hart `static ShootdownRing`s, the SSWI
//! handler that calls [`drain_shootdown_ring`], and the actual
//! `sfence.vma` instruction) live in kmain. Tests in
//! [orbit-core/tests/tlb_shootdown.rs](../../tests/tlb_shootdown.rs)
//! exercise the full push/wake/drain/ack loop with a fake `Hardware`
//! impl.
//!
//! # Capacity choice
//!
//! [`SHOOTDOWN_RING_CAP`] is 16. Each entry is `(u64, u64,
//! Arc<AtomicU32>)` ≈ 24 bytes; per hart that's 384 bytes of static.
//! With four harts that's 1.5 KiB total — negligible. The cap need
//! only exceed the maximum number of in-flight shootdowns the same
//! target hart can be hit with before it gets to drain. Today every
//! shootdown blocks the sender on the ack, so sustained back-pressure
//! is bounded by the number of senders × in-flight requests per
//! sender (≤ 1 since each sender is single-threaded). 16 is generous
//! headroom for the foreseeable future.
//!
//! # Ring-full policy
//!
//! Surfaced as [`ShootdownErr::RingFull`] so the caller can pick its
//! own fallback. kmain's policy is to fall back to a wider local
//! `sfence.vma` and panic-log the queue overflow — losing precision
//! is acceptable, losing correctness is not.

use process::AckCounter;
use thingbuf::StaticThingBuf;

use crate::Hardware;

/// One shootdown request as it sits on a target hart's queue.
///
/// `Empty` is the default for `StaticThingBuf` slot pre-init; the
/// drain loop matches on it as a no-op so a partially-populated ring
/// isn't a panic risk during early-boot races.
#[derive(Clone, Debug, Default)]
pub enum ShootdownEntry {
    #[default]
    Empty,
    Req {
        /// First VA in the range to invalidate. Page-aligned in
        /// practice but the protocol doesn't enforce it — the receiver
        /// passes whatever the sender wrote into `sfence.vma`'s rs1.
        va: u64,
        /// Length in bytes. `0` means "whole-ASID flush" — receivers
        /// translate to `sfence.vma x0, asid` when wired up; for now
        /// the orchestrator only emits per-page requests.
        len: u64,
        /// Sender's ack-counter clone. The receiver decrements after
        /// servicing; the sender spins on the underlying counter until
        /// it hits zero, at which point every target has fenced.
        ack: AckCounter,
    },
}

/// Per-hart shootdown ring depth. See module docs for the sizing
/// rationale.
pub const SHOOTDOWN_RING_CAP: usize = 16;

/// Per-hart shootdown ring type. Multi-producer (any hart can fire a
/// shootdown), single-consumer (the target hart drains in its SSWI
/// handler).
pub type ShootdownRing = StaticThingBuf<ShootdownEntry, SHOOTDOWN_RING_CAP>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShootdownErr {
    /// At least one target's ring was full at push time. The sender's
    /// caller should fall back to a coarser invalidation strategy and
    /// log — losing precision is OK; losing correctness is not.
    RingFull { failed: u32 },
}

/// Broadcast a shootdown of `[va, va + len)` to every target in
/// `targets` and block until all receivers have decremented their
/// share of the ack counter.
///
/// `n_targets` must equal the iterator's length. Passed explicitly
/// because we need the value before we start consuming the iterator
/// (the ack counter is constructed up front so receivers race-free
/// see a fully-initialized count when they pop their slot).
///
/// `targets` yields `(hart_id, &'static ShootdownRing)` pairs. The
/// orchestrator pushes one [`ShootdownEntry::Req`] onto each ring,
/// then asks the hardware to wake each target via
/// [`Hardware::wake_hart`] — the receiver's SSWI handler is
/// responsible for calling [`drain_shootdown_ring`].
///
/// Returns `Ok(())` once every target has acked. If one or more rings
/// were full, returns `Err(ShootdownErr::RingFull { failed })` after
/// waiting for the targets that *did* accept the request — a partial
/// shootdown is still progress.
///
/// # Self-fence
///
/// The orchestrator does *not* fence the local hart. The local hart
/// is the one running this code, so it's the caller's job to invoke
/// `sfence.vma` directly before calling here (or include itself in
/// `targets` and pop the entry off its own ring, but that's needless
/// IPI traffic when the sender is already in supervisor mode).
pub fn tlb_shootdown<I, H>(
    n_targets: usize,
    targets: I,
    va: u64,
    len: u64,
    hw: &mut H,
) -> Result<(), ShootdownErr>
where
    I: IntoIterator<Item = (usize, &'static ShootdownRing)>,
    H: Hardware,
{
    if n_targets == 0 {
        return Ok(());
    }

    let ack = AckCounter::new(n_targets);
    let mut failed: u32 = 0;

    for (hart_id, ring) in targets {
        match ring.push_ref() {
            Ok(mut slot) => {
                *slot = ShootdownEntry::Req {
                    va,
                    len,
                    ack: ack.clone(),
                };
                drop(slot); // publishes the slot to the consumer
                hw.wake_hart(hart_id);
            }
            Err(_) => {
                // Decrement immediately so the sender doesn't deadlock
                // waiting on a receiver that will never see the
                // request.
                ack.decrement();
                failed = failed.saturating_add(1);
            }
        }
    }

    ack.wait_zero_spin();

    if failed != 0 {
        Err(ShootdownErr::RingFull { failed })
    }
    else {
        Ok(())
    }
}

/// Drain `ring` until empty, calling `apply(va, len)` for each request
/// and decrementing the carried [`AckCounter`] afterwards. Returns
/// the number of entries serviced.
///
/// Intended call site: the SSWI handler in kmain's `s_trap` arm. The
/// `apply` callback is the place where `sfence.vma` actually runs;
/// passing it as a closure keeps this module hardware-free and lets
/// tests inject a recording fake instead.
///
/// # Ordering
///
/// The decrement happens *after* `apply` returns. If `apply` panics
/// (it shouldn't — kernel sfence is infallible), the ack stays
/// outstanding and the sender deadlocks. The kmain caller must keep
/// `apply` straight-line and panic-free.
pub fn drain_shootdown_ring(ring: &'static ShootdownRing, mut apply: impl FnMut(u64, u64)) -> u32 {
    let mut serviced = 0u32;
    while let Some(mut slot) = ring.pop_ref() {
        let entry = core::mem::take(&mut *slot);
        drop(slot); // releases the slot back to producers
        match entry {
            ShootdownEntry::Empty => {
                // Possible if a producer pushed but the receiver
                // raced past `push_ref`'s commit before the slot was
                // populated. thingbuf's Default-on-take prevents
                // double-fence by leaving Empty in the slot. Skip
                // without acking — there's no AckCounter to talk to
                // and the producer hasn't committed yet, so they'll
                // re-push as Req.
                //
                // In practice this shouldn't happen with our shape
                // (push_ref's drop publishes the slot atomically) but
                // handle it defensively rather than asserting.
                continue;
            }
            ShootdownEntry::Req { va, len, ack } => {
                apply(va, len);
                ack.decrement();
                serviced += 1;
            }
        }
    }
    serviced
}
