//! Bounded denial-event ring — the production [`DenialSink`].
//!
//! Wraps a `VecDeque<DenialEvent>` capped at
//! [`DENIAL_RING_CAPACITY`](orbit_abi::denial::DENIAL_RING_CAPACITY)
//! (50 entries today). On push when full, the oldest event is evicted
//! — best-effort retention; the diagnostic value is "what was the
//! kernel denying in the last few hundred ms," not "every denial
//! since boot."
//!
//! Threading: the ring is *not* internally synchronised. Production
//! usage parks it behind the kernel manager lock; the manager-side
//! `drain_denial_events` pass is the only writer (it consumes
//! events off the lock-free producer queue and pushes here). Holding
//! `&mut` is sufficient — Rust's borrow checker enforces single-
//! writer-at-a-time even without a `Mutex`.

use alloc::collections::VecDeque;
use orbit_abi::denial::{DENIAL_RING_CAPACITY, DenialEvent, DenialSink};

/// Production [`DenialSink`]. Bounded `VecDeque<DenialEvent>` —
/// pushes evict the oldest entry once at capacity, so the ring
/// always holds the most recent N events.
///
/// Iteration is chronological (oldest → newest); see
/// [`DenialRing::snapshot`] for the syscall-shaped accessor that
/// `query_denial_log` uses.
#[derive(Debug)]
pub struct DenialRing {
    /// Front = oldest, back = newest. `pop_front` on full + push at
    /// the back keeps insertion at the natural end.
    events: VecDeque<DenialEvent>,
}

impl DenialRing {
    /// Construct an empty ring. `const` so `Orbit::new` (also `const`)
    /// can name it. The first push allocates lazily — acceptable for
    /// the denial path, where pushes are rare and the steady-state
    /// capacity is bounded at [`DENIAL_RING_CAPACITY`].
    pub const fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    /// Number of events currently stored. Always `<= capacity()`.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True iff no events have been pushed (or all have been drained
    /// by some hypothetical future drain API — none today).
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Maximum entries the ring retains. Constant across the ring's
    /// lifetime; matches
    /// [`DENIAL_RING_CAPACITY`](orbit_abi::denial::DENIAL_RING_CAPACITY)
    /// from the ABI.
    pub const fn capacity() -> usize {
        DENIAL_RING_CAPACITY
    }

    /// Iterate events in chronological order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &DenialEvent> {
        self.events.iter()
    }

    /// Copy up to `buf.len()` events into `buf` in chronological
    /// order. Returns the number written. Used by `query_denial_log`
    /// to fill the user-side reply buffer.
    ///
    /// If `buf` is smaller than [`len`](Self::len), the *oldest*
    /// events are copied and the rest of the ring is dropped from
    /// the snapshot — chronological-from-oldest matches the wire
    /// shape callers will iterate. Smoke tests size their buffer for
    /// the full capacity, so truncation is a non-issue in the common
    /// path.
    pub fn snapshot(&self, buf: &mut [DenialEvent]) -> usize {
        let n = core::cmp::min(self.events.len(), buf.len());
        for (i, ev) in self.events.iter().take(n).enumerate() {
            buf[i] = *ev;
        }
        n
    }
}

impl Default for DenialRing {
    fn default() -> Self {
        Self::new()
    }
}

impl DenialSink for DenialRing {
    fn push(&mut self, event: DenialEvent) {
        if self.events.len() == DENIAL_RING_CAPACITY {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orbit_abi::denial::{DenialEvent, GateContext, deny_reason};

    fn perm_event(seq: u64) -> DenialEvent {
        // Synthetic event with `time_ticks = seq` so tests can identify
        // events by ordering without spinning up a clock.
        DenialEvent::PermDeny {
            required_class: 0x1,
            perms: 0x0,
            time_ticks: seq,
            tid: 1,
            sysno: 1,
            source_role: 1,
            pid: 1,
        }
    }

    fn role_event(seq: u64) -> DenialEvent {
        DenialEvent::RoleDeny {
            time_ticks: seq,
            _reserved: 0,
            tid: 1,
            source_role: 1,
            target_role: 2,
            deny_reason: deny_reason::TRANSITION_DENIED,
            pid: 1,
        }
    }

    #[test]
    fn new_ring_is_empty() {
        let r = DenialRing::new();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
        assert_eq!(DenialRing::capacity(), 50);
    }

    #[test]
    fn push_increments_len_until_capacity() {
        let mut r = DenialRing::new();
        for i in 0..10u64 {
            r.push(perm_event(i));
            assert_eq!(r.len() as u64, i + 1);
        }
        assert!(!r.is_empty());
    }

    #[test]
    fn iter_yields_chronological_order() {
        // Pushed in order 0..5; iter must produce 0..5 in the same
        // order. Pinned because query_denial_log relies on it.
        let mut r = DenialRing::new();
        for i in 0..5u64 {
            r.push(perm_event(i));
        }
        let times: alloc::vec::Vec<u64> = r
            .iter()
            .map(|e| match e {
                DenialEvent::PermDeny { time_ticks, .. } => *time_ticks,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(times, alloc::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn at_capacity_oldest_is_evicted_on_push() {
        // Push CAPACITY events, then one more — the first one
        // (time_ticks = 0) should be gone, and the new one
        // (time_ticks = CAPACITY) should be at the back.
        let mut r = DenialRing::new();
        for i in 0..(DENIAL_RING_CAPACITY as u64) {
            r.push(perm_event(i));
        }
        assert_eq!(r.len(), DENIAL_RING_CAPACITY);

        r.push(perm_event(DENIAL_RING_CAPACITY as u64));
        assert_eq!(
            r.len(),
            DENIAL_RING_CAPACITY,
            "len stays at cap after eviction"
        );

        let times: alloc::vec::Vec<u64> = r
            .iter()
            .map(|e| match e {
                DenialEvent::PermDeny { time_ticks, .. } => *time_ticks,
                _ => unreachable!(),
            })
            .collect();
        // First entry is now time_ticks=1 (originally 0 was evicted).
        assert_eq!(times[0], 1);
        // Last entry is the freshly pushed time_ticks=CAPACITY.
        assert_eq!(times[DENIAL_RING_CAPACITY - 1], DENIAL_RING_CAPACITY as u64);
    }

    #[test]
    fn snapshot_copies_chronologically_into_user_buffer() {
        let mut r = DenialRing::new();
        for i in 0..5u64 {
            r.push(perm_event(i));
        }
        let mut buf = [perm_event(99); 5]; // sentinel — should be overwritten
        let n = r.snapshot(&mut buf);
        assert_eq!(n, 5);
        for i in 0..5 {
            match buf[i] {
                DenialEvent::PermDeny { time_ticks, .. } => assert_eq!(time_ticks, i as u64),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn snapshot_truncates_when_buffer_is_smaller_than_ring() {
        // 10 events, buf=3 → snapshot writes the oldest 3. Documented
        // semantics: chronological-from-oldest. Smoke tests size for
        // full capacity, so this branch is rare in practice but the
        // shape needs to be deterministic.
        let mut r = DenialRing::new();
        for i in 0..10u64 {
            r.push(perm_event(i));
        }
        let mut buf = [perm_event(99); 3];
        let n = r.snapshot(&mut buf);
        assert_eq!(n, 3);
        for i in 0..3 {
            match buf[i] {
                DenialEvent::PermDeny { time_ticks, .. } => assert_eq!(time_ticks, i as u64),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn snapshot_pads_buffer_only_with_what_it_writes() {
        // Buf bigger than ring → only `len` slots are touched. Caller
        // reads `n` and ignores the tail. Pinned because the syscall
        // contract is "bytes_written tells you how much is valid."
        let mut r = DenialRing::new();
        r.push(perm_event(0));
        r.push(perm_event(1));
        let mut buf = [perm_event(99); 10];
        let n = r.snapshot(&mut buf);
        assert_eq!(n, 2);
        // Slot 2 should still hold the sentinel — snapshot doesn't
        // touch beyond `n`.
        match buf[2] {
            DenialEvent::PermDeny { time_ticks, .. } => assert_eq!(time_ticks, 99),
            _ => unreachable!(),
        }
    }

    #[test]
    fn ring_accepts_both_event_variants() {
        // Mixed PermDeny / RoleDeny in one ring — production gates push
        // both kinds to the same sink.
        let mut r = DenialRing::new();
        r.push(perm_event(0));
        r.push(role_event(1));
        r.push(perm_event(2));
        assert_eq!(r.len(), 3);
        let snapshot: alloc::vec::Vec<&DenialEvent> = r.iter().collect();
        assert!(matches!(snapshot[0], DenialEvent::PermDeny { .. }));
        assert!(matches!(snapshot[1], DenialEvent::RoleDeny { .. }));
        assert!(matches!(snapshot[2], DenialEvent::PermDeny { .. }));
    }

    #[test]
    fn ring_works_as_denial_sink_for_gate_functions() {
        // Integration-shaped sanity: pass &mut DenialRing where a
        // DenialSink is expected (Permissions::can_call). The denied
        // call's PermDeny event should land in the ring.
        use orbit_abi::perms::{ClassMask, Permissions, PermsRequest, class};

        let p = Permissions::ALL.pledge(PermsRequest {
            perms: ClassMask::from_raw(class::raw::ALL & !class::raw::NETCH),
            allowed_perms: class::ALL,
        });
        let mut ring = DenialRing::new();
        let ctx = GateContext {
            pid: 5,
            tid: 9,
            time_ticks: 1_000,
        };
        let ok = p.can_call(orbit_abi::syscall::CREATE_NETCH, ctx, &mut ring);
        assert!(!ok);
        assert_eq!(ring.len(), 1);
    }
}
