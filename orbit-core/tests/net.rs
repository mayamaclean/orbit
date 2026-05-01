use std::collections::BTreeMap;
use std::collections::VecDeque;

use orbit_core::net::{RevocableConn, drain_socket_deletions, prune_revoked_conns};

/// Minimal conn value — tests pin whether revocation is detected +
/// acted on correctly.
struct Conn {
    revoked: bool,
}

impl RevocableConn for Conn {
    fn is_revoked(&self) -> bool {
        self.revoked
    }
}

fn make_conns(entries: &[(u32, bool)]) -> BTreeMap<u32, Conn> {
    entries
        .iter()
        .map(|&(k, r)| (k, Conn { revoked: r }))
        .collect()
}

// ---------- prune_revoked_conns ----------

#[test]
fn prune_empty_is_noop() {
    let mut conns: BTreeMap<u32, Conn> = BTreeMap::new();
    let mut scratch = Vec::new();
    let mut removed: Vec<u32> = Vec::new();

    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));

    assert!(conns.is_empty());
    assert!(removed.is_empty());
    assert!(scratch.is_empty());
}

#[test]
fn prune_removes_only_revoked() {
    let mut conns = make_conns(&[(1, false), (2, true), (3, false), (4, true)]);
    let mut scratch = Vec::new();
    let mut removed: Vec<u32> = Vec::new();

    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));

    assert_eq!(conns.len(), 2);
    assert!(conns.contains_key(&1));
    assert!(conns.contains_key(&3));
    removed.sort();
    assert_eq!(removed, vec![2, 4]);
}

#[test]
fn prune_all_revoked_empties_conns() {
    let mut conns = make_conns(&[(1, true), (2, true), (3, true)]);
    let mut scratch = Vec::new();
    let mut removed: Vec<u32> = Vec::new();

    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));

    assert!(conns.is_empty());
    removed.sort();
    assert_eq!(removed, vec![1, 2, 3]);
}

#[test]
fn prune_none_revoked_leaves_conns() {
    let mut conns = make_conns(&[(1, false), (2, false)]);
    let mut scratch = Vec::new();
    let mut removed: Vec<u32> = Vec::new();

    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));

    assert_eq!(conns.len(), 2);
    assert!(removed.is_empty());
}

#[test]
fn prune_clears_scratch_on_entry() {
    let mut conns = make_conns(&[(5, true)]);
    let mut scratch: Vec<u32> = vec![99, 100, 101]; // stale junk
    let mut removed: Vec<u32> = Vec::new();

    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));

    // The stale contents must not have leaked through as fake removals.
    assert_eq!(removed, vec![5]);
    // And scratch should be drained before return, ready for reuse.
    assert!(scratch.is_empty());
}

#[test]
fn prune_reuses_scratch_across_calls() {
    // The explicit scratch parameter exists specifically so the net
    // thread can avoid a fresh Vec allocation per tick. Verify the
    // buffer's capacity survives the drain.
    let mut scratch: Vec<u32> = Vec::with_capacity(16);
    let cap_before = scratch.capacity();
    let mut removed: Vec<u32> = Vec::new();

    let mut conns = make_conns(&[(1, true), (2, true)]);
    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));
    assert!(
        scratch.capacity() >= cap_before,
        "capacity should not shrink on drain"
    );
}

// ---------- drain_socket_deletions ----------

#[test]
fn drain_empty_queue_is_noop() {
    let mut conns = make_conns(&[(1, false), (2, false)]);
    let mut deletions: VecDeque<u32> = VecDeque::new();
    let mut removed: Vec<u32> = Vec::new();

    drain_socket_deletions(&mut conns, || deletions.pop_front(), |k| removed.push(k));

    assert_eq!(conns.len(), 2);
    assert!(removed.is_empty());
}

#[test]
fn drain_removes_matching_handles() {
    let mut conns = make_conns(&[(1, false), (2, false), (3, false)]);
    let mut deletions: VecDeque<u32> = VecDeque::from([1, 3]);
    let mut removed: Vec<u32> = Vec::new();

    drain_socket_deletions(&mut conns, || deletions.pop_front(), |k| removed.push(k));

    assert_eq!(conns.len(), 1);
    assert!(conns.contains_key(&2));
    assert_eq!(removed, vec![1, 3]);
}

#[test]
fn drain_skips_remove_socket_for_absent_handles() {
    // Handle 99 is in the deletion queue but NOT in conns — simulates
    // the "revocation prune already handled this" race. `remove_socket`
    // MUST NOT be called for 99, because smoltcp's SocketSet::remove
    // would panic.
    let mut conns = make_conns(&[(1, false)]);
    let mut deletions: VecDeque<u32> = VecDeque::from([99, 1]);
    let mut removed: Vec<u32> = Vec::new();

    drain_socket_deletions(&mut conns, || deletions.pop_front(), |k| removed.push(k));

    // Only 1 got removed; 99 was silently skipped.
    assert!(conns.is_empty());
    assert_eq!(removed, vec![1]);
}

#[test]
fn drain_empties_the_queue_even_when_conns_empty() {
    // All deletions miss — drain still consumes the whole queue.
    let mut conns: BTreeMap<u32, Conn> = BTreeMap::new();
    let mut deletions: VecDeque<u32> = VecDeque::from([1, 2, 3]);
    let mut removed: Vec<u32> = Vec::new();

    drain_socket_deletions(&mut conns, || deletions.pop_front(), |k| removed.push(k));

    assert!(deletions.is_empty(), "drain must consume every queued item");
    assert!(removed.is_empty());
}

// ---------- prune + drain interaction ----------
// Mirrors the k_net loop ordering: drain runs first, then prune. The pair
// must leave state consistent even when the two sources overlap.

#[test]
fn prune_then_drain_on_same_handle_skips_double_remove() {
    // Conn 5 is revoked AND in the deletion queue. Prune runs first (as
    // the pure fns are invoked), removes it from conns and calls
    // remove_socket once. Drain then finds it absent in conns and MUST
    // skip its remove_socket call — smoltcp's SocketSet::remove panics
    // on an absent handle.
    let mut conns = make_conns(&[(1, false), (5, true)]);
    let mut scratch = Vec::new();
    let mut deletions: VecDeque<u32> = VecDeque::from([5]);
    let mut removed: Vec<u32> = Vec::new();

    prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));
    drain_socket_deletions(&mut conns, || deletions.pop_front(), |k| removed.push(k));

    assert_eq!(removed, vec![5], "remove_socket fires exactly once");
    assert!(conns.contains_key(&1));
    assert_eq!(conns.len(), 1);
}

#[test]
fn drain_with_duplicate_handle_only_removes_once() {
    // Queue has the same handle twice. First drain call removes it from
    // conns and fires remove_socket; second iteration finds it absent
    // and is a safe no-op.
    let mut conns = make_conns(&[(7, false)]);
    let mut deletions: VecDeque<u32> = VecDeque::from([7, 7]);
    let mut removed: Vec<u32> = Vec::new();

    drain_socket_deletions(&mut conns, || deletions.pop_front(), |k| removed.push(k));

    assert_eq!(removed, vec![7]);
    assert!(conns.is_empty());
    assert!(
        deletions.is_empty(),
        "both duplicate entries must be consumed"
    );
}

#[test]
fn scratch_buffer_reused_across_alternating_prune_drain_calls() {
    // k_net invokes prune on every tick. The caller-owned scratch must
    // survive intervening drain calls unchanged (drain doesn't touch
    // it) and still function on the next prune.
    let mut scratch: Vec<u32> = Vec::with_capacity(8);
    let mut removed: Vec<u32> = Vec::new();
    let mut deletions: VecDeque<u32> = VecDeque::from([99]);

    // Tick 1: prune finds one revoked.
    {
        let mut conns = make_conns(&[(1, true), (2, false)]);
        prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));
        assert!(scratch.is_empty(), "scratch drained");
    }

    // Unrelated drain tick between prunes — shouldn't touch scratch.
    {
        let mut conns: BTreeMap<u32, Conn> = BTreeMap::new();
        drain_socket_deletions(&mut conns, || deletions.pop_front(), |_| {});
        assert!(scratch.is_empty());
    }

    // Tick 2: prune works correctly with the reused scratch.
    {
        let mut conns = make_conns(&[(3, true), (4, true), (5, false)]);
        prune_revoked_conns(&mut conns, &mut scratch, |k| removed.push(k));
    }

    removed.sort();
    assert_eq!(removed, vec![1, 3, 4]);
}
