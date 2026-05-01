//! Min-heap of `(wake_time, *mut Thread)` for Suspended sleepers.
//!
//! Replaces the per-pass O(N_threads) scan in `get_runnable_thread`'s
//! Suspended arm with O(woken) at dispatch time. The manager peeks the
//! heap to find the earliest deadline (Phase D feeds this into
//! `stimecmp`), pops everything whose deadline has passed, and hands
//! the woken threads back via a callback.
//!
//! ## Lazy delete
//!
//! A Suspended thread can be woken early via [`Thread::wake_override`]
//! from a non-manager hart that has no access to this heap. We don't
//! try to remove the heap entry at that point — instead the entry
//! stays until the manager pops it and recognizes it as stale.
//!
//! Staleness rule at pop time, by `(seq_match, state)` combination:
//!
//! | seq match? | state              | verdict |
//! |---|---|---|
//! | no  | (any)                  | stale — re-parked since push |
//! | yes | Suspended              | live — check deadline |
//! | yes | Ready                  | stale — eagerly woken |
//! | yes | Exited                 | stale — being torn down |
//! | yes | Running/Assigned/Blocking | **transient** — leave entry, peek again next pass |
//!
//! The transient case covers `kthread_park`'s push-before-state-publish
//! window: the asm handoff publishes `state=Suspended` after the inbox
//! push, so a manager that drains during the gap would see seq matching
//! but state still Running. We can't drop (the park is real and pending)
//! and we can't fire (deadline-not-yet-elapsed could spuriously wake).
//! Solution: leave it in the heap; the next manager pass observes
//! state=Suspended and proceeds normally.
//!
//! The seq counter alone is insufficient because re-park flips state
//! back to Suspended with a new deadline. Without the seq check,
//! draining at time T1 < now < T2 (T1 = stale deadline, T2 = new
//! deadline) would mis-fire as a deadline-elapsed wake.
//!
//! ## BinaryHeap ordering invariant
//!
//! `std::collections::BinaryHeap` requires that an item's relative
//! order not change while in the heap. [`SleepEntry`] honors this by
//! comparing **owned** `wake_time` and `sleep_seq` u64 fields captured
//! at push time. The `*mut Thread` is carried only as the wake target;
//! the `Ord` impl never dereferences it. `Thread.wake_time` mutating
//! after a push has no effect on the entry's ordering.
//!
//! Concretely:
//!
//! ```ignore
//! // OK: ordering basis is owned, immutable for entry's lifetime
//! impl Ord for SleepEntry {
//!     fn cmp(&self, other: &Self) -> Ordering {
//!         self.wake_time.cmp(&other.wake_time)
//!     }
//! }
//!
//! // BUG: would dereference shared mutable Thread state
//! impl Ord for SleepEntry {
//!     fn cmp(&self, other: &Self) -> Ordering {
//!         unsafe { (*self.thread).wake_time.cmp(&(*other.thread).wake_time) }
//!     }
//! }
//! ```

use alloc::collections::BinaryHeap;
use core::cmp::{Ordering, Reverse};
use core::sync::atomic::Ordering as AtomicOrdering;

use process::{Thread, ThreadState};

/// One park instance of one thread. `wake_time` and `sleep_seq` are
/// snapshots taken at push time — see module doc for why they're
/// captured by value, not read through `thread`.
///
/// Companion field on [`Thread`]: `sleep_seq: AtomicU64`,
/// `fetch_add(1)`-ed on every Suspended transition. Currently *not*
/// in [process/src/lib.rs] — adding it is the prerequisite for wiring
/// this module into kmain.
#[derive(Eq, PartialEq)]
struct SleepEntry {
    wake_time: u64,
    sleep_seq: u64,
    thread: *mut Thread,
}

// SAFETY: SleepEntry is held only inside `SleepHeap`, which lives in
// `Orbit` and is mutated under `MANAGER_LOCK`. The `*mut Thread`
// points into the kernel thread registry; the registry frees the
// allocation in `cleanup_threads_and_processes`, which runs on the
// same manager hart in the same critical section as any heap
// drain — there's no window where a heap entry can outlive its
// thread.
unsafe impl Send for SleepEntry {}
unsafe impl Sync for SleepEntry {}

impl Ord for SleepEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Owned-field comparison only. Tie-break on sleep_seq so two
        // pushes with identical wake_time have a deterministic order
        // (matters for tests; doesn't matter for dispatch).
        self.wake_time
            .cmp(&other.wake_time)
            .then(self.sleep_seq.cmp(&other.sleep_seq))
    }
}

impl PartialOrd for SleepEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct SleepHeap {
    inner: BinaryHeap<Reverse<SleepEntry>>,
}

impl SleepHeap {
    pub const fn new() -> Self {
        Self {
            inner: BinaryHeap::new(),
        }
    }

    /// Push a park entry. Caller has just transitioned `thread` into
    /// Suspended and `fetch_add(1)`-ed `sleep_seq`; pass the
    /// post-increment seq value so re-parks of the same thread can be
    /// detected via mismatch on a later drain.
    ///
    /// `wake_time` is the absolute tick value the thread should wake
    /// at, in the same units as `riscv::register::time::read()`.
    pub fn push(&mut self, thread: *mut Thread, wake_time: u64, sleep_seq: u64) {
        self.inner.push(Reverse(SleepEntry {
            wake_time,
            sleep_seq,
            thread,
        }));
    }

    /// Earliest deadline currently in the heap, or `None` if empty.
    /// May reflect a stale entry — pruning happens at drain time.
    /// Callers using this for `stimecmp` programming get an
    /// over-conservative wake (fires earlier than strictly needed),
    /// which is harmless: the manager runs, drains the stale entry,
    /// and re-arms with the next live `next_wake` on the way out.
    pub fn next_wake(&self) -> Option<u64> {
        self.inner.peek().map(|Reverse(e)| e.wake_time)
    }

    /// Pop every entry whose deadline has passed and is still validly
    /// parked on this entry, calling `cb(thread)` once per woken
    /// thread. Stale entries are popped silently; live entries with
    /// future deadlines are left in place.
    ///
    /// Stops at the first live entry whose deadline has not passed.
    /// Stale entries with future-dated deadlines stay in the heap
    /// until popped naturally — they cost peek work but persistent
    /// growth would indicate a wake path that never marks Ready, not
    /// a heap problem.
    ///
    /// The callback is responsible for the state transition (consume
    /// `wake_override` → `last_wake_reason`, store `state = Ready`,
    /// push onto the ready queue in Phase B). Keeping that out of the
    /// heap keeps this module policy-light and host-testable.
    pub fn drain_woken<F: FnMut(*mut Thread)>(&mut self, now: u64, mut cb: F) {
        while let Some(Reverse(top)) = self.inner.peek() {
            // SAFETY: heap entries are valid for the lifetime of
            // their referenced Thread allocation — see SleepEntry's
            // Send/Sync justification. We read two atomics; no
            // aliasing concerns with concurrent producers.
            let verdict = unsafe {
                let t = &*top.thread;
                let live_seq = t.sleep_seq.load(AtomicOrdering::Acquire);
                let live_state = t.state.load(AtomicOrdering::Acquire);
                classify(live_seq, live_state, top.sleep_seq)
            };
            match verdict {
                Verdict::Stale => {
                    self.inner.pop();
                }
                Verdict::Transient => {
                    // Park-in-flight: state hasn't yet hit Suspended
                    // (kthread_park's pre-handoff push window).
                    // Stop here; revisit next pass. We can't pop
                    // (the park is real) and we can't fire (deadline
                    // logic depends on a settled state). Heap order
                    // is preserved since we only peeked.
                    break;
                }
                Verdict::Live => {
                    if top.wake_time > now {
                        break;
                    }
                    let entry = self.inner.pop().unwrap();
                    cb(entry.0.thread);
                }
            }
        }
    }

    /// Total entries in the heap, including stale ones. Diagnostic
    /// only — a healthy heap has size ≈ N_suspended; persistent
    /// growth points at a wake path that fails to consume entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for SleepHeap {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Entry refers to a previous park or a completed park (eagerly
    /// woken / exited). Drop and continue.
    Stale,
    /// Park is in flight but state hasn't reached Suspended yet (the
    /// `kthread_park` push-before-asm-publish window). Leave the
    /// entry; next manager pass will see it settled.
    Transient,
    /// Entry matches the current park instance and state has
    /// committed to Suspended. Apply the deadline check.
    Live,
}

fn classify(live_seq: u64, live_state: usize, entry_seq: u64) -> Verdict {
    if live_seq != entry_seq {
        return Verdict::Stale;
    }
    if live_state == ThreadState::Suspended as usize {
        return Verdict::Live;
    }
    if live_state == ThreadState::Ready as usize {
        return Verdict::Stale;
    }
    if live_state == ThreadState::Exited as usize {
        return Verdict::Stale;
    }
    // Running / Assigned / Blocking with matching seq: park is in
    // flight, asm handoff hasn't published Suspended yet.
    Verdict::Transient
}
