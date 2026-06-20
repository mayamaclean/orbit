//! The global scheduler lock + its RAII guard.
//!
//! Exactly one hart at a time holds this lock; it serializes all
//! manager-owned state — the thread registry, the ready queue, and the
//! wake/promote/resume orchestration. (Historically called "MANAGER_LOCK"
//! in prose; the guard is named [`SchedGuard`] for what it guards rather
//! than who holds it.)
//!
//! ## Why it lives in `process`
//!
//! The lock is forced low in the crate graph: the capability layer's
//! `ThreadHandle::as_manager(&SchedGuard)` (coming in the cap refactor)
//! touches sealed `Thread` fields and so must live in this crate, which
//! means it must *see* `SchedGuard` — so the guard, and the lock it
//! wraps, live here. The protected data (the kernel's `Orbit` /
//! registry / page tables) stays in the upper crates; this is the
//! standard "guard token below the data it protects" layering.
//!
//! ## Acquire / release discipline — scoped, not an owned guard
//!
//! The only public entry is [`SchedGuard::try_with`]: it wins the CAS,
//! runs your closure with a `&SchedGuard`, and releases on return. The
//! guard is **never handed to a caller frame as an owned value** — on
//! purpose. An owned guard is a binding you can accidentally still hold
//! when you call a `-> !` function (e.g. `enter_hart_context`, which
//! srets into a thread and abandons this stack via the `kptr`
//! long-jump). A stack-abandoning divergence skips `Drop` entirely (and
//! `panic = abort` never unwinds either), so an owned guard would leak
//! the lock across that call. The scoped form confines the guard to the
//! closure, so the diverging dispatch naturally lives *after* `try_with`
//! returns, where no guard is in scope.
//!
//! [`SchedGuard::new`] is private + `unsafe`, so any construction other
//! than `try_with`'s internal acquire is an explicit `unsafe` within
//! this crate, flagged for review — a guard that doesn't truly hold the
//! lock would let the capability layer mint aliasing thread handles.
//!
//! ## Residual leak path (not yet defended)
//!
//! The scoped form makes leaks *rare* (the common path releases on
//! normal return) but cannot *prevent* them: if the closure body itself
//! calls a `-> !` function, it never returns, so `Drop` never runs —
//! the same leak. The type system can't catch this (`!` coerces to any
//! return type, and you can't enumerate all diverging functions). The
//! planned cause-agnostic catch — a holder-id lock value plus a
//! leak-heal + `debug_assert` at the top of `k_hart_loop` (the chokepoint
//! all control flow re-enters) — is **not yet implemented**. Until then,
//! the rule "do not diverge inside the closure" + the caller's
//! `sstatus.SIE = 0` discipline are what hold the line.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, Ordering};

/// The one global scheduler lock. Private to this module — the only API
/// is acquire-via-guard / release-on-drop; no code reads the raw flag.
static SCHED_LOCK: AtomicBool = AtomicBool::new(false);

/// RAII proof that **this hart** exclusively holds the scheduler lock.
/// Releases on `Drop`.
///
/// `!Send + !Sync` via `PhantomData<*const ()>`: the lock is hart-local
/// while held, so the guard must never migrate to another hart.
pub struct SchedGuard {
    _not_send: PhantomData<*const ()>,
}

impl SchedGuard {
    /// # Safety
    /// The caller must have exclusively acquired [`SCHED_LOCK`] (won the
    /// CAS) and must not construct a second live guard before this one
    /// drops. Private to this module; [`SchedGuard::try_acquire`] is the
    /// only path that discharges the obligation.
    unsafe fn new() -> Self {
        Self {
            _not_send: PhantomData,
        }
    }

    /// Acquire the sched lock, run `f` with the guard, and release on
    /// return. Returns `Some(f`'s result)` if the lock was won, or `None`
    /// if it's held elsewhere (caller backs off / WFIs).
    ///
    /// Scoped on purpose — see the module docs. Keep diverging / dispatch
    /// calls (`enter_hart_context`, etc.) **outside** this closure; their
    /// natural home is after `try_with` returns.
    ///
    /// `#[must_use]`: discarding the `Option` would hide whether the
    /// section actually ran.
    #[must_use]
    pub fn try_with<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&SchedGuard) -> R,
    {
        match SCHED_LOCK.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed) {
            // SAFETY: we just won the CAS, so we hold the lock
            // exclusively and no other live guard exists — a second
            // winner is impossible until `guard`'s `Drop` stores `false`.
            Ok(_) => {
                let guard = unsafe { SchedGuard::new() };
                let r = f(&guard);
                // Explicit release point (equivalent to scope-end drop;
                // stated for clarity). On a normal return this always
                // runs; an in-closure `-> !` is the residual leak the
                // module docs describe.
                drop(guard);
                Some(r)
            }
            Err(_) => None,
        }
    }
}

impl Drop for SchedGuard {
    fn drop(&mut self) {
        // Pairs with the SeqCst CAS in `try_acquire`; Release so the next
        // acquirer observes all manager-side writes from this section.
        SCHED_LOCK.store(false, Ordering::Release);
    }
}
