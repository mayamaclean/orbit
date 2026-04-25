//! Generic wait/signal primitives.
//!
//! Two flavors that share a shape (atomic counter, one party waits) but
//! diverge on what "wait" means:
//!
//! - [`CompletionHandle`] — heap-allocated, refcounted, sleeping waiter.
//!   A thread parks in `ThreadState::Blocking` with a clone of the
//!   handle; signalers (manager, trap handlers, kernel threads) call
//!   [`CompletionHandle::signal`] to store an `isize` result and flip
//!   the state. The manager loop scans for signaled handles and wakes
//!   their threads.
//!
//! - [`AckCounter`] — refcounted `AtomicU32` for the
//!   "1-sender / N-receivers / sender spins" pattern (§10's TLB
//!   shootdowns). Sender allocates the counter at `n`, hands clones to
//!   receivers, each calls [`AckCounter::decrement`] when done; sender
//!   calls [`AckCounter::wait_zero_spin`].
//!
//! Both are trap-context-safe past construction: no allocations, no
//! locks. Construction allocates (Arc) and must run in a context where
//! the global allocator is reachable.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicI64, AtomicU8, AtomicU32, Ordering};

const STATE_PENDING: u8 = 0;
/// Transient: a signaler CAS-claimed the slot and is mid-write. The
/// reader treats this the same as `PENDING` (not yet ready) — only
/// `SIGNALED` means the rets/count are coherent.
const STATE_WRITING: u8 = 1;
const STATE_SIGNALED: u8 = 2;

/// Max number of return-value slots a handle can carry. Maps to a0..a3
/// when the manager unblocks the parked thread. Picked at 4 because
/// the RISC-V calling convention reserves a0..a7 for return values
/// and the upper half is unused so far; bump if a syscall ever needs
/// more.
pub const MAX_RET_SLOTS: usize = 4;

/// Inner state behind a [`CompletionHandle`]. Refcounted via `Arc`; all
/// clones see the same atomics.
///
/// `ret_count` says how many of `rets` are valid — the manager writes
/// exactly that many regs (`regs[10..10+ret_count]`) on resume and
/// leaves the rest alone, so user-side a-regs that the handler doesn't
/// claim retain their trap-entry snapshot. Critical for syscalls with
/// a 1-reg return: pre-§8 the kernel never touched a1, and at least
/// some user-mode code paths (orbit-loader against orbit-rt) depend on
/// that survival even though the inline-asm contract technically
/// permits the clobber.
#[derive(Debug)]
pub struct CompletionInner {
    state: AtomicU8,
    ret_count: AtomicU8,
    rets: [AtomicI64; MAX_RET_SLOTS],
}

impl CompletionInner {
    const fn new() -> Self {
        Self {
            state: AtomicU8::new(STATE_PENDING),
            ret_count: AtomicU8::new(0),
            rets: [
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
                AtomicI64::new(0),
            ],
        }
    }
}

/// One-shot wait/signal handle. Cheap to clone (Arc bump). The first
/// `signal` wins; subsequent signals are silently dropped — handlers
/// that may race with revocation can rely on this idempotence.
#[derive(Clone, Debug)]
pub struct CompletionHandle {
    inner: Arc<CompletionInner>,
}

impl CompletionHandle {
    /// Allocate a fresh, pending handle. Allocates one Arc.
    pub fn new() -> Self {
        Self { inner: Arc::new(CompletionInner::new()) }
    }

    /// Single-register return: `result` lands in `regs[10]` (a0); a1+
    /// retain their trap-entry snapshot. The common case for blocking
    /// syscalls.
    pub fn signal(&self, result: isize) {
        self.signal_n(&[result]);
    }

    /// Two-register return: `r0` → a0, `r1` → a1; a2+ retain their
    /// trap-entry snapshot. Used by `create_netch` returning
    /// `(vaddr, fd)`.
    pub fn signal_pair(&self, r0: isize, r1: isize) {
        self.signal_n(&[r0, r1]);
    }

    /// N-register return. `vals` length is clamped to
    /// [`MAX_RET_SLOTS`]; excess values are silently dropped.
    /// Trap-context-safe.
    ///
    /// Race shape: the first caller CAS-claims the slot
    /// (`PENDING → WRITING`), which excludes any concurrent signaler
    /// from touching `rets`/`ret_count`. Late callers observe
    /// non-PENDING and bail — the primitive is idempotent so a
    /// revoke racing a legitimate completion is safe. The owning
    /// writer commits with a Release store of `SIGNALED`, which
    /// publishes the rets/count writes to any Acquire reader.
    pub fn signal_n(&self, vals: &[isize]) {
        if self.inner.state.compare_exchange(
            STATE_PENDING,
            STATE_WRITING,
            Ordering::Acquire,
            Ordering::Acquire,
        ).is_err() {
            return;
        }
        let n = core::cmp::min(vals.len(), MAX_RET_SLOTS);
        for i in 0..n {
            self.inner.rets[i].store(vals[i] as i64, Ordering::Relaxed);
        }
        self.inner.ret_count.store(n as u8, Ordering::Relaxed);
        self.inner.state.store(STATE_SIGNALED, Ordering::Release);
    }

    /// `true` once any signaler has run. Acquire-ordered against the
    /// signaler's Release in `signal_n`.
    pub fn is_signaled(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) == STATE_SIGNALED
    }

    /// How many a-regs the manager should write on resume. Zero before
    /// signal; up to [`MAX_RET_SLOTS`] after.
    pub fn ret_count(&self) -> usize {
        self.inner.ret_count.load(Ordering::Acquire) as usize
    }

    /// Read the i-th return slot. Caller must respect
    /// `i < ret_count()` — out-of-range indices return whatever the
    /// slot was initialized to (zero).
    pub fn ret(&self, i: usize) -> isize {
        if i >= MAX_RET_SLOTS {
            return 0;
        }
        self.inner.rets[i].load(Ordering::Acquire) as isize
    }
}

impl Default for CompletionHandle {
    fn default() -> Self { Self::new() }
}

/// Refcounted counter for "1 sender / N receivers / sender spins".
/// §10's TLB-shootdown protocol uses one of these per request: sender
/// allocates with `new(n)`, hands clones to each target hart, receivers
/// `decrement` after servicing, sender `wait_zero_spin`s before
/// returning. Built here so the type is in place when §10 lands; §8
/// has no consumer.
#[derive(Clone, Debug)]
pub struct AckCounter {
    inner: Arc<AtomicU32>,
}

impl AckCounter {
    /// Allocate a counter starting at `n`. `n == 0` is legal (waiter
    /// returns immediately).
    pub fn new(n: u32) -> Self {
        Self { inner: Arc::new(AtomicU32::new(n)) }
    }

    /// Decrement by one. Saturates at zero — extra decrements are
    /// safe but should not happen in correct use. Trap-context-safe.
    pub fn decrement(&self) {
        let _ = self.inner.fetch_update(
            Ordering::Release,
            Ordering::Relaxed,
            |v| if v == 0 { None } else { Some(v - 1) },
        );
    }

    /// Spin until the counter reaches zero. Cheap on RISC-V: just a
    /// load + `spin_loop` hint. Caller must guarantee receivers will
    /// actually decrement, otherwise this is a deadlock.
    pub fn wait_zero_spin(&self) {
        while self.inner.load(Ordering::Acquire) != 0 {
            core::hint::spin_loop();
        }
    }

    /// Snapshot current count. Useful for logging/diagnostics.
    pub fn load(&self) -> u32 {
        self.inner.load(Ordering::Acquire)
    }
}
