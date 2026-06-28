//! Generic wait/signal primitives.
//!
//! Two flavors that share a shape (atomic counter, one party waits) but
//! diverge on what "wait" means:
//!
//! - [`CompletionHandle`] — heap-allocated, refcounted, sleeping waiter.
//!   A thread parks in `ThreadState::Blocking` with a clone of the
//!   handle; signalers (manager, trap handlers, kernel threads) call
//!   [`CompletionHandle::signal`] to store the result(s) and claim the
//!   waiter, which fires the registered wake hook (kmain's
//!   `wake_blocked_inline`) to marshal the rets and enqueue the thread
//!   Ready — there is no manager scan of signaled handles.
//!
//! - [`AckCounter`] — refcounted counter for the
//!   "1-sender / N-receivers / sender spins" pattern (TLB
//!   shootdowns). Sender allocates the counter at `n`, hands clones to
//!   receivers, each calls [`AckCounter::decrement`] when done; sender
//!   calls [`AckCounter::wait_zero_spin`].
//!
//! Both are trap-context-safe past construction: no allocations, no
//! locks. Construction allocates (Arc) and must run in a context where
//! the global allocator is reachable.

use alloc::sync::Arc;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicI64, AtomicPtr, AtomicU8, AtomicUsize, Ordering};

use crate::Thread;

/// Wake-hook signature. Called from `signal_n` with the parked thread
/// pointer that the signaler atomically claimed via `take_waiter`.
/// kmain registers an implementation that marshals the handle's rets
/// into the thread's frame, marks it Ready, and pushes onto the
/// per-hart ready inbox.
pub type WakeHook = fn(*mut Thread);

/// Storage for the registered hook. `0` is the "uninstalled" sentinel
/// — at boot time the hook isn't registered yet, and host tests
/// generally never install one. Round-tripping fn pointers through
/// `usize` is lossless on RV64; const-eval forbids the cast in a
/// static initializer though, hence the lazy install pattern.
static WAKE_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Install the wake hook called by `signal_n` for parked threads.
/// Call once at boot from kmain. Subsequent calls overwrite (the last
/// hook wins) — no current need for replacement so tests just
/// initialize once.
pub fn set_wake_hook(hook: WakeHook) {
    WAKE_HOOK.store(hook as usize, Ordering::Release);
}

fn invoke_wake_hook(t: *mut Thread) {
    let raw = WAKE_HOOK.load(Ordering::Acquire);
    if raw == 0 {
        // No hook installed (boot window or host test); silently
        // swallow. The signal still completed (state=SIGNALED) — a
        // future scan or polled re-check could still observe it.
        return;
    }
    // SAFETY: `raw != 0` means a `WakeHook` fn pointer was stored
    // via `set_wake_hook`. Function pointers round-trip through
    // `usize` losslessly on RV64.
    let hook: WakeHook = unsafe { core::mem::transmute(raw) };
    hook(t);
}

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
/// a 1-reg return: earlier the kernel never touched a1, and at least
/// some user-mode code paths (orbit-loader against orbit-rt) depend on
/// that survival even though the inline-asm contract technically
/// permits the clobber.
#[derive(Debug)]
pub struct CompletionInner {
    state: AtomicU8,
    ret_count: AtomicU8,
    rets: [AtomicI64; MAX_RET_SLOTS],
    /// Parked thread to wake when this handle signals. Set by the
    /// park path via [`CompletionHandle::set_waiter`] before publishing
    /// `state=Blocking`; consumed by either `signal_n` (signaler wins
    /// the race) or `take_waiter` from the park-time re-check
    /// (parker wins). `null` means no waiter is registered.
    waiter: AtomicPtr<Thread>,
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
            waiter: AtomicPtr::new(null_mut()),
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
        Self {
            inner: Arc::new(CompletionInner::new()),
        }
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
        if self
            .inner
            .state
            .compare_exchange(
                STATE_PENDING,
                STATE_WRITING,
                Ordering::Acquire,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        let n = core::cmp::min(vals.len(), MAX_RET_SLOTS);
        for i in 0..n {
            self.inner.rets[i].store(vals[i] as i64, Ordering::Relaxed);
        }
        self.inner.ret_count.store(n as u8, Ordering::Relaxed);
        self.inner.state.store(STATE_SIGNALED, Ordering::Release);
        // Race-free wake: atomically claim the waiter slot. If the
        // parker hasn't published its set_waiter yet we get null and
        // do nothing — the parker's post-park re-check will see
        // is_signaled and un-park itself. If it has, we own the
        // wake; invoke the registered hook.
        let waiter = self.inner.waiter.swap(null_mut(), Ordering::AcqRel);
        if !waiter.is_null() {
            invoke_wake_hook(waiter);
        }
    }

    /// Register `thread` as the parker on this handle. Caller is
    /// responsible for the post-publish re-check pattern: after this
    /// call, store `state=Blocking` and then call `take_waiter`; if
    /// the handle was already signaled (and `take_waiter` returned
    /// the parker's own ptr), un-park the thread inline.
    pub fn set_waiter(&self, thread: *mut Thread) {
        self.inner.waiter.store(thread, Ordering::Release);
    }

    /// Atomically claim and clear the waiter slot. Returns the
    /// previously-set ptr (or null if none / already taken). Used
    /// by the parker's post-park re-check to detect races where
    /// the signal arrived between `set_waiter` and the re-check.
    pub fn take_waiter(&self) -> *mut Thread {
        self.inner.waiter.swap(null_mut(), Ordering::AcqRel)
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

    /// Decompose into a raw `*const CompletionInner`. Caller takes
    /// ownership of the Arc strong count; pair with [`from_raw`]
    /// (or `Arc::from_raw`) to reclaim. Used by lock-free parked-
    /// reader slots that store handles as `AtomicPtr`.
    pub fn into_raw(self) -> *const CompletionInner {
        Arc::into_raw(self.inner)
    }

    /// Reclaim a handle previously emitted by [`into_raw`]. Each
    /// raw pointer must be reclaimed exactly once.
    ///
    /// # Safety
    /// `raw` must have come from `into_raw` (or `Arc::into_raw` on
    /// an `Arc<CompletionInner>`) and must not have been reclaimed
    /// already. The Arc strong count reverts to ownership of the
    /// returned handle.
    pub unsafe fn from_raw(raw: *const CompletionInner) -> Self {
        Self {
            inner: unsafe { Arc::from_raw(raw) },
        }
    }
}

impl Default for CompletionHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Refcounted counter for "1 sender / N receivers / sender spins".
/// The TLB-shootdown protocol uses one of these per request: sender
/// allocates with `new(n)`, hands clones to each target hart, receivers
/// `decrement` after servicing, sender `wait_zero_spin`s before
/// returning.
#[derive(Clone, Debug)]
pub struct AckCounter {
    inner: Arc<AtomicUsize>,
}

impl AckCounter {
    /// Allocate a counter starting at `n`. `n == 0` is legal (waiter
    /// returns immediately).
    pub fn new(n: usize) -> Self {
        Self {
            inner: Arc::new(AtomicUsize::new(n)),
        }
    }

    /// Decrement by one. Saturates at zero — extra decrements are
    /// safe but should not happen in correct use. Trap-context-safe.
    pub fn decrement(&self) {
        let _ = self
            .inner
            .fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
                if v == 0 { None } else { Some(v - 1) }
            });
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
    pub fn load(&self) -> usize {
        self.inner.load(Ordering::Acquire)
    }
}
