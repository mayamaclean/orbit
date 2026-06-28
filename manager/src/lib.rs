//! The kernel scheduling manager (Phase D extraction).
//!
//! Owns the **scheduling state** lifted out of kmain's `Orbit`
//! god-struct: the thread registry, the ready queue + sleep heap, the
//! wake/resume coordination, and the wake-queue statics. kmain's `Orbit`
//! keeps the **hardware state** (page tables, frame allocators, devices,
//! `page_cache`, the `processes` map) and holds a `manager: Manager`.
//!
//! Placement in the DAG is ABOVE orbit-core — it impls
//! `orbit_core::sched::Scheduler` and uses `ReadyQueue` / `PendingWork` /
//! the capability verbs from `process`:
//!
//! ```text
//! process  <-  orbit-core  <-  manager  <-  kmain
//! ```
//!
//! All thread mutation flows through the typestate capabilities
//! (`RunningThread` / `ManagerThread` / `ParkedMut` / `Runnable`) so the
//! bug-2 (live-frame write) and bug-4 (enqueue-without-marshal)
//! invariants hold by construction. This crate is the miri target for
//! the Phase E acceptance harness.

#![no_std]

extern crate alloc;

use alloc::collections::BTreeMap;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use orbit_core::pending_work::PendingWork;
use orbit_core::ready_queue::ReadyQueue;
use orbit_core::sleep_heap::SleepHeap;
use process::{Runnable, SchedGuard, Thread, ThreadHandle, ThreadState};
use thingbuf::StaticThingBuf;
use tracing::{error, info};

// ───────────────────────── wake-queue statics ─────────────────────────
//
// Relocated from kmain (Phase D). A `static`'s address is independent of
// the crate that declares it, so trap-context kmain producers reach
// these by path (`manager::X` / kmain's `crate::kernel` re-export) while
// the `Manager` methods drain them — no `&Manager` aliasing.

/// MPSC ring of `PendingWork` entries pushed by blocking-syscall paths;
/// drained by the manager's `drain_pending_work`. Cap 128 absorbs the
/// per-process startup burst (argv_envp per spawn) under the
/// wait_any_child smoke; a full ring EAGAINs the syscall (caller-visible)
/// so headroom is correctness-adjacent.
pub static MANAGER_WORK: StaticThingBuf<PendingWork, 128> = StaticThingBuf::new();

/// One completed virtio-blk chain: the packed `page_cache::CacheKey`
/// stashed at submit time plus the device status byte. `packed_key == 0`
/// is the empty/Default sentinel.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheFillEvent {
    pub packed_key: u64,
    pub status: u8,
}

/// Dedicated ring for virtio-blk completion events (producer: the blk
/// PLIC handler; consumer: the manager). Separate from `MANAGER_WORK` so
/// syscall pressure can never force the IRQ handler to drop a completion
/// (which would leave a page-cache slot `Loading` forever).
pub static CACHE_FILLS: StaticThingBuf<CacheFillEvent, 128> = StaticThingBuf::new();

/// Targeted "tickle a parked thread" events. Producers: PLIC IRQ
/// handlers, `update_tcp`, syscall paths publishing state a peer
/// sleep-polls. Consumer: the manager's `drain_wakes`. Not the cross-hart
/// IPI (that's `write_sswi`) — a "re-check the runnable predicate" signal.
pub static WAKE_QUEUE: StaticThingBuf<WakeEvent, 128> = StaticThingBuf::new();

/// High-water mark of `WAKE_QUEUE.len()` (monotonic via `fetch_max` in
/// [`wake_queue_push`]). Surfaces queue pressure for `query_stats`.
pub static WAKE_QUEUE_PEAK: AtomicU64 = AtomicU64::new(0);

/// Count of `WAKE_QUEUE.push()` attempts that EAGAIN'd (ring full). Each
/// drop is a missed wake; a non-zero counter says the cap is undersized.
pub static WAKE_QUEUE_DROPS: AtomicU64 = AtomicU64::new(0);

/// Stuck-thread watchdog state (temporary diagnostic — see
/// `Manager::check_stuck_threads`). `STUCK_TID` is the tid currently
/// observed parked-but-should-run; `STUCK_SINCE` is the tick it was first
/// seen. Manager-only access (under the scheduler lock).
pub static STUCK_TID: AtomicU32 = AtomicU32::new(0);
pub static STUCK_SINCE: AtomicU64 = AtomicU64::new(0);

/// Push a [`WakeEvent`] onto [`WAKE_QUEUE`] and update telemetry. Returns
/// `Err(ev)` if the queue is full — the caller decides whether to log,
/// retry, or coalesce. Trap-context-safe: two atomic ops on success.
pub fn wake_queue_push(ev: WakeEvent) -> Result<(), WakeEvent> {
    match WAKE_QUEUE.push(ev) {
        Ok(()) => {
            let depth = WAKE_QUEUE.len() as u64;
            let _ = WAKE_QUEUE_PEAK.fetch_max(depth, Ordering::Relaxed);
            Ok(())
        }
        Err(e) => {
            WAKE_QUEUE_DROPS.fetch_add(1, Ordering::Relaxed);
            Err(e.into_inner())
        }
    }
}

/// Lock-free MPSC ring of denial events produced by the dispatch-site
/// gate (any hart's `s_trap` on syscall denial). Consumer: the manager
/// folds each into the kernel-wide denial ring + the owning process's
/// counters. Lock-free is load-bearing (the trap path must not spin on
/// the scheduler lock to log a denial). Push-on-full drops + bumps
/// [`DENIAL_EVENTS_DROPPED`]. `None` is the Default empty slot.
pub static DENIAL_EVENT_QUEUE: StaticThingBuf<Option<orbit_abi::denial::DenialEvent>, 64> =
    StaticThingBuf::new();

/// Count of denial events dropped due to a full [`DENIAL_EVENT_QUEUE`].
pub static DENIAL_EVENTS_DROPPED: AtomicU64 = AtomicU64::new(0);

/// Targeted wake-up event. See [`WAKE_QUEUE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeEvent {
    /// Sentinel default — drained as a no-op (thingbuf needs `Default`).
    None,
    /// Wake k_net (targets the latched `net_thread_tid`; coarse pid=0
    /// fallback during the boot window).
    Net,
    /// Wake every thread of the given user pid (each re-checks + re-parks).
    Pid(u16),
    /// Wake a specific thread by tid.
    Tid(u32),
    /// Wake the thread parked on a process's key-event ring (tid latched
    /// at park time). Pushed from `input::dispatch`; `drain_wakes` runs
    /// `set_wake_reason_where(INPUT_IO, ..)` to eager-promote it.
    InputTid(u32),
    /// Wake the k_gpu compositor (targets `gpu_thread_tid`; coarse pid=0
    /// fallback during the boot window).
    Gpu,
    /// Wake the k_serial UART-drain thread (mirror of `Gpu`).
    Serial,
}

impl Default for WakeEvent {
    fn default() -> Self {
        WakeEvent::None
    }
}

/// One park notification queued by a parking hart for the manager to
/// fold into the sleep heap. The parking hart writes `Suspended` +
/// `fetch_add(1)`s `sleep_seq` first, then pushes this. `thread == null`
/// is the Default sentinel; the drain skips it.
#[derive(Clone, Copy)]
pub struct SleepNotice {
    pub wake_time: u64,
    pub sleep_seq: u64,
    pub thread: *mut Thread,
}

impl Default for SleepNotice {
    fn default() -> Self {
        Self {
            wake_time: 0,
            sleep_seq: 0,
            thread: core::ptr::null_mut(),
        }
    }
}

// SAFETY: `*mut Thread` points into the kernel thread registry, freed
// only from the manager's reap (same critical section as the inbox
// drain) — a notice always names a live allocation when popped.
unsafe impl Send for SleepNotice {}
unsafe impl Sync for SleepNotice {}

/// MPSC ring of [`SleepNotice`] entries pushed by parking harts, drained
/// into the manager's sleep heap. Cap 64 absorbs burst parks across harts.
pub static SLEEP_INBOX: StaticThingBuf<SleepNotice, 64> = StaticThingBuf::new();

/// Per-hart "thread just became Ready" notification, queued by
/// non-manager paths (e.g. preempt). The manager drains every per-hart
/// inbox into the ready queue at the head of each assign pass.
/// `thread == null` is the Default sentinel.
///
/// The `thread` pointer is **private**: a `ReadyNotice` can only be built
/// from a [`Runnable`] ([`Self::from_runnable`]) and the only way back to a
/// `Runnable` is [`Self::into_runnable`]. So the bug-4 guarantee ("only a
/// token marshaled under a won claim is dispatchable") survives the
/// per-hart inbox round-trip *by construction* — no code can fabricate a
/// dispatchable entry by stuffing a bare `*mut Thread` into the inbox.
#[derive(Clone, Copy)]
pub struct ReadyNotice {
    thread: *mut Thread,
}

impl Default for ReadyNotice {
    fn default() -> Self {
        Self {
            thread: core::ptr::null_mut(),
        }
    }
}

impl ReadyNotice {
    /// Stash a minted enqueue token as a bare pointer for the per-hart
    /// `READY_INBOX` (the thingbuf slot is `Copy`/`Default` and can't hold
    /// a non-`Clone` `Runnable`). Consumes the token; [`Self::into_runnable`]
    /// is the only way to get one back.
    #[inline]
    pub fn from_runnable(r: Runnable) -> Self {
        Self {
            thread: r.into_raw(),
        }
    }

    /// `true` for the `Default` sentinel (empty drained slot).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.thread.is_null()
    }

    /// Reconstitute the enqueue token drained from the inbox.
    ///
    /// # Safety
    /// Round-trips a token minted by [`Self::from_runnable`]: the pointer
    /// names the same live registry `Thread`, still genuinely dispatchable
    /// (the manager hasn't reaped it between push and drain). Must not be
    /// called on a `Default` (null) sentinel — guard with [`Self::is_empty`].
    #[inline]
    pub unsafe fn into_runnable(self) -> Runnable {
        unsafe { Runnable::from_raw(self.thread) }
    }
}

// SAFETY: same registry-lifetime argument as `SleepNotice`.
unsafe impl Send for ReadyNotice {}
unsafe impl Sync for ReadyNotice {}

/// Per-hart inbox of newly-Ready threads, indexed by hart id. SPSC per
/// hart (it pushes; manager pops); the manager drains all of them. Sized
/// by `orbit_core::MAX_HARTS` (re-homed there so this array compiles in
/// the manager crate). Cap 32 per hart is well above the working set.
pub static READY_INBOXES: [StaticThingBuf<ReadyNotice, 32>; orbit_core::MAX_HARTS] =
    [const { StaticThingBuf::new() }; orbit_core::MAX_HARTS];

/// The scheduling manager. Owns the thread registry, the ready queue +
/// sleep heap, and the latched driver-kthread tids — the scheduling
/// state lifted out of kmain's `Orbit`. kmain holds one of these as
/// `Orbit.manager` and threads the `SchedGuard` proof token through the
/// guard-bounded methods.
#[derive(Default)]
pub struct Manager {
    /// Thread registry, keyed by tid. The owning `ThreadHandle` wraps the
    /// `Box`-leaked `Thread`; `unregister` hands the box back to kmain for
    /// dealloc (which needs `kernel_pages` — kmain-only).
    threads: BTreeMap<u32, ThreadHandle>,

    /// FIFO of runnable threads. Populated by `drain_ready_inboxes`
    /// (per-hart inboxes), `drain_sleeps` (sleep-heap promotion),
    /// `set_wake_reason_where` (eager Suspended → Ready), and the
    /// creation paths (via `admit`). Drained by `get_runnable_thread`.
    ready: ReadyQueue,

    /// Min-heap of `(wake_time, sleep_seq, *mut Thread)` for Suspended
    /// sleepers; populated each pass by draining `SLEEP_INBOX`. See
    /// [orbit-core/src/sleep_heap.rs].
    sleeping: SleepHeap,

    /// TID of the k_net kernel thread, latched by `setup_igb`. `None`
    /// until then (and during the boot window before e1000 PLIC IRQs can
    /// fire); `WakeEvent::Net` falls back to a coarse pid=0 scan in that
    /// window. Once latched, `WakeEvent::Net` targets exactly this tid.
    net_thread_tid: Option<u32>,
    /// TID of the k_gpu compositor thread, latched in `setup_virtio_gpu`.
    /// Consumed by `WakeEvent::Gpu`; coarse pid=0 fallback before latch.
    gpu_thread_tid: Option<u32>,
    /// TID of the k_serial UART-drain thread, latched in
    /// `setup_serial_kthread`. Consumed by `WakeEvent::Serial`; coarse
    /// pid=0 fallback before latch.
    serial_thread_tid: Option<u32>,
}

impl Manager {
    pub const fn new() -> Self {
        Self {
            threads: BTreeMap::new(),
            ready: ReadyQueue::new(),
            sleeping: SleepHeap::new(),
            net_thread_tid: None,
            gpu_thread_tid: None,
            serial_thread_tid: None,
        }
    }

    // ───────────────────────── registry accessors ────────────────────
    //
    // The registry moved here from `Orbit`; these let the STAY-in-kmain
    // methods (creation, cleanup, resume, snapshot, pledge, exit-group)
    // reach threads by tid without owning the map. Reads return a
    // `&ThreadHandle` whose cap verbs (`peek`/`as_manager`) mutate through
    // the raw `*mut Thread`, never through the `&self.manager` borrow — so
    // a caller holding a disjoint `&mut self.processes` is fine.

    /// Insert a freshly-created thread into the registry.
    pub fn register(&mut self, tid: u32, handle: ThreadHandle) {
        self.threads.insert(tid, handle);
    }

    /// Remove and return the registry's owning handle for `tid`. kmain
    /// calls `into_box` on the result and runs `dealloc_thread`.
    pub fn unregister(&mut self, tid: u32) -> Option<ThreadHandle> {
        self.threads.remove(&tid)
    }

    /// Scrub every scheduler reference to a thread about to be freed (the
    /// reap path): its sleep-heap entry and any queued ready token. A
    /// freed `Box<Thread>` must leave no dangling entry — a later
    /// `drain_woken`/`pop_for` would otherwise dereference freed (and
    /// possibly recycled) memory, which can manifest as a permanently
    /// stalled sleep heap (every deeper sleeper frozen). Call this with
    /// the thread's pointer just before its `Box` is dropped; the registry
    /// removal is the separate [`Self::unregister`].
    pub fn forget_thread(&mut self, thread: *mut Thread) {
        self.sleeping.remove_thread(thread);
        self.ready.remove_thread(thread);
    }

    /// Borrow the registry handle for `tid`, if live.
    pub fn thread(&self, tid: u32) -> Option<&ThreadHandle> {
        self.threads.get(&tid)
    }

    /// Iterate the live registry (tid → handle). Used by the reaper,
    /// stats snapshot, and credential refresh walks.
    pub fn threads_iter(&self) -> alloc::collections::btree_map::Iter<'_, u32, ThreadHandle> {
        self.threads.iter()
    }

    /// Is `tid` already registered? Backs `next_tid`'s collision probe.
    pub fn tid_in_use(&self, tid: u32) -> bool {
        self.threads.contains_key(&tid)
    }

    /// Admit a runnable thread to the ready queue. The `Runnable` token
    /// proves the frame is marshaled (bug-4); the creation paths mint it
    /// fresh, the wake/resume paths via a `ParkedMut` verb.
    pub fn admit(&mut self, r: Runnable) {
        self.ready.push(r);
    }

    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    pub fn runnable_thread_count(&self) -> usize {
        self.threads
            .iter()
            .filter(|(_, t)| t.peek().state() == ThreadState::Ready as usize)
            .count()
    }

    // ────────────────────── latched driver-tid accessors ─────────────

    pub fn net_thread_tid(&self) -> Option<u32> {
        self.net_thread_tid
    }
    pub fn set_net_thread_tid(&mut self, tid: u32) {
        self.net_thread_tid = Some(tid);
    }
    pub fn gpu_thread_tid(&self) -> Option<u32> {
        self.gpu_thread_tid
    }
    pub fn set_gpu_thread_tid(&mut self, tid: u32) {
        self.gpu_thread_tid = Some(tid);
    }
    pub fn serial_thread_tid(&self) -> Option<u32> {
        self.serial_thread_tid
    }
    pub fn set_serial_thread_tid(&mut self, tid: u32) {
        self.serial_thread_tid = Some(tid);
    }

    // ───────────────────────── wake / resume drains ──────────────────

    /// Drain `WAKE_QUEUE`, folding each targeted wake into a
    /// `set_wake_reason_where` pass. Producers `fetch_or` into
    /// `wake_override`; the scheduler `swap(0)`s into `last_wake_reason`,
    /// so no two writers touch the same field. Coarse over-waking is
    /// harmless: each thread re-checks its own wait predicate on wake and
    /// re-parks if not actually ready.
    pub fn drain_wakes(&mut self, guard: &SchedGuard) {
        while let Some(mut slot) = WAKE_QUEUE.pop_ref() {
            let event = core::mem::take(&mut *slot);
            drop(slot);
            match event {
                WakeEvent::None => {}
                WakeEvent::Net => {
                    // Target k_net specifically once `setup_igb` has
                    // latched its tid. Before then (boot window), fall
                    // back to a coarse pid=0 scan — by the time anything
                    // pushes `WakeEvent::Net` for real (PLIC IRQ, user
                    // ch_yield) the latch has fired, so the fallback is
                    // just a safety net for self-pushes during k_net's
                    // own bringup.
                    match self.net_thread_tid {
                        Some(tid) => {
                            self.set_wake_reason_where(guard, process::wake_reason::TICKLE, |v| {
                                v.tid() == tid
                            })
                        }
                        None => {
                            self.set_wake_reason_where(guard, process::wake_reason::TICKLE, |v| {
                                v.pid() == 0
                            })
                        }
                    }
                }
                WakeEvent::Pid(pid) => {
                    self.set_wake_reason_where(guard, process::wake_reason::NET_IO, |v| {
                        v.pid() == pid
                    });
                }
                WakeEvent::Tid(tid) => {
                    self.set_wake_reason_where(guard, process::wake_reason::NET_IO, |v| {
                        v.tid() == tid
                    });
                }
                WakeEvent::InputTid(tid) => {
                    self.set_wake_reason_where(guard, process::wake_reason::INPUT_IO, |v| {
                        v.tid() == tid
                    });
                }
                WakeEvent::Gpu => {
                    // Mirror of the `WakeEvent::Net` branch: target k_gpu
                    // specifically once `setup_virtio_gpu` has latched its
                    // tid; coarse pid=0 fallback during the boot window.
                    match self.gpu_thread_tid {
                        Some(tid) => {
                            self.set_wake_reason_where(guard, process::wake_reason::TICKLE, |v| {
                                v.tid() == tid
                            })
                        }
                        None => {
                            self.set_wake_reason_where(guard, process::wake_reason::TICKLE, |v| {
                                v.pid() == 0
                            })
                        }
                    }
                }
                WakeEvent::Serial => {
                    // Same shape as `WakeEvent::Gpu` — target k_serial once
                    // `setup_serial_kthread` has latched its tid; coarse
                    // pid=0 fallback during the boot window.
                    match self.serial_thread_tid {
                        Some(tid) => {
                            self.set_wake_reason_where(guard, process::wake_reason::TICKLE, |v| {
                                v.tid() == tid
                            })
                        }
                        None => {
                            self.set_wake_reason_where(guard, process::wake_reason::TICKLE, |v| {
                                v.pid() == 0
                            })
                        }
                    }
                }
            }
        }
    }

    /// `fetch_or(reason)` into `wake_override` on every thread matching
    /// `pred`, eagerly promoting Suspended (and SIGNALED-Blocking) parkers
    /// to Ready in the same pass. The enqueue token is minted only by a
    /// `ParkedMut` verb under a claim, so "make dispatchable" is welded to
    /// "I marshaled the frame" (bug-4). `pub` because kmain's `nudge_*`
    /// helpers call it directly.
    pub fn set_wake_reason_where(
        &mut self,
        guard: &SchedGuard,
        reason: u64,
        mut pred: impl FnMut(&process::ThreadView) -> bool,
    ) {
        for (_, p) in self.threads.iter() {
            let view = p.peek();
            if !pred(&view) {
                continue;
            }
            // Snapshot the gating state *before* claiming. The split
            // mirrors the old `fetch_update` policy:
            //  - Suspended parkers (ms_sleep / read_key_event) always
            //    promote — they re-check their own condition on resume.
            //  - Blocking parkers (manager-resolved syscalls) promote
            //    ONLY with a SIGNALED completion slot; a Blocking + NONE
            //    wake is by construction stale (the parker's post-park
            //    re-check already won the take-CAS, or it's a doorbell
            //    misaimed at a blocking-syscall park). Promoting it would
            //    resume with `frame.regs` untouched — the 6001-as-byte-
            //    count QEMU repro.
            let st = view.state();
            let has_rets = view.pending_signaled();

            // OR the reason in for every matched thread — even a
            // non-promoted one records the bit for its next dispatch
            // (the old unconditional `fetch_or`).
            let mgr = p.as_manager(guard);
            mgr.note_wake(reason);

            // The enqueue token is minted only by a `ParkedMut` verb, so
            // "make dispatchable" is welded to "I marshaled the frame
            // under a claim" (**bug 4**). `resume_published` does the
            // take-CAS: it yields `None` if the parker's own re-check won
            // it, so exactly one of {this drain, the parker} enqueues.
            let runnable = if st == ThreadState::Suspended as usize {
                mgr.claim_parked().and_then(|parked| {
                    if has_rets {
                        parked.resume_published()
                    }
                    else {
                        Some(parked.promote_wake())
                    }
                })
            }
            else if st == ThreadState::Blocking as usize && has_rets {
                mgr.claim_parked()
                    .and_then(|parked| parked.resume_published())
            }
            else {
                None
            };
            if let Some(r) = runnable {
                // Just promoted Blocking/Suspended → Ready; queue it so
                // get_runnable_thread picks it up this same pass. Any
                // sleep-heap entry becomes stale (state mismatch) and is
                // reaped on the next drain_woken.
                self.ready.push(r);
            }
        }
    }

    /// Drain `SLEEP_INBOX` into the heap, then promote any sleepers whose
    /// deadline has passed to `Ready`. Called from kmain's `assign_threads`
    /// prelude so the dispatch that follows already sees freshly-promoted
    /// threads as Ready.
    pub fn drain_sleeps(&mut self, guard: &SchedGuard) {
        while let Some(mut slot) = SLEEP_INBOX.pop_ref() {
            let notice = core::mem::take(&mut *slot);
            drop(slot);
            if notice.thread.is_null() {
                continue;
            }
            // Race repair: if `set_wake_reason_where` ran while this
            // thread was mid-park (state=Running on its way to
            // Suspended), the eager-promote CAS failed but the
            // wake_override bit is set. Now that state has committed to
            // Suspended, check the bit before filing the entry — if
            // non-zero, eagerly promote here instead of letting the thread
            // wait for its deadline.
            //
            // SAFETY: heap/inbox entries name live registry threads —
            // freed only by the manager's own cleanup, in this same
            // critical section. The `guard` proves the lock is held.
            let mgr = unsafe { process::ManagerThread::from_raw(notice.thread, guard) };
            if mgr.view().has_pending_wake() {
                // `claim_parked` is the bug-2 guard + `promote_wake` mints
                // the enqueue token (bug-4). `None` ⇒ state was already
                // Ready (a concurrent promotion won); skip the heap push
                // either way (a Ready entry would be stale immediately).
                if let Some(parked) = mgr.claim_parked() {
                    self.ready.push(parked.promote_wake());
                }
                continue;
            }
            self.sleeping
                .push(notice.thread, notice.wake_time, notice.sleep_seq);
        }

        let now = riscv::register::time::read64();
        let ready = &mut self.ready;
        self.sleeping.drain_woken(now, |thread_ptr| {
            // SAFETY: heap entries name live registry threads — see
            // SLEEP_INBOX safety doc. We're under MANAGER_LOCK (the
            // `guard`); no other writer touches state/wake_override here.
            let mgr = unsafe { process::ManagerThread::from_raw(thread_ptr, guard) };
            // `promote_wake` consumes any pending wake_override bits into
            // last_wake_reason (so userspace can query why it woke;
            // timer-only wakes leave the bitmask 0), flips Ready, and
            // yields the enqueue token. `claim_parked` returns `None` if
            // the thread was already promoted this pass (e.g. by an eager
            // `set_wake_reason_where`) — skip the redundant push.
            if let Some(parked) = mgr.claim_parked() {
                ready.push(parked.promote_wake());
            }
        });
    }

    /// Drain every per-hart `READY_INBOXES` slot into the ready queue.
    /// Producers use these inboxes to publish Ready transitions without
    /// touching the ready queue directly (which is manager-only).
    pub fn drain_ready_inboxes(&mut self) {
        for inbox in READY_INBOXES.iter() {
            while let Some(mut slot) = inbox.pop_ref() {
                let notice = core::mem::take(&mut *slot);
                drop(slot);
                if notice.is_empty() {
                    continue;
                }
                // The notice round-trips a token `push_ready_notice` minted
                // from a `Runnable`. Reconstitute it to re-`push` onto the
                // ready queue.
                // SAFETY: the ptr was a legitimized `Runnable` at push time
                // and the thread is still live (not reaped under the lock).
                self.ready.push(unsafe { notice.into_runnable() });
            }
        }
    }

    /// Cycles until the earliest sleep-heap deadline, capped at the
    /// safety-net `cap` (so the manager still runs periodically and
    /// observes any new SLEEP_INBOX entries pushed after this read).
    /// Returns `cap` when the heap is empty or the earliest entry is
    /// further out than `cap`.
    ///
    /// Manager-only: callers must hold `MANAGER_LOCK` (the heap is not
    /// synchronized for concurrent peeks).
    pub fn next_sleep_in_cycles(&self, now: u64, cap: u64) -> u64 {
        match self.sleeping.next_wake() {
            Some(t) if t > now => (t - now).min(cap),
            Some(_) => 0,
            None => cap,
        }
    }

    /// Pop the next runnable thread for a hart, honoring affinity. All
    /// Ready transitions push onto the ready queue before this runs (the
    /// `assign_threads` prelude), so this is purely a queue pop.
    pub fn get_runnable_thread(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        self.ready.pop_for(hart_mask)
    }

    /// Publish completion results for a thread parked on a manager-
    /// resolved blocking syscall, then push `WakeEvent::Tid` so the wake
    /// drain covers the Suspended-parker case. Guard-free: it touches only
    /// the completion atoms (never the frame, never the state, mints no
    /// `Runnable`), so it can't trip bug-2/bug-4. No-op if the thread
    /// already exited.
    pub fn publish_pending_for_tid(&self, tid: u32, vals: &[isize]) {
        let Some(pt) = self.threads.get(&tid)
        else {
            // Thread exited mid-flight. No-op.
            return;
        };
        pt.publish_results(vals);
        let _ = wake_queue_push(WakeEvent::Tid(tid));
    }

    // ───────────────────────── diagnostics ───────────────────────────

    pub fn print_threads(&self) {
        for (_, t) in self.threads.iter() {
            let v = t.peek();
            info!("tid{}: state{}", v.tid(), v.state());
        }
    }

    /// **Stuck-thread watchdog (temporary diagnostic).** A thread that is
    /// `Blocking` with a `SIGNALED` completion slot, or `Blocking`/
    /// `Suspended` with a non-zero `wake_override`, *should* have been
    /// promoted to `Ready`. If the same tid stays in that "should-be-
    /// runnable but parked" condition longer than `THRESHOLD` ticks, a
    /// wake was lost — dump a full thread census. Called once per manager
    /// pass under the scheduler lock.
    pub fn check_stuck_threads(&self, now: u64) {
        // ~3 s at the 10 MHz `virt` timebase. A real cross-hart
        // park-commit transient resolves in microseconds, far under this.
        const THRESHOLD: u64 = 30_000_000;

        // First parked-but-should-run thread, if any. Three unambiguous
        // lost-wake signatures (a stale `wake_override` on a *Blocking*
        // waiter is deliberately NOT flagged — that's a legit waiter still
        // awaiting its publish):
        //   1. Blocking + SIGNALED   — published result never consumed.
        //   2. Suspended + wake_override — a tickle that wasn't promoted.
        //   3. Suspended + deadline passed — a sleep/yield_now never woken.
        let mut culprit: u32 = 0;
        for (_, p) in self.threads.iter() {
            let v = p.peek();
            let st = v.state();
            let blocking = st == ThreadState::Blocking as usize;
            let suspended = st == ThreadState::Suspended as usize;
            let stuck = (blocking && v.pending_signaled())
                || (suspended && v.has_pending_wake())
                || (suspended && (v.wake_time() as u64) <= now);
            if stuck {
                culprit = v.tid();
                break;
            }
        }

        if culprit == 0 {
            STUCK_TID.store(0, Ordering::Relaxed);
            return;
        }

        if STUCK_TID.load(Ordering::Relaxed) == culprit {
            if now.wrapping_sub(STUCK_SINCE.load(Ordering::Relaxed)) > THRESHOLD {
                error!(
                    "STUCK WATCHDOG: tid={culprit} parked but should be runnable (now={now}) — census:"
                );
                for (_, p) in self.threads.iter() {
                    let v = p.peek();
                    error!(
                        "  tid={} pid={} state={} signaled={} wake_pending={} wake_time={}",
                        v.tid(),
                        v.pid(),
                        v.state(),
                        v.pending_signaled(),
                        v.has_pending_wake(),
                        v.wake_time(),
                    );
                }
                // Re-arm the timer so we re-dump every THRESHOLD while stuck.
                STUCK_SINCE.store(now, Ordering::Relaxed);
            }
        }
        else {
            STUCK_TID.store(culprit, Ordering::Relaxed);
            STUCK_SINCE.store(now, Ordering::Relaxed);
        }
    }
}

impl orbit_core::sched::Scheduler for Manager {
    fn next_runnable(&mut self, hart_mask: u64) -> Option<*mut Thread> {
        // PThread wraps a raw ptr sourced from the thread registry (Box
        // allocations); returning it directly keeps provenance rooted at
        // that allocation — no `&mut` reborrow whose tag would be popped
        // on return (which would dangle the ptr stored in the target
        // hart's `current` slot).
        self.get_runnable_thread(hart_mask)
    }
}
