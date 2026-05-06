//! Block-keyed page cache. One page-aligned LBA → one cached page.
//!
//! Mediates every fs read in the kernel: cached pages copy out
//! synchronously, misses register a waiter on a Loading slot and
//! park the requesting tid until the DMA lands. Multiple concurrent
//! misses on the same key coalesce onto one in-flight DMA.
//!
//! Lifecycle:
//! 1. `lookup(key)` — read-only peek. Hit returns `Ready { frame,
//!    valid_bytes }`; coalescing returns `Loading`; miss returns
//!    `None`.
//! 2. On Hit, caller does the synchronous copy and calls
//!    `record_hit` + `touch_lru` to update stats / LRU order.
//! 3. On Loading, caller calls `register_waiter(key, w)` and parks
//!    the waiter's tid.
//! 4. On miss, caller calls `begin_load(key, valid_bytes, w)` to
//!    allocate a slot + frame, install the initial waiter, and
//!    obtain the DMA target PA. The caller then submits the DMA
//!    referencing `key` so the IRQ→manager `CacheFill` event lands
//!    here.
//! 5. On `CacheFill`, manager calls `complete_slot(key, status)`.
//!    On success, the slot transitions to Ready and the returned
//!    waiter list is dispatched (per-waiter copy + tid resume) by
//!    the caller. On failure, the slot is dropped and the frame
//!    recycled.
//!
//! All state mutations run under `MANAGER_LOCK` — no internal
//! locking. The cache is a `BTreeMap` over keys plus a `VecDeque`
//! tracking LRU order; both are tiny at the configured capacities
//! (typical: 64).
//!
//! Eviction targets only `Ready` slots: a `Loading` slot has an
//! in-flight DMA writing into its frame and waiters depending on
//! its eventual transition, so it cannot be reclaimed. If every
//! slot is `Loading` and the pool is empty, `begin_load` returns
//! `Err(PoolExhausted)` and the caller signals `-EAGAIN` to the
//! requesting thread; raise `capacity` if this becomes common.

use core::alloc::Layout;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use mmu::PAGE_SIZE;
use mmu::sv48::PhysAddr;
use tracing::warn;

use crate::kernel::memmap::KernelPages;
use crate::kernel::shared_frame::SharedFrame;

/// Cache key: `(dev_id, page-aligned lba)`. Today's single tarfs
/// mount pins `dev = 1`; multi-mount layers will assign distinct ids
/// without changing the on-disk layout.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct CacheKey {
    pub dev: u8,
    /// Page-aligned (multiple of `PAGE_SIZE / SECTOR_SIZE = 8`)
    /// sector index. Bits 0..=54 used; bits 55..=63 reserved by the
    /// `pack`/`unpack` layout (see [`pack`]).
    pub lba: u64,
}

/// Bit layout (low → high):
///   bits  0..=7    → dev (8 bits)
///   bits  8..=62   → lba (55 bits → 2^55 sectors × 512 B = 16 ZiB)
///   bit   63       → occupied flag (1 = in flight, 0 = empty)
const SLOT_OCCUPIED: u64 = 1 << 63;
const SLOT_DEV_MASK: u64 = 0xFF;
const SLOT_LBA_MASK: u64 = 0x7FFF_FFFF_FFFF_FF00;

/// Pack a `CacheKey` into the `AtomicU64` form used by the
/// virtio-blk `IN_FLIGHT` side-table. The OR with `SLOT_OCCUPIED`
/// makes any unpack-from-zero (cleared slot) cleanly distinguishable
/// from a packed key whose dev/lba happen to be zero.
pub const fn pack(key: CacheKey) -> u64 {
    SLOT_OCCUPIED | ((key.lba << 8) & SLOT_LBA_MASK) | key.dev as u64
}

/// Inverse of [`pack`]. Returns `None` for cleared slots (no
/// occupied bit set) so an `AtomicU64::swap(0, ...)` cleanly reports
/// "this slot held nothing" without a separate sentinel.
pub const fn unpack(packed: u64) -> Option<CacheKey> {
    if packed & SLOT_OCCUPIED == 0 {
        return None;
    }
    Some(CacheKey {
        dev: (packed & SLOT_DEV_MASK) as u8,
        lba: (packed & SLOT_LBA_MASK) >> 8,
    })
}

/// Where a completed-DMA waiter wants its bytes copied. Both
/// variants signal the waiting tid by writing the byte count into
/// `regs[10]` and transitioning Suspended → Runnable; only the copy
/// destination differs.
pub enum Waiter {
    /// User-buffer destination — fs_read syscall path. The manager
    /// installs a `UserPageWindow` (or KDMAP-aliases the user PA) and
    /// memcpys `cache_frame.kva()[intra..intra+len]` into
    /// `user_page_pa[user_page_off..user_page_off+len]`.
    ///
    /// `pid` rides on the waiter so the completion arm can skip the
    /// copy if the process exited mid-flight (its `user_page_pa`
    /// may have been reallocated to another tenant); `tid` is the
    /// resume target.
    User {
        tid: u32,
        pid: u16,
        intra: u32,
        user_page_pa: PhysAddr,
        user_page_off: u32,
        len: u32,
    },
    /// Kernel-buffer destination — path-mode spawn ELF read, future
    /// kernel-side fs reads. Manager memcpys directly into
    /// `dst_kva`.
    Kernel {
        tid: u32,
        intra: u32,
        dst_kva: usize,
        len: u32,
    },
}

/// Slot lifecycle. A key transitions Absent → Loading → Ready
/// (success) or Absent → Loading → Absent (failure / status != 0).
/// Eviction is Ready → Absent only — Loading slots are pinned by
/// their in-flight DMA and waiter list.
pub enum SlotState {
    Loading {
        /// DMA destination. Lives in the slot for the slot's
        /// lifetime; not handed to the IRQ side-table any more —
        /// the IRQ identifies the slot by `CacheKey` and the slot
        /// keeps the frame alive.
        frame: SharedFrame,
        /// Bytes that will be valid in the page once the DMA lands
        /// (clamped at file size by the caller at submit time).
        /// Stored alongside the frame so `complete_slot` doesn't
        /// need it threaded through the IRQ→manager path.
        valid_bytes: u32,
        waiters: Vec<Waiter>,
    },
    Ready {
        frame: SharedFrame,
        valid_bytes: u32,
    },
}

impl SlotState {
    pub fn frame(&self) -> &SharedFrame {
        match self {
            SlotState::Loading { frame, .. } => frame,
            SlotState::Ready { frame, .. } => frame,
        }
    }

    pub fn is_loading(&self) -> bool {
        matches!(self, SlotState::Loading { .. })
    }
}

/// Counters exposed via the kernel-stats syscall. All updates run
/// under `MANAGER_LOCK`, so plain `+=` is fine.
#[derive(Default, Clone, Copy)]
pub struct PageCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub coalesced: u64,
    pub evictions: u64,
    pub pool_exhausted: u64,
    pub io_errors: u64,
    pub inflight_pages: u64,
    pub ready_pages: u64,
}

#[derive(Debug)]
pub enum CacheErr {
    /// `register_waiter` called on a key the cache has never seen,
    /// or whose Loading slot was already drained by a prior
    /// `complete_slot`.
    NotFound,
    /// `register_waiter` called on a Ready slot — caller should have
    /// taken the synchronous-copy path instead.
    WrongState,
    /// `begin_load` couldn't acquire a frame: pool empty and every
    /// in-cache slot is Loading. Caller should signal `-EAGAIN` to
    /// the parked thread.
    PoolExhausted,
    /// `begin_load` called on a key already in the cache. Caller
    /// should have observed the existing slot via `lookup` and taken
    /// the Ready or register_waiter path.
    AlreadyPresent,
}

pub struct PageCache {
    entries: BTreeMap<CacheKey, SlotState>,
    /// MRU at front, LRU at back. **Ready slots only** —
    /// `begin_load` does not push, `complete_slot` pushes on the
    /// success transition, `complete_slot` failure does not touch
    /// (the key was never enqueued). Eviction is therefore a single
    /// `pop_back`; touch-on-hit is a linear remove + push_front.
    lru: VecDeque<CacheKey>,
    /// Pre-allocated frames recycled on eviction / load-failure.
    /// `Vec::pop` for acquire, `Vec::push` for return — order
    /// doesn't matter, all frames are interchangeable.
    pool: Vec<SharedFrame>,
    capacity: usize,
    pub stats: PageCacheStats,
}

impl PageCache {
    /// Allocate `capacity` page-sized frames from `kernel_pages` and
    /// stash them in the pool. Returns `None` if the kernel pool is
    /// exhausted partway through; partial allocations are dropped
    /// (their pages return to `kernel_pages` via SharedFrame's drop
    /// hook).
    pub fn with_capacity(kernel_pages: &mut KernelPages, capacity: usize) -> Option<Self> {
        let layout = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).ok()?;
        let mut pool = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            let Some(frame) = SharedFrame::alloc(kernel_pages, layout)
            else {
                warn!(
                    "page_cache: kernel_pages exhausted after {}/{capacity} frames",
                    pool.len()
                );
                return None;
            };
            pool.push(frame);
        }
        Some(Self {
            entries: BTreeMap::new(),
            lru: VecDeque::with_capacity(capacity),
            pool,
            capacity,
            stats: PageCacheStats::default(),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Snapshot the live counters. `inflight_pages` and
    /// `ready_pages` are derived from current state (not running
    /// tallies) so they're always self-consistent with `entries`.
    pub fn stats(&self) -> PageCacheStats {
        let mut s = self.stats;
        let mut inflight = 0u64;
        let mut ready = 0u64;
        for slot in self.entries.values() {
            match slot {
                SlotState::Loading { .. } => inflight += 1,
                SlotState::Ready { .. } => ready += 1,
            }
        }
        s.inflight_pages = inflight;
        s.ready_pages = ready;
        s
    }

    /// Read-only peek. Caller dispatches: `Some(Ready)` →
    /// synchronous copy + `record_hit` + `touch_lru`; `Some(Loading)`
    /// → `register_waiter`; `None` → `begin_load`.
    pub fn lookup(&self, key: CacheKey) -> Option<&SlotState> {
        self.entries.get(&key)
    }

    /// Bump a Ready key to MRU. No-op if the key is missing or
    /// Loading (Loading slots aren't in `lru` until `complete_slot`
    /// promotes them).
    pub fn touch_lru(&mut self, key: CacheKey) {
        if !matches!(self.entries.get(&key), Some(SlotState::Ready { .. })) {
            return;
        }
        // Linear scan — `capacity` is ~64; bypass the cost of a
        // hash-indexed LRU until profiling says otherwise.
        if let Some(pos) = self.lru.iter().position(|k| *k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_front(key);
    }

    /// Increment the hit counter. Caller invokes after copying out
    /// of a Ready slot. Decoupled from `lookup` so the cache doesn't
    /// have to know whether the caller actually went through with
    /// the copy (e.g., bailed on a process_alive check).
    pub fn record_hit(&mut self) {
        self.stats.hits += 1;
    }

    /// Append a waiter to an existing Loading slot.
    ///
    /// Caller must have observed `Loading` via `lookup` immediately
    /// before. Returns `WrongState` if the slot transitioned to
    /// Ready in between (would only happen if the caller dropped
    /// MANAGER_LOCK between lookup and register, which the
    /// architecture forbids), `NotFound` if the slot vanished.
    pub fn register_waiter(&mut self, key: CacheKey, waiter: Waiter) -> Result<(), CacheErr> {
        match self.entries.get_mut(&key) {
            None => Err(CacheErr::NotFound),
            Some(SlotState::Ready { .. }) => Err(CacheErr::WrongState),
            Some(SlotState::Loading { waiters, .. }) => {
                waiters.push(waiter);
                self.stats.coalesced += 1;
                Ok(())
            }
        }
    }

    /// Allocate a frame, install the slot as Loading with `waiter`
    /// as the first waiter and the given `valid_bytes`, and return
    /// the DMA destination PA.
    ///
    /// `valid_bytes` is the count of file-valid bytes the page will
    /// hold once the DMA lands (`min(PAGE_SIZE, file_size -
    /// page_off)` at submit time). It rides on the slot so
    /// `complete_slot` doesn't need it threaded through the
    /// IRQ→manager `CacheFill` payload.
    ///
    /// Errors:
    /// - `AlreadyPresent` — caller didn't lookup first; some other
    ///   waiter beat them to this key. Caller should retry the
    ///   lookup → register_waiter path.
    /// - `PoolExhausted` — pool empty and no Ready slot is
    ///   evictable. Caller signals `-EAGAIN` to the parked thread.
    pub fn begin_load(
        &mut self,
        key: CacheKey,
        valid_bytes: u32,
        waiter: Waiter,
    ) -> Result<PhysAddr, CacheErr> {
        if self.entries.contains_key(&key) {
            return Err(CacheErr::AlreadyPresent);
        }
        let frame = self.acquire_frame()?;
        let pa = frame.pa();
        self.entries.insert(
            key,
            SlotState::Loading {
                frame,
                valid_bytes,
                waiters: alloc::vec![waiter],
            },
        );
        // Loading slots stay out of `lru` so eviction sees only
        // valid candidates. `complete_slot` pushes on the success
        // transition.
        self.stats.misses += 1;
        Ok(pa)
    }

    /// Resolve a `CacheFill { key, status }` event. On success
    /// transitions the slot to Ready (using its stored
    /// `valid_bytes`) and bumps it to MRU; on failure removes the
    /// slot and recycles the frame. Either way returns the slot's
    /// waiter list so the caller can iterate, copy, and resume each
    /// waiter's tid.
    ///
    /// Returns an empty Vec if the slot was missing or already
    /// Ready (invariant violation — log + drop) so callers don't
    /// have to special-case those edges.
    pub fn complete_slot(&mut self, key: CacheKey, status: u8) -> Vec<Waiter> {
        let Some(slot) = self.entries.remove(&key)
        else {
            warn!("page_cache: complete_slot for absent key {key:?}");
            return Vec::new();
        };
        let (frame, valid_bytes, waiters) = match slot {
            SlotState::Loading {
                frame,
                valid_bytes,
                waiters,
            } => (frame, valid_bytes, waiters),
            SlotState::Ready { frame, valid_bytes } => {
                warn!("page_cache: complete_slot for Ready key {key:?} — putting it back");
                self.entries
                    .insert(key, SlotState::Ready { frame, valid_bytes });
                return Vec::new();
            }
        };

        if status == 0 {
            // First time this key enters `lru` — Loading slots are
            // never enqueued, so this is the canonical "newly
            // available, MRU" insertion.
            self.entries
                .insert(key, SlotState::Ready { frame, valid_bytes });
            self.lru.push_front(key);
        }
        else {
            self.stats.io_errors += 1;
            // Drop slot, recycle frame. No `lru` touch needed: the
            // key never made it in.
            self.pool.push(frame);
        }

        waiters
    }

    /// Acquire a frame: pool first, else evict the LRU Ready slot.
    /// `lru` contains only Ready keys by construction
    /// ([`begin_load`] doesn't enqueue, [`complete_slot`] enqueues
    /// on success), so `pop_back` is guaranteed to land on an
    /// evictable victim. Empty `lru` + empty pool means every slot
    /// is Loading → [`CacheErr::PoolExhausted`].
    fn acquire_frame(&mut self) -> Result<SharedFrame, CacheErr> {
        if let Some(f) = self.pool.pop() {
            return Ok(f);
        }
        let Some(key) = self.lru.pop_back()
        else {
            self.stats.pool_exhausted += 1;
            return Err(CacheErr::PoolExhausted);
        };
        let SlotState::Ready { frame, .. } = self
            .entries
            .remove(&key)
            .expect("victim key in lru must have an entry")
        else {
            unreachable!("lru should hold only Ready keys");
        };
        self.stats.evictions += 1;
        Ok(frame)
    }
}
