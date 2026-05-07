//! Per-process drawing-surface registry.
//!
//! Mirrors the [`stdin`](crate::kernel::stdin) module's pattern:
//! [`SURFACE_TABLE`] is a global `BTreeMap<pid, Arc<ProcessSurfaces>>`,
//! touched at:
//!
//! - process create — `register(pid)` inserts an empty `ProcessSurfaces`.
//! - `dealloc_process` — `unregister(pid)` removes the entry; the caller
//!   walks the returned `Arc` to free each surface's backing.
//! - manager-handled syscalls — `get(pid)` looks up the Arc to mutate.
//! - `fb_present` syscall — `get(pid)` looks up the Arc to copy out
//!   the metadata for one handle. Read-only on the hot path.
//!
//! The Display compositor in [`drivers::display`](crate::drivers::display)
//! holds a *snapshot* of the active source's surface entry (kdmap KVA +
//! dims + format) on its own state, so the per-frame blit doesn't go
//! through this table at all. A `fb_present` Cmd carries the snapshot
//! data with it; if the snapshot diverges from the current entry (e.g.
//! the user destroyed the surface mid-flight), the Cmd was already
//! validated against the current entry at syscall time, so we just blit
//! the bytes that exist and accept that a destroyed surface gets a
//! best-effort final paint.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use orbit_abi::fb::FbFormat;
use orbit_abi::layout::UPROC_SHARED_BASE;
use process::PhysBacking;
use spin::Mutex;

/// One drawing surface owned by a process. Backing frame travels with
/// the entry — when [`ProcessSurfaces::remove`] takes the entry out of
/// the per-process map, the caller is responsible for routing
/// `backing` back to `kernel_pages`.
#[derive(Debug)]
pub struct SurfaceEntry {
    /// User VA the surface is mapped at (in the calling process's
    /// shared range). Set at create time; immutable.
    pub user_va: u64,
    /// Kernel-side KDMAP alias of the same physical pages. Used by the
    /// compositor for the per-frame blit.
    pub kdmap_kva: u64,
    pub width: u32,
    pub height: u32,
    pub format: FbFormat,
    /// `width * height * bpp`, rounded up to `PAGE_SIZE`. The mapped
    /// region is exactly this many bytes.
    pub size_bytes: usize,
    /// Backing frame ownership. Surfaces own their own backing
    /// independent of `Process.heap_pages` so create/destroy stay
    /// O(1) (no linear scan of heap_pages on each destroy) and so
    /// `dealloc_process` walks one structure for surface cleanup.
    pub backing: PhysBacking,
}

/// Lightweight snapshot of [`SurfaceEntry`] that a [`Cmd::PresentSurface`]
/// carries to k_gpu without keeping the per-process Arc alive on the
/// fast path. Copied out under the inner `Mutex` once at syscall time.
///
/// [`Cmd::PresentSurface`]: crate::drivers::k_gpu::CmdKind::PresentSurface
#[derive(Debug, Clone, Copy)]
pub struct SurfaceSnapshot {
    pub kdmap_kva: u64,
    pub width: u32,
    pub height: u32,
    pub format: FbFormat,
}

pub struct ProcessSurfaces {
    /// Handle id → entry. Mutated only on the manager hart under
    /// `MANAGER_LOCK` (create/destroy + teardown); read on the
    /// `fb_present` sync path which takes the inner Mutex briefly to
    /// snapshot. Inner Mutex avoids requiring MANAGER_LOCK on
    /// every present.
    surfaces: Mutex<BTreeMap<u32, SurfaceEntry>>,
    /// Monotonically-incrementing id allocator. Process-local; rolling
    /// over a `u32` would require `2^32` create/destroy pairs in one
    /// process's lifetime — accept the wrap.
    next_id: AtomicU32,
    /// Bump-allocator cursor for shared-range VA assignment. Starts at
    /// [`UPROC_SHARED_BASE`] and only ever increases — destroy never
    /// reuses VAs. The shared range is 62 TiB so a process would need
    /// to create many millions of surfaces before exhausting it.
    /// Surface VAs land below NetChannel-mapped regions if they share
    /// this cursor with `mmap`; for v1 surfaces have their own cursor
    /// to avoid coupling with the existing `Process.mmap_cursor` (which
    /// is reserved for user-facing `mmap`).
    next_va_cursor: AtomicU64,
}

impl ProcessSurfaces {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            surfaces: Mutex::new(BTreeMap::new()),
            // Skip 0 so an uninitialised user-side `FbHandle::NONE`
            // never collides with a real handle.
            next_id: AtomicU32::new(1),
            next_va_cursor: AtomicU64::new(UPROC_SHARED_BASE),
        })
    }

    /// Reserve a fresh handle id without inserting an entry. Caller
    /// pairs with [`insert`](Self::insert) once the entry is fully
    /// constructed.
    pub fn alloc_id(&self) -> u32 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Bump-reserve a `size`-byte VA range in the shared user range.
    /// Caller must round `size` up to its mapping alignment (PAGE_SIZE
    /// or MEGAPAGE_SIZE). Returns the base VA; cursor is bumped by
    /// `size` regardless of whether the caller's mapping succeeds —
    /// failure leaks the VA range, which is fine for v1.
    pub fn alloc_va(&self, size: u64) -> u64 {
        self.next_va_cursor.fetch_add(size, Ordering::Relaxed)
    }

    pub fn insert(&self, id: u32, entry: SurfaceEntry) {
        self.surfaces.lock().insert(id, entry);
    }

    /// Take the entry out of the map. Caller routes `backing` back to
    /// `kernel_pages` and unmaps `user_va..user_va+size_bytes` from the
    /// process PT.
    pub fn remove(&self, id: u32) -> Option<SurfaceEntry> {
        self.surfaces.lock().remove(&id)
    }

    /// Cheap snapshot for the `fb_present` hot path. Returns `None` if
    /// the handle isn't registered.
    pub fn snapshot(&self, id: u32) -> Option<SurfaceSnapshot> {
        let g = self.surfaces.lock();
        g.get(&id).map(|e| SurfaceSnapshot {
            kdmap_kva: e.kdmap_kva,
            width: e.width,
            height: e.height,
            format: e.format,
        })
    }

    /// Drain every remaining surface. Used by `dealloc_process` to
    /// recover backings + unmap VAs after the process's teardown.
    pub fn drain_all(&self) -> alloc::vec::Vec<(u32, SurfaceEntry)> {
        let mut g = self.surfaces.lock();
        let drained: alloc::vec::Vec<(u32, SurfaceEntry)> = core::mem::take(&mut *g).into_iter().collect();
        drained
    }
}

/// Global pid → surface table. Insert on `register`, remove on
/// `unregister`. Lookup hands back an Arc clone so callers can drop the
/// outer lock before working with the per-process state.
pub static SURFACE_TABLE: Mutex<BTreeMap<u16, Arc<ProcessSurfaces>>> = Mutex::new(BTreeMap::new());

/// Register an empty surface slot for `pid`. Idempotent: a second
/// register on the same pid leaves the existing slot intact.
pub fn register(pid: u16) {
    let mut t = SURFACE_TABLE.lock();
    t.entry(pid).or_insert_with(ProcessSurfaces::new);
}

/// Remove the surface slot for `pid` and return the Arc so the caller
/// can iterate remaining entries to free their backings. Returns `None`
/// if the pid was never registered (covers re-entrant teardown paths).
pub fn unregister(pid: u16) -> Option<Arc<ProcessSurfaces>> {
    SURFACE_TABLE.lock().remove(&pid)
}

/// Look up `pid`'s surface table. Returns a clone of the Arc so the
/// caller can drop the outer lock before working with the per-process
/// state.
pub fn get(pid: u16) -> Option<Arc<ProcessSurfaces>> {
    SURFACE_TABLE.lock().get(&pid).cloned()
}
