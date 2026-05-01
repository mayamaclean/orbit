//! Orbit user runtime.
//!
//! The "stdlib for orbit" — the alloc-using user-side runtime that
//! sits on top of [`orbit-abi`](../orbit_abi/) and is consumed by both
//! no_std user binaries (umode, console, hello, orbit-loader, benches)
//! and std-on-orbit's PAL via the `rustc-dep-of-std` feature.
//!
//! Owns two pieces of memory machinery, matched to the kernel's priv/
//! shared VA split:
//!
//! - **`#[global_allocator]` (private heap).** Backed by
//!   [`dlmalloc::Dlmalloc`] with [`PrivMmapAllocator`] as the page
//!   source. `Box`/`Vec`/`String` claims land inside
//!   [`UPROC_PRIV_BASE`]..[`UPROC_PRIV_END`] via
//!   `mmap(share_with_kernel=false)`. Same shape std-on-orbit's PAL
//!   allocator uses ([`rust/library/std/src/sys/alloc/orbit.rs`]) —
//!   keeps the runtime story consistent across no_std and std builds.
//!
//! - **[`SharedVa`] — a buddy-style VA reservation in the shared range.**
//!   Hands out page-aligned VA chunks inside
//!   [`UPROC_SHARED_BASE`]..[`UPROC_SHARED_END`] without doing any
//!   mapping. Consumers map the returned range themselves: NetChannels
//!   via the `create_netch` syscall (kernel allocates + maps), direct
//!   shared mmaps via [`shared_mmap`] (user picks size, kernel maps).
//!   Frees come back via the buddy merger so the range can be reused as
//!   NetChannels open and close.
//!
//! Why a separate VA allocator instead of a second dlmalloc? Shared
//! allocations are heterogeneous: `create_netch` wants the kernel to
//! install the mapping at a hint VA, while `mmap(share=true)` does the
//! mapping itself. A heap-style allocator that *also* mmaps would
//! double-map the netchannel hint and silently leak the original
//! frames. Decoupling VA reservation from mapping lets each consumer
//! drive its own syscall.
//!
//! # Allocating shared memory directly
//!
//! ```ignore
//! use core::alloc::Layout;
//! use orbit_rt::shared_mmap;
//!
//! let region = shared_mmap(Layout::from_size_align(8192, 4096).unwrap())?;
//! let buf = unsafe { core::slice::from_raw_parts_mut(region.as_mut_ptr(), region.len()) };
//! // ... use buf ...
//! drop(region); // returns the VA range to SharedVa; munmap arrives in a future milestone.
//! ```
//!
//! # Reserving a VA hint for `create_netch`
//!
//! See [`crate::netch::NetCh::open`].
//!
//! # Locking
//!
//! Both the dlmalloc heap and [`SharedVa`] are guarded by an atomic
//! spinlock. Now that `create_thread` (syscall 5000) lets a single
//! process span multiple harts, a Vec push on one thread can race a
//! Box drop on another — concurrent allocator entry is no longer
//! hypothetical.
//!
//! Two specific safety requirements the lock alone doesn't cover:
//!
//! - **Don't allocate in trap context.** umode doesn't take async
//!   signals today, but the moment it does, a signal handler that
//!   allocates while the interrupted thread holds the heap lock
//!   deadlocks. The kernel's "no allocation in trap context" rule
//!   applies here too.
//! - **`PRIV_NEXT_VA` is bumped before the dlmalloc claim**, so two
//!   threads racing through `PrivMmapAllocator::alloc` get distinct
//!   VA ranges before either calls into dlmalloc. dlmalloc itself
//!   serializes its bookkeeping under the heap lock, so the two
//!   ranges land safely in distinct arenas.

#![no_std]
#![allow(static_mut_refs)]

// `alloc` is only needed by the full-runtime modules (netch + shared_va
// touch `alloc::vec::Vec`). The core orbit-rt surface (allocator,
// argv, atomic spinlock) stays alloc-free so std-on-orbit's build —
// which pulls orbit-rt with `default-features = false` and provides
// its own `alloc` workspace crate — doesn't end up with two `alloc`
// names colliding.
#[cfg(feature = "full-runtime")]
extern crate alloc;

pub mod argv;

// Modules that rely on the `mem` crate (FrameAllocator), the `alloc`
// crate (BTreeMap / Vec / String), or a `_start` symbol that would
// clash with std-on-orbit's PAL `_start`. Gated behind `full-runtime`
// so the std build (which sets `default-features = false`) sees a
// smaller orbit-rt surface and avoids the alloc / mem dep chain that
// isn't `rustc-dep-of-std`-friendly.
#[cfg(feature = "full-runtime")]
pub mod env;
#[cfg(feature = "full-runtime")]
pub mod netch;
#[cfg(feature = "full-runtime")]
pub mod start;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use orbit_abi::{
    layout::{PAGE_SIZE, UPROC_PRIV_BASE},
    user,
};

// Lockstep with `build.rs`'s hardcoded copies. build.rs can't depend
// on orbit-abi (resolver v1 unifies build-dep + normal-dep features,
// which would force in-tree-`core` recompilation under
// `rustc-dep-of-std` builds — see Cargo.toml). This compile-time
// assertion fires if the layout drifts.
const _: () = {
    const BUILD_RS_USER_TEXT_BASE: u64 = 0x2_2000_0000;
    const BUILD_RS_USER_ENVP_BASE: u64 = 0x2_FFFF_E000;
    assert!(
        BUILD_RS_USER_TEXT_BASE == orbit_abi::layout::USER_TEXT_BASE,
        "build.rs USER_TEXT_BASE diverged from orbit_abi::layout::USER_TEXT_BASE",
    );
    assert!(
        BUILD_RS_USER_ENVP_BASE == orbit_abi::layout::USER_ENVP_BASE,
        "build.rs USER_ENVP_BASE diverged from orbit_abi::layout::USER_ENVP_BASE",
    );
};

// R | W | U bits from mmu::PagePermissions. Kernel expects the raw 5-bit
// encoding today; the orbit-abi::mmap prot/flags redesign isn't wired up.
pub const PERMS_RW_U: usize = 0x2 | 0x4 | 0x10;

// Growth chunk for the private dlmalloc heap. mmap is a blocking
// manager round-trip today; fewer, larger claims keep that traffic
// down. Revisit when munmap lands.
const PRIV_GROWTH_CHUNK: usize = 2 * 1024 * 1024;

const PAGE_SIZE_USIZE: usize = PAGE_SIZE as usize;

// =====================================================================
// Atomic spinlock — replaces the lock_api / spinning_top dependency
// chain. Same shape as std-on-orbit's allocator lock so the two stay
// architecturally aligned.
// =====================================================================

struct SpinFlag(AtomicI32);

impl SpinFlag {
    const fn new() -> Self {
        Self(AtomicI32::new(0))
    }

    /// Spin until the flag is acquired. Returns a guard whose drop
    /// releases the lock. Caller must not park / yield while holding
    /// the lock — orbit doesn't yield from a locked section anyway,
    /// but a future `kthread_park`-shaped primitive would deadlock.
    fn lock(&self) -> SpinGuard<'_> {
        while self.0.swap(1, Ordering::Acquire) != 0 {
            core::hint::spin_loop();
        }
        SpinGuard(self)
    }
}

struct SpinGuard<'a>(&'a SpinFlag);

impl Drop for SpinGuard<'_> {
    fn drop(&mut self) {
        self.0.0.store(0, Ordering::Release);
    }
}

// =====================================================================
// Private heap — dlmalloc backed by the kernel's mmap syscall.
// =====================================================================

// Heap cursor starts at the bottom of the kernel-enforced private
// range. The kernel rejects any private mmap below this VA, so the
// allocator can't accidentally collide with the kernel-mapped stack
// or ELF regions even if a downstream caller mishandles the cursor.
static PRIV_NEXT_VA: AtomicUsize = AtomicUsize::new(UPROC_PRIV_BASE as usize);

/// dlmalloc page source. Each `alloc` request bumps `PRIV_NEXT_VA`
/// monotonically and asks the kernel to back the resulting range with
/// pages via `mmap(share_with_kernel=false)`.
pub struct PrivMmapAllocator;

unsafe impl Send for PrivMmapAllocator {}

unsafe impl dlmalloc::Allocator for PrivMmapAllocator {
    fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
        let need = size
            .max(PRIV_GROWTH_CHUNK)
            .next_multiple_of(PAGE_SIZE_USIZE);
        let va = PRIV_NEXT_VA.fetch_add(need, Ordering::Relaxed);

        // SAFETY: monotonic cursor + kernel-side bound check guarantees
        // a fresh range; `mmap` returns the kernel's own validation
        // result. Bump-then-check pattern avoids racing two growers
        // onto the same VA.
        if unsafe { user::mmap(va, need, PERMS_RW_U, false) }.is_err() {
            return (ptr::null_mut(), 0, 0);
        }
        // Strict-provenance-clean integer→pointer: kernel just mapped
        // this VA, so synthesizing a fresh provenance here is fine.
        (ptr::with_exposed_provenance_mut::<u8>(va), need, 0)
    }

    fn remap(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
        // No `mremap`-style syscall today; dlmalloc falls back to
        // alloc + copy + free.
        ptr::null_mut()
    }

    fn free_part(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize) -> bool {
        // No partial `munmap` today.
        false
    }

    fn free(&self, _ptr: *mut u8, _size: usize) -> bool {
        // No `munmap` syscall today; the kernel reclaims on process
        // exit. dlmalloc keeps the chunk and reuses it for future
        // allocations within this process.
        false
    }

    fn can_release_part(&self, _flags: u32) -> bool {
        false
    }

    fn allocates_zeros(&self) -> bool {
        // Kernel-mapped pages are zeroed.
        true
    }

    fn page_size(&self) -> usize {
        PAGE_SIZE_USIZE
    }
}

// dlmalloc isn't `Sync` on its own; we serialize all entry through
// `HEAP_LOCK` and access the cell via raw pointer. Wrapper struct
// carries the Sync attestation so the static is well-formed.
struct SyncDlmalloc(UnsafeCell<dlmalloc::Dlmalloc<PrivMmapAllocator>>);
unsafe impl Sync for SyncDlmalloc {}

static DLMALLOC: SyncDlmalloc = SyncDlmalloc(UnsafeCell::new(
    dlmalloc::Dlmalloc::new_with_allocator(PrivMmapAllocator),
));

static HEAP_LOCK: SpinFlag = SpinFlag::new();

/// Marker type the `#[global_allocator]` registration wires through.
/// Public so consumers can name the type if they ever need to (not
/// required for normal allocation — `alloc::alloc::alloc` / `Box::new`
/// route through whichever crate registered `#[global_allocator]`).
pub struct OrbitHeap;

unsafe impl GlobalAlloc for OrbitHeap {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _g = HEAP_LOCK.lock();
        unsafe { (*DLMALLOC.0.get()).malloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _g = HEAP_LOCK.lock();
        unsafe { (*DLMALLOC.0.get()).calloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _g = HEAP_LOCK.lock();
        unsafe { (*DLMALLOC.0.get()).free(ptr, layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _g = HEAP_LOCK.lock();
        unsafe { (*DLMALLOC.0.get()).realloc(ptr, layout.size(), layout.align(), new_size) }
    }
}

// `#[global_allocator]` only registered when orbit-rt is the runtime
// for a no_std user binary. When orbit-rt is pulled in as a
// `rustc-dep-of-std` dep, std provides its own #[global_allocator]
// (see `rust/library/std/src/sys/alloc/orbit.rs`); registering ours
// would be a duplicate and fail to link.
#[cfg(not(feature = "rustc-dep-of-std"))]
#[global_allocator]
static ORBIT_HEAP: OrbitHeap = OrbitHeap;

// =====================================================================
// Shared VA allocator (unchanged in shape; just swaps lock_api Mutex
// for the same SpinFlag pattern).
//
// Wrapped in `mod shared_va` and gated behind `full-runtime` because
// the buddy `FrameAllocator` lives in the `mem` crate, whose
// transitive deps (heapless, byteorder, ...) don't ship
// `rustc-dep-of-std` features and therefore can't be pulled into the
// std build. std-on-orbit doesn't use SharedVa today — its NetCh impl
// in [`rust/library/std/src/sys/net/connection/orbit.rs`] reserves
// shared VAs through its own path.
// =====================================================================
#[cfg(feature = "full-runtime")]
mod shared_va {
    use super::{PAGE_SIZE_USIZE, PERMS_RW_U, SpinFlag};
    use core::alloc::Layout;
    use core::cell::UnsafeCell;
    use mem::frame::FrameAllocator;
    use orbit_abi::{
        errno::{EINVAL, ENOMEM, Errno},
        layout::{UPROC_SHARED_BASE, UPROC_SHARED_END},
        user,
    };

    /// Buddy allocator covering the shared user range, addressed in
    /// **bytes** to match the rest of the codebase
    /// ([`kmain::kernel::memmap::TablePages`](../../kmain/src/kernel/memmap.rs)
    /// and friends feed `FrameAllocator` raw byte addresses, so
    /// `alloc_aligned` returns the address directly without any
    /// `* PAGE_SIZE` conversion). `Layout::from_size_align` callers pass
    /// byte sizes and byte alignments — same shape as the kmain pools.
    ///
    /// ORDER must satisfy `2^ORDER - 1 >= UPROC_SHARED_END -
    /// UPROC_SHARED_BASE`. Derived from the constants so a future shared
    /// range expansion fails the const assert below instead of silently
    /// running off the end. For the current 62 TiB shared range, ORDER
    /// resolves to 46.
    const SHARED_RANGE_BYTES: usize = (UPROC_SHARED_END - UPROC_SHARED_BASE) as usize;
    const SHARED_ORDER: usize = (usize::BITS - SHARED_RANGE_BYTES.leading_zeros()) as usize;
    const _: () = assert!(
        SHARED_RANGE_BYTES <= (1usize << SHARED_ORDER) - 1,
        "SHARED_ORDER too small for the shared VA range",
    );

    struct SharedVaInner {
        initialized: bool,
        fa: FrameAllocator<SHARED_ORDER>,
    }

    /// VA reservation pool for the kernel-shared user range. Hands out
    /// page-aligned VA chunks; does *not* install any mapping — the caller
    /// is responsible for that (typically via `user::mmap(share=true)` or
    /// `create_netch`). Free path returns frames through the buddy merger
    /// so the range survives long-running processes that open/close
    /// NetChannels.
    pub struct SharedVa {
        lock: SpinFlag,
        inner: UnsafeCell<SharedVaInner>,
    }

    // SharedVaInner holds a FrameAllocator that's `!Sync`, but every
    // access goes through `with` (which takes `&self.lock`). The
    // `unsafe impl Sync` carries that invariant.
    unsafe impl Sync for SharedVa {}

    impl SharedVa {
        const fn new() -> Self {
            Self {
                lock: SpinFlag::new(),
                inner: UnsafeCell::new(SharedVaInner {
                    initialized: false,
                    fa: FrameAllocator::new(),
                }),
            }
        }

        /// Run `f` against the inner allocator, lazy-initializing the free
        /// list on first call. Lazy because `FrameAllocator::insert` isn't
        /// `const`, but `static SHARED_VA` has to be const-constructible.
        /// The closure runs under the spinlock — keep it short and don't
        /// call back into `SharedVa` (single-threaded reentrancy would
        /// deadlock).
        fn with<R>(&self, f: impl FnOnce(&mut FrameAllocator<SHARED_ORDER>) -> R) -> R {
            let _g = self.lock.lock();
            // SAFETY: lock is held; we have exclusive access to the cell
            // for the duration of `_g`'s lifetime.
            let inner = unsafe { &mut *self.inner.get() };
            if !inner.initialized {
                // Byte-addressed: the buddy tracks the raw VA range
                // [UPROC_SHARED_BASE, UPROC_SHARED_END), same convention as
                // kmain's pool wrappers.
                inner
                    .fa
                    .insert(UPROC_SHARED_BASE as usize..UPROC_SHARED_END as usize);
                inner.initialized = true;
            }
            f(&mut inner.fa)
        }
    }

    /// Process-wide VA reservation pool for the kernel-shared range. See
    /// [`SharedVa`].
    pub static SHARED_VA: SharedVa = SharedVa::new();

    /// A reservation of contiguous VAs in the kernel-shared range. Owns
    /// the byte range; `Drop` returns it to [`SHARED_VA`] so the VAs can
    /// be reused.
    ///
    /// A `SharedRegion` is just a VA reservation — it does *not* own any
    /// kernel mapping. The caller is responsible for any `mmap` /
    /// `create_netch` / corresponding teardown (`munmap` /
    /// `close_handle`). Drop releases only the VA, not whatever mapping
    /// was installed at it; if the caller forgets to tear down its
    /// mapping, the orphaned PTEs persist until process exit.
    pub struct SharedRegion {
        va: usize,
        /// Layout passed to `alloc_aligned`. Stored verbatim because
        /// `dealloc_aligned` requires the exact same layout — the buddy's
        /// merge logic uses `max(size.next_power_of_two(), align)` to find
        /// the size class, so a mismatched layout corrupts the free list.
        layout: Layout,
    }

    impl SharedRegion {
        /// Reserve a VA range covering `layout`. `layout.size()` is rounded
        /// up to a whole page (since smaller granularity is unmappable);
        /// `layout.align()` is rounded up to `PAGE_SIZE` for the same
        /// reason.
        ///
        /// Returns `Err(EINVAL)` for a zero-sized layout, `Err(ENOMEM)` if
        /// the shared range is exhausted (or the buddy can't satisfy the
        /// alignment).
        pub fn reserve(layout: Layout) -> Result<Self, Errno> {
            if layout.size() == 0 {
                return Err(Errno::new(EINVAL));
            }
            // Pages are the unit the kernel can map, so round both fields
            // up to PAGE_SIZE before handing them to the buddy.
            let size = layout.size().next_multiple_of(PAGE_SIZE_USIZE);
            let align = layout.align().max(PAGE_SIZE_USIZE);
            let aligned = Layout::from_size_align(size, align).map_err(|_| Errno::new(EINVAL))?;

            // The buddy returns a byte-address from the
            // [UPROC_SHARED_BASE, UPROC_SHARED_END) range we seeded; cast
            // straight to a VA. `with` takes the spinlock internally.
            let va = SHARED_VA
                .with(|fa| fa.alloc_aligned(aligned))
                .ok_or(Errno::new(ENOMEM))?;

            Ok(Self {
                va,
                layout: aligned,
            })
        }

        /// Starting VA of the reservation. Page-aligned.
        pub fn va(&self) -> usize {
            self.va
        }

        /// Length of the reservation in bytes (page-rounded from the
        /// requesting layout).
        pub fn len(&self) -> usize {
            self.layout.size()
        }

        pub fn as_ptr(&self) -> *const u8 {
            self.va as *const u8
        }

        pub fn as_mut_ptr(&self) -> *mut u8 {
            self.va as *mut u8
        }

        /// Forget the reservation without returning its bytes to
        /// [`SHARED_VA`]. Used when ownership of the underlying VA range
        /// transfers to something else (the kernel takes over teardown via
        /// `close_handle`, etc.) so a later `Drop` would double-free.
        pub fn leak(self) -> usize {
            let va = self.va;
            core::mem::forget(self);
            va
        }
    }

    impl Drop for SharedRegion {
        fn drop(&mut self) {
            // `layout` is the exact value passed to `alloc_aligned` (we
            // stored it above), which is what `dealloc_aligned` requires
            // for buddy correctness. `with` takes the spinlock internally.
            SHARED_VA.with(|fa| fa.dealloc_aligned(self.va, self.layout));
        }
    }

    /// Reserve a VA range and ask the kernel to install a shared mapping at
    /// it. Convenience wrapper around `SharedRegion::reserve` +
    /// `user::mmap(share_with_kernel=true)` — keeps the pair atomic from
    /// the caller's perspective and unwinds the VA reservation on mmap
    /// failure.
    ///
    /// The returned [`SharedRegion`] frees its VA on drop, but doesn't
    /// (today) call `munmap` because the syscall doesn't exist yet — the
    /// kernel-installed PTEs persist until process exit. When `munmap`
    /// lands this should grow a `Drop` that issues it.
    pub fn shared_mmap(layout: Layout) -> Result<SharedRegion, Errno> {
        let region = SharedRegion::reserve(layout)?;
        if let Err(e) = unsafe { user::mmap(region.va(), region.len(), PERMS_RW_U, true) } {
            // mmap failed — drop returns the VA reservation. Propagate the
            // kernel's errno verbatim.
            drop(region);
            return Err(e);
        }
        Ok(region)
    }
}

#[cfg(feature = "full-runtime")]
pub use shared_va::{SHARED_VA, SharedRegion, SharedVa, shared_mmap};
