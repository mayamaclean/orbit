//! Orbit user runtime.
//!
//! Foundation crate for `riscv64gc-unknown-orbit` std (roadmap §9). Owns
//! two pieces of memory machinery, matched to the kernel's priv/shared
//! VA split:
//!
//! - **`#[global_allocator]` (private heap).** Backed by talc with
//!   [`PrivMmapSource`] as the growth hook. `Box`/`Vec`/`String` claims
//!   land inside [`UPROC_PRIV_BASE`]..[`UPROC_PRIV_END`] via
//!   `mmap(share_with_kernel=false)`. Byte-granular, fragmentation-
//!   handling — the standard heap-allocator shape.
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
//! Why a separate VA allocator instead of a second talc? Shared
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
//! Both the talc heap and [`SharedVa`] are guarded by
//! `spinning_top::RawSpinlock`. Now that `create_thread` (syscall 5000)
//! lets a single process span multiple harts, a Vec push on one thread
//! can race a Box drop on another — concurrent allocator entry is no
//! longer hypothetical.
//!
//! Two specific safety requirements the lock alone doesn't cover:
//!
//! - **Don't allocate in trap context.** umode doesn't take async
//!   signals today, but the moment it does, a signal handler that
//!   allocates while the interrupted thread holds the heap lock
//!   deadlocks. The kernel's "no allocation in trap context" rule
//!   applies here too.
//! - **`PRIV_NEXT_VA` is bumped before the talc claim**, so two
//!   threads racing through `PrivMmapSource::acquire` get distinct VA
//!   ranges before either calls `t.claim`. Talc itself serializes the
//!   claim under the heap lock, so the two ranges land safely in
//!   distinct arenas.

#![no_std]

extern crate alloc;

pub mod argv;
pub mod netch;
pub mod start;

use core::alloc::Layout;
use core::sync::atomic::{AtomicUsize, Ordering};

use lock_api::Mutex;
use mem::frame::FrameAllocator;
use orbit_abi::{
    errno::{Errno, EINVAL, ENOMEM},
    layout::{PAGE_SIZE, UPROC_PRIV_BASE, UPROC_SHARED_BASE, UPROC_SHARED_END},
    user,
};
use spinning_top::RawSpinlock;
use talc::base::binning::{Binning, DefaultBinning};
use talc::base::Talc;
use talc::source::Source;
use talc::sync::TalcLock;

// R | W | U bits from mmu::PagePermissions. Kernel expects the raw 5-bit
// encoding today; the orbit-abi::mmap prot/flags redesign isn't wired up.
pub const PERMS_RW_U: usize = 0x2 | 0x4 | 0x10;

// Growth chunk for the private talc heap. mmap is a blocking manager
// round-trip today; fewer, larger claims keep that traffic down. Revisit
// when munmap lands.
const PRIV_GROWTH_CHUNK: usize = 2 * 1024 * 1024;

const PAGE_SIZE_USIZE: usize = PAGE_SIZE as usize;

// --- private heap --------------------------------------------------------

// Heap cursor starts at the bottom of the kernel-enforced private
// range. The kernel rejects any private mmap below this VA, so the
// allocator can't accidentally collide with the kernel-mapped stack
// or ELF regions even if a downstream caller mishandles the cursor.
static PRIV_NEXT_VA: AtomicUsize = AtomicUsize::new(UPROC_PRIV_BASE as usize);

#[derive(Debug)]
pub struct PrivMmapSource;

unsafe impl Source for PrivMmapSource {
    fn acquire<B: Binning>(t: &mut Talc<Self, B>, layout: Layout) -> Result<(), ()> {
        let need = layout.size().max(PRIV_GROWTH_CHUNK).next_multiple_of(PAGE_SIZE_USIZE);
        let va = PRIV_NEXT_VA.fetch_add(need, Ordering::Relaxed);

        if unsafe { user::mmap(va, need, PERMS_RW_U, false) }.is_err() {
            return Err(());
        }

        let base = va as *mut u8;
        // SAFETY: kernel just mapped [base, base+need) user-RW for this
        // process. PRIV_NEXT_VA increases monotonically so the range is
        // disjoint from every prior claim.
        match unsafe { t.claim(base, need) } {
            Some(_) => Ok(()),
            None => Err(()),
        }
    }
}

// Spinlock-guarded talc heap. `create_thread` (syscall 5000) makes
// concurrent allocator entry from sibling threads a real possibility;
// `TalcLock` serializes on every `alloc`/`dealloc` so a Vec push on
// hart A and a Box drop on hart B don't corrupt the same arena. The
// uncontended path is a single atomic CAS — cheap relative to the talc
// bookkeeping it guards.
#[global_allocator]
static PRIV_HEAP: TalcLock<RawSpinlock, PrivMmapSource, DefaultBinning> =
    TalcLock::new(PrivMmapSource);

// --- shared VA allocator -------------------------------------------------

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
    inner: Mutex<RawSpinlock, SharedVaInner>,
}

impl SharedVa {
    const fn new() -> Self {
        Self {
            inner: Mutex::new(SharedVaInner {
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
        let mut g = self.inner.lock();
        if !g.initialized {
            // Byte-addressed: the buddy tracks the raw VA range
            // [UPROC_SHARED_BASE, UPROC_SHARED_END), same convention as
            // kmain's pool wrappers.
            g.fa.insert(UPROC_SHARED_BASE as usize..UPROC_SHARED_END as usize);
            g.initialized = true;
        }
        f(&mut g.fa)
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
        let va = SHARED_VA.with(|fa| fa.alloc_aligned(aligned))
            .ok_or(Errno::new(ENOMEM))?;

        Ok(Self { va, layout: aligned })
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
