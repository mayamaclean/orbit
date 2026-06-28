//! Userspace EventFd wrapper.
//!
//! Thin glue over the kernel's `eventfd(2)` syscall + the
//! [`EventFd`](orbit_abi::event_fd::EventFd) shared-memory layout.
//! Userspace operates on the count via atomic loads/stores against
//! the mapped region — only cross-thread wakes (signaling a reader
//! that's parked elsewhere) involve the kernel via `wake_tid`.
//!
//! ## Selector pattern
//!
//! The eventfd is the orbit-side analog of Linux's eventfd-as-mio-Waker:
//! - **Reactor** parks via `ch_yield(timeout)` after stashing its tid in
//!   selector-owned state. Reads `count > 0` from shared memory on wake.
//! - **Wakers** in other threads bump `count` via `signal(n)`, then read
//!   the parked-tid hint and issue `wake_tid` if non-zero.
//!
//! On the orbit selector path the eventfd's *own* `parked_tid` field
//! stays zero — the reactor isn't parking on a specific eventfd, it's
//! parking on `ch_yield` with the selector's tracked tid. The
//! eventfd's `parked_tid` is reserved for a direct
//! blocking-`read(eventfd)` path that the kernel does not implement
//! yet, so it stays zero in practice.

#![cfg(feature = "mem-alloc")]

use core::ptr::NonNull;
use core::sync::atomic::Ordering;

use orbit_abi::Errno;
use orbit_abi::errno::EINVAL;
use orbit_abi::event_fd::{EFD_ALL_FLAGS, EVENTFD_REGION_SIZE, EventFd as EventFdRegion};
use orbit_abi::user;

use crate::{PAGE_SIZE_USIZE, SharedRegion};

/// Userspace handle to a kernel-allocated EventFd region.
///
/// Construction reserves a page-sized VA slot in the process's shared
/// range, calls the kernel's `eventfd` syscall to allocate the backing
/// frame + map it shared, and stashes the resulting fd plus the user
/// pointer to the [`EventFdRegion`] shared header.
///
/// `Drop` issues `close_handle(fd)` — the kernel revokes the user
/// mapping and frees the backing frame.
#[derive(Debug)]
pub struct EventFd {
    /// Kernel-assigned slot in the calling process's handle table.
    /// `RawFd`-shaped (`i32`) so it round-trips through `AsRawFd`.
    fd: i32,
    /// Pointer to the mapped shared header. The kernel-installed
    /// mapping outlives any reference held to this struct because
    /// drop runs `close_handle` after the pointer goes out of scope
    /// at the language level.
    region: NonNull<EventFdRegion>,
    /// VA reservation in the shared range. `SharedRegion::drop` returns
    /// the bytes to the buddy allocator so future eventfds can reuse
    /// the slot. Held by value (not `Option`): our `Drop` runs
    /// `close_handle` first, then the compiler-generated field drop
    /// releases this reservation — after the PTE is already revoked, so
    /// the VA can't be reused while still mapped.
    _va: SharedRegion,
}

unsafe impl Send for EventFd {}
unsafe impl Sync for EventFd {}

impl EventFd {
    /// Allocate a new EventFd seeded at `initval` with `flags`
    /// (see [`orbit_abi::event_fd`] for the bit set).
    pub fn create(initval: u64, flags: u32) -> Result<Self, Errno> {
        if flags & !EFD_ALL_FLAGS != 0 {
            return Err(Errno::new(EINVAL));
        }

        // Reserve a page-aligned slot in the shared VA range. The
        // kernel will fail with EINVAL if the hint doesn't sit in
        // [UPROC_SHARED_BASE, UPROC_SHARED_END), so we lean on the
        // buddy allocator that already enforces that.
        let layout = core::alloc::Layout::from_size_align(EVENTFD_REGION_SIZE, PAGE_SIZE_USIZE)
            .map_err(|_| Errno::new(EINVAL))?;

        let va_region = SharedRegion::reserve(layout)?;
        let va = va_region.va();

        match user::eventfd(va, initval, flags) {
            Ok((mapped_va, fd)) => {
                debug_assert_eq!(mapped_va, va);
                let region =
                    NonNull::new(va as *mut EventFdRegion).expect("eventfd returned non-null VA");
                Ok(Self {
                    fd: fd as i32,
                    region,
                    _va: va_region,
                })
            }
            Err(e) => {
                // SharedRegion::drop returns the VA reservation on
                // bail-out, no manual cleanup needed.
                Err(e)
            }
        }
    }

    /// Bump the count by `n` and wake any parked reader (when the
    /// shared `parked_tid` hint is non-zero). `n == 0` is a no-op.
    ///
    /// On the orbit selector path the reactor doesn't park on this
    /// eventfd directly — the shared `parked_tid` stays at zero and
    /// the wake is delivered via a separate selector-owned tid that
    /// [`Self::signal_waker`] knows about.
    pub fn signal(&self, n: u64) {
        if n == 0 {
            return;
        }
        unsafe {
            self.region.as_ref().count.fetch_add(n, Ordering::Release);
            let tid = self.region.as_ref().parked_tid.load(Ordering::Acquire);
            if tid != 0 {
                let _ = user::wake_tid(tid);
            }
        }
    }

    /// Bump the count by `1` and additionally wake the explicit
    /// `waker_tid` if it's non-zero — used by mio-style selectors that
    /// own their reactor tid out-of-band rather than via the eventfd's
    /// shared `parked_tid` field.
    ///
    /// `waker_tid == 0` collapses to the same behavior as
    /// [`Self::signal`].
    pub fn signal_waker(&self, waker_tid: u32) {
        unsafe {
            self.region.as_ref().count.fetch_add(1, Ordering::Release);
        }
        if waker_tid != 0 {
            let _ = user::wake_tid(waker_tid);
        }
    }

    /// Atomically swap the count to `0` and return its prior value.
    /// `0` means "nothing pending"; non-zero is the total accumulated
    /// signal volume since the last consume.
    ///
    /// Pure-memory operation — no syscall.
    pub fn try_consume(&self) -> u64 {
        unsafe { self.region.as_ref().count.swap(0, Ordering::AcqRel) }
    }

    /// Snapshot the current counter without consuming. Used by
    /// selector-style readiness scans to decide whether the eventfd is
    /// "ready" without claiming the value.
    pub fn peek(&self) -> u64 {
        unsafe { self.region.as_ref().count.load(Ordering::Acquire) }
    }

    /// `RawFd`-shaped handle to the kernel slot. Same value `AsRawFd`
    /// would return on a std-wrapped fd.
    pub fn as_raw_fd(&self) -> i32 {
        self.fd
    }

    /// Pointer to the shared region. Useful for selectors that want to
    /// cache the readiness location at registration time and skip the
    /// `Arc` indirection on the hot scan.
    pub fn region_ptr(&self) -> NonNull<EventFdRegion> {
        self.region
    }
}

impl Drop for EventFd {
    fn drop(&mut self) {
        // Close the kernel slot first; the kernel revokes the user PTE
        // so future reads against the VA fault. `_va`'s Drop then
        // releases the VA reservation back to the buddy allocator.
        let _ = user::close_handle(self.fd as u32);
    }
}
