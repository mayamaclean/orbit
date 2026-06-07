//! EventFd — shared-memory counter, Linux-shaped surface.
//!
//! Linux-style `eventfd(2)` primitive, but the counter is exposed
//! directly in a kernel-allocated shared page so userspace bumps it
//! with an atomic add (no trap) and selectors poll it with a pure
//! memory load (no trap).
//!
//! # What's wired today
//!
//! - **Create / close.** `eventfd(2)` syscall allocates the region,
//!   maps it user-RW, installs a `Handle::EventFd` slot.
//! - **Signal (writer side).** Userspace `count.fetch_add(n)` — no
//!   syscall. Companion doorbell is a `wake_tid` call against a
//!   consumer-published parked tid (e.g. an mio `Selector`); see the
//!   `selector_parked` field on mio's `sys::orbit::Waker`. The doorbell
//!   tid lives in the *consumer*, not in this region.
//! - **Poll (reader side).** `count.load() > 0` from a selector scan.
//! - **`ch_inspect`.** Returns the region's `va`/`size` plus the
//!   create-time `flags` snapshot.
//!
//! # What's not wired
//!
//! - **No `read(fd)` syscall path.** A blocking `read` that drains via
//!   `swap(0)` (default) or `fetch_sub(1)` (`EFD_SEMAPHORE`) and parks
//!   on count==0 is the POSIX shape, but no kernel dispatch exists for
//!   it today. The [`parked_tid`](EventFd::parked_tid) field below is
//!   reserved for that future path. Until then it stays zero. Drain
//!   the counter from userspace with the same `swap(0)` / `fetch_sub`
//!   you'd want from `read(fd)` — `orbit_rt::event_fd::EventFd::try_consume`
//!   wraps this.
//! - **No `EFD_SEMAPHORE` enforcement.** The flag is accepted at
//!   create time and reflected back through `ch_inspect`, but no
//!   kernel path consumes the counter, so the flag's effect on read
//!   semantics is undefined until `read(fd)` lands.
//!
//! # Layout invariants
//!
//! This struct is the user/kernel ABI — do not reorder fields without
//! bumping the ABI surface.
//!
//! - The region is exactly one page (4 KiB).
//! - The header occupies a single 64-byte cache line; the rest of the
//!   page is reserved padding so future extensions don't require a new
//!   region size.
//! - `count` is RW from userspace. Writers `fetch_add` to signal;
//!   readers `swap(0)` or `fetch_sub(1)` to drain.
//! - `parked_tid` is reserved for the future POSIX `read(fd)` parking
//!   path (kernel-written, user-read at that point). Today it stays
//!   `0` after `init`. The mirroring kernel-side
//!   `EventFdSlot.kernel_parked_tid` shadow is also reserved for that
//!   path — see `kmain/src/kernel/handle.rs`.
//! - `flags` is set at create time and never mutated afterward.

use core::sync::atomic::{AtomicU32, AtomicU64};

/// One 4 KiB page; same shape as a single fs scratch page.
pub const EVENTFD_REGION_SIZE: usize = 4096;

/// Default mode: `read(fd)` swaps the counter to `0` and returns the
/// pre-swap value (Linux semantics). Mutually exclusive with
/// [`EFD_SEMAPHORE`].
pub const EFD_DEFAULT: u32 = 0;

/// `read(fd)` returns `EAGAIN` immediately when the counter is zero
/// instead of parking. Sets the slot's `nonblock` bit at create time.
pub const EFD_NONBLOCK: u32 = 1 << 0;

/// Semaphore mode: `read(fd)` decrements the counter by `1` and
/// returns `1` (Linux semantics). Reading is only allowed when the
/// counter is non-zero — zero either parks or returns `EAGAIN` per
/// `EFD_NONBLOCK`.
pub const EFD_SEMAPHORE: u32 = 1 << 1;

/// `FD_CLOEXEC` on the resulting slot — child processes spawned from
/// this fd onward will not inherit it. Honored by routing through the
/// slot's `cloexec: bool` flag.
pub const EFD_CLOEXEC: u32 = 1 << 2;

/// All flag bits the kernel currently recognizes. Bits outside this
/// mask in the `flags` argument cause `eventfd(2)` to return `EINVAL`.
pub const EFD_ALL_FLAGS: u32 = EFD_NONBLOCK | EFD_SEMAPHORE | EFD_CLOEXEC;

/// Shared region header. Lives at the base of the kernel-allocated
/// page. Padding after the header reserves the rest of the page for
/// future fields (`overflow_count` for `EFD_SEMAPHORE` overflow,
/// per-direction parked_tid for multi-reader fds, …) without changing
/// the region size.
#[repr(C, align(64))]
pub struct EventFd {
    /// Monotonic counter (semantically — `swap(0)` resets it on read).
    /// Writers `fetch_add` the value they want to signal; on
    /// `EFD_SEMAPHORE`-mode reads, the kernel decrements by 1.
    pub count: AtomicU64,

    /// Reserved for the future POSIX `read(fd)` parking path: kernel
    /// stamps the parked reader's tid here on suspend, clears on wake;
    /// writers read it (advisory) before issuing `wake_tid`. Today
    /// `read(fd)` is unimplemented, so the field stays `0` after
    /// [`init`](Self::init) and no userspace consumer should read or
    /// write it.
    ///
    /// Cross-thread reactor doorbells (mio Selector + Waker) publish
    /// their parked tid elsewhere — see `Selector::parked_tid` in the
    /// orbit mio fork.
    pub parked_tid: AtomicU32,

    /// Snapshot of the `flags` argument from `eventfd(2)`. Read by
    /// userspace to determine semaphore mode without a kernel
    /// round-trip; the kernel re-reads the slot's flag bits on each
    /// syscall (so changing `flags` here in userspace is ignored).
    pub flags: u32,

    /// Reserved. Pads the header to 64 bytes (one cache line).
    pub _reserved: [u8; 48],
}

impl EventFd {
    /// Zero-initialize the region in place. Caller passes a pointer to
    /// a zeroed 4 KiB page; this writes the header bytes the kernel
    /// guarantees on a fresh eventfd.
    ///
    /// # Safety
    /// `ptr` must point at the start of a zeroed `EVENTFD_REGION_SIZE`-
    /// byte allocation, no other reference exists, and the caller has
    /// established the kernel/userspace ABI agreement (page mapped, etc.).
    pub unsafe fn init(ptr: *mut u8, initval: u64, flags: u32) {
        unsafe {
            let p = ptr as *mut EventFd;
            (*p).count
                .store(initval, core::sync::atomic::Ordering::Release);
            (*p).parked_tid
                .store(0, core::sync::atomic::Ordering::Release);
            // `flags` is a plain field — write through; the volatile
            // isn't required here because the region is zero and no
            // observer reads before init returns.
            core::ptr::addr_of_mut!((*p).flags).write(flags);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fits_one_cache_line() {
        // ABI: header is 64 bytes — `count` (8) + `parked_tid` (4) +
        // `flags` (4) + `_reserved` (48) = 64.
        assert_eq!(core::mem::size_of::<EventFd>(), 64);
        assert_eq!(core::mem::align_of::<EventFd>(), 64);
    }

    #[test]
    fn region_size_is_one_page() {
        assert_eq!(EVENTFD_REGION_SIZE, 4096);
    }

    #[test]
    fn flag_bits_are_load_bearing() {
        assert_eq!(EFD_DEFAULT, 0);
        assert_eq!(EFD_NONBLOCK, 1);
        assert_eq!(EFD_SEMAPHORE, 2);
        assert_eq!(EFD_CLOEXEC, 4);
        assert_eq!(EFD_ALL_FLAGS, 7);
    }
}
