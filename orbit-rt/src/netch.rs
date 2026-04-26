//! NetChannel wrapper.
//!
//! Wraps the create_netch / listen_tcp / connect_tcp / send_tcp /
//! recv_tcp / close_handle dance behind a single owning handle.
//! Reservation of the VA hint goes through [`SharedRegion`] (in the
//! crate root) so the same shared-VA pool is shared with direct
//! `shared_mmap` callers — no double-mapping, no leaked frames when a
//! NetChannel closes.
//!
//! State machine the wrapper drives, mirroring `current_state.state`
//! in the shared region:
//!
//! ```text
//!         [Idle (state == 0)]
//!         /                  \
//!  start_connect          start_listen
//!     ↓                       ↓
//!  desired_state=1         desired_state=2
//!     ↓                       ↓
//!  (kernel acks)            (peer connects)
//!     ↓                       ↓
//!     └─→ [Connected (state > 0)] ←─┘
//!                  ↓
//!              reset()
//!                  ↓
//!         [Idle] (ready for fresh start_*)
//! ```
//!
//! Negative `current_state.state` values are sticky — the kernel signals
//! a connection failure by writing a negative value, and the next
//! `read`/`write` returns `EIO`. Recovery is `reset()` → fresh
//! `start_listen`/`start_connect`.

use core::{alloc::Layout, ptr::NonNull};

use net_channel::{NetChannel, NC_MAX_REGION_SIZE, NC_MIN_REGION_SIZE};
use orbit_abi::{
    Errno, Fd,
    errno::{EAGAIN, EBUSY, EINVAL, EIO},
    layout::{LARGE_PAGE, PAGE_SIZE},
    net::SockType,
    user::{close_handle, create_netch, sleep_ms},
};

use crate::SharedRegion;

/// Default poll cadence for the blocking `connect`/`listen`/`reset`
/// helpers. Matches orbit-loader's existing `POLL_SLEEP_MS`. Callers
/// that want a different cadence can fall back to the non-blocking
/// `start_*` variants and poll [`NetCh::state`] themselves.
pub const DEFAULT_POLL_MS: usize = 10;

/// Owning handle over a NetChannel: VA reservation + kernel handle. On
/// drop the kernel handle is closed and the VA reservation is returned
/// to [`crate::SHARED_VA`] so the range can be reused by a future
/// NetChannel or shared mmap.
///
/// Single-thread accessor: like the rest of orbit-rt, NetCh has no
/// internal locking — sound on single-threaded umode, needs revisiting
/// when umode grows threads (grep `FIXME(umode-threads)`).
pub struct NetCh {
    /// `None` iff the channel has been explicitly closed via
    /// [`NetCh::close`]. The wrapper is otherwise invariant: a live
    /// NetCh always has a live region and matching kernel handle.
    region: Option<SharedRegion>,
    descriptor: Fd,
    base: NonNull<NetChannel>,
}

impl NetCh {
    /// Open a NetChannel sized to hold at least `desired_ring_capacity`
    /// payload bytes per direction, of the given `sock_type`.
    ///
    /// Picks a region size that satisfies the request (rounded up to a
    /// page or a megapage as appropriate), reserves a VA hint inside
    /// `UPROC_SHARED_BASE..UPROC_SHARED_END` from [`crate::SHARED_VA`],
    /// and asks the kernel to install the NetChannel mapping at that
    /// VA. Returns `EINVAL` if `desired_ring_capacity` exceeds the
    /// per-NetChannel cap, `ENOMEM` if the shared range is exhausted,
    /// or whatever the create_netch syscall returns.
    pub fn open(desired_ring_capacity: usize, sock_type: SockType) -> Result<Self, Errno> {
        // Pick the smallest region size whose per-ring capacity covers
        // the request, then round up to the page or megapage boundary
        // so the VA reservation hits a clean alignment for the kernel.
        let region_size =
            pick_region_size(desired_ring_capacity).ok_or(Errno::new(EINVAL))?;
        let align = if region_size >= LARGE_PAGE as usize {
            LARGE_PAGE as usize
        } else {
            PAGE_SIZE as usize
        };
        let layout = Layout::from_size_align(region_size, align).map_err(|_| Errno::new(EINVAL))?;

        let region = SharedRegion::reserve(layout)?;
        let vaddr_hint = region.va();

        // create_netch returns the VA the kernel actually mapped at
        // (today always the hint, but the ABI doesn't require it).
        // Bail if it diverges so we don't end up with our hint reserved
        // but the mapping somewhere else.
        let (mapped_va, descriptor) = create_netch(vaddr_hint, region_size, sock_type as usize)?;
        if mapped_va != vaddr_hint {
            // Best-effort cleanup: tell the kernel to drop the mapping it
            // just installed elsewhere, then let `region` drop and return
            // the VA reservation. Either failure (close or drop) leaves
            // the process in a recoverable state.
            let _ = close_handle(descriptor);
            return Err(Errno::new(EIO));
        }

        // SAFETY: the kernel installed a NetChannel at `vaddr_hint`,
        // initialized via NetChannel::init, and the VA is in our
        // address space — same satp as everything else in this thread.
        let base = NonNull::new(vaddr_hint as *mut NetChannel)
            .expect("create_netch returned non-null VA");

        Ok(Self {
            region: Some(region),
            descriptor,
            base,
        })
    }

    /// Borrow the underlying [`NetChannel`]. Panics in debug if the
    /// channel has already been [`close`d](Self::close).
    pub fn channel(&self) -> &NetChannel {
        debug_assert!(self.region.is_some(), "NetCh used after close");
        // SAFETY: `base` was validated at open and the kernel mapping
        // stays live until close_handle. `region.is_some()` is the
        // wrapper's tombstone for "kernel handle still live"; callers
        // that bypass the debug_assert in release builds get UB, which
        // is what the panic in `unwrap` would catch in debug.
        unsafe { self.base.as_ref() }
    }

    /// Kernel-side handle for this NetChannel. Stable for the lifetime
    /// of the NetCh; revoked by [`Self::close`] / drop.
    pub fn fd(&self) -> Fd {
        self.descriptor
    }

    // ---- state inspection ------------------------------------------------

    /// Raw `current_state.state`. `0` is idle, positive is connected,
    /// negative is a sticky error signaled by the kernel.
    pub fn state(&self) -> i32 {
        self.channel()
            .current_state()
            .state
            .load(core::sync::atomic::Ordering::Acquire)
    }

    pub fn is_connected(&self) -> bool {
        self.state() > 0
    }

    pub fn is_idle(&self) -> bool {
        self.state() == 0
    }

    pub fn is_failed(&self) -> bool {
        self.state() < 0
    }

    pub fn readable(&self) -> usize {
        self.channel().readable()
    }

    pub fn writable(&self) -> usize {
        self.channel().writeable()
    }

    // ---- non-blocking state transitions ---------------------------------

    /// Move from idle → connecting. Returns `EBUSY` if the channel is
    /// not idle (already connecting/connected/in error). Non-blocking;
    /// poll [`Self::state`] until it goes positive (connected) or
    /// negative (failed), or use [`Self::connect`] for a blocking
    /// variant.
    pub fn start_connect(&self, addr: u32, port: u16) -> Result<(), Errno> {
        self.channel()
            .connect_tcp(addr, port)
            .map_err(|()| Errno::new(EBUSY))
    }

    /// Move from idle → listening. Same blocking semantics as
    /// [`Self::start_connect`].
    pub fn start_listen(&self, port: u16) -> Result<(), Errno> {
        self.channel()
            .listen_tcp(port)
            .map_err(|()| Errno::new(EBUSY))
    }

    // ---- blocking helpers ------------------------------------------------

    /// Connect to `(addr, port)` and block until the kernel reports
    /// success or failure. `EIO` on negative `current_state.state`.
    pub fn connect(&self, addr: u32, port: u16) -> Result<(), Errno> {
        self.start_connect(addr, port)?;
        self.wait_for_link()
    }

    /// Listen on `port` and block until a peer connects. `EIO` on
    /// negative `current_state.state`.
    pub fn listen(&self, port: u16) -> Result<(), Errno> {
        self.start_listen(port)?;
        self.wait_for_link()
    }

    fn wait_for_link(&self) -> Result<(), Errno> {
        loop {
            let s = self.state();
            if s > 0 {
                return Ok(());
            }
            if s < 0 {
                return Err(Errno::new(EIO));
            }
            sleep_ms(DEFAULT_POLL_MS)?;
        }
    }

    // ---- I/O -------------------------------------------------------------

    /// Read up to `dst.len()` bytes from the channel into `dst`.
    /// Non-blocking: returns `Ok(n)` for the bytes copied, `EAGAIN` if
    /// no data is currently available, `EIO` if the channel is in a
    /// non-positive state (idle or failed).
    pub fn read(&self, dst: &mut [u8]) -> Result<usize, Errno> {
        if !self.is_connected() {
            return Err(Errno::new(EIO));
        }
        self.channel()
            .recv_tcp(|src| {
                let n = src.len().min(dst.len());
                if n == 0 {
                    return 0;
                }
                src.sub(0, n).copy_to_slice(&mut dst[..n])
            })
            .map_err(map_io_err)
    }

    /// Write up to `src.len()` bytes to the channel.  Non-blocking:
    /// returns `Ok(n)` for the bytes accepted (which may be smaller
    /// than `src.len()` if the ring is partially full), `EAGAIN` if no
    /// space is currently available, `EIO` on non-positive state.
    pub fn write(&self, src: &[u8]) -> Result<usize, Errno> {
        if !self.is_connected() {
            return Err(Errno::new(EIO));
        }
        self.channel()
            .send_tcp(|dst| {
                let n = dst.len().min(src.len());
                if n == 0 {
                    return 0;
                }
                dst.sub(0, n).copy_from_slice(&src[..n])
            })
            .map_err(map_io_err)
    }

    /// Block until at least one byte transfers (or the channel breaks).
    /// Convenience over [`Self::read`] for callers that don't have
    /// their own poll loop. Distinct from `read_exact`-style: returns
    /// as soon as *any* data is delivered.
    pub fn read_some(&self, dst: &mut [u8]) -> Result<usize, Errno> {
        loop {
            match self.read(dst) {
                Ok(0) => sleep_ms(DEFAULT_POLL_MS)?,
                Ok(n) => return Ok(n),
                Err(Errno(e)) if e == EAGAIN => sleep_ms(DEFAULT_POLL_MS)?,
                Err(e) => return Err(e),
            }
        }
    }

    /// Block until every byte in `src` is delivered (or the channel
    /// breaks). `EIO` if the channel transitions to a non-positive
    /// state mid-flight.
    pub fn write_all(&self, src: &[u8]) -> Result<(), Errno> {
        let mut written = 0;
        while written < src.len() {
            match self.write(&src[written..]) {
                Ok(0) => sleep_ms(DEFAULT_POLL_MS)?,
                Ok(n) => written += n,
                Err(Errno(e)) if e == EAGAIN => sleep_ms(DEFAULT_POLL_MS)?,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    // ---- recycle / teardown ---------------------------------------------

    /// Run the reset handshake so this NetChannel can host a fresh
    /// `start_listen`/`start_connect`. Blocks until the kernel has
    /// torn down its half of the rings; callers that want non-blocking
    /// recycling can drop straight to `request_reset` /
    /// `complete_reset` on [`Self::channel`].
    pub fn reset(&self) -> Result<(), Errno> {
        self.channel()
            .request_reset()
            .map_err(|()| Errno::new(EBUSY))?;
        while self.state() != 0 {
            sleep_ms(DEFAULT_POLL_MS)?;
        }
        // SAFETY: NetCh is `&self`-only; the wrapper's no-thread
        // invariant means there are no outstanding `recv_tcp`/`send_tcp`
        // closures on this hart, and umode is single-threaded.
        // FIXME(umode-threads): callers that share a NetCh across
        // threads need to gate this on a "no in-flight closures" check.
        unsafe {
            self.channel().complete_reset();
        }
        Ok(())
    }

    /// Close the kernel handle and release the VA reservation.
    /// Equivalent to letting the NetCh drop, but propagates any errno
    /// from `close_handle` instead of swallowing it.
    pub fn close(mut self) -> Result<(), Errno> {
        self.do_close()
    }

    /// Internal close path. Idempotent: marks the wrapper as closed by
    /// dropping `region`, so `Drop` after `close()` is a no-op.
    fn do_close(&mut self) -> Result<(), Errno> {
        if self.region.is_none() {
            return Ok(());
        }
        // Order matters: close_handle revokes the kernel-installed PTE
        // *first*, so the VA frames we're about to free can't be
        // observed under the old mapping by a future SharedVa
        // consumer.
        let result = close_handle(self.descriptor);
        // Drop region — Drop on SharedRegion returns the frames to
        // SHARED_VA. Even if close_handle failed (e.g., kernel says
        // -EBADF because the handle was already torn down by process
        // teardown), the VA reservation is ours to release.
        self.region = None;
        result
    }
}

impl Drop for NetCh {
    fn drop(&mut self) {
        // Best-effort: process is unwinding (or the user forgot to
        // call close). Errors here are unrecoverable, so swallow.
        let _ = self.do_close();
    }
}

/// Map a `send_tcp`/`recv_tcp` negative-i32 return to an [`Errno`]:
/// `-4` (no slot) and `-5` (increments full) translate to `EAGAIN`;
/// every other negative value (channel state went non-positive
/// mid-call) translates to `EIO`.
fn map_io_err(e: isize) -> Errno {
    match e {
        -4 | -5 => Errno::new(EAGAIN),
        _ => Errno::new(EIO),
    }
}

/// Pick the smallest valid NetChannel region size whose per-ring
/// capacity is at least `desired`. Returns `None` if `desired` exceeds
/// the cap derived from [`NC_MAX_REGION_SIZE`].
fn pick_region_size(desired: usize) -> Option<usize> {
    // Two break points: one if a single page suffices (the floor), one
    // if a megapage is required (capacity scales linearly with region
    // size minus the fixed header).
    let min_capacity = NetChannel::capacity_for(NC_MIN_REGION_SIZE);
    if desired <= min_capacity {
        return Some(NC_MIN_REGION_SIZE);
    }
    let max_capacity = NetChannel::capacity_for(NC_MAX_REGION_SIZE);
    if desired > max_capacity {
        return None;
    }
    // capacity_for is monotonic in region size; normalize_region_size
    // rounds up to a page. Pick a region roughly 2x the requested
    // capacity (capacity is roughly half the region after the fixed
    // header) and let normalize_region_size align it.
    let approx = desired.saturating_mul(2).next_multiple_of(PAGE_SIZE as usize);
    NetChannel::normalize_region_size(approx)
}
