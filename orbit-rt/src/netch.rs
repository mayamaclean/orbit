//! NetChannel wrapper: declarative bindings + iterator-style sessions.
//!
//! The kernel-side reconciler (in [`net_channel::NetChannel::update_tcp`])
//! treats the binding as a sticky desired-state controller: server-mode
//! channels stay in `Listen` continuously across sessions; client-retain
//! channels keep `(addr, port)` connected with capped-exponential
//! reconnect; one-shot variants do exactly one cycle and then go
//! terminal. The binding is set once at [`NetCh::open`] and never
//! mutates after.
//!
//! Per-session lifecycle is driven by a single shared-memory flag —
//! [`NetChannelDesired::engaged`] — that the wrapper toggles on
//! [`NetCh::next_session`] entry and [`Session`] drop. The kernel
//! observes the `1 → 0` edge and recycles the smoltcp socket; for
//! retain bindings, recycle immediately re-arms the listen / reconnect.
//!
//! ```ignore
//! // server: re-listens automatically across peer sessions
//! let nc = NetCh::open(
//!     RING_CAPACITY, SockType::Tcp,
//!     BindSpec::ServerRetain { port: 7777 })?;
//!
//! loop {
//!     let s = nc.next_session()?;
//!     handle(&s);                  // read/write through `s`
//!     // dropping `s` disengages → kernel recycles, re-listens
//! }
//! ```
//!
//! Single-thread accessor. `NetCh` has no internal thread sync (matches
//! the rest of orbit-rt today); the [`AtomicBool`] in `in_session` only
//! guards against accidental nested-session entry, not against true
//! cross-thread races. Grep `FIXME(umode-threads)` for the wider issue.

extern crate alloc;

use core::{alloc::Layout, ptr::NonNull, sync::atomic::{AtomicBool, Ordering}};

use alloc::vec::Vec;

use net_channel::{BindSpec, NetChannel, NC_MAX_REGION_SIZE, NC_MIN_REGION_SIZE, channel_state};
use orbit_abi::{
    Errno, Fd,
    errno::{EAGAIN, EBUSY, EINVAL, EIO},
    layout::{LARGE_PAGE, PAGE_SIZE},
    net::SockType,
    user::{close_handle, create_netch, nc_yield},
};

use crate::SharedRegion;

/// Default poll cadence for the blocking `next_session` / `Session::drop`
/// helpers. Matches orbit-loader's prior `POLL_SLEEP_MS`. Callers that
/// want a different cadence can poll the underlying state byte directly
/// via [`NetCh::current_state`].
pub const DEFAULT_POLL_MS: usize = 50;

/// Owning handle over a NetChannel: VA reservation + kernel handle. On
/// drop the kernel handle is closed and the VA reservation is returned
/// to [`crate::SHARED_VA`] so the range can be reused by a future
/// NetChannel or shared mmap.
///
/// The binding is sticky for the life of the channel — there is no
/// rebind path. To switch from listening to dialing (or vice versa),
/// [`close`](Self::close) the channel and open a new one with the
/// desired [`BindSpec`].
pub struct NetCh {
    /// `None` iff the channel has been explicitly closed via
    /// [`NetCh::close`]. The wrapper is otherwise invariant: a live
    /// NetCh always has a live region, matching kernel handle, and
    /// fixed binding spec.
    region: Option<SharedRegion>,
    descriptor: Fd,
    base: NonNull<NetChannel>,
    /// Set while a [`Session`] guard is alive. Prevents nested
    /// [`next_session`](Self::next_session) calls from racing each other
    /// — single-threaded today, defense in depth for tomorrow.
    in_session: AtomicBool,
}

impl NetCh {
    /// Open a NetChannel sized to hold at least `desired_ring_capacity`
    /// payload bytes per direction, of the given `sock_type`, with the
    /// given sticky binding `spec`.
    ///
    /// Picks a region size that satisfies the request (rounded up to a
    /// page or a megapage as appropriate), reserves a VA hint inside
    /// `UPROC_SHARED_BASE..UPROC_SHARED_END` from [`crate::SHARED_VA`],
    /// and asks the kernel to install the NetChannel mapping at that
    /// VA *and* latch `spec` into its reconciler context. Returns
    /// `EINVAL` if `desired_ring_capacity` exceeds the per-NetChannel
    /// cap or `spec` was malformed, `ENOMEM` if the shared range is
    /// exhausted, or whatever the create_netch syscall returns.
    pub fn open(
        desired_ring_capacity: usize,
        sock_type: SockType,
        spec: BindSpec,
    ) -> Result<Self, Errno> {
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

        let (mapped_va, descriptor) = create_netch(
            vaddr_hint, region_size, sock_type as usize, spec.pack(),
        )?;
        if mapped_va != vaddr_hint {
            // Best-effort cleanup: tell the kernel to drop the mapping it
            // just installed elsewhere, then let `region` drop and return
            // the VA reservation.
            let _ = close_handle(descriptor);
            return Err(Errno::new(EIO));
        }

        // SAFETY: the kernel installed a NetChannel at `vaddr_hint`,
        // initialized via NetChannel::init, and the VA is in our address
        // space — same satp as everything else in this thread.
        let base = NonNull::new(vaddr_hint as *mut NetChannel)
            .expect("create_netch returned non-null VA");

        Ok(Self {
            region: Some(region),
            descriptor,
            base,
            in_session: AtomicBool::new(false),
        })
    }

    /// Borrow the underlying [`NetChannel`]. Panics in debug if the
    /// channel has already been [`close`d](Self::close).
    pub fn channel(&self) -> &NetChannel {
        debug_assert!(self.region.is_some(), "NetCh used after close");
        // SAFETY: `base` was validated at open; the kernel mapping
        // stays live until close_handle. `region.is_some()` is the
        // wrapper's tombstone for "kernel handle still live."
        unsafe { self.base.as_ref() }
    }

    /// Kernel-side handle for this NetChannel. Stable for the lifetime
    /// of the NetCh; revoked by [`Self::close`] / drop.
    pub fn fd(&self) -> Fd {
        self.descriptor
    }

    /// Raw `current.state`. See [`net_channel::channel_state`] for
    /// the value map: `IDLE` (0), `IN_FLIGHT` (1), `ACTIVE` (2),
    /// `CLOSING` (3 — graceful-close drain), `FAILED` (-1).
    /// Mostly useful for diagnostics; normal callers go through
    /// [`next_session`].
    pub fn current_state(&self) -> i32 {
        self.channel()
            .current()
            .state
            .load(Ordering::Acquire)
    }

    /// Sticky cause from [`current_state`](Self::current_state) when it
    /// returns negative, otherwise `0`. Values are the `EBIND_*`
    /// constants in [`net_channel`].
    pub fn fail_cause(&self) -> i32 {
        self.channel().current().fail_cause.load(Ordering::Acquire)
    }

    pub fn readable(&self) -> usize {
        self.channel().readable()
    }

    pub fn writable(&self) -> usize {
        self.channel().writeable()
    }

    /// Block until the kernel reports an active session (`current.state
    /// >= 2`), then return a guard the caller reads/writes through.
    /// The guard's `Drop` disengages, signalling the kernel to recycle
    /// the smoltcp socket — for retain bindings, recycle immediately
    /// re-arms the listen/connect; for one-shot bindings, recycle
    /// transitions the channel to its terminal state.
    ///
    /// Returns `EBUSY` if a [`Session`] from this `NetCh` is already
    /// alive, `EIO` if the channel is in a sticky-terminal state.
    pub fn next_session(&self) -> Result<Session<'_>, Errno> {
        if self.in_session.swap(true, Ordering::AcqRel) {
            return Err(Errno::new(EBUSY));
        }
        // Engage *first* so the kernel sees us claiming the upcoming
        // session before it considers any "no-claimer" recycle path
        // (the reconciler doesn't actually have such a path today —
        // recycling is gated on the 1→0 edge — but ordering this way
        // means future kernel changes can't strand us mid-claim).
        self.channel().engage();
        loop {
            let s = self.current_state();
            if s == channel_state::ACTIVE {
                return Ok(Session { nc: self });
            }
            if s == channel_state::FAILED {
                // Failed before we could claim: release the in-session
                // guard and surface the error.
                self.channel().disengage();
                self.in_session.store(false, Ordering::Release);
                return Err(Errno::new(EIO));
            }
            // nc_yield notifies k_net (in case it has work to do —
            // e.g. drive listen→Established) and parks us for up to
            // DEFAULT_POLL_MS, returning early on `WakeEvent::Pid`
            // when our channel state changes.
            nc_yield(DEFAULT_POLL_MS)?;
        }
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
        // close_handle revokes the kernel-installed PTE first, so the
        // VA frames we're about to free can't be observed under the old
        // mapping by a future SharedVa consumer.
        let result = close_handle(self.descriptor);
        // Even if close_handle failed (e.g. -EBADF because the handle
        // was already torn down by process teardown), the VA reservation
        // is ours to release.
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

/// Active-session guard handed back by [`NetCh::next_session`]. All I/O
/// goes through this type so the kernel can rely on `desired.engaged`
/// being a meaningful claim signal — drop the guard to release the
/// session, and the kernel auto-recycles.
///
/// `Session` is a `&NetCh` borrow, not `&mut`, so the underlying NetCh
/// stays usable for things like `current_state` queries during a
/// session. The single-session-at-a-time invariant is enforced by the
/// `NetCh::in_session` flag rather than the borrow checker.
pub struct Session<'a> {
    nc: &'a NetCh,
}

impl<'a> Session<'a> {
    /// Read up to `dst.len()` bytes from the channel into `dst`.
    /// Non-blocking: returns `Ok(n)` for the bytes copied, `EAGAIN` if
    /// no data is currently available, `EIO` if the channel transitions
    /// to a non-active state.
    pub fn read(&self, dst: &mut [u8]) -> Result<usize, Errno> {
        self.nc.channel()
            .recv_tcp(|src| {
                let n = src.len().min(dst.len());
                if n == 0 {
                    return 0;
                }
                src.sub(0, n).copy_to_slice(&mut dst[..n])
            })
            .map_err(map_io_err)
    }

    /// Write up to `src.len()` bytes to the channel. Non-blocking:
    /// returns `Ok(n)` for the bytes accepted (which may be smaller
    /// than `src.len()` if the ring is partially full), `EAGAIN` if no
    /// space is currently available, `EIO` on non-active state.
    pub fn write(&self, src: &[u8]) -> Result<usize, Errno> {
        self.nc.channel()
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
    /// their own poll loop.
    pub fn read_some(&self, dst: &mut [u8]) -> Result<usize, Errno> {
        loop {
            match self.read(dst) {
                Ok(0) => nc_yield(DEFAULT_POLL_MS)?,
                Ok(n) => return Ok(n),
                Err(Errno(e)) if e == EAGAIN => nc_yield(DEFAULT_POLL_MS)?,
                Err(e) => return Err(e),
            }
        }
    }

    /// Block until at least one byte transfers (or the channel breaks).
    /// Convenience over [`Self::read`] for callers that don't have
    /// their own poll loop.
    pub fn read_some_with_poll_timeout(&self, dst: &mut [u8], poll_ms: usize) -> Result<usize, Errno> {
        loop {
            match self.read(dst) {
                Ok(0) => nc_yield(poll_ms)?,
                Ok(n) => return Ok(n),
                Err(Errno(e)) if e == EAGAIN => nc_yield(poll_ms)?,
                Err(e) => return Err(e),
            }
        }
    }

    /// Append the next staged slice (up to `cap_bytes`) onto `dst` in
    /// one `recv_tcp` round-trip. Non-blocking: `Ok(0)` if no slice is
    /// staged, `EAGAIN` for ring transients, `EIO` on sticky failure.
    ///
    /// Drains slice-at-a-time rather than chunk-at-a-time — the kernel
    /// stages a slice covering the entire contiguous run of fresh rx
    /// data (potentially tens of KiB). Bounded only by `cap_bytes` so a
    /// huge staged slice can't OOM the consumer.
    ///
    /// Why not `Session::read`-into-a-`Vec` (resize first, then read)?
    /// `read` clamps to `dst.len()` which the caller fixed up-front.
    /// Slice-granularity drain needs the closure to see `src.len()` and
    /// size the destination accordingly — that's what this method does.
    pub fn read_into_vec(
        &self,
        dst: &mut Vec<u8>,
        cap_bytes: usize,
    ) -> Result<usize, Errno> {
        self.nc.channel()
            .recv_tcp(|src| {
                let n = src.len().min(cap_bytes);
                if n == 0 {
                    return 0;
                }
                let start = dst.len();
                // resize zeroes the appended region; copy_to_slice
                // overwrites it. The double-write is one extra memset
                // — cheap relative to the avoided round-trips.
                dst.resize(start + n, 0);
                src.sub(0, n).copy_to_slice(&mut dst[start..start + n])
            })
            .map_err(map_io_err)
    }

    /// Block until at least one byte is appended to `dst` (or the
    /// channel breaks). Convenience over [`Self::read_into_vec`] for
    /// drain-everything callers like the loader.
    pub fn read_into_vec_some(
        &self,
        dst: &mut Vec<u8>,
        cap_bytes: usize,
    ) -> Result<usize, Errno> {
        loop {
            match self.read_into_vec(dst, cap_bytes) {
                Ok(0) => nc_yield(DEFAULT_POLL_MS)?,
                Ok(n) => return Ok(n),
                Err(Errno(e)) if e == EAGAIN => nc_yield(DEFAULT_POLL_MS)?,
                Err(e) => return Err(e),
            }
        }
    }

    /// Block until every byte in `src` is delivered (or the channel
    /// breaks).
    pub fn write_all(&self, src: &[u8]) -> Result<(), Errno> {
        let mut written = 0;
        while written < src.len() {
            match self.write(&src[written..]) {
                Ok(0) => nc_yield(DEFAULT_POLL_MS)?,
                Ok(n) => written += n,
                Err(Errno(e)) if e == EAGAIN => nc_yield(DEFAULT_POLL_MS)?,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Peer's IPv4 address, populated by the kernel when the session
    /// became active. For server bindings: whoever connected. For
    /// client bindings: redundant with the bind params, exposed for
    /// symmetry.
    pub fn peer_addr(&self) -> u32 {
        self.nc.channel().current().peer_addr.load(Ordering::Acquire)
    }

    /// Peer's TCP port (paired with [`peer_addr`](Self::peer_addr)).
    pub fn peer_port(&self) -> u16 {
        self.nc.channel().current().peer_port.load(Ordering::Acquire)
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // Signal disengagement; the kernel reconciler picks up the
        // 1→0 edge on its next poll and starts recycling. Wait for
        // the kernel to acknowledge by transitioning *out of state
        // 2* — for ServerRetain the kernel goes 2 → 1 (re-listening
        // immediately), for ClientRetain 2 → 0 (idle, awaiting next
        // engage), for one-shot 2 → -1 (terminal). All of these
        // mean "kernel has aborted the smoltcp socket and reset its
        // ring halves," which is what we need before reset_user_side
        // can run safely.
        //
        // Cap the wait at ~100 polls (~1s at DEFAULT_POLL_MS=10) so
        // a wedged kernel can't hang the drop indefinitely.
        let nc = self.nc;
        nc.channel().disengage();

        for _ in 0..100 {
            // Wait for the kernel to leave both ACTIVE *and* CLOSING
            // — the older `!= ACTIVE` shape raced a still-draining
            // smoltcp socket against the next session's `engage` and
            // dropped the last `write_all`. See
            // [`net_channel::channel_state::CLOSING`].
            let s = nc.current_state();
            if s != channel_state::ACTIVE && s != channel_state::CLOSING {
                break;
            }
            // Drop can't propagate `?`; ignore EINTR-equivalent errors
            // from nc_yield — kernel cap means it's bounded. The yield
            // also nudges k_net so the disengage edge gets observed
            // promptly without waiting for k_net's own heartbeat.
            let _ = nc_yield(DEFAULT_POLL_MS);
        }

        // SAFETY: `Session` owned exclusive access to the rings while
        // it was alive, and the state-out-of-2 wait above observed
        // the kernel's recycle. Kernel has reset_kernel_side'd in the
        // disengage edge handler before flipping state away from 2.
        unsafe { nc.channel().reset_user_side(); }

        nc.in_session.store(false, Ordering::Release);
    }
}

/// Map a `send_tcp`/`recv_tcp` non-success return to an [`Errno`].
///
/// `-4` / `-5` are ring-internal transients (no staged slice, or the
/// increment ring is full) — caller retries.
///
/// `0` / `1` are *channel-state* transients — the channel was idle or
/// in-flight when we sampled `current.state`. The single-thread
/// invariant inside one process means an in-flight read can't normally
/// observe a 2→0 (recycle) or 2→1 (server-retain re-listen) edge — but
/// reserving these to EAGAIN keeps the wrapper correct under future
/// cross-thread use, and matches the right semantic: "transient, retry"
/// vs. "dead, give up." Without this, a benign mid-flight transition
/// from a recycle path would terminate `read_some`/`write_all` with
/// EIO instead of waiting it out.
///
/// Sticky-negative (anything else) is the real-failure path → EIO.
fn map_io_err(e: isize) -> Errno {
    match e {
        -4 | -5 => Errno::new(EAGAIN),
        0 | 1   => Errno::new(EAGAIN),
        _       => Errno::new(EIO),
    }
}

/// Pick the smallest valid NetChannel region size whose per-ring
/// capacity is at least `desired`. Returns `None` if `desired` exceeds
/// the cap derived from [`NC_MAX_REGION_SIZE`].
fn pick_region_size(desired: usize) -> Option<usize> {
    let min_capacity = NetChannel::capacity_for(NC_MIN_REGION_SIZE);
    if desired <= min_capacity {
        return Some(NC_MIN_REGION_SIZE);
    }
    let max_capacity = NetChannel::capacity_for(NC_MAX_REGION_SIZE);
    if desired > max_capacity {
        return None;
    }
    let approx = desired.saturating_mul(2).next_multiple_of(PAGE_SIZE as usize);
    NetChannel::normalize_region_size(approx)
}
