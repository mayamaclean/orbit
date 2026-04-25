#![no_std]

use core::marker::PhantomData;
use core::mem::size_of;
use core::sync::atomic::{AtomicI32, AtomicU16, AtomicU32, AtomicUsize, Ordering};

use mem::round_usize_up;

#[cfg(feature = "kernel")]
use core::net::Ipv4Addr;
#[cfg(feature = "kernel")]
use smoltcp::{iface::Interface, wire::IpAddress};
#[cfg(feature = "kernel")]
use smoltcp::socket::tcp::State as TcpState;

#[cfg(feature = "kernel")]
use tracing::{error};

/// Mutable view into a region of shared memory handed to `send_tcp`'s
/// closure. Writes go through [`core::ptr::write_volatile`], which
/// LLVM can't DCE the way it can writes through `&mut [u8]` — that
/// slice type carries `noalias`, so LLVM assumes nothing else can
/// observe the writes and deletes them (the bug that was silently
/// zeroing outbound TCP payloads).
pub struct VolSliceMut<'a> {
    ptr: *mut u8,
    len: usize,
    _m: PhantomData<&'a mut [u8]>,
}

impl<'a> VolSliceMut<'a> {
    /// # Safety
    /// `ptr` must be valid for writes of `len` bytes, and no other
    /// `&mut [u8]` may alias this region for the `'a` lifetime.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len, _m: PhantomData }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Copy up to `self.len()` bytes from `src`, returning the number
    /// actually written.
    pub fn copy_from_slice(&self, src: &[u8]) -> usize {
        let n = core::cmp::min(self.len, src.len());
        for i in 0..n {
            unsafe { self.ptr.add(i).write_volatile(src[i]); }
        }
        n
    }

    /// Read the byte at `i`. Panics if out of bounds (matches `slice[i]`).
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        assert!(i < self.len, "VolSliceMut::get: index {i} out of bounds (len {})", self.len);
        unsafe { self.ptr.add(i).read_volatile() }
    }

    /// Bounds-checked read. Returns `None` if out of range.
    #[inline]
    pub fn get_checked(&self, i: usize) -> Option<u8> {
        if i >= self.len { return None; }
        Some(unsafe { self.ptr.add(i).read_volatile() })
    }

    /// Write `b` at `i`. Panics if out of bounds.
    #[inline]
    pub fn set(&self, i: usize, b: u8) {
        assert!(i < self.len, "VolSliceMut::set: index {i} out of bounds (len {})", self.len);
        unsafe { self.ptr.add(i).write_volatile(b); }
    }

    /// Bounds-checked write. Returns `None` if out of range.
    #[inline]
    pub fn set_checked(&self, i: usize, b: u8) -> Option<()> {
        if i >= self.len { return None; }
        unsafe { self.ptr.add(i).write_volatile(b); }
        Some(())
    }

    /// Legacy name for [`set_checked`]. Kept for callers that already
    /// use the `write_at` API; new code should prefer `set` / `set_checked`.
    #[inline]
    pub fn write_at(&self, offset: usize, b: u8) -> Option<()> {
        self.set_checked(offset, b)
    }

    /// Sub-slice `[start, end)`. Panics if out of bounds.
    pub fn sub(&self, start: usize, end: usize) -> VolSliceMut<'_> {
        assert!(start <= end, "VolSliceMut::sub: start {start} > end {end}");
        assert!(end <= self.len, "VolSliceMut::sub: end {end} > len {}", self.len);
        unsafe {
            VolSliceMut::from_raw_parts(self.ptr.add(start), end - start)
        }
    }

    /// Read-only reborrow. Useful when a function wants a `VolSlice`
    /// but the caller only has a `VolSliceMut`.
    #[inline]
    pub fn as_readonly(&self) -> VolSlice<'_> {
        unsafe { VolSlice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Read-only view of a region of shared memory handed to `recv_tcp`'s
/// closure. Reads go through [`core::ptr::read_volatile`] — symmetric
/// with [`VolSliceMut`] so the compiler can't cache stale values that
/// may have been written by the other side of the channel.
pub struct VolSlice<'a> {
    ptr: *const u8,
    len: usize,
    _m: PhantomData<&'a [u8]>,
}

impl<'a> VolSlice<'a> {
    /// # Safety
    /// `ptr` must be valid for reads of `len` bytes.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self { ptr, len, _m: PhantomData }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Copy up to `dst.len()` bytes into `dst`, returning the number
    /// actually copied.
    pub fn copy_to_slice(&self, dst: &mut [u8]) -> usize {
        let n = core::cmp::min(self.len, dst.len());
        for i in 0..n {
            dst[i] = unsafe { self.ptr.add(i).read_volatile() };
        }
        n
    }

    /// Read the byte at `i`. Panics if out of bounds (matches `slice[i]`).
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        assert!(i < self.len, "VolSlice::get: index {i} out of bounds (len {})", self.len);
        unsafe { self.ptr.add(i).read_volatile() }
    }

    /// Bounds-checked read. Returns `None` if out of range.
    #[inline]
    pub fn get_checked(&self, i: usize) -> Option<u8> {
        if i >= self.len { return None; }
        Some(unsafe { self.ptr.add(i).read_volatile() })
    }

    /// Legacy name for [`get_checked`]. New code should prefer `get`
    /// (panic on OOB) or `get_checked`.
    #[inline]
    pub fn read_at(&self, offset: usize) -> Option<u8> {
        self.get_checked(offset)
    }

    /// Sub-slice `[start, end)`. Panics if out of bounds.
    pub fn sub(&self, start: usize, end: usize) -> VolSlice<'_> {
        assert!(start <= end, "VolSlice::sub: start {start} > end {end}");
        assert!(end <= self.len, "VolSlice::sub: end {end} > len {}", self.len);
        unsafe {
            VolSlice::from_raw_parts(self.ptr.add(start), end - start)
        }
    }

    /// True if the first `prefix.len()` bytes equal `prefix`.
    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        if self.len < prefix.len() { return false; }
        for i in 0..prefix.len() {
            if unsafe { self.ptr.add(i).read_volatile() } != prefix[i] {
                return false;
            }
        }
        true
    }

    /// Raw pointer, for syscalls that read this memory from a
    /// non-compiler context (e.g. `serial_print`, whose read happens
    /// in the kernel under SUM — the compiler can't elide the syscall).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 { self.ptr }
}

// SPSC queue lives in [`process::spsc`] (re-exported below) so other
// kernel sync paths (e.g. ProcessStdin) can share the implementation.
// The `#[repr(C)]` layout discipline that NetChannel's ABI relies on
// is preserved by the `process::SpscQueue` type definition.
pub use process::SpscQueue;

/// Fixed header offsets within a NetChannel region. The tx/rx queues follow
/// at `NC_TX_OFF` and `NC_TX_OFF + queue_len` respectively — `queue_len` is
/// runtime-selectable by the user at channel creation, up to
/// [`NC_MAX_REGION_SIZE`].
pub const NC_DESIRED_OFF: usize = 128;
pub const NC_CURRENT_OFF: usize = 256;
pub const NC_TX_OFF: usize = 384;

/// Minimum region size. Everything — states, tx, rx — packs into a single
/// 4 KiB page. Per-ring usable payload is
/// `(NC_MIN_REGION_SIZE - NC_TX_OFF) / 2 - size_of::<NetChannelQueue>() + 1`
/// (roughly ~1.7 KiB).
pub const NC_MIN_REGION_SIZE: usize = 4096;

/// Maximum region size. Cap at 256 KiB so misbehaving umode can't demand an
/// arbitrarily large kernel-side Shared allocation. Per-ring usable payload
/// at the cap is ~127 KiB.
pub const NC_MAX_REGION_SIZE: usize = 256 * 1024;

// Compile-time checks so layout invariants fail the build, not the boot.
// If these fire, `NC_TX_OFF` / `NetChannelQueue` layout / min-region math
// have drifted: either the fixed header offsets or `queue_len_for` need
// updating.
const _: () = {
    assert!(
        NC_TX_OFF % core::mem::align_of::<NetChannelQueue>() == 0,
        "NC_TX_OFF must be aligned for NetChannelQueue",
    );
    assert!(
        NetChannel::queue_len_for(NC_MIN_REGION_SIZE)
            % core::mem::align_of::<NetChannelQueue>() == 0,
        "queue_len at NC_MIN_REGION_SIZE must align the rx subregion",
    );
    assert!(
        NetChannel::capacity_for(NC_MIN_REGION_SIZE) > 0,
        "NC_MIN_REGION_SIZE leaves no room for a ring payload",
    );
};

#[repr(C, align(128))]
pub struct NetChannelState {
    pub state_addr: AtomicU32,
    pub state: AtomicI32,
    pub state_remote_port: AtomicU16,
    pub state_local_port: AtomicU16
}

/// Ring holding `(offset, len)` pairs pointing into [`NetChannelQueue::buf`].
/// N=2 → capacity 1.
type SliceQueue = SpscQueue<(usize, usize), 2>;
/// Ring of byte counts the consumer has advanced past. N=2 → capacity 1.
type IncrementQueue = SpscQueue<usize, 2>;

/// Shared producer/consumer queues + payload ring for one direction.
/// `#[repr(C)]` pins the field order so kernel- and user-side
/// compilations land at the same offsets.
#[repr(C)]
pub struct NetChannelQueue {
    slices: SliceQueue,
    increments: IncrementQueue,
    pub avail: AtomicUsize,
    capacity: usize,
    buf: u8,
}

impl NetChannelQueue {
    pub fn capacity(&self) -> usize { self.capacity }

    pub fn buf_ptr(&self) -> *mut u8 {
        &self.buf as *const u8 as *mut u8
    }

    /// # Safety
    /// Caller must be the single producer for `slices` on this queue.
    pub unsafe fn enqueue_slice(&self, v: (usize, usize)) -> Result<(), (usize, usize)> {
        unsafe { self.slices.enqueue(v) }
    }

    /// # Safety
    /// Caller must be the single consumer for `slices` on this queue.
    pub unsafe fn dequeue_slice(&self) -> Option<(usize, usize)> {
        unsafe { self.slices.dequeue() }
    }

    pub fn slices_is_empty(&self) -> bool { self.slices.is_empty() }
    pub fn slices_len(&self) -> usize { self.slices.len() }

    /// # Safety
    /// Caller must be the single producer for `increments` on this queue.
    pub unsafe fn enqueue_increment(&self, v: usize) -> Result<(), usize> {
        unsafe { self.increments.enqueue(v) }
    }

    /// # Safety
    /// Caller must be the single consumer for `increments` on this queue.
    pub unsafe fn dequeue_increment(&self) -> Option<usize> {
        unsafe { self.increments.dequeue() }
    }

    pub fn increments_is_full(&self) -> bool { self.increments.is_full() }
    pub fn increments_len(&self) -> usize { self.increments.len() }
}

/// Control header for a NetChannel region. Self-anchored: sub-region
/// accessors compute their targets from `self` + fixed offsets + the
/// runtime `queue_len`, so they resolve correctly under the user satp
/// *and* under the kernel satp (through KDMAP).
///
/// Never construct this directly — the kernel allocates the region and
/// calls [`NetChannel::init`].
#[repr(C)]
pub struct NetChannel {
    queue_len: usize,
}

impl NetChannel {
    /// Normalize a user-requested region size: clamp into
    /// `[NC_MIN_REGION_SIZE, NC_MAX_REGION_SIZE]`, round up to a page, and
    /// align so `queue_len` (half the post-header span) ends up
    /// `usize`-aligned.
    pub fn normalize_region_size(requested: usize) -> Option<usize> {
        if requested == 0 { return None }
        let clamped = requested.clamp(NC_MIN_REGION_SIZE, NC_MAX_REGION_SIZE);
        // Round up to page so each allocation fits cleanly in a whole
        // number of 4 KiB frames.
        let page_up = round_usize_up(clamped, 4096);
        if page_up > NC_MAX_REGION_SIZE { return None }
        if page_up < NC_MIN_REGION_SIZE { return None }
        // NC_TX_OFF is 16-aligned; dividing the remainder in half gives a
        // multiple of 8 as long as the region size is, which is already
        // guaranteed by page-rounding.
        Some(page_up)
    }

    /// Queue subregion length (header + ring buf) for a given total region
    /// size. Rounded down to `align_of::<NetChannelQueue>()` so
    /// `NC_TX_OFF + queue_len` (the rx subregion base) lands on an
    /// alignment valid for `*mut NetChannelQueue`. Wastes at most
    /// `align - 1` bytes of ring capacity; keeps the layout correct under
    /// any page-rounded `region_size`.
    pub const fn queue_len_for(region_size: usize) -> usize {
        let raw = (region_size - NC_TX_OFF) / 2;
        let align = core::mem::align_of::<NetChannelQueue>();
        raw & !(align - 1)
    }

    /// Per-ring usable payload capacity for a given total region size.
    pub const fn capacity_for(region_size: usize) -> usize {
        // `buf: u8` is the first byte of the ring, so the header
        // "overhead" is size_of - 1 and the rest is payload.
        Self::queue_len_for(region_size) - size_of::<NetChannelQueue>() + 1
    }

    /// Stamp the queue capacities on a freshly-allocated, zeroed region.
    ///
    /// # Safety
    /// - `base` must point at a zeroed, writable region of at least
    ///   `region_size` bytes, page-aligned.
    /// - `region_size` must be a value returned by
    ///   [`normalize_region_size`](Self::normalize_region_size).
    /// - No one else must observe the region between alloc and this call;
    ///   the kernel maps it into user VA only after init returns.
    pub unsafe fn init(base: *mut u8, region_size: usize) {
        let queue_len = Self::queue_len_for(region_size);
        let capacity = Self::capacity_for(region_size);
        unsafe {
            (*(base as *mut NetChannel)).queue_len = queue_len;

            let tx = base.add(NC_TX_OFF) as *mut NetChannelQueue;
            (*tx).capacity = capacity;

            let rx = base.add(NC_TX_OFF + queue_len) as *mut NetChannelQueue;
            (*rx).capacity = capacity;
        }
    }

    pub fn queue_len(&self) -> usize { self.queue_len }

    fn anchor(&self) -> *const u8 {
        self as *const Self as *const u8
    }

    pub fn desired_state(&self) -> &NetChannelState {
        unsafe { &*(self.anchor().add(NC_DESIRED_OFF) as *const NetChannelState) }
    }

    pub fn current_state(&self) -> &NetChannelState {
        unsafe { &*(self.anchor().add(NC_CURRENT_OFF) as *const NetChannelState) }
    }

    pub fn tx(&self) -> &NetChannelQueue {
        unsafe { &*(self.anchor().add(NC_TX_OFF) as *const NetChannelQueue) }
    }

    pub fn rx(&self) -> &NetChannelQueue {
        unsafe { &*(self.anchor().add(NC_TX_OFF + self.queue_len) as *const NetChannelQueue) }
    }

    /// Reset the kernel-owned halves of both rings so the next connection
    /// on this NetChannel starts clean. Kernel owns: tx.slices producer,
    /// tx.increments consumer, rx.slices producer, rx.increments consumer.
    /// `avail` fields are touched here too — both sides mutate them at
    /// steady state, but during reset the smoltcp socket is aborted and
    /// userspace is blocked on the state handshake, so kernel can zero
    /// them unilaterally.
    ///
    /// # Safety
    /// Must be called while the smoltcp socket is aborted and before
    /// the kernel releases `current_state.state = 0` — otherwise
    /// userspace may observe stale-then-zero indices out of order.
    #[cfg(feature = "kernel")]
    pub unsafe fn reset_kernel_side(&self) {
        let tx = self.tx();
        let rx = self.rx();
        unsafe {
            tx.slices.reset_producer();
            tx.increments.reset_consumer();
            rx.slices.reset_producer();
            rx.increments.reset_consumer();
        }
        tx.avail.store(0, Ordering::Release);
        rx.avail.store(0, Ordering::Release);
    }

    /// Reset the user-owned halves of both rings. Mirror of
    /// [`reset_kernel_side`]; user owns: tx.slices consumer,
    /// tx.increments producer, rx.slices consumer, rx.increments
    /// producer.
    ///
    /// # Safety
    /// Must be called after observing `current_state.state == 0` (which
    /// establishes that the kernel has already done its half) and
    /// before issuing a fresh `listen_tcp` / `connect_tcp`.
    #[cfg(not(feature = "kernel"))]
    pub unsafe fn reset_user_side(&self) {
        let tx = self.tx();
        let rx = self.rx();
        unsafe {
            tx.slices.reset_consumer();
            tx.increments.reset_producer();
            rx.slices.reset_consumer();
            rx.increments.reset_producer();
        }
    }

    /// Pump one poll cycle against `socket`. `pending_rx_ack` /
    /// `pending_tx_ack` are kernel-local state (held per socket in
    /// `SocketReq`): each is "a slice is enqueued and we haven't drained
    /// the matching increment yet." They gate re-enqueue so we don't
    /// double-post a slice in the window between the user's
    /// `dequeue_slice` and their `enqueue_increment`. Without them,
    /// `get_next_rx` / `get_next_tx` returns the same bytes (smoltcp
    /// hasn't been advanced yet) and we'd stage a duplicate that
    /// deadlocks the queue once the real ack drains.
    #[cfg(feature = "kernel")]
    pub fn update_tcp(
        &self,
        mut iface: Interface,
        socket: &mut smoltcp::socket::tcp::Socket,
        pending_rx_ack: &mut bool,
        pending_tx_ack: &mut bool,
        issued_desired: &mut i32,
    ) -> Interface {
        let current_state = self.current_state();
        let desired_state = self.desired_state();

        let channel_state = current_state.state.load(Ordering::Relaxed);

        // Reuse path: userspace has acknowledged the current connection
        // by dropping desired_state back to 0 while we're still marked
        // non-idle. Abort the socket, reset our half of the rings, then
        // release current_state = 0 so userspace can safely reset its
        // side and issue the next listen/connect. Negative channel_state
        // (error terminal) is also eligible — reset is how userspace
        // recovers without tearing down the NetChannel.
        if channel_state != 0
            && desired_state.state.load(Ordering::Acquire) == 0
        {
            socket.abort();
            unsafe { self.reset_kernel_side(); }
            *pending_rx_ack = false;
            *pending_tx_ack = false;
            *issued_desired = 0;
            // Release ordering: every reset store above is visible to a
            // userspace Acquire-load of current_state.state before it
            // reads any queue index.
            current_state.state.store(0, Ordering::Release);
            return iface;
        }

        if channel_state < 0 {
            return iface
        }
        else if channel_state == 0 {
            let port = desired_state.state_remote_port.load(Ordering::Acquire);
            let addr = desired_state.state_addr.load(Ordering::Acquire);
            let state = desired_state.state.load(Ordering::Acquire);

            // Level-triggered: only issue connect/listen when
            // `desired_state` changes. Without this, a peer RST drops the
            // socket back to CLOSED and we'd immediately re-call
            // `socket.connect` on the next poll, producing a SYN storm
            // when no one's listening. Userspace re-arms by requesting a
            // reset (which clears `issued_desired` to 0 above) before
            // re-writing a fresh desired_state.
            if !socket.get_timeout_status() && socket.state() == TcpState::Closed {
                if state != *issued_desired {
                    match state {
                        // connect
                        1 => {
                            let addr = IpAddress::Ipv4(Ipv4Addr::from_bits(addr));
                            if let Err(e) = socket.connect(
                                iface.context(),
                                (addr, port),
                                desired_state.state_local_port.load(Ordering::Acquire))
                            {
                                #[cfg(feature = "kernel")]
                                error!("tcp: failed to start connect: {e:?}");
                            }
                        },
                        // listen
                        2 => {
                            if let Err(e) = socket.listen(desired_state.state_local_port.load(Ordering::Acquire)) {
                                #[cfg(feature = "kernel")]
                                error!("tcp: failed to start listen: {e:?}");
                            }
                        }
                        _ => ()
                    }
                    *issued_desired = state;
                    return iface
                } else if *issued_desired > 0 {
                    // We already issued for this intent and smoltcp is
                    // back at CLOSED without ever reporting is_open().
                    // That's the terminal failure case — connect got
                    // RST'd, or the handshake never completed. Surface
                    // it to userspace so its poll loop breaks instead
                    // of spinning on state==0 forever. Userspace
                    // recovers via `request_reset` + re-issue.
                    current_state.state.store(-1, Ordering::Release);
                    return iface
                }
            }

            if socket.get_timeout_status() {
                current_state.state.store(-1, Ordering::Release);
                return iface
            }

            if socket.is_open() {
                current_state.state.store(state, Ordering::Release);
            }
        }

        let tx = self.tx();
        let rx = self.rx();

        if socket.may_recv() {
            // SAFETY: kernel is the sole consumer of rx.increments.
            while let Some(user_rx_count) = unsafe { rx.dequeue_increment() } {
                if let Err(e) = socket.recv(|_b| (user_rx_count, user_rx_count)) {
                    #[cfg(feature = "kernel")]
                    error!("tcp: failed recv: {e:?}");
                    current_state.state.store(-2, Ordering::Release);

                    return iface
                }
                // User has acknowledged the prior slice; clear so the
                // next `get_next_rx` can stage a fresh one.
                *pending_rx_ack = false;
            }

            if rx.slices_is_empty() && !*pending_rx_ack {
                let next_rx = socket.get_next_rx();
                if next_rx.1 > 0 {
                    // SAFETY: kernel is the sole producer of rx.slices.
                    let _r = unsafe { rx.enqueue_slice(next_rx) };

                    #[cfg(feature = "kernel")]
                    {
                        let _increments_len = rx.increments_len();
                        let _avail_len = rx.slices_len();

                        let _slice = unsafe {
                            core::slice::from_raw_parts(rx.buf_ptr(), next_rx.1)
                        };

                        //info!("tcp: next_rx={slice:02X?}, increments_len={increments_len}, avail_len={avail_len}");
                    }

                    core::sync::atomic::fence(Ordering::SeqCst);
                    rx.avail.store(next_rx.1, Ordering::Release);
                    *pending_rx_ack = true;
                }
            }
        }

        if socket.may_send() {
            // SAFETY: kernel is the sole consumer of tx.increments.
            while let Some(user_tx_count) = unsafe { tx.dequeue_increment() } {
                if let Err(e) = socket.send(|_b| (user_tx_count, user_tx_count)) {
                    #[cfg(feature = "kernel")]
                    error!("tcp: failed send: {e:?}");
                    current_state.state.store(-3, Ordering::Release);

                    return iface
                }
                *pending_tx_ack = false;
            }

            if tx.slices_is_empty() && !*pending_tx_ack {
                let next_tx = socket.get_next_tx();
                if next_tx.1 > 0 {
                    // SAFETY: kernel is the sole producer of tx.slices.
                    let _r = unsafe { tx.enqueue_slice(next_tx) };

                    #[cfg(feature = "kernel")]
                    {
                        let _increments_len = tx.increments_len();
                        let _avail_len = tx.slices_len();
                        //info!("tcp: next_tx={next_tx:08X?}, increments_len={increments_len}, avail_len={avail_len} r={r:08X?}");
                    }

                    core::sync::atomic::fence(Ordering::SeqCst);
                    tx.avail.store(next_tx.1, Ordering::Release);
                    *pending_tx_ack = true;
                }
            }
        }
        iface
    }

    #[cfg(not(feature = "kernel"))]
    pub fn send_tcp<F>(&self, f: F) -> Result<usize, isize>
        where F: FnOnce(&VolSliceMut) -> usize
    {
        let current_state = self.current_state();

        let channel_state = current_state.state.load(Ordering::Acquire);
        if channel_state <= 0 {
            return Err(channel_state as isize)
        }

        let tx = self.tx();

        if tx.slices_is_empty() {
            return Err(-4)
        }

        if tx.increments_is_full() {
            return Err(-5)
        }

        // SAFETY: user is the sole consumer of tx.slices and sole
        // producer of tx.increments.
        let (offset, len) = unsafe { tx.dequeue_slice() }.ok_or(-4isize)?;
        let buf = unsafe {
            VolSliceMut::from_raw_parts(tx.buf_ptr().add(offset), len)
        };

        let written = f(&buf);
        if written == 0 {
            return Ok(0)
        }

        unsafe {
            let _ = tx.enqueue_increment(written);
        }

        tx.avail.fetch_sub(written, Ordering::AcqRel);

        Ok(written)
    }

    #[cfg(not(feature = "kernel"))]
    pub fn recv_tcp<F>(&self, f: F) -> Result<usize, isize>
        where F: FnOnce(&VolSlice) -> usize
    {
        let current_state = self.current_state();

        let channel_state = current_state.state.load(Ordering::Acquire);
        if channel_state <= 0 {
            return Err(channel_state as isize)
        }

        let rx = self.rx();

        if rx.slices_is_empty() {
            return Err(-4)
        }

        if rx.increments_is_full() {
            return Err(-5)
        }

        // SAFETY: user is the sole consumer of rx.slices and sole
        // producer of rx.increments.
        let (offset, len) = unsafe { rx.dequeue_slice() }.ok_or(-4isize)?;
        let buf = unsafe {
            VolSlice::from_raw_parts(rx.buf_ptr().add(offset), len)
        };

        let written = f(&buf);
        if written == 0 {
            return Ok(0)
        }

        unsafe {
            let _ = rx.enqueue_increment(written);
        }

        rx.avail.fetch_sub(written, Ordering::AcqRel);

        Ok(written)
    }

    #[cfg(not(feature = "kernel"))]
    pub fn connect_tcp(&self, addr: u32, port: u16) -> Result<(), ()> {
        let current_state = self.current_state();
        if current_state.state.load(Ordering::Acquire) != 0 {
            return Err(())
        }

        let desired_state = self.desired_state();

        desired_state.state_remote_port.store(port, Ordering::Relaxed);
        desired_state.state_local_port.store(1337, Ordering::Relaxed);
        desired_state.state_addr.store(addr, Ordering::Relaxed);
        desired_state.state.store(1, Ordering::Release);

        Ok(())
    }

    #[cfg(not(feature = "kernel"))]
    pub fn listen_tcp(&self, port: u16) -> Result<(), ()> {
        let current_state = self.current_state();
        if current_state.state.load(Ordering::Acquire) != 0 {
            return Err(())
        }

        let desired_state = self.desired_state();

        desired_state.state_local_port.store(port, Ordering::Relaxed);
        desired_state.state.store(2, Ordering::Release);

        Ok(())
    }

    /// Begin recycling the NetChannel after the current connection has
    /// ended (normally, via peer FIN, or via a negative error state).
    /// Writes `desired_state.state = 0` — the kernel's update_tcp
    /// observes this, aborts the smoltcp socket, resets its half of the
    /// rings, and releases `current_state.state = 0`.
    ///
    /// This only starts the handshake. Poll `current_state.state` for 0
    /// (with [`wait_reset`]-style backoff of the caller's choosing), then
    /// call [`complete_reset`] before issuing a fresh `listen_tcp` /
    /// `connect_tcp`.
    ///
    /// Returns `Err(())` if the channel is already idle — the caller's
    /// state machine is out of sync.
    #[cfg(not(feature = "kernel"))]
    pub fn request_reset(&self) -> Result<(), ()> {
        if self.current_state().state.load(Ordering::Acquire) == 0 {
            return Err(());
        }
        self.desired_state().state.store(0, Ordering::Release);
        Ok(())
    }

    /// Finish the reset handshake started by [`request_reset`]: reset the
    /// user-owned half of the rings. Caller must have already observed
    /// `current_state.state == 0`; calling it earlier races the kernel's
    /// half of the reset.
    ///
    /// # Safety
    /// Caller must be this NetChannel's sole user-side accessor for the
    /// duration of the call, and must not have any outstanding
    /// `recv_tcp`/`send_tcp` closures executing.
    #[cfg(not(feature = "kernel"))]
    pub unsafe fn complete_reset(&self) {
        unsafe { self.reset_user_side(); }
    }

    #[cfg(feature = "kernel")]
    pub fn rings(&self) -> (&'static mut [u8], &'static mut [u8]) {
        let tx = self.tx();
        let rx = self.rx();
        unsafe {
            (
                core::slice::from_raw_parts_mut(tx.buf_ptr(), tx.capacity),
                core::slice::from_raw_parts_mut(rx.buf_ptr(), rx.capacity),
            )
        }
    }

    #[cfg(not(feature = "kernel"))]
    pub fn readable(&self) -> usize {
        self.rx().avail.load(Ordering::Acquire)
    }

    #[cfg(not(feature = "kernel"))]
    pub fn writeable(&self) -> usize {
        self.tx().avail.load(Ordering::Acquire)
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod vol_slice_tests {
    use super::{VolSlice, VolSliceMut};
    use std::boxed::Box;
    use std::vec::Vec;

    // miri extern — intentional test leaks register themselves as roots
    // so the leak checker accepts them without `-Zmiri-ignore-leaks`.
    #[cfg(miri)]
    unsafe extern "Rust" {
        fn miri_static_root(ptr: *const u8);
    }
    #[cfg(miri)]
    unsafe fn register_root(ptr: *const u8) {
        unsafe { miri_static_root(ptr); }
    }
    #[cfg(not(miri))]
    unsafe fn register_root(_ptr: *const u8) {}

    // Helper: leak a heap buffer of `len` bytes initialized with a
    // ramp (0, 1, 2, ...). Using the heap keeps pointer provenance
    // clean for miri — stack arrays have stricter reborrow rules in
    // some compiler versions.
    fn leaked_ramp(len: usize) -> *mut u8 {
        let v: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
        let boxed = v.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut u8;
        // `Vec::new().into_boxed_slice()` yields a dangling sentinel, not
        // a real allocation; only register real allocations with miri.
        if len > 0 {
            unsafe { register_root(ptr as *const u8); }
        }
        ptr
    }

    // ---- VolSliceMut basics ----

    #[test]
    fn volslice_mut_len_and_is_empty() {
        let ptr = leaked_ramp(10);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 10) };
        assert_eq!(m.len(), 10);
        assert!(!m.is_empty());

        let ptr2 = leaked_ramp(0);
        let empty = unsafe { VolSliceMut::from_raw_parts(ptr2, 0) };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn volslice_mut_copy_from_slice_truncates_to_self_len() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        // Source longer than self → truncates to 4.
        let written = m.copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        assert_eq!(written, 4);
        for i in 0..4 {
            assert_eq!(m.get(i), [0xAA, 0xBB, 0xCC, 0xDD][i]);
        }
    }

    #[test]
    fn volslice_mut_copy_from_slice_truncates_to_src_len() {
        let ptr = leaked_ramp(10);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 10) };
        // Source shorter than self → truncates to 3. Tail bytes untouched.
        let written = m.copy_from_slice(&[1, 2, 3]);
        assert_eq!(written, 3);
        assert_eq!(m.get(0), 1);
        assert_eq!(m.get(1), 2);
        assert_eq!(m.get(2), 3);
        // Index 3 retains the ramp initialization byte.
        assert_eq!(m.get(3), 3);
    }

    #[test]
    fn volslice_mut_set_get_roundtrip() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        m.set(0, 0x11);
        m.set(3, 0x44);
        assert_eq!(m.get(0), 0x11);
        assert_eq!(m.get(3), 0x44);
    }

    #[test]
    fn volslice_mut_checked_variants_return_none_on_oob() {
        let ptr = leaked_ramp(2);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 2) };
        assert!(m.get_checked(2).is_none());
        assert!(m.set_checked(2, 0).is_none());
        assert_eq!(m.set_checked(0, 7), Some(()));
        assert_eq!(m.get_checked(0), Some(7));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn volslice_mut_get_oob_panics() {
        let ptr = leaked_ramp(2);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 2) };
        let _ = m.get(5);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn volslice_mut_set_oob_panics() {
        let ptr = leaked_ramp(2);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 2) };
        m.set(2, 0);
    }

    // ---- VolSliceMut::sub + as_readonly ----

    #[test]
    fn volslice_mut_sub_in_bounds() {
        let ptr = leaked_ramp(8);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 8) };
        let s = m.sub(2, 6);
        assert_eq!(s.len(), 4);
        // Ramp: bytes [0,1,2,3,4,5,6,7]; sub [2..6] = [2,3,4,5].
        for i in 0..4 {
            assert_eq!(s.get(i), (i + 2) as u8);
        }
    }

    #[test]
    #[should_panic(expected = "end")]
    fn volslice_mut_sub_end_oob_panics() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        let _ = m.sub(0, 5);
    }

    #[test]
    #[should_panic(expected = "start")]
    fn volslice_mut_sub_start_after_end_panics() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        let _ = m.sub(3, 2);
    }

    #[test]
    fn volslice_mut_as_readonly_preserves_len() {
        let ptr = leaked_ramp(5);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 5) };
        let ro = m.as_readonly();
        assert_eq!(ro.len(), 5);
        // Ramp byte at index 2 = 2.
        assert_eq!(ro.get(2), 2);
    }

    // ---- VolSlice ----

    #[test]
    fn volslice_copy_to_slice_truncates_to_self_len() {
        let ptr = leaked_ramp(4);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 4) };
        let mut dst = [0u8; 10];
        let n = s.copy_to_slice(&mut dst);
        assert_eq!(n, 4);
        assert_eq!(&dst[..4], &[0, 1, 2, 3]);
        assert_eq!(&dst[4..], &[0u8; 6]); // tail untouched
    }

    #[test]
    fn volslice_copy_to_slice_truncates_to_dst_len() {
        let ptr = leaked_ramp(10);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 10) };
        let mut dst = [0u8; 3];
        let n = s.copy_to_slice(&mut dst);
        assert_eq!(n, 3);
        assert_eq!(dst, [0, 1, 2]);
    }

    #[test]
    fn volslice_checked_get_returns_none_on_oob() {
        let ptr = leaked_ramp(3);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 3) };
        assert_eq!(s.get_checked(0), Some(0));
        assert_eq!(s.get_checked(2), Some(2));
        assert!(s.get_checked(3).is_none());
        assert!(s.get_checked(999).is_none());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn volslice_get_oob_panics() {
        let ptr = leaked_ramp(2);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 2) };
        let _ = s.get(2);
    }
}

// `rings()` is gated to `feature = "kernel"`. The layout tests poke at
// ring byte addresses, so they run only when that feature is enabled —
// i.e. under `cargo test --features kernel`.
#[cfg(all(test, feature = "kernel"))]
mod netchannel_layout_tests {
    //! Tests for `NetChannel::init` + the header/ring layout.
    //!
    //! # Known miri caveat
    //!
    //! `NetChannel`'s accessor methods (`tx`, `rx`, `rings`, `anchor`,
    //! `desired_state`, `current_state`) cast `self: &NetChannel` to
    //! `*const u8` and then `.add(offset)` to reach well past the 8-byte
    //! struct footprint into the surrounding region. That pointer
    //! arithmetic is legal, but the dereference is UB under Stacked
    //! Borrows: the cast-derived ptr has SharedReadOnly provenance for
    //! only `sizeof(NetChannel)` bytes, not the full region.
    //!
    //! In practice the code works — nothing else reborrows against the
    //! narrow region, and the hardware doesn't track provenance — but
    //! strictly speaking it's UB. Fixing it properly would turn
    //! `NetChannel` into a DST `{ queue_len: usize, _rest: [u8] }` or
    //! move all accessors onto raw pointers taking the region base. Out
    //! of scope for the initial test sweep.
    //!
    //! Accessor-path tests are gated off under miri; the raw-access
    //! tests below directly read fields via offsets from the
    //! allocation-rooted base pointer and DO run under miri.

    use super::*;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    struct OwnedRegion {
        base: *mut u8,
        layout: Layout,
    }

    impl OwnedRegion {
        fn new(size: usize) -> Self {
            let layout = Layout::from_size_align(size, 4096).unwrap();
            let base = unsafe { alloc_zeroed(layout) };
            assert!(!base.is_null(), "alloc_zeroed returned null");
            // SAFETY: NetChannel::init reads/writes the whole region.
            unsafe { NetChannel::init(base, size) };
            Self { base, layout }
        }
        fn nc(&self) -> &NetChannel {
            unsafe { &*(self.base as *const NetChannel) }
        }
    }

    impl Drop for OwnedRegion {
        fn drop(&mut self) {
            unsafe { dealloc(self.base, self.layout) };
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // uses NetChannel accessors — see module docs
    fn init_stamps_queue_len_and_capacities() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        assert_eq!(nc.queue_len(), NetChannel::queue_len_for(NC_MIN_REGION_SIZE));
        assert_eq!(nc.tx().capacity(), NetChannel::capacity_for(NC_MIN_REGION_SIZE));
        assert_eq!(nc.rx().capacity(), NetChannel::capacity_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_return_buffers_of_expected_capacity() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let cap = NetChannel::capacity_for(NC_MIN_REGION_SIZE);
        let (tx, rx) = nc.rings();
        assert_eq!(tx.len(), cap);
        assert_eq!(rx.len(), cap);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_tx_and_rx_do_not_overlap() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let (tx, rx) = nc.rings();
        let tx_start = tx.as_ptr() as usize;
        let tx_end = tx_start + tx.len();
        let rx_start = rx.as_ptr() as usize;
        let rx_end = rx_start + rx.len();
        assert!(
            tx_end <= rx_start || rx_end <= tx_start,
            "tx [{tx_start:#x}..{tx_end:#x}] and rx [{rx_start:#x}..{rx_end:#x}] overlap"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_are_within_allocated_region() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let base = region.base as usize;
        let end = base + NC_MIN_REGION_SIZE;
        let (tx, rx) = nc.rings();
        let tx_start = tx.as_ptr() as usize;
        let rx_end = rx.as_ptr() as usize + rx.len();
        assert!(base <= tx_start && rx_end <= end,
            "rings escape region: base={base:#x} end={end:#x} tx_start={tx_start:#x} rx_end={rx_end:#x}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_are_writable_and_readable() {
        // Exercises raw-pointer derefs through the slice handle end to
        // end — miri validates each access against the region's
        // provenance.
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let (tx, rx) = nc.rings();

        tx[0] = 0xAA;
        tx[tx.len() - 1] = 0xBB;
        rx[0] = 0xCC;
        rx[rx.len() - 1] = 0xDD;

        assert_eq!(tx[0], 0xAA);
        assert_eq!(tx[tx.len() - 1], 0xBB);
        assert_eq!(rx[0], 0xCC);
        assert_eq!(rx[rx.len() - 1], 0xDD);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn tx_rx_offsets_match_headers() {
        // tx lives at NC_TX_OFF; rx lives at NC_TX_OFF + queue_len.
        // Verify the fixed-offset accessors against the raw base.
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let base = region.base as usize;
        let tx = nc.tx() as *const _ as usize;
        let rx = nc.rx() as *const _ as usize;
        assert_eq!(tx - base, NC_TX_OFF);
        assert_eq!(rx - base, NC_TX_OFF + nc.queue_len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn desired_and_current_states_sit_at_fixed_offsets() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let base = region.base as usize;
        let desired = nc.desired_state() as *const _ as usize;
        let current = nc.current_state() as *const _ as usize;
        assert_eq!(desired - base, NC_DESIRED_OFF);
        assert_eq!(current - base, NC_CURRENT_OFF);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_len_is_zero_or_more_always() {
        // Grid over several valid region sizes.
        for &sz in &[NC_MIN_REGION_SIZE, 8192, 16384, NC_MAX_REGION_SIZE] {
            let region = OwnedRegion::new(sz);
            let (tx, rx) = region.nc().rings();
            let cap = NetChannel::capacity_for(sz);
            assert_eq!(tx.len(), cap);
            assert_eq!(rx.len(), cap);
        }
    }

    // ---- Raw-access tests ----
    // Read fields via offsets from the allocation-rooted base pointer.
    // Stay miri-clean because provenance comes from `region.base` (the
    // alloc_zeroed return), not from a narrow `&NetChannel` reborrow.

    #[test]
    fn raw_queue_len_written_at_offset_zero() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let written = unsafe { *(region.base as *const usize) };
        assert_eq!(written, NetChannel::queue_len_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    fn raw_tx_capacity_written_at_nc_tx_off_plus_capacity_field() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let cap_field = core::mem::offset_of!(NetChannelQueue, capacity);
        let tx_cap_addr = unsafe { region.base.add(NC_TX_OFF + cap_field) as *const usize };
        let tx_cap = unsafe { *tx_cap_addr };
        assert_eq!(tx_cap, NetChannel::capacity_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    fn raw_rx_capacity_written_at_nc_tx_off_plus_queue_len() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let queue_len = NetChannel::queue_len_for(NC_MIN_REGION_SIZE);
        let cap_field = core::mem::offset_of!(NetChannelQueue, capacity);
        let rx_cap_addr =
            unsafe { region.base.add(NC_TX_OFF + queue_len + cap_field) as *const usize };
        let rx_cap = unsafe { *rx_cap_addr };
        assert_eq!(rx_cap, NetChannel::capacity_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    fn raw_region_is_fully_writable() {
        // Walk every byte. Provenance from one alloc_zeroed covers the
        // entire region — miri validates each access.
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        for off in 0..NC_MIN_REGION_SIZE {
            unsafe { region.base.add(off).write(0xA5); }
        }
        for off in 0..NC_MIN_REGION_SIZE {
            assert_eq!(unsafe { *region.base.add(off) }, 0xA5);
        }
    }
}

#[cfg(test)]
mod region_sizing_tests {
    use super::*;

    // ---- normalize_region_size ----

    #[test]
    fn zero_is_rejected() {
        assert_eq!(NetChannel::normalize_region_size(0), None);
    }

    #[test]
    fn min_requested_returns_min() {
        assert_eq!(
            NetChannel::normalize_region_size(NC_MIN_REGION_SIZE),
            Some(NC_MIN_REGION_SIZE)
        );
    }

    #[test]
    fn max_requested_returns_max() {
        assert_eq!(
            NetChannel::normalize_region_size(NC_MAX_REGION_SIZE),
            Some(NC_MAX_REGION_SIZE)
        );
    }

    #[test]
    fn below_min_clamps_up_to_min() {
        assert_eq!(
            NetChannel::normalize_region_size(1),
            Some(NC_MIN_REGION_SIZE)
        );
        assert_eq!(
            NetChannel::normalize_region_size(NC_MIN_REGION_SIZE - 1),
            Some(NC_MIN_REGION_SIZE)
        );
    }

    #[test]
    fn above_max_clamps_down_to_max() {
        assert_eq!(
            NetChannel::normalize_region_size(NC_MAX_REGION_SIZE + 1),
            Some(NC_MAX_REGION_SIZE)
        );
        assert_eq!(
            NetChannel::normalize_region_size(usize::MAX),
            Some(NC_MAX_REGION_SIZE)
        );
    }

    #[test]
    fn mid_range_rounds_up_to_page() {
        // 8192 + 1 should round to 12288 (still in [min, max])
        assert_eq!(
            NetChannel::normalize_region_size(8193),
            Some(12288)
        );
        // Already page-aligned passes through.
        assert_eq!(
            NetChannel::normalize_region_size(8192),
            Some(8192)
        );
    }

    #[test]
    fn result_is_always_page_aligned() {
        for &req in &[1usize, 100, 4095, 4096, 5000, 8192, 100_000] {
            let r = NetChannel::normalize_region_size(req).unwrap();
            assert_eq!(r % 4096, 0, "normalized size for {req} should be page-aligned, got {r}");
            assert!(r >= NC_MIN_REGION_SIZE && r <= NC_MAX_REGION_SIZE);
        }
    }

    // ---- queue_len_for ----

    #[test]
    fn queue_len_is_nc_queue_aligned() {
        let align = core::mem::align_of::<NetChannelQueue>();
        for &r in &[NC_MIN_REGION_SIZE, 8192, 16384, NC_MAX_REGION_SIZE] {
            assert_eq!(
                NetChannel::queue_len_for(r) % align,
                0,
                "queue_len_for({r}) must be aligned to align_of::<NetChannelQueue>() = {align}"
            );
        }
    }

    #[test]
    fn queue_len_leaves_room_for_both_halves() {
        for &r in &[NC_MIN_REGION_SIZE, 8192, 16384, NC_MAX_REGION_SIZE] {
            let q = NetChannel::queue_len_for(r);
            // tx + rx (each q) + header (NC_TX_OFF) must fit within r.
            assert!(NC_TX_OFF + 2 * q <= r, "tx+rx overrun region at size {r}");
        }
    }

    // ---- capacity_for ----

    #[test]
    fn capacity_equals_queue_len_minus_header() {
        for &r in &[NC_MIN_REGION_SIZE, 8192, NC_MAX_REGION_SIZE] {
            let q = NetChannel::queue_len_for(r);
            let c = NetChannel::capacity_for(r);
            // capacity = queue_len - size_of::<NetChannelQueue>() + 1
            assert_eq!(
                c,
                q - core::mem::size_of::<NetChannelQueue>() + 1,
                "capacity_for({r}) should equal queue_len - header + 1"
            );
        }
    }

    #[test]
    fn capacity_grows_with_region_size() {
        let c_min = NetChannel::capacity_for(NC_MIN_REGION_SIZE);
        let c_max = NetChannel::capacity_for(NC_MAX_REGION_SIZE);
        assert!(c_min < c_max, "larger region → larger per-ring capacity");
    }

    #[test]
    fn min_region_has_positive_capacity() {
        // If this fires, NC_MIN_REGION_SIZE is too small for the header
        // + one usable byte — the channel would be useless at the floor.
        assert!(NetChannel::capacity_for(NC_MIN_REGION_SIZE) > 0);
    }
}
