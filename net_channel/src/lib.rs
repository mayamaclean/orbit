#![no_std]

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::size_of;
use core::sync::atomic::AtomicUsize;
use core::{net::Ipv4Addr, sync::atomic::{AtomicI32, AtomicU16, AtomicU32, Ordering}};

use mem::round_usize_up;
use smoltcp::{iface::Interface, wire::IpAddress};
use smoltcp::socket::tcp::State as TcpState;

#[cfg(feature = "kernel")]
use tracing::{error, info};

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

/// Fixed-layout single-producer / single-consumer queue. Baked here
/// (rather than reusing `heapless::spsc`) because:
///
/// 1. The layout is part of the user/kernel NetChannel ABI. `heapless`'s
///    `Queue` is plain `repr(Rust)` — field ordering isn't guaranteed
///    stable across rustc versions or compilation flags. `#[repr(C)]`
///    here pins it.
/// 2. Slot reads/writes go through `read_volatile` / `write_volatile`.
///    Defensive against any alias-analysis-driven DCE of the slot write
///    (the bug we hit that sent us down this path).
///
/// Capacity is `N - 1` — one slot is always reserved so `head == tail`
/// unambiguously means empty.
#[repr(C)]
pub struct SpscQueue<T: Copy, const N: usize> {
    /// Consumer-owned: index of next slot to dequeue from.
    head: AtomicUsize,
    /// Producer-owned: index of next slot to enqueue into.
    tail: AtomicUsize,
    /// Backing ring storage. Raw `UnsafeCell<T>` so the slots can be
    /// written under `&self` via `.get()` → `*mut T`. Zero-init is a
    /// valid starting state since `head == tail == 0` marks empty and no
    /// slot is observed before being written.
    buffer: [UnsafeCell<T>; N],
}

// Producer and consumer are on different harts / threads; heads/tails
// are atomic, slots are synchronized via release/acquire.
unsafe impl<T: Copy + Send, const N: usize> Sync for SpscQueue<T, N> {}

impl<T: Copy, const N: usize> SpscQueue<T, N> {
    /// # Safety
    /// Caller must be the sole producer on this queue.
    #[inline]
    pub unsafe fn enqueue(&self, val: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next = (tail + 1) % N;
        if next == self.head.load(Ordering::Acquire) {
            return Err(val);
        }
        unsafe { self.buffer[tail].get().write_volatile(val); }
        self.tail.store(next, Ordering::Release);
        Ok(())
    }

    /// # Safety
    /// Caller must be the sole consumer on this queue.
    #[inline]
    pub unsafe fn dequeue(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        if head == self.tail.load(Ordering::Acquire) {
            return None;
        }
        let val = unsafe { self.buffer[head].get().read_volatile() };
        self.head.store((head + 1) % N, Ordering::Release);
        Some(val)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail + 1) % N == head
    }

    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail + N - head) % N
    }
}

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

    /// Pump one poll cycle against `socket`. `pending_rx_ack` /
    /// `pending_tx_ack` are kernel-local state (held per socket in
    /// `SocketReq`): each is "a slice is enqueued and we haven't drained
    /// the matching increment yet." They gate re-enqueue so we don't
    /// double-post a slice in the window between the user's
    /// `dequeue_slice` and their `enqueue_increment`. Without them,
    /// `get_next_rx` / `get_next_tx` returns the same bytes (smoltcp
    /// hasn't been advanced yet) and we'd stage a duplicate that
    /// deadlocks the queue once the real ack drains.
    pub fn update_tcp(
        &self,
        mut iface: Interface,
        socket: &mut smoltcp::socket::tcp::Socket,
        pending_rx_ack: &mut bool,
        pending_tx_ack: &mut bool,
    ) -> Interface {
        let current_state = self.current_state();

        let channel_state = current_state.state.load(Ordering::Relaxed);
        if channel_state < 0 {
            return iface
        }
        else if channel_state == 0 {
            let desired_state = self.desired_state();

            let port = desired_state.state_remote_port.load(Ordering::Acquire);
            let addr = desired_state.state_addr.load(Ordering::Acquire);
            let state = desired_state.state.load(Ordering::Acquire);

            if !socket.get_timeout_status() && socket.state() == TcpState::Closed {
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
                return iface
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
                        let increments_len = rx.increments_len();
                        let avail_len = rx.slices_len();

                        let slice = unsafe {
                            core::slice::from_raw_parts(rx.buf_ptr(), next_rx.1)
                        };

                        info!("tcp: next_rx={slice:02X?}, increments_len={increments_len}, avail_len={avail_len}");
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
                    let r = unsafe { tx.enqueue_slice(next_tx) };

                    #[cfg(feature = "kernel")]
                    {
                        let increments_len = tx.increments_len();
                        let avail_len = tx.slices_len();
                        info!("tcp: next_tx={next_tx:08X?}, increments_len={increments_len}, avail_len={avail_len} r={r:08X?}");
                    }

                    core::sync::atomic::fence(Ordering::SeqCst);
                    tx.avail.store(next_tx.1, Ordering::Release);
                    *pending_tx_ack = true;
                }
            }
        }
        iface
    }

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

    pub fn readable(&self) -> usize {
        self.rx().avail.load(Ordering::Acquire)
    }

    pub fn writeable(&self) -> usize {
        self.tx().avail.load(Ordering::Acquire)
    }
}
