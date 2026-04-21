#![no_std]

use core::cell::UnsafeCell;
use core::mem::size_of;
use core::sync::atomic::AtomicUsize;
use core::{net::Ipv4Addr, sync::atomic::{AtomicI32, AtomicU16, AtomicU32, Ordering}};

use mem::round_usize_up;
use smoltcp::{iface::Interface, wire::IpAddress};
use smoltcp::socket::tcp::State as TcpState;

#[cfg(feature = "kernel")]
use tracing::{error, info};

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

#[repr(C, align(128))]
pub struct NetChannelState {
    pub state_addr: AtomicU32,
    pub state: AtomicI32,
    pub state_remote_port: AtomicU16,
    pub state_local_port: AtomicU16
}

type SliceQueue = heapless::spsc::Queue<(usize, usize), 2>;
type IncrementQueue = heapless::spsc::Queue<usize, 2>;

/// Shared producer/consumer queues + payload ring for one direction.
///
/// `slices` and `increments` sit behind `UnsafeCell` because they live in
/// memory shared between kernel and user: each side gets `&self` (not
/// `&mut self`) and reaches through the cell. SPSC correctness — not
/// Rust's borrow checker — guarantees non-aliasing of slots.
#[repr(C)]
pub struct NetChannelQueue {
    slices: UnsafeCell<SliceQueue>,
    increments: UnsafeCell<IncrementQueue>,
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
        unsafe { (*self.slices.get()).enqueue(v) }
    }

    /// # Safety
    /// Caller must be the single consumer for `slices` on this queue.
    pub unsafe fn dequeue_slice(&self) -> Option<(usize, usize)> {
        unsafe { (*self.slices.get()).dequeue() }
    }

    pub fn slices_is_empty(&self) -> bool {
        unsafe { (*self.slices.get()).is_empty() }
    }

    pub fn slices_len(&self) -> usize {
        unsafe { (*self.slices.get()).len() }
    }

    /// # Safety
    /// Caller must be the single producer for `increments` on this queue.
    pub unsafe fn enqueue_increment(&self, v: usize) -> Result<(), usize> {
        unsafe { (*self.increments.get()).enqueue(v) }
    }

    /// # Safety
    /// Caller must be the single consumer for `increments` on this queue.
    pub unsafe fn dequeue_increment(&self) -> Option<usize> {
        unsafe { (*self.increments.get()).dequeue() }
    }

    pub fn increments_is_full(&self) -> bool {
        unsafe { (*self.increments.get()).is_full() }
    }

    pub fn increments_len(&self) -> usize {
        unsafe { (*self.increments.get()).len() }
    }
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
    /// size. Equal to `(region_size - NC_TX_OFF) / 2`.
    pub const fn queue_len_for(region_size: usize) -> usize {
        (region_size - NC_TX_OFF) / 2
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

    pub fn update_tcp(&self, mut iface: Interface, socket: &mut smoltcp::socket::tcp::Socket) -> Interface {
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
            }

            if rx.slices_is_empty() {
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
            }

            if tx.slices_is_empty() {
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
                }
            }
        }
        iface
    }

    pub fn send_tcp<F>(&self, f: F) -> Result<usize, isize>
        where F: FnOnce(&mut [u8]) -> usize
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

        // SAFETY: user is the sole consumer of tx.slices and sole producer
        // of tx.increments.
        let (offset, len) = unsafe { tx.dequeue_slice() }.ok_or(-4isize)?;
        let slice = unsafe {
            core::slice::from_raw_parts_mut(tx.buf_ptr().add(offset), len)
        };

        let written = f(slice);
        if written == 0 {
            return Ok(0)
        }

        tx.avail.fetch_sub(written, Ordering::AcqRel);

        unsafe {
            let _ = tx.enqueue_increment(written);
        }
        Ok(written)
    }

    pub fn recv_tcp<F>(&self, f: F) -> Result<usize, isize>
        where F: FnOnce(&[u8]) -> usize
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

        // SAFETY: user is the sole consumer of rx.slices and sole producer
        // of rx.increments.
        let (offset, len) = unsafe { rx.dequeue_slice() }.ok_or(-4isize)?;
        let slice = unsafe {
            core::slice::from_raw_parts(rx.buf_ptr().add(offset), len)
        };

        let written = f(slice);
        if written == 0 {
            return Ok(0)
        }

        rx.avail.fetch_sub(written, Ordering::AcqRel);

        unsafe {
            let _ = rx.enqueue_increment(written);
        }
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
