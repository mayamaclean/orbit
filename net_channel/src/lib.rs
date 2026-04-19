#![no_std]

use core::mem::size_of;
use core::sync::atomic::AtomicUsize;
use core::{net::Ipv4Addr, ptr::NonNull, sync::atomic::{AtomicI32, AtomicU16, AtomicU32, Ordering}};

use mem::round_usize_down;
use smoltcp::{iface::Interface, wire::IpAddress};
use smoltcp::socket::tcp::State as TcpState;

#[cfg(feature = "kernel")]
use tracing::{error, info};

#[repr(C, align(128))]
pub struct NetChannelState {
    pub state_addr: AtomicU32,
    pub state: AtomicI32,
    pub state_remote_port: AtomicU16,
    pub state_local_port: AtomicU16
}

type SliceQueue = heapless::spsc::Queue<(usize, usize), 2>;
type IncrementQueue = heapless::spsc::Queue<usize, 2>;

#[derive(Debug)]
#[repr(C)]
pub struct NetChannelQueue {
    slices: SliceQueue,
    increments: IncrementQueue,
    pub avail: AtomicUsize,
    capacity: usize,
    buf: u8,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NetChannel {
    pub desired_state: NonNull<NetChannelState>, // 8
    pub current_state: NonNull<NetChannelState>, // 16
    pub tx: NonNull<NetChannelQueue>, // 24
    pub rx: NonNull<NetChannelQueue>,  // 32
    
    // inline desired @ 128
    // inline current @ 256
    // end of states  @ 384
}

impl NetChannel {
    pub fn new(base: *mut u8, len: usize) -> Option<*mut Self> {
        if len < 4096 {
            return None
        }

        let addr = base as usize;
        if (addr % 128) != 0 {
            return None
        }

        let queue_len = round_usize_down((len - 384) / 2, 8);
        let buf_len = queue_len - size_of::<NetChannelQueue>() + 1;

        if buf_len == 0 {
            return None
        }

        let ptr = base as *mut Self;
        unsafe {
            let tx_ptr = base.add(384) as *mut NetChannelQueue;
            tx_ptr.as_mut_unchecked().capacity = buf_len;

            let rx_ptr = (tx_ptr as *mut u8).add(queue_len) as *mut NetChannelQueue;
            rx_ptr.as_mut_unchecked().capacity = buf_len;

            let nch = ptr.as_mut_unchecked();
            nch.tx = NonNull::new_unchecked(tx_ptr);
            nch.rx = NonNull::new_unchecked(rx_ptr);

            nch.desired_state = NonNull::new_unchecked(base.add(128) as *mut NetChannelState);
            nch.current_state = NonNull::new_unchecked(base.add(256) as *mut NetChannelState);
        }
        Some(ptr)
    }

    pub fn update_tcp(&mut self, mut iface: Interface, socket: &mut smoltcp::socket::tcp::Socket) -> Interface {
        let current_state = unsafe { self.current_state.as_ref() };

        let channel_state = current_state.state.load(Ordering::Relaxed);
        if channel_state < 0 {
            return iface
        }
        else if channel_state == 0 {
            let desired_state = unsafe { self.desired_state.as_ref() };

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

        let (tx, rx) = unsafe {
            (self.tx.as_mut(), self.rx.as_mut())
        };

        if socket.may_recv() {
            while let Some(user_rx_count) = rx.increments.dequeue() {
                if let Err(e) = socket.recv(|_b| (user_rx_count, user_rx_count)) {
                    #[cfg(feature = "kernel")]
                    error!("tcp: failed recv: {e:?}");
                    current_state.state.store(-2, Ordering::Release);

                    return iface
                }
            }

            if rx.slices.is_empty() {
                let next_rx = socket.get_next_rx();
                if next_rx.1 > 0 {
                    let _r = rx.slices.enqueue(next_rx);

                    #[cfg(feature = "kernel")]
                    {
                        let increments_len = rx.increments.len();
                        let avail_len = rx.slices.len();

                        let ptr = &rx.buf as *const u8;
                        let slice = unsafe {
                            core::slice::from_raw_parts(ptr, next_rx.1)
                        };

                        info!("tcp: next_rx={slice:02X?}, increments_len={increments_len}, avail_len={avail_len}");
                    }

                    core::sync::atomic::fence(Ordering::SeqCst);
                    rx.avail.store(next_rx.1, Ordering::Release);
                }
            }
        }

        if socket.may_send() {
            while let Some(user_tx_count) = tx.increments.dequeue() {
                if let Err(e) = socket.send(|_b| (user_tx_count, user_tx_count)) {
                    #[cfg(feature = "kernel")]
                    error!("tcp: failed send: {e:?}");
                    current_state.state.store(-3, Ordering::Release);

                    return iface
                }
            }

            if tx.slices.is_empty() {
                let next_tx = socket.get_next_tx();
                if next_tx.1 > 0 {
                    let r = tx.slices.enqueue(next_tx);

                    #[cfg(feature = "kernel")]
                    {
                        let increments_len = tx.increments.len();
                        let avail_len = tx.slices.len();
                        info!("tcp: next_tx={next_tx:08X?}, increments_len={increments_len}, avail_len={avail_len} r={r:08X?}");
                    }

                    core::sync::atomic::fence(Ordering::SeqCst);
                    tx.avail.store(next_tx.1, Ordering::Release);
                }
            }            
        }
        iface
    }

    pub fn send_tcp<F>(&mut self, f: F) -> Result<usize, isize>
        where F: FnOnce(&mut [u8]) -> usize
    {
        let current_state = unsafe {
            self.current_state.as_ref()
        };

        let channel_state = current_state.state.load(Ordering::Acquire);
        if channel_state <= 0 {
            return Err(channel_state as isize)
        }

        let tx = unsafe {
            self.tx.as_mut()
        };

        if tx.slices.is_empty() {
            return Err(-4)
        }

        if tx.increments.is_full() {
            return Err(-5)
        }

        let base = unsafe {
            &mut self.tx.as_mut().buf as *mut u8
        };

        let slice = unsafe {
            let (offset, len) = tx.slices.dequeue_unchecked();
            core::slice::from_raw_parts_mut(base.add(offset), len)
        };

        let written = f(slice);
        if written == 0 {
            return Ok(0)
        }

        tx.avail.fetch_sub(written, Ordering::AcqRel);

        unsafe {
            tx.increments.enqueue_unchecked(written);
        }
        Ok(written)
    }

    pub fn recv_tcp<F>(&mut self, f: F) -> Result<usize, isize>
        where F: FnOnce(&[u8]) -> usize
    {
        let current_state = unsafe {
            self.current_state.as_ref()
        };

        let channel_state = current_state.state.load(Ordering::Acquire);
        if channel_state <= 0 {
            return Err(channel_state as isize)
        }

        let rx = unsafe {
            self.rx.as_mut()
        };

        if rx.slices.is_empty() {
            return Err(-4)
        }

        if rx.increments.is_full() {
            return Err(-5)
        }

        let base = unsafe {
            &self.rx.as_ref().buf as *const u8
        };

        let slice = unsafe {
            let (offset, len) = rx.slices.dequeue_unchecked();
            core::slice::from_raw_parts(base.add(offset), len)
        };

        let written = f(slice);
        if written == 0 {
            return Ok(0)
        }

        rx.avail.fetch_sub(written, Ordering::AcqRel);

        unsafe {
            rx.increments.enqueue_unchecked(written);
        }
        Ok(written)
    }

    pub fn connect_tcp(&self, addr: u32, port: u16) -> Result<(), ()> {
        let current_state = unsafe { self.current_state.as_ref() };
        if current_state.state.load(Ordering::Acquire) != 0 {
            return Err(())
        }
        
        let desired_state = unsafe {
            self.desired_state.as_ref()
        };

        desired_state.state_remote_port.store(port, Ordering::Relaxed);
        desired_state.state_local_port.store(1337, Ordering::Relaxed);
        desired_state.state_addr.store(addr, Ordering::Relaxed);
        desired_state.state.store(1, Ordering::Release);

        Ok(())
    }

    pub fn rx_ring(&mut self) -> &mut [u8] {
        unsafe {
            let ptr = (&mut self.rx.as_mut().buf) as *mut _;
            let len = self.rx.as_ref().capacity;

            core::slice::from_raw_parts_mut(ptr, len)
        }
    }

    pub fn tx_ring(&mut self) -> &mut [u8] {
        unsafe {
            let ptr = (&mut self.tx.as_mut().buf) as *mut _;
            let len = self.tx.as_ref().capacity;

            core::slice::from_raw_parts_mut(ptr, len)
        }
    }

    pub fn rings(&mut self) -> (&'static mut [u8], &'static mut [u8]) {
        unsafe {
            let tx_ptr = (&mut self.tx.as_mut().buf) as *mut _;
            let tx_len = self.tx.as_ref().capacity;
            let rx_ptr = (&mut self.rx.as_mut().buf) as *mut _;
            let rx_len = self.rx.as_ref().capacity;

            (
                core::slice::from_raw_parts_mut(tx_ptr, tx_len),
                core::slice::from_raw_parts_mut(rx_ptr, rx_len)
            )
        }
    }

    pub fn readable(&self) -> usize {
        unsafe {
            self.rx.as_ref().avail.load(Ordering::Acquire)
        }
    }

    pub fn writeable(&self) -> usize {
        unsafe {
            self.tx.as_ref().avail.load(Ordering::Acquire)
        }
    }
}
