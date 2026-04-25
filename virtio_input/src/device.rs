//! virtio-input device driver. Pre-fills the eventq at boot, then
//! drains used buffers on demand from the IRQ handler.
//!
//! Buffer recycling relies on a [`virtio::queue::Virtqueue`] invariant
//! we assert at boot and again on each pop: `pop_used` sets
//! `free_head = head`, so an immediately-following `push_chain` reclaims
//! the same descriptor index. Combined with assigning arena slot `i`
//! to descriptor `i` at boot (push order matches free-list order
//! 0,1,2,…), the desc-index ⇔ arena-slot mapping is stable for the
//! lifetime of the device.

use tracing::{error, info};
use virtio::mmio::Mmio;
use virtio::queue::{Buf, Virtqueue, VirtqBacking};
use virtio::transport::{self, InitError};

use crate::proto::InputEvent;

/// Eventq is virtio-input's queue 0 (device → guest).
pub const EVENT_QUEUE: u32 = 0;

/// Wire size of a single evdev event. Each pre-posted buffer is exactly
/// this large; the device writes one event per descriptor.
pub const EVENT_SIZE: usize = core::mem::size_of::<InputEvent>();

#[derive(Debug)]
pub enum InputError {
    Init(InitError),
    QueueTooSmall { wanted: u16, got: u32 },
    ArenaTooSmall { needed: usize, got: usize },
    PrefillFailed,
}

impl From<InitError> for InputError {
    fn from(e: InitError) -> Self { InputError::Init(e) }
}

/// Backing handed to [`Input::new`]. The arena is one contiguous region
/// the device DMAs events into; caller must size it for at least
/// `eventq.size * EVENT_SIZE` bytes.
pub struct InputBacking {
    pub mmio: Mmio,
    pub eventq: VirtqBacking,
    pub arena_pa: u64,
    pub arena_kva: *mut u8,
    pub arena_size: usize,
}

pub struct Input {
    mmio: Mmio,
    eventq: Virtqueue,
    arena_pa: u64,
    arena_kva: *mut u8,
    queue_size: u16,
}

// SAFETY: Mmio + raw arena pointer alias KMMIO/KDMAP regions whose
// lifetimes outlast the Input. No interior aliasing — Input is
// single-consumer (the IRQ handler).
unsafe impl Send for Input {}

impl Input {
    /// Drive the virtio handshake, program the eventq, pre-fill it with
    /// `queue.size` write-buffers each pointing at a distinct arena
    /// slot, and flip DRIVER_OK. After this returns the device is live
    /// and will start writing events as soon as the host produces them.
    ///
    /// # Safety
    /// - `mmio` must alias a live virtio-mmio region whose `device_id ==
    ///   18`.
    /// - `eventq` backing must be zero-initialized with the spec's
    ///   alignments and exclusive to this queue.
    /// - `arena_pa` / `arena_kva` must cover at least
    ///   `eventq.size * EVENT_SIZE` contiguous bytes.
    pub unsafe fn new(backing: InputBacking) -> Result<Self, InputError> {
        let queue_size = backing.eventq.size;
        let needed = queue_size as usize * EVENT_SIZE;
        if backing.arena_size < needed {
            return Err(InputError::ArenaTooSmall { needed, got: backing.arena_size });
        }

        let mmio = backing.mmio;
        let mut eventq = unsafe { Virtqueue::new(backing.eventq) };

        unsafe {
            transport::init_device(&mmio, |dev| {
                // No optional features. virtio-input doesn't define
                // anything we'd want beyond VERSION_1.
                dev & transport::VIRTIO_F_VERSION_1
            })?;

            mmio.select_queue(EVENT_QUEUE);
            let qmax = mmio.queue_num_max();
            if qmax < queue_size as u32 {
                return Err(InputError::QueueTooSmall { wanted: queue_size, got: qmax });
            }
            mmio.set_queue_num(queue_size as u32);
            mmio.set_queue_desc(eventq.desc_pa());
            mmio.set_queue_driver(eventq.avail_pa());
            mmio.set_queue_device(eventq.used_pa());
            mmio.set_queue_ready(1);
        }

        // Pre-fill: push N single-buffer write descriptors. Because the
        // virtqueue's free-list is seeded 0→1→2→…, the i-th push pops
        // descriptor index i — so arena slot i is permanently bound to
        // descriptor i. We assert that to catch any future change to
        // Virtqueue::new's seeding.
        for i in 0..queue_size {
            let pa = backing.arena_pa + (i as u64) * EVENT_SIZE as u64;
            let head = eventq
                .push_chain(&[Buf { pa, len: EVENT_SIZE as u32, write: true }])
                .map_err(|_| InputError::PrefillFailed)?;
            assert!(
                head == i,
                "virtio-input: prefill desc index {head} != arena slot {i}",
            );
        }

        unsafe {
            mmio.notify_queue(EVENT_QUEUE);
            transport::set_driver_ok(&mmio);
        }

        info!("virtio-input: device ready (qsize={})", queue_size);

        Ok(Self {
            mmio,
            eventq,
            arena_pa: backing.arena_pa,
            arena_kva: backing.arena_kva,
            queue_size,
        })
    }

    /// Drain one event off the used ring and re-queue its buffer. Returns
    /// `None` when the ring is empty.
    ///
    /// # Safety
    /// Caller must serialize concurrent `pop_event` calls — the
    /// underlying Virtqueue is not thread-safe. Today the only caller
    /// is the PLIC handler, which runs single-hart.
    pub unsafe fn pop_event(&mut self) -> Option<InputEvent> {
        let (head, _len) = self.eventq.pop_used()?;

        // Read the event before re-queueing — once the descriptor is
        // back on the avail ring the device may overwrite the slot.
        let slot = unsafe {
            self.arena_kva
                .add(head as usize * EVENT_SIZE)
                .cast::<InputEvent>()
                .read_volatile()
        };

        // Re-queue the same arena slot. The Virtqueue invariant
        // (free_head == head right after pop_used) means push_chain
        // hands back the same descriptor index — assert in case the
        // queue impl ever changes.
        let pa = self.arena_pa + (head as u64) * EVENT_SIZE as u64;
        match self.eventq.push_chain(&[Buf { pa, len: EVENT_SIZE as u32, write: true }]) {
            Ok(reused) => debug_assert!(
                reused == head,
                "virtio-input: re-queue head drift {head} → {reused}",
            ),
            Err(_) => {
                // Should never happen — we just freed a slot.
                error!("virtio-input: re-queue push_chain returned Full");
            }
        }
        unsafe { self.mmio.notify_queue(EVENT_QUEUE); }

        Some(slot)
    }

    /// Read + ack the device's interrupt status. Call once per PLIC
    /// claim before draining events.
    ///
    /// # Safety
    /// MMIO touch — same alias-must-be-live precondition as the rest
    /// of the device API.
    pub unsafe fn ack_interrupt(&self) -> bool {
        let bits = unsafe { transport::ack_interrupts(&self.mmio) };
        bits.used_ring
    }

    pub fn queue_size(&self) -> u16 { self.queue_size }
}
