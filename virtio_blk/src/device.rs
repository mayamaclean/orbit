//! virtio-blk device driver. One virtqueue, three-descriptor chains,
//! per-request header+status slots in a fixed arena indexed by
//! descriptor head.
//!
//! Arena layout (one page total):
//! ```text
//! +0x000: [BlkReqHeader; QUEUE_SIZE]      // device reads
//! +0x400: [u8 status; QUEUE_SIZE] (padded)// device writes
//! +0x800: [u8; SECTOR_SIZE] sync_data     // bounce buf for read_blocks_blocking
//! ```
//! Slot[i] is used when a chain's head descriptor index equals `i`. The
//! head index is the only one the driver tracks — the data and status
//! descriptors land on whatever indices the queue's free list hands out
//! and don't index the arena.
//!
//! `submit_read` uses [`virtio::queue::Virtqueue::peek_free_head`] to
//! pre-populate `arena[head].header` *before* `push_chain` publishes
//! the chain to the device. Without the peek the header write would
//! race with the device's read of it.

use core::mem::size_of;

use tracing::{error, info, warn};
use virtio::mmio::Mmio;
use virtio::queue::{Buf, VirtqBacking, Virtqueue};
use virtio::transport::{self, InitError};

use crate::proto::*;

/// Request queue is virtio-blk's queue 0. Read-only configurations
/// have no other queues.
pub const REQUEST_QUEUE: u32 = 0;

/// 64 descriptors → 64 ÷ 3 = 21 in-flight 3-desc chains. Plenty of
/// slack for expected FS load (one outstanding read per parked
/// thread, hart count 4).
pub const QUEUE_SIZE: u16 = 64;

const HEADER_OFFSET: usize = 0;
const HEADER_STRIDE: usize = size_of::<BlkReqHeader>();
const STATUS_OFFSET: usize = QUEUE_SIZE as usize * HEADER_STRIDE;
const STATUS_STRIDE: usize = 1;
// Pad the status table out to 1 KiB so the sync-data buffer lands at a
// 1 KiB boundary; arena layout otherwise has no alignment constraint
// beyond the device-readable header (8-byte aligned, satisfied by the
// page alignment of the arena base).
const STATUS_TABLE_BYTES: usize = (QUEUE_SIZE as usize).next_multiple_of(1024);
const SYNC_DATA_OFFSET: usize = STATUS_OFFSET + STATUS_TABLE_BYTES;

/// Minimum arena size. Caller passes a page (4 KiB) which is more than
/// enough.
pub const ARENA_BYTES: usize = SYNC_DATA_OFFSET + SECTOR_SIZE;

#[derive(Debug)]
pub enum BlockError {
    Init(InitError),
    QueueTooSmall { wanted: u16, got: u32 },
    ArenaTooSmall { needed: usize, got: usize },
    QueueFull,
    BadStatus(u8),
    BadLength { wanted: usize, got: u32 },
    BadBufferLen(usize),
    OutOfRange { lba: u64, capacity: u64 },
    Timeout,
}

impl From<InitError> for BlockError {
    fn from(e: InitError) -> Self { BlockError::Init(e) }
}

pub struct BlockBacking {
    pub mmio: Mmio,
    pub reqq: VirtqBacking,
    pub arena_pa: u64,
    pub arena_kva: *mut u8,
    pub arena_size: usize,
}

pub struct Block {
    mmio: Mmio,
    reqq: Virtqueue,
    arena_pa: u64,
    arena_kva: *mut u8,
    /// Disk size in 512-byte sectors, snapshotted at handshake.
    capacity_sectors: u64,
}

// SAFETY: same constraints as virtio_input::Input — caller-vouched
// uniqueness of the raw pointers; single-consumer post-init (the IRQ
// handler in 12b, hart-pinned to hart 0).
unsafe impl Send for Block {}

impl Block {
    /// Drive the virtio handshake, program the request queue, snapshot
    /// disk capacity, and flip DRIVER_OK.
    ///
    /// # Safety
    /// - `mmio` must alias a live virtio-mmio register region for a
    ///   device whose `device_id == 2`.
    /// - `reqq` backing must be zero-initialized with the spec's
    ///   alignments and exclusive to this queue.
    /// - `arena_pa` / `arena_kva` must cover at least [`ARENA_BYTES`]
    ///   contiguous bytes.
    pub unsafe fn new(backing: BlockBacking) -> Result<Self, BlockError> {
        if backing.arena_size < ARENA_BYTES {
            return Err(BlockError::ArenaTooSmall {
                needed: ARENA_BYTES,
                got: backing.arena_size,
            });
        }

        let mmio = backing.mmio;
        let reqq = unsafe { Virtqueue::new(backing.reqq) };

        unsafe {
            transport::init_device(&mmio, |dev| {
                // Read-only single-sector v1: none of the optional
                // features (RO, BLK_SIZE, FLUSH, …) buy us anything yet.
                // VERSION_1 is the only required bit.
                dev & transport::VIRTIO_F_VERSION_1
            })?;

            mmio.select_queue(REQUEST_QUEUE);
            let qmax = mmio.queue_num_max();
            if qmax < QUEUE_SIZE as u32 {
                return Err(BlockError::QueueTooSmall {
                    wanted: QUEUE_SIZE,
                    got: qmax,
                });
            }
            mmio.set_queue_num(QUEUE_SIZE as u32);
            mmio.set_queue_desc(reqq.desc_pa());
            mmio.set_queue_driver(reqq.avail_pa());
            mmio.set_queue_device(reqq.used_pa());
            mmio.set_queue_ready(1);

            transport::set_driver_ok(&mmio);
        }

        let capacity_sectors = unsafe {
            let cfg = mmio.config_base() as *const BlkConfig;
            (*cfg).capacity
        };
        info!(
            "virtio-blk: device ready (qsize={}, capacity={} sectors = {} MiB)",
            QUEUE_SIZE,
            capacity_sectors,
            (capacity_sectors * SECTOR_SIZE as u64) >> 20,
        );

        Ok(Self {
            mmio,
            reqq,
            arena_pa: backing.arena_pa,
            arena_kva: backing.arena_kva,
            capacity_sectors,
        })
    }

    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    /// Predict the descriptor head that the next [`Self::submit_read`]
    /// will produce. Lets a caller publish per-head bookkeeping (e.g.
    /// the kernel's `IN_FLIGHT[head]` handle slot) *before* the submit
    /// notifies the device — without this, the IRQ-side completion
    /// could see an unregistered slot if the device finishes the chain
    /// faster than the submitter can publish.
    ///
    /// Non-mutating: the head is not consumed until `submit_read`
    /// actually calls `push_chain`. Returns `None` if the request
    /// queue is full.
    pub fn peek_next_head(&self) -> Option<u16> {
        self.reqq.peek_free_head()
    }

    fn header_slot(&self, head: u16) -> (*mut BlkReqHeader, u64) {
        let off = HEADER_OFFSET + head as usize * HEADER_STRIDE;
        let kva = unsafe { self.arena_kva.add(off) } as *mut BlkReqHeader;
        let pa = self.arena_pa + off as u64;
        (kva, pa)
    }

    fn status_slot(&self, head: u16) -> (*mut u8, u64) {
        let off = STATUS_OFFSET + head as usize * STATUS_STRIDE;
        let kva = unsafe { self.arena_kva.add(off) };
        let pa = self.arena_pa + off as u64;
        (kva, pa)
    }

    fn sync_data_slot(&self) -> (*mut u8, u64) {
        let kva = unsafe { self.arena_kva.add(SYNC_DATA_OFFSET) };
        let pa = self.arena_pa + SYNC_DATA_OFFSET as u64;
        (kva, pa)
    }

    /// Submit a single-sector read at `lba`; the device DMAs directly
    /// into `dst_pa`. Returns the descriptor head — caller stashes it
    /// to look up the in-flight request when [`drain_used`] reports
    /// completion.
    ///
    /// # Safety
    /// - `dst_pa` must cover `len` bytes of memory the kernel keeps
    ///   mapped until completion.
    /// - Caller must serialize concurrent submit/drain on the same
    ///   `Block`.
    pub unsafe fn submit_read(
        &mut self,
        lba: u64,
        dst_pa: u64,
        len: u32,
    ) -> Result<u16, BlockError> {
        if len as usize != SECTOR_SIZE {
            return Err(BlockError::BadLength {
                wanted: SECTOR_SIZE,
                got: len,
            });
        }
        if lba >= self.capacity_sectors {
            return Err(BlockError::OutOfRange {
                lba,
                capacity: self.capacity_sectors,
            });
        }

        // Header must be populated *before* push_chain publishes the
        // chain (push_chain inserts a SeqCst fence after descriptor
        // writes, and the device may read the header as soon as the
        // avail.idx bump becomes visible). Peek the upcoming head so
        // we know which arena slot to fill in.
        let predicted = self.reqq.peek_free_head().ok_or(BlockError::QueueFull)?;
        let (hdr_kva, hdr_pa) = self.header_slot(predicted);
        let (status_kva, status_pa) = self.status_slot(predicted);

        unsafe {
            hdr_kva.write_volatile(BlkReqHeader {
                ty: VIRTIO_BLK_T_IN,
                reserved: 0,
                sector: lba,
            });
            // Sentinel != S_OK so we notice if the device returns the
            // chain without writing a real status.
            status_kva.write_volatile(0xff);
        }

        let head = self
            .reqq
            .push_chain(&[
                Buf { pa: hdr_pa, len: HEADER_STRIDE as u32, write: false },
                Buf { pa: dst_pa, len, write: true },
                Buf { pa: status_pa, len: 1, write: true },
            ])
            .map_err(|_| BlockError::QueueFull)?;
        debug_assert!(
            head == predicted,
            "virtio-blk: peek_free_head ({predicted}) != push_chain head ({head})",
        );

        unsafe {
            self.mmio.notify_queue(REQUEST_QUEUE);
        }
        Ok(head)
    }

    /// Drain completed chains, calling `cb(head, status_byte)` for
    /// each. Status byte is the device-written value at
    /// `arena[head].status` — `VIRTIO_BLK_S_OK` on success.
    ///
    /// # Safety
    /// Caller must serialize concurrent drain calls.
    pub unsafe fn drain_used(&mut self, mut cb: impl FnMut(u16, u8)) {
        while let Some((head, _len)) = self.reqq.pop_used() {
            let (status_kva, _) = self.status_slot(head);
            let status = unsafe { status_kva.read_volatile() };
            cb(head, status);
        }
    }

    /// Polled-completion read of one sector at `lba` into `dst`. Used
    /// at mount time only (e.g. tarfs walking the archive header by
    /// header before IRQs are wired). `dst` must be exactly
    /// [`SECTOR_SIZE`] bytes.
    ///
    /// # Safety
    /// Caller must serialize concurrent calls. Mixing with the async
    /// path is only safe when no async submissions are in flight —
    /// mount-time bringup is the canonical safe window.
    pub unsafe fn read_blocks_blocking(
        &mut self,
        lba: u64,
        dst: &mut [u8],
    ) -> Result<(), BlockError> {
        if dst.len() != SECTOR_SIZE {
            return Err(BlockError::BadBufferLen(dst.len()));
        }
        let (sync_kva, sync_pa) = self.sync_data_slot();
        let head = unsafe { self.submit_read(lba, sync_pa, SECTOR_SIZE as u32)? };

        for _ in 0..10_000_000 {
            if let Some((h, _len)) = self.reqq.pop_used() {
                if h != head {
                    error!(
                        "virtio-blk: sync read got unexpected head {h} (expected {head})"
                    );
                    return Err(BlockError::Timeout);
                }
                let (status_kva, _) = self.status_slot(head);
                let status = unsafe { status_kva.read_volatile() };
                if status != VIRTIO_BLK_S_OK {
                    return Err(BlockError::BadStatus(status));
                }
                unsafe {
                    core::ptr::copy_nonoverlapping(sync_kva, dst.as_mut_ptr(), SECTOR_SIZE);
                }
                return Ok(());
            }
            core::hint::spin_loop();
        }

        warn!("virtio-blk: sync read timed out at lba={lba}");
        Err(BlockError::Timeout)
    }

    /// Read + ack the device's interrupt status. Call once per PLIC
    /// claim before [`drain_used`].
    ///
    /// # Safety
    /// MMIO touch — same alias-must-be-live precondition as the rest
    /// of the device API.
    pub unsafe fn ack_interrupt(&self) -> bool {
        let bits = unsafe { transport::ack_interrupts(&self.mmio) };
        bits.used_ring
    }
}
