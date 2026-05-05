//! virtio-blk driver. Read-only sector access over the shared
//! [`virtio`] transport.
//!
//! Single virtqueue (queue 0), three-descriptor chains
//! (header → data → status). Two completion modes:
//!
//! - [`Block::read_blocks_blocking`] — polled-completion read used at
//!   mount time, before IRQs are armed.
//! - [`Block::submit_read`] + [`Block::drain_used`] — IRQ-driven steady
//!   state; the kmain glue arranges completion-handle signalling.
//!
//! Multi-sector reads ride the same chain: one header, one data buffer
//! of `N * SECTOR_SIZE` bytes (`N` up to [`MAX_REQ_BYTES`]/`SECTOR_SIZE`),
//! one status. Caller supplies a physically-contiguous destination of
//! the matching length.

#![no_std]

pub mod device;
pub mod proto;

pub use device::{ARENA_BYTES, Block, BlockBacking, BlockError, QUEUE_SIZE};
pub use proto::{BlkConfig, BlkReqHeader, MAX_REQ_BYTES, SECTOR_SIZE, VIRTIO_BLK_DEVICE_ID};
