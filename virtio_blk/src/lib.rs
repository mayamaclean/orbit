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
//! Single-sector reads only. Multi-sector reads chunk at the caller —
//! tarfs in §12c walks the archive a sector at a time, then later FS
//! reads chunk by page.

#![no_std]

pub mod device;
pub mod proto;

pub use device::{ARENA_BYTES, Block, BlockBacking, BlockError, QUEUE_SIZE};
pub use proto::{BlkConfig, BlkReqHeader, SECTOR_SIZE, VIRTIO_BLK_DEVICE_ID};
