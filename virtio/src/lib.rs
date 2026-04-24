//! Shared virtio-mmio transport for Orbit.
//!
//! Device-agnostic layer split three ways:
//! - [`mmio`] — volatile register accessors over the virtio-mmio layout.
//! - [`queue`] — split-ring virtqueue (descriptor table, avail ring, used
//!   ring, free list).
//! - [`transport`] — status handshake and feature negotiation.
//! - [`discovery`] — DTB walker enumerating `virtio,mmio` slots.
//!
//! Device-specific command protocols (gpu/blk/…) live in separate
//! driver crates; this crate only speaks the wire-level transport.

#![no_std]

pub mod discovery;
pub mod mmio;
pub mod queue;
pub mod transport;
