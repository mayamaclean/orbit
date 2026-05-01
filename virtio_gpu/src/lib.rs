//! virtio-gpu 2D driver. Built on the shared [`virtio`] transport.
//!
//! Scope: single scanout, single 2D resource, control queue only. No
//! cursor queue, no 3D. The commands we emit are `GET_DISPLAY_INFO`,
//! `RESOURCE_CREATE_2D`, `RESOURCE_ATTACH_BACKING`, `SET_SCANOUT`,
//! `TRANSFER_TO_HOST_2D`, and `RESOURCE_FLUSH` — enough for an
//! in-kernel compositor to blit a framebuffer every frame.

#![no_std]

pub mod device;
pub mod proto;

pub use device::{ARENA_SIZE, DisplayInfo, Gpu, GpuBacking, GpuError};
pub use proto::FORMAT_B8G8R8A8_UNORM;
