//! virtio-input driver. Built on the shared [`virtio`] transport.
//!
//! Scope: keyboards (and any device-id 18 input device whose events we
//! can decode as evdev). Single eventq (device → guest). The statusq
//! (guest → device, for keyboard LEDs) is out of scope — we don't drive
//! caps/num/scroll lock yet.
//!
//! The eventq is pre-filled with N empty 8-byte buffers at boot; the
//! device writes [`InputEvent`]s into them and puts each onto the used
//! ring as it does. Consumers call [`Input::pop_event`] to drain one
//! event; the buffer is re-queued in the same call so the eventq stays
//! at full capacity.

#![no_std]

pub mod device;
pub mod proto;

pub use device::{EVENT_QUEUE, EVENT_SIZE, Input, InputBacking, InputError};
pub use proto::{InputEvent, VIRTIO_INPUT_DEVICE_ID};
