//! kmain-side glue for virtio-input: pick the device-id 18 slot off
//! [`virtio_probe`], allocate eventq + event arena from `kernel_pages`,
//! drive `Input::new` to bring the device live, register a PLIC handler
//! that drains used buffers and forwards each [`InputEvent`] to
//! [`crate::kernel::input::dispatch`].

use alloc::boxed::Box;
use core::alloc::Layout;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use tracing::{error, info};
use virtio::queue::VirtqBacking;
use virtio_input::{EVENT_SIZE, Input, InputBacking, VIRTIO_INPUT_DEVICE_ID};

use crate::drivers::{plic, virtio_probe};
use crate::kernel::input;
use crate::kernel::memmap::KernelPages;

// Queue sizing: 32 entries fit comfortably in a single page alongside
// the avail/used rings, and 32 in-flight buffers is plenty of slack for
// human-rate keyboard input. Layout (mirrors virtio_gpu_dev):
//   desc  = 32 × 16 = 512 B    → offset 0
//   avail = 4 + 32×2 + 2 = 70  → offset 1024 (1 KiB slot)
//   used  = 4 + 32×8 + 2 = 262 → offset 2048 (2 KiB slot)
pub const QUEUE_SIZE: u16 = 32;
pub const QUEUE_PAGE_SIZE: usize = 4096;
const DESC_OFFSET: u64 = 0;
const AVAIL_OFFSET: u64 = 1024;
const USED_OFFSET: u64 = 2048;

/// Arena = one event slot per descriptor. 32 × 8 = 256 B; rounded up
/// to a page for `kernel_pages::alloc_kdmap`.
const ARENA_BYTES: usize = QUEUE_SIZE as usize * EVENT_SIZE;

static INPUT_PTR: AtomicPtr<Input> = AtomicPtr::new(null_mut());

/// Access the installed input driver. Returns `None` until
/// [`setup_virtio_input`] has completed successfully.
fn input_dev() -> Option<&'static mut Input> {
    let p = INPUT_PTR.load(Ordering::Acquire);
    if p.is_null() {
        None
    }
    else {
        // SAFETY: INPUT_PTR is written exactly once from hart 0 during
        // setup_virtio_input. Single consumer post-init: the PLIC
        // handler, which is hart-pinned to hart 0.
        Some(unsafe { &mut *p })
    }
}

/// Discover the slot, install Input, and arm the IRQ. Returns true on
/// success. Requires [`virtio_probe::discover`] to have run first.
pub fn setup_virtio_input(kernel_pages: &mut KernelPages) -> bool {
    let Some(found) = virtio_probe::find(VIRTIO_INPUT_DEVICE_ID)
    else {
        info!("virtio-input: no device-id 18 slot present");
        return false;
    };
    let slot = found.slot;
    let mmio = found.mmio;
    info!(
        "virtio-input: selected slot @{:#x} irq={}",
        slot.pa_base, slot.irq
    );

    let queue_layout = match Layout::from_size_align(QUEUE_PAGE_SIZE, QUEUE_PAGE_SIZE) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let Some((q_frame, q_kva)) = kernel_pages.alloc_kdmap(queue_layout)
    else {
        error!("virtio-input: queue page alloc failed");
        return false;
    };
    let q_pa = q_frame.get_raw();
    unsafe {
        core::ptr::write_bytes(q_kva.as_mut_ptr::<u8>(), 0, QUEUE_PAGE_SIZE);
    }

    // Arena: one page (way more than 256 B needed) for the event slots.
    let arena_layout = match Layout::from_size_align(QUEUE_PAGE_SIZE, 4096) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let Some((arena_frame, arena_kva)) = kernel_pages.alloc_kdmap(arena_layout)
    else {
        error!("virtio-input: arena alloc failed");
        return false;
    };
    let arena_pa = arena_frame.get_raw();
    unsafe {
        core::ptr::write_bytes(arena_kva.as_mut_ptr::<u8>(), 0, QUEUE_PAGE_SIZE);
    }

    let q_kva_u64 = q_kva.as_mut_ptr::<u8>() as u64;
    let backing = InputBacking {
        mmio,
        eventq: VirtqBacking {
            desc_pa: q_pa + DESC_OFFSET,
            desc_kva: (q_kva_u64 + DESC_OFFSET) as *mut u8,
            avail_pa: q_pa + AVAIL_OFFSET,
            avail_kva: (q_kva_u64 + AVAIL_OFFSET) as *mut u8,
            used_pa: q_pa + USED_OFFSET,
            used_kva: (q_kva_u64 + USED_OFFSET) as *mut u8,
            size: QUEUE_SIZE,
        },
        arena_pa,
        arena_kva: arena_kva.as_mut_ptr::<u8>(),
        arena_size: ARENA_BYTES,
    };

    let dev = match unsafe { Input::new(backing) } {
        Ok(d) => d,
        Err(e) => {
            error!("virtio-input: init failed: {:?}", e);
            return false;
        }
    };

    let leaked: &'static mut Input = Box::leak(Box::new(dev));
    INPUT_PTR.store(leaked as *mut _, Ordering::Release);

    // Arm the IRQ. Pin to hart 0 (same as UART RX); rates are low enough
    // that load distribution doesn't matter.
    if plic::plic_register(slot.irq, virtio_input_handler, 0).is_err() {
        error!("virtio-input: plic_register failed for irq {}", slot.irq);
        return false;
    }

    info!("virtio-input: device live, irq {} armed", slot.irq);
    true
}

/// PLIC handler. Runs in trap context with SIE=0 on hart 0's tsp.
/// Acks the device interrupt, drains every used event-buffer (a burst
/// of keystrokes can deliver several per IRQ), and dispatches each
/// event through [`input::dispatch`]. Buffer re-queue happens inside
/// `Input::pop_event`.
fn virtio_input_handler(_src: u32) {
    let Some(dev) = input_dev()
    else {
        return;
    };
    unsafe {
        // Ack the interrupt status before draining so a fresh event
        // arriving mid-drain re-asserts the IRQ line and we'll see it
        // on the next claim.
        let _used = dev.ack_interrupt();
        while let Some(ev) = dev.pop_event() {
            input::dispatch(ev);
        }
    }
}
