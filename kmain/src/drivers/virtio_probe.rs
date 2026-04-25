//! One-shot virtio-mmio discovery shared by every device-specific
//! driver (gpu, input, …).
//!
//! `install_kmmio_alias` bump-allocates a fresh KMMIO VA on every call
//! — it is not idempotent — so we walk the DTB once at boot, alias
//! each `virtio,mmio` slot exactly once, probe its `device_id`, and
//! stash the resulting [`VirtioSlot`] in a leaked `Vec`. Drivers find
//! their device with [`find`].

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use dtoolkit::fdt::Fdt;
use mmu::mmap::{PageAlloc, RootTable};
use tracing::{error, info};
use virtio::discovery::{MmioSlot, for_each_virtio_mmio};
use virtio::mmio::Mmio;

use crate::kernel::memmap::{self, TablePages};

#[derive(Clone, Copy)]
pub struct VirtioSlot {
    pub slot: MmioSlot,
    pub mmio: Mmio,
    pub device_id: u32,
}

static SLOTS: AtomicPtr<Vec<VirtioSlot>> = AtomicPtr::new(null_mut());

/// Walk the DTB, install a KMMIO alias for every `virtio,mmio` node,
/// probe its `device_id`, and stash the result. Subsequent calls are
/// no-ops; `install_kmmio_alias` is not idempotent and a second walk
/// would burn duplicate VA slots.
pub fn discover(
    fdt: &Fdt<'_>,
    rt: &RootTable<'_>,
    table_pages: &mut TablePages,
) {
    if !SLOTS.load(Ordering::Acquire).is_null() {
        return;
    }

    // Phase 1: collect raw slots so we can drop the dtb walker borrow
    // before touching `table_pages` mutably.
    let mut raw: Vec<MmioSlot> = Vec::new();
    for_each_virtio_mmio(fdt, |s| raw.push(s));

    // Phase 2: alias + probe each.
    let mut out: Vec<VirtioSlot> = Vec::with_capacity(raw.len());
    for slot in raw {
        let kva = {
            let mut pa_alloc = PageAlloc::FA(table_pages.frames_mut());
            match unsafe {
                memmap::install_kmmio_alias(
                    rt,
                    &mut pa_alloc,
                    slot.pa_base..slot.pa_base + slot.size,
                )
            } {
                Ok(v) => v,
                Err(_) => {
                    error!("virtio_probe: failed to alias {:#x}", slot.pa_base);
                    continue;
                }
            }
        };
        riscv::asm::sfence_vma(0, 0);
        let mmio = unsafe { Mmio::new(kva) };
        let (magic, device_id) = unsafe { (mmio.magic(), mmio.device_id()) };
        info!(
            "virtio_mmio@{:#x} irq={} magic={:#x} device_id={}",
            slot.pa_base, slot.irq, magic, device_id,
        );
        out.push(VirtioSlot { slot, mmio, device_id });
    }

    let leaked: &'static mut Vec<VirtioSlot> = Box::leak(Box::new(out));
    SLOTS.store(leaked as *mut _, Ordering::Release);
}

/// First aliased slot whose `device_id` matches. Returns `None` if
/// [`discover`] hasn't run yet, or if no such device is on the bus.
pub fn find(device_id: u32) -> Option<VirtioSlot> {
    let p = SLOTS.load(Ordering::Acquire);
    if p.is_null() {
        return None;
    }
    // SAFETY: SLOTS is set exactly once by `discover` from hart 0
    // before any consumer is spawned. Read-only access from any hart
    // thereafter.
    let slots = unsafe { &*p };
    slots.iter().copied().find(|s| s.device_id == device_id)
}
