//! DTB walker for `virtio,mmio` nodes.
//!
//! The `virt` machine exposes 8 slots at `0x10001000..=0x10008000` with
//! IRQs 1..8. Each slot is a generic virtio-mmio stub; the actual
//! device-id is read from the MMIO register at `+0x08` after the slot
//! has been KMMIO-aliased.

use dtoolkit::fdt::{Fdt, FdtNode};
use dtoolkit::{Node, Property};

#[derive(Debug, Clone, Copy)]
pub struct MmioSlot {
    pub pa_base: u64,
    pub size: u64,
    /// The IRQ source number on the device's interrupt-parent
    /// (typically the PLIC). Use directly with
    /// `plic_register(irq, handler, hart)`.
    pub irq: u32,
}

fn is_virtio_mmio(n: &FdtNode<'_>) -> bool {
    let Some(compat) = n.property("compatible") else { return false };
    compat.as_str_list().any(|s| s == "virtio,mmio")
}

fn walk<'a, F: FnMut(MmioSlot)>(n: FdtNode<'a>, f: &mut F) {
    if is_virtio_mmio(&n) {
        if let Some(slot) = slot_from(&n) {
            f(slot);
        }
    }
    for c in n.children() {
        walk(c, f);
    }
}

fn slot_from(n: &FdtNode<'_>) -> Option<MmioSlot> {
    let mut regs = n.reg().ok()??;
    let reg = regs.next()?;
    let pa_base = reg.address::<u64>().ok()?;
    let size = reg.size::<u64>().ok()?;
    let irq = n
        .property("interrupts")?
        .as_u32()
        .ok()?;
    Some(MmioSlot { pa_base, size, irq })
}

/// Visit every `virtio,mmio` slot in the DTB.
pub fn for_each_virtio_mmio<F: FnMut(MmioSlot)>(fdt: &Fdt<'_>, mut f: F) {
    walk(fdt.root(), &mut f);
}
