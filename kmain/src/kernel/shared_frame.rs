//! Refcounted handle on a kernel-only `Shared`-pool page.
//!
//! Sibling of [`SharedUserPtr`](super::shared_user_ptr::SharedUserPtr)
//! for the case where a Shared-pool backing needs refcount + deferred
//! free *without* the user-mapping concerns (revoke walks, owner pid,
//! `SharedRevocable` PTE tags). The kernel allocates a page from
//! `kernel_pages` and wraps it in [`SharedFrame`]; clones can be
//! handed to in-flight DMA descriptors so close-mid-flight on the
//! originating fd doesn't UAF the page.
//!
//! When the last clone drops, the backing is queued onto
//! [`pending_frees`](super::pending_frees) — same drop-context-safe
//! path that `SharedUserPtr` uses, so callers can drop from anywhere
//! (manager, IRQ-deferred work, …).
//!
//! Today's only consumer is the per-fd scratch sector backing the
//! `fs_read` bounce path; future kernel-side IO buffers (async writes,
//! direct-block reads) will use the same shape.

use core::alloc::Layout;

use alloc::sync::Arc;

use mmu::sv48::PhysAddr;
use process::{Frame, Shared};
use tracing::debug;

use crate::kernel::memmap::{KdmapVa, KernelPages};
use crate::kernel::pending_frees;

struct SharedFrameInner {
    /// `Option` because Drop needs to move the frame out of `&mut
    /// self` into `pending_frees::push`, and `Frame<P>` isn't Copy.
    /// `Some` for the whole non-Drop lifetime of the Arc.
    frame: Option<Frame<Shared>>,
    layout: Layout,
    pa: PhysAddr,
    kva: KdmapVa,
}

impl Drop for SharedFrameInner {
    fn drop(&mut self) {
        if let Some(frame) = self.frame.take() {
            pending_frees::push(frame, self.layout);
        }
    }
}

/// Cheaply cloneable handle on a Shared-pool kernel page. Each clone
/// is one Arc strong-ref; the backing survives until the last clone
/// drops.
pub struct SharedFrame {
    inner: Arc<SharedFrameInner>,
}

impl Clone for SharedFrame {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl SharedFrame {
    /// Allocate a fresh page from `kernel_pages` and wrap it. Returns
    /// `None` if the pool is exhausted.
    pub fn alloc(kernel_pages: &mut KernelPages, layout: Layout) -> Option<Self> {
        let (frame, kva) = kernel_pages.alloc_kdmap(layout)?;
        let pa = PhysAddr::new(frame.get_raw());

        debug!("allocated sharedframe @ {pa:016X?}");

        Some(Self {
            inner: Arc::new(SharedFrameInner {
                frame: Some(frame),
                layout,
                pa,
                kva,
            }),
        })
    }

    /// Physical address of the page — DMA target.
    pub fn pa(&self) -> PhysAddr {
        self.inner.pa
    }

    /// KDMAP alias of the page — kernel-side VA for memcpys.
    pub fn kva(&self) -> KdmapVa {
        self.inner.kva
    }
}
