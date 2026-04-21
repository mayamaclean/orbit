//! Refcounted kernel-side handle on a user-shared `Shared`-pool page.
//!
//! The kernel allocates a page (e.g. a NetChannel) from the `Shared` pool,
//! maps it into the owner's user PT, and wraps it in
//! [`SharedUserPtr<T>`]. The resulting `Arc<SharedInner<T>>` can be
//! cloned into any subsystem that wants to drive the shared object
//! (k_net, console, …); when the last clone drops, the backing is queued
//! onto [`pending_frees`](super::pending_frees) and freed by the manager.
//!
//! Lifetime extension across process teardown is the point: if the
//! process dies while k_net is mid-TCP, the owning Arc in the registry
//! drops but k_net's clone keeps the page alive until k_net's next poll
//! drops it too. No stop-the-world coordination required.

use core::alloc::Layout;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, Ordering};

use alloc::sync::Arc;

use process::{Frame, Shared};

use crate::kernel::{memmap::FrameToKdmap, pending_frees};

struct SharedInner<T> {
    /// Strictly `Frame<Shared>` — only Shared-pool backings are legal for
    /// a SharedUserPtr (the kernel dereferences them via KDMAP).
    frame: Frame<Shared>,
    layout: Layout,
    user_va: u64,
    len: usize,
    owner_pid: u16,
    revoked: AtomicBool,
    _t: PhantomData<T>,
}

impl<T> Drop for SharedInner<T> {
    fn drop(&mut self) {
        // The backing lives in `Shared` pool, so the manager can reach
        // it through KDMAP to hand back to `kernel_pages`. We just
        // enqueue here — no allocator work from drop context.
        pending_frees::push(self.frame, self.layout);
    }
}

/// Kernel handle on a shared user page. Cheaply cloneable; the underlying
/// `Arc` keeps the backing alive until every clone drops.
pub struct SharedUserPtr<T> {
    inner: Arc<SharedInner<T>>,
}

impl<T> Clone for SharedUserPtr<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<T> SharedUserPtr<T> {
    /// Build a handle over a `Shared`-pool frame mapped at `user_va`
    /// spanning `len` bytes in the address space of `owner_pid`. The
    /// `Frame<Shared>` type requirement makes wrong-pool construction a
    /// compile error.
    pub fn new(frame: Frame<Shared>, layout: Layout, user_va: u64, len: usize, owner_pid: u16) -> Self {
        Self {
            inner: Arc::new(SharedInner {
                frame,
                layout,
                user_va,
                len,
                owner_pid,
                revoked: AtomicBool::new(false),
                _t: PhantomData,
            }),
        }
    }

    /// Dereference through the KDMAP alias of the backing. Does *not*
    /// consult `revoked` — callers who need revocation semantics should
    /// check [`is_revoked`](Self::is_revoked) first (revocation is a
    /// follow-up wired to the `supervisor_tag` PTE bit).
    pub fn as_ref(&self) -> &T {
        let kva = self.inner.frame.to_kdmap();
        unsafe { &*kva.as_ptr::<T>() }
    }

    pub fn revoke(&self) {
        self.inner.revoked.store(true, Ordering::Release);
    }

    pub fn is_revoked(&self) -> bool {
        self.inner.revoked.load(Ordering::Acquire)
    }

    pub fn user_va(&self) -> u64 { self.inner.user_va }
    pub fn len(&self) -> usize { self.inner.len }
    pub fn owner_pid(&self) -> u16 { self.inner.owner_pid }
}

impl<T> core::fmt::Debug for SharedUserPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SharedUserPtr")
            .field("user_va", &format_args!("{:#x}", self.inner.user_va))
            .field("len", &self.inner.len)
            .field("owner_pid", &self.inner.owner_pid)
            .field("refs", &Arc::strong_count(&self.inner))
            .field("revoked", &self.inner.revoked.load(Ordering::Acquire))
            .finish()
    }
}
