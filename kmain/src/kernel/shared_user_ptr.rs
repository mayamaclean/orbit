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

use mmu::{PAGE_SIZE, SupervisorTag};
use mmu::mmap::{RootTable, walk_to_table};
use mmu::sv48::VirtAddr;
use process::{Frame, Shared};

use crate::kernel::{memmap::FrameToKdmap, pending_frees};

/// Why a revoke walk couldn't complete. All variants leave the PTE(s)
/// untouched — revoke is all-or-nothing per call (though partial progress
/// may be visible for VA ranges spanning multiple leaves if we ever
/// extend this to keep going past a bad leaf; today we stop on the first
/// error).
#[derive(Debug, Clone, Copy)]
pub enum RevokeError {
    /// The walk hit an invalid or superpage PTE above L0 — something is
    /// mapping this VA at a granularity we don't expect. Shared mappings
    /// are always 4 KiB leaves, so this would mean the map path drifted.
    MissingIntermediate(u64),
    /// Leaf PTE is `V=0` or not a leaf. Either the mapping was never
    /// installed or something else already tore it down.
    NotMapped(u64),
    /// Leaf PTE exists but its RSW bits aren't `SharedRevocable`. Either
    /// the mapper forgot to tag a Shared mapping, or something else
    /// stomped this VA with a regular mapping after the fact. Refusing
    /// to clear it protects a legitimate non-shared tenant from being
    /// silently unmapped.
    WrongTag { va: u64, tag: u8 },
}

struct SharedInner<T> {
    /// Strictly `Frame<Shared>` — only Shared-pool backings are legal for
    /// a SharedUserPtr (the kernel dereferences them via KDMAP).
    ///
    /// `Option` because `Drop` needs to move the frame out of `&mut
    /// self` into `pending_frees::push`, and `Frame<P>` isn't `Copy`.
    /// Between `new` and `drop` the slot is always `Some`; `take()`
    /// happens exactly once, in `Drop`.
    frame: Option<Frame<Shared>>,
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
        if let Some(frame) = self.frame.take() {
            pending_frees::push(frame, self.layout);
        }
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
    ///
    /// Panics if `user_va` or `len` aren't 4 KiB-aligned. The revoke
    /// walker assumes one 4 KiB leaf per PAGE_SIZE step through the
    /// range; an unaligned input would walk past the end or start
    /// mid-page. The invariant holds implicitly today because every
    /// current construction path runs through `map_address_range`
    /// (which rejects unaligned VAs) and `normalize_region_size`
    /// (which page-rounds `len`), but this assert makes it explicit
    /// at the construction boundary instead of relying on callers.
    pub fn new(frame: Frame<Shared>, layout: Layout, user_va: u64, len: usize, owner_pid: u16) -> Self {
        assert!(
            user_va % PAGE_SIZE as u64 == 0,
            "SharedUserPtr::new: user_va {:#x} not page-aligned", user_va,
        );
        assert!(
            len % PAGE_SIZE == 0 && len > 0,
            "SharedUserPtr::new: len {len} must be a nonzero multiple of PAGE_SIZE",
        );
        Self {
            inner: Arc::new(SharedInner {
                frame: Some(frame),
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
    /// consult `revoked` — hot-path callers (k_net per-poll) have already
    /// decided they want to keep going regardless. Use [`try_as_ref`] if
    /// you want revocation to fail-closed.
    ///
    /// `self.inner.frame` is `Some` for the whole non-Drop lifetime of
    /// the Arc; the `expect` documents that invariant.
    pub fn as_ref(&self) -> &T {
        let kva = self.inner.frame.as_ref()
            .expect("SharedUserPtr::as_ref after frame.take() in Drop")
            .to_kdmap();
        unsafe { &*kva.as_ptr::<T>() }
    }

    /// `Some(&T)` if the handle is still live, `None` if the user mapping
    /// has been revoked. k_net-style consumers should prefer this over
    /// `as_ref` so a mid-poll revoke turns into a graceful socket
    /// teardown instead of continuing to drive a page the user can no
    /// longer see.
    pub fn try_as_ref(&self) -> Option<&T> {
        if self.is_revoked() {
            return None;
        }
        Some(self.as_ref())
    }

    /// Walk the owner's user PT and invalidate every leaf covering
    /// `[user_va, user_va + len)`. `root` must be the kernel-side
    /// [`RootTable`] built from the owner's satp
    /// (`kernel_root_from_pa(satp.ppn() * 4096)` — Orbit has the helper).
    ///
    /// Order: clear each leaf's V bit and `sfence.vma` before moving on,
    /// and flip `revoked` only after the last PTE is gone. So a concurrent
    /// observer of `is_revoked() == true` is guaranteed the user mapping
    /// is actually unreachable — "revoked" is a post-condition, not a
    /// plan.
    ///
    /// Local-hart sfence only. Cross-hart TLB shootdown is a follow-up;
    /// the rest of orbit's unmap paths have the same limitation today.
    pub fn revoke(&self, root: &RootTable<'_>) -> Result<(), RevokeError> {
        if self.is_revoked() {
            return Ok(());
        }

        let pid = self.inner.owner_pid;
        let start = self.inner.user_va;
        let end = start + self.inner.len as u64;

        let mut va = start;
        while va < end {
            let table = unsafe { walk_to_table(root, VirtAddr::new(va), 0) }
                .ok_or(RevokeError::MissingIntermediate(va))?;
            let idx = VirtAddr::new(va).vpn_n(0) as usize;
            let pte = &table.entries[idx];

            if !pte.is_valid() || !pte.is_leaf() {
                return Err(RevokeError::NotMapped(va));
            }

            let tag = pte.get_supervisor_bits();
            if tag != SupervisorTag::SharedRevocable as u8 {
                return Err(RevokeError::WrongTag { va, tag });
            }

            pte.set_raw(0);
            unsafe { riscv::asm::sfence_vma(pid as usize, va as usize); }

            va += PAGE_SIZE as u64;
        }

        self.inner.revoked.store(true, Ordering::Release);
        Ok(())
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
