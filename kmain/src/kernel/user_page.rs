//! Transient kernel-side window into a `user_pages` backing for setup-time
//! writes (stack zero, ELF copy, anon-mmap zero).
//!
//! `user_pages` has no KDMAP alias in any satp, so the kernel can't deref a
//! user-pool PA directly. `UserPageWindow` installs a leaf PTE at the
//! pre-materialized `KSCRATCH` window in the currently-active satp, lets
//! the caller write through it, then invalidates on drop. The window is
//! single-slot; the Orbit lock serializes access at the call-site level,
//! and the `WINDOW_ACTIVE` flag catches nesting bugs.

use core::sync::atomic::{AtomicBool, Ordering};

use mem::round_usize_up;
use mmu::PAGE_SIZE;
use mmu::PagePermissions;
use mmu::mmap::write_leaf_pte;
use mmu::sv48::{PhysAddr, VirtAddr};

use crate::kernel::memmap;

static WINDOW_ACTIVE: AtomicBool = AtomicBool::new(false);

pub struct UserPageWindow {
    kva: u64,
    len: usize,
    mapped_pages: usize,
}

impl UserPageWindow {
    /// # Safety
    /// - `pa` must be page-aligned and point into `layout.user_pages`.
    /// - `len > 0` and `[pa, pa + len)` must lie within the allocation
    ///   owned by the caller. Sub-page `len` is fine; the window maps the
    ///   pages spanning the request and exposes exactly `len` bytes.
    /// - The caller must hold the Orbit lock (single-slot serialization) and
    ///   must not switch satp for the lifetime of the returned window.
    pub unsafe fn map(pa: u64, len: usize) -> Self {
        assert!(pa as usize % PAGE_SIZE == 0, "UserPageWindow::map: pa not page-aligned");
        assert!(len > 0, "UserPageWindow::map: zero-length window");
        let mapped_len = round_usize_up(len, PAGE_SIZE);
        assert!(
            mapped_len as u64 <= memmap::KSCRATCH_SIZE,
            "UserPageWindow::map: span {mapped_len} exceeds KSCRATCH ({} bytes)",
            memmap::KSCRATCH_SIZE,
        );
        assert!(
            !WINDOW_ACTIVE.swap(true, Ordering::AcqRel),
            "UserPageWindow::map: another window is already active",
        );

        let base = memmap::kscratch_base();
        let perms = (PagePermissions::R as u64)
            | (PagePermissions::W as u64)
            | (PagePermissions::G as u64);
        let root = current_satp_root();
        let mapped_pages = mapped_len / PAGE_SIZE;
        for i in 0..mapped_pages {
            let off = (i * PAGE_SIZE) as u64;
            let vaddr = base + off;
            let paddr = pa + off;
            unsafe {
                write_leaf_pte(
                    &root,
                    VirtAddr::new(vaddr),
                    Some(PhysAddr::new(paddr)),
                    perms,
                ).expect("KSCRATCH intermediate missing — reserve_va_range not run?");
                riscv::asm::sfence_vma(0, vaddr as usize);
            }
        }

        Self { kva: base, len, mapped_pages }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.kva as *mut u8, self.len) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.kva as *mut u8
    }
}

impl Drop for UserPageWindow {
    fn drop(&mut self) {
        let base = memmap::kscratch_base();
        let root = current_satp_root();
        for i in 0..self.mapped_pages {
            let vaddr = base + (i * PAGE_SIZE) as u64;
            unsafe {
                write_leaf_pte(&root, VirtAddr::new(vaddr), None, 0)
                    .expect("KSCRATCH leaf walk failed on drop");
                riscv::asm::sfence_vma(0, vaddr as usize);
            }
        }
        WINDOW_ACTIVE.store(false, Ordering::Release);
    }
}

fn current_satp_root() -> mmu::mmap::RootTable<'static> {
    let satp = riscv::register::satp::read();
    let root_pa = (satp.ppn() * PAGE_SIZE) as u64;
    unsafe { memmap::kernel_root_from_pa(root_pa) }
}
