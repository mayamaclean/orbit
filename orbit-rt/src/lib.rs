//! Orbit user runtime.
//!
//! Foundation crate for `riscv64gc-unknown-orbit` std (roadmap §9). Today
//! it owns one thing: the `#[global_allocator]` every user process runs
//! against. The allocator is talc; heap growth is an Orbit `mmap` syscall
//! issued from talc's `Source` trait when a request can't be satisfied
//! from the existing arenas.
//!
//! # Locking
//!
//! Uses [`TalcSyncCell`] — lock-free, `!Sync` underneath but promoted to
//! `Sync` via an unsafe constructor. Sound because umode is
//! single-threaded today and Orbit has no async signal handlers. Once
//! umode grows threads (roadmap §7), swap for `TalcLock` with a
//! `spinning_top::RawSpinlock`.

#![no_std]

extern crate alloc;

use core::alloc::Layout;
use core::sync::atomic::{AtomicUsize, Ordering};

use orbit_abi::user;
use talc::base::binning::{Binning, DefaultBinning};
use talc::base::Talc;
use talc::cell::{TalcCell, TalcSyncCell};
use talc::source::Source;

// R | W | U bits from mmu::PagePermissions. Kernel expects the raw 5-bit
// encoding today; the orbit-abi::mmap prot/flags redesign isn't wired up.
const PERMS_RW_U: usize = 0x2 | 0x4 | 0x10;

// Heap VAs climb monotonically from here. Above USER_TEXT_BASE
// (0x2_2000_0000) and the umode netchannel-demo hint (0x2_4000_0000) so
// the three regions can't clip. No upper bound — the user half is huge
// and kernel mmap will reject anything out of range.
const HEAP_BASE: usize = 0x3_0000_0000;

// Growth chunk. mmap is a blocking manager round-trip today; fewer,
// larger claims keep that traffic down. Revisit when munmap lands.
const GROWTH_CHUNK: usize = 2 * 1024 * 1024;

static NEXT_VA: AtomicUsize = AtomicUsize::new(HEAP_BASE);

#[derive(Debug)]
pub struct MmapSource;

unsafe impl Source for MmapSource {
    fn acquire<B: Binning>(t: &mut Talc<Self, B>, layout: Layout) -> Result<(), ()> {
        let need = layout.size().max(GROWTH_CHUNK).next_multiple_of(4096);
        let va = NEXT_VA.fetch_add(need, Ordering::Relaxed);

        if unsafe { user::mmap(va, need, PERMS_RW_U, false) }.is_err() {
            return Err(());
        }

        let base = va as *mut u8;
        // SAFETY: kernel just mapped [base, base+need) user-RW for this
        // process. NEXT_VA increases monotonically so the range is
        // disjoint from every prior claim.
        match unsafe { t.claim(base, need) } {
            Some(_) => Ok(()),
            None => Err(()),
        }
    }
}

// SAFETY: TalcSyncCell::new is unsafe because concurrent allocator entry
// from another thread or an async signal handler is UB. umode is
// single-threaded; Orbit has no signals; S-mode trap entry switches satp
// so the kernel cannot re-enter this allocator while a user call is in
// flight. See crate docs.
#[global_allocator]
static ALLOCATOR: TalcSyncCell<MmapSource, DefaultBinning> =
    unsafe { TalcSyncCell::new(TalcCell::new(MmapSource)) };
