//! Goldfish RTC — wall-clock source on QEMU's RISC-V `virt` machine.
//!
//! 32-byte MMIO window at PA `0x101000` (`compatible =
//! "google,goldfish-rtc"` in the DTB). Two 32-bit registers carry the
//! current nanosecond count since the UNIX epoch:
//!
//! - `TIME_LOW  = 0x00` — low 32 bits. Reading this register *latches*
//!   the high half so the 64-bit value seen across the pair is
//!   consistent.
//! - `TIME_HIGH = 0x04` — high 32 bits, must be read after `TIME_LOW`.
//!
//! Driver is install-once / read-many. The KMMIO leaf at
//! `kmmio_rtc()` is mapped before [`init`] runs; reads from any hart
//! after that are lock-free volatile loads.

use core::ptr::{read_volatile, with_exposed_provenance};
use core::sync::atomic::{AtomicU64, Ordering};

const TIME_LOW: usize = 0x00;
const TIME_HIGH: usize = 0x04;

static RTC_VA: AtomicU64 = AtomicU64::new(0);

/// Record the kernel-side VA of the RTC's `TIME_LOW` register. Called
/// once on hart 0 from `rust_main` after `map_kernel_self` has placed
/// the KMMIO leaf.
pub fn init(va: u64) {
    RTC_VA.store(va, Ordering::Relaxed);
}

/// Nanoseconds since the UNIX epoch. Returns `0` if the driver hasn't
/// been initialized — in practice the syscall handler is unreachable
/// before init, so this branch only protects out-of-band callers.
pub fn now_nanos() -> u64 {
    let va = RTC_VA.load(Ordering::Relaxed);
    if va == 0 {
        return 0;
    }
    // SAFETY: KMMIO leaf is KRW, single-page, points at the Goldfish
    // RTC PA. The TIME_LOW-then-TIME_HIGH read order is required by
    // the device: TIME_LOW latches the upper half so the 64-bit view
    // is consistent across the two reads.
    unsafe {
        let lo = read_volatile(with_exposed_provenance::<u32>(va as usize + TIME_LOW));
        let hi = read_volatile(with_exposed_provenance::<u32>(va as usize + TIME_HIGH));
        ((hi as u64) << 32) | (lo as u64)
    }
}
