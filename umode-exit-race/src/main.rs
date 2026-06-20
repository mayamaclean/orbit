//! Repro for the **close-while-sibling-reads** race (the likely tid47
//! cause). `main` creates a closeable shared region (an eventfd — same
//! `close_handle` → `region.revoke()` path as a NetChannel), spawns a
//! worker that **tight-loops reading it**, then `close_handle`s the fd
//! while the worker is still reading. `run_close_req` revokes (unmaps) the
//! region immediately, so the worker's next load faults — a sibling thread
//! killed by a close it never made.
//!
//! Unlike the exit-ordering race, this is **deterministic**: the close
//! unmaps the VA synchronously, so the spinning worker faults within
//! microseconds — no timing window to lose. Expected kernel log:
//! `tidN killed: bad access cause=13 … stval=<region VA>`.
#![no_std]
#![no_main]

extern crate alloc;

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use orbit_abi::layout::UPROC_SHARED_BASE;
use orbit_abi::serialln;
use orbit_abi::user::{SerialWriter, close_handle, create_thread, eventfd, exit, sleep_ms};
use orbit_rt as _;

/// Region VA, published to the worker (process-private `.bss`).
static SHARED_VA: AtomicUsize = AtomicUsize::new(0);
/// Successful reads the worker has done — freezes the moment it faults.
static WORKER_READS: AtomicU64 = AtomicU64::new(0);

extern "C" fn worker_entry() -> ! {
    loop {
        let va = SHARED_VA.load(Ordering::Acquire);
        if va != 0 {
            let _ = unsafe { core::ptr::read_volatile(va as *const u64) };
            WORKER_READS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    serialln!("close-race: start");

    // eventfd's shared region lives in the shared range; `close_handle`
    // revokes it just like a NetChannel.
    let (va, fd) = match eventfd(UPROC_SHARED_BASE as usize, 0, 0) {
        Ok(p) => p,
        Err(e) => {
            serialln!("close-race: eventfd failed {e:?}");
            exit(1);
        }
    };
    // Confirm it's mapped from main first.
    let _ = unsafe { core::ptr::read_volatile(va as *const u64) };
    SHARED_VA.store(va, Ordering::Release);
    serialln!("close-race: eventfd region @ {va:#x} fd={fd}, spawning reader worker");

    if let Err(e) = create_thread(worker_entry, 0, 1u64 << 1) {
        serialln!("close-race: create_thread failed {e:?}");
        exit(1);
    }

    // Let the worker rack up reads.
    let _ = sleep_ms(30);
    let before = WORKER_READS.load(Ordering::Relaxed);
    serialln!("close-race: worker reads before close = {before}; closing fd={fd} (revokes the region)");

    // The smoking gun: revoke the region out from under the live sibling.
    if let Err(e) = close_handle(fd) {
        serialln!("close-race: close_handle failed {e:?}");
    }

    // Give the worker time to hit the now-unmapped VA.
    let _ = sleep_ms(100);
    let after = WORKER_READS.load(Ordering::Relaxed);
    if after == before {
        serialln!("close-race: worker reads FROZEN at {after} after close — the sibling faulted");
    }
    else {
        serialln!("close-race: worker still reading ({after}) after close — region NOT revoked?");
    }

    exit(0);
}

#[panic_handler]
fn panic(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "close-race panic: {p}");
    w.flush();
    exit(isize::MIN);
}
