//! `/bin/hello` — minimal user binary that proves §12e exec works
//! end-to-end. Runs out of the tarfs-served disk image; the console
//! `exec` builtin (or the §12d FS smoke) reads the bytes and hands
//! them to `create_process(4099)`.

#![no_std]
#![no_main]

extern crate alloc;

use core::panic::PanicInfo;

use orbit_abi::{
    serialln,
    user::{SerialWriter, exit},
};
use orbit_rt as _;

/// orbit-rt's `_start` calls this after eagerly resolving argv. The
/// return value is passed to `exit` — `42` keeps §13a.2's wait_pid
/// path well-tested through dealloc_process →
/// exit_waiter → signal_pair → wake_blocked_inline → user a1.
#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    // serialln! (not logln!) — short-lived processes lose their
    // framebuffer scrollback when the source gets torn down, so go
    // straight to the kernel serial log instead.
    serialln!("hello from /bin/hello");

    // §13a.3 — print argv. Each line is a separate serialln so
    // cross-hart interleaving can't split a single arg across
    // entries. `argc` is logged first so the smoke harness can pin
    // on it.
    let args = orbit_rt::argv::args();
    serialln!("hello argc={}", args.len());
    for (i, arg) in args.iter().enumerate() {
        let s = core::str::from_utf8(arg).unwrap_or("<non-utf8>");
        serialln!("hello argv[{i}]={s}");
    }

    #[cfg(feature = "scrollback-stress")]
    scrollback_bounding_test();

    42
}

/// Diagnostic for the kheap stats path. Writes a max-length line
/// (`MAX_LINE_LEN`-sized) into the framebuffer scrollback on every
/// iteration and prints the post-write `kernel_heap_bytes` to serial.
///
/// Expected shape: `kernel_heap_bytes` grows by ~256 B per iter for
/// the first ~`SCROLLBACK_LINES` (500) iters as the per-source
/// scrollback fills, then plateaus once `pop_front` starts firing in
/// lockstep with `push_back`. After this process exits, the scrollback
/// drops as part of `dealloc_process`'s `RemoveSource`, returning
/// kheap to the pre-spawn baseline.
///
/// Iteration count must exceed `SCROLLBACK_LINES` to demonstrate the
/// plateau — 500 iters is the minimum, 1000 leaves headroom to see
/// the steady-state behavior clearly.
#[cfg(feature = "scrollback-stress")]
fn scrollback_bounding_test() {
    use alloc::string::String;
    use orbit_abi::logln;

    let line: String = (0..256).map(|_| 'a').collect();
    for i in 0..1000 {
        logln!("{line}");

        if let Ok(s) = orbit_abi::user::query_stats() {
            serialln!("{i}: heap: {}B", s.kernel_heap_bytes);
        }
        else {
            serialln!("{i} failed to get stats");
        }

        orbit_abi::user::sleep_ms(100).expect("failed sleep");
    }
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "hello panic: {p}");
    w.flush();
    exit(isize::MIN);
}
