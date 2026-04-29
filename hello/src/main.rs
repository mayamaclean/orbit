//! `/bin/hello` — minimal user binary that proves §12e exec works
//! end-to-end. Runs out of the tarfs-served disk image; the console
//! `exec` builtin (or the §12d FS smoke) reads the bytes and hands
//! them to `create_process(4099)`.

#![no_std]
#![no_main]

use core::panic::PanicInfo;

use orbit_abi::{
    serialln,
    user::{exit, SerialWriter},
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

    42
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "hello panic: {p}");
    w.flush();
    exit(isize::MIN);
}
