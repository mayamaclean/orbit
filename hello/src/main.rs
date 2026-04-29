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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
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

    // Distinct value so §13a.2's wait_pid smoke can verify the
    // exit-code path through dealloc_process → exit_waiter →
    // signal_pair → wake_blocked_inline → user a1.
    exit(42);
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "hello panic: {p}");
    w.flush();
    exit(isize::MIN);
}
