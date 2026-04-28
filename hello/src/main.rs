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
    exit(0);
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "hello panic: {p}");
    w.flush();
    exit(isize::MIN);
}
