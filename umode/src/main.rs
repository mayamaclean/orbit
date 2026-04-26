#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use core::panic::PanicInfo;

use orbit_abi::errno::{Errno, EBADF, EFAULT, EINVAL};
use orbit_abi::net::SockType;
use orbit_abi::{logln, user::{close_handle, exit, sleep_ms, console_write, serial_print, SerialWriter}};
use orbit_rt::netch::NetCh;

/// Report a PASS/FAIL line for a single error-path scenario. The smoke
/// script greps for "PASS: <name>" lines to validate each branch fired.
fn check(name: &str, got: isize, want: isize) {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    if got == want {
        let _ = writeln!(w, "PASS: {name} got {got}");
    } else {
        let _ = writeln!(w, "FAIL: {name} want {want} got {got}");
    }
    w.flush();
}

/// Variant for syscalls that return `Result<T, Errno>`. Asserts the
/// call errored with `want`. The smoke script's grep is the same.
fn check_err<T: core::fmt::Debug>(name: &str, got: Result<T, Errno>, want: i32) {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    match got {
        Err(Errno(e)) if e == want => {
            let _ = writeln!(w, "PASS: {name} got Err(errno={e})");
        }
        other => {
            let _ = writeln!(w, "FAIL: {name} want Err(errno={want}) got {other:?}");
        }
    }
    w.flush();
}

/// Variant for syscalls that return `Result<T, Errno>`. Asserts the
/// call succeeded with `want`.
fn check_ok<T: core::fmt::Debug + PartialEq>(name: &str, got: Result<T, Errno>, want: T) {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    match got {
        Ok(v) if v == want => {
            let _ = writeln!(w, "PASS: {name} got Ok({v:?})");
        }
        other => {
            let _ = writeln!(w, "FAIL: {name} want Ok({want:?}) got {other:?}");
        }
    }
    w.flush();
}

/// Exercise the orbit-rt heap: first touch forces the talc `Source` to
/// mmap its first arena; subsequent pushes stay in that arena. Prints
/// PASS/FAIL for the smoke script to grep.
fn run_heap_smoke() {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::fmt::Write;

    let b = Box::new(0xABCDu32);
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "heap Box: {:#x} (want 0xabcd)", *b);
    w.flush();

    let mut v: Vec<u32> = Vec::new();
    for i in 0..1024 { v.push(i); }
    let sum: u64 = v.iter().map(|&x| x as u64).sum();
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "heap Vec sum: {sum} (want {})", (0u64..1024).sum::<u64>());
    w.flush();

    check("heap Box value", *b as isize, 0xABCD);
    check("heap Vec sum",   sum as isize, (0u64..1024).sum::<u64>() as isize);
}

/// Exercise syscall error paths that QEMU smoke otherwise never hits.
/// Each check prints a PASS/FAIL marker the smoke script verifies.
fn run_error_path_tests() {
    logln!("=== error path tests ===");

    // --- sleep_ms edge cases ---
    // The kernel caps sleep at 60*60*1000 ms. `>=` MAX returns EINVAL.
    check_err("sleep_ms at cap",    sleep_ms(60 * 60 * 1000),     EINVAL);
    check_err("sleep_ms above cap", sleep_ms(60 * 60 * 1000 + 1), EINVAL);

    // --- console_write / serial_print error paths ---
    // NULL-region VA (inside USER_NULL_GUARD_END) never translates → EFAULT.
    check_err("console_write null VA", console_write(0x1000, 5), EFAULT);
    check_err("serial_print null VA",  serial_print(0x1000, 5),  EFAULT);

    // len > PAGE_SIZE rejected with EINVAL before any memory is
    // touched, so the pointer just needs to be plausible.
    static FILLER: [u8; 16] = [b'x'; 16];
    check_err(
        "console_write too long",
        console_write(&FILLER as *const u8 as usize, 4097),
        EINVAL,
    );
    check_err(
        "serial_print too long",
        serial_print(&FILLER as *const u8 as usize, 4097),
        EINVAL,
    );

    // console_write doesn't validate UTF-8 — 4 bytes go through fine.
    // serial_print does, returns EINVAL on the same input.
    static BAD_UTF8: [u8; 4] = [0xFF, 0xFE, 0xFD, 0xFC];
    check_ok(
        "console_write non-utf8",
        console_write(&BAD_UTF8 as *const u8 as usize, 4),
        4usize,
    );
    check_err(
        "serial_print non-utf8",
        serial_print(&BAD_UTF8 as *const u8 as usize, 4),
        EINVAL,
    );

    // --- close_handle before any netchannel exists ---
    // No process_handles entry for this pid → EBADF.
    check_err("close_handle no registry", close_handle(7), EBADF);

    logln!("=== error path tests done ===");
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    // print to serial
    logln!("hello world!");

    run_heap_smoke();

    run_error_path_tests();

    let _ = sleep_ms(2000);

    // Open a NetChannel (smallest valid region — capacity=0 hits the
    // floor at NC_MIN_REGION_SIZE). NetCh reserves a VA in the shared
    // range from orbit_rt::SHARED_VA, then asks the kernel to install
    // the mapping at that VA.
    let nc = match NetCh::open(0, SockType::Tcp) {
        Ok(n) => n,
        Err(_) => {
            logln!("failed to create netchannel!");
            exit(-2isize);
        }
    };

    logln!("netchannel created!");

    // Bogus fd AFTER a netchannel has been created — process_handles
    // now has an entry for this pid, but fd 999 isn't in it → EBADF.
    // (The earlier `no registry` test hit the no-pid-entry branch.)
    check_err("close_handle bogus fd", close_handle(999), EBADF);

    if let Err(_) = nc.connect(u32::from_be_bytes([192,168,76,2]), 65535) {
        logln!("tcp connect failed!");
        exit(-2isize);
    }
    logln!("tcp connected!");

    // Send the greeting once, then drain any reply. The smoke
    // listener echoes "exit\n" to terminate this loop.
    if nc.write_all(b"Hello World!\n").is_err() {
        logln!("tcp write failed!");
        exit(-2isize);
    }

    let mut buf = [0u8; 1024];
    loop {
        match nc.read(&mut buf) {
            Ok(n) if n > 0 => {
                // console_write before the exit check so the smoke
                // script can grep the "exit\n" payload off serial —
                // console_write tee's to serial as `USER[pid]: ...`.
                let _ = console_write(buf.as_ptr() as usize, n);
                if buf[..n].starts_with(b"exit") {
                    break;
                }
            }
            Ok(_) => {
                // Spurious zero-byte read — back off briefly.
                let _ = sleep_ms(100);
            }
            Err(Errno(e)) if e == orbit_abi::errno::EAGAIN => {
                let _ = sleep_ms(100);
            }
            Err(_) => {
                logln!("tcp read failed (channel down)");
                exit(-99);
            }
        }
    }

    // Close the handle explicitly so we exercise the revoke path from
    // a live process, not just from teardown. NetCh::close consumes
    // self → kernel handle is closed and SharedRegion is freed back
    // to SHARED_VA (so a future NetCh::open can reuse the same VA).
    match nc.close() {
        Ok(()) => logln!("close_handle ok!"),
        Err(Errno(e)) => {
            logln!("close_handle failed: errno={e}");
            exit(-(e as isize));
        }
    }

    exit(0);
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "umode panic: {p}");
    w.flush();
    exit(isize::MIN);
}
