#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use core::{panic::PanicInfo, sync::atomic::Ordering};

use net_channel::NetChannel;
use orbit_abi::{logln, user::{close_handle, create_netch, exit, sleep_ms, console_write, serial_print, SerialWriter}};

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
    // The kernel caps sleep at 60*60*1000 ms. `>=` MAX returns -2.
    check("sleep_ms at cap",    sleep_ms(60 * 60 * 1000),     -2);
    check("sleep_ms above cap", sleep_ms(60 * 60 * 1000 + 1), -2);

    // --- console_write error paths ---
    // NULL-region VA (inside USER_NULL_GUARD_END) never translates → -2.
    check("console_write null VA", console_write(0x1000, 5), -2);
    check("serial_print null VA", serial_print(0x1000, 5),-2);

    // len > PAGE_SIZE rejected with -3 before any memory is touched,
    // so the pointer just needs to be plausible.
    static FILLER: [u8; 16] = [b'x'; 16];
    check(
        "console_write too long",
        console_write(&FILLER as *const u8 as usize, 4097),
        -3,
    );
    check(
        "serial_print too long",
        serial_print(&FILLER as *const u8 as usize, 4097),
        -3,
    );

    // Non-UTF-8 bytes rejected with -4. 0xFF is never a valid start byte.
    static BAD_UTF8: [u8; 4] = [0xFF, 0xFE, 0xFD, 0xFC];
    check(
        "console_write non-utf8",
        console_write(&BAD_UTF8 as *const u8 as usize, 4),
        4,
    );
    check(
        "serial_print non-utf8",
        serial_print(&BAD_UTF8 as *const u8 as usize, 4),
        -4,
    );

    // --- close_handle before any netchannel exists ---
    // No process_handles entry for this pid → -1.
    check("close_handle no registry", close_handle(7), -1);

    logln!("=== error path tests done ===");
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    // print to serial
    logln!("hello world!");

    run_heap_smoke();

    run_error_path_tests();

    sleep_ms(2000);

    // Ask the kernel to create a NetChannel. The hint is above
    // USER_TEXT_BASE (0x2_2000_0000) so it can't clip into the stack
    // region below. The kernel returns the actual VA it picked — today
    // that's always the hint, but readers should not rely on it.
    const AHINT: usize = 0x2_4000_0000;
    const NC_REGION_SIZE: usize = 4096;
    let (nc_vaddr, nc_fd) = match create_netch(AHINT, NC_REGION_SIZE, 0) {
        Ok(v) => v,
        Err(_) => {
            logln!("failed to create netchannel!");
            exit(-2isize);
        }
    };

    logln!("netchannel created!");

    // Bogus fd AFTER a netchannel has been created — process_handles
    // now has an entry for this pid, but fd 999 isn't in it → -2.
    // (The earlier `no registry` test hit the no-pid-entry branch.)
    check("close_handle bogus fd", close_handle(999), -2);

    let nc = unsafe { &*(nc_vaddr as *const NetChannel) };

    if let Err(_) = nc.connect_tcp(u32::from_be_bytes([192,168,76,2]), 65535) {
        logln!("bad failed nc tcp connect!");

        // exit call
        exit(-2isize);
    }

    loop {
        let state = nc.current_state().state.load(Ordering::Acquire);

        if state > 0 {
            logln!("tcp connected!");
            break
        }
        else if state < 0 {
            logln!("tcp connect failed!");
            break
        }
        else if state == 0 {
            // sleep for ms
            let _ = sleep_ms(10);
        }
    }

    //exit(0);

    let mut written = false;
    let mut br = false;
    loop {
        if !written && nc.writeable() > 0 {
            let wr = nc.send_tcp(|b| {
                let msg = b"Hello World!\n";
                b.copy_from_slice(msg)
            });

            if let Ok(n) = wr {
                if n > 0 {
                    written = true;
                }
            }
        }

        if nc.readable() > 0 {
            let r = nc.recv_tcp(|rx| {
                if rx.starts_with(b"exit") {
                    br = true;
                }
                console_write(rx.as_ptr() as usize, rx.len());
                rx.len()
            });

            match r {
                Err(e) if e > -4 => {
                    // exit call
                    exit(e);
                }
                _ => {}
            }

            if br {
                // Close the handle before exit so we exercise the
                // revoke path from a live process, not just from
                // teardown. After this returns, `nc` is invalid — the
                // user mapping has been torn down.
                let cr = close_handle(nc_fd);
                if cr != 0 {
                    logln!("close_handle failed!");
                    exit(cr);
                }

                logln!("close_handle ok!");

                let _ = unsafe {
                    core::ptr::read_volatile(nc as *const _ as *const u8);
                };
            }
        }
        else {
            // sleep for ms
            let _ = sleep_ms(100);
        }

        let state = nc.current_state().state.load(Ordering::Acquire);

        if state <= 0 {
            logln!("tcp connection failed!");
            break
        }
    }    
    exit(-99);
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "umode panic: {p}");
    w.flush();
    exit(isize::MIN);
}
