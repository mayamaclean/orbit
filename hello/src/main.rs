//! `/bin/hello` — minimal user binary that proves §12e exec works
//! end-to-end. Runs out of the tarfs-served disk image; the console
//! `exec` builtin (or the §12d FS smoke) reads the bytes and hands
//! them to `create_process(4099)`.

#![no_std]
#![no_main]

extern crate alloc;

use core::panic::PanicInfo;

use orbit_abi::{
    fs::{OPEN_RDONLY, Stat},
    serialln,
    user::{SerialWriter, exit, fs_open, fs_read, fs_stat},
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

    do_fs_read_test();

    #[cfg(feature = "scrollback-stress")]
    scrollback_bounding_test();

    42
}

fn do_fs_read_test() {
    const RSIZE: usize = 64 * 1024;

    let mut stat = Stat::default();
    if let Err(e) = fs_stat("/bin/hello-std", &mut stat) {
        serialln!("do_fs_read_test failed to stat: {e:?}");
        return;
    }

    serialln!("do_fs_read_test: {stat:?}");

    let fd = match fs_open("/bin/hello-std", OPEN_RDONLY) {
        Ok(f) => f,
        Err(e) => {
            serialln!("do_fs_read_test failed to open: {e:?}");
            return;
        }
    };

    let mut buf = [0u8; RSIZE];
    let file_len = stat.st_size;

    let mut total_reads = 0;

    let mut read_so_far = 0;
    while read_so_far < file_len {
        match fs_read(fd, &mut buf[..]) {
            Ok(read) => {
                read_so_far += read as i64;
                if read < RSIZE {
                    serialln!("do_fs_read_test only read {read}B");
                }
            }
            Err(e) => {
                serialln!("do_fs_read_test failed to read: {e:?}");
                return;
            }
        }
        total_reads += 1;
    }

    let expected_reads = if (file_len % RSIZE as i64) == 0 {
        file_len / RSIZE as i64
    }
    else {
        (file_len / RSIZE as i64) + 1
    };

    serialln!("do_fs_read_test total_reads: {total_reads}, expected: {expected_reads}");
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
