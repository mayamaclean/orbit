//! TCP throughput micro-benchmark.
//!
//! Connects out to `192.168.76.2:<port>` (the host gateway in QEMU's
//! user-mode NAT — same as the smoke listener), pumps `TARGET_BYTES`
//! of zeros as fast as `write_all` will accept them, and prints
//! elapsed ticks + computed throughput on serial. Repeats `ROUNDS`
//! times so we can eyeball variance.
//!
//! Tick basis: QEMU virt's `time` CSR runs at 10 MHz, matching kmain's
//! `TICKS_PER_MS = 10_000`. Throughput math uses that constant.
//!
//! Port: 7780 by default (see `BENCH_PORT`); the bench-tcp script
//! starts an `nc -l 7780 > /dev/null` on the host before launching
//! QEMU. No QEMU hostfwd needed since the kernel is the client.
//!
//! Send pattern: a fixed-size buffer (`BUF_SIZE`) of zeros, written
//! repeatedly via [`Session::write_all`]. write_all is the right
//! primitive here — it loops on partial writes and `nc_yield`s when
//! the netch tx ring is full, which exercises the parker → knet
//! signal path that Phase A+B+C+D restructured.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use core::panic::PanicInfo;

use orbit_abi::errno::{Errno, EAGAIN};
use orbit_abi::net::SockType;
use orbit_abi::{logln, user::{exit, get_micros, sleep_ms, SerialWriter}};
use net_channel::{BindSpec, NetChannel, NC_MAX_REGION_SIZE};
use orbit_rt::netch::NetCh;

/// Listener port on the host gateway (192.168.76.2 in QEMU's user-mode
/// NAT). Bench-tcp script starts `nc -l <BENCH_PORT>` before booting.
const BENCH_PORT: u16 = 7780;

/// Bytes to push per round. 8 MiB is enough to amortize connection
/// setup and warmup but small enough that a stalled scheduler shows
/// up as a long round, not a hang.
const TARGET_BYTES: usize = 8 * 1024 * 1024;

/// Per-write chunk. Page-sized matches the netch ring's natural
/// granularity; smaller chunks would amplify per-call overhead.
const BUF_SIZE: usize = 4096;

/// How many times to run the bench. Variance across rounds is the
/// signal we care about — a single bad round is rarely noise; a
/// consistent floor is the steady-state throughput.
const ROUNDS: usize = 4;

/// Microseconds per second — used to convert the syscall's μs deltas
/// into bytes/sec for throughput math.
const MICROS_PER_SECOND: u64 = 1_000_000;

fn run_round(round: usize) -> bool {
    use core::fmt::Write;

    // Fresh NetCh per round so each measurement starts from a known
    // state (no leftover ring contents skewing the first writes).
    // ClientOneShot dials 192.168.76.2:BENCH_PORT once via
    // next_session; the listener accepts and we start sending.
    //
    // Request the largest valid ring capacity. A small ring throttles
    // throughput by capping in-flight bytes — the bench would measure
    // ring-drain latency, not raw scheduler/TCP throughput. The cap
    // comes from NC_MAX_REGION_SIZE (256 KiB) minus header overhead.
    let nc = match NetCh::open(
        NetChannel::capacity_for(NC_MAX_REGION_SIZE),
        SockType::Tcp,
        BindSpec::ClientOneShot {
            addr: u32::from_be_bytes([192, 168, 76, 2]),
            port: BENCH_PORT,
        },
    ) {
        Ok(n) => n,
        Err(Errno(e)) => {
            logln!("BENCH round {round}: NetCh::open errno={e}");
            return false;
        }
    };

    let session = match nc.next_session() {
        Ok(s) => s,
        Err(Errno(e)) => {
            logln!("BENCH round {round}: next_session errno={e}");
            return false;
        }
    };

    let buf = [0u8; BUF_SIZE];
    let mut sent = 0usize;
    let t0 = get_micros();
    while sent < TARGET_BYTES {
        let take = core::cmp::min(BUF_SIZE, TARGET_BYTES - sent);
        match session.write_all(&buf[..take]) {
            Ok(()) => sent += take,
            Err(Errno(e)) if e == EAGAIN => {
                // write_all already loops on EAGAIN internally via
                // nc_yield, but defensively handle a surfaced one.
                continue;
            }
            Err(Errno(e)) => {
                logln!("BENCH round {round}: write_all errno={e} after {sent} bytes");
                return false;
            }
        }
    }
    let t1 = get_micros();
    let elapsed_us = t1 - t0;

    // Drop the session before close so the host's nc sees FIN and
    // stops reading — otherwise nc keeps the socket half-open and
    // the next round's connect can collide.
    drop(session);
    if let Err(Errno(e)) = nc.close() {
        logln!("BENCH round {round}: nc.close errno={e}");
        return false;
    }

    // bytes / (us / 1_000_000) = bytes * 1_000_000 / us. u128 to
    // avoid overflow on big transfers.
    let bps = if elapsed_us == 0 {
        0
    } else {
        ((sent as u128) * (MICROS_PER_SECOND as u128) / (elapsed_us as u128)) as u64
    };
    let mib_per_sec_x100 = (bps * 100) / (1024 * 1024);
    let elapsed_ms = elapsed_us / 1_000;

    let mut w = SerialWriter::new();
    let _ = writeln!(
        w,
        "BENCH round {round}: bytes={sent} elapsed_us={elapsed_us} \
         elapsed_ms={elapsed_ms} bps={bps} MiB/s={}.{:02}",
        mib_per_sec_x100 / 100,
        mib_per_sec_x100 % 100,
    );
    w.flush();
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    logln!("umode-tcp-bench: starting (target={TARGET_BYTES} bytes/round, rounds={ROUNDS})");

    // Brief settle after process create so the netch subsystem has
    // had a chance to initialize; matches the umode smoke test's
    // pre-NetCh sleep.
    let _ = sleep_ms(500);

    let mut ok_rounds = 0usize;
    for round in 0..ROUNDS {
        if run_round(round) {
            ok_rounds += 1;
        }
        // Brief gap between rounds to let the host nc reset cleanly.
        let _ = sleep_ms(200);
    }

    logln!("umode-tcp-bench: done ({ok_rounds}/{ROUNDS} rounds completed)");
    exit(0);
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "umode-tcp-bench panic: {p}");
    w.flush();
    exit(isize::MIN);
}
