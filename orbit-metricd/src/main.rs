//! orbit-metricd — push process + per-syscall metrics to a host-side
//! collector over TCP for offline analysis (CSV / pandas / plotting).
//!
//! # Why
//!
//! Iterating on the syscall-resolution path (the CompletionHandle →
//! on-thread completion migration) requires before/after measurement
//! across many syscalls. Eyeballing the console `syscall-stats` row
//! after each migration scales badly. metricd streams a JSON-Lines
//! sample per tick to the host, where Python can compute deltas,
//! diff configurations, and plot.
//!
//! # Topology
//!
//! orbit-metricd binds `0.0.0.0:<port>` and accepts host clients in a
//! loop (one stream at a time, ServerRetain auto-recycle behind the
//! scenes — see `std::net::TcpListener` docs in
//! `library/std/src/sys/net/connection/orbit.rs`). QEMU's user-net
//! forwards the host port to the guest via the `hostfwd` entry in
//! `bl/.cargo/config.toml`; out of the box `tcp::7800-:7800` is set
//! up to match this binary's default.
//!
//! On the host:
//! ```sh
//! python3 tools/orbit_metric_logger.py --port 7800 --csv runs/before.csv
//! ```
//!
//! # Wire format
//!
//! UTF-8, newline-delimited JSON. One sample per line, compact (no
//! whitespace). Counters are cumulative-since-boot — the host
//! computes deltas. Schema (one line, formatted here for clarity):
//!
//! ```json
//! {
//!   "t_orbit": <u64>,                  // `time` CSR ticks (10 MHz on qemu-virt)
//!   "proc": {                          // mirrors orbit_abi::stats::ProcessStats
//!     "pid": <u16>, "thread_count": <u16>, "cpu_ticks": <u64>,
//!     "context_switches": <u64>, "syscalls": <u64>,
//!     "resident_bytes": <u64>, "heap_bytes": <u64>,
//!     "kernel_kpages_bytes": <u64>, "kernel_user_pages_bytes": <u64>,
//!     "kernel_ktables_bytes": <u64>, "kernel_heap_bytes": <u64>,
//!     "syscall_ticks": <u64>,
//!     "hart_user_ticks": <u64>, "hart_kernel_ticks": <u64>,
//!     "hart_scheduler_ticks": <u64>, "hart_idle_ticks": <u64>,
//!     "perm_denials": <u64>, "role_denials": <u64>,
//!     "wake_queue_peak": <u64>, "wake_queue_drops": <u64>, "wake_queue_capacity": <u64>
//!   },
//!   "syscalls": [
//!     {"ord": <u32>, "name": <str>, "count": <u64>, "total_ticks": <u64>, "max_ticks": <u64>},
//!     ...
//!   ]
//! }
//! ```
//!
//! Schema bumps: append fields only. The host parser ignores unknown
//! keys and treats missing keys as zero (forward-compat both
//! directions).
//!
//! # Usage
//!
//! ```text
//! orbit-metricd [<port>] [<rate_hz>]
//! ```
//!
//! Defaults: port 7800, rate 5 Hz (capped at 100). The listener
//! survives client disconnects — bring up a host collector, kill it,
//! bring up another, all without rebooting orbit.

use std::env;
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
use std::process::ExitCode;
use std::time::Duration;

use orbit_abi::Sysno;
use orbit_abi::stats::ProcessStats;
use orbit_abi::syscall_stats::SyscallEntry;
use orbit_abi::{
    serialln,
    user::{query_stats, query_syscall_stats},
};

const DEFAULT_PORT: u16 = 7800;
/// Default sample rate when the caller doesn't specify one. Low
/// enough that a long session writes a manageable CSV (5 Hz × 60 s ×
/// 60 min = 18 k rows / hr) but high enough to catch transient
/// behavior on syscall floors that fire at human-trigger cadence.
const DEFAULT_RATE_HZ: u64 = 5;
/// Cap on the requested rate. Sampling does two syscalls
/// (`query_stats` + `query_syscall_stats`) plus a TCP send per tick;
/// at 100 Hz that's still well under the kernel's per-syscall
/// service-time floor and won't dominate observation overhead, but
/// past 100 the sampler starts measuring itself.
const MAX_RATE_HZ: u64 = 100;

fn main() -> ExitCode {
    let mut args = env::args();
    let _argv0 = args.next();
    let port = match args.next() {
        Some(s) => match s.parse::<u16>() {
            Ok(v) => v,
            Err(e) => {
                serialln!("orbit-metricd: invalid port {s:?}: {e}");
                return ExitCode::from(2);
            }
        },
        None => DEFAULT_PORT,
    };
    let rate_hz = match args.next() {
        Some(s) => match s.parse::<u64>() {
            Ok(v) => v.clamp(1, MAX_RATE_HZ),
            Err(e) => {
                serialln!("orbit-metricd: invalid rate {s:?}: {e}");
                return ExitCode::from(2);
            }
        },
        None => DEFAULT_RATE_HZ,
    };

    let bind_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    let listener = match TcpListener::bind(bind_addr) {
        Ok(l) => l,
        Err(e) => {
            serialln!("orbit-metricd: bind {bind_addr}: {e}");
            return ExitCode::from(1);
        }
    };
    serialln!("orbit-metricd: listening on {bind_addr} at {rate_hz} Hz");

    let period = Duration::from_nanos(1_000_000_000 / rate_hz);
    // Reusable line buffer — avoids per-tick allocation for the
    // common-size sample (~5 KiB at COUNT=52).
    let mut line = String::with_capacity(8 * 1024);
    // Scratch buffer for query_syscall_stats. Sized for the kernel's
    // native payload; the helper resizes if needed.
    let mut scratch = vec![0u8; orbit_abi::syscall_stats::payload_size()];

    // Outer loop: accept the next host client when the previous one
    // disconnects. ServerRetain semantics on the underlying NetCh
    // mean the kernel auto-recycles the listen between sessions.
    let mut sample_seq: u64 = 0;
    loop {
        let (mut stream, peer) = match listener.accept() {
            Ok(p) => p,
            Err(e) => {
                serialln!("orbit-metricd: accept: {e}");
                return ExitCode::from(1);
            }
        };
        serialln!("orbit-metricd: client connected from {peer}");

        // Inner loop: stream samples until the peer disconnects (or
        // any other write error), then return to accept.
        let session_err = loop {
            line.clear();
            let proc_stats = query_stats().unwrap_or_default();
            let syscall_payload = query_syscall_stats(&mut scratch);
            let now_ticks = read_time_ticks();
            sample_seq = sample_seq.wrapping_add(1);

            line.push_str("{\"seq\":");
            push_u64(&mut line, sample_seq);
            line.push_str(",\"t_orbit\":");
            push_u64(&mut line, now_ticks);
            line.push_str(",\"proc\":");
            write_proc(&mut line, &proc_stats);
            line.push_str(",\"syscalls\":");
            match syscall_payload {
                Ok((header, entries)) => {
                    let n = (header.count as usize).min(entries.len());
                    write_syscalls(&mut line, &entries[..n]);
                }
                Err(_) => line.push_str("[]"),
            }
            line.push_str("}\n");

            // Pre-write trace: visible on the orbit console so we can
            // correlate "what orbit-metricd intended to send" with
            // "what the host actually received."
            //serialln!(
            //    "orbit-metricd: tx seq={} bytes={}",
            //    sample_seq,
            //    line.len(),
            //);

            if let Err(e) = stream.write_all(line.as_bytes()) {
                break Some(e);
            }
            std::thread::sleep(period);
        };
        match session_err {
            Some(e) => serialln!("orbit-metricd: session ended ({peer}): {e}"),
            None => serialln!("orbit-metricd: session ended ({peer}): clean"),
        }
    }
}

/// Read the RISC-V `time` CSR via `get_micros`. We keep ticks (not
/// microseconds) so the host can convert with full precision.
/// 10 MHz on qemu-virt → divide by 10 for µs, by 10_000 for ms.
fn read_time_ticks() -> u64 {
    orbit_abi::user::get_micros().wrapping_mul(10)
}

// ─── JSON writers ────────────────────────────────────────────────────

fn push_u64(out: &mut String, v: u64) {
    use core::fmt::Write;
    let _ = write!(out, "{}", v);
}

fn push_u16(out: &mut String, v: u16) {
    use core::fmt::Write;
    let _ = write!(out, "{}", v);
}

fn write_proc(out: &mut String, p: &ProcessStats) {
    out.push_str("{\"pid\":");
    push_u16(out, p.pid);
    macro_rules! field {
        ($name:literal, $val:expr) => {{
            out.push(',');
            out.push('"');
            out.push_str($name);
            out.push_str("\":");
            push_u64(out, $val as u64);
        }};
    }
    field!("thread_count", p.thread_count);
    field!("cpu_ticks", p.cpu_ticks);
    field!("context_switches", p.context_switches);
    field!("syscalls", p.syscalls);
    field!("resident_bytes", p.resident_bytes);
    field!("heap_bytes", p.heap_bytes);
    field!("kernel_kpages_bytes", p.kernel_kpages_bytes);
    field!("kernel_user_pages_bytes", p.kernel_user_pages_bytes);
    field!("kernel_ktables_bytes", p.kernel_ktables_bytes);
    field!("kernel_heap_bytes", p.kernel_heap_bytes);
    field!("syscall_ticks", p.syscall_ticks);
    field!("hart_user_ticks", p.hart_user_ticks);
    field!("hart_kernel_ticks", p.hart_kernel_ticks);
    field!("hart_scheduler_ticks", p.hart_scheduler_ticks);
    field!("hart_idle_ticks", p.hart_idle_ticks);
    field!("perm_denials", p.perm_denials);
    field!("role_denials", p.role_denials);
    field!("wake_queue_peak", p.wake_queue_peak);
    field!("wake_queue_drops", p.wake_queue_drops);
    field!("wake_queue_capacity", p.wake_queue_capacity);
    out.push('}');
}

fn write_syscalls(out: &mut String, entries: &[SyscallEntry]) {
    out.push('[');
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"ord\":");
        push_u64(out, i as u64);
        out.push_str(",\"name\":\"");
        out.push_str(syscall_name(i));
        out.push_str("\",\"count\":");
        push_u64(out, e.count);
        out.push_str(",\"total_ticks\":");
        push_u64(out, e.total_ticks);
        out.push_str(",\"max_ticks\":");
        push_u64(out, e.max_ticks);
        out.push('}');
    }
    out.push(']');
}

/// Mirror of [`console`/`orbit-top-std`]'s table. Append a row
/// whenever `Sysno` grows.
fn syscall_name(ordinal: usize) -> &'static str {
    match ordinal {
        0 => "exit",
        1 => "serial_print",
        2 => "sleep_ms",
        3 => "console_write",
        4 => "read_stdin",
        5 => "set_affinity",
        6 => "get_affinity",
        7 => "get_hart_id",
        8 => "mmap",
        9 => "create_netch",
        10 => "close_handle",
        11 => "create_process",
        12 => "ch_yield",
        13 => "query_stats",
        14 => "query_syscall_stats",
        15 => "create_thread",
        16 => "get_micros",
        17 => "fs_open",
        18 => "fs_read",
        19 => "fs_stat",
        20 => "getpid",
        21 => "gettid",
        22 => "wait_pid",
        23 => "create_process_ex",
        24 => "argv_envp",
        25 => "futex_wait",
        26 => "futex_wake",
        27 => "fs_readdir",
        28 => "pledge",
        29 => "create_process_v2",
        30 => "query_denial_log",
        31 => "chdir",
        32 => "getcwd",
        33 => "fs_seek",
        34 => "fs_fstat",
        35 => "getuid",
        36 => "geteuid",
        37 => "getgid",
        38 => "getegid",
        39 => "getgroups",
        40 => "getlogin",
        41 => "setuid",
        42 => "setgid",
        43 => "setgroups",
        44 => "setlogin",
        45 => "get_realtime",
        46 => "thread_exit",
        47 => "fb_query",
        48 => "fb_surface_create",
        49 => "fb_surface_destroy",
        50 => "fb_present",
        51 => "read_key_event",
        52 => "wake_tid",
        53 => "dup",
        54 => "dup2",
        55 => "fcntl",
        56 => "fstat",
        57 => "eventfd",
        58 => "ch_inspect",
        _ => "?",
    }
}

const _: () = assert!(
    Sysno::COUNT == 59,
    "syscall_name table must be resized when Sysno::COUNT changes"
);
