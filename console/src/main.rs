//! In-tree shell. Owns whichever framebuffer pane is `Source::Process(pid)`
//! for this process; reads keystrokes from its stdin ring (fed by
//! `kmain`'s [`input::dispatch`] when this pane is active) and renders
//! prompt + echo + builtin output via `console_write`.
//!
//! Loop shape: `read_stdin` → `LineEditor::feed` per byte → on `\n`,
//! `dispatch(line)` runs a builtin and re-prints the prompt. Everything
//! is single-threaded; no allocator pressure beyond the editor's
//! line buffer (which lives across iterations).
//!
//! Builtins MVP: `echo`, `help`, `clear`. `ps` is gated on the
//! `ps_snapshot` syscall (deferred from §9) so it isn't wired today.
//!
//! Display compositor support (kmain/src/drivers/display.rs):
//! - `\x08` (backspace) — pops the last char from the in-progress line.
//! - `\x0c` (form feed) — clears this source's scrollback. Used by `clear`.
//! Other non-printables are still rendered as `?`.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use alloc::vec::Vec;
use core::panic::PanicInfo;

use core::fmt::Write;
use orbit_abi::{
    perms::{CreateProcessV2Args, role},
    syscall_stats::payload_size,
    user::{
        ConsoleWriter, console_write, create_process_v2, exit, query_stats, query_syscall_stats,
        read_stdin, sleep_ms, wait_pid,
    },
};

const PROMPT: &[u8] = b"console@orbit $ ";

/// One PIPE_BUF chunk. `console_write` rejects `len > 4096` with EINVAL,
/// so any longer payload has to be split before the syscall.
const CHUNK: usize = 4096;

/// Send `bytes` through `console_write`, splitting at the kernel's
/// 4 KiB PIPE_BUF boundary. Errors (ring full, etc.) are dropped on
/// the floor — `console_write` already retries internally for EAGAIN
/// via the ConsoleWriter shim used elsewhere; the console treats output
/// as best-effort to keep the read-edit loop responsive.
fn write_chunked(bytes: &[u8]) {
    let mut i = 0;
    while i < bytes.len() {
        let end = core::cmp::min(i + CHUNK, bytes.len());
        let _ = console_write(bytes[i..end].as_ptr() as usize, end - i);
        i = end;
    }
}

/// Append-with-backspace line editor. No mid-line cursor movement —
/// arrow keys and other ANSI sequences are swallowed by a tiny
/// `ESC [ X` state machine so they don't leak into the buffer or
/// echo as `?`s. Ctrl-C cancels the current line.
struct LineEditor {
    buf: Vec<u8>,
    /// 0 = idle, 1 = saw ESC, 2 = saw `ESC [` (next byte ends the seq).
    esc: u8,
}

impl LineEditor {
    const fn new() -> Self {
        Self {
            buf: Vec::new(),
            esc: 0,
        }
    }

    /// Feed one byte from the stdin ring. Returns `Some(line)` on `\n`
    /// (caller dispatches; the editor's buffer is reset by the take).
    fn feed(&mut self, b: u8) -> Option<Vec<u8>> {
        match self.esc {
            1 => {
                self.esc = if b == b'[' { 2 } else { 0 };
                return None;
            }
            2 => {
                self.esc = 0;
                return None;
            }
            _ => {}
        }
        match b {
            0x1b => {
                self.esc = 1;
                None
            }
            b'\n' => {
                let line = core::mem::take(&mut self.buf);
                write_chunked(b"\n");
                Some(line)
            }
            // BS (Ctrl-H) and DEL — different keyboards send different
            // bytes for the same intent. Both pop one char and echo
            // `\x08` so the compositor's pending line shrinks too.
            0x08 | 0x7f => {
                if self.buf.pop().is_some() {
                    write_chunked(b"\x08");
                }
                None
            }
            // Ctrl-C: discard the in-flight line and re-prompt. Echoing
            // `^C\n` matches what bash / dash do.
            0x03 => {
                self.buf.clear();
                write_chunked(b"^C\n");
                Some(Vec::new())
            }
            // Tab is fine to type — no completion in MVP, just a
            // literal whitespace byte.
            b if b.is_ascii_graphic() || b == b' ' || b == b'\t' => {
                self.buf.push(b);
                write_chunked(core::slice::from_ref(&b));
                None
            }
            // Everything else (Ctrl-letter we don't special-case, NUL,
            // etc.) is silently ignored — keeps the buffer ASCII-clean
            // for splitn.
            _ => None,
        }
    }
}

/// Snapshot the kernel's view of this process and dump it in a
/// `top`-ish two-column format. Time fields are displayed even when
/// the kernel returns 0 — Phase 2 wires the per-hart bucket state
/// machine that populates them.
fn stats_cmd() {
    let stats = match query_stats() {
        Ok(s) => s,
        Err(e) => {
            let mut w = LineWriter::new();
            let _ = writeln!(w, "stats: query_stats failed (errno {})", e.0);
            w.flush();
            return;
        }
    };

    // 10 MHz `time` CSR on qemu-virt — divide by 10_000 for ms.
    let ms = |ticks: u64| ticks / 10_000;

    let mut w = LineWriter::new();
    let _ = writeln!(w, "process:");
    let _ = writeln!(w, "  pid              {}", stats.pid);
    let _ = writeln!(w, "  threads          {}", stats.thread_count);
    let _ = writeln!(w, "  cpu_ms           {}", ms(stats.cpu_ticks));
    let _ = writeln!(w, "  syscall_ms       {}", ms(stats.syscall_ticks));
    let _ = writeln!(w, "  syscalls         {}", stats.syscalls);
    let _ = writeln!(w, "  ctx_switches     {}", stats.context_switches);
    let _ = writeln!(w, "  resident         {}", HumanBytes(stats.resident_bytes));
    let _ = writeln!(w, "  heap             {}", HumanBytes(stats.heap_bytes));
    let _ = writeln!(w, "kernel pools:");
    let _ = writeln!(
        w,
        "  kpages           {}",
        HumanBytes(stats.kernel_kpages_bytes)
    );
    let _ = writeln!(
        w,
        "  user_pages       {}",
        HumanBytes(stats.kernel_user_pages_bytes)
    );
    let _ = writeln!(
        w,
        "  ktables          {}",
        HumanBytes(stats.kernel_ktables_bytes)
    );
    let _ = writeln!(
        w,
        "  kheap            {}",
        HumanBytes(stats.kernel_heap_bytes)
    );
    let _ = writeln!(w, "harts (system-wide):");
    let _ = writeln!(w, "  user_ms          {}", ms(stats.hart_user_ticks));
    let _ = writeln!(w, "  kernel_ms        {}", ms(stats.hart_kernel_ticks));
    let _ = writeln!(w, "  scheduler_ms     {}", ms(stats.hart_scheduler_ticks));
    let _ = writeln!(w, "  idle_ms          {}", ms(stats.hart_idle_ticks));
    let _ = writeln!(w, "wake_queue:");
    let _ = writeln!(
        w,
        "  peak / cap       {} / {}",
        stats.wake_queue_peak, stats.wake_queue_capacity,
    );
    let _ = writeln!(w, "  drops            {}", stats.wake_queue_drops);
    w.flush();
}

/// Map a `Sysno::ordinal()` value back to a human-readable name.
/// Kept in sync with the match arms in `Sysno::ordinal` — appending a
/// new variant there means appending one row here.
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
        _ => "?",
    }
}

/// Snapshot the kernel's per-syscall counters and dump them as a table.
/// Buffer is sized for the kernel this binary was built against; a
/// newer kernel writes a prefix (header reports the row count it
/// actually filled), an older one fills fewer rows than `Sysno::COUNT`.
fn syscall_stats_cmd() {
    const BUF_LEN: usize = payload_size();
    let mut buf = [0u8; BUF_LEN];
    let (header, entries) = match query_syscall_stats(&mut buf) {
        Ok(r) => r,
        Err(e) => {
            let mut w = LineWriter::new();
            let _ = writeln!(w, "syscall-stats: query failed (errno {})", e.0);
            w.flush();
            return;
        }
    };

    let n = core::cmp::min(header.count as usize, entries.len());
    // 10 MHz `time` CSR — same conversion as stats_cmd.
    let to_ms = |ticks: u64| ticks / 10_000;

    let mut w = LineWriter::new();
    let _ = writeln!(
        w,
        "{:<22}{:>12}{:>14}{:>12}{:>12}",
        "syscall", "count", "total_ms", "avg_us", "max_us",
    );
    for i in 0..n {
        let e = &entries[i];
        let avg_us = if e.count > 0 {
            (e.total_ticks / e.count) / 10
        }
        else {
            0
        };
        let max_us = e.max_ticks / 10;
        let _ = writeln!(
            w,
            "{:<22}{:>12}{:>14}{:>12}{:>12}",
            syscall_name(i),
            e.count,
            to_ms(e.total_ticks),
            avg_us,
            max_us,
        );
    }
    w.flush();
}

/// `Display` shim for byte counts. Picks B / KiB / MiB based on the
/// magnitude — keeps the column width predictable without forcing the
/// user to count digits.
struct HumanBytes(u64);

impl core::fmt::Display for HumanBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const KIB: u64 = 1024;
        const MIB: u64 = 1024 * 1024;
        if self.0 >= MIB {
            write!(f, "{} MiB", self.0 / MIB)
        }
        else if self.0 >= KIB {
            write!(f, "{} KiB", self.0 / KIB)
        }
        else {
            write!(f, "{} B", self.0)
        }
    }
}

/// Line-buffered writer over `write_chunked`. ConsoleWriter goes through
/// the kernel serial back-channel; this one writes through the
/// framebuffer scrollback path that the rest of the console uses.
struct LineWriter {
    buf: [u8; 256],
    len: usize,
}

impl LineWriter {
    const fn new() -> Self {
        Self {
            buf: [0u8; 256],
            len: 0,
        }
    }
    fn flush(&mut self) {
        if self.len > 0 {
            write_chunked(&self.buf[..self.len]);
            self.len = 0;
        }
    }
}

impl core::fmt::Write for LineWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len == self.buf.len() {
                self.flush();
            }
            self.buf[self.len] = b;
            self.len += 1;
        }
        Ok(())
    }
}

/// Tokenize `line` on whitespace, slurp the first token as the ELF
/// path, and spawn it via `create_process_v2` with `stdout_capture=1`
/// so the child's `console_write` bytes route into *this* console's
/// pane (option-1 stdout capture — no fds, no pipes), then block on
/// `wait_pid` for the child's exit status. argv[0] is the program
/// path (POSIX convention); subsequent tokens are passed verbatim.
/// No quoting, no escapes — typing `eza "a b"` lands as three argv
/// entries. Foreground execution; the `&` background-spawn case is
/// a future addition.
fn exec_path(line: &str) {
    let tokens: Vec<&[u8]> = line.split_whitespace().map(str::as_bytes).collect();
    let path_bytes = match tokens.first() {
        Some(b) => *b,
        None => return,
    };
    let path = match core::str::from_utf8(path_bytes) {
        Ok(s) => s,
        Err(_) => {
            write_chunked(b"exec: path must be utf-8\n");
            return;
        }
    };

    // Pack argv into the wire format the kernel + orbit-rt expect.
    // 4 KiB scratch matches `ARGV_BLOB_MAX`; the kernel reads exactly
    // `argv_len` bytes, so no page alignment needed.
    let mut argv_buf = [0u8; orbit_abi::argv::ARGV_BLOB_MAX];
    let argv_len = match orbit_abi::argv::pack(&tokens, &mut argv_buf) {
        Some(n) => n,
        None => {
            write_chunked(b"exec: argv exceeds 4 KiB\n");
            return;
        }
    };

    let args = CreateProcessV2Args {
        // Path mode — the kernel does fs.open + X-bit check + fs.read
        // for us via the spawn_path_* fields below. Bytes mode would
        // EPERM here anyway since console is SHELL role, not LOADER.
        elf_vaddr: 0,
        elf_len: 0,
        allowed_affinity: 0,
        affinity: 0,
        // INHERIT keeps the parent's role + perms verbatim, matching
        // the fork-shaped semantic shells expect. Role-aware spawn
        // surfaces would name a concrete RoleId here.
        target_role: role::INHERIT,
        flags: 0,
        request_perms: 0,
        request_allowed_perms: 0,
        cwd_vaddr: 0,
        cwd_len: 0,
        argv_vaddr: argv_buf.as_ptr() as usize,
        argv_len,
        envp_vaddr: 0,
        // Route the child's `console_write` bytes into the console's
        // own pane so output lands inline like a normal shell.
        stdout_capture: 1,
        _pad2: 0,
        // Identity inherits — the console isn't LOADER and the kernel
        // would EPERM any non-inherit attempt.
        setuid_uid: CreateProcessV2Args::INHERIT_ID,
        setuid_gid: CreateProcessV2Args::INHERIT_ID,
        setlogin_vaddr: 0,
        setlogin_len: 0,
        groups_vaddr: 0,
        groups_count: 0,
        // Path mode — the console runs as SHELL role, not LOADER, so
        // bytes-mode would EPERM. The kernel does the fs.open + X-bit
        // check + fs.read for us, which also means we can drop the
        // userspace fs_open + fs_read of the binary above.
        spawn_path_vaddr: path.as_ptr() as usize,
        spawn_path_len: path.len(),
    };
    let pid = match create_process_v2(&args) {
        Ok(p) => p,
        Err(e) => {
            let mut w = LineWriter::new();
            let _ = writeln!(w, "exec: create_process_v2 {}: errno {}", path, e.0);
            w.flush();
            return;
        }
    };
    match wait_pid(pid) {
        Ok(code) => {
            let mut w = LineWriter::new();
            let _ = writeln!(w, "exec: {} exited {}", path, code);
            w.flush();
        }
        Err(e) => {
            let mut w = LineWriter::new();
            let _ = writeln!(w, "exec: wait_pid {} errored: errno {}", pid, e.0);
            w.flush();
        }
    }
}

/// Run a single command line (already stripped of its trailing `\n`).
/// Empty input → no-op (matches dash behavior — re-prompt only).
fn dispatch(line: &[u8]) {
    let s = match core::str::from_utf8(line) {
        Ok(s) => s.trim(),
        Err(_) => {
            write_chunked(b"console: input was not utf-8\n");
            return;
        }
    };
    if s.is_empty() {
        return;
    }
    // Anything starting with `/` is an explicit path to exec — no
    // PATH search yet. Matches what users type when they want to
    // pin a specific file.
    if s.starts_with('/') {
        exec_path(s);
        return;
    }
    let mut it = s.splitn(2, char::is_whitespace);
    let cmd = it.next().unwrap_or("");
    let args = it.next().unwrap_or("").trim_start();
    match cmd {
        "echo" => {
            write_chunked(args.as_bytes());
            write_chunked(b"\n");
        }
        "help" => {
            write_chunked(b"builtins: echo <text>, help, clear, stats, syscall-stats\n");
            write_chunked(b"exec: type a path (e.g. /bin/eza), or a bare name to look up in /bin (e.g. eza /tmp)\n");
        }
        "clear" => {
            // Form-feed: compositor clears this source's scrollback.
            write_chunked(b"\x0c");
        }
        "stats" => stats_cmd(),
        "syscall-stats" => syscall_stats_cmd(),
        // Fallback for bare names: rewrite `cmd args...` as
        // `/bin/cmd args...` and try to exec. No real PATH search —
        // /bin is the only directory consulted today. exec_path will
        // surface ENOENT etc. cleanly if the binary isn't there.
        _ => {
            let mut line = alloc::string::String::with_capacity(5 + s.len());
            line.push_str("/bin/");
            line.push_str(s);
            exec_path(&line);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    write_chunked(PROMPT);

    let mut buf = [0u8; 64];
    let mut editor = LineEditor::new();
    loop {
        let n = match read_stdin(buf.as_mut_ptr() as usize, buf.len(), 0) {
            Ok(n) => n,
            // Blocking read shouldn't EAGAIN in practice, and other
            // errors (EFAULT/EINVAL/EBUSY) shouldn't happen with a
            // stable stack-resident buffer. Yield briefly and retry.
            Err(_) => {
                let _ = sleep_ms(16);
                continue;
            }
        };
        for &b in &buf[..n] {
            if let Some(line) = editor.feed(b) {
                dispatch(&line);
                write_chunked(PROMPT);
            }
        }
    }
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "console panic: {p}");
    w.flush();
    exit(isize::MIN);
}
