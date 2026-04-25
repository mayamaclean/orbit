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

use orbit_abi::{
    logln,
    user::{console_write, exit, read_stdin, sleep_ms, SerialWriter},
};

const PROMPT: &[u8] = b"console$ ";

/// One PIPE_BUF chunk. `console_write` rejects `len > 4096` with EINVAL,
/// so any longer payload has to be split before the syscall.
const CHUNK: usize = 4096;

/// Send `bytes` through `console_write`, splitting at the kernel's
/// 4 KiB PIPE_BUF boundary. Errors (ring full, etc.) are dropped on
/// the floor — `console_write` already retries internally for EAGAIN
/// via the SerialWriter shim used elsewhere; the console treats output
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
        Self { buf: Vec::new(), esc: 0 }
    }

    /// Feed one byte from the stdin ring. Returns `Some(line)` on `\n`
    /// (caller dispatches; the editor's buffer is reset by the take).
    fn feed(&mut self, b: u8) -> Option<Vec<u8>> {
        match self.esc {
            1 => { self.esc = if b == b'[' { 2 } else { 0 }; return None; }
            2 => { self.esc = 0; return None; }
            _ => {}
        }
        match b {
            0x1b => { self.esc = 1; None }
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

/// Run a single command line (already stripped of its trailing `\n`).
/// Empty input → no-op (matches dash behavior — re-prompt only).
fn dispatch(line: &[u8]) {
    let s = match core::str::from_utf8(line) {
        Ok(s) => s.trim(),
        Err(_) => { write_chunked(b"console: input was not utf-8\n"); return; }
    };
    if s.is_empty() { return; }
    let mut it = s.splitn(2, char::is_whitespace);
    let cmd = it.next().unwrap_or("");
    let args = it.next().unwrap_or("").trim_start();
    match cmd {
        "echo" => {
            write_chunked(args.as_bytes());
            write_chunked(b"\n");
        }
        "help" => {
            write_chunked(b"builtins: echo <text>, help, clear\n");
        }
        "clear" => {
            // Form-feed: compositor clears this source's scrollback.
            write_chunked(b"\x0c");
        }
        _ => {
            write_chunked(b"unknown command: ");
            write_chunked(cmd.as_bytes());
            write_chunked(b"\n");
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    logln!("console: up");
    write_chunked(PROMPT);

    let mut buf = [0u8; 64];
    let mut editor = LineEditor::new();
    loop {
        let n = match read_stdin(buf.as_mut_ptr() as usize, buf.len(), 0) {
            Ok(n) => n,
            // Blocking read shouldn't EAGAIN in practice, and other
            // errors (EFAULT/EINVAL/EBUSY) shouldn't happen with a
            // stable stack-resident buffer. Yield briefly and retry.
            Err(_) => { let _ = sleep_ms(10); continue; }
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
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "console panic: {p}");
    w.flush();
    exit(isize::MIN);
}
