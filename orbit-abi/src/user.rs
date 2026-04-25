//! User-side syscall wrappers.
//!
//! Thin `ecall` shims over the syscall numbers in [`crate::syscall`].
//! Every user process that links this module gets the same surface — keep
//! the signatures synchronised with the dispatch arms in kmain's `s_trap`
//! and the per-syscall ABI docs in the sibling modules
//! ([`crate::mmap`], [`crate::net`]).
//!
//! Gated on `target_arch = "riscv64"` because inline `ecall` with `aN`
//! register operands doesn't parse on other targets — orbit-abi's host
//! unit tests wouldn't compile otherwise.

#![cfg(target_arch = "riscv64")]

use core::arch::asm;

use crate::errno::{Errno, EAGAIN};
use crate::syscall;

// --- low-level ecall primitives ------------------------------------------

/// Noreturn syscall with one argument. `a0 = code`, `a1 = arg0`. Used for
/// exit().
#[inline]
pub unsafe fn ecall1_noreturn(code: usize, arg0: usize) -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            options(noreturn),
        );
    }
}

/// Single-argument syscall returning an `isize` in `a0`.
#[inline]
pub unsafe fn ecall1(code: usize, arg0: usize) -> isize {
    let r: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            lateout("a0") r,
        );
    }
    r
}

/// Two-argument syscall returning an `isize` in `a0`.
#[inline]
pub unsafe fn ecall2(code: usize, arg0: usize, arg1: usize) -> isize {
    let r: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            lateout("a0") r,
        );
    }
    r
}

/// Three-argument syscall returning an `isize` in `a0`.
#[inline]
pub unsafe fn ecall3(code: usize, arg0: usize, arg1: usize, arg2: usize) -> isize {
    let r: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            lateout("a0") r,
        );
    }
    r
}

/// Four-argument syscall returning an `isize` in `a0`.
#[inline]
pub unsafe fn ecall4(code: usize, arg0: usize, arg1: usize, arg2: usize, arg3: usize) -> isize {
    let r: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            in("a4") arg3,
            lateout("a0") r,
        );
    }
    r
}

/// Four-argument syscall returning a pair of `isize` in `a0, a1`. Used by
/// create_netch to hand back `(vaddr, fd)` in one trap without a user
/// out-pointer — the kernel would otherwise have to resolve it through
/// KDMAP or a transient page window.
#[inline]
pub unsafe fn ecall4_ret2(
    code: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
) -> (isize, isize) {
    let r0: isize;
    let r1: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            in("a4") arg3,
            lateout("a0") r0,
            lateout("a1") r1,
        );
    }
    (r0, r1)
}

// --- high-level wrappers -------------------------------------------------

/// Terminate the current process with `code`. Never returns.
#[inline]
pub fn exit(code: isize) -> ! {
    unsafe { ecall1_noreturn(syscall::EXIT, code as usize) }
}

/// Print `len` bytes starting at `ptr` through the kernel's tagged
/// serial path. `Ok(n)` is the byte count the kernel acknowledged
/// (zero on the current shape — the call doesn't return a count, just
/// success/failure).
#[inline]
pub fn serial_print(ptr: usize, len: usize) -> Result<usize, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::SERIAL_PRINT, ptr, len) })
}

/// Append `len` bytes starting at `ptr` to the calling process's
/// framebuffer scrollback. Bytes are chunked at 4 KiB (POSIX
/// `PIPE_BUF` atomicity); oversize writes are rejected with `EINVAL`.
/// Returns the byte count accepted on success.
#[inline]
pub fn console_write(ptr: usize, len: usize) -> Result<usize, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::CONSOLE_WRITE, ptr, len) })
}

/// `flags` bit for [`read_stdin`]: return `EAGAIN` immediately when
/// the ring is empty instead of blocking until a keystroke arrives.
pub const READ_STDIN_NONBLOCK: usize = 1;

/// Read up to `len` bytes from the calling process's stdin ring into
/// the buffer at `ptr`. Stdin is fed by the kernel's input
/// dispatcher when this process is the active framebuffer source.
///
/// Returns the byte count drained on success. With `flags == 0` the
/// call blocks (the kernel parks the thread on a completion handle
/// and resumes it on the next keystroke); with
/// `flags & READ_STDIN_NONBLOCK` an empty ring returns `Err(EAGAIN)`.
///
/// Other errors:
/// - `EINVAL` — `len == 0` or `len > 4 KiB`.
/// - `EFAULT` — `ptr` doesn't translate under the caller's satp.
/// - `EBUSY`  — another reader is already parked on this process's
///   stdin (single-reader model violated).
#[inline]
pub fn read_stdin(ptr: usize, len: usize, flags: usize) -> Result<usize, Errno> {
    Errno::from_ret(unsafe { ecall3(syscall::READ_STDIN, ptr, len, flags) })
}

/// Block the calling thread for `ms` milliseconds. Kernel caps the
/// delay at one hour; requests at/above the cap return `Err(EINVAL)`.
#[inline]
pub fn sleep_ms(ms: usize) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::SLEEP_MS, ms) }).map(|_| ())
}

/// Ask the kernel for a user-accessible region at `hint_va` of `len`
/// bytes. `share_with_kernel` selects the backing pool (roadmap §3):
/// `false` → `user_pages` (no KDMAP alias), `true` → `kernel_pages`.
/// Returns the mapped VA on success.
///
/// # Safety
/// Caller must not already have a mapping covering `[hint_va, hint_va+len)`.
#[inline]
pub unsafe fn mmap(hint_va: usize, len: usize, perms: usize, share_with_kernel: bool) -> Result<usize, Errno> {
    Errno::from_ret(unsafe {
        ecall4(
            syscall::MMAP,
            hint_va,
            len,
            perms,
            share_with_kernel as usize,
        )
    })
}

/// Create a NetChannel region of `region_size` bytes at `vaddr_hint`,
/// as a socket of `sock_type`. On success returns `Ok((user_va, fd))`
/// — the VA the region landed at and the Fd the kernel assigned (pass
/// to [`close_handle`] to tear the channel down).
#[inline]
pub fn create_netch(
    vaddr_hint: usize,
    region_size: usize,
    sock_type: usize,
) -> Result<(usize, u32), Errno> {
    let (r0, r1) = unsafe {
        ecall4_ret2(syscall::CREATE_NETCH, vaddr_hint, region_size, sock_type, 0)
    };
    Errno::from_ret(r0).map(|va| (va, r1 as u32))
}

/// Release a handle returned by [`create_netch`]. Kernel revokes the
/// user mapping (subsequent accesses at the old VA fault) before
/// dropping its strong ref.
#[inline]
pub fn close_handle(fd: u32) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::CLOSE_HANDLE, fd as usize) }).map(|_| ())
}

/// Spawn a new process from an in-memory ELF image. `elf_ptr`/`elf_len`
/// describe a contiguous readable region in the caller's address space;
/// the kernel copies the bytes out, parses the ELF, and creates a
/// process whose first thread enters at `e_entry` with the default
/// stack size. Returns the new process's pid on success.
#[inline]
pub fn create_process(elf_ptr: *const u8, elf_len: usize) -> Result<u16, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::CREATE_PROCESS, elf_ptr as usize, elf_len) })
        .map(|p| p as u16)
}

pub struct SerialWriter {
    buf: [u8; 256],
    len: usize,
}

impl SerialWriter {
    pub const fn new() -> Self { Self { buf: [0u8; 256], len: 0 } }
    pub fn flush(&mut self) {
        if self.len == 0 {
            return;
        }
        // The kernel's CONSOLE_RING is small (8 slots, shared with
        // kernel ktrace). A burst of prints can fill it, in which case
        // console_write returns EAGAIN. Yield via sleep_ms(0) and
        // retry so output isn't silently dropped. Bounded so a
        // permanently-broken consumer doesn't deadlock the writer.
        const MAX_RETRIES: usize = 64;
        let mut attempts = 0;
        loop {
            match console_write(self.buf.as_ptr() as usize, self.len) {
                Ok(_) => break,
                Err(Errno(e)) if e == EAGAIN && attempts < MAX_RETRIES => {
                    attempts += 1;
                    let _ = sleep_ms(0);
                }
                Err(_) => break,
            }
        }
        self.len = 0;
    }
}

impl core::fmt::Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len >= self.buf.len() { self.flush(); }
            self.buf[self.len] = b;
            self.len += 1;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! logln {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let mut w = SerialWriter::new();
        let _ = writeln!(w, $($arg)*);
        w.flush();
    }};
}
