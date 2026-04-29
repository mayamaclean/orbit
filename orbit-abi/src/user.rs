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
use crate::stats::ProcessStats;
use crate::syscall;
use crate::syscall_stats::{SyscallEntry, SyscallStatsHeader};

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

/// Six-argument syscall returning an `isize` in `a0`. RISC-V's
/// calling convention has plenty of arg registers (a0..a7); used by
/// `create_process_with_argv` so the kernel can read elf + affinity
/// + argv-blob fields in one trap without the caller marshalling
/// them through user memory first.
#[inline]
pub unsafe fn ecall6(
    code: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
) -> isize {
    let r: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            in("a4") arg3,
            in("a5") arg4,
            in("a6") arg5,
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

/// Zero-argument syscall returning a pair of `isize` in `a0, a1`. Used
/// by `get_affinity` to hand back (current, allowed) in one trap.
#[inline]
pub unsafe fn ecall0_ret2(code: usize) -> (isize, isize) {
    let r0: isize;
    let r1: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            lateout("a0") r0,
            lateout("a1") r1,
        );
    }
    (r0, r1)
}

/// One-argument syscall returning a pair of `isize` in `a0, a1`.
/// Used by `wait_pid` to return `(status_or_errno, exit_code)` in one
/// trap — keeps exit-code encoding orthogonal to the errno-via-negative
/// convention.
#[inline]
pub unsafe fn ecall1_ret2(code: usize, arg0: usize) -> (isize, isize) {
    let r0: isize;
    let r1: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            lateout("a0") r0,
            lateout("a1") r1,
        );
    }
    (r0, r1)
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

/// Push a `WakeEvent::Net` so the kernel net thread wakes immediately
/// (instead of waiting up to its 10 ms heartbeat) — useful after a
/// NetCh ring-write where the kernel needs to drain an increment or
/// stage a slice for us. Then optionally park the caller for up to
/// `timeout_ms` milliseconds, returning early if the kernel marks the
/// caller's thread for wake-up first (e.g. via `WakeEvent::Pid` from
/// `update_tcp`'s `outcome.ring_progress`).
///
/// `timeout_ms == 0` skips the park: pure notification, returns
/// immediately. `timeout_ms > 0` is `sleep_ms(timeout_ms)` with the
/// notification bundled in. Same one-hour cap as `sleep_ms`.
///
/// Replaces the EAGAIN-park-then-syscall-again pattern with a single
/// syscall that can return as soon as the channel state changes,
/// avoiding the 10 ms timer-tick floor for request/response workloads.
#[inline]
pub fn nc_yield(timeout_ms: usize) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::NC_YIELD, timeout_ms) }).map(|_| ())
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
/// as a socket of `sock_type`. `bind_spec` is the `BindSpec::pack()`
/// representation of the sticky binding the kernel will latch for this
/// channel — the kernel rejects malformed packings at the syscall
/// boundary, so the wrapper just forwards the bits.
///
/// On success returns `Ok((user_va, fd))` — the VA the region landed at
/// and the Fd the kernel assigned (pass to [`close_handle`] to tear the
/// channel down).
#[inline]
pub fn create_netch(
    vaddr_hint: usize,
    region_size: usize,
    sock_type: usize,
    bind_spec: usize,
) -> Result<(usize, u32), Errno> {
    let (r0, r1) = unsafe {
        ecall4_ret2(syscall::CREATE_NETCH, vaddr_hint, region_size, sock_type, bind_spec)
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

/// Spawn a sibling thread in the calling process. `entry` is a function
/// pointer in the caller's address space; the new thread starts there
/// with a fresh stack and its own trap frame, sharing satp / heap /
/// open handles with the parent.
///
/// `allowed_affinity` and `affinity` follow the same rules as
/// [`create_process`] — pass `0` for either to mean "default to the
/// calling thread's allowed mask." `affinity` must be a subset of
/// `allowed_affinity` once both are resolved.
///
/// Returns the new tid on success. Async manager round-trip; the
/// caller blocks until the manager has wired the thread into the
/// scheduler.
#[inline]
pub fn create_thread(
    entry: extern "C" fn() -> !,
    allowed_affinity: u64,
    affinity: u64,
) -> Result<u32, Errno> {
    Errno::from_ret(unsafe { ecall3(
        syscall::CREATE_THREAD,
        entry as usize,
        allowed_affinity as usize,
        affinity as usize,
    )})
        .map(|t| t as u32)
}

/// Narrow the calling thread's per-hart eligibility mask. The new mask
/// must be non-zero and a subset of `allowed_affinity` (queryable via
/// [`get_affinity`]). Returns `Ok(())` on success, `Err(EINVAL)` if
/// `mask == 0`, `Err(EPERM)` if the mask escapes the cap.
///
/// Takes effect on the next scheduler dispatch; doesn't preempt the
/// caller. If the caller's current hart is no longer in `mask`, it
/// finishes its quantum there and migrates afterwards.
#[inline]
pub fn set_affinity(mask: u64) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::SET_AFFINITY, mask as usize) }).map(|_| ())
}

/// Return the hart id the caller is currently executing on. Useful for
/// self-checking after `set_affinity` (next-dispatch confirmation) and
/// for log markers in multi-hart tests. Cheap — pure read of the
/// per-hart context from the kernel, no scheduling decisions.
#[inline]
pub fn get_hart_id() -> u32 {
    let r = unsafe { ecall1(syscall::GET_HART_ID, 0) };
    r as u32
}

/// Return the calling process's pid. Stable for the process's
/// lifetime — unlike [`get_hart_id`], which changes whenever the
/// scheduler migrates the thread. Backs `std::process::id()`.
#[inline]
pub fn getpid() -> u16 {
    let r = unsafe { ecall1(syscall::GETPID, 0) };
    r as u16
}

/// Return the calling thread's tid. System-wide unique (not
/// per-process), stable for the thread's lifetime. Backs
/// `std::thread::current().id()`.
#[inline]
pub fn gettid() -> u32 {
    let r = unsafe { ecall1(syscall::GETTID, 0) };
    r as u32
}

/// Spawn a child process with command-line arguments. Same shape as
/// [`create_process`] otherwise; `argv_blob` is the packed bytes
/// described in [`crate::argv`] (header + offsets + string table).
/// Pass an empty slice for arg-less spawn (matches `create_process`).
///
/// # Safety
/// `elf_ptr`/`elf_len` must point to a valid mapped ELF range.
/// `argv_blob` must be a self-contained packed blob — the kernel
/// validates its `argc` field but trusts the offsets/strings to land
/// inside `argv_blob.len()`. Malformed offsets surface as `EINVAL`.
#[inline]
pub fn create_process_with_argv(
    elf_ptr: *const u8,
    elf_len: usize,
    allowed_affinity: u64,
    affinity: u64,
    argv_blob: &[u8],
) -> Result<u16, Errno> {
    Errno::from_ret(unsafe {
        ecall6(
            syscall::CREATE_PROCESS_EX,
            elf_ptr as usize,
            elf_len,
            allowed_affinity as usize,
            affinity as usize,
            argv_blob.as_ptr() as usize,
            argv_blob.len(),
        )
    })
    .map(|p| p as u16)
}

/// Return the user VA where the kernel mapped this process's argv
/// blob, or `0` if no argv was provided. Always reads the same value
/// for a given process — orbit-rt's startup caches the result. v1
/// returns either `0` or [`crate::layout::USER_ARGV_BASE`].
#[inline]
pub fn argv_envp() -> usize {
    let r = unsafe { ecall1(syscall::ARGV_ENVP, 0) };
    r as usize
}

/// Block the caller until child process `pid` exits, then return the
/// child's exit code. Errnos:
/// - `ECHILD` — `pid` doesn't exist (never existed or already reaped).
///   v1 has no zombies; a child whose parent never waited is reaped
///   immediately on exit, so a late `wait_pid` always sees ECHILD.
/// - `EPERM`  — caller is not the parent of `pid`.
/// - `EINVAL` — `pid == 0` or `pid == self`.
/// - `EBUSY`  — another thread already parked on this child (v1 is
///   single-waiter; futex lifts this).
///
/// On success returns the exit code passed to the child's `exit()`,
/// or `-1` if the child died from a fault rather than a clean exit.
/// The exit code lands in a separate register from the success/errno
/// signal so negative exit codes don't collide with the errno-as-
/// negative convention.
#[inline]
pub fn wait_pid(pid: u16) -> Result<i32, Errno> {
    let (r0, r1) = unsafe { ecall1_ret2(syscall::WAIT_PID, pid as usize) };
    Errno::from_ret(r0).map(|_| r1 as i32)
}

/// Absolute monotonic microseconds since system boot.
///
/// The base is opaque — only differences are meaningful. Backed by a
/// `csrr time` on the kernel side (RISC-V `time` runs at 10 MHz on
/// the QEMU virt machine; the syscall divides by 10 to give μs).
///
/// Use case: latency micro-benchmarks (sleep accuracy, RTT, throughput
/// timing) that don't want platform-coupled raw ticks. For wallclock,
/// add a future `get_realtime` syscall — `get_micros` is monotonic
/// only, no time-of-day offset.
///
/// A direct `csrr time` from U-mode (the §13a.4 zero-syscall idea)
/// is gated behind a CSR-emulation handler we don't have yet —
/// QEMU's virt machine traps `rdtime` to M-mode for emulation even
/// with `scounteren.TM` set, because the `time` CSR is spec'd as a
/// view onto CLINT mtime, not a hardware register. The syscall is
/// the canonical clock until that lands.
#[inline]
pub fn get_micros() -> u64 {
    let r = unsafe { ecall1(syscall::GET_MICROS, 0) };
    r as u64
}

/// Return `(current, allowed)` for the calling thread's affinity mask.
/// Modeled on Windows's `GetProcessAffinityMask` — the immutable cap is
/// returned alongside the current value so userspace can pick a valid
/// sub-mask without trial-and-error.
#[inline]
pub fn get_affinity() -> (u64, u64) {
    let (cur, allowed) = unsafe { ecall0_ret2(syscall::GET_AFFINITY) };
    (cur as u64, allowed as u64)
}

/// Spawn a new process from an in-memory ELF image. `elf_ptr`/`elf_len`
/// describe a contiguous readable region in the caller's address space;
/// the kernel copies the bytes out, parses the ELF, and creates a
/// process whose first thread enters at `e_entry` with the default
/// stack size. Returns the new process's pid on success.
///
/// `allowed_affinity` caps the harts the child may ever run on (the
/// child's `set_affinity` rejects anything outside this mask).
/// `affinity` is the initial mask the scheduler uses to pick a hart;
/// it must be a subset of `allowed_affinity`. Pass `0` for either to
/// mean "default to all harts" — the common case.
#[inline]
pub fn create_process(
    elf_ptr: *const u8,
    elf_len: usize,
    allowed_affinity: u64,
    affinity: u64,
) -> Result<u16, Errno> {
    Errno::from_ret(unsafe { ecall4(
        syscall::CREATE_PROCESS,
        elf_ptr as usize,
        elf_len,
        allowed_affinity as usize,
        affinity as usize,
    )})
        .map(|p| p as u16)
}

/// `fs_open(path, flags)` — resolve `path` against the mounted
/// filesystem and return an `Fd`. v1 is read-only tarfs; pass
/// [`crate::fs::OPEN_RDONLY`] (= 0) for `flags`. Errnos: `ENOENT`
/// (path not in archive), `EINVAL` (path too long / empty),
/// `EFAULT` (bad pointer), `EAGAIN` (manager work ring full).
#[inline]
pub fn fs_open(path: &str, flags: usize) -> Result<u32, Errno> {
    Errno::from_ret(unsafe {
        ecall3(syscall::FS_OPEN, path.as_ptr() as usize, path.len(), flags)
    })
    .map(|fd| fd as u32)
}

/// `fs_read(fd, buf)` — read one sector at the fd's current offset.
/// `buf.len()` must equal 512; the buffer must not straddle a 4 KiB
/// page boundary (sector-align it). Returns bytes considered valid
/// (up to 512); 0 at EOF. Auto-advances the fd's offset by 512 on
/// success, even at the file tail. Trailing bytes past the file size
/// in the final sector are zero-padded by the on-disk archive.
#[inline]
pub fn fs_read(fd: u32, buf: &mut [u8]) -> Result<usize, Errno> {
    Errno::from_ret(unsafe {
        ecall3(
            syscall::FS_READ,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
        )
    })
}

/// `fs_stat(path, &mut stat)` — fill `stat` with metadata for the
/// named entry. Layout matches Linux's generic-arch `struct stat`
/// (see [`crate::fs::Stat`]).
#[inline]
pub fn fs_stat(path: &str, stat: &mut crate::fs::Stat) -> Result<(), Errno> {
    Errno::from_ret(unsafe {
        ecall3(
            syscall::FS_STAT,
            path.as_ptr() as usize,
            path.len(),
            stat as *mut _ as usize,
        )
    })
    .map(|_| ())
}

/// Snapshot per-process and kernel-wide accounting. The wrapper owns
/// the buffer so callers don't have to think about ABI sizing — pass
/// `()`, get a struct back. On a kernel newer than this build,
/// trailing fields are silently dropped (kernel honours the caller's
/// smaller buffer); on a kernel older than this build, the local
/// struct is zero-initialised first so unwritten fields read as 0 and
/// the `size` prefix tells the caller how many bytes are valid.
#[inline]
pub fn query_stats() -> Result<ProcessStats, Errno> {
    let mut s = ProcessStats::default();
    let n = unsafe {
        ecall2(
            syscall::QUERY_STATS,
            &mut s as *mut _ as usize,
            core::mem::size_of::<ProcessStats>(),
        )
    };
    Errno::from_ret(n).map(|_| s)
}

/// Snapshot the system-wide per-syscall latency table into `buf`.
/// Returns `(header, entries)` borrowing from `buf`.
///
/// `header.count` may be smaller (older kernel) or larger (newer
/// kernel, but the wrapper-side buffer was too small to hold all
/// records) than the local [`Sysno::COUNT`](crate::Sysno::COUNT).
/// Iterate `min(header.count, entries.len())` and look up by
/// [`Sysno::ordinal`](crate::Sysno::ordinal).
///
/// Pass a buffer at least
/// [`syscall_stats::payload_size()`](crate::syscall_stats::payload_size)
/// bytes for a complete snapshot.
#[inline]
pub fn query_syscall_stats(
    buf: &mut [u8],
) -> Result<(SyscallStatsHeader, &[SyscallEntry]), Errno> {
    let n = unsafe {
        ecall2(
            syscall::QUERY_SYSCALL_STATS,
            buf.as_mut_ptr() as usize,
            buf.len(),
        )
    };
    let written = Errno::from_ret(n)? as usize;
    let hdr_size = core::mem::size_of::<SyscallStatsHeader>();
    if written < hdr_size {
        return Err(Errno(crate::errno::EINVAL));
    }
    // SAFETY: kernel guarantees `written` bytes of valid header+entries
    // were written into `buf`, and the layout is `#[repr(C)]` with no
    // padding before the entries array.
    let hdr = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const SyscallStatsHeader) };
    let entries_bytes = written - hdr_size;
    let entries_len = entries_bytes / core::mem::size_of::<SyscallEntry>();
    let entries = unsafe {
        core::slice::from_raw_parts(
            buf.as_ptr().add(hdr_size) as *const SyscallEntry,
            entries_len,
        )
    };
    Ok((hdr, entries))
}

pub struct ConsoleWriter {
    buf: [u8; 256],
    len: usize,
}

impl ConsoleWriter {
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

impl core::fmt::Write for ConsoleWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len >= self.buf.len() { self.flush(); }
            self.buf[self.len] = b;
            self.len += 1;
        }
        Ok(())
    }
}

/// Like [`ConsoleWriter`] but flushes via the `serial_print` syscall
/// instead of `console_write`. Output goes straight to the kernel's
/// serial back-channel, **not** to the per-process framebuffer
/// scrollback — useful when a short-lived process needs its output
/// to survive past its exit (the scrollback compositor may not have
/// rendered the message before the source gets torn down) or when
/// you want output to appear on the harness `-serial` log without
/// involving the gpu thread at all.
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
        // serial_print rejects len > PAGE_SIZE; our buffer is 256 B so
        // a single call is always safe. Errors (EINVAL on non-utf8,
        // EFAULT on bad ptr) shouldn't happen with stack-resident
        // buffer + correct callers — drop on the floor.
        let _ = serial_print(self.buf.as_ptr() as usize, self.len);
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

/// `logln!`-shaped macro that writes through `SerialWriter` instead
/// of `ConsoleWriter`. Use when output needs to survive on the
/// kernel serial log even if the calling process exits before the
/// framebuffer compositor can render the message.
#[macro_export]
macro_rules! serialln {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let mut w = $crate::user::SerialWriter::new();
        let _ = writeln!(w, $($arg)*);
        w.flush();
    }};
}

#[macro_export]
macro_rules! logln {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let mut w = $crate::user::ConsoleWriter::new();
        let _ = writeln!(w, $($arg)*);
        w.flush();
    }};
}
