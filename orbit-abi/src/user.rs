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

use crate::errno::{EAGAIN, Errno};
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
/// calling convention has plenty of arg registers (a0..a7), so wide
/// spawn-shaped syscalls can pass elf + affinity + blob fields in one
/// trap without marshalling them through user memory first.
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

/// Seven-argument syscall returning an `isize` in `a0`. Saturates the
/// RISC-V arg-register file (a0..a7 = code + 7 args). Used by
/// `create_process_with_argv_envp` to carry elf + affinity + argv +
/// envp pointers in one trap; envp blob length isn't passed because
/// the kernel always reads a full page at the envp VA — see
/// [`crate::layout::USER_ENVP_BASE`] and [`crate::envp`].
#[inline]
pub unsafe fn ecall7(
    code: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
    arg6: usize,
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
            in("a7") arg6,
            lateout("a0") r,
        );
    }
    r
}

/// Five-argument syscall returning an `isize` in `a0`. Used by
/// `fb_present(handle, x, y, w, h)` so the rect doesn't have to be
/// packed into a smaller arg.
#[inline]
pub unsafe fn ecall5(
    code: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
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
            lateout("a0") r,
        );
    }
    r
}

/// Three-argument syscall returning a pair of `isize` in `a0, a1`. Used
/// by `fb_surface_create(w, h, format)` to hand back `(handle, user_va)`
/// in one trap.
#[inline]
pub unsafe fn ecall3_ret2(code: usize, arg0: usize, arg1: usize, arg2: usize) -> (isize, isize) {
    let r0: isize;
    let r1: isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            lateout("a0") r0,
            lateout("a1") r1,
        );
    }
    (r0, r1)
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

/// Terminate the current process with `code`. Never returns. POSIX
/// `_exit(2)` shape — sibling threads of the calling process are
/// torn down by the kernel as part of the exit-group sweep.
///
/// To exit only the calling thread (e.g. from a worker's trampoline),
/// use [`thread_exit`] instead.
#[inline]
pub fn exit(code: isize) -> ! {
    unsafe { ecall1_noreturn(syscall::EXIT, code as usize) }
}

/// Terminate the calling thread, leaving sibling threads of the same
/// process running. Never returns. Used by std's thread trampoline
/// when a worker's closure returns; status surfaces through the
/// joiner's futex word, not the kernel.
#[inline]
pub fn thread_exit() -> ! {
    unsafe { ecall1_noreturn(syscall::THREAD_EXIT, 0) }
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
/// call blocks (the kernel parks the thread and resumes it on the next
/// keystroke); with `flags & READ_STDIN_NONBLOCK` an empty ring returns
/// `Err(EAGAIN)`.
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

/// Drain up to `count` `KeyEvent`s from the calling process's
/// structured-event ring into `buf`. Companion to [`read_stdin`] —
/// same producer, different encoding (no UTF-8 + ANSI round-trip).
///
/// `timeout_ms` selects the park shape (no NONBLOCK):
/// - `0` — peek; drain available, return synchronously with 0 if
///   the ring is empty.
/// - `usize::MAX` ([`READ_KEY_EVENT_INDEFINITE`]) — block until the
///   next event.
/// - any value in `1..(60*60*1000)` — block up to that many
///   milliseconds; return early on the next event.
///
/// `flags & READ_KEY_EVENT_NONBLOCK` overrides `timeout_ms` and
/// returns `EAGAIN` immediately on empty.
///
/// Errors:
/// - `EINVAL` — `count == 0`, buffer exceeds one page, or
///   `timeout_ms` is in the rejected band (≥ 1 hour and not the
///   sentinel).
/// - `EFAULT` — `buf` doesn't translate under the caller's satp.
/// - `EAGAIN` — empty + nonblock.
/// - `EBUSY`  — another reader is parked on the ring.
#[inline]
pub fn read_key_event(
    buf: *mut crate::input::KeyEvent,
    count: usize,
    flags: usize,
    timeout_ms: usize,
) -> Result<usize, Errno> {
    Errno::from_ret(unsafe {
        ecall4(
            syscall::READ_KEY_EVENT,
            buf as usize,
            count,
            flags,
            timeout_ms,
        )
    })
}

/// Block the calling thread for `ms` milliseconds. Kernel caps the
/// delay at one hour; requests at/above the cap return `Err(EINVAL)`.
#[inline]
pub fn sleep_ms(ms: usize) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::SLEEP_MS, ms) }).map(|_| ())
}

/// Push a `WakeEvent::Net` so the kernel net thread wakes immediately
/// (instead of waiting up to its ~100 ms heartbeat) — useful after a
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
pub fn ch_yield(timeout_ms: usize) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::CH_YIELD, timeout_ms) }).map(|_| ())
}

/// Ask the kernel for a user-accessible region at `hint_va` of `len`
/// bytes. `share_with_kernel` selects the backing pool:
/// `false` → `user_pages` (no KDMAP alias), `true` → `kernel_pages`.
/// Returns the mapped VA on success.
///
/// # Safety
/// Caller must not already have a mapping covering `[hint_va, hint_va+len)`.
#[inline]
pub unsafe fn mmap(
    hint_va: usize,
    len: usize,
    perms: usize,
    share_with_kernel: bool,
) -> Result<usize, Errno> {
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
        ecall4_ret2(
            syscall::CREATE_NETCH,
            vaddr_hint,
            region_size,
            sock_type,
            bind_spec,
        )
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

/// `ch_inspect(fd, &mut info)` — kind-aware per-fd metadata. Used by
/// mio's `Selector::register` to translate a `RawFd` into the shared
/// region pointer the scan loop reads from, and by future
/// `FromRawFd`-shaped consumers to rehydrate user-side handle state
/// without round-tripping through an `Arc<...>` cache.
///
/// The kernel writes exactly `size_of::<ChInfo>()` bytes into `info`.
/// Caller must place `info` so the struct fits inside a single
/// 4 KiB page (`(info as usize) % 64 == 0 && fits-in-page`) — same
/// constraint as the other small-struct syscalls.
#[inline]
pub fn ch_inspect(fd: u32, info: &mut crate::handle::ChInfo) -> Result<(), Errno> {
    Errno::from_ret(unsafe {
        ecall2(
            syscall::CH_INSPECT,
            fd as usize,
            info as *mut crate::handle::ChInfo as usize,
        )
    })
    .map(|_| ())
}

/// Allocate an EventFd backing page, map it shared at `vaddr_hint` in
/// the caller's shared range, and install a `Handle::EventFd` slot.
///
/// `initval` seeds the counter; `flags` is the bitwise OR of
/// [`event_fd::EFD_NONBLOCK`](crate::event_fd::EFD_NONBLOCK),
/// [`event_fd::EFD_SEMAPHORE`](crate::event_fd::EFD_SEMAPHORE), and
/// [`event_fd::EFD_CLOEXEC`](crate::event_fd::EFD_CLOEXEC).
///
/// `vaddr_hint` must be page-aligned and inside
/// `UPROC_SHARED_BASE..UPROC_SHARED_END`. Returns the mapped VA (which
/// equals `vaddr_hint` on success) plus the kernel-assigned fd.
#[inline]
pub fn eventfd(vaddr_hint: usize, initval: u64, flags: u32) -> Result<(usize, u32), Errno> {
    let (r0, r1) = unsafe {
        ecall3_ret2(
            syscall::EVENTFD,
            vaddr_hint,
            initval as usize,
            flags as usize,
        )
    };
    Errno::from_ret(r0).map(|va| (va, r1 as u32))
}

/// Push a `WakeEvent::Tid(tid)` onto the kernel wake queue. The kernel
/// validates that `tid` belongs to the calling process; cross-process
/// targets return `EPERM`. Best-effort — if the target isn't parked
/// when the manager drains the queue the wake is a no-op.
#[inline]
pub fn wake_tid(tid: u32) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::WAKE_TID, tid as usize) }).map(|_| ())
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
    create_thread_with_arg(
        unsafe {
            // SAFETY: layout of `extern "C" fn() -> !` and
            // `extern "C" fn(usize) -> !` is identical at the
            // calling-convention level; the new entry just ignores
            // its a0. The cast is purely a type-system relaxation so
            // existing call sites that don't care about `arg` keep
            // compiling.
            core::mem::transmute::<extern "C" fn() -> !, extern "C" fn(usize) -> !>(entry)
        },
        0,
        allowed_affinity,
        affinity,
    )
}

/// Spawn a sibling thread that enters at `entry` with `arg` in `a0`.
/// Same affinity rules as [`create_thread`]; the kernel writes `arg`
/// into the new thread's `a0` (x10) before its first sret, so the
/// entry can read it as its first C-ABI argument. `std::thread::spawn`
/// uses this to hand the new thread a `Box<ThreadInit>` pointer.
#[inline]
pub fn create_thread_with_arg(
    entry: extern "C" fn(usize) -> !,
    arg: usize,
    allowed_affinity: u64,
    affinity: u64,
) -> Result<u32, Errno> {
    Errno::from_ret(unsafe {
        ecall4(
            syscall::CREATE_THREAD,
            entry as usize,
            allowed_affinity as usize,
            affinity as usize,
            arg,
        )
    })
    .map(|t| t as u32)
}

/// Park the calling thread on `uaddr` if `*uaddr == expected`. The
/// compare-and-park is performed atomically with respect to
/// concurrent `futex_wake` calls — both go through the kernel
/// manager, which serializes the read-then-park against any wake.
///
/// The wait queue is keyed on the *physical* page+offset of `uaddr`,
/// so two threads in different processes that mapped the same
/// shared frame can rendezvous on the same word. `uaddr` must be
/// 4-byte aligned.
///
/// Returns:
/// - `Ok(())` — woken by a matching `futex_wake`.
/// - `Err(EAGAIN)` — `*uaddr != expected` at park time; caller
///   should re-load and retry.
/// - `Err(ETIMEDOUT)` — `timeout_ns > 0` deadline elapsed before a
///   wake. (v1: `timeout_ns == 0` means wait forever.)
/// - `Err(EFAULT)` — `uaddr` not mapped under the caller's satp.
/// - `Err(EINVAL)` — `uaddr` not 4-byte aligned, or kernel-half VA.
///
/// # Safety
/// `uaddr` must point at a 4-byte-aligned, mapped 32-bit word that
/// outlives the wait. `AtomicU32::as_ptr()` is the canonical source.
#[inline]
pub unsafe fn futex_wait(uaddr: *const u32, expected: u32, timeout_ns: u64) -> Result<(), Errno> {
    Errno::from_ret(unsafe {
        ecall3(
            syscall::FUTEX_WAIT,
            uaddr as usize,
            expected as usize,
            timeout_ns as usize,
        )
    })
    .map(|_| ())
}

/// Wake up to `n` threads parked on `uaddr` via [`futex_wait`].
/// Returns the number actually woken (0 if nobody was parked at the
/// time the wake was processed).
///
/// `uaddr` must be 4-byte aligned and follow the same physical-page
/// keying rules as [`futex_wait`].
///
/// # Safety
/// `uaddr` must be 4-byte-aligned and resolve to a mapped word.
#[inline]
pub unsafe fn futex_wake(uaddr: *const u32, n: u32) -> Result<u32, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::FUTEX_WAKE, uaddr as usize, n as usize) })
        .map(|c| c as u32)
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

/// Return the calling process's real uid. POSIX `getuid(2)` —
/// "shall always be successful and no return value is reserved to
/// indicate an error." A negative isize from the kernel (e.g. if the
/// caller pledged `PROC_LIFE` away) is silently re-cast and the
/// caller sees a uid with the high bit set; treat that the same as
/// any libc would. Backs `std::os::unix::fs::MetadataExt::uid` and
/// `nix::unistd::getuid`.
#[inline]
pub fn getuid() -> u32 {
    let r = unsafe { ecall1(syscall::GETUID, 0) };
    r as u32
}

/// Return the calling process's effective uid. POSIX `geteuid(2)`.
/// Same caveats as [`getuid`].
#[inline]
pub fn geteuid() -> u32 {
    let r = unsafe { ecall1(syscall::GETEUID, 0) };
    r as u32
}

/// Return the calling process's real gid. POSIX `getgid(2)`.
#[inline]
pub fn getgid() -> u32 {
    let r = unsafe { ecall1(syscall::GETGID, 0) };
    r as u32
}

/// Return the calling process's effective gid. POSIX `getegid(2)`.
#[inline]
pub fn getegid() -> u32 {
    let r = unsafe { ecall1(syscall::GETEGID, 0) };
    r as u32
}

/// Copy the calling process's supplementary group list into `buf`,
/// returning the number of entries written. POSIX `getgroups(2)`:
/// passing an empty slice (`buf.len() == 0`) returns the current
/// group count without writing — callers use this to size the real
/// call.
///
/// Errnos: `EFAULT`, `EINVAL` (buffer straddles a page), `ERANGE`
/// (non-empty buffer too small for the current list).
#[inline]
pub fn getgroups(buf: &mut [u32]) -> Result<usize, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::GETGROUPS, buf.as_mut_ptr() as usize, buf.len()) })
}

/// Copy the calling process's session login name into `buf` (no NUL
/// terminator). POSIX `getlogin_r(3)`-shaped — the bounded form,
/// since orbit doesn't carry the static-buffer flavor.
///
/// Errnos: `EFAULT`, `EINVAL` (buffer straddles a page), `ERANGE`
/// (buffer too small), `ENOENT` (no login name installed yet).
#[inline]
pub fn getlogin(buf: &mut [u8]) -> Result<usize, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::GETLOGIN, buf.as_mut_ptr() as usize, buf.len()) })
}

/// POSIX `setuid(uid)`. Mutate the calling process's uid triplet:
///   - euid == 0: stamp `uid` on all three slots (real/effective/saved)
///     — the privilege-drop path used by daemons after privsep startup.
///   - euid != 0: set only euid, IFF `uid` is one of the existing
///     ruid/suid (POSIX privilege-toggle rule).
///
/// Errnos: `EPERM` (non-root caller passed an unrelated uid).
#[inline]
pub fn setuid(uid: u32) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::SETUID, uid as usize) }).map(|_| ())
}

/// POSIX `setgid(gid)`. Same shape as [`setuid`] for the gid triplet.
#[inline]
pub fn setgid(gid: u32) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::SETGID, gid as usize) }).map(|_| ())
}

/// POSIX `setgroups(list)`. Replace the caller's supplementary group
/// list. Requires `euid == 0`. Capped at `process::NGROUPS_MAX = 16`
/// entries; longer lists yield `EINVAL`.
///
/// Errnos: `EPERM` (caller's `euid != 0`), `EINVAL` (list too long),
/// `EFAULT` (buffer doesn't translate).
#[inline]
pub fn setgroups(groups: &[u32]) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::SETGROUPS, groups.as_ptr() as usize, groups.len()) })
        .map(|_| ())
}

/// POSIX `setlogin(name)`. Stamp the calling process's session login
/// name. Requires `euid == 0`. Capped at `MAXLOGNAME = 32` bytes.
///
/// Errnos: `EPERM` (caller's `euid != 0`), `EINVAL` (non-UTF-8),
/// `ENAMETOOLONG`, `EFAULT`.
#[inline]
pub fn setlogin(name: &str) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::SETLOGIN, name.as_ptr() as usize, name.len()) })
        .map(|_| ())
}

/// Spawn a child process with command-line arguments. Same shape as
/// [`create_process`] otherwise; `argv_blob` is the packed bytes
/// described in [`crate::argv`] (header + offsets + string table).
/// Pass an empty slice for arg-less spawn (matches `create_process`).
///
/// Thin wrapper over [`create_process_with_argv_envp`] that passes
/// `0` for the envp VA (no environment installed).
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
    create_process_with_argv_envp(elf_ptr, elf_len, allowed_affinity, affinity, argv_blob, 0)
}

/// Spawn a child process with both command-line arguments and an
/// environment block. `argv_blob` follows the format in
/// [`crate::argv`]; pass an empty slice for arg-less spawn. `envp_va`
/// is the page-aligned user VA of an envp blob (same wire format as
/// argv — see [`crate::envp`]) or `0` for no envp.
///
/// The kernel always reads exactly one page at `envp_va`; callers
/// should hand over a page-resident, page-sized buffer (zero-padded
/// past the packed bytes). Subset of the trade-off keeping
/// `CREATE_PROCESS_EX` to seven user args (a1..a7).
///
/// # Safety
/// Same as [`create_process_with_argv`]. Additionally, when `envp_va
/// != 0` it must be page-aligned and the page must be mapped readable
/// for the caller's lifetime; otherwise the kernel returns `EFAULT`.
#[inline]
pub fn create_process_with_argv_envp(
    elf_ptr: *const u8,
    elf_len: usize,
    allowed_affinity: u64,
    affinity: u64,
    argv_blob: &[u8],
    envp_va: usize,
) -> Result<u16, Errno> {
    Errno::from_ret(unsafe {
        ecall7(
            syscall::CREATE_PROCESS_EX,
            elf_ptr as usize,
            elf_len,
            allowed_affinity as usize,
            affinity as usize,
            argv_blob.as_ptr() as usize,
            argv_blob.len(),
            envp_va,
        )
    })
    .map(|p| p as u16)
}

/// Return `(argv_va, envp_va)` — user VAs where the kernel mapped
/// this process's argv and envp blobs, with `0` in either slot
/// meaning "not installed" (process spawned via the bare
/// [`create_process`] path, or via [`create_process_with_argv`] with
/// no envp). Stable for the process's lifetime; orbit-rt's startup
/// caches the pair.
///
/// In v1 a non-zero argv VA is always [`crate::layout::USER_ARGV_BASE`]
/// and a non-zero envp VA is always [`crate::layout::USER_ENVP_BASE`].
#[inline]
pub fn argv_envp() -> (usize, usize) {
    let (r0, r1) = unsafe { ecall0_ret2(syscall::ARGV_ENVP) };
    (r0 as usize, r1 as usize)
}

/// Block the caller until child process `pid` exits, then return the
/// child's exit code. POSIX `waitpid(pid > 0, ...)`-shape. Errnos:
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
    // Specific-pid arm: the kernel returns `(0, exit_code)` on
    // success, `(-errno, 0)` on error. Cast the `u16` through `i32`
    // for the syscall arg — kernel reads it as i32 and dispatches
    // on sign.
    let (r0, r1) = unsafe { ecall1_ret2(syscall::WAIT_PID, pid as usize) };
    Errno::from_ret(r0).map(|_| r1 as i32)
}

/// Block the caller until *any* child exits, then return
/// `(child_pid, exit_code)`. POSIX `wait(&status)` / `waitpid(-1, ...)`
/// shape. Errnos:
/// - `ECHILD` — caller has no live children and an empty cache of
///   already-exited children. Either you never spawned anything that
///   could come back here, or every child you did spawn was
///   `DETACH`-flagged at creation.
/// - `EBUSY`  — another thread of this process already parked on
///   `waitpid(-1)` (v1 single-waiter; futex lifts this).
///
/// On success, the resolved child's pid is `> 0` and its exit code
/// lands in the second slot. The kernel drains the parent's
/// `dead_children` cache (lowest-pid wins) before parking — so
/// children that exited before you called still get reaped here.
///
/// Detached children (spawned with `CreateProcessV2Args::DETACH`)
/// are invisible to this call: their exits don't satisfy a parked
/// `wait_any_child` and they don't count toward the live-child probe
/// that gates ECHILD. Mirror of how `dead_children` already skips
/// detached spawns.
#[inline]
pub fn wait_any_child() -> Result<(u16, i32), Errno> {
    // Any-child arm: kernel returns `(child_pid, exit_code)` on
    // success (r0 positive = pid), `(-errno, 0)` on error. The `-1`
    // selector goes through the same WAIT_PID sysno as the
    // specific-pid path; the kernel dispatches on sign.
    let (r0, r1) = unsafe { ecall1_ret2(syscall::WAIT_PID, (-1isize) as usize) };
    if r0 < 0 {
        Err(Errno::new(-r0 as i32))
    }
    else {
        Ok((r0 as u16, r1 as i32))
    }
}

/// Absolute monotonic microseconds since system boot.
///
/// The base is opaque — only differences are meaningful. Backed by a
/// `csrr time` on the kernel side (RISC-V `time` runs at 10 MHz on
/// the QEMU virt machine; the syscall divides by 10 to give μs).
///
/// Use case: latency micro-benchmarks (sleep accuracy, RTT, throughput
/// timing) that don't want platform-coupled raw ticks. For wallclock,
/// see [`get_realtime`] — `get_micros` is monotonic only, no
/// time-of-day offset.
///
/// A direct `csrr time` from U-mode (the zero-syscall idea)
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

/// Wall-clock time since the UNIX epoch, returned as `(secs, nanos)`.
/// Backed by the kernel's Goldfish RTC driver on QEMU's `virt`
/// machine; `nanos ∈ [0, 999_999_999]`.
///
/// **Not monotonic.** The host RTC can step backward (suspend/resume,
/// NTP correction) — for interval timing use [`get_micros`].
#[inline]
pub fn get_realtime() -> (i64, u32) {
    let (s, ns) = unsafe { ecall0_ret2(syscall::GET_REALTIME) };
    (s as i64, ns as u32)
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
    Errno::from_ret(unsafe {
        ecall4(
            syscall::CREATE_PROCESS,
            elf_ptr as usize,
            elf_len,
            allowed_affinity as usize,
            affinity as usize,
        )
    })
    .map(|p| p as u16)
}

/// `pledge(req)` narrows this process's `perms` and `allowed_perms`
/// masks. Both axes are intersected with the corresponding
/// [`PermsRequest`] field; bits not present in `request.*` are
/// dropped, bits not present in the current permissions can't be
/// added back. Always succeeds (silent clamp, matching OpenBSD's
/// `pledge(promises, execpromises)` shape).
///
/// The kernel mutates `Process.permissions` and propagates the
/// narrowed snapshot to every live thread of the process, so the
/// dispatch-site gate EPERMs subsequent calls that needed a class
/// the caller just pledged away.
///
/// [`PermsRequest`]: crate::perms::PermsRequest
#[inline]
pub fn pledge(request: &crate::perms::PermsRequest) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::PLEDGE, request as *const _ as usize) }).map(|_| ())
}

/// `create_process_v2(args)` — role-aware spawn. Replaces
/// `create_process` for callers that need a `target_role` and
/// per-axis perms narrowing. The args struct lives in the caller's
/// memory; the kernel reads it once on entry.
///
/// On a denied transition (parent role's `transitions` bitset
/// doesn't include `target_role`) the kernel logs a `RoleDeny`
/// event into the kernel-wide ring, bumps the parent's
/// `role_denials` counter, and returns `-EPERM`. On success the
/// witness-derived perms are installed on the child.
///
/// [`CreateProcessV2Args`]: crate::perms::CreateProcessV2Args
#[inline]
pub fn create_process_v2(args: &crate::perms::CreateProcessV2Args) -> Result<u16, Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::CREATE_PROCESS_V2, args as *const _ as usize) })
        .map(|p| p as u16)
}

/// `query_denial_log(buf)` — copy the kernel-wide denial event ring
/// into `buf` in chronological order. Returns the number of events
/// actually written (`bytes / size_of::<DenialEvent>()`).
///
/// `buf` should be sized for at least
/// [`DENIAL_RING_CAPACITY`](crate::denial::DENIAL_RING_CAPACITY)
/// events for a complete snapshot; smaller buffers receive the
/// oldest prefix that fits.
///
/// **`buf` must not straddle a 4 KiB page** or the call returns
/// `EINVAL` (the kernel copies through a single page window — same as
/// `fs_read`). A full-snapshot buffer is ≈3 KiB, so page-align it
/// (e.g. wrap the array in a `#[repr(align(4096))]` struct); an
/// unaligned stack array crosses a page boundary more often than not.
///
/// **Cross-process disclosure.** The reply contains pids, tids,
/// syscall numbers, and (for `RoleDeny`) source/target roles for
/// *every* denial system-wide, not just the caller's. Acceptable
/// today (orbit is single-tenant); future multi-tenant workloads
/// will gate this behind a separate class.
#[inline]
pub fn query_denial_log(buf: &mut [crate::denial::DenialEvent]) -> Result<usize, Errno> {
    let event_size = core::mem::size_of::<crate::denial::DenialEvent>();
    Errno::from_ret(unsafe {
        ecall2(
            syscall::QUERY_DENIAL_LOG,
            buf.as_mut_ptr() as usize,
            buf.len() * event_size,
        )
    })
    .map(|bytes| bytes / event_size)
}

/// `fs_open(path, flags)` — resolve `path` against the mounted
/// filesystem and return an `Fd`. v1 is read-only tarfs; pass
/// [`crate::fs::OPEN_RDONLY`] (= 0) for `flags`. Errnos: `ENOENT`
/// (path not in archive), `EINVAL` (path too long / empty),
/// `EFAULT` (bad pointer), `EAGAIN` (manager work ring full).
#[inline]
pub fn fs_open(path: &str, flags: usize) -> Result<u32, Errno> {
    Errno::from_ret(unsafe { ecall3(syscall::FS_OPEN, path.as_ptr() as usize, path.len(), flags) })
        .map(|fd| fd as u32)
}

/// `fs_read(fd, buf)` — read up to `buf.len()` bytes at the fd's
/// current offset, served through the page cache. `buf.len()` may be
/// `1..=MAX_FS_READ_LEN` (64 KiB) and may span multiple pages (the
/// kernel walks the destination page by page). Returns the number of
/// bytes read (0 at EOF) and advances the fd's offset by exactly that
/// many bytes. Trailing bytes past the file size in the final page are
/// zero-padded by the cache.
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

/// `fs_readdir(fd, buf)` — pull a chunk of directory entries off `fd`
/// (which must come from `fs_open` on a directory). Returns bytes
/// written; `0` means end-of-directory.
///
/// The buffer is filled with packed [`crate::fs::DirEntry`] records.
/// Walk:
/// ```ignore
/// let mut p = 0;
/// while p < n {
///     let hdr = unsafe { core::ptr::read_unaligned(buf[p..].as_ptr() as *const DirEntry) };
///     let name = &buf[p + DIRENT_HDR_LEN .. p + DIRENT_HDR_LEN + hdr.d_namelen as usize];
///     p += hdr.d_reclen as usize;
/// }
/// ```
///
/// `buf.len()` must not span more than one 4 KiB page (the kernel
/// uses a single page-window for the copy-out, same constraint as
/// `fs_stat`). The cursor lives on the kernel-side `OpenFile`; pass
/// the same `fd` repeatedly until `0` to drain the directory.
///
/// Errnos: `EBADF` (fd not open / not a dir), `ENOTDIR` (fd is a
/// regular file), `EINVAL` (buffer crosses a page or is too small for
/// the next entry), `EFAULT` (bad pointer), `EAGAIN` (manager work
/// ring full).
#[inline]
pub fn fs_readdir(fd: u32, buf: &mut [u8]) -> Result<usize, Errno> {
    Errno::from_ret(unsafe {
        ecall3(
            syscall::FS_READDIR,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
        )
    })
}

/// `fs_fstat(fd, &mut Stat)` — fill `*stat` with metadata for the file
/// backing `fd`. Mirror of [`fs_stat`] keyed on an open fd, so callers
/// don't have to retain the path used at open. Backs
/// `std::fs::File::metadata` in the orbit std PAL.
///
/// Errnos: `EBADF` (fd not open), `EFAULT` (bad pointer), `EINVAL`
/// (stat straddles a page), `EIO` (backing fs lookup failed).
#[inline]
pub fn fs_fstat(fd: u32, stat: &mut crate::fs::Stat) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::FS_FSTAT, fd as usize, stat as *mut _ as usize) })
        .map(|_| ())
}

/// `fs_seek(fd, offset, whence)` — reposition the byte cursor on a
/// regular-file fd. `whence` is one of [`crate::fs::SEEK_SET`],
/// [`crate::fs::SEEK_CUR`], [`crate::fs::SEEK_END`]. Returns the new
/// absolute offset; never negative on success.
///
/// Errnos: `EBADF` (fd not a regular-file fd), `EINVAL` (bad whence
/// or resolved offset would be negative).
#[inline]
pub fn fs_seek(fd: u32, offset: i64, whence: u32) -> Result<u64, Errno> {
    Errno::from_ret(unsafe {
        ecall3(
            syscall::FS_SEEK,
            fd as usize,
            offset as usize,
            whence as usize,
        )
    })
    .map(|n| n as u64)
}

/// `chdir(path)` — replace the calling process's cwd with the
/// (absolute, UTF-8) `path`. The kernel validates the target
/// resolves to an existing directory in the active filesystem
/// before mutating cwd, so a successful return guarantees that
/// subsequent relative-path fs syscalls have a defined base.
///
/// Errnos: `EFAULT`, `EINVAL` (non-absolute / empty),
/// `ENAMETOOLONG`, `ENOENT` (target dir missing), `ENOTDIR`
/// (target exists but isn't a directory).
#[inline]
pub fn chdir(path: &str) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::CHDIR, path.as_ptr() as usize, path.len()) })
        .map(|_| ())
}

/// `getcwd(buf)` — copy the calling process's cwd into `buf` and
/// return the number of bytes written (no NUL terminator).
///
/// Errnos: `EFAULT`, `ERANGE` (buffer too small for current cwd —
/// caller can re-attempt with a larger buffer or fall back to a
/// page-sized scratch).
#[inline]
pub fn getcwd(buf: &mut [u8]) -> Result<usize, Errno> {
    Errno::from_ret(unsafe { ecall2(syscall::GETCWD, buf.as_mut_ptr() as usize, buf.len()) })
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
pub fn query_syscall_stats(buf: &mut [u8]) -> Result<(SyscallStatsHeader, &[SyscallEntry]), Errno> {
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

/// `fb_query(&mut info)` — fill `info` with active display dims and
/// pixel format. Stable for the system's lifetime in v1; cache the
/// result.
///
/// Errnos: `EFAULT` (`info` doesn't translate), `EINVAL` (straddles a
/// page), `EAGAIN` (display not yet initialized).
#[inline]
pub fn fb_query(info: &mut crate::fb::FbInfo) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::FB_QUERY, info as *mut _ as usize) }).map(|_| ())
}

/// `fb_surface_create(w, h, format)` — allocate a pixel surface and map
/// it into the calling process's shared range. Returns `(handle,
/// user_va)`. The mapping is user-writable BGRA8888; the kernel keeps a
/// KDMAP alias for the compositor.
///
/// Errnos: `EINVAL` (bad dims/format/size), `ENOMEM` (out of pages /
/// no shared VA), `EAGAIN` (manager ring full).
#[inline]
pub fn fb_surface_create(
    width: u32,
    height: u32,
    format: crate::fb::FbFormat,
) -> Result<(crate::fb::FbHandle, usize), Errno> {
    let (r0, r1) = unsafe {
        ecall3_ret2(
            syscall::FB_SURFACE_CREATE,
            width as usize,
            height as usize,
            format as u32 as usize,
        )
    };
    Errno::from_ret(r0).map(|h| (crate::fb::FbHandle(h as u32), r1 as usize))
}

/// `fb_surface_destroy(handle)` — release the surface, unmap its user
/// VA, and return the backing frame to `kernel_pages`.
///
/// Errnos: `EBADF` (unknown handle), `EAGAIN` (manager ring full).
#[inline]
pub fn fb_surface_destroy(handle: crate::fb::FbHandle) -> Result<(), Errno> {
    Errno::from_ret(unsafe { ecall1(syscall::FB_SURFACE_DESTROY, handle.raw() as usize) })
        .map(|_| ())
}

/// `fb_present(handle, x, y, w, h)` — submit a damage rect for the
/// surface. The compositor unions damage across multiple presents
/// between drains, then issues a single transfer + flush.
///
/// Errnos: `EBADF` (unknown handle), `EINVAL` (rect out of bounds /
/// zero-area), `EAGAIN` (compositor ring full).
#[inline]
pub fn fb_present(
    handle: crate::fb::FbHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<(), Errno> {
    Errno::from_ret(unsafe {
        ecall5(
            syscall::FB_PRESENT,
            handle.raw() as usize,
            x as usize,
            y as usize,
            width as usize,
            height as usize,
        )
    })
    .map(|_| ())
}

pub struct ConsoleWriter {
    buf: [u8; 256],
    len: usize,
}

impl ConsoleWriter {
    pub const fn new() -> Self {
        Self {
            buf: [0u8; 256],
            len: 0,
        }
    }
    pub fn flush(&mut self) {
        if self.len == 0 {
            return;
        }
        // The kernel's CONSOLE_RING is bounded (256 slots). A burst of
        // prints can fill it, in which case console_write returns
        // EAGAIN. Yield via sleep_ms(0) and
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
            if self.len >= self.buf.len() {
                self.flush();
            }
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
    pub const fn new() -> Self {
        Self {
            buf: [0u8; 256],
            len: 0,
        }
    }
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
            if self.len >= self.buf.len() {
                self.flush();
            }
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
