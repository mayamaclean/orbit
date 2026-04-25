//! Pure syscall handler bodies. Each function takes the resolved thread +
//! either extracted arguments or the raw [`TrapFrame`] + an effect handle,
//! mutates thread-local state, and returns a [`SyscallOutcome`] for the
//! kmain shim to apply.

use device::TrapFrame;
use process::{CompletionHandle, Thread, ThreadState};

use orbit_abi::errno::{Errno, EAGAIN, EFAULT, EINVAL, EIO};

use crate::{
    CloseHandleReq, CreateProcessReq, Hardware, MemMapReq, NetChannelCreationReq,
    PAGE_SIZE, PendingWork, SyscallOutcome,
};

/// Cap on `sleep_ms(ms)` arguments. Anything at or above this returns
/// `-EINVAL` without touching thread state.
pub const MAX_SLEEP_MS: usize = 60 * 60 * 1000;

/// `sleep_ms(ms)` — block the caller for `ms` milliseconds.
///
/// Stores the absolute wake tick on the thread and tells the shim to yield
/// into `Suspended`. The wake loop in the manager compares
/// `now_ticks() >= thread.wake_time` to decide when to mark the thread
/// runnable again.
pub fn ms_sleep<H: Hardware>(thread: &mut Thread, ms: usize, hw: &H) -> SyscallOutcome {
    if ms >= MAX_SLEEP_MS {
        return SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() };
    }

    let wake_time = (hw.now_ticks() as usize)
        .wrapping_add(ms.wrapping_mul(hw.ticks_per_ms() as usize));
    thread.wake_time = wake_time;

    SyscallOutcome::Yield {
        state: ThreadState::Suspended,
        ret: Some(0),
    }
}

/// `mmap(vaddr, size, perms, share_with_kernel)` — park the thread on a
/// fresh [`CompletionHandle`] and push a [`PendingWork::MemMap`] entry
/// onto the manager's work ring. Whichever hart next holds
/// `MANAGER_LOCK` runs the page-table mutation and signals the handle;
/// the next scheduler scan reads the result off the handle into a0 and
/// resumes the thread.
///
/// Returns `-EAGAIN` if the work ring is full so the caller can retry
/// — same convention as `console_write`.
pub fn mmap_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req = MemMapReq {
        vaddr: frame.regs[11],
        size: frame.regs[12],
        page_permissions: frame.regs[13] as u64,
        share_with_kernel: frame.regs[14] > 0,
    };
    let handle = CompletionHandle::new();
    let work = PendingWork::MemMap {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr() as u64,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `create_netch(vaddr_hint, region_size, nc_type)` — park on a handle
/// and push a [`PendingWork::NetChannelCreation`] entry. Manager runs
/// the allocation + smoltcp socket setup and signals the handle with
/// `(vaddr, fd)` via `signal_pair` — those land in `regs[10]` and
/// `regs[11]` when the thread resumes.
pub fn nc_create_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req = NetChannelCreationReq {
        nc_vaddr: frame.regs[11],
        region_size: frame.regs[12],
        nc_type: frame.regs[13],
    };
    let handle = CompletionHandle::new();
    let work = PendingWork::NetChannelCreation {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr() as u64,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `close_handle(fd)` — park on a handle and push a
/// [`PendingWork::CloseHandle`] entry. Manager looks up the fd, revokes
/// the underlying `SharedUserPtr` if any, drops the Arc, and signals.
pub fn close_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req = CloseHandleReq {
        fd: frame.regs[11] as u32,
    };
    let handle = CompletionHandle::new();
    let work = PendingWork::CloseHandle {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr() as u64,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `create_process(elf_vaddr, elf_len)` — park on a handle and push a
/// [`PendingWork::CreateProcess`] entry. Manager copies the ELF out of
/// user memory, parses it, spawns the new process, and signals the
/// handle with the new pid (or a negative errno on failure).
pub fn create_process_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req = CreateProcessReq {
        elf_vaddr: frame.regs[11],
        elf_len: frame.regs[12],
    };
    let handle = CompletionHandle::new();
    let work = PendingWork::CreateProcess {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr() as u64,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `serial_print(user_va, len)` — copy a UTF-8 string out of user memory
/// and write it to the kernel serial console. Yields `Ready` after, so the
/// scheduler decides whether this thread keeps running.
///
/// Return codes:
/// - `0`         — bytes written
/// - `-EFAULT`   — user VA doesn't translate (bad pointer)
/// - `-EINVAL`   — `len` exceeds a page, or bytes aren't valid UTF-8
/// - `-EIO`      — serial write failed
pub fn serial_print<H: Hardware>(
    thread: &Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let user_va = frame.regs[11] as u64;
    let len = frame.regs[12];

    if len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }

    if !hw.user_va_translates(thread.root_table_addr() as u64, user_va) {
        return ready(Errno::new(EFAULT).to_ret());
    }

    let mut buf = [0u8; PAGE_SIZE];
    hw.copy_from_user(user_va, &mut buf[..len]);

    let s = match core::str::from_utf8(&buf[..len]) {
        Ok(s) => s,
        Err(_) => return ready(Errno::new(EINVAL).to_ret()),
    };

    match hw.serial_write_user(thread.pid, thread.tid, s) {
        Ok(()) => ready(0),
        Err(()) => ready(Errno::new(EIO).to_ret()),
    }
}

/// Append `len` user bytes at `regs[11]` to the calling process's
/// framebuffer scrollback. Chunked at `PAGE_SIZE` (4 KiB atomic
/// unit, matches POSIX `PIPE_BUF`). Return value is the number of
/// bytes accepted on success; negative errno on failure.
///
/// - `-EFAULT` — user VA doesn't translate under the thread's satp
/// - `-EINVAL` — `len == 0` or overflows `PAGE_SIZE`
/// - `-EAGAIN` — ring full, retry after yield
pub fn console_write<H: Hardware>(
    thread: &Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let user_va = frame.regs[11] as u64;
    let len = frame.regs[12];

    if len == 0 || len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }
    if !hw.user_va_translates(thread.root_table_addr() as u64, user_va) {
        return ready(Errno::new(EFAULT).to_ret());
    }

    let mut buf = [0u8; PAGE_SIZE];
    hw.copy_from_user(user_va, &mut buf[..len]);

    match hw.console_write_user(thread.pid, &buf[..len]) {
        Ok(()) => ready(len as isize),
        Err(()) => ready(Errno::new(EAGAIN).to_ret()),
    }
}

#[inline]
fn ready(ret: isize) -> SyscallOutcome {
    SyscallOutcome::Yield {
        state: ThreadState::Ready,
        ret: Some(ret),
    }
}
