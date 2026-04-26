//! Pure syscall handler bodies. Each function takes the resolved thread +
//! either extracted arguments or the raw [`TrapFrame`] + an effect handle,
//! mutates thread-local state, and returns a [`SyscallOutcome`] for the
//! kmain shim to apply.

use device::TrapFrame;
use mmu::PagePermissions;
use process::{CompletionHandle, Thread, ThreadState};

use orbit_abi::errno::{Errno, EAGAIN, EBUSY, EFAULT, EINVAL, EIO, EPERM};
use orbit_abi::layout::{user_priv_range_ok, user_range_ok, user_shared_range_ok};

use crate::{
    CloseHandleReq, CreateProcessReq, CreateThreadReq, Hardware, MemMapReq,
    NetChannelCreationReq, PAGE_SIZE, PendingWork, SyscallOutcome,
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
    // Sanitize the user-supplied VA range before queueing manager work.
    // Otherwise umode could request a mapping at any kernel address
    // (KTEXT/KDMAP/KMMIO/the per-thread TrapFrame region) and the manager
    // would happily install PTEs on top of it. The check also enforces
    // the priv/shared split: a private mmap must land in the private
    // range, a shared mmap in the shared range, so the two pools never
    // share VAs.
    let range_ok = if req.share_with_kernel {
        user_shared_range_ok(req.vaddr as u64, req.size as u64)
    } else {
        user_priv_range_ok(req.vaddr as u64, req.size as u64)
    };

    if !range_ok {
        return SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() };
    }

    // No exec on shared mappings: shared frames keep a long-lived
    // writable KDMAP alias on the kernel side, so allowing X through
    // the user alias would set up a W^X violation across the two views
    // — kernel writes (e.g. net thread RX) would become executable code
    // in user.
    let is_exec = (req.page_permissions & PagePermissions::X) != 0;
    if req.share_with_kernel && is_exec {
        return SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() };
    }

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
    // NetChannels are always shared (the kernel keeps a KDMAP alias to
    // drive smoltcp). Reject any VA outside the shared range so the
    // priv/shared split holds: the private heap stays free of regions
    // with kernel aliases and forced revocation semantics.
    if !user_shared_range_ok(req.nc_vaddr as u64, req.region_size as u64) {
        return SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() };
    }
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
        allowed_affinity: frame.regs[13] as u64,
        affinity: frame.regs[14] as u64,
    };
    // Bound the source range before the manager copies bytes out. The
    // manager's per-page virt_to_phys would refuse a kernel VA today —
    // but only because user satps don't carry user PTEs for kernel
    // mappings. Keep the structural guarantee at the syscall boundary.
    if !user_range_ok(req.elf_vaddr as u64, req.elf_len as u64) {
        return SyscallOutcome::Return { ret: Errno::new(EFAULT).to_ret() };
    }
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

    if !user_range_ok(user_va, len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
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
    if !user_range_ok(user_va, len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
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

/// Bit set in `read_stdin`'s `flags` arg: return `EAGAIN` instead of
/// blocking when the ring is empty.
pub const READ_STDIN_NONBLOCK: usize = 1;

/// `read_stdin(buf, len, flags)` — drain up to `len` bytes of the
/// caller's stdin ring into `buf`. On non-empty: returns `Ok(n)`
/// synchronously. On empty + NONBLOCK: returns `Err(EAGAIN)`. On
/// empty + blocking: parks on a `CompletionHandle` and yields with
/// `YieldRetry` so the resumed thread re-executes the ecall (and
/// drains the bytes that woke it).
///
/// The park-then-recheck dance closes the race window between
/// observing an empty ring and storing the handle: if a producer
/// pushes a byte during that window, the recheck observes it and
/// the park is cancelled before yielding.
pub fn read_stdin<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let user_va = frame.regs[11] as u64;
    let user_len = frame.regs[12];
    let flags = frame.regs[13];

    if user_len == 0 || user_len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }
    if !user_range_ok(user_va, user_len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
    }
    if !hw.user_va_translates(thread.root_table_addr() as u64, user_va) {
        return ready(Errno::new(EFAULT).to_ret());
    }

    // Synchronous drain attempt. On any nonzero count we're done.
    let n = hw.read_stdin_drain(thread.pid, user_va, user_len);
    if n > 0 {
        return ready(n as isize);
    }

    if flags & READ_STDIN_NONBLOCK != 0 {
        return ready(Errno::new(EAGAIN).to_ret());
    }

    // Block path. Allocate a handle, park it on the per-process
    // slot, then re-check the ring to close the park-vs-signal
    // window before yielding.
    let handle = CompletionHandle::new();
    if !hw.park_stdin_reader(thread.pid, handle.clone()) {
        return ready(Errno::new(EBUSY).to_ret());
    }

    // Re-check: a byte that arrived between try_drain and park
    // would have observed `parked == null` and not signaled. By
    // re-draining after the park is visible, either we observe the
    // byte here (cancel the park, return synchronously) or we know
    // no producer raced us (yield safely).
    let n2 = hw.read_stdin_drain(thread.pid, user_va, user_len);
    if n2 > 0 {
        let _ = hw.unpark_stdin_reader(thread.pid);
        return ready(n2 as isize);
    }

    thread.handle = Some(handle);
    SyscallOutcome::YieldRetry { state: ThreadState::Blocking }
}

/// `create_thread(entry, allowed_affinity, affinity)` — spawn a sibling
/// thread in the calling process. Async manager round-trip: the
/// caller parks on a `CompletionHandle` while the manager allocates the
/// new thread, sets up its trap frame and stack, and inserts it into
/// `process.threads`; the handle is signaled with the new tid.
///
/// Sanitization happens here, not in the manager:
/// - `entry` must lie in the calling process's user range (the
///   broadest reasonable cap — finer-grained "must be inside .text"
///   would require process-state introspection that the syscall layer
///   deliberately doesn't have).
/// - `affinity & !allowed_affinity != 0` → `EINVAL` (well-formed but
///   structurally inconsistent — the requested initial mask escapes
///   the requested cap).
///
/// The "may not exceed parent's `allowed_affinity`" check happens at
/// the manager since that's the only place with access to the parent's
/// thread state. The manager rejects with `-EPERM` and the handle
/// surfaces the errno via the standard signal_n path.
pub fn create_thread<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req = CreateThreadReq {
        entry: frame.regs[11],
        allowed_affinity: frame.regs[12] as u64,
        affinity: frame.regs[13] as u64,
    };
    if !user_range_ok(req.entry as u64, 1) {
        return SyscallOutcome::Return { ret: Errno::new(EFAULT).to_ret() };
    }
    if req.allowed_affinity != 0
        && req.affinity != 0
        && req.affinity & !req.allowed_affinity != 0
    {
        return SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() };
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::CreateThread {
        req,
        pid: thread.pid,
        parent_allowed: thread.allowed_affinity,
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

/// `set_affinity(mask)` — narrow the calling thread's per-hart eligibility.
///
/// Validation order matches the docs in `process::Thread`:
/// 1. `mask == 0` → `EINVAL` (would orphan the thread; the scheduler
///    would never pick a hart for it).
/// 2. `mask & !allowed_affinity != 0` → `EPERM` (well-formed value, but
///    escapes the immutable cap set at thread construction).
///
/// On success, stores the new mask with `Release` so the next scheduler
/// pass sees it. The store doesn't preempt: if the calling thread is
/// running on a hart no longer in the new mask, it finishes its quantum
/// and migrates on the next dispatch.
pub fn set_affinity(thread: &Thread, frame: &TrapFrame) -> SyscallOutcome {
    let mask = frame.regs[11] as u64;
    if mask == 0 {
        return ready(Errno::new(EINVAL).to_ret());
    }
    if mask & !thread.allowed_affinity != 0 {
        return ready(Errno::new(EPERM).to_ret());
    }
    thread.affinity.store(mask, core::sync::atomic::Ordering::Release);
    ready(0)
}

/// `get_affinity()` — return `(current, allowed)` in `(a0, a1)`. Windows-shape:
/// the cap is exposed alongside the current mask so userspace can pick a
/// valid sub-mask without trial-and-error.
pub fn get_affinity(thread: &Thread) -> SyscallOutcome {
    let current = thread.affinity.load(core::sync::atomic::Ordering::Acquire);
    SyscallOutcome::Return2 {
        ret0: current as isize,
        ret1: thread.allowed_affinity as isize,
    }
}

#[inline]
fn ready(ret: isize) -> SyscallOutcome {
    SyscallOutcome::Yield {
        state: ThreadState::Ready,
        ret: Some(ret),
    }
}
