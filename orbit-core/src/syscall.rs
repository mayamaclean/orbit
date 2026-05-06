//! Pure syscall handler bodies. Each function takes the resolved thread +
//! either extracted arguments or the raw [`TrapFrame`] + an effect handle,
//! mutates thread-local state, and returns a [`SyscallOutcome`] for the
//! kmain shim to apply.

use device::TrapFrame;
use mmu::PagePermissions;
use orbit_abi::layout::UserVa;
use process::{CompletionHandle, Thread, ThreadState};

use orbit_abi::errno::{EAGAIN, EBUSY, EFAULT, EINVAL, EIO, EPERM, Errno};
use orbit_abi::layout::{user_priv_range_ok, user_range_ok, user_shared_range_ok};
use tracing::error;

use crate::{
    CloseHandleReq, CreateProcessExReq, CreateProcessReq, CreateProcessV2Req, CreateThreadReq,
    FsOpenReq, FsReadReq, FsReaddirReq, FsStatReq, FutexWaitReq, FutexWakeReq, Hardware,
    MAX_FS_PATH_LEN, MemMapReq, NetChannelCreationReq, PAGE_SIZE, PendingWork, PledgeReq,
    SyscallOutcome, WaitPidReq,
};
use net_channel::BindSpec;

/// Cap on `sleep_ms(ms)` arguments. Anything at or above this returns
/// `-EINVAL` without touching thread state.
pub const MAX_SLEEP_MS: usize = 60 * 60 * 1000;

/// True iff `[vaddr, vaddr + size)` lies within a single
/// [`PAGE_SIZE`]-aligned page. Used by syscalls that hand a small
/// fixed-size struct VA to the manager which then reads it via a
/// single `UserPageWindow` — straddling a page boundary would make
/// the second-page bytes silently come from a different physical
/// frame. Cheap (two arithmetic ops); call alongside the alignment
/// and `user_range_ok` checks.
#[inline]
fn struct_fits_in_one_page(vaddr: usize, size: usize) -> bool {
    if size == 0 {
        return true;
    }
    let page_mask = PAGE_SIZE - 1;
    (vaddr & !page_mask) == ((vaddr + size - 1) & !page_mask)
}

/// `sleep_ms(ms)` — block the caller for `ms` milliseconds.
///
/// Stores the absolute wake tick on the thread and tells the shim to yield
/// into `Suspended`. The wake loop in the manager compares
/// `now_ticks() >= thread.wake_time` to decide when to mark the thread
/// runnable again.
pub fn ms_sleep<H: Hardware>(thread: &mut Thread, ms: usize, hw: &H) -> SyscallOutcome {
    if ms >= MAX_SLEEP_MS {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }

    let wake_time =
        (hw.now_ticks() as usize).wrapping_add(ms.wrapping_mul(hw.ticks_per_ms() as usize));
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
pub fn mmap_req<H: Hardware>(thread: &mut Thread, frame: &TrapFrame, hw: &mut H) -> SyscallOutcome {
    let Ok(vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = MemMapReq {
        vaddr,
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
        user_shared_range_ok(req.vaddr.raw(), req.size as u64)
    }
    else {
        user_priv_range_ok(req.vaddr.raw(), req.size as u64)
    };

    if !range_ok {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }

    // No exec on shared mappings: shared frames keep a long-lived
    // writable KDMAP alias on the kernel side, so allowing X through
    // the user alias would set up a W^X violation across the two views
    // — kernel writes (e.g. net thread RX) would become executable code
    // in user.
    let is_exec = (req.page_permissions & PagePermissions::X) != 0;
    if req.share_with_kernel && is_exec {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }

    let handle = CompletionHandle::new();
    let work = PendingWork::MemMap {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
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
    // BindSpec is packed into a4 by the user-side wrapper. Reject
    // garbage at the boundary so the manager's run_nc_create_req can
    // assume well-formed input.
    let bind = match BindSpec::unpack(frame.regs[14]) {
        Some(b) => b,
        None => {
            return SyscallOutcome::Return {
                ret: Errno::new(EINVAL).to_ret(),
            };
        }
    };
    let Ok(nc_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = NetChannelCreationReq {
        nc_vaddr,
        region_size: frame.regs[12],
        nc_type: frame.regs[13],
        bind,
    };
    // NetChannels are always shared (the kernel keeps a KDMAP alias to
    // drive smoltcp). Reject any VA outside the shared range so the
    // priv/shared split holds: the private heap stays free of regions
    // with kernel aliases and forced revocation semantics.
    if !user_shared_range_ok(req.nc_vaddr.raw(), req.region_size as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::NetChannelCreation {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
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
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
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
    let Ok(elf_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        error!("bad elf vaddr: {:08X?}", frame.regs[11]);
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };

    let req = CreateProcessReq {
        elf_vaddr,
        elf_len: frame.regs[12],
        allowed_affinity: frame.regs[13] as u64,
        affinity: frame.regs[14] as u64,
    };
    // Bound the source range before the manager copies bytes out. The
    // manager's per-page virt_to_phys would refuse a kernel VA today —
    // but only because user satps don't carry user PTEs for kernel
    // mappings. Keep the structural guarantee at the syscall boundary.
    if !user_range_ok(req.elf_vaddr.raw(), req.elf_len as u64) {
        error!(
            "bad elf ranage: {:08X?}..{:08X?}",
            frame.regs[11],
            frame.regs[11] + req.elf_len
        );
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }

    let handle = CompletionHandle::new();
    let work = PendingWork::CreateProcess {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `wait_pid(pid) → exit_code | -errno` (`-ECHILD` if the target
/// doesn't exist or already reaped, `-EPERM` if the caller isn't its
/// parent, `-EINVAL` for self-wait, `-EBUSY` if a sibling already
/// parked on the target). Parks the caller on a fresh handle and
/// queues `PendingWork::WaitPid`; the manager either installs the
/// handle on the target's exit-waiter slot (alive case) or signals
/// the error sync.
pub fn wait_pid_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let target_pid = frame.regs[11] as u16;
    if target_pid == 0 || target_pid == thread.pid {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let req = WaitPidReq { target_pid };
    let handle = CompletionHandle::new();
    let work = PendingWork::WaitPid {
        req,
        pid: thread.pid,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `fs_open(path_ptr, path_len, flags) → fd | -errno`. Park the caller
/// on a fresh handle and queue the manager work; the manager copies
/// the path bytes, looks the inode up via the mounted filesystem, and
/// allocates an `Fd` in the calling pid's handle table.
pub fn fs_open_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(path_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = FsOpenReq {
        path_vaddr,
        path_len: frame.regs[12],
        flags: frame.regs[13],
    };
    if req.path_len == 0 || req.path_len > MAX_FS_PATH_LEN {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(req.path_vaddr.raw(), req.path_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::FsOpen {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// Max bytes per `fs_read` call. Backed by the page cache: a 64 KiB
/// read fans out into up to 16 cache-page slots, each contributing
/// up to two waiters (one per straddled user page). Sized at 16
/// pages so a single call comfortably fits inside the cache's
/// frame pool with slack for concurrent readers; raise alongside
/// `CACHE_PAGES` if larger reads ever pay off.
pub const MAX_FS_READ_LEN: usize = 16 * PAGE_SIZE;

/// `fs_read(fd, buf_ptr, len) → bytes | -errno`. The kernel reads at
/// the fd's current byte offset, returns up to `len` bytes (clipped
/// at EOF), and advances the offset by exactly the number of bytes
/// returned. `0` indicates EOF.
///
/// Buffer constraints: `len` is 1..=[`MAX_FS_READ_LEN`]; the buffer
/// VA range must pass `user_range_ok`. The kernel walks the
/// destination page-by-page, so multi-page buffers are supported —
/// only the total length matters.
///
/// Parks the calling thread; the manager wakes it (writing the byte
/// count into `regs[10]`) once every page that needs a DMA has
/// landed. Synchronous returns happen when every page is already a
/// cache hit. On any per-page failure the call resolves to `-EIO`.
pub fn fs_read_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(buf_vaddr) = UserVa::new(frame.regs[12] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = FsReadReq {
        fd: frame.regs[11] as u32,
        buf_vaddr,
        len: frame.regs[13],
    };
    if req.len == 0 || req.len > MAX_FS_READ_LEN {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(req.buf_vaddr.raw(), req.len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let work = PendingWork::FsRead {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        tid: thread.tid,
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `fs_stat(path_ptr, path_len, stat_ptr) → 0 | -errno`. Park on a
/// handle and queue manager work; the manager copies the path, looks
/// the inode up, and writes `size_of::<Stat>` bytes into the user's
/// stat buffer.
pub fn fs_stat_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(path_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let Ok(stat_vaddr) = UserVa::new(frame.regs[13] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = FsStatReq {
        path_vaddr,
        path_len: frame.regs[12],
        stat_vaddr,
    };
    if req.path_len == 0 || req.path_len > MAX_FS_PATH_LEN {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(req.path_vaddr.raw(), req.path_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if !user_range_ok(
        req.stat_vaddr.raw(),
        core::mem::size_of::<orbit_abi::fs::Stat>() as u64,
    ) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::FsStat {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `fs_readdir(fd, buf_ptr, len) → bytes | -errno`. Park on a handle
/// and queue manager work; the manager looks up the directory fd,
/// asks the filesystem to pack as many entries as fit into the user
/// buffer, and signals with bytes-written (`0` at end-of-dir).
///
/// v1 contract: `len` ≤ [`PAGE_SIZE`], buffer must not span more than
/// one page (single `UserPageWindow` for the copy-out).
pub fn fs_readdir_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(buf_vaddr) = UserVa::new(frame.regs[12] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = FsReaddirReq {
        fd: frame.regs[11] as u32,
        buf_vaddr,
        len: frame.regs[13],
    };
    if req.len == 0 || req.len > PAGE_SIZE {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(req.buf_vaddr.raw(), req.len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::FsReaddir {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `create_process_ex(elf_vaddr, elf_len, allowed_affinity, affinity,
/// argv_vaddr, argv_len) → pid | -errno`. §13a.3 extension to
/// `create_process` that carries an argv blob. Same async shape:
/// park the caller, queue manager work, return on signal.
///
/// Bound-checks the argv user-VA range so the manager can trust the
/// blob source. `argv_len == 0` (with any vaddr) is the "no args"
/// shorthand and falls through to a plain create_process.
pub fn create_process_ex_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(elf_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    // argv: skip pointer validation when len == 0. Rust hands out a
    // dangling-but-non-null pointer for `&[]` (e.g. `as_ptr() == 1`
    // for `&[u8]`), so a "no argv" caller would otherwise get EFAULT
    // here even though the kernel never dereferences the pointer.
    // envp uses an explicit `0` sentinel from userspace.
    let argv_raw = frame.regs[15] as u64;
    let argv_len = frame.regs[16];
    let argv_vaddr = if argv_len == 0 {
        unsafe { UserVa::new_unchecked(0) }
    }
    else {
        match UserVa::new(argv_raw) {
            Ok(v) => v,
            Err(_) => {
                return SyscallOutcome::Return {
                    ret: Errno::new(EFAULT).to_ret(),
                };
            }
        }
    };
    let envp_raw = frame.regs[17] as u64;
    let envp_vaddr = if envp_raw == 0 {
        unsafe { UserVa::new_unchecked(0) }
    }
    else {
        match UserVa::new(envp_raw) {
            Ok(v) => v,
            Err(_) => {
                return SyscallOutcome::Return {
                    ret: Errno::new(EFAULT).to_ret(),
                };
            }
        }
    };
    let req = CreateProcessExReq {
        elf_vaddr,
        elf_len: frame.regs[12],
        allowed_affinity: frame.regs[13] as u64,
        affinity: frame.regs[14] as u64,
        argv_vaddr,
        argv_len,
        envp_vaddr,
    };
    if !user_range_ok(req.elf_vaddr.raw(), req.elf_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if req.argv_len > 0 && !user_range_ok(req.argv_vaddr.raw(), req.argv_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if req.argv_len > orbit_abi::argv::ARGV_BLOB_MAX {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    // envp blob: the kernel always copies one page, so require a
    // page-aligned, page-resident VA. Mismatch surfaces as EFAULT
    // before the manager queues the work.
    if req.envp_vaddr.raw() != 0 {
        if req.envp_vaddr.raw() & (PAGE_SIZE as u64 - 1) != 0 {
            return SyscallOutcome::Return {
                ret: Errno::new(EINVAL).to_ret(),
            };
        }
        if !user_range_ok(req.envp_vaddr.raw(), PAGE_SIZE as u64) {
            return SyscallOutcome::Return {
                ret: Errno::new(EFAULT).to_ret(),
            };
        }
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::CreateProcessEx {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `futex_wait(uaddr, expected, timeout_ns)` — park on `uaddr` iff the
/// observed value equals `expected`. The compare-and-park happens on
/// the manager so a concurrent `futex_wake` can't slip between the
/// read and the queue insert.
///
/// The syscall layer's job is just to bound-check the user pointer
/// (4-byte aligned, mapped word) and queue the work; the manager
/// resolves uaddr → PA, reads `*uaddr`, and either signals
/// `-EAGAIN` (mismatch) or installs the handle on `futex_waiters[PA]`.
pub fn futex_wait_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let uaddr_raw = frame.regs[11];
    let expected = frame.regs[12] as u32;
    let timeout_ns = frame.regs[13] as u64;
    if uaddr_raw & 0b11 != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(uaddr_raw as u64, 4) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let Ok(uaddr) = UserVa::new(uaddr_raw as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = FutexWaitReq {
        uaddr,
        expected,
        timeout_ns,
    };
    let handle = CompletionHandle::new();
    let work = PendingWork::FutexWait {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `futex_wake(uaddr, n) → n_woken`. Manager resolves `uaddr` → PA,
/// drains up to `n` parked waiters from the futex table, signals
/// each with `0`, and returns the count.
///
/// Sync via the manager (not handler-thread) for the same reason as
/// `futex_wait`: serializing the table mutation against waiters and
/// against `dealloc_process` cleanup.
pub fn futex_wake_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let uaddr_raw = frame.regs[11];
    let n = frame.regs[12] as u32;
    if uaddr_raw & 0b11 != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(uaddr_raw as u64, 4) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let Ok(uaddr) = UserVa::new(uaddr_raw as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = FutexWakeReq { uaddr, n };
    let handle = CompletionHandle::new();
    let work = PendingWork::FutexWake {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
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
pub fn serial_print<H: Hardware>(thread: &Thread, frame: &TrapFrame, hw: &mut H) -> SyscallOutcome {
    let Ok(user_va) = UserVa::new(frame.regs[11] as u64)
    else {
        return ready(Errno::new(EFAULT).to_ret());
    };
    let len = frame.regs[12];

    if len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }

    if !user_range_ok(user_va.raw(), len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
    }

    if !hw.user_va_translates(thread.root_table_addr(), user_va) {
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
    let Ok(user_va) = UserVa::new(frame.regs[11] as u64)
    else {
        return ready(Errno::new(EFAULT).to_ret());
    };
    let len = frame.regs[12];

    if len == 0 || len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }
    if !user_range_ok(user_va.raw(), len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
    }
    if !hw.user_va_translates(thread.root_table_addr(), user_va) {
        return ready(Errno::new(EFAULT).to_ret());
    }

    let mut buf = [0u8; PAGE_SIZE];
    hw.copy_from_user(user_va, &mut buf[..len]);

    // Honor `stdout_capture=1` at spawn time — the calling thread's
    // snapshot redirects the bytes to the parent's pane. `None` is
    // the legacy/default path (writes go to the producer's own pane).
    let dest_pid = thread.stdout_redirect.unwrap_or(thread.pid);
    match hw.console_write_user(dest_pid, &buf[..len]) {
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
    let Ok(user_va) = UserVa::new(frame.regs[11] as u64)
    else {
        return ready(Errno::new(EFAULT).to_ret());
    };
    let user_len = frame.regs[12];
    let flags = frame.regs[13];

    if user_len == 0 || user_len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }
    if !user_range_ok(user_va.raw(), user_len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
    }
    if !hw.user_va_translates(thread.root_table_addr(), user_va) {
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
    SyscallOutcome::YieldRetry {
        state: ThreadState::Blocking,
    }
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
    let Ok(entry) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = CreateThreadReq {
        entry,
        allowed_affinity: frame.regs[12] as u64,
        affinity: frame.regs[13] as u64,
        arg: frame.regs[14],
    };
    if !user_range_ok(req.entry.raw(), 1) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if req.allowed_affinity != 0 && req.affinity != 0 && req.affinity & !req.allowed_affinity != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let handle = CompletionHandle::new();
    let work = PendingWork::CreateThread {
        req,
        pid: thread.pid,
        parent_allowed: thread.allowed_affinity,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `pledge(req: *const PermsRequest)` — narrow this process's
/// effective + cap masks. Park the caller on a fresh handle and
/// queue [`PendingWork::Pledge`]; the manager — sole writer of
/// `Process.permissions` — reads the request struct under the
/// caller's satp, applies the narrowing, and walks every live
/// thread of the process to rewrite each `Thread.permissions`
/// snapshot.
///
/// `req_vaddr` must be 8-byte aligned, bound-checked against the
/// caller's mappable range, and contained within a single page —
/// the manager reads via a single `UserPageWindow`, so a struct
/// straddling a page boundary would silently read garbage from
/// the next page. The request struct is 16 bytes (two `u64`s in
/// `PermsRequest`).
pub fn pledge_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req_vaddr_raw = frame.regs[11];
    let size = core::mem::size_of::<orbit_abi::perms::PermsRequest>();
    if req_vaddr_raw & 0b111 != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(req_vaddr_raw as u64, size as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if !struct_fits_in_one_page(req_vaddr_raw, size) {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let Ok(req_vaddr) = UserVa::new(req_vaddr_raw as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let req = PledgeReq { req_vaddr };
    let handle = CompletionHandle::new();
    let work = PendingWork::Pledge {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
    }
    thread.handle = Some(handle);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `create_process_v2(args: *const CreateProcessV2Args)` — role-aware
/// spawn with explicit perms narrowing. Park the caller on a fresh
/// handle and queue [`PendingWork::CreateProcessV2`]; the manager
/// copies the args struct + ELF, runs `check_transition`, and
/// signals the new pid on success or `-EPERM` (logged as a
/// `RoleDeny` audit event) on a denied transition.
///
/// The args struct must be 8-byte aligned, bound-checked against
/// the caller's mappable range, and contained within a single page
/// — same single-`UserPageWindow` read as `pledge_req` (see its
/// docs). Further validation (ELF range, affinity sanity) happens
/// manager-side after the args copy.
pub fn create_process_v2_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let args_vaddr_raw = frame.regs[11];
    let size = core::mem::size_of::<orbit_abi::perms::CreateProcessV2Args>();
    if args_vaddr_raw & 0b111 != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if !user_range_ok(args_vaddr_raw as u64, size as u64) {
        error!("invalid args addr");
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if !struct_fits_in_one_page(args_vaddr_raw, size) {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }

    let Ok(args_vaddr) = UserVa::new(args_vaddr_raw as u64)
    else {
        error!("invalid args addr");
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };

    let req = CreateProcessV2Req { args_vaddr };
    let handle = CompletionHandle::new();
    let work = PendingWork::CreateProcessV2 {
        req,
        pid: thread.pid,
        root_pa: thread.root_table_addr(),
        tid: thread.tid,
        handle: handle.clone(),
    };
    if hw.push_pending_work(work).is_err() {
        return SyscallOutcome::Return {
            ret: Errno::new(EAGAIN).to_ret(),
        };
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
    thread
        .affinity
        .store(mask, core::sync::atomic::Ordering::Release);
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
