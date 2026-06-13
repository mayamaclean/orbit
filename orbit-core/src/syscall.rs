//! Pure syscall handler bodies. Each function takes the resolved thread +
//! either extracted arguments or the raw [`TrapFrame`] + an effect handle,
//! mutates thread-local state, and returns a [`SyscallOutcome`] for the
//! kmain shim to apply.

use device::TrapFrame;
use mmu::PagePermissions;
use orbit_abi::layout::UserVa;
use process::{Thread, ThreadState};

use orbit_abi::errno::{EAGAIN, EBUSY, EFAULT, EINVAL, EIO, ENAMETOOLONG, EPERM, Errno};
use orbit_abi::layout::{user_priv_range_ok, user_range_ok, user_shared_range_ok};
use tracing::error;

use crate::{
    ChInspectReq, ChdirReq, CloseHandleReq, CreateProcessExReq, CreateProcessReq,
    CreateProcessV2Req, CreateThreadReq, EventFdCreateReq, FbSurfaceCreateReq, FbSurfaceDestroyReq,
    FsFstatReq, FsOpenReq, FsReadReq, FsReaddirReq, FsSeekReq, FsStatReq, FutexWaitReq,
    FutexWakeReq, GetCwdReq, GetGroupsReq, GetLoginReq, Hardware, MAX_FS_PATH_LEN, MAX_LOGIN_NAME,
    MemMapReq, NetChannelCreationReq, PAGE_SIZE, PendingWork, PledgeReq, SetGidReq, SetGroupsReq,
    SetLoginReq, SetUidReq, SyscallOutcome, WaitPidReq, WakeTidReq,
};
use net_channel::BindSpec;
use orbit_abi::event_fd::{EFD_ALL_FLAGS, EVENTFD_REGION_SIZE};
use orbit_abi::fb::FbFormat;

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

    let work = PendingWork::MemMap {
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

/// `fb_surface_create(w, h, format)` — park the calling thread on a
/// completion handle and queue a [`PendingWork::FbSurfaceCreate`]. The
/// manager validates dims/format, allocates a `kernel_pages` frame
/// sized to `w * h * bpp` (rounded up to `PAGE_SIZE`), maps it into
/// the user PT in the shared range, registers the per-process surface
/// table entry, and signals `(handle_id, user_va)` via `signal_pair`.
///
/// Validation here at the syscall boundary is light because the
/// allocator/mapper validates again — but we cheaply reject obviously
/// bad input (zero dims, unknown format) before queueing manager work.
pub fn fb_surface_create_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let width = frame.regs[11] as u32;
    let height = frame.regs[12] as u32;
    let format_raw = frame.regs[13] as u32;

    if width == 0 || height == 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if FbFormat::from_u32(format_raw).is_none() {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    // Size sanity check: `w * h * 4` must not overflow usize. The
    // manager will round up to PAGE_SIZE anyway; reject the hopeless
    // cases here so the manager arm doesn't have to defend against
    // them.
    let bpp = FbFormat::from_u32(format_raw)
        .map(|f| f.bytes_per_pixel())
        .unwrap_or(4);
    let Some(_total) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(bpp as usize))
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    };

    let req = FbSurfaceCreateReq {
        width,
        height,
        format_raw,
    };
    let work = PendingWork::FbSurfaceCreate {
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

/// `eventfd(vaddr_hint, initval, flags)` — park the caller and queue
/// a [`PendingWork::EventFdCreate`]. Manager allocates a `kernel_pages`
/// frame, initializes the [`EventFd`](orbit_abi::event_fd::EventFd)
/// header in-place, maps it user-RW + SharedRevocable at `vaddr_hint`,
/// and inserts a `Handle::EventFd` slot. Resumes via `signal_pair`
/// → `(vaddr, fd)`.
pub fn eventfd_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let vaddr_hint_raw = frame.regs[11] as u64;
    let initval = frame.regs[12] as u64;
    let flags = frame.regs[13] as u32;

    // EventFd regions are always one page; reject anything outside the
    // shared range. Same shape as the NetChannel boundary check.
    if !user_shared_range_ok(vaddr_hint_raw, EVENTFD_REGION_SIZE as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if vaddr_hint_raw & (PAGE_SIZE as u64 - 1) != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if flags & !EFD_ALL_FLAGS != 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let Ok(vaddr_hint) = UserVa::new(vaddr_hint_raw)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };

    let req = EventFdCreateReq {
        vaddr_hint,
        initval,
        flags,
    };
    let work = PendingWork::EventFdCreate {
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

/// `wake_tid(target_tid)` — push a `WakeEvent::Tid(target_tid)` once
/// the manager validates the target belongs to the calling process.
/// The doorbell primitive: lets a sibling thread nudge a reactor
/// parked in `ch_yield` / `eventfd`'s blocking read.
///
/// `target_tid == 0` (the sentinel "no reader parked") is a no-op
/// returning `0` synchronously without traversing the manager queue,
/// since EventFd writers use it as a fast-path skip.
pub fn wake_tid_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let target_tid = frame.regs[11] as u32;
    if target_tid == 0 {
        return SyscallOutcome::Return { ret: 0 };
    }
    let work = PendingWork::WakeTid {
        req: WakeTidReq { target_tid },
        pid: thread.pid,
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

/// Push `work` and park the caller `Blocking`, or return `-EAGAIN`
/// sync when the manager ring is full. The shared tail of every
/// converted-from-sync request below — see
/// `docs/dev/fd-unix-io-scope.md` item 0 for why these round-trip
/// instead of touching manager state from the trap path.
fn park_on_manager<H: Hardware>(hw: &mut H, work: PendingWork) -> SyscallOutcome {
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

/// `fs_seek(fd, offset, whence) → new_offset | -errno`. Queues
/// [`PendingWork::FsSeek`]; the manager owns `OpenFile.offset`.
/// Bad `whence` fails sync — no point round-tripping it.
pub fn fs_seek_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let whence = frame.regs[13] as u32;
    if whence > 2 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let req = FsSeekReq {
        fd: frame.regs[11] as u32,
        offset: frame.regs[12] as i64,
        whence,
    };
    park_on_manager(
        hw,
        PendingWork::FsSeek {
            req,
            pid: thread.pid,
            tid: thread.tid,
        },
    )
}

/// `fs_fstat(fd, &mut Stat) → 0 | -errno`. Queues
/// [`PendingWork::FsFstat`] after bounding the out-buffer.
pub fn fs_fstat_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(stat_vaddr) = UserVa::new(frame.regs[12] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let stat_size = core::mem::size_of::<orbit_abi::fs::Stat>() as u64;
    if !user_range_ok(stat_vaddr.raw(), stat_size) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    let req = FsFstatReq {
        fd: frame.regs[11] as u32,
        stat_vaddr,
    };
    park_on_manager(
        hw,
        PendingWork::FsFstat {
            req,
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `ch_inspect(fd, *mut ChInfo) → 0 | -errno`. Queues
/// [`PendingWork::ChInspect`]. The out-buffer must fit in one page
/// (single-`UserPageWindow` constraint) — rejected sync with EINVAL,
/// same contract as before the conversion.
pub fn ch_inspect_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let Ok(info_vaddr) = UserVa::new(frame.regs[12] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    let info_size = core::mem::size_of::<orbit_abi::handle::ChInfo>();
    if !user_range_ok(info_vaddr.raw(), info_size as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    if !struct_fits_in_one_page(info_vaddr.raw() as usize, info_size) {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let req = ChInspectReq {
        fd: frame.regs[11] as u32,
        info_vaddr,
    };
    park_on_manager(
        hw,
        PendingWork::ChInspect {
            req,
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `chdir(path_ptr, path_len) → 0 | -errno`. Queues
/// [`PendingWork::Chdir`]; manager validates the dir exists before
/// mutating `Process.cwd`.
pub fn chdir_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let path_len = frame.regs[12];
    if path_len == 0 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if path_len > MAX_FS_PATH_LEN {
        return SyscallOutcome::Return {
            ret: Errno::new(ENAMETOOLONG).to_ret(),
        };
    }
    let Ok(path_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    if !user_range_ok(path_vaddr.raw(), path_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    park_on_manager(
        hw,
        PendingWork::Chdir {
            req: ChdirReq {
                path_vaddr,
                path_len,
            },
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `getcwd(buf_ptr, buf_len) → bytes | -errno`. Queues
/// [`PendingWork::GetCwd`].
pub fn getcwd_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let buf_len = frame.regs[12];
    if buf_len == 0 || buf_len > PAGE_SIZE {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let Ok(buf_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    if !user_range_ok(buf_vaddr.raw(), buf_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    park_on_manager(
        hw,
        PendingWork::GetCwd {
            req: GetCwdReq { buf_vaddr, buf_len },
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `getgroups(buf_ptr, count) → count | -errno`. Queues
/// [`PendingWork::GetGroups`]. `count == 0` is the POSIX sizing call
/// — the buffer is ignored, so it's only validated for `count > 0`.
pub fn getgroups_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let buf_vaddr = frame.regs[11] as u64;
    let count = frame.regs[12];
    if count > process::NGROUPS_MAX {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if count > 0 && !user_range_ok(buf_vaddr, (count * core::mem::size_of::<u32>()) as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    park_on_manager(
        hw,
        PendingWork::GetGroups {
            req: GetGroupsReq { buf_vaddr, count },
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `getlogin(buf_ptr, buf_len) → bytes | -errno`. Queues
/// [`PendingWork::GetLogin`].
pub fn getlogin_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let buf_len = frame.regs[12];
    if buf_len == 0 || buf_len > PAGE_SIZE {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let Ok(buf_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    if !user_range_ok(buf_vaddr.raw(), buf_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    park_on_manager(
        hw,
        PendingWork::GetLogin {
            req: GetLoginReq { buf_vaddr, buf_len },
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `setuid(uid) → 0 | -EPERM`. Queues [`PendingWork::SetUid`] —
/// the manager applies POSIX triplet rules and refreshes sibling
/// threads' credential snapshots.
pub fn setuid_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    park_on_manager(
        hw,
        PendingWork::SetUid {
            req: SetUidReq {
                uid: frame.regs[11] as u32,
            },
            pid: thread.pid,
            tid: thread.tid,
        },
    )
}

/// `setgid(gid) → 0 | -EPERM`. Gid mirror of [`setuid_req`].
pub fn setgid_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    park_on_manager(
        hw,
        PendingWork::SetGid {
            req: SetGidReq {
                gid: frame.regs[11] as u32,
            },
            pid: thread.pid,
            tid: thread.tid,
        },
    )
}

/// `setgroups(buf_ptr, count) → 0 | -errno`. Queues
/// [`PendingWork::SetGroups`]. `count == 0` legally empties the list
/// with a (possibly null) ignored buffer.
pub fn setgroups_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let buf_vaddr = frame.regs[11] as u64;
    let count = frame.regs[12];
    if count > process::NGROUPS_MAX {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if count > 0 && !user_range_ok(buf_vaddr, (count * core::mem::size_of::<u32>()) as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    park_on_manager(
        hw,
        PendingWork::SetGroups {
            req: SetGroupsReq { buf_vaddr, count },
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `setlogin(name_ptr, name_len) → 0 | -errno`. Queues
/// [`PendingWork::SetLogin`].
pub fn setlogin_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let name_len = frame.regs[12];
    if name_len == 0 || name_len > MAX_LOGIN_NAME {
        return SyscallOutcome::Return {
            ret: Errno::new(ENAMETOOLONG).to_ret(),
        };
    }
    let Ok(name_vaddr) = UserVa::new(frame.regs[11] as u64)
    else {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    };
    if !user_range_ok(name_vaddr.raw(), name_len as u64) {
        return SyscallOutcome::Return {
            ret: Errno::new(EFAULT).to_ret(),
        };
    }
    park_on_manager(
        hw,
        PendingWork::SetLogin {
            req: SetLoginReq {
                name_vaddr,
                name_len,
            },
            pid: thread.pid,
            root_pa: thread.root_table_addr(),
            tid: thread.tid,
        },
    )
}

/// `argv_envp() → (argv_va | 0, envp_va | 0)`. Queues
/// [`PendingWork::ArgvEnvp`]. The blob VAs are fixed constants; the
/// round-trip exists only to read the install-presence flags off
/// `Process` safely (the sync version's `processes` map lookup raced
/// manager-side inserts of unrelated pids). Called once per process
/// startup by orbit-rt, so the park cost is invisible.
pub fn argv_envp_req<H: Hardware>(
    thread: &mut Thread,
    _frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    park_on_manager(
        hw,
        PendingWork::ArgvEnvp {
            pid: thread.pid,
            tid: thread.tid,
        },
    )
}

/// `fb_surface_destroy(handle)` — park the caller on a completion
/// handle and queue a [`PendingWork::FbSurfaceDestroy`]. Manager
/// looks up the surface, unmaps its user VA, removes the per-process
/// table entry, and frees the backing frame to `kernel_pages`.
pub fn fb_surface_destroy_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let req = FbSurfaceDestroyReq {
        handle: frame.regs[11] as u32,
    };
    let work = PendingWork::FbSurfaceDestroy {
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
    let work = PendingWork::NetChannelCreation {
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
    let work = PendingWork::CloseHandle {
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

    let work = PendingWork::CreateProcess {
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

/// `waitpid(pid)` — POSIX-shaped, polymorphic on pid:
///
/// - `pid >  0` — wait for that specific child. Returns `(0, exit_code)`
///   on success; r0 carries `-errno` on the error legs (`-ECHILD` for
///   never-existed / already-reaped, `-EPERM` if the caller isn't the
///   parent, `-EBUSY` if a sibling already parked on this target).
/// - `pid == -1` — wait for any child. Returns `(child_pid, exit_code)`
///   on success (r0 is the resolved child's pid, positive — fits in
///   `u16`); `-ECHILD` if the caller has no live children + empty
///   `dead_children`, `-EBUSY` if another thread already parked on the
///   parent's `any_child_waiter` slot.
/// - `pid ==  0` or `pid < -1` — reserved for future process-group
///   semantics; returns `-EINVAL` today.
///
/// Self-wait (caller passing its own pid as `> 0`) is rejected sync
/// with `-EINVAL` — would deadlock the calling thread on its own
/// `exit_waiter` slot.
///
/// Parks the caller and queues [`PendingWork::WaitPid`]; the manager
/// either installs the parker tid on the appropriate waiter slot or
/// signals the error / cached exit synchronously.
pub fn wait_pid_req<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    // Sign-extend the syscall arg through `i32`; the kernel-side
    // request struct carries the raw selector verbatim. Reject the
    // pgrp-reserved encodings sync — manager would have to do it
    // anyway and the work-queue round-trip is wasted.
    let target_pid = frame.regs[11] as isize as i32;
    if target_pid == 0 || target_pid < -1 {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    if target_pid > 0 && (target_pid as u32) == (thread.pid as u32) {
        return SyscallOutcome::Return {
            ret: Errno::new(EINVAL).to_ret(),
        };
    }
    let req = WaitPidReq { target_pid };
    let work = PendingWork::WaitPid {
        req,
        pid: thread.pid,
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
    let work = PendingWork::FsOpen {
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
    let work = PendingWork::FsStat {
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
    let work = PendingWork::FsReaddir {
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
    let work = PendingWork::CreateProcessEx {
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
    let work = PendingWork::FutexWait {
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
    let work = PendingWork::FutexWake {
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

    // Block path. Stamp the caller's tid on the per-process slot,
    // then re-check the ring to close the park-vs-push window before
    // yielding. No Arc allocation — `push_byte` swaps the slot back
    // to empty and returns the parked tid; the trap-context caller
    // (kmain's `input::dispatch`) issues
    // `WAKE_QUEUE.push(WakeEvent::InputTid(tid))` so the manager
    // resumes the parker.
    if !hw.park_stdin_reader(thread.pid, thread.tid) {
        return ready(Errno::new(EBUSY).to_ret());
    }

    // Re-check: a byte that arrived between try_drain and park
    // would have observed `parked_tid == 0` and not woken anyone. By
    // re-draining after the park is visible, either we observe the
    // byte here (cancel the park, return synchronously) or we know
    // no producer raced us (yield safely).
    let n2 = hw.read_stdin_drain(thread.pid, user_va, user_len);
    if n2 > 0 {
        let _ = hw.unpark_stdin_reader(thread.pid);
        return ready(n2 as isize);
    }

    // Park indefinitely. read_stdin has no timeout, so mirror
    // read_key_event's `READ_KEY_EVENT_INDEFINITE` shape: `wake_time =
    // usize::MAX` keeps the sleep-heap entry from ever popping on its
    // deadline (`drain_woken`'s `wake_time > now` is always true for
    // `u64::MAX`), so the only wake path is `wake_override`, set by
    // `input::dispatch`'s `WakeEvent::InputTid(tid)` doorbell.
    //
    // Suspended, NOT Blocking: the doorbell publishes no results, and
    // `set_wake_reason_where` promotes Suspended unconditionally but
    // gates Blocking on a SIGNALED completion slot (the canonical
    // publish-then-push shape every PendingWork syscall uses). A
    // Blocking park here would have its keystroke wake silently
    // dropped. YieldRetry's re-drain on resume makes a spurious or
    // duplicate wake harmless — an empty re-drain just re-parks. This
    // matches read_key_event, the only other YieldRetry parker.
    thread.wake_time = usize::MAX;
    SyscallOutcome::YieldRetry {
        state: ThreadState::Suspended,
    }
}

/// Bit set in `read_key_event`'s `flags` arg: return `EAGAIN` instead
/// of blocking when the ring is empty. Mirror of `READ_STDIN_NONBLOCK`.
pub const READ_KEY_EVENT_NONBLOCK: usize = 1;

/// Cap on `read_key_event(...)`'s `timeout_ms`. Mirrors `MAX_SLEEP_MS`
/// — anything at or above returns `-EINVAL`. Beyond the cap the caller
/// passes `usize::MAX` to mean "block indefinitely" (same idiom as
/// std's `Duration::MAX` on a condition variable).
pub const READ_KEY_EVENT_MAX_TIMEOUT_MS: usize = 60 * 60 * 1000;
/// Sentinel for "block indefinitely" in `timeout_ms`. The handler
/// programs `wake_time = u64::MAX` so the sleep_heap entry never
/// pops; only `wake_override` (set by `input::dispatch` on the
/// next event) wakes the thread.
pub const READ_KEY_EVENT_INDEFINITE: usize = usize::MAX;

/// `read_key_event(buf, count, flags, timeout_ms)` — drain up to
/// `count` 16-byte `KeyEvent`s from the caller's structured-event
/// ring into `buf`.
///
/// Park shape (same as `ch_yield`'s ms_sleep + wake_override):
/// - `flags & READ_KEY_EVENT_NONBLOCK` — never park; return
///   immediately with the events drained or `EAGAIN`.
/// - `timeout_ms == 0` (no NONBLOCK) — drain available; if empty,
///   return synchronously with 0. Effectively "peek".
/// - `timeout_ms == READ_KEY_EVENT_INDEFINITE` — block until the
///   next event arrives. `wake_time = u64::MAX`; only the
///   wake_override path wakes us.
/// - `0 < timeout_ms < MAX_TIMEOUT` — block up to `timeout_ms`. The
///   sleep_heap deadline OR the next event wakes us; whichever
///   fires first.
///
/// `count` is in events, so the byte length the kernel writes into
/// the user buffer is `count * 16`. The whole thing must fit in a
/// page — 256 events cap, which matches the ring's
/// [`process::ProcessKeyEvents`] capacity.
///
/// Returns `EBUSY` if another tid is already parked on this ring
/// (single-reader invariant).
pub fn read_key_event<H: Hardware>(
    thread: &mut Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    use orbit_abi::input::KeyEvent;
    const KEY_EVENT_SIZE: usize = core::mem::size_of::<KeyEvent>();

    let Ok(user_va) = UserVa::new(frame.regs[11] as u64)
    else {
        return ready(Errno::new(EFAULT).to_ret());
    };
    let count = frame.regs[12];
    let flags = frame.regs[13];
    let timeout_ms = frame.regs[14];

    if count == 0 {
        return ready(Errno::new(EINVAL).to_ret());
    }
    let Some(byte_len) = count.checked_mul(KEY_EVENT_SIZE)
    else {
        return ready(Errno::new(EINVAL).to_ret());
    };
    if byte_len > PAGE_SIZE {
        return ready(Errno::new(EINVAL).to_ret());
    }
    if !user_range_ok(user_va.raw(), byte_len as u64) {
        return ready(Errno::new(EFAULT).to_ret());
    }
    if !hw.user_va_translates(thread.root_table_addr(), user_va) {
        return ready(Errno::new(EFAULT).to_ret());
    }
    // Reject finite timeouts that match ch_yield's sleep cap. Anything
    // larger than the cap and *not* the sentinel is almost certainly a
    // caller treating ms_t as ns_t or similar — fail loud.
    if timeout_ms != READ_KEY_EVENT_INDEFINITE && timeout_ms >= READ_KEY_EVENT_MAX_TIMEOUT_MS {
        return ready(Errno::new(EINVAL).to_ret());
    }

    // Synchronous drain attempt.
    let n = hw.read_key_events_drain(thread.pid, user_va, count);
    if n > 0 {
        return ready(n as isize);
    }

    if flags & READ_KEY_EVENT_NONBLOCK != 0 {
        return ready(Errno::new(EAGAIN).to_ret());
    }
    if timeout_ms == 0 {
        // Peek shape: empty ring + no timeout = return 0 synchronously.
        return ready(0);
    }

    // Park path. Three branches based on the parker slot's prior state:
    //
    // - `Installed`: first park. Re-drain (close park-vs-push race),
    //   then yield Suspended with the deadline programmed.
    // - `AlreadyOwned`: re-entry from a timer wake (sleep_heap popped
    //   the deadline; no producer push cleared the slot). Clear the
    //   slot so the *next* userspace call can park afresh, then
    //   return 0 — that's the timeout signal a poll-with-timeout
    //   loop wants. Without the clear, the next call hits
    //   AlreadyOwned immediately and busy-loops. Without the
    //   AlreadyOwned branch entirely, the syscall re-parks into a
    //   fresh deadline forever and never returns on pure-timer
    //   wakes.
    // - `Busy`: another tid claimed the ring; single-reader violation.
    use process::key_events::ParkOutcome;
    match hw.set_key_event_parker(thread.pid, thread.tid) {
        ParkOutcome::Installed => {}
        ParkOutcome::AlreadyOwned => {
            let _ = hw.clear_key_event_parker_if(thread.pid, thread.tid);
            return ready(0);
        }
        ParkOutcome::Busy => return ready(Errno::new(EBUSY).to_ret()),
    }

    let n2 = hw.read_key_events_drain(thread.pid, user_va, count);
    if n2 > 0 {
        let _ = hw.clear_key_event_parker_if(thread.pid, thread.tid);
        return ready(n2 as isize);
    }

    // Schedule the deadline. Indefinite → u64::MAX (sleep_heap entry
    // never fires; wake_override is the only wake path).
    if timeout_ms == READ_KEY_EVENT_INDEFINITE {
        thread.wake_time = usize::MAX;
    }
    else {
        let now = hw.now_ticks() as usize;
        let ticks = timeout_ms.wrapping_mul(hw.ticks_per_ms() as usize);
        thread.wake_time = now.wrapping_add(ticks);
    }

    SyscallOutcome::YieldRetry {
        state: ThreadState::Suspended,
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
    let work = PendingWork::CreateThread {
        req,
        pid: thread.pid,
        parent_allowed: thread.allowed_affinity,
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
    let work = PendingWork::Pledge {
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
    let work = PendingWork::CreateProcessV2 {
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
