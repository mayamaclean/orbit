//! Manager work queue items.
//!
//! Each blocking syscall whose handler needs `&mut Orbit` (page-table
//! mutations, allocator access, process-table edits) drops a
//! [`PendingWork`] entry onto the kernel's `MANAGER_WORK` thingbuf via
//! [`crate::Hardware::push_pending_work`]. Each entry carries the
//! parked caller's `tid`. Whichever hart next holds the scheduler lock
//! drains the queue, runs the handler, and resumes the parked thread
//! via `publish_pending_for_tid` (writes the result a-regs, then pushes
//! `WakeEvent::Tid`). No `CompletionHandle` is carried in the work item.
//!
//! Some async waits don't need a work-queue entry at all: `read_stdin`
//! and `read_key_event` park on a per-process `parked_tid` slot and are
//! resumed by a `WakeEvent::InputTid` push from the producer side.

use mmu::sv48::PhysAddr;
use net_channel::BindSpec;
use orbit_abi::layout::UserVa;
// CompletionHandle is no longer carried in any PendingWork variant —
// all blocking syscalls now use the on-thread completion path
// (`tid: u32` carried here; manager calls `publish_pending_for_tid`).
// Import removed to silence the unused-import warning. Reintroduce if
// a future variant needs trap-context signaling (e.g. an IRQ-driven
// resume path).

#[derive(Debug, Clone, Copy)]
pub struct MemMapReq {
    pub vaddr: UserVa,
    pub size: usize,
    pub page_permissions: u64,
    pub share_with_kernel: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct NetChannelCreationReq {
    pub nc_vaddr: UserVa,
    pub region_size: usize,
    pub nc_type: usize,
    /// Sticky binding the kernel latches at channel creation. Sent
    /// packed in the syscall's a4 register (see [`BindSpec::pack`]) and
    /// validated/unpacked at the syscall boundary; the manager threads
    /// it into `SocketReq::ctx` and never reads it again from shared
    /// memory.
    pub bind: BindSpec,
}

#[derive(Debug, Clone, Copy)]
pub struct CloseHandleReq {
    pub fd: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct CreateThreadReq {
    /// User-VA function pointer the new thread enters at. Bound-checked
    /// against the calling process's private + ELF range at the syscall
    /// boundary; a kernel-half VA here would be a privilege escalation.
    pub entry: UserVa,
    /// Cap and initial mask for the new thread. Sentinel `0` means
    /// "inherit the parent's value." Manager validates the resolved
    /// pair against the parent's `allowed_affinity` so a thread can't
    /// be created with reach the parent itself doesn't have.
    pub allowed_affinity: u64,
    pub affinity: u64,
    /// Opaque value the kernel writes into the new thread's a0 (x10)
    /// at sret. By convention `std::thread::spawn` boxes the closure
    /// state and passes the boxed pointer here; bare `extern "C" fn()
    /// -> !` entries that don't read a0 ignore it. Not validated by
    /// the kernel — the entry is responsible for interpreting it.
    pub arg: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FsOpenReq {
    /// User VA of the path string.
    pub path_vaddr: UserVa,
    /// Length in bytes (no NUL). Capped at [`MAX_FS_PATH_LEN`] at the
    /// syscall boundary.
    pub path_len: usize,
    /// `OPEN_*` flag bits. v1 kernel ignores these (tarfs is read-only)
    /// but the field is reserved for future modes.
    pub flags: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FsReadReq {
    pub fd: u32,
    pub buf_vaddr: UserVa,
    pub len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FsStatReq {
    pub path_vaddr: UserVa,
    pub path_len: usize,
    /// User VA of the `Stat` out-buffer. The kernel writes
    /// `size_of::<Stat>` bytes — caller must reserve at least that
    /// much.
    pub stat_vaddr: UserVa,
}

#[derive(Debug, Clone, Copy)]
pub struct FsReaddirReq {
    /// Open directory fd (returned by `fs_open` on a directory path).
    pub fd: u32,
    /// User VA of the out-buffer. Filled with packed
    /// [`orbit_abi::fs::DirEntry`] records.
    pub buf_vaddr: UserVa,
    /// Buffer length in bytes. Capped at one page on the kernel side
    /// (single `UserPageWindow` for the copy-out, same constraint as
    /// `fs_stat`).
    pub len: usize,
}

/// Cap on user-supplied path lengths. Generous enough for tar's
/// `prefix + "/" + name` joint limit (155 + 1 + 100 = 256), tight
/// enough that the kernel can keep the copy on its own stack.
pub const MAX_FS_PATH_LEN: usize = 256;

#[derive(Debug, Clone, Copy)]
pub struct WaitPidReq {
    /// POSIX-shaped pid selector — same encoding as `waitpid(2)`'s
    /// first argument:
    ///
    /// - `>  0` — wait for that specific child. Manager validates the
    ///   caller is the parent (EPERM otherwise) and the pid currently
    ///   exists or is in the parent's `dead_children` (ECHILD covers
    ///   never-existed + already-reaped). Resolved against the
    ///   target's `exit_waiter` slot.
    /// - `== -1` — wait for any child. Manager drains
    ///   `parent.dead_children` first; on empty, parks the caller in
    ///   `parent.any_child_waiter`. ECHILD if the parent has no live
    ///   children and an empty `dead_children`.
    /// - `==  0` — reserved for "any child in the caller's process
    ///   group" once process groups land. EINVAL today.
    /// - `<  -1` — reserved for "any child in process group `|pid|`".
    ///   EINVAL today.
    ///
    /// Stored as `i32` (not `u16`) so the `-1` sentinel round-trips
    /// cleanly through the syscall arg. Specific-pid values are
    /// always in the `1..=u16::MAX` range, validated kernel-side.
    pub target_pid: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct CreateProcessReq {
    pub elf_vaddr: UserVa,
    pub elf_len: usize,
    /// Initial mask handed to the child's first thread. The manager
    /// substitutes the all-harts default for the sentinel value 0
    /// (umode passes 0 to mean "no preference"). Must be a subset of
    /// `allowed_affinity`.
    pub affinity: u64,
    /// Immutable upper bound the child's first thread will be
    /// constructed with. Sentinel 0 means "default to the all-harts
    /// mask"; the manager sanitizes both fields together so an
    /// affinity bit that escapes allowed_affinity surfaces as EINVAL
    /// rather than silently widening the cap.
    pub allowed_affinity: u64,
}

/// `futex_wait(uaddr, expected, timeout_ns)` request. The manager
/// resolves `uaddr` to a physical address via the caller's user PT
/// and uses the PA as the wait-queue key — two threads in different
/// processes that mapped the same shared frame can rendezvous.
///
/// The expected-value compare-then-park is also done on the manager
/// (after the PA resolve) so the read and the wait-queue insert are
/// atomic with respect to a concurrent `futex_wake` (the manager is
/// single-threaded — only one hart holds `MANAGER_LOCK`).
#[derive(Debug, Clone, Copy)]
pub struct FutexWaitReq {
    pub uaddr: UserVa,
    pub expected: u32,
    /// `0` → wait forever (current v1 contract). Reserved for a
    /// future timeout-scan path.
    pub timeout_ns: u64,
}

/// `futex_wake(uaddr, n)` request. Manager resolves `uaddr` to a PA
/// the same way `FutexWaitReq` does and drains up to `n` waiters
/// from `futex_waiters[PA]`, signaling each with `0`.
#[derive(Debug, Clone, Copy)]
pub struct FutexWakeReq {
    pub uaddr: UserVa,
    pub n: u32,
}

/// `pledge(req)` request. Carried by `PendingWork::Pledge` so the
/// manager — sole writer of `Process.permissions` — can apply the
/// narrowing under MANAGER_LOCK and propagate the new value to every
/// live `Thread.permissions` snapshot. Pure narrowing (`ClassMask`
/// has no widening operation), so the manager never EPERMs a
/// well-formed pledge today.
#[derive(Debug, Clone, Copy)]
pub struct PledgeReq {
    /// User VA of the [`orbit_abi::perms::PermsRequest`] struct.
    /// Bound-checked at the syscall boundary; manager copies the
    /// 16-byte payload via the standard boundary path.
    pub req_vaddr: UserVa,
}

/// `create_process_v2(args)` request — the role-aware spawn. Same
/// async shape as the older `CreateProcess` variants: park the
/// caller, queue manager work, return the resolved pid (or a
/// negative errno) on signal. The args struct lives in user memory
/// at `args_vaddr`; the manager copies it once on entry.
#[derive(Debug, Clone, Copy)]
pub struct CreateProcessV2Req {
    /// User VA of the [`orbit_abi::perms::CreateProcessV2Args`]
    /// struct. Bound-checked at the syscall boundary.
    pub args_vaddr: UserVa,
}

/// `CreateProcessReq` plus a packed argv blob and (optionally) a
/// packed envp blob. The two share a wire format (`orbit_abi::argv`
/// / `orbit_abi::envp` — header + offsets + strings) so the kernel
/// reuses one fixup helper for both.
///
/// `argv_len > 0` carries argv; `envp_vaddr != 0` carries envp. The
/// envp blob is always passed as a page-aligned, page-resident
/// buffer — the kernel reads `PAGE_SIZE` bytes — because the
/// syscall ABI (`CREATE_PROCESS_EX`) saturates the seven a-regs at
/// elf+affinity+argv+envp_vaddr and has no register left for
/// `envp_len`. Callers pad the unused tail with zeros; install-side
/// validation walks the header to ignore the padding.
#[derive(Debug, Clone, Copy)]
pub struct CreateProcessExReq {
    pub elf_vaddr: UserVa,
    pub elf_len: usize,
    pub allowed_affinity: u64,
    pub affinity: u64,
    /// User VA of the packed argv blob (see `orbit_abi::argv`).
    /// `0` / `len == 0` means "no argv" — equivalent to
    /// `CREATE_PROCESS`.
    pub argv_vaddr: UserVa,
    pub argv_len: usize,
    /// User VA of the packed envp blob (see `orbit_abi::envp`); `0`
    /// means "no envp." Must be page-aligned and page-resident; the
    /// kernel always copies one page from this VA.
    pub envp_vaddr: UserVa,
}

/// `fb_surface_create(w, h, format)` request. The manager allocates a
/// `kernel_pages` frame sized to `w * h * bytes_per_pixel(format)`
/// (rounded up to a page), maps it user-writable in the calling
/// process's shared range, registers the entry in the per-process
/// surface table, and resumes the caller with `(handle_id, user_va)`.
#[derive(Debug, Clone, Copy)]
pub struct FbSurfaceCreateReq {
    pub width: u32,
    pub height: u32,
    /// Encoded `crate::fb::FbFormat` discriminant (validated by the
    /// manager via `FbFormat::from_u32`).
    pub format_raw: u32,
}

/// `fb_surface_destroy(handle)` request. Manager looks up the surface
/// in the calling process's table, unmaps the user VA, drops the
/// table entry, and frees the backing frame back to `kernel_pages`.
/// Signals `0` on success or a negative errno.
#[derive(Debug, Clone, Copy)]
pub struct FbSurfaceDestroyReq {
    pub handle: u32,
}

/// `eventfd(vaddr_hint, initval, flags)` request — manager allocates
/// a `kernel_pages` frame, initializes the [`EventFd`](orbit_abi::event_fd::EventFd)
/// header, maps it shared-revocable into the caller's shared range,
/// installs a `Handle::EventFd` slot, and signals
/// `(vaddr, fd)` on the result tid.
#[derive(Debug, Clone, Copy)]
pub struct EventFdCreateReq {
    pub vaddr_hint: UserVa,
    pub initval: u64,
    pub flags: u32,
}

/// `wake_tid(target_tid)` request — manager validates that
/// `target_tid` belongs to the calling process and pushes a
/// `WakeEvent::Tid(target_tid)`. Signals `0` on success or `-ESRCH` /
/// `-EPERM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WakeTidReq {
    pub target_tid: u32,
}

/// Cap on `setlogin(2)` name length, shared between the boundary
/// check in `setlogin_req` and the manager body so the two can't
/// drift. POSIX leaves the bound implementation-defined; 32 matches
/// OpenBSD's `LOGIN_NAME_MAX`.
pub const MAX_LOGIN_NAME: usize = 32;

// The next eleven requests were converted from sync trap-path
// handlers: they read or
// mutate manager-owned state (`process_handles`, `Process` cwd /
// creds), and a lock-free trap-path access races the manager's
// BTreeMap mutations on other harts. Routing them through the work
// queue makes that state single-consumer by construction — no
// trap-path locks (a trap/exception while holding one re-creates the
// kptr-long-jump deadlock class documented on MANAGER_LOCK).

/// `fs_seek(fd, offset, whence)` request. Manager repositions the
/// per-fd `OpenFile.offset` and signals the new absolute offset.
/// `fs_read`/`fs_readdir` (the offset's only consumers) already run
/// on the manager, so this completes the single-consumer story for
/// per-fd state.
#[derive(Debug, Clone, Copy)]
pub struct FsSeekReq {
    pub fd: u32,
    pub offset: i64,
    /// POSIX whence: `SEEK_SET = 0`, `SEEK_CUR = 1`, `SEEK_END = 2`.
    pub whence: u32,
}

/// `fs_fstat(fd, &mut Stat)` request. Manager resolves the fd's
/// `OpenFile`, runs `Filesystem::stat`, and copies into the user
/// buffer via `UserPageWindow`.
#[derive(Debug, Clone, Copy)]
pub struct FsFstatReq {
    pub fd: u32,
    /// User VA of the `Stat` out-buffer (`size_of::<Stat>` bytes).
    pub stat_vaddr: UserVa,
}

/// `ch_inspect(fd, *mut ChInfo)` request. Manager reads the fd's
/// handle-table slot and copies a populated
/// [`ChInfo`](orbit_abi::handle::ChInfo) into the user buffer.
#[derive(Debug, Clone, Copy)]
pub struct ChInspectReq {
    pub fd: u32,
    /// User VA of the `ChInfo` out-buffer. Must sit inside a single
    /// page (single-`UserPageWindow` constraint, same as `fs_stat`).
    pub info_vaddr: UserVa,
}

/// `chdir(path_ptr, path_len)` request. Manager validates the dir
/// exists in the active fs, then replaces the caller's `Process.cwd`.
#[derive(Debug, Clone, Copy)]
pub struct ChdirReq {
    pub path_vaddr: UserVa,
    /// `1..=MAX_FS_PATH_LEN`, enforced at the boundary.
    pub path_len: usize,
}

/// `getcwd(buf_ptr, buf_len)` request. Manager copies the caller's
/// cwd (no NUL) into the user buffer; signals bytes written or
/// `-ERANGE`.
#[derive(Debug, Clone, Copy)]
pub struct GetCwdReq {
    pub buf_vaddr: UserVa,
    pub buf_len: usize,
}

/// `getgroups(buf_ptr, count)` request. `count == 0` is the POSIX
/// sizing call (returns the group count without writing).
#[derive(Debug, Clone, Copy)]
pub struct GetGroupsReq {
    /// Raw u64 (not `UserVa`): ignored — and legally null — when
    /// `count == 0`. Boundary validates it only for `count > 0`.
    pub buf_vaddr: u64,
    /// Slot count (`u32`s, not bytes).
    pub count: usize,
}

/// `getlogin(buf_ptr, buf_len)` request. Manager copies the session
/// login name (no NUL); signals bytes written or `-ENOENT`.
#[derive(Debug, Clone, Copy)]
pub struct GetLoginReq {
    pub buf_vaddr: UserVa,
    pub buf_len: usize,
}

/// `setuid(uid)` request. Manager applies the POSIX triplet rules and
/// refreshes every sibling thread's credential snapshot — which is
/// exactly why this must run on the manager: the snapshot walk
/// mutates `Thread` fields the trap path reads lock-free.
#[derive(Debug, Clone, Copy)]
pub struct SetUidReq {
    pub uid: u32,
}

/// `setgid(gid)` request. Gid mirror of [`SetUidReq`].
#[derive(Debug, Clone, Copy)]
pub struct SetGidReq {
    pub gid: u32,
}

/// `setgroups(buf_ptr, count)` request. Manager reads `count` `u32`s
/// from user memory and replaces the supplementary group list
/// (requires `euid == 0`; `count == 0` empties the list).
#[derive(Debug, Clone, Copy)]
pub struct SetGroupsReq {
    /// Raw u64 (not `UserVa`): ignored — and legally null — when
    /// `count == 0` (the POSIX empty-the-list call). Boundary
    /// validates it only for `count > 0`.
    pub buf_vaddr: u64,
    /// Slot count, `0..=NGROUPS_MAX` (boundary-enforced).
    pub count: usize,
}

/// `setlogin(name_ptr, name_len)` request. Manager stamps the session
/// login name (UTF-8, `1..=MAX_LOGIN_NAME` bytes, requires
/// `euid == 0`).
#[derive(Debug, Clone, Copy)]
pub struct SetLoginReq {
    pub name_vaddr: UserVa,
    pub name_len: usize,
}

/// One slot in the manager's MPSC work ring. Fixed-size by virtue of
/// the variants — the largest payload (`CreateProcessReq`) is two
/// words; the handle is one Arc.
///
/// The `Empty` default exists so `thingbuf::StaticThingBuf` can
/// pre-initialize slots; the manager treats it as a no-op when
/// drained.
#[derive(Clone, Debug, Default)]
pub enum PendingWork {
    #[default]
    Empty,
    MemMap {
        req: MemMapReq,
        pid: u16,
        root_pa: PhysAddr,
        /// Tid of the parked thread. Manager runs `run_mmap_req`,
        /// then resumes via `resume_thread_with_values(tid, &[result])`
        /// and pushes `WakeEvent::Tid(tid)` — the no-Arc replacement
        /// for the `CompletionHandle` round-trip. Stale tids (thread
        /// exited mid-flight) are silently dropped by the resume
        /// helper.
        tid: u32,
    },
    NetChannelCreation {
        req: NetChannelCreationReq,
        pid: u16,
        root_pa: PhysAddr,
        /// Caller's tid. Manager runs `run_nc_create_req` and resumes
        /// via `publish_pending_for_tid(tid, &[vaddr, fd])` — the
        /// two-register return analog of `signal_pair`.
        tid: u32,
    },
    CloseHandle {
        req: CloseHandleReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    CreateProcess {
        req: CreateProcessReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    CreateThread {
        req: CreateThreadReq,
        pid: u16,
        /// Parent thread's `allowed_affinity` snapshotted at syscall
        /// time. Manager uses this as the upper bound when resolving
        /// the new thread's `allowed_affinity`/`affinity` pair, so a
        /// thread can't widen the family's reach.
        parent_allowed: u64,
        tid: u32,
    },
    FsOpen {
        req: FsOpenReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    FsRead {
        req: FsReadReq,
        pid: u16,
        root_pa: PhysAddr,
        /// Calling thread's tid. The cache-driven path resumes the
        /// thread directly via `Orbit::resume_thread_with_value`
        /// once every page in the read has landed (or any one
        /// fails). This on-thread publish path (no `CompletionHandle`)
        /// is now how every converted blocking syscall resumes.
        tid: u32,
    },
    FsStat {
        req: FsStatReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    FsReaddir {
        req: FsReaddirReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    WaitPid {
        req: WaitPidReq,
        /// Caller's pid — manager checks this against the target's
        /// `parent_pid` for the EPERM gate.
        pid: u16,
        /// Caller's tid. On success the manager installs this on the
        /// target's `exit_waiter` slot and returns without publishing
        /// — `dealloc_process` later resumes via
        /// `publish_pending_for_tid(tid, &[0, exit_code])` when the
        /// child exits. Sync errors (ECHILD / EPERM / EBUSY) publish
        /// here in the manager arm.
        tid: u32,
    },
    CreateProcessEx {
        req: CreateProcessExReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    FutexWait {
        req: FutexWaitReq,
        pid: u16,
        root_pa: PhysAddr,
        /// Caller's tid. The manager either publishes synchronously
        /// with `-EAGAIN` (value mismatch) / `-EFAULT` (translation
        /// failure), or installs it on the per-PA waiter queue; a
        /// later `futex_wake` (or timeout scan) resumes via
        /// `publish_pending_for_tid(tid, &[0])` / `[-ETIMEDOUT]`.
        tid: u32,
    },
    FutexWake {
        req: FutexWakeReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `pledge(*const PermsRequest)` — narrow this process's
    /// effective + cap masks. Manager copies the request struct from
    /// user memory under the caller's satp, applies the narrowing to
    /// `Process.permissions`, then walks every live thread of the
    /// process and rewrites its `Thread.permissions` snapshot so the
    /// dispatch-site gate sees the new mask without locking.
    Pledge {
        req: PledgeReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `create_process_v2(*const CreateProcessV2Args)` — role-aware
    /// spawn. Manager copies the args struct, runs the role
    /// transition gate (logs a `RoleDeny` audit event and returns
    /// `-EPERM` on `Err`), copies the ELF, calls `derive_child_perms`,
    /// and installs the resulting `Permissions` on the freshly-
    /// spawned `Process` via `install_permissions`.
    CreateProcessV2 {
        req: CreateProcessV2Req,
        pid: u16,
        root_pa: PhysAddr,
        /// Calling thread's tid. Used as the `SpawnInProgress` key
        /// for path-mode spawns (each `CacheFill` for the spawn's
        /// kernel-buffer waiters dispatches via this tid), and as
        /// the resume target for both bytes-mode (synchronous, via
        /// `publish_pending_for_tid` in the manager arm) and
        /// path-mode (deferred, via `advance_spawn` /
        /// `issue_next_spawn_page` once all pages land or any read
        /// fails).
        tid: u32,
    },
    /// `fb_surface_create(w, h, format)` — manager allocates the
    /// pixel surface, maps it into the user PT, registers the
    /// per-process surface table entry, and resumes the caller via
    /// `publish_pending_for_tid(tid, &[handle_id, user_va])`.
    FbSurfaceCreate {
        req: FbSurfaceCreateReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `fb_surface_destroy(handle)` — manager unmaps + frees the
    /// surface and resumes via `publish_pending_for_tid` with `0`
    /// (or `-EBADF` if the handle was unknown).
    FbSurfaceDestroy {
        req: FbSurfaceDestroyReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `eventfd(vaddr_hint, initval, flags)` — manager allocates the
    /// backing frame, initializes the EventFd layout, maps it into the
    /// caller's shared range, and installs a `Handle::EventFd` slot.
    /// Resumes via `publish_pending_for_tid(tid, &[vaddr, fd])`.
    EventFdCreate {
        req: EventFdCreateReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `wake_tid(target_tid)` — manager validates same-process
    /// membership and pushes `WakeEvent::Tid(target_tid)`. Resumes via
    /// `publish_pending_for_tid(tid, &[result])`.
    WakeTid { req: WakeTidReq, pid: u16, tid: u32 },
    /// `fs_seek` — manager mutates the per-fd `OpenFile.offset`.
    /// Resumes via `publish_pending_for_tid(tid, &[new_offset])`.
    FsSeek { req: FsSeekReq, pid: u16, tid: u32 },
    /// `fs_fstat` — manager stats the file backing the fd and copies
    /// into the user buffer. Resumes with `&[result]`.
    FsFstat {
        req: FsFstatReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `ch_inspect` — manager fills a `ChInfo` from the fd's slot and
    /// copies into the user buffer. Resumes with `&[result]`.
    ChInspect {
        req: ChInspectReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `chdir` — manager validates + replaces `Process.cwd`.
    Chdir {
        req: ChdirReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `getcwd` — manager copies `Process.cwd` into the user buffer.
    GetCwd {
        req: GetCwdReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `getgroups` — manager copies the supplementary group list.
    GetGroups {
        req: GetGroupsReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `getlogin` — manager copies the session login name.
    GetLogin {
        req: GetLoginReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `setuid` — manager mutates the uid triplet + thread snapshots.
    SetUid { req: SetUidReq, pid: u16, tid: u32 },
    /// `setgid` — gid mirror of [`PendingWork::SetUid`].
    SetGid { req: SetGidReq, pid: u16, tid: u32 },
    /// `setgroups` — manager reads the group list from user memory
    /// and replaces `Process.groups`.
    SetGroups {
        req: SetGroupsReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `setlogin` — manager reads the name from user memory and
    /// stamps `Process.login_name`.
    SetLogin {
        req: SetLoginReq,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },
    /// `argv_envp` — manager reads the caller's `Process.argv_blob` /
    /// `envp_blob` presence flags and resumes with
    /// `&[argv_va_or_0, envp_va_or_0]`. Converted from a sync handler:
    /// the blobs themselves are install-once-immutable, but the
    /// `self.processes` BTreeMap lookup raced manager-side inserts of
    /// unrelated pids.
    ArgvEnvp { pid: u16, tid: u32 },
    /// Page-cache DMA completion. Posted by the virtio-blk IRQ when a
    /// chain submitted via the cached path (`submit_blk_read_cached`)
    /// finishes. Carries the packed `CacheKey` the manager uses to
    /// look up the in-flight slot; manager iterates the slot's waiter
    /// list, copies bytes (UserPageWindow for User waiters, memcpy
    /// for Kernel waiters), and resumes each waiter's tid.
    ///
    /// `packed_key` is the bit-packed form from `kernel::page_cache::pack`;
    /// orbit-core sees it as an opaque u64 since the layout is
    /// kmain-internal. `status` carries the virtio status byte
    /// (0 = OK, non-zero = device error).
    CacheFill { packed_key: u64, status: u8 },

    /// `_exit(2)` / `exit_group(2)` from the calling thread. Pushed by
    /// `handle_exit` immediately before the leader descends into
    /// `exit_thread_with_state(Exited)`. The manager runs
    /// `request_exit_group` under `MANAGER_LOCK`, marking every sibling
    /// thread `Exited` and IPI'ing any hart currently running one.
    ///
    /// No `tid` field — the leader is dying, so there is no resume
    /// target. Sync ordering with the leader's own state transition is
    /// loose: the leader is marked `Exited` by `exit_thread_with_state`
    /// regardless of whether the manager has processed this entry yet,
    /// and `request_exit_group` only writes sibling state, so the two
    /// can race freely.
    ExitGroup {
        pid: u16,
        leader_tid: u32,
        exit_code: i32,
    },

    /// `query_denial_log(buf, len) → bytes | -errno` request. Manager
    /// drains the producer queue, snapshots the kernel-wide ring,
    /// copies up to `buf_len` bytes into the caller's buffer via
    /// `UserPageWindow` against `root_pa`, and resumes via
    /// `publish_pending_for_tid(tid, &[bytes_written])`.
    ///
    /// Single-page constraint: the syscall boundary rejects buffers
    /// that straddle a 4 KiB page (matches `fs_stat` / `fs_readdir`),
    /// so the manager arm is one `UserPageWindow::map`.
    QueryDenials {
        buf_vaddr: UserVa,
        buf_len: usize,
        pid: u16,
        root_pa: PhysAddr,
        tid: u32,
    },

    /// `query_stats(buf, len) → bytes | -errno` request. Manager
    /// snapshots `Process` accounting for the calling thread's pid
    /// and copies the resulting [`orbit_abi::stats::ProcessStats`]
    /// into the caller's buffer; resumes via `publish_pending_for_tid`.
    ///
    /// Single-page constraint: same as `QueryDenials`. `ProcessStats`
    /// is well under one page so this never matters in practice.
    QueryStats {
        target_pid: u16,
        buf_vaddr: UserVa,
        buf_len: usize,
        root_pa: PhysAddr,
        tid: u32,
    },
}

/// Validated spawn context shared between bytes-mode (which uses it
/// inline in the manager) and path-mode (which drives a per-page
/// state machine on the manager via `Orbit::issue_next_spawn_page`
/// + `advance_spawn`, finalizing through `install_spawn` once every
/// page has been read into the in-progress blob).
///
/// All "I checked this and it's good" state from the v2 syscall's
/// pre-flight lives here so the install path (`Orbit::install_spawn`)
/// can be a pure function of `(blob, ctx)` regardless of which
/// front-door delivered it. Owned types throughout because the
/// path-mode in-progress entry outlives the originating handler.
#[derive(Clone, Debug)]
pub struct SpawnContext {
    /// Copy of the caller-provided args struct. Used by `install_spawn`
    /// to re-walk the caller's user memory for the cwd / argv / envp
    /// blobs (those copies need the still-alive parent's PT, which is
    /// only safe while the parent is still parked `Blocking` on the
    /// original spawn syscall).
    pub args: orbit_abi::perms::CreateProcessV2Args,
    pub parent_pid: u16,
    pub root_pa: PhysAddr,
    /// Witness from the role-transition gate. Carries the resolved
    /// `Permissions` for `Process::install_child` — never widens, only
    /// shows what the gate already decided.
    pub child_perms: orbit_abi::roles::ChildPerms,
    /// Pre-validated overrides. `Some(_)` only when the caller's
    /// identity-stamping check passed (LOADER role); plain inheritance
    /// stays `None`.
    pub setuid_override: Option<u32>,
    pub setgid_override: Option<u32>,
    pub login_override: Option<alloc::string::String>,
    pub groups_override: Option<alloc::vec::Vec<u32>>,
    /// Snapshot of the parent's credential triplet at the time the v2
    /// syscall was processed. Carried verbatim for the inherit path
    /// (the parent might mutate its own creds via setuid between
    /// snapshot and install — install_spawn pins the snapshot).
    pub parent_uid: u32,
    pub parent_euid: u32,
    pub parent_suid: u32,
    pub parent_gid: u32,
    pub parent_egid: u32,
    pub parent_sgid: u32,
    pub parent_login: Option<alloc::string::String>,
    pub parent_groups: alloc::vec::Vec<u32>,
}
