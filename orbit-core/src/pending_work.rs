//! Manager work queue items.
//!
//! Each blocking syscall whose handler needs `&mut Orbit` (page-table
//! mutations, allocator access, process-table edits) drops a
//! [`PendingWork`] entry onto the kernel's `MANAGER_WORK` thingbuf via
//! [`crate::Hardware::push_pending_work`]. Whichever hart next acquires
//! `MANAGER_LOCK` drains the queue, runs the handler, and signals the
//! paired [`CompletionHandle`] with the result; the blocked thread's
//! return value is read off the handle on the next scheduler scan.
//!
//! The queue replaces the per-thread `block_reason` enum: new async
//! waits (e.g. §9's `read_stdin`) only need a [`CompletionHandle`] on
//! the thread and a signaler somewhere — no work-queue entry, no
//! manager involvement.

use mmu::sv48::PhysAddr;
use net_channel::BindSpec;
use orbit_abi::layout::UserVa;
use process::CompletionHandle;

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
    /// Pid to wait on. Manager validates that the caller is the
    /// parent (returns EPERM otherwise) and that the pid currently
    /// exists (ECHILD if not — covers both never-existed and
    /// already-reaped).
    pub target_pid: u16,
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
    /// `0` → wait forever (current v1 contract). Reserved for the
    /// future timeout-scan path (see roadmap §13a.5).
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
        handle: CompletionHandle,
    },
    NetChannelCreation {
        req: NetChannelCreationReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    CloseHandle {
        req: CloseHandleReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    CreateProcess {
        req: CreateProcessReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    CreateThread {
        req: CreateThreadReq,
        pid: u16,
        /// Parent thread's `allowed_affinity` snapshotted at syscall
        /// time. Manager uses this as the upper bound when resolving
        /// the new thread's `allowed_affinity`/`affinity` pair, so a
        /// thread can't widen the family's reach.
        parent_allowed: u64,
        handle: CompletionHandle,
    },
    FsOpen {
        req: FsOpenReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    FsRead {
        req: FsReadReq,
        pid: u16,
        root_pa: PhysAddr,
        /// User thread's handle. The manager hands a clone to the
        /// virtio-blk IRQ slot, so the IRQ signals it directly with
        /// `bytes_read` (success) or `-EIO` (failure). On submit
        /// failure (queue full, bad fd, …) the manager signals the
        /// retained clone with the errno itself.
        handle: CompletionHandle,
    },
    FsStat {
        req: FsStatReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    FsReaddir {
        req: FsReaddirReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    WaitPid {
        req: WaitPidReq,
        /// Caller's pid — manager checks this against the target's
        /// `parent_pid` for the EPERM gate.
        pid: u16,
        /// Caller's handle. On success the manager installs this on
        /// the target's `exit_waiter` slot and returns without
        /// signaling — `dealloc_process` signals it later with the
        /// child's exit code. Sync errors (ECHILD / EPERM / EBUSY)
        /// signal here in the manager arm.
        handle: CompletionHandle,
    },
    CreateProcessEx {
        req: CreateProcessExReq,
        pid: u16,
        root_pa: PhysAddr,
        handle: CompletionHandle,
    },
    FutexWait {
        req: FutexWaitReq,
        pid: u16,
        root_pa: PhysAddr,
        /// Caller's handle. The manager either signals it
        /// synchronously with `-EAGAIN` (value mismatch) or installs
        /// it on the per-PA wait queue; a later `futex_wake` (or
        /// timeout scan) signals with `0` / `-ETIMEDOUT`.
        handle: CompletionHandle,
    },
    FutexWake {
        req: FutexWakeReq,
        pid: u16,
        root_pa: PhysAddr,
        /// Caller's handle. Signaled synchronously with the count of
        /// waiters actually woken (≤ `req.n`) or a negative errno on
        /// translation failure.
        handle: CompletionHandle,
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
        handle: CompletionHandle,
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
        handle: CompletionHandle,
    },
    /// Post-DMA completion step for `fs_read`. The virtio-blk IRQ
    /// enqueues this for every successfully-completed (or failed)
    /// bounce-path chain; the manager unboxes `notif_ptr`,
    /// performs the scratch→user copy + cache publish + signal
    /// under MANAGER_LOCK, and drops the box (which drops the
    /// SharedFrame clone inside the `Bounce` variant, possibly
    /// returning the scratch page to `pending_frees`).
    ///
    /// `notif_ptr` is the same `Box::into_raw` of a kmain-side
    /// `WorkNotification` that submit_blk_read stashed in the
    /// virtio-blk slot table — the IRQ forwards it through
    /// unchanged so we don't double-box. orbit-core sees it as an
    /// opaque transport pointer; valid for exactly one
    /// `Box::from_raw` on the manager side. The variant is always
    /// `Bounce` (Direct is signaled inline by the IRQ); the
    /// manager matches accordingly. `status` carries the
    /// underlying virtio status: 0 = success, non-zero = device
    /// error.
    FsReadCopy { notif_ptr: usize, status: u8 },
}
