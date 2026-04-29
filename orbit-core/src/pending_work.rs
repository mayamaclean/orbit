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

use net_channel::BindSpec;
use process::CompletionHandle;

#[derive(Debug, Clone, Copy)]
pub struct MemMapReq {
    pub vaddr: usize,
    pub size: usize,
    pub page_permissions: u64,
    pub share_with_kernel: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct NetChannelCreationReq {
    pub nc_vaddr: usize,
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
    pub entry: usize,
    /// Cap and initial mask for the new thread. Sentinel `0` means
    /// "inherit the parent's value." Manager validates the resolved
    /// pair against the parent's `allowed_affinity` so a thread can't
    /// be created with reach the parent itself doesn't have.
    pub allowed_affinity: u64,
    pub affinity: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct FsOpenReq {
    /// User VA of the path string.
    pub path_vaddr: usize,
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
    pub buf_vaddr: usize,
    pub len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FsStatReq {
    pub path_vaddr: usize,
    pub path_len: usize,
    /// User VA of the `Stat` out-buffer. The kernel writes
    /// `size_of::<Stat>` bytes — caller must reserve at least that
    /// much.
    pub stat_vaddr: usize,
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
    pub elf_vaddr: usize,
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

/// `CreateProcessReq` plus a packed argv blob. v1 carries no envp;
/// the field is reserved by the matching `argv_envp` syscall name
/// (the kernel can extend the blob format later without bumping the
/// syscall number).
#[derive(Debug, Clone, Copy)]
pub struct CreateProcessExReq {
    pub elf_vaddr: usize,
    pub elf_len: usize,
    pub allowed_affinity: u64,
    pub affinity: u64,
    /// User VA of the packed argv blob (see `orbit_abi::argv`).
    /// `0` / `len == 0` means "no argv" — equivalent to
    /// `CREATE_PROCESS`.
    pub argv_vaddr: usize,
    pub argv_len: usize,
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
        root_pa: u64,
        handle: CompletionHandle,
    },
    NetChannelCreation {
        req: NetChannelCreationReq,
        pid: u16,
        root_pa: u64,
        handle: CompletionHandle,
    },
    CloseHandle {
        req: CloseHandleReq,
        pid: u16,
        root_pa: u64,
        handle: CompletionHandle,
    },
    CreateProcess {
        req: CreateProcessReq,
        pid: u16,
        root_pa: u64,
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
        root_pa: u64,
        handle: CompletionHandle,
    },
    FsRead {
        req: FsReadReq,
        pid: u16,
        root_pa: u64,
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
        root_pa: u64,
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
        root_pa: u64,
        handle: CompletionHandle,
    },
}
