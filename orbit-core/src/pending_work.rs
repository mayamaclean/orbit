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
}
