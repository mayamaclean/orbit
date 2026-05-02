//! Per-process handle table. The manager assigns a `u32` ID at resource
//! creation (NetChannel today; sockets / files later) and stores the
//! owning handle here. Lookup by ID is the canonical way to name a
//! kernel-side resource from user space — validates existence, scopes
//! it to the calling pid, and gives us a stable key for future
//! close/configure syscalls.
//!
//! Ownership model:
//! - `Handle::NetChannel` holds a `SharedUserPtr<NetChannel>`. That's
//!   the manager's strong ref; k_net holds a separate clone via
//!   `SocketReq.netchan`.
//! - When the process exits, `dealloc_process` iterates this table,
//!   calls `revoke` on each Shared handle, then drops the table. k_net
//!   drops its clones when it processes socket_deletions, and the
//!   backing hits `pending_frees` when the last clone goes.
//!
//! IDs are monotonic per process; on wrap we'd need to handle
//! collisions but `u32` gives 4 billion allocations before that's a
//! concern.

use alloc::collections::BTreeMap;

use net_channel::NetChannel;

use crate::kernel::fs::Filesystem;
use crate::kernel::fs::Inode;
use crate::kernel::shared_frame::SharedFrame;
use crate::kernel::shared_user_ptr::SharedUserPtr;

/// Opaque per-process resource identifier. Not a hart-global value —
/// valid only in the process that was assigned it.
pub type Fd = u32;

/// One row in a process's handle table. Variants name the kind of
/// kernel-side resource the user is referring to. Adding a new
/// resource type = adding a variant here + whatever syscall surface
/// hands back the resulting Fd.
pub enum Handle {
    NetChannel(SharedUserPtr<NetChannel>),
    File(OpenFile),
}

/// FS-side per-fd state. Single mounted filesystem today, so
/// `fs` is a static reference; multi-mount would replace it with a
/// per-mount index.
pub struct OpenFile {
    pub fs: &'static dyn Filesystem,
    pub inode: Inode,

    /// Byte offset into the file. Advanced by `fs_read` by exactly
    /// the number of bytes copied to the caller (which can be less
    /// than a sector on sub-sector reads). Only meaningful when
    /// `inode` is a regular file.
    pub offset: u64,

    /// Opaque cursor for `fs_readdir`. The filesystem hands one back
    /// from each `readdir` call and we feed it forward. Only
    /// meaningful when `inode` is a directory; readdir on a
    /// regular-file fd returns ENOTDIR before the cursor is read.
    pub dir_cursor: u64,

    /// Per-fd sector cache for regular files. The kernel's
    /// underlying `fs_read` is sector-granular, but the syscall
    /// surface accepts arbitrary byte-aligned reads of arbitrary
    /// length up to one sector per call. The bridge is this cache:
    /// reads that fall within an already-cached sector copy out
    /// synchronously; reads that miss DMA into the scratch page,
    /// then a manager-side post-DMA step copies the requested slice
    /// to the user buffer.
    ///
    /// `None` for directories (use `fs_readdir` instead, no scratch
    /// needed) and on close-handle teardown after the backing has
    /// been freed.
    pub scratch: Option<ScratchSector>
}

/// Per-`OpenFile` 512-byte sector cache. Owns a [`SharedFrame`]
/// clone over the underlying `kernel_pages` page; the in-flight DMA
/// descriptor holds another clone, so a `close_handle` while the
/// DMA is mid-flight doesn't UAF the page — it just drops the
/// fd-side clone, and the descriptor's clone keeps it alive until
/// the manager finishes the post-DMA copy.
///
/// `loading` is the in-flight flag: true while a DMA into the
/// scratch is outstanding (scratch contents undefined; cache-hit
/// must not be served from the page). Cleared by
/// `run_fs_read_copy` (success or failure) and on submit-failure
/// rollback in `run_fs_read_req`. While loading, concurrent
/// readers on the same fd see EAGAIN — the kernel doesn't queue
/// per-fd waiters in v1.
pub struct ScratchSector {
    /// Refcounted backing. `frame.pa()` is the DMA target;
    /// `frame.kva()` is the kernel-side VA for the cache-hit
    /// memcpy into user buffers.
    pub frame: SharedFrame,
    /// File-relative sector index currently held in the scratch
    /// buffer (i.e. `byte_offset / 512`). `u64::MAX` means "empty /
    /// invalid"; treat as cache miss. Filesystem-agnostic: the
    /// underlying LBA is computed by the FS impl from inode +
    /// sector index. Only updated on successful DMA completion —
    /// never speculatively at submit time.
    pub cached_sector: u64,
    /// Bytes considered valid in the scratch buffer (≤ 512).
    /// Published alongside `cached_sector` from
    /// `run_fs_read_copy` on success.
    pub valid_bytes: u32,
    /// True iff a DMA into this scratch is currently outstanding.
    /// Read before the cache-hit check so a stale `cached_sector`
    /// can never be served while the page is being overwritten.
    pub loading: bool,
}

/// Per-process handle table + next-ID counter. Owned by `Orbit`, keyed
/// by pid — not stored inside `Process` itself because `Handle`
/// references kmain-level types (`SharedUserPtr`) that the `process`
/// crate doesn't see.
pub struct ProcessHandles {
    table: BTreeMap<Fd, Handle>,
    next_id: Fd,
}

impl ProcessHandles {
    pub fn new() -> Self {
        Self {
            table: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Allocate a fresh Fd and insert `handle` under it. Returns the
    /// Fd. Wraps at `u32::MAX`; collisions aren't handled today (we'd
    /// need a free list). At current allocation rates the counter
    /// can't realistically wrap before the process exits.
    pub fn insert(&mut self, handle: Handle) -> Fd {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.table.insert(id, handle);
        id
    }

    pub fn get(&self, fd: Fd) -> Option<&Handle> {
        self.table.get(&fd)
    }

    pub fn get_mut(&mut self, fd: Fd) -> Option<&mut Handle> {
        self.table.get_mut(&fd)
    }

    pub fn remove(&mut self, fd: Fd) -> Option<Handle> {
        self.table.remove(&fd)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Fd, &Handle)> {
        self.table.iter()
    }

    /// Consume and drain the table. Used at process teardown so the
    /// caller can walk each handle by value (e.g. to free per-fd
    /// scratch backings via `free_backing` before the BTreeMap
    /// drops).
    pub fn into_iter(self) -> impl Iterator<Item = (Fd, Handle)> {
        self.table.into_iter()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
    pub fn len(&self) -> usize {
        self.table.len()
    }
}
