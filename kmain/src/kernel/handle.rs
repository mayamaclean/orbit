//! Per-process handle table. The manager assigns an `Fd` (a positive
//! `u32` bounded by `i32::MAX` so the value round-trips losslessly to
//! `RawFd = c_int`) at resource creation (NetChannel, File, Stdin /
//! Stdout / Stderr at process spawn, EventFd, future Pipe / Pidfd) and
//! stores the owning [`Handle`] here behind a [`Slot`] that also tracks
//! per-fd flags (`cloexec`, `nonblock`).
//!
//! Lookup by Fd is the canonical way to name a kernel-side resource
//! from user space — validates existence, scopes it to the calling
//! pid, and gives us a stable key for `close` / `dup` / `fcntl` /
//! `fstat`-shaped syscalls.
//!
//! Ownership model:
//! - `Handle::NetChannel` holds a `SharedUserPtr<NetChannel>`. That's
//!   the manager's strong ref; k_net holds a separate clone via
//!   `SocketReq.netchan`.
//! - `Handle::File` holds an `OpenFile` with the per-fd cursor + inode
//!   ref. fs is shared and stateless, so this is the only place per-fd
//!   state lives.
//! - `Handle::Stdin` / `Stdout` / `Stderr` are zero-sized markers
//!   pre-seeded in slots 0 / 1 / 2 at process creation. There is no
//!   fd-based `read` / `write` dispatch yet — stdio I/O still goes
//!   through the dedicated `read_stdin` / `console_write` syscalls,
//!   which don't consult this table. The slots exist today to reserve
//!   the POSIX fd numbers (so the first real fd lands at 3) and to be
//!   reported by `ch_inspect`; a future generic `read`/`write` arm
//!   will route them into the console / key-event paths.
//! - When the process exits, `dealloc_process` iterates this table,
//!   calls `revoke` on each Shared handle, then drops the table. k_net
//!   drops its clones when it processes socket_deletions, and the
//!   backing hits `pending_frees` when the last clone goes.
//!
//! Fds are monotonic per process; on hitting `i32::MAX` (4096 short of
//! `u32::MAX` for the high-bit safety margin) we'd need a free list,
//! but no realistic workload approaches that.
//!
//! ## CLOEXEC default
//!
//! Per-slot `cloexec: bool` defaults to `false` (Linux semantics):
//! caller has to opt in via `O_CLOEXEC` at create or
//! `fcntl(F_SETFD, FD_CLOEXEC)`. Inheritance at spawn time is already
//! explicit on orbit (parent names the inherited fds in
//! `CreateProcessV2Args`), so the flag is mostly there to satisfy
//! libc-shaped consumers that read/set it programmatically.

use alloc::collections::BTreeMap;
use core::sync::atomic::AtomicU32;

use net_channel::NetChannel;
use orbit_abi::event_fd::EventFd as EventFdRegion;

use crate::kernel::fs::Filesystem;
use crate::kernel::fs::Inode;
use crate::kernel::shared_user_ptr::SharedUserPtr;

/// Opaque per-process resource identifier. Not a hart-global value —
/// valid only in the process that was assigned it.
///
/// Storage is `u32` for natural "ID counter" semantics, but every
/// allocated value is guaranteed `<= i32::MAX` so the value casts
/// losslessly to `RawFd = c_int` at the std PAL boundary. The
/// allocator panics on overflow rather than wrapping into the high
/// bit — wrap would silently produce a negative fd in user code.
pub type Fd = u32;

/// Upper bound on allocated [`Fd`] values. Matches `i32::MAX` so the
/// fd round-trips losslessly to `RawFd`.
pub const FD_MAX: Fd = i32::MAX as u32;

/// One row in a process's handle table. Variants name the kind of
/// kernel-side resource the user is referring to. Adding a new
/// resource type = adding a variant here + the matching kind tag in
/// [`orbit_abi::handle::HandleKind`] + dispatch arms in the generic
/// `read` / `write` / `fstat` syscalls.
pub enum Handle {
    NetChannel(SharedUserPtr<NetChannel>),
    File(OpenFile),
    /// Standard input — slot 0 at process creation. `read` dispatches
    /// to the cooked-mode line buffer / key-event ring; `write`
    /// returns `EBADF`.
    Stdin,
    /// Standard output — slot 1. `write` routes through the existing
    /// `console_write` path; `read` returns `EBADF`.
    Stdout,
    /// Standard error — slot 2. Same shape as `Stdout` today.
    Stderr,
    /// EventFd — Linux-shaped shared-memory counter. See [`EventFdSlot`].
    EventFd(EventFdSlot),
}

/// Per-fd kernel state for an EventFd. The shared counter + flags live
/// in user-mapped memory inside the [`SharedUserPtr<EventFdRegion>`];
/// this struct just owns that handle and the kernel-side parked-tid
/// shadow.
pub struct EventFdSlot {
    pub region: SharedUserPtr<EventFdRegion>,
    /// Authoritative kernel-side shadow of the parked-reader tid for
    /// the future POSIX `read(fd)` path. Mirrors the shared
    /// `EventFd.parked_tid` field so the kernel doesn't have to trust
    /// userspace-writable memory for state-machine work
    /// (close-while-parked cleanup, wake routing). Reserved — no
    /// kernel path currently parks a reader, so this stays `0`.
    pub kernel_parked_tid: AtomicU32,
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

    /// Snapshot of `S_IFREG`-ness at fs_open time. Lets the read /
    /// readdir paths reject the wrong kind without a fresh `stat`.
    /// (For tarfs this never changes mid-flight; once mutable
    /// filesystems land we'll re-evaluate.)
    pub is_regular: bool,
}

/// Per-slot wrapper carrying the handle plus the POSIX-shaped flags
/// that need to live next to it (`cloexec`, `nonblock`). Kept tight —
/// these are the bits `fcntl(F_GETFD)` / `F_GETFL` actually return.
pub struct Slot {
    pub handle: Handle,
    /// `FD_CLOEXEC` — when true, this slot is *not* cloned into a
    /// child's handle table at spawn time. Default `false` (Linux
    /// semantics).
    pub cloexec: bool,
    /// `O_NONBLOCK` — when true, `read` / `write` on this fd return
    /// `EAGAIN` instead of blocking on an empty / full ring. Default
    /// `false`.
    pub nonblock: bool,
}

impl Slot {
    pub fn new(handle: Handle) -> Self {
        Self {
            handle,
            cloexec: false,
            nonblock: false,
        }
    }

    pub fn with_flags(handle: Handle, cloexec: bool, nonblock: bool) -> Self {
        Self {
            handle,
            cloexec,
            nonblock,
        }
    }
}

/// Per-process handle table + next-ID counter. Owned by `Orbit`, keyed
/// by pid — not stored inside `Process` itself because `Handle`
/// references kmain-level types (`SharedUserPtr`) that the `process`
/// crate doesn't see.
pub struct ProcessHandles {
    table: BTreeMap<Fd, Slot>,
    next_id: Fd,
}

impl ProcessHandles {
    /// Construct an empty table. Stdio is *not* pre-seeded here; that
    /// happens at process creation via [`Self::seed_stdio`] so the
    /// caller (which knows whether the new process is a child of an
    /// existing process / what was inherited) can override.
    pub fn new() -> Self {
        Self {
            table: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Pre-seed slots 0 / 1 / 2 with [`Handle::Stdin`] / `Stdout` /
    /// `Stderr` and bump `next_id` past them.
    ///
    /// No-op on a populated table: overwriting an existing slot via
    /// `BTreeMap::insert` would drop a live `Handle` without revoking
    /// its backing — for a `NetChannel` that drops only the manager's
    /// strong ref while k_net keeps its own clone, so the backing never
    /// reaches `pending_frees` (leak). Returning early instead of
    /// asserting keeps the guard live in release builds.
    pub fn seed_stdio(&mut self) {
        if !self.table.is_empty() {
            return;
        }
        self.table.insert(0, Slot::new(Handle::Stdin));
        self.table.insert(1, Slot::new(Handle::Stdout));
        self.table.insert(2, Slot::new(Handle::Stderr));
        if self.next_id < 3 {
            self.next_id = 3;
        }
    }

    /// Allocate a fresh Fd and insert `handle` under it with default
    /// flags (`cloexec = false`, `nonblock = false`). Returns the Fd,
    /// or `None` when the namespace is exhausted ([`FD_MAX`] reached).
    pub fn insert(&mut self, handle: Handle) -> Option<Fd> {
        self.insert_with_flags(handle, false, false)
    }

    /// Allocate a fresh Fd and insert `handle` with explicit flags.
    /// Used by syscall handlers that honor `O_CLOEXEC` / `O_NONBLOCK`
    /// at create time (e.g. `eventfd(2)` with `EFD_CLOEXEC`).
    pub fn insert_with_flags(
        &mut self,
        handle: Handle,
        cloexec: bool,
        nonblock: bool,
    ) -> Option<Fd> {
        if self.next_id > FD_MAX {
            return None;
        }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1)?;
        self.table
            .insert(id, Slot::with_flags(handle, cloexec, nonblock));
        Some(id)
    }

    /// Insert `slot` at a caller-specified `fd`. If a slot already
    /// lives at `fd`, the old value is returned for the caller to
    /// close. Used by `dup2(oldfd, newfd)` to target a specific number.
    /// Bumps `next_id` past `fd` if needed so future
    /// [`insert`](Self::insert)s don't collide.
    pub fn insert_at(&mut self, fd: Fd, slot: Slot) -> Option<Slot> {
        if fd > FD_MAX {
            return None;
        }
        let old = self.table.insert(fd, slot);
        if self.next_id <= fd {
            self.next_id = fd.saturating_add(1);
        }
        old
    }

    pub fn get(&self, fd: Fd) -> Option<&Handle> {
        self.table.get(&fd).map(|s| &s.handle)
    }

    pub fn get_mut(&mut self, fd: Fd) -> Option<&mut Handle> {
        self.table.get_mut(&fd).map(|s| &mut s.handle)
    }

    pub fn get_slot(&self, fd: Fd) -> Option<&Slot> {
        self.table.get(&fd)
    }

    pub fn get_slot_mut(&mut self, fd: Fd) -> Option<&mut Slot> {
        self.table.get_mut(&fd)
    }

    pub fn remove(&mut self, fd: Fd) -> Option<Handle> {
        self.table.remove(&fd).map(|s| s.handle)
    }

    pub fn remove_slot(&mut self, fd: Fd) -> Option<Slot> {
        self.table.remove(&fd)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Fd, &Handle)> {
        self.table.iter().map(|(fd, s)| (fd, &s.handle))
    }

    pub fn iter_slots(&self) -> impl Iterator<Item = (&Fd, &Slot)> {
        self.table.iter()
    }

    /// Consume and drain the table. Used at process teardown so the
    /// caller can walk each handle by value (e.g. to free per-fd
    /// scratch backings via `free_backing` before the BTreeMap
    /// drops).
    pub fn into_iter(self) -> impl Iterator<Item = (Fd, Handle)> {
        self.table.into_iter().map(|(fd, s)| (fd, s.handle))
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
    pub fn len(&self) -> usize {
        self.table.len()
    }
}
