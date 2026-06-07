//! Per-process handle table kinds.
//!
//! Every fd-shaped resource the kernel hands back to userspace carries
//! a kind tag from this enum. The tag is part of the ABI: `fstat(fd)`
//! returns it so consumers (mio's `Selector::register`, libc `fstat`
//! shims, future `/proc/<pid>/fd/`-style tooling) can dispatch on what
//! the fd actually backs.
//!
//! **Numeric values are ABI-stable.** Reserve numbers up-front for
//! variants we don't ship yet so future tooling parsing kind tags
//! doesn't churn when a new variant lands.

/// Tag carried in [`Stat::kind`](crate::fs::Stat) and returned by the
/// kind-aware introspection paths. `repr(u8)` keeps the on-wire field
/// compact and ABI-stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HandleKind {
    /// NetChannel socket (TCP today; UDP/IPv6 future). Shared-memory
    /// region mapped in `UPROC_SHARED_BASE..UPROC_SHARED_END` carries
    /// the ring state; `fstat` returns the VA + region size + peer
    /// address + connection state for `FromRawFd` rehydration.
    NetChannel = 1,

    /// Regular file or directory on the active filesystem. `read` /
    /// `fstat` / `fs_seek` route to the fs layer; `write` returns
    /// `EROFS` while tarfs is read-only.
    File = 2,

    /// Standard input — seeded in slot 0 at process creation. `read`
    /// drains the cooked-mode line buffer / key-event ring; `write`
    /// returns `EBADF`.
    Stdin = 3,

    /// Standard output — seeded in slot 1. `write` routes to the
    /// console (k_gpu compositor + scrollback); `read` returns `EBADF`.
    Stdout = 4,

    /// Standard error — seeded in slot 2. Same shape as
    /// [`HandleKind::Stdout`] today; a future split may direct stderr
    /// to a separate sink.
    Stderr = 5,

    /// EventFd — shared-memory counter + parked-tid hint. `read(fd, &mut
    /// [u8; 8])` swaps the count to 0 (or decrements by 1 if
    /// `EFD_SEMAPHORE`); `write(fd, &val_le_bytes)` adds. Used for
    /// cross-thread doorbells (mio's `Waker`, future channel-based
    /// signaling primitives).
    EventFd = 6,

    /// Read end of a pipe. **Reserved**; lands with the
    /// `Stdio::MakePipe` / shell-pipeline milestone.
    PipeRead = 7,

    /// Write end of a pipe. **Reserved**; counterpart of
    /// [`HandleKind::PipeRead`].
    PipeWrite = 8,

    /// pidfd — refers to a child process. **Reserved**; lands with the
    /// `Child::kill` / `Child::try_wait` / `tokio::process` milestone.
    /// `read(fd)` blocks on child exit and returns the exit status;
    /// `close(fd)` releases the reference without affecting the child.
    Pidfd = 9,
}

impl HandleKind {
    pub const fn from_u8(n: u8) -> Option<Self> {
        Some(match n {
            1 => Self::NetChannel,
            2 => Self::File,
            3 => Self::Stdin,
            4 => Self::Stdout,
            5 => Self::Stderr,
            6 => Self::EventFd,
            7 => Self::PipeRead,
            8 => Self::PipeWrite,
            9 => Self::Pidfd,
            _ => return None,
        })
    }

    /// `true` when the kind is the reading side of a kernel-managed
    /// stream of bytes (file, stdin, pipe read, netchannel rx). Used
    /// by `read(fd)` dispatch to fail closed on kinds that don't carry
    /// readable bytes (`Stdout`, `Stderr`, `PipeWrite`).
    pub const fn is_readable(self) -> bool {
        matches!(
            self,
            Self::NetChannel
                | Self::File
                | Self::Stdin
                | Self::EventFd
                | Self::PipeRead
                | Self::Pidfd
        )
    }

    /// `true` when the kind accepts byte writes. Mirror of
    /// [`Self::is_readable`].
    pub const fn is_writable(self) -> bool {
        matches!(
            self,
            Self::NetChannel | Self::Stdout | Self::Stderr | Self::EventFd | Self::PipeWrite
        )
    }
}

/// `ch_inspect(fd, *mut ChInfo)` reply. 64-byte, cache-line-aligned,
/// fixed-layout per the orbit ABI surface — do not reorder fields or
/// repurpose padding without a coordinated ABI bump.
///
/// Population rules per kind:
///
/// - `NetChannel`: `region_va` + `region_size` point at the
///   `NetChannel` shared header; `peer_addr` / `peer_port` reflect
///   the remote endpoint (zeroed when no session is active);
///   `state` is the `net_channel::channel_state::*` constant.
///   `flags` is zero today.
///
/// - `EventFd`: `region_va` + `region_size` point at the EventFd
///   header. `flags` carries the `EFD_*` bits the fd was created
///   with. `peer_addr` / `peer_port` / `state` are zero.
///
/// - `File`: `region_va` = `region_size` = 0 (fs reads bounce
///   through per-fd scratch; userspace can't peek directly). `flags`
///   reserved.
///
/// - `Stdin` / `Stdout` / `Stderr`: zeros except `kind`.
#[repr(C, align(64))]
#[derive(Debug, Default, Clone, Copy)]
pub struct ChInfo {
    /// [`HandleKind`] tag for the slot at `fd`.
    pub kind: u8,
    pub _pad0: [u8; 7],

    /// User VA of the shared region backing the fd (NetChannel,
    /// EventFd). Zero for kinds without a userspace-visible mapping.
    pub region_va: u64,

    /// Length of the shared region in bytes. Zero when `region_va`
    /// is zero.
    pub region_size: u32,

    /// Kind-specific flag bits. EventFd: `EFD_NONBLOCK`/`SEMAPHORE`/
    /// `CLOEXEC` snapshot. Others: zero.
    pub flags: u32,

    /// NetChannel only: peer IPv4 address in network byte order
    /// (matches `NetChannelCurrent.peer_addr.load()`'s wire form).
    /// Zero for other kinds.
    pub peer_addr: u32,

    /// NetChannel only: peer port. Zero for other kinds.
    pub peer_port: u16,
    pub _pad1: [u8; 2],

    /// NetChannel only: session state (`channel_state::ACTIVE` etc.).
    /// Zero for other kinds.
    pub state: i32,

    /// Reserved for future fields (e.g. fd flags after `fcntl` lands,
    /// open-mode, file size). Always zero in v1.
    pub _reserved: [u8; 24],
}

const _: () = {
    assert!(core::mem::size_of::<ChInfo>() == 64);
    assert!(core::mem::align_of::<ChInfo>() == 64);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_tags_are_load_bearing() {
        // ABI: do not renumber.
        assert_eq!(HandleKind::NetChannel as u8, 1);
        assert_eq!(HandleKind::File as u8, 2);
        assert_eq!(HandleKind::Stdin as u8, 3);
        assert_eq!(HandleKind::Stdout as u8, 4);
        assert_eq!(HandleKind::Stderr as u8, 5);
        assert_eq!(HandleKind::EventFd as u8, 6);
        assert_eq!(HandleKind::PipeRead as u8, 7);
        assert_eq!(HandleKind::PipeWrite as u8, 8);
        assert_eq!(HandleKind::Pidfd as u8, 9);
    }

    #[test]
    fn from_u8_round_trips() {
        for k in [
            HandleKind::NetChannel,
            HandleKind::File,
            HandleKind::Stdin,
            HandleKind::Stdout,
            HandleKind::Stderr,
            HandleKind::EventFd,
            HandleKind::PipeRead,
            HandleKind::PipeWrite,
            HandleKind::Pidfd,
        ] {
            assert_eq!(HandleKind::from_u8(k as u8), Some(k));
        }
        assert_eq!(HandleKind::from_u8(0), None);
        assert_eq!(HandleKind::from_u8(10), None);
        assert_eq!(HandleKind::from_u8(255), None);
    }

    #[test]
    fn readable_writable_classification() {
        assert!(HandleKind::Stdin.is_readable());
        assert!(!HandleKind::Stdin.is_writable());
        assert!(!HandleKind::Stdout.is_readable());
        assert!(HandleKind::Stdout.is_writable());
        assert!(HandleKind::NetChannel.is_readable());
        assert!(HandleKind::NetChannel.is_writable());
        assert!(HandleKind::File.is_readable());
        assert!(!HandleKind::File.is_writable());
    }
}
