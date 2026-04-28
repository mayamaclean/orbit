//! VFS skeleton. One mounted filesystem at `/`, pluggable behind the
//! [`Filesystem`] trait. Today the only implementation is [`tar::Tarfs`]
//! (read-only ustar served from virtio-blk).
//!
//! Mount flow:
//! 1. [`crate::drivers::virtio_blk_dev::setup_virtio_blk`] brings up the
//!    device and synchronously reads LBA 0 to confirm the disk is
//!    healthy.
//! 2. Same function calls [`tar::Tarfs::mount`] which walks the
//!    archive sector-by-sector via `Block::read_blocks_blocking`,
//!    builds a `BTreeMap<String, TarInode>`, and returns the mounted
//!    filesystem.
//! 3. The filesystem is published via [`install`]; consumers
//!    (currently nobody, eventually 12d's syscall handlers) reach it
//!    through [`mounted`].

use orbit_abi::fs::Stat;
use process::CompletionHandle;

pub mod tar;

/// Filesystem-internal inode id. Stable for the lifetime of the mount.
/// 0 is reserved for "no inode".
pub type Inode = u32;

#[derive(Debug, Clone, Copy)]
pub enum FsErr {
    NotFound,
    NotRegular,
    BadInode,
    /// Underlying block device unavailable or returned an error.
    IoError,
    /// Argument outside the file's bounds (offset past EOF, len 0,
    /// non-sector-aligned where the FS requires alignment, …).
    BadRange,
}

pub trait Filesystem: Send + Sync {
    /// Resolve `path` to an inode. Path is normalized: leading `./`
    /// is stripped at parse time, lookup keys are absolute (`/foo`).
    /// Returns `NotFound` for paths the FS doesn't have.
    fn open(&self, path: &str) -> Result<Inode, FsErr>;

    /// Submit one sector-sized read. The implementation parks `handle`
    /// on the underlying block-device completion; signaling carries
    /// `0` on success or a negative error code.
    ///
    /// v1 contract:
    /// - `len` must equal 512.
    /// - `off` must be a 512-byte multiple.
    /// - `off + len` must not exceed the inode's size (rounded up to
    ///   the next sector — the last sector of a file is read fully and
    ///   the caller trims).
    ///
    /// Multi-sector reads chunk at the syscall layer (12d's `fs_read`).
    ///
    /// # Safety
    /// `dst_pa` must reference 512 bytes of memory the kernel keeps
    /// mapped until `handle` signals.
    unsafe fn read_async(
        &self,
        ino: Inode,
        off: u64,
        len: u32,
        dst_pa: u64,
        handle: CompletionHandle,
    ) -> Result<(), FsErr>;

    /// Fill `*out` with stat info for `ino`. Synchronous — tar's
    /// table is in-memory.
    fn stat(&self, ino: Inode) -> Result<Stat, FsErr>;

    /// File size in bytes (0 for directories). Mostly a convenience
    /// for the read syscall handler that needs the cap without the
    /// rest of the stat fields.
    fn size(&self, ino: Inode) -> Result<u64, FsErr>;
}

/// Single global mount slot. Write-once at boot from hart 0; readers
/// (eventually 12d's syscall handlers) Acquire-load.
static MOUNTED: spin::Once<&'static dyn Filesystem> = spin::Once::new();

/// Install the boot-mounted filesystem. Idempotent — a second call
/// from the same hart is a no-op (the first install wins).
pub fn install(fs: &'static dyn Filesystem) {
    MOUNTED.call_once(|| fs);
}

/// Return the boot-mounted filesystem, or `None` if no mount has
/// completed (early boot or the device wasn't present).
pub fn mounted() -> Option<&'static dyn Filesystem> {
    MOUNTED.get().copied()
}
