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
//! 3. The filesystem is published via [`install`]; consumers (the
//!    fs syscall handlers: fs_open / fs_stat / fs_read / readdir, and
//!    path-mode spawn) reach it through [`mounted`].

use orbit_abi::fs::Stat;

pub mod tar;

/// Filesystem-internal inode id. Stable for the lifetime of the mount.
/// 0 is reserved for "no inode".
pub type Inode = u32;

#[derive(Debug, Clone, Copy)]
pub enum FsErr {
    NotFound,
    NotRegular,
    /// fd is a regular file (or non-dir) where a directory was expected
    /// — `fs_readdir` on a `Kind::Reg` open file.
    NotADirectory,
    BadInode,
    /// Underlying block device unavailable or returned an error.
    IoError,
    /// Argument outside the file's bounds (offset past EOF, len 0,
    /// non-sector-aligned where the FS requires alignment, …). For
    /// `readdir` it also signals "buffer too small to hold the next
    /// entry" — caller grows the buffer or accepts the partial read.
    BadRange,
}

pub trait Filesystem: Send + Sync {
    /// Stable device id for this mount. Matches `Stat.st_dev` and
    /// keys the page cache (`(dev, lba)`); each mount must return a
    /// distinct value. Tarfs pins to `1` (single-mount today).
    fn dev_id(&self) -> u8;

    /// Translate a file-relative page index to the page-aligned
    /// LBA on the backing device. Drives both the cache key
    /// (computed at lookup time) and the DMA submission (PA derived
    /// from this LBA + the cache slot's frame).
    ///
    /// Tarfs: `entry.data_sector + page_idx * (PAGE_SIZE / SECTOR_SIZE)`.
    /// Future FSes (ext2/minix) walk indirect blocks here.
    ///
    /// Errors `BadInode` / `NotRegular` / `BadRange` mirror the
    /// page-cache fill path (`submit_blk_read_cached`).
    fn lba_for_page(&self, ino: Inode, page_idx: u64) -> Result<u64, FsErr>;

    /// Resolve `path` to an inode. Path is normalized: leading `./`
    /// is stripped at parse time, lookup keys are absolute (`/foo`).
    /// Returns `NotFound` for paths the FS doesn't have.
    fn open(&self, path: &str) -> Result<Inode, FsErr>;

    /// Fill `*out` with stat info for `ino`. Synchronous — tar's
    /// table is in-memory.
    fn stat(&self, ino: Inode) -> Result<Stat, FsErr>;

    /// File size in bytes (0 for directories). Mostly a convenience
    /// for the read syscall handler that needs the cap without the
    /// rest of the stat fields.
    fn size(&self, ino: Inode) -> Result<u64, FsErr>;

    /// Pack zero or more directory entries from `ino` into `out`,
    /// starting at `cursor`. Returns `(bytes_written, next_cursor)`
    /// — the manager stores `next_cursor` back on the `OpenFile` and
    /// returns `bytes_written` to userland. `bytes_written == 0`
    /// signals end-of-directory.
    ///
    /// Cursor is opaque to userland — every value the kernel returns
    /// here is fed back unchanged on the next `readdir` call. Today's
    /// tarfs implementation uses it as a sorted-children index;
    /// future filesystems can reinterpret as needed.
    ///
    /// Errors:
    /// - `NotADirectory` — `ino` is not a directory inode.
    /// - `BadInode` — `ino` is unknown.
    /// - `BadRange` — `out` is too small for even one entry. Caller
    ///   grows the buffer; cursor is not advanced.
    fn readdir(&self, ino: Inode, cursor: u64, out: &mut [u8]) -> Result<(usize, u64), FsErr>;
}

/// Single global mount slot. Write-once at boot from hart 0; the fs
/// syscall handlers Acquire-load.
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
