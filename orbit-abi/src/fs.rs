//! Filesystem syscall ABI types — `Stat`, `OpenFlags`, mode constants.
//! Shared between user wrappers (`crate::user::fs_*`) and the kernel
//! VFS layer (`kmain::kernel::fs`).
//!
//! Layout matches Linux's generic-arch `struct stat`
//! (`include/uapi/asm-generic/stat.h` — what riscv64 Linux uses).
//! 128 bytes on 64-bit. Picked over a hand-rolled shape so a future
//! POSIX `std::fs::Metadata` shim translates field-for-field.
//! Forward-compat path (`statx`-style new syscall + new struct) lives
//! in §12+ when we actually need a field that doesn't fit here.

/// `flags` argument to `fs_open`. v1 is read-only — no actual flag
/// bits today; the field is reserved for future O_NONBLOCK / O_CREAT
/// without a syscall renumber. Pass 0.
pub const OPEN_RDONLY: usize = 0;

// File-type bits for `Stat::st_mode`. POSIX-shape (octal). High bits
// encode the type; low 12 bits are permission + setuid/setgid/sticky.
pub const S_IFMT:   u32 = 0o170000;
pub const S_IFREG:  u32 = 0o100000;
pub const S_IFDIR:  u32 = 0o040000;
pub const S_IFLNK:  u32 = 0o120000;

/// Sector size used for `st_blocks` accounting. POSIX defines
/// `st_blocks` as "number of 512-byte units" regardless of the
/// filesystem's actual sector geometry.
pub const STAT_BLOCK_UNIT: u64 = 512;

/// POD `struct stat`. Layout pinned to Linux generic-arch — do not
/// reorder fields or change widths. Add new fields via a future
/// `fs_statx` syscall (separate, larger struct) rather than tail
/// growth here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stat {
    /// Filesystem device id. Single-mount today, so always 1.
    pub st_dev: u64,
    /// FS-internal inode number. Stable for the lifetime of the
    /// mount. 0 is reserved for "no inode".
    pub st_ino: u64,
    /// File type (`S_IF*` bits) packed with permission bits.
    pub st_mode: u32,
    /// Hard-link count. Tar v1 has no hard links → always 1.
    pub st_nlink: u32,
    /// Owning user id (parsed from the tar header).
    pub st_uid: u32,
    /// Owning group id (parsed from the tar header).
    pub st_gid: u32,
    /// Device id this file *represents* (only meaningful for char/
    /// block-device inodes). Always 0 in v1.
    pub st_rdev: u64,
    pub __pad1: u64,
    /// File size in bytes. 0 for directories.
    pub st_size: i64,
    /// FS preferred I/O block size. Tarfs is sector-granular → 512.
    pub st_blksize: i32,
    pub __pad2: i32,
    /// Number of [`STAT_BLOCK_UNIT`]-byte units the file occupies.
    /// `ceil(st_size / 512)` for regular files; 0 for dirs.
    pub st_blocks: i64,
    /// Last access time. Tarfs has no atime tracking → mirrors
    /// `st_mtime`.
    pub st_atime: i64,
    pub st_atime_nsec: u64,
    /// Last-modified time, parsed from the tar header (seconds since
    /// the Unix epoch).
    pub st_mtime: i64,
    pub st_mtime_nsec: u64,
    /// Last status-change time. Same source as `st_mtime` for tar.
    pub st_ctime: i64,
    pub st_ctime_nsec: u64,
    pub __unused4: u32,
    pub __unused5: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the on-wire size — the kernel side `fs_stat` writes
    /// `size_of::<Stat>` bytes into the user buffer, so changing
    /// the layout silently is an ABI break.
    #[test]
    fn size_matches_linux_generic() {
        assert_eq!(core::mem::size_of::<Stat>(), 128);
    }
}
