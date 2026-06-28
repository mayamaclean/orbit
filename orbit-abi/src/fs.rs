//! Filesystem syscall ABI types — `Stat`, `OpenFlags`, mode constants.
//! Shared between user wrappers (`crate::user::fs_*`) and the kernel
//! VFS layer (`kmain::kernel::fs`).
//!
//! Layout matches Linux's generic-arch `struct stat`
//! (`include/uapi/asm-generic/stat.h` — what riscv64 Linux uses).
//! 128 bytes on 64-bit. Picked over a hand-rolled shape so a future
//! POSIX `std::fs::Metadata` shim translates field-for-field.
//! A forward-compat path (`statx`-style new syscall + new struct) can
//! land if we ever need a field that doesn't fit here.

/// `flags` argument to `fs_open`. v1 is read-only — no actual flag
/// bits today; the field is reserved for future O_NONBLOCK / O_CREAT
/// without a syscall renumber. Pass 0.
pub const OPEN_RDONLY: usize = 0;

/// `whence` argument to [`crate::user::fs_seek`]. POSIX numbering
/// (`SEEK_SET = 0`, `SEEK_CUR = 1`, `SEEK_END = 2`) so a future libc
/// shim can pass the value through unchanged.
pub const SEEK_SET: u32 = 0;
pub const SEEK_CUR: u32 = 1;
pub const SEEK_END: u32 = 2;

// File-type bits for `Stat::st_mode`. POSIX-shape (octal). High bits
// encode the type; low 12 bits are permission + setuid/setgid/sticky.
pub const S_IFMT: u32 = 0o170000;
pub const S_IFREG: u32 = 0o100000;
pub const S_IFDIR: u32 = 0o040000;
pub const S_IFLNK: u32 = 0o120000;

/// Access-mode bits for `vaccess()`-style checks and the future POSIX
/// `access(2)` syscall. Numeric values match Linux/POSIX `R_OK` /
/// `W_OK` / `X_OK` so a future libc shim threads them through
/// unchanged. Composable: `ACCESS_R_OK | ACCESS_W_OK` checks both.
pub const ACCESS_R_OK: u32 = 4;
pub const ACCESS_W_OK: u32 = 2;
pub const ACCESS_X_OK: u32 = 1;

/// Sector size used for `st_blocks` accounting. POSIX defines
/// `st_blocks` as "number of 512-byte units" regardless of the
/// filesystem's actual sector geometry.
pub const STAT_BLOCK_UNIT: u64 = 512;

// `d_type` values for [`DirEntry`]. Numbers match Linux's `dirent.h`
// (`DT_*`) so a future POSIX shim doesn't have to translate. Only the
// types orbit can produce today are listed; add variants as the FS
// learns to emit them.
/// Type unknown — caller should `fs_stat` to disambiguate. v1 tarfs
/// never returns this, but the constant exists so future filesystems
/// without inline type info have a value to use.
pub const DT_UNKNOWN: u8 = 0;
/// Directory.
pub const DT_DIR: u8 = 4;
/// Regular file.
pub const DT_REG: u8 = 8;
/// Symbolic link. Reserved — tarfs v1 doesn't produce these.
pub const DT_LNK: u8 = 10;

/// Packed directory entry header. The kernel writes a stream of
/// `DirEntry` records back-to-back into the user buffer for
/// `fs_readdir`; each record carries its name immediately after the
/// header, then padding so the next record starts on an 8-byte
/// boundary.
///
/// Walk order: read header at offset 0, name spans `[12 ..
/// 12+d_namelen]`, advance by `d_reclen`. Repeat until `d_reclen`
/// would exit the returned byte count.
///
/// Layout matches Linux's `linux_dirent64` minus `d_off` — orbit keeps
/// the directory cursor on the kernel side (in the `OpenFile`'s
/// `dir_cursor`) rather than threading it through user buffers, so
/// there's nothing for `d_off` to carry.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct DirEntry {
    /// FS-internal inode number. Matches what `fs_stat` returns in
    /// `Stat::st_ino` for the same path.
    pub d_ino: u64,
    /// Total bytes from the start of this record to the start of the
    /// next, including padding. Always a multiple of 8.
    pub d_reclen: u16,
    /// `DT_*` type bits. `DT_UNKNOWN` means "fs_stat to find out".
    pub d_type: u8,
    /// Length of the name in bytes that follow this header. No NUL
    /// terminator — caller slices `[..d_namelen]` raw.
    pub d_namelen: u8,
}

/// Header size in bytes. Pinned for `fs_readdir` consumers that walk
/// records without dragging in `core::mem::size_of`.
pub const DIRENT_HDR_LEN: usize = 12;

/// Alignment that `d_reclen` is rounded up to. Matches Linux. Picked so
/// a `DirEntry` after the name lands on a u64 boundary, which keeps
/// the unaligned-load story simple for callers.
pub const DIRENT_ALIGN: usize = 8;

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

/// POSIX-style mode-bit access check. Returns `Ok(())` if the
/// `(euid, egid, supplementary groups)` credential triple has the
/// access bits in `want` (composition of [`ACCESS_R_OK`] /
/// [`ACCESS_W_OK`] / [`ACCESS_X_OK`]) against the file metadata in
/// `st`. Otherwise returns `EACCES`.
///
/// Rules (POSIX, matches OpenBSD `vaccess`):
///   - `euid == 0` (root): bypass, always `Ok`.
///   - `euid == st.st_uid`: check owner bits (`st_mode >> 6 & 7`).
///   - `egid == st.st_gid` OR `st.st_gid` ∈ `groups`: check group
///     bits (`st_mode >> 3 & 7`).
///   - Else: check other bits (`st_mode & 7`).
///
/// First matching branch wins — a process whose euid matches `st_uid`
/// uses owner bits even if its supplementary groups also include
/// `st_gid`. This matches POSIX (compare glibc `__access_internal`,
/// OpenBSD `vaccess`).
///
/// **Path-walk traversal isn't modeled here.** POSIX requires search
/// (X) on every parent directory of a path; this helper only checks
/// the final inode. tarfs's `open(path)` is a single-pass internal
/// walk that doesn't surface intermediate inodes — wiring per-segment
/// `vaccess(X)` waits on a vfs-layer rewrite.
pub fn vaccess(
    euid: u32,
    egid: u32,
    groups: &[u32],
    st: &Stat,
    want: u32,
) -> Result<(), crate::errno::Errno> {
    if euid == 0 {
        return Ok(());
    }
    let bits = if euid == st.st_uid {
        (st.st_mode >> 6) & 7
    }
    else if egid == st.st_gid || groups.contains(&st.st_gid) {
        (st.st_mode >> 3) & 7
    }
    else {
        st.st_mode & 7
    };
    if (bits & want) == want {
        Ok(())
    }
    else {
        Err(crate::errno::Errno::new(crate::errno::EACCES))
    }
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

    fn st(mode: u32, uid: u32, gid: u32) -> Stat {
        let mut s = Stat::default();
        s.st_mode = mode;
        s.st_uid = uid;
        s.st_gid = gid;
        s
    }

    #[test]
    fn vaccess_root_bypass_regardless_of_mode() {
        // Root reads a 0o000 file owned by someone else. Bypass should
        // fire before mode bits are consulted.
        let s = st(0o000, 1000, 1000);
        assert!(vaccess(0, 0, &[], &s, ACCESS_R_OK).is_ok());
        assert!(vaccess(0, 0, &[], &s, ACCESS_W_OK).is_ok());
        assert!(vaccess(0, 0, &[], &s, ACCESS_R_OK | ACCESS_W_OK | ACCESS_X_OK).is_ok());
    }

    #[test]
    fn vaccess_owner_branch_uses_owner_bits() {
        // Owner of a 0o600 file: rw permitted, x denied.
        let s = st(0o600, 1000, 1000);
        assert!(vaccess(1000, 1000, &[], &s, ACCESS_R_OK).is_ok());
        assert!(vaccess(1000, 1000, &[], &s, ACCESS_W_OK).is_ok());
        assert_eq!(
            vaccess(1000, 1000, &[], &s, ACCESS_X_OK).unwrap_err().0,
            crate::errno::EACCES
        );
    }

    #[test]
    fn vaccess_owner_branch_ignores_group_and_other() {
        // 0o007 = ---rwxrwx → owner bits are 0. Owner is denied even
        // though group/other would allow. POSIX first-match-wins.
        let s = st(0o007, 1000, 1000);
        assert_eq!(
            vaccess(1000, 1000, &[1000], &s, ACCESS_R_OK).unwrap_err().0,
            crate::errno::EACCES
        );
    }

    #[test]
    fn vaccess_group_branch_via_egid() {
        // Non-owner whose egid matches: 0o040 = ---r----- (group r only).
        let s = st(0o040, 1000, 100);
        assert!(vaccess(2000, 100, &[], &s, ACCESS_R_OK).is_ok());
        assert_eq!(
            vaccess(2000, 100, &[], &s, ACCESS_W_OK).unwrap_err().0,
            crate::errno::EACCES
        );
    }

    #[test]
    fn vaccess_group_branch_via_supplementary() {
        // Non-owner, egid != st_gid, but st_gid is in supplementary
        // groups. Group bits apply.
        let s = st(0o040, 1000, 100);
        assert!(vaccess(2000, 999, &[100], &s, ACCESS_R_OK).is_ok());
    }

    #[test]
    fn vaccess_other_branch_when_no_match() {
        // 0o644 file owned by uid=0 — non-owner, non-group caller
        // hits other branch (read OK, write denied).
        let s = st(0o644, 0, 0);
        assert!(vaccess(1000, 1000, &[], &s, ACCESS_R_OK).is_ok());
        assert_eq!(
            vaccess(1000, 1000, &[], &s, ACCESS_W_OK).unwrap_err().0,
            crate::errno::EACCES
        );
    }

    #[test]
    fn vaccess_owner_takes_precedence_over_group() {
        // 0o407 = r------rwx. Owner has r, group has nothing, other
        // has rwx. Owner caller is denied W (because owner bits don't
        // include W) even though other-bits would allow it.
        let s = st(0o407, 1000, 1000);
        assert_eq!(
            vaccess(1000, 1000, &[], &s, ACCESS_W_OK).unwrap_err().0,
            crate::errno::EACCES
        );
    }

    #[test]
    fn vaccess_zero_mode_denies_all_non_root() {
        // chmod 000 on a file: nobody but root can do anything.
        let s = st(0, 1000, 1000);
        for caller_uid in [1000u32, 1234, 9999] {
            for want in [ACCESS_R_OK, ACCESS_W_OK, ACCESS_X_OK] {
                assert!(
                    vaccess(caller_uid, 1000, &[1000], &s, want).is_err(),
                    "uid={caller_uid} want={want:#x} should EACCES on chmod 000",
                );
            }
        }
        // root still bypasses.
        assert!(vaccess(0, 0, &[], &s, ACCESS_R_OK | ACCESS_W_OK | ACCESS_X_OK).is_ok());
    }

    #[test]
    fn vaccess_combined_want_requires_all_bits() {
        // 0o500 = r-x------ (owner R+X). Asking for R+W requires both;
        // R alone OK, R+W denied (W not present in owner bits).
        let s = st(0o500, 1000, 1000);
        assert!(vaccess(1000, 1000, &[], &s, ACCESS_R_OK).is_ok());
        assert!(vaccess(1000, 1000, &[], &s, ACCESS_X_OK).is_ok());
        assert_eq!(
            vaccess(1000, 1000, &[], &s, ACCESS_R_OK | ACCESS_W_OK,)
                .unwrap_err()
                .0,
            crate::errno::EACCES
        );
    }

    /// Pin DirEntry layout — the kernel writes packed records keyed
    /// off `DIRENT_HDR_LEN`, and userland walks with `d_reclen` strides.
    /// Reordering or padding the struct silently breaks `fs_readdir`.
    #[test]
    fn direntry_header_layout() {
        assert_eq!(core::mem::size_of::<DirEntry>(), DIRENT_HDR_LEN);
        // `#[repr(C, packed)]` gives alignment 1; the kernel emits
        // records that *land* on 8-byte boundaries via padding, but
        // the struct itself is byte-packed so misaligned loads work.
        assert_eq!(core::mem::align_of::<DirEntry>(), 1);
    }
}
