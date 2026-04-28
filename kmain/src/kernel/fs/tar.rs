//! ustar-formatted read-only filesystem.
//!
//! Eager-mount: walk the archive sector-by-sector at boot via the
//! polled-completion `Block::read_blocks_blocking`, parse each header,
//! build a `BTreeMap<String, Inode>` index plus a `Vec<TarInode>`
//! table keyed by inode id. Once mount returns, the FS no longer
//! touches the block device synchronously — `read_async` submits
//! through the IRQ-driven path in
//! [`crate::drivers::virtio_blk_dev::submit_blk_read`].

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use orbit_abi::fs::{S_IFDIR, S_IFREG, STAT_BLOCK_UNIT, Stat};
use process::CompletionHandle;
use tracing::{info, warn};
use virtio_blk::{Block, BlockError, SECTOR_SIZE};

use crate::drivers::virtio_blk_dev;
use crate::kernel::fs::{Filesystem, FsErr, Inode};

const HEADER_NAME_LEN: usize = 100;
const HEADER_MODE_OFFSET: usize = 100;
const HEADER_MODE_LEN: usize = 8;
const HEADER_UID_OFFSET: usize = 108;
const HEADER_UID_LEN: usize = 8;
const HEADER_GID_OFFSET: usize = 116;
const HEADER_GID_LEN: usize = 8;
const HEADER_SIZE_OFFSET: usize = 124;
const HEADER_SIZE_LEN: usize = 12;
const HEADER_MTIME_OFFSET: usize = 136;
const HEADER_MTIME_LEN: usize = 12;
const HEADER_TYPEFLAG_OFFSET: usize = 156;
const HEADER_MAGIC_OFFSET: usize = 257;
const HEADER_PREFIX_OFFSET: usize = 345;
const HEADER_PREFIX_LEN: usize = 155;

const USTAR_MAGIC: &[u8; 6] = b"ustar\0";
// GNU tar emits "ustar  \0" (two spaces, then NUL at byte 7) — the
// magic at offset 257..263 is "ustar ". Tolerated.
const USTAR_MAGIC_GNU: &[u8; 6] = b"ustar ";

const TYPEFLAG_REG_NUL: u8 = 0;
const TYPEFLAG_REG_0: u8 = b'0';
const TYPEFLAG_DIR: u8 = b'5';

#[derive(Clone, Copy, Debug)]
enum Kind {
    Reg,
    Dir,
}

#[derive(Clone, Debug)]
struct TarInode {
    path: String,
    kind: Kind,
    /// First data sector. 0 for directories (no data).
    data_sector: u64,
    size: u64,
    /// Permission bits (low 12). Type bits (`S_IFREG` / `S_IFDIR`)
    /// land in the synthesized `st_mode` at `stat()` time.
    mode_perms: u32,
    uid: u32,
    gid: u32,
    /// Seconds since Unix epoch from the tar header. Mirrored into
    /// st_atime/st_mtime/st_ctime — tarfs has no separate atime/ctime
    /// tracking.
    mtime: i64,
}

#[derive(Debug)]
pub enum MountErr {
    Block(BlockError),
    BadHeader { lba: u64 },
    BadMagic { lba: u64 },
}

impl From<BlockError> for MountErr {
    fn from(e: BlockError) -> Self {
        MountErr::Block(e)
    }
}

pub struct Tarfs {
    /// inode id → entry. `inodes[0]` is reserved as the null sentinel;
    /// real ids start at 1.
    inodes: Vec<Option<TarInode>>,
    /// Path → inode id index. BTreeMap is O(log n) for lookup which is
    /// fine at the "few binaries" scale of v1.
    by_path: BTreeMap<String, Inode>,
}

impl Tarfs {
    /// Walk the archive, building the inode table. End-of-archive is
    /// two consecutive zero blocks per the ustar spec.
    ///
    /// Polled completion via `Block::read_blocks_blocking` — only safe
    /// because mount runs before `BLOCK_PTR` is published, so no
    /// async submitter can race.
    pub fn mount(dev: &mut Block) -> Result<Self, MountErr> {
        let mut inodes: Vec<Option<TarInode>> = Vec::new();
        inodes.push(None); // inode 0 = null sentinel
        let mut by_path: BTreeMap<String, Inode> = BTreeMap::new();

        let capacity = dev.capacity_sectors();
        let mut lba: u64 = 0;
        let mut prev_zero = false;

        while lba < capacity {
            let mut buf = [0u8; SECTOR_SIZE];
            unsafe { dev.read_blocks_blocking(lba, &mut buf)? };

            if buf.iter().all(|&b| b == 0) {
                if prev_zero {
                    break;
                }
                prev_zero = true;
                lba += 1;
                continue;
            }
            prev_zero = false;

            let magic = &buf[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 6];
            if magic != USTAR_MAGIC && magic != USTAR_MAGIC_GNU {
                return Err(MountErr::BadMagic { lba });
            }

            let name = nt_str(&buf[..HEADER_NAME_LEN]);
            let prefix =
                nt_str(&buf[HEADER_PREFIX_OFFSET..HEADER_PREFIX_OFFSET + HEADER_PREFIX_LEN]);
            let path = canonicalize_path(prefix, name);
            let typeflag = buf[HEADER_TYPEFLAG_OFFSET];
            let size = parse_octal(&buf[HEADER_SIZE_OFFSET..HEADER_SIZE_OFFSET + HEADER_SIZE_LEN])
                .ok_or(MountErr::BadHeader { lba })?;
            // Tar mode field carries permissions only — type bits are
            // in the typeflag. Mask to 12 bits (perms + setuid/setgid/
            // sticky) so a malformed header can't smuggle S_IF* bits in.
            let mode_perms =
                parse_octal(&buf[HEADER_MODE_OFFSET..HEADER_MODE_OFFSET + HEADER_MODE_LEN])
                    .unwrap_or(0) as u32
                    & 0o7777;
            let uid = parse_octal(&buf[HEADER_UID_OFFSET..HEADER_UID_OFFSET + HEADER_UID_LEN])
                .unwrap_or(0) as u32;
            let gid = parse_octal(&buf[HEADER_GID_OFFSET..HEADER_GID_OFFSET + HEADER_GID_LEN])
                .unwrap_or(0) as u32;
            let mtime = parse_octal(
                &buf[HEADER_MTIME_OFFSET..HEADER_MTIME_OFFSET + HEADER_MTIME_LEN],
            )
            .unwrap_or(0) as i64;
            let data_lba = lba + 1;
            let data_sectors = (size + SECTOR_SIZE as u64 - 1) / SECTOR_SIZE as u64;

            let kind_opt = match typeflag {
                TYPEFLAG_REG_NUL | TYPEFLAG_REG_0 => Some(Kind::Reg),
                TYPEFLAG_DIR => Some(Kind::Dir),
                _ => None,
            };

            if let Some(kind) = kind_opt
                && !path.is_empty()
            {
                let id = inodes.len() as Inode;
                inodes.push(Some(TarInode {
                    path: path.clone(),
                    kind,
                    data_sector: if matches!(kind, Kind::Reg) { data_lba } else { 0 },
                    size,
                    mode_perms,
                    uid,
                    gid,
                    mtime,
                }));
                by_path.insert(path, id);
            }

            lba = data_lba + data_sectors;
        }

        let registered = inodes.iter().skip(1).filter(|i| i.is_some()).count();
        info!(
            "tarfs: mounted {} entries (scanned to lba={})",
            registered, lba
        );
        for (i, ent) in inodes.iter().enumerate().skip(1) {
            if let Some(e) = ent {
                info!(
                    "tarfs: inode={} kind={:?} size={} mode={:#o} uid={} gid={} mtime={} path={}",
                    i, e.kind, e.size, e.mode_perms, e.uid, e.gid, e.mtime, e.path
                );
            }
        }

        Ok(Self { inodes, by_path })
    }

    fn entry(&self, ino: Inode) -> Result<&TarInode, FsErr> {
        self.inodes
            .get(ino as usize)
            .and_then(|e| e.as_ref())
            .ok_or(FsErr::BadInode)
    }
}

impl Filesystem for Tarfs {
    fn open(&self, path: &str) -> Result<Inode, FsErr> {
        self.by_path.get(path).copied().ok_or(FsErr::NotFound)
    }

    unsafe fn read_async(
        &self,
        ino: Inode,
        off: u64,
        len: u32,
        dst_pa: u64,
        handle: CompletionHandle,
    ) -> Result<(), FsErr> {
        let entry = self.entry(ino)?;
        if !matches!(entry.kind, Kind::Reg) {
            return Err(FsErr::NotRegular);
        }
        if len as usize != SECTOR_SIZE {
            return Err(FsErr::BadRange);
        }
        if off & (SECTOR_SIZE as u64 - 1) != 0 {
            return Err(FsErr::BadRange);
        }
        if off >= entry.size {
            return Err(FsErr::BadRange);
        }
        let lba = entry.data_sector + off / SECTOR_SIZE as u64;
        // Bytes considered valid in this read: the lesser of the
        // sector size and what's left in the file. Pinned at submit
        // time and stashed in the virtio-blk slot table so the IRQ
        // signals exactly this many bytes on success without needing
        // to know about FS state.
        let remaining = entry.size.saturating_sub(off);
        let valid = core::cmp::min(SECTOR_SIZE as u64, remaining) as isize;
        unsafe {
            virtio_blk_dev::submit_blk_read(lba, dst_pa, handle, valid).map_err(|e| {
                warn!("tarfs: submit_blk_read failed: {:?}", e);
                FsErr::IoError
            })?;
        }
        Ok(())
    }

    fn stat(&self, ino: Inode) -> Result<Stat, FsErr> {
        let entry = self.entry(ino)?;
        let kind_bits = match entry.kind {
            Kind::Reg => S_IFREG,
            Kind::Dir => S_IFDIR,
        };
        let st_size = match entry.kind {
            Kind::Reg => entry.size as i64,
            Kind::Dir => 0,
        };
        let st_blocks = match entry.kind {
            Kind::Reg => entry.size.div_ceil(STAT_BLOCK_UNIT) as i64,
            Kind::Dir => 0,
        };
        Ok(Stat {
            // Single-mount today; pin to 1 so consumers can match on
            // it without inferring "is the FS even up?".
            st_dev: 1,
            st_ino: ino as u64,
            st_mode: kind_bits | entry.mode_perms,
            st_nlink: 1,
            st_uid: entry.uid,
            st_gid: entry.gid,
            st_rdev: 0,
            __pad1: 0,
            st_size,
            st_blksize: SECTOR_SIZE as i32,
            __pad2: 0,
            st_blocks,
            st_atime: entry.mtime,
            st_atime_nsec: 0,
            st_mtime: entry.mtime,
            st_mtime_nsec: 0,
            st_ctime: entry.mtime,
            st_ctime_nsec: 0,
            __unused4: 0,
            __unused5: 0,
        })
    }

    fn size(&self, ino: Inode) -> Result<u64, FsErr> {
        Ok(self.entry(ino)?.size)
    }
}

/// NUL-terminated bytes off `slice` as a `&str`. Returns `""` if
/// `slice` starts with NUL or contains non-UTF-8.
fn nt_str(slice: &[u8]) -> &str {
    let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    core::str::from_utf8(&slice[..end]).unwrap_or("")
}

/// Parse an octal field. ustar fields are typically right-padded with
/// NULs or spaces; leading whitespace is also tolerated. Empty-or-pad
/// returns `Some(0)`. Non-octal digits return `None`.
fn parse_octal(bytes: &[u8]) -> Option<u64> {
    let mut acc: u64 = 0;
    let mut saw_digit = false;
    for &b in bytes {
        match b {
            b' ' | 0 => {
                if saw_digit {
                    return Some(acc);
                }
                // leading space/NUL — keep skipping
            }
            d @ b'0'..=b'7' => {
                acc = acc.checked_mul(8)?.checked_add((d - b'0') as u64)?;
                saw_digit = true;
            }
            _ => return None,
        }
    }
    Some(acc)
}

/// Build the canonical absolute path key from a tar header's
/// `prefix` + `name`.
///
/// - `tar -C rootfs .` yields `./README`, `./bin/hello.txt`, etc. The
///   leading `./` is stripped.
/// - The `prefix` field (used only for paths > 100 chars) joins to
///   `name` with a `/`.
/// - Trailing `/` on directory entries is stripped so lookup keys are
///   uniform: `/bin` rather than `/bin/`.
/// - The archive root (the bare `./` entry) collapses to `""` and
///   the caller skips it — there's no inode for the FS root in v1.
fn canonicalize_path(prefix: &str, name: &str) -> String {
    let mut joined = String::new();
    if !prefix.is_empty() {
        joined.push_str(prefix);
        if !prefix.ends_with('/') {
            joined.push('/');
        }
    }
    joined.push_str(name);

    let mut s = joined.as_str();
    if let Some(stripped) = s.strip_prefix("./") {
        s = stripped;
    } else if s == "." {
        return String::new();
    }
    let s = s.trim_end_matches('/');
    if s.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(s.len() + 1);
    out.push('/');
    out.push_str(s);
    out
}
