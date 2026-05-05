//! virtio-blk wire protocol (virtio spec §5.2).

pub const VIRTIO_BLK_DEVICE_ID: u32 = 2;

/// Standard logical block size for virtio-blk. Spec calls this 512 B
/// regardless of physical sector size — the device reports its own
/// geometry separately, but the queue still trades in 512-B units.
pub const SECTOR_SIZE: usize = 512;

/// Maximum bytes per request the driver will submit. The wire protocol
/// allows arbitrary multi-sector reads in one chain (one header, one
/// data buffer of `N * SECTOR_SIZE` bytes, one status); we cap at one
/// page so a single contiguous kernel scratch frame backs every
/// request without splitting into multiple data descriptors.
///
/// Eventually this should be `min(PAGE, BlkConfig.size_max)` once we
/// plumb the device-advertised limits through the handshake; QEMU is
/// generous so the static cap is safe today.
pub const MAX_REQ_BYTES: u32 = 4096;

// Request types (`BlkReqHeader::ty`).
pub const VIRTIO_BLK_T_IN: u32 = 0;
pub const VIRTIO_BLK_T_OUT: u32 = 1;
pub const VIRTIO_BLK_T_FLUSH: u32 = 4;

// Status bytes (last descriptor in chain, device-writes).
pub const VIRTIO_BLK_S_OK: u8 = 0;
pub const VIRTIO_BLK_S_IOERR: u8 = 1;
pub const VIRTIO_BLK_S_UNSUPP: u8 = 2;

/// Request header — first descriptor in every chain. Spec §5.2.6.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct BlkReqHeader {
    pub ty: u32,
    pub reserved: u32,
    pub sector: u64,
}

/// Device-specific config (read off `Mmio::config_base`). Spec §5.2.4
/// has more fields; we only consume `capacity` today.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct BlkConfig {
    /// Disk size in 512-byte sectors.
    pub capacity: u64,
}
