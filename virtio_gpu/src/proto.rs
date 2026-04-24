//! virtio-gpu wire protocol structs (virtio spec §5.7).

pub const VIRTIO_GPU_DEVICE_ID: u32 = 16;

// 2D command IDs.
pub const CMD_GET_DISPLAY_INFO: u32 = 0x0100;
pub const CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
pub const CMD_RESOURCE_UNREF: u32 = 0x0102;
pub const CMD_SET_SCANOUT: u32 = 0x0103;
pub const CMD_RESOURCE_FLUSH: u32 = 0x0104;
pub const CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
pub const CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
pub const CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;

// Response IDs.
pub const RESP_OK_NODATA: u32 = 0x1100;
pub const RESP_OK_DISPLAY_INFO: u32 = 0x1101;
pub const RESP_ERR_UNSPEC: u32 = 0x1200;

// Pixel formats. We use B8G8R8A8 (little-endian u32 bytes order
// reversed = 0xAARRGGBB on memory).
pub const FORMAT_B8G8R8A8_UNORM: u32 = 1;
pub const FORMAT_R8G8B8A8_UNORM: u32 = 67;

pub const MAX_SCANOUTS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CtrlHdr {
    pub ty: u32,
    pub flags: u32,
    pub fence_id: u64,
    pub ctx_id: u32,
    pub ring_idx: u8,
    pub _padding: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DisplayOne {
    pub r: Rect,
    pub enabled: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RespDisplayInfo {
    pub hdr: CtrlHdr,
    pub pmodes: [DisplayOne; MAX_SCANOUTS],
}

impl Default for RespDisplayInfo {
    fn default() -> Self {
        Self {
            hdr: CtrlHdr::default(),
            pmodes: [DisplayOne::default(); MAX_SCANOUTS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ResourceCreate2d {
    pub hdr: CtrlHdr,
    pub resource_id: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MemEntry {
    pub addr: u64,
    pub length: u32,
    pub _padding: u32,
}

/// Single-entry attach_backing. virtio-gpu supports multi-entry but
/// our framebuffer is one physically contiguous region so one entry
/// suffices.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ResourceAttachBacking {
    pub hdr: CtrlHdr,
    pub resource_id: u32,
    pub nr_entries: u32,
    pub entry: MemEntry,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SetScanout {
    pub hdr: CtrlHdr,
    pub r: Rect,
    pub scanout_id: u32,
    pub resource_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct TransferToHost2d {
    pub hdr: CtrlHdr,
    pub r: Rect,
    pub offset: u64,
    pub resource_id: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ResourceFlush {
    pub hdr: CtrlHdr,
    pub r: Rect,
    pub resource_id: u32,
    pub _padding: u32,
}
