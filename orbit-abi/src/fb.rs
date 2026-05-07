//! Framebuffer / drawing-surface ABI.
//!
//! User processes can allocate pixel surfaces via [`FB_SURFACE_CREATE`],
//! draw into them through the shared mapping the syscall hands back, and
//! present a damaged region to the screen via [`FB_PRESENT`]. The kernel
//! compositor (k_gpu) blits the active source's surface into the
//! virtio-gpu scanout on its next drain pass.
//!
//! [`FB_SURFACE_CREATE`]: crate::syscall::FB_SURFACE_CREATE
//! [`FB_PRESENT`]: crate::syscall::FB_PRESENT

/// Pixel format. Only [`Self::Bgra8888`] is wired in v1 — matches the
/// virtio-gpu scanout format used by the in-kernel framebuffer driver.
/// Encoded as a `u32` on the wire so future formats can be added without
/// renumbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FbFormat {
    /// 32-bit pixels packed `0xAA_RR_GG_BB` (so the in-memory byte order
    /// on little-endian RISC-V is `BB GG RR AA`). Same packing the
    /// kernel's `fb::rgb()` helper uses.
    Bgra8888 = 1,
}

impl FbFormat {
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Bgra8888 => 4,
        }
    }

    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Bgra8888),
            _ => None,
        }
    }
}

/// Reply payload for [`crate::syscall::FB_QUERY`]. Snapshotted by the
/// kernel at boot from the virtio-gpu `GET_DISPLAY_INFO` reply; stable
/// for the lifetime of the system in v1 (no display-mode changes yet).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct FbInfo {
    pub width: u32,
    pub height: u32,
    /// Stored as `u32` so the struct is `repr(C)` clean. Decode via
    /// [`FbFormat::from_u32`]; an unknown value means a kernel newer
    /// than this build introduced a format the caller doesn't know.
    pub format: u32,
    /// Reserved for future flags (HDR, DPI scale, etc.). Kernel writes
    /// `0` today; readers should ignore unknown bits.
    pub flags: u32,
}

/// Surface handle returned by [`crate::syscall::FB_SURFACE_CREATE`].
/// Process-local — the kernel maps `(pid, handle)` to its `SurfaceEntry`,
/// so a handle is meaningless outside the process that created it.
/// Reserved value `0` means "no surface" (the default for an
/// uninitialised handle slot in user code); kernel never returns 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FbHandle(pub u32);

impl FbHandle {
    pub const NONE: Self = Self(0);

    pub const fn raw(self) -> u32 {
        self.0
    }
}
