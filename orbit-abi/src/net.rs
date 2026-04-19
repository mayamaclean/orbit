//! NetChannel ABI.
//!
//! The kernel allocates the NetChannel region, maps it into the caller's
//! address space, and returns `(fd, vaddr)`. Both sides use the constants
//! in this module to locate substructures inside the region — neither side
//! trusts pointer fields inside shared memory.
//!
//! Syscall signature:
//!
//! ```text
//! a0 = REGISTER_NETCH
//! a1 = sock_type  (SockType)
//! -> a0 = fd on success, -errno on failure
//!    a1 = vaddr of mapped region (on success)
//! ```

/// Total size of a NetChannel region. One control page plus two ring pages.
pub const NC_SIZE:        usize = 3 * 4096;

/// Offset of the `NetChannel` header (placed at region base).
pub const NC_HEADER_OFF:  usize = 0;
pub const NC_DESIRED_OFF: usize = 128;
pub const NC_CURRENT_OFF: usize = 256;
pub const NC_TX_OFF:      usize = 4096;
pub const NC_RX_OFF:      usize = 8192;

/// Per-ring usable payload capacity. Derived so each ring fits exactly in one
/// page alongside its `NetChannelQueue` header.
pub const NC_RING_BYTES:  usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SockType {
    Tcp = 0,
    Udp = 1,
}

impl SockType {
    pub const fn from_usize(v: usize) -> Option<Self> {
        Some(match v {
            0 => Self::Tcp,
            1 => Self::Udp,
            _ => return None,
        })
    }
}
