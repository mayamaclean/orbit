//! NetChannel ABI.
//!
//! The kernel allocates the NetChannel region, initializes its control
//! fields and queue headers, maps it into the caller's address space, and
//! registers it with the net thread — all in a single syscall. Neither
//! side stores absolute pointers into shared memory; offsets are anchored
//! off the region base, which [`NetChannel`](net_channel::NetChannel)
//! accessors compute at runtime.
//!
//! Syscall signature (number
//! [`CREATE_NETCH`](crate::syscall::CREATE_NETCH)):
//!
//! ```text
//! a0 = CREATE_NETCH  (4097)
//! a1 = user_vaddr    (mapping hint, must be page-aligned)
//! a2 = region_size   (bytes; kernel clamps to [NC_MIN_REGION_SIZE,
//!                     NC_MAX_REGION_SIZE] and rounds up to a page)
//! a3 = sock_type     (SockType)
//! a4 = bind_spec     (packed BindSpec; required)
//! -> a0 = user_va, a1 = fd on success; a0 = -errno on failure
//! ```

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
