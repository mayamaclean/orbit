//! Errno values.
//!
//! Numbers chosen to match Linux on RISC-V so `io::Error::from_raw_os_error`
//! in a future `std::sys::orbit` can reuse the unix translation table.
//!
//! Convention at the syscall boundary: success returns a non-negative `isize`,
//! failure returns `-(errno as isize)`. Callers use [`Errno::from_ret`] to
//! split the return value.

pub const EPERM:  i32 =  1;
pub const ENOENT: i32 =  2;
pub const ESRCH:  i32 =  3;
pub const EIO:    i32 =  5;
pub const EBADF:  i32 =  9;
pub const EAGAIN: i32 = 11;
pub const ENOMEM: i32 = 12;
pub const EACCES: i32 = 13;
pub const EFAULT: i32 = 14;
pub const EBUSY:  i32 = 16;
pub const EEXIST: i32 = 17;
pub const ENODEV: i32 = 19;
pub const EINVAL: i32 = 22;
pub const ENFILE: i32 = 23;
pub const ENOSYS: i32 = 38;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Errno(pub i32);

impl Errno {
    pub const fn new(code: i32) -> Self { Self(code) }

    /// Decode a syscall return value. Non-negative = Ok, negative = Err(errno).
    pub const fn from_ret(r: isize) -> Result<usize, Self> {
        if r < 0 { Err(Self((-r) as i32)) } else { Ok(r as usize) }
    }

    /// Encode an errno back into a syscall return value.
    pub const fn to_ret(self) -> isize { -(self.0 as isize) }
}
