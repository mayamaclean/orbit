//! Errno values.
//!
//! Numbers chosen to match Linux on RISC-V so `io::Error::from_raw_os_error`
//! in a future `std::sys::orbit` can reuse the unix translation table.
//!
//! Convention at the syscall boundary: success returns a non-negative `isize`,
//! failure returns `-(errno as isize)`. Callers use [`Errno::from_ret`] to
//! split the return value.

pub const EPERM: i32 = 1;
pub const ENOENT: i32 = 2;
pub const ESRCH: i32 = 3;
pub const EIO: i32 = 5;
pub const ENOEXEC: i32 = 8;
pub const EBADF: i32 = 9;
pub const ECHILD: i32 = 10;
pub const EAGAIN: i32 = 11;
pub const ENOMEM: i32 = 12;
pub const EACCES: i32 = 13;
pub const EFAULT: i32 = 14;
pub const EBUSY: i32 = 16;
pub const EEXIST: i32 = 17;
pub const ENODEV: i32 = 19;
pub const ENOTDIR: i32 = 20;
pub const EISDIR: i32 = 21;
pub const EINVAL: i32 = 22;
pub const ENFILE: i32 = 23;
pub const EMFILE: i32 = 24;
pub const ERANGE: i32 = 34;
pub const ENAMETOOLONG: i32 = 36;
pub const ENOSYS: i32 = 38;
pub const ETIMEDOUT: i32 = 110;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Errno(pub i32);

impl Errno {
    pub const fn new(code: i32) -> Self {
        Self(code)
    }

    /// Decode a syscall return value. Non-negative = Ok, negative = Err(errno).
    pub const fn from_ret(r: isize) -> Result<usize, Self> {
        if r < 0 {
            Err(Self((-r) as i32))
        }
        else {
            Ok(r as usize)
        }
    }

    /// Encode an errno back into a syscall return value.
    pub const fn to_ret(self) -> isize {
        -(self.0 as isize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_negative_decodes_as_ok() {
        assert_eq!(Errno::from_ret(0), Ok(0));
        assert_eq!(Errno::from_ret(1), Ok(1));
        assert_eq!(Errno::from_ret(42), Ok(42));
        assert_eq!(Errno::from_ret(isize::MAX), Ok(isize::MAX as usize));
    }

    #[test]
    fn negative_decodes_as_err() {
        assert_eq!(Errno::from_ret(-EPERM as isize), Err(Errno(EPERM)));
        assert_eq!(Errno::from_ret(-ENOENT as isize), Err(Errno(ENOENT)));
        assert_eq!(Errno::from_ret(-EINVAL as isize), Err(Errno(EINVAL)));
    }

    #[test]
    fn to_ret_round_trips_through_from_ret() {
        for &code in &[
            EPERM, ENOENT, EIO, EAGAIN, ENOMEM, EFAULT, EBUSY, EEXIST, ENODEV, EINVAL, ENFILE,
            ENOSYS,
        ] {
            let ret = Errno::new(code).to_ret();
            assert!(ret < 0, "errno {code} encoded as non-negative {ret}");
            assert_eq!(Errno::from_ret(ret), Err(Errno(code)));
        }
    }

    #[test]
    fn well_known_codes_match_linux_riscv() {
        // Pin the numeric values — consumers like a future `std::sys::orbit`
        // rely on these being identical to Linux's errno table.
        assert_eq!(EPERM, 1);
        assert_eq!(ENOENT, 2);
        assert_eq!(EAGAIN, 11);
        assert_eq!(EINVAL, 22);
        assert_eq!(ENOSYS, 38);
    }
}
