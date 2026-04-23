//! Syscall numbers.
//!
//! Numbers are load-bearing: the kernel dispatch table in `s_trap` matches on
//! them directly. Do not renumber an existing entry; add new ones at the end.

pub const EXIT:            usize = 0;
pub const SERIAL_PRINT:    usize = 1;
pub const SLEEP_MS:        usize = 2;

pub const MMAP:            usize = 4096;
pub const CREATE_NETCH:    usize = 4097;
pub const CLOSE_HANDLE:    usize = 4098;
pub const CREATE_PROCESS:  usize = 4099;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Sysno {
    Exit           = EXIT,
    SerialPrint    = SERIAL_PRINT,
    SleepMs        = SLEEP_MS,
    Mmap           = MMAP,
    CreateNetch    = CREATE_NETCH,
    CloseHandle    = CLOSE_HANDLE,
    CreateProcess  = CREATE_PROCESS,
}

impl Sysno {
    pub const fn from_usize(n: usize) -> Option<Self> {
        Some(match n {
            EXIT           => Self::Exit,
            SERIAL_PRINT   => Self::SerialPrint,
            SLEEP_MS       => Self::SleepMs,
            MMAP           => Self::Mmap,
            CREATE_NETCH   => Self::CreateNetch,
            CLOSE_HANDLE   => Self::CloseHandle,
            CREATE_PROCESS => Self::CreateProcess,
            _              => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_defined_number_decodes() {
        assert_eq!(Sysno::from_usize(EXIT),         Some(Sysno::Exit));
        assert_eq!(Sysno::from_usize(SERIAL_PRINT), Some(Sysno::SerialPrint));
        assert_eq!(Sysno::from_usize(SLEEP_MS),     Some(Sysno::SleepMs));
        assert_eq!(Sysno::from_usize(MMAP),         Some(Sysno::Mmap));
        assert_eq!(Sysno::from_usize(CREATE_NETCH), Some(Sysno::CreateNetch));
        assert_eq!(Sysno::from_usize(CLOSE_HANDLE), Some(Sysno::CloseHandle));
        assert_eq!(Sysno::from_usize(CREATE_PROCESS), Some(Sysno::CreateProcess));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(Sysno::from_usize(3), None);
        assert_eq!(Sysno::from_usize(4095), None);
        assert_eq!(Sysno::from_usize(4100), None);
        assert_eq!(Sysno::from_usize(usize::MAX), None);
    }

    #[test]
    fn variant_discriminant_matches_constant() {
        assert_eq!(Sysno::Exit          as usize, EXIT);
        assert_eq!(Sysno::SerialPrint   as usize, SERIAL_PRINT);
        assert_eq!(Sysno::SleepMs       as usize, SLEEP_MS);
        assert_eq!(Sysno::Mmap          as usize, MMAP);
        assert_eq!(Sysno::CreateNetch   as usize, CREATE_NETCH);
        assert_eq!(Sysno::CloseHandle   as usize, CLOSE_HANDLE);
        assert_eq!(Sysno::CreateProcess as usize, CREATE_PROCESS);
    }

    #[test]
    fn numbers_are_load_bearing_do_not_renumber() {
        // Pin the on-wire numbers — kmain's dispatch table matches on
        // these directly. Renumbering breaks the user/kernel ABI.
        assert_eq!(EXIT, 0);
        assert_eq!(SERIAL_PRINT, 1);
        assert_eq!(SLEEP_MS, 2);
        assert_eq!(MMAP, 4096);
        assert_eq!(CREATE_NETCH, 4097);
        assert_eq!(CLOSE_HANDLE, 4098);
        assert_eq!(CREATE_PROCESS, 4099);
    }
}
