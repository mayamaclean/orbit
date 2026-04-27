//! Syscall numbers.
//!
//! Numbers are load-bearing: the kernel dispatch table in `s_trap` matches on
//! them directly. Do not renumber an existing entry; add new ones at the end.

pub const EXIT:            usize = 0;
pub const SERIAL_PRINT:    usize = 1;
pub const SLEEP_MS:        usize = 2;
pub const CONSOLE_WRITE:   usize = 3;
pub const READ_STDIN:      usize = 4;
pub const SET_AFFINITY:    usize = 5;
pub const GET_AFFINITY:    usize = 6;
pub const GET_HART_ID:     usize = 7;

pub const MMAP:            usize = 4096;
pub const CREATE_NETCH:    usize = 4097;
pub const CLOSE_HANDLE:    usize = 4098;
pub const CREATE_PROCESS:  usize = 4099;
pub const NC_YIELD:        usize = 4100;

// 5000+ — multi-thread / SMP control plane. Numbered out of the 4096
// block so the categorical split is obvious in dispatch tables and so
// future single-process-spanning syscalls (futex wake/wait, etc.)
// share the same range.
pub const CREATE_THREAD:   usize = 5000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Sysno {
    Exit           = EXIT,
    SerialPrint    = SERIAL_PRINT,
    SleepMs        = SLEEP_MS,
    ConsoleWrite   = CONSOLE_WRITE,
    ReadStdin      = READ_STDIN,
    SetAffinity    = SET_AFFINITY,
    GetAffinity    = GET_AFFINITY,
    GetHartId      = GET_HART_ID,
    Mmap           = MMAP,
    CreateNetch    = CREATE_NETCH,
    CloseHandle    = CLOSE_HANDLE,
    CreateProcess  = CREATE_PROCESS,
    NcYield        = NC_YIELD,
    CreateThread   = CREATE_THREAD,
}

impl Sysno {
    pub const fn from_usize(n: usize) -> Option<Self> {
        Some(match n {
            EXIT           => Self::Exit,
            SERIAL_PRINT   => Self::SerialPrint,
            SLEEP_MS       => Self::SleepMs,
            CONSOLE_WRITE  => Self::ConsoleWrite,
            READ_STDIN     => Self::ReadStdin,
            SET_AFFINITY   => Self::SetAffinity,
            GET_AFFINITY   => Self::GetAffinity,
            GET_HART_ID    => Self::GetHartId,
            MMAP           => Self::Mmap,
            CREATE_NETCH   => Self::CreateNetch,
            CLOSE_HANDLE   => Self::CloseHandle,
            CREATE_PROCESS => Self::CreateProcess,
            NC_YIELD       => Self::NcYield,
            CREATE_THREAD  => Self::CreateThread,
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
        assert_eq!(Sysno::from_usize(CONSOLE_WRITE), Some(Sysno::ConsoleWrite));
        assert_eq!(Sysno::from_usize(READ_STDIN),    Some(Sysno::ReadStdin));
        assert_eq!(Sysno::from_usize(SET_AFFINITY),  Some(Sysno::SetAffinity));
        assert_eq!(Sysno::from_usize(GET_AFFINITY),  Some(Sysno::GetAffinity));
        assert_eq!(Sysno::from_usize(GET_HART_ID),   Some(Sysno::GetHartId));
        assert_eq!(Sysno::from_usize(MMAP),         Some(Sysno::Mmap));
        assert_eq!(Sysno::from_usize(CREATE_NETCH), Some(Sysno::CreateNetch));
        assert_eq!(Sysno::from_usize(CLOSE_HANDLE), Some(Sysno::CloseHandle));
        assert_eq!(Sysno::from_usize(CREATE_PROCESS), Some(Sysno::CreateProcess));
        assert_eq!(Sysno::from_usize(NC_YIELD),       Some(Sysno::NcYield));
        assert_eq!(Sysno::from_usize(CREATE_THREAD),  Some(Sysno::CreateThread));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(Sysno::from_usize(8), None);
        assert_eq!(Sysno::from_usize(4095), None);
        assert_eq!(Sysno::from_usize(4101), None);
        assert_eq!(Sysno::from_usize(4999), None);
        assert_eq!(Sysno::from_usize(5001), None);
        assert_eq!(Sysno::from_usize(usize::MAX), None);
    }

    #[test]
    fn variant_discriminant_matches_constant() {
        assert_eq!(Sysno::Exit          as usize, EXIT);
        assert_eq!(Sysno::SerialPrint   as usize, SERIAL_PRINT);
        assert_eq!(Sysno::SleepMs       as usize, SLEEP_MS);
        assert_eq!(Sysno::ConsoleWrite  as usize, CONSOLE_WRITE);
        assert_eq!(Sysno::ReadStdin     as usize, READ_STDIN);
        assert_eq!(Sysno::SetAffinity   as usize, SET_AFFINITY);
        assert_eq!(Sysno::GetAffinity   as usize, GET_AFFINITY);
        assert_eq!(Sysno::GetHartId     as usize, GET_HART_ID);
        assert_eq!(Sysno::Mmap          as usize, MMAP);
        assert_eq!(Sysno::CreateNetch   as usize, CREATE_NETCH);
        assert_eq!(Sysno::CloseHandle   as usize, CLOSE_HANDLE);
        assert_eq!(Sysno::CreateProcess as usize, CREATE_PROCESS);
        assert_eq!(Sysno::NcYield       as usize, NC_YIELD);
        assert_eq!(Sysno::CreateThread  as usize, CREATE_THREAD);
    }

    #[test]
    fn numbers_are_load_bearing_do_not_renumber() {
        // Pin the on-wire numbers — kmain's dispatch table matches on
        // these directly. Renumbering breaks the user/kernel ABI.
        assert_eq!(EXIT, 0);
        assert_eq!(SERIAL_PRINT, 1);
        assert_eq!(SLEEP_MS, 2);
        assert_eq!(CONSOLE_WRITE, 3);
        assert_eq!(READ_STDIN, 4);
        assert_eq!(SET_AFFINITY, 5);
        assert_eq!(GET_AFFINITY, 6);
        assert_eq!(GET_HART_ID, 7);
        assert_eq!(MMAP, 4096);
        assert_eq!(CREATE_NETCH, 4097);
        assert_eq!(CLOSE_HANDLE, 4098);
        assert_eq!(CREATE_PROCESS, 4099);
        assert_eq!(NC_YIELD, 4100);
    }
}
