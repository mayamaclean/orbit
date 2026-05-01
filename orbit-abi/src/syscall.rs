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
/// `get_micros() -> u64` — absolute monotonic microseconds since
/// system boot. Cheap unprivileged tick read on the kernel side
/// (RISC-V `time` CSR / 10 since QEMU virt clocks at 10 MHz),
/// returned in `a0`. Opaque base: only differences are meaningful.
pub const GET_MICROS:      usize = 8;

/// `pledge(req: *const PermsRequest) -> 0 | -errno` — narrow this
/// process's `perms` and `allowed_perms` masks. PR2 ships this as
/// data-only: the kernel mutates `Process.permissions` for keeps,
/// so a subsequent `derive_child_perms` clamps real children
/// against the narrowed cap, but the dispatch gate stays in
/// log-only shadow mode. PR3 flips the gate to EPERM.
///
/// `req` is a `*const orbit_abi::perms::PermsRequest` in user
/// memory; the kernel reads both `ClassMask` fields via the
/// standard boundary-deserializer path. Errors:
/// - `EFAULT` — `req` doesn't translate under the caller's satp.
/// - `EPERM` (PR3+) — caller has pledged `class::PLEDGE` away.
pub const PLEDGE:          usize = 9;

pub const MMAP:            usize = 4096;
pub const CREATE_NETCH:    usize = 4097;
pub const CLOSE_HANDLE:    usize = 4098;
pub const CREATE_PROCESS:  usize = 4099;
pub const NC_YIELD:        usize = 4100;
pub const QUERY_STATS:         usize = 4101;
pub const QUERY_SYSCALL_STATS: usize = 4102;
pub const CREATE_PROCESS_EX:   usize = 4103;
pub const ARGV_ENVP:           usize = 4104;
/// `create_process_v2(args: *const CreateProcessV2Args) -> pid | -errno`
/// — role-aware spawn with explicit permission narrowing. Replaces
/// `CREATE_PROCESS` and `CREATE_PROCESS_EX` for callers that need
/// a `target_role` and `PermsRequest` (the older numbers stay live
/// for ABI compat; they spawn into `BOOTSTRAP` with `Permissions::ALL`).
/// Args struct in user memory because the call carries enough
/// fields to overflow the `a1..a7` register window comfortably.
///
/// PR2 ships in shadow mode: `check_transition` runs, on `Err` the
/// kernel falls through to `install_child_shadow` (logged) instead
/// of EPERMing. PR3 flips the `Err` arm to EPERM.
pub const CREATE_PROCESS_V2:   usize = 4105;
/// `query_shadow_log(buf: *mut u8, buf_len: usize) -> bytes | -errno`
/// — copy the kernel-wide shadow event ring into a user buffer in
/// chronological order. Reply layout is a packed sequence of
/// `ShadowEvent`s; `bytes_written / size_of::<ShadowEvent>()`
/// gives the count. Buffer too small: returns the prefix that fits
/// + `bytes_written` reflecting the partial copy.
///
/// Unlike `query_stats`, the ring isn't per-process — it covers
/// every gate event system-wide. Sized at
/// `SHADOW_RING_CAPACITY * size_of::<ShadowEvent>()` for a complete
/// snapshot (≈2.4 KiB at the PR2 cap of 50 events).
///
/// **Cross-process disclosure.** The reply contains pids, tids,
/// syscall numbers, and (for `RoleDeny`) source/target roles for
/// *every* would-be denial system-wide, not just the caller's. A
/// process holding `class::STATS` can correlate which other
/// processes are calling which syscalls and where they're being
/// gated. Acceptable today (orbit is single-tenant); once
/// multi-tenant workloads land, consider gating this behind a
/// dedicated `class::SHADOW_LOG` so a low-trust observer role can
/// hold STATS without seeing other processes' denials.
///
/// **Lifecycle.** Introduced in PR2 alongside the shadow ring.
/// Whether it survives PR3 is open: enforcement-mode auditing may
/// keep the ring (relabelling "would-be denials" to "actual
/// denials"), or the ring may be deleted alongside
/// `install_child_shadow`. Decide when PR3 lands; until then the
/// syscall number stays reserved.
pub const QUERY_SHADOW_LOG:    usize = 4106;

// 5000+ — multi-thread / SMP control plane. Numbered out of the 4096
// block so the categorical split is obvious in dispatch tables and so
// future single-process-spanning syscalls (futex wake/wait, etc.)
// share the same range.
pub const CREATE_THREAD:   usize = 5000;
pub const GETPID:          usize = 5001;
pub const GETTID:          usize = 5002;
pub const WAIT_PID:        usize = 5003;
pub const FUTEX_WAIT:      usize = 5004;
pub const FUTEX_WAKE:      usize = 5005;

// 6000+ — filesystem. v1 is read-only tarfs; close re-uses
// `CLOSE_HANDLE = 4098` (handle table is shared across NetCh / file
// fds) so there's no FS_CLOSE here.
pub const FS_OPEN:         usize = 6000;
pub const FS_READ:         usize = 6001;
pub const FS_STAT:         usize = 6002;
pub const FS_READDIR:      usize = 6003;

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
    GetMicros      = GET_MICROS,
    Pledge         = PLEDGE,
    Mmap           = MMAP,
    CreateNetch    = CREATE_NETCH,
    CloseHandle    = CLOSE_HANDLE,
    CreateProcess  = CREATE_PROCESS,
    NcYield        = NC_YIELD,
    QueryStats        = QUERY_STATS,
    QuerySyscallStats = QUERY_SYSCALL_STATS,
    CreateProcessEx = CREATE_PROCESS_EX,
    ArgvEnvp        = ARGV_ENVP,
    CreateProcessV2 = CREATE_PROCESS_V2,
    QueryShadowLog  = QUERY_SHADOW_LOG,
    CreateThread   = CREATE_THREAD,
    GetPid         = GETPID,
    GetTid         = GETTID,
    WaitPid        = WAIT_PID,
    FutexWait      = FUTEX_WAIT,
    FutexWake      = FUTEX_WAKE,
    FsOpen         = FS_OPEN,
    FsRead         = FS_READ,
    FsStat         = FS_STAT,
    FsReaddir      = FS_READDIR,
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
            GET_MICROS     => Self::GetMicros,
            PLEDGE         => Self::Pledge,
            MMAP           => Self::Mmap,
            CREATE_NETCH   => Self::CreateNetch,
            CLOSE_HANDLE   => Self::CloseHandle,
            CREATE_PROCESS => Self::CreateProcess,
            NC_YIELD       => Self::NcYield,
            QUERY_STATS         => Self::QueryStats,
            QUERY_SYSCALL_STATS => Self::QuerySyscallStats,
            CREATE_PROCESS_EX => Self::CreateProcessEx,
            ARGV_ENVP      => Self::ArgvEnvp,
            CREATE_PROCESS_V2 => Self::CreateProcessV2,
            QUERY_SHADOW_LOG  => Self::QueryShadowLog,
            CREATE_THREAD  => Self::CreateThread,
            GETPID         => Self::GetPid,
            GETTID         => Self::GetTid,
            WAIT_PID       => Self::WaitPid,
            FUTEX_WAIT     => Self::FutexWait,
            FUTEX_WAKE     => Self::FutexWake,
            FS_OPEN        => Self::FsOpen,
            FS_READ        => Self::FsRead,
            FS_STAT        => Self::FsStat,
            FS_READDIR     => Self::FsReaddir,
            _              => return None,
        })
    }

    /// Stable, dense ordinal for stats tables. Tied to the order of
    /// `match` arms below — append-only, never reorder. The raw
    /// syscall-number space is sparse (0-7, 4096-4102, 5000) so we
    /// can't use it as an array index.
    pub const fn ordinal(self) -> usize {
        match self {
            Self::Exit              => 0,
            Self::SerialPrint       => 1,
            Self::SleepMs           => 2,
            Self::ConsoleWrite      => 3,
            Self::ReadStdin         => 4,
            Self::SetAffinity       => 5,
            Self::GetAffinity       => 6,
            Self::GetHartId         => 7,
            Self::Mmap              => 8,
            Self::CreateNetch       => 9,
            Self::CloseHandle       => 10,
            Self::CreateProcess     => 11,
            Self::NcYield           => 12,
            Self::QueryStats        => 13,
            Self::QuerySyscallStats => 14,
            Self::CreateThread      => 15,
            Self::GetMicros         => 16,
            Self::FsOpen            => 17,
            Self::FsRead            => 18,
            Self::FsStat            => 19,
            Self::GetPid            => 20,
            Self::GetTid            => 21,
            Self::WaitPid           => 22,
            Self::CreateProcessEx   => 23,
            Self::ArgvEnvp          => 24,
            Self::FutexWait         => 25,
            Self::FutexWake         => 26,
            Self::FsReaddir         => 27,
            Self::Pledge            => 28,
            Self::CreateProcessV2   => 29,
            Self::QueryShadowLog    => 30,
        }
    }

    /// Number of distinct ordinals returned by [`Self::ordinal`]. Pinned
    /// so the per-syscall stats table size is part of the ABI; bump
    /// when adding a `Sysno` variant. Older userland with a smaller
    /// COUNT reads a prefix of the kernel's table; newer userland with
    /// a larger COUNT treats the kernel's missing slots as zero.
    pub const COUNT: usize = 31;
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
        assert_eq!(Sysno::from_usize(GET_MICROS),    Some(Sysno::GetMicros));
        assert_eq!(Sysno::from_usize(MMAP),         Some(Sysno::Mmap));
        assert_eq!(Sysno::from_usize(CREATE_NETCH), Some(Sysno::CreateNetch));
        assert_eq!(Sysno::from_usize(CLOSE_HANDLE), Some(Sysno::CloseHandle));
        assert_eq!(Sysno::from_usize(CREATE_PROCESS), Some(Sysno::CreateProcess));
        assert_eq!(Sysno::from_usize(NC_YIELD),       Some(Sysno::NcYield));
        assert_eq!(Sysno::from_usize(QUERY_STATS),         Some(Sysno::QueryStats));
        assert_eq!(Sysno::from_usize(QUERY_SYSCALL_STATS), Some(Sysno::QuerySyscallStats));
        assert_eq!(Sysno::from_usize(CREATE_THREAD),  Some(Sysno::CreateThread));
        assert_eq!(Sysno::from_usize(FS_OPEN),  Some(Sysno::FsOpen));
        assert_eq!(Sysno::from_usize(FS_READ),  Some(Sysno::FsRead));
        assert_eq!(Sysno::from_usize(FS_STAT),  Some(Sysno::FsStat));
        assert_eq!(Sysno::from_usize(FS_READDIR), Some(Sysno::FsReaddir));
        assert_eq!(Sysno::from_usize(GETPID),   Some(Sysno::GetPid));
        assert_eq!(Sysno::from_usize(GETTID),   Some(Sysno::GetTid));
        assert_eq!(Sysno::from_usize(WAIT_PID), Some(Sysno::WaitPid));
        assert_eq!(Sysno::from_usize(FUTEX_WAIT), Some(Sysno::FutexWait));
        assert_eq!(Sysno::from_usize(FUTEX_WAKE), Some(Sysno::FutexWake));
        assert_eq!(Sysno::from_usize(CREATE_PROCESS_EX), Some(Sysno::CreateProcessEx));
        assert_eq!(Sysno::from_usize(ARGV_ENVP),         Some(Sysno::ArgvEnvp));
        assert_eq!(Sysno::from_usize(PLEDGE),            Some(Sysno::Pledge));
        assert_eq!(Sysno::from_usize(CREATE_PROCESS_V2), Some(Sysno::CreateProcessV2));
        assert_eq!(Sysno::from_usize(QUERY_SHADOW_LOG),  Some(Sysno::QueryShadowLog));
    }

    #[test]
    fn unknown_returns_none() {
        // 9 was reserved for PLEDGE in PR2 — used to be a hole, now decodes.
        assert_eq!(Sysno::from_usize(10), None);
        assert_eq!(Sysno::from_usize(4095), None);
        // 4105/4106 are now CREATE_PROCESS_V2 / QUERY_SHADOW_LOG.
        assert_eq!(Sysno::from_usize(4107), None);
        assert_eq!(Sysno::from_usize(4999), None);
        assert_eq!(Sysno::from_usize(5006), None);
        assert_eq!(Sysno::from_usize(5999), None);
        assert_eq!(Sysno::from_usize(6004), None);
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
        assert_eq!(Sysno::GetMicros     as usize, GET_MICROS);
        assert_eq!(Sysno::Mmap          as usize, MMAP);
        assert_eq!(Sysno::CreateNetch   as usize, CREATE_NETCH);
        assert_eq!(Sysno::CloseHandle   as usize, CLOSE_HANDLE);
        assert_eq!(Sysno::CreateProcess as usize, CREATE_PROCESS);
        assert_eq!(Sysno::NcYield           as usize, NC_YIELD);
        assert_eq!(Sysno::QueryStats        as usize, QUERY_STATS);
        assert_eq!(Sysno::QuerySyscallStats as usize, QUERY_SYSCALL_STATS);
        assert_eq!(Sysno::CreateThread      as usize, CREATE_THREAD);
        assert_eq!(Sysno::FsOpen            as usize, FS_OPEN);
        assert_eq!(Sysno::FsRead            as usize, FS_READ);
        assert_eq!(Sysno::FsStat            as usize, FS_STAT);
        assert_eq!(Sysno::FsReaddir         as usize, FS_READDIR);
        assert_eq!(Sysno::GetPid            as usize, GETPID);
        assert_eq!(Sysno::GetTid            as usize, GETTID);
        assert_eq!(Sysno::WaitPid           as usize, WAIT_PID);
        assert_eq!(Sysno::FutexWait         as usize, FUTEX_WAIT);
        assert_eq!(Sysno::FutexWake         as usize, FUTEX_WAKE);
        assert_eq!(Sysno::CreateProcessEx   as usize, CREATE_PROCESS_EX);
        assert_eq!(Sysno::ArgvEnvp          as usize, ARGV_ENVP);
        assert_eq!(Sysno::Pledge            as usize, PLEDGE);
        assert_eq!(Sysno::CreateProcessV2   as usize, CREATE_PROCESS_V2);
        assert_eq!(Sysno::QueryShadowLog    as usize, QUERY_SHADOW_LOG);
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
        assert_eq!(GET_MICROS, 8);
        assert_eq!(MMAP, 4096);
        assert_eq!(CREATE_NETCH, 4097);
        assert_eq!(CLOSE_HANDLE, 4098);
        assert_eq!(CREATE_PROCESS, 4099);
        assert_eq!(NC_YIELD, 4100);
        assert_eq!(QUERY_STATS, 4101);
        assert_eq!(QUERY_SYSCALL_STATS, 4102);
        assert_eq!(CREATE_THREAD, 5000);
        assert_eq!(FS_OPEN, 6000);
        assert_eq!(FS_READ, 6001);
        assert_eq!(FS_STAT, 6002);
        assert_eq!(FS_READDIR, 6003);
        assert_eq!(GETPID, 5001);
        assert_eq!(GETTID, 5002);
        assert_eq!(WAIT_PID, 5003);
        assert_eq!(FUTEX_WAIT, 5004);
        assert_eq!(FUTEX_WAKE, 5005);
        assert_eq!(CREATE_PROCESS_EX, 4103);
        assert_eq!(ARGV_ENVP, 4104);
        assert_eq!(PLEDGE, 9);
        assert_eq!(CREATE_PROCESS_V2, 4105);
        assert_eq!(QUERY_SHADOW_LOG, 4106);
    }

    #[test]
    fn ordinals_are_dense_and_unique() {
        // Iterate every variant via from_usize so we can't forget to
        // update the test when adding a Sysno.
        let all = [
            Sysno::Exit, Sysno::SerialPrint, Sysno::SleepMs,
            Sysno::ConsoleWrite, Sysno::ReadStdin, Sysno::SetAffinity,
            Sysno::GetAffinity, Sysno::GetHartId, Sysno::Mmap,
            Sysno::CreateNetch, Sysno::CloseHandle, Sysno::CreateProcess,
            Sysno::NcYield, Sysno::QueryStats, Sysno::QuerySyscallStats,
            Sysno::CreateThread, Sysno::GetMicros, Sysno::FsOpen,
            Sysno::FsRead, Sysno::FsStat, Sysno::GetPid, Sysno::GetTid,
            Sysno::WaitPid, Sysno::CreateProcessEx, Sysno::ArgvEnvp,
            Sysno::FutexWait, Sysno::FutexWake, Sysno::FsReaddir,
            Sysno::Pledge, Sysno::CreateProcessV2, Sysno::QueryShadowLog,
        ];
        assert_eq!(all.len(), Sysno::COUNT);
        let mut seen = [false; Sysno::COUNT];
        for s in all {
            let o = s.ordinal();
            assert!(o < Sysno::COUNT, "ordinal {} >= COUNT {}", o, Sysno::COUNT);
            assert!(!seen[o], "ordinal {} repeated", o);
            seen[o] = true;
        }
        assert!(seen.iter().all(|x| *x), "ordinal range has gaps");
    }
}
