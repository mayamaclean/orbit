//! Syscall numbers.
//!
//! Numbers are load-bearing: the kernel dispatch table in `s_trap` matches on
//! them directly. Do not renumber an existing entry; add new ones at the end.

pub const EXIT: usize = 0;
pub const SERIAL_PRINT: usize = 1;
pub const SLEEP_MS: usize = 2;
pub const CONSOLE_WRITE: usize = 3;
pub const READ_STDIN: usize = 4;
pub const SET_AFFINITY: usize = 5;
pub const GET_AFFINITY: usize = 6;
pub const GET_HART_ID: usize = 7;
/// `get_micros() -> u64` — absolute monotonic microseconds since
/// system boot. Cheap unprivileged tick read on the kernel side
/// (RISC-V `time` CSR / 10 since QEMU virt clocks at 10 MHz),
/// returned in `a0`. Opaque base: only differences are meaningful.
pub const GET_MICROS: usize = 8;

/// `get_realtime() -> (secs: i64, nsec: u32)` — wall-clock time
/// since the UNIX epoch. Two-return ecall: seconds in `a0`, nanoseconds
/// in `a1`. Backed by the Goldfish RTC on QEMU's `virt` machine
/// (nanosecond resolution at the device; the syscall does the
/// `divmod 1_000_000_000` so callers can build a `(secs, nsec)`
/// `SystemTime` directly).
///
/// `nsec ∈ [0, 999_999_999]`. No errno path — the device read can't
/// fail. Time can step backward across host suspend/resume; for
/// monotonic intervals use [`GET_MICROS`].
pub const GET_REALTIME: usize = 10;

/// `thread_exit() -> !` — terminate just the calling thread, leaving
/// sibling threads of the same process running. Used by std's thread
/// trampoline (and any future `pthread_exit`-shaped caller) when a
/// worker's closure returns; the *process* is only torn down via
/// [`EXIT`], which is now exit-group.
///
/// No exit code: thread-level status surfaces through std's
/// `JoinHandle` futex word (`EXITED`/`RUNNING`), not through the
/// kernel. The arg slot is reserved for a future `pthread_exit_value`
/// pattern — for now any value passed is ignored.
pub const THREAD_EXIT: usize = 11;

/// `pledge(req: *const PermsRequest) -> 0 | -errno` — narrow this
/// process's `perms` and `allowed_perms` masks. The kernel mutates
/// `Process.permissions` and propagates the narrowed snapshot to
/// every live thread of the process, so the dispatch-site gate
/// EPERMs subsequent calls that needed a class the caller just
/// pledged away.
///
/// `req` is a `*const orbit_abi::perms::PermsRequest` in user
/// memory; the kernel reads both `ClassMask` fields via the
/// standard boundary-deserializer path. Errors:
/// - `EFAULT` — `req` doesn't translate under the caller's satp.
/// - `EPERM` — caller has pledged `class::PLEDGE` away.
pub const PLEDGE: usize = 9;

pub const MMAP: usize = 4096;
pub const CREATE_NETCH: usize = 4097;
pub const CLOSE_HANDLE: usize = 4098;
pub const CREATE_PROCESS: usize = 4099;
pub const NC_YIELD: usize = 4100;
pub const QUERY_STATS: usize = 4101;
pub const QUERY_SYSCALL_STATS: usize = 4102;
pub const CREATE_PROCESS_EX: usize = 4103;
pub const ARGV_ENVP: usize = 4104;
/// `create_process_v2(args: *const CreateProcessV2Args) -> pid | -errno`
/// — role-aware spawn with explicit permission narrowing. Replaces
/// `CREATE_PROCESS` and `CREATE_PROCESS_EX` for callers that need
/// a `target_role` and `PermsRequest` (the older numbers stay live
/// for ABI compat; they spawn into `BOOTSTRAP` with `Permissions::ALL`).
/// Args struct in user memory because the call carries enough
/// fields to overflow the `a1..a7` register window comfortably.
///
/// `check_transition` runs against the parent's role; on `Err(_)`
/// the kernel records a `DenialEvent::RoleDeny` audit entry, bumps
/// the parent's `role_denials` counter, and returns `-EPERM`. On
/// success the witness-derived `Permissions` are installed on the
/// child.
pub const CREATE_PROCESS_V2: usize = 4105;
/// `query_denial_log(buf: *mut u8, buf_len: usize) -> bytes | -errno`
/// — copy the kernel-wide denial event ring into a user buffer in
/// chronological order. Reply layout is a packed sequence of
/// `DenialEvent`s; `bytes_written / size_of::<DenialEvent>()`
/// gives the count. Buffer too small: returns the prefix that fits
/// + `bytes_written` reflecting the partial copy.
///
/// Unlike `query_stats`, the ring isn't per-process — it covers
/// every gate event system-wide. Sized at
/// `DENIAL_RING_CAPACITY * size_of::<DenialEvent>()` for a complete
/// snapshot (≈2.4 KiB at the cap of 50 events).
///
/// **Cross-process disclosure.** The reply contains pids, tids,
/// syscall numbers, and (for `RoleDeny`) source/target roles for
/// *every* denial system-wide, not just the caller's. A process
/// holding `class::STATS` can correlate which other processes are
/// calling which syscalls and where they're being gated.
/// Acceptable today (orbit is single-tenant); once multi-tenant
/// workloads land, consider gating this behind a dedicated
/// `class::DENIAL_LOG` so a low-trust observer role can hold STATS
/// without seeing other processes' denials.
pub const QUERY_DENIAL_LOG: usize = 4106;
/// `chdir(path_ptr: *const u8, path_len: usize) -> 0 | -errno` —
/// replace the calling process's `cwd` with the (absolute, UTF-8)
/// path at `[path_ptr, path_ptr+path_len)`. v1 rejects relative
/// paths with `EINVAL` (no cwd-relative chdir until path-walk
/// resolution lands). The new value is the literal bytes passed —
/// the kernel does not canonicalize, so callers should pre-strip
/// trailing slashes if they want `getcwd` to echo the same shape.
///
/// Errnos:
/// - `EFAULT` — buffer doesn't translate under the caller's satp.
/// - `EINVAL` — non-absolute path, non-UTF-8, or empty length.
/// - `ENAMETOOLONG` — path exceeds the kernel-side cwd cap (4 KiB).
/// - `ENOENT` — path doesn't resolve to an existing directory in
///   the active filesystem (validated at chdir time so subsequent
///   relative-path syscalls don't dangle).
pub const CHDIR: usize = 4107;
/// `getcwd(buf_ptr: *mut u8, buf_len: usize) -> bytes_written | -errno`
/// — copy the calling process's `cwd` into the user buffer. The
/// returned byte count is the cwd's length (no NUL terminator).
///
/// Errnos:
/// - `EFAULT` — buffer doesn't translate under the caller's satp.
/// - `ERANGE` — buffer too small for the current cwd.
pub const GETCWD: usize = 4108;

/// `getuid() -> uid` — POSIX `getuid(2)`. Returns the calling
/// process's real uid. Reads the per-thread credential snapshot
/// without locking. Never fails — uid is a `u32` so the return
/// always fits in the positive range of `isize`.
pub const GETUID: usize = 4109;

/// `geteuid() -> euid` — POSIX `geteuid(2)`. Returns the calling
/// process's effective uid. Same shape as [`GETUID`] but reads
/// `Thread.euid`. Splitting effective from real avoids the negative-
/// `isize` hazard a bundled `(real << 32) | effective` return would
/// have for uids ≥ 0x8000_0000.
pub const GETEUID: usize = 4110;

/// `getgid() -> gid` — POSIX `getgid(2)`. Real gid counterpart to
/// [`GETUID`].
pub const GETGID: usize = 4111;

/// `getegid() -> egid` — POSIX `getegid(2)`. Effective gid counterpart
/// to [`GETEUID`].
pub const GETEGID: usize = 4112;

/// `getgroups(buf_ptr: *mut u32, count: usize) -> count | -errno` —
/// POSIX `getgroups(2)`. Copy the caller's supplementary group list
/// into the user buffer (one `u32` per slot) and return the number of
/// entries copied. POSIX special case: `count == 0` (regardless of
/// `buf_ptr`) returns the current group count *without* writing —
/// callers use this to size the real call.
///
/// `count` is in `u32` slots, not bytes. Buffer length in bytes is
/// implicitly `count * 4`. Maximum entries the kernel will write is
/// [`process::NGROUPS_MAX`].
///
/// Errnos:
/// - `EFAULT` — `buf_ptr` doesn't translate when `count > 0`.
/// - `EINVAL` — buffer straddles a page boundary.
/// - `ERANGE` — `count > 0` but smaller than the current group count.
pub const GETGROUPS: usize = 4113;

/// `getlogin(buf_ptr: *mut u8, buf_len: usize) -> bytes_written | -errno`
/// — POSIX `getlogin_r(3)` shape (Rust prefers the bounded form over
/// `getlogin(3)`'s static-buffer return). Copy the calling process's
/// session login name (no NUL terminator) into the user buffer.
///
/// Errnos:
/// - `EFAULT` — `buf_ptr` doesn't translate.
/// - `EINVAL` — buffer straddles a page boundary.
/// - `ERANGE` — buffer too small for the current login name.
/// - `ENOENT` — no login name installed (initial process state, before
///   any `setlogin` has run).
pub const GETLOGIN: usize = 4114;

/// `setuid(uid) -> 0 | -errno` — POSIX `setuid(2)`. Mutates the
/// calling process's uid triplet under the standard rules:
///   - euid == 0: set all three (real, effective, saved) to `uid` —
///     the privilege-drop path.
///   - euid != 0: set only euid, IFF `uid ∈ {ruid, suid}` (the
///     privilege-toggle path used by setuid-bit binaries; real and
///     saved are unchanged).
///
/// Per-thread credential snapshots refreshed in the same call by
/// walking the calling process's thread set, so subsequent syscalls
/// from sibling threads observe the new identity. Gated on
/// [`orbit_abi::perms::class::PROC_CRED`]; pledging it away locks the
/// caller's identity for the rest of its lifetime.
///
/// Errnos:
/// - `EPERM` — non-root caller passed a uid that isn't in
///   `{ruid, suid}`.
pub const SETUID: usize = 4115;

/// `setgid(gid) -> 0 | -errno` — POSIX `setgid(2)`. Same shape as
/// [`SETUID`] for the gid triplet. EPERM rules apply against
/// `{rgid, sgid}` rather than the uid pair, but the privilege-drop
/// vs privilege-toggle distinction is identical.
pub const SETGID: usize = 4116;

/// `setgroups(buf_ptr: *const u32, count: usize) -> 0 | -errno` —
/// POSIX `setgroups(2)`. Replace the caller's supplementary group
/// list with the `count` `u32`s at `buf_ptr`. Requires `euid == 0`
/// (matches POSIX); a caller that has dropped privilege via
/// `setuid(N)` for non-zero N gets `EPERM`.
///
/// Errnos:
/// - `EPERM` — caller's `euid != 0`.
/// - `EINVAL` — `count > process::NGROUPS_MAX` (16).
/// - `EFAULT` — `buf_ptr` doesn't translate.
pub const SETGROUPS: usize = 4117;

/// `setlogin(name_ptr: *const u8, name_len: usize) -> 0 | -errno` —
/// POSIX `setlogin(2)`. Stamp the calling process's session login
/// name. Caller must have `euid == 0` (matches OpenBSD; the syscall
/// is meant for `login(1)` to install the authenticated user's name
/// on the session). Capped at `MAXLOGNAME = 32` bytes.
///
/// Errnos:
/// - `EPERM` — caller's `euid != 0`.
/// - `EINVAL` — non-UTF-8 input.
/// - `ENAMETOOLONG` — `name_len > 32`.
/// - `EFAULT` — `name_ptr` doesn't translate.
pub const SETLOGIN: usize = 4118;

// 5000+ — multi-thread / SMP control plane. Numbered out of the 4096
// block so the categorical split is obvious in dispatch tables and so
// future single-process-spanning syscalls (futex wake/wait, etc.)
// share the same range.
pub const CREATE_THREAD: usize = 5000;
pub const GETPID: usize = 5001;
pub const GETTID: usize = 5002;
pub const WAIT_PID: usize = 5003;
pub const FUTEX_WAIT: usize = 5004;
pub const FUTEX_WAKE: usize = 5005;

// 6000+ — filesystem. v1 is read-only tarfs; close re-uses
// `CLOSE_HANDLE = 4098` (handle table is shared across NetCh / file
// fds) so there's no FS_CLOSE here.
pub const FS_OPEN: usize = 6000;
pub const FS_READ: usize = 6001;
pub const FS_STAT: usize = 6002;
pub const FS_READDIR: usize = 6003;
/// `fs_fstat(fd, &mut Stat) -> 0 | -errno` — fill `*stat` with metadata
/// for the file backing `fd`. Mirror of `FS_STAT` but keyed on an
/// already-open fd, so callers don't have to retain the path used at
/// open. Backs `std::fs::File::metadata`.
///
/// Errnos:
/// - `EBADF` — `fd` not open in the calling process.
/// - `EFAULT` — `stat` doesn't translate.
/// - `EINVAL` — `stat` straddles a page (same constraint as `FS_STAT`).
/// - `EIO`   — backing fs lookup failed.
pub const FS_FSTAT: usize = 6005;

/// `fs_seek(fd, offset, whence) -> new_offset | -errno` — reposition
/// the byte cursor on a regular-file fd. `whence` follows POSIX:
/// `SEEK_SET = 0` (absolute), `SEEK_CUR = 1` (relative to current
/// offset), `SEEK_END = 2` (relative to file size). `offset` is
/// `i64`; it sign-extends the syscall arg. The return value is the
/// resulting absolute offset.
///
/// Errnos:
/// - `EBADF`  — `fd` not open, or not a regular-file fd (directories
///   use `fs_readdir`'s opaque cursor instead).
/// - `EINVAL` — invalid `whence`, or the resolved offset would be
///   negative. Past-EOF is allowed (POSIX hole semantics — orbit's
///   read-only fs returns 0 on those reads).
pub const FS_SEEK: usize = 6004;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Sysno {
    Exit = EXIT,
    SerialPrint = SERIAL_PRINT,
    SleepMs = SLEEP_MS,
    ConsoleWrite = CONSOLE_WRITE,
    ReadStdin = READ_STDIN,
    SetAffinity = SET_AFFINITY,
    GetAffinity = GET_AFFINITY,
    GetHartId = GET_HART_ID,
    GetMicros = GET_MICROS,
    GetRealtime = GET_REALTIME,
    ThreadExit = THREAD_EXIT,
    Pledge = PLEDGE,
    Mmap = MMAP,
    CreateNetch = CREATE_NETCH,
    CloseHandle = CLOSE_HANDLE,
    CreateProcess = CREATE_PROCESS,
    NcYield = NC_YIELD,
    QueryStats = QUERY_STATS,
    QuerySyscallStats = QUERY_SYSCALL_STATS,
    CreateProcessEx = CREATE_PROCESS_EX,
    ArgvEnvp = ARGV_ENVP,
    CreateProcessV2 = CREATE_PROCESS_V2,
    QueryDenialLog = QUERY_DENIAL_LOG,
    Chdir = CHDIR,
    Getcwd = GETCWD,
    GetUid = GETUID,
    GetEuid = GETEUID,
    GetGid = GETGID,
    GetEgid = GETEGID,
    GetGroups = GETGROUPS,
    GetLogin = GETLOGIN,
    SetUid = SETUID,
    SetGid = SETGID,
    SetGroups = SETGROUPS,
    SetLogin = SETLOGIN,
    CreateThread = CREATE_THREAD,
    GetPid = GETPID,
    GetTid = GETTID,
    WaitPid = WAIT_PID,
    FutexWait = FUTEX_WAIT,
    FutexWake = FUTEX_WAKE,
    FsOpen = FS_OPEN,
    FsRead = FS_READ,
    FsStat = FS_STAT,
    FsReaddir = FS_READDIR,
    FsSeek = FS_SEEK,
    FsFstat = FS_FSTAT,
}

impl Sysno {
    pub const fn from_usize(n: usize) -> Option<Self> {
        Some(match n {
            EXIT => Self::Exit,
            SERIAL_PRINT => Self::SerialPrint,
            SLEEP_MS => Self::SleepMs,
            CONSOLE_WRITE => Self::ConsoleWrite,
            READ_STDIN => Self::ReadStdin,
            SET_AFFINITY => Self::SetAffinity,
            GET_AFFINITY => Self::GetAffinity,
            GET_HART_ID => Self::GetHartId,
            GET_MICROS => Self::GetMicros,
            GET_REALTIME => Self::GetRealtime,
            THREAD_EXIT => Self::ThreadExit,
            PLEDGE => Self::Pledge,
            MMAP => Self::Mmap,
            CREATE_NETCH => Self::CreateNetch,
            CLOSE_HANDLE => Self::CloseHandle,
            CREATE_PROCESS => Self::CreateProcess,
            NC_YIELD => Self::NcYield,
            QUERY_STATS => Self::QueryStats,
            QUERY_SYSCALL_STATS => Self::QuerySyscallStats,
            CREATE_PROCESS_EX => Self::CreateProcessEx,
            ARGV_ENVP => Self::ArgvEnvp,
            CREATE_PROCESS_V2 => Self::CreateProcessV2,
            QUERY_DENIAL_LOG => Self::QueryDenialLog,
            CHDIR => Self::Chdir,
            GETCWD => Self::Getcwd,
            GETUID => Self::GetUid,
            GETEUID => Self::GetEuid,
            GETGID => Self::GetGid,
            GETEGID => Self::GetEgid,
            GETGROUPS => Self::GetGroups,
            GETLOGIN => Self::GetLogin,
            SETUID => Self::SetUid,
            SETGID => Self::SetGid,
            SETGROUPS => Self::SetGroups,
            SETLOGIN => Self::SetLogin,
            CREATE_THREAD => Self::CreateThread,
            GETPID => Self::GetPid,
            GETTID => Self::GetTid,
            WAIT_PID => Self::WaitPid,
            FUTEX_WAIT => Self::FutexWait,
            FUTEX_WAKE => Self::FutexWake,
            FS_OPEN => Self::FsOpen,
            FS_READ => Self::FsRead,
            FS_STAT => Self::FsStat,
            FS_READDIR => Self::FsReaddir,
            FS_SEEK => Self::FsSeek,
            FS_FSTAT => Self::FsFstat,
            _ => return None,
        })
    }

    /// Stable, dense ordinal for stats tables. Tied to the order of
    /// `match` arms below — append-only, never reorder. The raw
    /// syscall-number space is sparse (0-7, 4096-4102, 5000) so we
    /// can't use it as an array index.
    pub const fn ordinal(self) -> usize {
        match self {
            Self::Exit => 0,
            Self::SerialPrint => 1,
            Self::SleepMs => 2,
            Self::ConsoleWrite => 3,
            Self::ReadStdin => 4,
            Self::SetAffinity => 5,
            Self::GetAffinity => 6,
            Self::GetHartId => 7,
            Self::Mmap => 8,
            Self::CreateNetch => 9,
            Self::CloseHandle => 10,
            Self::CreateProcess => 11,
            Self::NcYield => 12,
            Self::QueryStats => 13,
            Self::QuerySyscallStats => 14,
            Self::CreateThread => 15,
            Self::GetMicros => 16,
            Self::FsOpen => 17,
            Self::FsRead => 18,
            Self::FsStat => 19,
            Self::GetPid => 20,
            Self::GetTid => 21,
            Self::WaitPid => 22,
            Self::CreateProcessEx => 23,
            Self::ArgvEnvp => 24,
            Self::FutexWait => 25,
            Self::FutexWake => 26,
            Self::FsReaddir => 27,
            Self::Pledge => 28,
            Self::CreateProcessV2 => 29,
            Self::QueryDenialLog => 30,
            Self::Chdir => 31,
            Self::Getcwd => 32,
            Self::FsSeek => 33,
            Self::FsFstat => 34,
            Self::GetUid => 35,
            Self::GetEuid => 36,
            Self::GetGid => 37,
            Self::GetEgid => 38,
            Self::GetGroups => 39,
            Self::GetLogin => 40,
            Self::SetUid => 41,
            Self::SetGid => 42,
            Self::SetGroups => 43,
            Self::SetLogin => 44,
            Self::GetRealtime => 45,
            Self::ThreadExit => 46,
        }
    }

    /// Number of distinct ordinals returned by [`Self::ordinal`]. Pinned
    /// so the per-syscall stats table size is part of the ABI; bump
    /// when adding a `Sysno` variant. Older userland with a smaller
    /// COUNT reads a prefix of the kernel's table; newer userland with
    /// a larger COUNT treats the kernel's missing slots as zero.
    pub const COUNT: usize = 47;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_defined_number_decodes() {
        assert_eq!(Sysno::from_usize(EXIT), Some(Sysno::Exit));
        assert_eq!(Sysno::from_usize(SERIAL_PRINT), Some(Sysno::SerialPrint));
        assert_eq!(Sysno::from_usize(SLEEP_MS), Some(Sysno::SleepMs));
        assert_eq!(Sysno::from_usize(CONSOLE_WRITE), Some(Sysno::ConsoleWrite));
        assert_eq!(Sysno::from_usize(READ_STDIN), Some(Sysno::ReadStdin));
        assert_eq!(Sysno::from_usize(SET_AFFINITY), Some(Sysno::SetAffinity));
        assert_eq!(Sysno::from_usize(GET_AFFINITY), Some(Sysno::GetAffinity));
        assert_eq!(Sysno::from_usize(GET_HART_ID), Some(Sysno::GetHartId));
        assert_eq!(Sysno::from_usize(GET_MICROS), Some(Sysno::GetMicros));
        assert_eq!(Sysno::from_usize(GET_REALTIME), Some(Sysno::GetRealtime));
        assert_eq!(Sysno::from_usize(THREAD_EXIT), Some(Sysno::ThreadExit));
        assert_eq!(Sysno::from_usize(MMAP), Some(Sysno::Mmap));
        assert_eq!(Sysno::from_usize(CREATE_NETCH), Some(Sysno::CreateNetch));
        assert_eq!(Sysno::from_usize(CLOSE_HANDLE), Some(Sysno::CloseHandle));
        assert_eq!(
            Sysno::from_usize(CREATE_PROCESS),
            Some(Sysno::CreateProcess)
        );
        assert_eq!(Sysno::from_usize(NC_YIELD), Some(Sysno::NcYield));
        assert_eq!(Sysno::from_usize(QUERY_STATS), Some(Sysno::QueryStats));
        assert_eq!(
            Sysno::from_usize(QUERY_SYSCALL_STATS),
            Some(Sysno::QuerySyscallStats)
        );
        assert_eq!(Sysno::from_usize(CREATE_THREAD), Some(Sysno::CreateThread));
        assert_eq!(Sysno::from_usize(FS_OPEN), Some(Sysno::FsOpen));
        assert_eq!(Sysno::from_usize(FS_READ), Some(Sysno::FsRead));
        assert_eq!(Sysno::from_usize(FS_STAT), Some(Sysno::FsStat));
        assert_eq!(Sysno::from_usize(FS_READDIR), Some(Sysno::FsReaddir));
        assert_eq!(Sysno::from_usize(GETPID), Some(Sysno::GetPid));
        assert_eq!(Sysno::from_usize(GETTID), Some(Sysno::GetTid));
        assert_eq!(Sysno::from_usize(WAIT_PID), Some(Sysno::WaitPid));
        assert_eq!(Sysno::from_usize(FUTEX_WAIT), Some(Sysno::FutexWait));
        assert_eq!(Sysno::from_usize(FUTEX_WAKE), Some(Sysno::FutexWake));
        assert_eq!(
            Sysno::from_usize(CREATE_PROCESS_EX),
            Some(Sysno::CreateProcessEx)
        );
        assert_eq!(Sysno::from_usize(ARGV_ENVP), Some(Sysno::ArgvEnvp));
        assert_eq!(Sysno::from_usize(PLEDGE), Some(Sysno::Pledge));
        assert_eq!(
            Sysno::from_usize(CREATE_PROCESS_V2),
            Some(Sysno::CreateProcessV2)
        );
        assert_eq!(
            Sysno::from_usize(QUERY_DENIAL_LOG),
            Some(Sysno::QueryDenialLog)
        );
        assert_eq!(Sysno::from_usize(CHDIR), Some(Sysno::Chdir));
        assert_eq!(Sysno::from_usize(GETCWD), Some(Sysno::Getcwd));
        assert_eq!(Sysno::from_usize(FS_SEEK), Some(Sysno::FsSeek));
        assert_eq!(Sysno::from_usize(FS_FSTAT), Some(Sysno::FsFstat));
        assert_eq!(Sysno::from_usize(GETUID), Some(Sysno::GetUid));
        assert_eq!(Sysno::from_usize(GETEUID), Some(Sysno::GetEuid));
        assert_eq!(Sysno::from_usize(GETGID), Some(Sysno::GetGid));
        assert_eq!(Sysno::from_usize(GETEGID), Some(Sysno::GetEgid));
        assert_eq!(Sysno::from_usize(GETGROUPS), Some(Sysno::GetGroups));
        assert_eq!(Sysno::from_usize(GETLOGIN), Some(Sysno::GetLogin));
        assert_eq!(Sysno::from_usize(SETUID), Some(Sysno::SetUid));
        assert_eq!(Sysno::from_usize(SETGID), Some(Sysno::SetGid));
        assert_eq!(Sysno::from_usize(SETGROUPS), Some(Sysno::SetGroups));
        assert_eq!(Sysno::from_usize(SETLOGIN), Some(Sysno::SetLogin));
    }

    #[test]
    fn unknown_returns_none() {
        // 9 is PLEDGE, 10 is GET_REALTIME, 11 is THREAD_EXIT — used to
        // be holes below 4096.
        assert_eq!(Sysno::from_usize(12), None);
        assert_eq!(Sysno::from_usize(4095), None);
        // 4105..=4118 are now CREATE_PROCESS_V2 / QUERY_DENIAL_LOG /
        // CHDIR / GETCWD / GETUID / GETEUID / GETGID / GETEGID /
        // GETGROUPS / GETLOGIN / SETUID / SETGID / SETGROUPS / SETLOGIN.
        assert_eq!(Sysno::from_usize(4119), None);
        assert_eq!(Sysno::from_usize(4999), None);
        assert_eq!(Sysno::from_usize(5006), None);
        assert_eq!(Sysno::from_usize(5999), None);
        // 6004 / 6005 are now FS_SEEK / FS_FSTAT.
        assert_eq!(Sysno::from_usize(6006), None);
        assert_eq!(Sysno::from_usize(usize::MAX), None);
    }

    #[test]
    fn variant_discriminant_matches_constant() {
        assert_eq!(Sysno::Exit as usize, EXIT);
        assert_eq!(Sysno::SerialPrint as usize, SERIAL_PRINT);
        assert_eq!(Sysno::SleepMs as usize, SLEEP_MS);
        assert_eq!(Sysno::ConsoleWrite as usize, CONSOLE_WRITE);
        assert_eq!(Sysno::ReadStdin as usize, READ_STDIN);
        assert_eq!(Sysno::SetAffinity as usize, SET_AFFINITY);
        assert_eq!(Sysno::GetAffinity as usize, GET_AFFINITY);
        assert_eq!(Sysno::GetHartId as usize, GET_HART_ID);
        assert_eq!(Sysno::GetMicros as usize, GET_MICROS);
        assert_eq!(Sysno::GetRealtime as usize, GET_REALTIME);
        assert_eq!(Sysno::ThreadExit as usize, THREAD_EXIT);
        assert_eq!(Sysno::Mmap as usize, MMAP);
        assert_eq!(Sysno::CreateNetch as usize, CREATE_NETCH);
        assert_eq!(Sysno::CloseHandle as usize, CLOSE_HANDLE);
        assert_eq!(Sysno::CreateProcess as usize, CREATE_PROCESS);
        assert_eq!(Sysno::NcYield as usize, NC_YIELD);
        assert_eq!(Sysno::QueryStats as usize, QUERY_STATS);
        assert_eq!(Sysno::QuerySyscallStats as usize, QUERY_SYSCALL_STATS);
        assert_eq!(Sysno::CreateThread as usize, CREATE_THREAD);
        assert_eq!(Sysno::FsOpen as usize, FS_OPEN);
        assert_eq!(Sysno::FsRead as usize, FS_READ);
        assert_eq!(Sysno::FsStat as usize, FS_STAT);
        assert_eq!(Sysno::FsReaddir as usize, FS_READDIR);
        assert_eq!(Sysno::GetPid as usize, GETPID);
        assert_eq!(Sysno::GetTid as usize, GETTID);
        assert_eq!(Sysno::WaitPid as usize, WAIT_PID);
        assert_eq!(Sysno::FutexWait as usize, FUTEX_WAIT);
        assert_eq!(Sysno::FutexWake as usize, FUTEX_WAKE);
        assert_eq!(Sysno::CreateProcessEx as usize, CREATE_PROCESS_EX);
        assert_eq!(Sysno::ArgvEnvp as usize, ARGV_ENVP);
        assert_eq!(Sysno::Pledge as usize, PLEDGE);
        assert_eq!(Sysno::CreateProcessV2 as usize, CREATE_PROCESS_V2);
        assert_eq!(Sysno::QueryDenialLog as usize, QUERY_DENIAL_LOG);
        assert_eq!(Sysno::Chdir as usize, CHDIR);
        assert_eq!(Sysno::Getcwd as usize, GETCWD);
        assert_eq!(Sysno::FsSeek as usize, FS_SEEK);
        assert_eq!(Sysno::FsFstat as usize, FS_FSTAT);
        assert_eq!(Sysno::GetUid as usize, GETUID);
        assert_eq!(Sysno::GetEuid as usize, GETEUID);
        assert_eq!(Sysno::GetGid as usize, GETGID);
        assert_eq!(Sysno::GetEgid as usize, GETEGID);
        assert_eq!(Sysno::GetGroups as usize, GETGROUPS);
        assert_eq!(Sysno::GetLogin as usize, GETLOGIN);
        assert_eq!(Sysno::SetUid as usize, SETUID);
        assert_eq!(Sysno::SetGid as usize, SETGID);
        assert_eq!(Sysno::SetGroups as usize, SETGROUPS);
        assert_eq!(Sysno::SetLogin as usize, SETLOGIN);
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
        assert_eq!(GET_REALTIME, 10);
        assert_eq!(THREAD_EXIT, 11);
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
        assert_eq!(QUERY_DENIAL_LOG, 4106);
        assert_eq!(CHDIR, 4107);
        assert_eq!(GETCWD, 4108);
        assert_eq!(FS_SEEK, 6004);
        assert_eq!(FS_FSTAT, 6005);
        assert_eq!(GETUID, 4109);
        assert_eq!(GETEUID, 4110);
        assert_eq!(GETGID, 4111);
        assert_eq!(GETEGID, 4112);
        assert_eq!(GETGROUPS, 4113);
        assert_eq!(GETLOGIN, 4114);
        assert_eq!(SETUID, 4115);
        assert_eq!(SETGID, 4116);
        assert_eq!(SETGROUPS, 4117);
        assert_eq!(SETLOGIN, 4118);
    }

    #[test]
    fn ordinals_are_dense_and_unique() {
        // Iterate every variant via from_usize so we can't forget to
        // update the test when adding a Sysno.
        let all = [
            Sysno::Exit,
            Sysno::SerialPrint,
            Sysno::SleepMs,
            Sysno::ConsoleWrite,
            Sysno::ReadStdin,
            Sysno::SetAffinity,
            Sysno::GetAffinity,
            Sysno::GetHartId,
            Sysno::Mmap,
            Sysno::CreateNetch,
            Sysno::CloseHandle,
            Sysno::CreateProcess,
            Sysno::NcYield,
            Sysno::QueryStats,
            Sysno::QuerySyscallStats,
            Sysno::CreateThread,
            Sysno::GetMicros,
            Sysno::FsOpen,
            Sysno::FsRead,
            Sysno::FsStat,
            Sysno::GetPid,
            Sysno::GetTid,
            Sysno::WaitPid,
            Sysno::CreateProcessEx,
            Sysno::ArgvEnvp,
            Sysno::FutexWait,
            Sysno::FutexWake,
            Sysno::FsReaddir,
            Sysno::Pledge,
            Sysno::CreateProcessV2,
            Sysno::QueryDenialLog,
            Sysno::Chdir,
            Sysno::Getcwd,
            Sysno::FsSeek,
            Sysno::FsFstat,
            Sysno::GetUid,
            Sysno::GetEuid,
            Sysno::GetGid,
            Sysno::GetEgid,
            Sysno::GetGroups,
            Sysno::GetLogin,
            Sysno::SetUid,
            Sysno::SetGid,
            Sysno::SetGroups,
            Sysno::SetLogin,
            Sysno::GetRealtime,
            Sysno::ThreadExit,
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
