//! Permissions, role IDs, and the syscall→class mapping.
//!
//! Two orthogonal axes encoded in [`Permissions`]:
//!
//! - **Syscall-class bitmask** — `perms` (effective for this process) and
//!   `allowed_perms` (cap for direct children). Monotonic narrowing only —
//!   `pledge` is the sole mutator post-spawn. Bit positions in [`class`].
//! - **Role identifier** — opaque `u32` indexed into the kernel-side
//!   role registry (which gates create_process_v2 transitions). Constants
//!   in [`role`].
//!
//! See [docs/dev/permissions-roles.md](../../../docs/dev/permissions-roles.md)
//! for the design rationale, the role transition matrix, and the
//! migration plan. This module owns the *types* and the *syscall→class*
//! lookup; the kernel-side registry + clamping logic live in
//! `orbit-core::roles`.
//!
//! # ABI shape
//!
//! `Permissions` is `#[repr(C)]` and 32 bytes. The `_pad` and `_reserved`
//! tail are part of the wire shape — they exist so future axes (MLS
//! level, label fingerprint) can be added without renumbering or growing
//! the struct on disk. New code MUST initialize them to zero.

use crate::syscall;

/// Process-wide permission state. Lives on `Process` in the kernel; user
/// code constructs instances to pass to `pledge` and `create_process_v2`.
///
/// Invariant maintained by `pledge` and `derive_child_perms`:
/// `perms & !allowed_perms == 0` (effective is always a subset of cap).
/// Code that constructs a `Permissions` directly is responsible for
/// preserving this — prefer [`Permissions::new`] which clamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Permissions {
    /// Effective syscall-class bitmask for this process. Bit positions
    /// in [`class`].
    pub perms: u64,
    /// Cap on what direct children may receive at create_process_v2.
    /// Always `>= perms` (kernel maintains; user-supplied violations are
    /// clamped, not rejected, mirroring OpenBSD's pledge).
    pub allowed_perms: u64,
    /// Role identifier — index into the kernel role registry. `0` =
    /// [`role::NOROLE`] (terminal, no outbound transitions).
    pub role: u32,
    /// Reserved padding; keeps the struct's 8-byte alignment explicit.
    /// Must be zero.
    pub _pad: u32,
    /// Reserved for future axes (MLS level, label fingerprint, RBAC
    /// supplemental groups). Must be zero today; non-zero is rejected
    /// by the kernel so a future client doesn't silently downgrade
    /// against an old kernel.
    pub _reserved: [u64; 2],
}

/// Syscall-class bitmask values. Bits are part of the user/kernel ABI
/// — append-only, never reorder, never recycle a freed bit. See
/// [`Permissions::class_for`] for which class each syscall maps to.
pub mod class {
    /// `serial_print`, `console_write`, `read_stdin`, `get_micros`,
    /// `close_handle`. The last is in stdio rather than its own class
    /// because closing handles is a destructor — denying it strands
    /// resources, never a useful sandboxing tool.
    pub const STDIO: u64 = 1 << 0;

    /// `exit`, `getpid`, `gettid`, `wait_pid`, `argv_envp`. The
    /// "what process am I" surface — a process that can't exit cleanly
    /// is a worse outcome than one that can.
    pub const PROC_LIFE: u64 = 1 << 1;

    /// `create_process`, `create_process_ex`, `create_process_v2`,
    /// `create_thread`. Coarse — a future split into PROC_SPAWN_PROC
    /// vs PROC_SPAWN_THREAD is foreseeable but not motivated yet.
    pub const PROC_SPAWN: u64 = 1 << 2;

    /// `sleep_ms`, `set_affinity`, `get_affinity`, `get_hart_id`,
    /// `nc_yield`. Anything that hands the scheduler a parking
    /// or migration request.
    pub const SCHED: u64 = 1 << 3;

    /// `mmap` with `share_with_kernel = false`. Private user pages,
    /// no kernel alias.
    pub const VMEM: u64 = 1 << 4;

    /// `mmap` with `share_with_kernel = true`. Pages with a long-lived
    /// kernel KDMAP alias — used by NetChannel rings and any future
    /// kernel-readable region. Separate class because the W^X / alias
    /// hazard is materially different from VMEM.
    pub const VMEM_SHARED: u64 = 1 << 5;

    /// `create_netch`. NetChannel allocation; gates network reach
    /// orthogonally to VMEM_SHARED (which only enables the *backing*
    /// pool — without NETCH you can have shared memory but no socket).
    pub const NETCH: u64 = 1 << 6;

    /// `fs_open` (read-only, today's only mode), `fs_read`, `fs_stat`,
    /// future readdir. Becomes `FS_RW` once writes land — at which
    /// point this bit narrows to "read paths only" and a new bit
    /// covers the writing surface.
    pub const FS_RO: u64 = 1 << 7;

    /// `futex_wait`, `futex_wake`. Cheap to deny in single-threaded
    /// processes; required for any threaded workload.
    pub const FUTEX: u64 = 1 << 8;

    /// `query_stats`, `query_syscall_stats`. Observability.
    pub const STATS: u64 = 1 << 9;

    /// `pledge` (the syscall itself). Default-on for the `BOOTSTRAP`
    /// and `LOADER` roles so they can self-narrow; a process that
    /// pledges this away can never re-narrow itself afterwards (which
    /// is fine — the typical caller does it once at startup).
    pub const PLEDGE: u64 = 1 << 10;

    /// Union of every defined class. Equal to a process's `perms` in
    /// the `BOOTSTRAP` role. Update when adding a class — the test
    /// `all_is_union_of_classes` pins the invariant.
    pub const ALL: u64 = STDIO
        | PROC_LIFE
        | PROC_SPAWN
        | SCHED
        | VMEM
        | VMEM_SHARED
        | NETCH
        | FS_RO
        | FUTEX
        | STATS
        | PLEDGE;
}

/// Well-known role IDs. Indices into the kernel-side `ROLES` table in
/// `orbit-core::roles`. The ID space is `u32`; only the lower 6 bits
/// are usable today because role.transitions packs target IDs into a
/// `u64` bitset. Widen to a multi-word bitset if we need >64 roles —
/// the field types here are forward-compatible with that.
pub mod role {
    /// Type alias used at every API boundary. Keeps the role-vs-uid
    /// distinction explicit at call sites.
    pub type RoleId = u32;

    /// Sentinel — no outbound transitions, all permissions zero.
    /// Default for `Permissions::ZERO`; never assigned by the loader
    /// or by `derive_child_perms` to a real process.
    pub const NOROLE: RoleId = 0;

    /// Initial role of `orbit-loader` and any process kmain spawns
    /// directly. Wide caps (essentially `ALL`) so a single self-pledge
    /// can shed anything; transitions to `LOADER` only.
    pub const BOOTSTRAP: RoleId = 1;

    /// `orbit-loader`. Wide caps + wide transitions — the single
    /// authority that converts signer identity into a role assignment.
    pub const LOADER: RoleId = 2;

    /// Interactive shell (today: `console`). Spawning hub: can
    /// transition to NET_CLIENT, FS_TOOL, WORKER. Itself excludes
    /// NETCH/VMEM_SHARED/FUTEX from `perms` but keeps them in
    /// `allowed_perms` so children that need them can be granted.
    pub const SHELL: RoleId = 3;

    /// Network-using leaf processes. Has NETCH; cannot pass NETCH to
    /// its own children (default_allowed excludes it). May spawn
    /// WORKER children but no further reach.
    pub const NET_CLIENT: RoleId = 4;

    /// Filesystem-using leaf. FS_RO + STDIO + VMEM. Same shape as
    /// NET_CLIENT but with FS_RO in place of NETCH.
    pub const FS_TOOL: RoleId = 5;

    /// Pure compute leaf — STDIO + VMEM + PLEDGE only, no spawn,
    /// no I/O. The "default sandbox" target.
    pub const WORKER: RoleId = 6;

    /// Long-lived service (future: signed daemons). NET + FS_RO +
    /// FUTEX + STATS. Spawn limited to WORKER children.
    pub const SERVICE: RoleId = 7;

    /// Total number of defined roles. Sized to fit the static `ROLES`
    /// table in `orbit-core::roles`; bump when adding a role.
    pub const COUNT: usize = 8;
}

impl Permissions {
    /// All-zero state — `NOROLE`, no perms, no allowed perms. Default
    /// for `#[derive(Default)]`-shaped construction; useful as a base
    /// for tests and as the fail-safe for any code path that needs to
    /// strand a process while the manager decides what to do with it.
    pub const ZERO: Self = Self {
        perms: 0,
        allowed_perms: 0,
        role: role::NOROLE,
        _pad: 0,
        _reserved: [0; 2],
    };

    /// `BOOTSTRAP` role with every class set on both perms and
    /// allowed_perms. Convenience for kmain's first-process spawn —
    /// callers that want the LOADER or SHELL defaults should consult
    /// the role registry instead.
    pub const ALL: Self = Self {
        perms: class::ALL,
        allowed_perms: class::ALL,
        role: role::BOOTSTRAP,
        _pad: 0,
        _reserved: [0; 2],
    };

    /// Construct a `Permissions` clamping `perms ⊆ allowed_perms`. The
    /// clamp prevents the caller from accidentally violating the
    /// kernel's invariant; if you want to detect the violation instead
    /// of silently fixing it, compare your inputs first.
    pub const fn new(perms: u64, allowed_perms: u64, role: role::RoleId) -> Self {
        Self {
            perms: perms & allowed_perms,
            allowed_perms,
            role,
            _pad: 0,
            _reserved: [0; 2],
        }
    }

    /// Permission class associated with `sysno`. Returns `0` for
    /// unknown / unmapped syscalls; the dispatch site should treat
    /// `0` as "no class — never allowed by any non-`ALL` mask," which
    /// also matches the desired behaviour for typo'd syscall numbers.
    ///
    /// Some syscalls (`mmap`, future `fs_open` with write flags) have
    /// an argument-conditional class — the coarse class returned here
    /// is the *minimum* needed to enter the handler, and the handler
    /// performs an extra check on the args. `mmap` returns `VMEM`;
    /// the handler upgrades to also requiring `VMEM_SHARED` when
    /// `share_with_kernel = true`.
    pub const fn class_for(sysno: usize) -> u64 {
        match sysno {
            syscall::EXIT => class::PROC_LIFE,
            syscall::SERIAL_PRINT => class::STDIO,
            syscall::SLEEP_MS => class::SCHED,
            syscall::CONSOLE_WRITE => class::STDIO,
            syscall::READ_STDIN => class::STDIO,
            syscall::SET_AFFINITY => class::SCHED,
            syscall::GET_AFFINITY => class::SCHED,
            syscall::GET_HART_ID => class::SCHED,
            syscall::GET_MICROS => class::STDIO,
            syscall::MMAP => class::VMEM,
            syscall::CREATE_NETCH => class::NETCH,
            syscall::CLOSE_HANDLE => class::STDIO,
            syscall::CREATE_PROCESS => class::PROC_SPAWN,
            syscall::NC_YIELD => class::SCHED,
            syscall::QUERY_STATS => class::STATS,
            syscall::QUERY_SYSCALL_STATS => class::STATS,
            syscall::CREATE_PROCESS_EX => class::PROC_SPAWN,
            syscall::ARGV_ENVP => class::PROC_LIFE,
            syscall::CREATE_THREAD => class::PROC_SPAWN,
            syscall::GETPID => class::PROC_LIFE,
            syscall::GETTID => class::PROC_LIFE,
            syscall::WAIT_PID => class::PROC_LIFE,
            syscall::FUTEX_WAIT => class::FUTEX,
            syscall::FUTEX_WAKE => class::FUTEX,
            syscall::FS_OPEN => class::FS_RO,
            syscall::FS_READ => class::FS_RO,
            syscall::FS_STAT => class::FS_RO,
            _ => 0,
        }
    }

    /// Does this process's `perms` permit `sysno`? Unknown sysnos are
    /// rejected (class is `0`, mask `& 0` is `0`). A class with multiple
    /// bits would require all of them; today every class is a single
    /// bit so this collapses to a single `&`.
    pub const fn allows(&self, sysno: usize) -> bool {
        let cls = Self::class_for(sysno);
        cls != 0 && (self.perms & cls) == cls
    }

    /// Pledge-style narrowing. The result is `self` clamped down to
    /// the requested masks: bits not present in `requested_*` are
    /// dropped; bits not present in `self.*` cannot be added back.
    /// Always succeeds — passing a broader mask is a no-op on those
    /// bits, mirroring OpenBSD's pledge (which silently clamps rather
    /// than EPERMing on attempted-broaden).
    ///
    /// Maintains `perms ⊆ allowed_perms` even if the caller's two
    /// masks would otherwise violate it: `perms` is intersected with
    /// the *new* `allowed_perms` after that's been computed.
    ///
    /// `role` and `_reserved` are preserved verbatim — pledge is
    /// permission narrowing, not a role change.
    pub const fn pledge(&self, requested_perms: u64, requested_allowed: u64) -> Self {
        let new_allowed = self.allowed_perms & requested_allowed;
        let new_perms = self.perms & requested_perms & new_allowed;
        Self {
            perms: new_perms,
            allowed_perms: new_allowed,
            role: self.role,
            _pad: self._pad,
            _reserved: self._reserved,
        }
    }
}

impl Default for Permissions {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the union — class::ALL must equal the OR of every
    /// individual constant. If a new class is added, this test fails
    /// until ALL is updated, which is the point.
    #[test]
    fn all_is_union_of_classes() {
        let union = class::STDIO
            | class::PROC_LIFE
            | class::PROC_SPAWN
            | class::SCHED
            | class::VMEM
            | class::VMEM_SHARED
            | class::NETCH
            | class::FS_RO
            | class::FUTEX
            | class::STATS
            | class::PLEDGE;
        assert_eq!(class::ALL, union);
    }

    #[test]
    fn class_constants_are_distinct_single_bits() {
        let all = [
            class::STDIO,
            class::PROC_LIFE,
            class::PROC_SPAWN,
            class::SCHED,
            class::VMEM,
            class::VMEM_SHARED,
            class::NETCH,
            class::FS_RO,
            class::FUTEX,
            class::STATS,
            class::PLEDGE,
        ];
        for c in all {
            assert!(c.is_power_of_two(), "class {c:#x} is not a single bit");
        }
        let mut accum = 0u64;
        for c in all {
            assert_eq!(accum & c, 0, "class {c:#x} overlaps an earlier class");
            accum |= c;
        }
    }

    #[test]
    fn zero_is_empty_and_norole() {
        assert_eq!(Permissions::ZERO.perms, 0);
        assert_eq!(Permissions::ZERO.allowed_perms, 0);
        assert_eq!(Permissions::ZERO.role, role::NOROLE);
        assert_eq!(Permissions::ZERO._pad, 0);
        assert_eq!(Permissions::ZERO._reserved, [0; 2]);
    }

    #[test]
    fn all_is_full_caps_under_bootstrap() {
        assert_eq!(Permissions::ALL.perms, class::ALL);
        assert_eq!(Permissions::ALL.allowed_perms, class::ALL);
        assert_eq!(Permissions::ALL.role, role::BOOTSTRAP);
    }

    #[test]
    fn new_clamps_perms_to_allowed_perms() {
        let p = Permissions::new(class::ALL, class::STDIO | class::SCHED, role::WORKER);
        assert_eq!(p.perms, class::STDIO | class::SCHED);
        assert_eq!(p.allowed_perms, class::STDIO | class::SCHED);
        assert_eq!(p.role, role::WORKER);
    }

    /// Every defined sysno must map to a non-zero class. Catches new
    /// syscall additions that forgot to extend `class_for`.
    #[test]
    fn every_known_syscall_has_a_class() {
        use crate::syscall::*;
        let all = [
            EXIT,
            SERIAL_PRINT,
            SLEEP_MS,
            CONSOLE_WRITE,
            READ_STDIN,
            SET_AFFINITY,
            GET_AFFINITY,
            GET_HART_ID,
            GET_MICROS,
            MMAP,
            CREATE_NETCH,
            CLOSE_HANDLE,
            CREATE_PROCESS,
            NC_YIELD,
            QUERY_STATS,
            QUERY_SYSCALL_STATS,
            CREATE_PROCESS_EX,
            ARGV_ENVP,
            CREATE_THREAD,
            GETPID,
            GETTID,
            WAIT_PID,
            FUTEX_WAIT,
            FUTEX_WAKE,
            FS_OPEN,
            FS_READ,
            FS_STAT,
        ];
        for s in all {
            let cls = Permissions::class_for(s);
            assert!(cls != 0, "sysno {s} has no class — extend class_for");
            assert!(
                cls.is_power_of_two() || cls == 0,
                "sysno {s} maps to multi-bit class {cls:#x} — that's fine, but update this test if intentional"
            );
        }
    }

    #[test]
    fn unknown_syscall_has_no_class() {
        assert_eq!(Permissions::class_for(usize::MAX), 0);
        assert_eq!(Permissions::class_for(99999), 0);
    }

    #[test]
    fn allows_respects_perms_mask() {
        let p = Permissions::new(class::STDIO | class::FS_RO, class::ALL, role::FS_TOOL);
        assert!(p.allows(crate::syscall::SERIAL_PRINT));
        assert!(p.allows(crate::syscall::FS_OPEN));
        assert!(!p.allows(crate::syscall::CREATE_NETCH));
        assert!(!p.allows(crate::syscall::MMAP));
    }

    #[test]
    fn allows_rejects_unknown_syscall() {
        let p = Permissions::ALL;
        // Even with full caps, an unmapped sysno isn't allowed —
        // class_for returns 0, and `0 != 0` short-circuits the check.
        assert!(!p.allows(usize::MAX));
        assert!(!p.allows(99999));
    }

    #[test]
    fn pledge_only_narrows_perms() {
        let start = Permissions::ALL;
        let narrow = start.pledge(class::STDIO | class::VMEM, class::ALL);
        assert_eq!(narrow.perms, class::STDIO | class::VMEM);
        assert_eq!(narrow.allowed_perms, class::ALL);
        assert_eq!(narrow.role, start.role);
    }

    #[test]
    fn pledge_can_narrow_allowed_perms() {
        let start = Permissions::ALL;
        let narrow = start.pledge(class::ALL, class::STDIO);
        assert_eq!(narrow.allowed_perms, class::STDIO);
        // perms gets clamped to new allowed.
        assert_eq!(narrow.perms, class::STDIO);
    }

    #[test]
    fn pledge_cannot_expand_perms() {
        let p = Permissions::new(class::STDIO, class::STDIO | class::VMEM, role::SHELL);
        // Asking for ALL doesn't grow the mask — only intersection.
        let q = p.pledge(class::ALL, class::ALL);
        assert_eq!(q.perms, class::STDIO);
        assert_eq!(q.allowed_perms, class::STDIO | class::VMEM);
    }

    #[test]
    fn pledge_cannot_expand_allowed_perms() {
        let p = Permissions::new(class::STDIO, class::STDIO, role::WORKER);
        let q = p.pledge(class::ALL, class::ALL);
        assert_eq!(q.allowed_perms, class::STDIO);
    }

    #[test]
    fn pledge_is_idempotent() {
        let p = Permissions::new(class::STDIO | class::VMEM, class::ALL, role::SHELL);
        let q = p.pledge(class::STDIO, class::STDIO | class::VMEM);
        let r = q.pledge(class::STDIO, class::STDIO | class::VMEM);
        assert_eq!(q, r);
    }

    #[test]
    fn pledge_preserves_role_and_reserved() {
        let p = Permissions {
            perms: class::ALL,
            allowed_perms: class::ALL,
            role: role::SERVICE,
            _pad: 0,
            _reserved: [0xAA, 0xBB], // synthetic non-zero — we don't validate here
        };
        let q = p.pledge(class::STDIO, class::STDIO);
        assert_eq!(q.role, role::SERVICE);
        assert_eq!(q._reserved, [0xAA, 0xBB]);
    }

    #[test]
    fn pledge_maintains_perms_subset_allowed_invariant() {
        // Even if the caller's two masks would individually let perms
        // exceed allowed, the result clamps perms ⊆ allowed.
        let p = Permissions::new(class::ALL, class::ALL, role::BOOTSTRAP);
        let q = p.pledge(class::ALL, class::STDIO);
        assert_eq!(q.perms & !q.allowed_perms, 0);
    }

    /// Every role ID below COUNT should be distinct.
    #[test]
    fn role_constants_dense_and_distinct() {
        let all = [
            role::NOROLE,
            role::BOOTSTRAP,
            role::LOADER,
            role::SHELL,
            role::NET_CLIENT,
            role::FS_TOOL,
            role::WORKER,
            role::SERVICE,
        ];
        assert_eq!(all.len(), role::COUNT);
        let mut seen = [false; role::COUNT];
        for r in all {
            let i = r as usize;
            assert!(i < role::COUNT, "role id {i} >= COUNT");
            assert!(!seen[i], "role id {i} repeated");
            seen[i] = true;
        }
    }

    #[test]
    fn struct_layout_is_pinned() {
        // Wire-shape pin: perms(8) + allowed_perms(8) + role(4) +
        // _pad(4) + _reserved[2](16) = 40 bytes, 8-byte aligned.
        // If this changes, the kernel-side Process layout and any
        // serialization need re-checking.
        assert_eq!(core::mem::size_of::<Permissions>(), 40);
        assert_eq!(core::mem::align_of::<Permissions>(), 8);
    }
}
