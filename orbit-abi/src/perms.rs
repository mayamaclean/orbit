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
//! # ABI shape vs kernel-internal type
//!
//! [`Permissions`] is `#[repr(C)]` and 40 bytes — the on-the-wire shape
//! the syscall boundary deserializes into / out of. The fields stay
//! plain `u64` so the layout is layout-stable across user/kernel.
//!
//! Kernel-internal code should *not* operate on the raw `u64` fields.
//! Instead it works in terms of the [`ClassMask`] newtype, which has
//! no widening operations — only [`ClassMask::narrow`] (intersection)
//! and [`ClassMask::contains`]. That eliminates `process.perms |= bit`
//! style accidents at the type level: code that only holds a
//! `ClassMask` literally cannot construct a wider mask.
//!
//! Wide masks are still needed at static-init sites (the per-role
//! defaults, `class::ALL`). Those go through
//! [`ClassMask::from_raw`] + the raw bit constants in [`class::raw`].
//! Both are public — there's no way to make them otherwise without
//! breaking the orbit-abi/orbit-core boundary — but the discipline
//! is "the only places that bitor raw u64 bits are static
//! initializers," and `grep from_raw` makes that property auditable.
//!
//! The `_pad` and `_reserved` tail are part of the wire shape — they
//! exist so future axes (MLS level, label fingerprint) can be added
//! without renumbering or growing the struct on disk. New code MUST
//! initialize them to zero.

use crate::syscall;

/// A typed view onto the syscall-class bitmask. `repr(transparent)`
/// over `u64`, so it costs nothing at runtime.
///
/// Operations: [`narrow`](Self::narrow) (intersection),
/// [`contains`](Self::contains), [`is_empty`](Self::is_empty),
/// [`raw`](Self::raw). There is deliberately **no `BitOr` /
/// `BitOrAssign`** — the only way a held `ClassMask` value changes
/// is to be replaced by a narrower one. This makes "permission
/// widening" un-spellable in code that holds the kernel-internal
/// type.
///
/// Construction:
/// - At runtime, derive a narrower mask from a wider one via
///   `narrow`.
/// - At static-init, [`ClassMask::from_raw`] takes a raw `u64` bit
///   pattern. Used by [`class`] (per-class constants), `class::ALL`,
///   and the per-role defaults in `orbit-core::roles`. Grep
///   `from_raw` to find every wide-mask construction site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ClassMask(u64);

impl ClassMask {
    /// The empty mask — no classes set. Default for "I haven't
    /// granted anything yet" states.
    pub const EMPTY: Self = Self(0);

    /// Wrap a raw `u64` bit pattern. The escape hatch for static
    /// initializers and for materialising masks read off the wire.
    /// Runtime kernel code that wants to derive a narrower mask
    /// should prefer [`narrow`](Self::narrow) instead — it can't
    /// accidentally widen.
    pub const fn from_raw(bits: u64) -> Self {
        Self(bits)
    }

    /// Underlying `u64` view. Used by serialization at the syscall
    /// boundary (writing into `Permissions.perms`) and by the rare
    /// site that must hand a mask to a non-typed API.
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Intersection — drop any bit not present in `other`.
    /// Monotonic: `result.raw() <= self.raw()` (bitwise). The
    /// load-bearing operation for `pledge` and `derive_child_perms`.
    pub const fn narrow(self, other: ClassMask) -> Self {
        Self(self.0 & other.0)
    }

    /// Does `self` contain every bit set in `other`? Used by the
    /// dispatch gate to test "are all required classes present?"
    /// — typically `process_perms.contains(class_for(sysno))`.
    pub const fn contains(self, other: ClassMask) -> bool {
        (self.0 & other.0) == other.0
    }

    /// True iff no bits are set. Used by `class_for` callers to
    /// distinguish "no class assigned" (unknown sysno) from a real
    /// permission check.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Syscall-class bitmask values. Bits are part of the user/kernel ABI
/// — append-only, never reorder, never recycle a freed bit. See
/// [`Permissions::class_for`] for which class each syscall maps to.
///
/// Each `class::*` constant is a `ClassMask`. The underlying bit
/// patterns live in the [`class::raw`] submodule so that wide-mask
/// composition (`class::ALL`, per-role defaults) can express
/// "STDIO | NETCH | VMEM" via plain `u64` bitor — there's no
/// `ClassMask` widening operation, by design.
pub mod class {
    use super::ClassMask;

    /// Raw `u64` bit patterns underlying the [`class`] constants.
    /// Used at static-init to compose wide masks via plain bitor;
    /// runtime code should reach for [`ClassMask::narrow`] instead
    /// (which can only narrow) or use the typed [`class`] constants.
    /// The "wide-construction" surface is exactly `from_raw` + this
    /// module — `grep raw::` finds every static composition.
    pub mod raw {
        pub const STDIO: u64 = 1 << 0;
        pub const PROC_LIFE: u64 = 1 << 1;
        pub const PROC_SPAWN: u64 = 1 << 2;
        pub const SCHED: u64 = 1 << 3;
        pub const VMEM: u64 = 1 << 4;
        pub const VMEM_SHARED: u64 = 1 << 5;
        pub const NETCH: u64 = 1 << 6;
        pub const FS_RO: u64 = 1 << 7;
        pub const FUTEX: u64 = 1 << 8;
        pub const STATS: u64 = 1 << 9;
        pub const PLEDGE: u64 = 1 << 10;

        /// Union of every defined class's raw bit. Update when adding
        /// a class — the `all_is_union_of_classes` test pins this.
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

    /// `serial_print`, `console_write`, `read_stdin`, `get_micros`,
    /// `close_handle`. The last is in stdio rather than its own class
    /// because closing handles is a destructor — denying it strands
    /// resources, never a useful sandboxing tool.
    pub const STDIO: ClassMask = ClassMask::from_raw(raw::STDIO);

    /// `exit`, `getpid`, `gettid`, `wait_pid`, `argv_envp`. The
    /// "what process am I" surface — a process that can't exit cleanly
    /// is a worse outcome than one that can.
    pub const PROC_LIFE: ClassMask = ClassMask::from_raw(raw::PROC_LIFE);

    /// `create_process`, `create_process_ex`, `create_process_v2`,
    /// `create_thread`. Coarse — a future split into PROC_SPAWN_PROC
    /// vs PROC_SPAWN_THREAD is foreseeable but not motivated yet.
    pub const PROC_SPAWN: ClassMask = ClassMask::from_raw(raw::PROC_SPAWN);

    /// `sleep_ms`, `set_affinity`, `get_affinity`, `get_hart_id`,
    /// `nc_yield`. Anything that hands the scheduler a parking
    /// or migration request.
    pub const SCHED: ClassMask = ClassMask::from_raw(raw::SCHED);

    /// `mmap` with `share_with_kernel = false`. Private user pages,
    /// no kernel alias.
    pub const VMEM: ClassMask = ClassMask::from_raw(raw::VMEM);

    /// `mmap` with `share_with_kernel = true`. Pages with a long-lived
    /// kernel KDMAP alias — used by NetChannel rings and any future
    /// kernel-readable region. Separate class because the W^X / alias
    /// hazard is materially different from VMEM.
    pub const VMEM_SHARED: ClassMask = ClassMask::from_raw(raw::VMEM_SHARED);

    /// `create_netch`. NetChannel allocation; gates network reach
    /// orthogonally to VMEM_SHARED (which only enables the *backing*
    /// pool — without NETCH you can have shared memory but no socket).
    pub const NETCH: ClassMask = ClassMask::from_raw(raw::NETCH);

    /// `fs_open` (read-only, today's only mode), `fs_read`, `fs_stat`,
    /// future readdir. Becomes `FS_RW` once writes land — at which
    /// point this bit narrows to "read paths only" and a new bit
    /// covers the writing surface.
    pub const FS_RO: ClassMask = ClassMask::from_raw(raw::FS_RO);

    /// `futex_wait`, `futex_wake`. Cheap to deny in single-threaded
    /// processes; required for any threaded workload.
    pub const FUTEX: ClassMask = ClassMask::from_raw(raw::FUTEX);

    /// `query_stats`, `query_syscall_stats`. Observability.
    pub const STATS: ClassMask = ClassMask::from_raw(raw::STATS);

    /// `pledge` (the syscall itself). Default-on for the `BOOTSTRAP`
    /// and `LOADER` roles so they can self-narrow; a process that
    /// pledges this away can never re-narrow itself afterwards (which
    /// is fine — the typical caller does it once at startup).
    pub const PLEDGE: ClassMask = ClassMask::from_raw(raw::PLEDGE);

    /// Union of every defined class. Equal to a process's `perms` in
    /// the `BOOTSTRAP` role. The composition lives in [`raw::ALL`]
    /// (plain `u64` bitor), wrapped here once.
    pub const ALL: ClassMask = ClassMask::from_raw(raw::ALL);
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

    /// Migration default: legacy `CREATE_PROCESS` and
    /// `CREATE_PROCESS_EX` callers stamp every spawned `Process`
    /// with `BOOTSTRAP` so they keep working until the loader
    /// switches to `CREATE_PROCESS_V2` with explicit role
    /// assignment (kmain → loader → console). Survives long-term
    /// as a rescue/test sentinel — transitions to `LOADER` and
    /// `SHELL`, wide caps so a single self-`pledge` can shed
    /// anything. See `ROLES[BOOTSTRAP]` in `orbit-core::roles`.
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

    /// Sentinel passed as `target_role` to `create_process_v2` meaning
    /// "spawn a child with my own role and perms exactly — no role
    /// transition, no perm narrowing." Outside the registry range, so
    /// `role_def(INHERIT)` returns `None`; `check_transition`
    /// short-circuits and `derive_child_perms` returns the parent's
    /// `Permissions` verbatim.
    ///
    /// This is the escape hatch for callers that want fork-shaped
    /// semantics rather than the role-aware downgrade lattice — the
    /// shape `std::process::Command` and any other libc-style spawn
    /// surface needs. Real role-aware spawn paths (the loader chain,
    /// service supervisors) pass a concrete `RoleId`.
    ///
    /// **TODO (role-metadata milestone):** every kernel/userland call
    /// site that currently passes `INHERIT` is a placeholder. Once
    /// per-binary role metadata exists (signed manifest, ELF note,
    /// or tarfs sidecar — TBD), each `INHERIT` should be replaced
    /// with the concrete `target_role` derived from that metadata so
    /// the role-transition gate runs for real. Search this codebase
    /// for `role::INHERIT` to find the full list of sites that need
    /// promotion.
    pub const INHERIT: RoleId = u32::MAX;
}

/// Process-wide permission state. Lives on `Process` in the kernel; user
/// code constructs instances to pass to `pledge` and `create_process_v2`.
///
/// **Wire shape, not the kernel-internal type.** The `perms` and
/// `allowed_perms` fields are plain `u64` for layout stability across
/// the syscall boundary. Kernel code that operates on these masks
/// should reach for [`Permissions::perms_mask`] /
/// [`Permissions::allowed_perms_mask`] which return the typed
/// [`ClassMask`] view — that's the surface where widening is
/// un-spellable.
///
/// `perms` and `allowed_perms` are *independent axes*. `perms` is what
/// THIS process can do; `allowed_perms` is the cap on what direct
/// children can be granted at `create_process_v2`. They are not in a
/// subset relationship — a process can hold a permission for itself
/// without being able to pass it down (a leaf role like `NET_CLIENT`),
/// or hold an `allowed_perms` cap covering classes it doesn't itself
/// run with (a `SHELL` that grants NETCH to children but doesn't speak
/// network itself). Mirrors OpenBSD's `pledge(promises, execpromises)`
/// where the two masks are similarly orthogonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Permissions {
    /// Effective syscall-class bitmask for this process. Wire shape;
    /// kernel-internal access via [`Permissions::perms_mask`].
    pub perms: u64,
    /// Cap on what direct children may receive at create_process_v2.
    /// Independent of `perms` (see struct docs). Wire shape;
    /// kernel-internal access via [`Permissions::allowed_perms_mask`].
    pub allowed_perms: u64,
    /// Role identifier — index into the kernel role registry. `0` =
    /// [`role::NOROLE`] (terminal, no outbound transitions).
    pub role: u32,
    /// Reserved padding; keeps the struct's 8-byte alignment explicit.
    /// Must be zero.
    pub _pad: u32,
    /// Reserved for future axes (MLS level, label fingerprint, RBAC
    /// supplemental groups). Must be zero today. Once a future axis
    /// starts using a slot, the spawn boundary will reject non-zero
    /// `_reserved` from old clients so they can't silently downgrade
    /// against a newer kernel; `pledge` and `Permissions::new` preserve
    /// it verbatim today since there's nothing to validate yet.
    pub _reserved: [u64; 2],
}

/// Two-axis mask request carried across function boundaries that
/// touch both `perms` and `allowed_perms`. Named-field struct so
/// callers can't silently swap the two args — `perms` is *always*
/// "what I (or this child) can do," `allowed_perms` is *always*
/// "what I (or this child) can pass down."
///
/// Used by [`Permissions::pledge`] and (in `orbit-core::roles`) by
/// `derive_child_perms`. Cheap — `Copy + repr(C)` aligned to 8 bytes,
/// 16 bytes total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PermsRequest {
    /// Requested effective mask. Pledge intersects against
    /// `self.perms_mask()`; spawn intersects against role default.
    pub perms: ClassMask,
    /// Requested cap mask. Independent of `perms` — see [`Permissions`]
    /// struct docs.
    pub allowed_perms: ClassMask,
}

impl PermsRequest {
    /// Request that opens nothing — a `narrow`-with-this is a
    /// strip-everything pledge. Useful for "give me the minimum cap"
    /// patterns.
    pub const EMPTY: Self = Self {
        perms: ClassMask::EMPTY,
        allowed_perms: ClassMask::EMPTY,
    };

    /// Request both masks set to `class::ALL`. Combined with `narrow`
    /// in `pledge`, this is a no-op pledge — useful as a default
    /// before the caller has decided what to drop.
    pub const ALL: Self = Self {
        perms: class::ALL,
        allowed_perms: class::ALL,
    };
}

/// `create_process_v2(args: *const CreateProcessV2Args) -> pid | -errno`
/// argument bundle. Lives in user memory because the call carries
/// more fields than the `a1..a7` register window can hold
/// comfortably; the kernel reads via the standard boundary
/// deserializer (one bounded copy on entry, no further user reads).
///
/// Fields are append-only — never reorder, never repurpose, never
/// shrink. `_pad` exists so the struct's natural 8-byte alignment
/// is explicit; future role-axis additions can repurpose the slot
/// before extending the struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CreateProcessV2Args {
    /// User VA of the child's ELF blob. Validated by the syscall
    /// boundary against `user_range_ok` before the manager copies.
    pub elf_vaddr: usize,
    /// Length of the ELF blob in bytes.
    pub elf_len: usize,
    /// Cap mask the child's first thread is constructed with —
    /// sentinel `0` defers to the kernel's all-harts default. Same
    /// shape as `CREATE_PROCESS`'s `allowed_affinity`.
    pub allowed_affinity: u64,
    /// Initial affinity for the child's first thread. Sentinel `0`
    /// → resolved `allowed_affinity`.
    pub affinity: u64,
    /// Role to install on the child. Validated against the parent's
    /// transition bitset by the kernel-side gate. On `Err(_)` the
    /// gate logs a `RoleDeny` audit event, bumps the parent's
    /// `role_denials` counter, and the syscall returns `-EPERM` —
    /// no child is created.
    pub target_role: role::RoleId,
    /// Reserved — must be zero. Pinning the alignment without an
    /// implicit padding slot keeps the wire shape grep-able.
    pub _pad: u32,
    /// Caller-requested narrowing of the child's effective perms.
    /// Intersected with the role default + parent's allowed_perms.
    pub request_perms: u64,
    /// Caller-requested narrowing of the child's allowed_perms cap.
    /// Independent axis — see [`Permissions`] struct docs.
    pub request_allowed_perms: u64,
    /// Optional cwd override for the child. `cwd_vaddr == 0` (or
    /// `cwd_len == 0`) means "inherit the parent's cwd verbatim",
    /// matching POSIX fork+exec semantics; non-zero gives the child
    /// a starting cwd of the bytes at `[cwd_vaddr, cwd_vaddr+cwd_len)`,
    /// which the kernel validates as absolute UTF-8 + existing dir
    /// before installing.
    pub cwd_vaddr: usize,
    /// Length of the cwd override in bytes; ignored when `cwd_vaddr
    /// == 0`. Capped at 4 KiB; longer requests yield `ENAMETOOLONG`.
    pub cwd_len: usize,
    /// User VA of an argv blob in [`crate::argv`] wire format, or `0`
    /// for a child with no argv. Capped at one page. Mirrors the
    /// `argv_vaddr` arg of the legacy `CREATE_PROCESS_EX` syscall —
    /// V2 carries the field directly so it's a strict superset.
    pub argv_vaddr: usize,
    /// Length of the argv blob in bytes; ignored when `argv_vaddr ==
    /// 0`. Must fit in one page.
    pub argv_len: usize,
    /// User VA of a page-aligned envp blob in [`crate::envp`] wire
    /// format, or `0` for "no envp." The kernel always reads exactly
    /// one page from this VA; callers must pass a page-resident,
    /// page-sized buffer (zero-padded past the packed bytes).
    pub envp_vaddr: usize,
    /// Stdout-routing override for the child. Determines where
    /// `console_write` from the new process lands.
    ///
    /// - `0` — child writes go to its own scrollback pane (today's
    ///   behavior; legacy `CREATE_PROCESS` / `CREATE_PROCESS_EX` both
    ///   resolve to this).
    /// - `1` — child writes route to the *parent's* pane. Lets a
    ///   shell-style caller (`console`) run a foreground child and
    ///   have its output land inline in the caller's scrollback,
    ///   without an fd table or pipes. Defined as parent-routed
    ///   rather than fd-1 redirected so the field has a meaning even
    ///   while fds (option 2) don't exist.
    /// - other values — reserved. The kernel rejects with `-EINVAL`.
    ///
    /// When fds land this slot can carry an fd index instead, with
    /// `0`/`1` retaining the current meaning as compat shims.
    pub stdout_capture: u32,
    /// Reserved — must be zero. Pads `stdout_capture` up to the next
    /// 8-byte boundary so the struct's natural alignment is explicit.
    pub _pad2: u32,
}

impl CreateProcessV2Args {
    /// Typed view of the request masks, for the kernel-side
    /// `derive_child_perms` clamp.
    pub const fn request(&self) -> PermsRequest {
        PermsRequest {
            perms: ClassMask::from_raw(self.request_perms),
            allowed_perms: ClassMask::from_raw(self.request_allowed_perms),
        }
    }
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

    /// `BOOTSTRAP` role with every class set on both `perms` and
    /// `allowed_perms`. Migration default for legacy
    /// `CREATE_PROCESS` / `CREATE_PROCESS_EX` callers and the
    /// shape used in host tests; `CREATE_PROCESS_V2` resolves
    /// `default_perms` / `default_allowed` through the registry
    /// (see `orbit-core::roles`) so the result matches the
    /// chosen role's policy.
    pub const ALL: Self = Self {
        perms: class::raw::ALL,
        allowed_perms: class::raw::ALL,
        role: role::BOOTSTRAP,
        _pad: 0,
        _reserved: [0; 2],
    };

    pub const LOADER: Self = Self {
        perms: class::raw::ALL,
        allowed_perms: class::raw::ALL,
        role: role::LOADER,
        _pad: 0,
        _reserved: [0; 2],
    };

    /// Construct from raw `u64` masks — the boundary-deserializer
    /// shape, used by code reading a `Permissions` off the wire. No
    /// clamping (the two axes are independent; see struct docs); the
    /// spawn path validates instead of clamping.
    ///
    /// **Bypass hazard.** This constructor is `pub` because boundary
    /// code reading a wire `Permissions` needs it. Kernel-side, it
    /// would also let a misbehaved caller fabricate a `Permissions`
    /// with no gate having run — for example,
    /// `process.permissions = Permissions::new(class::raw::ALL, …)`
    /// installs full perms with no `derive_child_perms` invocation.
    /// The mitigation is *not* visibility on this constructor
    /// (boundary code legitimately needs it), but the `Process`
    /// API surface: the only setter is
    /// `Process::install_permissions(Permissions)`, called only by
    /// the `create_process_v2` handler with a witness-derived
    /// value. Adding a wider setter would be a security-class
    /// regression — reviewers police direct `install_permissions`
    /// calls as part of any spawn-path change.
    pub const fn new(perms: u64, allowed_perms: u64, role: role::RoleId) -> Self {
        Self {
            perms,
            allowed_perms,
            role,
            _pad: 0,
            _reserved: [0; 2],
        }
    }

    /// Construct from typed [`ClassMask`]s — the kernel-internal
    /// constructor used by `derive_child_perms` and by static
    /// initializers that want the typed surface. Same bypass-hazard
    /// caveats as [`new`](Self::new): the `pub` constructor doesn't
    /// itself enforce gating, and the kernel-side `Process` API
    /// surface is what makes the witness path the only path.
    pub const fn from_masks(
        perms: ClassMask,
        allowed_perms: ClassMask,
        role: role::RoleId,
    ) -> Self {
        Self::new(perms.raw(), allowed_perms.raw(), role)
    }

    /// Typed view onto `perms`. Returns a [`ClassMask`] which has no
    /// widening operations — code that holds the result can only
    /// derive narrower masks via `narrow`.
    pub const fn perms_mask(&self) -> ClassMask {
        ClassMask::from_raw(self.perms)
    }

    /// Typed view onto `allowed_perms`. See [`perms_mask`](Self::perms_mask).
    pub const fn allowed_perms_mask(&self) -> ClassMask {
        ClassMask::from_raw(self.allowed_perms)
    }

    /// Permission class associated with `sysno`. Returns
    /// [`ClassMask::EMPTY`] for unknown / unmapped syscalls; the
    /// dispatch site treats `EMPTY` as "no class — never allowed,"
    /// which also matches the desired behaviour for typo'd syscall
    /// numbers.
    ///
    /// Some syscalls (`mmap`, future `fs_open` with write flags) have
    /// an argument-conditional class — the coarse class returned here
    /// is the *minimum* needed to enter the handler, and the handler
    /// performs an extra check on the args. `mmap` returns `VMEM`;
    /// the handler upgrades to also requiring `VMEM_SHARED` when
    /// `share_with_kernel = true`.
    pub const fn class_for(sysno: usize) -> ClassMask {
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
            syscall::PLEDGE => class::PLEDGE,
            syscall::MMAP => class::VMEM,
            syscall::CREATE_NETCH => class::NETCH,
            syscall::CLOSE_HANDLE => class::STDIO,
            syscall::CREATE_PROCESS => class::PROC_SPAWN,
            syscall::NC_YIELD => class::SCHED,
            syscall::QUERY_STATS => class::STATS,
            syscall::QUERY_SYSCALL_STATS => class::STATS,
            syscall::CREATE_PROCESS_EX => class::PROC_SPAWN,
            syscall::ARGV_ENVP => class::PROC_LIFE,
            syscall::CREATE_PROCESS_V2 => class::PROC_SPAWN,
            syscall::QUERY_DENIAL_LOG => class::STATS,
            syscall::CREATE_THREAD => class::PROC_SPAWN,
            syscall::GETPID => class::PROC_LIFE,
            syscall::GETTID => class::PROC_LIFE,
            syscall::WAIT_PID => class::PROC_LIFE,
            syscall::FUTEX_WAIT => class::FUTEX,
            syscall::FUTEX_WAKE => class::FUTEX,
            syscall::FS_OPEN => class::FS_RO,
            syscall::FS_READ => class::FS_RO,
            syscall::FS_STAT => class::FS_RO,
            syscall::FS_READDIR => class::FS_RO,
            syscall::FS_SEEK => class::FS_RO,
            syscall::FS_FSTAT => class::FS_RO,
            // chdir validates the target dir exists in the active fs
            // before mutating cwd, so it needs FS_RO. getcwd is pure
            // process-state read, like getpid.
            syscall::CHDIR => class::FS_RO,
            syscall::GETCWD => class::PROC_LIFE,
            _ => ClassMask::EMPTY,
        }
    }

    /// Does this process's `perms` permit `sysno`? Unknown sysnos are
    /// rejected (class is `EMPTY`, `contains(EMPTY)` is `true`-but-
    /// vacuous so we short-circuit on `is_empty`). A class with
    /// multiple bits would require all of them; today every class is
    /// a single bit so this collapses to a single `&`.
    ///
    /// **Use [`can_call`](Self::can_call) instead at production
    /// dispatch sites.** This bare check doesn't push a
    /// [`DenialEvent`](crate::denial::DenialEvent) on denial, so the
    /// audit log won't show up in `query_denial_log` or the
    /// `perm_denials` counter. `allows` is for tests and for the
    /// rare context where the caller already knows a denial
    /// shouldn't be recorded (e.g. a speculative check).
    pub const fn allows(&self, sysno: usize) -> bool {
        let cls = Self::class_for(sysno);
        if cls.is_empty() {
            return false;
        }
        self.perms_mask().contains(cls)
    }

    /// Dispatch-gate entrypoint. Equivalent to
    /// [`allows`](Self::allows) plus a [`DenialEvent::PermDeny`]
    /// push to `sink` on denial. Returns `true` if the syscall is
    /// allowed; `false` (with the audit event already pushed) if
    /// denied — the caller is expected to short-circuit with
    /// `-EPERM`.
    ///
    /// "You can't gate without recording" is type-enforced here:
    /// the gate function takes the sink by mutable reference, so
    /// any code path that wants to invoke the gate must have a
    /// sink in scope. The kernel's production sink is the bounded
    /// `DenialRing` in `orbit-core::denial_ring`; tests pass a
    /// Vec-backed sink (or `()` for "no recording, I'm checking
    /// idempotently").
    ///
    /// `ctx` carries the caller-side metadata (pid/tid/time_ticks)
    /// that lands in the event's matching fields. The dispatch site
    /// reads it from the hart context once per syscall and threads
    /// it through.
    pub fn can_call(
        &self,
        sysno: usize,
        ctx: crate::denial::GateContext,
        sink: &mut impl crate::denial::DenialSink,
    ) -> bool {
        if self.allows(sysno) {
            return true;
        }
        sink.push(crate::denial::DenialEvent::PermDeny {
            required_class: Self::class_for(sysno).raw(),
            perms: self.perms,
            time_ticks: ctx.time_ticks,
            tid: ctx.tid,
            sysno: sysno as u32,
            source_role: self.role,
            pid: ctx.pid,
        });
        false
    }

    /// Pledge-style narrowing. Each axis is intersected with its
    /// corresponding field of [`PermsRequest`] independently — bits
    /// not present in `request.*` are dropped, bits not present in
    /// `self.*` cannot be added back. Always succeeds; passing a
    /// broader mask is a no-op on those bits, mirroring OpenBSD's
    /// `pledge(promises, execpromises)` (silent clamp rather than
    /// EPERM on attempted-broaden).
    ///
    /// `role` and `_reserved` are preserved verbatim — pledge is
    /// permission narrowing, not a role change.
    pub const fn pledge(&self, request: PermsRequest) -> Self {
        Self {
            perms: self.perms_mask().narrow(request.perms).raw(),
            allowed_perms: self
                .allowed_perms_mask()
                .narrow(request.allowed_perms)
                .raw(),
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

    // Helper for tests that want to compose multi-class ClassMasks at
    // runtime (analogous to a u64 bitor). Lives only in tests so the
    // production widening surface stays "from_raw + raw::*".
    fn union(a: ClassMask, b: ClassMask) -> ClassMask {
        ClassMask::from_raw(a.raw() | b.raw())
    }

    /// Pin the union — class::ALL must equal the OR of every
    /// individual constant. If a new class is added, this test fails
    /// until ALL is updated, which is the point.
    #[test]
    fn all_is_union_of_classes() {
        let computed = ClassMask::from_raw(
            class::raw::STDIO
                | class::raw::PROC_LIFE
                | class::raw::PROC_SPAWN
                | class::raw::SCHED
                | class::raw::VMEM
                | class::raw::VMEM_SHARED
                | class::raw::NETCH
                | class::raw::FS_RO
                | class::raw::FUTEX
                | class::raw::STATS
                | class::raw::PLEDGE,
        );
        assert_eq!(class::ALL, computed);
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
            assert!(
                c.raw().is_power_of_two(),
                "class {:#x} is not a single bit",
                c.raw()
            );
        }
        let mut accum = ClassMask::EMPTY;
        for c in all {
            assert!(
                accum.narrow(c).is_empty(),
                "class {:#x} overlaps an earlier class",
                c.raw()
            );
            accum = union(accum, c);
        }
    }

    #[test]
    fn classmask_no_widening_via_narrow() {
        // `narrow` always produces a result with a subset of `self`'s bits.
        // Pinned so a future "convenience" `union` method on ClassMask
        // can't sneak in without updating this and the API docs.
        let wide = class::ALL;
        let narrowed = wide.narrow(class::STDIO);
        assert_eq!(narrowed, class::STDIO);
        // narrow with a wider mask returns self (intersection).
        let renarrowed = class::STDIO.narrow(class::ALL);
        assert_eq!(renarrowed, class::STDIO);
    }

    #[test]
    fn classmask_contains_works_for_subset_and_disjoint() {
        let stdio_and_netch = union(class::STDIO, class::NETCH);
        assert!(stdio_and_netch.contains(class::STDIO));
        assert!(stdio_and_netch.contains(class::NETCH));
        assert!(!stdio_and_netch.contains(class::VMEM));
        assert!(class::ALL.contains(stdio_and_netch));
    }

    #[test]
    fn zero_is_empty_and_norole() {
        assert_eq!(Permissions::ZERO.perms, 0);
        assert_eq!(Permissions::ZERO.allowed_perms, 0);
        assert_eq!(Permissions::ZERO.role, role::NOROLE);
        assert_eq!(Permissions::ZERO._pad, 0);
        assert_eq!(Permissions::ZERO._reserved, [0; 2]);
        assert!(Permissions::ZERO.perms_mask().is_empty());
        assert!(Permissions::ZERO.allowed_perms_mask().is_empty());
    }

    #[test]
    fn all_is_full_caps_under_bootstrap() {
        assert_eq!(Permissions::ALL.perms_mask(), class::ALL);
        assert_eq!(Permissions::ALL.allowed_perms_mask(), class::ALL);
        assert_eq!(Permissions::ALL.role, role::BOOTSTRAP);
    }

    #[test]
    fn from_masks_roundtrips_through_raw_fields() {
        let p = Permissions::from_masks(
            union(class::STDIO, class::NETCH),
            class::STDIO,
            role::NET_CLIENT,
        );
        assert_eq!(p.perms_mask(), union(class::STDIO, class::NETCH));
        assert_eq!(p.allowed_perms_mask(), class::STDIO);
        assert_eq!(p.role, role::NET_CLIENT);
        // Wire view stays in sync — boundary serializers see the same bits.
        assert_eq!(p.perms, class::raw::STDIO | class::raw::NETCH);
        assert_eq!(p.allowed_perms, class::raw::STDIO);
    }

    #[test]
    fn new_does_not_clamp_axes_against_each_other() {
        // The two axes are independent — `new` just stores them.
        // A NET_CLIENT-shaped pair (own NETCH, no NETCH for children)
        // should round-trip exactly.
        let p = Permissions::new(
            class::raw::STDIO | class::raw::NETCH,
            class::raw::STDIO,
            role::NET_CLIENT,
        );
        assert_eq!(p.perms_mask(), union(class::STDIO, class::NETCH));
        assert_eq!(p.allowed_perms_mask(), class::STDIO);
        assert_eq!(p.role, role::NET_CLIENT);
    }

    /// Every defined sysno must map to a non-empty class. Catches new
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
            PLEDGE,
            MMAP,
            CREATE_NETCH,
            CLOSE_HANDLE,
            CREATE_PROCESS,
            NC_YIELD,
            QUERY_STATS,
            QUERY_SYSCALL_STATS,
            CREATE_PROCESS_EX,
            ARGV_ENVP,
            CREATE_PROCESS_V2,
            QUERY_DENIAL_LOG,
            CREATE_THREAD,
            GETPID,
            GETTID,
            WAIT_PID,
            FUTEX_WAIT,
            FUTEX_WAKE,
            FS_OPEN,
            FS_READ,
            FS_STAT,
            FS_READDIR,
            FS_SEEK,
            FS_FSTAT,
            CHDIR,
            GETCWD,
        ];
        for s in all {
            let cls = Permissions::class_for(s);
            assert!(!cls.is_empty(), "sysno {s} has no class — extend class_for");
            assert!(
                cls.raw().is_power_of_two(),
                "sysno {s} maps to multi-bit class {:#x} — that's fine, but update this test if intentional",
                cls.raw()
            );
        }
    }

    #[test]
    fn unknown_syscall_has_empty_class() {
        assert!(Permissions::class_for(usize::MAX).is_empty());
        assert!(Permissions::class_for(99999).is_empty());
    }

    #[test]
    fn allows_respects_perms_mask() {
        let p =
            Permissions::from_masks(union(class::STDIO, class::FS_RO), class::ALL, role::FS_TOOL);
        assert!(p.allows(crate::syscall::SERIAL_PRINT));
        assert!(p.allows(crate::syscall::FS_OPEN));
        assert!(!p.allows(crate::syscall::CREATE_NETCH));
        assert!(!p.allows(crate::syscall::MMAP));
    }

    #[test]
    fn allows_rejects_unknown_syscall() {
        let p = Permissions::ALL;
        // Even with full caps, an unmapped sysno isn't allowed —
        // class_for returns EMPTY, and `is_empty()` short-circuits.
        assert!(!p.allows(usize::MAX));
        assert!(!p.allows(99999));
    }

    #[test]
    fn pledge_only_narrows_perms() {
        let start = Permissions::ALL;
        let narrow = start.pledge(PermsRequest {
            perms: union(class::STDIO, class::VMEM),
            allowed_perms: class::ALL,
        });
        assert_eq!(narrow.perms_mask(), union(class::STDIO, class::VMEM));
        assert_eq!(narrow.allowed_perms_mask(), class::ALL);
        assert_eq!(narrow.role, start.role);
    }

    #[test]
    fn pledge_narrows_allowed_independently_of_perms() {
        // Axes are independent: narrowing allowed_perms doesn't touch perms.
        // This is exactly the "I keep using the network but my children
        // can't" pattern — the parent stays capable, the cap shrinks.
        let start = Permissions::ALL;
        let narrow = start.pledge(PermsRequest {
            perms: class::ALL,
            allowed_perms: class::STDIO,
        });
        assert_eq!(narrow.allowed_perms_mask(), class::STDIO);
        assert_eq!(narrow.perms_mask(), class::ALL);
    }

    #[test]
    fn pledge_cannot_expand_perms() {
        let p =
            Permissions::from_masks(class::STDIO, union(class::STDIO, class::VMEM), role::SHELL);
        // Asking for ALL doesn't grow the mask — only intersection.
        let q = p.pledge(PermsRequest::ALL);
        assert_eq!(q.perms_mask(), class::STDIO);
        assert_eq!(q.allowed_perms_mask(), union(class::STDIO, class::VMEM));
    }

    #[test]
    fn pledge_cannot_expand_allowed_perms() {
        let p = Permissions::from_masks(class::STDIO, class::STDIO, role::WORKER);
        let q = p.pledge(PermsRequest::ALL);
        assert_eq!(q.allowed_perms_mask(), class::STDIO);
    }

    #[test]
    fn pledge_is_idempotent() {
        let p = Permissions::from_masks(union(class::STDIO, class::VMEM), class::ALL, role::SHELL);
        let req = PermsRequest {
            perms: class::STDIO,
            allowed_perms: union(class::STDIO, class::VMEM),
        };
        let q = p.pledge(req);
        let r = q.pledge(req);
        assert_eq!(q, r);
    }

    #[test]
    fn pledge_preserves_role_and_reserved() {
        let p = Permissions {
            perms: class::raw::ALL,
            allowed_perms: class::raw::ALL,
            role: role::SERVICE,
            _pad: 0,
            _reserved: [0xAA, 0xBB], // synthetic non-zero — we don't validate here
        };
        let q = p.pledge(PermsRequest {
            perms: class::STDIO,
            allowed_perms: class::STDIO,
        });
        assert_eq!(q.role, role::SERVICE);
        assert_eq!(q._reserved, [0xAA, 0xBB]);
    }

    #[test]
    fn pledge_can_produce_perms_with_bits_outside_allowed() {
        // The two axes are orthogonal. Narrowing allowed_perms below
        // perms is legal — exactly the "I keep my own caps but won't
        // pass them to children" pattern (NET_CLIENT etc).
        let p = Permissions::from_masks(union(class::STDIO, class::NETCH), class::ALL, role::SHELL);
        let q = p.pledge(PermsRequest {
            perms: class::ALL,
            allowed_perms: class::STDIO,
        });
        assert_eq!(q.perms_mask(), union(class::STDIO, class::NETCH));
        assert_eq!(q.allowed_perms_mask(), class::STDIO);
        // Bits in perms but not in allowed_perms — independence in action.
        assert_ne!(q.perms_mask().raw() & !q.allowed_perms_mask().raw(), 0);
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

    #[test]
    fn classmask_is_transparent_over_u64() {
        // repr(transparent) — same size and alignment as u64. Pinned
        // so the type is always free to round-trip through the wire
        // shape without bit-manipulation.
        assert_eq!(core::mem::size_of::<ClassMask>(), 8);
        assert_eq!(core::mem::align_of::<ClassMask>(), 8);
    }

    #[test]
    fn perms_request_layout_is_pinned() {
        // Two ClassMasks back-to-back, no padding.
        assert_eq!(core::mem::size_of::<PermsRequest>(), 16);
        assert_eq!(core::mem::align_of::<PermsRequest>(), 8);
    }

    #[test]
    fn pledge_to_zero_perms_yields_zero() {
        let p = Permissions::ALL;
        let q = p.pledge(PermsRequest {
            perms: ClassMask::EMPTY,
            allowed_perms: class::ALL,
        });
        assert!(q.perms_mask().is_empty());
        assert_eq!(q.allowed_perms_mask(), class::ALL);
    }

    #[test]
    fn pledge_to_zero_allowed_yields_zero_allowed() {
        let p = Permissions::ALL;
        let q = p.pledge(PermsRequest {
            perms: class::ALL,
            allowed_perms: ClassMask::EMPTY,
        });
        assert!(q.allowed_perms_mask().is_empty());
        assert_eq!(q.perms_mask(), class::ALL);
    }

    #[test]
    fn pledge_to_zero_both_yields_full_zero_masks() {
        let p = Permissions::ALL;
        let q = p.pledge(PermsRequest::EMPTY);
        assert!(q.perms_mask().is_empty());
        assert!(q.allowed_perms_mask().is_empty());
        // Role survives even when masks go to zero.
        assert_eq!(q.role, role::BOOTSTRAP);
    }

    #[test]
    fn zero_then_pledge_is_still_zero() {
        // Pledge can't add bits, so pledging ZERO with full masks stays
        // at ZERO — fail-safe state is sticky under any future pledge.
        let p = Permissions::ZERO;
        let q = p.pledge(PermsRequest::ALL);
        assert_eq!(q, Permissions::ZERO);
    }

    #[test]
    fn pledge_composition_equals_intersected_single_pledge() {
        // pledge(a1, b1).pledge(a2, b2) == pledge(a1 & a2, b1 & b2).
        // Algebraically: pledge is a left-fold of bitwise-AND over a
        // sequence of mask pairs.
        let p = Permissions::from_masks(class::ALL, class::ALL, role::SHELL);
        let two_step = p
            .pledge(PermsRequest {
                perms: union(union(class::STDIO, class::VMEM), class::NETCH),
                allowed_perms: union(union(class::STDIO, class::VMEM), class::NETCH),
            })
            .pledge(PermsRequest {
                perms: union(class::STDIO, class::VMEM),
                allowed_perms: class::STDIO,
            });
        let one_step = p.pledge(PermsRequest {
            perms: union(class::STDIO, class::VMEM),
            allowed_perms: class::STDIO,
        });
        assert_eq!(two_step, one_step);
    }

    #[test]
    fn pledge_is_commutative_through_and() {
        // Two pledges commute: applying mask pairs in either order
        // yields the same result, since both reduce to the same
        // intersection.
        let p = Permissions::from_masks(class::ALL, class::ALL, role::LOADER);
        let req_ab = PermsRequest {
            perms: union(class::STDIO, class::NETCH),
            allowed_perms: class::STDIO,
        };
        let req_cd = PermsRequest {
            perms: union(class::STDIO, class::VMEM),
            allowed_perms: class::ALL,
        };
        let ab_then_cd = p.pledge(req_ab).pledge(req_cd);
        let cd_then_ab = p.pledge(req_cd).pledge(req_ab);
        assert_eq!(ab_then_cd, cd_then_ab);
    }

    #[test]
    fn allows_respects_pledge_narrowing() {
        // After pledge drops NETCH, allows() rejects CREATE_NETCH but
        // still permits other surviving classes.
        let p = Permissions::ALL;
        assert!(p.allows(crate::syscall::CREATE_NETCH));
        // ALL minus NETCH expressed via raw bits — the test composes
        // wide-without-one-bit which has no narrow form on ClassMask.
        let q = p.pledge(PermsRequest {
            perms: ClassMask::from_raw(class::raw::ALL & !class::raw::NETCH),
            allowed_perms: class::ALL,
        });
        assert!(!q.allows(crate::syscall::CREATE_NETCH));
        assert!(q.allows(crate::syscall::SERIAL_PRINT));
        assert!(q.allows(crate::syscall::MMAP));
    }

    #[test]
    fn allows_is_false_for_perms_zero_regardless_of_syscall() {
        // ZERO is the fail-safe state — every syscall is denied. Pin
        // this so a future refactor can't accidentally make a
        // zero-perms process syscall-permissive.
        use crate::syscall::*;
        let p = Permissions::ZERO;
        for s in [
            SERIAL_PRINT,
            MMAP,
            CREATE_NETCH,
            FS_OPEN,
            FUTEX_WAIT,
            CREATE_PROCESS,
            EXIT,
        ] {
            assert!(!p.allows(s), "ZERO.allows({s}) should be false");
        }
    }

    #[test]
    fn new_preserves_role_argument_verbatim() {
        // Permissions::new doesn't validate the role — it's a thin
        // wire-shape constructor. Any in-range RoleId round-trips.
        for r in [
            role::NOROLE,
            role::BOOTSTRAP,
            role::LOADER,
            role::SHELL,
            role::NET_CLIENT,
            role::FS_TOOL,
            role::WORKER,
            role::SERVICE,
        ] {
            let p = Permissions::new(class::raw::ALL, class::raw::ALL, r);
            assert_eq!(p.role, r);
            // from_masks (the typed surface) round-trips the same way.
            let q = Permissions::from_masks(class::ALL, class::ALL, r);
            assert_eq!(q.role, r);
        }
    }

    #[test]
    fn new_zeroes_pad_and_reserved_unconditionally() {
        // Both constructors always produce _pad=0 and _reserved=[0;2]
        // — there's no way to inject garbage via the public API. Basis
        // for the "derived child has clean tail" property tested in
        // roles.rs.
        let p = Permissions::new(class::raw::ALL, class::raw::STDIO, role::WORKER);
        assert_eq!(p._pad, 0);
        assert_eq!(p._reserved, [0; 2]);
        let q = Permissions::from_masks(class::ALL, class::STDIO, role::WORKER);
        assert_eq!(q._pad, 0);
        assert_eq!(q._reserved, [0; 2]);
    }

    #[test]
    fn new_accepts_arbitrary_unrelated_axes() {
        // Constructor doesn't impose subset relations between perms
        // and allowed_perms — each axis is independent. Any pair of
        // u64 masks (including future bits beyond class::raw::ALL)
        // round-trips exactly.
        let cases = [
            (0u64, 0u64),
            (class::raw::ALL, 0),
            (0, class::raw::ALL),
            (class::raw::NETCH, class::raw::STDIO), // perms with NETCH, allowed without
            (class::raw::STDIO, class::raw::NETCH), // inverse
            (!0u64, !0u64),                         // future bits — stored verbatim
        ];
        for (perms, allowed) in cases {
            let p = Permissions::new(perms, allowed, role::SHELL);
            assert_eq!(p.perms, perms);
            assert_eq!(p.allowed_perms, allowed);
        }
    }

    #[test]
    fn pledge_chains_to_zero_through_disjoint_masks() {
        // Two pledges with disjoint perms masks intersect to zero —
        // a process that accidentally pledges away "everything I need"
        // ends up at the empty perms state.
        let p = Permissions::from_masks(class::ALL, class::ALL, role::SHELL);
        let q = p.pledge(PermsRequest {
            perms: class::STDIO,
            allowed_perms: class::ALL,
        });
        let r = q.pledge(PermsRequest {
            perms: class::NETCH,
            allowed_perms: class::ALL,
        });
        assert!(r.perms_mask().is_empty());
        // allowed_perms wasn't touched.
        assert_eq!(r.allowed_perms_mask(), class::ALL);
    }

    /// Vec-backed DenialSink for can_call() tests. Captures pushed
    /// events for inspection — the production sink is in orbit-core
    /// (the bounded ring) and isn't reachable from this crate's tests.
    #[derive(Default)]
    struct CapturingSink(alloc::vec::Vec<crate::denial::DenialEvent>);

    impl crate::denial::DenialSink for CapturingSink {
        fn push(&mut self, event: crate::denial::DenialEvent) {
            self.0.push(event);
        }
    }

    extern crate alloc;

    fn ctx() -> crate::denial::GateContext {
        crate::denial::GateContext {
            pid: 7,
            tid: 11,
            time_ticks: 99_000,
        }
    }

    #[test]
    fn can_call_returns_true_and_does_not_push_when_allowed() {
        // Happy path — perms permit the syscall, no event pushed.
        // The sink is a witness that no spurious events landed.
        let p = Permissions::ALL;
        let mut sink = CapturingSink::default();
        let ok = p.can_call(crate::syscall::CREATE_NETCH, ctx(), &mut sink);
        assert!(ok);
        assert!(sink.0.is_empty(), "no event should fire on the allow path");
    }

    #[test]
    fn can_call_returns_false_and_pushes_perm_deny_when_denied() {
        // Pledge away NETCH, then attempt CREATE_NETCH — gate returns
        // false AND pushes one PermDeny carrying the right class /
        // perms / sysno / pid / tid / time_ticks.
        use crate::denial::DenialEvent;
        let p = Permissions::ALL.pledge(PermsRequest {
            perms: ClassMask::from_raw(class::raw::ALL & !class::raw::NETCH),
            allowed_perms: class::ALL,
        });
        let mut sink = CapturingSink::default();
        let ok = p.can_call(crate::syscall::CREATE_NETCH, ctx(), &mut sink);
        assert!(!ok, "denied call must return false");
        assert_eq!(sink.0.len(), 1, "exactly one event per denied call");
        match sink.0[0] {
            DenialEvent::PermDeny {
                required_class,
                perms,
                time_ticks,
                tid,
                sysno,
                source_role,
                pid,
            } => {
                assert_eq!(required_class, class::raw::NETCH);
                assert_eq!(perms, class::raw::ALL & !class::raw::NETCH);
                assert_eq!(time_ticks, 99_000);
                assert_eq!(tid, 11);
                assert_eq!(sysno as usize, crate::syscall::CREATE_NETCH);
                // Permissions::ALL has role::BOOTSTRAP — the gate populates
                // source_role from self.role for symmetric "which role
                // hit the gate" diagnosis.
                assert_eq!(source_role, role::BOOTSTRAP);
                assert_eq!(pid, 7);
            }
            other => panic!("expected PermDeny, got {other:?}"),
        }
    }

    #[test]
    fn can_call_for_unknown_syscall_pushes_event_with_empty_class() {
        // Unknown sysno → class_for returns EMPTY, allows returns false.
        // The gate still pushes a PermDeny so reviewers can spot the
        // "syscall N has no class — extend class_for" miss in the log.
        use crate::denial::DenialEvent;
        let p = Permissions::ALL;
        let mut sink = CapturingSink::default();
        let ok = p.can_call(99_999, ctx(), &mut sink);
        assert!(!ok);
        assert_eq!(sink.0.len(), 1);
        match sink.0[0] {
            DenialEvent::PermDeny {
                required_class,
                sysno,
                source_role,
                ..
            } => {
                assert_eq!(required_class, 0);
                assert_eq!(sysno, 99_999);
                // Even on unknown-sysno path the source role is captured.
                assert_eq!(source_role, role::BOOTSTRAP);
            }
            other => panic!("expected PermDeny, got {other:?}"),
        }
    }

    #[test]
    fn can_call_with_unit_sink_compiles_and_returns_correct_bool() {
        // `()` is a no-op DenialSink — the gate still computes the
        // bool correctly under enforcement-shaped contexts where
        // logging is redundant.
        let p = Permissions::ALL;
        let mut sink = ();
        assert!(p.can_call(crate::syscall::SERIAL_PRINT, ctx(), &mut sink));
        let q = Permissions::ZERO;
        assert!(!q.can_call(crate::syscall::SERIAL_PRINT, ctx(), &mut sink));
    }

    #[test]
    fn can_call_does_not_double_push_on_repeated_denials() {
        // Each invocation pushes exactly one event. Two back-to-back
        // denials produce two events — caller-side dedup, if any,
        // happens at the ring not at the gate.
        let p = Permissions::ZERO;
        let mut sink = CapturingSink::default();
        let _ = p.can_call(crate::syscall::SERIAL_PRINT, ctx(), &mut sink);
        let _ = p.can_call(crate::syscall::CREATE_NETCH, ctx(), &mut sink);
        assert_eq!(sink.0.len(), 2);
    }

    #[test]
    fn allows_is_true_for_all_perms_on_every_known_syscall() {
        // Symmetric to the above: a process with class::ALL should be
        // able to invoke every known sysno. Catches regressions in the
        // class table where a new sysno's class isn't in class::ALL.
        use crate::syscall::*;
        let p = Permissions::ALL;
        for s in [
            EXIT,
            SERIAL_PRINT,
            SLEEP_MS,
            CONSOLE_WRITE,
            READ_STDIN,
            SET_AFFINITY,
            GET_AFFINITY,
            GET_HART_ID,
            GET_MICROS,
            PLEDGE,
            MMAP,
            CREATE_NETCH,
            CLOSE_HANDLE,
            CREATE_PROCESS,
            NC_YIELD,
            QUERY_STATS,
            QUERY_SYSCALL_STATS,
            CREATE_PROCESS_EX,
            ARGV_ENVP,
            CREATE_PROCESS_V2,
            QUERY_DENIAL_LOG,
            CREATE_THREAD,
            GETPID,
            GETTID,
            WAIT_PID,
            FUTEX_WAIT,
            FUTEX_WAKE,
            FS_OPEN,
            FS_READ,
            FS_STAT,
            FS_SEEK,
            FS_FSTAT,
            CHDIR,
            GETCWD,
        ] {
            assert!(p.allows(s), "ALL.allows({s}) should be true");
        }
    }
}
