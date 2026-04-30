//! Role registry, transition gating, and `create_process_v2` clamp math.
//!
//! Pure-logic counterpart to [`orbit_abi::perms`]. Owns:
//!
//! - The [`ROLES`] table — per-role default perms, default cap, and the
//!   bitset of target roles a process may transition into via
//!   `create_process_v2`. Defaults are typed [`ClassMask`]s.
//! - [`check_transition`] — gate 1: validates `(parent_role, target_role)`
//!   against the registry. On success returns a [`TransitionAllowed`]
//!   witness; that witness is the *only* legitimate source for the
//!   transition argument to [`derive_child_perms`]. PR3 deletes the
//!   `Err` arm of the transition check at the call site (it becomes
//!   `EPERM`); PR2's shadow path uses [`install_child_shadow`] to
//!   bypass the witness when the would-be denial is being shadowed.
//! - [`derive_child_perms`] — gate 2: with a proven transition,
//!   computes the child's clamped [`Permissions`] and wraps them in a
//!   [`ChildPerms`] witness. The kernel's `Process::install_child` will
//!   take a [`ChildPerms`] (and, alongside it, a [`TransitionAllowed`])
//!   so the type system enforces "every spawn went through both gates."
//! - [`install_child_shadow`] — the PR2 escape hatch. Computes the
//!   child's would-be [`Permissions`] *as if* the transition were
//!   allowed. Used only on the shadow-mode `Err(TransitionDenied)`
//!   branch; PR3 deletes both the function and its sole caller.
//!
//! See [docs/dev/permissions-roles.md](../../../docs/dev/permissions-roles.md)
//! for the design rationale, the role transition diagram, and how this
//! module fits into the broader create_process_v2 flow.
//!
//! No `Hardware` dependency — the registry is static data and the
//! clamping math is pure. Host-testable in unit tests below.

use orbit_abi::perms::{
    class,
    role::{self, RoleId},
    ClassMask, Permissions, PermsRequest,
};

/// Per-role policy entry. The `ROLES` array is indexed by [`RoleId`];
/// looking up via [`role_def`] keeps the bounds-check explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoleDef {
    /// Default `perms` for a process spawned into this role, before the
    /// parent-cap and caller-requested clamps. The role's "natural"
    /// permission set.
    pub default_perms: ClassMask,
    /// Default `allowed_perms` — the cap that propagates to
    /// grandchildren. Encodes the "can re-grant?" axis: a role with
    /// `default_allowed` excluding NETCH cannot pass NETCH down even
    /// if it holds NETCH itself.
    pub default_allowed: ClassMask,
    /// Bitset over `RoleId` space — bit `i` set ⇔ a process in this
    /// role may transition to role `i` via `create_process_v2`.
    /// Zero ⇒ terminal (no spawning, even of the same role).
    pub transitions: u64,
}

impl RoleDef {
    /// Does this role permit a transition to `target`? `target >= 64`
    /// is rejected because the bitset can't represent it (we'd need
    /// to widen `transitions` first).
    pub const fn allows_transition(&self, target: RoleId) -> bool {
        if target >= 64 {
            return false;
        }
        (self.transitions & (1u64 << target)) != 0
    }
}

/// Helper for table literals — `bit(role::FOO)` is more readable than
/// `1 << role::FOO`.
const fn bit(role: RoleId) -> u64 {
    1u64 << (role as u32)
}

// ── Per-role permission masks ────────────────────────────────────────────
//
// Wide masks composed from raw u64 bits via `ClassMask::from_raw`. This is
// the static-init escape hatch — runtime code can't widen a `ClassMask`,
// but the registry needs to declare role defaults that cover multiple
// classes. Using `class::raw::*` here makes the widening surface
// explicit and grep-able.

/// SHELL's effective set: stdio, lifecycle, spawn, sched, private mmap,
/// fs reads, stats, pledge. Excludes NETCH / VMEM_SHARED / FUTEX from
/// `perms` — the shell itself doesn't speak network or share kernel
/// memory; it's a single-threaded process spawning children that do.
const SHELL_PERMS: ClassMask = ClassMask::from_raw(
    class::raw::STDIO
        | class::raw::PROC_LIFE
        | class::raw::PROC_SPAWN
        | class::raw::SCHED
        | class::raw::VMEM
        | class::raw::FS_RO
        | class::raw::STATS
        | class::raw::PLEDGE,
);

/// SHELL's child-cap: every class. The shell is a privilege-passing
/// authority — children may receive NETCH, VMEM_SHARED, FUTEX, etc.
/// even though SHELL itself doesn't use them.
const SHELL_ALLOWED: ClassMask = class::ALL;

/// NET_CLIENT effective: stdio + network + futex (for any internal
/// async runtime) + private mmap + pledge. No FS_RO, no spawn-by-default
/// — those have to be granted explicitly by the loader if needed.
const NET_CLIENT_PERMS: ClassMask = ClassMask::from_raw(
    class::raw::STDIO
        | class::raw::NETCH
        | class::raw::FUTEX
        | class::raw::VMEM
        | class::raw::PLEDGE,
);

/// NET_CLIENT cap: deliberately excludes NETCH. Grandchildren of a
/// network client get no network reach by default — the network
/// capability dies at the first generation. This is the load-bearing
/// example of role-based propagation control.
const NET_CLIENT_ALLOWED: ClassMask = ClassMask::from_raw(
    class::raw::STDIO | class::raw::VMEM | class::raw::PLEDGE,
);

/// FS_TOOL effective: stdio + read-only filesystem + private mmap +
/// pledge. Same shape as NET_CLIENT but with FS_RO instead of NETCH.
const FS_TOOL_PERMS: ClassMask = ClassMask::from_raw(
    class::raw::STDIO | class::raw::FS_RO | class::raw::VMEM | class::raw::PLEDGE,
);

/// FS_TOOL cap: same shape as NET_CLIENT_ALLOWED — fs reach doesn't
/// propagate either.
const FS_TOOL_ALLOWED: ClassMask = ClassMask::from_raw(
    class::raw::STDIO | class::raw::VMEM | class::raw::PLEDGE,
);

/// WORKER effective + cap: stdio + private mmap + pledge. The "default
/// sandbox" — no I/O, no networking, no spawn. Used as the grandchild
/// fallthrough for NET_CLIENT and FS_TOOL since their `default_allowed`
/// matches WORKER_PERMS exactly.
const WORKER_PERMS: ClassMask = ClassMask::from_raw(
    class::raw::STDIO | class::raw::VMEM | class::raw::PLEDGE,
);

/// SERVICE effective: long-lived daemon shape — stdio + network +
/// fs reads + futex + private mmap + pledge + stats. The "service"
/// counterpart to NET_CLIENT for processes that aren't shell-children.
const SERVICE_PERMS: ClassMask = ClassMask::from_raw(
    class::raw::STDIO
        | class::raw::NETCH
        | class::raw::FS_RO
        | class::raw::FUTEX
        | class::raw::VMEM
        | class::raw::PLEDGE
        | class::raw::STATS,
);

/// SERVICE cap: same as NET_CLIENT — no I/O propagation to spawned
/// children.
const SERVICE_ALLOWED: ClassMask = ClassMask::from_raw(
    class::raw::STDIO | class::raw::VMEM | class::raw::PLEDGE,
);

/// Role registry, indexed by [`RoleId`]. Order is load-bearing — the
/// indices are the `role::*` constants. Adding a role: append to the
/// end of [`role`] in orbit-abi, bump `role::COUNT`, append a row
/// here, and update any source roles whose `transitions` should
/// include the new target.
pub static ROLES: [RoleDef; role::COUNT] = [
    /* NOROLE     */
    RoleDef {
        default_perms: ClassMask::EMPTY,
        default_allowed: ClassMask::EMPTY,
        transitions: 0,
    },
    /* BOOTSTRAP  */
    RoleDef {
        default_perms: class::ALL,
        default_allowed: class::ALL,
        // Boot path: kmain spawns LOADER. Direct shell spawn from
        // bootstrap is allowed too, for tests / rescue scenarios.
        transitions: bit(role::LOADER) | bit(role::SHELL),
    },
    /* LOADER     */
    RoleDef {
        default_perms: class::ALL,
        default_allowed: class::ALL,
        transitions: bit(role::SHELL)
            | bit(role::SERVICE)
            | bit(role::WORKER)
            | bit(role::NET_CLIENT)
            | bit(role::FS_TOOL),
    },
    /* SHELL      */
    RoleDef {
        default_perms: SHELL_PERMS,
        default_allowed: SHELL_ALLOWED,
        transitions: bit(role::NET_CLIENT) | bit(role::FS_TOOL) | bit(role::WORKER),
    },
    /* NET_CLIENT */
    RoleDef {
        default_perms: NET_CLIENT_PERMS,
        default_allowed: NET_CLIENT_ALLOWED,
        transitions: bit(role::WORKER),
    },
    /* FS_TOOL    */
    RoleDef {
        default_perms: FS_TOOL_PERMS,
        default_allowed: FS_TOOL_ALLOWED,
        transitions: bit(role::WORKER),
    },
    /* WORKER     */
    RoleDef {
        default_perms: WORKER_PERMS,
        default_allowed: WORKER_PERMS, // perms == allowed: leaves grant nothing
        transitions: 0,                // terminal
    },
    /* SERVICE    */
    RoleDef {
        default_perms: SERVICE_PERMS,
        default_allowed: SERVICE_ALLOWED,
        transitions: bit(role::WORKER),
    },
];

// Compile-time invariants for the registry. Failures here are caught
// at `cargo build`, not at `cargo test` — useful when someone adds
// a role to `role::*` and forgets to extend ROLES, or vice versa.
// Iteration-heavy invariants (transitions point at valid roles, leaf
// roles have perms == allowed) stay as runtime tests since
// `const fn` for-loops over slice iterators are still patchy on the
// pinned nightly; revisit during PR2.
const _: () = {
    assert!(
        ROLES.len() == role::COUNT,
        "ROLES table length must match role::COUNT — extend ROLES when adding a role::* constant"
    );
    // NOROLE must be the zero state — every field zeroed. Pinned so
    // a future "let me give NOROLE some default perms" change has to
    // explicitly reckon with the sentinel-role contract.
    let nr = &ROLES[role::NOROLE as usize];
    assert!(nr.default_perms.raw() == 0, "NOROLE default_perms must be 0");
    assert!(nr.default_allowed.raw() == 0, "NOROLE default_allowed must be 0");
    assert!(nr.transitions == 0, "NOROLE must have no transitions");
};

/// Look up a role definition by ID. `None` if `role` is past the
/// registry. The kernel treats `None` as a corrupt parent state
/// (impossible if the parent was spawned through `derive_child_perms`)
/// or an invalid target (caller bug — surfaced as `EPERM`).
pub const fn role_def(role: RoleId) -> Option<&'static RoleDef> {
    if (role as usize) < role::COUNT {
        Some(&ROLES[role as usize])
    } else {
        None
    }
}

/// Reasons a spawn may be rejected. Each maps to a distinct
/// kernel-side log message; the syscall returns `EPERM` for all of
/// them — the caller doesn't get to learn the discriminant, since
/// that would leak policy information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnDeny {
    /// Parent's `role` field doesn't index into `ROLES`. Indicates
    /// corruption, since a process should only ever have a role that
    /// was assigned through this same code path.
    UnknownParentRole,
    /// Caller's requested `target_role` is out of range.
    UnknownTargetRole,
    /// Parent's role registry entry doesn't permit a transition into
    /// `target_role`. The common case for "your shell can't spawn
    /// that kind of process."
    TransitionDenied,
}

/// Witness that a `(source_role, target_role)` transition was
/// validated against the registry. The only constructor is
/// [`check_transition`]; the inner fields are private so external
/// code can't fabricate one. Pass to [`derive_child_perms`] to
/// produce a [`ChildPerms`] without re-running the check.
///
/// Carries the resolved `&'static RoleDef` for the target alongside
/// the role ID, so [`derive_child_perms`] doesn't need to redo the
/// `role_def(target)` lookup — both eliminating the `expect("validated
/// by check_transition")` panic path *and* tightening the witness
/// contract: the registry entry referenced here is exactly the one
/// `check_transition` validated, by construction.
///
/// Cheap — `Copy`, 24 bytes (8 for `&RoleDef`, 4+4 for the role IDs,
/// no padding). Still passes by register pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct TransitionAllowed {
    source: RoleId,
    target: RoleId,
    /// The resolved registry entry for `target`. `pub(crate)` so
    /// [`derive_child_perms`] can read it without an `expect`;
    /// external code uses [`target`](Self::target) (the role ID).
    /// `&'static` lifetime — the registry is a `static`.
    target_def: &'static RoleDef,
}

impl TransitionAllowed {
    /// Source role of the validated transition (the parent process's
    /// role at check time). Read-only.
    pub const fn source(&self) -> RoleId {
        self.source
    }

    /// Target role of the validated transition (the role the child
    /// will inhabit). Read-only.
    pub const fn target(&self) -> RoleId {
        self.target
    }

    /// Resolved registry entry for `target`. Crate-private — used by
    /// [`derive_child_perms`] to skip the registry re-lookup.
    pub(crate) const fn target_def(&self) -> &'static RoleDef {
        self.target_def
    }
}

/// Witness that a child's [`Permissions`] were produced by
/// [`derive_child_perms`] — i.e., that they were clamped against the
/// parent's `allowed_perms`, the target role's defaults, and any
/// caller-requested narrowing. The only constructor is
/// [`derive_child_perms`]; the inner field is private. Kernel-side
/// `Process::install_child` should require a `ChildPerms` so the
/// type system enforces "every spawned child went through the clamp."
///
/// **`Copy` semantics — proof-of-provenance, not one-shot.** A
/// `ChildPerms` value attests "the gate ran for these `Permissions`,"
/// not "this is the unique installation token." Cloning / copying
/// the witness is fine; the threat model is "did a gate run?", not
/// "could the same result be installed twice?" The "install once"
/// property is structural in the kernel side — PR2's `Process` API
/// exposes `install_child(c: ChildPerms)` (witness path) and
/// `install_child_via_shadow(p: Permissions)` (PR2-only, deleted in
/// PR3) and *no* `set_permissions(Permissions)` overload. Without
/// such a setter, post-PR3 the witness path is the only path to a
/// populated child, regardless of how many times a `ChildPerms` got
/// copied along the way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ChildPerms(Permissions);

impl ChildPerms {
    /// Read-only view of the underlying [`Permissions`]. Useful for
    /// logging / shadow-event push without consuming the witness.
    ///
    /// **Bypass hazard.** This returns an owned [`Permissions`]
    /// because `Permissions: Copy`. A caller could route the result
    /// through a non-witness install path. The mitigation is the
    /// kernel-side `Process` API surface (see struct-level docs):
    /// no `set_permissions` overload exists, so the only paths to a
    /// populated child are `Process::install_child` (witness) and
    /// `Process::install_child_via_shadow` (deleted in PR3).
    pub const fn permissions(&self) -> Permissions {
        self.0
    }

    /// Consume the witness, yielding the [`Permissions`] for
    /// installation onto a fresh `Process`. Same bypass hazard as
    /// [`permissions`](Self::permissions) — the witness is
    /// proof-of-provenance, not a one-shot install token. The
    /// kernel-side discipline (no `set_permissions` overload) is
    /// what makes the witness path the only path post-PR3.
    pub const fn into_permissions(self) -> Permissions {
        self.0
    }
}

/// Validate a `(parent_role, target_role)` transition against the
/// registry. Returns a [`TransitionAllowed`] witness on success.
///
/// Errors:
/// - `UnknownParentRole` — `parent_role` is out of registry range.
///   Indicates corruption; should never happen for a process that
///   was spawned through this code path.
/// - `UnknownTargetRole` — `target_role` is out of registry range.
///   Caller bug.
/// - `TransitionDenied` — registry entry for `parent_role` doesn't
///   include `target_role` in its `transitions` bitset. Common case
///   for legitimate denials.
pub fn check_transition(
    parent_role: RoleId,
    target_role: RoleId,
) -> Result<TransitionAllowed, SpawnDeny> {
    let parent_def = role_def(parent_role).ok_or(SpawnDeny::UnknownParentRole)?;
    let target_def = role_def(target_role).ok_or(SpawnDeny::UnknownTargetRole)?;
    if !parent_def.allows_transition(target_role) {
        return Err(SpawnDeny::TransitionDenied);
    }
    Ok(TransitionAllowed {
        source: parent_role,
        target: target_role,
        target_def,
    })
}

/// Compute the child's [`Permissions`] for a `create_process_v2` call.
/// Infallible — the transition has already been validated and the
/// witness consumed; this is just the clamp math.
///
/// Three-way clamp on each axis, *independently*:
///
/// - `child.perms = target_role.default_perms
///                ∩ parent.allowed_perms
///                ∩ request.perms`
/// - `child.allowed_perms = target_role.default_allowed
///                        ∩ parent.allowed_perms
///                        ∩ request.allowed_perms`
///
/// Both axes are clamped against `parent.allowed_perms` — that's the
/// load-bearing piece (the parent cap propagates) — but they are NOT
/// clamped against each other. A child can legitimately end up with
/// `perms ⊋ allowed_perms` (the NET_CLIENT case: own NETCH but no
/// NETCH cap to pass to its own children).
///
/// `request` is the caller's narrowing requests. Passing
/// [`PermsRequest::ALL`] for both means "give me whatever the role
/// default and my parent's cap permit." Passing
/// [`PermsRequest::EMPTY`] gives a child with no permissions — legal
/// but useless.
pub fn derive_child_perms(
    parent: &Permissions,
    transition: TransitionAllowed,
    request: PermsRequest,
) -> ChildPerms {
    // No expect / no panic path: the witness carries the resolved
    // RoleDef directly (populated by check_transition). Even a forged
    // witness via `mem::transmute` would only produce undefined
    // behaviour at the transmute site, not here — and `&'static
    // RoleDef` can't be fabricated to a valid value without already
    // having access to the registry, so a forge can't slip a
    // pretend-RoleDef past us either.
    let target_def = transition.target_def();
    let parent_cap = parent.allowed_perms_mask();

    let child_perms = target_def
        .default_perms
        .narrow(parent_cap)
        .narrow(request.perms);
    let child_allowed = target_def
        .default_allowed
        .narrow(parent_cap)
        .narrow(request.allowed_perms);

    ChildPerms(Permissions::from_masks(
        child_perms,
        child_allowed,
        transition.target,
    ))
}

/// **PR2 shadow-only.** Compute the child's would-be [`Permissions`]
/// *as if* the transition had been allowed. Used by the shadow-mode
/// `create_process_v2` handler on the `Err(TransitionDenied)` branch
/// of [`check_transition`], so the spawn proceeds with realistic
/// child perms while the kernel logs the would-be denial.
///
/// Returns raw [`Permissions`] (no [`ChildPerms`] witness) — the
/// shadow path bypasses the type-level enforcement that production
/// spawns require. Kernel-side, this means the manager has a
/// separate `install_child_via_shadow` path that takes a plain
/// `Permissions`. Both the function here and that kernel-side path
/// are deleted in PR3 when `Err(TransitionDenied)` becomes EPERM.
///
/// Errors:
/// - `UnknownTargetRole` if `target_role` is out of registry range.
///   Even in shadow mode we don't pretend an out-of-range role is
///   real — that's a caller bug, not a policy denial.
pub fn install_child_shadow(
    parent: &Permissions,
    target_role: RoleId,
    request: PermsRequest,
) -> Result<Permissions, SpawnDeny> {
    let target_def = role_def(target_role).ok_or(SpawnDeny::UnknownTargetRole)?;
    let parent_cap = parent.allowed_perms_mask();

    let child_perms = target_def
        .default_perms
        .narrow(parent_cap)
        .narrow(request.perms);
    let child_allowed = target_def
        .default_allowed
        .narrow(parent_cap)
        .narrow(request.allowed_perms);

    Ok(Permissions::from_masks(child_perms, child_allowed, target_role))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent_in_role(r: RoleId) -> Permissions {
        let def = role_def(r).expect("known role");
        Permissions::from_masks(def.default_perms, def.default_allowed, r)
    }

    fn full_request() -> PermsRequest {
        PermsRequest::ALL
    }

    fn try_spawn(
        parent: &Permissions,
        target: RoleId,
        request: PermsRequest,
    ) -> Result<ChildPerms, SpawnDeny> {
        let transition = check_transition(parent.role, target)?;
        Ok(derive_child_perms(parent, transition, request))
    }

    // ── transition matrix ────────────────────────────────────────────────

    #[test]
    fn worker_is_terminal() {
        let worker = parent_in_role(role::WORKER);
        for target in 0..role::COUNT as RoleId {
            let r = try_spawn(&worker, target, full_request());
            assert!(
                matches!(r, Err(SpawnDeny::TransitionDenied)),
                "WORKER → {target} should be denied; got {r:?}"
            );
        }
    }

    #[test]
    fn shell_can_spawn_net_client_fs_tool_worker() {
        let shell = parent_in_role(role::SHELL);
        for target in [role::NET_CLIENT, role::FS_TOOL, role::WORKER] {
            assert!(
                try_spawn(&shell, target, full_request()).is_ok(),
                "SHELL → {target} should be allowed"
            );
        }
    }

    #[test]
    fn shell_cannot_spawn_loader_or_shell_or_service() {
        let shell = parent_in_role(role::SHELL);
        for target in [role::LOADER, role::SHELL, role::SERVICE, role::BOOTSTRAP] {
            let r = try_spawn(&shell, target, full_request());
            assert!(
                matches!(r, Err(SpawnDeny::TransitionDenied)),
                "SHELL → {target} should be denied; got {r:?}"
            );
        }
    }

    #[test]
    fn loader_can_spawn_all_normal_roles() {
        let loader = parent_in_role(role::LOADER);
        for target in [
            role::SHELL,
            role::SERVICE,
            role::WORKER,
            role::NET_CLIENT,
            role::FS_TOOL,
        ] {
            assert!(
                try_spawn(&loader, target, full_request()).is_ok(),
                "LOADER → {target} should be allowed"
            );
        }
    }

    #[test]
    fn loader_cannot_spawn_loader_or_bootstrap_or_norole() {
        let loader = parent_in_role(role::LOADER);
        for target in [role::LOADER, role::BOOTSTRAP, role::NOROLE] {
            assert!(
                matches!(
                    try_spawn(&loader, target, full_request()),
                    Err(SpawnDeny::TransitionDenied)
                ),
                "LOADER → {target} should be denied"
            );
        }
    }

    #[test]
    fn unknown_target_role_is_rejected() {
        let r = check_transition(role::SHELL, 9999);
        assert_eq!(r, Err(SpawnDeny::UnknownTargetRole));
    }

    #[test]
    fn unknown_parent_role_is_rejected() {
        let r = check_transition(9999, role::WORKER);
        assert_eq!(r, Err(SpawnDeny::UnknownParentRole));
    }

    #[test]
    fn check_transition_witness_carries_source_and_target() {
        let t = check_transition(role::SHELL, role::WORKER).unwrap();
        assert_eq!(t.source(), role::SHELL);
        assert_eq!(t.target(), role::WORKER);
    }

    #[test]
    fn net_client_can_only_spawn_worker() {
        let nc = parent_in_role(role::NET_CLIENT);
        for target in 0..role::COUNT as RoleId {
            let r = try_spawn(&nc, target, full_request());
            if target == role::WORKER {
                assert!(r.is_ok(), "NET_CLIENT → {target} should succeed; got {r:?}");
            } else {
                assert!(
                    matches!(r, Err(SpawnDeny::TransitionDenied)),
                    "NET_CLIENT → {target} should be TransitionDenied; got {r:?}"
                );
            }
        }
    }

    #[test]
    fn fs_tool_can_only_spawn_worker() {
        let ft = parent_in_role(role::FS_TOOL);
        for target in 0..role::COUNT as RoleId {
            let r = try_spawn(&ft, target, full_request());
            if target == role::WORKER {
                assert!(r.is_ok(), "FS_TOOL → {target} should succeed; got {r:?}");
            } else {
                assert!(
                    matches!(r, Err(SpawnDeny::TransitionDenied)),
                    "FS_TOOL → {target} should be TransitionDenied; got {r:?}"
                );
            }
        }
    }

    #[test]
    fn service_can_only_spawn_worker() {
        let svc = parent_in_role(role::SERVICE);
        for target in 0..role::COUNT as RoleId {
            let r = try_spawn(&svc, target, full_request());
            if target == role::WORKER {
                assert!(r.is_ok(), "SERVICE → {target} should succeed; got {r:?}");
            } else {
                assert!(
                    matches!(r, Err(SpawnDeny::TransitionDenied)),
                    "SERVICE → {target} should be TransitionDenied; got {r:?}"
                );
            }
        }
    }

    #[test]
    fn bootstrap_can_only_spawn_loader_or_shell() {
        // BOOTSTRAP transitions to LOADER (canonical first child) and
        // SHELL (test/rescue). Everything else denied.
        let bs = parent_in_role(role::BOOTSTRAP);
        for target in 0..role::COUNT as RoleId {
            let r = try_spawn(&bs, target, full_request());
            let expected_ok = target == role::LOADER || target == role::SHELL;
            if expected_ok {
                assert!(r.is_ok(), "BOOTSTRAP → {target} should succeed; got {r:?}");
            } else {
                assert!(
                    matches!(r, Err(SpawnDeny::TransitionDenied)),
                    "BOOTSTRAP → {target} should be denied; got {r:?}"
                );
            }
        }
    }

    #[test]
    fn norole_cannot_spawn_anything() {
        // NOROLE is the sentinel (zero perms, zero allowed, zero
        // transitions). A process holding it is stranded by design.
        let nr = parent_in_role(role::NOROLE);
        for target in 0..role::COUNT as RoleId {
            let r = try_spawn(&nr, target, full_request());
            assert!(
                matches!(r, Err(SpawnDeny::TransitionDenied)),
                "NOROLE → {target} should be denied; got {r:?}"
            );
        }
    }

    #[test]
    fn norole_is_not_a_valid_spawn_target() {
        // No defined role transitions to NOROLE, so check_transition
        // always returns TransitionDenied (not UnknownTargetRole — 0 is
        // in range). Pinning this makes the "NOROLE is unreachable from
        // outside" property explicit.
        for parent_role in 0..role::COUNT as RoleId {
            let parent = parent_in_role(parent_role);
            let r = try_spawn(&parent, role::NOROLE, full_request());
            assert!(
                matches!(r, Err(SpawnDeny::TransitionDenied)),
                "spawn target=NOROLE from parent={parent_role} should be TransitionDenied; got {r:?}"
            );
        }
    }

    #[test]
    fn full_transition_matrix_matches_registry_bits() {
        // For every (parent, target) pair, check_transition's outcome
        // must agree with `parent.transitions & bit(target)`. Adding a
        // role or flipping a transition bit can't drift from the test
        // suite without this firing.
        for parent_role in 0..role::COUNT as RoleId {
            let parent_def = role_def(parent_role).unwrap();
            for target_role in 0..role::COUNT as RoleId {
                let r = check_transition(parent_role, target_role);
                let bit_set = parent_def.allows_transition(target_role);
                match (bit_set, &r) {
                    (true, Ok(_)) => {}
                    (false, Err(SpawnDeny::TransitionDenied)) => {}
                    _ => panic!(
                        "matrix mismatch: parent={parent_role} target={target_role} \
                         transition_bit={bit_set} got={r:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn check_transition_in_range_only_returns_ok_or_transition_denied() {
        // The bounded matrix can't produce UnknownParentRole or
        // UnknownTargetRole — those only fire for out-of-range IDs.
        // Pinning this distinguishes "policy denial" from "caller bug"
        // for the in-range case.
        for parent in 0..role::COUNT as RoleId {
            for target in 0..role::COUNT as RoleId {
                let r = check_transition(parent, target);
                match r {
                    Ok(_) | Err(SpawnDeny::TransitionDenied) => {}
                    Err(other) => panic!(
                        "unexpected variant for ({parent}, {target}): {other:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn allows_transition_rejects_out_of_range_targets() {
        let shell = role_def(role::SHELL).unwrap();
        // Anything past the bitset width is rejected without UB or
        // wraparound (the function early-returns on target >= 64).
        assert!(!shell.allows_transition(64));
        assert!(!shell.allows_transition(100));
        assert!(!shell.allows_transition(u32::MAX));
    }

    #[test]
    fn allows_transition_handles_bit_63_safely() {
        // Bit 63 is the topmost bit of the u64 transitions bitset. No
        // defined role uses it today, so every role should return false
        // for target=63 — and crucially no shift overflow.
        for r in 0..role::COUNT as RoleId {
            assert!(!role_def(r).unwrap().allows_transition(63));
        }
    }

    #[test]
    fn transition_witness_target_matches_resulting_child_role() {
        // The TransitionAllowed witness's target() must equal the
        // child's role field — pinning that derive_child_perms doesn't
        // accidentally route through a different role.
        let shell = parent_in_role(role::SHELL);
        let t = check_transition(role::SHELL, role::NET_CLIENT).unwrap();
        assert_eq!(t.target(), role::NET_CLIENT);
        assert_eq!(t.source(), role::SHELL);
        let child = derive_child_perms(&shell, t, full_request());
        assert_eq!(child.permissions().role, t.target());
    }

    // ── three-way clamp ──────────────────────────────────────────────────

    #[test]
    fn child_inherits_role_defaults_when_parent_and_caller_are_full() {
        let shell = parent_in_role(role::SHELL);
        let child = try_spawn(&shell, role::NET_CLIENT, full_request()).unwrap();
        let p = child.permissions();
        assert_eq!(p.role, role::NET_CLIENT);
        // NET_CLIENT default has NETCH; SHELL_ALLOWED is ALL; caller is ALL.
        // → NETCH survives.
        assert!(p.perms_mask().contains(class::NETCH));
        // NET_CLIENT default doesn't have FS_RO, so FS_RO is not in child
        // perms even though SHELL_ALLOWED includes it.
        assert!(!p.perms_mask().contains(class::FS_RO));
    }

    #[test]
    fn parent_cap_clamps_child_perms() {
        // Synthetic parent: SHELL role, but allowed_perms narrowed via
        // a hypothetical pledge to exclude NETCH. Spawning NET_CLIENT
        // should then yield a child without NETCH.
        let shell_def = role_def(role::SHELL).unwrap();
        let parent = Permissions::from_masks(
            shell_def.default_perms,
            ClassMask::from_raw(shell_def.default_allowed.raw() & !class::raw::NETCH),
            role::SHELL,
        );
        let child = try_spawn(&parent, role::NET_CLIENT, full_request()).unwrap();
        let p = child.permissions();
        assert!(
            !p.perms_mask().contains(class::NETCH),
            "parent's narrowed cap must clamp NETCH out of child"
        );
        assert!(!p.allowed_perms_mask().contains(class::NETCH));
    }

    #[test]
    fn caller_request_clamps_child_perms() {
        // Parent SHELL with full caps, but caller asks for child perms
        // = STDIO only. Even though NET_CLIENT.default_perms has more,
        // the caller's request narrows.
        let shell = parent_in_role(role::SHELL);
        let req = PermsRequest {
            perms: class::STDIO,
            allowed_perms: class::ALL,
        };
        let child = try_spawn(&shell, role::NET_CLIENT, req).unwrap();
        let p = child.permissions();
        assert_eq!(p.perms_mask(), class::STDIO);
        // allowed stays at NET_CLIENT.default_allowed since requested_allowed = ALL.
        assert_eq!(
            p.allowed_perms_mask(),
            role_def(role::NET_CLIENT).unwrap().default_allowed
        );
    }

    #[test]
    fn caller_can_narrow_allowed_independently_of_perms() {
        let shell = parent_in_role(role::SHELL);
        let req = PermsRequest {
            perms: class::ALL,
            allowed_perms: class::STDIO,
        };
        let child = try_spawn(&shell, role::NET_CLIENT, req).unwrap();
        let p = child.permissions();
        // allowed_perms = target.default_allowed (STDIO|VMEM|PLEDGE)
        //               & parent.allowed_perms (ALL)
        //               & requested_allowed (STDIO)
        // → STDIO.
        assert_eq!(p.allowed_perms_mask(), class::STDIO);
        // perms is independent — NET_CLIENT default keeps its NETCH +
        // FUTEX + VMEM + STDIO + PLEDGE bits regardless of the narrow
        // allowed mask. This is the NET_CLIENT-shape we want: own
        // network reach, no network reach for grandchildren.
        let net_client_def = role_def(role::NET_CLIENT).unwrap();
        assert_eq!(p.perms_mask(), net_client_def.default_perms);
    }

    #[test]
    fn perms_axis_is_independent_of_allowed_axis() {
        // The two masks compute independently. Demonstrating: for a
        // shell parent spawning NET_CLIENT, perms always equals the
        // clamp of (target_default_perms ∩ parent_allowed ∩ requested_perms);
        // allowed_perms always equals (target_default_allowed ∩ parent_allowed ∩ requested_allowed).
        let shell = parent_in_role(role::SHELL);
        let nc_def = role_def(role::NET_CLIENT).unwrap();
        let inputs = [
            (class::ALL, class::ALL),
            (class::ALL, class::STDIO),
            (class::STDIO, class::ALL),
            (class::NETCH, class::STDIO),
            (ClassMask::EMPTY, class::ALL),
            (class::ALL, ClassMask::EMPTY),
        ];
        for (rp, ra) in inputs {
            let req = PermsRequest {
                perms: rp,
                allowed_perms: ra,
            };
            let child = try_spawn(&shell, role::NET_CLIENT, req).unwrap();
            let p = child.permissions();
            assert_eq!(
                p.perms_mask(),
                nc_def
                    .default_perms
                    .narrow(shell.allowed_perms_mask())
                    .narrow(rp),
                "perms wrong for requested=({:#x}, {:#x})",
                rp.raw(),
                ra.raw(),
            );
            assert_eq!(
                p.allowed_perms_mask(),
                nc_def
                    .default_allowed
                    .narrow(shell.allowed_perms_mask())
                    .narrow(ra),
                "allowed wrong for requested=({:#x}, {:#x})",
                rp.raw(),
                ra.raw(),
            );
        }
    }

    #[test]
    fn requested_zero_produces_empty_axes() {
        // PermsRequest::EMPTY → child with both masks empty. Legal but
        // useless — the child can do nothing and grant nothing.
        let shell = parent_in_role(role::SHELL);
        let child = try_spawn(&shell, role::WORKER, PermsRequest::EMPTY).unwrap();
        let p = child.permissions();
        assert!(p.perms_mask().is_empty());
        assert!(p.allowed_perms_mask().is_empty());
        assert_eq!(p.role, role::WORKER);
    }

    #[test]
    fn requested_cannot_widen_beyond_role_defaults() {
        // WORKER_PERMS has no NETCH/FS_RO; requesting full mask doesn't
        // bring those bits into the child.
        let shell = parent_in_role(role::SHELL);
        let child = try_spawn(&shell, role::WORKER, full_request()).unwrap();
        let p = child.permissions();
        assert!(!p.perms_mask().contains(class::NETCH));
        assert!(!p.allowed_perms_mask().contains(class::NETCH));
        assert!(!p.perms_mask().contains(class::FS_RO));
    }

    #[test]
    fn parent_perms_does_not_affect_child() {
        // Only parent.allowed_perms propagates. A parent with
        // perms=EMPTY (which the dispatch gate would normally reject
        // upstream — this test exercises the math in isolation) still
        // produces a full-default child.
        let shell_def = role_def(role::SHELL).unwrap();
        let zero_perms_parent =
            Permissions::from_masks(ClassMask::EMPTY, shell_def.default_allowed, role::SHELL);
        let child =
            try_spawn(&zero_perms_parent, role::WORKER, full_request()).unwrap();
        let p = child.permissions();
        let worker_def = role_def(role::WORKER).unwrap();
        assert_eq!(p.perms_mask(), worker_def.default_perms);
        assert_eq!(p.allowed_perms_mask(), worker_def.default_allowed);
    }

    #[test]
    fn requested_allowed_does_not_affect_child_perms() {
        // requested.allowed_perms = EMPTY zeroes child.allowed_perms but
        // leaves child.perms at the role default.
        let shell = parent_in_role(role::SHELL);
        let req = PermsRequest {
            perms: class::ALL,
            allowed_perms: ClassMask::EMPTY,
        };
        let child = try_spawn(&shell, role::NET_CLIENT, req).unwrap();
        let p = child.permissions();
        let nc_def = role_def(role::NET_CLIENT).unwrap();
        assert_eq!(p.perms_mask(), nc_def.default_perms);
        assert!(p.allowed_perms_mask().is_empty());
    }

    #[test]
    fn requested_perms_does_not_affect_child_allowed() {
        // Symmetric: requested.perms = EMPTY zeroes child.perms but
        // leaves child.allowed_perms at the role default.
        let shell = parent_in_role(role::SHELL);
        let req = PermsRequest {
            perms: ClassMask::EMPTY,
            allowed_perms: class::ALL,
        };
        let child = try_spawn(&shell, role::NET_CLIENT, req).unwrap();
        let p = child.permissions();
        let nc_def = role_def(role::NET_CLIENT).unwrap();
        assert!(p.perms_mask().is_empty());
        assert_eq!(p.allowed_perms_mask(), nc_def.default_allowed);
    }

    #[test]
    fn child_inherits_target_role_not_parent_role() {
        let shell = parent_in_role(role::SHELL);
        let child = try_spawn(&shell, role::WORKER, full_request()).unwrap();
        assert_eq!(child.permissions().role, role::WORKER);
        assert_ne!(child.permissions().role, role::SHELL);
    }

    #[test]
    fn derived_child_has_zero_pad_and_reserved() {
        // Even with garbage in the parent's tail fields, derive routes
        // through Permissions::new (via from_masks), which always
        // zeroes _pad and _reserved. Hygiene against state leakage
        // from a corrupted parent.
        let parent = Permissions {
            perms: class::raw::ALL,
            allowed_perms: class::raw::ALL,
            role: role::SHELL,
            _pad: 0xDEAD_BEEF,
            _reserved: [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444],
        };
        let child = try_spawn(&parent, role::WORKER, full_request()).unwrap();
        let p = child.permissions();
        assert_eq!(p._pad, 0);
        assert_eq!(p._reserved, [0; 2]);
    }

    #[test]
    fn net_client_naturally_has_perms_outside_allowed() {
        // The structural payoff of independent axes: NET_CLIENT child
        // has NETCH and FUTEX in perms but not in allowed_perms.
        let shell = parent_in_role(role::SHELL);
        let nc = try_spawn(&shell, role::NET_CLIENT, full_request()).unwrap();
        let p = nc.permissions();
        assert!(p.perms_mask().contains(class::NETCH));
        assert!(!p.allowed_perms_mask().contains(class::NETCH));
        assert!(p.perms_mask().contains(class::FUTEX));
        assert!(!p.allowed_perms_mask().contains(class::FUTEX));
    }

    #[test]
    fn fs_tool_naturally_has_perms_outside_allowed() {
        let shell = parent_in_role(role::SHELL);
        let ft = try_spawn(&shell, role::FS_TOOL, full_request()).unwrap();
        let p = ft.permissions();
        assert!(p.perms_mask().contains(class::FS_RO));
        assert!(!p.allowed_perms_mask().contains(class::FS_RO));
    }

    #[test]
    fn service_naturally_has_perms_outside_allowed() {
        // SERVICE is the most asymmetric role: NETCH, FS_RO, FUTEX,
        // STATS all in perms but not allowed. Only LOADER can spawn
        // SERVICE, so the parent has to be LOADER.
        let loader = parent_in_role(role::LOADER);
        let svc = try_spawn(&loader, role::SERVICE, full_request()).unwrap();
        let p = svc.permissions();
        assert!(p.perms_mask().contains(class::NETCH));
        assert!(!p.allowed_perms_mask().contains(class::NETCH));
        assert!(p.perms_mask().contains(class::FS_RO));
        assert!(!p.allowed_perms_mask().contains(class::FS_RO));
        assert!(p.perms_mask().contains(class::FUTEX));
        assert!(!p.allowed_perms_mask().contains(class::FUTEX));
        assert!(p.perms_mask().contains(class::STATS));
        assert!(!p.allowed_perms_mask().contains(class::STATS));
    }

    #[test]
    fn child_perms_peek_and_consume_yield_same_value() {
        // permissions() (peek) and into_permissions() (consume) must
        // observe the same Permissions. ChildPerms is Copy, so peek
        // doesn't drop the witness.
        let shell = parent_in_role(role::SHELL);
        let cp = try_spawn(&shell, role::WORKER, full_request()).unwrap();
        let peeked = cp.permissions();
        let consumed = cp.into_permissions();
        assert_eq!(peeked, consumed);
    }

    // ── grandchild scenario ──────────────────────────────────────────────

    /// The load-bearing example from the design doc: a shell spawns a
    /// network client; the network client can spawn a worker, but the
    /// worker doesn't inherit NETCH because NET_CLIENT's
    /// `default_allowed` excludes it.
    #[test]
    fn netch_dies_at_grandchild_boundary() {
        let shell = parent_in_role(role::SHELL);
        let net_client = try_spawn(&shell, role::NET_CLIENT, full_request())
            .unwrap()
            .into_permissions();
        // Sanity: net_client itself has NETCH.
        assert!(net_client.perms_mask().contains(class::NETCH));

        // Net client spawns a worker, requesting full perms — kernel
        // should still strip NETCH because NET_CLIENT.default_allowed
        // doesn't include it.
        let grandchild = try_spawn(&net_client, role::WORKER, full_request())
            .unwrap()
            .into_permissions();
        assert_eq!(grandchild.role, role::WORKER);
        assert!(
            !grandchild.perms_mask().contains(class::NETCH),
            "NETCH must not propagate to a worker grandchild"
        );
        assert!(!grandchild.allowed_perms_mask().contains(class::NETCH));
    }

    #[test]
    fn fs_ro_dies_at_grandchild_boundary_via_fs_tool() {
        let shell = parent_in_role(role::SHELL);
        let fs_tool = try_spawn(&shell, role::FS_TOOL, full_request())
            .unwrap()
            .into_permissions();
        assert!(fs_tool.perms_mask().contains(class::FS_RO));
        let grandchild = try_spawn(&fs_tool, role::WORKER, full_request())
            .unwrap()
            .into_permissions();
        assert!(!grandchild.perms_mask().contains(class::FS_RO));
    }

    #[test]
    fn worker_grandchild_cannot_spawn_at_all() {
        let shell = parent_in_role(role::SHELL);
        let worker = try_spawn(&shell, role::WORKER, full_request())
            .unwrap()
            .into_permissions();
        let r = try_spawn(&worker, role::WORKER, full_request());
        assert!(matches!(r, Err(SpawnDeny::TransitionDenied)));
    }

    #[test]
    fn service_to_worker_clamps_to_worker_default() {
        // SERVICE.default_allowed equals WORKER's default_perms
        // exactly, so a SERVICE-spawned WORKER lands at WORKER's
        // default verbatim — no leakage in either direction.
        let svc = parent_in_role(role::SERVICE);
        let worker = try_spawn(&svc, role::WORKER, full_request())
            .unwrap()
            .into_permissions();
        let worker_def = role_def(role::WORKER).unwrap();
        assert_eq!(worker.perms_mask(), worker_def.default_perms);
        assert_eq!(worker.allowed_perms_mask(), worker_def.default_allowed);
    }

    #[test]
    fn loader_to_shell_to_net_client_chain() {
        // Two-hop spawn: LOADER → SHELL → NET_CLIENT. Verifies caps
        // don't leak across hops and the final child lands at the
        // expected NET_CLIENT defaults — including the asymmetric
        // shape (NETCH in perms, not in allowed).
        let loader = parent_in_role(role::LOADER);
        let shell = try_spawn(&loader, role::SHELL, full_request())
            .unwrap()
            .into_permissions();
        let nc = try_spawn(&shell, role::NET_CLIENT, full_request())
            .unwrap()
            .into_permissions();
        let nc_def = role_def(role::NET_CLIENT).unwrap();
        assert_eq!(nc.perms_mask(), nc_def.default_perms);
        assert_eq!(nc.allowed_perms_mask(), nc_def.default_allowed);
        assert!(nc.perms_mask().contains(class::NETCH));
        assert!(!nc.allowed_perms_mask().contains(class::NETCH));
    }

    // ── pledge interaction ───────────────────────────────────────────────

    #[test]
    fn pledge_then_spawn_propagates_narrowed_cap() {
        // End-to-end "pledge restricts what children get": a SHELL
        // pledges NETCH out of allowed_perms, then spawns a NET_CLIENT.
        // The child must not have NETCH despite NET_CLIENT.default_perms
        // including it — the parent's narrowed cap propagates through
        // both axes of the three-way clamp.
        let shell = parent_in_role(role::SHELL);
        let narrowed = shell.pledge(PermsRequest {
            perms: class::ALL,
            allowed_perms: ClassMask::from_raw(class::raw::ALL & !class::raw::NETCH),
        });
        let child = try_spawn(&narrowed, role::NET_CLIENT, full_request())
            .unwrap()
            .into_permissions();
        assert!(!child.perms_mask().contains(class::NETCH));
        assert!(!child.allowed_perms_mask().contains(class::NETCH));
        // FUTEX (also in NET_CLIENT.default_perms) survives — only
        // NETCH was removed from the cap.
        assert!(child.perms_mask().contains(class::FUTEX));
    }

    #[test]
    fn pledge_to_zero_then_spawn_yields_empty_child() {
        // After pledge(EMPTY, EMPTY), parent.allowed_perms is empty.
        // Any subsequent spawn produces a child with empty masks since
        // the three-way clamp intersects everything against the zero
        // parent cap.
        let shell = parent_in_role(role::SHELL);
        let starved = shell.pledge(PermsRequest::EMPTY);
        let child = try_spawn(&starved, role::WORKER, full_request())
            .unwrap()
            .into_permissions();
        assert!(child.perms_mask().is_empty());
        assert!(child.allowed_perms_mask().is_empty());
        assert_eq!(child.role, role::WORKER);
    }

    #[test]
    fn pledge_does_not_affect_role_or_transitions() {
        // Pledge narrows permissions, never role identity. After a
        // self-pledge, the parent's role and the transitions it can
        // produce are unchanged.
        let shell = parent_in_role(role::SHELL);
        let narrowed = shell.pledge(PermsRequest {
            perms: class::STDIO,
            allowed_perms: class::STDIO,
        });
        assert_eq!(narrowed.role, role::SHELL);
        // SHELL → WORKER still allowed.
        assert!(try_spawn(&narrowed, role::WORKER, full_request()).is_ok());
        // SHELL → SHELL still denied (not in SHELL.transitions).
        assert!(matches!(
            try_spawn(&narrowed, role::SHELL, full_request()),
            Err(SpawnDeny::TransitionDenied)
        ));
    }

    #[test]
    fn spawn_then_pledge_can_narrow_below_role_defaults() {
        // A spawned child's pledge isn't bounded by its role's default
        // perms — it can shed any bit the role granted at spawn.
        let shell = parent_in_role(role::SHELL);
        let nc = try_spawn(&shell, role::NET_CLIENT, full_request())
            .unwrap()
            .into_permissions();
        assert!(nc.perms_mask().contains(class::NETCH));
        let narrow = nc.pledge(PermsRequest {
            perms: class::STDIO,
            allowed_perms: class::STDIO,
        });
        assert_eq!(narrow.perms_mask(), class::STDIO);
        assert!(!narrow.perms_mask().contains(class::NETCH));
    }

    #[test]
    fn pledge_is_monotonic_against_self() {
        // Repeated pledge calls only narrow further — anything still
        // present in the result must have been in the original perms.
        let p = parent_in_role(role::LOADER);
        let q = p.pledge(PermsRequest {
            perms: ClassMask::from_raw(class::raw::ALL & !class::raw::FS_RO),
            allowed_perms: class::ALL,
        });
        let r = q.pledge(PermsRequest {
            perms: ClassMask::from_raw(class::raw::ALL & !class::raw::NETCH),
            allowed_perms: class::ALL,
        });
        assert!(!r.perms_mask().contains(class::FS_RO));
        assert!(!r.perms_mask().contains(class::NETCH));
        // r.perms ⊆ p.perms (no bits in r that weren't in p).
        let p_raw = p.perms_mask().raw();
        let r_raw = r.perms_mask().raw();
        assert_eq!(r_raw & !p_raw, 0);
    }

    // ── shadow-path equivalence ──────────────────────────────────────────

    #[test]
    fn install_child_shadow_matches_enforcement_path_when_transition_would_be_allowed() {
        // Even though install_child_shadow exists for the *denied*
        // case, when the transition WOULD have succeeded, both paths
        // must produce identical Permissions. Catches divergence
        // between the shadow math and the enforcement math.
        let shell = parent_in_role(role::SHELL);
        let enforced = try_spawn(&shell, role::NET_CLIENT, full_request())
            .unwrap()
            .into_permissions();
        let shadow =
            install_child_shadow(&shell, role::NET_CLIENT, full_request()).unwrap();
        assert_eq!(enforced, shadow);
    }

    #[test]
    fn install_child_shadow_produces_realistic_perms_for_denied_transition() {
        // Worker tries to spawn worker (denied in production). Shadow
        // path produces what the child WOULD have had — full WORKER
        // defaults clamped against parent's allowed_perms.
        let worker = parent_in_role(role::WORKER);
        let shadowed =
            install_child_shadow(&worker, role::WORKER, full_request()).unwrap();
        assert_eq!(shadowed.role, role::WORKER);
        // worker's allowed_perms = WORKER_PERMS, target default = WORKER_PERMS,
        // request = ALL → child perms = WORKER_PERMS.
        assert_eq!(shadowed.perms_mask(), WORKER_PERMS);
    }

    #[test]
    fn install_child_shadow_rejects_unknown_target() {
        let shell = parent_in_role(role::SHELL);
        // Even shadow mode rejects out-of-range role IDs — that's
        // caller bug territory, not a policy denial.
        let r = install_child_shadow(&shell, 9999, full_request());
        assert_eq!(r, Err(SpawnDeny::UnknownTargetRole));
    }

    // ── registry sanity ──────────────────────────────────────────────────

    #[test]
    fn roles_table_has_one_entry_per_role_id() {
        assert_eq!(ROLES.len(), role::COUNT);
    }

    #[test]
    fn norole_is_zero_everywhere() {
        let nr = &ROLES[role::NOROLE as usize];
        assert!(nr.default_perms.is_empty());
        assert!(nr.default_allowed.is_empty());
        assert_eq!(nr.transitions, 0);
    }

    #[test]
    fn leaf_roles_have_perms_equal_to_allowed() {
        // A "leaf" role is one whose default_perms equals its
        // default_allowed — meaning every cap it has is also a cap
        // it can grant. Everything that's terminal (transitions = 0)
        // should be a leaf, since there's no point granting bits to
        // children that can never be spawned. Mostly a sanity check
        // on the table, not a hard rule.
        for (i, def) in ROLES.iter().enumerate() {
            if def.transitions == 0 && i != role::NOROLE as usize {
                assert_eq!(
                    def.default_perms, def.default_allowed,
                    "terminal role {i} should have perms == allowed (no caps to hold for ungrantable children)"
                );
            }
        }
    }

    #[test]
    fn transitions_only_target_valid_role_ids() {
        for (i, def) in ROLES.iter().enumerate() {
            for target in 0..64 {
                if def.transitions & (1u64 << target) != 0 {
                    assert!(
                        (target as usize) < role::COUNT,
                        "role {i} has transition to undefined role {target}"
                    );
                    // No transitions to NOROLE — that role is the
                    // sentinel, never an actual destination.
                    assert_ne!(
                        target as RoleId,
                        role::NOROLE,
                        "role {i} declares a transition to NOROLE"
                    );
                }
            }
        }
    }

    // ── witness shape ────────────────────────────────────────────────────

    #[test]
    fn child_perms_witness_layout_is_transparent() {
        // ChildPerms is repr(transparent) over Permissions — same
        // size, no overhead. Protects the "witness is free" property.
        assert_eq!(
            core::mem::size_of::<ChildPerms>(),
            core::mem::size_of::<Permissions>()
        );
    }

    #[test]
    fn transition_allowed_witness_is_small() {
        // source(4) + target(4) + target_def: &'static RoleDef(8) = 16
        // bytes on a 64-bit target. Carrying the resolved RoleDef
        // ref lets `derive_child_perms` skip the registry lookup and
        // eliminates its `expect` panic path.
        assert_eq!(core::mem::size_of::<TransitionAllowed>(), 16);
    }
}
