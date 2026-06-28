//! Audit logging for permission denials — the events the gates push
//! when they EPERM a syscall.
//!
//! Two surfaces:
//!
//! - [`DenialEvent`] — the wire shape for a single denial. `repr(C)`
//!   so the kernel-side ring can serialize bytes straight into a
//!   user buffer via the `query_denial_log` syscall. Two variants:
//!   [`PermDeny`](DenialEvent::PermDeny) for the dispatch-site
//!   bitmask gate, [`RoleDeny`](DenialEvent::RoleDeny) for the
//!   role-transition gate inside `create_process_v2`.
//! - [`DenialSink`] — the trait the gates push into. Any
//!   `&mut impl DenialSink` will do; the production implementor is
//!   the kernel-wide bounded ring
//!   `orbit-core::denial_ring::DenialRing`, tests use a `Vec`-backed
//!   sink.
//!
//! Entries are drained by `query_denial_log`.

use crate::perms::role::RoleId;

/// Maximum entries the kernel-wide denial ring retains. Chosen for
/// "enough to capture a regression's worth of context" while staying
/// small enough that a full snapshot fits in a single ~3 KiB user
/// buffer for `query_denial_log` (64 × 48 B = 3072 B, no header —
/// the reply is just a packed sequence of `DenialEvent`s). Older
/// events are evicted on push.
///
/// Also pinned as the wire-shape upper bound for `query_denial_log`'s
/// reply, so user code can size its buffer up-front.
pub const DENIAL_RING_CAPACITY: usize = 64;

/// A single permission denial. Pushed onto the kernel-wide denial
/// ring (and the per-process counters incremented) whenever a gate
/// EPERMs a syscall.
///
/// `repr(C, u32)` so the layout is stable across user/kernel and the
/// variant tag is grep-able in a hex dump. Fields within each variant
/// are laid out **largest-alignment-first** — that lets `repr(C)`
/// compute the natural padding without any explicit `_pad` fields,
/// keeping the source readable while still pinning the wire shape.
/// Each variant occupies 48 bytes (4 discriminant + 4 trailing pad +
/// 40 payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, u32)]
pub enum DenialEvent {
    /// The dispatch-site bitmask gate EPERMed the syscall.
    /// Recorded by the gate before it short-circuits the syscall
    /// with `-EPERM`; no handler runs.
    PermDeny {
        /// Class bit the syscall required, from `Permissions::class_for`.
        /// A reader can match this against `class::raw::*` to identify
        /// which permission was missing.
        required_class: u64,
        /// Effective `perms` mask of the calling process at the
        /// moment of the denial. The complement of `required_class`
        /// against this is "what role/pledge change would have made
        /// the call succeed."
        perms: u64,
        /// Monotonic ticks at the time of the event. Same time domain
        /// as `get_micros` / `query_stats` — readers should not
        /// interpret as wallclock.
        time_ticks: u64,
        /// Thread ID of the caller.
        tid: u32,
        /// Syscall number the gate denied.
        sysno: u32,
        /// Role of the calling process at the time of the gate. Symmetric
        /// with [`RoleDeny`](Self::RoleDeny)'s `source_role` — lets a
        /// log reader answer "which role's `default_perms` is too
        /// tight?" without joining against external process metadata.
        /// Lives in the u32 group between `sysno` and `pid` to fit the
        /// existing variant size (40 B payload, 48 B total) without
        /// bumping the layout pin.
        source_role: RoleId,
        /// Process ID of the caller.
        pid: u16,
        // 2 bytes of trailing padding to align the variant size to 8.
    } = 0,

    /// `create_process_v2`'s role-transition gate EPERMed the
    /// spawn. Recorded by the manager-side handler before it
    /// returns `-EPERM`; no child is created.
    RoleDeny {
        /// Monotonic ticks at the time of the event.
        time_ticks: u64,
        /// Reserved for future axes (e.g. additional `SpawnDeny`
        /// variants, label fingerprints). Must be zero today. Same
        /// forward-compat *idea* as
        /// [`crate::perms::Permissions::_reserved`] — old kernels
        /// stamp zero, new readers can repurpose — but a single
        /// `u64` slot here vs. that field's `[u64; 2]`. Bump the
        /// variant size pin in `denial_event_layout_is_pinned` if
        /// you grow this slot.
        _reserved: u64,
        /// Parent thread's tid (the one that issued the spawn).
        tid: u32,
        /// Parent's role at the time of the call.
        source_role: RoleId,
        /// Role the spawn was targeting.
        target_role: RoleId,
        /// Discriminant of the [`crate::roles::SpawnDeny`]-shaped reason
        /// for the denial: `0` = `UnknownParentRole`, `1` =
        /// `UnknownTargetRole`, `2` = `TransitionDenied`. Inlined as
        /// `u32` so the wire layout doesn't depend on a Rust enum's
        /// internal layout.
        deny_reason: u32,
        /// Parent process's pid.
        pid: u16,
        // 2 bytes of trailing padding to align the variant size to 8.
    } = 1,
}

/// `deny_reason` values for [`DenialEvent::RoleDeny`]. Mirror the
/// discriminants of `orbit_core::roles::SpawnDeny`; pinned here as
/// `u32` constants because the on-the-wire shape can't depend on a
/// Rust enum's representation.
pub mod deny_reason {
    pub const UNKNOWN_PARENT_ROLE: u32 = 0;
    pub const UNKNOWN_TARGET_ROLE: u32 = 1;
    pub const TRANSITION_DENIED: u32 = 2;
}

/// Sink for denial events. Production sites use the kernel-wide
/// bounded ring (`orbit-core::denial_ring::DenialRing`); tests use
/// a `Vec`-backed sink to inspect the event stream.
///
/// Threading model: implementors are free to require external
/// synchronization. The kernel-side ring lives behind the manager
/// lock, which serializes both gates by construction.
pub trait DenialSink {
    /// Push an event. Implementors that have a fixed capacity (e.g.
    /// the production ring) evict the oldest event on push. Drops
    /// from the sink-side don't need to be observable to the
    /// pusher — the ring is best-effort.
    fn push(&mut self, event: DenialEvent);
}

/// `()` is the no-op sink. Useful for tests where the gate is being
/// exercised purely for its return value and the audit push is
/// uninteresting.
impl DenialSink for () {
    fn push(&mut self, _event: DenialEvent) {}
}

/// Caller-side metadata needed to populate a [`DenialEvent`]. The
/// gate functions themselves don't have access to the kernel's
/// `(pid, tid, time)` triple; the dispatch site reads it from the
/// hart context and bundles it here.
///
/// Internal-only — doesn't cross the syscall boundary, so no
/// `repr(C)` / wire-shape concerns. Implementors of the gate
/// functions can ignore the fields they don't need (pure-logic
/// tests typically pass `GateContext::ZERO`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateContext {
    /// PID of the calling process. Lands in the event's `pid` field.
    pub pid: u16,
    /// TID of the calling thread. Lands in the event's `tid` field.
    pub tid: u32,
    /// Monotonic ticks at the time of the gate call. Lands in the
    /// event's `time_ticks` field.
    pub time_ticks: u64,
}

impl GateContext {
    /// All-zero context. Useful for tests; production callers
    /// populate from the hart context.
    pub const ZERO: Self = Self {
        pid: 0,
        tid: 0,
        time_ticks: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the variant sizes — both `PermDeny` and `RoleDeny` are
    /// 48 bytes including the 4-byte discriminant + 4 bytes of
    /// trailing pad (40-byte payload), so 64 events fit in 3 KiB.
    /// If a future field grows the variant, this fails loudly so
    /// the wire-shape impact is visible.
    #[test]
    fn denial_event_layout_is_pinned() {
        // `repr(C, u32)` enum layout: 4-byte discriminant + variant
        // payload, padded to the variant's alignment. The largest
        // variant determines size_of::<DenialEvent>().
        assert_eq!(core::mem::size_of::<DenialEvent>(), 48);
        assert_eq!(core::mem::align_of::<DenialEvent>(), 8);
    }

    #[test]
    fn denial_ring_capacity_pinned() {
        // DENIAL_RING_CAPACITY is part of the ABI: user code sizing a
        // query_denial_log buffer relies on this number.
        assert_eq!(DENIAL_RING_CAPACITY, 64);
    }

    #[test]
    fn deny_reason_constants_match_documented_values() {
        // Pinned: kernel-side log readers and on-host analysis tools
        // assume these specific u32 values for SpawnDeny variants.
        assert_eq!(deny_reason::UNKNOWN_PARENT_ROLE, 0);
        assert_eq!(deny_reason::UNKNOWN_TARGET_ROLE, 1);
        assert_eq!(deny_reason::TRANSITION_DENIED, 2);
    }

    /// Vec-backed sink for tests — the production sink (`DenialRing`
    /// in orbit-core) requires alloc + manager-lock context, so unit
    /// tests at the orbit-abi level use this.
    #[derive(Default)]
    struct VecSink(alloc::vec::Vec<DenialEvent>);

    impl DenialSink for VecSink {
        fn push(&mut self, event: DenialEvent) {
            self.0.push(event);
        }
    }

    #[test]
    fn vec_sink_records_events_in_order() {
        // DenialSink trait sanity — a sink that just collects should
        // see events in the order push was called. Plain Vec semantics,
        // pinned because the production ring promises chronological
        // order in the snapshot reply.
        let mut sink = VecSink::default();
        let e0 = DenialEvent::PermDeny {
            required_class: 0x40,
            perms: 0x1,
            time_ticks: 100,
            tid: 1,
            sysno: 4097,
            source_role: 6,
            pid: 1,
        };
        let e1 = DenialEvent::RoleDeny {
            time_ticks: 200,
            _reserved: 0,
            tid: 1,
            source_role: 6,
            target_role: 3,
            deny_reason: deny_reason::TRANSITION_DENIED,
            pid: 1,
        };
        sink.push(e0);
        sink.push(e1);
        assert_eq!(sink.0.len(), 2);
        assert_eq!(sink.0[0], e0);
        assert_eq!(sink.0[1], e1);
    }

    #[test]
    fn unit_sink_is_a_noop() {
        // `impl DenialSink for ()` — push compiles and is a no-op.
        // Used in enforcement-mode contexts where logging is redundant.
        let mut sink = ();
        sink.push(DenialEvent::PermDeny {
            required_class: 0,
            perms: 0,
            time_ticks: 0,
            tid: 0,
            sysno: 0,
            source_role: 0,
            pid: 0,
        });
    }

    extern crate alloc;
}
