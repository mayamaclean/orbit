//! Layer-2 wiring for [`orbit_core::tlb_shootdown`].
//!
//! Owns the per-hart [`ShootdownRing`] statics and the kmain-side glue:
//! [`broadcast`] (orchestrator entry point — sender side) and
//! [`drain_local`] (SSWI receiver — drains and `sfence.vma`s).
//!
//! The orbit-core protocol module is hardware-free; this is where the
//! actual `sfence.vma` instruction lives. Producers call [`broadcast`]
//! after modifying a user PTE, the SSWI handler in `s_trap` calls
//! [`drain_local`] before returning to the interrupted thread.
//!
//! # Self-fence rule
//!
//! [`broadcast`] does **not** flush the calling hart's TLB —
//! `tlb_shootdown` deliberately excludes the local hart so we don't
//! waste an IPI on ourselves. Every caller is responsible for issuing
//! its own local `sfence.vma` (typically the same one it was already
//! doing pre-shootdown).

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use orbit_core::tlb_shootdown::{FlushScope, ShootdownEntry, ShootdownRing, drain_shootdown_ring};
use process::AckCounter;
use tracing::warn;

use crate::kernel::context::get_hart_context;

/// Compile-time cap on hart count. Homed in orbit-core (the lowest crate
/// the `manager` crate's `READY_INBOXES` can also reach); re-exported
/// here so existing `shootdown::MAX_HARTS` references keep working.
/// Bump there + re-verify the `RING_INITIALIZER` block below if a real
/// platform pushes past it.
pub use orbit_core::MAX_HARTS;

/// Per-hart shootdown ring. Index by `hart_id`. Producers (any hart)
/// push via the orchestrator; the consumer (target hart) drains in
/// its SSWI cause-1 handler.
///
/// Static-array shape (vs. one ring per `HartContext`) keeps the
/// orchestrator caller-agnostic — `SharedUserPtr::revoke` runs in the
/// manager thread without `&Orbit` and still needs to broadcast.
pub static SHOOTDOWN_RINGS: [ShootdownRing; MAX_HARTS] = [
    ShootdownRing::new(),
    ShootdownRing::new(),
    ShootdownRing::new(),
    ShootdownRing::new(),
    ShootdownRing::new(),
    ShootdownRing::new(),
    ShootdownRing::new(),
    ShootdownRing::new(),
];

/// Online hart count, set by [`init`] during boot. Used to bound the
/// broadcast target iterator. Atomic so [`broadcast`] doesn't need
/// `&Orbit`. Written once at boot, read on every shootdown.
pub static CPU_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Capture the boot-time online hart count. Must run on hart 0 in
/// `rust_main` after the DTB walk has resolved the value, *before*
/// any user PTE modification can fire a broadcast.
pub fn init(cpu_count: usize) {
    assert!(
        cpu_count as usize <= MAX_HARTS,
        "shootdown::init: cpu_count={} exceeds MAX_HARTS={}",
        cpu_count,
        MAX_HARTS,
    );
    CPU_COUNT.store(cpu_count, Ordering::Release);
}

/// Flips to `true` once hart 0 has issued the SECONDARY_GO release
/// + IPI kicks in `k_smpstart`. Until then the secondary harts spin
/// in `secondary_rust_setup` and can't drain their shootdown rings,
/// so a [`broadcast`] would `wait_zero_spin` forever waiting for an
/// ack that never arrives.
///
/// Pre-flip there's also no risk of stale TLB entries on remote
/// harts (none have run any user PTE yet), so the broadcast is
/// unnecessary. Both correctness and liveness say "no-op until
/// secondaries are alive."
static SECONDARIES_KICKED: AtomicBool = AtomicBool::new(false);

/// Hart 0 calls this exactly once after `SECONDARY_GO`/`supervisor_wake_hart`
/// in `k_smpstart`, signalling that subsequent broadcasts may safely
/// wait for acks. Idempotent — second call is a no-op.
pub fn mark_secondaries_kicked() {
    SECONDARIES_KICKED.store(true, Ordering::Release);
}

/// Send a per-process TLB-shootdown to every hart other than the caller
/// and block until each acks. `asid` is the target process's pid; the
/// receiver honors:
///
/// - `va == 0` → whole-ASID flush (`sfence.vma x0, asid`): drop every
///   cached translation for that process. Used for post-mmap,
///   post-revoke, and teardown invalidations.
/// - otherwise → single page (`sfence.vma va, asid`). `len` is
///   informational today; a future range variant would loop the
///   receiver over `[va, va + len)`.
///
/// For a whole-TLB flush across every address space use
/// [`broadcast_all`] instead.
///
/// Caller is responsible for the local-hart fence — the orchestrator
/// excludes the calling hart from `targets` so we don't waste an IPI on
/// ourselves.
///
/// No-op if [`init`] hasn't run (`cpu_count == 0`) or if there's only
/// one hart online — useful for early-boot mmap that happens before
/// secondary harts come up.
pub fn broadcast(asid: u16, va: u64, len: u64) {
    broadcast_scope(FlushScope::Asid(asid), va, len);
}

/// Whole-TLB cross-hart shootdown: every other hart drops every cached
/// translation in every address space (`sfence.vma x0, x0`). The blunt
/// instrument — reserved for kernel-global mapping changes that aren't
/// scoped to a single process. Per-process invalidations should pass
/// their pid to [`broadcast`].
pub fn broadcast_all() {
    broadcast_scope(FlushScope::All, 0, 0);
}

fn broadcast_scope(scope: FlushScope, va: u64, len: u64) {
    let n = CPU_COUNT.load(Ordering::Acquire);
    if n <= 1 {
        return;
    }
    // Boot-window guard: skip until hart 0 has woken the secondaries.
    // Pre-kick they can't drain the shootdown ring (still spinning in
    // `secondary_rust_setup`), and they have no TLB entries to
    // invalidate yet anyway, so the IPI is both deadlock-prone and
    // unnecessary.
    if !SECONDARIES_KICKED.load(Ordering::Acquire) {
        return;
    }
    let self_id = get_hart_context().hart_id as usize;

    // Inlined `tlb_shootdown` orchestrator so we can spin on the ack
    // counter ourselves and surface a warn! when the wait exceeds a
    // threshold — diagnostics for the silent-hang scenario where a
    // remote hart never delivers our SSWI and the manager wedges
    // here forever (fans pegged, no logs).
    let ack = AckCounter::new(n - 1);
    let mut failed: u32 = 0;

    for hart_id in 0..n {
        if hart_id == self_id {
            continue;
        }
        let ring: &'static ShootdownRing = &SHOOTDOWN_RINGS[hart_id];
        match ring.push_ref() {
            Ok(mut slot) => {
                *slot = ShootdownEntry::Req {
                    scope,
                    va,
                    len,
                    ack: ack.clone(),
                };
                drop(slot);
                crate::supervisor_wake_hart(hart_id);
            }
            Err(_) => {
                ack.decrement();
                failed = failed.saturating_add(1);
            }
        }
    }

    // Spin with a stuck-detection threshold. ~1M iterations of the
    // tight load+spin_loop on QEMU is roughly 10–50 ms of wall time —
    // a real shootdown completes in microseconds, so anything above
    // this is a genuine wedge worth telling about. After the first
    // warn we keep spinning (the wedge may be transient and we can
    // still recover) but bump the threshold geometrically so the log
    // doesn't flood.
    const FIRST_WARN_AT: u64 = 1_000_000;
    let mut spins: u64 = 0;
    let mut next_warn: u64 = FIRST_WARN_AT;
    while ack.load() != 0 {
        core::hint::spin_loop();
        spins = spins.wrapping_add(1);
        if spins == next_warn {
            warn!(
                "shootdown::broadcast wedged on hart {} after {} spins: \
                 va={:#x} len={:#x} pending_acks={} failed={}",
                self_id,
                spins,
                va,
                len,
                ack.load(),
                failed,
            );
            next_warn = next_warn.saturating_mul(4);
        }
    }
    if spins >= FIRST_WARN_AT {
        warn!(
            "shootdown::broadcast on hart {} unblocked after {} spins \
             (va={:#x} len={:#x} failed={})",
            self_id, spins, va, len, failed,
        );
    }

    // Errors are RingFull only — failed targets already ack-
    // decremented above so we don't deadlock; missing shootdowns
    // mean a target hart will see a stale TLB entry until its next
    // natural eviction. TODO: fall back to a coarser local policy
    // (forced full-flush) once we have data showing the ring
    // saturates in practice.
    let _ = failed;
}

/// Drain the local hart's shootdown ring, executing one `sfence.vma`
/// per request and decrementing each carried [`AckCounter`]. Called
/// from `s_trap`'s SSWI cause-1 arm.
///
/// Trap-context-safe: only atomic stores + `sfence.vma`. No
/// allocations, no locks, no reentry.
pub fn drain_local() {
    let hart_id = get_hart_context().hart_id as usize;
    debug_assert!(
        hart_id < MAX_HARTS,
        "shootdown::drain_local: hart_id={} >= MAX_HARTS",
        hart_id,
    );
    let ring = &SHOOTDOWN_RINGS[hart_id];
    drain_shootdown_ring(ring, |scope, va, _len| {
        // SAFETY: SFENCE.VMA is a fence, not a memory access — always
        // safe to issue. The arms differ in scope only.
        match scope {
            FlushScope::All => {
                // Whole-TLB flush: every leaf in every address space.
                riscv::asm::sfence_vma_all();
            }
            FlushScope::Asid(asid) if va == 0 => {
                // Whole-ASID flush — drop every leaf for this process.
                crate::kernel::tlb::flush_asid(asid as usize);
            }
            FlushScope::Asid(asid) => {
                // Single page in one ASID. Both operands are non-zero
                // (a live ASID is never 0 and `va != 0` here), so the
                // crate's register-form helper encodes the right scope.
                riscv::asm::sfence_vma(asid as usize, va as usize);
            }
        }
    });
}
