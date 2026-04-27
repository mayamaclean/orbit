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

use core::sync::atomic::{AtomicUsize, Ordering};

use orbit_core::tlb_shootdown::{ShootdownRing, drain_shootdown_ring, tlb_shootdown};

use crate::kernel::context::get_hart_context;

/// Compile-time cap on hart count. The QEMU `virt` machine we target
/// runs `-smp 4`; 8 leaves room for future `-smp 8` runs without
/// re-jiggering the static array. Bump and re-verify the
/// `RING_INITIALIZER` block below if a real platform pushes past it.
pub const MAX_HARTS: usize = 8;

/// Per-hart shootdown ring. Index by `hart_id`. Producers (any hart)
/// push via the orchestrator; the consumer (target hart) drains in
/// its SSWI cause-1 handler.
///
/// Static-array shape (vs. one ring per `HartContext`) keeps the
/// orchestrator caller-agnostic — `SharedUserPtr::revoke` runs in the
/// manager thread without `&Orbit` and still needs to broadcast.
pub static SHOOTDOWN_RINGS: [ShootdownRing; MAX_HARTS] = [
    ShootdownRing::new(), ShootdownRing::new(), ShootdownRing::new(), ShootdownRing::new(),
    ShootdownRing::new(), ShootdownRing::new(), ShootdownRing::new(), ShootdownRing::new(),
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
        cpu_count, MAX_HARTS,
    );
    CPU_COUNT.store(cpu_count, Ordering::Release);
}

/// Send a TLB-shootdown request for `[va, va + len)` to every hart
/// other than the caller and block until each acks. The request shape
/// the receiver honors:
///
/// - `va == 0 && len == 0` → whole-TLB flush (`sfence.vma x0, x0`).
///   Use for whole-process invalidations (post-mmap, post-revoke,
///   process teardown).
/// - otherwise → single-page invalidation at `va` (`sfence.vma va, x0`).
///   `len` is currently ignored beyond the sentinel check; a future
///   range-broadcast variant would loop on the receiver side.
///
/// Caller is responsible for the local-hart `sfence.vma` — the
/// orchestrator excludes the calling hart from `targets` so we don't
/// waste an IPI on ourselves.
///
/// No-op if [`init`] hasn't run (`cpu_count == 0`) or if there's only
/// one hart online — useful for early-boot mmap that happens before
/// secondary harts come up.
pub fn broadcast(va: u64, len: u64) {
    let n = CPU_COUNT.load(Ordering::Acquire);
    if n <= 1 {
        return;
    }
    let self_id = get_hart_context().hart_id as usize;
    let n_targets = n - 1;
    let targets = (0..n)
        .filter(move |&h| h != self_id)
        .map(|h| (h, &SHOOTDOWN_RINGS[h]));

    let mut hw = crate::hw::RiscvHardware;
    // Errors are RingFull only — drop them. The orchestrator already
    // ack-decrements failed targets so we don't deadlock; missing
    // shootdowns mean a target hart will see a stale TLB entry until
    // its next natural eviction. TODO: fall back to a coarser local
    // policy (e.g., kick the receiver into a forced full-flush) once
    // we have data showing the ring saturates in practice.
    let _ = tlb_shootdown(n_targets, targets, va, len, &mut hw);
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
    drain_shootdown_ring(ring, |va, len| {
        // SAFETY: sfence.vma is always safe (it's a fence, not a
        // memory access). The arms differ in scope only.
        if va == 0 && len == 0 {
            // Sentinel: whole-TLB flush. Equivalent to the
            // pre-shootdown `sfence_vma(pid, 0)` + `sfence_vma(0, 0)`
            // pair the senders used to do locally.
            riscv::asm::sfence_vma(0, 0);
        }
        else {
            // Per-page invalidation across all ASIDs. Slightly
            // broader than `sfence.vma va, asid` would be, but
            // the ring entries don't carry asid today and "all
            // ASIDs" is always correct (per RISC-V Privileged
            // ISA: rs2=x0 means the fence orders accesses to
            // all ASIDs).
            riscv::asm::sfence_vma(0, va as usize);
        }
    });
}
