//! `k_serial` kernel thread + the thingbuf MPSC ring that producers
//! push trace/log lines into from any hart.
//!
//! Mirror of [`k_gpu`](super::k_gpu) for the UART side. Before this
//! existed, every `serial::print!` call serialized on `serial::SERIAL`'s
//! spinlock. At trace volume that turned the UART path into a global
//! bottleneck — concrete repro: orbit-top-std under eza-stress wedged
//! console echo because three harts contended on the lock for tens of
//! milliseconds at a time. Routing tracing through a lock-free ring +
//! single-consumer drain folds N concurrent producers into one drain
//! pass that takes the lock once.
//!
//! Flow: tracing's `emit` (in `crate::ktrace`) formats into a
//! [`LineBuf`](crate::ktrace), then [`push_chunk`] copies the bytes
//! into a free [`SerialChunk`] slot via `thingbuf::StaticThingBuf`'s
//! lock-free `push_ref`. The manager nudges k_serial at end-of-pass
//! ([`crate::kernel::Orbit::nudge_serial_if_pending`]) the same way it
//! nudges k_gpu — one TICKLE per pass, not one per push, so a syscall-
//! trace burst doesn't wake-flood the scheduler.
//!
//! Producers that fire pre-scheduler (early bl/kmain trampoline,
//! [`crate::ktrace::emit`] before [`mark_ready`] runs) fall back to
//! `serial::print!` directly via the `is_ready` check — there's no
//! kthread to drain the ring during that window.

use core::sync::atomic::{AtomicBool, Ordering};

use process::ThreadState;
use thingbuf::StaticThingBuf;
use tracing::info;

use crate::exit_thread_with_state;

/// Max bytes per chunk. Sized to fit `ktrace::LINE_BUF_LEN` (512) with
/// headroom for any future longer trace line; producers that need
/// more bytes split across chunks like k_gpu's `WriteChunk` does.
pub const CHUNK_BYTES: usize = 1024;

/// Depth of the ring. 64 slots × ~1 KiB ≈ 64 KiB total. Bursts during
/// the densest trace activity (multi-hart eza-stress runs) peaked
/// around ~30 lines/ms; 64 slots gives ~2 ms of buffering at that rate
/// before producers fall back to the spinlock path on overflow.
pub const RING_CAP: usize = 64;

/// One queue slot. Fixed-size so slot reuse (`push_ref`/`pop_ref`)
/// doesn't need allocator traffic.
#[derive(Clone)]
pub struct SerialChunk {
    /// Valid byte count in `bytes`. `0` is the [`Default`] sentinel
    /// the drain skips without touching the UART.
    pub len: u16,
    pub bytes: [u8; CHUNK_BYTES],
}

impl Default for SerialChunk {
    fn default() -> Self {
        Self { len: 0, bytes: [0u8; CHUNK_BYTES] }
    }
}

/// Global producer ring. Written by any hart from any context (trap,
/// manager, kthread); drained only by `k_serial`.
pub static SERIAL_RING: StaticThingBuf<SerialChunk, RING_CAP> = StaticThingBuf::new();

/// `true` once `k_serial` has started running and is draining the ring.
/// Producers consult this to decide whether to push (kthread up) or
/// fall back to the synchronous `serial::print!` spinlock path
/// (pre-scheduler / boot window).
static READY: AtomicBool = AtomicBool::new(false);

/// `true` once the k_serial kthread is live and draining.
pub fn is_ready() -> bool {
    READY.load(Ordering::Acquire)
}

/// Mark k_serial ready. Called once from the kthread entrypoint after
/// the first park/drain cycle is set up.
fn mark_ready() {
    READY.store(true, Ordering::Release);
}

/// Push `bytes` (≤ [`CHUNK_BYTES`]) onto the ring as one chunk.
/// Returns `false` if the ring was full or the chunk would be empty;
/// caller should fall back to the synchronous serial path on `false`
/// to keep the line from being dropped.
///
/// Lock-free; safe from any context including trap handlers.
pub fn push_chunk(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let Ok(mut slot) = SERIAL_RING.push_ref()
    else {
        return false;
    };
    let len = core::cmp::min(bytes.len(), CHUNK_BYTES);
    slot.len = len as u16;
    slot.bytes[..len].copy_from_slice(&bytes[..len]);
    true
}

/// Kernel-thread entrypoint. Spawned once via
/// [`crate::kernel::Orbit::setup_serial_kthread`] before device init
/// fans out trace volume. Never returns (exits as `Exited` on fatal
/// error, but there's no path that produces one today).
#[unsafe(no_mangle)]
pub extern "C" fn k_serial(_a0: usize) {
    unsafe {
        riscv::register::sstatus::clear_sie();
    }

    info!("k_serial: ready");
    mark_ready();

    unsafe { serial::acquire_serial() };
    loop {
        unsafe {
            riscv::register::sstatus::clear_sie();
        }
        while let Some(chunk) = SERIAL_RING.pop_ref() {
            let len = chunk.len as usize;
            if len == 0 {
                continue;
            }
            let bytes = &chunk.bytes[..len];
            // Producers (ktrace::emit) format via core::fmt::Write
            // into UTF-8 first, so this from_utf8 is normally Ok.
            // On a non-UTF-8 chunk we drop the line — that's a bug
            // at the call site, not something to paper over here.
            if let Ok(s) = core::str::from_utf8(bytes) {
                unsafe { serial::print_no_crit(format_args!("{s}")) };
            }
        }

        // Park with a ~50 ms wake deadline (timebase = 10 MHz →
        // 500k ticks). A producer that pushes onto SERIAL_RING can
        // wake us via `WakeEvent::Serial` (handled by the manager's
        // end-of-pass nudge in `Orbit::nudge_serial_if_pending`).
        let wake_at = riscv::register::time::read64().wrapping_add(500_000) as usize;
        crate::kernel::context::kthread_park(ThreadState::Suspended, wake_at);
    }

    // Unreachable today; keep the symmetry with k_gpu in case a
    // fatal-error branch is added later.
    #[allow(unreachable_code)]
    unsafe {
        serial::release_serial();
        exit_thread_with_state(ThreadState::Exited)
    };
}
