//! `k_gpu` kernel thread + the thingbuf MPSC ring that producers push
//! into from any hart.
//!
//! Flow: `console_write` syscalls (and the kernel's trace shim) build
//! a [`Cmd`] and push it onto [`CONSOLE_RING`] via
//! `thingbuf::StaticThingBuf::push_ref`, then SSWI `k_gpu`'s hart. The
//! UART-RX PLIC handler enqueues a `CycleActive` command the same way.
//! `k_gpu` drains, mutates its owned [`Display`], and if anything
//! changed issues a `TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH` on the
//! virtio-gpu control queue before parking.

use alloc::boxed::Box;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use process::ThreadState;
use thingbuf::StaticThingBuf;
use tracing::{error, info};

use crate::drivers::display::{Display, Source};
use crate::drivers::virtio_gpu_dev;
use crate::exit_thread_with_state;

/// Max bytes per `Cmd`. Matches POSIX `PIPE_BUF` atomicity — writes
/// up to this size are committed in one ring slot; larger writes are
/// split across multiple pushes by the syscall layer.
pub const CMD_BYTES: usize = 4096;

/// Depth of the ring. 8 slots × (4 KiB + small header) ≈ 32 KiB total.
pub const RING_CAP: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmdKind {
    Noop,
    WriteChunk,
    CycleActive,
    InsertSource,
    RemoveSource,
}

/// One queue slot. Fixed-size so slot-reuse (`push_ref` / `pop_ref`)
/// works without allocator traffic.
#[derive(Clone)]
pub struct Cmd {
    pub kind: CmdKind,
    /// Only meaningful for `WriteChunk`.
    pub source: Source,
    /// Valid length in `bytes`; 0 for non-chunk commands.
    pub len: u16,
    pub bytes: [u8; CMD_BYTES],
}

impl Default for Cmd {
    fn default() -> Self {
        Self {
            kind: CmdKind::Noop,
            source: Source::Kernel,
            len: 0,
            bytes: [0u8; CMD_BYTES],
        }
    }
}

/// Global producer ring. Written by any hart (syscall path, trap
/// handler, ktrace shim); drained only by `k_gpu`.
pub static CONSOLE_RING: StaticThingBuf<Cmd, RING_CAP> = StaticThingBuf::new();

pub struct GpuPackage {
    pub display: Display,
    /// Set by `virtio_gpu_dev::setup_virtio_gpu` — the 2D resource id
    /// whose backing is the scanout framebuffer.
    pub fb_resource_id: u32,
}

static PKG_PTR: AtomicPtr<GpuPackage> = AtomicPtr::new(null_mut());

/// Leak the package into a `'static` slot so `k_gpu` can access it
/// without a borrowed reference.
pub fn install_package(pkg: GpuPackage) {
    let leaked: &'static mut GpuPackage = Box::leak(Box::new(pkg));
    PKG_PTR.store(leaked as *mut GpuPackage, Ordering::Release);
}

fn package() -> Option<&'static mut GpuPackage> {
    let p = PKG_PTR.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        // SAFETY: installed once from hart 0; only `k_gpu` reads it
        // post-install. Single-consumer invariant.
        Some(unsafe { &mut *p })
    }
}

/// `true` if the global gpu package has been installed.
pub fn is_ready() -> bool {
    !PKG_PTR.load(Ordering::Acquire).is_null()
}

/// Push a single chunk command. Returns false if the ring was full.
pub fn push_chunk(source: Source, bytes: &[u8]) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref() else {
        return false;
    };
    let len = core::cmp::min(bytes.len(), CMD_BYTES);
    slot.kind = CmdKind::WriteChunk;
    slot.source = source;
    slot.len = len as u16;
    slot.bytes[..len].copy_from_slice(&bytes[..len]);
    true
}

/// Push a cycle-active command. Returns false if the ring was full.
pub fn push_cycle_active() -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref() else {
        return false;
    };
    slot.kind = CmdKind::CycleActive;
    slot.len = 0;
    true
}

/// Push an insert-source command (new process came up). Returns false
/// if the ring was full.
pub fn push_insert_source(source: Source) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref() else {
        return false;
    };
    slot.kind = CmdKind::InsertSource;
    slot.source = source;
    slot.len = 0;
    true
}

/// Push a remove-source command (process exited). Returns false if the
/// ring was full.
pub fn push_remove_source(source: Source) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref() else {
        return false;
    };
    slot.kind = CmdKind::RemoveSource;
    slot.source = source;
    slot.len = 0;
    true
}

/// Kernel thread entry. Spawned once after virtio-gpu init. Never
/// returns (exits as `ThreadState::Exited` on fatal error).
#[unsafe(no_mangle)]
pub extern "C" fn k_gpu(_a0: usize) {
    unsafe { riscv::register::sstatus::clear_sie(); }

    let pkg = match package() {
        Some(p) => p,
        None => {
            error!("k_gpu: package not installed");
            unsafe { exit_thread_with_state(ThreadState::Exited) };
        }
    };

    info!("k_gpu: ready, resource_id={}", pkg.fb_resource_id);

    loop {
        unsafe { riscv::register::sstatus::clear_sie(); }

        // Drain every pending command, batching redraws.
        while let Some(cmd) = CONSOLE_RING.pop_ref() {
            match cmd.kind {
                CmdKind::WriteChunk => {
                    let len = cmd.len as usize;
                    pkg.display.append(cmd.source, &cmd.bytes[..len]);
                }
                CmdKind::CycleActive => {
                    pkg.display.cycle_active();
                }
                CmdKind::InsertSource => {
                    pkg.display.insert_source(cmd.source);
                }
                CmdKind::RemoveSource => {
                    pkg.display.remove_source(cmd.source);
                }
                CmdKind::Noop => {}
            }
        }

        // One transfer + flush per drain pass. `repaint` returns
        // `false` when nothing changed, in which case we skip the
        // round-trip entirely.
        if pkg.display.repaint() {
            if let Some(gpu) = virtio_gpu_dev::gpu() {
                let w = pkg.display.fb.width();
                let h = pkg.display.fb.height();
                unsafe {
                    if let Err(e) = gpu.transfer_to_host_2d(pkg.fb_resource_id, 0, 0, w, h) {
                        error!("k_gpu: transfer failed: {:?}", e);
                    }
                    if let Err(e) = gpu.flush(pkg.fb_resource_id, 0, 0, w, h) {
                        error!("k_gpu: flush failed: {:?}", e);
                    }
                }
            }
        }

        // Park. Matches the k_net pattern: mark suspended with a
        // ~50 ms wake deadline (timebase = 10 MHz → 500k ticks) so
        // the thread yields cleanly to the scheduler. A producer
        // that pushes onto CONSOLE_RING can still wake us via SSWI
        // before the deadline.
        unsafe {
            let hart_context = (riscv::register::sscratch::read()
                as *mut device::HartContext)
                .as_mut_unchecked();
            let this_thread = (hart_context.current.load(Ordering::Acquire)
                as *mut process::Thread)
                .as_mut_unchecked();

            hart_context.cscratch2 = 1;
            this_thread.ticks = 0;
            this_thread.wake_time =
                (riscv::register::time::read64().wrapping_add(500_000)) as usize;
            this_thread
                .state
                .store(ThreadState::Suspended as usize, Ordering::Release);

            riscv::register::sstatus::set_sie();
            core::arch::asm!("ebreak", "nop");
        }
    }
}
