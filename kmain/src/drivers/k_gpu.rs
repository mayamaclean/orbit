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
pub const RING_CAP: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmdKind {
    Noop,
    WriteChunk,
    CycleActive,
    InsertSource,
    RemoveSource,
    /// User-side `fb_present(handle, rect)` request. Carries an
    /// embedded snapshot of the surface (kdmap KVA, dims, format) so
    /// `k_gpu`'s drain loop never has to touch the per-process surface
    /// table. The user-side syscall handler validates the handle +
    /// rect at submission time.
    PresentSurface,
}

/// Args carried by a [`CmdKind::PresentSurface`] slot. Snapshot of the
/// surface metadata + the damage rect to compose. Fixed-size so the
/// outer `Cmd` stays `Default + Clone` for `StaticThingBuf` slot reuse.
#[derive(Clone, Copy, Debug, Default)]
pub struct PresentArgs {
    /// Kernel-side KDMAP alias of the surface's first byte. The
    /// compositor reads pixels straight from here for the per-row blit.
    pub kdmap_kva: u64,
    /// Surface dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Damage rect inside the surface, validated against `(width,
    /// height)` at submit time.
    pub rect_x: u32,
    pub rect_y: u32,
    pub rect_w: u32,
    pub rect_h: u32,
    /// `FbFormat` discriminant. `1 = Bgra8888` is the only value v1
    /// emits; reserved for future formats.
    pub format_raw: u32,
}

/// One queue slot. Fixed-size so slot-reuse (`push_ref` / `pop_ref`)
/// works without allocator traffic.
#[derive(Clone)]
pub struct Cmd {
    pub kind: CmdKind,
    /// Only meaningful for `WriteChunk` / `PresentSurface` /
    /// `InsertSource` / `RemoveSource`. The compositor uses this to
    /// route per-source state mutations.
    pub source: Source,
    /// Valid length in `bytes`; 0 for non-chunk commands.
    pub len: u16,
    pub bytes: [u8; CMD_BYTES],
    /// Only meaningful for `PresentSurface`. Zeroed for other kinds.
    pub present: PresentArgs,
}

impl Default for Cmd {
    fn default() -> Self {
        Self {
            kind: CmdKind::Noop,
            source: Source::Kernel,
            len: 0,
            bytes: [0u8; CMD_BYTES],
            present: PresentArgs::default(),
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
    }
    else {
        // SAFETY: installed once from hart 0; only `k_gpu` reads it
        // post-install. Single-consumer invariant.
        Some(unsafe { &mut *p })
    }
}

/// `true` if the global gpu package has been installed.
pub fn is_ready() -> bool {
    !PKG_PTR.load(Ordering::Acquire).is_null()
}

/// Snapshot of the active framebuffer dimensions. Returns `None` until
/// `install_package` runs at boot. Stable for the life of the system
/// in v1 (no display-mode changes).
pub fn fb_size() -> Option<(u32, u32)> {
    let p = PKG_PTR.load(Ordering::Acquire);
    if p.is_null() {
        return None;
    }
    // SAFETY: PKG_PTR is install-once; the FrameBuffer dims are
    // immutable after install. Reading width/height through the
    // shared reference races nothing.
    let pkg = unsafe { &*p };
    Some((pkg.display.fb.width(), pkg.display.fb.height()))
}

/// Push a single chunk command. Returns false if the ring was full.
pub fn push_chunk(source: Source, bytes: &[u8]) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref()
    else {
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
    let Ok(mut slot) = CONSOLE_RING.push_ref()
    else {
        return false;
    };
    slot.kind = CmdKind::CycleActive;
    slot.len = 0;
    true
}

/// Push an insert-source command (new process came up). Returns false
/// if the ring was full.
pub fn push_insert_source(source: Source) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref()
    else {
        return false;
    };
    slot.kind = CmdKind::InsertSource;
    slot.source = source;
    slot.len = 0;
    true
}

/// Push a present-surface command. The args carry an immutable
/// snapshot of the surface (`kdmap_kva`, dims, format) plus the damage
/// rect inside the surface; the syscall handler is responsible for
/// validating the handle and rect bounds before submission. Returns
/// `false` if the ring was full.
pub fn push_present(source: Source, args: PresentArgs) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref()
    else {
        return false;
    };
    slot.kind = CmdKind::PresentSurface;
    slot.source = source;
    slot.len = 0;
    slot.present = args;
    true
}

/// Push a remove-source command (process exited). Returns false if the
/// ring was full.
pub fn push_remove_source(source: Source) -> bool {
    let Ok(mut slot) = CONSOLE_RING.push_ref()
    else {
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
    unsafe {
        riscv::register::sstatus::clear_sie();
    }

    let pkg = match package() {
        Some(p) => p,
        None => {
            error!("k_gpu: package not installed");
            unsafe { exit_thread_with_state(ThreadState::Exited) };
        }
    };

    info!("k_gpu: ready, resource_id={}", pkg.fb_resource_id);

    loop {
        unsafe {
            riscv::register::sstatus::clear_sie();
        }

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
                CmdKind::PresentSurface => {
                    pkg.display.present_surface(cmd.source, &cmd.present);
                }
                CmdKind::Noop => {}
            }
        }

        // One transfer + flush per drain pass. `repaint` returns
        // `None` when nothing changed; otherwise we transfer the
        // damage rect (smaller than full-screen for surface-mode
        // partial updates).
        if let Some(rect) = pkg.display.repaint() {
            if let Some(gpu) = virtio_gpu_dev::gpu() {
                unsafe {
                    if let Err(e) = gpu.transfer_to_host_2d(
                        pkg.fb_resource_id,
                        rect.x,
                        rect.y,
                        rect.w,
                        rect.h,
                    ) {
                        error!("k_gpu: transfer failed: {:?}", e);
                    }
                    if let Err(e) =
                        gpu.flush(pkg.fb_resource_id, rect.x, rect.y, rect.w, rect.h)
                    {
                        error!("k_gpu: flush failed: {:?}", e);
                    }
                }
            }
        }

        // Park with a ~50 ms wake deadline (timebase = 10 MHz →
        // 500k ticks) so the thread yields cleanly to the
        // scheduler. A producer that pushes onto CONSOLE_RING can
        // still wake us via wake_override before the deadline.
        // kthread_park's stack-switch-then-publish ordering closes
        // the double-dispatch race the prior cscratch2=1; ebreak
        // workaround was fencing off — see kernel::context.
        let wake_at = riscv::register::time::read64().wrapping_add(500_000) as usize;
        crate::kernel::context::kthread_park(ThreadState::Suspended, wake_at);
    }
}
