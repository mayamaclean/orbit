//! kmain-side glue for virtio-gpu: discover the MMIO slot, install
//! the KMMIO alias, allocate ctrl-queue / command arena / framebuffer
//! from `kernel_pages`, and run the boot-time init sequence
//! (`GET_DISPLAY_INFO` → `CREATE_2D` → `ATTACH_BACKING` → `SET_SCANOUT`
//! → initial `TRANSFER_TO_HOST_2D` + `FLUSH`).
//!
//! After this the device is live and holds a single 2D resource
//! (resource id = 1) whose backing is our kernel-owned framebuffer.
//! Subsequent steps (glyph blit, k_gpu thread) write pixels into the
//! kdmap-aliased framebuffer and issue further transfer+flush pairs.

use alloc::boxed::Box;
use core::alloc::Layout;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use tracing::{error, info};
use virtio::queue::VirtqBacking;
use virtio_gpu::{Gpu, GpuBacking, ARENA_SIZE, FORMAT_B8G8R8A8_UNORM};
use virtio_gpu::proto::VIRTIO_GPU_DEVICE_ID;

use crate::drivers::virtio_probe;
use crate::kernel::memmap::KernelPages;

// Queue sizing: 64 entries is plenty for gpu ctrl-queue traffic
// (bursty but never many in-flight at once). Fits in one page with
// room to spare:
//   desc  = 64 × 16 = 1024 B   → offset 0
//   avail = 4 + 64×2 + 2 = 134 → rounded to offset 1024 (1 KiB slot)
//   used  = 4 + 64×8 + 2 = 518 → rounded to offset 2048 (2 KiB slot)
pub const QUEUE_SIZE: u16 = 64;
pub const QUEUE_PAGE_SIZE: usize = 4096;
const DESC_OFFSET: u64 = 0;
const AVAIL_OFFSET: u64 = 1024;
const USED_OFFSET: u64 = 2048;

/// Hard cap on framebuffer size (8 MiB = 1920×1080×4 + slack) to
/// prevent a bogus `GET_DISPLAY_INFO` from draining the kernel pool.
const MAX_FB_BYTES: usize = 8 * 1024 * 1024;

/// Resource ID used for our one 2D surface. virtio-gpu reserves 0;
/// anything else works.
const FB_RESOURCE_ID: u32 = 1;

static GPU_PTR: AtomicPtr<Gpu> = AtomicPtr::new(null_mut());

/// Access the installed gpu driver. Returns `None` until
/// [`setup_virtio_gpu`] has completed successfully.
pub fn gpu() -> Option<&'static mut Gpu> {
    let p = GPU_PTR.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        // SAFETY: GPU_PTR is written exactly once, from hart 0 in
        // `setup_virtio_gpu`. Later accesses happen from `k_gpu`
        // (single consumer) and from boot-time init before k_gpu is
        // spawned. Callers must serialize themselves — currently
        // that's trivially true because k_gpu is the only post-boot
        // consumer.
        Some(unsafe { &mut *p })
    }
}

/// Installation result so the caller knows whether the framebuffer
/// pipeline is live. Returned by value rather than stored so Orbit
/// can decide whether to dual-route ktrace, spawn k_gpu, etc.
pub struct InstallOutcome {
    pub width: u32,
    pub height: u32,
    pub fb_pa: u64,
    pub fb_kva: u64,
    pub fb_size: usize,
    pub resource_id: u32,
}

/// Build a [`Gpu`] over the already-aliased virtio-gpu slot, run boot
/// init, and return framebuffer coordinates for the caller to
/// integrate into the display pipeline. Requires
/// [`virtio_probe::discover`] to have run first.
pub fn setup_virtio_gpu(
    kernel_pages: &mut KernelPages,
) -> Option<InstallOutcome> {
    let found = virtio_probe::find(VIRTIO_GPU_DEVICE_ID)?;
    let slot = found.slot;
    let mmio = found.mmio;
    info!("virtio-gpu: selected slot @{:#x} irq={}", slot.pa_base, slot.irq);

    // Phase 3: allocate ctrl queue page + command arena page.
    let ctrl_layout = Layout::from_size_align(QUEUE_PAGE_SIZE, QUEUE_PAGE_SIZE).ok()?;
    let (ctrl_frame, ctrl_kva) = kernel_pages.alloc_kdmap(ctrl_layout)?;
    let ctrl_pa = ctrl_frame.get_raw();
    unsafe { core::ptr::write_bytes(ctrl_kva.as_mut_ptr::<u8>(), 0, QUEUE_PAGE_SIZE); }

    let arena_layout = Layout::from_size_align(ARENA_SIZE, 4096).ok()?;
    let (arena_frame, arena_kva) = kernel_pages.alloc_kdmap(arena_layout)?;
    let arena_pa = arena_frame.get_raw();
    unsafe { core::ptr::write_bytes(arena_kva.as_mut_ptr::<u8>(), 0, ARENA_SIZE); }

    let ctrl_kva_u64 = ctrl_kva.as_mut_ptr::<u8>() as u64;
    let ctrl_backing = VirtqBacking {
        desc_pa: ctrl_pa + DESC_OFFSET,
        desc_kva: (ctrl_kva_u64 + DESC_OFFSET) as *mut u8,
        avail_pa: ctrl_pa + AVAIL_OFFSET,
        avail_kva: (ctrl_kva_u64 + AVAIL_OFFSET) as *mut u8,
        used_pa: ctrl_pa + USED_OFFSET,
        used_kva: (ctrl_kva_u64 + USED_OFFSET) as *mut u8,
        size: QUEUE_SIZE,
    };

    // Phase 4: run the device init handshake.
    let mut gpu = match unsafe {
        Gpu::new(GpuBacking {
            mmio,
            ctrl: ctrl_backing,
            arena_pa,
            arena_kva: arena_kva.as_mut_ptr::<u8>(),
            arena_size: ARENA_SIZE,
        })
    } {
        Ok(g) => g,
        Err(e) => {
            error!("virtio-gpu: init failed: {:?}", e);
            return None;
        }
    };

    // Phase 5: query resolution and sanity-check it fits our cap.
    let info = unsafe {
        match gpu.get_display_info() {
            Ok(i) => i,
            Err(e) => {
                error!("virtio-gpu: get_display_info failed: {:?}", e);
                return None;
            }
        }
    };
    info!("virtio-gpu: scanout 0 = {}x{}", info.width, info.height);
    let fb_size = (info.width as usize) * (info.height as usize) * 4;
    if fb_size == 0 || fb_size > MAX_FB_BYTES {
        error!("virtio-gpu: fb_size {} out of range", fb_size);
        return None;
    }

    // Phase 6: allocate the framebuffer. 2 MiB-aligned so the buddy
    // allocator hands back a single contiguous megapage-ish chunk —
    // virtio-gpu wants one physically contiguous backing entry.
    let fb_layout = Layout::from_size_align(fb_size, 2 * 1024 * 1024).ok()?;
    let (fb_frame, fb_kva) = kernel_pages.alloc_kdmap(fb_layout)?;
    let fb_pa = fb_frame.get_raw();

    // Prime the framebuffer with a solid dark-gray background plus a
    // greeting so we can visually confirm the blit path works before
    // the k_gpu thread + Scrollback land.
    let fb = unsafe {
        crate::drivers::fb::FrameBuffer::new(
            fb_kva.as_ptr::<u8>() as u64,
            info.width,
            info.height,
        )
    };
    fb.fill(crate::drivers::fb::DARK_GRAY);
    fb.blit_text(
        16,
        16,
        "orbit framebuffer online",
        crate::drivers::fb::WHITE,
        crate::drivers::fb::DARK_GRAY,
    );

    // Phase 7: tell the device about the resource + scanout + initial
    // contents.
    unsafe {
        if let Err(e) = gpu.create_2d_resource(
            FB_RESOURCE_ID, info.width, info.height, FORMAT_B8G8R8A8_UNORM,
        ) {
            error!("virtio-gpu: create_2d failed: {:?}", e);
            return None;
        }
        if let Err(e) = gpu.attach_backing(FB_RESOURCE_ID, fb_pa, fb_size as u32) {
            error!("virtio-gpu: attach_backing failed: {:?}", e);
            return None;
        }
        if let Err(e) = gpu.set_scanout(0, FB_RESOURCE_ID, info.width, info.height) {
            error!("virtio-gpu: set_scanout failed: {:?}", e);
            return None;
        }
        if let Err(e) = gpu.transfer_to_host_2d(
            FB_RESOURCE_ID, 0, 0, info.width, info.height,
        ) {
            error!("virtio-gpu: transfer failed: {:?}", e);
            return None;
        }
        if let Err(e) = gpu.flush(FB_RESOURCE_ID, 0, 0, info.width, info.height) {
            error!("virtio-gpu: flush failed: {:?}", e);
            return None;
        }
    }

    info!(
        "virtio-gpu: initialized, fb pa={:#x} kva={:#x} size={}KB",
        fb_pa,
        fb_kva.as_ptr::<u8>() as u64,
        fb_size / 1024,
    );

    // Stash the Gpu for later consumers (k_gpu, console_write). Leaked
    // intentionally — the kernel never tears down its gpu state.
    let gpu_leaked: &'static mut Gpu = Box::leak(Box::new(gpu));
    GPU_PTR.store(gpu_leaked as *mut Gpu, Ordering::Release);

    Some(InstallOutcome {
        width: info.width,
        height: info.height,
        fb_pa,
        fb_kva: fb_kva.as_ptr::<u8>() as u64,
        fb_size,
        resource_id: FB_RESOURCE_ID,
    })
}
