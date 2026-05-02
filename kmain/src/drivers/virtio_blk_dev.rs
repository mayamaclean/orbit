//! kmain-side glue for virtio-blk: pick the device-id 2 slot off
//! [`virtio_probe`], allocate request queue + arena from `kernel_pages`,
//! drive `Block::new` to bring the device live, register a PLIC handler
//! that drains used chains and dispatches each completion through the
//! caller-supplied [`WorkNotification`].
//!
//! Lookup table for in-flight async reads is keyed by descriptor head:
//! [`submit_blk_read`] boxes the caller's `WorkNotification` and stashes
//! the raw pointer at `IN_FLIGHT[head]`; the IRQ handler swaps it back
//! to null and runs the appropriate completion shape exactly once.
//! Single-mutator on each slot via the atomic swap, so there's no
//! locking.

use alloc::boxed::Box;
use core::alloc::Layout;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use mmu::sv48::PhysAddr;
use tracing::{error, info, warn};
use virtio::queue::VirtqBacking;
use virtio_blk::{
    ARENA_BYTES, Block, BlockBacking, BlockError, QUEUE_SIZE, SECTOR_SIZE, VIRTIO_BLK_DEVICE_ID,
    proto::VIRTIO_BLK_S_OK,
};

use orbit_core::PendingWork;
use process::CompletionHandle;

use crate::drivers::{plic, virtio_probe};
use crate::kernel::memmap::KernelPages;
use crate::kernel::shared_frame::SharedFrame;

// Queue page layout matches virtio_input_dev / virtio_gpu_dev — one
// page holds desc / avail / used with comfortable slack at the chosen
// 64-deep ring (desc 1 KiB, avail 138 B at +1 KiB, used 522 B at +2 KiB).
pub const QUEUE_PAGE_SIZE: usize = 4096;
const DESC_OFFSET: u64 = 0;
const AVAIL_OFFSET: u64 = 1024;
const USED_OFFSET: u64 = 2048;

/// What to do when a chain completes. `submit_blk_read` boxes one of
/// these and stashes the pointer at `IN_FLIGHT[head]`; the IRQ takes
/// ownership and dispatches.
///
/// Two shapes today, extensible:
/// - [`WorkNotification::Direct`] — IRQ signals `handle` directly with
///   `success_value` on `VIRTIO_BLK_S_OK` or `-1` otherwise. For
///   future raw kernel-side reads where the IRQ has everything it
///   needs to complete the request.
/// - [`WorkNotification::Bounce`] — IRQ enqueues a
///   [`PendingWork::FsReadCopy`] from the carried [`CopyDescriptor`];
///   the manager performs the scratch→user copy + cache publish +
///   signal under MANAGER_LOCK (UserPageWindow can't run in IRQ
///   context). The current `fs_read` path uses this exclusively.
pub enum WorkNotification {
    Direct {
        handle: CompletionHandle,
        success_value: isize,
    },
    Bounce(CopyDescriptor),
}

/// Bounce-path completion metadata. Carries everything the manager
/// needs to (1) copy `scratch[intra..intra + len]` into the user's
/// buffer at `user_page_pa[user_page_off..]`, and (2) publish the
/// new cache state on the originating `OpenFile` (clear `loading`,
/// set `cached_sector` / `valid_bytes`).
///
/// The `handle` and `scratch` ride inside the descriptor so the
/// side-table holds a single allocation per in-flight request, and
/// the [`SharedFrame`] clone keeps the scratch page alive across
/// `close_handle` / process teardown — last clone drops via
/// `pending_frees::push` after the manager finishes the post-DMA
/// copy.
pub struct CopyDescriptor {
    pub handle: CompletionHandle,
    /// pid that owns the originating fd. Used by the manager to find
    /// the `OpenFile` in `process_handles` (and to skip the
    /// user-page copy if the process has exited).
    pub pid: u16,
    /// fd within `pid`'s handle table.
    pub fd: u32,
    /// File-relative sector index DMA was filling. Becomes
    /// `cached_sector` on success.
    pub target_sector: u64,
    /// Bytes considered valid in the just-DMA'd sector. Becomes
    /// `valid_bytes` on success.
    pub valid_bytes: u32,
    /// Refcount clone keeping the scratch page alive across the
    /// IRQ → manager hand-off. The OpenFile holds the fd-side
    /// clone; close_handle / process teardown can drop theirs at
    /// any time without UAF.
    pub scratch: SharedFrame,
    pub user_page_pa: PhysAddr,
    pub user_page_off: u32,
    pub intra: u32,
    pub len: u32,
}

/// Per-descriptor-head completion slot table. Each entry holds a
/// `Box::into_raw`'d [`WorkNotification`] while the chain is in
/// flight, null otherwise. Single-mutator per slot via atomic swap.
static IN_FLIGHT: [AtomicPtr<WorkNotification>; QUEUE_SIZE as usize] = {
    const NULL: AtomicPtr<WorkNotification> = AtomicPtr::new(null_mut());
    [NULL; QUEUE_SIZE as usize]
};

static BLOCK_PTR: AtomicPtr<Block> = AtomicPtr::new(null_mut());

/// Access the installed block driver. Returns `None` until
/// [`setup_virtio_blk`] has completed successfully.
pub fn block_dev() -> Option<&'static mut Block> {
    let p = BLOCK_PTR.load(Ordering::Acquire);
    if p.is_null() {
        None
    }
    else {
        // SAFETY: BLOCK_PTR is set exactly once from hart 0 during
        // setup_virtio_blk. Single mutator post-init: callers must
        // serialize themselves (queue is not internally locked).
        Some(unsafe { &mut *p })
    }
}

/// Discover the slot, install Block, do a smoke sync read of LBA 0 to
/// prove end-to-end transport works, and arm the IRQ. Returns true on
/// success. Requires [`virtio_probe::discover`] to have run first.
pub fn setup_virtio_blk(kernel_pages: &mut KernelPages) -> bool {
    let Some(found) = virtio_probe::find(VIRTIO_BLK_DEVICE_ID)
    else {
        info!("virtio-blk: no device-id 2 slot present");
        return false;
    };
    let slot = found.slot;
    let mmio = found.mmio;
    info!(
        "virtio-blk: selected slot @{:#x} irq={}",
        slot.pa_base, slot.irq
    );

    let queue_layout = match Layout::from_size_align(QUEUE_PAGE_SIZE, QUEUE_PAGE_SIZE) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let Some((q_frame, q_kva)) = kernel_pages.alloc_kdmap(queue_layout)
    else {
        error!("virtio-blk: queue page alloc failed");
        return false;
    };
    let q_pa = q_frame.get_raw();
    unsafe {
        core::ptr::write_bytes(q_kva.as_mut_ptr::<u8>(), 0, QUEUE_PAGE_SIZE);
    }

    // Arena: per-head header (16 B) + status (1 B) + a sector-sized
    // bounce slot for the sync path. ARENA_BYTES is 2.5 KiB; round up
    // to one page for the allocator.
    let arena_pages = ARENA_BYTES.next_multiple_of(QUEUE_PAGE_SIZE);
    let arena_layout = match Layout::from_size_align(arena_pages, 4096) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let Some((arena_frame, arena_kva)) = kernel_pages.alloc_kdmap(arena_layout)
    else {
        error!("virtio-blk: arena alloc failed");
        return false;
    };
    let arena_pa = arena_frame.get_raw();
    unsafe {
        core::ptr::write_bytes(arena_kva.as_mut_ptr::<u8>(), 0, arena_pages);
    }

    let q_kva_u64 = q_kva.as_mut_ptr::<u8>() as u64;
    let backing = BlockBacking {
        mmio,
        reqq: VirtqBacking {
            desc_pa: q_pa + DESC_OFFSET,
            desc_kva: (q_kva_u64 + DESC_OFFSET) as *mut u8,
            avail_pa: q_pa + AVAIL_OFFSET,
            avail_kva: (q_kva_u64 + AVAIL_OFFSET) as *mut u8,
            used_pa: q_pa + USED_OFFSET,
            used_kva: (q_kva_u64 + USED_OFFSET) as *mut u8,
            size: QUEUE_SIZE,
        },
        arena_pa,
        arena_kva: arena_kva.as_mut_ptr::<u8>(),
        arena_size: arena_pages,
    };

    let mut dev = match unsafe { Block::new(backing) } {
        Ok(d) => d,
        Err(e) => {
            error!("virtio-blk: init failed: {:?}", e);
            return false;
        }
    };

    // Sync-read smoke: read LBA 0 (the first ustar header) and log the
    // first 8 bytes. SIE/PLIC aren't armed yet at this point in boot
    // (setup_interrupts runs in k_harthello, which fires after
    // get_environment_info), so we use the polled `read_blocks_blocking`
    // path. The async IRQ path lands here as code but isn't exercised
    // until 12c's tarfs reads, after the scheduler is up.
    let mut buf = [0u8; SECTOR_SIZE];
    match unsafe { dev.read_blocks_blocking(0, &mut buf) } {
        Ok(()) => {
            info!(
                "virtio-blk: sync read ok lba=0 first8={:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            );
        }
        Err(e) => {
            error!("virtio-blk: sync read of lba 0 failed: {:?}", e);
            return false;
        }
    }

    // Mount tarfs while we still own `&mut dev` exclusively. After
    // BLOCK_PTR is published below, only the IRQ-driven path may touch
    // the device (no further sync reads).
    match crate::kernel::fs::tar::Tarfs::mount(&mut dev) {
        Ok(fs) => {
            let leaked: &'static dyn crate::kernel::fs::Filesystem = Box::leak(Box::new(fs));
            crate::kernel::fs::install(leaked);
        }
        Err(e) => {
            error!("tarfs: mount failed: {:?}", e);
            // Continue anyway — the block driver is still useful for
            // diagnostics, just no FS mounted.
        }
    }

    let leaked: &'static mut Block = Box::leak(Box::new(dev));
    BLOCK_PTR.store(leaked as *mut _, Ordering::Release);

    if plic::plic_register(slot.irq, virtio_blk_handler, 0).is_err() {
        error!("virtio-blk: plic_register failed for irq {}", slot.irq);
        return false;
    }

    info!("virtio-blk: device live, irq {} armed", slot.irq);
    true
}

/// Submit an asynchronous single-sector read at `lba` into `dst_pa`.
/// The IRQ handler dispatches `notif` once the chain completes:
/// [`WorkNotification::Direct`] signals the carried handle inline;
/// [`WorkNotification::Bounce`] enqueues a [`PendingWork::FsReadCopy`]
/// for the manager to copy scratch→user and signal.
///
/// Returns the descriptor head used. On `Err`, the boxed `notif`
/// is dropped — including any `CompletionHandle` it owned.
///
/// # Safety
/// - `dst_pa` must reference `SECTOR_SIZE` bytes the kernel keeps
///   mapped until completion.
/// - Caller serializes concurrent submitters on the same `Block`.
pub unsafe fn submit_blk_read(
    lba: u64,
    dst_pa: u64,
    notif: WorkNotification,
) -> Result<u16, BlockError> {
    let dev = block_dev().ok_or(BlockError::QueueFull)?;

    // Pre-publish: peek the head that submit_read will use, then
    // populate IN_FLIGHT[head] *before* submit_read runs (which
    // notifies the device). Without this the device can complete the
    // chain and fire the IRQ on another hart while we're still inside
    // submit_read; the IRQ handler then sees a null `IN_FLIGHT[head]`
    // and the parked thread strands waiting for a signal that never
    // comes. The race window is short on QEMU but fundamental on real
    // hardware. peek_next_head is non-consuming: submit_read's own
    // peek_free_head + push_chain land on the same head we predicted.
    let head = dev.peek_next_head().ok_or(BlockError::QueueFull)?;

    // Stash the notification for the IRQ handler to reclaim. swap,
    // not store: a previous in-flight chain at this descriptor index
    // must have completed (drain_used cleared the slot to null) before
    // its index re-entered the free list, so this swap should always
    // see null. If it doesn't, something violated the queue invariant
    // — log and drop the prior notification to avoid leaking the Arc.
    let raw = Box::into_raw(Box::new(notif));
    let prev = IN_FLIGHT[head as usize].swap(raw, Ordering::AcqRel);
    if !prev.is_null() {
        error!(
            "virtio-blk: IN_FLIGHT[{head}] was non-null at submit — pre-existing notification leaked then dropped"
        );
        unsafe {
            drop(Box::from_raw(prev));
        }
    }

    // Now actually submit. push_chain must produce the same head we
    // pre-published; if not, our notification is stranded at the
    // wrong slot. submit_read debug_asserts head == predicted
    // internally, so a mismatch panics in dev builds; in release
    // it's silently wrong.
    match unsafe { dev.submit_read(lba, dst_pa, SECTOR_SIZE as u32) } {
        Ok(actual) => {
            debug_assert!(
                actual == head,
                "submit_blk_read: predicted head={head} but submit produced {actual}",
            );
            Ok(head)
        }
        Err(e) => {
            // Reclaim the notification (and the handle inside it)
            // since the chain never went out.
            let raw = IN_FLIGHT[head as usize].swap(core::ptr::null_mut(), Ordering::AcqRel);
            if !raw.is_null() {
                unsafe {
                    drop(Box::from_raw(raw));
                }
            }
            Err(e)
        }
    }
}

/// PLIC handler. Acks the device interrupt, drains every completed
/// chain, and dispatches the corresponding [`WorkNotification`].
///
/// For [`WorkNotification::Bounce`] the IRQ forwards the same
/// `Box<WorkNotification>` pointer that `submit_blk_read` stashed
/// in IN_FLIGHT — `PendingWork::FsReadCopy { notif_ptr, status }`
/// — so the box (with its [`SharedFrame`] clone inside) survives
/// the IRQ → manager hand-off without re-allocation.
fn virtio_blk_handler(_src: u32) {
    let Some(dev) = block_dev()
    else {
        return;
    };
    unsafe {
        let _used = dev.ack_interrupt();
        dev.drain_used(|head, status| {
            let raw = IN_FLIGHT[head as usize].swap(null_mut(), Ordering::AcqRel);
            if raw.is_null() {
                // No notification registered. Either the submitter
                // raced and hasn't published yet (impossible: we
                // publish before notify_queue) or the chain was a
                // stray. Log once and drop.
                error!("virtio-blk: completion for head={head} with no registered notification");
                return;
            }
            // Peek the variant without consuming the box: bounce
            // path forwards the same pointer through PendingWork
            // (manager unboxes), direct path unboxes here and
            // signals inline.
            //
            // SAFETY: `raw` was just swapped out of IN_FLIGHT; no
            // other thread observes it. The `&*raw` read is valid
            // for this match arm; we either consume the box
            // (Direct) or transfer ownership through PendingWork
            // (Bounce).
            match &*raw {
                WorkNotification::Direct { .. } => {
                    let notif = *Box::from_raw(raw);
                    let WorkNotification::Direct {
                        handle,
                        success_value,
                    } = notif
                    else {
                        unreachable!()
                    };
                    let result: isize = if status == VIRTIO_BLK_S_OK {
                        success_value
                    }
                    else {
                        -1
                    };
                    handle.signal(result);
                }
                WorkNotification::Bounce(_) => {
                    let notif_ptr = raw as usize;
                    let work = PendingWork::FsReadCopy { notif_ptr, status };
                    // If MANAGER_WORK is full the parked thread
                    // would otherwise hang forever — recover the
                    // box, signal -EIO inline, drop. The
                    // SharedFrame clone drops with the box and the
                    // scratch page heads to pending_frees if this
                    // was the last clone.
                    match crate::kernel::MANAGER_WORK.push_ref() {
                        Ok(mut slot) => {
                            *slot = work;
                        }
                        Err(_) => {
                            warn!(
                                "virtio-blk: MANAGER_WORK full — signaling -EIO for head={head}"
                            );
                            let recovered = Box::from_raw(raw);
                            if let WorkNotification::Bounce(desc) = *recovered {
                                desc.handle.signal(-(orbit_abi::errno::EIO as isize));
                            }
                        }
                    }
                }
            }
        });
    }
}
