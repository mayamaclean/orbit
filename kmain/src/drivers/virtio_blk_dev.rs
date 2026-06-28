//! kmain-side glue for virtio-blk: pick the device-id 2 slot off
//! [`virtio_probe`], allocate request queue + arena from `kernel_pages`,
//! drive `Block::new` to bring the device live, register a PLIC handler
//! that drains used chains and forwards each completion to the manager
//! via [`PendingWork::CacheFill`].
//!
//! Submit path is exclusively the cache-driven one
//! ([`submit_blk_read_cached`]): the caller stashes a packed
//! [`crate::kernel::page_cache::CacheKey`] at `IN_FLIGHT[head]` before
//! `submit_read` notifies the device; the IRQ handler swaps the slot
//! back to zero and pushes one `CacheFill { packed_key, status }` per
//! completed chain. No allocation on the IRQ side, no per-chain
//! `Box<WorkNotification>`.

use alloc::boxed::Box;
use core::alloc::Layout;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use tracing::{error, info};
use virtio::queue::VirtqBacking;
use virtio_blk::{
    ARENA_BYTES, Block, BlockBacking, BlockError, QUEUE_SIZE, SECTOR_SIZE, VIRTIO_BLK_DEVICE_ID,
};

use crate::drivers::{plic, virtio_probe};
use crate::kernel::memmap::{KernelPages, phys_to_kdmap};
use crate::kernel::shootdown::CPU_COUNT;
use mmu::sv48::PhysAddr;

// Queue page layout matches virtio_input_dev / virtio_gpu_dev — one
// page holds desc / avail / used with comfortable slack at the chosen
// 64-deep ring: desc fills 1 KiB at offset 0, avail is 134 B at +1 KiB,
// used is 518 B at +2 KiB.
pub const QUEUE_PAGE_SIZE: usize = 4096;
const DESC_OFFSET: u64 = 0;
const AVAIL_OFFSET: u64 = 1024;
const USED_OFFSET: u64 = 2048;

/// Per-descriptor-head completion slot table. Each entry holds a
/// packed [`crate::kernel::page_cache::CacheKey`] (see
/// [`crate::kernel::page_cache::pack`]) while the chain is in flight,
/// `0` otherwise. Single-mutator per slot via atomic swap, with
/// [`QUEUE_LOCK`] excluding the IRQ from the submitter's
/// peek-stash-submit window.
static IN_FLIGHT: [AtomicU64; QUEUE_SIZE as usize] = {
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; QUEUE_SIZE as usize]
};

/// Spinlock around `Virtqueue::free_head` mutation and the matching
/// `IN_FLIGHT` slot publish/clear. Held by `submit_blk_read_cached`
/// across its peek-stash-submit window and by [`virtio_blk_handler`]
/// across `pop_used` + slot-swap.
///
/// **Why a real spinlock and not just SIE masking:** the manager is
/// greedy (any hart can drain MANAGER_WORK), but the virtio-blk IRQ
/// is PLIC-pinned to one hart (the last, `CPU_COUNT - 1`). When the
/// manager submits from a different hart, masking SIE locally has no
/// effect on the IRQ hart's handler — the IRQ's `pop_used` mutates `Virtqueue::free_head`
/// concurrently with the submitter's `peek_next_head`, leaving the
/// stashed `IN_FLIGHT[predicted]` at the wrong slot and producing
/// "completion for head=X with no registered key" / "non-zero at
/// submit" error pairs.
///
/// IRQ-side hold time: one `pop_used` iteration + one `swap` per
/// completed chain. Submit-side hold time: one `peek_next_head` +
/// one `swap` + one `submit_read` (push_chain writes 3 descs +
/// notifies). Both are short; brief IRQ-context spinning is
/// acceptable.
static QUEUE_LOCK: spin::Mutex<()> = spin::Mutex::new(());

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

    if plic::plic_register(
        slot.irq,
        virtio_blk_handler,
        CPU_COUNT.load(Ordering::Relaxed).saturating_sub(1),
    )
    .is_err()
    {
        error!("virtio-blk: plic_register failed for irq {}", slot.irq);
        return false;
    }

    info!("virtio-blk: device live, irq {} armed", slot.irq);
    true
}

/// Submit an asynchronous multi-sector read at `lba` into `dst_pa`.
/// Stashes the packed `CacheKey` at `IN_FLIGHT[head]`; the IRQ
/// handler swaps it out and pushes
/// [`PendingWork::CacheFill { packed_key, status }`] for the manager
/// to dispatch via `run_cache_fill`.
///
/// `packed_key` must be a non-zero packed `CacheKey` (the occupied
/// bit is what makes the swap-vs-zero discrimination work in the IRQ).
/// `len` must satisfy `virtio_blk::Block::submit_read`'s contract:
/// non-zero, multiple of `SECTOR_SIZE`, and at most
/// [`virtio_blk::MAX_REQ_BYTES`]. Today's callers pass exactly one
/// page.
///
/// # Safety
/// - `dst_pa` must reference `len` bytes of physically-contiguous
///   memory the cache slot keeps alive (the `Loading` slot's
///   `SharedFrame`) until the manager runs `complete_slot`.
/// - Caller serializes concurrent submitters on the same `Block`.
pub unsafe fn submit_blk_read_cached(
    lba: u64,
    dst_pa: u64,
    len: u32,
    packed_key: u64,
) -> Result<u16, BlockError> {
    debug_assert!(packed_key != 0, "submit_blk_read_cached: empty key");
    let dev = block_dev().ok_or(BlockError::QueueFull)?;

    // Take QUEUE_LOCK across the peek-stash-submit window so the IRQ
    // handler can't pop_used (and mutate `Virtqueue::free_head`)
    // mid-sequence. See QUEUE_LOCK's doc for the cross-hart race
    // this guards against.
    let _g = QUEUE_LOCK.lock();

    // Clamp the DMA to disk capacity. tar packs files at 512-byte
    // boundaries, so a file's last page can ask for sectors past the
    // disk end. Without the clamp `dev.submit_read` rejects with
    // `OutOfRange` and the consumer surfaces EIO at the file tail —
    // breaks any file whose last page would overrun. The slot's
    // `valid_bytes` already bounds what consumers read out, so the
    // bytes past the actual disk are just hygiene-zeros for accidental
    // over-reads. Pre-fills via the kdmap alias before the DMA
    // launches — the device only touches the leading `actual_len`
    // bytes; the trailing zeros stay intact.
    let cap_sectors = dev.capacity_sectors();
    let sector_size = SECTOR_SIZE as u64;
    debug_assert!(
        len as u64 % sector_size == 0,
        "submit_blk_read_cached: len {len} not sector-multiple"
    );
    let req_sectors = len as u64 / sector_size;
    let avail_sectors = cap_sectors.saturating_sub(lba);
    if avail_sectors == 0 {
        // First sector itself is past the disk — that's a real bug
        // (page cache asked for a page entirely past EOF), surface
        // it. Caller's `complete_slot(_, err)` tears down the slot.
        return Err(BlockError::OutOfRange {
            lba,
            capacity: cap_sectors,
        });
    }
    let actual_sectors = req_sectors.min(avail_sectors);
    let actual_len = (actual_sectors * sector_size) as u32;
    let zero_bytes = ((req_sectors - actual_sectors) * sector_size) as usize;
    if zero_bytes > 0 {
        // SAFETY: `dst_pa` references a page-cache slot frame the
        // caller pinned via `begin_load`'s `SharedFrame`. The frame
        // is `len` bytes; we write to its tail starting at
        // `actual_len`. kdmap maps every kernel-pool frame at boot,
        // so `phys_to_kdmap(dst_pa + actual_len)` is a valid kernel
        // VA for the duration of this call.
        let zero_pa = PhysAddr::new(dst_pa + actual_len as u64);
        let dst = phys_to_kdmap(zero_pa).as_mut_ptr::<u8>();
        unsafe {
            core::ptr::write_bytes(dst, 0, zero_bytes);
        }
    }

    let head = dev.peek_next_head().ok_or(BlockError::QueueFull)?;
    let prev = IN_FLIGHT[head as usize].swap(packed_key, Ordering::AcqRel);
    if prev != 0 {
        // Slot was non-zero at submit time — should be unreachable
        // now that the lock excludes the IRQ from the same window.
        // Logged so any regression is visible.
        error!(
            "virtio-blk: IN_FLIGHT[{head}] was non-zero ({prev:#x}) at submit — pre-existing key dropped"
        );
    }

    match unsafe { dev.submit_read(lba, dst_pa, actual_len) } {
        Ok(actual) => {
            debug_assert!(
                actual == head,
                "submit_blk_read_cached: predicted head={head} but submit produced {actual}",
            );
            Ok(head)
        }
        Err(e) => {
            // Submit failed; clear the slot we speculatively
            // populated. Caller's cache slot is torn down
            // separately by `complete_slot(key, status=err)`.
            IN_FLIGHT[head as usize].store(0, Ordering::Release);
            Err(e)
        }
    }
}

/// PLIC handler. Acks the device interrupt, drains every completed
/// chain, and pushes one [`PendingWork::CacheFill`] per chain for
/// the manager to dispatch via `run_cache_fill`. No allocation, no
/// per-chain box — the side-table holds the packed key inline.
fn virtio_blk_handler(_src: u32) {
    let Some(dev) = block_dev()
    else {
        return;
    };
    // Take QUEUE_LOCK across pop_used + slot-swap. ack_interrupt is
    // outside the lock — it only touches the MMIO interrupt-status
    // register, which the submitter never reads.
    let _g = QUEUE_LOCK.lock();
    unsafe {
        let _used = dev.ack_interrupt();
        dev.drain_used(|head, status| {
            let packed = IN_FLIGHT[head as usize].swap(0, Ordering::AcqRel);
            if packed == 0 {
                // Either the submitter raced and hasn't published
                // yet (impossible: we publish before notify_queue
                // under QUEUE_LOCK) or this is a stray completion.
                error!("virtio-blk: completion for head={head} with no registered key");
                return;
            }
            // Dedicated completion ring — never MANAGER_WORK. Sharing
            // the syscall ring meant a burst of blocking syscalls
            // could fill it and force this handler to *drop* the
            // completion, leaving the cache slot Loading forever and
            // every fs_read waiter on it parked permanently. The ring
            // is sized at 2× QUEUE_SIZE, so with ≤ QUEUE_SIZE chains
            // in flight this push cannot fail unless the manager has
            // stopped draining entirely — which is unrecoverable
            // anyway, hence error-level and no fallback.
            let ev = crate::kernel::CacheFillEvent {
                packed_key: packed,
                status,
            };
            if crate::kernel::CACHE_FILLS
                .push_ref()
                .map(|mut s| *s = ev)
                .is_err()
            {
                error!("virtio-blk: CACHE_FILLS full — dropping CacheFill for head={head} (manager stalled?)");
            }
        });
    }
}
