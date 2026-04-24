//! Split-ring virtqueue (virtio spec §2.6).
//!
//! The queue owns descriptor free-list bookkeeping and wraps the three
//! rings (`desc` / `avail` / `used`) that the caller allocated.
//! Allocation is the caller's problem: hand in PAs for the device to
//! see and KDMAP VAs the kernel will write through.

use core::sync::atomic::{Ordering, fence};

// Descriptor flags.
pub const VIRTQ_DESC_F_NEXT: u16 = 1;
pub const VIRTQ_DESC_F_WRITE: u16 = 2;
pub const VIRTQ_DESC_F_INDIRECT: u16 = 4;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VirtqDesc {
    pub addr: u64,
    pub len: u32,
    pub flags: u16,
    pub next: u16,
}

#[repr(C)]
pub struct VirtqAvail {
    pub flags: u16,
    pub idx: u16,
    pub ring: [u16; 0], // trailing array, queue_size entries
    // pub used_event: u16 — after ring, when EVENT_IDX feature negotiated
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VirtqUsedElem {
    pub id: u32,
    pub len: u32,
}

#[repr(C)]
pub struct VirtqUsed {
    pub flags: u16,
    pub idx: u16,
    pub ring: [VirtqUsedElem; 0], // trailing array, queue_size entries
}

/// One buffer in a descriptor chain.
#[derive(Clone, Copy, Debug)]
pub struct Buf {
    pub pa: u64,
    pub len: u32,
    /// `true` = device writes to us; `false` = device reads from us.
    pub write: bool,
}

/// Caller-provided backing. PAs go to the device; KVAs are how the
/// kernel reads/writes the ring structures.
pub struct VirtqBacking {
    pub desc_pa: u64,
    pub desc_kva: *mut u8,
    pub avail_pa: u64,
    pub avail_kva: *mut u8,
    pub used_pa: u64,
    pub used_kva: *mut u8,
    pub size: u16,
}

/// Returned from `push_chain` when the free-list is exhausted.
#[derive(Debug, Clone, Copy)]
pub struct Full;

pub struct Virtqueue {
    desc: *mut VirtqDesc,
    avail: *mut u8, // points at VirtqAvail header
    used: *mut u8,  // points at VirtqUsed header
    size: u16,
    free_head: u16,
    num_free: u16,
    avail_idx: u16,
    last_used_idx: u16,
    // Stash PAs so the transport layer can program queue registers.
    desc_pa: u64,
    avail_pa: u64,
    used_pa: u64,
}

// SAFETY: caller vouches for uniqueness of the raw pointers through
// `unsafe fn new`. Internal accesses are non-aliased volatile writes
// to memory backed by device-visible PAs.
unsafe impl Send for Virtqueue {}

impl Virtqueue {
    /// # Safety
    /// `backing` must point at three distinct, valid, zero-initialized
    /// memory regions sized for `size` entries. Caller retains
    /// ownership; `Virtqueue` just borrows the raw pointers for its
    /// lifetime.
    pub unsafe fn new(backing: VirtqBacking) -> Self {
        let desc = backing.desc_kva as *mut VirtqDesc;
        let size = backing.size;

        // Seed the descriptor free-list: desc[i].next = i + 1, last
        // wraps to 0 (sentinel "end of free list").
        unsafe {
            for i in 0..size {
                let d = desc.add(i as usize);
                (*d).addr = 0;
                (*d).len = 0;
                (*d).flags = 0;
                (*d).next = if i + 1 < size { i + 1 } else { 0 };
            }
        }

        Self {
            desc,
            avail: backing.avail_kva,
            used: backing.used_kva,
            size,
            free_head: 0,
            num_free: size,
            avail_idx: 0,
            last_used_idx: 0,
            desc_pa: backing.desc_pa,
            avail_pa: backing.avail_pa,
            used_pa: backing.used_pa,
        }
    }

    pub fn size(&self) -> u16 { self.size }
    pub fn desc_pa(&self) -> u64 { self.desc_pa }
    pub fn avail_pa(&self) -> u64 { self.avail_pa }
    pub fn used_pa(&self) -> u64 { self.used_pa }

    /// Build a descriptor chain from `bufs`, publish the head on the
    /// avail ring, and return the head index. Caller must call
    /// `notify()` on the transport separately.
    pub fn push_chain(&mut self, bufs: &[Buf]) -> Result<u16, Full> {
        if bufs.is_empty() || bufs.len() as u16 > self.num_free {
            return Err(Full);
        }

        // Pop `bufs.len()` descriptors from the free list, remember
        // them in order.
        let n = bufs.len();
        let mut descs: [u16; 32] = [0; 32];
        if n > descs.len() {
            return Err(Full);
        }
        let mut head = self.free_head;
        descs[0] = head;
        for i in 1..n {
            let next = unsafe { (*self.desc.add(descs[i - 1] as usize)).next };
            descs[i] = next;
        }
        let new_free_head = unsafe { (*self.desc.add(descs[n - 1] as usize)).next };
        self.free_head = new_free_head;
        self.num_free -= n as u16;

        // Populate the descriptors in order.
        for i in 0..n {
            let buf = bufs[i];
            let last = i == n - 1;
            let mut flags = 0u16;
            if buf.write {
                flags |= VIRTQ_DESC_F_WRITE;
            }
            if !last {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            let next = if !last { descs[i + 1] } else { 0 };
            unsafe {
                let d = self.desc.add(descs[i] as usize);
                (*d).addr = buf.pa;
                (*d).len = buf.len;
                (*d).flags = flags;
                (*d).next = next;
            }
        }

        head = descs[0];

        // Publish head onto avail.ring[avail.idx % size].
        let ring_slot = self.avail_idx % self.size;
        unsafe {
            // avail layout: [flags:u16][idx:u16][ring: u16 × size]
            let ring_base = self.avail.add(4) as *mut u16;
            ring_base.add(ring_slot as usize).write_volatile(head);
        }
        // Memory barrier: ensure descriptor writes and ring slot write
        // are visible to the device before we bump the idx.
        fence(Ordering::SeqCst);
        self.avail_idx = self.avail_idx.wrapping_add(1);
        unsafe {
            (self.avail.add(2) as *mut u16).write_volatile(self.avail_idx);
        }

        Ok(head)
    }

    /// Pop one completed descriptor chain off the used ring. Returns
    /// `(head_index, bytes_written_by_device)`; frees the whole chain
    /// back to the free list.
    pub fn pop_used(&mut self) -> Option<(u16, u32)> {
        let used_idx = unsafe { (self.used.add(2) as *const u16).read_volatile() };
        if used_idx == self.last_used_idx {
            return None;
        }
        // Establish a before-after fence on used ring access.
        fence(Ordering::SeqCst);

        let slot = self.last_used_idx % self.size;
        let elem: VirtqUsedElem = unsafe {
            // used layout: [flags:u16][idx:u16][ring: VirtqUsedElem × size]
            let ring_base = self.used.add(4) as *const VirtqUsedElem;
            ring_base.add(slot as usize).read_volatile()
        };
        self.last_used_idx = self.last_used_idx.wrapping_add(1);

        let head = elem.id as u16;

        // Walk the chain and push every descriptor back onto the free
        // list. Re-link by adjusting their `next` fields and splicing
        // at `free_head`.
        let mut idx = head;
        let mut count = 0u16;
        loop {
            count += 1;
            let (flags, next) = unsafe {
                let d = self.desc.add(idx as usize);
                ((*d).flags, (*d).next)
            };
            if flags & VIRTQ_DESC_F_NEXT == 0 {
                // Splice: tail.next = old free_head.
                unsafe {
                    (*self.desc.add(idx as usize)).next = self.free_head;
                }
                break;
            }
            idx = next;
        }
        self.free_head = head;
        self.num_free += count;

        Some((head, elem.len))
    }

    /// `true` if the used ring has entries we haven't consumed.
    pub fn has_used(&self) -> bool {
        let used_idx = unsafe { (self.used.add(2) as *const u16).read_volatile() };
        used_idx != self.last_used_idx
    }
}
