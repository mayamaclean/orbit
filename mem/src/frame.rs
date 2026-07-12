// Adapted from the `buddy_system_allocator` crate's FrameAllocator (~v0.11):
// https://github.com/rcore-os/buddy_system_allocator
// Copyright 2019-2020 Jiajie Chen. MIT license — full text in
// THIRD_PARTY_NOTICES.md at the repo root.
//
// Orbit changes vs upstream: dropped LockedFrameAllocator and alloc_at,
// const-fn new(), added allocated()/total() accounting, added host unit
// tests.

use super::prev_power_of_two;
use core::alloc::Layout;
use core::cmp::{max, min};
use core::ops::Range;

use alloc::collections::BTreeSet;

/// A frame allocator that uses buddy system, requiring a global allocator.
///
/// The max order of the allocator is determined by the const generic parameter `ORDER` (`MAX_ORDER = ORDER - 1`).
/// The frame allocator will only be able to allocate ranges of size up to 2<sup>MAX_ORDER</sup>, out of a total
/// range of size at most 2<sup>MAX_ORDER + 1</sup> - 1.
///
/// # Usage
///
/// Create a frame allocator and add some frames to it:
/// ```
/// use mem::frame::FrameAllocator;
/// // Notice that the max order is `ORDER - 1`.
/// let mut frame = FrameAllocator::<33>::new();
/// assert!(frame.alloc(1).is_none());
///
/// frame.add_frame(0, 3);
/// let num = frame.alloc(1);
/// assert_eq!(num, Some(2));
/// let num = frame.alloc(2);
/// assert_eq!(num, Some(0));
/// ```
pub struct FrameAllocator<const ORDER: usize = 33> {
    // buddy system with max order of `ORDER - 1`
    free_list: [BTreeSet<usize>; ORDER],

    // statistics
    allocated: usize,
    total: usize,
}

impl<const ORDER: usize> FrameAllocator<ORDER> {
    /// Create an empty frame allocator
    pub const fn new() -> Self {
        Self {
            free_list: [const { BTreeSet::new() }; ORDER],
            allocated: 0,
            total: 0,
        }
    }

    /// Add a range of frame number [start, end) to the allocator
    pub fn add_frame(&mut self, start: usize, end: usize) {
        assert!(start <= end);

        let mut total = 0;
        let mut current_start = start;

        while current_start < end {
            let lowbit = if current_start > 0 {
                current_start & (!current_start + 1)
            }
            else {
                32
            };
            let size = min(
                min(lowbit, prev_power_of_two(end - current_start)),
                1 << (ORDER - 1),
            );
            total += size;

            self.free_list[size.trailing_zeros() as usize].insert(current_start);
            //.expect("failed to add frame to free list");

            current_start += size;
        }

        self.total += total;
    }

    /// Add a range of frames to the allocator.
    pub fn insert(&mut self, range: Range<usize>) {
        self.add_frame(range.start, range.end);
    }

    /// Sum of `size` values from every outstanding allocation. Same
    /// units the caller passed to `add_frame` / `alloc_aligned` —
    /// kmain wraps these with byte-address ranges, so for kmain users
    /// this is "bytes outstanding."
    pub fn allocated(&self) -> usize {
        self.allocated
    }

    /// Total capacity added via `add_frame` / `insert`. Same units as
    /// [`Self::allocated`].
    pub fn total(&self) -> usize {
        self.total
    }

    /// Allocate a range of frames from the allocator, returning the first frame of the allocated
    /// range.
    pub fn alloc(&mut self, count: usize) -> Option<usize> {
        let size = count.next_power_of_two();
        self.alloc_power_of_two(size)
    }

    /// Allocate a range of frames with the given size and alignment from the allocator, returning
    /// the first frame of the allocated range.
    /// The allocated size is the maximum of the next power of two of the given size and the
    /// alignment.
    pub fn alloc_aligned(&mut self, layout: Layout) -> Option<usize> {
        let size = max(layout.size().next_power_of_two(), layout.align());
        self.alloc_power_of_two(size)
    }

    /// Allocate a range of frames of the given size from the allocator. The size must be a power of
    /// two. The allocated range will have alignment equal to the size.
    fn alloc_power_of_two(&mut self, size: usize) -> Option<usize> {
        let class = size.trailing_zeros() as usize;
        for i in class..self.free_list.len() {
            // Find the first non-empty size class
            if !self.free_list[i].is_empty() {
                // Split buffers
                for j in (class + 1..i + 1).rev() {
                    if let Some(block_ref) = self.free_list[j].iter().next() {
                        let block = *block_ref;
                        self.free_list[j - 1].insert(block + (1 << (j - 1)));
                        //.expect("failed to add frame to free list");

                        self.free_list[j - 1].insert(block);
                        //.expect("failed to add frame to free list");

                        self.free_list[j].remove(&block);
                    }
                    else {
                        return None;
                    }
                }

                let result = self.free_list[class].iter().next();
                if let Some(result_ref) = result {
                    let result = *result_ref;
                    self.free_list[class].remove(&result);
                    self.allocated += size;
                    return Some(result);
                }
                else {
                    return None;
                }
            }
        }
        None
    }

    /// Deallocate a range of frames [frame, frame+count) from the frame allocator.
    ///
    /// The range should be exactly the same when it was allocated, as in heap allocator
    pub fn dealloc(&mut self, start_frame: usize, count: usize) {
        let size = count.next_power_of_two();
        self.dealloc_power_of_two(start_frame, size)
    }

    /// Deallocate a range of frames which was previously allocated by [`alloc_aligned`].
    ///
    /// The layout must be exactly the same as when it was allocated.
    pub fn dealloc_aligned(&mut self, start_frame: usize, layout: Layout) {
        let size = max(layout.size().next_power_of_two(), layout.align());
        self.dealloc_power_of_two(start_frame, size)
    }

    /// Deallocate a range of frames with the given size from the allocator. The size must be a
    /// power of two.
    fn dealloc_power_of_two(&mut self, start_frame: usize, size: usize) {
        let class = size.trailing_zeros() as usize;

        // Merge free buddy lists
        let mut current_ptr = start_frame;
        let mut current_class = class;
        while current_class < self.free_list.len() {
            let buddy = current_ptr ^ (1 << current_class);
            if self.free_list[current_class].remove(&buddy) {
                // Free buddy found
                current_ptr = min(current_ptr, buddy);
                current_class += 1;
            }
            else {
                self.free_list[current_class].insert(current_ptr);
                //.expect("failed to add frame to free list");

                break;
            }
        }

        self.allocated -= size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_allocator_returns_none() {
        let mut fa = FrameAllocator::<10>::new();
        assert!(fa.alloc(1).is_none());
        assert!(fa.alloc(8).is_none());
        assert_eq!(fa.total, 0);
        assert_eq!(fa.allocated, 0);
    }

    #[test]
    fn add_frame_updates_total() {
        let mut fa = FrameAllocator::<10>::new();
        fa.add_frame(0, 8);
        assert_eq!(fa.total, 8);
        assert_eq!(fa.allocated, 0);
    }

    #[test]
    fn alloc_round_trip_dealloc_restores_availability() {
        let mut fa = FrameAllocator::<10>::new();
        fa.add_frame(0, 8);

        let a = fa.alloc(1).unwrap();
        assert_eq!(fa.allocated, 1);

        fa.dealloc(a, 1);
        assert_eq!(fa.allocated, 0);

        // Can alloc again.
        let b = fa.alloc(1).unwrap();
        assert_eq!(fa.allocated, 1);
        fa.dealloc(b, 1);
    }

    #[test]
    fn alloc_count_rounds_up_to_power_of_two() {
        let mut fa = FrameAllocator::<10>::new();
        fa.add_frame(0, 16);
        // Request 3 → rounds to 4, consuming 4 frames.
        let _ = fa.alloc(3).unwrap();
        assert_eq!(fa.allocated, 4);
    }

    #[test]
    fn alloc_aligned_satisfies_alignment() {
        use core::alloc::Layout;
        let mut fa = FrameAllocator::<20>::new();
        fa.add_frame(0, 1024);

        let layout = Layout::from_size_align(3, 16).unwrap();
        let a = fa.alloc_aligned(layout).unwrap();
        // The returned frame number is aligned to max(size.next_power_of_two, align) = 16.
        assert_eq!(a % 16, 0);
    }

    #[test]
    fn exhaust_then_dealloc_all_restores_initial() {
        let mut fa = FrameAllocator::<10>::new();
        fa.add_frame(0, 8);

        let mut allocations = alloc::vec::Vec::new();
        while let Some(a) = fa.alloc(1) {
            allocations.push(a);
        }
        assert_eq!(allocations.len(), 8, "should exhaust the 8-frame range");
        assert_eq!(fa.allocated, 8);
        assert!(fa.alloc(1).is_none());

        for a in allocations {
            fa.dealloc(a, 1);
        }
        assert_eq!(fa.allocated, 0);

        // Whole range should be available again.
        let big = fa.alloc(8).unwrap();
        assert_eq!(fa.allocated, 8);
        fa.dealloc(big, 8);
    }

    #[test]
    fn dealloc_merges_buddies_back_into_larger_class() {
        // If the merge path works, after freeing two adjacent 1-frame
        // allocs from a fresh 2-frame range, we should be able to
        // allocate a 2-frame block.
        let mut fa = FrameAllocator::<10>::new();
        fa.add_frame(0, 2);

        let a = fa.alloc(1).unwrap();
        let b = fa.alloc(1).unwrap();
        assert!(fa.alloc(1).is_none(), "exhausted");

        fa.dealloc(a, 1);
        fa.dealloc(b, 1);

        // Both freed + merged → 2-frame alloc should succeed.
        let two = fa.alloc(2);
        assert!(two.is_some(), "merged buddies should form a 2-frame class");
        fa.dealloc(two.unwrap(), 2);
    }

    #[test]
    fn insert_range_matches_add_frame() {
        let mut a = FrameAllocator::<10>::new();
        a.add_frame(4, 12);
        let mut b = FrameAllocator::<10>::new();
        b.insert(4..12);
        assert_eq!(a.total, b.total);
    }
}
