use core::{
    alloc::GlobalAlloc,
    ptr::null_mut,
    sync::atomic::{AtomicUsize, Ordering},
};

use linked_list_allocator::LockedHeap;

#[global_allocator]
pub static KHEAP: TrackedHeap = TrackedHeap::empty();

pub struct TrackedHeap {
    inner: LockedHeap,
    allocated: AtomicUsize,
    capacity: AtomicUsize,
}

impl TrackedHeap {
    pub const fn empty() -> Self {
        Self {
            inner: LockedHeap::empty(),
            allocated: AtomicUsize::new(0),
            capacity: AtomicUsize::new(0),
        }
    }

    /// safety: only to be called during kernel init, prior to other harts
    /// coming online and requesting heap allocations
    pub unsafe fn init(&self, heap_bottom: *mut u8, heap_size: usize) {
        self.capacity.store(heap_size, Ordering::Relaxed);
        self.allocated.store(0, Ordering::Relaxed);
        unsafe {
            self.inner
                .make_guard_unchecked()
                .init(heap_bottom, heap_size);
        }
    }

    pub fn free_bytes(&self) -> usize {
        self.capacity.load(Ordering::Relaxed) - self.allocated.load(Ordering::Relaxed)
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

unsafe impl GlobalAlloc for TrackedHeap {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };

        if ptr == null_mut() {
            return ptr;
        }

        self.allocated.fetch_add(layout.size(), Ordering::Relaxed);

        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        unsafe {
            self.inner.dealloc(ptr, layout);
        }
        self.allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}
