//! Hart-local SPSC queues for `Shared`-pool backings whose last
//! `SharedUserPtr` Arc just dropped.
//!
//! Drop can fire from any kernel-thread context — including `k_net` on a
//! hart that does *not* hold the Orbit lock, so calling the frame
//! allocator inline is unsound. Instead, [`SharedInner::drop`] pushes the
//! `FreeItem` (`(Frame<Shared>, Layout)`) here and the manager drains via [`drain`] from
//! [`Orbit::cleanup_threads_and_processes`], under the Orbit lock.
//!
//! Each hart has its own `heapless::spsc::Queue<FreeItem, CAP>` split
//! into a `Producer` (owned by that hart) and `Consumer` (owned by the
//! manager). SPSC invariants hold by construction:
//!
//! - Producer side: only the owning hart pushes, since [`push`] indexes
//!   the producer array by `sscratch` → `HartContext.hart_id`.
//! - Consumer side: only one hart at a time drains, because the Orbit
//!   lock serializes "being the manager".
//!
//! The ring array is heap-allocated once at boot, sized to the detected
//! CPU count; the `AtomicPtr` lets later accessors (which run after the
//! `Release` store in [`init`]) see it with `Acquire` ordering.

use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicPtr, Ordering};

use alloc::boxed::Box;
use alloc::vec::Vec;
use heapless::spsc::{Consumer, Producer, Queue};

use process::{Frame, Shared};

use crate::kernel::context::get_hart_context;

/// Deferred-free item: a `Shared`-pool frame + its Layout. The pool tag
/// is a compile-time guarantee — only Shared frames ever queue here
/// (UserOnly backings are freed synchronously from `proc.heap_pages` at
/// teardown, not through the refcount path).
pub type FreeItem = (Frame<Shared>, Layout);

/// Per-hart free-ring capacity. Sized well above orbit's current
/// drop rate (single-digit per process teardown); overflow is a signal
/// to revisit rather than absorb silently.
pub const FREE_RING_CAP: usize = 64;

type FreeProducer = Producer<'static, FreeItem>;
type FreeConsumer = Consumer<'static, FreeItem>;

struct FreeRings {
    /// One slot per hart, indexed by `HartContext.hart_id`. Only the
    /// owning hart accesses `producers[hartid]`; `UnsafeCell` lets us
    /// get `&mut Producer` from `&FreeRings` under that invariant.
    producers: Box<[UnsafeCell<FreeProducer>]>,
    /// Mirrored consumer array. Only the current manager (serialized by
    /// the Orbit lock) accesses these, so single-consumer invariant
    /// holds even though consumers from multiple rings are drained in a
    /// single pass.
    consumers: Box<[UnsafeCell<FreeConsumer>]>,
}

// The per-hart / manager-only access pattern above means the UnsafeCells
// are never aliased across threads — so FreeRings is Sync despite
// carrying `!Sync` cells.
unsafe impl Sync for FreeRings {}

static FREE_RINGS: AtomicPtr<FreeRings> = AtomicPtr::new(null_mut());

/// Allocate and install the per-hart rings. Must be called once, from
/// hart 0 during boot, before any SharedUserPtr is constructed (which
/// means: before the first `Orbit` operation that could alloc a Shared
/// backing). A second call panics.
pub fn init(num_harts: usize) {
    assert!(num_harts > 0, "pending_frees::init: zero-hart system");
    assert!(
        FREE_RINGS.load(Ordering::Acquire).is_null(),
        "pending_frees::init: already initialized",
    );

    let mut producers = Vec::with_capacity(num_harts);
    let mut consumers = Vec::with_capacity(num_harts);
    for _ in 0..num_harts {
        // Leak the Queue so the (Producer, Consumer) borrows are 'static.
        // Drop on shutdown is not a concern — the kernel doesn't shut down.
        let queue: &'static mut Queue<FreeItem, FREE_RING_CAP> = Box::leak(Box::new(Queue::new()));
        let (prod, cons) = queue.split();
        producers.push(UnsafeCell::new(prod));
        consumers.push(UnsafeCell::new(cons));
    }

    let rings = Box::leak(Box::new(FreeRings {
        producers: producers.into_boxed_slice(),
        consumers: consumers.into_boxed_slice(),
    }));

    FREE_RINGS.store(rings as *mut FreeRings, Ordering::Release);
}

fn rings() -> &'static FreeRings {
    let p = FREE_RINGS.load(Ordering::Acquire);
    assert!(!p.is_null(), "pending_frees accessed before init");
    unsafe { &*p }
}

/// Push a Shared-pool frame + layout onto the current hart's ring. Safe
/// to call from any kernel context; indexes the producer array by the
/// hart's `sscratch`.
pub fn push(frame: Frame<Shared>, layout: Layout) {
    let hartid = get_hart_context().hart_id as usize;
    let rings = rings();
    let producer = unsafe { &mut *rings.producers[hartid].get() };
    if let Err((returned_frame, _)) = producer.enqueue((frame, layout)) {
        panic!(
            "pending_frees: hart{} ring exhausted; dropped {:?}. Raise FREE_RING_CAP or throttle drops.",
            hartid, returned_frame,
        );
    }
}

/// Drain every hart's ring, calling `f` on each (frame, layout). The
/// manager invokes this under the Orbit lock, which guarantees it is
/// the sole consumer across all rings for the duration of the call.
pub fn drain<F: FnMut(Frame<Shared>, Layout)>(mut f: F) {
    let rings = rings();
    for cell in rings.consumers.iter() {
        let consumer = unsafe { &mut *cell.get() };
        while let Some((frame, layout)) = consumer.dequeue() {
            f(frame, layout);
        }
    }
}
