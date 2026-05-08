//! Per-process structured key-event registry.
//!
//! Mirrors [`crate::kernel::stdin`] but for `ProcessKeyEvents`. The
//! producer side (`input::dispatch`) decides which pid is the active
//! pane via [`crate::kernel::stdin::active_pid`] and routes events to
//! that pid's ring; the consumer side is the `read_key_event` syscall
//! handler.
//!
//! Wake mechanism is the wake_override-via-`WAKE_QUEUE` shape, same as
//! `nc_yield` and `update_tcp` use for net I/O. See
//! [`process::ProcessKeyEvents`] for the producer/consumer ordering
//! contract.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use orbit_abi::input::KeyEvent;
use process::ProcessKeyEvents;
use spin::Mutex;

use crate::kernel::{WAKE_QUEUE, WakeEvent};

/// pid → key-event ring. Manager-side mutations (register/unregister)
/// and producer/consumer reads (input::dispatch + read_key_event) all
/// take this lock briefly.
pub static KEY_EVENTS_TABLE: Mutex<BTreeMap<u16, Arc<ProcessKeyEvents>>> =
    Mutex::new(BTreeMap::new());

/// Insert a fresh event slot for `pid`. Idempotent.
pub fn register(pid: u16) {
    let mut t = KEY_EVENTS_TABLE.lock();
    t.entry(pid).or_insert_with(ProcessKeyEvents::new);
}

/// Remove `pid`'s event slot. If a tid was parked on the ring, push
/// a `WakeEvent::InputTid` so the manager wakes that thread (the
/// resumed thread re-enters `read_key_event`, finds the missing
/// entry, and falls through to its error path — the dying process's
/// own teardown then reaps it).
pub fn unregister(pid: u16) {
    let entry = KEY_EVENTS_TABLE.lock().remove(&pid);
    if let Some(events) = entry {
        if let Some(tid) = events.take_parker() {
            let _ = WAKE_QUEUE.push(WakeEvent::InputTid(tid));
        }
        // The Arc drops here; ring + parked_tid go with it.
    }
}

/// Look up a pid's event ring.
pub fn get(pid: u16) -> Option<Arc<ProcessKeyEvents>> {
    KEY_EVENTS_TABLE.lock().get(&pid).cloned()
}

/// Producer-side fan-out. Push `ev` onto `pid`'s ring; if a reader
/// was parked, queue a kernel wake for that tid. Trap-safe — only
/// atomics + a thingbuf push.
pub fn push_and_wake(pid: u16, ev: KeyEvent) {
    let Some(events) = get(pid)
    else {
        return;
    };
    if let Some(tid) = events.push_event(ev) {
        let _ = WAKE_QUEUE.push(WakeEvent::InputTid(tid));
    }
}
