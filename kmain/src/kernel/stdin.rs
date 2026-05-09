//! Per-process stdin registry + active-pid mirror.
//!
//! [`STDIN_TABLE`] is the global map of `pid → Arc<ProcessStdin>`. The
//! Arc is what producers ([`crate::kernel::input::dispatch`]) and the
//! consumer (`read_stdin` syscall handler on the user's hart) share.
//! The map is touched at:
//!
//! - process create — `register(pid)` inserts a fresh `ProcessStdin`.
//! - `dealloc_process` — `unregister(pid)` removes the entry, signaling
//!   any parked reader so the blocked thread doesn't dangle.
//! - syscall / input fan-out — `get(pid)` looks up the Arc.
//!
//! [`ACTIVE_PID`] mirrors the [`Display`]'s active source as an atomic
//! `i32` (-1 = `Source::Kernel`; ≥0 = `Source::Process(pid)`). Updated
//! by [`set_active`] from k_gpu's drain loop, read lock-free by
//! `input::dispatch` to decide which pid to fan keystrokes out to.

use core::sync::atomic::{AtomicI32, Ordering};

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use process::ProcessStdin;
use spin::Mutex;

use crate::drivers::display::Source;

/// Sentinel for "no active process — keystrokes go nowhere."
pub const NO_ACTIVE: i32 = -1;

/// Atomic mirror of `Display::active`. Stored as i32 so a single
/// atomic load can encode both the "no process" case (-1) and the
/// "Process(pid)" case (≥0). Lock-free reads from any context;
/// updated by [`set_active`] only from k_gpu.
pub static ACTIVE_PID: AtomicI32 = AtomicI32::new(NO_ACTIVE);

/// The pid → stdin map. Manager-side mutations (register/unregister)
/// and producer/consumer reads (input::dispatch + read_stdin) all
/// take this lock briefly. Critical sections are O(log n) BTreeMap
/// lookups and an Arc clone — short enough that brief trap-context
/// spin is acceptable.
pub static STDIN_TABLE: Mutex<BTreeMap<u16, Arc<ProcessStdin>>> = Mutex::new(BTreeMap::new());

/// Insert a fresh stdin slot for `pid`. Idempotent: if `pid` already
/// has an entry (shouldn't happen post-§9 but cheap to defend), the
/// existing entry is kept.
pub fn register(pid: u16) {
    let mut t = STDIN_TABLE.lock();
    t.entry(pid).or_insert_with(ProcessStdin::new);
}

/// Remove `pid`'s stdin slot. If a reader was parked on it, push a
/// `WakeEvent::InputTid(tid)` so the manager unblocks the thread;
/// the resumed thread re-enters `read_stdin` which sees the missing
/// entry and returns an error. Mirrors
/// `key_events::unregister`'s shape now that both rings use the
/// on-thread completion path.
pub fn unregister(pid: u16) {
    let entry = STDIN_TABLE.lock().remove(&pid);
    if let Some(stdin) = entry {
        if let Some(tid) = stdin.unpark() {
            let _ = crate::kernel::wake_queue_push(crate::kernel::WakeEvent::InputTid(tid));
        }
        // `stdin` (the Arc) drops here. No Arc-reclaim sleight of
        // hand needed post-Phase-6 — the parked-tid slot is just an
        // AtomicU32, no resources tied to it.
    }
}

/// Look up a pid's stdin. Returns a clone of the Arc so the caller
/// can drop the lock before working with the ring.
pub fn get(pid: u16) -> Option<Arc<ProcessStdin>> {
    STDIN_TABLE.lock().get(&pid).cloned()
}

/// Update the active-pid mirror to match `Display::active`. Called
/// from k_gpu's drain loop after every mutation that could change
/// `active` (cycle / insert / remove). One atomic store; cheap.
pub fn set_active(source: Source) {
    let v = match source {
        Source::Kernel => NO_ACTIVE,
        Source::Process(pid) => pid as i32,
    };
    ACTIVE_PID.store(v, Ordering::Release);
}

/// Read the active pid lock-free. `None` = the kernel pane is active
/// (or display hasn't initialized yet); keystrokes should be floored.
pub fn active_pid() -> Option<u16> {
    let v = ACTIVE_PID.load(Ordering::Acquire);
    if v < 0 { None } else { Some(v as u16) }
}
