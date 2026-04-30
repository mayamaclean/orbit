//! Process environment variables — orbit-rt's user-facing env table.
//!
//! Lazy `BTreeMap<Vec<u8>, Vec<u8>>` seeded from the kernel-mapped
//! envp blob at [`orbit_abi::layout::USER_ENVP_BASE`] on first access.
//! Subsequent reads and writes go through the map; the kernel-mapped
//! page is read-only and never re-scanned, so mutations don't escape
//! the calling process.
//!
//! # Bytes, not strings
//!
//! Keys and values are byte slices. POSIX env entries are
//! conventionally UTF-8 `KEY=VALUE` but the wire format itself is
//! 8-bit clean — keeping the API in bytes preserves whatever the
//! parent packed and lets a `std`-flavored layer wrap a UTF-8 facade
//! on top without fighting our types. `OsString`-equivalent.
//!
//! # Inheritance
//!
//! Modifications stay process-local. To propagate the current env to
//! a child, snapshot via [`vars`] (or pack via
//! [`orbit_abi::envp::pack`]) into a page-aligned, page-sized buffer
//! and hand the VA to
//! [`orbit_abi::user::create_process_with_argv_envp`]. Each process
//! ends up with its own fresh envp page — POSIX-shaped inheritance
//! without shared memory.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::UnsafeCell;

use crate::SpinFlag;
use crate::argv;

struct EnvInner {
    /// First-access seeding flag. We can't pre-fill the map at
    /// `static` init time because [`BTreeMap`] doesn't have a `const`
    /// constructor reachable from `no_std` consumers, and the seed
    /// data lives at a kernel-mapped VA that isn't valid until the
    /// process is running. Lazy init under the spinlock keeps this
    /// race-free across multi-thread first access.
    initialized: bool,
    map: BTreeMap<Vec<u8>, Vec<u8>>,
}

struct EnvCell {
    lock: SpinFlag,
    inner: UnsafeCell<EnvInner>,
}

// Every access to `inner` goes through `with`, which takes the
// spinlock. The `unsafe impl Sync` carries that invariant.
unsafe impl Sync for EnvCell {}

static ENV: EnvCell = EnvCell {
    lock: SpinFlag::new(),
    inner: UnsafeCell::new(EnvInner {
        initialized: false,
        map: BTreeMap::new(),
    }),
};

/// Run `f` against the env map, lazy-initializing on first call.
/// The closure runs under the spinlock — keep it short and don't
/// reenter `with` (single-threaded reentrancy would deadlock).
fn with<R>(f: impl FnOnce(&mut BTreeMap<Vec<u8>, Vec<u8>>) -> R) -> R {
    let _g = ENV.lock.lock();
    // SAFETY: lock is held; we have exclusive access to the cell for
    // the duration of `_g`'s lifetime.
    let inner = unsafe { &mut *ENV.inner.get() };
    if !inner.initialized {
        // Each entry is `KEY=VALUE` bytes (the convention enforced by
        // the parent's `pack`). Split on the first `=`; entries with
        // no `=` are dropped — there's no key to bind them to.
        let blob = argv::envp();
        for entry in blob.iter() {
            if let Some(eq) = entry.iter().position(|&b| b == b'=') {
                let key = entry[..eq].to_vec();
                let value = entry[eq + 1..].to_vec();
                inner.map.insert(key, value);
            }
        }
        inner.initialized = true;
    }
    f(&mut inner.map)
}

/// Return the value of `key`, or `None` if it isn't set. Returns an
/// owned `Vec<u8>` so the caller can drop the spinlock immediately
/// after the lookup.
pub fn var(key: &[u8]) -> Option<Vec<u8>> {
    with(|m| m.get(key).cloned())
}

/// Snapshot every (key, value) pair currently in the env. Returned
/// vector is independent of the map — safe to iterate without
/// re-entering [`var`] / [`set_var`].
pub fn vars() -> Vec<(Vec<u8>, Vec<u8>)> {
    with(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

/// Set `key` to `value`, replacing any existing entry. Empty keys
/// are still inserted — this layer doesn't enforce POSIX's "no `=`
/// in key" rule because the kernel doesn't either; callers
/// constructing envp blobs for children are responsible for the
/// stricter validation if they need it.
pub fn set_var(key: &[u8], value: &[u8]) {
    with(|m| {
        m.insert(key.to_vec(), value.to_vec());
    });
}

/// Remove `key` from the env. No-op if it wasn't set.
pub fn remove_var(key: &[u8]) {
    with(|m| {
        m.remove(key);
    });
}
