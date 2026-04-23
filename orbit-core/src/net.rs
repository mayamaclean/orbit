//! Pure-policy pieces of the `k_net` loop: revocation prune and the
//! deletion-queue drain. The driver-coupled smoltcp `iface.poll` call,
//! per-conn `update_tcp` fan-out, and smoltcp socket-construction all
//! stay in kmain — this module is the surface where the net thread's
//! bookkeeping rots silently on bugs.
//!
//! Functions are generic over the handle type `K` and the conn value
//! type `V` so orbit-core doesn't pull in smoltcp or net_channel.
//! kmain provides a one-line impl of [`RevocableConn`] for `SocketReq`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A connection entry that may have been revoked out-of-band (today by
/// the user calling `close_handle` on the underlying channel). The k_net
/// pre-poll pass uses this to find conns it should drop.
pub trait RevocableConn {
    fn is_revoked(&self) -> bool;
}

/// Remove any conn whose channel has been revoked. `scratch` is the
/// caller's reusable scratch buffer — it's cleared on entry and drained
/// before return, so repeated calls reuse the same allocation.
///
/// Two-pass because the revocation check borrows `conns` immutably for
/// the iterator, but the removal needs `&mut`. Collecting keys first
/// avoids the borrow conflict without collecting a fresh `Vec` per tick.
pub fn prune_revoked_conns<K, V, F>(
    conns: &mut BTreeMap<K, V>,
    scratch: &mut Vec<K>,
    mut remove_socket: F,
) where
    K: Copy + Ord,
    V: RevocableConn,
    F: FnMut(K),
{
    scratch.clear();
    for (k, v) in conns.iter() {
        if v.is_revoked() {
            scratch.push(*k);
        }
    }
    for k in scratch.drain(..) {
        conns.remove(&k);
        remove_socket(k);
    }
}

/// Drain the deletion queue, removing each handle from both `conns` and
/// the smoltcp socket set. Defensive: only invokes `remove_socket` if
/// the handle was actually in `conns` — the revocation prune above may
/// have removed it first, and smoltcp's `SocketSet::remove` panics on
/// an absent handle.
pub fn drain_socket_deletions<K, V, D, R>(
    conns: &mut BTreeMap<K, V>,
    mut next_deletion: D,
    mut remove_socket: R,
) where
    K: Copy + Ord,
    D: FnMut() -> Option<K>,
    R: FnMut(K),
{
    while let Some(k) = next_deletion() {
        if conns.remove(&k).is_some() {
            remove_socket(k);
        }
    }
}
