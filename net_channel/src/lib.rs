#![no_std]

mod spsc;

use core::marker::PhantomData;
use core::mem::size_of;
use core::sync::atomic::{AtomicI32, AtomicU16, AtomicU32, AtomicUsize, Ordering};

pub use spsc::SpscQueue;

/// Round `value` up to the next multiple of `alignment` (which must be
/// a power of two). Inlined here so the user-side surface of this crate
/// has zero non-core dependencies — that's the precondition for
/// path-depping it from `library/std` under `rustc-dep-of-std`.
#[inline]
fn round_usize_up(value: usize, alignment: usize) -> usize {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(feature = "kernel")]
use core::net::Ipv4Addr;
#[cfg(feature = "kernel")]
use smoltcp::socket::tcp::State as TcpState;
#[cfg(feature = "kernel")]
use smoltcp::{iface::Interface, wire::IpAddress};

#[cfg(feature = "kernel")]
use tracing::{error, info, trace};

/// Mutable view into a region of shared memory handed to `send_tcp`'s
/// closure. Writes go through [`core::ptr::write_volatile`], which
/// LLVM can't DCE the way it can writes through `&mut [u8]` — that
/// slice type carries `noalias`, so LLVM assumes nothing else can
/// observe the writes and deletes them (the bug that was silently
/// zeroing outbound TCP payloads).
pub struct VolSliceMut<'a> {
    ptr: *mut u8,
    len: usize,
    _m: PhantomData<&'a mut [u8]>,
}

impl<'a> VolSliceMut<'a> {
    /// # Safety
    /// `ptr` must be valid for writes of `len` bytes, and no other
    /// `&mut [u8]` may alias this region for the `'a` lifetime.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _m: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy up to `self.len()` bytes from `src`, returning the number
    /// actually written.
    pub fn copy_from_slice(&self, src: &[u8]) -> usize {
        let n = core::cmp::min(self.len, src.len());
        for i in 0..n {
            unsafe {
                self.ptr.add(i).write_volatile(src[i]);
            }
        }
        n
    }

    /// Read the byte at `i`. Panics if out of bounds (matches `slice[i]`).
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        assert!(
            i < self.len,
            "VolSliceMut::get: index {i} out of bounds (len {})",
            self.len
        );
        unsafe { self.ptr.add(i).read_volatile() }
    }

    /// Bounds-checked read. Returns `None` if out of range.
    #[inline]
    pub fn get_checked(&self, i: usize) -> Option<u8> {
        if i >= self.len {
            return None;
        }
        Some(unsafe { self.ptr.add(i).read_volatile() })
    }

    /// Write `b` at `i`. Panics if out of bounds.
    #[inline]
    pub fn set(&self, i: usize, b: u8) {
        assert!(
            i < self.len,
            "VolSliceMut::set: index {i} out of bounds (len {})",
            self.len
        );
        unsafe {
            self.ptr.add(i).write_volatile(b);
        }
    }

    /// Bounds-checked write. Returns `None` if out of range.
    #[inline]
    pub fn set_checked(&self, i: usize, b: u8) -> Option<()> {
        if i >= self.len {
            return None;
        }
        unsafe {
            self.ptr.add(i).write_volatile(b);
        }
        Some(())
    }

    /// Legacy name for [`set_checked`]. Kept for callers that already
    /// use the `write_at` API; new code should prefer `set` / `set_checked`.
    #[inline]
    pub fn write_at(&self, offset: usize, b: u8) -> Option<()> {
        self.set_checked(offset, b)
    }

    /// Sub-slice `[start, end)`. Panics if out of bounds.
    pub fn sub(&self, start: usize, end: usize) -> VolSliceMut<'_> {
        assert!(start <= end, "VolSliceMut::sub: start {start} > end {end}");
        assert!(
            end <= self.len,
            "VolSliceMut::sub: end {end} > len {}",
            self.len
        );
        unsafe { VolSliceMut::from_raw_parts(self.ptr.add(start), end - start) }
    }

    /// Read-only reborrow. Useful when a function wants a `VolSlice`
    /// but the caller only has a `VolSliceMut`.
    #[inline]
    pub fn as_readonly(&self) -> VolSlice<'_> {
        unsafe { VolSlice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Read-only view of a region of shared memory handed to `recv_tcp`'s
/// closure. Reads go through [`core::ptr::read_volatile`] — symmetric
/// with [`VolSliceMut`] so the compiler can't cache stale values that
/// may have been written by the other side of the channel.
pub struct VolSlice<'a> {
    ptr: *const u8,
    len: usize,
    _m: PhantomData<&'a [u8]>,
}

impl<'a> VolSlice<'a> {
    /// # Safety
    /// `ptr` must be valid for reads of `len` bytes.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _m: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy up to `dst.len()` bytes into `dst`, returning the number
    /// actually copied.
    pub fn copy_to_slice(&self, dst: &mut [u8]) -> usize {
        let n = core::cmp::min(self.len, dst.len());
        for i in 0..n {
            dst[i] = unsafe { self.ptr.add(i).read_volatile() };
        }
        n
    }

    /// Read the byte at `i`. Panics if out of bounds (matches `slice[i]`).
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        assert!(
            i < self.len,
            "VolSlice::get: index {i} out of bounds (len {})",
            self.len
        );
        unsafe { self.ptr.add(i).read_volatile() }
    }

    /// Bounds-checked read. Returns `None` if out of range.
    #[inline]
    pub fn get_checked(&self, i: usize) -> Option<u8> {
        if i >= self.len {
            return None;
        }
        Some(unsafe { self.ptr.add(i).read_volatile() })
    }

    /// Legacy name for [`get_checked`]. New code should prefer `get`
    /// (panic on OOB) or `get_checked`.
    #[inline]
    pub fn read_at(&self, offset: usize) -> Option<u8> {
        self.get_checked(offset)
    }

    /// Sub-slice `[start, end)`. Panics if out of bounds.
    pub fn sub(&self, start: usize, end: usize) -> VolSlice<'_> {
        assert!(start <= end, "VolSlice::sub: start {start} > end {end}");
        assert!(
            end <= self.len,
            "VolSlice::sub: end {end} > len {}",
            self.len
        );
        unsafe { VolSlice::from_raw_parts(self.ptr.add(start), end - start) }
    }

    /// True if the first `prefix.len()` bytes equal `prefix`.
    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        if self.len < prefix.len() {
            return false;
        }
        for i in 0..prefix.len() {
            if unsafe { self.ptr.add(i).read_volatile() } != prefix[i] {
                return false;
            }
        }
        true
    }

    /// Raw pointer, for syscalls that read this memory from a
    /// non-compiler context (e.g. `serial_print`, whose read happens
    /// in the kernel under SUM — the compiler can't elide the syscall).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

// SPSC queue lives in this crate's [`spsc`] module — `pub use`
// re-exported above. Used to live in `process::spsc`, but pulling
// `process` into `library/std` (`rustc-dep-of-std`) is impractical
// because of its kernel-side deps (mmu, riscv, smoltcp). The `process`
// crate keeps an independent identical copy for kernel-side use; the
// two layouts (`#[repr(C)]`) are byte-for-byte compatible by
// construction. Other kernel sync paths (e.g. `ProcessStdin`) still
// import `process::SpscQueue`.

/// Fixed header offsets within a NetChannel region. The tx/rx queues follow
/// at `NC_TX_OFF` and `NC_TX_OFF + queue_len` respectively — `queue_len` is
/// runtime-selectable by the user at channel creation, up to
/// [`NC_MAX_REGION_SIZE`].
pub const NC_DESIRED_OFF: usize = 128;
pub const NC_CURRENT_OFF: usize = 256;
pub const NC_TX_OFF: usize = 384;

/// Minimum region size. Everything — states, tx, rx — packs into a single
/// 4 KiB page. Per-ring usable payload is
/// `(NC_MIN_REGION_SIZE - NC_TX_OFF) / 2 - size_of::<NetChannelQueue>() + 1`
/// (roughly ~1.7 KiB).
pub const NC_MIN_REGION_SIZE: usize = 4096;

/// Maximum region size. Cap at 8 MiB so misbehaving umode can't demand an
/// arbitrarily large kernel-side Shared allocation. Per-ring usable payload
/// at the cap is ~4 MiB.
pub const NC_MAX_REGION_SIZE: usize = 8 * 1024 * 1024;

// Compile-time checks so layout invariants fail the build, not the boot.
// If these fire, `NC_TX_OFF` / `NetChannelQueue` layout / min-region math
// have drifted: either the fixed header offsets or `queue_len_for` need
// updating.
const _: () = {
    assert!(
        NC_TX_OFF % core::mem::align_of::<NetChannelQueue>() == 0,
        "NC_TX_OFF must be aligned for NetChannelQueue",
    );
    assert!(
        NetChannel::queue_len_for(NC_MIN_REGION_SIZE) % core::mem::align_of::<NetChannelQueue>()
            == 0,
        "queue_len at NC_MIN_REGION_SIZE must align the rx subregion",
    );
    assert!(
        NetChannel::capacity_for(NC_MIN_REGION_SIZE) > 0,
        "NC_MIN_REGION_SIZE leaves no room for a ring payload",
    );
    // Each control struct must fit in its 128-byte slot and own its
    // cache line — both for false-sharing isolation and so the
    // per-slot `add(NC_*_OFF)` accessor lands on a properly-aligned
    // pointer for the struct's `repr(align(128))`.
    assert!(
        core::mem::size_of::<NetChannelDesired>() == 128,
        "NetChannelDesired must occupy exactly its 128-byte slot",
    );
    assert!(
        core::mem::align_of::<NetChannelDesired>() == 128,
        "NetChannelDesired must be cache-line aligned",
    );
    assert!(
        core::mem::size_of::<NetChannelCurrent>() == 128,
        "NetChannelCurrent must occupy exactly its 128-byte slot",
    );
    assert!(
        core::mem::align_of::<NetChannelCurrent>() == 128,
        "NetChannelCurrent must be cache-line aligned",
    );
    // The two control slots must not overlap each other or the rings.
    assert!(
        NC_DESIRED_OFF + core::mem::size_of::<NetChannelDesired>() <= NC_CURRENT_OFF,
        "NetChannelDesired runs into NetChannelCurrent slot",
    );
    assert!(
        NC_CURRENT_OFF + core::mem::size_of::<NetChannelCurrent>() <= NC_TX_OFF,
        "NetChannelCurrent runs into the tx ring",
    );
};

/// User-side per-session signal. The user writes `engaged = 1` when
/// they want to claim the next session that lands; they write `0` to
/// release it (typically via `Session::drop`). The kernel observes a
/// `1 → 0` transition and recycles the smoltcp socket — for retain
/// bindings, recycling immediately re-arms (re-listen / re-connect);
/// for one-shot bindings, recycling moves to the terminal state.
///
/// Only this single flag lives in the desired slot; per-binding
/// addresses and ports live kernel-side in `ChannelCtx::bind` (set at
/// channel creation, not user-mutable).
#[repr(C, align(128))]
pub struct NetChannelDesired {
    pub engaged: AtomicU32,
    _pad: [u8; 124],
}

/// User-visible session-state values published in
/// [`NetChannelCurrent::state`]. The numeric values are part of the
/// kernel/user ABI — user code polls the state byte through the
/// shared NetChannel mapping, kernel code writes it via
/// [`NetChannel::update_tcp`]. Don't renumber.
///
/// Use the constants below rather than literal numbers; the older
/// ad-hoc `if state == 2` checks made it easy to forget about
/// transient states like [`CLOSING`] (state 3).
pub mod channel_state {
    /// Idle. Server retain: kernel listening but no peer yet
    /// connected. Client retain: between reconnect attempts. Client
    /// one-shot: kernel waiting for `engaged = 1`.
    pub const IDLE: i32 = 0;
    /// In flight. Server: listen freshly armed but no peer. Client:
    /// dialing (smoltcp `SynSent`).
    pub const IN_FLIGHT: i32 = 1;
    /// Active session: peer connected, rings are live, user may
    /// `send_tcp` / `recv_tcp`.
    pub const ACTIVE: i32 = 2;
    /// Closing/draining. User has disengaged and the kernel is
    /// driving smoltcp through the graceful-close handshake (FIN +
    /// final ACKs) so any data the user wrote just before drop
    /// actually reaches the peer. Set on the disengage edge in
    /// `update_tcp`; cleared when the kernel observes
    /// `socket.state() == Closed` and transitions to whatever the
    /// binding wants next ([`IN_FLIGHT`] for server retain re-listen,
    /// [`IDLE`] for client retain re-dial pending, [`FAILED`] for
    /// one-shot terminal).
    ///
    /// User code that wants "kernel has fully released the rings"
    /// must wait for state to leave both [`ACTIVE`] *and* [`CLOSING`]
    /// — the older "wait for state != ACTIVE" idiom raced
    /// `close_handle` against the in-flight close handshake and
    /// dropped the last `write_all`.
    pub const CLOSING: i32 = 3;
    /// Sticky terminal failure. Consult
    /// [`super::NetChannelCurrent::fail_cause`] for the
    /// `EBIND_*`-flavoured reason.
    pub const FAILED: i32 = -1;
}

/// Kernel-published per-session observation. Layout invariants:
/// - `state` is the only field user code polls in `wait_for_session`-
///   style loops; it's the first field for cache-friendliness.
/// - `peer_addr` / `peer_port` are populated when `state` reaches
///   [`channel_state::ACTIVE`] and reflect the connection's *remote*
///   endpoint (server bindings: the peer that connected; client
///   bindings: redundant with the bind params, exposed for symmetry).
/// - `fail_cause` is meaningful only when `state == channel_state::FAILED`;
///   otherwise 0.
///
/// See [`channel_state`] for the state-value constants and their
/// transitions.
#[repr(C, align(128))]
pub struct NetChannelCurrent {
    pub state: AtomicI32,
    pub peer_addr: AtomicU32,
    pub peer_port: AtomicU16,
    _pad0: [u8; 2],
    pub fail_cause: AtomicI32,
    _pad1: [u8; 112],
}

/// Sticky binding spec — set once at channel creation, latched into
/// `ChannelCtx::bind` kernel-side and never user-mutable thereafter.
/// Encoded into a single `usize` for register passing through the
/// `create_netch` syscall (see [`Self::pack`] / [`Self::unpack`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindSpec {
    /// Connect once to `(addr, port)`. After the user disengages or the
    /// connect fails, the binding goes to the terminal state.
    ClientOneShot { addr: u32, port: u16 },
    /// Connect to `(addr, port)`; reconnect after disconnects with
    /// capped exponential backoff. Channel maintains the connection
    /// from create until close.
    ClientRetain { addr: u32, port: u16 },
    /// Listen on `port`, accept one peer, then go terminal once the
    /// user disengages.
    ServerOneShot { port: u16 },
    /// Listen on `port`; after each session ends, immediately re-arm the
    /// listen so back-to-back peers don't race a closed-window.
    ServerRetain { port: u16 },
}

impl BindSpec {
    /// Pack into 64 bits for `create_netch` register passing:
    /// - bits  0..8   mode tag (1..=4; 0 reserved for "invalid")
    /// - bits  8..24  port (u16; local-port for server, remote for client)
    /// - bits 24..56  IPv4 addr (u32; 0 for server modes)
    /// - bits 56..64  reserved (must be 0)
    pub fn pack(self) -> usize {
        match self {
            BindSpec::ClientOneShot { addr, port } => {
                1 | ((port as usize) << 8) | ((addr as usize) << 24)
            }
            BindSpec::ClientRetain { addr, port } => {
                2 | ((port as usize) << 8) | ((addr as usize) << 24)
            }
            BindSpec::ServerOneShot { port } => 3 | ((port as usize) << 8),
            BindSpec::ServerRetain { port } => 4 | ((port as usize) << 8),
        }
    }

    /// Inverse of [`pack`]. Returns `None` for an unknown mode tag, a
    /// zero port (TCP wildcard isn't supported here), or non-zero
    /// reserved bits — those indicate either a stale sender or a
    /// future ABI extension we don't understand.
    pub fn unpack(packed: usize) -> Option<Self> {
        if packed >> 56 != 0 {
            return None;
        }
        let mode = packed & 0xff;
        let port = ((packed >> 8) & 0xffff) as u16;
        let addr = ((packed >> 24) & 0xffff_ffff) as u32;
        if port == 0 {
            return None;
        }
        match mode {
            1 => Some(BindSpec::ClientOneShot { addr, port }),
            2 => Some(BindSpec::ClientRetain { addr, port }),
            3 if addr == 0 => Some(BindSpec::ServerOneShot { port }),
            4 if addr == 0 => Some(BindSpec::ServerRetain { port }),
            _ => None,
        }
    }

    pub fn is_server(self) -> bool {
        matches!(
            self,
            BindSpec::ServerOneShot { .. } | BindSpec::ServerRetain { .. }
        )
    }

    pub fn is_retain(self) -> bool {
        matches!(
            self,
            BindSpec::ClientRetain { .. } | BindSpec::ServerRetain { .. }
        )
    }
}

/// Kernel-side reconciler phase, paired 1:1 with each NetChannel by the
/// kernel-side [`ChannelCtx`]. Tracks where smoltcp is in its state
/// machine relative to the user's binding, so [`NetChannel::update_tcp`]
/// can drive transitions without re-deriving them from smoltcp's
/// `Socket::state()` every poll.
#[cfg(feature = "kernel")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// First poll after channel creation, or post-recycle-into-retain.
    /// Server bindings transition straight to `Listening`; client retain
    /// transitions to `Connecting` (subject to backoff); client one-shot
    /// waits here until `engaged == 1`.
    FreshIdle,
    /// smoltcp is in `Listen` for a server binding.
    Listening,
    /// smoltcp is in `SynSent` for a client binding.
    Connecting,
    /// smoltcp is past handshake (Established or a close-state with
    /// buffered data); `current.state == 2`. Stays here until the user
    /// disengages.
    Active,
    /// Graceful-close drain. User just disengaged; we issued
    /// `socket.close()` (FIN-on-empty) and are waiting for smoltcp to
    /// drive the close handshake to `Closed` so any data the
    /// application wrote just before drop actually reaches the peer.
    ///
    /// Skipping this phase (the old "abort + reset on the disengage
    /// edge" path) discarded any tx queued in the final `write_all` —
    /// the response a listener wrote immediately before drop never
    /// made it to the peer.
    Closing,
    /// Sticky terminal. `current.state == -1` and `fail_cause` carries
    /// the errno-flavored explanation. One-shot bindings end here; retain
    /// bindings only enter Failed on a non-recoverable error
    /// (e.g. `socket.listen()` rejected at bind).
    Failed,
}

/// Kernel-owned per-channel reconciler state. Lives inside the kernel's
/// `SocketReq`; threaded into [`NetChannel::update_tcp`] each poll so
/// the reconciler can advance phases without owning the SocketReq.
#[cfg(feature = "kernel")]
#[derive(Debug)]
pub struct ChannelCtx {
    pub bind: BindSpec,
    pub phase: Phase,
    /// "A slice is enqueued on rx.slices and we haven't yet drained the
    /// matching increment." Mirror of the user-side ack flag — gates
    /// re-enqueue so we don't deposit a duplicate slice while smoltcp
    /// hasn't been told the prior bytes were consumed.
    pub pending_rx_ack: bool,
    /// Same invariant on the tx side.
    pub pending_tx_ack: bool,
    /// Last engaged value observed. We recycle on the `1 → 0` edge
    /// rather than on every poll where engaged is `0`, otherwise a
    /// fresh-but-unclaimed session would get torn down before the user
    /// ever saw it.
    pub last_engaged: bool,
    /// Capped-exponential backoff for client-retain reconnects, in
    /// milliseconds. `0` means "next attempt is immediate."
    pub backoff_ms: u32,
    /// Microsecond timestamp (matches the iface clock) at which a
    /// client-retain reconnect is allowed to run. `0` if not waiting.
    pub next_attempt_at_us: u64,
}

#[cfg(feature = "kernel")]
impl ChannelCtx {
    pub fn new(bind: BindSpec) -> Self {
        Self {
            bind,
            phase: Phase::FreshIdle,
            pending_rx_ack: false,
            pending_tx_ack: false,
            last_engaged: false,
            backoff_ms: 0,
            next_attempt_at_us: 0,
        }
    }
}

/// User-visible side effects of a single [`NetChannel::update_tcp`]
/// poll. The kernel net loop reads this to decide whether the
/// channel's owner thread should be woken now (via the kernel's
/// `WAKE_QUEUE`) instead of waiting for the user thread's own poll
/// cadence. `#[must_use]` catches the "called update_tcp and forgot
/// to act on the outcome" mistake at compile time — the kind of
/// thing that costs a week of mystery latency.
#[cfg(feature = "kernel")]
#[must_use]
#[derive(Default, Clone, Copy, Debug)]
pub struct UpdateOutcome {
    /// `current.state` was written to a new value (Listening→Active,
    /// recycle, Failed, etc.). User threads parked on `next_session`
    /// or `wait_for_link` care about this.
    pub session_state_changed: bool,
    /// User-visible ring state moved: a fresh rx slice was staged
    /// (user can read more), or a tx increment was drained / tx
    /// avail grew (user's blocked write has space). User threads
    /// in `read_some` / `write_all`'s sleep loops care.
    pub ring_progress: bool,
}

#[cfg(feature = "kernel")]
impl UpdateOutcome {
    /// True if the owner thread should be woken (i.e. the kernel
    /// observed something it would otherwise have spin-polled for
    /// at the user-side `sleep_ms` cadence).
    pub fn should_wake_user(self) -> bool {
        self.session_state_changed || self.ring_progress
    }

    pub const fn new(session_state_changed: bool, ring_progress: bool) -> Self {
        Self {
            session_state_changed,
            ring_progress,
        }
    }
}

/// Capped exponential backoff for client-retain reconnects. Doubles per
/// failure starting at 100 ms, caps at 30 s.
#[cfg(feature = "kernel")]
const RETAIN_BACKOFF_MIN_MS: u32 = 100;
#[cfg(feature = "kernel")]
const RETAIN_BACKOFF_CAP_MS: u32 = 30_000;

// Sticky-failure cause codes reported via `NetChannelCurrent::fail_cause`
// when `state == -1`. These are *not* posix errno values — they're a
// channel-local namespace so userspace can distinguish "the bind itself
// was rejected" from "the connect failed" from "an in-flight read/write
// hit a smoltcp error." Currently used only kernel-side (the user-side
// netch wrapper in orbit-rt translates them when surfacing results).
pub const EBIND_LISTEN: u16 = 1; // smoltcp listen() rejected — bad port
pub const EBIND_CONNECT: u16 = 2; // connect failed (RST / timeout) — one-shot
pub const EBIND_DONE: u16 = 3; // one-shot session completed; binding spent
pub const EBIND_IO: u16 = 4; // smoltcp recv/send returned an error mid-session

#[cfg(feature = "kernel")]
fn try_connect(
    iface: &mut Interface,
    socket: &mut smoltcp::socket::tcp::Socket,
    addr: u32,
    port: u16,
) -> Result<(), ()> {
    let dst = IpAddress::Ipv4(Ipv4Addr::from_bits(addr));
    // Local port: ephemeral. Smoltcp doesn't auto-allocate; pick a fixed
    // value for now — collisions across multiple client-retain channels
    // would matter, but we have at most one user per process today.
    // Future: take from a per-process ephemeral allocator.
    match socket.connect(iface.context(), (dst, port), 49152u16) {
        Ok(()) => Ok(()),
        Err(e) => {
            error!("tcp: failed to start connect: {e:?}");
            Err(())
        }
    }
}

/// Snapshot smoltcp's remote endpoint into the user-visible `peer_*`
/// fields. Wrapped in a helper because the IpAddress-variant match is
/// irrefutable under our smoltcp build (no `proto-ipv6`), but a future
/// IPv6 cutover should fail to compile a single match arm rather than
/// drop both call sites' allow-attributes.
#[cfg(feature = "kernel")]
#[allow(irrefutable_let_patterns)]
fn publish_peer(socket: &smoltcp::socket::tcp::Socket, cur: &NetChannelCurrent) {
    if let Some(ep) = socket.remote_endpoint() {
        if let IpAddress::Ipv4(v4) = ep.addr {
            cur.peer_addr.store(u32::from(v4), Ordering::Relaxed);
        }
        cur.peer_port.store(ep.port, Ordering::Relaxed);
    }
}

#[cfg(feature = "kernel")]
fn schedule_retry(ctx: &mut ChannelCtx, now_us: u64) {
    let next = if ctx.backoff_ms == 0 {
        RETAIN_BACKOFF_MIN_MS
    }
    else {
        ctx.backoff_ms.saturating_mul(2).min(RETAIN_BACKOFF_CAP_MS)
    };
    ctx.backoff_ms = next;
    ctx.next_attempt_at_us = now_us.saturating_add((next as u64) * 1000);
}

/// Ring holding `(offset, len)` pairs pointing into [`NetChannelQueue::buf`].
///
/// Sized to match the e1000 RX ring (8 descriptors): a PLIC IRQ batch
/// can land 8 frames into smoltcp at once, and we want the staging path
/// to absorb that without forcing a `ch_yield` round-trip per frame.
/// N=16 → capacity 15 (one slot reserved for the empty/full sentinel),
/// so up to 15 staged slices in flight before the user has to wait on
/// the kernel to drain.
type SliceQueue = SpscQueue<(usize, usize), 16>;
/// Ring of byte counts the consumer has advanced past. Sized in lockstep
/// with `SliceQueue` so the pipelining is symmetric: every staged slice
/// has room for its eventual increment without backpressuring user-side
/// reads. N=16 → capacity 15.
type IncrementQueue = SpscQueue<usize, 16>;

/// Shared producer/consumer queues + payload ring for one direction.
/// `#[repr(C)]` pins the field order so kernel- and user-side
/// compilations land at the same offsets.
#[repr(C)]
pub struct NetChannelQueue {
    slices: SliceQueue,
    increments: IncrementQueue,
    pub avail: AtomicUsize,
    capacity: usize,
    buf: u8,
}

impl NetChannelQueue {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn buf_ptr(&self) -> *mut u8 {
        &self.buf as *const u8 as *mut u8
    }

    /// # Safety
    /// Caller must be the single producer for `slices` on this queue.
    pub unsafe fn enqueue_slice(&self, v: (usize, usize)) -> Result<(), (usize, usize)> {
        unsafe { self.slices.enqueue(v) }
    }

    /// # Safety
    /// Caller must be the single consumer for `slices` on this queue.
    pub unsafe fn dequeue_slice(&self) -> Option<(usize, usize)> {
        unsafe { self.slices.dequeue() }
    }

    pub fn slices_is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// `true` when the slice ring is full and a producer would block
    /// on the next `enqueue_slice`. Used by selector-style readiness
    /// scans to determine if the queue is currently writable without
    /// issuing a syscall — pure shared-memory load against the SPSC
    /// head/tail indices.
    pub fn slices_is_full(&self) -> bool {
        self.slices.is_full()
    }
    pub fn slices_len(&self) -> usize {
        self.slices.len()
    }

    /// # Safety
    /// Caller must be the single producer for `increments` on this queue.
    pub unsafe fn enqueue_increment(&self, v: usize) -> Result<(), usize> {
        unsafe { self.increments.enqueue(v) }
    }

    /// # Safety
    /// Caller must be the single consumer for `increments` on this queue.
    pub unsafe fn dequeue_increment(&self) -> Option<usize> {
        unsafe { self.increments.dequeue() }
    }

    pub fn increments_is_full(&self) -> bool {
        self.increments.is_full()
    }
    pub fn increments_len(&self) -> usize {
        self.increments.len()
    }
}

/// Control header for a NetChannel region. Self-anchored: sub-region
/// accessors compute their targets from `self` + fixed offsets + the
/// runtime `queue_len`, so they resolve correctly under the user satp
/// *and* under the kernel satp (through KDMAP).
///
/// Never construct this directly — the kernel allocates the region and
/// calls [`NetChannel::init`].
#[repr(C)]
pub struct NetChannel {
    queue_len: usize,
}

impl NetChannel {
    /// Normalize a user-requested region size: clamp into
    /// `[NC_MIN_REGION_SIZE, NC_MAX_REGION_SIZE]`, round up to a page, and
    /// align so `queue_len` (half the post-header span) ends up
    /// `usize`-aligned.
    pub fn normalize_region_size(requested: usize) -> Option<usize> {
        if requested == 0 {
            return None;
        }
        let clamped = requested.clamp(NC_MIN_REGION_SIZE, NC_MAX_REGION_SIZE);

        // Round up to page so each allocation fits cleanly in a whole
        // number of 4 KiB frames.
        let page_up = round_usize_up(clamped, 4096);
        if page_up > NC_MAX_REGION_SIZE {
            return None;
        }
        if page_up < NC_MIN_REGION_SIZE {
            return None;
        }

        // NC_TX_OFF is 16-aligned; dividing the remainder in half gives a
        // multiple of 8 as long as the region size is, which is already
        // guaranteed by page-rounding.
        Some(page_up)
    }

    /// Queue subregion length (header + ring buf) for a given total region
    /// size. Rounded down to `align_of::<NetChannelQueue>()` so
    /// `NC_TX_OFF + queue_len` (the rx subregion base) lands on an
    /// alignment valid for `*mut NetChannelQueue`. Wastes at most
    /// `align - 1` bytes of ring capacity; keeps the layout correct under
    /// any page-rounded `region_size`.
    pub const fn queue_len_for(region_size: usize) -> usize {
        let raw = (region_size - NC_TX_OFF) / 2;
        let align = core::mem::align_of::<NetChannelQueue>();
        raw & !(align - 1)
    }

    /// Per-ring usable payload capacity for a given total region size.
    pub const fn capacity_for(region_size: usize) -> usize {
        // `buf: u8` is the first byte of the ring, so the header
        // "overhead" is size_of - 1 and the rest is payload.
        Self::queue_len_for(region_size) - size_of::<NetChannelQueue>() + 1
    }

    /// Stamp the queue capacities on a freshly-allocated, zeroed region.
    ///
    /// # Safety
    /// - `base` must point at a zeroed, writable region of at least
    ///   `region_size` bytes, page-aligned.
    /// - `region_size` must be a value returned by
    ///   [`normalize_region_size`](Self::normalize_region_size).
    /// - No one else must observe the region between alloc and this call;
    ///   the kernel maps it into user VA only after init returns.
    pub unsafe fn init(base: *mut u8, region_size: usize) {
        let queue_len = Self::queue_len_for(region_size);
        let capacity = Self::capacity_for(region_size);
        unsafe {
            (*(base as *mut NetChannel)).queue_len = queue_len;

            let tx = base.add(NC_TX_OFF) as *mut NetChannelQueue;
            (*tx).capacity = capacity;

            let rx = base.add(NC_TX_OFF + queue_len) as *mut NetChannelQueue;
            (*rx).capacity = capacity;
        }
    }

    pub fn queue_len(&self) -> usize {
        self.queue_len
    }

    fn anchor(&self) -> *const u8 {
        self as *const Self as *const u8
    }

    pub fn desired(&self) -> &NetChannelDesired {
        unsafe { &*(self.anchor().add(NC_DESIRED_OFF) as *const NetChannelDesired) }
    }

    pub fn current(&self) -> &NetChannelCurrent {
        unsafe { &*(self.anchor().add(NC_CURRENT_OFF) as *const NetChannelCurrent) }
    }

    pub fn tx(&self) -> &NetChannelQueue {
        unsafe { &*(self.anchor().add(NC_TX_OFF) as *const NetChannelQueue) }
    }

    pub fn rx(&self) -> &NetChannelQueue {
        unsafe { &*(self.anchor().add(NC_TX_OFF + self.queue_len) as *const NetChannelQueue) }
    }

    /// Reset the kernel-owned halves of both rings so the next connection
    /// on this NetChannel starts clean. Kernel owns: tx.slices producer,
    /// tx.increments consumer, rx.slices producer, rx.increments consumer.
    /// `avail` fields are touched here too — both sides mutate them at
    /// steady state, but during reset the smoltcp socket is aborted and
    /// userspace is blocked on the state handshake, so kernel can zero
    /// them unilaterally.
    ///
    /// # Safety
    /// Must be called while the smoltcp socket is aborted and before
    /// the kernel releases `current.state = 0` — otherwise userspace may
    /// observe stale-then-zero indices out of order.
    #[cfg(feature = "kernel")]
    pub unsafe fn reset_kernel_side(&self) {
        let tx = self.tx();
        let rx = self.rx();
        unsafe {
            tx.slices.reset_producer();
            tx.increments.reset_consumer();
            rx.slices.reset_producer();
            rx.increments.reset_consumer();
        }
        tx.avail.store(0, Ordering::Release);
        rx.avail.store(0, Ordering::Release);
    }

    /// Reset the user-owned halves of both rings. Mirror of
    /// [`reset_kernel_side`]; user owns: tx.slices consumer,
    /// tx.increments producer, rx.slices consumer, rx.increments
    /// producer.
    ///
    /// # Safety
    /// Must be called after observing `current.state == 0` (which
    /// establishes that the kernel has already done its half) and
    /// before re-engaging on a fresh session.
    #[cfg(not(feature = "kernel"))]
    pub unsafe fn reset_user_side(&self) {
        let tx = self.tx();
        let rx = self.rx();
        unsafe {
            tx.slices.reset_consumer();
            tx.increments.reset_producer();
            rx.slices.reset_consumer();
            rx.increments.reset_producer();
        }
    }

    /// Pump one poll cycle against `socket`. The reconciler reads the
    /// user's per-session engagement flag, drives smoltcp through
    /// listen/connect/abort transitions on its own (binding params come
    /// from `ctx.bind`, which was latched at channel creation), and
    /// surfaces handshake completion / failure via `current.state`.
    ///
    /// `now_us` is microseconds since boot, matching the iface clock —
    /// used only for client-retain backoff scheduling.
    ///
    /// Pre-handshake transitions never write `current.state = 2`; only
    /// once smoltcp is past `Established` (i.e. `may_send || may_recv`)
    /// does the user see "session active." Gating on `is_open()` would
    /// have flipped the flag in `SynSent`/`SynReceived`/`Listen`, which
    /// is exactly the bug this redesign closes.
    #[cfg(feature = "kernel")]
    pub fn update_tcp(
        &self,
        mut iface: Interface,
        socket: &mut smoltcp::socket::tcp::Socket,
        ctx: &mut ChannelCtx,
        now_us: u64,
    ) -> (Interface, UpdateOutcome) {
        let cur = self.current();
        let des = self.desired();
        let mut outcome = UpdateOutcome::default();

        let engaged = des.engaged.load(Ordering::Acquire) != 0;
        let was_engaged = ctx.last_engaged;
        ctx.last_engaged = engaged;

        if engaged != was_engaged {
            info!(
                "netch[{:?}]: engaged {}->{} phase={:?}",
                ctx.bind, was_engaged as u32, engaged as u32, ctx.phase,
            );
        }

        // Sticky terminal: nothing the reconciler does can leave Failed
        // until the channel itself is torn down.
        if matches!(ctx.phase, Phase::Failed) {
            return (iface, outcome);
        }

        // ── Disengage edge: user just released the session ──────────────
        // Only act on the 1→0 edge, not on any poll where engaged==0;
        // otherwise a fresh session that the user hasn't claimed yet
        // would get torn down before they ever observed `state == 2`.
        //
        // Two-phase tear-down: enter `Closing` (graceful FIN, drain
        // smoltcp tx, observe peer FIN-ACK) instead of immediately
        // aborting + recycling. The old single-phase path discarded
        // any tx queued in the final `write_all` — see roadmap.
        if was_engaged && !engaged && matches!(ctx.phase, Phase::Active) {
            // Drain any tx the user wrote between the previous poll
            // and this disengage. Without this the bytes sit in the
            // user-side tx ring and `socket.close()` below queues a
            // FIN that races them — peer sees an empty FIN, our
            // pending data is discarded on recycle. Mirrors the
            // Phase::Active drain block but runs on this last poll
            // before we leave Active.
            if socket.may_send() {
                let tx = self.tx();
                // SAFETY: kernel is the sole consumer of tx.increments.
                while let Some(user_tx_count) = unsafe { tx.dequeue_increment() } {
                    let _ = socket.send(|_b| (user_tx_count, user_tx_count));
                    ctx.pending_tx_ack = false;
                }
            }

            info!(
                "netch[{:?}]: disengage edge → Closing (sock_state={:?})",
                ctx.bind,
                socket.state()
            );
            // `close()` queues a FIN once the existing tx queue
            // drains; smoltcp continues to send buffered data + the
            // FIN before transitioning the socket to `Closed`.
            socket.close();
            ctx.phase = Phase::Closing;
            // Move state to `CLOSING` (3) so the user-side
            // `disengage_and_release` spin can wait for it to leave
            // *both* `ACTIVE` *and* `CLOSING` before issuing
            // `close_handle`. Using a distinct sentinel (rather than
            // just != ACTIVE) lets the user race a still-draining
            // smoltcp socket against `close_handle`'s revoke and
            // drop the last `write_all`. See
            // [`channel_state::CLOSING`].
            cur.state.store(channel_state::CLOSING, Ordering::Release);
            outcome.session_state_changed = true;
            return (iface, outcome);
        }

        // ── Closing: drain smoltcp's tx + close handshake ───────────────
        // Stay in this phase until smoltcp reaches `Closed`. Then
        // reset_kernel_side and transition to whatever the binding
        // semantics dictate (re-listen, re-dial, terminal).
        if matches!(ctx.phase, Phase::Closing) {
            // smoltcp's `state()` for a fully-closed connection. Any
            // residual buffered tx has by now been ACK'd or RST'd.
            if !matches!(socket.state(), TcpState::Closed) {
                // Still flushing — try again on the next poll.
                return (iface, outcome);
            }

            info!(
                "netch[{:?}]: drain complete, recycling (sock_state={:?})",
                ctx.bind,
                socket.state()
            );
            // Even a graceful Closed leaves smoltcp with empty
            // buffers — abort() is a no-op functionally but resets
            // any internal state we'd otherwise carry into the next
            // session.
            socket.abort();
            unsafe {
                self.reset_kernel_side();
            }
            ctx.pending_rx_ack = false;
            ctx.pending_tx_ack = false;
            // peer_addr/port are advisory; clear so the next session's
            // peer info doesn't leak through.
            cur.peer_addr.store(0, Ordering::Relaxed);
            cur.peer_port.store(0, Ordering::Relaxed);

            match ctx.bind {
                BindSpec::ServerRetain { port } => {
                    if let Err(e) = socket.listen(port) {
                        error!("tcp: re-listen({port}) failed after recycle: {e:?}");
                        cur.fail_cause.store(EBIND_LISTEN as i32, Ordering::Release);
                        cur.state.store(channel_state::FAILED, Ordering::Release);
                        outcome.session_state_changed = true;
                        ctx.phase = Phase::Failed;
                        return (iface, outcome);
                    }
                    info!(
                        "netch[ServerRetain port={port}]: re-armed listen, phase=Listening state=1"
                    );
                    ctx.phase = Phase::Listening;
                    cur.state.store(channel_state::IN_FLIGHT, Ordering::Release);
                    outcome.session_state_changed = true;
                }
                BindSpec::ClientRetain { .. } => {
                    info!(
                        "netch[{:?}]: phase=FreshIdle (retain re-dial pending)",
                        ctx.bind
                    );
                    ctx.phase = Phase::FreshIdle;
                    ctx.backoff_ms = 0;
                    ctx.next_attempt_at_us = 0;
                    cur.state.store(channel_state::IDLE, Ordering::Release);
                    outcome.session_state_changed = true;
                }
                BindSpec::ServerOneShot { .. } | BindSpec::ClientOneShot { .. } => {
                    info!("netch[{:?}]: one-shot done, phase=Failed", ctx.bind);
                    cur.fail_cause.store(EBIND_DONE as i32, Ordering::Release);
                    cur.state.store(channel_state::FAILED, Ordering::Release);
                    outcome.session_state_changed = true;
                    ctx.phase = Phase::Failed;
                    return (iface, outcome);
                }
            }
            return (iface, outcome);
        }

        // ── First poll for this binding: arm the smoltcp side ──────────
        if matches!(ctx.phase, Phase::FreshIdle) {
            match ctx.bind {
                BindSpec::ServerOneShot { port } | BindSpec::ServerRetain { port } => {
                    if let Err(e) = socket.listen(port) {
                        error!("tcp: listen({port}) failed at bind: {e:?}");
                        cur.fail_cause.store(EBIND_LISTEN as i32, Ordering::Release);
                        cur.state.store(channel_state::FAILED, Ordering::Release);
                        outcome.session_state_changed = true;
                        ctx.phase = Phase::Failed;
                        return (iface, outcome);
                    }
                    info!(
                        "netch[{:?}]: armed listen({port}), phase=Listening state=1",
                        ctx.bind
                    );
                    ctx.phase = Phase::Listening;
                    cur.state.store(channel_state::IN_FLIGHT, Ordering::Release);
                    outcome.session_state_changed = true;
                }
                BindSpec::ClientRetain { addr, port } => {
                    if now_us >= ctx.next_attempt_at_us {
                        if try_connect(&mut iface, socket, addr, port).is_ok() {
                            info!(
                                "netch[ClientRetain addr={addr:#x} port={port}]: dialed, phase=Connecting state=1"
                            );
                            ctx.phase = Phase::Connecting;
                            cur.state.store(channel_state::IN_FLIGHT, Ordering::Release);
                            outcome.session_state_changed = true;
                        }
                        else {
                            schedule_retry(ctx, now_us);
                            info!(
                                "netch[ClientRetain addr={addr:#x} port={port}]: dial failed, retry in {}ms",
                                ctx.backoff_ms
                            );
                        }
                    }
                }
                BindSpec::ClientOneShot { addr, port } => {
                    if engaged {
                        if try_connect(&mut iface, socket, addr, port).is_ok() {
                            info!(
                                "netch[ClientOneShot addr={addr:#x} port={port}]: dialed, phase=Connecting state=1"
                            );
                            ctx.phase = Phase::Connecting;
                            cur.state.store(channel_state::IN_FLIGHT, Ordering::Release);
                            outcome.session_state_changed = true;
                        }
                        else {
                            error!(
                                "netch[ClientOneShot addr={addr:#x} port={port}]: dial failed at bind, phase=Failed"
                            );
                            cur.fail_cause
                                .store(EBIND_CONNECT as i32, Ordering::Release);
                            cur.state.store(channel_state::FAILED, Ordering::Release);
                            outcome.session_state_changed = true;
                            ctx.phase = Phase::Failed;
                            return (iface, outcome);
                        }
                    }
                }
            }
        }

        // ── Listening: peer has connected once smoltcp is past Established
        if matches!(ctx.phase, Phase::Listening) && (socket.may_send() || socket.may_recv()) {
            publish_peer(socket, cur);
            let pa = cur.peer_addr.load(Ordering::Relaxed);
            let pp = cur.peer_port.load(Ordering::Relaxed);
            info!(
                "netch[{:?}]: peer connected addr={:#x} port={} sock_state={:?}, phase=Active state=2",
                ctx.bind,
                pa,
                pp,
                socket.state()
            );
            ctx.phase = Phase::Active;
            cur.state.store(channel_state::ACTIVE, Ordering::Release);
            outcome.session_state_changed = true;
        }

        // ── Connecting: handshake done OR fell back to Closed (RST) ────
        if matches!(ctx.phase, Phase::Connecting) {
            if socket.may_send() || socket.may_recv() {
                publish_peer(socket, cur);
                let pa = cur.peer_addr.load(Ordering::Relaxed);
                let pp = cur.peer_port.load(Ordering::Relaxed);
                info!(
                    "netch[{:?}]: handshake complete peer={:#x}:{} sock_state={:?}, phase=Active state=2",
                    ctx.bind,
                    pa,
                    pp,
                    socket.state()
                );
                ctx.phase = Phase::Active;
                ctx.backoff_ms = 0;
                ctx.next_attempt_at_us = 0;
                cur.state.store(channel_state::ACTIVE, Ordering::Release);
                outcome.session_state_changed = true;
            }
            else if socket.state() == TcpState::Closed {
                match ctx.bind {
                    BindSpec::ClientOneShot { .. } => {
                        info!(
                            "netch[{:?}]: connect failed (RST/timeout), phase=Failed",
                            ctx.bind
                        );
                        cur.fail_cause
                            .store(EBIND_CONNECT as i32, Ordering::Release);
                        cur.state.store(channel_state::FAILED, Ordering::Release);
                        outcome.session_state_changed = true;
                        ctx.phase = Phase::Failed;
                        return (iface, outcome);
                    }
                    BindSpec::ClientRetain { .. } => {
                        ctx.phase = Phase::FreshIdle;
                        schedule_retry(ctx, now_us);
                        cur.state.store(channel_state::IDLE, Ordering::Release);
                        outcome.session_state_changed = true;
                        info!(
                            "netch[{:?}]: connect failed, retry in {}ms",
                            ctx.bind, ctx.backoff_ms
                        );
                    }
                    _ => {}
                }
            }
        }

        // ── Drain rings only when the session is active ────────────────
        if matches!(ctx.phase, Phase::Active) {
            let tx = self.tx();
            let rx = self.rx();

            // `may_recv()` is true while the *transport* allows new data
            // (Established / FIN-WAIT-1/2) OR when the rx_buffer still has
            // queued octets — smoltcp folds the "data already buffered"
            // case into may_recv via its `_ if self.can_recv()` arm. We
            // OR `recv_queue() > 0` explicitly anyway: it's documentation
            // for the case that actually matters here (CloseWait with
            // bytes still pending) and a guard if a future smoltcp tightens
            // may_recv to the strict-state-only definition.
            if socket.may_recv() || socket.recv_queue() > 0 {
                // SAFETY: kernel is the sole consumer of rx.increments.
                while let Some(user_rx_count) = unsafe { rx.dequeue_increment() } {
                    if let Err(e) = socket.recv(|_b| (user_rx_count, user_rx_count)) {
                        error!("tcp: failed recv: {e:?}");
                        cur.fail_cause.store(EBIND_IO as i32, Ordering::Release);
                        cur.state.store(channel_state::FAILED, Ordering::Release);
                        outcome.session_state_changed = true;
                        ctx.phase = Phase::Failed;
                        return (iface, outcome);
                    }
                    ctx.pending_rx_ack = false;
                }

                if !ctx.pending_rx_ack {
                    let next_rx = socket.get_next_rx();
                    if next_rx.1 > 0 {
                        // SAFETY: kernel is the sole producer of rx.slices.
                        let _r = unsafe { rx.enqueue_slice(next_rx) };
                        core::sync::atomic::fence(Ordering::SeqCst);
                        rx.avail.store(next_rx.1, Ordering::Release);
                        ctx.pending_rx_ack = true;
                        // New rx slice is visible to the user — anyone
                        // parked in `read_some` should run.
                        outcome.ring_progress = true;
                    }
                }
            }

            if socket.may_send() {
                // SAFETY: kernel is the sole consumer of tx.increments.
                while let Some(user_tx_count) = unsafe { tx.dequeue_increment() } {
                    if let Err(e) = socket.send(|_b| (user_tx_count, user_tx_count)) {
                        error!("tcp: failed send: {e:?}");
                        cur.fail_cause.store(EBIND_IO as i32, Ordering::Release);
                        cur.state.store(channel_state::FAILED, Ordering::Release);
                        outcome.session_state_changed = true;
                        ctx.phase = Phase::Failed;
                        return (iface, outcome);
                    }
                    ctx.pending_tx_ack = false;
                }

                if tx.slices_is_empty() && !ctx.pending_tx_ack {
                    let next_tx = socket.get_next_tx();
                    if next_tx.1 > 0 {
                        // SAFETY: kernel is the sole producer of tx.slices.
                        let _r = unsafe { tx.enqueue_slice(next_tx) };
                        core::sync::atomic::fence(Ordering::SeqCst);
                        tx.avail.store(next_tx.1, Ordering::Release);
                        ctx.pending_tx_ack = true;
                        // Fresh tx slice = user has writeable space —
                        // anyone parked in `write_all` should run.
                        outcome.ring_progress = true;
                    }
                }
            }
        }

        (iface, outcome)
    }

    #[cfg(not(feature = "kernel"))]
    pub fn send_tcp<F>(&self, f: F) -> Result<usize, isize>
    where
        F: FnOnce(&VolSliceMut) -> usize,
    {
        let cur = self.current();

        // Only `ACTIVE` permits ring traffic. `IDLE` (0), `IN_FLIGHT`
        // (1), `CLOSING` (3), and `FAILED` (-1) all reject — the user
        // wrapper maps the raw value to an errno via the
        // negative-isize convention. Note that `CLOSING` looks
        // numerically like a live state (3 > 2) but the kernel is
        // already driving graceful close on the smoltcp socket; the
        // user side must not write more.
        let channel_state = cur.state.load(Ordering::Acquire);
        if channel_state != channel_state::ACTIVE {
            return Err(channel_state as isize);
        }

        let tx = self.tx();

        if tx.slices_is_empty() {
            return Err(-4);
        }

        if tx.increments_is_full() {
            return Err(-5);
        }

        // SAFETY: user is the sole consumer of tx.slices and sole
        // producer of tx.increments.
        let (offset, len) = unsafe { tx.dequeue_slice() }.ok_or(-4isize)?;
        let buf = unsafe { VolSliceMut::from_raw_parts(tx.buf_ptr().add(offset), len) };

        let written = f(&buf);
        if written == 0 {
            return Ok(0);
        }

        unsafe {
            let _ = tx.enqueue_increment(written);
        }

        tx.avail.fetch_sub(written, Ordering::AcqRel);

        Ok(written)
    }

    #[cfg(not(feature = "kernel"))]
    pub fn recv_tcp<F>(&self, f: F) -> Result<usize, isize>
    where
        F: FnOnce(&VolSlice) -> usize,
    {
        let cur = self.current();

        let channel_state = cur.state.load(Ordering::Acquire);
        if channel_state < channel_state::ACTIVE {
            return Err(channel_state as isize);
        }

        let rx = self.rx();

        if rx.slices_is_empty() {
            return Err(-4);
        }

        if rx.increments_is_full() {
            return Err(-5);
        }

        // SAFETY: user is the sole consumer of rx.slices and sole
        // producer of rx.increments.
        let (offset, len) = unsafe { rx.dequeue_slice() }.ok_or(-4isize)?;
        let buf = unsafe { VolSlice::from_raw_parts(rx.buf_ptr().add(offset), len) };

        let written = f(&buf);
        if written == 0 {
            return Ok(0);
        }

        unsafe {
            let _ = rx.enqueue_increment(written);
        }

        rx.avail.fetch_sub(written, Ordering::AcqRel);

        Ok(written)
    }

    /// Mark the channel as engaged with the upcoming session. Writes
    /// `desired.engaged = 1`; the kernel's reconciler treats engagement
    /// as the user's claim on whatever session lands. For client one-
    /// shot bindings, the engage transition is also the gate that lets
    /// the kernel issue the dial.
    ///
    /// Idempotent — calling twice is harmless. The kernel only acts on
    /// the `1 → 0` transition (via [`disengage`](Self::disengage)).
    #[cfg(not(feature = "kernel"))]
    pub fn engage(&self) {
        self.desired().engaged.store(1, Ordering::Release);
    }

    /// Release the current session. Writes `desired.engaged = 0`; the
    /// reconciler observes the `1 → 0` edge and tears down the smoltcp
    /// socket (for retain bindings, immediately re-arms; for one-shot
    /// bindings, transitions to the terminal `Failed` state).
    ///
    /// After calling this, the user must wait for `current.state` to
    /// drop to 0 before resetting the user-owned ring halves and
    /// engaging again. The blocking sequence is owned by the wrapper in
    /// orbit-rt; this method is just the signal.
    #[cfg(not(feature = "kernel"))]
    pub fn disengage(&self) {
        self.desired().engaged.store(0, Ordering::Release);
    }

    #[cfg(feature = "kernel")]
    pub fn rings(&self) -> (&'static mut [u8], &'static mut [u8]) {
        let tx = self.tx();
        let rx = self.rx();
        unsafe {
            (
                core::slice::from_raw_parts_mut(tx.buf_ptr(), tx.capacity),
                core::slice::from_raw_parts_mut(rx.buf_ptr(), rx.capacity),
            )
        }
    }

    #[cfg(not(feature = "kernel"))]
    pub fn readable(&self) -> usize {
        self.rx().avail.load(Ordering::Acquire)
    }

    #[cfg(not(feature = "kernel"))]
    pub fn writeable(&self) -> usize {
        self.tx().avail.load(Ordering::Acquire)
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod vol_slice_tests {
    use super::{VolSlice, VolSliceMut};
    use std::boxed::Box;
    use std::vec::Vec;

    // miri extern — intentional test leaks register themselves as roots
    // so the leak checker accepts them without `-Zmiri-ignore-leaks`.
    #[cfg(miri)]
    unsafe extern "Rust" {
        fn miri_static_root(ptr: *const u8);
    }
    #[cfg(miri)]
    unsafe fn register_root(ptr: *const u8) {
        unsafe {
            miri_static_root(ptr);
        }
    }
    #[cfg(not(miri))]
    unsafe fn register_root(_ptr: *const u8) {}

    // Helper: leak a heap buffer of `len` bytes initialized with a
    // ramp (0, 1, 2, ...). Using the heap keeps pointer provenance
    // clean for miri — stack arrays have stricter reborrow rules in
    // some compiler versions.
    fn leaked_ramp(len: usize) -> *mut u8 {
        let v: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
        let boxed = v.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut u8;
        // `Vec::new().into_boxed_slice()` yields a dangling sentinel, not
        // a real allocation; only register real allocations with miri.
        if len > 0 {
            unsafe {
                register_root(ptr as *const u8);
            }
        }
        ptr
    }

    // ---- VolSliceMut basics ----

    #[test]
    fn volslice_mut_len_and_is_empty() {
        let ptr = leaked_ramp(10);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 10) };
        assert_eq!(m.len(), 10);
        assert!(!m.is_empty());

        let ptr2 = leaked_ramp(0);
        let empty = unsafe { VolSliceMut::from_raw_parts(ptr2, 0) };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn volslice_mut_copy_from_slice_truncates_to_self_len() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        // Source longer than self → truncates to 4.
        let written = m.copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        assert_eq!(written, 4);
        for i in 0..4 {
            assert_eq!(m.get(i), [0xAA, 0xBB, 0xCC, 0xDD][i]);
        }
    }

    #[test]
    fn volslice_mut_copy_from_slice_truncates_to_src_len() {
        let ptr = leaked_ramp(10);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 10) };
        // Source shorter than self → truncates to 3. Tail bytes untouched.
        let written = m.copy_from_slice(&[1, 2, 3]);
        assert_eq!(written, 3);
        assert_eq!(m.get(0), 1);
        assert_eq!(m.get(1), 2);
        assert_eq!(m.get(2), 3);
        // Index 3 retains the ramp initialization byte.
        assert_eq!(m.get(3), 3);
    }

    #[test]
    fn volslice_mut_set_get_roundtrip() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        m.set(0, 0x11);
        m.set(3, 0x44);
        assert_eq!(m.get(0), 0x11);
        assert_eq!(m.get(3), 0x44);
    }

    #[test]
    fn volslice_mut_checked_variants_return_none_on_oob() {
        let ptr = leaked_ramp(2);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 2) };
        assert!(m.get_checked(2).is_none());
        assert!(m.set_checked(2, 0).is_none());
        assert_eq!(m.set_checked(0, 7), Some(()));
        assert_eq!(m.get_checked(0), Some(7));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn volslice_mut_get_oob_panics() {
        let ptr = leaked_ramp(2);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 2) };
        let _ = m.get(5);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn volslice_mut_set_oob_panics() {
        let ptr = leaked_ramp(2);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 2) };
        m.set(2, 0);
    }

    // ---- VolSliceMut::sub + as_readonly ----

    #[test]
    fn volslice_mut_sub_in_bounds() {
        let ptr = leaked_ramp(8);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 8) };
        let s = m.sub(2, 6);
        assert_eq!(s.len(), 4);
        // Ramp: bytes [0,1,2,3,4,5,6,7]; sub [2..6] = [2,3,4,5].
        for i in 0..4 {
            assert_eq!(s.get(i), (i + 2) as u8);
        }
    }

    #[test]
    #[should_panic(expected = "end")]
    fn volslice_mut_sub_end_oob_panics() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        let _ = m.sub(0, 5);
    }

    #[test]
    #[should_panic(expected = "start")]
    fn volslice_mut_sub_start_after_end_panics() {
        let ptr = leaked_ramp(4);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 4) };
        let _ = m.sub(3, 2);
    }

    #[test]
    fn volslice_mut_as_readonly_preserves_len() {
        let ptr = leaked_ramp(5);
        let m = unsafe { VolSliceMut::from_raw_parts(ptr, 5) };
        let ro = m.as_readonly();
        assert_eq!(ro.len(), 5);
        // Ramp byte at index 2 = 2.
        assert_eq!(ro.get(2), 2);
    }

    // ---- VolSlice ----

    #[test]
    fn volslice_copy_to_slice_truncates_to_self_len() {
        let ptr = leaked_ramp(4);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 4) };
        let mut dst = [0u8; 10];
        let n = s.copy_to_slice(&mut dst);
        assert_eq!(n, 4);
        assert_eq!(&dst[..4], &[0, 1, 2, 3]);
        assert_eq!(&dst[4..], &[0u8; 6]); // tail untouched
    }

    #[test]
    fn volslice_copy_to_slice_truncates_to_dst_len() {
        let ptr = leaked_ramp(10);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 10) };
        let mut dst = [0u8; 3];
        let n = s.copy_to_slice(&mut dst);
        assert_eq!(n, 3);
        assert_eq!(dst, [0, 1, 2]);
    }

    #[test]
    fn volslice_checked_get_returns_none_on_oob() {
        let ptr = leaked_ramp(3);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 3) };
        assert_eq!(s.get_checked(0), Some(0));
        assert_eq!(s.get_checked(2), Some(2));
        assert!(s.get_checked(3).is_none());
        assert!(s.get_checked(999).is_none());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn volslice_get_oob_panics() {
        let ptr = leaked_ramp(2);
        let s = unsafe { VolSlice::from_raw_parts(ptr, 2) };
        let _ = s.get(2);
    }
}

// `rings()` is gated to `feature = "kernel"`. The layout tests poke at
// ring byte addresses, so they run only when that feature is enabled —
// i.e. under `cargo test --features kernel`.
#[cfg(all(test, feature = "kernel"))]
mod netchannel_layout_tests {
    //! Tests for `NetChannel::init` + the header/ring layout.
    //!
    //! # Known miri caveat
    //!
    //! `NetChannel`'s accessor methods (`tx`, `rx`, `rings`, `anchor`,
    //! `desired_state`, `current_state`) cast `self: &NetChannel` to
    //! `*const u8` and then `.add(offset)` to reach well past the 8-byte
    //! struct footprint into the surrounding region. That pointer
    //! arithmetic is legal, but the dereference is UB under Stacked
    //! Borrows: the cast-derived ptr has SharedReadOnly provenance for
    //! only `sizeof(NetChannel)` bytes, not the full region.
    //!
    //! In practice the code works — nothing else reborrows against the
    //! narrow region, and the hardware doesn't track provenance — but
    //! strictly speaking it's UB. Fixing it properly would turn
    //! `NetChannel` into a DST `{ queue_len: usize, _rest: [u8] }` or
    //! move all accessors onto raw pointers taking the region base. Out
    //! of scope for the initial test sweep.
    //!
    //! Accessor-path tests are gated off under miri; the raw-access
    //! tests below directly read fields via offsets from the
    //! allocation-rooted base pointer and DO run under miri.

    use super::*;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    struct OwnedRegion {
        base: *mut u8,
        layout: Layout,
    }

    impl OwnedRegion {
        fn new(size: usize) -> Self {
            let layout = Layout::from_size_align(size, 4096).unwrap();
            let base = unsafe { alloc_zeroed(layout) };
            assert!(!base.is_null(), "alloc_zeroed returned null");
            // SAFETY: NetChannel::init reads/writes the whole region.
            unsafe { NetChannel::init(base, size) };
            Self { base, layout }
        }
        fn nc(&self) -> &NetChannel {
            unsafe { &*(self.base as *const NetChannel) }
        }
    }

    impl Drop for OwnedRegion {
        fn drop(&mut self) {
            unsafe { dealloc(self.base, self.layout) };
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // uses NetChannel accessors — see module docs
    fn init_stamps_queue_len_and_capacities() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        assert_eq!(
            nc.queue_len(),
            NetChannel::queue_len_for(NC_MIN_REGION_SIZE)
        );
        assert_eq!(
            nc.tx().capacity(),
            NetChannel::capacity_for(NC_MIN_REGION_SIZE)
        );
        assert_eq!(
            nc.rx().capacity(),
            NetChannel::capacity_for(NC_MIN_REGION_SIZE)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_return_buffers_of_expected_capacity() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let cap = NetChannel::capacity_for(NC_MIN_REGION_SIZE);
        let (tx, rx) = nc.rings();
        assert_eq!(tx.len(), cap);
        assert_eq!(rx.len(), cap);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_tx_and_rx_do_not_overlap() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let (tx, rx) = nc.rings();
        let tx_start = tx.as_ptr() as usize;
        let tx_end = tx_start + tx.len();
        let rx_start = rx.as_ptr() as usize;
        let rx_end = rx_start + rx.len();
        assert!(
            tx_end <= rx_start || rx_end <= tx_start,
            "tx [{tx_start:#x}..{tx_end:#x}] and rx [{rx_start:#x}..{rx_end:#x}] overlap"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_are_within_allocated_region() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let base = region.base as usize;
        let end = base + NC_MIN_REGION_SIZE;
        let (tx, rx) = nc.rings();
        let tx_start = tx.as_ptr() as usize;
        let rx_end = rx.as_ptr() as usize + rx.len();
        assert!(
            base <= tx_start && rx_end <= end,
            "rings escape region: base={base:#x} end={end:#x} tx_start={tx_start:#x} rx_end={rx_end:#x}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_are_writable_and_readable() {
        // Exercises raw-pointer derefs through the slice handle end to
        // end — miri validates each access against the region's
        // provenance.
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let (tx, rx) = nc.rings();

        tx[0] = 0xAA;
        tx[tx.len() - 1] = 0xBB;
        rx[0] = 0xCC;
        rx[rx.len() - 1] = 0xDD;

        assert_eq!(tx[0], 0xAA);
        assert_eq!(tx[tx.len() - 1], 0xBB);
        assert_eq!(rx[0], 0xCC);
        assert_eq!(rx[rx.len() - 1], 0xDD);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn tx_rx_offsets_match_headers() {
        // tx lives at NC_TX_OFF; rx lives at NC_TX_OFF + queue_len.
        // Verify the fixed-offset accessors against the raw base.
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let base = region.base as usize;
        let tx = nc.tx() as *const _ as usize;
        let rx = nc.rx() as *const _ as usize;
        assert_eq!(tx - base, NC_TX_OFF);
        assert_eq!(rx - base, NC_TX_OFF + nc.queue_len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn desired_and_current_states_sit_at_fixed_offsets() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let nc = region.nc();
        let base = region.base as usize;
        let desired = nc.desired() as *const _ as usize;
        let current = nc.current() as *const _ as usize;
        assert_eq!(desired - base, NC_DESIRED_OFF);
        assert_eq!(current - base, NC_CURRENT_OFF);
    }

    #[test]
    fn netchannel_state_structs_have_expected_size_and_align() {
        // Belt-and-braces: even though const_assert checks these at
        // build time, surface them in the test suite so a layout
        // regression shows up as a test failure rather than a build
        // failure that's harder to bisect.
        assert_eq!(core::mem::size_of::<NetChannelDesired>(), 128);
        assert_eq!(core::mem::align_of::<NetChannelDesired>(), 128);
        assert_eq!(core::mem::size_of::<NetChannelCurrent>(), 128);
        assert_eq!(core::mem::align_of::<NetChannelCurrent>(), 128);
    }

    #[test]
    fn bind_spec_round_trips_through_pack() {
        let cases = [
            BindSpec::ClientOneShot {
                addr: 0xC0A8_4C02,
                port: 65535,
            },
            BindSpec::ClientRetain {
                addr: 0x0A00_0001,
                port: 80,
            },
            BindSpec::ServerOneShot { port: 7777 },
            BindSpec::ServerRetain { port: 22 },
        ];
        for &c in &cases {
            assert_eq!(
                BindSpec::unpack(c.pack()),
                Some(c),
                "round-trip failed for {c:?}"
            );
        }
    }

    #[test]
    fn bind_spec_unpack_rejects_invalid() {
        // Mode 0 or unknown.
        assert_eq!(BindSpec::unpack(0), None);
        assert_eq!(BindSpec::unpack(99 | (80 << 8)), None);
        // Port 0 isn't a valid TCP endpoint here.
        assert_eq!(BindSpec::unpack(1 | (0 << 8)), None);
        // Server with a non-zero addr — that'd be a stale or malformed sender.
        assert_eq!(BindSpec::unpack(3 | (80 << 8) | (0xC0A8_0001 << 24)), None,);
        // Reserved high bits set.
        assert_eq!(
            BindSpec::unpack(BindSpec::ServerRetain { port: 22 }.pack() | (1usize << 56)),
            None
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rings_len_is_zero_or_more_always() {
        // Grid over several valid region sizes.
        for &sz in &[NC_MIN_REGION_SIZE, 8192, 16384, NC_MAX_REGION_SIZE] {
            let region = OwnedRegion::new(sz);
            let (tx, rx) = region.nc().rings();
            let cap = NetChannel::capacity_for(sz);
            assert_eq!(tx.len(), cap);
            assert_eq!(rx.len(), cap);
        }
    }

    // ---- Raw-access tests ----
    // Read fields via offsets from the allocation-rooted base pointer.
    // Stay miri-clean because provenance comes from `region.base` (the
    // alloc_zeroed return), not from a narrow `&NetChannel` reborrow.

    #[test]
    fn raw_queue_len_written_at_offset_zero() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let written = unsafe { *(region.base as *const usize) };
        assert_eq!(written, NetChannel::queue_len_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    fn raw_tx_capacity_written_at_nc_tx_off_plus_capacity_field() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let cap_field = core::mem::offset_of!(NetChannelQueue, capacity);
        let tx_cap_addr = unsafe { region.base.add(NC_TX_OFF + cap_field) as *const usize };
        let tx_cap = unsafe { *tx_cap_addr };
        assert_eq!(tx_cap, NetChannel::capacity_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    fn raw_rx_capacity_written_at_nc_tx_off_plus_queue_len() {
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        let queue_len = NetChannel::queue_len_for(NC_MIN_REGION_SIZE);
        let cap_field = core::mem::offset_of!(NetChannelQueue, capacity);
        let rx_cap_addr =
            unsafe { region.base.add(NC_TX_OFF + queue_len + cap_field) as *const usize };
        let rx_cap = unsafe { *rx_cap_addr };
        assert_eq!(rx_cap, NetChannel::capacity_for(NC_MIN_REGION_SIZE));
    }

    #[test]
    fn raw_region_is_fully_writable() {
        // Walk every byte. Provenance from one alloc_zeroed covers the
        // entire region — miri validates each access.
        let region = OwnedRegion::new(NC_MIN_REGION_SIZE);
        for off in 0..NC_MIN_REGION_SIZE {
            unsafe {
                region.base.add(off).write(0xA5);
            }
        }
        for off in 0..NC_MIN_REGION_SIZE {
            assert_eq!(unsafe { *region.base.add(off) }, 0xA5);
        }
    }
}

#[cfg(test)]
mod region_sizing_tests {
    use super::*;

    // ---- normalize_region_size ----

    #[test]
    fn zero_is_rejected() {
        assert_eq!(NetChannel::normalize_region_size(0), None);
    }

    #[test]
    fn min_requested_returns_min() {
        assert_eq!(
            NetChannel::normalize_region_size(NC_MIN_REGION_SIZE),
            Some(NC_MIN_REGION_SIZE)
        );
    }

    #[test]
    fn max_requested_returns_max() {
        assert_eq!(
            NetChannel::normalize_region_size(NC_MAX_REGION_SIZE),
            Some(NC_MAX_REGION_SIZE)
        );
    }

    #[test]
    fn below_min_clamps_up_to_min() {
        assert_eq!(
            NetChannel::normalize_region_size(1),
            Some(NC_MIN_REGION_SIZE)
        );
        assert_eq!(
            NetChannel::normalize_region_size(NC_MIN_REGION_SIZE - 1),
            Some(NC_MIN_REGION_SIZE)
        );
    }

    #[test]
    fn above_max_clamps_down_to_max() {
        assert_eq!(
            NetChannel::normalize_region_size(NC_MAX_REGION_SIZE + 1),
            Some(NC_MAX_REGION_SIZE)
        );
        assert_eq!(
            NetChannel::normalize_region_size(usize::MAX),
            Some(NC_MAX_REGION_SIZE)
        );
    }

    #[test]
    fn mid_range_rounds_up_to_page() {
        // 8192 + 1 should round to 12288 (still in [min, max])
        assert_eq!(NetChannel::normalize_region_size(8193), Some(12288));
        // Already page-aligned passes through.
        assert_eq!(NetChannel::normalize_region_size(8192), Some(8192));
    }

    #[test]
    fn result_is_always_page_aligned() {
        for &req in &[1usize, 100, 4095, 4096, 5000, 8192, 100_000] {
            let r = NetChannel::normalize_region_size(req).unwrap();
            assert_eq!(
                r % 4096,
                0,
                "normalized size for {req} should be page-aligned, got {r}"
            );
            assert!(r >= NC_MIN_REGION_SIZE && r <= NC_MAX_REGION_SIZE);
        }
    }

    // ---- queue_len_for ----

    #[test]
    fn queue_len_is_nc_queue_aligned() {
        let align = core::mem::align_of::<NetChannelQueue>();
        for &r in &[NC_MIN_REGION_SIZE, 8192, 16384, NC_MAX_REGION_SIZE] {
            assert_eq!(
                NetChannel::queue_len_for(r) % align,
                0,
                "queue_len_for({r}) must be aligned to align_of::<NetChannelQueue>() = {align}"
            );
        }
    }

    #[test]
    fn queue_len_leaves_room_for_both_halves() {
        for &r in &[NC_MIN_REGION_SIZE, 8192, 16384, NC_MAX_REGION_SIZE] {
            let q = NetChannel::queue_len_for(r);
            // tx + rx (each q) + header (NC_TX_OFF) must fit within r.
            assert!(NC_TX_OFF + 2 * q <= r, "tx+rx overrun region at size {r}");
        }
    }

    // ---- capacity_for ----

    #[test]
    fn capacity_equals_queue_len_minus_header() {
        for &r in &[NC_MIN_REGION_SIZE, 8192, NC_MAX_REGION_SIZE] {
            let q = NetChannel::queue_len_for(r);
            let c = NetChannel::capacity_for(r);
            // capacity = queue_len - size_of::<NetChannelQueue>() + 1
            assert_eq!(
                c,
                q - core::mem::size_of::<NetChannelQueue>() + 1,
                "capacity_for({r}) should equal queue_len - header + 1"
            );
        }
    }

    #[test]
    fn capacity_grows_with_region_size() {
        let c_min = NetChannel::capacity_for(NC_MIN_REGION_SIZE);
        let c_max = NetChannel::capacity_for(NC_MAX_REGION_SIZE);
        assert!(c_min < c_max, "larger region → larger per-ring capacity");
    }

    #[test]
    fn min_region_has_positive_capacity() {
        // If this fires, NC_MIN_REGION_SIZE is too small for the header
        // + one usable byte — the channel would be useless at the floor.
        assert!(NetChannel::capacity_for(NC_MIN_REGION_SIZE) > 0);
    }
}
