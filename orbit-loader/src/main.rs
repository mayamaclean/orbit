//! orbit-loader — u-mode listener that accepts an ELF over TCP and asks
//! the kernel to spawn it via `create_process`. Replaces `include_bytes!`
//! rebuild cycles for test binaries; kmain embeds this loader once and
//! iteration happens over the wire.
//!
//! Wire protocol (per incoming connection):
//! ```text
//! [u32 LE len] [u32 LE !len] [cbor body: len bytes]
//! ```
//! where the CBOR body decodes as [`Payload`]: a map of `elf` (byte
//! string) and `name` (text string). The inverse-length check rejects
//! obvious corruption before we allocate.
//!
//! One [`NetCh`] serves the loader's lifetime; after each client we
//! call `reset()` to recycle the channel rather than tearing down and
//! reallocating between clients.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_abi::user::sleep_ms;
use orbit_rt as _;

use alloc::string::String;
use alloc::vec::Vec;
use core::panic::PanicInfo;

use minicbor::{Decode, Encode};
use net_channel::{BindSpec, NetChannel, NC_MAX_REGION_SIZE};
use orbit_abi::errno::Errno;
use orbit_abi::net::SockType;
use orbit_abi::{logln, user::{create_process, exit, ConsoleWriter}};
use orbit_rt::netch::{NetCh, Session};

const LISTEN_PORT: u16 = 7777;
/// Size the rings to the maximum the kernel will allocate, so a large
/// CBOR payload doesn't ping-pong through the rings more than necessary.
const RING_CAPACITY: usize = NetChannel::capacity_for(NC_MAX_REGION_SIZE);
// Matches the kernel-side cap in handle_create_process_req. Rejecting
// before we allocate the buffer keeps a bogus header from forcing us
// to grow the heap to 4 MiB and then get -1'd by the syscall.
const MAX_ELF_BYTES: usize = 4 * 1024 * 1024;

// In-tree shell — spawned at loader startup so the user has a usable
// pane on Ctrl+Tab without first having to push a payload over TCP.
// Mirrors how kmain embeds this loader: build console first, then
// orbit-loader picks up the latest release ELF.
const CONSOLE_ELF: &[u8] = include_bytes!(
    "../../console/target/riscv64gc-unknown-none-elf/release/console"
);

// `map` (not the derive default `array`) so new optional fields can be
// added later without breaking existing senders — map entries are keyed
// by their `#[n(N)]` index, so missing keys are tolerated rather than
// shifting every subsequent field.
#[derive(Decode, Encode, Debug)]
#[cbor(map)]
struct Payload<'a> {
    #[n(0)] #[cbor(with = "minicbor::bytes")] elf: &'a [u8],
    #[b(1)] name: &'a str,
    /// Optional CPU affinity cap. Sentinel `0` (also the absent-key
    /// default since minicbor leaves missing primitives at zero) tells
    /// the kernel to use the all-harts default.
    #[n(2)] #[cbor(default)] allowed_affinity: u64,
    /// Optional initial affinity. Sentinel `0` → defaults to whatever
    /// `allowed_affinity` resolves to. Must be a subset of
    /// `allowed_affinity` once both are resolved (kernel enforces).
    #[n(3)] #[cbor(default)] affinity: u64,
}

#[derive(Debug)]
#[allow(dead_code)] // payload fields are read via Debug in logln!
enum LoaderErr {
    Framing,
    TooLarge(u32),
    Cbor,
    NetCh(Errno),
    Syscall(Errno),
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    // Spawn the in-tree console first so a user has a pane to flip to
    // on Ctrl+Tab while the loader is still negotiating its NetChannel.
    // Failure here isn't fatal — the loader is still useful for
    // sending ELFs over TCP without an interactive shell.
    // Console gets the all-harts default — passing 0 for both affinity
    // fields tells the kernel "no preference."
    match create_process(CONSOLE_ELF.as_ptr(), CONSOLE_ELF.len(), 0, 0) {
        Ok(pid) => logln!("orbit-loader: spawned console pid={pid}"),
        Err(e)  => logln!("orbit-loader: console spawn failed: {e:?}"),
    }

    if let Err(e) = sleep_ms(2000) {
        return e.to_ret() as i32;
    }

    // ServerRetain: the kernel keeps `socket.listen(LISTEN_PORT)` armed
    // continuously across sessions. We just engage on `next_session`,
    // drain the payload, and disengage on `Session` drop — the kernel
    // re-arms the listen as part of its recycle, so back-to-back peers
    // never race a closed-window.
    let nc = match NetCh::open(
        RING_CAPACITY,
        SockType::Tcp,
        BindSpec::ServerRetain { port: LISTEN_PORT },
    ) {
        Ok(n) => n,
        Err(e) => {
            logln!("orbit-loader: NetCh::open failed: {e:?}");
            exit(-2);
        }
    };

    logln!("orbit-loader: listening on :{LISTEN_PORT}");

    loop {
        match accept_and_load(&nc) {
            Ok((pid, name)) => logln!("orbit-loader: spawned pid={pid} name={name:?}"),
            Err(e)          => logln!("orbit-loader: iteration failed: {e:?}"),
        }
        // Session is dropped at end of accept_and_load — disengagement
        // and kernel-side relisten happen there.
    }
}

/// One listen-accept-recv-spawn cycle on an already-opened NetCh.
/// Holds a [`Session`] for the duration; dropping it on return signals
/// the kernel to recycle into a fresh listen for the next iteration.
fn accept_and_load(nc: &NetCh) -> Result<(u16, String), LoaderErr> {
    let session = nc.next_session().map_err(LoaderErr::NetCh)?;
    let (payload_bytes, name) = recv_payload(&session)?;
    let pid = spawn(&payload_bytes)?;
    Ok((pid, name))
}

/// Read the full framed message into a heap Vec. Returns (buf, name)
/// where `buf` is the trimmed CBOR body and `name` is decoded from it.
/// The returned bytes are handed verbatim to the kernel.
fn recv_payload(s: &Session<'_>) -> Result<(Vec<u8>, String), LoaderErr> {
    let mut scratch: Vec<u8> = Vec::new();

    // Fill enough for the 8-byte header.
    while scratch.len() < 8 {
        drain_some(s, &mut scratch)?;
    }

    let len = u32::from_le_bytes([scratch[0], scratch[1], scratch[2], scratch[3]]);
    let inv = u32::from_le_bytes([scratch[4], scratch[5], scratch[6], scratch[7]]);
    if len ^ inv != u32::MAX { return Err(LoaderErr::Framing); }
    if (len as usize) > MAX_ELF_BYTES { return Err(LoaderErr::TooLarge(len)); }

    let total = 8 + len as usize;
    scratch.reserve(total.saturating_sub(scratch.len()));
    while scratch.len() < total {
        drain_some(s, &mut scratch)?;
    }

    let body = &scratch[8..total];

    // Decode name eagerly so we can log it; elf bytes stay in `scratch`
    // and the kernel reads them via create_process.
    let payload: Payload = minicbor::decode(body).map_err(|_| LoaderErr::Cbor)?;
    let name = String::from(payload.name);

    // Trim to just the CBOR body so the returned buffer's ptr + len
    // line up with what the syscall needs. (The ELF inside is a view
    // over these bytes; we pass the whole body.)
    let body_only = scratch[8..total].to_vec();
    Ok((body_only, name))
}

/// Pull at least one byte from the session into `out` (blocking via
/// `Session::read_some`, which sleeps the hart until the ring has data
/// or the channel breaks).
fn drain_some(s: &Session<'_>, out: &mut Vec<u8>) -> Result<(), LoaderErr> {
    let mut tmp = [0u8; 128 * 1024];
    let n = s.read_some_with_poll_timeout(&mut tmp, 100).map_err(LoaderErr::NetCh)?;
    out.extend_from_slice(&tmp[..n]);
    Ok(())
}

/// Parse the already-received CBOR body a second time to locate the ELF
/// byte-string inside `body_only`, then syscall. The double-decode cost
/// is negligible vs the wire transfer and keeps the interface between
/// recv and spawn a plain `&[u8]`.
fn spawn(body_only: &[u8]) -> Result<u16, LoaderErr> {
    let payload: Payload = minicbor::decode(body_only).map_err(|_| LoaderErr::Cbor)?;
    let elf = payload.elf;
    let head = [
        elf.first().copied().unwrap_or(0),
        elf.get(1).copied().unwrap_or(0),
        elf.get(2).copied().unwrap_or(0),
        elf.get(3).copied().unwrap_or(0),
    ];
    logln!("orbit-loader: spawn ptr={:p} len={} head={:02x?} \
           allowed_affinity={:#x} affinity={:#x}",
           elf.as_ptr(), elf.len(), head,
           payload.allowed_affinity, payload.affinity);
    create_process(
        elf.as_ptr(),
        elf.len(),
        payload.allowed_affinity,
        payload.affinity,
    ).map_err(LoaderErr::Syscall)
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "orbit-loader: panic: {p}");
    w.flush();
    exit(isize::MIN);
}
