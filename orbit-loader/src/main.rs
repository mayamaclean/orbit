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
//! One NetChannel allocation serves the loader's lifetime; after each
//! client we drive a shared-memory reset handshake (`request_reset` →
//! poll `current_state == 0` → `complete_reset`) and re-listen, rather
//! than tearing down and reallocating between clients.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use alloc::string::String;
use alloc::vec::Vec;
use core::{panic::PanicInfo, sync::atomic::Ordering};

use minicbor::{Decode, Encode};
use net_channel::NetChannel;
use orbit_abi::{logln, user::{create_netch, create_process, exit, sleep_ms, SerialWriter}};

const LISTEN_PORT: u16 = 7777;
const NC_VADDR_HINT: usize = 0x2_4000_0000;
const NC_REGION_SIZE: usize = net_channel::NC_MAX_REGION_SIZE;
// Matches the kernel-side cap in handle_create_process_req. Rejecting
// before we allocate the buffer keeps a bogus header from forcing us
// to grow the heap to 4 MiB and then get -1'd by the syscall.
const MAX_ELF_BYTES: usize = 4 * 1024 * 1024;
const POLL_SLEEP_MS: usize = 10;

// `map` (not the derive default `array`) so new optional fields can be
// added later without breaking existing senders — map entries are keyed
// by their `#[n(N)]` index, so missing keys are tolerated rather than
// shifting every subsequent field.
#[derive(Decode, Encode, Debug)]
#[cbor(map)]
struct Payload<'a> {
    #[n(0)] #[cbor(with = "minicbor::bytes")] elf: &'a [u8],
    #[b(1)] name: &'a str,
}

#[derive(Debug)]
#[allow(dead_code)] // payload fields are read via Debug in logln!
enum LoaderErr {
    Framing,
    TooLarge(u32),
    Cbor,
    ConnClosed(i32),
    Listen,
    Syscall(isize),
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    logln!("orbit-loader: listening on :{LISTEN_PORT}");

    let (nc_vaddr, _nc_fd) = match create_netch(NC_VADDR_HINT, NC_REGION_SIZE, 0) {
        Ok(v) => v,
        Err(e) => {
            logln!("orbit-loader: create_netch failed: {e}");
            exit(-2);
        }
    };
    // SAFETY: kernel just mapped NC_REGION_SIZE user-RW at nc_vaddr and
    // this task owns the region for its lifetime.
    let nc = unsafe { &*(nc_vaddr as *const NetChannel) };

    loop {
        match accept_and_load(nc) {
            Ok((pid, name)) => logln!("orbit-loader: spawned pid={pid} name={name:?}"),
            Err(e)          => logln!("orbit-loader: iteration failed: {e:?}"),
        }

        // Return the NetChannel to idle before re-listening. This is
        // the shared-memory reset handshake: request → wait for kernel
        // to drop current_state to 0 → finish by clearing our half.
        if let Err(e) = recycle(nc) {
            // If the channel is already idle (e.g., listen_tcp itself
            // failed), recycle returns Err — ignore and loop; a bad
            // listen will re-surface as a LoaderErr::Listen next pass.
            logln!("orbit-loader: recycle skipped: {e:?}");
        }
    }
}

/// One listen-accept-recv-spawn cycle on an already-registered
/// NetChannel. Doesn't touch the NetChannel allocation; the caller
/// handles recycling.
fn accept_and_load(nc: &NetChannel) -> Result<(u16, String), LoaderErr> {
    if nc.listen_tcp(LISTEN_PORT).is_err() {
        return Err(LoaderErr::Listen);
    }

    wait_established(nc)?;
    let (payload_bytes, name) = recv_payload(nc)?;
    let pid = spawn(&payload_bytes)?;
    Ok((pid, name))
}

/// Drive the reset handshake: write `desired_state = 0`, poll
/// `current_state == 0`, then clear our own ring indices. After this
/// the channel is safe for a fresh `listen_tcp`. Returns `Err` if the
/// channel was already idle (nothing to reset) — caller decides whether
/// that's a problem.
fn recycle(nc: &NetChannel) -> Result<(), LoaderErr> {
    logln!("orbit-loader: recycling...");

    nc.request_reset().map_err(|()| LoaderErr::ConnClosed(0))?;
    loop {
        let st = nc.current_state().state.load(Ordering::Acquire);
        if st == 0 { break; }
        sleep_ms(POLL_SLEEP_MS);
    }
    // SAFETY: we are the sole user-side accessor for this channel and
    // have no outstanding recv_tcp/send_tcp closures in flight.
    unsafe { nc.complete_reset(); }
    Ok(())
}

fn wait_established(nc: &NetChannel) -> Result<(), LoaderErr> {
    loop {
        let st = nc.current_state().state.load(Ordering::Acquire);
        if st > 0 { return Ok(()); }
        if st < 0 { return Err(LoaderErr::ConnClosed(st)); }
        sleep_ms(POLL_SLEEP_MS);
    }
}

/// Read the full framed message into a heap Vec. Returns (buf, name)
/// where `buf` is the trimmed CBOR body and `name` is decoded from it.
/// The returned bytes are handed verbatim to the kernel.
fn recv_payload(nc: &NetChannel) -> Result<(Vec<u8>, String), LoaderErr> {
    let mut scratch: Vec<u8> = Vec::new();

    // Fill enough for the 8-byte header.
    while scratch.len() < 8 {
        drain_once(nc, &mut scratch)?;
    }

    let len = u32::from_le_bytes([scratch[0], scratch[1], scratch[2], scratch[3]]);
    let inv = u32::from_le_bytes([scratch[4], scratch[5], scratch[6], scratch[7]]);
    if len ^ inv != u32::MAX { return Err(LoaderErr::Framing); }
    if (len as usize) > MAX_ELF_BYTES { return Err(LoaderErr::TooLarge(len)); }

    let total = 8 + len as usize;
    scratch.reserve(total.saturating_sub(scratch.len()));
    while scratch.len() < total {
        drain_once(nc, &mut scratch)?;
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

/// Pull whatever's available from the ring into `out`. Blocks (polling
/// via sleep_ms) until at least one byte arrives or the connection
/// drops. `recv_tcp` returns `Ok(0)` when the closure consumes nothing,
/// so we bail on that too to avoid live-lock on a full-but-empty ring.
fn drain_once(nc: &NetChannel, out: &mut Vec<u8>) -> Result<(), LoaderErr> {
    loop {
        if nc.readable() > 0 {
            let r = nc.recv_tcp(|rx| {
                let start = out.len();
                out.resize(start + rx.len(), 0);
                let n = rx.copy_to_slice(&mut out[start..]);
                out.truncate(start + n);
                n
            });
            match r {
                Ok(n) if n > 0 => return Ok(()),
                Ok(_)          => {}
                Err(e)         => return Err(LoaderErr::ConnClosed(e as i32)),
            }
        }
        let st = nc.current_state().state.load(Ordering::Acquire);
        if st <= 0 { return Err(LoaderErr::ConnClosed(st)); }
        sleep_ms(POLL_SLEEP_MS);
    }
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
    logln!("orbit-loader: spawn ptr={:p} len={} head={:02x?}",
           elf.as_ptr(), elf.len(), head);
    match create_process(elf.as_ptr(), elf.len()) {
        Ok(pid) => Ok(pid),
        Err(e)  => Err(LoaderErr::Syscall(e)),
    }
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "orbit-loader panic: {p}");
    w.flush();
    exit(isize::MIN);
}
