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
use net_channel::{BindSpec, NC_MAX_REGION_SIZE, NetChannel};
use orbit_abi::errno::Errno;
use orbit_abi::fs::Stat;
use orbit_abi::net::SockType;
use orbit_abi::{
    logln,
    user::{
        ConsoleWriter, close_handle, create_process, create_process_with_argv_envp, exit, fs_open,
        fs_read, fs_stat,
    },
};
use orbit_rt::netch::{NetCh, Session};

const LISTEN_PORT: u16 = 7777;
/// Size the rings to the maximum the kernel will allocate, so a large
/// CBOR payload doesn't ping-pong through the rings more than necessary.
const RING_CAPACITY: usize = NetChannel::capacity_for(NC_MAX_REGION_SIZE);
// Matches the kernel-side cap in handle_create_process_req. Rejecting
// before we allocate the buffer keeps a bogus header from forcing us
// to grow the heap to 4 MiB and then get -1'd by the syscall.
const MAX_ELF_BYTES: usize = 4 * 1024 * 1024;

/// Cap on init-binary size we'll fs_read off the disk. Generous enough
/// for hello-std-shaped binaries (~300 KiB today) with headroom; the
/// loader's heap grows on demand via dlmalloc so a smaller cap doesn't
/// save memory until we hit it.
const MAX_INIT_ELF_BYTES: usize = 4 * 1024 * 1024;

// `map` (not the derive default `array`) so new optional fields can be
// added later without breaking existing senders — map entries are keyed
// by their `#[n(N)]` index, so missing keys are tolerated rather than
// shifting every subsequent field.
#[derive(Decode, Encode, Debug)]
#[cbor(map)]
struct Payload<'a> {
    #[n(0)]
    #[cbor(with = "minicbor::bytes")]
    elf: &'a [u8],
    #[b(1)]
    name: &'a str,
    /// Optional CPU affinity cap. Sentinel `0` (also the absent-key
    /// default since minicbor leaves missing primitives at zero) tells
    /// the kernel to use the all-harts default.
    #[n(2)]
    #[cbor(default)]
    allowed_affinity: u64,
    /// Optional initial affinity. Sentinel `0` → defaults to whatever
    /// `allowed_affinity` resolves to. Must be a subset of
    /// `allowed_affinity` once both are resolved (kernel enforces).
    #[n(3)]
    #[cbor(default)]
    affinity: u64,
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
    // Boot init: read argv[1] (init path), fs_load it from tarfs, and
    // create_process. Replaces the previous `include_bytes!` of the
    // console binary — kmain now passes the init path as argv when
    // spawning the loader, so the same loader image serves
    // default/console, smoke/umode, and hello-std bringups by just
    // pointing at a different `/bin/<name>` entry.
    //
    // Failures (no argv, missing path, fs not mounted, oversize ELF,
    // ...) log and fall through. The loader's TCP listener is still
    // useful for ad-hoc binary delivery without a working init.
    log_boot_env();
    spawn_init_from_argv();

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
            Err(e) => logln!("orbit-loader: iteration failed: {e:?}"),
        }
        // Session is dropped at end of accept_and_load — disengagement
        // and kernel-side relisten happen there.
    }
}

/// One-line dump of the boot envp the kernel installed for us. Lets
/// `./smoke` confirm Phase 5 delivery without poking at internal log
/// markers. Cheap: a single iter over the seeded BTreeMap.
fn log_boot_env() {
    let entries = orbit_rt::env::vars();
    if entries.is_empty() {
        logln!("orbit-loader: boot env empty");
        return;
    }
    logln!("orbit-loader: boot env ({} entries):", entries.len());
    for (k, v) in &entries {
        let key = core::str::from_utf8(k).unwrap_or("<non-utf8>");
        let val = core::str::from_utf8(v).unwrap_or("<non-utf8>");
        logln!("  {key}={val}");
    }
}

/// Read argv[1] (init path), pull the ELF off tarfs, hand to
/// `create_process`. All failure modes log and return — the caller
/// continues into the TCP listen loop regardless.
fn spawn_init_from_argv() {
    let args = orbit_rt::argv::args();
    let path_bytes = match args.get(1) {
        Some(p) if !p.is_empty() => p,
        _ => {
            logln!("orbit-loader: no init path in argv (skipping init spawn)");
            return;
        }
    };
    let path = match core::str::from_utf8(path_bytes) {
        Ok(s) => s,
        Err(_) => {
            logln!("orbit-loader: init path is not utf-8 (skipping)");
            return;
        }
    };

    // fs_stat to size the heap allocation. Without this we'd grow the
    // Vec by sector, which works but wastes a few resizes for the
    // common multi-hundred-KiB ELF.
    let mut st = Stat::default();
    if let Err(e) = fs_stat(path, &mut st) {
        logln!("orbit-loader: fs_stat({path}) failed: {e:?} (skipping init)");
        return;
    }
    let total = st.st_size as usize;
    if total == 0 || total > MAX_INIT_ELF_BYTES {
        logln!(
            "orbit-loader: init {path} size={total} out of range (cap={}); skipping",
            MAX_INIT_ELF_BYTES,
        );
        return;
    }

    let fd = match fs_open(path, 0) {
        Ok(fd) => fd,
        Err(e) => {
            logln!("orbit-loader: fs_open({path}) failed: {e:?} (skipping init)");
            return;
        }
    };

    // Sector-aligned scratch buf. fs_read demands `len == 512` and
    // rejects buffers that straddle a 4 KiB page; aligning to 512
    // keeps the allocation page-resident.
    #[repr(align(512))]
    struct AlignedBuf([u8; 512]);
    let mut scratch = AlignedBuf([0u8; 512]);
    let mut elf: Vec<u8> = Vec::with_capacity(total);

    while elf.len() < total {
        match fs_read(fd, &mut scratch.0) {
            Ok(0) => break, // EOF before total — surface below.
            Ok(n) => {
                let take = core::cmp::min(n, total - elf.len());
                elf.extend_from_slice(&scratch.0[..take]);
            }
            Err(e) => {
                logln!(
                    "orbit-loader: fs_read({path}) at offset {} failed: {e:?}",
                    elf.len(),
                );
                let _ = close_handle(fd);
                return;
            }
        }
    }
    let _ = close_handle(fd);

    if elf.len() != total {
        logln!(
            "orbit-loader: init {path} read {} bytes, expected {total} (skipping spawn)",
            elf.len(),
        );
        return;
    }

    // Page-aligned, page-sized envp scratch. Stack-local because
    // spawn_init_from_argv runs once at boot and a 4 KiB stack burst
    // is well within the umode default stack. The VA must be
    // page-aligned (kernel rejects misaligned envp_va with EINVAL)
    // and the kernel always reads exactly one page from it.
    #[repr(C, align(4096))]
    struct EnvPage([u8; 4096]);
    let mut env_page = EnvPage([0u8; 4096]);
    let envp_va = pack_env_for_child(&mut env_page.0);

    // Init gets the all-harts default — passing 0 for both affinity
    // fields tells the kernel "no preference." Argv stays empty for
    // v1 inits (they don't need command-line args from the loader).
    // Envp is propagated from our own kernel-installed env so the
    // child sees the same baseline (PATH/HOME/TERM, plus anything we
    // add later).
    let argv_blob: &[u8] = &[];
    match create_process_with_argv_envp(elf.as_ptr(), elf.len(), 0, 0, argv_blob, envp_va) {
        Ok(pid) => logln!(
            "orbit-loader: spawned init {path} pid={pid} bytes={total} envp={}",
            if envp_va == 0 { "skipped" } else { "inherited" },
        ),
        Err(e) => logln!("orbit-loader: init {path} create_process failed: {e:?}"),
    }
}

/// Snapshot the current process env (seeded from the kernel envp at
/// boot) and pack it into `buf` for handoff to a child via
/// `create_process_with_argv_envp`. Returns the page-aligned VA, or
/// `0` if there's nothing to install (empty env or pack failure).
///
/// `buf` must be exactly one page and page-aligned — the kernel-side
/// copy reads `PAGE_SIZE` bytes from the returned VA. The call zeros
/// `buf` first so unused tail bytes don't carry uninitialised stack
/// data into the child's env page.
fn pack_env_for_child(buf: &mut [u8; 4096]) -> usize {
    let entries = orbit_rt::env::vars();
    if entries.is_empty() {
        return 0;
    }
    // Flatten each (k, v) pair into "KEY=VALUE" bytes. The pack
    // helper takes `&[&[u8]]`, so we own one Vec per entry and
    // collect refs over the owned vec.
    let kvs: Vec<Vec<u8>> = entries
        .into_iter()
        .map(|(mut k, v)| {
            k.reserve(1 + v.len());
            k.push(b'=');
            k.extend_from_slice(&v);
            k
        })
        .collect();
    let refs: Vec<&[u8]> = kvs.iter().map(|e| e.as_slice()).collect();

    buf.fill(0);
    match orbit_abi::envp::pack(&refs, buf) {
        Some(_) => buf.as_ptr() as usize,
        None => {
            // Total entries exceed one page — drop envp inheritance
            // entirely rather than truncate (truncation would silently
            // corrupt KEY=VALUE structure mid-string).
            logln!("orbit-loader: env doesn't fit in one page; init spawned without envp");
            0
        }
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
    if len ^ inv != u32::MAX {
        return Err(LoaderErr::Framing);
    }
    if (len as usize) > MAX_ELF_BYTES {
        return Err(LoaderErr::TooLarge(len));
    }

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
    let n = s
        .read_some_with_poll_timeout(&mut tmp, 100)
        .map_err(LoaderErr::NetCh)?;
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
    logln!(
        "orbit-loader: spawn ptr={:p} len={} head={:02x?} \
           allowed_affinity={:#x} affinity={:#x}",
        elf.as_ptr(),
        elf.len(),
        head,
        payload.allowed_affinity,
        payload.affinity
    );
    create_process(
        elf.as_ptr(),
        elf.len(),
        payload.allowed_affinity,
        payload.affinity,
    )
    .map_err(LoaderErr::Syscall)
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "orbit-loader: panic: {p}");
    w.flush();
    exit(isize::MIN);
}
