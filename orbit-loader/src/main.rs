//! orbit-loader — u-mode listener that accepts an ELF over TCP and asks
//! the kernel to spawn it via `create_process_v2`. Replaces `include_bytes!`
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
//! One [`NetCh`] serves the loader's lifetime; each client runs in a
//! `Session` whose `Drop` recycles the channel, so we don't tear down
//! and reallocate between clients.

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
use orbit_abi::perms::{CreateProcessV2Args, role};
use orbit_abi::{
    logln,
    user::{ConsoleWriter, close_handle, create_process_v2, exit, fs_open, fs_read, fs_stat},
};

/// Default identity stamped on every payload the loader spawns, until
/// real `/etc/passwd` lookup + login flow lands in a later milestone.
/// Picked as 1000/1000 to match the conventional "first non-root user"
/// allocation on Linux/BSD — gives `whoami`/`id` an interesting answer
/// without a passwd file.
const DEFAULT_PAYLOAD_UID: i64 = 1000;
const DEFAULT_PAYLOAD_GID: i64 = 1000;
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
    /// Optional argv. Each element is a borrowed UTF-8 string from
    /// the CBOR body; the loader converts to `&[u8]` and hands the
    /// packed blob through `CreateProcessV2Args.argv_vaddr`. Empty /
    /// missing → child sees argc=0. Strings (rather than byte
    /// strings) keep the CBOR shape simple and work for every CLI
    /// argv we've shipped so far.
    #[n(4)]
    #[cbor(default)]
    argv: alloc::vec::Vec<&'a str>,
}

#[derive(Debug)]
#[allow(dead_code)] // payload fields are read via Debug in logln!
enum LoaderErr {
    Framing,
    TooLarge(u32),
    Cbor,
    NetCh(Errno),
    Syscall(Errno),
    BadAck(isize),
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    // Boot init: read argv[1] (init path), fs_load it from tarfs, and
    // create_process_v2. Replaces the previous `include_bytes!` of the
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

    if let Err(e) = sleep_ms(1000) {
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
/// `create_process_v2`. All failure modes log and return — the caller
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

    // Sector-sized scratch buf. fs_read now accepts any length up to
    // 64 KiB across multiple pages; we keep a 512-byte aligned buffer
    // and read a sector at a time for simplicity.
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
    // add later). Identity stamps via the same loader-default path
    // as network payloads — init runs as 1000:1000 too.
    let args = CreateProcessV2Args {
        elf_vaddr: elf.as_ptr() as usize,
        elf_len: elf.len(),
        allowed_affinity: 0,
        affinity: 0,
        target_role: role::INHERIT,
        // Init is the one process we *would* want to observe; keep it
        // reapable.
        flags: 0,
        request_perms: 0,
        request_allowed_perms: 0,
        cwd_vaddr: 0,
        cwd_len: 0,
        argv_vaddr: 0,
        argv_len: 0,
        envp_vaddr: envp_va,
        stdout_capture: 0,
        _pad2: 0,
        setuid_uid: DEFAULT_PAYLOAD_UID,
        setuid_gid: DEFAULT_PAYLOAD_GID,
        setlogin_vaddr: 0,
        setlogin_len: 0,
        groups_vaddr: 0,
        groups_count: 0,
        spawn_path_len: 0,
        spawn_path_vaddr: 0,
    };
    match create_process_v2(&args) {
        Ok(pid) => logln!(
            "orbit-loader: spawned init {path} pid={pid} bytes={total} envp={}",
            if envp_va == 0 { "skipped" } else { "inherited" },
        ),
        Err(e) => logln!("orbit-loader: init {path} create_process_v2 failed: {e:?}"),
    }
}

/// Snapshot the current process env (seeded from the kernel envp at
/// boot) and pack it into `buf` for handoff to a child via
/// `create_process_v2` (`CreateProcessV2Args.envp_vaddr`). Returns the page-aligned VA, or
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

    s.write_all(&[0xFF])
        .map_err(|e| LoaderErr::BadAck(e.to_ret()))?;

    let body = &scratch[8..total];

    // Decode name eagerly so we can log it; elf bytes stay in `scratch`
    // and the kernel reads them via create_process_v2.
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
        .read_some_with_poll_timeout(&mut tmp, 25)
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
           allowed_affinity={:#x} affinity={:#x} argc={}",
        elf.as_ptr(),
        elf.len(),
        head,
        payload.allowed_affinity,
        payload.affinity,
        payload.argv.len()
    );
    // Pack argv into the same wire format orbit-rt's `_start` expects.
    // 4 KiB scratch matches the kernel's argv-page mapping. Empty
    // argv is fine — pack returns a zero-string header and we pass
    // argv_vaddr=0/argv_len=0 below.
    let argv_bytes: alloc::vec::Vec<&[u8]> = payload.argv.iter().map(|s| s.as_bytes()).collect();
    let mut argv_buf = [0u8; 4096];
    let argv_blob: &[u8] = if payload.argv.is_empty() {
        &[]
    }
    else {
        let Some(argv_len) = orbit_abi::argv::pack(&argv_bytes, &mut argv_buf)
        else {
            return Err(LoaderErr::Cbor);
        };
        &argv_buf[..argv_len]
    };

    // Stamp default uid/gid on every payload — orbit-loader is the
    // single privileged spawner today, so this is the only path that
    // produces non-root user processes. Once a real login pipeline
    // lands the loader will defer identity to that path; until then
    // every payload runs as 1000:1000.
    let args = CreateProcessV2Args {
        elf_vaddr: elf.as_ptr() as usize,
        elf_len: elf.len(),
        allowed_affinity: payload.allowed_affinity,
        affinity: payload.affinity,
        // INHERIT keeps loader's role + perms verbatim. A future
        // signed-manifest path would name a concrete RoleId here.
        target_role: role::INHERIT,
        // Detach: orbit-loader is fire-and-forget per network payload
        // and never `wait_pid`s its children. Without this, every
        // payload exit accumulates an entry in the loader's
        // `dead_children` map (BTreeMap<u16, i32>), which under
        // long-running stress (eza_stress.py) grows the kernel-heap
        // allocations indefinitely and was the trigger for the
        // jump-through-bad-fnptr fault inside the BTreeMap insert path.
        flags: CreateProcessV2Args::DETACH,
        request_perms: 0,
        request_allowed_perms: 0,
        cwd_vaddr: 0,
        cwd_len: 0,
        argv_vaddr: argv_blob.as_ptr() as usize,
        argv_len: argv_blob.len(),
        envp_vaddr: 0,
        stdout_capture: 0,
        _pad2: 0,
        setuid_uid: DEFAULT_PAYLOAD_UID,
        setuid_gid: DEFAULT_PAYLOAD_GID,
        setlogin_vaddr: 0,
        setlogin_len: 0,
        groups_vaddr: 0,
        groups_count: 0,
        spawn_path_len: 0,
        spawn_path_vaddr: 0,
    };
    create_process_v2(&args).map_err(LoaderErr::Syscall)
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "orbit-loader: panic: {p}");
    w.flush();
    exit(isize::MIN);
}
