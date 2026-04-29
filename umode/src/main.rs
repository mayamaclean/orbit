#![no_std]
#![no_main]
#![feature(thread_local)]

extern crate alloc;
use orbit_rt as _;

use core::cell::Cell;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use orbit_abi::errno::{Errno, EBADF, ECHILD, EFAULT, EINVAL, ENOENT, ENOTDIR, EPERM};
use orbit_abi::fs::{
    DIRENT_HDR_LEN, DT_DIR, DT_REG, DirEntry, S_IFDIR, S_IFMT, S_IFREG, Stat,
};
use orbit_abi::net::SockType;
use orbit_abi::{logln, user::{close_handle, create_process_with_argv, create_thread, exit, fs_open, fs_read, fs_readdir, fs_stat, futex_wait, futex_wake, get_affinity, get_hart_id, getpid, gettid, set_affinity, sleep_ms, console_write, serial_print, wait_pid, ConsoleWriter}};
use net_channel::BindSpec;
use orbit_rt::netch::NetCh;

// =====================================================================
// §11 TLS isolation probe.
//
// Two `#[thread_local]` statics — one zero-init (lands in .tbss), one
// initialized to a sentinel value (lands in .tdata). Each thread sees
// its own copy: the per-thread TLS block is allocated and copy-init'd
// from the binary's PT_TLS template at create_thread time; tp points
// at it. `MY_TICK` (Cell) lets each thread bump a private counter to
// confirm writes don't bleed across threads.
// =====================================================================

#[thread_local]
static TLS_SENTINEL: Cell<u32> = Cell::new(0xC0FFEEu32);

#[thread_local]
static MY_TICK: Cell<u32> = Cell::new(0);

// Communication out of the worker thread back to main, in shared
// process memory (NOT thread-local). Main reads these to confirm the
// worker observed the .tdata template and the writes were isolated.
static TLS_WORKER_SEEN_INIT: AtomicU32 = AtomicU32::new(0);
static TLS_WORKER_FINAL_TICK: AtomicU32 = AtomicU32::new(0);
static TLS_WORKER_DONE: AtomicU32 = AtomicU32::new(0);

extern "C" fn tls_worker_entry() -> ! {
    // First read of TLS_SENTINEL — should be the .tdata initial value
    // (0xC0FFEE) the kernel copied in from the PT_TLS template, not
    // the value main wrote on its own copy.
    TLS_WORKER_SEEN_INIT.store(TLS_SENTINEL.get(), Ordering::Release);

    // Bump a private tick a few times. If TLS isn't isolated, this
    // would race main's tick on the same memory.
    for i in 1..=5u32 {
        MY_TICK.set(i);
    }
    TLS_WORKER_FINAL_TICK.store(MY_TICK.get(), Ordering::Release);

    TLS_WORKER_DONE.store(1, Ordering::Release);
    exit(0);
}

fn run_tls_isolation_probe() {
    let (_cur, allowed) = get_affinity();
    if allowed.count_ones() < 2 {
        logln!("TLS isolation probe skipped (only one hart)");
        return;
    }

    // Main: write a distinct value into TLS_SENTINEL (overwrites the
    // .tdata initial value on this thread's copy only). Bump MY_TICK
    // to a different cadence than the worker.
    TLS_SENTINEL.set(0xDEAD_BEEFu32);
    for i in 100..=110u32 {
        MY_TICK.set(i);
    }

    // Spawn worker pinned to hart 1 — same shape as the create_thread
    // probe above. allowed=0 sentinel inherits parent's mask.
    let target_bit = 1u64 << 1;
    TLS_WORKER_DONE.store(0, Ordering::Release);
    if let Err(Errno(e)) = create_thread(tls_worker_entry, 0, target_bit) {
        logln!("FAIL: TLS probe create_thread errno={e}");
        return;
    }

    // Wait for worker to publish results.
    let mut tries = 0u32;
    while TLS_WORKER_DONE.load(Ordering::Acquire) == 0 {
        if tries >= 100 {
            logln!("FAIL: TLS probe worker didn't finish");
            return;
        }
        let _ = sleep_ms(10);
        tries += 1;
    }

    let worker_init = TLS_WORKER_SEEN_INIT.load(Ordering::Acquire);
    let worker_tick = TLS_WORKER_FINAL_TICK.load(Ordering::Acquire);
    let main_sentinel = TLS_SENTINEL.get();
    let main_tick = MY_TICK.get();

    let mut ok = true;
    if worker_init != 0xC0FFEE {
        logln!("FAIL: TLS probe worker saw 0x{worker_init:x} (want 0xc0ffee — main's write should not have been visible)");
        ok = false;
    }
    if worker_tick != 5 {
        logln!("FAIL: TLS probe worker tick = {worker_tick} (want 5)");
        ok = false;
    }
    if main_sentinel != 0xDEAD_BEEF {
        logln!("FAIL: TLS probe main sentinel = 0x{main_sentinel:x} (want 0xdeadbeef — worker overwrote main's TLS)");
        ok = false;
    }
    if main_tick != 110 {
        logln!("FAIL: TLS probe main tick = {main_tick} (want 110)");
        ok = false;
    }

    if ok {
        logln!("PASS: TLS isolation — main(sentinel=0x{main_sentinel:x},tick={main_tick}) worker(init=0x{worker_init:x},tick={worker_tick})");
    }
}

// Worker thread publishes its hart_id here once it starts. `u32::MAX`
// is the sentinel for "not yet observed." Atomic so the main thread
// can spin-poll across the cross-hart memory barrier.
static WORKER_HART: AtomicU32 = AtomicU32::new(u32::MAX);

/// Entry point for the create_thread smoke probe's worker. Single
/// responsibility: read its own hart id, publish it, exit. No heap
/// access, no syscalls beyond `get_hart_id` + `exit`, so this thread
/// has minimal interactions with the rest of the runtime.
extern "C" fn worker_entry() -> ! {
    let hart = get_hart_id();
    WORKER_HART.store(hart, Ordering::Release);
    exit(0);
}

// =====================================================================
// Layer-3 cross-hart TLB-shootdown probe.
//
// Two threads in the same process, deliberately pinned to different
// harts. The setup makes the *cross-hart* path the only way the
// worker can observe the post-revoke invariant: if the kernel's
// shootdown machinery is wired correctly, the worker's stale TLB
// entry on its own hart gets flushed by the broadcast and a post-
// revoke read faults. If shootdown is broken (e.g., broadcast
// stubbed out), the worker's TLB still holds the translation and
// the read silently succeeds — we detect that and FAIL.
//
// Choreography:
//   1. Main opens a NetChannel; its shared region is the SharedUserPtr
//      whose user PTEs revoke will clear.
//   2. Main publishes the channel base VA via REVOKED_VA, spawns the
//      worker pinned to hart 1.
//   3. Worker reads one byte at REVOKED_VA — warms the hart-1 TLB
//      with that translation. Sets PHASE=1.
//   4. Main sees PHASE=1, calls nc.close(). Manager runs the revoke
//      from whichever hart drains MANAGER_WORK; with main pinned to
//      hart 0, that's almost always hart 0 (greedy manager). The
//      revoke clears the user PTEs on the manager's hart and
//      `crate::kernel::shootdown::broadcast(0, 0)` fans out to every
//      other hart — including hart 1, which the worker is on.
//   5. Main sets PHASE=2, signaling worker to retry.
//   6. Worker re-reads the same VA. With shootdown working, the TLB
//      is empty for that mapping and the load page-faults; the kernel
//      kills the worker thread (process keeps running). Without
//      shootdown, the read returns stale data and the worker sets
//      WORKER_SURVIVED=1.
//   7. Main sleeps a settling interval, checks WORKER_SURVIVED.
// =====================================================================

static SD_VA: AtomicUsize = AtomicUsize::new(0);
static SD_PHASE: AtomicU32 = AtomicU32::new(0);
static SD_WORKER_SURVIVED: AtomicU32 = AtomicU32::new(0);

extern "C" fn shootdown_worker_entry() -> ! {
    // Pre-warm wait can yield freely — TLB doesn't matter yet.
    while SD_VA.load(Ordering::Acquire) == 0 {
        let _ = sleep_ms(1);
    }
    let va = SD_VA.load(Ordering::Acquire);

    // Pre-revoke read: warm the hart-1 TLB with the translation.
    // read_volatile so the compiler can't elide it.
    let _warm = unsafe { core::ptr::read_volatile(va as *const u8) };
    SD_PHASE.store(1, Ordering::Release);

    // Tight spin (no yield, no syscall) for phase 2. Yielding here
    // would let the kernel context-switch the worker out and either
    // sfence hart 1's TLB on the way back in or simply let the entry
    // age out — either way the post-revoke read would fault for
    // reasons unrelated to shootdown, and the test would pass for
    // the wrong reason. The spin keeps hart 1 hot on this thread so
    // the *only* way the TLB entry leaves is via the shootdown IPI.
    while SD_PHASE.load(Ordering::Acquire) != 2 {
        core::hint::spin_loop();
    }

    // Post-revoke read. With shootdown wired correctly this faults;
    // the kernel kills this thread mid-load and we never reach the
    // store below. If we do reach it, the broadcast didn't reach
    // hart 1's TLB and we report the failure from main.
    let _stale = unsafe { core::ptr::read_volatile(va as *const u8) };
    SD_WORKER_SURVIVED.store(1, Ordering::Release);
    exit(0);
}

fn run_shootdown_probe() {
    let (_cur, allowed) = get_affinity();
    if allowed.count_ones() < 2 {
        logln!("shootdown probe skipped (only one hart)");
        return;
    }

    // Pin main to hart 0 so the manager that runs the revoke is also
    // hart 0 (greedy-manager picks the idle hart that's holding the
    // lock; main parks on the close handle, freeing hart 0). Worker
    // gets hart 1, ensuring the revoke happens on a hart != worker's.
    let _ = set_affinity(1u64 << 0);

    // The shootdown probe only needs a SharedUserPtr<NetChannel> to
    // revoke; it never moves data over TCP. Pick a binding the kernel
    // can wire up cheaply but that won't talk to anyone — ClientOneShot
    // sits in `FreshIdle` until the user engages, and we never do.
    let nc = match NetCh::open(
        0,
        SockType::Tcp,
        BindSpec::ClientOneShot { addr: 0x0100_007F, port: 1 },
    ) {
        Ok(n) => n,
        Err(Errno(e)) => {
            logln!("FAIL: shootdown probe NetCh::open errno={e}");
            return;
        }
    };
    let nc_va = nc.channel() as *const _ as usize;

    // Reset state so re-runs (none today, but defensive) start clean.
    SD_PHASE.store(0, Ordering::Release);
    SD_WORKER_SURVIVED.store(0, Ordering::Release);
    SD_VA.store(nc_va, Ordering::Release);

    let target_bit = 1u64 << 1;
    if let Err(Errno(e)) = create_thread(shootdown_worker_entry, 0, target_bit) {
        logln!("FAIL: shootdown probe create_thread errno={e}");
        let _ = nc.close();
        return;
    }

    // Wait for worker's pre-revoke read to complete.
    let mut tries = 0u32;
    while SD_PHASE.load(Ordering::Acquire) != 1 {
        if tries >= 100 {
            logln!("FAIL: shootdown probe worker never warmed TLB");
            let _ = nc.close();
            return;
        }
        let _ = sleep_ms(10);
        tries += 1;
    }

    // Trigger revoke. The orbit-rt drop path also closes, but we want
    // ordering relative to SD_PHASE so do it explicitly here.
    if let Err(Errno(e)) = nc.close() {
        logln!("FAIL: shootdown probe nc.close errno={e}");
        return;
    }

    // Tell worker to attempt the post-revoke read.
    SD_PHASE.store(2, Ordering::Release);

    // Settle. Worker either faults (kernel kills its thread; we never
    // see SURVIVED=1) or reads stale (sets SURVIVED=1 within a few
    // syscall round-trips). 200ms is a generous bound for both paths.
    let _ = sleep_ms(200);

    if SD_WORKER_SURVIVED.load(Ordering::Acquire) == 1 {
        logln!("FAIL: shootdown probe — worker survived post-revoke read \
               (cross-hart TLB still cached the translation)");
    } else {
        logln!("PASS: shootdown probe — worker faulted on post-revoke read");
    }

    // Restore main's affinity for the rest of the test flow.
    let _ = set_affinity(allowed);
}

/// End-to-end check that `create_thread` (syscall 5000) parks-and-wakes
/// correctly and that the new thread observes its `affinity` arg. Pins
/// the worker to hart 1 (always present once `cpu_count >= 2`); the
/// scheduler's affinity gate must steer it there even though the main
/// thread is running on a different hart and competing for dispatch.
fn run_create_thread_probe() {
    let (_cur, allowed) = get_affinity();

    // Test needs at least 2 harts. Skip silently if cpu_count < 2 — the
    // assertion would be vacuous and we'd flake against a config we
    // didn't intend to support yet.
    let target_bit = 1u64 << 1;
    if allowed & target_bit == 0 {
        logln!("create_thread probe skipped (only one hart)");
        return;
    }

    WORKER_HART.store(u32::MAX, Ordering::Release);

    // allowed=0 sentinel → manager substitutes parent's allowed mask.
    // affinity=target_bit pins worker to hart 1 specifically.
    match create_thread(worker_entry, 0, target_bit) {
        Ok(tid) => logln!("create_thread: spawned tid={tid}"),
        Err(Errno(e)) => {
            logln!("FAIL: create_thread spawn errno={e}");
            return;
        }
    }

    // Bounded poll. Worker runs `get_hart_id + atomic store + exit` — a
    // few syscalls' worth of work. 1s ceiling is generous; if the
    // scheduler can't wake the worker in that window, something is
    // structurally wrong (and the smoke wall-clock cap would catch it
    // anyway).
    let mut tries = 0u32;
    while WORKER_HART.load(Ordering::Acquire) == u32::MAX {
        if tries >= 100 {
            logln!("FAIL: create_thread worker never published its hart");
            return;
        }
        let _ = sleep_ms(10);
        tries += 1;
    }

    let observed = WORKER_HART.load(Ordering::Acquire);
    if observed == 1 {
        logln!("PASS: create_thread worker ran on hart 1 (target_bit=0x{target_bit:x})");
    } else {
        logln!("FAIL: create_thread worker ran on hart {observed} (want 1)");
    }
}

// =====================================================================
// §13a.5 futex probe.
//
// Two threads in the same process share a 4-byte counter at a fixed
// process-private VA (a `static AtomicU32`). The worker `futex_wait`s
// while the counter is `0`; main bumps the counter to `1` and
// `futex_wake`s. The worker's `futex_wait` then returns Ok(()), it
// publishes a "I woke" marker, and main verifies the EAGAIN fast-
// path separately by waiting on a value that doesn't match.
//
// Two threads in the same process is the minimum non-trivial test —
// a single-thread test couldn't `wake` itself. The cross-process
// shared-frame case (different satps, same PA) is part of the
// smoke's design intent but is gated on shared mmap from another
// process, which is its own bringup; covered in a future smoke.
// =====================================================================
static FUTEX_COUNTER: AtomicU32 = AtomicU32::new(0);
static FUTEX_WORKER_RAN: AtomicU32 = AtomicU32::new(0);
static FUTEX_WORKER_RESULT: AtomicU32 = AtomicU32::new(u32::MAX);

extern "C" fn futex_worker_entry() -> ! {
    FUTEX_WORKER_RAN.store(1, Ordering::Release);
    // Park while counter is still 0. The kernel re-reads the value
    // under the manager lock; if main already bumped it before we
    // get here we'd return EAGAIN immediately and the test would
    // FAIL by way of FUTEX_WORKER_RESULT not being 0. Main sleeps
    // for 50 ms after spawning to leave us time to land in the
    // wait queue before bumping the counter.
    let r = unsafe { futex_wait(FUTEX_COUNTER.as_ptr(), 0, 0) };
    let code = match r {
        Ok(()) => 0,
        Err(Errno(e)) => e as u32,
    };
    FUTEX_WORKER_RESULT.store(code, Ordering::Release);
    exit(0);
}

fn run_futex_probe() {
    let (_cur, allowed) = get_affinity();
    if allowed.count_ones() < 2 {
        logln!("futex probe skipped (only one hart)");
        return;
    }

    // Reset state — defensive; we only run this once today.
    FUTEX_COUNTER.store(0, Ordering::Release);
    FUTEX_WORKER_RAN.store(0, Ordering::Release);
    FUTEX_WORKER_RESULT.store(u32::MAX, Ordering::Release);

    // EAGAIN fast path: counter is 0, but we ask for `expected=42`.
    // The kernel reads `*counter`, sees `0 != 42`, and returns
    // -EAGAIN sync without parking.
    let eagain = unsafe { futex_wait(FUTEX_COUNTER.as_ptr(), 42, 0) };
    match eagain {
        Err(Errno(e)) if e == orbit_abi::errno::EAGAIN => {
            logln!("PASS: futex_wait value mismatch got Err(errno={e})");
        }
        other => {
            logln!("FAIL: futex_wait value mismatch want EAGAIN got {other:?}");
            return;
        }
    }

    // Spawn the parker on hart 1 — a different hart than main, so the
    // wake path crosses the IPI boundary (same shape as
    // create_thread_probe).
    let target_bit = 1u64 << 1;
    if let Err(Errno(e)) = create_thread(futex_worker_entry, 0, target_bit) {
        logln!("FAIL: futex probe create_thread errno={e}");
        return;
    }

    // Wait for worker to enter futex_wait. The RAN flag is set just
    // before the syscall — there's a short window between RAN=1 and
    // the kernel actually inserting the waiter. The 50ms sleep below
    // covers that window with margin.
    let mut tries = 0u32;
    while FUTEX_WORKER_RAN.load(Ordering::Acquire) == 0 {
        if tries >= 100 {
            logln!("FAIL: futex probe worker never entered wait");
            return;
        }
        let _ = sleep_ms(10);
        tries += 1;
    }
    let _ = sleep_ms(50);

    // Bump the counter and wake one waiter. Order matters: the user-
    // observable invariant is "counter != 0 implies the waker has
    // run" — the wake itself is an event signal, not the data
    // delivery, so the store goes first.
    FUTEX_COUNTER.store(1, Ordering::Release);
    let woken = unsafe { futex_wake(FUTEX_COUNTER.as_ptr(), 1) };
    match woken {
        Ok(n) => {
            if n != 1 {
                logln!("FAIL: futex_wake want 1 woke got {n}");
                return;
            }
            logln!("PASS: futex_wake n=1 woke {n}");
        }
        Err(Errno(e)) => {
            logln!("FAIL: futex_wake errno={e}");
            return;
        }
    }

    // Wait for worker to publish its return code. With wake working,
    // the worker's `futex_wait` returns Ok(()) (code 0).
    let mut tries = 0u32;
    while FUTEX_WORKER_RESULT.load(Ordering::Acquire) == u32::MAX {
        if tries >= 200 {
            logln!("FAIL: futex probe worker never resumed after wake");
            return;
        }
        let _ = sleep_ms(10);
        tries += 1;
    }
    let r = FUTEX_WORKER_RESULT.load(Ordering::Acquire);
    if r == 0 {
        logln!("PASS: futex_wait worker resumed (code=0)");
    } else {
        logln!("FAIL: futex_wait worker code={r} (want 0)");
        return;
    }

    // Wake on an empty queue: counter has nobody parked anymore.
    // Returns 0 woken.
    let woken_empty = unsafe { futex_wake(FUTEX_COUNTER.as_ptr(), 1) };
    match woken_empty {
        Ok(0) => logln!("PASS: futex_wake empty queue woke 0"),
        other => logln!("FAIL: futex_wake empty queue want Ok(0) got {other:?}"),
    }
}

/// Report a PASS/FAIL line for a single error-path scenario. The smoke
/// script greps for "PASS: <name>" lines to validate each branch fired.
fn check(name: &str, got: isize, want: isize) {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    if got == want {
        let _ = writeln!(w, "PASS: {name} got {got}");
    } else {
        let _ = writeln!(w, "FAIL: {name} want {want} got {got}");
    }
    w.flush();
}

/// Variant for syscalls that return `Result<T, Errno>`. Asserts the
/// call errored with `want`. The smoke script's grep is the same.
fn check_err<T: core::fmt::Debug>(name: &str, got: Result<T, Errno>, want: i32) {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    match got {
        Err(Errno(e)) if e == want => {
            let _ = writeln!(w, "PASS: {name} got Err(errno={e})");
        }
        other => {
            let _ = writeln!(w, "FAIL: {name} want Err(errno={want}) got {other:?}");
        }
    }
    w.flush();
}

/// Variant for syscalls that return `Result<T, Errno>`. Asserts the
/// call succeeded with `want`.
fn check_ok<T: core::fmt::Debug + PartialEq>(name: &str, got: Result<T, Errno>, want: T) {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    match got {
        Ok(v) if v == want => {
            let _ = writeln!(w, "PASS: {name} got Ok({v:?})");
        }
        other => {
            let _ = writeln!(w, "FAIL: {name} want Ok({want:?}) got {other:?}");
        }
    }
    w.flush();
}

/// Exercise the orbit-rt heap: first touch forces the talc `Source` to
/// mmap its first arena; subsequent pushes stay in that arena. Prints
/// PASS/FAIL for the smoke script to grep.
fn run_heap_smoke() {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::fmt::Write;

    let b = Box::new(0xABCDu32);
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "heap Box: {:#x} (want 0xabcd)", *b);
    w.flush();

    let mut v: Vec<u32> = Vec::new();
    for i in 0..1024 { v.push(i); }
    let sum: u64 = v.iter().map(|&x| x as u64).sum();
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "heap Vec sum: {sum} (want {})", (0u64..1024).sum::<u64>());
    w.flush();

    check("heap Box value", *b as isize, 0xABCD);
    check("heap Vec sum",   sum as isize, (0u64..1024).sum::<u64>() as isize);
}

/// Exercise syscall error paths that QEMU smoke otherwise never hits.
/// Each check prints a PASS/FAIL marker the smoke script verifies.
fn run_error_path_tests() {
    logln!("=== error path tests ===");

    // --- sleep_ms edge cases ---
    // The kernel caps sleep at 60*60*1000 ms. `>=` MAX returns EINVAL.
    check_err("sleep_ms at cap",    sleep_ms(60 * 60 * 1000),     EINVAL);
    check_err("sleep_ms above cap", sleep_ms(60 * 60 * 1000 + 1), EINVAL);

    // --- console_write / serial_print error paths ---
    // NULL-region VA (inside USER_NULL_GUARD_END) never translates → EFAULT.
    check_err("console_write null VA", console_write(0x1000, 5), EFAULT);
    check_err("serial_print null VA",  serial_print(0x1000, 5),  EFAULT);

    // len > PAGE_SIZE rejected with EINVAL before any memory is
    // touched, so the pointer just needs to be plausible.
    static FILLER: [u8; 16] = [b'x'; 16];
    check_err(
        "console_write too long",
        console_write(&FILLER as *const u8 as usize, 4097),
        EINVAL,
    );
    check_err(
        "serial_print too long",
        serial_print(&FILLER as *const u8 as usize, 4097),
        EINVAL,
    );

    // console_write doesn't validate UTF-8 — 4 bytes go through fine.
    // serial_print does, returns EINVAL on the same input.
    static BAD_UTF8: [u8; 4] = [0xFF, 0xFE, 0xFD, 0xFC];
    check_ok(
        "console_write non-utf8",
        console_write(&BAD_UTF8 as *const u8 as usize, 4),
        4usize,
    );
    check_err(
        "serial_print non-utf8",
        serial_print(&BAD_UTF8 as *const u8 as usize, 4),
        EINVAL,
    );

    // --- close_handle before any netchannel exists ---
    // No process_handles entry for this pid → EBADF.
    check_err("close_handle no registry", close_handle(7), EBADF);

    // --- affinity ---
    // Windows-shape: (current, allowed). Out of the box the loader hands
    // umode the all-harts default, so current == allowed and both have
    // bits matching the runtime's cpu_count. Smoke just asserts they're
    // equal and non-zero rather than pinning a specific value (cpu_count
    // is QEMU-config-dependent).
    let (cur, allowed) = get_affinity();
    {
        use core::fmt::Write;
        let mut w = ConsoleWriter::new();
        if cur != 0 && cur == allowed {
            let _ = writeln!(w,
                "PASS: get_affinity initial cur=0x{cur:x} allowed=0x{allowed:x}");
        } else {
            let _ = writeln!(w,
                "FAIL: get_affinity initial cur=0x{cur:x} allowed=0x{allowed:x}");
        }
        w.flush();
    }

    // Empty mask must not orphan the thread.
    check_err("set_affinity zero", set_affinity(0), EINVAL);

    // Bit outside the allowed cap (one above the highest set bit) must
    // not be silently masked. allowed has at least one bit set; the
    // first bit above its high bit is always outside the cap.
    let outside_bit = allowed
        .checked_shl((64 - allowed.leading_zeros()) as u32)
        .unwrap_or(0);
    if outside_bit != 0 {
        check_err("set_affinity outside cap", set_affinity(outside_bit), EPERM);
    }

    // Self-pin to bit 0 (always present; "all-harts" can't be empty).
    // Then sleep so the scheduler has multiple chances to dispatch — on
    // a working impl, get_hart_id must return 0 after the wake.
    let _ = set_affinity(1);
    let _ = sleep_ms(50);
    let pinned_to = get_hart_id();
    {
        use core::fmt::Write;
        let mut w = ConsoleWriter::new();
        if pinned_to == 0 {
            let _ = writeln!(w, "PASS: set_affinity pinned to hart 0 (got hart {pinned_to})");
        } else {
            let _ = writeln!(w, "FAIL: set_affinity pinned to hart 0 (got hart {pinned_to})");
        }
        w.flush();
    }

    // Restore so the rest of the test runs unconstrained — the
    // followup TCP sleep/recv loop expects scheduler flexibility.
    let _ = set_affinity(allowed);

    logln!("=== error path tests done ===");
}

/// FS smoke: stat / open / read / close against the boot-mounted
/// tarfs. Validates the §12d syscall stack end-to-end:
///   /README is 217 bytes → one short sector; first byte is 'O'
///     (the README starts with "Orbit rootfs.").
///   /bin/hello.txt is 26 bytes → one short sector starting with 'h'.
///   /bin is a directory → S_IFDIR in stat.
///   missing path → ENOENT.
///   read past EOF → 0.
fn run_fs_smoke() {
    // stat /README — confirm Linux-shape Stat fields wire up.
    let mut st = Stat::default();
    match fs_stat("/README", &mut st) {
        Ok(()) => {
            let kind = st.st_mode & S_IFMT;
            if kind == S_IFREG && st.st_size == 217 && st.st_blksize == 512 {
                logln!(
                    "PASS: fs_stat /README size={} mode={:#o} ino={} blocks={}",
                    st.st_size, st.st_mode, st.st_ino, st.st_blocks,
                );
            } else {
                logln!(
                    "FAIL: fs_stat /README unexpected size={} mode={:#o} blksize={}",
                    st.st_size, st.st_mode, st.st_blksize,
                );
            }
        }
        Err(e) => logln!("FAIL: fs_stat /README errored: {e:?}"),
    }

    // stat /bin — directory.
    let mut sd = Stat::default();
    match fs_stat("/bin", &mut sd) {
        Ok(()) => {
            if sd.st_mode & S_IFMT == S_IFDIR {
                logln!("PASS: fs_stat /bin dir mode={:#o}", sd.st_mode);
            } else {
                logln!("FAIL: fs_stat /bin not a dir, mode={:#o}", sd.st_mode);
            }
        }
        Err(e) => logln!("FAIL: fs_stat /bin errored: {e:?}"),
    }

    // open + read /bin/hello.txt. Buffer is sector-aligned via the
    // explicit alignment; the kernel rejects buffers that straddle a
    // 4 KiB page boundary.
    #[repr(align(512))]
    struct AlignedBuf([u8; 512]);
    let mut buf = AlignedBuf([0; 512]);

    let fd = match fs_open("/bin/hello.txt", 0) {
        Ok(fd) => fd,
        Err(e) => {
            logln!("FAIL: fs_open /bin/hello.txt: {e:?}");
            return;
        }
    };
    logln!("PASS: fs_open /bin/hello.txt fd={fd}");

    match fs_read(fd, &mut buf.0) {
        Ok(n) if n == 26 && &buf.0[..n] == b"hello from /bin/hello.txt\n" => {
            logln!("PASS: fs_read /bin/hello.txt n={n} matches");
        }
        Ok(n) => logln!(
            "FAIL: fs_read /bin/hello.txt n={n} got {:?}",
            core::str::from_utf8(&buf.0[..n.min(64)]).unwrap_or("<non-utf8>"),
        ),
        Err(e) => logln!("FAIL: fs_read /bin/hello.txt: {e:?}"),
    }

    // Past-EOF read returns 0 (the kernel sees offset >= file_size
    // after the previous successful read auto-advanced to 512).
    match fs_read(fd, &mut buf.0) {
        Ok(0) => logln!("PASS: fs_read past EOF returns 0"),
        Ok(n) => logln!("FAIL: fs_read past EOF returned {n}"),
        Err(e) => logln!("FAIL: fs_read past EOF errored: {e:?}"),
    }

    let _ = close_handle(fd);
    logln!("PASS: fs close /bin/hello.txt fd={fd}");

    // Missing path → ENOENT.
    match fs_open("/does-not-exist", 0) {
        Err(Errno(e)) if e == ENOENT => {
            logln!("PASS: fs_open missing got Err(errno={ENOENT})");
        }
        Ok(fd) => {
            logln!("FAIL: fs_open missing returned fd={fd}");
            let _ = close_handle(fd);
        }
        Err(e) => logln!("FAIL: fs_open missing errored: {e:?}"),
    }

    run_fs_readdir_smoke();

    logln!("=== fs smoke done ===");
}

/// `fs_readdir` smoke. The known rootfs (built from /rootfs/) has
/// exactly two top-level entries (`/README` regular, `/bin` dir) and
/// two entries under `/bin` (`hello`, `hello.txt`). Verify:
///   - readdir on `/` yields both top-level names with the right d_type
///   - a follow-up readdir returns 0 (end-of-directory)
///   - readdir on `/bin` yields the two child names
///   - readdir on a regular-file fd returns ENOTDIR
fn run_fs_readdir_smoke() {
    // ---- / ----
    let fd_root = match fs_open("/", 0) {
        Ok(fd) => fd,
        Err(e) => {
            logln!("FAIL: fs_readdir / open: {e:?}");
            return;
        }
    };
    logln!("PASS: fs_readdir / opened fd={fd_root}");

    let mut buf = [0u8; 256];
    let n = match fs_readdir(fd_root, &mut buf) {
        Ok(n) => n,
        Err(e) => {
            logln!("FAIL: fs_readdir / read: {e:?}");
            let _ = close_handle(fd_root);
            return;
        }
    };

    let mut saw_readme_reg = false;
    let mut saw_bin_dir = false;
    let mut count_root = 0usize;
    if !walk_dirents(&buf[..n], |name, d_type, _ino| {
        count_root += 1;
        if name == "README" && d_type == DT_REG {
            saw_readme_reg = true;
        }
        if name == "bin" && d_type == DT_DIR {
            saw_bin_dir = true;
        }
    }) {
        logln!("FAIL: fs_readdir / packed records malformed");
        let _ = close_handle(fd_root);
        return;
    }
    if saw_readme_reg && saw_bin_dir && count_root == 2 {
        logln!("PASS: fs_readdir / count=2 README+bin with right d_type");
    } else {
        logln!(
            "FAIL: fs_readdir / count={count_root} readme_reg={saw_readme_reg} bin_dir={saw_bin_dir}",
        );
    }

    // EOD: a second call past the cursor returns 0.
    match fs_readdir(fd_root, &mut buf) {
        Ok(0) => logln!("PASS: fs_readdir / EOD returns 0"),
        Ok(n) => logln!("FAIL: fs_readdir / EOD returned {n}"),
        Err(e) => logln!("FAIL: fs_readdir / EOD errored: {e:?}"),
    }
    let _ = close_handle(fd_root);

    // ---- /bin ----
    let fd_bin = match fs_open("/bin", 0) {
        Ok(fd) => fd,
        Err(e) => {
            logln!("FAIL: fs_readdir /bin open: {e:?}");
            return;
        }
    };
    let n = match fs_readdir(fd_bin, &mut buf) {
        Ok(n) => n,
        Err(e) => {
            logln!("FAIL: fs_readdir /bin read: {e:?}");
            let _ = close_handle(fd_bin);
            return;
        }
    };
    let mut saw_hello = false;
    let mut saw_hello_txt = false;
    let mut count_bin = 0usize;
    if !walk_dirents(&buf[..n], |name, d_type, _ino| {
        count_bin += 1;
        if name == "hello" && d_type == DT_REG {
            saw_hello = true;
        }
        if name == "hello.txt" && d_type == DT_REG {
            saw_hello_txt = true;
        }
    }) {
        logln!("FAIL: fs_readdir /bin packed records malformed");
        let _ = close_handle(fd_bin);
        return;
    }
    if saw_hello && saw_hello_txt && count_bin == 2 {
        logln!("PASS: fs_readdir /bin count=2 hello+hello.txt");
    } else {
        logln!(
            "FAIL: fs_readdir /bin count={count_bin} hello={saw_hello} hello_txt={saw_hello_txt}",
        );
    }
    let _ = close_handle(fd_bin);

    // ---- ENOTDIR on a regular-file fd ----
    let fd_file = match fs_open("/README", 0) {
        Ok(fd) => fd,
        Err(e) => {
            logln!("FAIL: fs_readdir ENOTDIR open /README: {e:?}");
            return;
        }
    };
    match fs_readdir(fd_file, &mut buf) {
        Err(Errno(e)) if e == ENOTDIR => {
            logln!("PASS: fs_readdir on regular file got Err(errno={ENOTDIR})");
        }
        Ok(n) => logln!("FAIL: fs_readdir on regular file returned {n}"),
        Err(e) => logln!("FAIL: fs_readdir on regular file errored: {e:?}"),
    }
    let _ = close_handle(fd_file);
}

/// Walk a packed-record buffer produced by `fs_readdir`, calling
/// `visit(name, d_type, d_ino)` for each entry. Returns `false` if the
/// stream is malformed (header runs off the end, name overflows the
/// record, non-utf8 name, d_reclen smaller than header+name).
fn walk_dirents(
    buf: &[u8],
    mut visit: impl FnMut(&str, u8, u64),
) -> bool {
    let mut p = 0usize;
    while p < buf.len() {
        if p + DIRENT_HDR_LEN > buf.len() {
            return false;
        }
        let hdr = unsafe {
            core::ptr::read_unaligned(buf[p..].as_ptr() as *const DirEntry)
        };
        // Copy out packed fields by value (taking refs into a packed
        // struct is UB; copies are fine).
        let reclen = hdr.d_reclen as usize;
        let nlen = hdr.d_namelen as usize;
        let d_type = hdr.d_type;
        let ino = hdr.d_ino;
        if reclen < DIRENT_HDR_LEN + nlen || p + reclen > buf.len() {
            return false;
        }
        let name_start = p + DIRENT_HDR_LEN;
        let name_bytes = &buf[name_start..name_start + nlen];
        let name = match core::str::from_utf8(name_bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };
        visit(name, d_type, ino);
        p += reclen;
    }
    true
}

/// §13a.1 identity probe. Calls `getpid` and `gettid` from the main
/// thread, asserts pid is the boot pid (1) and tid is non-zero +
/// stable across calls. tid is system-global (matches Linux's
/// `gettid()` shape), so its absolute value depends on how many
/// kernel threads — k_net etc. — got allocated tids before umode
/// started; assert nonzero and stability instead of a fixed value.
fn run_identity_probe() {
    let pid = getpid();
    if pid == 1 {
        logln!("PASS: getpid main got {pid}");
    } else {
        logln!("FAIL: getpid main got {pid} (want 1)");
    }

    let tid_a = gettid();
    let tid_b = gettid();
    if tid_a > 0 && tid_a == tid_b {
        logln!("PASS: gettid main got {tid_a} (stable across calls)");
    } else {
        logln!("FAIL: gettid main got {tid_a} then {tid_b}");
    }
}

/// §12e exec smoke: read `/bin/hello` (a real ELF on the disk image)
/// and hand it to `create_process`. Spawn-only flavor — no `wait_pid`
/// yet, so we just sleep briefly for the child to print its marker.
///
/// Mirrors what the console's `exec` builtin does at the prompt, but
/// runs from umode so the existing smoke harness can validate it
/// without driving an interactive shell.
fn run_exec_smoke() {
    extern crate alloc;
    use alloc::vec::Vec;

    // Stat to size the buffer.
    let mut st = Stat::default();
    if let Err(e) = fs_stat("/bin/hello", &mut st) {
        logln!("FAIL: exec_smoke fs_stat /bin/hello: {e:?}");
        return;
    }
    if st.st_size <= 0 {
        logln!("FAIL: exec_smoke /bin/hello unexpected size {}", st.st_size);
        return;
    }
    let total = st.st_size as usize;
    logln!("PASS: exec_smoke fs_stat /bin/hello size={total}");

    // Open + chunked read into a heap buffer. Sector-aligned scratch
    // buf for the read syscall (kernel rejects buffers that straddle
    // a 4 KiB page).
    let fd = match fs_open("/bin/hello", 0) {
        Ok(fd) => fd,
        Err(e) => {
            logln!("FAIL: exec_smoke fs_open /bin/hello: {e:?}");
            return;
        }
    };

    #[repr(align(512))]
    struct AlignedBuf([u8; 512]);
    let mut scratch = AlignedBuf([0; 512]);
    let mut elf: Vec<u8> = Vec::with_capacity(total);

    while elf.len() < total {
        match fs_read(fd, &mut scratch.0) {
            Ok(0) => break, // EOF
            Ok(n) => elf.extend_from_slice(&scratch.0[..n]),
            Err(e) => {
                logln!("FAIL: exec_smoke fs_read at offset {}: {e:?}", elf.len());
                let _ = close_handle(fd);
                return;
            }
        }
    }
    let _ = close_handle(fd);

    if elf.len() != total {
        logln!("FAIL: exec_smoke read {} bytes, expected {total}", elf.len());
        return;
    }
    logln!("PASS: exec_smoke read {total} bytes from /bin/hello");

    // Sanity: ELF magic at byte 0.
    if elf.get(0..4) != Some(&[0x7f, b'E', b'L', b'F']) {
        logln!("FAIL: exec_smoke /bin/hello missing ELF magic");
        return;
    }
    logln!("PASS: exec_smoke ELF magic present");

    // §13a.3 — pack argv ["world", "peace"] (hello's own path lands
    // implicitly as argv[0] later if a convention emerges; v1 just
    // packs whatever umode hands in). Spawn via create_process_ex.
    let mut argv_buf = [0u8; 256];
    let argv_args: [&[u8]; 3] = [b"/bin/hello", b"world", b"peace"];
    let argv_len = orbit_abi::argv::pack(&argv_args, &mut argv_buf)
        .expect("argv blob fits in 256 bytes");
    let argv_blob = &argv_buf[..argv_len];

    let child_pid = match create_process_with_argv(elf.as_ptr(), elf.len(), 0, 0, argv_blob) {
        Ok(pid) => {
            logln!("PASS: exec_smoke create_process_with_argv pid={pid}");
            pid
        }
        Err(e) => {
            logln!("FAIL: exec_smoke create_process_with_argv: {e:?}");
            return;
        }
    };

    // §13a.2 — block until the child exits. Validates the full chain:
    // dealloc_process takes exit_waiter, signal_pair fires the wake
    // hook, the parked thread resumes with exit_code in a1.
    match wait_pid(child_pid) {
        Ok(42) => logln!("PASS: exec_smoke wait_pid pid={child_pid} got 42"),
        Ok(n) => logln!("FAIL: exec_smoke wait_pid got {n} (want 42)"),
        Err(e) => logln!("FAIL: exec_smoke wait_pid errored: {e:?}"),
    }

    // wait_pid error paths. Self-wait → EINVAL, missing pid → ECHILD,
    // post-reap of the same child → ECHILD (no zombies in v1 — once
    // dealloc_process runs, the Process is gone).
    match wait_pid(getpid()) {
        Err(Errno(e)) if e == EINVAL => logln!("PASS: wait_pid self got Err(errno={EINVAL})"),
        other => logln!("FAIL: wait_pid self got {other:?}"),
    }
    match wait_pid(9999) {
        Err(Errno(e)) if e == ECHILD => logln!("PASS: wait_pid missing got Err(errno={ECHILD})"),
        other => logln!("FAIL: wait_pid missing got {other:?}"),
    }
    match wait_pid(child_pid) {
        Err(Errno(e)) if e == ECHILD => logln!("PASS: wait_pid post-reap got Err(errno={ECHILD})"),
        other => logln!("FAIL: wait_pid post-reap got {other:?}"),
    }

    logln!("=== exec smoke done ===");
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    // print to serial
    logln!("hello world!");

    run_heap_smoke();

    // §13a.1 identity probe — runs before any worker thread spawns so
    // the main thread's tid is deterministic (= 1, the first tid the
    // kernel allocates). umode is the boot pid, so getpid() == 1.
    run_identity_probe();

    run_fs_smoke();

    run_exec_smoke();

    run_error_path_tests();

    // Cross-hart probe — must run after run_error_path_tests, which
    // restores affinity to all-harts; pinning the main thread first
    // would force the worker and main onto the same hart and defeat
    // the point of the test.
    run_create_thread_probe();

    // §11 TLS isolation — verifies #[thread_local] statics are
    // per-thread (each thread sees its own copy). Must run before
    // the shootdown probe pins main to hart 0.
    run_tls_isolation_probe();

    // §13a.5 futex round trip — must run before the shootdown probe
    // pins main to hart 0; the worker pinned to hart 1 needs main on
    // a different hart so the wake crosses the IPI boundary.
    run_futex_probe();

    // §10 layer-3: cross-hart TLB shootdown probe. Pins itself and
    // restores affinity on the way out. Must run before the TCP
    // listener flow below opens its real netchannel — they're
    // sequential users of the per-process NetCh slot.
    run_shootdown_probe();

    let _ = sleep_ms(2000);

    // Open a NetChannel as a one-shot client to the smoke listener.
    // The kernel latches the BindSpec at create time and waits for us
    // to engage before dialing — `next_session` does both.
    let nc = match NetCh::open(
        0,
        SockType::Tcp,
        BindSpec::ClientOneShot {
            addr: u32::from_be_bytes([192, 168, 76, 2]),
            port: 65535,
        },
    ) {
        Ok(n) => n,
        Err(_) => {
            logln!("failed to create netchannel!");
            exit(-2isize);
        }
    };

    logln!("netchannel created!");

    // Bogus fd AFTER a netchannel has been created — process_handles
    // now has an entry for this pid, but fd 999 isn't in it → EBADF.
    // (The earlier `no registry` test hit the no-pid-entry branch.)
    check_err("close_handle bogus fd", close_handle(999), EBADF);

    let session = match nc.next_session() {
        Ok(s) => s,
        Err(_) => {
            logln!("tcp connect failed!");
            exit(-2isize);
        }
    };
    logln!("tcp connected!");

    // Send the greeting once, then drain any reply. The smoke
    // listener echoes "exit\n" to terminate this loop.
    if session.write_all(b"Hello World!\n").is_err() {
        logln!("tcp write failed!");
        exit(-2isize);
    }

    let mut buf = [0u8; 1024];
    loop {
        match session.read(&mut buf) {
            Ok(n) if n > 0 => {
                // console_write before the exit check so the smoke
                // script can grep the "exit\n" payload off serial —
                // console_write tee's to serial as `USER[pid]: ...`.
                let _ = console_write(buf.as_ptr() as usize, n);
                if buf[..n].starts_with(b"exit") {
                    break;
                }
            }
            Ok(_) => {
                // Spurious zero-byte read — back off briefly.
                let _ = sleep_ms(100);
            }
            Err(Errno(e)) if e == orbit_abi::errno::EAGAIN => {
                let _ = sleep_ms(100);
            }
            Err(_) => {
                logln!("tcp read failed (channel down)");
                exit(-99);
            }
        }
    }
    drop(session);

    // Close the handle explicitly so we exercise the revoke path from
    // a live process, not just from teardown. NetCh::close consumes
    // self → kernel handle is closed and SharedRegion is freed back
    // to SHARED_VA (so a future NetCh::open can reuse the same VA).
    match nc.close() {
        Ok(()) => logln!("close_handle ok!"),
        Err(Errno(e)) => {
            logln!("close_handle failed: errno={e}");
            return -(e as i32);
        }
    }

    0
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = ConsoleWriter::new();
    let _ = writeln!(w, "umode panic: {p}");
    w.flush();
    exit(isize::MIN);
}
