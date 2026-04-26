#![no_std]
#![no_main]
#![feature(thread_local)]

extern crate alloc;
use orbit_rt as _;

use core::cell::Cell;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use orbit_abi::errno::{Errno, EBADF, EFAULT, EINVAL, EPERM};
use orbit_abi::net::SockType;
use orbit_abi::{logln, user::{close_handle, create_thread, exit, get_affinity, get_hart_id, set_affinity, sleep_ms, console_write, serial_print, SerialWriter}};
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

    let nc = match NetCh::open(0, SockType::Tcp) {
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

/// Report a PASS/FAIL line for a single error-path scenario. The smoke
/// script greps for "PASS: <name>" lines to validate each branch fired.
fn check(name: &str, got: isize, want: isize) {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
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
    let mut w = SerialWriter::new();
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
    let mut w = SerialWriter::new();
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
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "heap Box: {:#x} (want 0xabcd)", *b);
    w.flush();

    let mut v: Vec<u32> = Vec::new();
    for i in 0..1024 { v.push(i); }
    let sum: u64 = v.iter().map(|&x| x as u64).sum();
    let mut w = SerialWriter::new();
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
        let mut w = SerialWriter::new();
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
        let mut w = SerialWriter::new();
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    // print to serial
    logln!("hello world!");

    run_heap_smoke();

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

    // §10 layer-3: cross-hart TLB shootdown probe. Pins itself and
    // restores affinity on the way out. Must run before the TCP
    // listener flow below opens its real netchannel — they're
    // sequential users of the per-process NetCh slot.
    run_shootdown_probe();

    let _ = sleep_ms(2000);

    // Open a NetChannel (smallest valid region — capacity=0 hits the
    // floor at NC_MIN_REGION_SIZE). NetCh reserves a VA in the shared
    // range from orbit_rt::SHARED_VA, then asks the kernel to install
    // the mapping at that VA.
    let nc = match NetCh::open(0, SockType::Tcp) {
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

    if let Err(_) = nc.connect(u32::from_be_bytes([192,168,76,2]), 65535) {
        logln!("tcp connect failed!");
        exit(-2isize);
    }
    logln!("tcp connected!");

    // Send the greeting once, then drain any reply. The smoke
    // listener echoes "exit\n" to terminate this loop.
    if nc.write_all(b"Hello World!\n").is_err() {
        logln!("tcp write failed!");
        exit(-2isize);
    }

    let mut buf = [0u8; 1024];
    loop {
        match nc.read(&mut buf) {
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

    // Close the handle explicitly so we exercise the revoke path from
    // a live process, not just from teardown. NetCh::close consumes
    // self → kernel handle is closed and SharedRegion is freed back
    // to SHARED_VA (so a future NetCh::open can reuse the same VA).
    match nc.close() {
        Ok(()) => logln!("close_handle ok!"),
        Err(Errno(e)) => {
            logln!("close_handle failed: errno={e}");
            exit(-(e as isize));
        }
    }

    exit(0);
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "umode panic: {p}");
    w.flush();
    exit(isize::MIN);
}
