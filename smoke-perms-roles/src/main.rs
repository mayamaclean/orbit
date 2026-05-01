#![no_std]
#![no_main]

use core::panic::PanicInfo;

use orbit_abi::denial::{DENIAL_RING_CAPACITY, DenialEvent, deny_reason};
use orbit_abi::errno::{EPERM, Errno};
use orbit_abi::layout::{UPROC_SHARED_BASE, UPROC_SHARED_END};
use orbit_abi::net::SockType;
use orbit_abi::perms::{ClassMask, CreateProcessV2Args, PermsRequest, class, role};
use orbit_abi::serialln;
use orbit_abi::user::{
    create_netch, create_process_v2, exit, getpid, pledge, query_denial_log, query_stats,
};
use orbit_rt as _;

// Stub child ELF for the role-gate test. Built by `smoke-stub-child`
// — a degenerate orbit user binary whose `main()` returns 0 so
// orbit-rt's `_start` calls `exit(0)`. Under enforcement the role
// gate EPERMs the spawn before the kernel parses these bytes, but
// the syscall layer still bound-checks the buffer and the manager
// still copies it before the gate fires (it needs `target_role` out
// of `args`), so the bytes have to be a valid ELF. Older kernels
// running in legacy shadow mode would actually load + spawn the
// child; under enforcement the bytes are just along for the ride.
//
// `--release` build because the kernel's `MAX_ELF_BYTES` is 4 MiB
// and a debug build of even a degenerate user binary lands at
// ~4.5 MiB once orbit-rt + dlmalloc are linked in. Release strips
// it to ~20 KiB.
//
// Build order: `(cd smoke-stub-child && cargo build --release)`
// before this crate compiles. The path is relative to *this*
// `main.rs`.
static STUB_ELF: &[u8] = include_bytes!(
    "../../smoke-stub-child/target/riscv64gc-unknown-none-elf/release/smoke-stub-child"
);

/// Mode the smoke is running against, deduced from the observed
/// behavior of the first denied syscall. Current kernels are
/// `Enforcement` (gate returns `-EPERM`); `Shadow` survives as a
/// branch in case the smoke is run against an older kernel that
/// logged + fell through to the handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Shadow,
    Enforcement,
}

fn detect_mode(rc_perm: isize) -> Mode {
    if rc_perm == -(EPERM as isize) {
        Mode::Enforcement
    }
    else {
        Mode::Shadow
    }
}

/// Count entries in `events` matching `(self_pid, predicate)`. Used
/// to filter the system-wide ring down to events caused by this
/// smoke binary; concurrent activity from orbit-loader / k_net etc.
/// would otherwise pollute the count.
fn count_matching(
    events: &[DenialEvent],
    self_pid: u16,
    mut pred: impl FnMut(&DenialEvent) -> bool,
) -> usize {
    events
        .iter()
        .filter(|e| match **e {
            DenialEvent::PermDeny { pid, .. } => pid == self_pid && pred(e),
            DenialEvent::RoleDeny { pid, .. } => pid == self_pid && pred(e),
        })
        .count()
}

/// Print a one-line PASS/FAIL banner over the kernel serial path
/// and exit with the matching status. Lives in `serialln!` (not
/// `logln!`/console) so the result survives even if the framebuffer
/// compositor is wedged.
fn finish(passed: bool, label: &str) -> ! {
    if passed {
        serialln!("PASS smoke-perms-roles: {label}");
        exit(0);
    }
    else {
        serialln!("FAIL smoke-perms-roles: {label}");
        exit(1);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    let self_pid = getpid();
    serialln!("smoke-perms-roles starting (pid={self_pid})");

    // ── Phase A — baseline ────────────────────────────────────────────
    // Capture the per-process counters and a snapshot of the ring
    // before we trigger anything. Both readbacks are STATS-class
    // syscalls which we still hold (only NETCH gets pledged away
    // below), so they go through cleanly.
    let stats_0 = match query_stats() {
        Ok(s) => s,
        Err(Errno(e)) => {
            serialln!("FAIL: query_stats baseline failed errno={e}");
            exit(1);
        }
    };
    let perm_0 = stats_0.perm_denials;
    let role_0 = stats_0.role_denials;

    // Buffer sized for a full snapshot (~2.4 KiB at the cap of 50).
    // Pre-fill with a sentinel `PermDeny{ pid: u16::MAX }` so an
    // unwritten slot never accidentally matches `self_pid`.
    let mut ring_a: [DenialEvent; DENIAL_RING_CAPACITY] = [DenialEvent::PermDeny {
        required_class: 0,
        perms: 0,
        time_ticks: 0,
        tid: 0,
        sysno: 0,
        source_role: 0,
        pid: u16::MAX,
    }; DENIAL_RING_CAPACITY];
    let n_ring_a = match query_denial_log(&mut ring_a) {
        Ok(n) => n,
        Err(Errno(e)) => {
            serialln!("FAIL: query_denial_log baseline failed errno={e}");
            exit(1);
        }
    };
    serialln!("baseline: perm={perm_0} role={role_0} ring_n={n_ring_a}");

    // ── Phase B — PermDeny path ───────────────────────────────────────
    // Pledge away NETCH (drop the bit from `perms`; leave
    // `allowed_perms` alone — pledge clamps each axis independently).
    // Then call create_netch and observe the gate's effect.
    let pledge_req = PermsRequest {
        perms: ClassMask::from_raw(class::raw::ALL & !class::raw::NETCH),
        allowed_perms: class::ALL,
    };
    if let Err(Errno(e)) = pledge(&pledge_req) {
        serialln!("FAIL: pledge failed errno={e}");
        exit(1);
    }

    // Pick a sentinel hint inside the user-controlled shared range.
    // The exact VA doesn't matter — the gate fires before the
    // handler's arg check; under shadow the handler may then return
    // success or some downstream error, neither of which we assert
    // on. Under enforcement the dispatch gate returns -EPERM before
    // arg processing.
    const HINT: usize = (UPROC_SHARED_BASE + 0x100_0000) as usize;
    let _ = UPROC_SHARED_END; // silence unused if the const ever moves
    let rc_perm = match create_netch(HINT, 4096, SockType::Tcp as usize, 0) {
        Ok((va, fd)) => {
            // Shadow-mode happy(ish) path: the gate fired and was
            // logged, the handler then ran and returned a valid pair.
            // Treat as a non-EPERM return → Mode::Shadow.
            serialln!("create_netch returned va={va:#x} fd={fd}");
            0
        }
        Err(Errno(e)) => {
            serialln!("create_netch returned errno={e}");
            -(e as isize)
        }
    };

    let mode = detect_mode(rc_perm);
    serialln!("detected mode: {mode:?}");

    // ── Phase C — RoleDeny path ───────────────────────────────────────
    // BOOTSTRAP's `transitions` bitset includes LOADER and SHELL only
    // — WORKER is denied, so the role gate fires inside the manager-
    // side `create_process_v2` handler.
    let v2_args = CreateProcessV2Args {
        elf_vaddr: STUB_ELF.as_ptr() as usize,
        elf_len: STUB_ELF.len(),
        allowed_affinity: 0,
        affinity: 0,
        target_role: role::WORKER,
        _pad: 0,
        request_perms: class::raw::ALL,
        request_allowed_perms: class::raw::ALL,
    };
    let rc_role = match create_process_v2(&v2_args) {
        Ok(child_pid) => {
            serialln!("create_process_v2 returned child_pid={child_pid}");
            child_pid as isize
        }
        Err(Errno(e)) => {
            serialln!("create_process_v2 returned errno={e}");
            -(e as isize)
        }
    };

    // ── Phase D — readback ────────────────────────────────────────────
    let stats_1 = match query_stats() {
        Ok(s) => s,
        Err(Errno(e)) => {
            serialln!("FAIL: query_stats post-gate failed errno={e}");
            exit(1);
        }
    };
    let perm_delta = stats_1.perm_denials - perm_0;
    let role_delta = stats_1.role_denials - role_0;
    serialln!("delta: perm={perm_delta} role={role_delta}");

    let mut ring_b: [DenialEvent; DENIAL_RING_CAPACITY] = [DenialEvent::PermDeny {
        required_class: 0,
        perms: 0,
        time_ticks: 0,
        tid: 0,
        sysno: 0,
        source_role: 0,
        pid: u16::MAX,
    }; DENIAL_RING_CAPACITY];
    let n_ring_b = match query_denial_log(&mut ring_b) {
        Ok(n) => n,
        Err(Errno(e)) => {
            serialln!("FAIL: query_denial_log post-gate failed errno={e}");
            exit(1);
        }
    };
    let events_b = &ring_b[..n_ring_b];

    // Match a PermDeny matching this run: same pid, sysno =
    // CREATE_NETCH, source_role = BOOTSTRAP, required_class = NETCH.
    let perm_hits = count_matching(events_b, self_pid, |e| match *e {
        DenialEvent::PermDeny {
            sysno,
            required_class,
            source_role,
            ..
        } => {
            sysno as usize == orbit_abi::syscall::CREATE_NETCH
                && required_class == class::raw::NETCH
                && source_role == role::BOOTSTRAP
        }
        _ => false,
    });

    // Match a RoleDeny: same pid, source_role = BOOTSTRAP,
    // target_role = WORKER, deny_reason = TRANSITION_DENIED.
    let role_hits = count_matching(events_b, self_pid, |e| match *e {
        DenialEvent::RoleDeny {
            source_role,
            target_role,
            deny_reason,
            ..
        } => {
            source_role == role::BOOTSTRAP
                && target_role == role::WORKER
                && deny_reason == deny_reason::TRANSITION_DENIED
        }
        _ => false,
    });
    serialln!("ring matches: perm_hits={perm_hits} role_hits={role_hits}");

    // ── Assertions ────────────────────────────────────────────────────
    //
    // Counter + ring assertions are identical across modes: the
    // audit log was kept under enforcement (relabelled "actual
    // denials"), so the same observability holds. The mode-
    // specific bits are the syscall return values and whether the
    // v2 spawn produced a child.

    if perm_delta != 1 {
        finish(false, "perm_denials delta != 1");
    }
    if role_delta != 1 {
        finish(false, "role_denials delta != 1");
    }
    if perm_hits != 1 {
        finish(false, "PermDeny event missing or duplicated in ring");
    }
    if role_hits != 1 {
        finish(false, "RoleDeny event missing or duplicated in ring");
    }

    match mode {
        Mode::Shadow => {
            // Historical: shadow mode logged + fell through, so the
            // v2 spawn returned a positive child pid. PR3 deleted
            // the fall-through; if anything still reports Shadow,
            // it should be because the smoke is running against an
            // older kernel.
            if rc_role <= 0 {
                finish(
                    false,
                    "shadow: create_process_v2 should have spawned a child",
                );
            }
        }
        Mode::Enforcement => {
            // Both gates EPERM. Mode detection already saw rc_perm
            // == -EPERM; assert the role-side return matches.
            if rc_role != -(EPERM as isize) {
                finish(
                    false,
                    "enforcement: create_process_v2 should have returned -EPERM",
                );
            }
        }
    }

    finish(
        true,
        match mode {
            Mode::Shadow => "shadow: counters + ring match expectations",
            Mode::Enforcement => "enforcement: both gates returned -EPERM, counters + ring match",
        },
    );
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    serialln!("smoke-perms-roles panic: {p}");
    exit(isize::MIN)
}
