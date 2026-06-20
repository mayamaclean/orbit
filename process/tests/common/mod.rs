//! Shared host-test fixtures for the capability tests. Included via
//! `mod common;`. Not a test itself.
//!
//! Mirrors `orbit-core/tests/common`'s thread builder, trimmed to what the
//! cap-safety tests need. Requires the `test-helpers` feature (for
//! `Thread::transition_to_unchecked`) — `./test` runs with `--all-features`.
#![allow(dead_code)]

use std::alloc::{Layout, alloc_zeroed};

use device::{Stack, TrapFrame};
use process::{SchedGuard, Thread, ThreadInit, ThreadState};
use riscv::register::satp::Satp;
use riscv::register::sstatus::SPP;

/// Spin-acquire the process-global scheduler guard, then run `f` under it.
/// `SchedGuard::try_with` is non-blocking by design (the kernel never spins
/// on the lock); cargo runs a binary's test fns in parallel threads, which
/// momentarily contend the single global lock, so the fixture retries until
/// it wins. `f` is `Fn` (not `FnOnce`) so it can be re-offered across
/// retries — `try_with` only runs it on the acquiring attempt.
pub fn with_guard<R>(f: impl Fn(&SchedGuard) -> R) -> R {
    loop {
        if let Some(r) = SchedGuard::try_with(&f) {
            return r;
        }
        std::hint::spin_loop();
    }
}

// Register leaked allocations with miri's leak checker so `cargo miri test`
// doesn't flag the deliberately-leaked fixture frames/stacks. No-op off-miri.
#[cfg(miri)]
unsafe extern "Rust" {
    fn miri_static_root(ptr: *const u8);
}
#[cfg(miri)]
unsafe fn register_root(ptr: *const u8) {
    unsafe { miri_static_root(ptr) }
}
#[cfg(not(miri))]
unsafe fn register_root(_ptr: *const u8) {}

/// Build a minimal `Thread` on the test heap, then stamp `state` via the
/// unchecked fixture setter (the checked `transition_to` would reject the
/// non-production edges fixtures need, e.g. `Ready → Suspended`). The
/// `frame`/`stack` are zero-initialized leaked allocations — sufficient for
/// pure-logic / cap tests that never execute asm.
pub fn make_thread(state: ThreadState) -> Thread {
    unsafe {
        let frame_ptr = alloc_zeroed(Layout::new::<TrapFrame>()) as *mut TrapFrame;
        let stack_ptr = alloc_zeroed(Layout::new::<Stack>()) as *mut Stack;
        register_root(frame_ptr as *const u8);
        register_root(stack_ptr as *const u8);
        let frame = &mut *frame_ptr;
        let stack = &mut *stack_ptr;
        let t = Thread::new(ThreadInit {
            entrypoint: 0,
            satp: Satp::from_bits(0),
            mode: SPP::User,
            tid: 1,
            pid: 1,
            frame,
            stack,
            kernel_stack: None,
            kernel_trap_frame: None,
            slot: None,
            allowed_affinity: u64::MAX,
            affinity: u64::MAX,
            permissions: orbit_abi::perms::Permissions::ZERO,
            uid: 0,
            euid: 0,
            suid: 0,
            gid: 0,
            egid: 0,
            sgid: 0,
            stdout_redirect: None,
        });
        t.transition_to_unchecked(state);
        t
    }
}

/// Leak a `Thread` to a stable `*mut Thread`, the shape the caps consume.
/// Tests never free it (the cap lifecycle ends at the leaked allocation);
/// the leak is registered with miri so the leak checker accepts it.
pub fn leak_thread(state: ThreadState) -> *mut Thread {
    let ptr = Box::into_raw(Box::new(make_thread(state)));
    unsafe { register_root(ptr as *const u8) };
    ptr
}
