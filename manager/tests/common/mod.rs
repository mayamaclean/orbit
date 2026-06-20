//! Host-test fixtures for the manager scheduling tests. Included via
//! `mod common;`. Builds a fresh `Ready` thread on the test heap.
#![allow(dead_code)]

use std::alloc::{Layout, alloc_zeroed};

use device::{Stack, TrapFrame};
use process::{Thread, ThreadInit};
use riscv::register::satp::Satp;
use riscv::register::sstatus::SPP;

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

/// Leak a fresh thread to a stable `*mut Thread`. `Thread::new` starts the
/// thread `Ready` with a valid (zeroed) frame — exactly the shape the ready
/// queue / dispatch path consume, so no state stamping is needed. All three
/// leaked allocations (frame, stack, thread box) are registered with miri.
pub fn leak_ready_thread(tid: u32) -> *mut Thread {
    unsafe {
        let frame = alloc_zeroed(Layout::new::<TrapFrame>()) as *mut TrapFrame;
        let stack = alloc_zeroed(Layout::new::<Stack>()) as *mut Stack;
        register_root(frame as *const u8);
        register_root(stack as *const u8);
        let t = Thread::new(ThreadInit {
            entrypoint: 0,
            satp: Satp::from_bits(0),
            mode: SPP::User,
            tid,
            pid: 1,
            frame: &mut *frame,
            stack: &mut *stack,
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
        let ptr = Box::into_raw(Box::new(t));
        register_root(ptr as *const u8);
        ptr
    }
}
