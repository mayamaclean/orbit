//! Shared host-test fixtures. Included via `mod common;` in each
//! integration test. Not a binary, not a test itself.
#![allow(dead_code)]

use std::alloc::{Layout, alloc_zeroed};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;

use device::{Stack, TrapFrame};
use process::{Thread, ThreadBlockReason, ThreadState};
use riscv::register::satp::Satp;
use riscv::register::sstatus::SPP;

use orbit_core::Hardware;

/// Build a minimal runnable `Thread` on the test heap. `frame` and `stack`
/// are zero-initialized leaked allocations — sufficient for pure-logic
/// tests that don't execute any asm.
pub fn make_thread(state: ThreadState, mode: SPP) -> Thread {
    unsafe {
        let frame = &mut *(alloc_zeroed(Layout::new::<TrapFrame>()) as *mut TrapFrame);
        let stack = &mut *(alloc_zeroed(Layout::new::<Stack>()) as *mut Stack);
        Thread {
            pc: AtomicUsize::new(0),
            state: AtomicUsize::new(state as usize),
            wake_time: 0,
            frame,
            stack,
            satp: Satp::from_bits(0),
            mode,
            block_reason: ThreadBlockReason::NotBlocking,
            tid: 1,
            pid: 1,
            ticks: 0,
            slot: None,
            fault_info: None,
        }
    }
}

/// A blank trap frame on the test heap. Callers mutate regs in place.
pub fn make_frame() -> TrapFrame {
    TrapFrame::empty()
}

/// Configurable fake [`Hardware`] for host tests. Every knob is a plain
/// field; tests mutate directly between calls.
pub struct FakeHw {
    pub now_ticks: u64,
    pub ticks_per_ms: u64,

    /// Value returned from `user_va_translates`. Flip to exercise the
    /// bad-pointer path.
    pub translates: bool,

    /// Simulated user memory, keyed by user VA. `copy_from_user` looks the
    /// VA up and copies into the caller's dst.
    pub user_mem: BTreeMap<u64, Vec<u8>>,

    /// Accumulated `(pid, tid, text)` tuples captured by
    /// `serial_write_user`. Tests inspect this directly.
    pub user_prints: Vec<(u16, u32, String)>,

    /// If false, `serial_write_user` returns `Err(())` — exercises the
    /// `-5` return-code path.
    pub serial_ok: bool,

    /// Ordered hart ids received by `wake_hart`. Scheduler tests read
    /// this to assert which remotes got IPIs and in what order.
    pub wakes: Vec<u32>,
}

impl Default for FakeHw {
    fn default() -> Self {
        Self {
            now_ticks: 0,
            ticks_per_ms: 10_000,
            translates: true,
            user_mem: BTreeMap::new(),
            user_prints: Vec::new(),
            serial_ok: true,
            wakes: Vec::new(),
        }
    }
}

impl Hardware for FakeHw {
    fn now_ticks(&self) -> u64 {
        self.now_ticks
    }
    fn ticks_per_ms(&self) -> u64 {
        self.ticks_per_ms
    }
    fn user_va_translates(&self, _root_table_pa: u64, _user_va: u64) -> bool {
        self.translates
    }
    fn copy_from_user(&mut self, user_va: u64, dst: &mut [u8]) {
        let bytes = self
            .user_mem
            .get(&user_va)
            .expect("FakeHw::copy_from_user: no user_mem registered at user_va");
        dst.copy_from_slice(&bytes[..dst.len()]);
    }
    fn serial_write_user(&mut self, pid: u16, tid: u32, text: &str) -> Result<(), ()> {
        if self.serial_ok {
            self.user_prints.push((pid, tid, text.to_string()));
            Ok(())
        } else {
            Err(())
        }
    }
    fn wake_hart(&mut self, hart_id: u32) {
        self.wakes.push(hart_id);
    }
}
