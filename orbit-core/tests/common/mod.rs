//! Shared host-test fixtures. Included via `mod common;` in each
//! integration test. Not a binary, not a test itself.
#![allow(dead_code)]

use std::alloc::{Layout, alloc_zeroed};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use device::{HartContext, Stack, TrapFrame};
use mmu::sv48::PhysAddr;
use orbit_abi::layout::UserVa;
use process::{Thread, ThreadState};
use riscv::register::satp::Satp;
use riscv::register::sstatus::SPP;

use orbit_core::{Hardware, PendingWork};
use process::CompletionHandle;

#[cfg(miri)]
unsafe extern "Rust" {
    fn miri_static_root(ptr: *const u8);
}

#[cfg(miri)]
unsafe fn register_root(ptr: *const u8) {
    unsafe {
        miri_static_root(ptr);
    }
}

#[cfg(not(miri))]
unsafe fn register_root(_ptr: *const u8) {}

/// Build a minimal runnable `Thread` on the test heap. `frame` and `stack`
/// are zero-initialized leaked allocations — sufficient for pure-logic
/// tests that don't execute any asm. Each allocation is registered with
/// miri so the leak checker accepts it; on non-miri builds
/// `register_root` is a no-op and the leak is simply tolerated.
pub fn make_thread(state: ThreadState, mode: SPP) -> Thread {
    unsafe {
        let frame_ptr = alloc_zeroed(Layout::new::<TrapFrame>()) as *mut TrapFrame;
        let stack_ptr = alloc_zeroed(Layout::new::<Stack>()) as *mut Stack;
        register_root(frame_ptr as *const u8);
        register_root(stack_ptr as *const u8);
        let frame = &mut *frame_ptr;
        let stack = &mut *stack_ptr;
        Thread {
            pc: AtomicUsize::new(0),
            state: AtomicUsize::new(state as usize),
            wake_time: 0,
            wake_override: AtomicU64::new(0),
            last_wake_reason: AtomicU64::new(0),
            sleep_seq: AtomicU64::new(0),
            frame,
            stack,
            // Test threads don't go through the kernel-thread allocator
            // path that owns these — None matches the user-thread shape
            // used in production for non-pid-0 threads.
            kernel_stack: None,
            kernel_trap_frame: None,
            satp: Satp::from_bits(0),
            mode,
            handle: None,
            tid: 1,
            pid: 1,
            ticks: 0,
            slot: None,
            fault_info: None,
            // Test threads default to "any hart" — affinity-specific
            // behavior is asserted by tests that mutate the field
            // directly after construction.
            allowed_affinity: u64::MAX,
            affinity: AtomicU64::new(u64::MAX),
            context_switches: AtomicU64::new(0),
            cpu_ticks_total: AtomicU64::new(0),
            syscall_count: AtomicU64::new(0),
            syscall_ticks: AtomicU64::new(0),
            permissions: orbit_abi::perms::Permissions::ZERO,
            stdout_redirect: None,
            egid: 0, euid: 0, gid: 0,
            sgid: 0, suid: 0, uid: 0
        }
    }
}

/// A blank trap frame on the test heap. Callers mutate regs in place.
pub fn make_frame() -> TrapFrame {
    TrapFrame::empty()
}

/// Zero-initialized `HartContext` on the test heap, returned as a
/// `'static` reference so tests can share it across helper closures
/// without lifetime gymnastics. Total allocation is ~4 MiB
/// (two embedded `Stack` arrays at 2 MiB each); each instance is
/// leaked and registered with miri.
///
/// All atomic fields read as zero, all stack bytes read as zero,
/// `current` is null. Tests that need a non-null `current` write
/// through the atomic explicitly.
pub fn make_hart_context() -> &'static HartContext {
    unsafe {
        let ptr = alloc_zeroed(Layout::new::<HartContext>()) as *mut HartContext;
        assert!(
            !ptr.is_null(),
            "alloc_zeroed::<HartContext>() returned null"
        );
        register_root(ptr as *const u8);
        &*ptr
    }
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
    pub wakes: Vec<usize>,

    /// Accumulated `(pid, bytes)` tuples captured by
    /// `console_write_user`. Tests inspect this directly.
    pub console_writes: Vec<(u16, Vec<u8>)>,

    /// If false, `console_write_user` returns `Err(())` — exercises
    /// the `-7` ring-full path.
    pub console_ok: bool,

    /// Accumulated `PendingWork` entries pushed by syscall handlers
    /// during a test. Tests inspect this to assert what the manager
    /// would receive.
    pub pending_work: Vec<PendingWork>,

    /// If false, `push_pending_work` returns `Err(work)` — exercises
    /// the EAGAIN-on-full-ring path.
    pub pending_work_ok: bool,

    /// Pre-staged stdin payloads keyed by pid. The first
    /// `read_stdin_drain` call for a pid pops the head entry and
    /// returns its length (writes are recorded in
    /// `stdin_drain_writes`). Subsequent calls drain the next entry,
    /// or return 0 if the queue is empty. Lets tests script
    /// "first try empty, second try non-empty" race scenarios.
    pub stdin_ready: BTreeMap<u16, Vec<Vec<u8>>>,

    /// Records of `(pid, user_va, drained)` from `read_stdin_drain`.
    /// Tests can confirm the SUM-copy was attempted with the right
    /// destination.
    pub stdin_drain_writes: Vec<(u16, u64, Vec<u8>)>,

    /// Records of `(pid, handle)` from `park_stdin_reader` calls.
    pub stdin_parked: Vec<(u16, CompletionHandle)>,

    /// Records of pids passed to `unpark_stdin_reader`.
    pub stdin_unparked: Vec<u16>,

    /// If false, `park_stdin_reader` returns `false` for all calls
    /// — exercises the EBUSY path.
    pub stdin_park_ok: bool,
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
            console_writes: Vec::new(),
            console_ok: true,
            pending_work: Vec::new(),
            pending_work_ok: true,
            stdin_ready: BTreeMap::new(),
            stdin_drain_writes: Vec::new(),
            stdin_parked: Vec::new(),
            stdin_unparked: Vec::new(),
            stdin_park_ok: true,
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
    fn user_va_translates(&self, _root_table_pa: PhysAddr, _user_va: UserVa) -> bool {
        self.translates
    }
    fn copy_from_user(&mut self, user_va: UserVa, dst: &mut [u8]) {
        let bytes = self
            .user_mem
            .get(&user_va.raw())
            .expect("FakeHw::copy_from_user: no user_mem registered at user_va");
        dst.copy_from_slice(&bytes[..dst.len()]);
    }
    fn serial_write_user(&mut self, pid: u16, tid: u32, text: &str) -> Result<(), ()> {
        if self.serial_ok {
            self.user_prints.push((pid, tid, text.to_string()));
            Ok(())
        }
        else {
            Err(())
        }
    }
    fn wake_hart(&mut self, hart_id: usize) {
        self.wakes.push(hart_id);
    }
    fn console_write_user(&mut self, pid: u16, bytes: &[u8]) -> Result<(), ()> {
        if self.console_ok {
            self.console_writes.push((pid, bytes.to_vec()));
            Ok(())
        }
        else {
            Err(())
        }
    }
    fn read_stdin_drain(&mut self, pid: u16, user_va: UserVa, max_len: usize) -> usize {
        let head = self.stdin_ready.get_mut(&pid).and_then(|q| {
            if q.is_empty() {
                None
            }
            else {
                Some(q.remove(0))
            }
        });
        match head {
            Some(mut bytes) => {
                bytes.truncate(max_len);
                let n = bytes.len();
                self.stdin_drain_writes.push((pid, user_va.raw(), bytes));
                n
            }
            None => 0,
        }
    }
    fn park_stdin_reader(&mut self, pid: u16, handle: CompletionHandle) -> bool {
        if !self.stdin_park_ok {
            return false;
        }
        self.stdin_parked.push((pid, handle));
        true
    }
    fn unpark_stdin_reader(&mut self, pid: u16) -> bool {
        self.stdin_unparked.push(pid);
        // Reflect the reverse of any park: drop the most recent matching
        // park record so the test sees parked.len() == 0 after a cancel.
        if let Some(idx) = self.stdin_parked.iter().rposition(|(p, _)| *p == pid) {
            self.stdin_parked.remove(idx);
            true
        }
        else {
            false
        }
    }
    fn push_pending_work(&mut self, work: PendingWork) -> Result<(), PendingWork> {
        if self.pending_work_ok {
            self.pending_work.push(work);
            Ok(())
        }
        else {
            Err(work)
        }
    }
}
