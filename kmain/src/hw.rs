//! Concrete [`orbit_core::Hardware`] impl for the RISC-V S-mode kernel.
//!
//! Zero-sized; constructed on demand at each syscall entry. Methods route
//! to the real CSRs / MMIO sites.

use mmu::PAGE_SIZE;
use mmu::sv48::{PhysAddr, VirtAddr};
use orbit_abi::layout::UserVa;
use orbit_core::{Hardware, PendingWork};
// CompletionHandle no longer referenced — the on-thread completion
// path replaced the stdin park-handle protocol in Phase 6.

use crate::UserAccess;
use crate::kernel::MANAGER_WORK;

pub struct RiscvHardware;

impl Hardware for RiscvHardware {
    fn now_ticks(&self) -> u64 {
        riscv::register::time::read() as u64
    }

    fn ticks_per_ms(&self) -> u64 {
        10_000
    }

    fn user_va_translates(&self, root_table_pa: PhysAddr, user_va: UserVa) -> bool {
        unsafe {
            let root = crate::kernel::memmap::kernel_root_from_pa(root_table_pa);
            mmu::mmap::virt_to_phys(&root, VirtAddr::new(user_va.raw())).is_some()
        }
    }

    fn copy_from_user(&mut self, user_va: UserVa, dst: &mut [u8]) {
        // Syscall path runs under the user's satp (no satp-swap at trap
        // entry), so the user VA is directly addressable while SUM is set.
        // Caller has already validated the range via `user_va_translates`.
        let guard = UserAccess::enter();
        unsafe {
            let slice = guard.slice(user_va, dst.len());
            dst.copy_from_slice(slice);
        }
        drop(guard);
    }

    fn serial_write_user(&mut self, pid: u16, tid: u32, text: &str) -> Result<(), ()> {
        // `{t}t USER[pid.tid]: {text}` matches the tracing-subscriber
        // layout used by kernel info!/debug! lines — keeps user output
        // visually aligned with kernel logs in the smoke/debug streams.
        crate::serialln!(
            "{}t USER[{}.{}]: {text}",
            riscv::register::time::read64(),
            pid,
            tid
        );

        Ok(())
    }

    fn wake_hart(&mut self, hart_id: usize) {
        crate::supervisor_wake_hart(hart_id);
    }

    fn console_write_user(&mut self, dest_pid: u16, bytes: &[u8]) -> Result<(), ()> {
        use crate::drivers::{display::Source, k_gpu};
        if !k_gpu::is_ready() {
            // Framebuffer path not live — fall back to the serial
            // back-channel so the bytes aren't silently dropped.
            if let Ok(s) = core::str::from_utf8(bytes) {
                crate::serialln!(
                    "{}t USER[{}]: {}",
                    riscv::register::time::read64(),
                    dest_pid,
                    s
                );
            }
            return Ok(());
        }
        if k_gpu::push_chunk(Source::Process(dest_pid), bytes) {
            Ok(())
        }
        else {
            Err(())
        }
    }

    fn push_pending_work(&mut self, work: PendingWork) -> Result<(), PendingWork> {
        // thingbuf push_ref returns the slot to write into; we move
        // `work` in via the `*slot = ...` assignment. Drop releases the
        // slot back to the queue for the manager to pop.
        match MANAGER_WORK.push_ref() {
            Ok(mut slot) => {
                *slot = work;
                Ok(())
            }
            Err(_) => Err(work),
        }
    }

    fn read_stdin_drain(&mut self, pid: u16, user_va: UserVa, max_len: usize) -> usize {
        let Some(stdin) = crate::kernel::stdin::get(pid)
        else {
            return 0;
        };
        // Drain into a kernel-side scratch slice first so the SUM
        // window only covers the copy step (and so a partial read
        // doesn't leave the user buffer half-written if the pid
        // disappears mid-drain — though that can't happen with the
        // current single-thread-per-process model).
        let mut scratch = [0u8; PAGE_SIZE];
        let n = stdin.try_drain(&mut scratch[..max_len]);
        if n == 0 {
            return 0;
        }
        let guard = UserAccess::enter();
        unsafe {
            let dst = guard.slice_mut(user_va, n);
            dst.copy_from_slice(&scratch[..n]);
        }
        drop(guard);
        n
    }

    fn park_stdin_reader(&mut self, pid: u16, tid: u32) -> bool {
        let Some(stdin) = crate::kernel::stdin::get(pid)
        else {
            return false;
        };
        stdin.park(tid)
    }

    fn unpark_stdin_reader(&mut self, pid: u16) -> bool {
        let Some(stdin) = crate::kernel::stdin::get(pid)
        else {
            return false;
        };
        stdin.unpark().is_some()
    }

    fn read_key_events_drain(&mut self, pid: u16, user_va: UserVa, max_count: usize) -> usize {
        use orbit_abi::input::KeyEvent;
        let Some(events) = crate::kernel::key_events::get(pid)
        else {
            return 0;
        };
        // Drain into a kernel-side scratch buffer first so the SUM
        // window only covers the copy step. Cap at the syscall's
        // page-bounded count (read_key_event already validated this).
        const KEY_EVENT_SIZE: usize = core::mem::size_of::<KeyEvent>();
        const MAX_EVENTS: usize = PAGE_SIZE / KEY_EVENT_SIZE;
        let cap = max_count.min(MAX_EVENTS);
        // SAFETY: KeyEvent is `#[repr(C)]` with all-u32 fields; zeroed
        // is a valid initialized value (decodes to KeyCode::Char with
        // codepoint 0, mods empty, kind Press — but we never read
        // these slots before overwriting in try_drain).
        let mut scratch: [KeyEvent; MAX_EVENTS] = unsafe { core::mem::zeroed() };
        let n = events.try_drain(&mut scratch[..cap]);
        if n == 0 {
            return 0;
        }
        let byte_len = n * KEY_EVENT_SIZE;
        let guard = UserAccess::enter();
        unsafe {
            let dst = guard.slice_mut(user_va, byte_len);
            // Reinterpret the events slice as bytes. `KeyEvent` is
            // `#[repr(C)]` and contains only `u32`s — no padding.
            let src_bytes = core::slice::from_raw_parts(scratch.as_ptr() as *const u8, byte_len);
            dst.copy_from_slice(src_bytes);
        }
        drop(guard);
        n
    }

    fn set_key_event_parker(&mut self, pid: u16, tid: u32) -> process::key_events::ParkOutcome {
        let Some(events) = crate::kernel::key_events::get(pid)
        else {
            return process::key_events::ParkOutcome::Busy;
        };
        events.set_parker(tid)
    }

    fn clear_key_event_parker_if(&mut self, pid: u16, tid: u32) -> bool {
        let Some(events) = crate::kernel::key_events::get(pid)
        else {
            return false;
        };
        events.clear_parker_if(tid)
    }
}
