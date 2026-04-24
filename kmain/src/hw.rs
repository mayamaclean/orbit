//! Concrete [`orbit_core::Hardware`] impl for the RISC-V S-mode kernel.
//!
//! Zero-sized; constructed on demand at each syscall entry. Methods route
//! to the real CSRs / MMIO sites.

use mmu::sv48::VirtAddr;
use orbit_core::Hardware;

use crate::UserAccess;

pub struct RiscvHardware;

impl Hardware for RiscvHardware {
    fn now_ticks(&self) -> u64 {
        riscv::register::time::read() as u64
    }

    fn ticks_per_ms(&self) -> u64 {
        10_000
    }

    fn user_va_translates(&self, root_table_pa: u64, user_va: u64) -> bool {
        unsafe {
            let root = crate::kernel::memmap::kernel_root_from_pa(root_table_pa);
            mmu::mmap::virt_to_phys(&root, VirtAddr::new(user_va)).is_some()
        }
    }

    fn copy_from_user(&mut self, user_va: u64, dst: &mut [u8]) {
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
        serial::print!(
            "{}t USER[{}.{}]: {text}",
            riscv::register::time::read64(),
            pid,
            tid
        );
        Ok(())
    }

    fn wake_hart(&mut self, hart_id: u32) {
        crate::supervisor_wake_hart(hart_id as usize);
    }

    fn console_write_user(&mut self, pid: u16, bytes: &[u8]) -> Result<(), ()> {
        use crate::drivers::{display::Source, k_gpu};
        if !k_gpu::is_ready() {
            // Framebuffer path not live — fall back to the serial
            // back-channel so the bytes aren't silently dropped.
            if let Ok(s) = core::str::from_utf8(bytes) {
                serial::print!("{}t USER[{}]: {}", riscv::register::time::read64(), pid, s);
            }
            return Ok(());
        }
        if k_gpu::push_chunk(Source::Process(pid), bytes) {
            Ok(())
        } else {
            Err(())
        }
    }
}
