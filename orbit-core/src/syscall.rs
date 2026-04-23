//! Pure syscall handler bodies. Each function takes the resolved thread +
//! either extracted arguments or the raw [`TrapFrame`] + an effect handle,
//! mutates thread-local state, and returns a [`SyscallOutcome`] for the
//! kmain shim to apply.

use device::TrapFrame;
use process::{CloseHandleReq, MemMapReq, NetChannelCreationReq, Thread, ThreadBlockReason, ThreadState};

use crate::{Hardware, PAGE_SIZE, SyscallOutcome};

/// Cap on `sleep_ms(ms)` arguments. Anything at or above this returns -2
/// without touching thread state.
pub const MAX_SLEEP_MS: usize = 60 * 60 * 1000;

/// `sleep_ms(ms)` — block the caller for `ms` milliseconds.
///
/// Stores the absolute wake tick on the thread and tells the shim to yield
/// into `Suspended`. The wake loop in the manager compares
/// `now_ticks() >= thread.wake_time` to decide when to mark the thread
/// runnable again.
pub fn ms_sleep<H: Hardware>(thread: &mut Thread, ms: usize, hw: &H) -> SyscallOutcome {
    if ms >= MAX_SLEEP_MS {
        return SyscallOutcome::Return { ret: -2 };
    }

    let wake_time = (hw.now_ticks() as usize)
        .wrapping_add(ms.wrapping_mul(hw.ticks_per_ms() as usize));
    thread.wake_time = wake_time;

    SyscallOutcome::Yield {
        state: ThreadState::Suspended,
        ret: Some(0),
    }
}

/// `mmap(vaddr, size, perms, share_with_kernel)` — enter Blocking with a
/// [`MemMapReq`] that the manager consumes on its next pass.
///
/// Syscall returns are written by the manager at unblock time (the shim
/// leaves `regs[10]` alone — see [`SyscallOutcome::Yield`] docs).
pub fn mmap_req(thread: &mut Thread, frame: &TrapFrame) -> SyscallOutcome {
    let req = MemMapReq {
        vaddr: frame.regs[11],
        size: frame.regs[12],
        page_permissions: frame.regs[13] as u64,
        share_with_kernel: frame.regs[14] > 0,
    };
    thread.block_reason = ThreadBlockReason::MemMap(req);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `create_netch(vaddr_hint, region_size, nc_type)` — enter Blocking with a
/// [`NetChannelCreationReq`] for the manager.
pub fn nc_create_req(thread: &mut Thread, frame: &TrapFrame) -> SyscallOutcome {
    let req = NetChannelCreationReq {
        nc_vaddr: frame.regs[11],
        region_size: frame.regs[12],
        nc_type: frame.regs[13],
    };
    thread.block_reason = ThreadBlockReason::NetChannelCreation(req);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `close_handle(fd)` — enter Blocking with a [`CloseHandleReq`] for the
/// manager's handle-registry teardown path.
pub fn close_req(thread: &mut Thread, frame: &TrapFrame) -> SyscallOutcome {
    let req = CloseHandleReq {
        fd: frame.regs[11] as u32,
    };
    thread.block_reason = ThreadBlockReason::CloseHandle(req);
    SyscallOutcome::Yield {
        state: ThreadState::Blocking,
        ret: None,
    }
}

/// `serial_print(user_va, len)` — copy a UTF-8 string out of user memory
/// and write it to the kernel serial console. Yields `Ready` after, so the
/// scheduler decides whether this thread keeps running.
///
/// Return codes:
/// - `0`  — bytes written
/// - `-2` — user VA doesn't translate (bad pointer)
/// - `-3` — `len` exceeds a page
/// - `-4` — bytes aren't valid UTF-8
/// - `-5` — serial write failed
pub fn serial_print<H: Hardware>(
    thread: &Thread,
    frame: &TrapFrame,
    hw: &mut H,
) -> SyscallOutcome {
    let user_va = frame.regs[11] as u64;
    let len = frame.regs[12];

    if len > PAGE_SIZE {
        return ready(-3);
    }

    if !hw.user_va_translates(thread.root_table_addr() as u64, user_va) {
        return ready(-2);
    }

    let mut buf = [0u8; PAGE_SIZE];
    hw.copy_from_user(user_va, &mut buf[..len]);

    let s = match core::str::from_utf8(&buf[..len]) {
        Ok(s) => s,
        Err(_) => return ready(-4),
    };

    match hw.serial_write_user(thread.pid, thread.tid, s) {
        Ok(()) => ready(0),
        Err(()) => ready(-5),
    }
}

#[inline]
fn ready(ret: isize) -> SyscallOutcome {
    SyscallOutcome::Yield {
        state: ThreadState::Ready,
        ret: Some(ret),
    }
}
