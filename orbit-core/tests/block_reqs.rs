mod common;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{Errno, EAGAIN, EFAULT, EINVAL};
use orbit_abi::layout::{
    UPROC_PRIV_BASE, UPROC_PRIV_END, UPROC_SHARED_BASE, UPROC_SHARED_END,
    USER_TEXT_BASE, USER_VA_END,
};
use orbit_core::{PendingWork, SyscallOutcome, syscall};

use common::{FakeHw, make_frame, make_thread};

/// VA in the kernel high half. The trap-frame region at USER_VA_END is
/// off-limits, but anything above is also unambiguously kernel space.
const KERNEL_VA: usize = 0xFFFF_FFC0_0000_0000;

/// Convenient priv/shared anchors for tests. UPROC_PRIV_BASE and
/// UPROC_SHARED_BASE are page-aligned so a 4 KiB request is always valid.
const PRIV_VA: usize = UPROC_PRIV_BASE as usize;
const SHARED_VA: usize = UPROC_SHARED_BASE as usize;

#[test]
fn mmap_req_shared_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = SHARED_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0x17; // V|R|W|U — X (0x8) cleared; shared+exec is rejected
    frame.regs[14] = 1;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert!(t.handle.is_some(), "thread should be parked on a handle");
    assert_eq!(hw.pending_work.len(), 1);
    match &hw.pending_work[0] {
        PendingWork::MemMap { req, pid, handle, .. } => {
            assert_eq!(req.vaddr.raw(), SHARED_VA as u64);
            assert_eq!(req.size, 4096);
            assert_eq!(req.page_permissions, 0x17);
            assert!(req.share_with_kernel);
            assert_eq!(*pid, t.pid);
            // Handle on the queue and on the thread are clones of the
            // same Arc — signaling either side wakes the other.
            handle.signal(0);
            assert!(t.handle.as_ref().unwrap().is_signaled());
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn mmap_req_priv_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = PRIV_VA;
    frame.regs[12] = 4096;
    frame.regs[14] = 0;

    let _ = syscall::mmap_req(&mut t, &frame, &mut hw);

    match &hw.pending_work[0] {
        PendingWork::MemMap { req, .. } => {
            assert_eq!(req.vaddr.raw(), PRIV_VA as u64);
            assert!(!req.share_with_kernel);
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn mmap_req_returns_eagain_when_ring_full() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = PRIV_VA;
    frame.regs[12] = 4096;
    hw.pending_work_ok = false;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() });
    assert!(t.handle.is_none(), "no parking on push failure");
    assert!(hw.pending_work.is_empty());
}

#[test]
fn nc_create_req_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = SHARED_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0;
    let bind = net_channel::BindSpec::ServerRetain { port: 7777 };
    frame.regs[14] = bind.pack();

    let outcome = syscall::nc_create_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert!(t.handle.is_some());
    match &hw.pending_work[0] {
        PendingWork::NetChannelCreation { req, pid, .. } => {
            assert_eq!(req.nc_vaddr.raw(), SHARED_VA as u64);
            assert_eq!(req.region_size, 4096);
            assert_eq!(req.nc_type, 0);
            assert_eq!(req.bind, bind);
            assert_eq!(*pid, t.pid);
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn nc_create_req_rejects_malformed_bind_spec() {
    // Mode tag 0 (or any unknown value) in the packed BindSpec must be
    // rejected at the syscall boundary so the manager never sees a
    // bogus `req.bind`.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = SHARED_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0;
    frame.regs[14] = 0;  // mode 0 = invalid

    let outcome = syscall::nc_create_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() }
    );
    assert!(t.handle.is_none());
    assert!(hw.pending_work.is_empty());
}

#[test]
fn close_req_marshals_fd_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = 7;

    let outcome = syscall::close_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert!(t.handle.is_some());
    match &hw.pending_work[0] {
        PendingWork::CloseHandle { req, pid, .. } => {
            assert_eq!(req.fd, 7);
            assert_eq!(*pid, t.pid);
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn close_req_truncates_fd_to_u32() {
    // frame.regs[11] is usize; CloseHandleReq::fd is u32. Values above u32::MAX
    // should truncate, matching the existing `as u32` cast in the shim.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = 0x1_0000_0005;

    let _ = syscall::close_req(&mut t, &frame, &mut hw);

    match &hw.pending_work[0] {
        PendingWork::CloseHandle { req, .. } => assert_eq!(req.fd, 5),
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn create_process_req_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = 0x2_2000_0000;
    frame.regs[12] = 0x4000;

    let outcome = syscall::create_process_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert!(t.handle.is_some(), "thread should be parked on a handle");
    assert_eq!(hw.pending_work.len(), 1);
    match &hw.pending_work[0] {
        PendingWork::CreateProcess { req, pid, handle, .. } => {
            assert_eq!(req.elf_vaddr.raw(), 0x2_2000_0000);
            assert_eq!(req.elf_len, 0x4000);
            assert_eq!(*pid, t.pid);
            handle.signal(7);
            assert!(t.handle.as_ref().unwrap().is_signaled());
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn create_thread_req_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.allowed_affinity = 0xF;
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = USER_TEXT_BASE as usize + 0x100; // entry inside user text
    frame.regs[12] = 0xF;                              // allowed_affinity
    frame.regs[13] = 0x4;                              // affinity (subset of allowed)

    let outcome = syscall::create_thread(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert!(t.handle.is_some(), "thread should be parked on a handle");
    assert_eq!(hw.pending_work.len(), 1);
    match &hw.pending_work[0] {
        PendingWork::CreateThread { req, pid, parent_allowed, handle } => {
            assert_eq!(req.entry.raw(), USER_TEXT_BASE + 0x100);
            assert_eq!(req.allowed_affinity, 0xF);
            assert_eq!(req.affinity, 0x4);
            assert_eq!(*pid, t.pid);
            assert_eq!(*parent_allowed, 0xF);
            handle.signal(99);
            assert!(t.handle.as_ref().unwrap().is_signaled());
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn create_thread_req_rejects_kernel_entry() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = KERNEL_VA;
    frame.regs[12] = 0xF;
    frame.regs[13] = 0x1;

    let outcome = syscall::create_thread(&mut t, &frame, &mut hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: Errno::new(EFAULT).to_ret() });
    assert!(t.handle.is_none());
    assert!(hw.pending_work.is_empty());
}

#[test]
fn create_thread_req_rejects_affinity_outside_allowed() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = USER_TEXT_BASE as usize + 0x100;
    frame.regs[12] = 0x3;       // allowed = bits 0,1
    frame.regs[13] = 0x4;       // affinity = bit 2 — outside allowed

    let outcome = syscall::create_thread(&mut t, &frame, &mut hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: Errno::new(EINVAL).to_ret() });
    assert!(t.handle.is_none());
    assert!(hw.pending_work.is_empty());
}

#[test]
fn create_thread_req_accepts_zero_sentinel_pair() {
    // Both 0 → "inherit parent's mask." Sanitization at syscall layer
    // doesn't know the parent's mask, only the manager does, so the
    // syscall-side check must accept (0, 0) and let the manager
    // resolve.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = USER_TEXT_BASE as usize + 0x100;
    frame.regs[12] = 0;
    frame.regs[13] = 0;

    let outcome = syscall::create_thread(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert_eq!(hw.pending_work.len(), 1);
}

#[test]
fn create_thread_req_returns_eagain_when_ring_full() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = USER_TEXT_BASE as usize + 0x100;
    frame.regs[12] = 0xF;
    frame.regs[13] = 0x1;
    hw.pending_work_ok = false;

    let outcome = syscall::create_thread(&mut t, &frame, &mut hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() });
    assert!(t.handle.is_none());
    assert!(hw.pending_work.is_empty());
}

#[test]
fn create_process_req_returns_eagain_when_ring_full() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = 0x2_2000_0000;
    frame.regs[12] = 0x4000;
    hw.pending_work_ok = false;

    let outcome = syscall::create_process_req(&mut t, &frame, &mut hw);

    assert_eq!(outcome, SyscallOutcome::Return { ret: Errno::new(EAGAIN).to_ret() });
    assert!(t.handle.is_none(), "no parking on push failure");
    assert!(hw.pending_work.is_empty());
}

// ---- syscall-VA sanitization ----
//
// Each blocking syscall that takes a user VA must reject kernel-half
// addresses *before* pushing to the manager work ring. Otherwise umode
// can name a kernel address (KTEXT/KDMAP/KMMIO/the per-thread TrapFrame
// region) and the manager will act on it. These tests pin that
// contract at the syscall boundary so a regression that drops the
// check is caught here, not when QEMU hands the kernel a user-installed
// PTE on top of its own text.

fn assert_rejected_no_work(outcome: SyscallOutcome, hw: &FakeHw, t: &process::Thread, ret: isize) {
    assert_eq!(outcome, SyscallOutcome::Return { ret });
    assert!(hw.pending_work.is_empty(), "no work pushed when arg is bad");
    assert!(t.handle.is_none(), "no parking on rejected request");
}

#[test]
fn mmap_req_rejects_kernel_vaddr() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = KERNEL_VA;
    frame.regs[12] = 4096;
    frame.regs[14] = 0; // private

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    // `UserVa::new` is the first gate; a kernel-half VA fails its
    // user-mappable check and surfaces as EFAULT before mmap_req's
    // pool-mismatch (EINVAL) check ever runs.
    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EFAULT).to_ret());
}

#[test]
fn mmap_req_rejects_trap_frame_region() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = USER_VA_END as usize;
    frame.regs[12] = 4096;
    frame.regs[14] = 1; // shared — checked against shared range, still rejects

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    // Same as above — the trap-frame region is outside user-mappable
    // space, so `UserVa::new` rejects with EFAULT before the
    // shared-range gate (EINVAL) runs.
    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EFAULT).to_ret());
}

#[test]
fn mmap_req_rejects_kernel_managed_user_regions() {
    // Stacks and the ELF image are kernel-installed at process
    // creation; user mmap must never aim at them. Both have
    // user-accessible PTEs, so the only thing stopping a malicious
    // mmap from shadowing them is the syscall gate.
    for vaddr in [USER_TEXT_BASE as usize, 0x1000_0000usize /* stack region */] {
        let mut t = make_thread(ThreadState::Running, SPP::User);
        let mut frame = make_frame();
        let mut hw = FakeHw::default();
        frame.regs[11] = vaddr;
        frame.regs[12] = 4096;
        frame.regs[14] = 0; // private — closest match for user buffers

        let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

        assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
    }
}

#[test]
fn mmap_req_priv_rejects_shared_vaddr() {
    // share_with_kernel=false but the VA is in the shared range —
    // the priv/shared split is what makes per-pool teardown safe, so
    // crossing it must be rejected even though the address is
    // otherwise legal.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = SHARED_VA;
    frame.regs[12] = 4096;
    frame.regs[14] = 0;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn mmap_req_shared_rejects_exec_perm() {
    // Shared frames carry a long-lived writable KDMAP alias on the
    // kernel side, so allowing X through the user alias would give a
    // W^X violation across the two views — kernel writes (e.g. net
    // thread RX) would become executable in user. The syscall layer
    // must reject before the request reaches the manager work ring.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = SHARED_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0x8; // PTE X bit
    frame.regs[14] = 1;   // shared

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn mmap_req_priv_allows_exec_perm() {
    // Private mappings have no kernel-side alias, so X is fine — the
    // shared+exec rejection must not leak into the priv path.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = PRIV_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0xA; // R|X
    frame.regs[14] = 0;   // private

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert_eq!(hw.pending_work.len(), 1);
}

#[test]
fn mmap_req_shared_rejects_priv_vaddr() {
    // Mirror of the above: share_with_kernel=true with a private VA.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = PRIV_VA;
    frame.regs[12] = 4096;
    frame.regs[14] = 1;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn mmap_req_rejects_overflowing_size() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = PRIV_VA;
    frame.regs[12] = usize::MAX;
    frame.regs[14] = 0;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn mmap_req_priv_rejects_range_crossing_priv_end() {
    // Range starts in priv but reaches into shared.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = (UPROC_PRIV_END - 4096) as usize;
    frame.regs[12] = 8192;
    frame.regs[14] = 0;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn mmap_req_shared_rejects_range_crossing_shared_end() {
    // Range starts at the top of the shared range and reaches into
    // the trap-frame region.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = (UPROC_SHARED_END - 4096) as usize;
    frame.regs[12] = 8192;
    frame.regs[14] = 1;

    let outcome = syscall::mmap_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn nc_create_req_rejects_kernel_vaddr() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = KERNEL_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0;

    let outcome = syscall::nc_create_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn nc_create_req_rejects_priv_vaddr() {
    // NetChannels live in the shared range only — a priv VA must be
    // rejected even though it's otherwise legal user space.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = PRIV_VA;
    frame.regs[12] = 4096;
    frame.regs[13] = 0;

    let outcome = syscall::nc_create_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EINVAL).to_ret());
}

#[test]
fn create_process_req_rejects_kernel_vaddr() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = KERNEL_VA;
    frame.regs[12] = 0x1000;

    let outcome = syscall::create_process_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EFAULT).to_ret());
}

#[test]
fn create_process_req_rejects_overflowing_len() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = USER_TEXT_BASE as usize;
    frame.regs[12] = usize::MAX;

    let outcome = syscall::create_process_req(&mut t, &frame, &mut hw);

    assert_rejected_no_work(outcome, &hw, &t, Errno::new(EFAULT).to_ret());
}
