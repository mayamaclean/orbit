mod common;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{Errno, EAGAIN};
use orbit_core::{PendingWork, SyscallOutcome, syscall};

use common::{FakeHw, make_frame, make_thread};

#[test]
fn mmap_req_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[11] = 0x1_0000_0000;
    frame.regs[12] = 4096;
    frame.regs[13] = 0x1F;
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
            assert_eq!(req.vaddr, 0x1_0000_0000);
            assert_eq!(req.size, 4096);
            assert_eq!(req.page_permissions, 0x1F);
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
fn mmap_req_zero_share_flag() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
    frame.regs[14] = 0;

    let _ = syscall::mmap_req(&mut t, &frame, &mut hw);

    match &hw.pending_work[0] {
        PendingWork::MemMap { req, .. } => assert!(!req.share_with_kernel),
        other => panic!("unexpected pending work: {other:?}"),
    }
}

#[test]
fn mmap_req_returns_eagain_when_ring_full() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    let mut hw = FakeHw::default();
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
    frame.regs[11] = 0x2_4000_0000;
    frame.regs[12] = 4096;
    frame.regs[13] = 0;

    let outcome = syscall::nc_create_req(&mut t, &frame, &mut hw);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    assert!(t.handle.is_some());
    match &hw.pending_work[0] {
        PendingWork::NetChannelCreation { req, pid, .. } => {
            assert_eq!(req.nc_vaddr, 0x2_4000_0000);
            assert_eq!(req.region_size, 4096);
            assert_eq!(req.nc_type, 0);
            assert_eq!(*pid, t.pid);
        }
        other => panic!("unexpected pending work: {other:?}"),
    }
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
