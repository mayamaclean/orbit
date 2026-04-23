mod common;

use process::{ThreadBlockReason, ThreadState};
use riscv::register::sstatus::SPP;

use orbit_core::{SyscallOutcome, syscall};

use common::{make_frame, make_thread};

#[test]
fn mmap_req_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[11] = 0x1_0000_0000;
    frame.regs[12] = 4096;
    frame.regs[13] = 0x1F;
    frame.regs[14] = 1;

    let outcome = syscall::mmap_req(&mut t, &frame);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    match t.block_reason {
        ThreadBlockReason::MemMap(req) => {
            assert_eq!(req.vaddr, 0x1_0000_0000);
            assert_eq!(req.size, 4096);
            assert_eq!(req.page_permissions, 0x1F);
            assert!(req.share_with_kernel);
        }
        other => panic!("unexpected block reason: {other:?}"),
    }
}

#[test]
fn mmap_req_zero_share_flag() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[14] = 0;

    let _ = syscall::mmap_req(&mut t, &frame);

    match t.block_reason {
        ThreadBlockReason::MemMap(req) => assert!(!req.share_with_kernel),
        other => panic!("unexpected block reason: {other:?}"),
    }
}

#[test]
fn nc_create_req_marshals_args_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[11] = 0x2_4000_0000;
    frame.regs[12] = 4096;
    frame.regs[13] = 0;

    let outcome = syscall::nc_create_req(&mut t, &frame);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    match t.block_reason {
        ThreadBlockReason::NetChannelCreation(req) => {
            assert_eq!(req.nc_vaddr, 0x2_4000_0000);
            assert_eq!(req.region_size, 4096);
            assert_eq!(req.nc_type, 0);
        }
        other => panic!("unexpected block reason: {other:?}"),
    }
}

#[test]
fn close_req_marshals_fd_and_blocks() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[11] = 7;

    let outcome = syscall::close_req(&mut t, &frame);

    assert_eq!(
        outcome,
        SyscallOutcome::Yield { state: ThreadState::Blocking, ret: None }
    );
    match t.block_reason {
        ThreadBlockReason::CloseHandle(req) => assert_eq!(req.fd, 7),
        other => panic!("unexpected block reason: {other:?}"),
    }
}

#[test]
fn close_req_truncates_fd_to_u32() {
    // frame.regs[11] is usize; CloseHandleReq::fd is u32. Values above u32::MAX
    // should truncate, matching the existing `as u32` cast in the shim.
    let mut t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[11] = 0x1_0000_0005;

    let _ = syscall::close_req(&mut t, &frame);

    match t.block_reason {
        ThreadBlockReason::CloseHandle(req) => assert_eq!(req.fd, 5),
        other => panic!("unexpected block reason: {other:?}"),
    }
}
