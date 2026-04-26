mod common;

use process::ThreadState;
use riscv::register::sstatus::SPP;

use orbit_abi::errno::{Errno, EFAULT, EINVAL, EIO};
use orbit_core::{PAGE_SIZE, SyscallOutcome, syscall};

use common::{FakeHw, make_frame, make_thread};

const UVA: u64 = 0x2_0000_1000;

fn frame_with(len: usize) -> device::TrapFrame {
    let mut f = make_frame();
    f.regs[11] = UVA as usize;
    f.regs[12] = len;
    f
}

fn ready(ret: isize) -> SyscallOutcome {
    SyscallOutcome::Yield { state: ThreadState::Ready, ret: Some(ret) }
}

#[test]
fn prints_ascii_and_yields_ready_zero() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(13);
    let mut hw = FakeHw::default();
    hw.user_mem.insert(UVA, b"hello world!\n".to_vec());

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(0));
    assert_eq!(hw.user_prints, vec![(t.pid, t.tid, "hello world!\n".to_string())]);
}

#[test]
fn rejects_len_above_page() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(PAGE_SIZE + 1);
    let mut hw = FakeHw::default();

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EINVAL).to_ret()));
    assert!(hw.user_prints.is_empty(), "nothing should be written on reject");
}

#[test]
fn accepts_len_exactly_page() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(PAGE_SIZE);
    let mut hw = FakeHw::default();
    hw.user_mem.insert(UVA, vec![b'a'; PAGE_SIZE]);

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(0));
    assert_eq!(hw.user_prints.len(), 1);
    assert_eq!(hw.user_prints[0].2.len(), PAGE_SIZE);
}

#[test]
fn bad_user_va_returns_efault() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(5);
    let mut hw = FakeHw { translates: false, ..Default::default() };

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EFAULT).to_ret()));
    assert!(hw.user_prints.is_empty());
}

#[test]
fn non_utf8_returns_einval() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(4);
    let mut hw = FakeHw::default();
    // 0xFF is never a valid start byte in UTF-8.
    hw.user_mem.insert(UVA, vec![0xFF, 0xFE, 0xFD, 0xFC]);

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EINVAL).to_ret()));
    assert!(hw.user_prints.is_empty(), "no partial write on utf8 failure");
}

#[test]
fn valid_prefix_then_invalid_byte_returns_einval() {
    // Catches a regression where `from_utf8` gets swapped for
    // `from_utf8_unchecked` plus a length check — the prefix would
    // pass any cheap "first byte ASCII" inspection.
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(5);
    let mut hw = FakeHw::default();
    hw.user_mem.insert(UVA, vec![b'h', b'i', b'!', b'\n', 0xFF]);

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EINVAL).to_ret()));
    assert!(hw.user_prints.is_empty(), "no partial write on utf8 failure");
}

#[test]
fn serial_failure_returns_eio() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(3);
    let mut hw = FakeHw { serial_ok: false, ..Default::default() };
    hw.user_mem.insert(UVA, b"abc".to_vec());

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EIO).to_ret()));
}

#[test]
fn empty_len_still_succeeds() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(0);
    let mut hw = FakeHw::default();
    hw.user_mem.insert(UVA, Vec::new());

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(0));
    assert_eq!(hw.user_prints, vec![(t.pid, t.tid, String::new())]);
}

#[test]
fn check_order_len_before_translate() {
    // If both are bad, -EINVAL (length check) wins over -EFAULT
    // (translate check). Defense-in-depth ordering — bound the range
    // before walking page tables.
    let t = make_thread(ThreadState::Running, SPP::User);
    let frame = frame_with(PAGE_SIZE + 100);
    let mut hw = FakeHw { translates: false, ..Default::default() };

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EINVAL).to_ret()));
}

#[test]
fn rejects_kernel_vaddr_without_translating() {
    // VA in the kernel high half. Even if the user satp shadows kernel
    // mappings, the syscall must reject the address structurally — and
    // it must do so without consulting the page table (defense in
    // depth + cheaper rejection).
    let t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[11] = 0xFFFF_FFC0_0000_0000;
    frame.regs[12] = 4;
    let mut hw = FakeHw::default();

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EFAULT).to_ret()));
    assert!(hw.user_prints.is_empty());
}

#[test]
fn rejects_null_guard_vaddr() {
    let t = make_thread(ThreadState::Running, SPP::User);
    let mut frame = make_frame();
    frame.regs[11] = 0x0;
    frame.regs[12] = 4;
    let mut hw = FakeHw::default();

    let outcome = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(outcome, ready(Errno::new(EFAULT).to_ret()));
    assert!(hw.user_prints.is_empty());
}

#[test]
fn carries_pid_tid_to_serial() {
    let mut t = make_thread(ThreadState::Running, SPP::User);
    t.pid = 7;
    t.tid = 42;
    let frame = frame_with(5);
    let mut hw = FakeHw::default();
    hw.user_mem.insert(UVA, b"abcde".to_vec());

    let _ = syscall::serial_print(&t, &frame, &mut hw);

    assert_eq!(hw.user_prints, vec![(7, 42, "abcde".to_string())]);
}
