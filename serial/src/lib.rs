#![no_std]

use core::{cell::UnsafeCell, fmt::{Arguments, Write}, mem::MaybeUninit, sync::atomic::{AtomicU64, Ordering}};

use ns16550a::Uart;

#[repr(align(128))]
pub struct CriticalVal {
    pub inner: AtomicU64
}

impl CriticalVal {
    pub fn acquire(&self) {
        while let Err(_) = self.inner.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {}
    }

    pub fn release(&self) {
        self.inner.store(0, Ordering::Release);
    }
}

#[repr(align(4096))]
pub struct MpUart {
    pub crit: CriticalVal,
    pub uart: UnsafeCell<MaybeUninit<Uart>>
}

impl MpUart {
    pub fn print(&self, s: Arguments) {
        self.crit.acquire();
        unsafe { self.print_no_crit(s); }
        self.crit.release();
    }

    pub unsafe fn print_no_crit(&self, s: Arguments) {
        let uart = unsafe { self.uart.get().as_mut_unchecked().assume_init_mut() };
        let _ = uart.write_fmt(s);
    }

    pub unsafe fn print_str(&self, s: &str) -> core::fmt::Result {
        self.crit.acquire();
        let uart = unsafe { self.uart.get().as_mut_unchecked().assume_init_mut() };
        let r = uart.write_str(s);
        self.crit.release();

        r
    }
}

unsafe impl Sync for MpUart {}

pub static SERIAL: MpUart = MpUart{
    crit: CriticalVal{inner:AtomicU64::new(0)},
    uart: UnsafeCell::new(MaybeUninit::zeroed())
};

pub fn print(s: Arguments) {
    SERIAL.print(s);
}

pub unsafe fn print_no_crit(s: Arguments) {
    unsafe { SERIAL.print_no_crit(s); }
}

pub unsafe fn acquire_serial() {
    SERIAL.crit.acquire();
}

pub unsafe fn release_serial() {
    SERIAL.crit.release();
}

pub unsafe fn init_serial(addr: usize) {
    let _ = unsafe { *SERIAL.uart.get().as_mut_unchecked().write(Uart::new(addr)) };
}

pub fn print_loc() -> usize {
    let l = &SERIAL as *const _ as usize;
    l
}

#[macro_export]
macro_rules! print {
    // The pattern to match: accepts a format string and any number of arguments
    ($($arg:tt)*) => {{
        unsafe {
            serial::print(format_args!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! println {
    // The pattern to match: accepts a format string and any number of arguments
    ($($arg:tt)*) => {{
        unsafe {
            serial::acquire_serial();
            serial::print_no_crit(format_args!($($arg)*));
            serial::print_no_crit(format_args!("\n"));
            serial::release_serial();
        }
    }};
}
