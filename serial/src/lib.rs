#![no_std]

use core::{
    cell::UnsafeCell,
    fmt::{Arguments, Write},
    mem::MaybeUninit,
};

use ns16550a::Uart;
use spin::{MutexGuard, mutex::Mutex};

#[repr(align(4096))]
pub struct MpUart {
    pub(crate) uart: UnsafeCell<spin::Mutex<MaybeUninit<Uart>>>,
}

impl MpUart {
    pub fn print(&self, s: Arguments) {
        unsafe {
            let _ = self
                .uart
                .get()
                .as_ref_unchecked()
                .lock()
                .assume_init_mut()
                .write_fmt(s);
        }
    }

    /// **Panic-handler-only.** Bypass the UART mutex entirely and write
    /// straight to the hardware. Forms a `&mut Mutex<...>` while the
    /// lock may be held by another hart (notably k_serial, which holds
    /// it for life), so concurrent calls + concurrent ring-drains can
    /// produce aliasing `&mut`s — UB at the language level. We accept
    /// that for the panic path because the alternative is the panic
    /// being silently swallowed when k_serial owns the lock; everything
    /// else (`ktrace::emit`, the `serial_print` syscall, direct
    /// `println!`) MUST go through the locked `print` / `acquire_serial`
    /// path or the ring instead.
    ///
    /// Output may interleave with k_serial's drain bytes — cosmetic
    /// only; the panic message still reaches the UART.
    pub unsafe fn print_lock_bypass(&self, s: Arguments) {
        let _ = unsafe {
            self.uart
                .get()
                .as_mut_unchecked()
                .get_mut()
                .assume_init_mut()
                .write_fmt(s)
        };
    }

    pub unsafe fn print_str(&self, s: &str) -> core::fmt::Result {
        let int_on = riscv::register::sstatus::read().sie();
        unsafe {
            riscv::register::sstatus::clear_sie();
        }
        let r = unsafe {
            self.uart
                .get()
                .as_ref_unchecked()
                .lock()
                .assume_init_mut()
                .write_str(s)
        };
        if int_on {
            unsafe {
                riscv::register::sstatus::set_sie();
            }
        }
        r
    }
}

unsafe impl Sync for MpUart {}

pub static SERIAL: MpUart = MpUart {
    uart: UnsafeCell::new(Mutex::new(MaybeUninit::zeroed())),
};

pub fn print(s: Arguments) {
    SERIAL.print(s);
}

/// Panic-handler-only free-function shim around
/// [`MpUart::print_lock_bypass`]. See that method's doc for the safety
/// rationale and why this isn't a general fallback.
pub unsafe fn print_lock_bypass(s: Arguments) {
    unsafe {
        SERIAL.print_lock_bypass(s);
    }
}

pub unsafe fn init_serial(addr: usize) {
    let _ = unsafe {
        *SERIAL
            .uart
            .get()
            .as_mut_unchecked()
            .get_mut()
            .write(Uart::new(addr))
    };
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

pub unsafe fn acquire_serial() -> MutexGuard<'static, MaybeUninit<Uart>> {
    unsafe {
        let int_on = riscv::register::sstatus::read().sie();
        riscv::register::sstatus::clear_sie();
        let g = SERIAL.uart.get().as_ref_unchecked().lock();
        if int_on {
            riscv::register::sstatus::set_sie();
        }
        g
    }
}

#[macro_export]
macro_rules! println {
    // The pattern to match: accepts a format string and any number of arguments
    ($($arg:tt)*) => {{
        unsafe {
            use core::fmt::Write;
            let mut g = serial::acquire_serial();
            let _ = g.assume_init_mut().write_fmt(format_args!($($arg)*));
            let _ = g.assume_init_mut().write_str("\n");
            core::mem::drop(g);
        }
    }};
}

#[macro_export]
/// **Panic-handler-only.** Writes directly to the UART without taking
/// the spin::Mutex — the only way to get bytes out when k_serial owns
/// the lock for life. See [`MpUart::print_lock_bypass`] for the safety
/// caveats. Don't use this from non-panic code; it's UB-adjacent on
/// concurrent calls.
macro_rules! panic_println {
    ($($arg:tt)*) => {{
        unsafe {
            serial::print_lock_bypass(format_args!($($arg)*));
            serial::print_lock_bypass(format_args!("\n"));
        }
    }};
}
