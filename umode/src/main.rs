#![no_std]
#![no_main]

use core::{arch::asm, panic::PanicInfo, sync::atomic::Ordering};

use net_channel::NetChannel;

extern "C" fn syscall_arg0_noret(code: usize, arg0: usize) -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            options(noreturn)
        );
    }
}

extern "C" fn syscall_arg0(code: usize, arg0: usize) -> isize {
    let mut r = 0isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            lateout("a0") r
        );
    }
    r
}

extern "C" fn syscall_arg1(code: usize, arg0: usize, arg1: usize) -> isize {
    let mut r = 0isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            lateout("a0") r
        );
    }
    r
}

#[allow(dead_code)]
extern "C" fn syscall_arg4(code: usize, arg0: usize, arg1: usize, arg2: usize, arg3: usize) -> isize {
    let mut r = 0isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            in("a4") arg3,
            lateout("a0") r
        );
    }
    r
}

/// Two-return variant: primary in a0, secondary in a1. Used by
/// `create_netch` to hand back both the mapped VA (a0) and the Fd the
/// kernel assigned (a1) in one trap — avoids needing a user
/// out-pointer, which the kernel would have to resolve through KDMAP
/// or a transient page window.
extern "C" fn syscall_arg4_ret2(code: usize, arg0: usize, arg1: usize, arg2: usize, arg3: usize, fd: &mut u32) -> isize {
    let mut r0 = 0isize;
    let mut r1 = 0isize;
    unsafe {
        asm!(
            "ecall",
            in("a0") code,
            in("a1") arg0,
            in("a2") arg1,
            in("a3") arg2,
            in("a4") arg3,
            lateout("a0") r0,
            lateout("a1") r1,
        );
    }
    
    if r0 == 0 {
        *fd = r1 as u32;
    }
    r0
}

fn sleep_ms(ms: usize) -> isize {
    syscall_arg0(2, ms)
}

fn serial_print(ptr: usize, len: usize) -> isize {
    syscall_arg1(1, ptr, len)
}

#[allow(dead_code)]
fn mmap(addr: usize, len: usize, permissions: usize, share_with_kernel: bool) -> isize {
    syscall_arg4(4096, addr, len, permissions, share_with_kernel as usize)
}

fn exit(code: isize) -> ! {
    syscall_arg0_noret(0, code as usize)
}

/// Ask the kernel to allocate a NetChannel region of `region_size` bytes,
/// map it at `vaddr_hint`, initialize its headers, and register it with
/// the net thread as a socket of `sock_type`. On success returns
/// `(user_va, fd)` — the VA the region landed at and the Fd the kernel
/// assigned (pass this to `close_handle` to tear down the channel).
/// On failure returns a negative errno.
fn create_netch(vaddr_hint: usize, region_size: usize, sock_type: usize) -> Result<(usize, u32), ()> {
    let mut fd = 0;
    let va = syscall_arg4_ret2(4097, vaddr_hint, region_size, sock_type, 0, &mut fd);
    if va < 0 { Err(()) } else { Ok((va as usize, fd as u32)) }
}

/// Release a handle previously returned by `create_netch`. Kernel
/// revokes the user mapping (future accesses to the old VA fault)
/// before dropping its strong ref. Returns 0 on success, negative on
/// error.
fn close_handle(fd: u32) -> isize {
    syscall_arg0(4098, fd as usize)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    // print to serial
    const TEST: &'static str = "hello world!\n";
    let _ = serial_print(TEST.as_ptr() as usize, TEST.len());

    sleep_ms(5000);

    // Ask the kernel to create a NetChannel. The hint is above
    // USER_TEXT_BASE (0x2_2000_0000) so it can't clip into the stack
    // region below. The kernel returns the actual VA it picked — today
    // that's always the hint, but readers should not rely on it.
    const AHINT: usize = 0x2_4000_0000;
    const NC_REGION_SIZE: usize = 4096;
    let (nc_vaddr, nc_fd) = match create_netch(AHINT, NC_REGION_SIZE, 0) {
        Ok(v) => v,
        Err(_) => {
            const NO_NC: &'static str = "failed to create netchannel!\n";
            let _ = serial_print(NO_NC.as_ptr() as usize, NO_NC.len());
            exit(-2isize);
        }
    };

    const OK: &'static str = "netchannel created!\n";
    let _ = serial_print(OK.as_ptr() as usize, OK.len());

    let nc = unsafe { &*(nc_vaddr as *const NetChannel) };

    if let Err(_) = nc.connect_tcp(u32::from_be_bytes([192,168,76,2]), 65535) {
        const NC_NO_CONNECT: &'static str = "bad failed nc tcp connect!\n";
        let _ = serial_print(NC_NO_CONNECT.as_ptr() as usize, NC_NO_CONNECT.len());

        // exit call
        exit(-2isize);
    }

    loop {
        let state = nc.current_state().state.load(Ordering::Acquire);

        if state > 0 {
            const TCP_CONNECTED: &'static str = "tcp connected!\n";
            let _ = serial_print(TCP_CONNECTED.as_ptr() as usize, TCP_CONNECTED.len());
            break
        }
        else if state < 0 {
            const TCP_FAILURE: &'static str = "tcp connect failed!\n";
            let _ = serial_print(TCP_FAILURE.as_ptr() as usize, TCP_FAILURE.len());
            break
        }
        else if state == 0 {
            // sleep for ms
            let _ = sleep_ms(10);
        }
    }

    //exit(0);

    let mut written = false;
    let mut br = false;
    loop {
        if !written && nc.writeable() > 0 {
            let wr = nc.send_tcp(|b| {
                let msg = b"Hello World!\n";
                b.copy_from_slice(msg)
            });

            if let Ok(n) = wr {
                if n > 0 {
                    written = true;
                }
            }
        }

        if nc.readable() > 0 {
            let r = nc.recv_tcp(|rx| {
                if rx.starts_with(b"exit") {
                    br = true;
                }
                serial_print(rx.as_ptr() as usize, rx.len());
                rx.len()
            });

            match r {
                Err(e) if e > -4 => {
                    // exit call
                    exit(e);
                }
                _ => {}
            }

            if br {
                // Close the handle before exit so we exercise the
                // revoke path from a live process, not just from
                // teardown. After this returns, `nc` is invalid — the
                // user mapping has been torn down.
                let cr = close_handle(nc_fd);
                if cr != 0 {
                    const CLOSE_FAIL: &'static str = "close_handle failed!\n";
                    let _ = serial_print(CLOSE_FAIL.as_ptr() as usize, CLOSE_FAIL.len());
                    exit(cr);
                }

                const CLOSE_OK: &'static str = "close_handle ok!\n";
                let _ = serial_print(CLOSE_OK.as_ptr() as usize, CLOSE_OK.len());

                let _ = unsafe {
                    core::ptr::read_volatile(nc as *const _ as *const u8);
                };
            }
        }
        else {
            // sleep for ms
            let _ = sleep_ms(100);
        }

        let state = nc.current_state().state.load(Ordering::Acquire);

        if state <= 0 {
            const TCP_CONN_FAILURE: &'static str = "tcp connection failed!\n";
            let _ = serial_print(TCP_CONN_FAILURE.as_ptr() as usize, TCP_CONN_FAILURE.len());
            break
        }
    }    
    exit(-99);
}

/// Tiny `fmt::Write` that buffers into a 256-byte stack array and
/// flushes via the serial-print syscall. Enough to get a panic message
/// out before exit without any allocator.
struct SerialWriter {
    buf: [u8; 256],
    len: usize,
}

impl SerialWriter {
    const fn new() -> Self { Self { buf: [0u8; 256], len: 0 } }
    fn flush(&mut self) {
        if self.len > 0 {
            let _ = serial_print(self.buf.as_ptr() as usize, self.len);
            self.len = 0;
        }
    }
}

impl core::fmt::Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len >= self.buf.len() {
                self.flush();
            }
            self.buf[self.len] = b;
            self.len += 1;
        }
        Ok(())
    }
}

#[panic_handler]
fn panic_time(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "umode panic: {p}");
    w.flush();
    syscall_arg0_noret(0, isize::MIN as usize);
}
