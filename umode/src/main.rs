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

fn sleep_ms(ms: usize) -> isize {
    syscall_arg0(2, ms)
}

fn serial_print(ptr: usize, len: usize) -> isize {
    syscall_arg1(1, ptr, len)
}

fn mmap(addr: usize, len: usize, permissions: usize, share_with_kernel: bool) -> isize {
    let r = syscall_arg4(4096, addr, len, permissions, share_with_kernel as usize);
    if r == 0 {
        unsafe {
            core::ptr::write_bytes(addr as *mut u8, 0, len);
        }
    }
    r  
}

fn exit(code: isize) -> ! {
    syscall_arg0_noret(0, code as usize)
}

fn register_netchannel(nc: *mut NetChannel, sock_type: usize) -> isize {
    syscall_arg1(4097, nc as usize, sock_type)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    // print to serial
    const TEST: &'static str = "hello world!\n";
    let _ = serial_print(TEST.as_ptr() as usize, TEST.len());

    sleep_ms(5000);

    //let n = 0 as *const u64;
    //core::ptr::read_volatile(n);

    // map 4096 bytes to addr with read/write + share with kernel. Pick a
    // hint above USER_TEXT_BASE (0x2_2000_0000 post-higher-half) so it can't
    // clip into the stack region below.
    const AHINT: usize = 0x2_4000_0000;
    let ptr = AHINT as *mut u64;
    if mmap(AHINT, 4096, 0x6, true) == 0 {
        unsafe {
            core::ptr::write_bytes(ptr as *mut u8, 0, 4096);
        }

        const OK: &'static str = "mmapped!\n";
        let _ = serial_print(OK.as_ptr() as usize, OK.len());
    }
    else {
        const FAILED: &'static str = "failed to mmap!\n";
        let _ = serial_print(FAILED.as_ptr() as usize, FAILED.len());

        exit(-1isize);
    }

    let nc = match NetChannel::new(ptr as *mut u8, 4096) {
        Some(n) => n,
        None => {
            const NO_NC: &'static str = "bad netchannel allocation!\n";
            let _ = serial_print(NO_NC.as_ptr() as usize, NO_NC.len());

            // exit call
            exit(-2isize);
        }
    };

    // register netchannel as tcp (arg1=0) channel
    let nc_ok = register_netchannel(nc, 0) == 0;
    if !nc_ok {
        const NO_NC_MAP: &'static str = "bad netchannel share!\n";
        let _ = serial_print(NO_NC_MAP.as_ptr() as usize, NO_NC_MAP.len());

        // exit call
        exit(-2isize);
    }

    let nc = unsafe {
        nc.as_mut_unchecked()
    };

    if let Err(_) = nc.connect_tcp(u32::from_be_bytes([192,168,76,2]), 65535) {
        const NC_NO_CONNECT: &'static str = "bad failed nc tcp connect!\n";
        let _ = serial_print(NC_NO_CONNECT.as_ptr() as usize, NC_NO_CONNECT.len());

        // exit call
        exit(-2isize);
    }

    loop {
        let state = unsafe {
            nc.current_state.as_ref().state.load(Ordering::Acquire)
        };
        
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

    let mut written = false;
    loop {
        if !written && nc.writeable() > 0 {
            let wr = nc.send_tcp(|b| {
                let msg = b"Hello World!\n";
                let len = core::cmp::min(b.len(), msg.len());
                b[..len].copy_from_slice(&msg[..]);

                len
            });

            if wr.is_ok() {
                written = true;
            }
        }

        if nc.readable() > 0 {
            const TCP_CONNECTED: &'static str = "tcp readable!\n";
            let _ = serial_print(TCP_CONNECTED.as_ptr() as usize, TCP_CONNECTED.len());

            let r = nc.recv_tcp(|rx| {
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
        }
        else {
            // sleep for ms
            let _ = sleep_ms(10);
        }
    }    
    exit(0);
}

#[panic_handler]
fn panic_time(_p: &PanicInfo) -> ! {
    syscall_arg0_noret(0, isize::MIN as usize);
}
