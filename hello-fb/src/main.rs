//! `/bin/hello-fb` — drawing-surface smoke test for milestone M1.
//!
//! Loop:
//! 1. `fb_query` to read the active framebuffer dims.
//! 2. `fb_surface_create(w, h, Bgra8888)` for a full-screen surface.
//! 3. Fill the mapped pixels with a diagonal RGB gradient — proves the
//!    user-writable mapping landed at the returned VA and that the
//!    pixels reach the compositor.
//! 4. `fb_present(handle, 0, 0, w, h)` to ask the compositor to blit.
//! 5. Sleep 5 s so a human can eyeball the result.
//! 6. Bump the gradient phase, present a partial damage rect to verify
//!    partial-update path.
//! 7. `fb_surface_destroy(handle)` and exit.

#![no_std]
#![no_main]

extern crate alloc;
use orbit_rt as _;

use core::panic::PanicInfo;

use orbit_abi::fb::{FbFormat, FbInfo};
use orbit_abi::{
    serialln,
    user::{
        SerialWriter, exit, fb_present, fb_query, fb_surface_create, fb_surface_destroy, sleep_ms,
    },
};

fn pack_bgra(r: u8, g: u8, b: u8) -> u32 {
    // FrameBuffer expects 0xAA_RR_GG_BB packing (see kmain fb::rgb).
    0xFF_00_00_00 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Diagonal RGB gradient: red across X, green across Y, blue from a
/// constant phase. Cheap, visually obvious, and covers every pixel of
/// the surface so a missed write stands out.
unsafe fn fill_gradient(base: *mut u32, width: u32, height: u32, phase: u8) {
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = phase;
            let idx = y as usize * width as usize + x as usize;
            unsafe {
                base.add(idx).write_volatile(pack_bgra(r, g, b));
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    serialln!("hello-fb: starting");

    let mut info = FbInfo::default();
    if let Err(e) = fb_query(&mut info) {
        serialln!("hello-fb: fb_query failed errno={}", e.0);
        return 1;
    }
    serialln!(
        "hello-fb: display {}x{} format={}",
        info.width,
        info.height,
        info.format
    );

    let format = match FbFormat::from_u32(info.format) {
        Some(f) => f,
        None => {
            serialln!("hello-fb: unknown format {}", info.format);
            return 2;
        }
    };

    let (handle, user_va) = match fb_surface_create(info.width, info.height, format) {
        Ok(p) => p,
        Err(e) => {
            serialln!("hello-fb: fb_surface_create failed errno={}", e.0);
            return 3;
        }
    };
    serialln!(
        "hello-fb: surface handle={} user_va=0x{:X}",
        handle.raw(),
        user_va
    );

    // Initial full-screen gradient.
    let base = user_va as *mut u32;
    unsafe {
        fill_gradient(base, info.width, info.height, 0x40);
    }
    if let Err(e) = fb_present(handle, 0, 0, info.width, info.height) {
        serialln!("hello-fb: fb_present full failed errno={}", e.0);
        return 4;
    }
    serialln!("hello-fb: presented full {}x{}", info.width, info.height);

    let _ = sleep_ms(5_000);

    // Partial update — overwrite the center quarter with phase 0xC0,
    // present only that rect. Tests the damage-rect path.
    let qx = info.width / 4;
    let qy = info.height / 4;
    let qw = info.width / 2;
    let qh = info.height / 2;
    unsafe {
        for y in qy..qy + qh {
            for x in qx..qx + qw {
                let idx = y as usize * info.width as usize + x as usize;
                let r = (x * 255 / info.width) as u8;
                let g = (y * 255 / info.height) as u8;
                base.add(idx).write_volatile(pack_bgra(r, g, 0xC0));
            }
        }
    }
    if let Err(e) = fb_present(handle, qx, qy, qw, qh) {
        serialln!("hello-fb: fb_present partial failed errno={}", e.0);
        return 5;
    }
    serialln!("hello-fb: presented partial rect ({qx},{qy} {qw}x{qh})");

    let _ = sleep_ms(30_000);

    if let Err(e) = fb_surface_destroy(handle) {
        serialln!("hello-fb: destroy failed errno={}", e.0);
        return 6;
    }
    serialln!("hello-fb: surface destroyed; exiting");
    0
}

#[panic_handler]
fn panic_handler(p: &PanicInfo) -> ! {
    use core::fmt::Write;
    let mut w = SerialWriter::new();
    let _ = writeln!(w, "hello-fb panic: {p}");
    w.flush();
    exit(isize::MIN);
}
