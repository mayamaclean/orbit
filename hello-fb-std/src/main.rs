//! `/bin/hello-fb-std` — anti-aliased TTF text via orbit-text glyph cache.
//!
//! Std-on-orbit binary. Loads Liberation Mono off `/usr/share/fonts/`,
//! constructs an `orbit-text` `GlyphCache`, and draws several lines at
//! varying scales onto a full-screen BGRA8888 surface. The cache
//! amortizes ab_glyph rasterization across the four lines (and across
//! repeats of the same character — most ASCII glyphs land in two of
//! them).

use ab_glyph::{FontVec, PxScale};
use orbit_abi::fb::{FbFormat, FbInfo};
use orbit_abi::user::{fb_present, fb_query, fb_surface_create, fb_surface_destroy};
use orbit_text::{GlyphCache, SurfaceMut, render_str};

const FONT_PATH: &str = "/usr/share/fonts/LiberationMono-Regular.ttf";

fn main() {
    println!("hello-fb-std: starting");

    let font_bytes = match std::fs::read(FONT_PATH) {
        Ok(b) => b,
        Err(e) => {
            println!("hello-fb-std: read({FONT_PATH}) failed: {e}");
            return;
        }
    };
    println!("hello-fb-std: loaded font, {} bytes", font_bytes.len());

    let font = match FontVec::try_from_vec(font_bytes) {
        Ok(f) => f,
        Err(e) => {
            println!("hello-fb-std: FontVec parse failed: {e}");
            return;
        }
    };

    let mut info = FbInfo::default();
    if let Err(e) = fb_query(&mut info) {
        println!("hello-fb-std: fb_query failed errno={}", e.0);
        return;
    }
    println!(
        "hello-fb-std: display {}x{} format={}",
        info.width, info.height, info.format
    );

    let format = match FbFormat::from_u32(info.format) {
        Some(f) => f,
        None => {
            println!("hello-fb-std: unknown format {}", info.format);
            return;
        }
    };

    let (handle, user_va) = match fb_surface_create(info.width, info.height, format) {
        Ok(p) => p,
        Err(e) => {
            println!("hello-fb-std: fb_surface_create failed errno={}", e.0);
            return;
        }
    };
    println!(
        "hello-fb-std: surface handle={} user_va=0x{:X}",
        handle.raw(),
        user_va
    );

    // Wrap the kernel-mapped pixel range in a SurfaceMut for the
    // lifetime of the draw. The unsafe is bounded to the slice
    // construction; orbit-text only sees a `&mut [u32]`.
    let pixel_count = info.width as usize * info.height as usize;
    let pixels: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(user_va as *mut u32, pixel_count) };
    let mut surf = match SurfaceMut::new(pixels, info.width, info.height) {
        Some(s) => s,
        None => {
            println!("hello-fb-std: SurfaceMut::new mismatched len");
            return;
        }
    };

    // Dark navy background; off-white text. Subjective but high
    // enough contrast that the AA edges show clearly.
    let bg = SurfaceMut::pack_bgra(0x10, 0x18, 0x28);
    let fg = (0xE6u8, 0xE6, 0xE6);
    let accent = (0x6Au8, 0xC8, 0xFF);
    surf.fill(bg);

    let mut cache = GlyphCache::new();

    // Three lines at three sizes — proves scaled rendering and
    // cross-scale cache reuse (the same glyph at 18 and 28 px is two
    // distinct entries).
    let xs = PxScale::from(12.0);
    let small = PxScale::from(18.0);
    let medium = PxScale::from(28.0);
    let large = PxScale::from(56.0);

    let _ = render_str(
        &mut surf,
        &font,
        large,
        80.0,
        140.0,
        accent,
        "hello, orbit",
        &mut cache,
    );

    let _ = render_str(
        &mut surf,
        &font,
        medium,
        80.0,
        220.0,
        fg,
        "Liberation Mono via ab_glyph + orbit-text",
        &mut cache,
    );

    let _ = render_str(
        &mut surf,
        &font,
        small,
        80.0,
        260.0,
        fg,
        "anti-aliased TTF on the kernel framebuffer",
        &mut cache,
    );

    let _ = render_str(
        &mut surf,
        &font,
        xs,
        80.0,
        300.0,
        fg,
        "another line of text down here",
        &mut cache,
    );

    println!(
        "hello-fb-std: cached {} glyph entries (~{} bytes)",
        cache.len(),
        cache.approx_bytes()
    );

    if let Err(e) = fb_present(handle, 0, 0, info.width, info.height) {
        println!("hello-fb-std: fb_present failed errno={}", e.0);
        return;
    }
    println!("hello-fb-std: presented");

    std::thread::sleep(std::time::Duration::from_secs(30));

    if let Err(e) = fb_surface_destroy(handle) {
        println!("hello-fb-std: destroy failed errno={}", e.0);
    }
    println!("hello-fb-std: done");
}
