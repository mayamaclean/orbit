//! Linear framebuffer primitives.
//!
//! Wraps the KDMAP-aliased pixel buffer the virtio-gpu scanout is
//! pointing at. All writes are volatile u32 pixels in BGRA8888 layout
//! (`format = VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM`), which on
//! little-endian RISC-V means the in-memory byte order is `BB GG RR AA`
//! — pack colors as `0xAA_RR_GG_BB` when writing.
//!
//! Glyph blit uses Terminus 8×16. Each glyph is 16 rows × 8 cols, bit
//! 7 of each row byte = leftmost pixel (MSB-first, BDF/PSF
//! convention). Indexed by Latin-1 codepoint.

use crate::drivers::fonts::terminus::TERMINUS_8X16;

pub const GLYPH_W: u32 = 8;
pub const GLYPH_H: u32 = 16;

/// Pack `(r, g, b)` as a BGRA8888 pixel with full alpha.
#[inline]
pub const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    0xFF_00_00_00 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

pub const BLACK: u32 = rgb(0, 0, 0);
pub const WHITE: u32 = rgb(0xFF, 0xFF, 0xFF);
pub const DARK_GRAY: u32 = rgb(0x20, 0x20, 0x20);
pub const CYAN: u32 = rgb(0, 0xCC, 0xCC);

/// Wrapper around a linear BGRA8888 framebuffer.
#[derive(Clone, Copy)]
pub struct FrameBuffer {
    base: *mut u32,
    width: u32,
    height: u32,
}

// SAFETY: the pointer is into KDMAP-aliased `kernel_pages` memory,
// valid for the lifetime of the kernel. Writes are volatile u32s with
// no interior aliasing concerns beyond what the caller enforces.
unsafe impl Send for FrameBuffer {}
unsafe impl Sync for FrameBuffer {}

impl FrameBuffer {
    /// # Safety
    /// `base_kva` must be a writable KDMAP VA covering at least
    /// `width * height * 4` bytes.
    pub const unsafe fn new(base_kva: u64, width: u32, height: u32) -> Self {
        Self { base: base_kva as *mut u32, width, height }
    }

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn size_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }

    #[inline]
    unsafe fn put(&self, x: u32, y: u32, color: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = y as usize * self.width as usize + x as usize;
        unsafe { self.base.add(idx).write_volatile(color) }
    }

    /// Fill the entire framebuffer with a solid color.
    pub fn fill(&self, color: u32) {
        let n = self.width as usize * self.height as usize;
        for i in 0..n {
            unsafe { self.base.add(i).write_volatile(color) }
        }
    }

    /// Fill a rectangle (clipped to framebuffer bounds).
    pub fn fill_rect(&self, x: u32, y: u32, w: u32, h: u32, color: u32) {
        let x1 = x.saturating_add(w).min(self.width);
        let y1 = y.saturating_add(h).min(self.height);
        for row in y.min(self.height)..y1 {
            for col in x.min(self.width)..x1 {
                unsafe { self.put(col, row, color); }
            }
        }
    }

    /// Blit one 8×16 glyph at pixel `(x, y)`. Pixels whose bit is set
    /// in the glyph get `fg`, unset get `bg`. Bit 7 = leftmost pixel.
    #[inline]
    pub fn blit_glyph(
        &self,
        x: u32,
        y: u32,
        glyph: &[u8; GLYPH_H as usize],
        fg: u32,
        bg: u32,
    ) {
        for (row, &byte) in glyph.iter().enumerate() {
            let py = y + row as u32;
            if py >= self.height {
                break;
            }
            for col in 0..GLYPH_W {
                let px = x + col;
                if px >= self.width {
                    break;
                }
                // MSB-first: bit 7 = col 0, bit 0 = col 7.
                let on = (byte >> (7 - col)) & 1 != 0;
                unsafe { self.put(px, py, if on { fg } else { bg }); }
            }
        }
    }

    /// Render `text` left-to-right starting at pixel `(x, y)`. Bytes
    /// outside Latin-1 aren't reachable through our `&str` path, but
    /// unmapped codepoints render as whatever's in the slot (zeros
    /// for most of U+0080..U+00FF in the Terminus ISO10646-1 subset).
    pub fn blit_text(&self, x: u32, y: u32, text: &str, fg: u32, bg: u32) {
        let mut cx = x;
        for b in text.bytes() {
            let glyph = &TERMINUS_8X16[b as usize];
            self.blit_glyph(cx, y, glyph, fg, bg);
            cx = cx.saturating_add(GLYPH_W);
            if cx >= self.width {
                break;
            }
        }
    }
}
