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

// ANSI 16-color palette. Indices match SGR 30..37 (standard) and
// 90..97 (bright). Values cribbed from the VS Code default dark
// theme — close enough to xterm that eza/ripgrep output reads
// correctly while staying legible against `DARK_GRAY` background.
pub const ANSI_BLACK: u32 = rgb(0, 0, 0);
pub const ANSI_RED: u32 = rgb(0xCD, 0x31, 0x31);
pub const ANSI_GREEN: u32 = rgb(0x0D, 0xBC, 0x79);
pub const ANSI_YELLOW: u32 = rgb(0xE5, 0xE5, 0x10);
pub const ANSI_BLUE: u32 = rgb(0x24, 0x72, 0xC8);
pub const ANSI_MAGENTA: u32 = rgb(0xBC, 0x3F, 0xBC);
pub const ANSI_CYAN: u32 = rgb(0x11, 0xA8, 0xCD);
pub const ANSI_WHITE: u32 = rgb(0xE5, 0xE5, 0xE5);
pub const ANSI_BRIGHT_BLACK: u32 = rgb(0x66, 0x66, 0x66);
pub const ANSI_BRIGHT_RED: u32 = rgb(0xF1, 0x4C, 0x4C);
pub const ANSI_BRIGHT_GREEN: u32 = rgb(0x23, 0xD1, 0x8B);
pub const ANSI_BRIGHT_YELLOW: u32 = rgb(0xF5, 0xF5, 0x43);
pub const ANSI_BRIGHT_BLUE: u32 = rgb(0x3B, 0x8E, 0xEA);
pub const ANSI_BRIGHT_MAGENTA: u32 = rgb(0xD6, 0x70, 0xD6);
pub const ANSI_BRIGHT_CYAN: u32 = rgb(0x29, 0xB8, 0xDB);
pub const ANSI_BRIGHT_WHITE: u32 = rgb(0xFF, 0xFF, 0xFF);

/// Map an SGR fg parameter (`30..37` standard, `90..97` bright) to a
/// concrete BGRA pixel. Returns `None` for any other code so the
/// parser can swallow it without changing color state.
pub const fn ansi_fg(code: u8) -> Option<u32> {
    match code {
        30 => Some(ANSI_BLACK),
        31 => Some(ANSI_RED),
        32 => Some(ANSI_GREEN),
        33 => Some(ANSI_YELLOW),
        34 => Some(ANSI_BLUE),
        35 => Some(ANSI_MAGENTA),
        36 => Some(ANSI_CYAN),
        37 => Some(ANSI_WHITE),
        90 => Some(ANSI_BRIGHT_BLACK),
        91 => Some(ANSI_BRIGHT_RED),
        92 => Some(ANSI_BRIGHT_GREEN),
        93 => Some(ANSI_BRIGHT_YELLOW),
        94 => Some(ANSI_BRIGHT_BLUE),
        95 => Some(ANSI_BRIGHT_MAGENTA),
        96 => Some(ANSI_BRIGHT_CYAN),
        97 => Some(ANSI_BRIGHT_WHITE),
        _ => None,
    }
}

/// Same shape for SGR background codes (`40..47`, `100..107`).
pub const fn ansi_bg(code: u8) -> Option<u32> {
    match code {
        40 => Some(ANSI_BLACK),
        41 => Some(ANSI_RED),
        42 => Some(ANSI_GREEN),
        43 => Some(ANSI_YELLOW),
        44 => Some(ANSI_BLUE),
        45 => Some(ANSI_MAGENTA),
        46 => Some(ANSI_CYAN),
        47 => Some(ANSI_WHITE),
        100 => Some(ANSI_BRIGHT_BLACK),
        101 => Some(ANSI_BRIGHT_RED),
        102 => Some(ANSI_BRIGHT_GREEN),
        103 => Some(ANSI_BRIGHT_YELLOW),
        104 => Some(ANSI_BRIGHT_BLUE),
        105 => Some(ANSI_BRIGHT_MAGENTA),
        106 => Some(ANSI_BRIGHT_CYAN),
        107 => Some(ANSI_BRIGHT_WHITE),
        _ => None,
    }
}

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
        Self {
            base: base_kva as *mut u32,
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Raw KDMAP pointer to the first pixel. Exposed for the
    /// surface-mode compositor's per-row blit, which writes pixel
    /// strides directly via `write_volatile`. All accesses must
    /// stay within `width * height * 4` bytes; the caller is
    /// responsible for clipping.
    pub fn base_ptr(&self) -> *mut u32 {
        self.base
    }
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
                unsafe {
                    self.put(col, row, color);
                }
            }
        }
    }

    /// Blit one 8×16 glyph at pixel `(x, y)`. Pixels whose bit is set
    /// in the glyph get `fg`, unset get `bg`. Bit 7 = leftmost pixel.
    #[inline]
    pub fn blit_glyph(&self, x: u32, y: u32, glyph: &[u8; GLYPH_H as usize], fg: u32, bg: u32) {
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
                unsafe {
                    self.put(px, py, if on { fg } else { bg });
                }
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
