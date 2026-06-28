//! `ratatui-core` `Backend` impl that paints into an orbit framebuffer
//! surface via `orbit-text`.
//!
//! Pipeline: ratatui's `Terminal::draw` produces a stream of
//! `(col, row, &Cell)` tuples; `OrbitBackend::draw` resolves each cell's
//! fg/bg through `Color::to_rgb`, calls `orbit_text::render_cell`
//! into the surface, and `flush` syncs by issuing `fb_present` to the
//! kernel compositor.
//!
//! The crate is `no_std` + `alloc` so widget libraries can pull it in
//! without the std-feature flags on `ratatui-core`. The demo binary
//! that consumes it does run under std-on-orbit (font loading via
//! `std::fs::read`, signal handling), but that's the consumer's choice.
//!
//! # Scope
//!
//! - **Static frames work; events do not yet** — `Backend` only owns
//!   output. Input wiring (key events, mouse) lands with the
//!   `read_key_event` syscall.
//! - **Whole-surface present on flush.** A future revision tracks
//!   damage in the surface and presents tight rects; v1 just blasts
//!   the full screen, which is fine while frames are O(60Hz) at most.
//! - **First-char-of-symbol rendering.** ratatui's `Cell::symbol()`
//!   returns a grapheme cluster; we render only its first char.
//!   Box-drawing, basic Latin, and most TUI glyphs are single
//!   codepoints, so this covers the common case. Multi-cp emoji and
//!   combining marks do not render correctly.
//! - **Modifiers are mostly ignored.** `REVERSED` swaps fg/bg; bold /
//!   italic / underline / blink fall through. ab_glyph + a single
//!   regular face is what's loaded; bold/italic faces would need a
//!   separate font + cache.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use ab_glyph::{Font, PxScale};
use orbit_abi::fb::FbHandle;
use orbit_text::{CellMetrics, GlyphCache, SurfaceMut, render_cell};
use ratatui_core::backend::{Backend, ClearType, WindowSize};
use ratatui_core::buffer::Cell;
use ratatui_core::layout::{Position, Size};
use ratatui_core::style::{Color, Modifier};

// `orbit_abi::user::fb_present` is gated on `target_arch = "riscv64"`
// (the `ecall` inline asm only parses there). Mirror that gate so
// host builds of this crate still compile — useful for unit tests of
// the color-conversion helpers below. The `Backend` impl pulls
// `fb_present` in unconditionally; on non-riscv64 hosts that path
// stubs out to a panic-on-call so the type still resolves.
#[cfg(target_arch = "riscv64")]
use orbit_abi::user::fb_present;

#[cfg(not(target_arch = "riscv64"))]
fn fb_present(
    _handle: FbHandle,
    _x: u32,
    _y: u32,
    _w: u32,
    _h: u32,
) -> Result<(), orbit_abi::errno::Errno> {
    unreachable!("fb_present called on non-riscv64 host build");
}

/// Errors the orbit backend produces. Kept narrow — the only thing
/// that can really fail at runtime is the `fb_present` syscall, and
/// even that only happens when the kernel-side handle is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendError {
    /// `fb_present` returned a non-zero errno.
    PresentFailed(i32),
    /// `clear_region` was called with a `ClearType` we don't support.
    /// We service `ClearType::All` only; ratatui's higher-level
    /// `Terminal::clear` only ever asks for that.
    UnsupportedClear,
}

impl core::fmt::Display for BackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PresentFailed(e) => write!(f, "fb_present failed errno={e}"),
            Self::UnsupportedClear => write!(f, "unsupported clear region"),
        }
    }
}

impl core::error::Error for BackendError {}

/// Ratatui backend that paints into an orbit framebuffer surface.
///
/// Owns: the pixel slice (as `SurfaceMut`), the font reference, the
/// pre-computed cell metrics, and a `GlyphCache`. Holds the kernel
/// `FbHandle` so `flush` can `fb_present` without the consumer
/// threading it through.
pub struct OrbitBackend<'a, F: Font> {
    surface: SurfaceMut<'a>,
    font: &'a F,
    scale: PxScale,
    metrics: CellMetrics,
    cache: GlyphCache,
    cell_cols: u16,
    cell_rows: u16,
    cursor_pos: Position,
    cursor_visible: bool,
    handle: FbHandle,
    /// Surface dims in pixels — captured up front so `flush`'s
    /// `fb_present` rect doesn't need to refetch them through
    /// `surface`.
    surface_w: u32,
    surface_h: u32,
    default_fg: (u8, u8, u8),
    default_bg: (u8, u8, u8),
}

impl<'a, F: Font> OrbitBackend<'a, F> {
    /// Construct a backend over an already-allocated framebuffer
    /// surface. The caller owns the surface lifecycle —
    /// `fb_surface_create`, this wrapper, then `fb_surface_destroy`
    /// once the backend is dropped.
    pub fn new(
        surface: SurfaceMut<'a>,
        font: &'a F,
        scale: PxScale,
        handle: FbHandle,
        default_fg: (u8, u8, u8),
        default_bg: (u8, u8, u8),
    ) -> Self {
        let metrics = CellMetrics::from_font(font, scale);
        let surface_w = surface.width();
        let surface_h = surface.height();
        let cell_cols = (surface_w / metrics.width).min(u16::MAX as u32) as u16;
        let cell_rows = (surface_h / metrics.height).min(u16::MAX as u32) as u16;
        Self {
            surface,
            font,
            scale,
            metrics,
            cache: GlyphCache::new(),
            cell_cols,
            cell_rows,
            cursor_pos: Position { x: 0, y: 0 },
            cursor_visible: true,
            handle,
            surface_w,
            surface_h,
            default_fg,
            default_bg,
        }
    }

    /// Cell-grid dimensions in `(cols, rows)`. Useful for callers that
    /// want to size their layouts without going through `Backend::size`.
    pub fn grid(&self) -> (u16, u16) {
        (self.cell_cols, self.cell_rows)
    }

    /// Cell metrics in pixels. Same data passed to `orbit-text`.
    pub fn metrics(&self) -> CellMetrics {
        self.metrics
    }

    /// Glyph cache stats — for sanity-checking growth.
    pub fn cache_entries(&self) -> usize {
        self.cache.len()
    }

    fn resolve_fg(&self, color: Color) -> (u8, u8, u8) {
        match color {
            Color::Reset => self.default_fg,
            other => color_to_rgb(other),
        }
    }

    fn resolve_bg(&self, color: Color) -> (u8, u8, u8) {
        match color {
            Color::Reset => self.default_bg,
            other => color_to_rgb(other),
        }
    }
}

impl<F: Font> Backend for OrbitBackend<'_, F> {
    type Error = BackendError;

    fn draw<'b, I>(&mut self, content: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = (u16, u16, &'b Cell)>,
    {
        for (col, row, cell) in content {
            // Skip cells past the grid edge (caller may produce them
            // when the buffer is wider than our derived grid). Cheaper
            // than letting `render_cell` fill_rect-clip them.
            if col >= self.cell_cols || row >= self.cell_rows {
                continue;
            }

            let mut fg = self.resolve_fg(cell.fg);
            let mut bg = self.resolve_bg(cell.bg);
            if cell.modifier.contains(Modifier::REVERSED) {
                core::mem::swap(&mut fg, &mut bg);
            }

            // Pull the first codepoint out of the symbol — ratatui's
            // typical TUI glyphs are single codepoints, and ab_glyph's
            // shaping doesn't speak grapheme clusters anyway. Empty
            // symbol renders as a space (fills bg, no glyph).
            let ch = cell.symbol().chars().next().unwrap_or(' ');

            render_cell(
                &mut self.surface,
                self.font,
                self.scale,
                &self.metrics,
                col as u32,
                row as u32,
                ch,
                fg,
                bg,
                &mut self.cache,
            );
        }
        Ok(())
    }

    fn hide_cursor(&mut self) -> Result<(), Self::Error> {
        self.cursor_visible = false;
        Ok(())
    }

    fn show_cursor(&mut self) -> Result<(), Self::Error> {
        self.cursor_visible = true;
        Ok(())
    }

    fn get_cursor_position(&mut self) -> Result<Position, Self::Error> {
        Ok(self.cursor_pos)
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> Result<(), Self::Error> {
        self.cursor_pos = position.into();
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        let bg = self.default_bg;
        self.surface.fill(SurfaceMut::pack_bgra(bg.0, bg.1, bg.2));
        Ok(())
    }

    fn clear_region(&mut self, clear_type: ClearType) -> Result<(), Self::Error> {
        match clear_type {
            ClearType::All => self.clear(),
            _ => Err(BackendError::UnsupportedClear),
        }
    }

    fn size(&self) -> Result<Size, Self::Error> {
        Ok(Size {
            width: self.cell_cols,
            height: self.cell_rows,
        })
    }

    fn window_size(&mut self) -> Result<WindowSize, Self::Error> {
        Ok(WindowSize {
            columns_rows: Size {
                width: self.cell_cols,
                height: self.cell_rows,
            },
            pixels: Size {
                width: (self.surface_w.min(u16::MAX as u32)) as u16,
                height: (self.surface_h.min(u16::MAX as u32)) as u16,
            },
        })
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // Optional cursor block — drawn just before present so it doesn't
        // get overwritten by subsequent cell draws within the same frame.
        if self.cursor_visible
            && (self.cursor_pos.x as u32) < self.cell_cols as u32
            && (self.cursor_pos.y as u32) < self.cell_rows as u32
        {
            let (cx, cy) = self
                .metrics
                .cell_origin(self.cursor_pos.x as u32, self.cursor_pos.y as u32);
            let fg = self.default_fg;
            self.surface.fill_rect(
                cx,
                cy,
                self.metrics.width,
                self.metrics.height,
                SurfaceMut::pack_bgra(fg.0, fg.1, fg.2),
            );
        }
        match fb_present(self.handle, 0, 0, self.surface_w, self.surface_h) {
            Ok(()) => Ok(()),
            Err(e) => Err(BackendError::PresentFailed(e.0)),
        }
    }
}

/// Convert a ratatui `Color` to BGRA8888-friendly `(r, g, b)`. `Reset`
/// is *not* handled here — the backend's per-call resolver substitutes
/// the default fg/bg before calling this.
fn color_to_rgb(color: Color) -> (u8, u8, u8) {
    match color {
        Color::Reset | Color::Black => (0x00, 0x00, 0x00),
        Color::Red => (0x80, 0x00, 0x00),
        Color::Green => (0x00, 0x80, 0x00),
        Color::Yellow => (0x80, 0x80, 0x00),
        Color::Blue => (0x00, 0x00, 0x80),
        Color::Magenta => (0x80, 0x00, 0x80),
        Color::Cyan => (0x00, 0x80, 0x80),
        Color::Gray => (0xC0, 0xC0, 0xC0),
        Color::DarkGray => (0x80, 0x80, 0x80),
        Color::LightRed => (0xFF, 0x40, 0x40),
        Color::LightGreen => (0x40, 0xFF, 0x40),
        Color::LightYellow => (0xFF, 0xFF, 0x40),
        Color::LightBlue => (0x40, 0x80, 0xFF),
        Color::LightMagenta => (0xFF, 0x40, 0xFF),
        Color::LightCyan => (0x40, 0xFF, 0xFF),
        Color::White => (0xFF, 0xFF, 0xFF),
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Indexed(i) => indexed_to_rgb(i),
    }
}

/// xterm-256 palette: 0..=15 ANSI, 16..=231 6×6×6 cube, 232..=255 grayscale.
fn indexed_to_rgb(i: u8) -> (u8, u8, u8) {
    if i < 16 {
        // Mirror the named-color palette (kept in sync with `color_to_rgb`).
        const ANSI16: [(u8, u8, u8); 16] = [
            (0x00, 0x00, 0x00),
            (0x80, 0x00, 0x00),
            (0x00, 0x80, 0x00),
            (0x80, 0x80, 0x00),
            (0x00, 0x00, 0x80),
            (0x80, 0x00, 0x80),
            (0x00, 0x80, 0x80),
            (0xC0, 0xC0, 0xC0),
            (0x80, 0x80, 0x80),
            (0xFF, 0x40, 0x40),
            (0x40, 0xFF, 0x40),
            (0xFF, 0xFF, 0x40),
            (0x40, 0x80, 0xFF),
            (0xFF, 0x40, 0xFF),
            (0x40, 0xFF, 0xFF),
            (0xFF, 0xFF, 0xFF),
        ];
        ANSI16[i as usize]
    }
    else if i < 232 {
        // 6×6×6 cube; per-channel ramp matches xterm's default
        // (0, 95, 135, 175, 215, 255).
        const RAMP: [u8; 6] = [0, 95, 135, 175, 215, 255];
        let i = i - 16;
        let r = RAMP[(i / 36) as usize];
        let g = RAMP[((i / 6) % 6) as usize];
        let b = RAMP[(i % 6) as usize];
        (r, g, b)
    }
    else {
        // 24 grayscale steps. xterm formula: 8 + (i-232)*10.
        let v = 8u16 + (i as u16 - 232) * 10;
        let v = v.min(255) as u8;
        (v, v, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_cube_corners() {
        // 16 → (0,0,0) — first cube corner
        assert_eq!(indexed_to_rgb(16), (0, 0, 0));
        // 231 → (255,255,255) — last cube corner
        assert_eq!(indexed_to_rgb(231), (255, 255, 255));
    }

    #[test]
    fn indexed_grayscale_bounds() {
        // 232 → 8 (xterm's minimum gray); 255 → 238 (8 + 23*10).
        assert_eq!(indexed_to_rgb(232), (8, 8, 8));
        assert_eq!(indexed_to_rgb(255), (238, 238, 238));
    }

    #[test]
    fn named_color_matches_indexed_alias() {
        // First 16 entries of the indexed palette must match
        // `color_to_rgb` for the corresponding named colors.
        assert_eq!(color_to_rgb(Color::Red), indexed_to_rgb(1));
        assert_eq!(color_to_rgb(Color::White), indexed_to_rgb(15));
    }
}
