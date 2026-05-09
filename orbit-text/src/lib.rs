//! Glyph cache + single-line text composition for orbit framebuffer
//! surfaces.
//!
//! Cache amortizes ab_glyph rasterization across repeated draws of the
//! same `(glyph_id, scale)` pair: outline once, store the coverage
//! bytes, composite many times. Surfaces are BGRA8888 with the
//! `0xAA_RR_GG_BB` packing the kernel framebuffer driver uses.
//!
//! # Quick smoke
//!
//! ```ignore
//! use ab_glyph::{FontVec, PxScale};
//! use orbit_text::{GlyphCache, SurfaceMut, render_str};
//!
//! let font = FontVec::try_from_vec(std::fs::read("/path/to/font.ttf")?)?;
//! let mut cache = GlyphCache::new();
//! // pixels: &mut [u32] of length width * height
//! let mut surf = SurfaceMut::new(pixels, width, height).unwrap();
//! surf.fill(SurfaceMut::pack_bgra(0x10, 0x18, 0x28));
//! render_str(
//!     &mut surf, &font, PxScale::from(28.0),
//!     80.0, 140.0, (0xE6, 0xE6, 0xE6),
//!     "hello, orbit", &mut cache,
//! );
//! ```
//!
//! # Scope (and what's not in it)
//!
//! - **Single line per call.** Layout (wrap, alignment, multi-line) is
//!   the consumer's job — `render_str` returns the final caret x so a
//!   simple loop suffices for paragraphs.
//! - **Whole-pixel positioning.** Cache key elides subpixel phase;
//!   text snaps to integer pixels. At ≥ 16 px scales the visual
//!   difference is negligible. Add a 4×4 subpixel grid when small text
//!   demands it.
//! - **No LRU eviction.** ASCII × a handful of scales is < 100 KB; the
//!   cache just grows. Add bounds when a real workload pushes past a
//!   few MB.
//! - **No shaping.** Latin-1-style codepoint → glyph_id only. RTL,
//!   Indic, contextual ligatures need a shaping layer (rustybuzz or
//!   similar) on top.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use ab_glyph::{Font, GlyphId, PxScale, ScaleFont, point};
use alloc::boxed::Box;
use alloc::vec;
use hashbrown::HashMap;

/// Pixel-space integer rectangle. Derived from ab_glyph's `Rect` by
/// rounding `min` outward (`floor`) and `max` outward (`ceil`), so the
/// stored bounds always cover every pixel ab_glyph might write to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PxRect {
    /// X of the top-left corner relative to the glyph's position
    /// (typically negative for left-bearing characters like 'j').
    pub x_min: i32,
    /// Y of the top-left corner relative to the baseline. Negative for
    /// glyphs that extend above the baseline (most of them).
    pub y_min: i32,
    /// Width and height in pixels. Stored as `u16` since no realistic
    /// font + scale combination produces glyphs above ~64 K pixels per
    /// side.
    pub w: u16,
    pub h: u16,
}

impl PxRect {
    pub fn area(&self) -> usize {
        self.w as usize * self.h as usize
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

impl From<ab_glyph::Rect> for PxRect {
    fn from(r: ab_glyph::Rect) -> Self {
        // `f32::floor` / `f32::ceil` are intrinsics under std but
        // routed through libm under no_std — go through libm directly
        // so the same code compiles either way.
        let x_min = libm::floorf(r.min.x) as i32;
        let y_min = libm::floorf(r.min.y) as i32;
        let x_max = libm::ceilf(r.max.x) as i32;
        let y_max = libm::ceilf(r.max.y) as i32;
        let w = (x_max - x_min).max(0).min(u16::MAX as i32) as u16;
        let h = (y_max - y_min).max(0).min(u16::MAX as i32) as u16;
        Self { x_min, y_min, w, h }
    }
}

/// Cache key. Quantizes `scale.x` to 1/4 px so common scales (`16.0`,
/// `28.0`, `56.0`) hash exactly. Assumes uniform scaling — `PxScale`
/// constructed via `From<f32>` always satisfies that, which is the
/// common case in TUI use.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct CacheKey {
    /// `ab_glyph::GlyphId.0`
    pub glyph_id: u16,
    /// `(scale.x * 4.0) as u32`
    pub scale_q: u32,
}

impl CacheKey {
    #[inline]
    pub fn new(glyph_id: GlyphId, scale: PxScale) -> Self {
        Self {
            glyph_id: glyph_id.0,
            scale_q: (scale.x * 4.0) as u32,
        }
    }
}

/// One rasterized glyph at a fixed scale.
#[derive(Clone, Debug)]
pub struct CachedGlyph {
    /// Bounding box relative to a glyph positioned at `point(0, 0)`.
    /// At composition time the consumer adds the caret position to
    /// `(x_min, y_min)` to get the destination pixel origin.
    pub bounds: PxRect,
    /// `bounds.w * bounds.h` coverage bytes, row-major, 0..=255. The
    /// f32 coverage from ab_glyph's rasterizer is mapped via
    /// `(c.clamp(0,1) * 255.0) as u8`. Renders independent of fg
    /// color — composition picks the color when blitting.
    pub coverage: Box<[u8]>,
}

/// Glyph rasterization cache. Map keyed on `(glyph_id, scale_q)` to a
/// rasterized coverage buffer (or `None` for glyphs without an outline
/// — spaces, control chars).
pub struct GlyphCache {
    entries: HashMap<CacheKey, Option<CachedGlyph>>,
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Total entries cached (including outline-less ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Approximate bytes the cache currently holds (key + coverage
    /// only; HashMap bucket overhead not counted). Good enough for
    /// sanity-checking growth in tests.
    pub fn approx_bytes(&self) -> usize {
        let key_size = core::mem::size_of::<CacheKey>();
        let entry_size = core::mem::size_of::<Option<CachedGlyph>>();
        let mut sum = self.entries.len() * (key_size + entry_size);
        for entry in self.entries.values() {
            if let Some(g) = entry {
                sum += g.coverage.len();
            }
        }
        sum
    }

    /// Fetch (or rasterize) a glyph. Returns `None` if the glyph has
    /// no outline at this scale (whitespace, control chars). The
    /// borrow extends for the lifetime of the returned reference; once
    /// it ends, `&mut self` is free again for the next call.
    pub fn get_or_render<F: Font>(
        &mut self,
        font: &F,
        glyph_id: GlyphId,
        scale: PxScale,
    ) -> Option<&CachedGlyph> {
        let key = CacheKey::new(glyph_id, scale);
        // `Entry::or_insert_with` lets us cache the negative result
        // (`None`) too — we never re-call `outline_glyph` for the
        // same key after the first miss.
        self.entries
            .entry(key)
            .or_insert_with(|| {
                let glyph = glyph_id.with_scale_and_position(scale, point(0.0, 0.0));
                font.outline_glyph(glyph).map(|outlined| {
                    let bounds: PxRect = outlined.px_bounds().into();
                    let mut coverage: Box<[u8]> = vec![0u8; bounds.area()].into_boxed_slice();
                    let w = bounds.w as usize;
                    if w > 0 {
                        let h = bounds.h as usize;
                        outlined.draw(|gx, gy, c| {
                            let gx = gx as usize;
                            let gy = gy as usize;
                            if gx < w && gy < h {
                                // Manual clamp — `f32::clamp` is fine
                                // under core but we keep math
                                // primitive-clean for clarity.
                                let c = if c < 0.0 {
                                    0.0
                                }
                                else if c > 1.0 {
                                    1.0
                                }
                                else {
                                    c
                                };
                                coverage[gy * w + gx] = (c * 255.0) as u8;
                            }
                        });
                    }
                    CachedGlyph { bounds, coverage }
                })
            })
            .as_ref()
    }
}

/// Lifetime-checked wrapper over a BGRA8888 framebuffer slice. The
/// slice length must match `width * height` exactly; `new` enforces
/// it. All composition helpers take `&mut SurfaceMut<'_>`.
pub struct SurfaceMut<'a> {
    pixels: &'a mut [u32],
    width: u32,
    height: u32,
}

impl<'a> SurfaceMut<'a> {
    /// Construct from a writable slice + dimensions. Returns `None` if
    /// `pixels.len() != width * height`. The slice lifetime gates
    /// access — typical use is wrapping a `slice::from_raw_parts_mut`
    /// at a `fb_surface_create` user_va right before drawing.
    pub fn new(pixels: &'a mut [u32], width: u32, height: u32) -> Option<Self> {
        if pixels.len() == width as usize * height as usize {
            Some(Self {
                pixels,
                width,
                height,
            })
        }
        else {
            None
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Pack `(r, g, b)` into the BGRA8888 word the framebuffer expects.
    /// Layout `0xAA_RR_GG_BB`; little-endian bytes `BB GG RR AA`.
    /// Matches `fb::rgb` in kmain.
    #[inline]
    pub const fn pack_bgra(r: u8, g: u8, b: u8) -> u32 {
        0xFF_00_00_00 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    }

    /// Fill the surface with a solid BGRA color.
    pub fn fill(&mut self, color: u32) {
        for px in self.pixels.iter_mut() {
            *px = color;
        }
    }

    /// Fill an axis-aligned rect. Clipped to surface bounds; out-of-bounds
    /// rects are no-ops on the missing region.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: u32, h: u32, color: u32) {
        let surf_w = self.width as i32;
        let surf_h = self.height as i32;
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w as i32).min(surf_w);
        let y1 = (y + h as i32).min(surf_h);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        for py in y0..y1 {
            let row = py as usize * self.width as usize;
            for px in x0..x1 {
                self.pixels[row + px as usize] = color;
            }
        }
    }
}

/// Composite a single cached glyph onto the surface at `(dst_x, dst_y)`.
/// `dst_*` are absolute surface coordinates of the glyph bounding box's
/// top-left corner. Coverage is interpreted as straight alpha; bg is
/// read back from the surface and blended with `fg` per pixel.
///
/// `clip = (clip_x, clip_y, clip_w, clip_h)` is an optional bounding
/// rect (surface coords) outside which no pixels are written. Pass
/// `None` for "clip only to the surface bounds" (the original
/// behavior; appropriate for free-flowing `render_str` output). Pass
/// `Some(...)` to constrain the blit to a smaller region — `render_cell`
/// uses this with the cell rect so a glyph whose bounding box is taller
/// than `metrics.height` (block-drawing characters at certain scales,
/// glyphs with descenders larger than the line gap) can't bleed pixels
/// into the next cell or into the trailing surface strip below the
/// last cell row. Without this clip the bleed pixels persist until
/// `clear()` runs, which never happens during incremental ratatui
/// updates — the visible artifact is "stale row of pixels under a
/// shrinking bar gauge."
fn blit_glyph(
    surface: &mut SurfaceMut<'_>,
    glyph: &CachedGlyph,
    dst_x: i32,
    dst_y: i32,
    fg: (u8, u8, u8),
    clip: Option<(i32, i32, u32, u32)>,
) {
    if glyph.bounds.is_empty() {
        return;
    }
    let w = glyph.bounds.w as i32;
    let h = glyph.bounds.h as i32;
    let surf_w = surface.width as i32;
    let surf_h = surface.height as i32;

    // Effective clip = caller's clip ∩ surface. Caller's clip defaults
    // to "the whole surface" when None.
    let (cx0, cy0, cx1, cy1) = match clip {
        Some((cx, cy, cw, ch)) => (
            cx.max(0),
            cy.max(0),
            (cx + cw as i32).min(surf_w),
            (cy + ch as i32).min(surf_h),
        ),
        None => (0, 0, surf_w, surf_h),
    };
    if cx1 <= cx0 || cy1 <= cy0 {
        return;
    }

    // Pre-clip rows + cols so the inner loop has no per-pixel branches.
    // Translate the clip rect into glyph-local space — `gy/gx` index
    // into `glyph.coverage`, `dst_y + gy` is the surface y.
    let gy_lo = (cy0 - dst_y).max(0);
    let gy_hi = (cy1 - dst_y).min(h);
    let gx_lo = (cx0 - dst_x).max(0);
    let gx_hi = (cx1 - dst_x).min(w);
    if gy_hi <= gy_lo || gx_hi <= gx_lo {
        return;
    }

    let stride = surface.width as usize;
    let cov = &glyph.coverage;
    let cov_w = w as usize;

    for gy in gy_lo..gy_hi {
        let py = (dst_y + gy) as usize;
        let row_base = py * stride;
        let cov_row = gy as usize * cov_w;
        for gx in gx_lo..gx_hi {
            let c = cov[cov_row + gx as usize];
            if c == 0 {
                continue; // fully transparent — common for AA edges
            }
            let px = (dst_x + gx) as usize;
            let idx = row_base + px;
            let bg = surface.pixels[idx];
            let bg_r = ((bg >> 16) & 0xFF) as u16;
            let bg_g = ((bg >> 8) & 0xFF) as u16;
            let bg_b = (bg & 0xFF) as u16;
            let cov16 = c as u16;
            let inv = 255u16 - cov16;
            // Straight-alpha blend. `* cov + bg * inv` peaks at
            // 255 * 255 = 65_025, fits in u16 with room. The
            // `+ 127` rounds toward nearest instead of toward zero.
            let r = ((fg.0 as u16 * cov16 + bg_r * inv + 127) / 255) as u8;
            let g = ((fg.1 as u16 * cov16 + bg_g * inv + 127) / 255) as u8;
            let b = ((fg.2 as u16 * cov16 + bg_b * inv + 127) / 255) as u8;
            surface.pixels[idx] = SurfaceMut::pack_bgra(r, g, b);
        }
    }
}

/// Cell-grid metrics derived from a `(font, scale)` pair. Sized to the
/// `'M'` advance for cell width — correct for monospace fonts (every
/// glyph has the same advance) and the right "design intent" answer for
/// proportional fonts forced into a cell grid.
///
/// Heights round up so consecutive rows never overlap. Baseline is the
/// per-cell top-down offset where glyphs anchor: `ascent` rounded up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellMetrics {
    /// Pixel width of one cell.
    pub width: u32,
    /// Pixel height of one cell (one line's vertical advance).
    pub height: u32,
    /// Offset from the cell's top edge to the glyph baseline.
    pub baseline: u32,
}

impl CellMetrics {
    /// Compute cell metrics from a font + scale. Uses `'M'` as the
    /// width reference (any printable ASCII would work for a monospace
    /// font; `'M'` is conventional). Returns sane non-zero values even
    /// when the font lacks `'M'` — falls back to scale.x.
    pub fn from_font<F: Font>(font: &F, scale: PxScale) -> Self {
        let scaled = font.as_scaled(scale);
        let m_id = scaled.glyph_id('M');
        let raw_w = scaled.h_advance(m_id);
        let width = if raw_w > 0.0 {
            libm::ceilf(raw_w) as u32
        }
        else {
            libm::ceilf(scale.x) as u32
        }
        .max(1);
        let ascent = scaled.ascent();
        let descent = scaled.descent();
        let line_gap = scaled.line_gap();
        let height = libm::ceilf(ascent - descent + line_gap) as u32;
        let height = height.max(1);
        let baseline = libm::ceilf(ascent) as u32;
        Self {
            width,
            height,
            baseline,
        }
    }

    /// Top-left pixel coordinate of cell `(col, row)`.
    #[inline]
    pub fn cell_origin(&self, col: u32, row: u32) -> (i32, i32) {
        ((col * self.width) as i32, (row * self.height) as i32)
    }
}

/// Render one character into a cell of `(col, row)` on the grid implied
/// by `metrics`. Fills the cell with `bg`, then blits the glyph in `fg`
/// at the cell's baseline. Out-of-bounds cells are clipped by
/// `fill_rect` and `blit_glyph`.
///
/// Use this for ratatui-style cell-by-cell rendering. For proportional
/// flowing text, use `render_str` instead.
pub fn render_cell<F: Font>(
    surface: &mut SurfaceMut<'_>,
    font: &F,
    scale: PxScale,
    metrics: &CellMetrics,
    col: u32,
    row: u32,
    ch: char,
    fg: (u8, u8, u8),
    bg: (u8, u8, u8),
    cache: &mut GlyphCache,
) {
    let (cx, cy) = metrics.cell_origin(col, row);
    surface.fill_rect(
        cx,
        cy,
        metrics.width,
        metrics.height,
        SurfaceMut::pack_bgra(bg.0, bg.1, bg.2),
    );

    let scaled = font.as_scaled(scale);
    let glyph_id = scaled.glyph_id(ch);
    if let Some(cached) = cache.get_or_render(font, glyph_id, scale) {
        let baseline_int = cy + metrics.baseline as i32;
        let dst_x = cx + cached.bounds.x_min;
        let dst_y = baseline_int + cached.bounds.y_min;
        // Clip to the cell rect so a glyph whose bounds extend below
        // `metrics.height` (typical for block-drawing chars used by
        // ratatui's bar gauges) doesn't bleed pixels into adjacent
        // cells or — for the bottom row — into the surface's trailing
        // strip that no cell ever covers.
        blit_glyph(
            surface,
            cached,
            dst_x,
            dst_y,
            fg,
            Some((cx, cy, metrics.width, metrics.height)),
        );
    }
}

/// Render `text` left-to-right starting at the baseline `(x_start,
/// baseline_y)` (surface coords). Foreground color `fg = (r, g, b)`;
/// background reads through the surface (caller fills it first). Uses
/// `cache` to amortize rasterization across repeated calls.
///
/// Returns the resulting caret x — chain `render_str` calls to draw
/// multiple lines at known x positions, or use the return for
/// alignment math.
pub fn render_str<F: Font>(
    surface: &mut SurfaceMut<'_>,
    font: &F,
    scale: PxScale,
    x_start: f32,
    baseline_y: f32,
    fg: (u8, u8, u8),
    text: &str,
    cache: &mut GlyphCache,
) -> f32 {
    let scaled = font.as_scaled(scale);
    let mut caret_x = x_start;
    let mut prev_id: Option<GlyphId> = None;

    for ch in text.chars() {
        let glyph_id = scaled.glyph_id(ch);

        // Apply kerning between the previous glyph and this one. Tiny
        // for monospace fonts, meaningful for proportional ones.
        if let Some(prev) = prev_id {
            caret_x += scaled.kern(prev, glyph_id);
        }

        if let Some(cached) = cache.get_or_render(font, glyph_id, scale) {
            // Snap caret to integer pixels (no subpixel positioning
            // for v1). `floor` (via libm) ensures negative caret
            // values round toward `-∞` consistently — `as i32` would
            // truncate toward zero.
            let caret_int = libm::floorf(caret_x) as i32;
            let baseline_int = libm::floorf(baseline_y) as i32;
            let dst_x = caret_int + cached.bounds.x_min;
            let dst_y = baseline_int + cached.bounds.y_min;
            // No cell-level clip for free-flowing text — let the glyph
            // span as much as its bounds dictate, only clipped to the
            // surface itself.
            blit_glyph(surface, cached, dst_x, dst_y, fg, None);
        }

        caret_x += scaled.h_advance(glyph_id);
        prev_id = Some(glyph_id);
    }

    caret_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn px_rect_from_ab_glyph_rect_rounds_outward() {
        let r = ab_glyph::Rect {
            min: ab_glyph::point(-1.3, 4.7),
            max: ab_glyph::point(8.2, 19.1),
        };
        let p: PxRect = r.into();
        assert_eq!(p.x_min, -2);
        assert_eq!(p.y_min, 4);
        // x: -2..=9 covers [-1.3, 8.2]; w = 11
        assert_eq!(p.w, 11);
        // y: 4..=20 covers [4.7, 19.1]; h = 16
        assert_eq!(p.h, 16);
    }

    #[test]
    fn px_rect_empty_when_zero_dim() {
        let p = PxRect {
            x_min: 0,
            y_min: 0,
            w: 0,
            h: 5,
        };
        assert!(p.is_empty());
    }

    #[test]
    fn cache_key_quantization() {
        // Same int scale collides — desired.
        assert_eq!(
            CacheKey::new(GlyphId(42), PxScale::from(16.0)),
            CacheKey::new(GlyphId(42), PxScale::from(16.0)),
        );
        // 1/4 px granularity: 16.0 and 16.25 differ.
        assert_ne!(
            CacheKey::new(GlyphId(42), PxScale::from(16.0)),
            CacheKey::new(GlyphId(42), PxScale::from(16.25)),
        );
        // Different glyph_id at same scale — distinct.
        assert_ne!(
            CacheKey::new(GlyphId(1), PxScale::from(16.0)),
            CacheKey::new(GlyphId(2), PxScale::from(16.0)),
        );
    }

    #[test]
    fn surface_mut_rejects_mismatched_len() {
        let mut buf = vec![0u32; 32];
        assert!(SurfaceMut::new(&mut buf, 8, 4).is_some());
        let mut buf = vec![0u32; 31];
        assert!(SurfaceMut::new(&mut buf, 8, 4).is_none());
    }

    #[test]
    fn surface_mut_fill_rect_clips() {
        let mut buf = vec![0u32; 16]; // 4x4
        let mut surf = SurfaceMut::new(&mut buf, 4, 4).unwrap();
        let red = SurfaceMut::pack_bgra(0xFF, 0, 0);
        // Rect that extends past every edge — should fill exactly the
        // clipped intersection (the entire surface in this case).
        surf.fill_rect(-2, -2, 10, 10, red);
        assert!(buf.iter().all(|&p| p == red));

        // Out-of-bounds rect — no-op.
        let mut buf = vec![0u32; 16];
        let mut surf = SurfaceMut::new(&mut buf, 4, 4).unwrap();
        surf.fill_rect(10, 10, 4, 4, red);
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn surface_mut_pack_bgra_layout() {
        // 0xAA_RR_GG_BB packing — alpha always 0xFF.
        assert_eq!(SurfaceMut::pack_bgra(0xFF, 0, 0), 0xFFFF_0000);
        assert_eq!(SurfaceMut::pack_bgra(0, 0xFF, 0), 0xFF00_FF00);
        assert_eq!(SurfaceMut::pack_bgra(0, 0, 0xFF), 0xFF00_00FF);
        assert_eq!(SurfaceMut::pack_bgra(0x12, 0x34, 0x56), 0xFF12_3456);
    }
}
