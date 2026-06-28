//! Scrollback + active-source selector + framebuffer compositor.
//!
//! Single-consumer: only `k_gpu` touches this state post-init. All
//! producers (`console_write` syscalls from any hart, the kernel's
//! own tracing shim) funnel through a `thingbuf` MPSC ring that
//! `k_gpu` drains and then calls [`Display::append`] on each pop.
//!
//! The model is i3/sway fullscreen: one source fills the screen at a
//! time; a keystroke cycles through live sources. Each source has its
//! own `Scrollback` so output that arrives while off-screen is kept
//! until the user flips to it.
//!
//! `Scrollback::append` runs an inline ANSI SGR parser that
//! recognizes the standard 16-color foreground/background codes and
//! drops every other escape sequence (bold, italic, underline,
//! 256-color, truecolor, cursor moves, …) silently — those map to
//! attributes the bitmap blit doesn't render.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;
use tracing::info;

use crate::drivers::fb::{self, FrameBuffer, GLYPH_H, GLYPH_W};
use crate::drivers::k_gpu::PresentArgs;
use orbit_abi::fb::FbFormat;

/// Axis-aligned rect in framebuffer pixel coordinates. Used as the
/// repaint damage hint: the compositor unions all rects from the drain
/// pass and uses the bounding box for the virtio-gpu `transfer_to_host_2d`
/// + `flush` round trip, instead of always uploading the whole screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub fn empty() -> Self {
        Self {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// Bounding-box union with `other`. Empty rects act as identity.
    pub fn union(self, other: Rect) -> Rect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.w).max(other.x + other.w);
        let y1 = (self.y + self.h).max(other.y + other.h);
        Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    /// Clip to the framebuffer's bounds.
    pub fn clip(self, fb_w: u32, fb_h: u32) -> Rect {
        let x = self.x.min(fb_w);
        let y = self.y.min(fb_h);
        let w = (self.x + self.w).min(fb_w).saturating_sub(x);
        let h = (self.y + self.h).min(fb_h).saturating_sub(y);
        Rect { x, y, w, h }
    }
}

/// Per-source surface state. Carries the kernel-side KDMAP alias so the
/// compositor can blit pixels straight from the surface into the
/// framebuffer without touching the user PT.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceState {
    pub kdmap_kva: u64,
    pub width: u32,
    pub height: u32,
    pub format: FbFormat,
}

/// Max bytes a single scrollback line retains. Long kernel debug
/// output past this gets truncated until the next `\n`.
pub const MAX_LINE_LEN: usize = 256;
/// Depth of the per-source history ring.
pub const SCROLLBACK_LINES: usize = 500;

/// Output source. `Kernel` is seeded at boot; `Process(pid)` entries
/// are added/removed by `Orbit::create_process` / `dealloc_process`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Source {
    Kernel,
    Process(u16),
}

/// Foreground + background color for a run of characters. Stored on
/// each chunk so the painter doesn't need to re-walk the parser
/// state. Defaults to "white on dark gray" to match what the prior
/// non-colored scrollback rendered.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Style {
    pub fg: u32,
    pub bg: u32,
}

impl Style {
    pub const DEFAULT: Style = Style {
        fg: fb::WHITE,
        bg: fb::DARK_GRAY,
    };
}

impl Default for Style {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One run of same-styled text. Emitted by the parser and consumed
/// by the painter; lines are `Vec<StyledChunk>` so a single render
/// row can have multiple foreground colors.
#[derive(Clone)]
pub struct StyledChunk {
    pub style: Style,
    pub text: String,
}

/// A line that has been wholly received (i.e. terminated by `\n`)
/// or is still being assembled (`pending`). `len` tracks the total
/// number of visible chars across all chunks so the parser can
/// enforce [`MAX_LINE_LEN`] without summing.
#[derive(Clone, Default)]
pub struct StyledLine {
    pub chunks: Vec<StyledChunk>,
    pub len: usize,
}

impl StyledLine {
    fn clear(&mut self) {
        self.chunks.clear();
        self.len = 0;
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append `ch` under `style`. Coalesces with the trailing chunk
    /// if its style matches — keeps allocation pressure flat for the
    /// common "one color per line" case.
    fn push_char(&mut self, style: Style, ch: char) {
        if let Some(last) = self.chunks.last_mut() {
            if last.style == style {
                last.text.push(ch);
                self.len += 1;
                return;
            }
        }
        let mut text = String::new();
        text.push(ch);
        self.chunks.push(StyledChunk { style, text });
        self.len += 1;
    }

    fn pop_char(&mut self) {
        while let Some(last) = self.chunks.last_mut() {
            if last.text.pop().is_some() {
                self.len -= 1;
                if last.text.is_empty() {
                    self.chunks.pop();
                }
                return;
            }
            self.chunks.pop();
        }
    }
}

/// Inline ANSI-escape parser state. Only `ESC [ params ... m` (SGR)
/// is interpreted; every other CSI command (cursor moves, screen
/// clears, OSC, …) is consumed silently.
#[derive(Clone)]
enum AnsiState {
    Normal,
    Esc,
    /// Inside a CSI sequence. `params` accumulates digits + `;` from
    /// the parameter bytes; the final byte (anything in the range
    /// 0x40..0x7E) terminates the sequence.
    Csi {
        params: String,
    },
}

/// Per-source line history + an in-progress partial line + style
/// state carried across `append` calls.
pub struct Scrollback {
    lines: VecDeque<StyledLine>,
    pending: StyledLine,
    style: Style,
    ansi: AnsiState,
}

impl Default for Scrollback {
    fn default() -> Self {
        Self::new()
    }
}

impl Scrollback {
    pub fn new() -> Self {
        Self {
            lines: VecDeque::with_capacity(SCROLLBACK_LINES),
            pending: StyledLine::default(),
            style: Style::DEFAULT,
            ansi: AnsiState::Normal,
        }
    }

    /// `true` if no completed lines and no in-progress line. Used by
    /// `Display::remove_source` to decide whether a soon-to-exit
    /// surface-mode source has any text content worth preserving as
    /// a tombstone — empty scrollbacks for surface-only panes get
    /// dropped entirely on exit.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty() && self.pending.is_empty()
    }

    /// Append a chunk of bytes. Splits on `\n`; drops non-printable
    /// bytes (bitmap font is ASCII-only); truncates any line beyond
    /// [`MAX_LINE_LEN`] until the next `\n` resets. Two control
    /// bytes are special-cased so the in-tree console can drive
    /// interactive editing through this same path: `\x08` (BS) pops
    /// the last char from the pending line, and `\x0c` (FF) clears
    /// both completed history and the pending line for this source.
    /// `\x1b` opens an ANSI escape sequence (see [`AnsiState`]).
    pub fn append(&mut self, chunk: &[u8]) {
        for &b in chunk {
            // ANSI parsing has priority — a byte mid-sequence isn't
            // a printable. Extract the trailing-`m`-applies-SGR work
            // out of the borrow scope so `apply_sgr` can take a
            // fresh `&mut self`.
            let mut sgr_params: Option<String> = None;
            match &mut self.ansi {
                AnsiState::Normal => {
                    if b == 0x1b {
                        self.ansi = AnsiState::Esc;
                        continue;
                    }
                }
                AnsiState::Esc => {
                    self.ansi = if b == b'[' {
                        AnsiState::Csi {
                            params: String::new(),
                        }
                    }
                    else {
                        // Two-byte escape (e.g. `ESC c`) or junk —
                        // swallow the next byte and reset.
                        AnsiState::Normal
                    };
                    continue;
                }
                AnsiState::Csi { params } => {
                    // Parameter bytes: digits, `;`, `:`, and the
                    // private-marker prefixes `<` / `=` / `>` / `?`.
                    if (0x30..=0x3f).contains(&b) {
                        params.push(b as char);
                        continue;
                    }
                    // Intermediate bytes (rarely emitted by CLIs).
                    if (0x20..=0x2f).contains(&b) {
                        continue;
                    }
                    // Final byte — terminator. Only `m` (SGR) carries
                    // anything we render; everything else is dropped.
                    if (0x40..=0x7e).contains(&b) {
                        if b == b'm' {
                            sgr_params = Some(core::mem::take(params));
                        }
                        self.ansi = AnsiState::Normal;
                        if let Some(p) = sgr_params {
                            self.apply_sgr(&p);
                        }
                        continue;
                    }
                    // Unknown byte — abandon.
                    self.ansi = AnsiState::Normal;
                    continue;
                }
            }

            if b == b'\n' {
                let completed = core::mem::take(&mut self.pending);
                if self.lines.len() >= SCROLLBACK_LINES {
                    self.lines.pop_front();
                }
                self.lines.push_back(completed);
                continue;
            }
            if b == 0x08 {
                self.pending.pop_char();
                continue;
            }
            if b == 0x0c {
                self.lines.clear();
                self.pending.clear();
                continue;
            }
            if self.pending.len >= MAX_LINE_LEN {
                continue;
            }
            let ch = if b.is_ascii_graphic() || b == b' ' || b == b'\t' {
                if b == b'\t' { ' ' } else { b as char }
            }
            else {
                continue;
            };
            self.pending.push_char(self.style, ch);
        }
    }

    /// Apply one CSI-m (SGR) parameter list. Accepts any combination
    /// of `;`-separated decimal codes; recognized:
    /// - `0` (or empty)         → reset to default
    /// - `30..37` / `90..97`    → standard / bright fg
    /// - `40..47` / `100..107`  → standard / bright bg
    /// - `39` / `49`            → reset fg / bg
    ///
    /// Anything else is swallowed silently (bold, italic, underline,
    /// 256-color, truecolor, …). 256-color and truecolor sub-
    /// sequences (`38;5;n`, `38;2;r;g;b`) consume their following
    /// codes as a side effect of the linear walk so they don't
    /// accidentally match a later `30..37` and recolor.
    fn apply_sgr(&mut self, params: &str) {
        // Split on `;` and parse to u8. Empty params → `[0]` per spec.
        let mut iter = params.split(';');
        // Manual parse: an empty string (e.g. plain `ESC [ m`) means
        // a single `0` parameter.
        let mut codes: Vec<u8> = Vec::new();
        if params.is_empty() {
            codes.push(0);
        }
        else {
            while let Some(p) = iter.next() {
                let n: u32 = p.parse().unwrap_or(0);
                codes.push((n & 0xff) as u8);
            }
        }

        let mut i = 0;
        while i < codes.len() {
            let c = codes[i];
            match c {
                0 => self.style = Style::DEFAULT,
                30..=37 | 90..=97 => {
                    if let Some(fg) = fb::ansi_fg(c) {
                        self.style.fg = fg;
                    }
                }
                40..=47 | 100..=107 => {
                    if let Some(bg) = fb::ansi_bg(c) {
                        self.style.bg = bg;
                    }
                }
                39 => self.style.fg = Style::DEFAULT.fg,
                49 => self.style.bg = Style::DEFAULT.bg,
                // 256-color / truecolor: skip the indicator (5 or 2)
                // and its data so a trailing 30..37 doesn't get
                // mis-applied.
                38 | 48 => {
                    if let Some(&kind) = codes.get(i + 1) {
                        match kind {
                            5 => i += 2, // 38;5;n  → consume n
                            2 => i += 4, // 38;2;r;g;b → consume r,g,b
                            _ => i += 1,
                        }
                    }
                }
                _ => {} // bold/italic/underline/etc — swallow.
            }
            i += 1;
        }
    }

    /// Yield up to `rows` visible lines: at most `rows-1` completed
    /// lines plus the `pending` line if it's non-empty, in
    /// oldest-first order suitable for top-to-bottom blit.
    pub fn view(&self, rows: usize) -> impl Iterator<Item = &StyledLine> {
        let pending_slot = if self.pending.is_empty() { 0 } else { 1 };
        let line_rows = rows.saturating_sub(pending_slot);
        let skip = self.lines.len().saturating_sub(line_rows);
        self.lines
            .iter()
            .skip(skip)
            .chain(Some(&self.pending).filter(|s| !s.is_empty()))
    }
}

pub struct ProcScrollback {
    pub(super) scrollback: Scrollback,
    pub(super) title: String,
}

impl ProcScrollback {
    pub fn new(src: Source) -> Self {
        Self {
            scrollback: Scrollback::new(),
            title: match src {
                Source::Kernel => " kernel ".into(),
                Source::Process(pid) => alloc::format!(" pid {pid} "),
            },
        }
    }
}

/// The i3-style display: one framebuffer, one currently-active source,
/// a scrollback per source. Owns no virtio state — rendering writes
/// pixels into `fb`; the caller is responsible for the
/// `TRANSFER_TO_HOST_2D` + `FLUSH` pair afterward.
pub struct Display {
    pub fb: FrameBuffer,
    pub active: Source,
    scrollbacks: BTreeMap<Source, ProcScrollback>,
    /// Per-source pixel surfaces. A source with an entry here is in
    /// "surface mode" — repaint blits from `kdmap_kva` instead of
    /// re-rendering the scrollback. Mutated only by `present_surface`
    /// (insert/update with damage) and `remove_source` (drop).
    /// Coexistence with the same source's scrollback is allowed: an
    /// app that prints once via `console_write` and then takes over
    /// with `fb_present` keeps the scrollback in case the user wants
    /// to dismiss the surface and read back the prints. Today
    /// repaint always prefers the surface when present.
    surfaces: BTreeMap<Source, SurfaceState>,
    /// `true` if the next `repaint` needs to redraw something. Set by
    /// any state mutation (`append`, `present_surface`, `cycle_active`,
    /// `insert_source`, `remove_source`); cleared by `repaint`.
    dirty: bool,
    /// Pending damage rect in framebuffer coords — union of every
    /// damage-bearing event since the last repaint. `repaint` clips
    /// against the fb dims and returns it. Full-screen redraws
    /// (cycling, text repaints) reset this to the full bounds.
    damage: Rect,
}

impl Display {
    pub fn new(fb: FrameBuffer) -> Self {
        let mut scrollbacks = BTreeMap::new();
        scrollbacks.insert(Source::Kernel, ProcScrollback::new(Source::Kernel));
        let full = Rect {
            x: 0,
            y: 0,
            w: fb.width(),
            h: fb.height(),
        };
        let s = Self {
            fb,
            active: Source::Kernel,
            scrollbacks,
            surfaces: BTreeMap::new(),
            dirty: true,
            damage: full,
        };
        crate::kernel::stdin::set_active(s.active);
        s
    }

    /// Mark the entire framebuffer as needing a redraw on next repaint.
    /// Used when the active source changes or for text-mode redraws,
    /// where the change isn't expressible as a tight rect.
    fn mark_full_dirty(&mut self) {
        self.dirty = true;
        self.damage = Rect {
            x: 0,
            y: 0,
            w: self.fb.width(),
            h: self.fb.height(),
        };
    }

    /// Extend the pending damage by `rect`, clipped to fb bounds.
    fn extend_damage(&mut self, rect: Rect) {
        let clipped = rect.clip(self.fb.width(), self.fb.height());
        if clipped.is_empty() {
            return;
        }
        self.dirty = true;
        self.damage = self.damage.union(clipped);
    }

    /// Add a new source (new process came up). Idempotent.
    pub fn insert_source(&mut self, source: Source) {
        self.scrollbacks
            .entry(source)
            .or_insert(ProcScrollback::new(source));
    }

    /// Mark a source as exited (process gone). Two paths:
    ///
    /// - **Text-mode pane with content** — keep the scrollback as a
    ///   tombstone, append `(exited)` to the title. The user can
    ///   cycle back to read a short-lived process's output (e.g.
    ///   an `eza` listing) after the fact.
    /// - **Surface-only pane** (or text-mode pane with empty
    ///   scrollback) — drop the entry entirely. Surface-mode panes
    ///   have no useful tombstone view: the surface backing is gone
    ///   and the scrollback is empty, so leaving the pane in the
    ///   cycle is just dead weight.
    ///
    /// The kernel pane is unkillable.
    ///
    /// Surface state is dropped in either case. The backing frame is
    /// freed by the caller (`dealloc_process` walking the surface
    /// table) — `Display` only holds raw KVA + dims, no ownership.
    pub fn remove_source(&mut self, source: Source) {
        if matches!(source, Source::Kernel) {
            return;
        }
        let had_surface = self.surfaces.remove(&source).is_some();

        // Decide tombstone vs full drop. Surface-only sources (no
        // text content ever written) don't leave anything worth
        // looking at, so they exit the cycle entirely. Text-mode
        // panes with real content keep the existing tombstone
        // semantics.
        let drop_pane = match self.scrollbacks.get(&source) {
            Some(sb) => had_surface && sb.scrollback.is_empty(),
            None => false,
        };

        if drop_pane {
            self.scrollbacks.remove(&source);
            if source == self.active {
                // Active pane is going away — advance to the next
                // surviving pane. BTreeMap order is stable; falling
                // back to Kernel covers the empty-map edge case.
                self.active = self
                    .scrollbacks
                    .keys()
                    .next()
                    .copied()
                    .unwrap_or(Source::Kernel);
                crate::kernel::stdin::set_active(self.active);
                self.mark_full_dirty();
            }
            return;
        }

        let Some(sb) = self.scrollbacks.get_mut(&source)
        else {
            if had_surface && source == self.active {
                self.mark_full_dirty();
            }
            return;
        };
        if sb.title.contains("(exited)") {
            return;
        }
        sb.title.push_str("(exited)");
        if self.active == source {
            self.mark_full_dirty();
        }
    }

    /// Append bytes to `source`'s scrollback. Creates the source's
    /// scrollback implicitly if it doesn't already exist.
    ///
    /// If the source is currently in surface mode (has a `SurfaceState`
    /// entry), the bytes still accumulate in scrollback but don't
    /// trigger a repaint — the surface keeps owning the screen until
    /// the source is removed or its surface is destroyed. This lets a
    /// process print boot diagnostics via `console_write`, then take
    /// over with `fb_present`, without losing the early prints.
    pub fn append(&mut self, source: Source, chunk: &[u8]) {
        let sb = self
            .scrollbacks
            .entry(source)
            .or_insert(ProcScrollback::new(source));

        sb.scrollback.append(chunk);
        if source == self.active && !self.surfaces.contains_key(&source) {
            self.mark_full_dirty();
        }
    }

    /// Advance `active` to the next source in key order (wraps).
    /// Used by the Ctrl+Tab "cycle pane" keybind (virtio-input).
    pub fn cycle_active(&mut self) {
        info!("cycling pane");

        if self.scrollbacks.is_empty() {
            return;
        }

        let next = self
            .scrollbacks
            .keys()
            .skip_while(|src| **src != self.active)
            .nth(1)
            .copied()
            .unwrap_or(Source::Kernel);

        if next != self.active {
            self.active = next;
            self.mark_full_dirty();
            crate::kernel::stdin::set_active(self.active);
        }
    }

    /// Submit a damage rect from a `fb_present` syscall. Carries an
    /// immutable snapshot of the surface metadata (kdmap KVA, dims,
    /// format) — installs it as the source's `SurfaceState` if absent
    /// or updates the entry otherwise, then folds the damage rect into
    /// the pending damage if the source is currently active.
    ///
    /// First-call semantics: the source switches to surface mode. The
    /// scrollback for that source stays alive but is no longer rendered
    /// while the surface is registered.
    pub fn present_surface(&mut self, source: Source, args: &PresentArgs) {
        let format = match FbFormat::from_u32(args.format_raw) {
            Some(f) => f,
            None => return, // submitted with a future format we don't know about
        };
        // Surface metadata is immutable post-create today — but record
        // each time anyway so a future "resize" path is a one-liner.
        self.surfaces.insert(
            source,
            SurfaceState {
                kdmap_kva: args.kdmap_kva,
                width: args.width,
                height: args.height,
                format,
            },
        );

        // Lazy-insert a scrollback entry so cycle_active (which
        // iterates `scrollbacks.keys()`) can reach this source. A
        // process spawned with `stdout_capture=1` doesn't get an
        // automatic insert at create time — its `console_write`
        // bytes route to the parent's pane — but a `fb_present` is
        // an explicit "give me my own pane" gesture, so we surface
        // the source key here. The scrollback itself stays unused
        // while the surface is registered (surface wins on repaint).
        self.scrollbacks
            .entry(source)
            .or_insert_with(|| ProcScrollback::new(source));

        if source != self.active {
            return;
        }
        // Surface coordinates already align with framebuffer
        // coordinates 1:1 in v1 (single-source fullscreen, no scaling
        // or offsets). When we add a title bar overlay or scaling, the
        // mapping between (rect_*) and fb coords moves here.
        let rect = Rect {
            x: args.rect_x,
            y: args.rect_y,
            w: args.rect_w,
            h: args.rect_h,
        };
        self.extend_damage(rect);
    }

    /// Re-blit the active source onto the framebuffer if anything has
    /// changed since the last repaint. Returns `Some(rect)` describing
    /// the smallest framebuffer-coord region whose pixels were touched
    /// — caller forwards this to `transfer_to_host_2d` + `flush` to
    /// avoid uploading the whole screen on small updates. `None` =
    /// nothing changed since the last call.
    pub fn repaint(&mut self) -> Option<Rect> {
        if !self.dirty {
            return None;
        }
        self.dirty = false;
        let damage = core::mem::replace(&mut self.damage, Rect::empty())
            .clip(self.fb.width(), self.fb.height());
        if damage.is_empty() {
            return None;
        }

        // Surface mode — the active source has registered a pixel
        // surface. Blit the damage rect from the surface's KDMAP alias
        // into the framebuffer, then return the damage to the caller.
        if let Some(surf) = self.surfaces.get(&self.active).copied() {
            self.blit_surface_rect(&surf, damage);
            return Some(damage);
        }

        // Text mode — full-screen redraw of the active source's
        // scrollback. Damage stays whatever it was (caller can use it
        // verbatim; a partial rect would only cover the lines that
        // changed but the existing scrollback layout doesn't track
        // line-level dirty regions).
        self.fb.fill(fb::DARK_GRAY);

        // Title bar at the top showing which source is visible.
        let title = &self.scrollbacks[&self.active].title[..];
        self.fb
            .fill_rect(0, 0, self.fb.width(), GLYPH_H, fb::DARK_GRAY);
        self.fb
            .blit_text(0, 0, title, fb::rgb(192, 0, 192), fb::DARK_GRAY);

        // Body: blit scrollback lines starting under the title bar.
        // Each line is a sequence of styled chunks; walk them
        // left-to-right, advancing `cx` by `chunk.text.len() *
        // GLYPH_W` after each blit. Truncate at the right edge.
        let cols = (self.fb.width() / GLYPH_W) as usize;
        let rows_avail = ((self.fb.height() - GLYPH_H) / GLYPH_H) as usize;
        let Some(sb) = self.scrollbacks.get(&self.active)
        else {
            return Some(damage);
        };

        for (row_idx, line) in sb.scrollback.view(rows_avail).enumerate() {
            let y = GLYPH_H + row_idx as u32 * GLYPH_H;
            let mut col = 0usize;
            for chunk in &line.chunks {
                if col >= cols {
                    break;
                }
                let remaining = cols - col;
                let slice = if chunk.text.len() > remaining {
                    &chunk.text[..remaining]
                }
                else {
                    chunk.text.as_str()
                };
                let x = col as u32 * GLYPH_W;
                self.fb
                    .blit_text(x, y, slice, chunk.style.fg, chunk.style.bg);
                col += slice.len();
            }
        }

        Some(damage)
    }

    /// Per-row memcpy from the surface's KDMAP alias into the
    /// framebuffer. `rect` is in framebuffer coordinates (which v1
    /// maps 1:1 onto surface coordinates) and has been clipped to
    /// fb bounds by the caller; this routine clips again to surface
    /// bounds so a present that names a rect past the surface dims
    /// just no-ops on the missing rows.
    fn blit_surface_rect(&self, surf: &SurfaceState, rect: Rect) {
        if rect.is_empty() {
            return;
        }
        // Clip to surface as well — surface dims may be smaller than
        // the framebuffer (centered / stretched layouts come later).
        let x_end = rect.x.saturating_add(rect.w).min(surf.width);
        let y_end = rect.y.saturating_add(rect.h).min(surf.height);
        if x_end <= rect.x || y_end <= rect.y {
            return;
        }
        let bpp = surf.format.bytes_per_pixel() as usize;
        let surf_pitch_bytes = surf.width as usize * bpp;
        let fb_w = self.fb.width() as usize;

        // SAFETY: `surf.kdmap_kva` was installed by
        // `run_fb_surface_create_req` from a Frame<Shared>; it is
        // valid for the surface's full size_bytes for the lifetime of
        // the entry, and `Display` only holds the snapshot — the
        // backing is owned by the per-process surface table. fb base
        // pointer is the install-time KDMAP pointer set in `fb::new`,
        // valid for the system lifetime. Volatile copy avoids racing
        // with the user-side draw — userland may still be writing
        // into the surface concurrently with our read; in v1 we
        // accept tearing within a row (interactive UIs rarely notice).
        unsafe {
            let fb_base = self.fb.base_ptr() as *mut u32;
            let src_base = surf.kdmap_kva as *const u32;
            for row in rect.y..y_end {
                let cols = x_end - rect.x;
                let src_row = src_base.add(row as usize * surf.width as usize + rect.x as usize);
                let dst_row = fb_base.add(row as usize * fb_w + rect.x as usize);
                for col in 0..cols as usize {
                    let px = src_row.add(col).read_volatile();
                    dst_row.add(col).write_volatile(px);
                }
            }
            // Suppress unused warning for surf_pitch_bytes — kept for
            // future memcpy-by-row optimization where we'd write
            // bpp-sized strides instead of u32 pixels.
            let _ = surf_pitch_bytes;
        }
    }
}
