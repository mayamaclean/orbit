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
    Csi { params: String },
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
                        AnsiState::Csi { params: String::new() }
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
    /// `true` if the next `repaint` needs to re-blit the active
    /// source. Writes to the active source set it; cycling set it.
    dirty: bool,
}

impl Display {
    pub fn new(fb: FrameBuffer) -> Self {
        let mut scrollbacks = BTreeMap::new();
        scrollbacks.insert(Source::Kernel, ProcScrollback::new(Source::Kernel));
        let s = Self {
            fb,
            active: Source::Kernel,
            scrollbacks,
            dirty: true,
        };
        crate::kernel::stdin::set_active(s.active);
        s
    }

    /// Add a new source (new process came up). Idempotent.
    pub fn insert_source(&mut self, source: Source) {
        self.scrollbacks
            .entry(source)
            .or_insert(ProcScrollback::new(source));
    }

    /// Mark a source as exited (process gone). Keeps the scrollback
    /// in place — the user might want to cycle back to a short-lived
    /// process's output (e.g. an `eza` listing) after the fact —
    /// just appends `(exited)` to the title and stays put on the
    /// active pane. The kernel pane is unkillable.
    pub fn remove_source(&mut self, source: Source) {
        if matches!(source, Source::Kernel) {
            return;
        }
        let Some(sb) = self.scrollbacks.get_mut(&source)
        else {
            return;
        };
        if sb.title.contains("(exited)") {
            return;
        }
        sb.title.push_str("(exited)");
        if self.active == source {
            self.dirty = true;
        }
    }

    /// Append bytes to `source`'s scrollback. Creates the source's
    /// scrollback implicitly if it doesn't already exist.
    pub fn append(&mut self, source: Source, chunk: &[u8]) {
        let sb = self
            .scrollbacks
            .entry(source)
            .or_insert(ProcScrollback::new(source));

        sb.scrollback.append(chunk);
        if source == self.active {
            self.dirty = true;
        }
    }

    /// Advance `active` to the next source in key order (wraps).
    /// Used by the UART-RX "cycle pane" keybind.
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
            self.dirty = true;
            crate::kernel::stdin::set_active(self.active);
        }
    }

    /// Re-blit the active source onto the framebuffer if anything has
    /// changed since the last repaint. Returns `true` if pixels were
    /// touched (so the caller knows to run a transfer + flush).
    pub fn repaint(&mut self) -> bool {
        if !self.dirty {
            return false;
        }
        self.dirty = false;

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
            return true;
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
                self.fb.blit_text(x, y, slice, chunk.style.fg, chunk.style.bg);
                col += slice.len();
            }
        }

        true
    }
}
