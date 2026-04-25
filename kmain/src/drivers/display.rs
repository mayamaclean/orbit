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

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
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

/// Per-source line history + an in-progress partial line.
pub struct Scrollback {
    lines: VecDeque<String>,
    pending: String,
}

impl Default for Scrollback {
    fn default() -> Self { Self::new() }
}

impl Scrollback {
    pub fn new() -> Self {
        Self {
            lines: VecDeque::with_capacity(SCROLLBACK_LINES),
            pending: String::new(),
        }
    }

    /// Append a chunk of bytes. Splits on `\n`, drops non-printable
    /// bytes (bitmap font is ASCII-only for now), truncates any line
    /// that exceeds [`MAX_LINE_LEN`] until the next `\n` resets.
    pub fn append(&mut self, chunk: &[u8]) {
        for &b in chunk {
            if b == b'\n' {
                let completed = core::mem::take(&mut self.pending);
                if self.lines.len() >= SCROLLBACK_LINES {
                    self.lines.pop_front();
                }
                self.lines.push_back(completed);
                continue;
            }
            if self.pending.len() >= MAX_LINE_LEN {
                continue;
            }
            let ch = if b.is_ascii_graphic() || b == b' ' || b == b'\t' {
                if b == b'\t' { ' ' } else { b as char }
            } else {
                '?'
            };
            self.pending.push(ch);
        }
    }

    /// Yield up to `rows` visible lines: at most `rows-1` completed
    /// lines plus the `pending` line if it's non-empty, in
    /// oldest-first order suitable for top-to-bottom blit.
    pub fn view(&self, rows: usize) -> impl Iterator<Item = &str> {
        let pending_slot = if self.pending.is_empty() { 0 } else { 1 };
        let line_rows = rows.saturating_sub(pending_slot);
        let skip = self.lines.len().saturating_sub(line_rows);
        self.lines
            .iter()
            .skip(skip)
            .map(String::as_str)
            .chain(
                Some(self.pending.as_str())
                    .filter(|s| !s.is_empty()),
            )
    }
}

pub struct ProcScrollback {
    pub(super) scrollback: Scrollback,
    pub(super) title: String
}

impl ProcScrollback {
    pub fn new(src: Source) -> Self {
        Self {
            scrollback: Scrollback::new(),
            title: match src {
                Source::Kernel => " kernel ".into(),
                Source::Process(pid) => alloc::format!(" pid {pid} ")
            }
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
        self.scrollbacks.entry(source).or_insert(ProcScrollback::new(source));
    }

    /// Remove a source (process exited). If it was active, advance to
    /// the next live source (falling back to `Kernel`).
    pub fn remove_source(&mut self, source: Source) {
        if self.scrollbacks.remove(&source).is_none() {
            return;
        }
        if self.active == source {
            self.active = self
                .scrollbacks
                .keys()
                .next()
                .copied()
                .unwrap_or(Source::Kernel);

            self.dirty = true;
            crate::kernel::stdin::set_active(self.active);
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
            return
        }

        let next = self.scrollbacks.keys()
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
        self.fb.fill_rect(0, 0, self.fb.width(), GLYPH_H, fb::DARK_GRAY);
        self.fb.blit_text(0, 0, title, fb::rgb(192, 0, 192), fb::DARK_GRAY);

        // Body: blit scrollback lines starting under the title bar.
        let cols = (self.fb.width() / GLYPH_W) as usize;
        let rows_avail = ((self.fb.height() - GLYPH_H) / GLYPH_H) as usize;
        let Some(sb) = self.scrollbacks.get(&self.active) else {
            return true;
        };

        for (row_idx, line) in sb.scrollback.view(rows_avail).enumerate() {
            let y = GLYPH_H + row_idx as u32 * GLYPH_H;
            // blit_text also clamps per-pixel, but slicing here
            // avoids walking glyphs we'd throw away.
            let slice = if line.len() > cols { &line[..cols] } else { line };
            self.fb.blit_text(0, y, slice, fb::WHITE, fb::DARK_GRAY);
        }

        true
    }
}
