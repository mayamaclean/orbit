//! `/bin/hello-ratatui-std` — static-frame ratatui demo running on
//! orbit's framebuffer.
//!
//! Wires the `OrbitBackend` from `orbit-ratatui` into a stock
//! `ratatui::Terminal`, draws a single frame composed of `Block`,
//! `Paragraph`, and `List` widgets across a two-column layout, presents
//! it, sleeps so the human can look at it, and tears down. No event
//! loop yet — input wiring lands with the kernel-side
//! `read_key_event` syscall (M3).
//!
//! Run via the loader: bundled into `disk.img` by `tools/build-disk.sh`,
//! launched from the console with `hello-ratatui-std`.

use ab_glyph::{FontVec, PxScale};
use orbit_abi::fb::{FbFormat, FbInfo};
use orbit_abi::input::{KeyCode, KeyEvent, KeyEventKind, READ_KEY_EVENT_INDEFINITE, decoded_char};
use orbit_abi::user::{fb_query, fb_surface_create, fb_surface_destroy, read_key_event};
use orbit_ratatui::OrbitBackend;
use orbit_text::SurfaceMut;
use ratatui::Terminal;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

const FONT_PATH: &str = "/usr/share/fonts/LiberationMono-Regular.ttf";
/// Pixel size for the cell grid font. 18 px works out to a ~70-column
/// grid at 1280×720, which leaves room for borders + a paragraph.
const CELL_PX: f32 = 18.0;

fn main() {
    println!("hello-ratatui-std: starting");

    let font_bytes = match std::fs::read(FONT_PATH) {
        Ok(b) => b,
        Err(e) => {
            println!("hello-ratatui-std: read({FONT_PATH}) failed: {e}");
            return;
        }
    };
    let font = match FontVec::try_from_vec(font_bytes) {
        Ok(f) => f,
        Err(e) => {
            println!("hello-ratatui-std: FontVec parse failed: {e}");
            return;
        }
    };

    let mut info = FbInfo::default();
    if let Err(e) = fb_query(&mut info) {
        println!("hello-ratatui-std: fb_query failed errno={}", e.0);
        return;
    }
    let format = match FbFormat::from_u32(info.format) {
        Some(f) => f,
        None => {
            println!("hello-ratatui-std: unknown format {}", info.format);
            return;
        }
    };
    println!(
        "hello-ratatui-std: display {}x{} format={}",
        info.width, info.height, info.format
    );

    let (handle, user_va) = match fb_surface_create(info.width, info.height, format) {
        Ok(p) => p,
        Err(e) => {
            println!("hello-ratatui-std: fb_surface_create failed errno={}", e.0);
            return;
        }
    };

    // Wrap the kernel-mapped surface in a `&mut [u32]`. Same pattern
    // as hello-fb-std — the slice lifetime ends when `surf` (and the
    // backend it feeds) goes out of scope.
    let pixel_count = info.width as usize * info.height as usize;
    let pixels: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(user_va as *mut u32, pixel_count) };
    let surf = match SurfaceMut::new(pixels, info.width, info.height) {
        Some(s) => s,
        None => {
            println!("hello-ratatui-std: SurfaceMut::new mismatched len");
            let _ = fb_surface_destroy(handle);
            return;
        }
    };

    // Dark slate bg + soft off-white fg as defaults — `Color::Reset`
    // in ratatui resolves to these.
    let default_bg = (0x10, 0x14, 0x1F);
    let default_fg = (0xE0, 0xE0, 0xE6);

    let backend = OrbitBackend::new(
        surf,
        &font,
        PxScale::from(CELL_PX),
        handle,
        default_fg,
        default_bg,
    );
    let (cols, rows) = backend.grid();
    println!(
        "hello-ratatui-std: ratatui grid {} cols × {} rows (cell {}×{} px)",
        cols,
        rows,
        backend.metrics().width,
        backend.metrics().height
    );

    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            println!("hello-ratatui-std: Terminal::new failed: {e:?}");
            let _ = fb_surface_destroy(handle);
            return;
        }
    };

    if let Err(e) = terminal.clear() {
        println!("hello-ratatui-std: terminal.clear failed: {e:?}");
    }

    let draw_result = terminal.draw(|frame| {
        let area = frame.area();

        // Top title strip + a body that splits into a wide paragraph
        // and a narrower stat list. 3-line title gives the paragraph
        // a clear anchor.
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(outer[1]);

        // Title bar.
        let title = Paragraph::new(Line::from(vec![
            Span::styled(
                " orbit ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                "ratatui demo",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  —  "),
            Span::styled(
                "kernel framebuffer · TTF cells · static frame",
                Style::default().fg(Color::Gray),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(title, outer[0]);

        // Left pane: a paragraph that exercises wrapping + mixed styles.
        let lines = vec![
            Line::from(vec![
                Span::styled("hello, ", Style::default().fg(Color::White)),
                Span::styled(
                    "ratatui",
                    Style::default()
                        .fg(Color::LightMagenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" on "),
                Span::styled("orbit", Style::default().fg(Color::LightCyan)),
                Span::raw("."),
            ]),
            Line::from(""),
            Line::from(
                "this frame is rendered cell-by-cell into a kernel-allocated framebuffer surface, \
                 with anti-aliased TTF glyphs cached in a hashbrown map keyed on (glyph_id, scale).",
            ),
            Line::from(""),
            Line::from(vec![
                Span::raw("colors: "),
                Span::styled(" red ", Style::default().bg(Color::Red).fg(Color::White)),
                Span::styled(
                    " green ",
                    Style::default().bg(Color::Green).fg(Color::Black),
                ),
                Span::styled(
                    " yellow ",
                    Style::default().bg(Color::Yellow).fg(Color::Black),
                ),
                Span::styled(" blue ", Style::default().bg(Color::Blue).fg(Color::White)),
                Span::styled(
                    " magenta ",
                    Style::default().bg(Color::Magenta).fg(Color::White),
                ),
                Span::styled(
                    " cyan ",
                    Style::default().bg(Color::Cyan).fg(Color::Black),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("box drawing: "),
                Span::styled(
                    "┌─┬─┐  ╭───╮  ╔═══╗",
                    Style::default().fg(Color::LightGreen),
                ),
            ]),
            Line::from(vec![
                Span::raw("             "),
                Span::styled(
                    "│ │ │  │   │  ║   ║",
                    Style::default().fg(Color::LightGreen),
                ),
            ]),
            Line::from(vec![
                Span::raw("             "),
                Span::styled(
                    "└─┴─┘  ╰───╯  ╚═══╝",
                    Style::default().fg(Color::LightGreen),
                ),
            ]),
        ];
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .title(" notes ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );
        frame.render_widget(para, body[0]);

        // Right pane: a list of milestones (some checked, some not).
        let items = vec![
            ListItem::new(Line::from(vec![
                Span::styled("[x] ", Style::default().fg(Color::Green)),
                Span::raw("M1 surfaces + present"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[x] ", Style::default().fg(Color::Green)),
                Span::raw("M2 orbit-text + ab_glyph"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[x] ", Style::default().fg(Color::Green)),
                Span::raw("M4 ratatui Backend"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[ ] ", Style::default().fg(Color::Yellow)),
                Span::raw("M3 raw key events"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[ ] ", Style::default().fg(Color::Yellow)),
                Span::raw("M5 event polling"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("[ ] ", Style::default().fg(Color::Gray)),
                Span::raw("M6 htop-shaped demo"),
            ])),
        ];
        let list = List::new(items).block(
            Block::default()
                .title(" milestones ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(list, body[1]);
    });

    if let Err(e) = draw_result {
        println!("hello-ratatui-std: terminal.draw failed: {e:?}");
    }

    println!(
        "hello-ratatui-std: drew frame, glyph cache holds {} entries — press q or Esc to exit",
        terminal.backend().cache_entries()
    );

    // Blocking event loop. `READ_KEY_EVENT_INDEFINITE` parks the
    // thread until a key arrives (no timer wakeup). For a frame-rate
    // shaped loop pass a finite ms instead — e.g.
    // `read_key_event(buf, len, 0, 16)` is a 60-Hz tick budget that
    // wakes early on input. Filter on Press so a held key exits cleanly
    // on first transition (Repeat / Release ignored).
    let mut events = [KeyEvent::default(); 16];
    'outer: loop {
        let n = match read_key_event(
            events.as_mut_ptr(),
            events.len(),
            0,
            READ_KEY_EVENT_INDEFINITE,
        ) {
            Ok(n) => n,
            Err(e) => {
                println!("hello-ratatui-std: read_key_event errno={}", e.0);
                break;
            }
        };
        for ev in &events[..n] {
            if ev.event_kind() != Some(KeyEventKind::Press) {
                continue;
            }
            match ev.key_code() {
                Some(KeyCode::Escape) => break 'outer,
                Some(KeyCode::Char) => {
                    if let Some(c) = decoded_char(ev.code) {
                        if c == 'q' || c == 'Q' {
                            break 'outer;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if let Err(e) = fb_surface_destroy(handle) {
        println!("hello-ratatui-std: destroy failed errno={}", e.0);
    }
    println!("hello-ratatui-std: done");
}
