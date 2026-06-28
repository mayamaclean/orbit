//! `/bin/orbit-top-std` — system monitor TUI on the orbit framebuffer.
//!
//! Refreshes 4× per second; computes hart-utilization, kernel-memory,
//! self-process and top-syscall views from `query_stats` +
//! `query_syscall_stats` deltas. Quits on `q` / `Q` / `Esc`.
//!
//! No process enumeration — `query_stats` only returns the caller's
//! own counters today. The "this process" pane shows the demo's own
//! state; everything else is system-wide. A real `htop`-shaped
//! process table is gated on the `query_proc_list` syscall.
//!
//! Loop shape:
//!
//! - `read_key_event(buf, n, 0, REFRESH_MS)` — single syscall that
//!   blocks up to one tick OR returns early on input.
//! - On wake (timer or key): drain events, snapshot stats, redraw.

use ab_glyph::{FontVec, PxScale};
use orbit_abi::Sysno;
use orbit_abi::fb::{FbFormat, FbInfo};
use orbit_abi::input::{KeyCode, KeyEvent, KeyEventKind, decoded_char};
use orbit_abi::stats::ProcessStats;
use orbit_abi::syscall_stats::{SyscallEntry, payload_size};
use orbit_abi::user::{
    fb_query, fb_surface_create, fb_surface_destroy, query_stats, query_syscall_stats,
    read_key_event,
};
use orbit_ratatui::OrbitBackend;
use orbit_text::SurfaceMut;
use ratatui::Terminal;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

const FONT_PATH: &str = "/usr/share/fonts/LiberationMono-Regular.ttf";
const CELL_PX: f32 = 18.0;
/// Refresh tick. 4 Hz is enough for human-readable monitor output;
/// going faster mostly burns CPU on the redraw path.
const REFRESH_MS: usize = 250;

fn main() {
    println!("orbit-top-std: starting");

    let font_bytes = match std::fs::read(FONT_PATH) {
        Ok(b) => b,
        Err(e) => {
            println!("orbit-top-std: read({FONT_PATH}) failed: {e}");
            return;
        }
    };
    let font = match FontVec::try_from_vec(font_bytes) {
        Ok(f) => f,
        Err(e) => {
            println!("orbit-top-std: FontVec parse failed: {e}");
            return;
        }
    };

    let mut info = FbInfo::default();
    if let Err(e) = fb_query(&mut info) {
        println!("orbit-top-std: fb_query failed errno={}", e.0);
        return;
    }
    let format = match FbFormat::from_u32(info.format) {
        Some(f) => f,
        None => {
            println!("orbit-top-std: unknown format {}", info.format);
            return;
        }
    };

    let (handle, user_va) = match fb_surface_create(info.width, info.height, format) {
        Ok(p) => p,
        Err(e) => {
            println!("orbit-top-std: fb_surface_create failed errno={}", e.0);
            return;
        }
    };

    let pixel_count = info.width as usize * info.height as usize;
    let pixels: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(user_va as *mut u32, pixel_count) };
    let surf = match SurfaceMut::new(pixels, info.width, info.height) {
        Some(s) => s,
        None => {
            println!("orbit-top-std: SurfaceMut::new mismatched len");
            let _ = fb_surface_destroy(handle);
            return;
        }
    };

    let backend = OrbitBackend::new(
        surf,
        &font,
        PxScale::from(CELL_PX),
        handle,
        (0xE0, 0xE0, 0xE6),
        (0x10, 0x14, 0x1F),
    );

    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            println!("orbit-top-std: Terminal::new failed: {e:?}");
            let _ = fb_surface_destroy(handle);
            return;
        }
    };

    if let Err(e) = terminal.clear() {
        println!("orbit-top-std: terminal.clear failed: {e:?}");
    }

    // Two-snapshot history. The first frame can't compute deltas (no
    // prior sample), so it shows zero rates — the second tick onward
    // produces real numbers.
    let mut prev: Option<Snapshot> = None;
    let mut event_buf = [KeyEvent::default(); 16];
    let mut syscall_buf = vec![0u8; payload_size()];
    let mut tick: u64 = 0;

    'outer: loop {
        // Snapshot before draw so the displayed values correspond to
        // the moment the frame was rendered. Cheap (~µs).
        let curr = Snapshot::take(&mut syscall_buf);

        let draw_result = terminal.draw(|frame| ui(frame, &curr, prev.as_ref(), tick));
        if let Err(e) = draw_result {
            println!("orbit-top-std: draw failed: {e:?}");
            break;
        }

        // Park up to REFRESH_MS or until a key arrives.
        let n = match read_key_event(event_buf.as_mut_ptr(), event_buf.len(), 0, REFRESH_MS) {
            Ok(n) => n,
            Err(e) => {
                println!("orbit-top-std: read_key_event errno={}", e.0);
                break;
            }
        };
        for ev in &event_buf[..n] {
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

        prev = Some(curr);
        tick = tick.wrapping_add(1);
    }

    if let Err(e) = fb_surface_destroy(handle) {
        println!("orbit-top-std: destroy failed errno={}", e.0);
    }
    println!("orbit-top-std: done");
}

/// One sample of system + self stats. Two consecutive snapshots feed
/// delta-based rate computations (cpu %, syscalls/s, etc.).
struct Snapshot {
    proc: ProcessStats,
    /// Per-ordinal syscall counters/total_ticks. Owned `Vec` so the
    /// snapshot outlives the syscall scratch buffer.
    syscall: Vec<SyscallEntry>,
}

impl Snapshot {
    fn take(scratch: &mut [u8]) -> Self {
        let proc = query_stats().unwrap_or_default();
        let syscall = match query_syscall_stats(scratch) {
            Ok((_hdr, entries)) => entries.to_vec(),
            Err(_) => Vec::new(),
        };
        Self { proc, syscall }
    }
}

/// Top-level layout: title strip, hart gauges, two-column kernel/proc
/// summary, syscall histogram. All `Length`-constrained except the
/// histogram which takes the rest.
fn ui(frame: &mut ratatui::Frame, curr: &Snapshot, prev: Option<&Snapshot>, tick: u64) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Length(7), // hart gauges (1 title + 4 bars + spacing+borders)
            Constraint::Length(8), // kernel mem + this proc
            Constraint::Min(0),    // syscall histogram
        ])
        .split(area);

    render_title(frame, chunks[0], tick);
    render_harts(frame, chunks[1], curr, prev);
    render_summary(frame, chunks[2], curr);
    render_syscalls(frame, chunks[3], curr, prev);
}

fn render_title(frame: &mut ratatui::Frame, area: Rect, tick: u64) {
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
            "monitor",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  —  "),
        Span::styled(format!("tick {tick}"), Style::default().fg(Color::Gray)),
        Span::raw("  ·  "),
        Span::styled(
            format!("{} Hz", 1000 / REFRESH_MS),
            Style::default().fg(Color::Gray),
        ),
        Span::raw("  ·  "),
        Span::styled("q to quit", Style::default().fg(Color::DarkGray)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(title, area);
}

/// Four horizontal gauges showing system-wide hart-time partition over
/// the interval since the previous snapshot. Buckets are partition-
/// disjoint, so the four percentages sum to ~100%.
fn render_harts(frame: &mut ratatui::Frame, area: Rect, curr: &Snapshot, prev: Option<&Snapshot>) {
    let block = Block::default()
        .title(" harts ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Float ratios (0..=1) per bucket and the raw delta-tick counts
    // alongside, so the per-bar label can show three-decimal precision
    // — the scheduler bucket runs in the µs range per pass and would
    // otherwise round to integer 0% even with real activity.
    let (ratios, deltas, total) = match prev {
        Some(p) => {
            let du = curr
                .proc
                .hart_user_ticks
                .saturating_sub(p.proc.hart_user_ticks);
            let dk = curr
                .proc
                .hart_kernel_ticks
                .saturating_sub(p.proc.hart_kernel_ticks);
            let ds = curr
                .proc
                .hart_scheduler_ticks
                .saturating_sub(p.proc.hart_scheduler_ticks);
            let di = curr
                .proc
                .hart_idle_ticks
                .saturating_sub(p.proc.hart_idle_ticks);
            let total = du.saturating_add(dk).saturating_add(ds).saturating_add(di);
            let ratio = |v: u64| -> f64 {
                if total == 0 {
                    0.0
                }
                else {
                    (v as f64 / total as f64).clamp(0.0, 1.0)
                }
            };
            (
                [ratio(du), ratio(dk), ratio(ds), ratio(di)],
                [du, dk, ds, di],
                total,
            )
        }
        None => ([0.0; 4], [0u64; 4], 0u64),
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let names = ["user  ", "kernel", "sched ", "idle  "];
    let colors = [
        Color::LightGreen,
        Color::LightYellow,
        Color::LightMagenta,
        Color::DarkGray,
    ];
    for (i, row) in rows.iter().enumerate() {
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(8),  // name
                Constraint::Min(0),     // gauge
                Constraint::Length(20), // numeric label
            ])
            .split(*row);
        let label = Paragraph::new(Line::from(Span::styled(
            names[i],
            Style::default().fg(Color::Gray),
        )));
        frame.render_widget(label, split[0]);
        let g = Gauge::default()
            .gauge_style(Style::default().fg(colors[i]))
            .ratio(ratios[i])
            .label("");
        frame.render_widget(g, split[1]);
        // Three-decimal % + raw tick delta. The scheduler bucket
        // runs in the µs range per pass; integer % would round to 0
        // and hide whether the bracket is live.
        let pct_text = if total == 0 {
            "      n/a".to_string()
        }
        else {
            format!("{:>7.3}%", ratios[i] * 100.0)
        };
        let pct_widget = Paragraph::new(Line::from(vec![
            Span::styled(pct_text, Style::default().fg(Color::White)),
            Span::raw(" "),
            Span::styled(
                format!("{:>10}t", deltas[i]),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        frame.render_widget(pct_widget, split[2]);
    }
}

/// Two-column block: left = system-wide kernel pools, right = this
/// process's counters.
fn render_summary(frame: &mut ratatui::Frame, area: Rect, curr: &Snapshot) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let mem = Paragraph::new(vec![
        kv("kpages", fmt_bytes(curr.proc.kernel_kpages_bytes)),
        kv("upages", fmt_bytes(curr.proc.kernel_user_pages_bytes)),
        kv("ktables", fmt_bytes(curr.proc.kernel_ktables_bytes)),
        kv("kheap", fmt_bytes(curr.proc.kernel_heap_bytes)),
        kv(
            "wake_q",
            format!(
                "peak {}/{}  ·  drops {}",
                curr.proc.wake_queue_peak,
                curr.proc.wake_queue_capacity,
                curr.proc.wake_queue_drops,
            ),
        ),
    ])
    .block(
        Block::default()
            .title(" kernel runtime ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(mem, cols[0]);

    let proc = Paragraph::new(vec![
        kv(
            "pid",
            format!("{}     threads {}", curr.proc.pid, curr.proc.thread_count),
        ),
        kv("cpu", fmt_ticks_ms(curr.proc.cpu_ticks)),
        kv(
            "memory",
            format!(
                "rss {}  ·  heap {}",
                fmt_bytes(curr.proc.resident_bytes),
                fmt_bytes(curr.proc.heap_bytes)
            ),
        ),
        kv(
            "denials",
            format!(
                "perm {}  ·  role {}",
                curr.proc.perm_denials, curr.proc.role_denials
            ),
        ),
    ])
    .block(
        Block::default()
            .title(" this process ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(proc, cols[1]);

    // Total syscalls counter under the bottom of the right column has
    // no native ratatui placement — fold it into the proc block above
    // by appending; layout already accommodates 4 lines.
    // (kept as a TODO if more lines need to land here)
    let _ = curr.proc.syscalls;
}

/// Top syscalls by *count delta* since the previous snapshot, rendered
/// as a unicode bar chart with absolute count and ms/tick service time
/// alongside. Falls back to absolute counts on the first frame.
fn render_syscalls(
    frame: &mut ratatui::Frame,
    area: Rect,
    curr: &Snapshot,
    prev: Option<&Snapshot>,
) {
    let block = Block::default()
        .title(" top syscalls (Δ count over interval) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if curr.syscall.is_empty() {
        let p = Paragraph::new("no syscall data yet").style(Style::default().fg(Color::DarkGray));
        frame.render_widget(p, inner);
        return;
    }

    // Build (ordinal, delta_count, total_count, total_ticks, max_ticks) tuples.
    let mut rows: Vec<(usize, u64, u64, u64, u64)> = curr
        .syscall
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let delta = match prev {
                Some(p) if i < p.syscall.len() => e.count.saturating_sub(p.syscall[i].count),
                _ => e.count,
            };
            (i, delta, e.count, e.total_ticks, e.max_ticks)
        })
        .filter(|(_, _, total, _, _)| *total > 0)
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    let max_delta = rows
        .iter()
        .map(|(_, d, _, _, _)| *d)
        .max()
        .unwrap_or(1)
        .max(1);

    // Column budget: name 20, gap 1, Δcount 6, gap 2, total 7, gap 2,
    // total_ms 12, gap 2, max_us 9. Longest sysno_name today is
    // `query_syscall_stats` (19 chars); pin to 20 with `{:<20.20}` so a
    // future longer name truncates rather than shoves the bar right.
    const NAME_W: usize = 20;
    const TAIL_BUDGET: usize =
        1 /*gap*/ + 6 /*Δ*/ + 2 + 7 /*total*/ + 2 + 12 /*total_ms*/ + 2 + 9 /*max_us*/;
    let row_count = (inner.height as usize).min(rows.len());
    let lines: Vec<Line> = rows
        .iter()
        .take(row_count)
        .map(|(ord, delta, total, ticks, max_ticks)| {
            let name = sysno_name(*ord);
            let bar_width = (inner.width as usize).saturating_sub(NAME_W + TAIL_BUDGET);
            let filled = if max_delta == 0 {
                0
            }
            else {
                (*delta as usize * bar_width) / max_delta as usize
            };
            let filled = filled.min(bar_width);
            let bar = "█".repeat(filled);
            let pad = " ".repeat(bar_width.saturating_sub(filled));
            // `time` CSR is 10 MHz on qemu-virt — 10 ticks per µs.
            let max_us = max_ticks / 10;
            Line::from(vec![
                Span::styled(
                    format!("{:<NAME_W$.NAME_W$}", name),
                    Style::default().fg(Color::White),
                ),
                Span::styled(bar, Style::default().fg(Color::LightGreen)),
                Span::styled(pad, Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(format!("Δ{:>5}", delta), Style::default().fg(Color::Yellow)),
                Span::raw("  "),
                Span::styled(format!("{:>7}", total), Style::default().fg(Color::Gray)),
                Span::raw("  "),
                Span::styled(
                    format!("{:>12}", fmt_ticks_ms(*ticks)),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{:>6}µs", max_us),
                    Style::default().fg(Color::LightRed),
                ),
            ])
        })
        .collect();

    let p = Paragraph::new(lines);
    frame.render_widget(p, inner);
}

fn kv(k: &str, v: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {:<8}", k), Style::default().fg(Color::Gray)),
        Span::styled(v.into(), Style::default().fg(Color::White)),
    ])
}

fn pct_u16(num: u64, denom: u64) -> u16 {
    if denom == 0 {
        return 0;
    }
    let p = (num.saturating_mul(100)) / denom;
    p.min(100) as u16
}

/// Format bytes into `KiB`/`MiB`/`GiB` with one decimal where useful.
/// Sticks to power-of-two units so the numbers line up with what the
/// kernel allocator pools actually carve out.
fn fmt_bytes(b: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    if b < KIB {
        format!("{} B", b)
    }
    else if b < MIB {
        format!("{} KiB", b / KIB)
    }
    else if b < GIB {
        format!("{:.1} MiB", b as f64 / MIB as f64)
    }
    else {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    }
}

/// Convert ticks (10 MHz on qemu-virt) to a milliseconds string.
fn fmt_ticks_ms(t: u64) -> String {
    // 10_000 ticks / ms per ABI doc on ProcessStats.
    let ms = t / 10_000;
    let us = (t / 10) % 1_000;
    format!("{ms}.{us:03} ms")
}

/// Map a `Sysno::ordinal` value back to a short display name. Mirrors
/// the table in [`console/src/main.rs`](../console/src/main.rs)
/// `syscall_name`. Append entries when new syscalls land — same
/// ordinal map, same names.
fn sysno_name(ord: usize) -> &'static str {
    match ord {
        0 => "exit",
        1 => "serial_print",
        2 => "sleep_ms",
        3 => "console_write",
        4 => "read_stdin",
        5 => "set_affinity",
        6 => "get_affinity",
        7 => "get_hart_id",
        8 => "mmap",
        9 => "create_netch",
        10 => "close_handle",
        11 => "create_process",
        12 => "ch_yield",
        13 => "query_stats",
        14 => "query_syscall_stats",
        15 => "create_thread",
        16 => "get_micros",
        17 => "fs_open",
        18 => "fs_read",
        19 => "fs_stat",
        20 => "getpid",
        21 => "gettid",
        22 => "wait_pid",
        23 => "create_process_ex",
        24 => "argv_envp",
        25 => "futex_wait",
        26 => "futex_wake",
        27 => "fs_readdir",
        28 => "pledge",
        29 => "create_process_v2",
        30 => "query_denial_log",
        31 => "chdir",
        32 => "getcwd",
        33 => "fs_seek",
        34 => "fs_fstat",
        35 => "getuid",
        36 => "geteuid",
        37 => "getgid",
        38 => "getegid",
        39 => "getgroups",
        40 => "getlogin",
        41 => "setuid",
        42 => "setgid",
        43 => "setgroups",
        44 => "setlogin",
        45 => "get_realtime",
        46 => "thread_exit",
        47 => "fb_query",
        48 => "fb_surface_create",
        49 => "fb_surface_destroy",
        50 => "fb_present",
        51 => "read_key_event",
        52 => "wake_tid",
        53 => "dup",
        54 => "dup2",
        55 => "fcntl",
        56 => "fstat",
        57 => "eventfd",
        58 => "ch_inspect",
        _ => "?",
    }
}

// Compile-time guard: the local `sysno_name` table covers every
// ordinal the kernel knows about. Bump alongside `Sysno::COUNT`.
const _ASSERT_SYSNO_COUNT: () = assert!(Sysno::COUNT == 59);
