#!/usr/bin/env python3
"""Parse a kmain serial log into a netch/smoltcp event timeline.

Skips early boot output (everything before the configurable
`--start-marker`), then bucketizes the trace stream into fixed-width
time windows and counts/sums per-class events. Output is one CSV row
per bucket so the result loads cleanly into pandas / matplotlib.

Why
---
We're chasing why steady-state netch transfers oscillate between
"keep-alive cadence" (one burst per 1 s timer) and "wire-rate fast"
modes during long eza_stress.py runs. To find what flipped, we need
to correlate per-event rates over time:

  * `rx_buffer.{enqueueing,dequeueing}` octets   → wire throughput
  * `e1000 IRQ`                                  → device pressure
  * `sending ACK`, `keep-alive timer expired`    → smoltcp tx events
  * `delayed ack timer …`                        → ACK batching state
  * `[set_wake_reason] thread #N not in …`       → kthread spin signal
  * netch phase transitions (Active / recycle)   → session lifecycle
  * `create_process_v2: spawned`                 → loader iteration boundaries

A bucket of 10 ms is dense enough to see individual TCP segments
(MSS=1446 → multiple per ms at link rate) without making the CSV
ridiculously long. Tweak with `--bucket-ms` if you want coarser/finer
resolution.

Usage
-----
    python3 tools/netch_timeline.py runs/eza-stress0-logs.txt \\
        --csv runs/eza-stress0.timeline.csv --bucket-ms 10

    # Quick eyeball without CSV:
    python3 tools/netch_timeline.py runs/eza-stress0-logs.txt --print

Notes
-----
QEMU-virt's `time` CSR runs at 10 MHz, so each tick = 100 ns and
ticks/10 = microseconds. The script normalizes timestamps relative to
the first event after the start marker so bucket 0 = the start of
useful data.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_START_MARKER = "ort=0xFFFFFFD077E00000"
# kmain trace lines look like: "<ticks>t <LEVEL>: <message>"
# Other shapes (cargo output, e1000_X tracing macros without the
# leading-tick format) are silently skipped.
LINE_RE = re.compile(r"^(\d+)t\s+(\w+):\s+(.*?)\s*$")

# (label, regex, group_for_byte_count_or_None).
# Order matters: first match wins, so put more-specific patterns first.
# byte-count classes accumulate the captured integer; everything else
# increments by 1.
CLASSES: list[tuple[str, re.Pattern[str], int | None]] = [
    ("rx_enqueue_bytes", re.compile(r"rx buffer: enqueueing (\d+) octets"), 1),
    ("rx_dequeue_bytes", re.compile(r"rx buffer: dequeueing (\d+) octets"), 1),
    ("tx_enqueue_bytes", re.compile(r"tx buffer: enqueueing (\d+) octets"), 1),
    ("tx_dequeue_bytes", re.compile(r"tx buffer: dequeueing (\d+) octets"), 1),
    ("e1000_irq", re.compile(r"e1000 IRQ:"), None),
    ("ack_sent", re.compile(r"sending ACK\b"), None),
    ("keep_alive_send", re.compile(r"sending a keep-alive"), None),
    ("keep_alive_expired", re.compile(r"keep-alive timer expired"), None),
    ("delayed_ack_force", re.compile(r"delayed ack timer already (started|force-expired)"), None),
    ("delayed_ack_start", re.compile(r"starting delayed ack timer"), None),
    ("delayed_ack_stop", re.compile(r"stop delayed ack timer"), None),
    ("rtte_sample", re.compile(r"rtte: sample="), None),
    ("set_wake_skip", re.compile(r"\[set_wake_reason\] thread #\d+ not in"), None),
    ("netch_peer_connected", re.compile(r"peer connected addr="), None),
    ("netch_handshake_complete", re.compile(r"handshake complete peer="), None),
    ("netch_disengage", re.compile(r"disengage edge"), None),
    ("netch_recycling", re.compile(r"drain complete, recycling"), None),
    ("netch_armed_listen", re.compile(r"armed listen\("), None),
    ("netch_engaged", re.compile(r"engaged 0->1"), None),
    ("create_process", re.compile(r"create_process(?:_v2|_ex)?: spawned"), None),
    ("nc_creation_req", re.compile(r"handling nc creation req"), None),
    ("nc_close_req", re.compile(r"handling close req"), None),
    ("smoltcp_state_change", re.compile(r"^state=[A-Z\-]+=>"), None),
    ("assembler", re.compile(r"^assembler:"), None),
]
CLASS_NAMES = [c[0] for c in CLASSES]


def parse_log(
    path: Path, start_marker: str
) -> tuple[int, list[tuple[int, str, str, str]]]:
    """Return (base_tick, events). Events are (tick, level, msg, raw_line)
    tuples starting from the first marker-line onwards (inclusive).
    `raw_line` is the original log line (newline-stripped) — needed by
    `--logs-out` to dump original lines per bucket. base_tick is the
    tick value of the first event so callers can normalize."""
    events: list[tuple[int, str, str, str]] = []
    base_tick: int | None = None
    started = False
    with open(path) as f:
        for line in f:
            if not started:
                if start_marker in line:
                    started = True
                else:
                    continue
            stripped = line.rstrip("\n")
            m = LINE_RE.match(stripped)
            if not m:
                continue
            tick = int(m.group(1))
            if base_tick is None:
                base_tick = tick
            events.append((tick, m.group(2), m.group(3), stripped))
    if base_tick is None:
        base_tick = 0
    return base_tick, events


def bucketize(
    events: list[tuple[int, str, str, str]],
    base_tick: int,
    bucket_ticks: int,
) -> dict[int, dict[str, int]]:
    """Group events into time buckets and accumulate per-class counts."""
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tick, _level, msg, _raw in events:
        b = (tick - base_tick) // bucket_ticks
        for label, rx, byte_group in CLASSES:
            m = rx.search(msg)
            if m is None:
                continue
            if byte_group is not None:
                buckets[b][label] += int(m.group(byte_group))
            else:
                buckets[b][label] += 1
            break
    return buckets


def write_logs_out(
    out_path: Path,
    events: list[tuple[int, str, str, str]],
    base_tick: int,
    bucket_ticks: int,
    bucket_ms: float,
) -> None:
    """Dump raw log lines tagged with their bucket index, so a CSV row
    that looks interesting can be cross-referenced against the original
    serial output. Tag format is grep-friendly:

        [bucket=853 t=42650.0ms] <original line>

    Use `grep -E '^\\[bucket=853 '` to pull a bucket's lines back out."""
    with open(out_path, "w") as f:
        for tick, _level, _msg, raw in events:
            b = (tick - base_tick) // bucket_ticks
            t_ms = b * bucket_ms
            f.write(f"[bucket={b} t={t_ms:.1f}ms] {raw}\n")


def write_csv(
    out_path: Path,
    buckets: dict[int, dict[str, int]],
    bucket_ms: float,
) -> int:
    fieldnames = ["t_ms"] + CLASS_NAMES + ["any_event"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in sorted(buckets):
            row: dict[str, float | int] = {"t_ms": round(b * bucket_ms, 3)}
            total = 0
            for c in CLASS_NAMES:
                v = buckets[b].get(c, 0)
                row[c] = v
                # `any_event` is the count of trace lines that matched
                # *any* class — a coarse "is the kernel actually doing
                # something this bucket?" signal, useful when scanning
                # the CSV for quiet periods.
                if "bytes" not in c:
                    total += v
            row["any_event"] = total
            w.writerow(row)
    return len(buckets)


def fmt_throughput(bytes_in_bucket: int, bucket_ms: float) -> str:
    """Render bucket-bytes as `<KiB this bucket> (<mbit/s rate>mbit/s)`.

    Mbit/s is the wire-throughput convention (decimal megabits per
    second), not MiB/s — a sample reading 16.4mbit/s ≈ 2 MiB/s. Pairing
    a bucket-local size with a normalized rate lets the eye distinguish
    "200 KiB landed in this single bucket" (one batch) from "200 KiB/s
    sustained over many buckets" (steady-state) at a glance.
    """
    if bytes_in_bucket == 0:
        return ""
    kib = bytes_in_bucket / 1024.0
    # bucket_ms is ms → bucket_ms/1000 = sec; *8 for bits; /1e6 for mbit
    mbits = (bytes_in_bucket * 8.0) / (bucket_ms / 1000.0) / 1_000_000.0
    return f"{kib:.1f}KiB ({mbits:.1f}mbit/s)"


def print_timeline(
    buckets: dict[int, dict[str, int]],
    bucket_ms: float,
    only_active: bool,
) -> None:
    """Compact human-readable timeline. Each rx/tx column shows
    `<KiB-this-bucket> (<mbit/s>mbit/s)` — local volume + normalized
    wire rate, paired so single-bucket bursts read differently from
    sustained transfers.
    """
    rx_w = tx_w = 22  # width budget for "1234.5KiB (1234.5mbit/s)"
    print(
        f"{'t_ms':>10}  {'rx':>{rx_w}}  {'tx':>{tx_w}}  "
        f"{'irq':>5}  {'ack':>5}  {'ka':>4}  {'wake_skip':>9}  {'events':>6}"
    )
    print("-" * (10 + 2 + rx_w + 2 + tx_w + 2 + 5 + 2 + 5 + 2 + 4 + 2 + 9 + 2 + 6))
    for b in sorted(buckets):
        events = buckets[b]
        rx_b = events.get("rx_enqueue_bytes", 0)
        tx_b = events.get("tx_enqueue_bytes", 0)
        irq = events.get("e1000_irq", 0)
        ack = events.get("ack_sent", 0)
        ka = events.get("keep_alive_send", 0) + events.get("keep_alive_expired", 0)
        wake_skip = events.get("set_wake_skip", 0)
        any_ev = sum(
            v for k, v in events.items() if "bytes" not in k
        )
        if only_active and any_ev == 0 and rx_b == 0 and tx_b == 0:
            continue
        rx_str = fmt_throughput(rx_b, bucket_ms)
        tx_str = fmt_throughput(tx_b, bucket_ms)
        t_ms = b * bucket_ms
        print(
            f"{t_ms:>10.1f}  {rx_str:>{rx_w}}  {tx_str:>{tx_w}}  "
            f"{irq:>5}  {ack:>5}  {ka:>4}  {wake_skip:>9}  {any_ev:>6}"
        )


def write_plots(
    out_dir: Path,
    buckets: dict[int, dict[str, int]],
    bucket_ms: float,
) -> int:
    """Render PNGs into `out_dir`. Returns the number of plots written.
    Falls back to text summary on stderr if matplotlib isn't available
    — keeps the script usable in venv-less environments."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[plot] matplotlib not available; skipping. "
            "Install with `pip install matplotlib` or use the CSV output.",
            file=sys.stderr,
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_keys = sorted(buckets)
    if not sorted_keys:
        return 0
    t_ms = [b * bucket_ms for b in sorted_keys]
    rate_div = bucket_ms / 1000.0  # bucket → seconds for rate conversion

    def series(name: str) -> list[float]:
        return [buckets[b].get(name, 0) for b in sorted_keys]

    def per_sec(name: str) -> list[float]:
        return [buckets[b].get(name, 0) / rate_div for b in sorted_keys]

    def kib_per_sec(name: str) -> list[float]:
        return [(buckets[b].get(name, 0) / 1024.0) / rate_div for b in sorted_keys]

    n_written = 0

    # ── 1. Throughput: rx + tx bytes per second over time. ────────
    # Primary diagnostic for "did things speed up and when?". Both
    # directions on one axis so an rx burst followed by a tx burst
    # (e.g. eza payload landing → metricd flushing samples) reads as
    # a clear handoff rather than two separate plots to flip between.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, kib_per_sec("rx_enqueue_bytes"), label="rx (KiB/s)", linewidth=1)
    ax.plot(t_ms, kib_per_sec("tx_enqueue_bytes"), label="tx (KiB/s)", linewidth=1)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("KiB/s")
    ax.set_title(f"netch throughput (bucket={bucket_ms} ms)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput.png", dpi=120)
    plt.close(fig)
    n_written += 1

    # ── 2. Wake/IRQ activity — diagnostic for "is k_net spinning?" ──
    # set_wake_skip is the canonical "spurious wake" signal: if that's
    # high *and* throughput is low, k_net is spinning without making
    # progress. e1000_irq tells us how often the device is poking us.
    # ack_sent + keep_alive_send tell us if smoltcp is generating tx.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, per_sec("e1000_irq"), label="e1000 IRQ/s", linewidth=1)
    ax.plot(t_ms, per_sec("ack_sent"), label="ACK/s", linewidth=1)
    ax.plot(t_ms, per_sec("set_wake_skip"), label="wake_skip/s", linewidth=1)
    ax.plot(t_ms, per_sec("keep_alive_send"), label="keep-alive/s", linewidth=1)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("events/s")
    ax.set_title(f"wake / IRQ / ACK activity (bucket={bucket_ms} ms)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "wake-activity.png", dpi=120)
    plt.close(fig)
    n_written += 1

    # ── 3. Netch session lifecycle — markers for boundary events. ──
    # Plotted as scatter so each session boundary is a clear discrete
    # point to align against the throughput plot. `create_process`
    # markers tag the iteration boundary in eza_stress runs.
    fig, ax = plt.subplots(figsize=(12, 4))
    lifecycle_events = [
        ("netch_engaged", "engaged", "tab:blue"),
        ("netch_armed_listen", "armed listen", "tab:cyan"),
        ("netch_peer_connected", "peer connect", "tab:green"),
        ("netch_disengage", "disengage", "tab:orange"),
        ("netch_recycling", "recycle", "tab:red"),
        ("create_process", "create_process", "tab:purple"),
    ]
    for i, (cls, label, color) in enumerate(lifecycle_events):
        s = series(cls)
        xs = [t for t, v in zip(t_ms, s) if v > 0]
        ys = [i] * len(xs)
        ax.scatter(xs, ys, label=label, color=color, s=20, marker="|")
    ax.set_yticks(range(len(lifecycle_events)))
    ax.set_yticklabels([e[1] for e in lifecycle_events])
    ax.set_xlabel("time (ms)")
    ax.set_title(f"netch lifecycle events (bucket={bucket_ms} ms)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "lifecycle.png", dpi=120)
    plt.close(fig)
    n_written += 1

    # ── 4. Combined dashboard — throughput + wake_skip on same x. ──
    # Most useful single-glance plot for the slow→fast question:
    # if `wake_skip` drops at the same time throughput jumps, k_net
    # was spinning beforehand and stopped (e.g. starvation cleared).
    # If throughput jumps without a wake_skip change, the trigger is
    # elsewhere (delayed-ack timer, congestion window, recycle).
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(t_ms, kib_per_sec("rx_enqueue_bytes"), label="rx", linewidth=1)
    axs[0].plot(t_ms, kib_per_sec("tx_enqueue_bytes"), label="tx", linewidth=1)
    axs[0].set_ylabel("KiB/s")
    axs[0].set_title("throughput")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(t_ms, per_sec("set_wake_skip"), color="tab:red", linewidth=1)
    axs[1].set_ylabel("wake_skip/s")
    axs[1].set_title("kthread spurious-wake rate")
    axs[1].grid(True, alpha=0.3)
    axs[2].plot(t_ms, per_sec("e1000_irq"), color="tab:blue", label="IRQ", linewidth=1)
    axs[2].plot(t_ms, per_sec("ack_sent"), color="tab:green", label="ACK", linewidth=1)
    axs[2].set_xlabel("time (ms)")
    axs[2].set_ylabel("events/s")
    axs[2].set_title("device + ACK rate")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "dashboard.png", dpi=120)
    plt.close(fig)
    n_written += 1

    return n_written


def detect_phase_transition_for_series(
    buckets: dict[int, dict[str, int]],
    sorted_keys: list[int],
    series_name: str,
    label: str,
    bucket_ms: float,
    window_buckets: int,
) -> None:
    """Heuristic phase finder for a single byte-counter series. Compute
    a rolling-mean throughput over `window_buckets` buckets; report the
    bucket where throughput jumps by ≥4× over the prior window mean.
    Direction-agnostic — caller picks the series (rx or tx)."""
    series = [buckets[b].get(series_name, 0) for b in sorted_keys]
    if len(series) < 2 * window_buckets:
        print(
            f"\n[phase:{label}] not enough data for transition detection "
            f"(need {2 * window_buckets} buckets, have {len(series)})",
            file=sys.stderr,
        )
        return
    window_s = window_buckets * bucket_ms / 1000.0
    print(
        f"\n[phase:{label}] looking for ≥4× throughput jump over "
        f"{window_buckets}-bucket / {window_s:.2f}s window "
        f"(rates in KiB/s)...",
        file=sys.stderr,
    )
    # Suppress consecutive matches: once a window crosses the threshold,
    # subsequent buckets in the *same* sustained transition will also
    # match (the window slid by 1, the averages barely changed). Require
    # the window to have fully advanced past the last hit before we
    # report again — i.e. gap ≥ window_buckets.
    found_any = False
    last_reported_idx = -1_000_000
    rate_factor_kib_s = 1.0 / 1024.0 / (bucket_ms / 1000.0)
    for i in range(window_buckets, len(series) - window_buckets + 1):
        prev = sum(series[i - window_buckets : i]) / window_buckets
        cur = sum(series[i : i + window_buckets]) / window_buckets
        if prev > 0 and cur >= 4 * prev and cur > 1024:  # ignore noise
            if i - last_reported_idx < window_buckets:
                continue
            t_ms = sorted_keys[i] * bucket_ms
            print(
                f"  t={t_ms:.0f} ms  "
                f"prev={prev * rate_factor_kib_s:.1f} KiB/s  "
                f"new={cur * rate_factor_kib_s:.1f} KiB/s  "
                f"({cur / max(prev, 1):.1f}×)",
                file=sys.stderr,
            )
            found_any = True
            last_reported_idx = i
    if not found_any:
        print("  no transition detected", file=sys.stderr)


def detect_phase_transition(
    buckets: dict[int, dict[str, int]],
    bucket_ms: float,
    window_buckets: int,
) -> None:
    """Run phase-transition detection on both rx and tx byte streams.
    Each direction is searched independently — a host-bound burst
    (metricd flushing samples) and a guest-bound burst (eza payload
    landing) get their own report so they don't get tangled."""
    if not buckets:
        return
    sorted_keys = sorted(buckets)
    detect_phase_transition_for_series(
        buckets, sorted_keys, "rx_enqueue_bytes", "rx", bucket_ms, window_buckets
    )
    detect_phase_transition_for_series(
        buckets, sorted_keys, "tx_enqueue_bytes", "tx", bucket_ms, window_buckets
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="kmain serial log to parse")
    ap.add_argument(
        "--start-marker",
        default=DEFAULT_START_MARKER,
        help="parse begins at the line containing this substring "
        f"(default {DEFAULT_START_MARKER!r}, the first kernel `info!` "
        "after boot)",
    )
    ap.add_argument(
        "--bucket-ms",
        type=float,
        default=10.0,
        help="time bucket width in milliseconds (default 10)",
    )
    ap.add_argument("--csv", help="write the full timeline CSV here")
    ap.add_argument(
        "--print",
        action="store_true",
        help="print a compact human-readable timeline to stdout",
    )
    ap.add_argument(
        "--only-active",
        action="store_true",
        help="when --print, suppress empty buckets",
    )
    ap.add_argument(
        "--phase-window",
        type=int,
        default=20,
        help="rolling-mean window size (in buckets) for phase-transition "
        "detection (default 20)",
    )
    ap.add_argument(
        "--plot",
        metavar="DIR",
        help="write PNG plots into DIR (throughput, wake-activity, "
        "lifecycle, dashboard). Falls back to text summary if matplotlib "
        "isn't installed.",
    )
    ap.add_argument(
        "--logs-out",
        metavar="PATH",
        help="dump every parsed log line tagged with its bucket index "
        "to PATH. Useful for cross-referencing CSV rows: "
        "`grep '^\\[bucket=853 ' PATH` returns that bucket's raw lines.",
    )
    args = ap.parse_args()

    bucket_ticks = int(args.bucket_ms * 10_000)  # 10 MHz timer → 10 000 ticks/ms
    base_tick, events = parse_log(Path(args.path), args.start_marker)
    if not events:
        print(
            f"no events found after marker {args.start_marker!r} in {args.path}",
            file=sys.stderr,
        )
        return 1
    print(
        f"parsed {len(events)} events spanning "
        f"{(events[-1][0] - base_tick) / 10_000_000:.2f} s",
        file=sys.stderr,
    )

    buckets = bucketize(events, base_tick, bucket_ticks)
    print(f"bucketized into {len(buckets)} non-empty buckets "
          f"of {args.bucket_ms} ms each", file=sys.stderr)

    if args.csv:
        n = write_csv(Path(args.csv), buckets, args.bucket_ms)
        print(f"wrote {n} rows to {args.csv}", file=sys.stderr)

    if args.logs_out:
        write_logs_out(
            Path(args.logs_out), events, base_tick, bucket_ticks, args.bucket_ms
        )
        print(
            f"wrote {len(events)} bucket-tagged lines to {args.logs_out}",
            file=sys.stderr,
        )

    if args.plot:
        n = write_plots(Path(args.plot), buckets, args.bucket_ms)
        if n > 0:
            print(f"wrote {n} plots to {args.plot}", file=sys.stderr)

    if args.print:
        print_timeline(buckets, args.bucket_ms, args.only_active)

    detect_phase_transition(buckets, args.bucket_ms, args.phase_window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
