#!/usr/bin/env python3
"""Render plots from an orbit-metricd CSV captured by orbit_metric_logger.py.

Usage:
    python3 tools/orbit_metric_plot.py <csv> [--out <dir>] [--top-n <int>]
                                              [--syscalls a,b,c] [--show]

Plots produced (one PNG each, dropped into <out>):

  syscall-rate.png   — per-syscall calls/sec over time, top-N by total count
  syscall-mean.png   — per-syscall mean service time (µs) over time, top-N
  syscall-max.png    — per-syscall max service time (µs) over time, top-N
  hart-buckets.png   — system-wide hart-time partition (user/kernel/sched/idle)
                       as stacked area, ratios over time
  wake-queue.png     — wake_queue depth peak and drops counter over time

The script computes per-sample deltas internally (orbit-metricd emits
cumulative-since-boot counters; deltas-per-sample give rates/intervals).
With `--top-n N` the syscall plots only show the N flavors with the
highest cumulative count over the whole capture (everything else is
noise on a log-scale plot anyway). Override with `--syscalls` if you
care about specific flavors regardless of activity.

If matplotlib isn't on PATH, the script falls back to writing a small
text summary table to stderr — useful as a sanity check before sinking
time into a plotting venv.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


# 10 MHz `time` CSR on qemu-virt → 10 ticks per microsecond.
TICKS_PER_US = 10
TICKS_PER_S = TICKS_PER_US * 1_000_000


def load_csv(path: str) -> tuple[list[str], list[dict]]:
    """Read the CSV into a list of dicts, coercing all values to int.

    Empty cells (forward-compat: kernel emitted a field newer logger
    doesn't know, or vice versa) become 0.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for r in reader:
            # All known fields are integer counters. csv.DictReader
            # gives strings; coerce here so downstream math is clean.
            for k, v in list(r.items()):
                try:
                    r[k] = int(v) if v != "" else 0
                except ValueError:
                    # Leave non-int strings alone (future text fields).
                    pass
            rows.append(r)
    return fieldnames, rows


def syscall_flavors(fieldnames: list[str]) -> list[str]:
    """Pull the unique flavor names from columns shaped `sys.<name>.<metric>`."""
    seen: set[str] = set()
    for col in fieldnames:
        if col.startswith("sys.") and col.endswith(".count"):
            seen.add(col[len("sys."):-len(".count")])
    return sorted(seen)


def pick_top_flavors(rows: list[dict], flavors: list[str], n: int) -> list[str]:
    """Top-N flavors by terminal cumulative count (last row's count column)."""
    if not rows or n >= len(flavors):
        return flavors
    last = rows[-1]
    scored = [(last.get(f"sys.{f}.count", 0), f) for f in flavors]
    scored.sort(reverse=True)
    return [f for _, f in scored[:n]]


def deltas(rows: list[dict], col: str) -> list[float]:
    """Per-sample deltas of a cumulative counter. First sample → 0."""
    out = [0.0]
    for i in range(1, len(rows)):
        out.append(float(rows[i].get(col, 0) - rows[i - 1].get(col, 0)))
    return out


def t_seconds(rows: list[dict]) -> list[float]:
    """Convert `t_orbit` ticks to seconds, anchored at the first sample."""
    if not rows:
        return []
    base = rows[0].get("t_orbit", 0)
    return [(r.get("t_orbit", 0) - base) / TICKS_PER_S for r in rows]


def text_summary(rows: list[dict], flavors: list[str]) -> None:
    """Tiny text fallback when matplotlib isn't around — prints a table."""
    if not rows:
        print("(no rows)", file=sys.stderr)
        return
    last = rows[-1]
    duration_s = (last.get("t_orbit", 0) - rows[0].get("t_orbit", 0)) / TICKS_PER_S
    print(
        f"{len(rows)} samples over {duration_s:.2f} s "
        f"(seq {rows[0].get('seq', 0)} .. {last.get('seq', 0)})",
        file=sys.stderr,
    )
    print(
        f"  wake_queue peak={last.get('proc.wake_queue_peak', 0)} "
        f"drops={last.get('proc.wake_queue_drops', 0)} "
        f"cap={last.get('proc.wake_queue_capacity', 0)}",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(f"{'syscall':<24}{'count':>10}{'mean_us':>10}{'max_us':>10}",
          file=sys.stderr)
    by_count = sorted(
        flavors,
        key=lambda f: last.get(f"sys.{f}.count", 0),
        reverse=True,
    )
    for f in by_count:
        c = last.get(f"sys.{f}.count", 0)
        if c == 0:
            continue
        tot = last.get(f"sys.{f}.total_ticks", 0)
        mx = last.get(f"sys.{f}.max_ticks", 0)
        mean_us = (tot / c) / TICKS_PER_US if c else 0.0
        max_us = mx / TICKS_PER_US
        print(f"{f:<24}{c:>10}{mean_us:>10.1f}{max_us:>10.1f}",
              file=sys.stderr)


def plot_all(rows: list[dict], flavors: list[str], out_dir: str,
             show: bool) -> None:
    """Render the standard plot set into out_dir using matplotlib."""
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t = t_seconds(rows)

    # ---- per-syscall calls/sec rate over time (top-N by activity) ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for f in flavors:
        d_count = deltas(rows, f"sys.{f}.count")
        d_t = [t[i] - t[i - 1] if i > 0 else 1.0 for i in range(len(t))]
        rate = [(d_count[i] / d_t[i]) if d_t[i] > 0 else 0.0
                for i in range(len(t))]
        if max(rate, default=0.0) > 0:
            ax.plot(t, rate, label=f, linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("calls / sec")
    ax.set_title("syscall rate (Δcount / Δt)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "syscall-rate.png"), dpi=120)
    plt.close(fig)

    # ---- per-syscall mean service time (µs) per interval -----------
    fig, ax = plt.subplots(figsize=(11, 6))
    for f in flavors:
        d_count = deltas(rows, f"sys.{f}.count")
        d_total = deltas(rows, f"sys.{f}.total_ticks")
        mean_us = [
            (d_total[i] / d_count[i]) / TICKS_PER_US if d_count[i] > 0 else None
            for i in range(len(t))
        ]
        # Skip flavors with zero activity over the whole window.
        if not any(v is not None for v in mean_us):
            continue
        # Replace None gaps with the previous valid sample to keep the
        # line continuous (idle intervals look like flat sections).
        last_v = 0.0
        ys = []
        for v in mean_us:
            if v is not None:
                last_v = v
            ys.append(last_v)
        ax.plot(t, ys, label=f, linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean service time (µs)")
    ax.set_title("syscall mean service time (Δtotal_ticks / Δcount)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "syscall-mean.png"), dpi=120)
    plt.close(fig)

    # ---- per-syscall running max service time (µs) -----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for f in flavors:
        ys = [r.get(f"sys.{f}.max_ticks", 0) / TICKS_PER_US for r in rows]
        if max(ys, default=0.0) > 0:
            ax.plot(t, ys, label=f, linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("max service time (µs, cumulative)")
    ax.set_title("syscall worst-case service time (max_ticks)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "syscall-max.png"), dpi=120)
    plt.close(fig)

    # ---- system-wide hart-time partition (stacked area) ------------
    bucket_cols = [
        ("user", "proc.hart_user_ticks"),
        ("kernel", "proc.hart_kernel_ticks"),
        ("scheduler", "proc.hart_scheduler_ticks"),
        ("idle", "proc.hart_idle_ticks"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    series = {}
    for name, col in bucket_cols:
        d = deltas(rows, col)
        series[name] = d
    totals = [sum(series[name][i] for name, _ in bucket_cols)
              for i in range(len(t))]
    ratios = {
        name: [
            (series[name][i] / totals[i]) if totals[i] > 0 else 0.0
            for i in range(len(t))
        ]
        for name, _ in bucket_cols
    }
    ax.stackplot(t,
                 [ratios[name] for name, _ in bucket_cols],
                 labels=[name for name, _ in bucket_cols],
                 alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fraction of hart-time")
    ax.set_title("hart-time partition (Δ buckets per sample)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hart-buckets.png"), dpi=120)
    plt.close(fig)

    # ---- wake_queue telemetry --------------------------------------
    fig, (ax_peak, ax_drops) = plt.subplots(2, 1, figsize=(11, 6),
                                            sharex=True)
    peak = [r.get("proc.wake_queue_peak", 0) for r in rows]
    drops = [r.get("proc.wake_queue_drops", 0) for r in rows]
    cap = rows[-1].get("proc.wake_queue_capacity", 0) if rows else 0
    ax_peak.plot(t, peak, label="peak", linewidth=1.5)
    ax_peak.axhline(cap, color="red", linestyle="--", linewidth=1,
                    label=f"cap={cap}")
    ax_peak.set_ylabel("WAKE_QUEUE depth")
    ax_peak.set_title("wake_queue telemetry")
    ax_peak.legend(loc="upper right", fontsize=9)
    ax_peak.grid(alpha=0.3)
    ax_drops.plot(t, drops, label="drops (cumulative)",
                  color="orange", linewidth=1.5)
    ax_drops.set_xlabel("time (s)")
    ax_drops.set_ylabel("drop count")
    ax_drops.legend(loc="upper right", fontsize=9)
    ax_drops.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wake-queue.png"), dpi=120)
    if show:
        plt.show()
    plt.close(fig)

    print(f"wrote plots → {out_dir}/", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="path to a CSV captured by orbit_metric_logger")
    ap.add_argument("--out", default=None,
                    help="output dir for PNGs (default: <csv>.plots/)")
    ap.add_argument("--top-n", type=int, default=10,
                    help="show only the top-N syscalls by total count "
                         "in syscall-* plots (default 10)")
    ap.add_argument("--syscalls", default=None,
                    help="comma-separated explicit flavor list, "
                         "overrides --top-n")
    ap.add_argument("--show", action="store_true",
                    help="open the last plot interactively (requires "
                         "matplotlib's interactive backend)")
    args = ap.parse_args()

    fieldnames, rows = load_csv(args.csv)
    flavors_all = syscall_flavors(fieldnames)
    if args.syscalls:
        flavors = [f.strip() for f in args.syscalls.split(",") if f.strip()]
    else:
        flavors = pick_top_flavors(rows, flavors_all, args.top_n)

    out_dir = args.out or args.csv + ".plots"
    try:
        plot_all(rows, flavors, out_dir, args.show)
    except ImportError as e:
        print(
            f"matplotlib unavailable ({e}); falling back to text summary.\n"
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        text_summary(rows, flavors_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
