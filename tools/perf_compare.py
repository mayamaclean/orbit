#!/usr/bin/env python3
"""Diff per-syscall avg/max ticks across two metricd CSV captures.

Used for post-migration verification: pick a "before" CSV (e.g. the
post-mmap baseline) and an "after" CSV (e.g. the post-Phase-6 capture
or the latest run with all fixes in place), and the script prints a
table showing which syscalls got faster, slower, or stayed flat.

Why
---
Phase 8 of the CompletionHandle migration is "verify the migration
delivered the latency wins we expected." metricd's CSV gives us
cumulative-since-boot counters (count, total_ticks, max_ticks) per
syscall flavor. Final-row values are the lifetime totals for that
capture. avg = total_ticks / count, max = max_ticks (already a peak
across the run). Comparing two final-row snapshots tells us how the
service-time distribution shifted between configurations.

Caveats
-------
* Captures must come from comparable workloads — a baseline that ran
  for 30 s and a current run that did 100 eza iterations aren't
  directly comparable. Look at counts to gauge similar workload
  intensity.
* `max_ticks` is a single-sample peak, not a percentile — one
  preemption-induced spike skews it. Look at avg first; max as a
  sanity check that the tail didn't regress dramatically.
* Cumulative counters don't decay. If a process / metricd kept
  running through a hangs-and-recovers period, that period's slow
  syscalls are baked into the average.
* Some syscalls only fire for specific binaries (e.g. `fb_present`
  only when a surface-mode app like orbit-top-std runs); a row
  missing from one capture but present in the other is "not
  exercised" rather than "got infinitely fast/slow."

Usage
-----
    python3 tools/perf_compare.py BASELINE.csv CURRENT.csv

    # Filter to a specific syscall:
    python3 tools/perf_compare.py BASELINE.csv CURRENT.csv --grep mmap

    # Only show syscalls with ≥10% delta in either direction:
    python3 tools/perf_compare.py BASELINE.csv CURRENT.csv --threshold 0.1

10 MHz `time` CSR convention (qemu-virt): ticks/10 = microseconds.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# 10 MHz CSR → 10 ticks per microsecond.
TICKS_PER_US = 10


def load_final_row(path: Path) -> dict[str, int]:
    """Return the last row of `path`'s CSV as a dict, with values
    coerced to int. Missing/empty cells become 0."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        last: dict[str, str] = {}
        for row in reader:
            last = row
        if not last:
            print(f"error: {path} has no data rows", file=sys.stderr)
            sys.exit(2)
    out: dict[str, int] = {}
    for k, v in last.items():
        if v == "" or v is None:
            out[k] = 0
            continue
        try:
            out[k] = int(v)
        except ValueError:
            # Non-int strings (future text fields): drop. We only need
            # numeric counters.
            pass
    return out


def syscall_summary(row: dict[str, int]) -> dict[str, dict[str, int]]:
    """Group `sys.<name>.{count,total_ticks,max_ticks}` columns into
    per-flavor dicts. Returns `{flavor: {count, total_ticks, max_ticks}}`."""
    out: dict[str, dict[str, int]] = {}
    for k, v in row.items():
        m = re.match(r"^sys\.(.+)\.(count|total_ticks|max_ticks)$", k)
        if m is None:
            continue
        flavor, field = m.group(1), m.group(2)
        out.setdefault(flavor, {"count": 0, "total_ticks": 0, "max_ticks": 0})
        out[flavor][field] = v
    return out


def fmt_us(ticks: float | int) -> str:
    """Render ticks as µs to one decimal."""
    if ticks == 0:
        return "—"
    return f"{ticks / TICKS_PER_US:.1f}"


def fmt_delta_pct(before: float, after: float) -> str:
    """Render the percentage change `before → after`. Negative = faster
    (avg/max went down); positive = slower."""
    if before == 0 and after == 0:
        return ""
    if before == 0:
        return "(new)"
    pct = (after - before) / before * 100.0
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", help="metricd CSV captured before the change")
    ap.add_argument("current", help="metricd CSV captured after the change")
    ap.add_argument(
        "--grep",
        help="case-insensitive substring filter on syscall name",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="suppress rows whose avg-delta percentage is below this "
        "(absolute) threshold. Default 0 = show all rows. Common: 0.1 "
        "for only 10pct-or-bigger changes.",
    )
    ap.add_argument(
        "--sort",
        choices=("name", "avg-delta", "max-delta", "count"),
        default="name",
        help="row sort order (default: alphabetical by syscall name)",
    )
    args = ap.parse_args()

    base_row = load_final_row(Path(args.baseline))
    cur_row = load_final_row(Path(args.current))
    base = syscall_summary(base_row)
    cur = syscall_summary(cur_row)
    flavors = sorted(set(base.keys()) | set(cur.keys()))

    rows: list[tuple[str, int, int, float, float, int, int]] = []
    for flavor in flavors:
        if args.grep and args.grep.lower() not in flavor.lower():
            continue
        b = base.get(flavor, {"count": 0, "total_ticks": 0, "max_ticks": 0})
        c = cur.get(flavor, {"count": 0, "total_ticks": 0, "max_ticks": 0})

        b_count = b["count"]
        c_count = c["count"]
        b_avg = b["total_ticks"] / b_count if b_count else 0.0
        c_avg = c["total_ticks"] / c_count if c_count else 0.0
        b_max = b["max_ticks"]
        c_max = c["max_ticks"]

        # Skip rows that are entirely zero — happens when filter +
        # neither capture exercised this syscall.
        if b_count == 0 and c_count == 0:
            continue

        if args.threshold > 0 and b_avg > 0:
            pct_delta = abs(c_avg - b_avg) / b_avg
            if pct_delta < args.threshold:
                continue

        rows.append((flavor, b_count, c_count, b_avg, c_avg, b_max, c_max))

    sort_keys = {
        "name": lambda r: r[0],
        # Negative because we want biggest improvements first when
        # sorting by delta. Default tiebreaker stays alphabetical.
        "avg-delta": lambda r: (
            -((r[4] - r[3]) / r[3]) if r[3] > 0 else 0.0,
            r[0],
        ),
        "max-delta": lambda r: (
            -((r[6] - r[5]) / r[5]) if r[5] > 0 else 0.0,
            r[0],
        ),
        "count": lambda r: (-(r[1] + r[2]), r[0]),
    }
    rows.sort(key=sort_keys[args.sort])

    if not rows:
        print("no syscalls matched filter", file=sys.stderr)
        return 1

    # Column layout. Pad to longest flavor name.
    name_w = max(len(r[0]) for r in rows)
    name_w = max(name_w, len("syscall"))

    print(
        f"{'syscall':<{name_w}}  "
        f"{'count_b':>9}  {'count_a':>9}  "
        f"{'avg_b_us':>9}  {'avg_a_us':>9}  {'Δavg':>7}  "
        f"{'max_b_us':>9}  {'max_a_us':>9}  {'Δmax':>7}"
    )
    print("-" * (name_w + 2 + 9 + 2 + 9 + 2 + 9 + 2 + 9 + 2 + 7 + 2 + 9 + 2 + 9 + 2 + 7))
    for flavor, b_count, c_count, b_avg, c_avg, b_max, c_max in rows:
        print(
            f"{flavor:<{name_w}}  "
            f"{b_count:>9}  {c_count:>9}  "
            f"{fmt_us(b_avg):>9}  {fmt_us(c_avg):>9}  {fmt_delta_pct(b_avg, c_avg):>7}  "
            f"{fmt_us(b_max):>9}  {fmt_us(c_max):>9}  {fmt_delta_pct(b_max, c_max):>7}"
        )

    # Quick proc.* summary at the bottom — context for interpreting
    # the per-syscall numbers (was the workload comparable? did the
    # wake queue saturate?).
    print()
    print("--- proc context ---")
    proc_keys = (
        "proc.context_switches",
        "proc.syscalls",
        "proc.syscall_ticks",
        "proc.hart_user_ticks",
        "proc.hart_kernel_ticks",
        "proc.hart_idle_ticks",
        "proc.wake_queue_peak",
        "proc.wake_queue_drops",
    )
    print(f"{'metric':<28}  {'baseline':>14}  {'current':>14}")
    print("-" * 60)
    for k in proc_keys:
        b = base_row.get(k, 0)
        c = cur_row.get(k, 0)
        if b == 0 and c == 0:
            continue
        print(f"{k:<28}  {b:>14}  {c:>14}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
