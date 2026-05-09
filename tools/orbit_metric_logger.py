#!/usr/bin/env python3
"""Connect to orbit-metricd, log JSONL samples to a CSV.

orbit-metricd binds 0.0.0.0:7800 inside the guest; QEMU's user-net
forwards `localhost:7800` on the host. This script connects to the
forwarded port, reads newline-delimited JSON samples, flattens them,
and writes a CSV with one row per sample.

Usage:
    python3 tools/orbit_metric_logger.py [--host HOST] [--port PORT]
                                         [--csv FILE] [--quiet]

Pandas-friendly columns: `t_orbit`, `t_host` (host monotonic ns),
each ProcessStats field as a top-level column, and per-syscall
columns named `<flavor>.count|total_ticks|max_ticks`. Cumulative
counters land as-is — compute deltas with `df.diff()` when plotting.

Stop with Ctrl-C; the CSV is flushed on every line so a partial run
is still readable.
"""
import argparse
import csv
import json
import socket
import sys
import time


def run(host: str, port: int, csv_path: str | None, quiet: bool) -> int:
    sock = socket.create_connection((host, port))
    if not quiet:
        print(f"connected to {host}:{port}", file=sys.stderr)

    sock_file = sock.makefile("r", encoding="utf-8", newline="\n")

    csv_writer = None
    csv_file = None
    fieldnames: list[str] | None = None
    sample_count = 0
    dup_count = 0
    last_seq: int | None = None
    t_start = time.monotonic_ns()

    try:
        for line in sock_file:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip malformed line: {e}", file=sys.stderr)
                continue

            # Seq-based dedupe. orbit-metricd's `sample_seq` increments
            # monotonically per process lifetime starting at 1; a sample
            # with seq <= last_seq is a netch / smoltcp retransmit
            # (known transient on first-write-after-listen, see metricd
            # docstring) and should not contribute to the CSV.
            #
            # Backward-compat: pre-seq metricd builds didn't emit the
            # field at all, so .get() returns 0 there and dedupe is a
            # no-op (every sample passes). New metricd builds always
            # emit seq >= 1.
            seq = sample.get("seq", 0)
            if seq and last_seq is not None and seq <= last_seq:
                dup_count += 1
                if not quiet and dup_count <= 5:
                    print(
                        f"dedupe: skipping seq={seq} (last_seq={last_seq})",
                        file=sys.stderr,
                    )
                continue
            if seq:
                last_seq = seq

            row = flatten(sample)
            row["t_host_ns"] = time.monotonic_ns() - t_start

            if csv_path is not None:
                if csv_writer is None:
                    fieldnames = sorted(row.keys())
                    csv_file = open(csv_path, "w", newline="")
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    csv_writer.writeheader()
                else:
                    # Tolerate new fields appearing mid-stream
                    # (forward-compat schema bumps): rewrite the
                    # header is not free, so we stick to the initial
                    # fieldnames and silently drop any later-added
                    # keys. Document in the metricd schema doc that
                    # appending is safe but mid-run additions are
                    # ignored by old loggers.
                    pass
                csv_writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()

            sample_count += 1
            if not quiet and sample_count % 50 == 0:
                print(f"  {sample_count} samples", file=sys.stderr)
    except KeyboardInterrupt:
        if not quiet:
            print(
                f"\nstopped after {sample_count} samples "
                f"({dup_count} dedup'd)",
                file=sys.stderr,
            )
    finally:
        if csv_file is not None:
            csv_file.close()
        sock.close()
    return 0


def flatten(sample: dict) -> dict:
    """Flatten one JSONL sample into a flat dict suitable for a CSV row."""
    row: dict = {
        "seq": sample.get("seq", 0),
        "t_orbit": sample.get("t_orbit", 0),
    }
    proc = sample.get("proc", {}) or {}
    for k, v in proc.items():
        row[f"proc.{k}"] = v
    for entry in sample.get("syscalls", []) or []:
        name = entry.get("name", f"ord{entry.get('ord', '?')}")
        row[f"sys.{name}.count"] = entry.get("count", 0)
        row[f"sys.{name}.total_ticks"] = entry.get("total_ticks", 0)
        row[f"sys.{name}.max_ticks"] = entry.get("max_ticks", 0)
    return row


def raw_dump(host: str, port: int, max_bytes: int) -> int:
    """Bypass JSON parsing — print every byte received as a Python repr.
    Use to diagnose framing issues (duplicate lines, partial lines,
    stray bytes) by showing exactly what's on the wire.
    """
    sock = socket.create_connection((host, port))
    print(f"connected to {host}:{port}, dumping up to {max_bytes} bytes",
          file=sys.stderr)
    total = 0
    try:
        while total < max_bytes:
            chunk = sock.recv(min(4096, max_bytes - total))
            if not chunk:
                print(f"\n--- EOF after {total} bytes ---", file=sys.stderr)
                break
            sys.stdout.write(repr(chunk)[2:-1])  # strip b'' wrapper
            sys.stdout.write(f"\n--- recv {len(chunk)} bytes ---\n")
            sys.stdout.flush()
            total += len(chunk)
    except KeyboardInterrupt:
        print(f"\nstopped after {total} bytes", file=sys.stderr)
    finally:
        sock.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1",
                    help="orbit-metricd host (default 127.0.0.1, "
                         "via QEMU's hostfwd of guest:7800)")
    ap.add_argument("--port", type=int, default=7800)
    ap.add_argument("--csv", default=None,
                    help="write a CSV here; omit to just connect "
                         "and discard (smoke check)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--raw-dump", type=int, metavar="BYTES", default=None,
                    help="diagnostic: print each recv() chunk verbatim "
                         "until BYTES total received, then exit")
    args = ap.parse_args()
    if args.raw_dump is not None:
        return raw_dump(args.host, args.port, args.raw_dump)
    return run(args.host, args.port, args.csv, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
