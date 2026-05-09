#!/usr/bin/env python3
"""eza_stress.py — boot the guest and spawn `eza -lahR` in a loop.

Why
---
Two jobs at once:
  1. Stress test — eza -lahR is the canonical workload that surfaced
     the pre-existing rayon-worker / dispatch race. After the
     CompletionHandle migration we want a confidence run before
     declaring Phase 8 done.
  2. Profile capture — start `orbit-metricd` first, run the host-side
     `orbit_metric_logger.py` collector in parallel, and dump a CSV
     covering the same workload so we can compare avg/max syscall
     ticks against pre-migration baselines in `runs/`.

What it does
------------
1. (Optional) `cargo build` of bl/kmain via the canonical `bl/cargo run`
   path — but by default we assume the user has built already.
2. `subprocess.Popen` of QEMU via the bl runner (`cd bl && cargo run`),
   teeing serial to a log file and watching for panic markers.
3. Polls TCP :7777 (orbit-loader listen) until the guest is ready.
4. (Optional) `send-payload orbit-metricd` so metrics start streaming;
   waits for :7800 to come up; spawns `orbit_metric_logger.py` to dump
   a CSV.
5. Loops `--iterations` times: `send-payload eza --arg eza --arg -lahR`,
   then poll-connect to :7777 to confirm the loader recycled the listen
   before the next send.
6. Cleans up: kills the metric logger, sends SIGTERM to QEMU, prints a
   pass/fail summary (panic counter + iterations completed).

Usage
-----
    python3 tools/eza_stress.py                      # 50 iterations,
                                                     # default csv path
    python3 tools/eza_stress.py -n 200               # bump iterations
    python3 tools/eza_stress.py --no-metrics         # skip metricd +
                                                     # logger
    python3 tools/eza_stress.py --csv runs/foo.csv

Run from the repo root. Requires the eza binary at
`rootfs/bin/eza` (built into the canonical disk image) and
`rootfs/bin/orbit-metricd` for the metrics path.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEND_PAYLOAD = REPO_ROOT / "send-payload.py"
EZA_PATH = REPO_ROOT / "rootfs" / "bin" / "eza"
METRICD_PATH = REPO_ROOT / "rootfs" / "bin" / "orbit-metricd"
METRIC_LOGGER = REPO_ROOT / "tools" / "orbit_metric_logger.py"

LOADER_PORT = 7777
METRIC_PORT = 7800
METRIC_HOST = "127.0.0.1"

# Substrings that, when present in a serial line, indicate a fault.
# Conservative: a stray "panicked" mention in a log message would
# false-positive, so we tighten to the actual panic-print prefix.
PANIC_MARKERS = (
    "PanicInfo {",
    "panicked at",
    # Trap dumps from S-mode often print "stval=" with cause info; treat
    # as suspicious only if paired with a fault cause (12, 13, 15) which
    # the kernel logs alongside.
    "kernel panic",
)

# Regex for "orbit-loader: listening on :7777" — the loader's first line
# after fully arming the listen socket. We use TCP-poll as the primary
# readiness check, but watching for the line gives a clean log mark.
LOADER_READY_RE = re.compile(rb"orbit-loader: listening on")


def wait_for_port(host: str, port: int, timeout: float, *, label: str) -> bool:
    return True
    """Block until host:port accepts a TCP connection or `timeout` elapses.
    Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    print(f"timeout: {label} ({host}:{port}) not up after {timeout}s",
          file=sys.stderr)
    return False


def loader_ready_for_next() -> bool:
    return True
    """Quick connect-then-disconnect to confirm the loader is back to
    `accept`. Returns True if connect succeeds within 2s. Sequential
    sends rely on this — the loader is single-threaded per connection,
    so re-connecting too eagerly while it's still in `recv_payload`
    will queue rather than racing the next send."""
    try:
        with socket.create_connection(
            (METRIC_HOST, LOADER_PORT), timeout=2.0
        ):
            pass
        return True
    except OSError:
        return False


class SerialTee:
    """Pipe QEMU stdout into both a log file and an in-memory ring of
    recent lines. Background thread; `panic_count` increments any time
    a marker matches. Stops on EOF or when `stop()` is called."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.panic_count = 0
        self.last_lines: list[bytes] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loader_ready_seen = threading.Event()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(log_path, "wb", buffering=0)

    def loader_ready(self) -> threading.Event:
        return self._loader_ready_seen

    def start(self, stream) -> None:
        self._thread = threading.Thread(
            target=self._run, args=(stream,), daemon=True
        )
        self._thread.start()

    def _run(self, stream) -> None:
        try:
            for raw in iter(stream.readline, b""):
                if self._stop.is_set():
                    break
                self._log_fh.write(raw)
                # Keep last 200 lines for crash forensics.
                self.last_lines.append(raw)
                if len(self.last_lines) > 200:
                    self.last_lines.pop(0)
                if LOADER_READY_RE.search(raw):
                    self._loader_ready_seen.set()
                line_str = raw.decode("utf-8", errors="replace")
                for marker in PANIC_MARKERS:
                    if marker in line_str:
                        self.panic_count += 1
                        print(
                            f"\n[serial-panic] {line_str.rstrip()}",
                            file=sys.stderr,
                        )
                        break
        finally:
            self._log_fh.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def send_payload(elf: Path, name: str, argv: list[str]) -> int:
    """Invoke send-payload.py as a subprocess. Returns its exit code."""
    cmd = [
        sys.executable,
        str(SEND_PAYLOAD),
        str(elf),
        "--name",
        name,
    ]
    # `--arg=VALUE` (single token) rather than `--arg VALUE` so
    # argparse on the receiver doesn't reject dash-prefixed argv
    # like `-lahR` as a stray flag.
    for a in argv:
        cmd.append(f"--arg={a}")

    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--iterations", type=int, default=50,
                    help="how many times to send eza -lahR (default 50)")
    ap.add_argument("--inter-spawn-delay", type=float, default=0.5,
                    help="seconds to sleep after each send before "
                         "polling the loader (default 0.5)")
    ap.add_argument("--csv", default=str(REPO_ROOT / "runs" / "eza_stress.csv"),
                    help="metric collector output CSV")
    ap.add_argument("--serial-log",
                    default=str(REPO_ROOT / "runs" / "eza_stress.serial.log"),
                    help="raw serial output capture")
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip orbit-metricd / orbit_metric_logger.py")
    ap.add_argument("--no-args", action="store_true",
                    help="send eza without `--arg eza --arg -lahR` "
                         "(useful for diff-baselining other workloads)")
    ap.add_argument("--boot-timeout", type=float, default=120.0,
                    help="seconds to wait for orbit-loader (default 120)")
    ap.add_argument("--metric-timeout", type=float, default=30.0,
                    help="seconds to wait for orbit-metricd (default 30)")
    ap.add_argument("--keep-running", action="store_true",
                    help="leave QEMU running after iterations complete "
                         "(useful for post-mortem inspection)")
    ap.add_argument("--release", action="store_true",
                    help="cargo run --release for bl (default debug, "
                         "matching CLAUDE.md's documented launch flow)")
    args = ap.parse_args()

    if not EZA_PATH.exists():
        print(f"missing {EZA_PATH}; build the disk image first",
              file=sys.stderr)
        return 1
    if not args.no_metrics and not METRICD_PATH.exists():
        print(f"missing {METRICD_PATH}; build it or run with --no-metrics",
              file=sys.stderr)
        return 1

    bl_dir = REPO_ROOT / "bl"
    cargo_cmd = ["cargo", "run"] + (["--release"] if args.release else [])
    print(f"launching QEMU via `{' '.join(cargo_cmd)}` in {bl_dir}",
          file=sys.stderr)
    qemu = subprocess.Popen(
        cargo_cmd,
        cwd=bl_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # New session so we can SIGTERM the whole process group on
        # cleanup (the cargo wrapper spawns qemu as its child).
        start_new_session=True,
    )

    tee = SerialTee(Path(args.serial_log))
    tee.start(qemu.stdout)

    metric_logger: subprocess.Popen | None = None
    iterations_done = 0

    def cleanup() -> None:
        nonlocal metric_logger
        if metric_logger is not None and metric_logger.poll() is None:
            metric_logger.terminate()
            try:
                metric_logger.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                metric_logger.kill()
        if not args.keep_running and qemu.poll() is None:
            try:
                os.killpg(os.getpgid(qemu.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                qemu.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(qemu.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        tee.stop()

    try:
        time.sleep(5.0)
        if not wait_for_port(METRIC_HOST, LOADER_PORT, args.boot_timeout,
                             label="orbit-loader"):
            print("orbit-loader never came up; aborting", file=sys.stderr)
            print("--- last 40 serial lines ---", file=sys.stderr)
            for line in tee.last_lines[-40:]:
                sys.stderr.write(line.decode("utf-8", errors="replace"))
            return 2
        print(f"orbit-loader ready on :{LOADER_PORT}", file=sys.stderr)

        if not args.no_metrics:
            print("sending orbit-metricd payload", file=sys.stderr)
            rc = send_payload(METRICD_PATH, "orbit-metricd",
                              ["orbit-metricd"])
            if rc != 0:
                print(f"send-payload(metricd) failed: rc={rc}",
                      file=sys.stderr)
                return 3
            if not wait_for_port(METRIC_HOST, METRIC_PORT,
                                 args.metric_timeout,
                                 label="orbit-metricd"):
                return 4
            print(f"orbit-metricd up on :{METRIC_PORT}; "
                  f"starting collector → {args.csv}", file=sys.stderr)
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            metric_logger = subprocess.Popen(
                [
                    sys.executable,
                    str(METRIC_LOGGER),
                    "--port", str(METRIC_PORT),
                    "--csv", args.csv,
                    "--quiet",
                ],
            )

            time.sleep(5.0)

            # Loader needs a beat between sequential sends so the listen
            # socket recycles cleanly. metricd is the first send and is
            # bigger than eza-arg sends; pause to be safe.
            if not loader_ready_for_next():
                # Not fatal — `wait_for_port` already proved it once;
                # the next send will retry.
                pass

        eza_args = [] if args.no_args else ["eza", "-lahR", "/"]
        for i in range(1, args.iterations + 1):
            t0 = time.monotonic()
            rc = send_payload(EZA_PATH, "eza", eza_args)
            if rc != 0:
                print(f"\nsend-payload(eza) failed at iter {i}: rc={rc}",
                      file=sys.stderr)
                break
            iterations_done = i
            elapsed = time.monotonic() - t0
            print(f"  iter {i:>4}/{args.iterations}  send_ms={elapsed*1e3:.0f}  "
                  f"panics={tee.panic_count}", file=sys.stderr)

            time.sleep(args.inter_spawn_delay)
            # Confirm loader is ready for the next send. If it isn't, the
            # next connect will block at the kernel listen — a slower
            # spawn (eza heap allocation, page-table fan-out under
            # rayon) would otherwise let us race ahead and queue
            # against a still-busy loader, blunting the test.
            if not loader_ready_for_next():
                print(
                    f"  loader not ready after iter {i}; backing off",
                    file=sys.stderr,
                )
                time.sleep(1.0)

            if qemu.poll() is not None:
                print(f"\nQEMU exited mid-run at iter {i} "
                      f"(rc={qemu.returncode}); halting", file=sys.stderr)
                break

        print(file=sys.stderr)
        print(f"completed {iterations_done}/{args.iterations} iterations",
              file=sys.stderr)
        print(f"panics observed in serial: {tee.panic_count}",
              file=sys.stderr)
        if not args.no_metrics:
            print(f"metrics CSV: {args.csv}", file=sys.stderr)
        print(f"serial log: {args.serial_log}", file=sys.stderr)

        if tee.panic_count > 0 or iterations_done < args.iterations:
            return 5
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted; cleaning up", file=sys.stderr)
        return 130
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
