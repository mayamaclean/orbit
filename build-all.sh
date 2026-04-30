#!/usr/bin/env bash
#
# Build every Orbit artifact in dependency order. Post §13e tarfs-init,
# the privilege chain is:
#
#   bl ──include_bytes──► kmain ──include_bytes──► orbit-loader
#
# Everything else (console, smoke=umode, hello-std, hello, plus any
# bench binaries shipped via send-payload.py) builds independently.
# orbit-loader fs_loads the init binary off disk.img at runtime; the
# disk image is repacked by tools/build-disk.sh, which also builds the
# user binaries it stages under rootfs/bin/.
#
# Build order:
#   1. orbit-loader (kmain include_bytes!s it)
#   2. kmain (PIE; via build.sh for the right RUSTFLAGS)
#   3. bl (M-mode bootloader)
#   4. disk.img (builds hello, console, smoke=umode, hello-std internally)
#   5. ad-hoc bench binaries (sent over TCP via send-payload.py;
#      not staged in disk.img today)
#
# Pass --features <list> to forward kmain build features
# (`smoke` / `hello-std` change the boot argv k_smpstart packs;
# default is `/bin/console`).
# Anything else after `--` is forwarded to every cargo build invocation.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
KMAIN_FEATURES=""
EXTRA_CARGO_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --features) KMAIN_FEATURES="$2"; shift 2 ;;
    --) shift; EXTRA_CARGO_ARGS+=("$@"); break ;;
    *)  EXTRA_CARGO_ARGS+=("$1"); shift ;;
  esac
done

note() { printf '>> %s\n' "$*"; }

# 1. orbit-loader — kmain include_bytes!s the resulting ELF.
note "building orbit-loader (release)"
( cd "$ROOT/orbit-loader" && time cargo build --release --bin orbit-loader "${EXTRA_CARGO_ARGS[@]}" >/dev/null )

# 2. kmain (PIE; must go through build.sh for the right RUSTFLAGS).
note "building kmain (release via build.sh${KMAIN_FEATURES:+ --features $KMAIN_FEATURES})"
if [ -n "$KMAIN_FEATURES" ]; then
  ( cd "$ROOT/kmain" && time ./build.sh --features "$KMAIN_FEATURES" )
else
  ( cd "$ROOT/kmain" && time ./build.sh )
fi

# 3. bl (M-mode bootloader binary `launch`).
note "building bl (release)"
( cd "$ROOT/bl" && time cargo build --release --bin launch "${EXTRA_CARGO_ARGS[@]}" >/dev/null )

BL_BIN="$ROOT/bl/target/riscv64gc-unknown-none-elf/release/launch"
[ -x "$BL_BIN" ] || { echo "FAIL: bl binary missing at $BL_BIN" >&2; exit 1; }

# 4. disk.img — builds hello + console + smoke=umode + hello-std and
# tars rootfs/. hello-std is built only if the orbit-stage1 toolchain
# is linked (see CLAUDE.md "Building Orbit's std").
note "building disk.img from rootfs/ (console, smoke=umode, hello-std, hello)"
"$ROOT/tools/build-disk.sh" >/dev/null
[ -s "$ROOT/disk.img" ] || { echo "FAIL: disk.img missing at $ROOT/disk.img" >&2; exit 1; }

# 5. Ad-hoc bench binaries — not in tarfs; shipped via
# orbit-loader/tools/send-payload.py at runtime. Built last so a
# failure here (these crates predate orbit-rt's `_start` and may have
# unmigrated entrypoints) doesn't strand the kernel/disk artifacts.
BENCH_APPS=(umode-sleep-bench umode-tcp-bench)
for app in "${BENCH_APPS[@]}"; do
  if [ -d "$ROOT/$app" ]; then
    note "building $app (release)"
    if ! ( cd "$ROOT/$app" && cargo build --release "${EXTRA_CARGO_ARGS[@]}" >/dev/null 2>&1 ); then
      printf 'warn: %s build failed (skipping; rerun by hand for the error)\n' "$app" >&2
    fi
  fi
done

note "OK — launch=$BL_BIN  disk=$ROOT/disk.img"
