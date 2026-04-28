#!/usr/bin/env bash
#
# Build every Orbit artifact in the order the include_bytes! chain
# requires:
#   1. umode apps (release)        — shipped to a running guest via
#                                    orbit-loader/tools/send-payload.py
#   2. disk.img (ustar of rootfs/) — also builds hello and stages it
#                                    under rootfs/bin/ before tarring
#   3. kmain (release, via build.sh) — embeds orbit-loader's ELF
#   4. bl (release)                — embeds kmain's ELF, links to launch
#
# Pass --features <list> to forward kmain build features (e.g. `smoke`).
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

# 1. umode apps. hello is built by tools/build-disk.sh in stage 2.
UMODE_APPS=(umode umode-sleep-bench umode-tcp-bench)
for app in "${UMODE_APPS[@]}"; do
  note "building $app (release)"
  ( cd "$ROOT/$app" && time cargo build --release "${EXTRA_CARGO_ARGS[@]}" >/dev/null )
done

# 2. disk.img — builds hello and tars rootfs/.
note "building disk.img from rootfs/"
"$ROOT/tools/build-disk.sh" >/dev/null
[ -s "$ROOT/disk.img" ] || { echo "FAIL: disk.img missing at $ROOT/disk.img" >&2; exit 1; }

# 3. kmain (PIE; must go through build.sh for the right RUSTFLAGS).
note "building kmain (release via build.sh${KMAIN_FEATURES:+ --features $KMAIN_FEATURES})"
if [ -n "$KMAIN_FEATURES" ]; then
  ( cd "$ROOT/kmain" && time ./build.sh --features "$KMAIN_FEATURES" )
else
  ( cd "$ROOT/kmain" && time ./build.sh )
fi

# 4. bl (M-mode bootloader binary `launch`).
note "building bl (release)"
( cd "$ROOT/bl" && time cargo build --release --bin launch "${EXTRA_CARGO_ARGS[@]}" >/dev/null )

BL_BIN="$ROOT/bl/target/riscv64gc-unknown-none-elf/release/launch"
[ -x "$BL_BIN" ] || { echo "FAIL: bl binary missing at $BL_BIN" >&2; exit 1; }

note "OK — launch=$BL_BIN  disk=$ROOT/disk.img"
