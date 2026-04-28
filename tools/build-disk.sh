#!/usr/bin/env bash
#
# Build disk.img from rootfs/ as a ustar archive. Served to the guest
# over virtio-blk; consumed by §12c's tarfs mount. Rebuilt fresh every
# call — disk.img is gitignored.
#
# Builds the in-tree user binaries that ship on the disk and stages
# them under rootfs/bin/ before tarring. Today: `hello` (§12e exec
# smoke target).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -d rootfs ] || { echo "FAIL: $ROOT/rootfs does not exist" >&2; exit 1; }

# Build /bin/hello.
( cd hello && cargo build --release >/dev/null )
mkdir -p rootfs/bin
cp hello/target/riscv64gc-unknown-none-elf/release/hello rootfs/bin/hello

tar --format=ustar -cf disk.img -C rootfs .
printf 'built %s/disk.img (%s bytes) from rootfs/\n' "$ROOT" "$(stat -c %s disk.img)"
