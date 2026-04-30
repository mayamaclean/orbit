#!/usr/bin/env bash
#
# Build disk.img from rootfs/ as a ustar archive. Served to the guest
# over virtio-blk; consumed by §12c's tarfs mount. Rebuilt fresh every
# call — disk.img is gitignored.
#
# Builds the in-tree user binaries that ship on the disk and stages
# them under rootfs/bin/ before tarring:
#   /bin/hello      — §12e exec smoke target
#   /bin/console    — interactive shell, default init
#   /bin/smoke      — umode test harness (renamed from `umode`); fed to
#                     orbit-loader as init when kmain is built with the
#                     `smoke` cargo feature
#   /bin/hello-std  — std-on-orbit smoke. Optional; built only if the
#                     orbit-stage1 rustup toolchain is linked (the rust
#                     fork at ./rust ships its own libstd).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -d rootfs ] || { echo "FAIL: $ROOT/rootfs does not exist" >&2; exit 1; }
mkdir -p rootfs/bin

# §12e: simple no_std exec target.
( cd hello && cargo build --release >/dev/null )
cp hello/target/riscv64gc-unknown-none-elf/release/hello rootfs/bin/hello

# Default init.
( cd console && cargo build --release >/dev/null )
cp console/target/riscv64gc-unknown-none-elf/release/console rootfs/bin/console

# Smoke harness — renamed at install time so kmain's `feature = "smoke"`
# argv (`/bin/smoke`) matches what the disk holds. Keeping the crate
# name as `umode` avoids touching its Cargo.toml; the rename happens
# only at the disk-staging step.
( cd umode && cargo build --release >/dev/null )
cp umode/target/riscv64gc-unknown-none-elf/release/umode rootfs/bin/smoke

# §13e std-on-orbit binary. Requires the `orbit-stage1` rustup toolchain
# linked from rust/build/x86_64-unknown-linux-gnu/stage1 (see CLAUDE.md
# under "Building Orbit's std"). Skip-with-warning if absent so a
# fresh checkout without the rust fork built can still run ./smoke.
if rustup toolchain list 2>/dev/null | grep -q '^orbit-stage1\b'; then
    ( cd hello-std && cargo +orbit-stage1 build --release >/dev/null )
    cp hello-std/target/riscv64gc-unknown-orbit/release/hello-std rootfs/bin/hello-std
else
    rm -f rootfs/bin/hello-std
    printf 'warn: orbit-stage1 toolchain not linked; skipping /bin/hello-std\n' >&2
fi

tar --format=ustar -cf disk.img -C rootfs .
printf 'built %s/disk.img (%s bytes) from rootfs/\n' "$ROOT" "$(stat -c %s disk.img)"
