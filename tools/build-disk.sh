#!/usr/bin/env bash
#
# Build disk.img from rootfs/ as a ustar archive. Served to the guest
# over virtio-blk; consumed by §12c's tarfs mount. Rebuilt fresh every
# call — disk.img is gitignored.
#
# Two-tier discovery for the in-tree user binaries staged under
# rootfs/bin/:
#
# 1. Simple crates (default toolchain, default target, install name ==
#    crate name) are auto-discovered by the presence of a
#    `[package.metadata.disk]` section in their Cargo.toml. Add a new
#    simple disk binary by adding that marker to the new crate; no
#    edit here required. Currently:
#      /bin/hello      — §12e exec smoke target
#      /bin/console    — interactive shell, default init
#
# 2. Bespoke crates (rename, special toolchain, optional skip) stay as
#    explicit blocks below since each needs custom handling:
#      /bin/smoke      — umode test harness (renamed from `umode`);
#                        fed to orbit-loader as init when kmain is
#                        built with the `smoke` cargo feature
#      /bin/hello-std  — std-on-orbit smoke. Optional; built only if
#                        the orbit-stage1 rustup toolchain is linked
#                        (the rust fork at ./rust ships its own libstd).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -d rootfs ] || { echo "FAIL: $ROOT/rootfs does not exist" >&2; exit 1; }
mkdir -p rootfs/bin

note() { printf '>> %s\n' "$*"; }

# Tier 1 — auto-discover simple disk binaries via the
# `[package.metadata.disk]` marker. grep is enough; no TOML parser
# needed for the empty-section presence check.
mapfile -t SIMPLE_DISK_BINS < <(
  git ls-files '*Cargo.toml' | while read -r m; do
    if grep -q '^\[package\.metadata\.disk\]' "$m"; then
      printf '%s\n' "${m%/Cargo.toml}"
    fi
  done
)

for crate in "${SIMPLE_DISK_BINS[@]}"; do
  note "building $crate (release) → /bin/$crate"
  ( cd "$crate" && cargo build --release >/dev/null )
  cp "$crate/target/riscv64gc-unknown-none-elf/release/$crate" "rootfs/bin/$crate"
done

# Tier 2 — bespoke binaries.

# Smoke harness — renamed at install time so kmain's `feature = "smoke"`
# argv (`/bin/smoke`) matches what the disk holds. Keeping the crate
# name as `umode` avoids touching its Cargo.toml; the rename happens
# only at the disk-staging step.
note "building umode (release) → /bin/smoke"
( cd umode && cargo build --release >/dev/null )
cp umode/target/riscv64gc-unknown-none-elf/release/umode rootfs/bin/smoke

# §13e std-on-orbit binary. Requires the `orbit-stage1` rustup toolchain
# linked from rust/build/x86_64-unknown-linux-gnu/stage1 (see CLAUDE.md
# under "Building Orbit's std"). Skip-with-warning if absent so a
# fresh checkout without the rust fork built can still run ./smoke.
if rustup toolchain list 2>/dev/null | grep -q '^orbit-stage1\b'; then
    note "building hello-std (+orbit-stage1, release) → /bin/hello-std"
    ( cd hello-std && cargo +orbit-stage1 build --release >/dev/null )
    cp hello-std/target/riscv64gc-unknown-orbit/release/hello-std rootfs/bin/hello-std
else
    rm -f rootfs/bin/hello-std
    printf 'warn: orbit-stage1 toolchain not linked; skipping /bin/hello-std\n' >&2
fi

# eza — modern `ls` replacement, out-of-tree at ../eza. Built against
# orbit-stage1 with --no-default-features (drops git2). The crate's
# Cargo.toml carries [patch.crates-io] entries pointing at the
# vendored libc / backtrace / terminal_size / uzers under
# rust/library/; see docs/dev/unix-surface.md. Skip-with-warning if
# either the toolchain or the eza checkout is missing so contributors
# without that sibling repo can still run ./smoke.
EZA_SRC="$ROOT/../eza"
if rustup toolchain list 2>/dev/null | grep -q '^orbit-stage1\b' \
   && [ -d "$EZA_SRC" ]; then
    note "building eza (+orbit-stage1, release, --no-default-features) → /bin/eza"
    ( cd "$EZA_SRC" && cargo +orbit-stage1 build --release --no-default-features >/dev/null )
    cp "$EZA_SRC/target/riscv64gc-unknown-orbit/release/eza" rootfs/bin/eza
else
    rm -f rootfs/bin/eza
    printf 'warn: orbit-stage1 toolchain or %s missing; skipping /bin/eza\n' "$EZA_SRC" >&2
fi

tar --format=ustar -cf disk.img -C rootfs .
printf 'built %s/disk.img (%s bytes) from rootfs/\n' "$ROOT" "$(stat -c %s disk.img)"
