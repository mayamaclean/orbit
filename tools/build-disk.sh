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
# under "Building Orbit's std") AND the rust fork checkout in this tree
# (hello-std's [patch.crates-io] libc points at rust/library/libc, so a
# globally-linked toolchain isn't enough on its own). Skip-with-warning
# if either is absent so a fresh checkout can still boot: the prebuilt
# copies tracked in rootfs/bin/ stay in place and land in disk.img.
if rustup toolchain list 2>/dev/null | grep -q '^orbit-stage1\b' \
   && [ -d "$ROOT/rust/library/libc" ]; then
    note "building hello-std (+orbit-stage1, release) → /bin/hello-std"
    ( cd hello-std && cargo +orbit-stage1 build --release >/dev/null )
    cp hello-std/target/riscv64gc-unknown-orbit/release/hello-std rootfs/bin/hello-std

    note "building hello-fb-std (+orbit-stage1, release) → /bin/hello-fb-std"
    ( cd hello-fb-std && cargo +orbit-stage1 build --release >/dev/null )
    cp hello-fb-std/target/riscv64gc-unknown-orbit/release/hello-fb-std rootfs/bin/hello-fb-std

    note "building hello-ratatui-std (+orbit-stage1, release) → /bin/hello-ratatui-std"
    ( cd hello-ratatui-std && cargo +orbit-stage1 build --release >/dev/null )
    cp hello-ratatui-std/target/riscv64gc-unknown-orbit/release/hello-ratatui-std rootfs/bin/hello-ratatui-std

    note "building orbit-top-std (+orbit-stage1, release) → /bin/orbit-top-std"
    ( cd orbit-top-std && cargo +orbit-stage1 build --release >/dev/null )
    cp orbit-top-std/target/riscv64gc-unknown-orbit/release/orbit-top-std rootfs/bin/orbit-top-std

    note "building orbit-metricd (+orbit-stage1, release) → /bin/orbit-metricd"
    ( cd orbit-metricd && cargo +orbit-stage1 build --release >/dev/null )
    cp orbit-metricd/target/riscv64gc-unknown-orbit/release/orbit-metricd rootfs/bin/orbit-metricd
else
    printf 'warn: orbit-stage1 toolchain or rust/ fork checkout missing; staging prebuilt rootfs/bin std binaries as-is (hello-fb-std, hello-ratatui-std, orbit-top-std, orbit-metricd; hello-std only if previously built)\n' >&2
fi

# eza — modern `ls` replacement, out-of-tree at ../eza (orbit branch
# of https://github.com/mayamaclean/eza). Built against
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

# ripgrep — `rg` recursive grep, out-of-tree at ../ripgrep (orbit
# branch of https://github.com/mayamaclean/ripgrep). Same
# build pattern as eza but uses the `release-lto` profile (defined
# in ripgrep's Cargo.toml) to get a stripped + LTO'd binary instead
# of the 29 MB debug-info-laden default release. Patches in
# [patch.crates-io] cover same-file + memmap2 (in addition to the
# eza set); see docs/dev/unix-surface.md §6.
RG_SRC="$ROOT/../ripgrep"
if rustup toolchain list 2>/dev/null | grep -q '^orbit-stage1\b' \
   && [ -d "$RG_SRC" ]; then
    note "building ripgrep (+orbit-stage1, release-lto, --no-default-features) → /bin/rg"
    ( cd "$RG_SRC" && cargo +orbit-stage1 build --profile release-lto --no-default-features >/dev/null )
    cp "$RG_SRC/target/riscv64gc-unknown-orbit/release-lto/rg" rootfs/bin/rg
else
    rm -f rootfs/bin/rg
    printf 'warn: orbit-stage1 toolchain or %s missing; skipping /bin/rg\n' "$RG_SRC" >&2
fi

# Two-pass archive build so /etc/ entries land with owner=root:0 in
# the tar headers despite the rootfs/ files being owned by the build
# user (tar would otherwise stamp the owner from stat()). The kernel-
# side `vaccess()` check needs at least one file unreadable to a
# non-root caller for the deny-path smoke; system files (passwd,
# shadow, etc.) are the natural home and POSIX-shape callers expect
# uid=0 ownership there. Add new system-owned subtrees by extending
# this block — the default tar pass handles everything else with
# the build user's uid.
if [ -d rootfs/etc ]; then
  tar --format=ustar -cf disk.img -C rootfs --exclude='./etc' .
  tar --format=ustar --append -f disk.img -C rootfs \
      --owner=0 --group=0 etc
else
  tar --format=ustar -cf disk.img -C rootfs .
fi

# No tail padding needed — `submit_blk_read_cached` clamps the DMA to
# the disk's `capacity_sectors` and zero-fills the cache slot's tail
# via the kdmap alias. tar lays files at 512-byte boundaries, so a
# file's last page can ask for sectors past the disk end; the kernel
# absorbs that natively now. See [kmain/src/drivers/virtio_blk_dev.rs].
printf 'built %s/disk.img (%s bytes) from rootfs/\n' "$ROOT" "$(stat -c %s disk.img)"
