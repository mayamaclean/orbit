#!/usr/bin/env bash
#
# Rebuild the orbit-forked rustc's std for both the orbit target and the
# host target, then clean every in-tree std-on-orbit consumer's `target/`
# directory so cargo refetches their rmetas against the freshly-built
# stage1 sysroots.
#
# # Why
#
# `./x build library --stage 1 --target riscv64gc-unknown-orbit` is the
# canonical orbit-side rebuild step (see CLAUDE.md "Building Orbit's
# std"). Two reasons it isn't quite enough on its own:
#
# 1. **Both target sysroots need to stay in sync.** Stage-1 rustc itself
#    runs on the host; proc-macros and build.rs scripts in std-on-orbit
#    consumers link against the host stage-1 std. Rebuilding only the
#    orbit target leaves the host stage-1 stale, and `cargo
#    +orbit-stage1 build` against a consumer fails later with bizarre
#    type-mismatch errors (the host std rustc loads doesn't match what
#    the host stage-1 was built against). Pass both `--target` flags.
#
# 2. **Cargo's incremental cache doesn't track sysroot changes.** After
#    a clean std rebuild, every consumer's per-crate `target/` still
#    has rmeta files (e.g. `libab_glyph-*.rmeta`) that were produced
#    against the previous sysroot. Cargo marks them `Fresh` and reuses
#    them, but the new stage1 rustc rejects them — symptom is `error:
#    can't find crate for X` even though X is fully built and visible
#    in `target/.../deps/`. The only reliable fix is `cargo clean` per
#    consumer; selective cleans (`-p X`) don't catch every transitive
#    that depends on something through std.
#
# # Scope
#
# Cleans the std-on-orbit consumers that live in this repo:
#   hello-std, hello-fb-std, hello-ratatui-std,
#   orbit-top-std, orbit-metricd
# plus orbit-text (no_std library, but has its own target dir when built
# directly). Each consumer's `target/` is wiped wholesale; the next
# `build-all.sh` invocation rebuilds them from a clean slate.
#
# Out-of-tree consumers (eza at ../eza, ripgrep at ../ripgrep) are
# **not** cleaned automatically — they have large dep trees and
# rebuilding from scratch costs minutes. If you hit `can't find crate
# for X` errors on them after this script, run `cargo clean` in the
# affected source tree by hand.
#
# # Usage
#
#   ./rebuild-std.sh
#
# No flags. Idempotent — safe to re-run if it failed mid-way.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

note() { printf '>> %s\n' "$*"; }

# Host triple — matches what `rustup toolchain link orbit-stage1` was
# pointed at in CLAUDE.md ("Building Orbit's std" section).
HOST_TRIPLE="x86_64-unknown-linux-gnu"
ORBIT_TRIPLE="riscv64gc-unknown-orbit"

note "rebuilding std (stage 1) for $ORBIT_TRIPLE + $HOST_TRIPLE"
( cd "$ROOT/rust" && time ./x build library --stage 1 \
    --target "$ORBIT_TRIPLE","$HOST_TRIPLE" )

# In-tree std-on-orbit consumers. Order doesn't matter — `cargo clean`
# is independent per crate. Add new consumers to this list as they
# appear; out-of-tree paths (../eza, ../ripgrep) intentionally not
# included — see header comment.
STD_CONSUMERS=(
    orbit-text
    hello-std
    hello-fb-std
    hello-ratatui-std
    orbit-top-std
    orbit-metricd
)

for crate in "${STD_CONSUMERS[@]}"; do
    if [ -d "$ROOT/$crate" ]; then
        note "cargo clean: $crate"
        ( cd "$ROOT/$crate" && cargo clean )
    else
        # Not fatal — a future rename / removal shouldn't fail the script.
        printf 'warn: %s missing, skipping clean\n' "$crate" >&2
    fi
done

note "OK — std rebuilt, consumers cleaned. Next: ./build-all.sh"
