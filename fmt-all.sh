#!/usr/bin/env bash
#
# Run `cargo fmt` on every crate tracked in this repo. Discovers crates
# by `git ls-files` (so untracked junk under e.g. `rust/build/` is
# skipped) and excludes `rust/` — that's the upstream rustc fork, not
# our code.
#
# There is no top-level workspace so each crate is formatted in its own
# `cargo` invocation. Continues past per-crate failures and reports the
# list at the end so a single broken Cargo.toml doesn't mask the rest.

set -uo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

note() { printf '>> %s\n' "$*"; }

mapfile -t MANIFESTS < <(
  cd "$ROOT" && git ls-files '*Cargo.toml' | grep -v '^rust/' | sort
)

if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  echo "fmt-all: no Cargo.toml files found via git ls-files" >&2
  exit 1
fi

failed=()
for manifest in "${MANIFESTS[@]}"; do
  crate_dir=$(dirname "$manifest")
  note "cargo fmt — $crate_dir"
  if ! ( cd "$ROOT/$crate_dir" && cargo fmt "$@" ); then
    failed+=("$crate_dir")
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "fmt-all: ${#failed[@]} crate(s) failed:" >&2
  for c in "${failed[@]}"; do printf '  - %s\n' "$c" >&2; done
  exit 1
fi

note "OK — formatted ${#MANIFESTS[@]} crate(s)"
