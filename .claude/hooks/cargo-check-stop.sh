#!/usr/bin/env bash
# Stop hook: run clippy and the rustdoc link check on the Rust crate once per
# turn, but only when there are uncommitted .rs changes (so pure-conversation /
# Python-only turns pay nothing). clippy compiles, so it also serves as the
# type-check; the pair mirrors two of CI's `rust_check` gates
# (`cargo clippy --all-targets --all-features -- -D warnings` and
# `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features
# --document-private-items`).
#
# Why rustdoc runs here (angr-z3672): the rustdoc lane existed ONLY in CI, and
# this box has no network to GitHub, so nothing in the local loop ever ran it —
# broken `[`Foo::bar`]` intra-doc links accumulated silently between human CI
# pushes. angr-f9pct cleared 8, then 9 more grew back before angr-xtjm8 caught
# them. It is cheap: ~4.3s when only the crate's own sources changed (the dev
# profile target dir clippy just warmed is shared), ~0.3s when nothing did.
# `--document-private-items` is load-bearing (angr-sqfj8.144) — nearly every
# item here is `pub(crate)`/private, so without it rustdoc never visits their
# docs. Do not drop it.
#
# On failure: exit 2 — this blocks the stop and feeds the trimmed tool output
# back to Claude as a reason to keep working, surfacing type/lint/doc-link
# errors before the turn ends. On success or when there's nothing to check:
# exit 0.
set -u

input=$(cat)

# Loop guard: if we already blocked once this turn, don't block again.
if [ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false' 2>/dev/null)" = "true" ]; then
  exit 0
fi

# Repo root: prefer the hook env var, else derive from this script's location
# (.claude/hooks/<script> -> ../.. is the repo root). No hardcoded user paths.
repo="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$repo" 2>/dev/null || exit 0

# Skip unless Rust files are dirty (modified, staged, or new/untracked).
git status --porcelain -- '*.rs' 2>/dev/null | grep -q . || exit 0

export PATH="$HOME/.cargo/bin:$PATH"
if ! out=$(cargo clippy --manifest-path native/angr/Cargo.toml --all-targets --all-features -- -D warnings 2>&1); then
  {
    echo "cargo clippy failed — fix the Rust errors/warnings before finishing:"
    printf '%s\n' "$out" | tail -40
  } >&2
  exit 2
fi

if ! out=$(RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path native/angr/Cargo.toml \
    --no-deps --all-features --document-private-items 2>&1); then
  {
    echo "rustdoc link check failed — fix the broken intra-doc links before finishing."
    echo "Rules of thumb: link a same-impl method as [\`m\`](Self::m); a link from a"
    echo "public item's docs to a private one trips private_intra_doc_links even"
    echo "though the target exists — demote those to a plain code span."
    printf '%s\n' "$out" | tail -40
  } >&2
  exit 2
fi

exit 0
