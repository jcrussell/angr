#!/usr/bin/env bash
# Stop hook: run clippy on the Rust crate once per turn, but only when there are
# uncommitted .rs changes (so pure-conversation / Python-only turns pay nothing).
# clippy compiles, so this also serves as the type-check; it mirrors CI's gate
# `cargo clippy --all-targets --all-features -- -D warnings`.
#
# On failure: exit 2 — this blocks the stop and feeds the trimmed clippy output
# back to Claude as a reason to keep working, surfacing type/lint errors before
# the turn ends. On success or when there's nothing to check: exit 0.
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

exit 0
