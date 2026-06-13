#!/usr/bin/env bash
# PostToolUse(Edit|Write) hook: auto-format the file Claude just edited.
#   .rs  -> cargo fmt (whole crate; idempotent, fast)
#   .py  -> ruff format + ruff check --fix  (only if ruff is installed)
#
# Best-effort and NON-blocking: always exits 0. Formatting failures (e.g. a
# file that doesn't parse) are intentionally ignored here — the Stop hook's
# `cargo check` is the gate that surfaces real type errors.
#
# The .py branch is a best-effort no-op unless `ruff` is on PATH. On this env
# ruff 0.15.15 lives at ~/.cargo/bin/ruff, installed from the PyPI wheel
# (the venv pip is broken; see bd memory ruff-install-recipe). Fresh checkouts
# without ruff simply skip Python.
set -u

input=$(cat)
f=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -z "$f" ] && exit 0

export PATH="$HOME/.cargo/bin:$PATH"
# Repo root: prefer the hook env var, else derive from this script's location
# (.claude/hooks/<script> -> ../.. is the repo root). No hardcoded user paths.
repo="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

case "$f" in
  *.rs)
    cargo fmt --manifest-path "$repo/native/angr/Cargo.toml" >/dev/null 2>&1
    ;;
  *.py)
    if command -v ruff >/dev/null 2>&1; then
      # check --fix BEFORE format (pre-commit order) so lint fixes that
      # restructure code get reformatted, matching the repo's clean state.
      ruff check --fix "$f" >/dev/null 2>&1
      ruff format "$f" >/dev/null 2>&1
    fi
    ;;
esac

exit 0
