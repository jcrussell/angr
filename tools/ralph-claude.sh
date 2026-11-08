#!/bin/sh
# tools/ralph-claude.sh — runner wrapper for ralph.
#
# Sets the PATH and venv the rust-symex loop agent needs (~/.cargo/bin for
# cargo/rustc, .venv/bin for python/pytest/pip), then execs claude. Ralph
# passes the rendered prompt on stdin; `exec` preserves it.

set -e
export PATH="$HOME/.cargo/bin:$PATH"
# A cold build of the angr crate at full parallelism outgrows the 4G
# memory_limit_bytes in .ralph/config.toml; -j2 peaked at 3.4 GB
# (angr-mx00p.122). An explicit CARGO_BUILD_JOBS still wins.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
# shellcheck disable=SC1091
. /home/ubuntu/repos/angr/.venv/bin/activate
# The model comes from [runner] model in .ralph/config.toml, which ralph
# appends to "$@" as --model.
exec claude --dangerously-skip-permissions --output-format=json "$@"
