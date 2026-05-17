#!/bin/sh
# tools/ralph-claude.sh — runner wrapper for ralph.
#
# Sets the PATH and venv the rust-symex loop agent needs (~/.cargo/bin for
# cargo/rustc, .venv/bin for python/pytest/pip), then execs claude. Ralph
# passes the rendered prompt on stdin; `exec` preserves it.

set -e
export PATH="$HOME/.cargo/bin:$PATH"
# shellcheck disable=SC1091
. /home/ubuntu/repos/angr/.venv/bin/activate
exec claude --dangerously-skip-permissions --output-format=json "$@"
