#!/usr/bin/env bash
# tools/restore-venv.sh — restore a working .venv after corruption / wipe.
#
# Tries (in order):
#   1. Restore from a tarball backup at ~/repos/angr.tar.gz (or $ANGR_BACKUP_TAR)
#   2. Fresh install from PyPI via `pip install -e .`
#
# After restore, verifies that `import angr` works and runs a quick
# test smoke-check.
#
# Usage: ./tools/restore-venv.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/.venv"
BACKUP_TAR="${ANGR_BACKUP_TAR:-$HOME/repos/angr.tar.gz}"

cd "$REPO_DIR"

# Refuse to clobber a working venv unless --force.
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "import angr" >/dev/null 2>&1; then
    if [[ "${1:-}" != "--force" ]]; then
        echo "venv already works ($VENV/bin/python imports angr). pass --force to rebuild."
        exit 0
    fi
fi

# Wipe whatever's left of the broken venv before restore.
if [[ -d "$VENV" ]]; then
    echo "removing existing $VENV"
    rm -rf "$VENV"
fi

# --- Option A: restore from tarball -----------------------------------------
if [[ -f "$BACKUP_TAR" ]]; then
    echo "extracting .venv from $BACKUP_TAR"
    if tar -tzf "$BACKUP_TAR" 2>/dev/null | grep -q '^angr/\.venv/bin/python$'; then
        (cd "$(dirname "$REPO_DIR")" && tar -xzf "$BACKUP_TAR" "$(basename "$REPO_DIR")/.venv")
        echo "tarball extract done"
    else
        echo "tarball does not contain angr/.venv/bin/python — skipping option A"
        rm -rf "$VENV"
    fi
fi

# --- Option B: fresh install ------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "creating fresh venv (Option B)"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip setuptools setuptools-rust
    "$VENV/bin/pip" install -e .
    "$VENV/bin/pip" install -e . --no-build-isolation --no-deps
fi

# --- Verify -----------------------------------------------------------------
echo
echo "=== verifying ==="
"$VENV/bin/python" -c "import angr; print('angr at', angr.__file__)"
"$VENV/bin/python" -c "import claripy, pyvex, archinfo, cle; print('claripy', claripy.__version__, 'pyvex', pyvex.__version__, 'archinfo', archinfo.__version__, 'cle', cle.__version__)"

# Quick smoke-check (skips full suite — that's hundreds of tests).
"$VENV/bin/python" -m pytest tests/engines/test_rust_exploration.py -k "test_fauxware or test_state_creation" --tb=short -q 2>&1 | tail -5

echo
echo "venv restored at $VENV"
echo "next: source $VENV/bin/activate"
