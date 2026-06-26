#!/usr/bin/env bash
# tools/rebuild-rust.sh — clean rebuild of the Rust extension.
#
# Use this when:
#   - A previous `pip install -e .` failed midway and left a stale
#     angr/rustylib.cpython-*.so
#   - Tests load the wrong .so (wrong feature flags, missing symbols,
#     unexpected hash mismatches)
#   - You suspect cargo's incremental cache is stale
#
# What it does (in order):
#   1. Remove angr/rustylib*.so so a missed rebuild surfaces as ImportError
#   2. Remove build/ (setuptools intermediate dir)
#   3. cargo clean (workspace-wide; native/angr is the only member)
#   4. pip install -e . --no-build-isolation --no-deps  (rebuild the .so)
#
# Modes:
#   --keep-cargo-cache   Skip step 3. Faster; useful when only setuptools-rust
#                        / Python wrapper state is suspect, not Rust codegen.
#   --cargo-only         Skip pip; build via `cargo build --release` and copy
#                        target/release/librustylib.so to angr/rustylib.<tag>.so.
#                        Use when the venv's pip/setuptools is broken but
#                        cargo+rustc work. The PyO3 .so exports the same module,
#                        so `import angr.rustylib` works without setup.py.
#   --fast               Inner-loop dev rebuild via [profile.release-fast]
#                        (lto=off, codegen-units=16). ~3x faster warm rebuild
#                        than --cargo-only. Implies --keep-cargo-cache and the
#                        cargo-direct copy path. NOT for benchmark gates or
#                        release artifacts — use the standard path for those.
#                        Runtime impact on fast-tier benches was within 5-sample
#                        noise; see bd memory `build-perf-spike-codegen-units`.
#
# Usage:
#   ./tools/rebuild-rust.sh
#   ./tools/rebuild-rust.sh --keep-cargo-cache
#   ./tools/rebuild-rust.sh --cargo-only
#   ./tools/rebuild-rust.sh --fast

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/.venv"
MANIFEST="$REPO_DIR/native/angr/Cargo.toml"

KEEP_CARGO_CACHE=0
CARGO_ONLY=0
FAST=0
for arg in "$@"; do
    case "$arg" in
        --keep-cargo-cache) KEEP_CARGO_CACHE=1 ;;
        --cargo-only)       CARGO_ONLY=1 ;;
        --fast)             FAST=1; CARGO_ONLY=1; KEEP_CARGO_CACHE=1 ;;
        -h|--help)
            sed -n '2,/^set /p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

cd "$REPO_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "no venv at $VENV — run tools/restore-venv.sh first" >&2
    exit 1
fi

export PATH="$HOME/.cargo/bin:$PATH"

# Compute the .so target path the way setuptools-rust would name it.
# E.g. angr/rustylib.cpython-312-x86_64-linux-gnu.so
PY_TAG="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SO_TARGET="$REPO_DIR/angr/rustylib${PY_TAG}"

echo "=== removing stale .so files ==="
shopt -s nullglob
SO_FILES=("$REPO_DIR"/angr/rustylib*.so)
if (( ${#SO_FILES[@]} )); then
    rm -fv "${SO_FILES[@]}"
else
    echo "(no rustylib*.so to remove)"
fi
shopt -u nullglob

echo
echo "=== removing setuptools build/ ==="
if [[ -d "$REPO_DIR/build" ]]; then
    rm -rf "$REPO_DIR/build"
    echo "removed $REPO_DIR/build"
else
    echo "(no build/ to remove)"
fi

if (( KEEP_CARGO_CACHE == 0 )); then
    echo
    echo "=== cargo clean ==="
    cargo clean --manifest-path "$REPO_DIR/Cargo.toml"
else
    echo
    echo "=== skipping cargo clean (--keep-cargo-cache) ==="
fi

if (( CARGO_ONLY == 1 )); then
    echo
    if (( FAST == 1 )); then
        echo "=== cargo build --profile release-fast (--fast inner-loop) ==="
        CARGO_PROFILE_FLAG=(--profile release-fast)
        CARGO_PROFILE_DIR=release-fast
    else
        echo "=== cargo build --release (--cargo-only fallback) ==="
        CARGO_PROFILE_FLAG=(--release)
        CARGO_PROFILE_DIR=release
    fi
    # z3-sys needs the C header. Honor an existing override; otherwise probe
    # the same locations setup.py::_resolve_z3_header does (venv → pkg-config
    # → system paths). If none match, leave Z3_SYS_Z3_HEADER unset and let
    # z3-sys's own pkg-config probe report the error.
    if [[ -z "${Z3_SYS_Z3_HEADER:-}" ]]; then
        CANDIDATES=("$VENV/lib/python3.12/site-packages/z3/include/z3.h")
        if command -v pkg-config >/dev/null 2>&1; then
            PCINC="$(pkg-config --variable=includedir z3 2>/dev/null)"
            [[ -n "$PCINC" ]] && CANDIDATES+=("$PCINC/z3.h")
        fi
        CANDIDATES+=(/usr/include/z3.h /usr/local/include/z3.h /opt/homebrew/include/z3.h /opt/local/include/z3.h)
        for C in "${CANDIDATES[@]}"; do
            if [[ -f "$C" ]]; then
                export Z3_SYS_Z3_HEADER="$C"
                echo "using Z3_SYS_Z3_HEADER=$Z3_SYS_Z3_HEADER"
                break
            fi
        done
    fi
    # build.rs runpaths the .so to whatever libz3 `python3` (on PATH) resolves —
    # which in --cargo-only mode is the SYSTEM python3, not the venv. Linking the
    # system libz3 while claripy loads the venv libz3 loads TWO libz3 instances
    # and double-frees at interpreter teardown (`free(): invalid pointer`,
    # SIGABRT/SIGSEGV at process exit — see bd cargo-direct-so-teardown-segfault).
    # Pin the runpath to the venv's libz3 so Python and Rust share one instance.
    if [[ -z "${Z3_LIBRARY_PATH_OVERRIDE:-}" ]]; then
        VENV_Z3_LIB="$VENV/lib/python3.12/site-packages/z3/lib"
        if [[ -f "$VENV_Z3_LIB/libz3.so" || -f "$VENV_Z3_LIB/libz3.dylib" ]]; then
            export Z3_LIBRARY_PATH_OVERRIDE="$VENV_Z3_LIB"
            echo "using Z3_LIBRARY_PATH_OVERRIDE=$Z3_LIBRARY_PATH_OVERRIDE"
        fi
    fi
    cargo build --manifest-path "$MANIFEST" "${CARGO_PROFILE_FLAG[@]}"
    SRC="$REPO_DIR/target/$CARGO_PROFILE_DIR/librustylib.so"
    if [[ ! -f "$SRC" ]]; then
        echo "ERROR: cargo build did not produce $SRC" >&2
        exit 1
    fi
    cp -v "$SRC" "$SO_TARGET"
else
    echo
    echo "=== pip install -e . --no-build-isolation --no-deps ==="
    # Some venvs ship without a `pip` executable; `python -m pip` always works.
    "$VENV/bin/python" -m pip install -e . --no-build-isolation --no-deps
fi

echo
echo "=== verifying ==="
NEW_SO=("$REPO_DIR"/angr/rustylib*.so)
if (( ${#NEW_SO[@]} == 0 )); then
    echo "ERROR: rebuild produced no rustylib*.so" >&2
    exit 1
fi
echo "built: ${NEW_SO[*]}"
"$VENV/bin/python" -c "import angr; from angr.rustylib import vex_engine; print('angr at', angr.__file__)"

echo
echo "rebuild complete"
