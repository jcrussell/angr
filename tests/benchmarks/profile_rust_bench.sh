#!/usr/bin/env bash
# Profile a Rust criterion bench with the best profiler available on this host.
#
# Usage:
#   tests/benchmarks/profile_rust_bench.sh [--secs N] [--filter PATTERN] [--tool auto|perf|flamegraph|callgrind]
#   tests/benchmarks/profile_rust_bench.sh --list
#   tests/benchmarks/profile_rust_bench.sh --help
#
# What this does:
#   1. Detects which profiler is available: cargo-flamegraph, perf, callgrind.
#   2. Builds the vex_engine bench in --release with frame pointers and full
#      debuginfo so symbols resolve.
#   3. Runs `cargo bench` with criterion's `--profile-time N` flag (criterion
#      disables its own measurement loop and just runs the bench body for ~N
#      seconds — exactly what an external sampling profiler wants).
#   4. Wraps the run with the detected tool. Output goes under
#      target/profile/<filter-or-all>/ at the workspace root.
#
# Notes:
#   - perf needs perf_event_paranoid <= 2 (current setting is in
#     /proc/sys/kernel/perf_event_paranoid). On hosts with paranoid >= 3 the
#     script will detect this and refuse to use perf. Lower it with
#     `sudo sysctl -w kernel.perf_event_paranoid=2`.
#   - cargo-flamegraph (cargo install flamegraph) wraps perf+inferno end to end
#     and is the friendliest option when available.
#   - callgrind is slow (10-50x) but works without kernel-event privileges.
#   - Workspace [profile.release] sets strip="symbols", which the bench profile
#     inherits. We override via --config so perf can resolve Rust symbols.
#   - z3-sys build.rs needs Z3_SYS_Z3_HEADER. If the .venv z3 package is
#     missing include/z3.h we fall back to /usr/include/z3.h automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$REPO_ROOT/native/angr/Cargo.toml"
# Workspace target dir (cargo writes here, not into the package's own target/).
TARGET_DIR="$REPO_ROOT/target"

SECS=10
FILTER=""
TOOL="auto"

# The workspace [profile.release] strips symbols, which makes flamegraphs
# unreadable. Force-disable strip and force-enable debuginfo for the bench
# profile via --config overrides (no permanent edit to Cargo.toml).
CARGO_PROFILE_OVERRIDES=(
    --config 'profile.bench.strip=false'
    --config 'profile.bench.debug=true'
)

# z3-sys build.rs reads Z3_SYS_Z3_HEADER for the header to bind. The .venv
# z3 package may be missing include/z3.h — fall back to the system header so
# `cargo bench --no-run` does not panic on bindgen.
if [[ -z "${Z3_SYS_Z3_HEADER:-}" ]]; then
    venv_header="$REPO_ROOT/.venv/lib/python3.12/site-packages/z3/include/z3.h"
    if [[ ! -e "$venv_header" && -e /usr/include/z3.h ]]; then
        export Z3_SYS_Z3_HEADER=/usr/include/z3.h
    fi
fi

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}"
}

list_groups() {
    cat <<'EOF'
Available bench groups (see native/angr/benches/vex_engine.rs):
  rustbv_concrete       — RustBV concrete add/sub/concat/extract/reverse
  rustbv_symbolic       — RustBV symbolic add/concat/extract/reverse
  rustbv_z3             — RustBV → z3::ast::BV build
  symcontext_fork       — SymContext fork cost
  symcontext_fork_scaling — Fork cost vs. constraint count
  symcontext_check_branch — Branch feasibility check
  symcontext_assume     — assume_true cost
  symcontext_push_pop   — solver scope push/pop
  memory_concrete       — page-aligned concrete store/load
  memory_symbolic_load  — symbolic-address load
  memory_fork           — SymbolicMemory fork
  state_fork            — RustSimState fork
  rustbv_neon_ops       — NEON Mul8x16 / VGetElem / VSetElem (concrete + symbolic)

Pass any prefix (e.g. --filter rustbv_) to scope to a subset, or omit to run all.
EOF
}

detect_tool() {
    if command -v cargo-flamegraph >/dev/null 2>&1; then
        echo "flamegraph"
        return
    fi
    if command -v perf >/dev/null 2>&1; then
        local paranoid
        paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)
        if [[ "$paranoid" -le 2 ]]; then
            echo "perf"
            return
        fi
    fi
    if command -v valgrind >/dev/null 2>&1; then
        echo "callgrind"
        return
    fi
    echo "none"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --secs) SECS="$2"; shift 2 ;;
        --filter) FILTER="$2"; shift 2 ;;
        --tool) TOOL="$2"; shift 2 ;;
        --list) list_groups; exit 0 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "$TOOL" == "auto" ]]; then
    TOOL=$(detect_tool)
fi

if [[ "$TOOL" == "none" ]]; then
    echo "ERROR: no profiler available on this host." >&2
    echo "  - cargo-flamegraph: not installed (cargo install flamegraph)" >&2
    echo "  - perf: needs perf_event_paranoid<=2 (currently $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown))" >&2
    echo "  - valgrind: not installed (apt install valgrind)" >&2
    exit 1
fi

OUT_NAME="${FILTER:-all}"
OUT_DIR="$TARGET_DIR/profile/$OUT_NAME"
mkdir -p "$OUT_DIR"

echo "==> profiler: $TOOL"
echo "==> output:   $OUT_DIR"
echo "==> filter:   ${FILTER:-(all)}"
echo "==> seconds:  $SECS"

export RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes -C debuginfo=2"

# Build the bench binary first so the profiler wraps only the workload, not
# rustc itself. `cargo bench --no-run` builds the bench and prints the path.
echo "==> building bench (release+debuginfo)..."
BENCH_BIN=$(
    cargo bench --manifest-path "$MANIFEST" "${CARGO_PROFILE_OVERRIDES[@]}" \
        --bench vex_engine --no-run --message-format=json 2>/dev/null \
        | python3 -c '
import sys, json
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if msg.get("reason") == "compiler-artifact" and msg.get("target", {}).get("kind") == ["bench"] \
       and msg.get("target", {}).get("name") == "vex_engine":
        exe = msg.get("executable")
        if exe:
            print(exe)
'
)

if [[ -z "$BENCH_BIN" || ! -x "$BENCH_BIN" ]]; then
    echo "ERROR: could not locate built vex_engine bench binary." >&2
    exit 1
fi
echo "==> bench bin: $BENCH_BIN"

BENCH_ARGS=(--bench --profile-time "$SECS")
if [[ -n "$FILTER" ]]; then
    BENCH_ARGS+=("$FILTER")
fi

case "$TOOL" in
    flamegraph)
        # cargo-flamegraph supports an `--bench` mode but invoking the binary
        # directly via `flamegraph` keeps us in control of the args.
        flamegraph -o "$OUT_DIR/flamegraph.svg" -- "$BENCH_BIN" "${BENCH_ARGS[@]}"
        echo "==> flamegraph: $OUT_DIR/flamegraph.svg"
        ;;
    perf)
        perf record -g --call-graph=dwarf -F 997 -o "$OUT_DIR/perf.data" -- \
            "$BENCH_BIN" "${BENCH_ARGS[@]}"
        perf report -i "$OUT_DIR/perf.data" --stdio --sort=overhead,symbol \
            > "$OUT_DIR/perf.txt" 2>&1 || true
        perf script -i "$OUT_DIR/perf.data" > "$OUT_DIR/perf.script" 2>/dev/null || true
        echo "==> perf.data:   $OUT_DIR/perf.data"
        echo "==> perf.txt:    $OUT_DIR/perf.txt (top symbols)"
        echo "==> for SVG: cat $OUT_DIR/perf.script | inferno-collapse-perf | inferno-flamegraph > $OUT_DIR/flamegraph.svg"
        ;;
    callgrind)
        valgrind --tool=callgrind --callgrind-out-file="$OUT_DIR/callgrind.out" \
            "$BENCH_BIN" "${BENCH_ARGS[@]}"
        if command -v callgrind_annotate >/dev/null 2>&1; then
            callgrind_annotate --auto=yes "$OUT_DIR/callgrind.out" \
                > "$OUT_DIR/callgrind.txt" 2>&1 || true
            echo "==> callgrind.txt: $OUT_DIR/callgrind.txt"
        fi
        echo "==> callgrind.out: $OUT_DIR/callgrind.out"
        ;;
    *)
        echo "ERROR: unknown tool '$TOOL'" >&2
        exit 2
        ;;
esac

echo "==> done."
