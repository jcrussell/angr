#!/bin/sh
# tools/ralph-gate.sh — the shared body of the ralph clean/ and dirty/ state gates.
#
# Both states can produce commits, so both must run the same checks; keeping the
# body here stops the two hooks from drifting (they already had: dirty/gate was
# missing --retry-failures 2).
#
# Two checks, cheapest-and-most-fundamental first:
#
#   1. cargo test --release — the Rust unit + integration suite. Added after
#      angr-8kk32, where a real correctness regression (NativeRead minting fresh
#      symbolic bytes over a write-demoted file) was caught by a Rust unit test,
#      failed for a full iteration, and nothing noticed: the gate ran pytest +
#      benches only, and CI's `rust_test` job never fires in the loop (the box
#      has no network to GitHub). See bd memory `cargo-test-not-in-ralph-gate`.
#
#      Cost: ~15s when the test binaries are warm (any iteration that did not
#      touch native/), ~2m when they must be rebuilt (any iteration that did).
#      Note that `cargo clippy --all-targets` does NOT warm them — clippy's
#      driver produces separate fingerprints from rustc's.
#
#      Runs first: a failing unit test invalidates the bench numbers anyway, and
#      failing fast keeps the gate short in the case we most want to catch.
#
#   2. run_regression.py — the fast-tier benchmark regression check, mirroring
#      .github/workflows/ci.yml::benchmark_regression
#      (--rust-only --skip-bimodal --threshold 0.15).
#
#      --retry-failures 2: sub-second fast-tier benches are noise-dominated. Per
#      the `benchmark-regression-noise-floor` memory, a single failing run is not
#      a real regression — only believe it if the same bench fails twice
#      consecutively. This flag mechanizes that rule.
#
# Gate semantics come from .ralph/config.toml: run_when="commits-only" (so this
# only fires on iterations that committed) and soft_fail=true (a failure surfaces
# in the next prompt's {{.GateResult}}; the FSM does not enter failed{}).
#
# Env: RALPH_REPO RALPH_ITER RALPH_STATE RALPH_PROMPT_FILE RALPH_ITER_JSON

set -e
export PATH="$HOME/.cargo/bin:$PATH"
# shellcheck disable=SC1091
. /home/ubuntu/repos/angr/.venv/bin/activate

echo "== ralph gate [1/2]: cargo test --release =="
cargo test --release

echo "== ralph gate [2/2]: fast-tier benchmark regression =="
exec python tests/benchmarks/run_regression.py \
    --rust-only --skip-bimodal --threshold 0.15 --retry-failures 2
