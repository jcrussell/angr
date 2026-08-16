#!/bin/sh
# tools/ralph-gate.sh — the shared body of the ralph clean/ and dirty/ state gates.
#
# Both states can produce commits, so both must run the same checks; keeping the
# body here stops the two hooks from drifting (they already had: dirty/gate was
# missing --retry-failures 2).
#
# Three checks, cheapest-and-most-fundamental first:
#
#   1. make check-no-z3 — `cargo check -D warnings --all-targets` over the four
#      `--no-default-features` combos CI's `rust_feature_flags` job gates.
#      Added after angr-nk8p0. That surface had rotted and been repaired four
#      times (angr-1yge9.14, angr-cagbn, angr-c7xno.99, angr-9hkr6) for a
#      structural reason: nothing in the local loop ever compiled a
#      `#[cfg(not(feature = "vex-engine-z3"))]` arm. Check 2 below is
#      default-features-only, and the Stop hook's clippy runs `--all-features`,
#      which is the exact opposite of what those arms need; CI covers it but
#      never fires here (no network to GitHub). So each red sat in the branch
#      until some later iteration happened to remember `make check-no-z3`.
#
#      Runs first because it is by far the cheapest: 0.7s when nothing changed
#      since the last run, ~13s when the crate's sources did (measured
#      2026-08-16 / iter57, all four combos). It builds into the *dev* profile
#      target dir, a separate fingerprint set from both `release-checked`
#      (check 2) and clippy's driver, so the first run after this lands pays a
#      one-time from-scratch dependency build of that profile.
#
#   2. cargo test --profile release-checked — the Rust unit + integration
#      suite. Added after angr-8kk32, where a real correctness regression
#      (NativeRead minting fresh symbolic bytes over a write-demoted file) was
#      caught by a Rust unit test, failed for a full iteration, and nothing
#      noticed: the gate ran pytest + benches only, and CI's `rust_test` job
#      never fires in the loop (the box has no network to GitHub). See bd
#      memory `cargo-test-not-in-ralph-gate`.
#
#      Uses `release-checked` (Cargo.toml: inherits release, lto="off",
#      codegen-units=16, overflow-checks=true, debug-assertions=true — the
#      last one added by angr-03vl4.84, so the debug-only Z3 guards such as
#      `SymContext::to_snapshot`'s residual/assumed bound run here too)
#      rather than plain `--release`
#      (fat LTO + codegen-units=1) — matches .github/workflows/ci.yml's
#      `rust_test` job, which already made this switch. Measured on iteration
#      211 (a touched-native iteration), plain `--release` cost ~181s of
#      relinking 4 LTO'd test binaries (1 unittest + 3 native/angr/tests/*.rs
#      integration binaries) from scratch every time; `release-checked` avoids
#      the LTO relink entirely. Measured 2026-08-07: ~43s once
#      target/release-checked/ is warm (dominated by actually running the
#      ~2500 tests, not by compiling — relinking is near-instant when nothing
#      changed), ~89s for a from-scratch dependency build of that profile
#      (paid once, the first time this runs after switching profiles, since
#      it's a separate target dir from the plain `target/release/` the .so
#      build uses). Both are well under the old ~181s touched-native figure.
#      Note that `cargo clippy --all-targets` does NOT warm it — clippy's
#      driver produces separate fingerprints from rustc's.
#
#      Side effect, not just a speedup: plain `--release` never set
#      overflow-checks, so this is the first time the loop's own local gate
#      (as opposed to CI's rust_test job, which never fires here — no GitHub
#      network) runs the suite with overflow-checks on.
#
#      Runs first: a failing unit test invalidates the bench numbers anyway, and
#      failing fast keeps the gate short in the case we most want to catch.
#
#      Hard-capped at CARGO_TEST_TIMEOUT_SECS (default 600s): a hung test (observed 2026-07-21,
#      context_tests_solver::test_check_branch_feasibility_keeps_branch_on_timeout
#      — Z3_solver_check itself never returns, so the timeout-under-test never
#      fires) otherwise blocks every future iteration's gate forever, since
#      ralph's own iteration timeout does not reliably tear down the process
#      tree cargo forks (a killed cargo/gate process leaves the compiled test
#      binary orphaned and still running). `timeout`'s SIGKILL only reaches the
#      process it directly launched, not that already-forked grandchild, so we
#      also sweep any surviving `rustylib-*` test binary by name after a
#      timeout/failure.
#
#   3. run_regression.py — the fast-tier benchmark regression check, mirroring
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

echo "== ralph gate [1/3]: make check-no-z3 (four --no-default-features combos) =="
make check-no-z3

echo "== ralph gate [2/3]: cargo test --profile release-checked =="
: "${CARGO_TEST_TIMEOUT_SECS:=600}"
# `rc=$?` must be captured on the || side, not inside `if ! cmd`: the `!`
# rewrites $? to 0 in the then-block, so the gate would exit 0 on a failure.
rc=0
timeout --kill-after=30s --signal=TERM "${CARGO_TEST_TIMEOUT_SECS}s" cargo test --profile release-checked || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "cargo test --profile release-checked failed or timed out after ${CARGO_TEST_TIMEOUT_SECS}s (rc=$rc);" \
        "sweeping any orphaned test binaries" >&2
    pkill -KILL -f 'target/release-checked/deps/rustylib-' 2>/dev/null || true
    exit "$rc"
fi

echo "== ralph gate [3/3]: fast-tier benchmark regression =="
exec python tests/benchmarks/run_regression.py \
    --rust-only --skip-bimodal --threshold 0.15 --retry-failures 2
