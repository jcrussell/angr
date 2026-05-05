# Loop session notes (2026-05-05, forty-ninth loop session)

## Task: angr-2aih (P0) — DONE
Re-run 22-benchmark sweep, refresh baseline_timings.json, confirm
214/214 RustExplorationManager tests + 391 Rust unit tests still pass.

## What I did
1. cargo check --release: clean
2. pip install -e . --no-build-isolation --no-deps: rebuilt .so
3. cargo test --release --lib: 391 passing
4. pytest tests/engines/test_rust_exploration.py: 214 passing
5. python tests/benchmarks/run_regression.py --full --update — 22/22 passed
6. Diffed baseline_timings.json: see project_benchmark_status memory.

## Result
- All 22 benchmarks correct.
- Only 1 SLA warning: mma_howtouse 0.63x (long-standing
  Callable-heavy workload, expected).
- All 9 prior regressors recovered or determined to be stale baselines:
  - sym-write: TIMEOUT 60s → 0.42s ✓
  - codegate_2017-angrybird: stable across commits; old baseline was stale
  - securityfest_fairlight 16.05s → 12.83s (FP-theory remainder)
  - ekopartyctf2016_rev250 2.25s → 2.03s ✓
  - flareon2015_5 8.33s → 6.56s ✓
  - mma_howtouse 7.32s → 6.74s ✓ (SLA WARN persists)
  - whitehatvn2015_re400 1.52s → 1.26s ✓
  - flareon2015_2 4.52s → 3.62s ✓
  - unmapped_analysis 2.49s → 2.15s ✓
- Bonus: hackcon2016_angry-reverser 35.0s → 12.0s (-66%, prior baseline stale)

## Files modified
- tests/benchmarks/baseline_timings.json (refreshed via --update)
- .claude/loop-session.md
- memory/project_benchmark_status.md (rewritten to reflect post-fix state)
- new memory: benchmark-2026-05-05-sweep

## Beads
- angr-2aih: claimed → close after commit.

## Next-up
- angr-fbxi (P1) — single-bench baseline refresh for unbreakable_1
  and hackcon angry-reverser. Likely already covered by this sweep
  refresh (both bumped during --update). Verify and close.
- Optimization-side ready candidates (P1): angr-8kht (avoid double-Z3
  in solutions+range), angr-nwbx (cache claripy↔Z3 conversion for
  register sync), angr-6uhh (eliminate per-step clones).
- Investigation candidates: angr-eygl (differential test harness),
  angr-pufm (concretization fallback for intractable solution sets).
