# Loop session notes (2026-05-01, twelfth session)

## Closed this session

### angr-dja4 — Expand benchmark baseline tracking

Pre-fix: `tests/benchmarks/baseline_timings.json` had 16 entries, only
7 with full algorithmic stats (callback_count, state_creations, steps,
peak_memory_mb). The 9 entries with python_time/rust_time only were
populated before the algorithmic-tracking commit (c6515f061) and never
backfilled. Several entries (codegate_2017-angrybird, csgames2018,
whitehatvn2015_re400) were also "orphaned" — listed in the baseline
but not in any `FAST_SUITE`/`MEDIUM_SUITE` so they were not exercised
by `run_regression.py`.

Fix: extended suite tuple format to support a per-entry `rust_only`
flag `(name, timeout, [strategy, [rust_only]])` and re-populated the
baseline:

- `rust_only=True` is set for entries where Python is unreliable under
  the 4GB memory limit (OOM/timeout) OR where Z3 model nondeterminism
  causes benign output divergence (multiple valid Z3 models for the
  same find/avoid set). In both cases Rust timing + algorithmic
  metrics are still tracked, just no Python comparison or output check.
- Marked all FAST_SUITE entries that historically had `python_time:
  null` as `rust_only=True` (fauxware, defcamp_r100, ais3_crackme,
  google2016_unbreakable_0/1, strcpy_find, flareon2015_2,
  defcamp_r100__dfs).
- Added 7 new tracked entries: unmapped_analysis,
  defcon2016quals_baby-re, csgames2018, whitehatvn2015_re400,
  ekopartyctf2016_sokohashv2, mma_howtouse,
  hackcon2016_angry-reverser; pulled codegate_2017-angrybird in from
  the orphan pool.

Result: 22 entries (was 16), all with full stats (was 7). Regression
suite runs all 22 in ~120s and detects algorithmic regressions via
`--check-counts`.

**Verification:**
- `python tests/benchmarks/run_regression.py --update --full --check-counts`
  → 22 passed, 0 failed
- `python -m pytest tests/engines/test_rust_exploration.py` → 208/208

## Ready P-tasks remaining

- angr-vt0t (P3 categorize remaining ~280 except blocks)
- angr-8em4 (P3 panic audit — 543 sites, must split)
- angr-3ijo (P3 bincode for IRSB serialization spike)
- angr-bgv0 (P3 Z3 FP theory)
- angr-awm3 (P3 CAS/LLSC statement handling)
- angr-v4db (P3 extract god-methods)
- angr-4dxi (P4 memory permission enforcement)
- angr-borb (P4 StateId / Address newtypes)
- angr-7x45 (P4 perf SLA enforcement in CI)
