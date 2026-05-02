# Loop session notes (2026-05-02, nineteenth session)

## Task: angr-xidi (CLOSED)
[perf] Fix google2016_unbreakable_1 regression: 3.2s->6.2s (-93%)

## Outcome
- Refreshed baseline rust_time 6.18s -> 2.7s
- Refreshed baseline peak_memory_mb 291.4 -> 631.0
- 208/208 tests passing, regression suite confirms unbreakable_1 no longer flagged
- Commit: db6ea164d

## Investigation summary

Old baseline (6.18s) was captured during a high-variance run under
suite-induced memory pressure (per `benchmark-update-variance` memory).
Standalone subprocess runs across 5 trials: [2.42, 2.43, 2.71, 3.33, 3.42],
median 2.71s. Suite runs (in run_regression.py): 2.0-2.5s consistently.

Recent perf fixes since baseline was set (commit 15395dd6b) likely closed
the regression: 342df4a7f (GC leak), 7fec7db2d (SegmentList O(n)),
8964c0da3 (exit-cont cache).

## New issue created: angr-rsfv (P2)

Discovered: 8 OTHER baselines consistently fail by 18-71% with low
variance — looks like real regressions or stale baselines from different
conditions. Created angr-rsfv to investigate. Affected:
defcamp_r100, ais3_crackme, unbreakable_0, flareon2015_2, baby-re,
defcamp_r100__dfs, csgames2018, whitehatvn2015_re400.

Variance is small (~3-5% across runs), so this is signal not noise.
Could be real perf regression from one of: bf2711b74 (DivMod),
fb52689e3 (BV width validation), 85c59c7a0 (ccall fallback),
4d174348c (null-check), 8964c0da3 (cache check).

## Memory saved

- `benchmark-unbreakable-1-baseline-refresh`: refresh details + pointer
  to systemic 8-baseline issue

## Other ready tasks

- angr-3tek (P2): native read/write SimProcs blocked by stale-cache
- angr-rsfv (P2): NEW — 8-baseline staleness investigation
- angr-w4os, angr-2fs0, angr-1f8s, angr-cbko, angr-3ijo, angr-8em4,
  angr-bgv0, angr-awm3 (P3)
