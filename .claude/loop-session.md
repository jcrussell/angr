## Session log: 2026-05-09 — angr-6a7u (187th loop session)

Closed: refresh stale benchmark baselines after recent perf wins.

### Approach
Ran `tests/benchmarks/run_regression.py --full --rust-only --no-sla --update`
once to capture current Rust timings, then ran several verification rounds
without --update to validate stability.

### Real wins (baseline shrinks reflect actual perf gains)
- google2016_unbreakable_1: 3.523 → ~1.7s typical (but bimodal up to ~3.3s)
- unmapped_analysis: 2.148 → 0.789s (much faster after recent FxHash work)
- csaw_wyvern: 1.257 → 0.94s
- ekopartyctf2016_sokohashv2: 12.091 → ~9.5s typical (bimodal, can hit 15s+)
- flareon2015_5: 6.563 → 5.567s
- securityfest_fairlight: bimodal — runs ~7.8s OR ~15.0s

### Conservative bumps for high-variance Z3-nondeterministic benches
Three benchmarks have bimodal/noisy distributions due to Z3 model
nondeterminism affecting executed paths. Set baselines at the slow mode +
some margin so the 15% regression threshold absorbs noise rather than
flagging false positives:

- google2016_unbreakable_1: --update wrote 1.683 → bumped to 3.5
  (max observed across 8 runs: 3.26s; 3.5*1.15=4.0s budget)
- securityfest_fairlight: --update wrote 15.096 → bumped to 16.0
  (max observed: 15.10s; 16.0*1.15=18.4s budget)
- ekopartyctf2016_sokohashv2: --update wrote 10.083 → bumped to 16.0
  (max observed: 15.36s; 16.0*1.15=18.4s budget)

### Verification
5+ consecutive `--full --rust-only --no-sla` runs all green after bumps.

### Files modified
- tests/benchmarks/baseline_timings.json
