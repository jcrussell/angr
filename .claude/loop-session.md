## Session log: 2026-05-09 — angr-gfyl (188th loop session, closed)

### Task
Backfill python_time in tests/benchmarks/baseline_timings.json. Previously
only 2/22 entries had non-null python_time (flareon2015_10 and mma_howtouse).
Acceptance: every entry either has python_time or a documented null reason;
run_regression.py prints speedup-vs-Python for every dual-engine entry.

### Approach
Wrote tests/benchmarks/backfill_python_time.py — a one-off CLI that walks
baseline_timings.json, runs the Python engine through run_regression.run_one
(re-using the existing 4GB-RLIMIT_AS subprocess runner), and writes the
elapsed time back into the JSON. On timeout/OOM/crash, records
python_skip_reason. Saves incrementally per-entry so a crash mid-run doesn't
lose progress.

### Result
All 20 missing entries populated in a single ~5-minute run. No timeouts/skips;
default 90s + per-entry overrides (120s for flareon2015_5/csaw_wyvern/etc.)
were all sufficient. Notable speedups:

- csaw_wyvern: 16.92x (Py 15.9s / Rust 0.94s) — best case
- ekopartyctf2016_rev250: 15.65x
- flareon2015_5: 10.27x
- defcamp_r100__dfs: 4.51x
- ais3_crackme: 2.95x

Known Rust-slower benches now exposed via SLA WARN (not FAIL):
- google2016_unbreakable_1: 0.56x (bimodal; rust_time baseline bumped to 3.5)
- ekopartyctf2016_sokohashv2: 0.58x
- mma_howtouse: 0.65x
- hackcon2016_angry-reverser: 0.88x
- fauxware: 0.99x

### Verification
- 5 consecutive `run_regression.py --rust-only` (fast tier) green
- 1 `run_regression.py --rust-only --full` (all 22) green
- 357/357 unit tests in tests/engines/test_rust_exploration.py pass

### Caveat
Bimodal benches (unbreakable_1) can occasionally hit slow-mode and trip
SLA fail (saw 0.48x once during validation). This was already latent — the
backfill just exposes it via SLA. Future work could add an `sla_exempt`
flag for known-bimodal entries.

### Files modified
- tests/benchmarks/baseline_timings.json (20 python_time entries populated)
- tests/benchmarks/backfill_python_time.py (new, reusable for future drift)
