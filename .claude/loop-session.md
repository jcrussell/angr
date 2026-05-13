## Session log: 2026-05-13 — angr-3hzg (Property-based differential fuzzer)

### Task

**angr-3hzg** (P1) — Property-based differential fuzzer: Rust ≡ Python
on random binaries/seeds.

### Changes

- `tests/benchmarks/property_fuzzer.py` (new, 320 lines). Samples
  random (example, strategy, seed) tuples from the rust_ok=True
  catalog, runs both engines in subprocesses (reusing
  `run_single._run_in_child` with the 4 GB RLIMIT_AS), and compares
  normalized stdout. Classifies each trial as `pass`,
  `diverge`, `expected-diverge`, `py-fail`, `rust-fail`, or
  `both-fail`. `expected-diverge` is for examples on the known
  rust_only/bimodal allow-list (derived programmatically from
  `run_regression.FAST_SUITE`/`MEDIUM_SUITE` rust_only=True entries
  plus `BIMODAL_BENCHMARKS`) — divergence on those is benign, not a
  regression. Strict mode (`--strict`) fails only on NEW divergences
  or Rust crashes.

  Module docstring includes the 3-step triage workflow (reproduce →
  classify benign/path/crash → fix-and-reverify). Subprocess timeouts,
  memory limits, seeded sampling, JSON report (`--report`), eligibility
  listing (`--list`), and `--only` override all wired up.

  Located in `tests/benchmarks/` rather than `tests/` (where the task
  spec suggested) because `tests/types/` shadows the stdlib `types`
  module when sys.path[0] is `tests/`. The benchmark dir is on the
  same conventional level as `run_single.py` / `run_regression.py`.

- `.github/workflows/nightly-ci.yml`: new `property_fuzzer` job. Runs
  `--trials 50 --seed 1 --strict` (~5 min wall) and uploads the JSON
  report as an artifact even on failure. 20 min job timeout.

### Validation

- `python tests/benchmarks/property_fuzzer.py --list` — lists 17
  eligible examples.
- `--trials 2 --only fauxware --seed 42`: 2 EXPECTED-DIVERGE trials
  (fauxware is rust_only=True). Strict mode would not fail.
- `--trials 5 --seed 1`: produced mix of PASS (defcamp_r100,
  mma_howtouse) and EXPECTED-DIVERGE (csgames2018, unmapped_analysis).
- `--trials 3 --seed 7 --report /tmp/fuzz.json` (captured run):
  1 PASS (google2016_unbreakable_0), 1 EXPECTED-DIVERGE
  (codegate_2017-angrybird), 1 PY-FAIL (flareon2015_5 timeout). JSON
  report well-formed.
- `pytest tests/engines/test_rust_exploration.py -q --tb=no`:
  389 passed, 3 pre-existing failures (`test_dcas_cmpxchg16b_no_match_keeps_memory`,
  `test_pipe_native_dispatch_creates_two_fds`,
  `test_dup2_native_dispatch_redirects_stdin`) — same as documented
  on `bd memories pre-existing-dcas-test-failure`. No regressions
  from this change (it only adds a new file).

### Observations worth saving as memory

- The fuzzer revealed that `defcamp_r100` in BFS mode actually
  produces matching output between engines — yet it's flagged
  `rust_only=True` in `run_regression.py`. Either it was marked
  conservatively, or the divergence was DFS-only / Z3-seed-dependent.
  Worth a follow-up to audit the rust_only=True list and promote
  cases that consistently pass. Memory under
  `fuzzer-rust_only-audit-candidate`.
- Only 2 of 17 eligible examples (`flareon2015_10`, `mma_howtouse`)
  are NOT on the known-divergence list, so under default sampling
  ~88% of trials will be `expected-diverge`. The fuzzer is still
  useful as a regression guard (any NEW divergence fails strict
  mode), but the "signal" trials are sparse. Long-term, the
  rust_only=True list needs an audit to identify which examples
  can be promoted to first-class output comparison. Memory under
  `fuzzer-eligible-pass-set-small`.

### Files touched

- `tests/benchmarks/property_fuzzer.py` (new)
- `.github/workflows/nightly-ci.yml` (new `property_fuzzer` job)
- `.claude/loop-session.md` (this file)

### Next ready P1 candidates

- `angr-qdwt` — Pre-PR final docs sweep (intentionally LAST)
