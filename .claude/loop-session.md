## Session log: 2026-05-13 — angr-1583 (PR-time benchmark regression gate)

### Task closed

**angr-1583** (P1, "PR-time fast-tier benchmark regression gate
(<2m, 15% threshold)") — closed at this session.

### Changes

- `.github/workflows/ci.yml`: new `benchmark_regression` job on PR.
  Checkouts `angr`, `angr/binaries`, `angr/angr-examples`; builds the
  Rust .so; runs `tests/benchmarks/run_regression.py --rust-only
  --skip-bimodal --threshold 0.15`. Job timeout 15m (Rust build
  dominates cold); bench step timeout 3m to enforce the <2m bench-work
  SLA.
- `.github/workflows/nightly-ci.yml`: added the missing
  `angr/angr-examples` checkout + `ANGR_EXAMPLES_DIR` env on the
  existing nightly bench step. That step was silently broken — without
  the corpus, `run_regression.py` exits with `Missing examples`.
- `tests/benchmarks/run_regression.py`:
  - `BIMODAL_BENCHMARKS = {unbreakable_1, fairlight, sokohashv2}`
  - new `--skip-bimodal` flag, applied after suite normalization so
    `--full --skip-bimodal` works too.
- `tests/benchmarks/run_single.py`: `EXAMPLES_DIR` now honors
  `ANGR_EXAMPLES_DIR` env var; falls back to `~/repos/angr-examples`.
- `CLAUDE.md`: documented the two CI gates and the new env override.

### Validation

- `cargo check --release` — clean.
- `python tests/benchmarks/run_regression.py --help` shows
  `--skip-bimodal` correctly.
- `--skip-bimodal` filter verified: 22-bench full suite → 19 after skip,
  and only the 3 bimodal names are removed.
- `ANGR_EXAMPLES_DIR=/tmp/foo python -c ...` confirms env override.
- `pytest tests/engines/test_rust_exploration.py`: 389 passed, 3
  pre-existing failures (`test_dcas_cmpxchg16b_no_match_keeps_memory`,
  `test_pipe_native_dispatch_creates_two_fds`,
  `test_dup2_native_dispatch_redirects_stdin`) — same as documented at
  HEAD in the prior loop-session.md and on `bd memories
  pre-existing-dcas-test-failure`.

### Observations worth saving as memory

- Nightly bench step was latently broken: it checks out `angr/binaries`
  but not `angr/angr-examples`, and `run_regression.py` resolves the
  corpus from `~/repos/angr-examples/examples`. Without the env-var
  override path, the nightly job's bench step has been exiting with
  `Missing examples`. Worth a memory under `latent-nightly-bench-broken`.
- Local PR-gate dry run on this dev box flagged 0-5 sub-second
  benchmarks per run with 13-29% relative deltas (defcamp_r100,
  google2016_unbreakable_0, strcpy_find, flareon2015_2). The deltas
  are consistent with system-load jitter on a sub-second baseline —
  i.e. 60ms-150ms absolute variance is normal at this scale. CI runners
  are usually more stable, so the 15% gate is expected to be reliable
  in CI; but the team should be ready to either rebaseline or
  introduce an `--abs-threshold` floor if flakes show up in the wild.
  Worth a `pr-bench-gate-jitter-risk` memory.

### Next ready P1 candidates (for the next session)

- `angr-3hzg` — Property-based differential fuzzer
- `angr-qdwt` — Pre-PR final docs sweep (intentionally LAST)
