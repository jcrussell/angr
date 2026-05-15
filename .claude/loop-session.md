## Session log: 2026-05-15 — Perf dashboard landed (angr-myty)

### Closed task

**angr-myty** — Daytime perf dashboard (GitHub Pages from nightly artifacts).
End-to-end pipeline shipped: nightly emits a timestamped JSON record →
artifact upload → aggregator merges last 50 commits → static Chart.js
dashboard publishes to GitHub Pages.

### What landed

**Pipeline**

1. `tests/benchmarks/run_regression.py` — new `--history-record PATH` flag.
   Writes `{schema_version, timestamp, commit, branch, results}` JSON
   (commit/branch resolved from `GITHUB_SHA`/`GITHUB_REF_NAME` first,
   `git rev-parse` fallback). Activates only when `results` is non-empty
   so a failed run doesn't poison the time series.
2. `.github/workflows/nightly-ci.yml::benchmark_regression` — now passes
   `--history-record bench_history_record.json` and uploads it as
   `bench-history-${{ github.sha }}` (90-day retention, `if-no-files-found:
   warn`, `if: always()` so partial-failure runs still publish their
   partial results).
3. `tools/aggregate_bench_history.py` — NEW. Walks a directory of records,
   de-duplicates by commit SHA (newest wins), sorts chronologically,
   truncates to `--max-points` (default 50), emits
   `{generated, max_points, benchmarks, series}` JSON. Skips malformed
   files with a warning rather than failing.
4. `docs/dashboard/index.html` + `dashboard.css` + `dashboard.js` — NEW
   static dashboard. Featured grid renders fauxware / ais3_crackme /
   csaw_wyvern rust_time histories. Detail section has bench + metric
   selectors (rust_time / python_time / speedup / peak_memory_mb /
   callback_count / state_creations / steps). Chart.js loaded from CDN
   (jsdelivr pinned to 4.4.1).
5. `.github/workflows/perf-dashboard.yml` — NEW. Trigger: `workflow_run`
   after Nightly CI success, `workflow_dispatch`, plus `push` on
   dashboard sources. Downloads up to 60 most-recent nightly artifacts
   via `gh run list`/`gh run download`, runs the aggregator, copies the
   site to `_site/`, uploads via `actions/upload-pages-artifact@v3`,
   deploys with `actions/deploy-pages@v4`.
6. `.gitignore` — added `docs/dashboard/data.json` so the generated
   payload doesn't get committed.

### Validation

- Help: `run_regression.py --help` lists `--history-record PATH`.
- Direct call to `save_history_record()` produces a record with
  `commit=73e049186f...`, `branch=rust-symex`, valid ISO Z timestamp.
- Aggregator end-to-end with 5 synthetic records: outputs
  `{series: 5 points, benchmarks: 4}`, chronologically sorted, dedup'd
  by commit, last entry == latest timestamp.
- Edge cases: empty input dir → `{series:[], benchmarks:[]}`; malformed
  JSON / wrong schema_version → warned + skipped.
- HTML parses cleanly via `html.parser`.
- JS parses cleanly under node (DOM ReferenceError at runtime is
  expected — Chart.js loads in-browser).
- YAML lint: both workflows parse with `yaml.safe_load`.
- `tests/engines/test_rust_exploration.py -k test_basic` — 1/1 passed
  (smoke: Rust .so still loads).

### Bootstrap note

`.venv/bin/` and `pyvenv.cfg` were missing on session start. Restored
from `~/repos/angr.tar.gz` (`tar -xzf angr.tar.gz angr/.venv/bin
angr/.venv/pyvenv.cfg angr/.venv/include`). Site-packages were intact.
This is the `env-venv-corruption` pattern that lives in bd memories.

### Caveats / follow-ups

- First nightly after merge will publish a 1-point dashboard; the
  series fills out over ~50 days.
- The `gh run download` loop runs sequentially — fine for 60 runs but
  could be parallelised if we ever raise the window past 200.
- `workflow_run` does NOT pass `GITHUB_REF` matching the target
  workflow's ref, so the dashboard always reflects whatever was
  committed at the time the publish workflow ran. That's intentional
  for `master`-only mode; revisit if we ever want per-branch dashboards.
