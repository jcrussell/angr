# Loop session notes (2026-05-06, eighty-sixth loop session — DONE)

## Status: COMPLETE — angr-lbze closed

Loop benchmark gate was broken: every iteration shipped commits without
functional regression validation (28+ logged iterations).

## Root cause

`tests/benchmarks/run_regression.py` uses
`multiprocessing.get_context("spawn").Pool(1)` to isolate each benchmark.
`spawn` re-execs Python with `sys.path` inherited from the parent. When
the orchestrator launches `python tests/benchmarks/run_regression.py ...`,
Python sets the parent's `sys.path[0]` to that script's directory
(`tests/benchmarks/`) — REPO_DIR is NOT on `sys.path`. The spawn child
inherits this, so `import angr` fails inside the worker.

The angr editable install also has no `.pth` in site-packages — only the
`__editable__*finder.pyc`. Without the matching `.pth` import side-effect,
that finder never registers. So `import angr` only works when REPO_DIR is
implicitly in `sys.path` (e.g. cwd matches via the `''` entry on `python -c`).

## Fix

`run_optimization_loop.py:run_benchmark_gate` now prepends REPO_DIR to
`PYTHONPATH` in the env passed to systemd-run. PYTHONPATH lands on every
interpreter in the scope, including spawn workers, and is independent of
sys.path[0].

Also added gate-broken detection: when every failure matches an
import/harness pattern (`No module named`, `ImportError`,
`ModuleNotFoundError`, `subprocess crashed`), the result dict now has
`gate_broken=True`. The orchestrator's main loop logs `log.error("BENCHMARK
GATE BROKEN ...")` instead of `log.warning("BENCHMARK REGRESSIONS ...")`
in this case, so a future re-break is loud rather than silent.

## Verification

- Manual call to `run_benchmark_gate("4G")` now reports `passed=11
  failed=1 gate_broken=False`. The single failure is `csgames2018:
  Rust engine failed: list index out of range` — a pre-existing real bug,
  not the gate.
- Manual repro with PYTHONPATH stripped reports `failed=12
  gate_broken=True` with the original `No module named 'angr'` failures.
- pytest tests/engines/test_rust_exploration.py: 243/243 pass.

## Files modified

- `run_optimization_loop.py` (run_benchmark_gate env fix + gate_broken flag,
  main-loop branch in benchmark gate handling)
