# Loop session notes (2026-05-02, eighteenth session)

## Task: angr-mboi (CLOSED — fixed)
[perf] Fix mma_howtouse regression: 0.7x vs Python, 1.3GB memory

## Outcome
- mma_howtouse peak_mem: **1606MB → 285MB** (5.6x reduction)
- Tests: 208/208 passing
- Wall time unchanged (still ~6.5s vs Python ~4.2s)
  → Remaining gap is AST cache lookup overhead, NOT memory leak.
    See `mma-howtouse-cache-clear-speedup` memory.
- Commit: 342df4a7f

## Root cause (recorded in pyo3-pyclass-cycle-leak memory)

PyO3 `#[pyclass]` does NOT implement `__traverse__`/`__clear__` by
default. If a pyclass holds `Py<PyAny>` refs that participate in a
cycle through Python objects, the cycle is invisible to Python's
cycle-GC and leaks permanently — `gc.collect()` does NOT help.

Cycle in this codebase:
  `mgr -> _callbacks (PyO3 PythonCallbacks) -> bound method -> mgr`
  AND
  `mgr -> _rust_mgr (PyO3 RustExplorationManager).callbacks -> bound method -> mgr`

Both held strong refs to bound methods (`mgr._cb_*`). Both pyclasses
needed GC support for the cycle to be collectible.

## Fix

Added `__traverse__` and `__clear__` to both `PythonCallbacks` (callbacks.rs)
and `RustExplorationManager` (exploration/mod.rs). Factored the
per-field traversal/clearing into helpers `traverse_fields` /
`clear_fields` on `PythonCallbacks` so the manager's `__traverse__`
can delegate for its embedded clone (set_callbacks moves a cloned
copy into the manager).

## Failed approach (recorded in avoid-weakref-callbacks memory)

Tried wrapping callbacks in weakref-based closures on the Python side.
Worked for memory (same 285MB) but caused 19-72% perf regression across
8 benchmarks due to per-callback overhead. Reverted in favor of the
Rust-side GC fix which has zero per-call overhead.

## New memories saved

- `pyo3-pyclass-cycle-leak`: root cause + fix pattern
- `avoid-weakref-callbacks`: failed approach
- `benchmark-mma-howtouse-leak-fix`: before/after numbers

## Other ready tasks (unchanged)

- angr-3tek (P2): native read/write SimProcs blocked by stale-cache
- angr-xidi (P2): likely false positive (suite-induced variance)
- angr-w4os, angr-2fs0, angr-1f8s, angr-cbko, angr-3ijo, angr-8em4 (P3)

## Pre-existing concerns (unchanged)

- Pre-existing baseline timing variance in `run_regression.py`
  (per `benchmark-update-variance` memory). Same 8 failures occur
  WITH or WITHOUT this commit — verified by stashing and re-running.
- mma_howtouse wall time still 0.65x. The 1.5s AST cache overhead
  (per `mma-howtouse-cache-clear-speedup`) remains a separate fix.
