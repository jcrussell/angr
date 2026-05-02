# Loop session notes (2026-05-02, seventeenth session)

## Task: angr-mboi (investigated, NOT closed)
[perf] Fix mma_howtouse regression: 0.7x vs Python, 1.3GB memory

Did deep investigation, did NOT ship a fix. Task left open with
detailed findings. Two new memories saved.

## Findings

Standalone reproduction:
- Python engine: 4.18s, 207MB RSS, 21MB Python heap (FLAT — no growth across 45 calls)
- Rust engine: 6.50s, 1606MB RSS, 1108MB Python heap
  (linear growth +24MB Python + ~8MB Rust per callable() invocation)

Top tracemalloc growth points (Rust engine, since i=0):
- `ultra_page.py:30,34` (bytearray(page_size)): +359MB each, 181368 pages
  → ~4030 fresh angr memory pages allocated PER callable() invocation
- `dirty_addrs_mixin.py:10` (set updates): +170MB (2.7M blocks)
- `sortedcontainers/sorteddict.py:154` (page index): +29MB

Cause: each callable() creates fresh `factory.call_state` + fresh
RustExplorationManager. The states aren't being freed across calls
when Rust engine is in use. Python alone GC's them fine.

Suspected cycle (rust_state_export.py:52 `RustSolverFallback.attach`):
- state -> solver.eval (bound method) -> wrapper -> state
- Plus state.scratch.rust_mgr = self._rust_mgr

Eliminated as causes:
- Thread-local AST caches (size stays 0 — concrete benchmark)
- Global SymbolicIdentityRegistry (size stays 0)
- Callable.result_path_group / result_state retention (probe set them
  to None explicitly; no change)

Side finding: clearing thread-local AST caches between calls saves
~1.5s of wall time (6.6s → 5.1s, ~23%). Memory unaffected.

## Next session pickup options

1. **Fix angr-mboi**: implement proper teardown of `RustSolverFallback.attach`
   - Add a detach() method that restores `state.solver.eval` etc to the
     originals and removes `state.scratch.rust_mgr`/`rust_found_state_id`.
   - Call detach() on all cached states when RustExplorationManager is
     dropped (e.g. via `__del__`).
   - Verify with: `python tests/benchmarks/run_single.py mma_howtouse --both`
     (Python should be flat, Rust currently grows ~32MB/call in RSS)

2. **angr-xidi (likely false positive)**: standalone runs of
   google2016_unbreakable_1 are 2.2-3.3s — well under the 6.18s
   baseline. The "regression" was suite-induced variance. Could close
   with notes (per `benchmark-update-variance` memory).

3. **AST cache clear on manager Drop**: 23% speedup on mma_howtouse.
   Risk: clearing CLARIPY_AST_CACHE / EXPRESSION_CACHE may break AST
   identity if user holds expressions across manager boundaries. Check
   the test suite carefully.

## Other ready tasks (unchanged from sixteenth session)

- angr-3tek (P2): native read/write SimProcs blocked by stale-cache
  issue — needs Rust→Python sync per `avoid-enabling-native-read`
- angr-mboi (P2): ABOVE
- angr-xidi (P2): see above
- angr-w4os, angr-2fs0, angr-1f8s, angr-cbko, angr-3ijo, angr-8em4 (P3)

## Pre-existing concerns

- Pre-existing baseline timing variance in `run_regression.py`
  (per `benchmark-update-variance` memory).
