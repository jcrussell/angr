## Session log: 2026-05-17 — angr-97l8 per-simprocedure fallback telemetry

### Status: CLOSED — commit 156cdd64d. 438/438 pass (+1 net test).

### Task

**angr-97l8 (P3, task)** — "Per-SimProcedure fallback telemetry in
get_solver_stats()". Add `Dict[str, int]` of procedure_name →
fallback_count so we know which native procedures need implementing
next.

### What landed

- Added `simprocedure_fallback_by_name: HashMap<String, u64>` to
  `RustExplorationManager` (alongside scalar
  `simprocedure_python_fallback_count`).
- Incremented at the TWO Python-fallback sites:
  - `native/angr/src/exploration/run_loop.rs:305` (top-of-step hook)
  - `native/angr/src/exploration/stepping.rs:597` (interpreter exit)
- Exposed via `_stats()` and `_get_fallback_stats()` under
  `simprocedure_fallback_by_name` (alongside the scalar).
- Surfaced in `rust_manager.py::get_performance_summary` — prints top 10
  procedures by fallback count below the scalar line.
- Tests: extended `test_python_procedure_symbolic_arg_falls_back_to_python`
  to assert by-name records "sym_proc" + sum-equals-scalar invariant.
  Added `test_simprocedure_fallback_by_name_empty` for fresh-manager.

### Acceptance criterion note

bd ticket said `get_solver_stats()` returns `simprocedure_fallback_by_name`,
but that helper returns process-wide Z3 counters only. The per-manager
fallback counters naturally live under `_stats()` /
`_get_fallback_stats()`, where the scalar `simprocedure_python_fallback_count`
already lived. Wired there. Close-reason on the bd ticket documents
the divergence.

### Files modified

- native/angr/src/exploration/mod.rs (+7)
- native/angr/src/exploration/run_loop.rs (+4)
- native/angr/src/exploration/stats_api.rs (+10)
- native/angr/src/exploration/stepping.rs (+4)
- angr/exploration/rust_manager.py (+5)
- tests/engines/test_rust_exploration.py (+28)

### Memories saved

- `invariant-simprocedure-fallback-two-sites` — both run_loop.rs and
  stepping.rs increment the counters; future fallback bookkeeping
  changes must touch both.
- `telemetry-simprocedure-fallback-by-name` — pointer to the new
  Dict[str, int] surface and how to use it (prioritizing native
  SimProcedure implementations).

### Followup work for next session

- None directly tied to this ticket. The new telemetry is a tool
  available for the next round of native SimProcedure picking — run
  any benchmark, read `mgr.get_fallback_stats()['simprocedure_fallback_by_name']`,
  the highest-count names are the candidates.
- Carryover from previous session: mma_howtouse remains 0.58x; root
  cause is no longer AST cache lookup. Fresh profiling investigation
  warranted if there's appetite. Also the angr-7vcx
  `_scan_symbolic_pages` followup is still open.
