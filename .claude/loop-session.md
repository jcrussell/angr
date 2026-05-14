## Session log: 2026-05-14 — Phase 1.4 wired (angr-5zw8)

### Closed task

**angr-5zw8** — Phase 1.4: wire MultiwriteAnnotation detection in
`_cb_memory_store_symbolic_full` to the Rust Multi-cell store path
(`store_symbolic_unified_multi`). This closes the last sub-bead of
angr-czph (parent Phase 1 lazy LOAD).

### What landed

**Rust side**

- `native/angr/src/state.rs` — added
  `memory_store_symbolic_multi(addr, value)` mirroring
  `memory_store_symbolic` but dispatching to
  `store_symbolic_unified_multi`.
- `native/angr/src/exploration/state_api.rs` — added
  `_state_memory_store_symbolic_multi(py, state_id, addr_ast, data_ast)`.
  Returns `bool` (true on success). Converts addr/data via
  `claripy_to_rustbv` inside a scoped immutable solver borrow, then
  drops the borrow so the mutable `memory_store_symbolic_multi` call
  is borrow-checker clean.
- `native/angr/src/exploration/mod.rs` — pyclass method
  `state_memory_store_symbolic_multi(state_id, addr_ast, data_ast)`
  wrapping the above. Documents the strchr / gets / fgets
  MultiwriteAnnotation use case.

**Python side**

- `angr/exploration/rust_manager.py`:
  - `_cb_memory_store_symbolic_full` now first tries
    `_try_multi_cell_store(addr_ast, data_ast)`. On True the Rust
    Multi-cell write took the data; on False we fall through to the
    existing `state.memory.store(...)` Python path so writes are not
    lost.
  - `_try_multi_cell_store`: detects `MultiwriteAnnotation` via
    `addr_ast.has_annotation_type(...)`. Gates on
    `_get_stepping_state_id()` (no current Rust state → fall through).
    Routes via `_rust_state_memory_store_symbolic_multi`. Swallows
    AttributeError (old Rust build) and generic exceptions (Rust-side
    failures) so the Python fallback path can retry.
  - `_rust_state_memory_store_symbolic_multi` is a thin Python wrapper
    around the pyclass method — solely so tests can monkey-patch
    routing (PyO3 methods are read-only at the binding level).

**Tests**

- `tests/engines/test_rust_exploration.py` — 7 new tests in the
  `TestErrorRecovery` block:
  * `test_try_multi_cell_store_skips_when_no_annotation`
  * `test_try_multi_cell_store_skips_when_state_id_unknown`
  * `test_try_multi_cell_store_routes_to_pyo3_when_annotated`
  * `test_try_multi_cell_store_falls_through_on_pyo3_failure`
  * `test_try_multi_cell_store_falls_through_on_pyo3_exception`
  * `test_cb_memory_store_symbolic_full_routes_through_multi`
  * `test_cb_memory_store_symbolic_full_falls_back_when_multi_fails`

### Validation

- `pytest tests/engines/test_rust_exploration.py`: **403/403** (up
  from 396 — +7 new tests).
- `cargo test --release --lib`: 756/756.
- `run_single.py fauxware --engine rust`: still solves, finds
  SOSNEAKY password.

### Why this is the right scope for Phase 1.4

The bead description called for "wire one Python SimProcedure path
... via the existing memory_store_symbolic_full callback bypass."
With Rust's default write_range_limit=128 matching Python's
MultiwriteAnnotation default, in practice the callback rarely fires
for strchr-style stores in production today — Rust hits `Multiple`
internally and goes eager. The Phase 1.4 wiring lands the plumbing
contract so that when Phase 2 (angr-qh5u) flips the default
store path to use Multi cells (or when a user widens
write_range_limit past the threshold and TooLarge fires), the
MultiwriteAnnotation routing is in place and tested.

The annotation IS preserved across the bridge for SimProcedure-
originated ASTs via `EXPRESSION_BY_OPERANDS_PTR` reverse cache
(`native/angr/src/claripy_bridge.rs:80-83, 1030-1034`). So the
Python-side `has_annotation_type` check on `addr_ast` works without
needing to teach the RustBV layer about annotations.

### Memories saved this session

(See bd remember section below.)

### Open follow-up

- Phase 2 (angr-qh5u): flip `store_symbolic_unified`
  Multiple/Strided branches to call `store_symbolic_unified_multi`
  unconditionally. Will trigger the load-side Multi collapse on
  every symbolic store automatically and should be where the
  sym-write 2× speedup target lands.
- Phase 1.4 baseline measurement is moot in practice (callback
  rarely fires today, see above). Real before/after numbers will
  surface when Phase 2 lands.

### Next ready (`bd ready` after close):

- angr-myty (P3, daytime perf dashboard) — Python/CI work
- angr-qh5u (P3, Phase 2 lazy STORE) — direct successor; flips
  the default store path to Multi cells. Significantly larger
  surface than Phase 1.4 (touches store_conditional_multiple,
  store_strided, the eager ITE path). Plan for a fresh session.
