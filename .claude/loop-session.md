## Session log: 2026-05-16 — angr-uq4n.2 inspect marshalling layer

### Status: ready-to-close

### Task

**angr-uq4n.2** — "Python callback marshalling layer for inspect events."

Build the Rust → PyO3 → Python state.inspect.action() round-trip
for mem_read / mem_write inspect events. Acceptance: skeleton
marshalling layer that delivers a stub event to a registered
Python callable, even without a real event source yet.

### Plan

Per the survey from uq4n.1 (memory `inspect-mem-dispatch-surfaces`):

**Rust side (`native/angr/src/callbacks.rs`):**
- Add `inspect_mem_read`, `inspect_mem_write` `Option<Py<PyAny>>` fields on `PythonCallbacks`
- Add `inspect_enabled: u8` bitmask field (zero-overhead skip)
- Add setters + getter + `call_inspect_mem_{read,write}` helpers
- Wire `__traverse__`/`__clear__`

**Python side (`angr/exploration/rust_state_proxy.py`):**
- Replace `_NoOpInspectProxy` with a `RustInspectProxy` that
  supports mem_read / mem_write BPs (other events still raise).
- Backed by manager-level BP storage (single set for all states).

**Python side (`angr/exploration/rust_manager.py`):**
- Add `_inspect_breakpoints` dict + global `_inspect_proxy`.
- Add `_cb_inspect_mem_read` / `_cb_inspect_mem_write` dispatchers.
- Add `_update_inspect_bitmask` aggregator.
- Wire callbacks in `_setup_callbacks`.

**Tests:** add tests in `test_rust_exploration.py` that exercise
the round-trip by calling `_cb_inspect_mem_*` directly with
synthetic args.

### Files modified

- native/angr/src/callbacks.rs — added `inspect_mem_{read,write}` callback slots, `inspect_enabled` bitmask, setters/getter, `call_inspect_mem_{read,write}` helpers, wired GC (`__traverse__`/`__clear__`).
- angr/exploration/rust_state_proxy.py — added `RustInspectProxy` (manager-shared facade with SimInspector-compatible `b`/`make_breakpoint`/`add_breakpoint`/`remove_breakpoint`/`action` for mem_read / mem_write; other events still raise). Kept `_NoOpInspectProxy` for the no-manager fallback. `RustStateProxy.inspect` now routes to `mgr._get_inspect_proxy()` when a manager is attached.
- angr/exploration/rust_manager.py — added `_inspect_breakpoints` storage, `_INSPECT_EVENT_BITS`, `_inspect_dispatch_depth` reentrancy guard, `_get_inspect_proxy`/`_update_inspect_bitmask`/`_make_inspect_state_for`/`_dispatch_inspect_event`/`_cb_inspect_mem_{read,write}` dispatchers. Wired in `_setup_callbacks`.
- tests/engines/test_rust_exploration.py — added `TestRustInspectMarshalling` (10 tests: callbacks slots, BP registration + bitmask, unsupported-event rejection, mem_read dispatch, mem_write w/ value AST, no-BP fast path, reentrancy guard, Rust-side `call_inspect_*` invocation, unset-callback no-op, proxy routing).

### Test result

426 / 426 passing (added 10 new tests on top of previous 416 count, plus
renamed 1 existing).

### Followup

- uq4n.3: instrument the 5 Rust dispatch sites identified in uq4n.1's
  survey to actually fire these callbacks.
- uq4n.5: bd memory `invariant-rust-inspect-unsupported` needs an update
  once dispatch wiring lands.
