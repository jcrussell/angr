## Session log: 2026-05-16 — angr-uq4n.4 mem_write dispatch wiring

### Status: CLOSED — bd angr-uq4n.4 done

### Task

**angr-uq4n.4** — "inspect.4: mem_write event dispatch + reentrancy
guard test." Wire IRStmt::Store in the Rust VEX interpreter to fire
the mem_write inspect callback into Python via the marshalling layer
already built in uq4n.2.

### Implementation

1. **CallbackInterpreter::current_state_id (`i64`, default -1)** — new
   pub field on `interpreter_cb/mod.rs`. Forwarded by stepping.rs
   `run_interpreter_step` from `state.state_id() as i64` so dispatch
   sites know which state owns the firing event. Inherited on fork().
2. **IRStmt::Store dispatch** in `interpreter_cb/statements.rs`:
   - Destructure `endness` from the `IRStmt::Store` variant.
   - After a successful store (both `try_rust_memory_store` true-path
     and `fallback_to_python_store` path), call new private helper
     `dispatch_mem_write_inspect`.
   - Helper gates on `callbacks.inspect_event_enabled(1)` (bit 1 =
     `InspectEvent::MemWrite`); concrete-address only for MVP (symbolic
     dispatch is a follow-up); converts data_val to claripy AST via
     `rustbv_to_claripy`; fires `call_inspect_mem_write(state_id,
     "after", addr_u64, data_size, Some(&value_ast), endness_str)`.
   - Errors are swallowed — the Python dispatcher logs BP failures
     itself, and the engine must not halt because of a user BP error.
3. **Bitmask sync fix** (this was the blocker for the first test attempt):
   `PythonCallbacks` derives `Clone`, and `RustExplorationManager::
   set_callbacks` takes by value — PyO3 clones the struct, so the Rust
   manager holds a different copy than the Python `mgr._callbacks`.
   `inspect_enabled: u8` was a primitive that doesn't share updates;
   bumped to `Arc<AtomicU8>` so both copies see the same byte.
   `py_set_inspect_enabled` now takes `&self` (atomic store), getter +
   `inspect_event_enabled` use atomic load (Relaxed ordering).
4. **3 new tests** in `tests/engines/test_rust_exploration.py`
   (`TestRustInspectMemWriteDispatch` class):
   - `test_mem_write_fires_during_exploration`: register mem_write BP
     on fauxware entry_state, `mgr.run(max_steps=5)`, expect at least
     one event with valid endness ("Iend_LE"/"Iend_BE") and length 1-16.
   - `test_mem_write_skipped_when_no_bp`: bitmask stays 0, run does
     not raise (zero-overhead common case).
   - `test_mem_write_reentrancy_with_proxy_access`: BP reads `s.addr`
     (which fails because state is held by interp during step) inside
     the action, swallows; exploration must continue without deadlock.

### Files modified

- `native/angr/src/callbacks.rs` — `inspect_enabled` → `Arc<AtomicU8>`,
  setter/getter/helper use atomic ops, added
  `get_inspect_enabled_for_debug` for tracing.
- `native/angr/src/exploration/stepping.rs` — set
  `interp.current_state_id = state.state_id() as i64` in
  `run_interpreter_step`.
- `native/angr/src/interpreter_cb/mod.rs` — add `current_state_id: i64`
  field on `CallbackInterpreter` (default -1), wired in both
  constructors (new and fork).
- `native/angr/src/interpreter_cb/statements.rs` — destructure `endness`
  from `IRStmt::Store`, fire dispatch helper after store; new
  `dispatch_mem_write_inspect` method.
- `tests/engines/test_rust_exploration.py` — `TestRustInspectMemWriteDispatch`
  class (3 tests).

### Test result

429 / 429 passing (was 426; added 3 mem_write dispatch tests).

### Followup

- Same instrumentation for mem_read (IRExpr::Load fast paths — survey
  in `inspect-mem-dispatch-surfaces` memory enumerates 8 sites) needs
  uq4n.3 to be unblocked; that task was auto-deferred.
- IRStmt::StoreG path (guarded store) is also a mem_write site — not
  wired in this PR. Symbolic-address dispatch deferred.
- state.rs wrappers (memory_store / memory_store_symbolic / multi) —
  the SimProcedure mem_write surface — needs a separate plumbing
  approach since state.rs doesn't see PythonCallbacks today.
