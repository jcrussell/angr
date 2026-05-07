# Loop session notes (2026-05-07, 100th loop session)

## Task: angr-b1qq — Wire memory_{store,load}_symbolic_full callbacks (CLOSED)

Bead closed (commit 30702fa5a). First split of the deferred angr-pufm P1 work.

### Background

`bd memories invariant-symbolic-full-callbacks-unset` (now updated to STALE):
- `memory_store_symbolic_full` and `memory_load_symbolic_full` had Rust setters
  (callbacks.rs:560/566) and consumers at statements.rs:368/444/710/1174 and
  expressions.rs:149, but were NEVER wired from Python.
- Result: any TooLarge ConcretizationResult on a symbolic-address load/store
  errored out as `CbExecutionError::Unsupported`.

### Implementation

Two new callback methods on `RustExplorationManager`
(`angr/exploration/rust_manager.py`, next to the existing
`_cb_memory_store_symbolic_value`):

- `_cb_memory_store_symbolic_full(addr_ast, data_ast)` — calls
  `state.memory.store(addr_ast, data_ast, endness=..., inspect=False,
  disable_actions=True)`.
- `_cb_memory_load_symbolic_full(addr_ast, size)` — calls
  `state.memory.load(addr_ast, size, endness=..., inspect=False,
  disable_actions=True)` and returns the AST.

Both swallow `SimError`/`ClaripyError` per the angr-8e81 / angr-2f7o convention.
The load-side fallback returns `claripy.BVS(f"sym_load_full_fail_{size}",
size*8)` so Rust's expressions.rs:149 path can wrap it into a `sym_pyref_*`
placeholder.

`_init_callbacks` registers both via `set_memory_store_symbolic_full` /
`set_memory_load_symbolic_full` (with `hasattr` guards for older .so builds).

### Files changed

- `angr/exploration/rust_manager.py`: 2 new callback methods + 4 lines in
  `_init_callbacks`.
- `tests/engines/test_rust_exploration.py`: 7 new regression tests in
  `TestErrorRecovery` covering wiring, round-trip (at 0x4000 — 0x1000 is
  already mapped by `load_shellcode`), Sim-swallow, and non-Sim-propagate.

### Verification

- pytest `tests/engines/test_rust_exploration.py` → 254 passed (was 247).
- End-to-end benchmarks could not run (still the same `.venv` editable-install
  issue from session 99 — `__editable_*_finder.pyc` exists but no `.pth`,
  so spawned subprocesses can't `import angr`).

### Memories updated

- `invariant-symbolic-full-callbacks-unset` → STALE (now points to commit
  30702fa5a and notes the callbacks are wired).
- `invariant-symbolic-full-callbacks-wired` → NEW. Captures the new wiring
  contract: bound-method presence is the wiring check (has_* aren't exposed
  to Python), the load-side BVS fallback width matches `size*8`, and tests
  must use 0x4000+ to avoid colliding with `load_shellcode`'s 0x1000 mapping.

## Bead state

`angr-b1qq` CLOSED. `angr-pufm` (P1) still open — its remaining piece is the
lazy guarded-entries optimization (write records an address constraint instead
of enumerating).

## Suggested next slices

- `angr-pufm` lazy guarded-entries piece (now that the fallback callbacks are
  wired, this can be tackled independently — but it interacts with
  `lazy-memory-load-overlay-fails` and `symwrite-eager-vs-lazy-memory`
  memories, so plan carefully first).
- `angr-3zs6` — FallbackStrategy enum walk (cosmetic, well-scoped).
- `angr-b6og` — Wire flamegraph/pprof into criterion benches.
