## Session log: 2026-05-08, 166th loop session

### Task: angr-t3l3 (closed) — RustStateProxy missing state.options/globals/heap

Read-only proxy gaps: state.options returned `set()`, state.globals returned
`{}`, state.heap was completely absent. User code in find/avoid predicates
and exploration techniques that read or mutated these plugins observed
silent wrong behavior.

### What changed

**Storage (angr/exploration/rust_manager.py)**
- Added `_py_state_options: Dict[int, set]` and `_py_state_globals: Dict[int, dict]`
  to `RustExplorationManager.__init__`.
- Seeded in `_add_rust_state` after `actual_state_id` resolves. Pull enabled
  options via `{name for name, v in state.options._options.items() if v is True}`
  — `set(state.options)` raises SimStateOptionsError (custom dict-backed type
  iterates non-set-like).
- New accessors `get_state_options_py(sid)` / `get_state_globals_py(sid)` with
  parent-walk fallback via `_rust_mgr.get_state_root(sid)` so children forked
  inside Rust inherit a shallow copy from the root on first access.
- `_cleanup_state_cache` extended to drop entries for sids no longer in any
  live stash (root entries pinned).

**Proxy (angr/exploration/rust_state_proxy.py)**
- New `RustHeapProxy` exposing `mmap_base` (read/write via existing
  `get_state_mmap_base` / `set_state_mmap_base` PyO3 accessors), `allocations`
  and `freed` (via `get_state_heap_metadata`).
- `RustStateProxy.__init__` now takes `python_mgr=None`; threaded through
  `copy()` and `RustSimulationManagerProxy._wrap_state`.
- `state.options` / `state.globals` delegate to `python_mgr.get_state_*_py()`
  when wired; fall back to empty stand-ins for low-level unit-test paths
  (e.g., the existing `RustStateProxy(mgr, sid)` pattern in TestSolverProxyTimeout).
- `state.heap` lazily constructs `RustHeapProxy`.

**Construction sites updated to pass python_mgr** (all 5 user-facing call sites):
- `rust_manager.py:2944` (filter fallback) — `python_mgr=self`
- `rust_manager.py:2778` (`RustSimulationManagerProxy` factory) — `python_mgr=self`
- `rust_state_cache.py:264` (predicate eval) — `python_mgr=self`
- `rust_callback_dispatch.py:1340, 1390` (find/avoid predicate callbacks) — `python_mgr=self`
- `rust_techniques.py:243` (technique filter) — `python_mgr=mgr`

### Why the first test run failed and what it taught

`set(state.options)` looks like the obvious extraction but raises on
SimStateOptions because that class lazily validates every iterated key —
including numeric internal keys like `'0'`. The seed code's `except` block
swallowed it and `_py_state_options` stayed empty. DEBUG log surfaced
`SimStateOptionsError: The state option '0' does not exist.` Saved as memory
`avoid-set-of-simstateoptions`.

### Files changed

- `angr/exploration/rust_manager.py` (+85)
- `angr/exploration/rust_state_proxy.py` (+82)
- `angr/exploration/rust_state_cache.py` (+1)
- `angr/exploration/rust_callback_dispatch.py` (+2)
- `angr/exploration/rust_techniques.py` (+1)
- `tests/engines/test_rust_exploration.py` (+116, new `TestStatePluginsProxy`)

### Tests

339/339 passing on `tests/engines/test_rust_exploration.py` (332 baseline + 7 new
under `TestStatePluginsProxy`):
- `test_options_seeded_from_source_state`
- `test_options_writes_persist`
- `test_globals_seeded_and_mutable`
- `test_options_inherited_by_forked_states`
- `test_heap_mmap_base_exposed`
- `test_heap_allocations_and_freed_lists`
- `test_options_globals_fallback_without_python_mgr`

### Commit

`832e81900` on rust-symex.

### Memories saved

- `avoid-set-of-simstateoptions` — extract via `_options.items()`, not `set()`
- `invariant-rust-state-proxy-plugins` — what's exposed, what's still
  unsupported (state.libc, state.scratch tracked in angr-jco0)
