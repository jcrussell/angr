# Loop session notes (2026-05-08, 140th loop session)

## Task: angr-p8o3 — Move _state_metadata storage into Rust per-state struct (closed)

### Status: complete; closed

### Summary
Migrated three per-state Python dicts (`symbolic_pages`,
`hook_symbolic_memory`, `addr_to_ast`) from a Python-side
`Dict[int, StateMetadata]` map into `RustSimState` itself. Storage and
state lifetime are now unified — when Rust drops a state, its metadata
maps drop with it. This was the prerequisite for angr-qm7w (lazy
`_state_cache` populate/evict).

### Changes
- **state.rs** — added three `HashMap<u64, Py<PyAny>>` fields to
  `RustSimState`. Initialized in all 3 constructors. Added
  `clone_py_metadata()` helper that uses `Python::attach + clone_ref`
  to bump PyObject ref counts under GIL on every fork. Wired the helper
  into all 5 fork sites (`fork`, `fork_true`, `fork_false`,
  `fork_from_snapshot`, `merge`).
- **exploration/mod.rs** — added 7 PyO3 methods on the manager:
  `get_state_{symbolic_pages,hook_symbolic_memory,addr_to_ast}` (return
  empty dict for missing state — preserves the old `dict.get()` falsy
  semantics), `set_state_{symbolic_pages,hook_symbolic_memory,addr_to_ast}`
  (whole-dict and single-entry setters), and `clear_state_metadata`.
- **rust_manager.py** — deleted the `StateMetadata` import + the
  `_state_metadata: Dict[int, StateMetadata]` dict. Updated 3 callback
  sites: `_cb_memory_load` (~746), `_cb_sync_constraints`
  addr_concretize/mem reconstructions (~875/895), and the two
  `_extract_symbolic_pages` writes in `_add_rust_state` (~1870/1919) —
  the addr-map iterations collapsed to `addr_map.get(addr)` because
  they were already exact-match lookups.
- **rust_state_cache.py** — deleted `_state_md()` helper and the
  `StateMetadata` import. `_register_handle` now calls
  `set_state_addr_to_ast`. `_cleanup_symbolic_pages_cache` is now a
  no-op shim (Rust state lifetime handles eviction implicitly).
  `_cleanup_state_cache`/`_cleanup_state_refs` call
  `clear_state_metadata`.
- **rust_state_sync.py** — `_restore_symbolic_pages` and
  `_restore_hook_symbolic_memory` now call
  `get_state_{symbolic_pages,hook_symbolic_memory}` for each of
  direct/root/ancestor candidates. The ancestry walk pattern preserved.
- **rust_state_export.py** — 4 read sites converted to
  `get_state_addr_to_ast` / `get_state_hook_symbolic_memory`.
- **_state_metadata.py** — deleted.
- **test_rust_exploration.py** — `TestStateMetadataStorage` (9 new
  tests): round-trip for all three maps, replace-whole-dict, unknown
  state handling, setter raises, clear drops all maps, no-op for
  unknown clear, safe stale clear, no-aliasing-after-fork.

### Verification
- `cargo build --release` clean.
- `pytest tests/engines/test_rust_exploration.py` — **285/285 passing**
  (was 276 before; +9 new metadata-storage tests).
- fauxware benchmark: 0.31s (no regression).

### Memories saved
- `invariant-rust-py-metadata-storage` — per-state Python AST refs now
  live in `RustSimState`; access goes through the manager's
  `get_state_*` / `set_state_*` PyO3 methods.
- `pattern-pyobject-fork-clone-via-attach` — fork sites must call
  `Python::attach + clone_ref` per entry; PyO3 `Clone` on `Py<T>`
  requires the GIL.

### .venv side trip
Same vendored-pip strip as last session — fixed with the recipe in
`fragile-venv-recovery` memory: force-reinstall pip + setuptools<81 +
semantic_version. After that, also had to `cargo build --release` and
`cp` the `.so` because `pip install -e .` did not rebuild the Rust
extension on its own (cached editable install).

### Files changed
- native/angr/src/state.rs (+114 −0)
- native/angr/src/exploration/mod.rs (+122 −0)
- angr/exploration/rust_manager.py (~30 modified)
- angr/exploration/rust_state_cache.py (~25 modified)
- angr/exploration/rust_state_export.py (~10 modified)
- angr/exploration/rust_state_sync.py (~30 modified)
- angr/exploration/_state_metadata.py (deleted)
- tests/engines/test_rust_exploration.py (+155)

### Unblocks
- angr-qm7w (lazy `_state_cache` populate/evict) — was blocked on this
  task for clean ownership semantics on the metadata side.
