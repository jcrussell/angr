## Session log: 2026-05-09, 174th loop session

### Task: angr-4j5u.5.3 (closed) — Extract pending callback API into pending_api.rs

Third of the 4-way split of angr-4j5u.5. Continues mod.rs<800-line decomposition.

Sibling status:
- 4j5u.5.1 — Extract run() into run_loop.rs ✓ (172nd session)
- 4j5u.5.2 — Extract resume_after_* into resume.rs ✓ (173rd session)
- 4j5u.5.3 — Extract pending callback API into pending_api.rs ✓ THIS SESSION
- 4j5u.5.4 — Extract state inspection API into state_api.rs (open, ready)

### What was already done in a prior session

Previous session left native/angr/src/exploration/pending_api.rs (499 lines)
already created with `pub(crate) _method_name` bodies for all 35 pyclass-exposed
pending callback methods, but had NOT wired up mod.rs forwarders or added the
`mod pending_api;` declaration. The task came in as in_progress with this
half-done state.

### What changed this session

**native/angr/src/exploration/mod.rs (2806 → 2532 lines, -274)**
- Added `mod pending_api;` next to other children.
- Replaced 35 method bodies with 1-line forwarders to `_method_name` in
  pending_api.rs. Kept #[pymethods] declaration and #[pyo3(signature = ...)]
  attributes on the wrappers.

Methods covered (35 total):
- PC / memory mapping: set_pending_state_pc, pending_state_map_memory,
  active_states_map_memory, pending_memory_map_data
- Branch condition / register / history / jumpkind:
  get_pending_branch_condition, get_pending_register{,_ast},
  get_pending_history, get_pending_jumpkind,
  set_pending_register{,_symbolic{,_ast}}
- Symbolic memory import: import_symbolic_to_state, import_symbolic_memory
- Pending memory get/set + dirty pages: get/set_pending_memory,
  get_pending_dirty_pages, clear_pending_dirty_tracking,
  get_pending_mapped_pages, pending_memory_load{,_page}, pending_memory_store
- Constraints / handles / snapshots / ancestry:
  export_pending_constraints, get_active_handle_ids, export_pending_state,
  get_pending_root_state_id, get_pending_ancestry, export_callback_bundle
- Solver fork/borrow + constraint sync: fork_pending_solver,
  borrow_pending_solver, add_constraints_to_pending, pending_constraint_count
- Skip-hook stack: set/clear_skip_hook_addr, clear_skip_hook_for_addr

### Build/test

- Venv pip is broken (avoid-broken-venv-pip-fallback-cargo-build):
  `cargo build --manifest-path native/angr/Cargo.toml --release` then
  `cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`.
- `cargo check --release` clean (only pre-existing warnings).
- 342/342 tests passing in 19.14s.

### Memories saved

- `invariant-pending-api-extension-impl` — points readers at pending_api.rs
  for pending-callback API semantics changes; reaffirms thin-wrapper pattern.

### Acceptance criteria delta

Acceptance said "mod.rs shrinks by ~960 lines" but actual delta is 274 lines.
The estimate was based on raw body LOC, not net delta — the wrappers retain
doc comments and #[pyo3(signature)] attributes, which adds back ~200-300
lines. pending_api.rs at 499 lines is well under the 1100-line target.

### Next session

Pick up `angr-4j5u.5.4` (Extract state inspection API into state_api.rs).
Same pattern. Target methods are roughly 1060 lines (~lines 1561-2622 of
mod.rs after this session — verify with grep before editing) covering
get_state_*/set_state_*/eval_in_state/state_satisfiable/export_state.
mod.rs target after .5.4: 2532 → ~1500 lines.

NOTE: 5.4 is a larger task because the bodies are not pre-extracted.
Plan to spend the full session on it.
