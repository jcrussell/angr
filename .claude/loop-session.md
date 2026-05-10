## Session log: 2026-05-10 — angr-r4r7 (audit unbounded Python shadow structures, COMPLETE)

### Task
Audit Python-side bookkeeping dicts/sets in `rust_manager.py` and friends for
unbounded growth. For each, decide: (a) bounded by lifetime, (b) needs explicit
clear, (c) needs LRU cap, (d) shadows Rust and should be deleted.

### Audit result

| Structure | Verdict |
|-----------|---------|
| `_state_cache` (cap=8, LRU) | (a) bounded |
| `_ast_handle_cache` (cap=10000) | (a) bounded |
| `_z3_ptr_cache` (cap=1024) | (a) bounded |
| `_py_state_options`, `_py_state_globals` | (a) bounded — pruned in `_cleanup_state_cache` |
| `_predicate_eval_cache` | (a) bounded — popped in `_cleanup_state_refs` |
| `_state_roots` | **(b) FIXED — now pruned in `_cleanup_state_cache`** |
| `_predicate_matched_ids` | **(b) FIXED — now `intersection_update(any_stash)`** |
| `_registered_hooks`, `_exit_continuation_addrs` | (a) bounded by binary |
| `_predicate_found` | (a) bounded by found stash |
| `_pending_procedure_data` | (a) bounded by binary's continuation set |
| `_warned_rejected_options` | (a) bounded by sim_options enum |
| `_stdin_content` | (a) bounded by stdin size |
| `_procedure_times` | (a) bounded by SimProcedure name set |
| `_active_techniques` | (a) bounded by user input |

### Fix
`_cleanup_state_cache` (rust_manager.py:2980) now reads `avoid` + `deadended`
stashes in addition to `active` + `found`, builds `any_stash = active | found |
avoid | deadended`, and prunes `_state_roots` keys and `_predicate_matched_ids`
members not present in `any_stash`. Rust state IDs are monotonically allocated
and never reused so dropped entries can never become relevant.

### Files changed
- `angr/exploration/rust_manager.py` (+43): broader `any_stash` set, two new
  prune blocks, doc on `_state_roots` declaration.
- `tests/engines/test_rust_exploration.py` (+47): two regression tests under
  `TestStateMetadataStorage`:
  - `test_cleanup_state_cache_prunes_state_roots`
  - `test_cleanup_state_cache_prunes_predicate_matched_ids`

### Verification
- `pytest -k cleanup_state_cache`: 5 passed (3 existing + 2 new).
- Full test suite: 378 passed (was 376), 3 pre-existing failures unchanged
  (dcas/pipe/dup2 — same set as sessions 215 & 216).
- `run_single.py fauxware --engine rust`: fauxware still runs correctly.

### Memories saved
- `invariant-state-id-never-reused` (new): the load-bearing invariant for any
  future shadow-structure prune.
- `audit-shadow-structures-2026-05-10` (new): full audit table for future
  reference, so this work doesn't have to be redone.

### Closed beads
- `angr-r4r7`.

### Commit
7dd834b78  fix(rust-symex): prune dead _state_roots / _predicate_matched_ids — angr-r4r7

### Status
COMPLETE.
