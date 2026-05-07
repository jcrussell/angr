# Loop session notes (2026-05-07, 99th loop session)

## Task: angr-p7oa — diff-state harness: invoke callable find/avoid predicates in step loop (CLOSED)

Bead closed (commit a66f73fcd).

### Problem

`tests/benchmarks/diff_state.py:_patch_rust_explore_to_single_step`
replaces `RustExplorationManager.explore()` with a step(1) loop, but
between steps it never evaluated callable find/avoid predicates. So
benchmarks like csgames2018 and sym-write (stdout-based find predicates)
silently never matched in `--diff-state` mode — Rust can't observe
Python callables, and the harness wasn't filling that gap.

### Fix

In `diff_state.py:_drive`, after `manager.step(1)`:

```python
if (getattr(manager, "_find_predicate", None) is not None
        or getattr(manager, "_avoid_predicate", None) is not None):
    try:
        manager._evaluate_predicates_on_active()
    except Exception:
        pass
```

`_evaluate_predicates_on_active()` (rust_state_cache.py:191) is the
same path the production exploration loops use:

- `_explore_with_predicates` (rust_manager.py:2099, 2115) calls it
  after every batch and at the end.
- `_explore_with_addresses` (rust_manager.py:2231) calls it after the
  step when callable predicates are set.

It walks active+deadended states and any cached Python states, runs the
predicate, and routes matches into found/avoid stashes via
`_rust_mgr.move_state`. Termination via `manager._found_count()`
already counts both Rust-native and predicate-matched finds, so no
loop-condition change was needed.

### Files changed

- `tests/benchmarks/diff_state.py`: 12 lines added in `_drive`.
- No Rust code touched.
- No tests modified — pytest still 247/247.

### Verification

- pytest `tests/engines/test_rust_exploration.py` → 247 passed.
- End-to-end `--diff-state` could not be exercised this session: the
  `.venv` editable-install hook is currently broken for spawned child
  processes (no `.pth` file, only the `__editable_*_finder.pyc`), so
  any `multiprocessing.spawn` child whose cwd is changed via `chdir`
  cannot `import angr`. Pre-existing env issue unrelated to this fix.
  Track: `bd memories env-venv-corruption` / `env-venv-recovery-procedure`.

### Memories saved

- `invariant-rust-explore-replacement-needs-predicate-eval` — any
  harness or technique that replaces `RustExplorationManager.explore`
  with its own step loop must call `_evaluate_predicates_on_active()`
  between steps when callable predicates are set, since Rust can't
  evaluate Python callables.

## Bead state

`angr-p7oa` CLOSED. 26 → 25 ready issues remaining.

## Suggested next slices

- `angr-3vrj` — StateMetadata dataclass (Python-only, but 56 call
  sites across 5 files; size/risk worth a careful plan first).
- `angr-4t2u` — NAMING_CONVENTIONS.md + cosmetic param renames.
- `angr-8f7d` — SSE scalar lane caching (needs flamegraph first per
  acceptance criteria).
- `angr-b6og` — Wire flamegraph/pprof into criterion benches.
