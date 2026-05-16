## Session log: 2026-05-16 — Phase 5 (angr-o6ef): state-export sync cache

### Status: closed

### Investigation findings

Initial hypothesis (per bead): per-byte state-export eval cost dominates.
**Wrong.** Diagnostic counter showed only 1 Python-driven eval_upto call,
all 255 inner checks came from `sm.found[0].solver.eval_upto(u, 256)` in
solve.py:42 (the user-level post-explore code).

Real bottleneck (measured via /tmp/sym_write_time.py):
- `len(mgr.found)`: **1183ms** (first state-export pass, 2 states)
- `mgr.found[0]`:    **646ms** (second pass — re-runs all syncs)
- `eval_upto(u, 256)`: 196ms
- explore loop: 99ms

Each property access to `mgr.found` / `mgr.active` / etc. re-ran
`_get_stash_states`, which calls `_restore_plugins`,
`_inject_rust_stdout`, `_attach_rust_solver_fallback`,
`_sync_rust_memory_to_state`, `_sync_rust_registers_to_state`,
`_sync_rust_callstack_to_state`, … on every cached state — even when
the Rust state had not advanced.

### Implementation

Added a per-state `rust_fully_synced` sentinel on `state.scratch`,
set at the tail of each sync path in `_get_stash_states`. Subsequent
visits short-circuit to the cached SimState. Cleared by
`_invalidate_state_export_cache()`, called at the top of `step()`
and `explore()` (the only Rust-state-mutating entry points;
`_explore_with_addresses` is reached via `explore()`, so it inherits
the invalidation).

SSoT preserved: Rust remains authoritative. The cache only
short-circuits redundant Python mirror re-syncs between re-entries
into Rust.

### Validation

- `test_rust_exploration.py`: 404 passed (was 403; new test
  `test_state_export_cache_invalidated_on_step` exercises both the
  cache-hit identity invariant and the step/explore invalidation).
- `/tmp/sym_write_time.py`: second-access `mgr.found[0]` 646ms → **0.0ms**.
- `run_regression.py --rust-only --skip-bimodal --threshold 0.15`:
  pre-existing host-induced regressions present both with and without
  this change (compared via `git stash`); the cache does not introduce
  new ones. Sym-write bench wall-clock 1.74–1.83s (variance), unchanged
  vs the 1.70s post-mmdh baseline — bench is a single-pass workflow,
  the cache helps repeated property access not single reads.

### Diagnostic instrumentation

Loop-session note about `Z3_PY_EVAL_UPTO_*` counters in
`symbolic/context.rs` / `solver.rs::eval_upto`: confirmed absent
from the working tree (`git diff native/` empty). Nothing to revert.
