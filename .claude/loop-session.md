## Session log: 2026-05-16 — angr-n129 closed (EFFICIENT_STATE_MERGING → raise)

### Status: closed

### Task

**angr-n129** — "Raise NotImplementedError for state-merging
SimOptions in Rust engine."

Bead description called state merging "unimplemented; option silently
ignored". Reality is more nuanced: Rust DOES implement `merge_states`
at the low level (`native/angr/src/exploration/state_lifecycle.rs:85`,
calling `RustSimState::merge` at `native/angr/src/state.rs:1724`), and
`RustExplorationManager.merge()` exports states to Python and uses
Python `state.merge()` per group (`rust_manager.py:3856`). What IS
silently ignored is the `EFFICIENT_STATE_MERGING` SimOption, which the
Python engine consults from `SimStateHistory.set_strongref_state`
(`state_plugins/history.py:131`) to retain ancestor refs for plugin
merging. Rust never drives that path.

### Implementation

- `angr/exploration/rust_manager.py`:
  - Added `EFFICIENT_STATE_MERGING` to `_RAISE_OPTION_NAMES`.
  - Added a paragraph rationale block above the literal (matches the
    CONCRETIZE / DO_RET_EMULATION comment style), explicitly noting
    why the paired `SIMPLIFY_MERGED_CONSTRAINTS` is NOT being promoted
    (default-bundle member).
- `tests/engines/test_rust_exploration.py`:
  - Added `test_efficient_state_merging_option_raises_at_construction`
    in `TestEdgeCases`, mirroring the cf9h / gmrc test pattern (asserts
    `NotImplementedError`, option name in message, "Python engine"
    pointer in message).
- `docs/advanced-topics/rust_engine.rst`:
  - Removed the `EFFICIENT_STATE_MERGING` row from "Ignored — no-op"
    (was "Rust does not yet support state merge").
  - Added an `EFFICIENT_STATE_MERGING` row to the (c)-raise category
    in "Ignored — divergence-risk", explicitly citing the Veritesting
    auto-add at `step_state` and explaining the
    `SIMPLIFY_MERGED_CONSTRAINTS` asymmetry.
  - Added a "Followup (angr-n129, 2026-05-16)" note in the
    implementation-note callout.

### Why not SIMPLIFY_MERGED_CONSTRAINTS

`SIMPLIFY_MERGED_CONSTRAINTS` is a member of the `simplification` set
inside `common_options` inside the default `symbolic` /
`symbolic_approximating` mode bundles (`sim_options.py:370-379, 391-392`).
Adding it to `_RAISE_OPTION_NAMES` would break every `entry_state()`.
The option is only read inside Python `SimStateHistory.merge()`, which
IS reached by `RustExplorationManager.merge()` since that method
exports states to Python and calls Python `state.merge()`. So the
option is effectively honored on the only path that touches it.

The merge interface methods themselves (`merge_states`, `manager.merge`)
were left alone — both have working implementations with passing tests
(`test_merge_states_*` at `test_rust_exploration.py:7046+`).

### Tests

416 tests pass (was 415; +1 from the new raise test). Full suite ran
in 49s with no regressions.

### Memory updates

- `invariant-rust-raise-option-names` updated to include
  `EFFICIENT_STATE_MERGING` as the 6th member. Critical note added
  about `SIMPLIFY_MERGED_CONSTRAINTS` being a default-bundle member
  that must never be promoted. Listed `EFFICIENT_STATE_MERGING` vs
  `SIMPLIFY_MERGED_CONSTRAINTS` as a new asymmetric-promotion
  precedent.

### Commit

`8d4617562` — feat(rust-symex): raise NotImplementedError for
EFFICIENT_STATE_MERGING (angr-n129)
