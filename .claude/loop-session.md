## Session log: 2026-05-16 — angr-xghv: raise NotImplementedError for TRACK_*_ACTIONS

### Status: closing

### Task

Make TRACK_*_ACTIONS family (TRACK_MEMORY_ACTIONS, TRACK_REGISTER_ACTIONS,
TRACK_TMP_ACTIONS, TRACK_JMP_ACTIONS, TRACK_OP_ACTIONS, TRACK_ACTION_HISTORY)
raise NotImplementedError at RustExplorationManager construction instead of
silent warn-once. Rust never emits SimAction records — loud failure beats
hours of debugging an empty `state.history.actions`.

### Implementation

- Added `_RAISE_OPTION_NAMES` frozenset alongside `_REJECTED_OPTION_NAMES`
  in angr/exploration/rust_manager.py. The 6 TRACK_*_ACTIONS names moved
  from `_REJECTED_OPTION_NAMES` (warn) into `_RAISE_OPTION_NAMES` (raise).
- New `_check_raise_options()` method raises `NotImplementedError` listing
  all offending option names and pointing users at the Python engine.
- Hooked in at the two existing call sites: `__init__` (initial states) and
  `_add_rust_state` (late additions). Check runs *before* the warn-once
  pass so we fail clean without emitting an unrelated warning first.
- Updated docs/advanced-topics/rust_engine.rst: new (c) classification for
  the raise set; TRACK_CONSTRAINT_ACTIONS split out as (b) (default-mode
  bundle, warn-on-read via `_RustOwnedSimStateHistory`).

### Tests

- Updated existing `test_rejected_options_emit_warning` to use CONCRETIZE
  + DO_RET_EMULATION (TRACK_MEMORY_ACTIONS now raises, not warns).
- Added `test_action_tracking_options_raise_at_construction` parametrized
  across all 6 options.
- Added `test_action_tracking_options_raise_lists_all` to verify multi-
  option case lists all names in the error message.

411 tests pass (was 404 before; +7 from the new parametrized + multi tests).

### Notes for future sessions

- `_RAISE_OPTION_NAMES` is the natural attach point for the sibling tasks
  (angr-gmrc CONCRETIZE, angr-csmm CONSERVATIVE_WRITE_STRATEGY,
  angr-cf9h DO_RET_EMULATION+CALLLESS). They currently warn via
  `_REJECTED_OPTION_NAMES`; each could be promoted to raise by moving the
  name. Default-mode bundle members (TRACK_CONSTRAINT_ACTIONS,
  TRACK_MEMORY_MAPPING) must NOT be promoted (would break every
  `entry_state()`).
- angr-n129 (state-merging) is structurally different: needs a check at
  `state.merge()` call site, not in option set. Distinct mechanism.
