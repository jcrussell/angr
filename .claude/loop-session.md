## Session log: 2026-05-11 — angr-383x (silent-divergence: TRACK_CONSTRAINT_ACTIONS / TRACK_MEMORY_MAPPING, COMPLETE)

### Task
Close the silent-divergence gap for `TRACK_CONSTRAINT_ACTIONS` and
`TRACK_MEMORY_MAPPING` which were intentionally excluded from
`_REJECTED_OPTION_NAMES` (rust_manager.py:189-200) because they ship in the
default `symbolic` mode bundle (sim_options.py:391, 374) and would warn on
every `entry_state()`.

### Detour: angr-m2hf
First claimed angr-m2hf (CommonErrorReason refactor). Verified the audit
memo's deferral was correct — only `Unsupported(String)` actually appears in
3+ enums (CbExecutionError, ExecutionError, LiftError); `SymbolicArgument` /
`Other` only in 2 each. Re-deferred with updated rationale + memory.

### Strategy
Warn-once on first read of `state.history.actions` / `state.history.events`
via `state.history.__class__` reassignment:

- New `_RustOwnedSimStateHistory(SimStateHistory)` in `rust_state_export.py`
  overrides `.actions` and `.events` properties with a process-wide `_WARNED`
  class flag (one warning per Python process across all managers and states).
- `_install_rust_history_warning()` swaps `state.history.__class__` once per
  materialized Rust-owned state.
- Hooked in `_restore_plugins_to_state` so every state returned by
  `mgr.active` / `.found` / `.deadended` / etc. inherits the warning.
- Default-bundle users who never touch `.actions` / `.events` see nothing.
- Users who do touch them get one clear `UserWarning` pointing at the
  Python engine and `docs/RUST_SIMOPTION_COVERAGE.md`.

### Files changed
- `angr/exploration/rust_state_export.py` (+70): subclass + install function
  + hook call in `_restore_plugins_to_state`.
- `tests/engines/test_rust_exploration.py` (+125): 4 regression tests under
  `TestEdgeCases`:
  - `test_history_actions_read_warns_under_rust`
  - `test_history_events_read_warns_under_rust`
  - `test_history_actions_warn_once_process_wide`
  - `test_history_warning_does_not_fire_when_actions_not_read`
- `docs/RUST_SIMOPTION_COVERAGE.md` (+10): followup note on the warn-on-read
  mechanism for default-bundle (b) options.

### Verification
- New tests: 4/4 pass.
- Full suite: 382 passed, 3 failed. The 3 failures (dcas/pipe/dup2) are
  the same pre-existing set as sessions 215–217. No regressions from
  this change.

### Memories saved
- `strategy-default-bundle-silent-divergence` (new): the warn-on-read pattern
  for SimOptions that can't be warn-on-add'd because they ship in default
  bundles.
- `python-class-reassignment-for-history-plugin` (new): records that
  `state.history.__class__ = SubClass` works cleanly on `SimStateHistory`
  (no `__slots__`).
- `avoid-deferred-m2hf-error-trait` (updated): re-deferral rationale for
  CommonErrorReason refactor, with the actual variant-overlap counts.

### Closed beads
- `angr-383x`.

### Re-deferred beads
- `angr-m2hf` (back to deferred with updated note).

### Commit
07630755c  feat(rust-symex): warn-on-read for state.history.actions/.events — angr-383x

### Status
COMPLETE.
