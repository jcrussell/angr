## Session log: 2026-05-16 — angr-gmrc + angr-csmm: promote two SimOptions to raise

### Status: closing (2 tasks done)

### Tasks

Both were near-mechanical follow-ups to angr-xghv: move a SimOption name
from `_REJECTED_OPTION_NAMES` to `_RAISE_OPTION_NAMES` so it raises
`NotImplementedError` at `RustExplorationManager` construction instead
of warning once.

1. **angr-gmrc — CONCRETIZE.** Python routes it through
   `SimSolver.BatchedConcretizationBacker` to eagerly concretize every
   fresh symbol; Rust has no equivalent hook. Silent ignore meant
   symbolic-driven analyses behaved as if the option were absent.
2. **angr-csmm — CONSERVATIVE_WRITE_STRATEGY.** Python's
   `SimSymbolicMemory.concretize_write_addr` honors it; Rust's
   `SymbolicMemory` always concretizes within strategy limits. Silent
   ignore defeated the user's intent to keep the analysis conservative.

### Implementation

- `angr/exploration/rust_manager.py`: moved both names out of
  `_REJECTED_OPTION_NAMES` and added them to `_RAISE_OPTION_NAMES`.
  Added a comment block per option explaining the Python-side mechanism
  and why silent acceptance would diverge.
- `tests/engines/test_rust_exploration.py`:
  - `test_rejected_options_emit_warning` switched from `CONCRETIZE +
    DO_RET_EMULATION` to `CALLLESS + DO_RET_EMULATION` (CONCRETIZE now
    raises).
  - Added `test_concretize_option_raises_at_construction` and
    `test_conservative_write_strategy_raises_at_construction`. Both
    assert `NotImplementedError`, that the option name is in the
    message, and that the message points users to the Python engine.
- `docs/advanced-topics/rust_engine.rst`:
  - Moved the `CONCRETIZE` row in the divergence-risk table from (b) to
    (c).
  - Split the `CONSERVATIVE_WRITE_STRATEGY / CONSERVATIVE_READ_STRATEGY`
    row into two: write becomes (c) raise, read stays (b) warn. The
    asymmetry tracks ticket scope (only the write variant was filed),
    not behavioral difference.
  - Added two new "Followup" notes inside the implementation-note
    callout, one per task.

### Tests

413 tests pass (was 411 going into the session; +1 per new test, +0
deletions). Both targeted runs and the full suite green.

### Notes for future sessions

- **One sibling task left**: `angr-cf9h` (DO_RET_EMULATION + CALLLESS).
  Same pattern. **Watch out** — `test_rejected_options_emit_warning`
  currently uses `CALLLESS + DO_RET_EMULATION` to verify warnings. If
  cf9h promotes both at once, replace those with another stable
  warn-only pair. Good candidates that remain in `_REJECTED_OPTION_NAMES`:
  `UNINITIALIZED_ACCESS_AWARENESS + BEST_EFFORT_MEMORY_STORING`.
- `angr-n129` (state-merging) is structurally different and not part of
  this option-set sweep.
- Memory `invariant-rust-raise-option-names` updated to reflect the
  current state (3 entries: TRACK_*_ACTIONS, CONCRETIZE,
  CONSERVATIVE_WRITE_STRATEGY) and to record the doc-row-splitting
  convention for asymmetric promotions.
