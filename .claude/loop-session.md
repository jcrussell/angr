## Session log: 2026-05-16 — angr-cf9h: promote DO_RET_EMULATION + CALLLESS to raise

### Status: closing (1 task done)

### Task

**angr-cf9h** — Final task of the option-set sweep started by angr-xghv
(TRACK_*_ACTIONS), angr-gmrc (CONCRETIZE), and angr-csmm
(CONSERVATIVE_WRITE_STRATEGY). Move `DO_RET_EMULATION` and `CALLLESS`
from `_REJECTED_OPTION_NAMES` (warn-once) to `_RAISE_OPTION_NAMES`
(raise NotImplementedError at construction).

- **DO_RET_EMULATION:** Python emits an emulated ret successor at every
  ret site; Rust does not emulate rets at all. Silent ignore changes the
  successor set — typically breaks Callable workflows.
- **CALLLESS:** Python replaces each call with unconstraining of the
  return register so Callable can short-circuit function bodies. Rust
  has no equivalent path and steps into the callee, breaking the
  Callable contract.

`TRUE_RET_EMULATION_GUARD` (paired with DO_RET_EMULATION) **stays** in
`_REJECTED_OPTION_NAMES`. Alone it's just a guard tweak with no effect;
the case where it matters (paired with DO_RET_EMULATION) now raises
before the guard is consulted. Asymmetric promotion follows the
CONSERVATIVE_WRITE_STRATEGY / CONSERVATIVE_READ_STRATEGY precedent.

### Implementation

- `angr/exploration/rust_manager.py`:
  - Dropped `DO_RET_EMULATION` and `CALLLESS` from
    `_REJECTED_OPTION_NAMES`; left `TRUE_RET_EMULATION_GUARD` there with
    an updated comment noting its now-orphan status.
  - Added both names to `_RAISE_OPTION_NAMES` with paragraph rationale
    blocks above the literal (matching the CONCRETIZE / CONSERVATIVE
    comment style).
- `tests/engines/test_rust_exploration.py`:
  - `test_rejected_options_emit_warning` swapped its option pair from
    `CALLLESS + DO_RET_EMULATION` (both now raise) to
    `UNINITIALIZED_ACCESS_AWARENESS + BEST_EFFORT_MEMORY_STORING` (still
    warn-only). Inline comment cites angr-cf9h.
  - `test_rejected_options_warn_once_per_manager` swapped sole option
    from `CALLLESS` to `UNINITIALIZED_ACCESS_AWARENESS`. Same reason.
  - Added `test_do_ret_emulation_option_raises_at_construction` and
    `test_callless_option_raises_at_construction`. Both assert
    `NotImplementedError`, option name in message, and "Python engine"
    pointer in message — matching the established pattern from xghv /
    gmrc / csmm.
- `docs/advanced-topics/rust_engine.rst`:
  - Added a new "Followup (angr-cf9h, 2026-05-16)" callout inside the
    implementation-note block, summarizing the promotion and the
    TRUE_RET_EMULATION_GUARD asymmetry.
  - Split the `DO_RET_EMULATION, TRUE_RET_EMULATION_GUARD` table row
    into two rows: the first promoted to (c), the second updated to (b)
    with the new "only meaningful when paired" rationale.
  - Promoted the `CALLLESS` row from (b) to (c).

### Tests

415 tests pass (was 413; +2 from the two new raise-at-construction
tests). Targeted run of the 13 option-related tests passed; full
suite passed in 49s.

### Memory updates

`invariant-rust-raise-option-names` updated: `_RAISE_OPTION_NAMES` now
holds 5 conceptual entries (TRACK_*_ACTIONS family, CONCRETIZE,
CONSERVATIVE_WRITE_STRATEGY, DO_RET_EMULATION, CALLLESS). Option-set
sweep is **complete** — no further sibling tasks remain in this
mini-epic (angr-n129 state-merging is structurally different).
