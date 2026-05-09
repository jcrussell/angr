## Session log: 2026-05-09, 179th loop session

### Task: angr-lwpt — max_history retroactive cap + FIFO + fork inheritance — CLOSED

Discovered `set_max_history` was non-converging when shrinking the cap
below the current buffer length: `add_to_history` and `add_history_entry`
only remove ONE entry per push when over cap, so a state with 100
entries pushed at cap=5 would oscillate between 100-101 forever. The
manager-level docstring at `exploration/mod.rs:664-668` promised "applied
to every state already in any stash" — but state.set_max_history was
just a field assignment.

### Fix

In `state.rs:1799` `set_max_history` now FIFO-drains both `history` and
`detailed_history` down to the new cap immediately. `max=0` means
unlimited and explicitly does NOT trim.

### Files modified

- native/angr/src/state.rs (+81 lines)
  * `set_max_history` now trims retroactively
  * Added 4 PyRustSimState helpers: `get_max_history`, `detailed_history`,
    `add_history`, `add_detailed_history` (testing surface)
  * Added 2 Rust unit tests: `test_set_max_history_trims_retroactively`,
    `test_set_max_history_zero_no_trim`
- tests/engines/test_rust_exploration.py (+85 lines)
  * `test_max_history_retroactive_trim`
  * `test_max_history_fifo_eviction_via_add`
  * `test_max_history_inherited_through_fork`

### Verification

- Rust unit tests: `state::tests::test_set_max_history_*` 2/2 pass
- Python: `pytest tests/engines/test_rust_exploration.py` 345/345 pass
- Commit: `27441a00e`

### Memories saved

- `set-max-history-non-converging-bug` — root cause + fix.
- `invariant-fork-needs-python-init` — RustSimState::fork() panics in
  pure cargo test (PyO3 0.27 needs Py interpreter initialized); known
  pre-existing failures (test_state_fork etc.). Write fork tests in
  Python (test_rust_exploration.py) so pytest initializes the harness.

### Next session

`bd ready` shows P3 follow-ups: angr-w6ry callstack proxy fork tests,
angr-orc9 ARM/AArch64/MIPS proc round-trip, angr-f58x DCAS counter
test. Larger P2 items remain (angr-wqao split rust_manager.py,
angr-4j5u decompose god struct) but auto-defer policy still applies.
