# Loop session notes (2026-05-07, 97th loop session)

## Task: angr-cz5f — Make state.max_history configurable (CLOSED)

Bead closed (commit b91fc4e81).

### Problem

`RustSimState.max_history` defaulted to 1000 but had no Python-facing
knob and — more importantly — was being silently bypassed during
exploration. The interpreter accumulates `detailed_history` for an
entire step, then `stepping.rs` calls `state.set_detailed_history(...)`
which previously just *assigned* the buffer. So long blocks could
exceed the cap without ever triggering FIFO eviction.

### Changes

- `native/angr/src/exploration/mod.rs`:
  - Added `max_history: usize` field on `RustExplorationManager`
    (default 1000), plus `set_max_history` / `get_max_history` PyO3
    methods. The setter retroactively applies to every state already
    in any stash.
  - `create_state` and `add_state` now call `state.set_max_history(...)`
    so newly created/added states inherit the manager's cap.
- `native/angr/src/state.rs`:
  - `set_detailed_history` now drains the oldest entries when the
    incoming buffer exceeds the cap (FIFO eviction). This was the
    actual blast radius of the cap not working.
  - Added two cargo unit tests
    (`test_set_detailed_history_honors_cap`,
    `test_set_detailed_history_unlimited`).
- `angr/exploration/rust_manager.py`:
  - Added `max_history=1000` kwarg to `RustExplorationManager.__init__`,
    only pushed to native side when non-default.
- `tests/engines/test_rust_exploration.py`:
  - Three new tests in `TestDetailedHistory`:
    - `test_max_history_get_set_default`
    - `test_max_history_caps_recorded_history` (uses fauxware,
      cap=5 — would have caught the set_detailed_history bypass)
    - `test_max_history_default_bounds_long_run`

### Test counts

- `cargo test --release --lib` → 517/517 (was 515, +2)
- `pytest tests/engines/test_rust_exploration.py` → 247/247 (was 244, +3)

## Memories saved

- `invariant-set-detailed-history-cap` — any cap on detailed_history
  must trim in `set_detailed_history` too, not just
  `add_to_history`/`add_history_entry`. Interpreter writeback bypasses
  the per-push checks.
- `venv-z3-headers-workaround` — `.venv/lib/python3.12/site-packages/z3/include/`
  was missing on this machine. Build with
  `Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build ...` and copy
  `target/release/librustylib.so` → `angr/rustylib.cpython-312-x86_64-linux-gnu.so`
  manually because `pip install -e .` is broken on this venv
  (resolvelib import error).

## Bead state

`angr-cz5f` CLOSED.

## Suggested next slices (P3, contained)

- `angr-xvnz` — Wire-or-remove dirty_registers. Profile first.
- `angr-3vrj` — StateMetadata dataclass to replace positional dicts/tuples (Python-only).
- `angr-4t2u` — NAMING_CONVENTIONS.md + minor renames (doc + small refactor).
- `angr-p7oa` — diff-state harness for callable find/avoid predicates.
- Memory module continuation (uncreated): extract `flush_pending_writes` and
  `merge` into separate files in `memory/`. Pure cuts.
