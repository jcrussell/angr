## Session log: 2026-05-17 — angr-dcva Implement ENABLE_NX in Rust engine

### Status: CLOSED — commit b99b3810a. 440/440 pass (+2 net tests).

### Task

**angr-dcva (P2, feature)** — Wire ENABLE_NX SimOption to Rust engine.
Previously, Rust collapsed STRICT_PAGE_ACCESS and ENABLE_NX into a single
`enforce_permissions` flag that gated both R/W AND X checks. Python angr
separates them: X check fires only when BOTH STRICT_PAGE_ACCESS AND ENABLE_NX
are set (`angr/engines/vex/heavy/heavy.py:115-124`).

### What landed

Rust side:
- `SymbolicMemory::enforce_nx` field + `set_enforce_nx`/`enforce_nx`
  accessors (`native/angr/src/memory/mod.rs`); propagated through `fork()`.
- `check_executable` now requires BOTH `enforce_permissions` AND `enforce_nx`
  to fire (matches Python heavy VEX engine semantics).
- `RustSimState` + PyO3 wrappers (`native/angr/src/state.rs:1344-1357,
  2128-2141`) expose the flag to Python.
- `RustExplorationManager::state_enforce_nx(state_id)` accessor
  (`native/angr/src/exploration/mod.rs:1748-1752` + `state_api.rs:483-485`).

Python side:
- `rust_manager.py::_add_rust_state` sets `enforce_nx=True` when
  `o.ENABLE_NX in state.options` (alongside existing STRICT_PAGE_ACCESS).
- `_apply_state_metadata`'s allow-list extended:
  `(LAZY_SOLVES, STRICT_PAGE_ACCESS, ENABLE_NX)` — add+discard mirroring
  prevents cache reuse from leaking or dropping the flag.

Tests (+2 net):
- Updated `test_strict_page_access_blocks_nx_block_fetch` to also call
  `set_enforce_nx(True)` (since the single-flag wiring is gone).
- Added `test_strict_page_access_alone_does_not_block_nx_fetch` —
  verifies Python's gating semantics (STRICT alone must not block X).
- Added `test_enable_nx_propagates_to_rust` — verifies the option-to-flag
  wiring through the high-level `RustExplorationManager`.

Docs:
- `docs/advanced-topics/rust_engine.rst`: ENABLE_NX moved from "implement"
  to the honored matrix with note about the STRICT_PAGE_ACCESS gate.

### Memories saved

- `invariant-enable-nx-gated-on-strict` — both flags required for X check
  to fire. Future memory-permission work must respect this.
- `invariant-rust-honored-simoptions` (updated) — now 6 honored options
  (added ENABLE_NX). Lists access pattern for both flags from Python.

### Files modified

- native/angr/src/memory/mod.rs (+27/-9)
- native/angr/src/state.rs (+27)
- native/angr/src/exploration/state_api.rs (+4)
- native/angr/src/exploration/mod.rs (+6)
- angr/exploration/rust_manager.py (+18/-13)
- tests/engines/test_rust_exploration.py (+66/-3)
- docs/advanced-topics/rust_engine.rst (+12/-4)

### Followup work for next session

Sister tasks still open (P2) that share the same wiring pattern:
- angr-yl5n — NO_IP_CONCRETIZATION
- angr-ph9z — KEEP_IP_SYMBOLIC

These are different in scope: they require routing through symbolic IP
in the interpreter loop, not just a memory flag. Not as straightforward
as ENABLE_NX.
