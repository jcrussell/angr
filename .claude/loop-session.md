## Session log: 2026-05-17 — angr-dmqr closed (strcpy_find integration test fixed)

### Status: CLOSED

### Task

**angr-dmqr (P3, bug)** — Fix strcpy_find integration test: CFG jumptable
resolver creates RustExplorationManager with DO_RET_EMULATION.

### Resolution

`tests/engines/test_rust_integration.py::patched_simulation_manager` now
mirrors the call-stack guard already present in
`tests/benchmarks/run_single.py:136-158`. When invoked from a frame under
`/angr/analyses/` or `/angr/exploration_techniques/`, it falls through to
the original `simulation_manager`. This stops CFG/jumptable resolver paths
from constructing a `RustExplorationManager` over an internal state with
`DO_RET_EMULATION` (which `_check_raise_options` rejects).

### Files modified

- `tests/engines/test_rust_integration.py` (+9 lines guard)

### Verification

- `tests/engines/test_rust_integration.py`: 22 pass / 2 xfail (codegate
  correctness + perf, expected). strcpy_find now passes both
  correctness and the 3x perf check.
- `tests/engines/test_rust_exploration.py`: 444/444 still pass.

### Followup work for next session

`bd ready` candidates:
- angr-prem (P2) / angr-prem.1 (P3) — MemoryLayer trait. Deferred
  multiple times; parent notes say premise is wrong.
- angr-fk0m (P2) — refactor sync/cache/export mixins. "Purely
  cosmetic"; deferred multiple times.
- angr-kcf (P2, bug) — parent of just-closed kcf.2; codegate stdin
  variable tracking. Real fix would need a concrete plan to align
  Rust+Python branch decisions on symbolic-memory loads.
- angr-34w.12 (P2, bug) — grub OOM / constraint explosion. Big. See
  bd memory `grub-investigation-depth`.

No new beads created this session.
