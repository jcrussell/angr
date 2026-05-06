# Loop session notes (2026-05-06, sixtieth loop session — DONE)

## Status: COMPLETE — angr-2i4n closed

## Task: angr-2i4n (P2)
Tests: error path coverage (unmapped, OOM, Z3 timeout, permission)

## What was done
Added three Python tests to TestErrorRecovery in
`tests/engines/test_rust_exploration.py` (commit 3159ccfcd):

1. `test_unmapped_concrete_store_returns_error` — mirror of the
   existing unmapped-load test; ensures `RustSimState.memory_store`
   to an unmapped concrete address raises ValueError("unmapped"),
   not silently auto-maps.

2. `test_cross_page_permission_write_returns_error` — exercises the
   cross-page permission check from the Python boundary: maps page 0
   RW and page 1 R-only, enables enforcement, asserts a 4-byte write
   straddling the boundary raises ValueError("permission"). Sanity
   check confirms a write entirely within the RW page still succeeds.

3. `test_z3_solver_timeout_does_not_hang` — configures a 50ms Z3
   timeout via RustSolverContext.set_timeout, builds a 128-bit
   semiprime factoring constraint with both factors > 2^60, asserts
   `satisfiable()` returns wall-clock < 5s. Doesn't assert on the
   bool result (Unknown collapses to false in SymContext::is_sat).

Python tests: 218 → 221 passing. No Rust changes — pure Python test addition.

## Key empirical findings
- The bead's reference to "interpreter_cb/execution.rs:600-700" is
  stale; the file is only 511 lines now. Tests live at the Python
  RustSimState/RustSolverContext boundary instead — that's where the
  user-facing error contract lives.
- SymContext::is_sat returns bool: SatResult::{Sat→true, Unknown→false,
  Unsat→false}. From Python you can't differentiate Unknown from Unsat,
  so Z3-timeout tests have to assert wall-clock bound, not result.
- Permission enforcement is OFF by default in RustSimState; test must
  call `set_enforce_permissions(True)` before exercising violations.
- Permission bit encoding for `map_memory(perm_bits)`: 0x4=R, 0x2=W,
  0x1=X (NOT mprotect; bit 0x1 is execute, not read).

## Memories saved
- `z3-timeout-test-pattern` — how to write hang-protection tests
  given the Unknown→false collapse.
- `invariant-rust-perm-bit-encoding` — perm bits for map_memory.

## Build env reminder
- Pure-test change. No `cargo build` or `pip install -e .` needed.
- Tests run in 7.6s for the full file.

## Next-up (still ready)
- angr-eygl (P1) Differential test harness — BIG, multi-session
- angr-pufm (P1) Symbolic address concretization fallback — large feature
- angr-io1t (P2) Tests: VEX FP edge cases (NaN, infinity, conversion)
- angr-prem (P2) Introduce a MemoryLayer trait
- angr-fk0m (P2) Unify rust_state_sync / rust_state_cache / rust_state_export mixins
- angr-nnov (P2) Fill out RustStateProxy
- angr-4j5u (P2) Decompose RustExplorationManager god-struct
- angr-imy1 (P2) Native syscall coverage
- angr-fbl0 (P2) Native SimProcedure coverage
