# Loop session notes (2026-05-03, twenty-third session)

## Task: angr-07tg
Audit deferred fork base in SimProc-native success path (stepping.rs).

## Status
- DONE. Commit 184a61f76. Closed angr-07tg.

## What landed
- SimProc-native success path now calls `process_deferred_forks_into()`
  instead of inlining its own ~40-line deferred-fork loop with a `fork_base`
  clone. Same call signature as the existing P21 / SymbolicJumpTarget paths.
- Net: -43 lines, +9 lines.

## Audit conclusion (recorded in memory)
1. **Primary path (always snapshot-based in practice)**: equivalent.
   `fork_from_snapshot` REPLACES the solver/registers/memory entirely with
   the snapshot's state. The parent state for the call (`successors[0]` vs
   `fork_base`) only affects inherited fields (history, call_stack, etc.),
   and those fields are identical at the cloning point in both code paths.
2. **No-snapshot fallback (unreachable — every deferred fork creates a
   snapshot at interpreter_cb/statements.rs:389)**:
   - MaxBlocks/BlockEnd: would produce UNSAT (F_1..F_n + !F_n) → pruned.
   - SimProc-native: would produce spurious state (only !F_n, missing prior).
   Both wrong but unreachable.
3. Therefore safe to unify on MaxBlocks/BlockEnd / process_deferred_forks_into.

## Bonuses from unification
- SimProc-native gains UNSAT-pruning to STASH_PRUNED.
- SimProc-native gains P15 conservative-fork for missing-condition path.
- Eliminates `let fork_base = successors[0].fork()` clone.

## Tests
- 208/208 passing in 6.79s.
- fauxware --engine rust completes in 0.38s, finds SOSNEAKY (no regression).

## Files modified
- native/angr/src/exploration/stepping.rs (+9, −43)

## Memories saved
- `invariant-deferred-fork-snapshot-primary`: fork_from_snapshot replaces
  solver/registers/memory entirely; parent only affects inherited fields.
- `invariant-symcontext-push-not-cache-aware`: SymContext::push/pop is
  solver-only; the z3_assertions_local cache survives a pop. fork() reads
  the cache, so snapshots inside a pushed frame capture the pushed
  constraints (this is what makes the block-level snapshot model work).

## Other ready tasks (P3)
- angr-xl2f: decompose stepping.rs 13-arm match into per-result methods
- angr-w4os: Python bridge cleanup
- angr-2fs0: decompose _handle_simprocedure_callback (~478 lines)
- angr-cbko: native exit/abort SimProcs
- angr-8em4: replace panic patterns
- angr-3ijo: bincode for VEX IRSB serialization
- angr-bgv0: Z3 floating point theory
- angr-awm3: CAS/LLSC statement handling
- angr-v4db: extract god-methods in Python bridge layer

## Other ready blocked
- angr-3tek (P2): native read/write SimProcs blocked by stale-cache
