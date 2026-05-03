# Loop session notes (2026-05-03, thirtieth session)

## Task: angr-awm3 — CLOSED
"Implement CAS/LLSC statement handling in Rust VEX interpreter"

## Outcome
Both `IRStmt::CAS` and `IRStmt::LLSC` previously returned `Unsupported`
in the callback interpreter (statements.rs:836-839), forcing a Python
VEX fallback via `RunResult::NeedPythonVEX`. Implemented the simplified
non-atomic single-state semantics that mirror angr's Python light engine
(angr/engines/vex/light/light.py:289-528).

### Coverage
- LL  (LLSC, storedata=None): load → result temp.
- SC  (LLSC, storedata=Some): store; write 1 → result temp.
- CAS (single): load current, cmp = current == expd; conditional store
  (concrete cmp → unconditional / skip; symbolic cmp → ITE(cmp, data, current)).
  Write current → oldLo.

### Falls back to Python (still Unsupported)
- Double-CAS (oldHi/expdHi/dataHi present).
- CAS with symbolic address when cmp is also symbolic
  (would require addr concretization + per-branch store synthesis).

### Implementation note
The clean reuse pattern — synthesize an `IRStmt::Store` and recurse via
`execute_stmt_with_callbacks` — only works when the value to store is a
concrete `IRExpr` (the original `dataLo`). For symbolic-cmp CAS the value
is a *computed* `RustBV` (the ITE) and cannot be wrapped back into
`IRExpr` without mutating `tyenv` mid-execution. So that case does the
store directly via callbacks/pending_stores (restricted to concrete addr).

## Changes (commit 799981d20)
- native/angr/src/interpreter_cb/statements.rs (+139 / −3)

## Verification
- `cargo check --release` clean.
- 208/208 Python rust-exploration tests pass.
- 370/370 Rust unit tests pass.
- fauxware benchmark: no regression visible at single-run granularity.

## Memories saved
- `cas-llsc-recursion-limit`: why we can't recurse to Store for symbolic-cmp CAS
- `invariant-cas-llsc-coverage`: what's covered vs. still-deferred-to-Python

## Next ready (P3, no blockers)
- angr-cbko: native exit/abort SimProcs (warning: caused infinite zero-length
  hook loops in v2; needs find/avoid sync)
- angr-bgv0: Z3 floating point theory support (30+ ops in vex/ops.rs)

## Remaining big methods (still untracked, for a future refactor session)
From last session's note — top unsplit methods:

  rust_state_export.py:
    _get_stash_states               139 lines
    _sync_exported_constraints       98 lines
    _sync_rust_memory_to_state       79 lines
  rust_state_sync.py:
    _extract_wide_symbolic_regions  111 lines
    _extract_register_changes        71 lines
    _sync_registers_to_rust          66 lines
  rust_manager.py:
    _cb_resolve_function            100 lines
    _cb_sync_constraints             87 lines
    merge                            76 lines
