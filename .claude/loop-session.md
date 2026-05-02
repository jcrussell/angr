# Loop session notes (2026-05-02, fourteenth session)

## Closed: angr-te1b

[bug] Z3 AST pointer extraction in solver.rs lacks null/validity check
before unsafe use.

### Root cause

`extract_z3_ast_ptr` (solver.rs:19-30) returned `Ok(ptr)` regardless of
whether the extracted pointer was null. Three call sites guarded with
`if z3_ptr != 0` before passing the value to
`NonNull::new_unchecked`. A future caller forgetting that guard would
invoke UB.

### Fix

Move validation to the source. `extract_z3_ast_ptr` now returns Err on
null. Removed the three redundant `!= 0` guards at call sites
(`add_constraint_ast`, `eval`, `eval_upto`) and added a SAFETY comment
referencing the invariant.

### Verification

- `cargo check --release` clean (no warnings).
- `python -m pytest tests/engines/test_rust_exploration.py` →
  208/208 passing.
- `run_single.py fauxware --engine rust` → behaves identically.

### Memory saved

`invariant-extract-z3-ast-ptr` — future work that touches the function
must preserve the null-Err contract or reinstate guards at all 3
unsafe call sites.

## Carryover from session 13

- Pre-existing baseline timing variance in `run_regression.py` (ais3 +75%,
  re400 +62%, etc.). Need to either re-record or widen tolerances.
- Ready P-tasks: angr-210j, angr-3tek, angr-mboi, angr-xidi.
- P3 refactors: angr-w4os, angr-2fs0, angr-1f8s, angr-4knw.
