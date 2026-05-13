## Session log: 2026-05-13 — angr-w2je (unsat_core() FFI completion)

### Task
P2: `RustSolverContext.unsat_core()` was exposed via PyO3
(solver.rs:532) but always returned `[]` — `add_constraint_ast()` uses
`add_constraint_raw()` which calls `solver.assert()` (untracked), not
`solver.assert_and_track()`. The `constraint_trackers` vector stayed
empty, so the post-UNSAT lookup found no matches.

### What was done
- `native/angr/src/solver.rs` — added
  `add_constraint_tracked_ast(ast) -> int`. Mirrors
  `add_constraint_ast` but routes through the new
  `SymContext::add_constraint_tracked_indexed` and returns the
  assigned tracker index. Both the fast (raw Z3 ptr) and slow
  (claripy → RustBV → to_z3_bool) paths are supported; without z3
  feature the method returns `PyRuntimeError`.
- `native/angr/src/symbolic/context.rs` — renamed
  `add_constraint_tracked` → `add_constraint_tracked_indexed` and
  changed return type from `()` to `usize` (tracker index). No
  existing callers, safe rename. Hot-path `add_constraint`/_raw paths
  untouched.
- Added three test cases under `TestSolverOperations`:
  - `test_unsat_core_reports_contributing_indices`: idx returned from
    add_constraint_tracked_ast, asserts core contains the contradicting
    pair.
  - `test_unsat_core_empty_when_untracked`: locks down the silent-
    empty behaviour for the untracked fast path (matches the
    `avoid-rust-tracking-actions-silent-ignore` memory).
  - `test_unsat_core_empty_when_sat`: tracked but SAT → core is [].

### Verification
- `cargo check --release` — clean.
- `tools/rebuild-rust.sh --cargo-only` — succeeded (pip in this
  venv is broken, per the prior session note).
- pytest: 389 passed, 3 pre-existing failures
  (`test_pipe_native_dispatch_creates_two_fds`,
  `test_dup2_native_dispatch_redirects_stdin`,
  `test_dcas_cmpxchg16b_no_match_keeps_memory`). Baseline at HEAD was
  386 passed + 3 failed; new tests add +3. No regressions.

### Architectural note
The fast path (extract_z3_ast_ptr) preserves claripy's original Z3
AST structure. The slow path goes through claripy_to_rustbv +
to_z3_bool — width 1 BV is treated as a bool directly; wider BV is
converted via ne(0). Both paths feed
`SymContext::add_constraint_tracked_indexed` which is the single
write point for `constraint_trackers`.

### Prior session
`angr-dxsf` (gitignore) closed clean at 58e50aaf0.
