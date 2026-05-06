# Loop session notes (2026-05-06, seventy-eighth loop session — DONE)

## Status: COMPLETE — angr-zrq1 closed

## Task: angr-zrq1 (P2) — Dirty-call coverage: symbolic guard, symbolic args, missing-handler stubs

Closed the carry-over hard-error paths in
`native/angr/src/interpreter_cb/statements.rs::IRStmt::Dirty`:

1. **Symbolic guard** (was Err): now `check_branch_feasibility` →
   - `!cb_true` → skip (provably false guard)
   - `cb_false` → `assume_true(guard)` (concretize-to-taken; lossy but
     unblocks; matches angr Python's existing pattern, which doesn't
     even check the guard).
   - `!cb_false` → guard is provably true; just fall through.
2. **Symbolic args** (was Err): eager-concretize via `ctx.eval` +
   `assume_true(arg.eq(concrete))`. Native dispatch and Python
   callback both see concrete u64 args afterward.
3. **Missing handler + no Python callback** (was Err): warn-log,
   write a fresh symbolic tmp (when `dirty.tmp` exists) at the
   correct width, continue. Mirrors Python `_cb_dirty_call` stub
   at rust_manager.py:958 which returns zero-bytes + False.

### Verification
- pytest tests/engines/test_rust_exploration.py: 243/243 pass (8.24s)
- cargo test --lib --release --features vex-engine-z3: 463/463 pass
- Quick benchmarks: fauxware 0.10s, defcamp_r100 0.22s,
  securityfest_fairlight 14.81s (all within baseline variance).

### Commit
- 1e901a407 fix(vex/dirty): graceful fallback for symbolic guard/args/missing handler

### Memories saved
- `invariant-vex-dirty-symbolic-guard` — semantics of symbolic guard
  on dirty calls; concretize-to-taken is at least as careful as angr
  Python which ignores the guard outright.
- `invariant-eager-concretize-pattern` — the standard ctx.eval +
  assume_true(eq) recipe for symbolic-must-be-concrete sites.
- `invariant-rust-python-dirty-callback-symmetry` — _cb_dirty_call
  is registered unconditionally; stub-on-no-handler is defense-only
  but mirrors Python-side behavior on unknown handler names.

### Build environment notes (carry-over)
- `Z3_SYS_Z3_HEADER=/usr/include/z3.h` (.venv has pyc-only z3)
- `cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`
- pytest needs `PYTHONPATH=.`
- pip install in this venv currently broken (resolvelib import error);
  use the cargo-build + cp .so path instead.

## Next-up (still ready)
- angr-eygl (P1) Differential test harness — substantial infra
- angr-pufm (P1) Symbolic address concretization fallback — architectural
- angr-prem (P2) MemoryLayer trait
- angr-fk0m (P2) Unify rust_state_sync / cache / export mixins
- angr-imy1 (P2) Native syscall coverage (exit, write, read, mmap, brk, mprotect)
- angr-fbl0 (P2) Native SimProcedure coverage gaps (audit needed)
- angr-3zs6 (P2) FallbackStrategy enum
- angr-nnov (P2) Fill out RustStateProxy
- angr-4j5u (P2) Decompose Rust-side RustExplorationManager
