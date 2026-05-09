## Session log: 2026-05-09, 181st loop session

### Task: angr-agvl — Constraint-freshness fork mutex audit / regression test — CLOSED

The Rust SymContext::fork() (native/angr/src/symbolic/context.rs:1823)
freezes assumed_constraints_local + z3_assertions_local into shared Arc
when push_level==0, with an in-place drain via `Arc::get_mut` when
uniquely owned (commit 589b814e9). No regression test asserted that
post-fork constraints stay isolated between parent and sibling.

### Implementation

Added two regression tests to `TestSolverOperations` in
tests/engines/test_rust_exploration.py:

1. `test_fork_constraint_bidirectional_isolation` (level-0 freeze path):
   - ctx_a asserts X=5 → fork into ctx_b → A asserts Y=10, B asserts Z=20
   - Both sat; X visible to both (frozen prefix); A.eval(y)==10, B.eval(z)==20
   - Negative proof: adding z!=20 to A and y!=10 to B must remain SAT
     (UNSAT would prove leakage)

2. `test_fork_inside_push_isolation` (in-transaction freeze path):
   - ctx_a asserts X=5; ctx_a.push(); ctx_a asserts Y=10; fork into B
   - A then asserts Z=20 (still in push frame); B asserts Z=99 (level 0)
   - Both sat with their own Z; A.pop() then z!=20 still SAT
   - Exercises the `in_transaction` branch of freeze_z3_assertions
     (context.rs:2060) — must allocate fresh merged Vec, not drain local
   - Locks `invariant-symcontext-push-not-cache-aware`: pop unwinds the
     solver but cache keeps the assertion; we test solver behaviour only

### Verification

- Both new tests pass in 1.22s
- Full suite: 348/348 pass in 18.64s (was 346 — added 2 isolation tests)

### Files modified

- tests/engines/test_rust_exploration.py (+62 lines, two new tests)

### Outcome

Tests passed first try. Fork constraint isolation invariant is now
locked. No code change to context.rs needed — the existing freeze logic
correctly isolates parent and sibling constraint vectors in both the
level-0 (drain via Arc::get_mut) and in-transaction (allocate fresh
merged Vec) freeze paths.
