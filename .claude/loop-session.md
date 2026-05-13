## Session log: 2026-05-13 — angr-ruay (Pre-existing test failures) — CLOSED

### Task

**angr-ruay** (P2 bug) — Three tests were failing on rust-symex HEAD
(post-7419bf545) before any local changes:

- `TestErrorRecovery::test_dcas_cmpxchg16b_no_match_keeps_memory`
- `TestNativeFileDescriptorProcedures::test_pipe_native_dispatch_creates_two_fds`
- `TestNativeFileDescriptorProcedures::test_dup2_native_dispatch_redirects_stdin`

All three exhibited "no state in any stash" — exploration emitted no
states at all, in any stash bucket.

### Root cause

Commit `974e2daeb fix(rust-symex): route ret-with-empty-call-stack to
unconstrained — angr-3uye.2` (May 10) changed where a `ret` with an
empty call stack routes its state: previously it ended up in
`deadended` (via UnmodeledCall → unmodeled_call_generic_skip), now it
goes to `unconstrained` (matching Python's `_eval_target_brutal`).

The three tests were added BEFORE that fix:
- pipe/dup2 tests in `41428af17` on May 8
- DCAS no-match test in `ed249907a` on May 9

By default `RustExplorationManager` is built with
`save_unconstrained=False`, which causes `_explore_with_addresses` to
call `_rust_mgr.clear_stash('unconstrained')` after every batch
(rust_manager.py:2871-2878). So the state lands in `unconstrained`
inside Rust, then Python wipes it. The Rust debug log confirmed:
`State 1 moved to unconstrained stash`.

### Fix

Updated the three tests to:
1. Build the manager with `save_unconstrained=True` so the post-step
   state stays alive for inspection.
2. Include `unconstrained` in the list of stashes the test inspects.

For DCAS test: `survived` now includes `mgr.unconstrained`.
For pipe/dup2 tests: `_all_fds` now iterates `("active", "deadended",
"errored", "unconstrained")` instead of dropping the last one.

### Verification

- 3 previously failing tests pass individually.
- Full `tests/engines/test_rust_exploration.py` is 392/392 (was 389/392).
- No Rust code changed, no rebuild needed.

### Memories planned

- Update/replace `pre-existing-dcas-test-failure` with the resolution.
- NEW `invariant-ret-empty-callstack-save-unconstrained` — when writing
  a test that steps a shellcode whose final instruction is `ret` (with
  no preceding `call`), or whose return address points outside the
  binary, expect the state in `unconstrained` (per angr-3uye.2 routing)
  and build the manager with `save_unconstrained=True`.

### Files changed

- `tests/engines/test_rust_exploration.py` (3 tests fixed)
