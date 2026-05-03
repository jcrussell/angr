# Loop session notes (2026-05-03, thirty-second session)

## Task: angr-cbko — CLOSED
"Enable native exit/abort SimProcedures (needs find/avoid address sync with Python)"

## Outcome
Re-enabled NativeExit / NativeUnderscoreExit / NativeAbort. Commit 7cd3d7431.

## Root Cause (was non-obvious)
The VEX interpreter has TWO native SimProcedure dispatch paths:
  1. `stepping.rs:handle_simprocedure` — called when a block ends with
     `call extern_hook` and the interpreter dispatches the hook inline.
  2. `mod.rs:run` top-of-loop check — when the next state's PC is at a
     hook address before stepping.

Both paths advanced PC to `return_addr` and popped SP after the native call,
ignoring `no_return`. For NativeExit, that "return address" in fauxware is
0x40071d (after `call exit@plt` at 0x400718) — which by coincidence is the
START of main() (rejected() and main are adjacent in this binary). So
deadending the wrong way caused infinite re-entry into main.

## Fix
Both dispatch sites now early-deadend when `no_return` is true:
- Skip PC/SP/return-value update.
- Run `process_deferred_forks_into` so unexplored branches still spawn.
- Push the main state to STASH_DEADENDED instead of returning it as a
  successor.

## Files changed
- native/angr/src/exploration/stepping.rs (handle_simprocedure)
- native/angr/src/exploration/mod.rs (run loop top-of-loop hook)
- native/angr/src/procedures/mod.rs (registration)

## Verification
- 208/208 RustExplorationManager tests pass.
- The previously-hanging test `test_step_func_stops_when_no_active`
  (fauxware step+step_func, n=100000) now finishes in 18 steps,
  matching the no-native-exit baseline.
- Regression test: 12/12 correct, 1 SLA WARN on flareon2015_2
  (pre-existing variance — runs identically with/without my change).

## Memories saved
- invariant-no-return-deadend (the rule)
- avoid-fixing-only-one-native-dispatch-path (only fixing mod.rs misses
  the interpreter-driven path in stepping.rs)
- fauxware-exit-overlaps-main (the binary quirk that makes this bug visible)
- no-native-exit-simprocs (updated — supersedes the old "AVOID" memory)
