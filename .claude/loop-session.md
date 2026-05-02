# Loop session notes (2026-05-02, thirteenth session)

## Closed: angr-y07x

[bug] Exit-only continuation cache race:
`_exit_continuation_addrs` incorrectly deadended state-dependent
SimProcedures (`rust_callback_dispatch.py:618-628`).

### Root cause

The cache was keyed on address only. When a SimProcedure callback
produced ALL-Ijk_Exit successors on first observation, the address
was cached and all future callbacks at that address were
fast-deadended without recreating the SimState or running the
procedure. But Ijk_Exit-on-all-successors is **state-dependent**: a
generic SimProcedure could exit for one state and return normally for
another. The cache would then incorrectly deadend the second state.

### Fix

Single-line change: require `proc.NO_RET=True` before adding an
address to `_exit_continuation_addrs`. Rationale: continuations
inherit NO_RET from the canonical procedure (`make_continuation` does
`copy.copy(self.canonical)`), so `__libc_start_main.after_main`
(NO_RET=True) is still cached. State-dependent procedures default to
NO_RET=False and are correctly NOT cached.

### Verification

- `python -m pytest tests/engines/test_rust_exploration.py` →
  208/208 passing.
- `run_single.py ais3_crackme --engine rust` → 47 callbacks
  (matches the cached benchmark in
  `bd memories exit-continuation-cache`).
- `run_regression.py` shows pre-existing timing regressions vs
  baseline that ALSO appear without my change (ran with `git stash`
  to confirm). System load variance, not regression from my fix.

### Memory saved

`invariant-exit-continuation-cache` — future work must respect that
the cache ONLY accepts NO_RET=True procedures.

## Observation: pre-existing baseline timing variance

`run_regression.py` flagged 8 timing regressions today (ais3 +75%,
re400 +62%, etc.) that appear identically with and without my change.
Either system load is heavier than when 12th session set the
baselines, or the baselines need a re-record.

NOT my fix's fault. May warrant a follow-up bead to either re-record
or widen tolerances.

## Ready P-tasks remaining

- angr-te1b (P2 Z3 AST pointer extraction null check)
- angr-210j (P2 CCall returns concrete(0) instead of Python fallback)
- angr-3tek (P2 native read/write SimProcedures disabled)
- angr-mboi (P2 mma_howtouse 0.7x regression)
- angr-xidi (P2 google2016_unbreakable_1 3.2s→6.2s)
- angr-w4os, angr-2fs0, angr-1f8s, angr-4knw (P3 refactors)
- and the P-tasks tracked in 12th-session notes.
