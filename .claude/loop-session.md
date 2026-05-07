# Loop session notes (2026-05-07, 133rd loop session)

## Task: angr-q7r0 — sym-write regressed +660% (3.22s vs 0.42s baseline)

### Status: closed (fixed)

### Root cause
sym-write regression bottleneck was NOT memory_*_symbolic_full callbacks
(audit hypothesis). Only 3 memory_load callbacks fire during sym-write.

True bottleneck: callback-time register sync of `edx`. After main runs,
`edx` contains a RustBV::Expression with 142,133 expanded nodes but only
**25 unique Arc-shared subtrees** (DAG with heavy reuse).
`rustbv_to_claripy` recursively walked the DAG without memoization,
fanning it into a tree and creating ~142k Python claripy objects (~2.7s
for one register).

### Fix
commit c88947e65: memoize `rustbv_to_claripy` by RustBV pointer identity.
Operands stored inline inside a shared `Arc<[RustBV]>` have stable
addresses, so sibling references hit the cache. Per-call
`HashMap<*const RustBV, Py<PyAny>>`. Memoization gated on Expression
variant only.

### Results
- sym-write: **3.22s → 0.44s** (matches 0.42s baseline)
- 261/261 tests pass
- 12/12 regression-suite benchmarks pass (19.7s total)

### Memories saved
- rustbv-to-claripy-memoization
- symwrite-regression-q7r0-root-cause
- invariant-rustbv-arc-operands-shared
- benchmark-q7r0-symwrite-recovered

### Follow-on
- angr-491g (concrete-addr fast path): premise no longer holds for
  sym-write. Annotated bead with note. mma_howtouse motivation should be
  re-profiled before that work is taken on.
