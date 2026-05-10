## Session log: 2026-05-10 — angr-1w75 (215th loop session, COMPLETE)

### Task
Close Liskov gaps in RustStateProxy:
1. `RustRegisterProxy.load(offset_int[, size])` — was NotImplementedError.
2. `RustMemoryProxy.load(symbolic_addr, size)` — was NotImplementedError.

Depended on angr-25za (SSOT) which landed last session.

### Implementation

`angr/exploration/rust_state_proxy.py`:

**Registers (line ~306):** `load()` now accepts `(offset:int, size:int|None)`
and resolves through `arch.register_size_names[(offset, size)]` (size
defaults to `arch.bytes`, matching SimMemory). Unknown `(offset, size)`
raises NotImplementedError with the arch name in the message; non-str/non-int
input raises TypeError. String-name path unchanged.

**Memory (line ~325):** `load()` with a non-concrete claripy addr now
calls `self._mgr.fork_state_solver(state_id)` once (lazy-cached on
RustMemoryProxy) and eval()'s the address under the state's constraints,
then falls through to the existing concrete-read path. Unsat addrs raise
`claripy.errors.UnsatError` — same convention as `RustSolverProxy.eval`.

### Tests

Added `TestProxyLiskovGaps` (6 tests) in tests/engines/test_rust_exploration.py:
- register load by offset (rax via offset 16, default size 8)
- register load with sub-register size (eax via offset 16, size 4)
- bad offset raises NotImplementedError mentioning arch
- wrong-type arg (float) raises TypeError
- symbolic-addr memory load resolves under constraint to concrete read
  (matches `SimState.memory.load()` reference value at the same addr)
- unsat symbolic addr raises UnsatError (contradictory eq constraints)

### Verification

- `pytest tests/engines/test_rust_exploration.py`: 376 passed, 3 pre-existing
  failed (dcas/pipe/dup2). Zero regressions. (370 → 376 = 6 new.)
- New tests run in 1.48s.
- Implementation diff is pure Python; no Rust rebuild needed.

### Files changed

- `angr/exploration/rust_state_proxy.py` (+39 lines)
- `tests/engines/test_rust_exploration.py` (+111 lines, new TestProxyLiskovGaps class)

### Memories saved

- `invariant-rust-proxy-load-by-offset` — documents the (offset, size) →
  name resolution path via arch.register_size_names; future archs that
  add new registers must keep that table accurate.
- `invariant-rust-proxy-symbolic-load-eval` — symbolic-addr loads use a
  single-solution eval, not an ITE over possible addresses; callers
  needing multi-solution semantics must reach for the solver themselves.

### Closed beads

- `angr-1w75`.

### Status

COMPLETE.
