## Session log: 2026-05-16 — angr-uq4n.1 closed (mem_read/mem_write inspect survey)

### Status: closed

### Task

**angr-uq4n.1** — "survey hook points in VEX interpreter + memory
model for mem_read/mem_write dispatch."

Pure research / design-note task. No code changes. Output is a
design note attached as bd note + memory entry.

### Findings (KEY)

**Two dispatch surfaces in Rust, not one:**

1. `state.rs` wrappers (lines 1350/1356/1361/1368/1385) —
   `memory_load`, `memory_store`, `memory_load_symbolic`,
   `memory_store_symbolic`, `memory_store_symbolic_multi`. These
   transitively cover all 20 native SimProcedures (~106 sites) — no
   per-procedure instrumentation needed.

2. `interpreter_cb/` VEX sites that BYPASS state wrappers:
   - `expressions.rs:49` IRExpr::Load (8+ fast paths: pending_stores,
     pending_symbolic_stores, flushed stores, prefetch cache,
     concrete cache, callback fallback, ITE builds, symbolic-full)
   - `statements.rs:54` IRStmt::Store
   - `statements.rs:302` IRStmt::StoreG (guarded)
   - `statements.rs:549` IRStmt::LoadG (guarded)

**Critical anti-pattern**: do NOT instrument
`native/angr/src/memory/{load,store,mod}.rs` backends. They would
(a) miss VEX fast paths (events never fire because fast paths skip
the backend entirely), and (b) double-fire for SimProcedures.

**Plumbing requirement**: `interpreter_cb` sites have no
`&mut RustSimState`. Need (1) new `set_inspect_mem_read/write`
methods on `PythonCallbacks`, (2) `inspect_enabled: u8` bitmask field
on `PythonCallbacks` for zero-overhead skip when no breakpoint
registered (the common case), (3) Python-side dispatcher that
reads `state.inspect._breakpoints['mem_read']` and fires each BP.

`interpreter.rs` simple `VEXInterpreter` (Load at 366, Store at 255)
is legacy/test (only used by `RustVEXEngine::execute_cached_block`,
engine.rs:1406) — out of MVP scope.

### Deliverables

- **bd note** on `angr-uq4n.1` — full 138-line design note with
  per-site table + Python attribute shape + phasing recommendation.
- **bd memory** `inspect-mem-dispatch-surfaces` — distilled
  call-site list + the anti-pattern + the plumbing requirement.

### Phasing recommendation for the rest of the epic

- **uq4n.2** (next, unblocked now) — Python callback marshalling
  layer: add `set_inspect_mem_read/write` on `PythonCallbacks`, wire
  `inspect_enabled` bitmask, replace `_NoOpInspectProxy` registration
  with real wiring.
- **uq4n.3** — Instrument the 5 sites (3 state.rs + 4 interpreter_cb).
- **uq4n.4** — Integration tests.
- **uq4n.5** — Update `invariant-rust-inspect-unsupported` memory +
  `docs/advanced-topics/rust_engine.rst`.

### Commit

No code change. Task is research-only.
