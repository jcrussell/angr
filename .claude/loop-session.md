## Session log: 2026-05-11 — angr-z8xa (profile _sync_registers_to_rust, RATIONALE)

### Task
Profile `_sync_registers_to_rust` (claimed 7.8 ms / 46 % of warm
`_add_rust_state`) and decide between optimization or written rationale.

### Profile (wall-clock, fauxware, 100 fresh entry_states)
TOTAL `_sync_registers_to_rust` slow path: **4.40 ms / state**
  - `getattr(state.regs, name)` × 43:  **2.98 ms (68 %)**
  - `z3_backend.convert` × 26 sym:     0.76 ms (17 %, on top of getattr)
  - `set_registers_bulk` FFI:          0.003 ms (~0 %)
  - `set_register_symbolic` × 26 FFI:  0.007 ms (~0 %)

Disk-cache fast path (precomputed_regs dict): **0.007 ms / state** — already
optimal.

The 7.8 ms claim in the bead came from cProfile-inflated `_add_rust_state`
measurements; real wall-clock cost dropped further after l9h7's symbolic-
page fix.

### Decision: rationale, no code change
1. 68 % of cost is `getattr` walking angr's 14-level memory-mixin chain
   (view → name_resolution → ... → ultra_page.load + default_filler).
   This is fundamental angr cost; not safely addressable from the Rust
   extension.
2. `_add_rust_state` is called **once per seed state add** (and per merge),
   NOT per exploration step or per callback. On real benchmarks the 4 ms
   cost is < 0.5 % of total runtime.
3. The disk-cache fast path already short-circuits the slow path entirely
   for repeat runs.
4. Alternatives considered and declined:
   - Bulk-load register file (fragile, mixed concrete/symbolic bytes).
   - Skip unconstrained-default regs (Python ↔ Rust symbol divergence).
   - Cache precomputed dict across seeds (entry_state mints new BVS IDs
     each call, so cached AST pointers would carry stale names).

### Tests
382/385 pass — same 3 pre-existing failures as baseline (no regressions,
no code changes).

### Memory saved
`z8xa-sync-regs-bottleneck` — full profile breakdown + decision rationale.

### Status
COMPLETE — bead angr-z8xa closed as no-action.
