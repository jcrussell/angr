## Session log: 2026-05-14 — Phase 3 per-load Multi-cell collapse cache landed (angr-j0n4)

### Closed task

**angr-j0n4** — Phase 3 per-load collapse cache for `MultiPayload`. The
cache is plumbed and tested; the Phase 2 gate stays OFF because the cache
alone does not reach parity with the eager `store_conditional_multiple`
path on sym-write.

### What landed

**Rust side**

- `memory/multi.rs`:
  - New `CachedCollapse { default_byte: u8, bv: RustBV }` struct (private).
  - `MultiPayload` gains `cached_collapse: RefCell<Option<CachedCollapse>>`.
    Manual `Clone` impl propagates the cache (Z3 ASTs are refcounted, so
    cloning is cheap and the fork path benefits).
  - New `MultiPayload::collapse(&self, default_byte, ctx) -> RustBV` —
    returns the cached BV when `default_byte` matches, otherwise rebuilds
    via the same right-fold as before and caches the result.
  - `push` now invalidates the cache (defensive: `set_multi_alternatives`
    always replaces the entry with a fresh payload, but the standalone
    push path stays sound).
  - `flush_multi_cells` uses `payload.collapse(...)` so a load that
    populated the cache pays zero extra Z3 work at flush.
- `memory/load.rs`:
  - `assemble_load_with_multi` replaces the inline right-fold with
    `payload.collapse(concrete_byte, ctx)`. The `record_mem_ite_depth`
    contract still fires per-byte to keep the counter semantics.
- `memory/mod.rs`:
  - Updated the `use_multi_cell_stores` field doc to reflect Phase 3
    findings (gate stays off; cache reduces gate-on cost from 1.76s to
    1.72s but ~10% regression vs gate-off remains; residual cost is
    per-byte assembly + per-byte export, not load-time ITE rebuild).

**Tests** (`memory/tests.rs`, 4 new `test_phase3_*`):

- `collapse_cache_hit_after_load` — cache empty pre-load, populated after
  first load, second load returns equivalent BV.
- `collapse_cache_invalidated_on_push` — `MultiPayload::push` clears the
  cache.
- `collapse_cache_invalidated_on_default_byte_change` — different default
  byte produces a rebuilt collapse with the new ELSE leaf.
- `collapse_cache_clones_with_payload` — `Clone` carries the cache;
  mutating the clone does not touch the original.

### Validation

- `cargo test --release --lib`: **766/766** (+4 new from 762).
- `pytest tests/engines/test_rust_exploration.py`: **403/403**.
- `run_single.py sym-write --engine rust` (gate OFF default): 1.55s × 3
  (parity with pre-Phase 3 baseline 1.56-1.58s).
- `run_single.py sym-write --engine rust` (gate ON, transiently flipped):
  1.71-1.74s × 3 (down from 1.76s pre-Phase 3, ~2% improvement; still
  ~10% slower than gate-off 1.55-1.58s).
- `run_single.py fauxware --engine rust`: 0.28s (parity).
- `run_single.py ais3_crackme --engine rust`: 2.03s (parity).
- `run_single.py defcamp_r100 --engine rust`: 0.27s (parity).

### Why the gate still stays off

The Phase 3 cache eliminates load-time ITE rebuild, which was the
documented Phase 2 soft blocker. But sym-write's z3_check counter is
unchanged (z3_check ≈ 144 ms either way) and the load_stmt budget is tiny
(8 ms of 1.55 s). The residual ~10% gap is downstream of memory:

- Per-byte iteration in `assemble_load_with_multi` — even cache hits do
  N page lookups + N cache reads + N byte concatenations per load,
  versus the eager path's single `symbolic_objects.get(addr)` returning
  a width-N BV.
- `flush_multi_cells` writes per-byte `symbolic_objects` entries; the
  state-export pipeline serialises each entry. Eager stores write one
  width-N entry per candidate address. The per-byte expansion blows up
  export volume by 4× for 32-bit loads / 8× for 64-bit loads.

A wider-load collapse cache (cache the N-byte assembled BV keyed on the
start address + size + payload version) would address (1). Coalescing
per-byte symbolic_objects entries on flush would address (2). Both are
out of scope for angr-j0n4; the cache work itself is correct, tested,
and landed.

### Follow-up

New bead created for wider-load collapse cache + per-byte symbolic_object
coalescing on flush. Saved memory `phase3-cache-residual-gap` with the
profile data so the next investigation does not have to re-discover that
load-time ITE rebuild is no longer the bottleneck.
