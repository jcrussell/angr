## Session log: 2026-05-14 — Phase 2 lazy STORE landed gated (angr-qh5u)

### Closed task

**angr-qh5u** — Phase 2 lazy symbolic STORE for very large solution
sets. The Multi-cell write path is wired into the default unified
store entry points (`store_symbolic_unified`,
`store_with_concretization`), but the production code keeps the
eager `store_conditional_multiple` / `store_strided` path until a
runtime gate is flipped. The gate is OFF by default per the design
doc's soft-blocker rule because the load-time ITE rebuild regresses
sym-write 10% in this environment.

### What landed (commit bce90fef4)

**Rust side**

- `memory/store.rs`:
  - `install_multi_for_candidates_safe(addr_expr, value, addrs, ctx)`
    — Phase 2's production-safe variant. Returns
    `UnmappedPageInRegion` if any candidate page is unmapped inside a
    declared lazy region (interpreter fetches from Python); silently
    filters non-lazy unmapped candidates (matches
    `prepare_addresses_for_ite` skip semantics); auto-maps nothing.
  - `store_symbolic_unified` and `store_with_concretization` Multiple
    and Strided branches now branch on
    `self.use_multi_cell_stores`: when on, call
    `install_multi_for_candidates_safe`; when off, retain the eager
    `store_conditional_multiple` / `store_strided` path.
  - `store_conditional_multiple` becomes dead code under the gate-on
    path but is kept around for the gate-off path.
- `memory/multi.rs`:
  - `flush_multi_cells(ctx)` collapses every Multi byte to a 1-byte
    `symbolic_objects[byte_addr]` entry via the same right-fold as
    `assemble_load_with_multi`. After flush the page bitmap shows the
    byte as Symbolic (not Multi) so the state export pipeline
    (`_sync_rust_symbolic_objects_to_state`) picks it up. Without
    this step Multi bytes export as concrete `data[]` (usually
    zero), losing the lazy alternatives.
- `memory/mod.rs`:
  - New `use_multi_cell_stores: bool` field on `SymbolicMemory`
    (default false) + setter / getter; propagated through `fork`.
  - `flush_pending_writes` calls `flush_multi_cells` unconditionally
    on the path — a no-op when `multi_objects` is empty (always true
    with the gate off).
- `state.rs`:
  - `set_use_multi_cell_stores` / `use_multi_cell_stores` on
    `RustSimState`, plus PyO3 names so Python can toggle the gate
    once a SimOption hook lands.

**Tests**

- `memory/tests.rs` — 6 new `test_phase2_*` tests:
  * `gate_off_default_eager_store` — default no Multi cells.
  * `gate_on_installs_multi` — flip flag, Multi cells appear, load
    round-trips.
  * `safe_install_lazy_region_signals` — unmapped lazy candidate
    returns `UnmappedPageInRegion { page_addr }`.
  * `safe_install_skips_unmapped_non_lazy` — non-lazy unmapped
    candidate silently filtered, mapped candidate still gets a Multi.
  * `flush_multi_to_symbolic_objects` — flush produces correct
    per-byte ITE under each candidate's concretization constraint.
  * `fork_independence_via_safe_install` — parent / child Multi
    cells diverge after fork.

### Validation

- `cargo test --release --lib`: **762/762** (+6 new).
- `pytest tests/engines/test_rust_exploration.py`: **403/403**.
- `run_single.py sym-write --engine rust`: 1.58s gate-off vs 1.58s
  baseline HEAD (parity, within noise). Gate-on: ~1.76s.
- `run_single.py fauxware --engine rust`: 0.30s (unchanged).
- `run_single.py {ais3_crackme, strcpy_find, defcamp_r100} --engine rust`:
  within 6% of stashed HEAD; baseline_timings.json itself is stale on
  this machine (see new bd memory `baseline-timings-stale-2026-05-14`).

### Acceptance vs the bead

- **≥2× sym-write speedup target**: NOT MET — soft blocker triggered.
  Gate stays off; Phase 2 ships as plumbing-only.
- **No regression on benchmarks outside slower-than-1.0× table**: met
  with gate off (default). Pairwise comparison against pre-Phase 2
  HEAD shows <6% drift on tested benchmarks.
- **389 Python tests pass**: met (403/403).
- **New Rust unit tests cover fork / merge / export**: partial — fork
  and export covered; merge of Multi cells across states still has no
  test (Rust does not have a Multi merge path today; the design doc
  notes merge correctness for Rust-Rust only).

### Why gated

The Multi-cell load collapse runs the alternatives right-fold on every
load to a Multi byte, building fresh Z3 ITEs each time. The eager
`store_conditional_multiple` builds one ITE at store time and the cached
BV serves every subsequent load. For read-heavy symbolic-memory
workloads (sym-write fits this) the eager amortization wins. The fix is
a per-load collapse cache inside `MultiPayload` that memoizes the
collapsed BV and invalidates on alternative append — tracked as
**angr-j0n4** (Phase 3, P3, created this session).

### Memories saved this session

- `qh5u-soft-blocker` — root cause of the 10% gate-on regression
  and the per-load cache fix path.
- `baseline-timings-stale-2026-05-14` — `baseline_timings.json` is
  stale vs current env. Pairwise compare against HEAD with the change
  stashed, not against the JSON, when assessing PR perf.
- `invariant-multi-cell-routing-safe` — production callers must use
  `install_multi_for_candidates_safe`, not the bare installer.

### Open follow-ups

- **angr-j0n4** (Phase 3, P3) — per-load Multi-cell collapse cache.
  Acceptance: gate-on sym-write at parity-or-better with gate-off;
  ideally ≥2× improvement over the eager path. Once met, flip the
  `SymbolicMemory` default and remove the gate.
- The Phase 1.4 `_try_multi_cell_store` annotation gate
  (`rust_manager.py:1632`) is now redundant for in-Rust stores once
  the `use_multi_cell_stores` flag is on. Keep as-is for now; revisit
  when the gate flips in Phase 3.
- `baseline_timings.json` needs a refresh on the current env so the
  PR-time regression gate stops firing on env drift alone.

### Next ready (`bd ready` after close)

- angr-myty (P3, daytime perf dashboard) — Python/CI work.
- angr-j0n4 (P3, Phase 3 collapse cache) — direct follow-up, but
  unproven without measurement infrastructure (Phase 0 ite-depth
  counter already lands the data we need).
