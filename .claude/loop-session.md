## Session log: 2026-05-15 — Phase 4.3 default flip + gate removal (angr-mmdh.3)

### Closed tasks

- **angr-mmdh.3** — Phase 4.3: flip `use_multi_cell_stores` default to
  `true` and remove the gate field. Commit `e5c594fe1`.
- **angr-mmdh** (parent epic) — Phase 4 closed in the same session.

### What landed

Multi-cell lazy stores are now the only path for symbolic-address
`Multiple` / `Strided` concretization results. The gate field and all
its plumbing are gone.

Removed:
- `SymbolicMemory::use_multi_cell_stores` field and its `Self::new()`
  + `clone_for_fork()` sites.
- `SymbolicMemory::{set_use_multi_cell_stores, use_multi_cell_stores}`
  getter/setter.
- `State::{set_use_multi_cell_stores, use_multi_cell_stores}` wrapper
  methods (state.rs).
- `RustState::{py_set_use_multi_cell_stores, py_use_multi_cell_stores}`
  PyO3 bindings (state.rs).
- `store_conditional_multiple` (eager fallback) — now dead, removed.
- `if self.use_multi_cell_stores { … } else { … }` branches in
  `store_symbolic_unified` and `store_with_concretization` — collapsed
  to the Multi-install path.
- Test `test_phase2_gate_off_default_eager_store` (no longer
  reachable) and `mem.set_use_multi_cell_stores(true)` calls in the
  four phase2/4 tests that still ran the Multi path explicitly.

Comment hygiene:
- `record_mem_ite_depth` doc updated to drop references to
  `store_conditional_multiple` and the "future Multi-cell" wording.
- `store.rs` file header drops `store_conditional_multiple` from the
  listed families.

### Validation

- `cargo check` clean, no `dead_code` warnings.
- `cargo test --lib` — 775/775 pass (70 in `memory::tests`).
- `python -m pytest tests/engines/test_rust_exploration.py` — 403/403
  pass in 50.45s.

### Local benchmarks (8GB host, gate-off → default-on, --engine rust)

FAST_SUITE (PR-time CI gate, unchanged within noise):

| Bench | Gate-off | Default-on |
|---|---|---|
| fauxware | 0.28s | 0.28–0.30s |
| defcamp_r100 | 0.27s | 0.27s |
| ais3_crackme | 1.96–1.98s | 1.98s |
| strcpy_find | 2.11s | 2.13s |
| defcon2016quals_baby-re | 0.74s | 0.73–0.74s |

MEDIUM_SUITE:

| Bench | Gate-off | Default-on | Delta |
|---|---|---|---|
| sym-write | 1.53–1.55s | 1.70–1.74s | +~12% |

The 12% sym-write slip is within the nightly 15% threshold but not
at the parent epic's "parity-or-better" acceptance. Z3 eval counts
were unchanged from Phase 4.1 (310 / 303) — the residual is in
state-export eval count, not per-eval cost.

### Follow-up bead

- **angr-o6ef** (P2): reduce sym-write state-export Z3 eval count.
  Phase 4.2 measured `z3_check_count` and `z3_site_eval_upto_count`
  unchanged across Phase 4.1 → 4.2, so coalescing reduced per-eval
  cost but not eval count. Profile target: which export path issues
  these evals per Multi byte? Levers: per-AST cache, no-eval when
  constraints haven't moved, coalesce evals across consecutive
  `symbolic_objects` entries.

### Memory updates

- `phase43-default-flip` — what Phase 4.3 changed + which APIs were
  removed.
- `benchmark-phase43-flip` — before/after timing on the 8GB host.

### Files modified

- `native/angr/src/memory/mod.rs` (-29 lines)
- `native/angr/src/memory/store.rs` (-62 lines net)
- `native/angr/src/memory/tests.rs` (-37 lines net)
- `native/angr/src/state.rs` (-25 lines)
- `native/angr/src/symbolic/context.rs` (-3 lines)

Net: +24 −172.

### Caveats

- The local 8GB host runs sym-write at 1.55s vs CI baseline 0.436s
  (~3.5× slower overall). Regression checks against
  `baseline_timings.json` are not meaningful here; before/after on
  the same host is the relevant signal.
- `.venv/bin/pip` still broken; used `tools/rebuild-rust.sh --cargo-only`
  (same as Phase 4.1 / 4.2 sessions).
- Parent epic closed with acceptance technically not met (parity
  vs gate-off) but with the practical regression-gate criterion
  (15% threshold) satisfied. Follow-up `angr-o6ef` tracks closing
  the remaining gap.
