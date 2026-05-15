## Session log: 2026-05-15 — Phase 4.2 flush_multi_cells coalescing (angr-mmdh.2)

### Closed task

**angr-mmdh.2** — Coalesce per-byte symbolic_objects on flush_multi_cells.
Phase 4.2 of the Multi-cell lazy memory work (parent `angr-mmdh`).
Landed at commit `278fc5223`.

### What landed

**Core change in `native/angr/src/memory/multi.rs`**

- `flush_multi_cells` now scans `multi_objects` in address order and
  greedily extends runs of consecutive byte addresses whose
  `MultiPayload` cond fingerprints match. For runs >= 2 bytes (capped
  at `COALESCE_MAX_RUN = 16`):
  - Per-candidate wider value = endian-aware concat of per-byte
    `MultiAlternative::value` BVs.
  - Wider default = endian-aware concat of page bytes.
  - Right-fold (cond, wider_value) into one ITE BV of width 8*N.
  - Insert at `symbolic_objects[run_start]`; populate
    `symbolic_spans` for interior bytes.
- Singletons and runs with any unmapped page fall through to the
  byte-identical pre-Phase-4.2 per-byte path.
- New helpers: `cond_fingerprint(&RustBV) -> u64` (Arc-ptr identity
  for Expression; value/id hash for the other variants);
  `payload_cond_fingerprint(&MultiPayload) -> Vec<u64>`;
  `build_wider_value(&[RustBV], Endness, &SymContext) -> RustBV`.

**Key invariant**

`install_multi_for_candidates` builds one cond per candidate then
clones it into each byte — so neighbouring bytes from the same
store-call share `Arc::as_ptr(operands)`. Future store paths that
emit Multi bytes must follow this "build cond once, clone into each
byte" pattern or coalescing will silently fall back to per-byte
(correctness preserved, perf only).

**Tests** (`memory/tests.rs`, 5 new):

- `test_phase42_flush_coalesces_le_multi_byte_run`: 4-byte LE store
  at 2 candidates -> 2 wider symbolic_objects (width 32) + interior
  spans; load round-trips correctly under each concretization.
- `test_phase42_flush_coalesces_be_multi_byte_run`: BE variant.
- `test_phase42_flush_singleton_no_coalesce`: 1-byte store -> 2
  per-byte entries unchanged.
- `test_phase42_flush_fingerprint_mismatch_breaks_run`: a later
  partial store appends an extra alt to byte 0x1002 -> run is broken
  there; expected output is 1 coalesced width-16 entry at 0x1000
  plus 2 singleton width-8 entries at 0x1002/0x1003.
- `test_phase42_flush_run_length_cap`: 24-byte store coalesces into
  one width-128 entry at 0x1000 + one width-64 entry at 0x1010.

### Validation

- `cargo test --lib memory::` — 71/71 pass.
- `python -m pytest tests/engines/test_rust_exploration.py` — 403/403
  pass.
- Regression-suite failures verified pre-existing: ran
  `run_regression.py --rust-only --skip-bimodal` both with stash
  popped and unpopped; same set of failures and near-identical times
  on both.

### Benchmarks (sym-write, --engine rust, avg of 4 runs)

| Config | Wall | z3_check | z3_site_eval_upto |
|---|---|---|---|
| Gate OFF (Phase 4.2 dormant) | 1.55s | 86ms  | 55ms  |
| Gate ON (Phase 4.1 baseline) | 1.78s | 158ms | 136ms |
| Gate ON (Phase 4.2)          | 1.70s | 147ms | 125ms |

Phase 4.2 closes ~35% of the prior gate-on/gate-off gap (1.78->1.70
of 1.78->1.62 baseline). Residual ~10% gap (1.70 vs 1.55) is in
state-export downstream evals: `z3_check_count` and
`z3_site_eval_upto_count` were unchanged (310/303 both Phase 4.1 and
Phase 4.2) — coalescing reduced per-eval cost, not eval count.

Gate-off remains unchanged at 1.55s (Phase 4.2 code path is dormant
when `use_multi_cell_stores=false`, which is still the default).

### Memory updates

- `phase42-bottleneck` — what Phase 4.2 does, the residual gap, the
  metric deltas.
- `invariant-coalesce-fingerprint` — Arc-ptr identity contract and
  the 16-byte cap rationale.
- `benchmark-phase42-symwrite` — before/after numbers and which
  counts moved vs which stayed flat.

### Files modified

- `native/angr/src/memory/multi.rs` (+218 -40 net)
- `native/angr/src/memory/tests.rs` (+211 -0)

### Caveats / follow-ups

- The `use_multi_cell_stores` default stays `false`. Flipping it
  (Phase 4.3, `angr-mmdh.3`) is the next bead but the ~10% residual
  on sym-write may still gate it; downstream eval reductions
  (state-export eval coalescing, or Rust-side
  `_get_state_symbolic_z3_asts` early-out) are the next likely lever.
- `z3_check_count` did NOT drop after coalescing — interesting; the
  state-export pipeline appears to issue a similar number of evals
  regardless of `symbolic_objects` entry count. Worth profiling
  with `--rust-profile` and a Python-side trace before designing
  Phase 4.3.
- Run cap of 16 bytes is conservative; `RustBV::concat_into`
  silently truncates widths > 120+8 via its u128 fast path. Lifting
  the cap requires auditing `concat_into` for wider-than-128
  safety.
- `.venv/bin/pip` still missing on this machine; used
  `tools/rebuild-rust.sh --cargo-only`. Same pattern as recorded in
  `env-venv-corruption` / `venv-rebuild-cargo-direct-copy` memories.

### Quick smoke verification (gate-off, post-change)

- ais3_crackme: 2.00s (matches pre-existing regression-baseline
  failure list)
- defcamp_r100: 0.27s (matches baseline)
