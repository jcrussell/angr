## Session log: 2026-05-15 — Phase 4.1 wider-load collapse cache (angr-mmdh.1)

### Closed task

**angr-mmdh.1** — Wider-load collapse cache in `assemble_load_with_multi`.
Phase 4.1 of the Multi-cell lazy memory work (parent `angr-mmdh`).
Landed at commit `a1d9f5593`.

### What landed

**New types in `native/angr/src/memory/mod.rs`**

- `ByteFingerprint::{Multi { version, default_byte }, Concrete { byte }}`
  — per-byte snapshot used to detect whether a cached wider-load result
  is still valid. Plain Symbolic bytes are not cached (return None
  from `compute_wider_load_fingerprint` → bypass cache).
- `CachedWiderLoad { byte_fingerprints, total_ite_depth, bv }` — cache
  entry. `total_ite_depth` replays `record_mem_ite_depth` on hits to
  preserve `invariant-mem-ite-depth-counter`.
- `WIDER_LOAD_CACHE_CAP = 1024` — soft cap; on insertion past cap we
  evict an arbitrary entry (`HashMap::keys().next()`).

**Cache state on `SymbolicMemory`**

- `multi_versions: FxHashMap<u64, u64>` — per-byte monotonic version
  counter, bumped by `set_multi_alternatives`, `clear_multi_at` (only
  when a payload was present), and per byte inside `flush_multi_cells`.
- `wider_load_cache: RefCell<FxHashMap<(u64, u32), CachedWiderLoad>>`.
- `fork()` and `new()` updated; cache cloned on fork (refcounted BVs
  make the clone cheap).

**Load-path change in `memory/load.rs::assemble_load_with_multi`**

1. Compute fingerprint (returns `None` if any byte is plain Symbolic
   or any page is unmapped — falls through to existing build+error
   handling).
2. If fingerprint exists AND cache has an entry at `(addr, size)`
   whose fingerprints match, replay
   `record_mem_ite_depth(cached.total_ite_depth)` and return
   `cached.bv.clone()`.
3. Otherwise: existing per-byte build (tracking `total_ite_depth`
   along the way), then on success insert into cache.
4. `size==1` bypasses the cache (no concat to amortize).

**Tests** (`memory/tests.rs`, 5 new):

- `test_phase4_wider_load_cache_hit`: prime cache, second load reuses.
- `test_phase4_wider_load_cache_skips_size_one`: size==1 not cached.
- `test_phase4_wider_load_cache_invalidated_on_multi_install`: bumps
  version → fingerprint mismatch → rebuild (uses two independent
  addr vars so eval pins the new alt).
- `test_phase4_wider_load_cache_skips_symbolic_bytes`: plain Symbolic
  byte in range → not cached.
- `test_phase4_wider_load_cache_fork_independence`: parent cache
  survives child mutation.

### Validation

- `cargo test --lib memory::` — 66/66 pass.
- `python -m pytest tests/engines/test_rust_exploration.py` — 403/403
  pass.
- `tests/benchmarks/run_regression.py --rust-only --skip-bimodal`
  failures verified pre-existing (same failures with identical
  baseline_timings.json before and after the change).

### Benchmarks (sym-write, --engine rust)

| Config | Wall | load_stmt | z3_check | z3_site_eval_upto |
|---|---|---|---|---|
| Gate OFF | 1.62s avg | 9ms | 86ms | 55ms |
| Gate ON (Phase 4.1) | 1.78s avg | 10ms | 158ms | 136ms |

Phase 4.1's stated goal — "load-time portion no longer dominates the
gate-on regression vs gate-off (~1.55s baseline)" — is met. `load_stmt`
gap is now 1ms. The remaining wall-time gap is downstream Z3 work in
`z3_check` + `z3_site_eval_upto` driven by per-byte `symbolic_objects`
entries from `flush_multi_cells`. That is Phase 4.2's scope (`angr-mmdh.2`).

### Memory updates

- `phase41-bottleneck` — Phase 4.1 closes load-time portion; residual
  is in Z3 work (state-export concretization), Phase 4.2 territory.
- `invariant-multi-versions` — any future write path that mutates
  `multi_objects` must call `self.bump_multi_version(addr)` or the
  wider-load cache will serve stale BVs.
- `benchmark-phase41-symwrite` — before/after numbers.

### Files modified

- `native/angr/src/memory/mod.rs` (+150 -7)
- `native/angr/src/memory/load.rs` (+48 -2)
- `native/angr/src/memory/multi.rs` (+15 -3)
- `native/angr/src/memory/tests.rs` (+191 -0)

### Caveats / follow-ups

- The `use_multi_cell_stores` default stays `false`. Flipping it
  (Phase 4.3, `angr-mmdh.3`) is blocked on Phase 4.2.
- Wider-load cache benefit on the sym-write bench is small because
  the gate-on regression isn't bottlenecked on load-time anyway. The
  cache will likely show up more clearly on workloads with heavy
  repeated loads over the same Multi regions.
- Cache eviction is arbitrary (no LRU). At cap=1024 entries per state,
  this should not matter; if a future workload trips it, add LRU
  bookkeeping.
- `.venv/bin/pip` is missing on this machine (only `python` and
  `python3.12` symlinks). Used `tools/rebuild-rust.sh --cargo-only` as
  the build path — same pattern as the recorded `env-venv-corruption`
  / `venv-rebuild-cargo-direct-copy` memories.
