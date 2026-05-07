# Loop session notes (2026-05-07, 117th loop session)

## Task: angr-teo2 — FxHash swap for remaining hot Arc<HashMap/HashSet>  ✓ CLOSED

### What changed (commit a37018771)

Swapped 3 Arc-shared maps in `native/angr/src/interpreter_cb/mod.rs`
plus the source HashMap in `native/angr/src/exploration/mod.rs:448`:

| Before | After |
| --- | --- |
| `hook_addrs: Arc<HashSet<u64>>` (line 512) | `Arc<FxHashSet<u64>>` |
| `simprocedure_registry: Arc<HashMap<u64, SimProcedureInfo>>` (592) | `Arc<FxHashMap<...>>` |
| `vex_opt_level_overrides: Arc<HashMap<u64, i32>>` (636) | `Arc<FxHashMap<u64, i32>>` |
| `RustExplorationManager.vex_opt_level_overrides: HashMap<u64, i32>` | `FxHashMap<u64, i32>` |

Each is looked up per-block during execution
(`execution.rs:32,134,178,358,394` for the registry/overrides;
`mod.rs:1175` for hook_addrs.contains). FxHasher's faster integer hash
helps every per-block lookup; Arc-share on fork is unaffected (already
cheap).

### Validation

- `cargo check --release`: clean.
- `cargo build --release`: clean.
- `261/261` pytest tests pass (`tests/engines/test_rust_exploration.py`).
- Benchmark regression: 11/12 (csgames2018 fails — pre-existing, see
  angr-4pkm).

### Memory update

`bd remember --key fxhash-interpreter-cb-arc-maps` records the swap
and lists remaining candidates (CLARIPY_AST_CACHE,
SymbolicIdentityRegistry.rust_id_to_py, RustSymbolTable.symbols, stash
maps).

## Side-effect: angr-4pkm filed

csgames2018 used to pass at 0.96s/292MB (per
project_benchmark_status.md, 2026-05-05) but now fails with `list index
out of range` on the rust engine. Verified to reproduce on HEAD prior
to angr-teo2 — pre-existing regression, NOT caused by this work.

Filed as P2 bug `angr-4pkm` with bisect candidates noted (commits
between 2026-05-05 and 2026-05-07: 7f769212d, fad930315, 4de453a9a,
0bb4389fc, 833761f1b).

## Status: complete
