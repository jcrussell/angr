## Session log: 2026-05-14 (Phase 0 instrumentation)

### Task: angr-0nme — Phase 0: mem_ite_depth_max counter — CLOSED

Added two global atomics (`mem_ite_depth_max`, `mem_ite_depth_total`)
to `native/angr/src/symbolic/context.rs`, plumbed through the existing
`get_solver_stats()` / `reset_solver_stats()` API. Wired
`record_mem_ite_depth(depth)` into the three eager symbolic-store
sites that produce ITE chains:

- `store_strided` (memory/store.rs) — depth = `count`
- `store_conditional_multiple` (memory/store.rs) — depth = `addrs.len()`
- `store_symbolic` Multiple branch (memory/store.rs) — depth = `addrs.len()`

`run_single.py` now prints any non-zero `mem_ite_*` keys under a new
"symbolic memory ite-depth:" section. Baseline numbers captured:

| Workload     | mem_ite_depth_max | mem_ite_depth_total |
|--------------|------------------:|--------------------:|
| sym-write    | 2                 | 16                  |
| strcpy_find  | 0                 | 0                   |
| fauxware     | 0                 | 0                   |

Two new Rust unit tests in memory/tests.rs cover the wiring and the
helper. Both use delta-based assertions — the atomics are
process-global and cargo test runs in parallel.

### Files modified

- native/angr/src/symbolic/context.rs (counters + record_mem_ite_depth + reset/get)
- native/angr/src/symbolic/mod.rs (re-export)
- native/angr/src/memory/store.rs (3 call sites)
- native/angr/src/memory/tests.rs (2 new tests)
- tests/benchmarks/run_single.py (print mem_ite_* keys)

### Validation

- `cargo check --release` clean
- `cargo test --lib memory::` 36/36 pass
- `pytest tests/engines/test_rust_exploration.py` 396/396 pass
- Counters confirmed reachable via Python `_REM.get_solver_stats()`

### Commit

0b959b98c feat(rust-symex): mem_ite_depth_max counter for lazy-memory baseline (angr-0nme)

### What this unblocks

- angr-czph (lazy LOAD, Phase 1) — Multi-cell inserts must also call
  `record_mem_ite_depth()` so before/after comparison is direct. The
  bd memory `invariant-mem-ite-depth-counter` captures this contract.
- angr-qh5u (lazy STORE, Phase 2) — same requirement.

### Findings worth remembering

- Eager ITE chain depths from current sym-write are surprisingly
  shallow (max 2, total 16) — the workload's slowdown vs Python is
  likely driven more by sheer NUMBER of stores than per-chain depth.
  Phase 1 may need an additional metric (e.g. count of Multi
  insertions, or per-cell average) before claiming improvement.
- `strcpy_find` triggers zero eager multi-stores under Rust — its
  2.3× speedup is already captured by other paths.
- Venv's pip is currently broken (`ImportError: RequirementInformation
  from pip._vendor.resolvelib.structs` — pip 24.0 bytecode cache
  mismatch). Use `tools/rebuild-rust.sh --cargo-only` to rebuild the
  .so directly. CLAUDE.md's "venv-rebuild-cargo-direct-copy" warned
  about this exact path.
