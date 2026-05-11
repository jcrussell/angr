# Rust Engine — Known Slower Benchmarks

A handful of benchmarks tracked in `tests/benchmarks/baseline_timings.json`
run slower under the Rust engine than under pure Python. This file documents
the root cause for each one so future investigators do not re-derive the
same findings, and so the maintainer can decide whether the gap is worth
closing.

Speedup column is `python_time / rust_time` from the live baseline file.
Numbers refresh whenever `baseline_timings.json` is regenerated; treat the
values here as snapshots, not invariants.

## mma_howtouse — 0.65x (Py 4.25s / Rust 6.51s)

**Workload:** `Callable` FFI demo (`angr-examples/mma_howtouse/solve.py`)
invokes `howtouse(i)` 45 times in a loop. Each `callable(i)` constructs a
fresh `RustExplorationManager` via the run_single.py monkey-patch on
`AngrObjectFactory.simulation_manager`. The exploration itself is short
(concrete-only, ~45 short call_state runs).

**Root cause — thread-local AST cache lookup overhead.** angr's claripy
backend caches AST simplification results in thread-local hashmaps. The
caches accumulate entries across all 45 manager instances (they outlive
any single manager) so per-lookup cost grows roughly with total AST count.
Manually calling `claripy.clear_all_caches()` between `callable()`
invocations cuts wall time from 6.62s → 5.07s (a ~23% reduction)
without affecting correctness — but is not a default because it would
penalize the common single-long-exploration case.

**Memory:** the original 1606MB peak was fixed independently in commit
`342df4a7f` (2026-05-02), bringing peak to ~285MB. The current 0.65x
slowdown is purely CPU time.

**Why not chase a Rust fix?** The slow path is in the Python-side
hashmap (per-`callable()` AST equality probes), not in the Rust engine.
A real fix would need either (a) a per-manager AST cache scope so cached
entries do not leak across manager lifetimes, or (b) opt-in
`clear_all_caches()` plumbed through Callable. Both are non-trivial and
the benchmark itself is a degenerate stress test (45 isolated invocations)
that does not reflect typical symbolic-execution workloads.

**Relevant memories:** `mma-howtouse-leak-source`,
`mma-howtouse-cache-clear-speedup`, `benchmark-mma-howtouse-leak-fix`.

## ekopartyctf2016_sokohashv2 — 0.36x (Py 5.83s / Rust 16.0s baseline; ~9.5s typical)

**Workload:** Sokoban-style CTF binary that uses x87 transcendentals
(`fyl2x`, `fscale`, `f2xm1`) inside its hash routine. The baseline 16.0s
value is set at slow-mode to absorb known bimodal variance — actual fast
runs land near 9.5s (~21% improvement since commit a571d4634 from
FxHash adoption, freeze-local-assertions, and Arc-wrap fork fields).

**Root cause — two separate factors:**

1. **x87 transcendental fallback to Python.** `fyl2x`, `fscale`, and
   `f2xm1` are not implemented natively in `vex/ops.rs`. Previously these
   silently returned 0 (see memory `avoid-silent-zero-raw-fallback`,
   fixed in angr-n28w / commit `ef020d101`). Now they correctly route
   to Python claripy operations, which means each transcendental crosses
   the FFI boundary. The benchmark hits these on every hash iteration.

2. **Bimodal Z3 nondeterminism.** The solver picks one of two model
   shapes per run, producing either ~9.5s or ~15.4s wall time.
   `rust_only=True` is set in `run_regression.py` because the same
   nondeterminism causes output divergence between Rust and Python.
   The 0.36x baseline reflects the slow mode; the typical 0.58x figure
   reflects the fast mode.

**Memory:** Peak memory is not an issue here; the slowdown is pure CPU
in the x87 fallbacks plus solver nondeterminism.

**Why not chase a Rust fix?** Implementing x87 transcendentals natively
in Rust would close part of the gap but the binary is the only known
benchmark exercising them — high implementation cost for a niche win.
Z3 nondeterminism is structural; nothing in the engine controls it.

**Relevant memories:** `avoid-silent-zero-raw-fallback`,
`invariant-bimodal-variance-benchmarks`, `benchmark-perf-wins-2026-05-09`.

## Other benchmarks below 1.0x

These are listed in the CLAUDE.md performance table but were not the focus
of angr-ed7j. Captured here for completeness so future drift can be tracked.

| Benchmark | Speedup | Root cause |
|-----------|---------|------------|
| `google2016_unbreakable_1` | 0.46x | Bimodal Z3 nondeterminism (~1.6–3.3s Rust vs 1.6s Py). See `invariant-sla-bimodal-unbreakable1` + `benchmark-google2016-unbreakable-1-regression`. Regressed from a previous 3.3x peak; cause not yet isolated. |
| `hackcon2016_angry-reverser` | 0.87x | Close to parity, no formal investigation yet. |
| `securityfest_fairlight` | 0.98x | At parity in fast mode; bimodal (7.8s OR 15s baseline). See `invariant-bimodal-variance-benchmarks`. |
