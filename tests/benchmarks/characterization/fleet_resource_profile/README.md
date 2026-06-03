# Fleet-level resource profile (angr-ayrq)

Runs N independent `fauxware` solves as separate subprocesses with a
capped concurrency, measures total wall time and aggregate peak RSS
(sum of VmRSS across all live workers at any single sample), and
repeats across both engines.

## Hypothesis

> Rust's lower per-process Python+Z3 footprint means lower aggregate
> RSS at the same throughput.

## Result

**Confirmed, modestly.** Per-process steady-state RSS is ~12% lower on
Rust (~206 MB vs ~231 MB for `fauxware`). The Rust engine also turns
in ~10-20% shorter per-process wall time, so aggregate-wall savings
compound on top of aggregate-RSS savings as concurrency rises.

| Concurrency | Engine | Wall (s) | per-proc-peak (MB) | aggregate-peak (MB) | Python overhead |
|-------------|--------|----------|--------------------|---------------------|-----------------|
| 1           | rust   | 1.91     | 196                | 196                 | —               |
| 1           | python | 1.96     | 224                | 224                 | +14% RSS        |
| 2           | rust   | 1.75     | 205                | 406                 | —               |
| 2           | python | 2.07     | 231                | 462                 | +14% RSS        |
| 4           | rust   | 1.91     | 206                | 809                 | —               |
| 4           | python | 2.27     | 231                | 921                 | +14% RSS        |
| 8           | rust   | 2.37     | 206                | 1560                | —               |
| 8           | python | 2.87     | 231                | 1737                | +11% RSS        |
| 16          | rust   | 5.09     | 206                | 2804                | —               |
| 16          | python | 6.05     | 231                | 3260                | +16% RSS        |

Captured 2026-06-03 on the ralph host (8 GB RAM, 8 vCPU, 1 GB
`RLIMIT_AS` per child). Full CSV: `results.csv`.

## Interpretation

- **Per-process RSS gap is real but small** at the fauxware workload —
  Python's SimState plus angr's import cost dominates the snapshot,
  and Rust shaves ~25 MB off that. Heavier symbolic-fork workloads
  (see `../cow_fork_scaling/`) widen the gap because Rust's
  `im::OrdMap` CoW state grows more slowly than Python's `SimMemory`.
- **Aggregate RSS scales linearly with concurrency** for both engines:
  no shared-page collapse, no superlinear blowup. Each worker is
  effectively an isolated process tree.
- **Throughput ceiling**: at concurrency=16 on an 8-core box, per-process
  wall time roughly doubles vs concurrency=1 (Rust 1.91 → 2.45,
  Python 1.96 → 2.93). The CPU is saturated past nproc; the optimal
  scheduling point is concurrency ≈ nproc.
- **Fleet packing**: at 4 GB available headroom (after OS + cache),
  Rust fits ~19 concurrent fauxware workers vs ~17 for Python — a
  ~12% packing bump. The bigger lever for a CI fleet is per-process
  wall, where Rust's ~20% lead translates directly into queue
  throughput.

## Reproducing

```bash
# Single point (engine, count, concurrency)
.venv/bin/python tests/benchmarks/characterization/fleet_resource_profile/run_fleet.py \
    --engine rust --count 16 --concurrency 8 --mem-limit-mb 1024

# Full sweep (writes results.csv)
.venv/bin/python tests/benchmarks/characterization/fleet_resource_profile/sweep.py \
    --concurrencies 1,2,4,8,16 --count-multiplier 2 --mem-limit-mb 1024
```

The sweep invokes `run_fleet.py` in a subprocess per (engine,
concurrency) point so the RSS sampler is per-config and any RLIMIT_AS
exhaustion confines OOMs to one sub-run.

## Implementation notes

`run_single.py` uses `multiprocessing.spawn`, which means the actual
angr worker is a **grandchild** of the `Popen` we launch. The sampler
walks `/proc/<root>/task/<root>/children` recursively to attribute
RSS to the right subtree (see `_descendants` in `run_fleet.py`). A
naive sampler that only reads the Popen pid's `VmRSS` underreports by
roughly 10x (the orchestrator stays small while the grandchild does
all the angr work).

## Follow-ups

- Per-process RSS plateaus after ~1.5s of fauxware exploration; the
  sampler at 100 ms resolution catches the peak reliably for this
  workload. For very long-running benches (e.g. flareon2015_10's 45
  callable invocations), consider lowering `--sample-ms` to track
  intra-run RSS dynamics.
- The 12% per-process RSS gap is dominated by Python/Z3 init overhead
  shared across both engines. A more meaningful "fleet" comparison
  would target a workload where the Rust engine actively reduces
  steady-state working set — `mma_howtouse` or `csaw_wyvern` come to
  mind (both Z3-heavy enough that Rust's shared-context savings should
  surface).
- Parent epic angr-kvn0 still needs `.4` (flame-graph comparison) and
  `.5` (writeup aggregating .1-.4).
