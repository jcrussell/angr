# CoW fork scaling characterisation (angr-uf0g)

A parameterised fork-tree binary that creates `2^N` independent symbolic
branches. Sweeping `N` and recording wall time + peak RSS for each engine
lets us see how the Rust engine's `im::OrdMap`-backed state fork scales
against angr's Python `SimState` copy.

## Hypothesis

> Python deep-copies state on fork (hockey-stick memory);
> Rust uses `im::OrdMap` structural sharing (linear-ish).

## Result

**Both engines scale linearly in number of leaf states, not exponentially.**
Python does not exhibit a hockey-stick curve. The shared `SimMemory`
page-level CoW handles the fork cost on the Python side. The interesting
delta is in **per-state cost**:

| Engine | Per-state CPU (steady-state) | Per-state RSS (steady-state) |
|--------|------------------------------|-------------------------------|
| Rust   | ~10 ms/state                 | ~3.7 MB/state                 |
| Python | ~150 ms/state                | ~7.2 MB/state                 |

Rust is ~15× faster per state and uses ~2× less RAM per state. Both grow
roughly linearly with leaf-state count.

## Raw data

Captured 2026-06-03 on the ralph host (8 GB RAM, 4 GB `RLIMIT_AS` per
subprocess, gcc 12 `-O0 -no-pie`). Full CSV: `results.csv`.

| N  | 2^N | Rust wall (s) | Rust RSS (MB) | Python wall (s) | Python RSS (MB) | Speedup |
|----|-----|---------------|---------------|------------------|------------------|---------|
| 2  | 4   | 0.88          | 162           | 1.20             | 171              | 1.4×    |
| 3  | 8   | 0.64          | 175           | 1.76             | 199              | 2.7×    |
| 4  | 16  | 0.71          | 202           | 2.92             | 251              | 4.1×    |
| 5  | 32  | 0.84          | 255           | 5.34             | 356              | 6.4×    |
| 6  | 64  | 1.13          | 360           | 10.06            | 567              | 8.9×    |
| 7  | 128 | 1.69          | 572           | 19.78            | 989              | 11.7×   |
| 8  | 256 | 2.80          | 996           | 38.77            | 1835             | 13.8×   |
| 9  | 512 | 5.10          | 1848          | OOM (Z3)         | — (>2.7 GB)      | n/a     |

Python OOMs inside the Z3 SAT solver before exploration completes at
N=9 under a 4 GB cap. Rust completes the same workload using ~1.9 GB.

## Reproducing

```bash
# Single (engine, N) point
.venv/bin/python tests/benchmarks/characterization/cow_fork_scaling/run_one.py \
    --engine rust --n 6 --build-dir /tmp/fork_tree_build

# Full sweep, both engines, plot disabled (matplotlib not in venv)
.venv/bin/python tests/benchmarks/characterization/cow_fork_scaling/sweep.py \
    --n-min 2 --n-max 9 --no-plot
```

`sweep.py` invokes `run_one.py` in a subprocess for each (engine, N)
pair so peak RSS via `getrusage(RUSAGE_SELF)` is per-config and the
4 GB `RLIMIT_AS` confines OOMs to a single sub-run.

## Files

- `fork_tree_template.c` — C template; `run_one.py` substitutes the
  branch block based on `--n`.
- `run_one.py` — compiles the binary on demand, runs one (engine, N)
  configuration, prints `N=… engine=… wall_s=… rss_kb=… terminal=…`.
- `sweep.py` — drives `run_one.py` across an N range and both engines;
  writes `results.csv` (and optionally `scaling.png` if matplotlib is
  available).
- `results.csv` — sweep output from the 2026-06-03 run.

## Follow-ups

- **Bench candidate**: this same binary (N=6) is a useful synthetic
  regression bench — pure symbolic fork without any libc procedure
  noise. Worth adding to `baseline_timings.json` once the corpus
  expansion epic (angr-vx8p) lands.
- **N=9+ behaviour**: Z3 OOM on the Python side is independent of
  state-fork cost — it's the per-state constraint solver allocation.
  Worth a follow-on bead if we want to push the high end of this curve
  (raise `RLIMIT_AS` for that sub-sweep only).
- **Hypothesis disproved**: hockey-stick growth for Python isn't there.
  Update `core-goal-design-philosophy` framing if it still claims fork
  is the bottleneck — the real wins are constant-factor CPU per
  symbolic step (interpreter loop, less Python boundary work).
