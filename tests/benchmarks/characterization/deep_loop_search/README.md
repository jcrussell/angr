# Deep-loop search-strategy characterisation (angr-xel4)

First slice of the angr-xel4 spike ("BFS vs DFS vs prioritized search
for grub OOM"). Grub itself is on the project's OOM avoid list, so this
slice substitutes a parameterised **grub-proxy** binary that exercises
the same exploration topology — a tight input-driven loop with N+1
depth-to-vulnerability — without grub's I/O / SimProc surface.

## Hypothesis

Grub's depth-14 vulnerability (13 backspaces + CR) is unreachable
under BFS at the current 4 GB RSS budget because BFS expands a wide
frontier of low-depth states. **DFS should reach the target depth at
lower RSS and lower step count.** Confirming this on the proxy
justifies switching the grub-class search strategy (the follow-on
implementation work is angr-smxp).

## Result

**Confirmed: DFS reaches deeper at lower memory than BFS for deep-loop
binaries.** On the synthetic grub-proxy with iteration cap N:

| N  | step budget | strategy | max depth | active at end | peak RSS | wall  |
|----|-------------|----------|-----------|---------------|----------|-------|
|  4 | 100         | bfs      | 11        | 4             | 302 MB   | 1.30s |
|  4 | 100         | dfs      | 12        | 4             | 302 MB   | 1.04s |
|  8 | 100         | bfs      |  **9**    | 65            | 429 MB   | 1.36s |
|  8 | 100         | dfs      | **20**    | 17            | 331 MB   | 1.10s |
|  8 | 300         | bfs      | 12        | 187           | 967 MB   | 2.84s |
|  8 | 300         | dfs      | 20        | 15            | 605 MB   | 1.88s |
| 14 | 100         | bfs      |  **9**    | 65            | 429 MB   | 1.37s |
| 14 | 100         | dfs      | **32**    | 34            | 371 MB   | 1.22s |
| 14 | 300         | bfs      | 11        | 191           | 976 MB   | 2.84s |
| 14 | 300         | dfs      | **32**    | 31            | 645 MB   | 2.00s |

Key observations:

1. **BFS depth plateaus at 9-12** regardless of step budget or
   iteration cap. The wide low-depth frontier consumes the budget; the
   exploration never drills deep.
2. **DFS reaches the iteration cap and beyond** (depth 32 at N=14
   reflects the inlined loop body's basic-block count, not 32
   independent iterations — but it is clearly past depth 14).
3. **DFS uses less memory than BFS at the same step budget**: at
   N=14/300 steps, BFS holds 191 active states (1 GB RSS); DFS holds
   31 (645 MB).
4. **The grub-target depth of 14 is unreachable under BFS** in this
   step budget — BFS at N=14/300 caps at depth 11. DFS reaches depth
   32 at the same budget.

This confirms the 2026-05-17 root-cause hypothesis on the angr-34w.12
bug bead: native uniqueness + BFS terminates before depth 14.

## Implication for angr-smxp

The follow-on implementation bead (angr-smxp) should adopt DFS as the
default for grub-class binaries (deep input loops with vulnerability
depth >= 10). A simple heuristic: switch to DFS when the binary's
detected input-loop bound exceeds the BFS plateau depth (~10 blocks).
A more nuanced approach can be deferred until DFS lands and we see
which binaries it does NOT close.

## Reproducing

```bash
# Single (engine, strategy, N) point
.venv/bin/python tests/benchmarks/characterization/deep_loop_search/run_one.py \
    --engine rust --strategy dfs --n 14 --max-steps 300 \
    --build-dir /tmp/deep_loop_build

# Full sweep (Rust only, mem-safe defaults)
.venv/bin/python tests/benchmarks/characterization/deep_loop_search/sweep.py \
    --n-list 4,8,14 --max-steps-list 100,300 \
    --engines rust --strategies bfs,dfs \
    --mem-limit-mb 2048 --timeout-s 90
```

`sweep.py` invokes `run_one.py` in a subprocess for each (engine,
strategy, N, max_steps) cell so peak RSS via `getrusage(RUSAGE_SELF)`
is per-config and the `RLIMIT_AS` confines OOMs to a single sub-run.

## Files

- `deep_loop_template.c` — C template; `run_one.py` substitutes
  `__N__` (iteration cap).
- `run_one.py` — compiles the binary on demand, runs one
  (engine, strategy, N, max-steps) cell, prints metrics line.
- `sweep.py` — drives `run_one.py` across the chosen grid;
  writes `results.csv`.
- `results.csv` — 2026-06-06 Rust-only sweep on the ralph host.

## Limitations / follow-ups

1. **Proxy ≠ grub.** The deep-loop binary captures the input-loop
   topology but not grub's libc surface (puts/getline/strlen).
   Quantitative numbers won't match grub exactly. The qualitative
   conclusion (DFS dominates BFS for deep-input binaries) should.
2. **No prioritized-search strategy yet.** The bd description asked
   for BFS + DFS + prioritized. Prioritized = depth-greedy state
   selection, which the Rust manager doesn't expose today. Deferring
   that comparison until a prioritized-search hook lands (or until
   the smxp implementation defines what "prioritized" means).
3. **Python engine not measured.** Rust is the focus; the proxy under
   Python at N>=8 will likely OOM the same way (BFS frontier
   blow-up). Worth a follow-up if angr-smxp wants Python-parity
   data.
4. **Grub itself not measured in this slice.** Requires raising the
   per-subprocess `RLIMIT_AS` past 4 GB or running on a beefier
   host. The harness here generalises: drop in a real grub project
   factory in place of the synthetic one and the same depth/RSS
   metrics fall out.
