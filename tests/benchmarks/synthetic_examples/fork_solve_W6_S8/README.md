# fork_solve_W6_S8 — wide-AND-slow synthetic (parallel-symex GO gate)

A parameterised fork/solve binary purpose-built to be the one thing the rest of
the benchmark corpus lacks: a workload that is **wide** (many independent active
states at the same time) **and** slow (each state carries a real solver check).
`run_width_audit.py` computes `frac_ge3` (fraction of dispatched steps with >= 3
concurrent states) and `ms/task` (wall / dispatched tasks); the GO shape for the
in-process parallel wave loop is `frac_ge3 >= 0.20 AND ms/task >= 50` under
< 3 GB RSS. The checked-in W6/S8/M20 variant clears it: `frac_ge3 ~= 0.80`,
`ms/task` in the thousands, peak RSS ~385 MB.

## Shape

Knobs are baked into the C source (`build.py` substitutes them into
`fork_solve_template.c`, exactly like `cow_fork_scaling`'s `__N__`):

- **`W` (width)** — `W` independent `if (b[i] != 0) s += (1<<i);` branches over a
  fully symbolic 32-byte stdin. A symbolic input reaches all `2^W` leaves, and
  BFS keeps them active concurrently → sustained frontier width.
- **`S` (mixing depth)** — after the branch region every leaf runs `S` rounds of
  nonlinear mixing (`acc = (acc*1103515245u + 12345u) ^ (acc>>3); acc += b[k&31];`),
  unrolled straight-line so mixing costs few dispatched steps but builds a deep,
  hash-like accumulator.
- **`M` (match-mask width, `build.py --m`)** — the closing
  `if ((acc & MASK) == PATTERN) reach_target();` requires the low `M` bits of the
  mixed accumulator to equal a fixed pattern. This is the per-state solve-cost
  knob: wider `M` (and deeper `S`) makes the find-address `satisfiable()` harder,
  but the match stays *satisfiable*, so cost is bounded.

`solve.py` runs `sm.explore(find=reach_target, num_find=1)`: BFS materializes the
whole `2^W` frontier before any leaf reaches the find address, so the first
find-address `satisfiable()` fires with the frontier already wide, then
exploration stops — bounding the (very expensive) slow work to the first wave so
the bench finishes in ~1 min under the audit timeout.

### Why a partial mask and not a full equality

The original design gated `reach_target` on a full-word `acc == CONST`. That
forces Z3 to prove **UNSAT** for the many leaves whose branch pattern cannot hit
an exact 32-bit constant — pathologically slow and unpredictable (W4/S4 = 16
leaves took **94 s**; W6 timed out). A partial `M`-bit match over a well-mixing
function is essentially always satisfiable, giving a real but *bounded* per-leaf
check. Memory is structurally bounded: at most `2^W` leaves, cost in constraint
*depth* not state count — so `W <= 6` stays well under the 4 GB cap.

## Files

- `fork_solve_template.c` — C template with `__BRANCHES__` / `__MIX__` / `__GATE__`
  markers.
- `build.py` — substitutes `W`/`S`/`M`, compiles `gcc -O0 -no-pie` x86-64
  (matches `cow_fork_scaling` flags), writes `fork_solve_W<W>_S<S>[.c]` (the mask
  width `M` is baked into the `.c`, kept out of the file name).
- `fork_solve_W6_S8.c` — the exact substituted source of the checked-in binary.
- `fork_solve_W6_S8` — prebuilt ELF (W=6 → 64 leaves, S=8 mixing rounds, M=20).
- `solve.py` — pre-stocks 32 symbolic stdin bytes, resolves `main` / find target
  `reach_target` by symbol, runs `sm.explore(find=reach_target, num_find=1)`, and
  `test()` asserts a solution is found (correctness gate). Built through
  `proj.factory.simulation_manager` so `run_single.py --engine rust` swaps in
  `RustExplorationManager` transparently.

## Rebuilding / retuning

```bash
python build.py --w 6 --s 8 --m 20     # the checked-in variant
python build.py --w 6 --s 12 --m 24    # harder per-state solve
```

If `frac_ge3` is too low, raise `W` (keep <= 6). If `ms/task` is too low (solves
too cheap), raise `M` (and/or `S`). If the bench times out or peak RSS approaches
3 GB, lower `M`/`S`/`W`. Tuning knobs `FORK_SOLVE_VARIANT` (which ELF to load) and
`FORK_SOLVE_NUM_FIND` are read from the environment by `solve.py`. After settling
on a variant, update `W`/`S` in `solve.py`, rename the directory (the bench name
must equal its directory under `synthetic_examples/`), and update the
`run_single.py` `EXAMPLE_CATALOG` entry and `baseline_timings.json`.
