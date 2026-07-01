# fork_solve_trap_W5_S8_M12 — migration-dominated parallel regression gate

A **reconstruction** of a lost synthetic benchmark that is the acceptance gate
for **angr-nkoct**. It exhibits a *migration-dominated* parallel regression:
fast-ish single-threaded, but currently **slower at `workers > 1`** because the
level-synchronous parallel wave loop re-migrates the whole active frontier
across every Python-callback / wave boundary. The angr-nkoct fix is expected to
later turn this bench into a speedup; until then it must reproduce the *shape*
of the regression.

## Mechanism

This is the `fork_solve_W6_S8` wide-AND-slow base (see that dir's README) with
one addition: a **Python-bounce trap**.

- **Width (`W=5`)** — `W` independent `if (b[i] != 0)` branches over a fully
  symbolic 32-byte stdin fork into exactly `2^W = 32` leaves that BFS keeps
  active concurrently. W=5 is the *exact-drain* value: `num_find=32` collects
  every leaf, so `found == 32` at every worker count.
- **Slow (`S=8`, `M=12`)** — `S` rounds of nonlinear mixing plus a closing
  `if ((acc & MASK) == PATTERN) reach_target();` make each leaf's find-address
  `satisfiable()` a real but bounded Z3 solve.
- **Trap (`T=4`)** — `T` sequential `acc = trap_point(acc);` call sites placed
  **after** the mixing region, so all 32 leaves reach every one. `solve.py`
  hooks `trap_point` with an **identity `SimProcedure`** (`run(self, x): return
  x`). Returning the symbolic accumulator unchanged keeps the find-gate solve
  exactly as hard as the base — the trap adds *migration* cost, not solver cost.

Why this reproduces the regression (verified root cause): a parallel wave runs
its subtree to quiescence worker-locally (normal successors stay in the
worker's local queue — zero serde). The **only** thing that repopulates
`STASH_ACTIVE` and forces another full-frontier drain + re-migrate is a state
hitting a Python SimProcedure/hook — a *bounce*. Each `trap_point` call bounces
all 32 leaves to Python (`Hook`/`SimProcedurePython` -> `NeedCallback`), forcing
`run()` re-entry. At `workers > 1` each re-drain detaches + reattaches the whole
~32-wide frontier via a full Z3-AST serde round-trip (`StateMigrationPayload`).
`T` sequential trap call sites therefore drive ~`T` full-frontier
re-migrations, so wall time grows with worker count instead of shrinking.

The base bench alone (no trap) does **not** reproduce this: its frontier runs to
quiescence worker-locally with no bounce, so `workers>1` is non-monotonic
(~1.6x max), not the required monotonic blowup.

## Measured shape (Rust engine, this box)

`num_find=32`, `.venv/bin/python tests/benchmarks/run_single.py
fork_solve_trap_W5_S8_M12 --engine rust`, `RUST_PARALLEL_WORKERS` swept (this
box, pre-angr-nkoct-fix HEAD, single sample each):

| workers | wall (s)          | found | peak RSS |
|---------|-------------------|-------|----------|
| 1       | ~34               | 32    | ~0.7 GB  |
| 2       | TIMEOUT (>180)    | 32    | < 3 GB   |
| 4       | ~45               | 32    | ~0.9 GB  |

Both `workers=2` and `workers=4` are slower than `workers=1` — the
migration-dominated regression angr-nkoct must cure. Two caveats:

- **Non-monotonic / bimodal.** `workers=2` is *dramatically* worse than
  `workers=4` (2 workers serialize the offload/re-migration churn; 4 absorb it
  faster) and single-sample wall times swing with the documented bimodal-Z3
  variance (see `docs/advanced-topics/rust_bimodal_variance.rst`). Do NOT gate
  on an absolute cross-worker threshold.
- **Use same-config before/after.** The robust acceptance measurement is the
  SAME worker count before vs after the fix (e.g. `workers=4`: ~45s -> target
  well under `w1/2`), which cancels the bimodal variance. The fix's counter
  proof (`parallel_reattaches` independent of bounce-level count) is the
  primary GO signal; wall time is the secondary, variance-prone one.

## Files

- `fork_solve_trap_template.c` — C template: base `__BRANCHES__` / `__MIX__` /
  `__GATE__` markers **plus** a non-inlined `trap_point()` and a `__TRAP__`
  marker after the mixing region.
- `build_trap.py` — substitutes `W`/`S`/`M`/`T`, compiles `gcc -O0 -no-pie`
  x86-64. Writes `fork_solve_trap_W<W>_S<S>_M<M>[.c]` (the trap count `T` is
  baked into the `.c`, kept out of the file name).
- `fork_solve_trap_W5_S8_M12.c` — the exact substituted source of the ELF.
- `fork_solve_trap_W5_S8_M12` — prebuilt ELF (W=5 -> 32 leaves, S=8, M=12,
  T=4).
- `solve.py` — pre-stocks 32 symbolic stdin bytes, hooks `trap_point` with the
  identity `SimProcedure`, resolves `main` / `reach_target` by symbol, runs
  `sm.explore(find=reach_target, num_find=32)`, and `test()` asserts a solution
  is found. Built through `proj.factory.simulation_manager` so
  `run_single.py --engine rust` swaps in `RustExplorationManager` transparently.

## Rebuilding / retuning

```bash
python build_trap.py --w 5 --s 8 --m 12 --t 4      # the checked-in variant
```

Tuning knobs (`FORK_SOLVE_VARIANT`, `FORK_SOLVE_NUM_FIND`) are read from the
environment by `solve.py`. The regression strength scales with `T` (more trap
call sites = more re-migration levels). Keep `W=5` so `found` stays the
exact-drain 32; raising `T` raises both the single-thread wall and the
worker>1 blowup. Keep peak RSS < 3 GB (the `run_single.py` subprocess enforces
a 4 GB `RLIMIT_AS`).
