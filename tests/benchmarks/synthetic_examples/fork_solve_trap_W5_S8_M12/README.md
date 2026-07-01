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
box, pre-angr-nkoct-fix HEAD):

| workers | wall              | found |
|---------|-------------------|-------|
| 1       | ~34 s             | 32    |
| 2       | TIMEOUT (>180 s)  | 32    |
| 4       | TIMEOUT (>180 s)  | 32    |

`workers>1` is *catastrophically* slower — a **super-linear** callback-boundary
pathology, not a linear "migration overhead". Characterized on a trivial-solve
variant (`--s 2 --m 8`, so Z3 cost is negligible and the callback/migration
machinery is isolated):

- `w1` completes in **~1.7 s** (181 callbacks, found=32).
- `w4` wall **explodes** as the trap count `T` rises: `T=1` 2.5 s -> `T=2` 3.4 s
  -> `T=4` **TIMEOUT (>90 s)** — a >26x cliff for a ~2x rise in callbacks.
- Across that cliff `parallel_reattaches` stays ~linear (~1 per bounce: 43 at
  T=1, 75 at T=2) and `gil_work_time_ns` is sub-millisecond. So the blow-up is
  **neither reattach volume nor GIL work** — it is the per-Python-callback
  `run()` re-entry (each of ~T*2^W bounces surfaces one callback, returns to
  Python, and re-enters the parallel coordinator). That is the cost angr-nkoct
  targets by keeping frontiers worker-local across the callback boundary.

**Measurement guidance.** Wall time is bimodal (`docs/advanced-topics/
rust_bimodal_variance.rst`) and often a timeout pre-fix, so gate on
(a) *completion* — the fix must make `w2`/`w4` finish well under `w1` — and
(b) the deterministic counters (`parallel_reattaches`,
`parallel_bounce_roundtrips`, `python_callback_count`), not an absolute
cross-worker wall threshold. Use the same worker count before vs after.

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
