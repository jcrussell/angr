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

The **deterministic** shape (isolated on a trivial-solve variant `--s 2 --m 8`,
so Z3 cost is negligible and the callback/migration machinery is what's
measured), workers=4, sweeping the trap count `T` (each ~1 wave per trap level):

| T (bounce levels) | w1     | w4     |
|-------------------|--------|--------|
| 1                 | —      | 2.36 s |
| 2                 | —      | 3.26 s |
| 3                 | —      | 4.25 s |
| 4                 | ~1.7 s | ~5.0 s (4.97/5.05/5.20, 3 repeats) |

So the parallel overhead is **LINEAR** in the callback count — ~1 s per trap
level at w4 — a roughly **~3x** slowdown vs w1, `peak_mem` flat (~621 MB),
`parallel_reattaches` ~1 per bounce, `gil_work_time_ns` sub-millisecond. The
cost is the per-wave frontier re-migration (each callback level re-seeds the
~40-wide frontier: detach+reattach serde), NOT a super-linear pathology.

**Do NOT trust single-sample wall times.** On the full-solve `S8/M12` variant
the same run swings from ~45 s to a >180 s "TIMEOUT" run-to-run: that is the
documented **bimodal-Z3 variance** (`docs/advanced-topics/rust_bimodal_variance.rst`),
not a deterministic cliff. (An earlier revision of this file mis-read those
bimodal timeouts as a ">26x super-linear cliff" — corrected: it is linear.) Gate
on (a) *completion* under the same worker count before vs after the fix and
(b) the deterministic counters (`parallel_reattaches`,
`parallel_bounce_roundtrips`), never an absolute cross-worker wall threshold.

**Scope note.** Because *every* leaf of this bench bounces to Python at every
trap level, each state must reattach in the coordinator's Z3 context to run its
callback — so its per-bounce migration is largely *inherent* and the
persistent-frontier fix helps it only modestly. The fix's real target is
**partial-bounce** workloads, where most states continue worker-locally (zero
serde) and only a few bounce; there the eliminated per-wave re-seed is the whole
cost. This bench is the migration *stress* case, not the ideal *win* case.

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
