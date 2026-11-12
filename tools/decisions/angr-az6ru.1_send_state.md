# Decision brief: a `RustSimState` that holds no Z3 objects and is `Send`

## Status

- **Bead:** angr-az6ru.1 (spike), parent angr-az6ru (engine direction).
- **State:** spike complete. **GO**, in the order under "Implementation
  sketch". Awaiting owner sign-off before angr-az6ru.2 starts.
- **Bounded by:** `solver_pool_design.md` (no shared solver pool, no
  hot-ratio valve). Nothing here reopens that.

## Context

Parallel exploration is built and correct but is not a wall-clock win
(`docs/advanced-topics/rust_parallel_design.rst`, "Where it landed"). A
state cannot move between workers: `RustSimState` is `!Send`, so a steal goes
through `StateMigrationPayload` serialization and an AST and solver rebuild on
the receiving thread. The spike asked three questions: what exactly makes the
state `!Send`, what a parked state's memory is made of, and what it would cost
to answer queries without a solver owned by the state.

## Findings

### 1. One field makes the state `!Send`, and most of what is behind it is cache

A compile probe (`fn p<T: Send>(_: &T)` applied to every field) fails on
exactly one `RustSimState` field: `solver: Rc<RefCell<SymContext>>`. Every
other field passes, including `Box<dyn Arch>`, `AddressConcretizer`,
`FileSystem` and the three `OverlayMap<HostAst>` overlays.

Inside `SymContext` the probe fails on these fields:

| Field | Role | Rebuildable from the constraint list? |
|-------|------|----------------------------------------|
| `solver` | the materialized `z3::Solver` | yes: `SymContext::solver` already rebuilds it lazily after `fork` |
| `z3_assertions_shared`, `local_constraints` (`z3_assertions`, `dedup_set`) | Z3 form of every constraint | yes for the assume class, which has a clarirs log beside it (`assumed`, `assumed_constraints_shared`) |
| `non_bv_assertions_shared`, `local_constraints.non_bv_assertions` | the residual class: raw, bv-eq and merge-guard constraints | **no**: Z3 is the only form |
| `constraint_trackers` | `(track_bool, constraint)` pairs for unsat cores | **no**: Z3 is the only form |
| `model_cache`, `alt_model_cache` | `Rc<z3::Model>` witnesses | yes: a cache, invalidated on every constraint add |
| `lineage`, `scope_path` | the opt-in `SharedLineageSolver` | yes: already dropped on migration |

So the work reduces to giving the residual class and the trackers a clarirs
form. The Python-origin residual sites already hold one: `solver.rs` and
`exploration/constraints.rs` reach `add_constraint_raw` through
`lower_claripy_bool` / `claripy_z3_ptr`, which lower a clarirs AST the caller
still has in hand. They use the raw sink only because `claripy_to_rustbv` does
not model the op (floating point and similar), not because the term is
Z3-only. `to_snapshot` shows the split is already real: it rebuilds the assume
class from clarirs and dumps only the residual as SMT-LIB2.

### 2. A parked state's memory is its Z3 solver

`tests/benchmarks/characterization/cow_fork_scaling/z3_share.py` runs the fork
tree without building any Python state and compares RSS growth with Z3's own
allocation estimate:

| Leaves | RSS per live state | Z3 per live state | Z3 share |
|--------|--------------------|-------------------|----------|
| 256 | 815 KB | 753 KB | 92% |
| 512 | 801 KB | 754 KB | 94% |
| 1024 | 795 KB | 754 KB | 95% |

Every live state has built one solver (`z3_materialize_count` equals the live
count), so a materialized `z3::Solver` costs about 750 KB however few
constraints it holds, and everything else the state owns is about 45 KB.
Turning on `use_shared_lineage_solver` does not change the peak on this
workload.

The per-state figure in that directory's README (3.7 MB) is from an older
engine; the slope today is about 0.8 MB.

### 3. A per-state solver is used once or twice before it is discarded

Single runs of `run_single.py <bench> --engine rust --counters-json`:

| Bench | Wall | Solvers built (time) | Checks (time) | Checks per solver |
|-------|------|----------------------|---------------|-------------------|
| `cow_fork_scaling` | 0.45 s | 256 (74 ms) | 263 (210 ms) | 1.0 |
| `fork_solve_pbounce_W6_S8_M12_B2` | 7.22 s | 128 (66 ms) | 128 (6649 ms) | 1.0 |
| `ekopartyctf2016_rev250` | 1.00 s | 90 (220 ms) | 155 (302 ms) | 1.7 |
| `cmu_binary_bomb_partial` | 0.36 s | 25 (8 ms) | 45 (4 ms) | 1.8 |
| `CADET_00001_partial` | 3.22 s | 281 (254 ms) | 554 (1433 ms) | 2.0 |
| `codegate_2017-angrybird` | 1.23 s | 424 (999 ms) | 1 (28 ms) | — |

The state-owned solver's benefit is incremental reuse across checks on the
same state. At one to two checks per solver there is little to reuse. If every
check built its own solver with no reuse at all, the extra builds would cost
roughly 160 ms on rev250 and 250 ms on CADET at the measured per-build cost;
a worker that keeps its most recent solver recovers the back-to-back case.

`codegate_2017-angrybird` runs under `LAZY_SOLVES` and builds 424 solvers to
make one check. Why constraint adds force a build there was not traced; if
the build is avoidable, most of that bench's wall time is.

## Options

**A. Z3-free state.** The clarirs constraint list becomes the record. Z3
terms, solvers and models become worker-owned caches. `RustSimState` is
`Send`; a steal is a move.

- Gains: migration without serialization; about 750 KB less per parked
  state; `fork` stops scheduling an assertion replay.
- Costs: every residual sink and the trackers need a clarirs form; unsat-core
  indices and bare `push`/`pop` have to be redefined against the list instead
  of Z3 frames; `dedup_set` moves from Z3 pointer identity to clarirs
  hash-cons identity.
- Not verifiable offline: the query-time cost on workloads with many checks
  per state, which none of the benches above has.

**B. Keep Z3 in the state, evict solvers.** Set a parked state's `solver`
back to `None` under memory pressure; it rebuilds on demand as it does after
`fork`.

- Gains: most of the memory win for a small change.
- Costs: the state stays `!Send`, so parallelism is where it is today.

**C. Status quo.**

## Recommendation

**A.** Finding 1 says the blocker is narrower than the `!Send` label
suggests, finding 2 says the thing being removed is nearly all of a state's
memory, and finding 3 says what it buys today is small. B's eviction is step
3 of A, so nothing is lost by taking A.

What would change this: a workload where one state answers many checks
between forks, where a retained incremental solver matters. Step 3 should
measure checks per solver on whatever real target Phase 2 picks.

## Implementation sketch

Each step lands on its own and keeps the gates green.

1. **Clarirs form for the residual class and trackers.** Carry the clarirs
   Bool alongside the Z3 term at the Python-origin sinks (`solver.rs`,
   `exploration/constraints.rs`, `exploration/state_api.rs`), and build merge
   guards as clarirs terms in `snapshot_fork_ops.rs`. Snapshots stop needing
   `residual_smtlib2`; keep reading it for old snapshots.
2. **Make the clarirs list the record.** The Z3 logs become a derived cache
   filled through `z3_lower::lower`, whose per-thread `LoweredCache` already
   makes re-lowering a live term cheap. Move `dedup_set` to clarirs identity.
3. **Move the solver and models to the worker.** A small worker-local cache
   keyed by constraint-list identity: reuse on the same list or an extension
   of it, rebuild otherwise. This is deliberately not `switch_to`-style scope
   surgery. Models are held worker-side or stored as concrete assignments.
4. **Replace `Rc<RefCell<SymContext>>`** with an owned context and add
   `assert_send::<RustSimState>()` beside the existing payload assertion in
   `state/migration.rs`. Retire the `arc_shared` suppression rationale.
5. **Lineage.** The opt-in `SharedLineageSolver` becomes worker-local, as it
   is already dropped on migration.

Acceptance numbers for angr-az6ru.2: `z3_share.py` per-state RSS falls toward
the non-Z3 remainder; `z3_materialize_count` no longer tracks the live-state
count; the fast-tier regression gate and `tests/engines/rust` stay green.

## Limits of this spike

- Timings are single runs on a shared box, not medians; the counts are
  deterministic.
- The extension measured was built before the two most recent engine commits.
- The cost of step 3 on query-heavy states is an estimate from per-build
  cost, not a prototype.
