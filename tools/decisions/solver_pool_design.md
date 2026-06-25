# Design spike: per-fork Z3 solver-pool redesign — go/no-go

- **Bead:** angr-mawkv (spike) — parent angr-ovqja (perf epic)
- **Consolidates:** angr-2yyao (H3 prefix-share, DEFERRED) and angr-ovqja.5
  (qfbv_smart push/pop, DEFERRED/NOGO)
- **Status:** COMPLETE — **NO-GO on a new pool; the recommended redesign already
  exists** as `SharedLineageSolver`. See "Decision" below.
- **Date:** 2026-06-25 (iter48)

## 1. Why this spike exists

The `angr-2yyao` iter9 feasibility analysis (and bd memory
`per-fork-solver-share-infeasible`) concluded that the naive
"share the asserted shared-prefix solver across sibling forks via push/pop"
idea is **semantically infeasible** for angr's stash model, and closed with:

> "A genuinely viable design is a solver POOL with a single active base +
> swap-in/swap-out + cache-invalidation tracking — a much larger redesign than
> this task scopes... Recommend DEFER pending a dedicated design spike; not
> drainable as written."

That dedicated spike was never beaded. This document is it.

## 2. The cost being attacked

`RustSolverContext::solver()` (`native/angr/src/symbolic/context.rs`, `fn solver`)
lazily builds a fresh `z3::Solver` on first post-fork access and re-asserts the
ENTIRE assertion vector — the `Arc`-shared prefix (`z3_assertions_shared`) then
the per-context delta (`local_constraints.z3_assertions`):

```rust
let new_solver = build_solver(self.timeout_ms.load(Ordering::SeqCst));
let shared = Arc::clone(&self.z3_assertions_shared.lock());
for constraint in shared.iter() { new_solver.assert(constraint); }   // <- prefix re-assert
let local = self.local_constraints.lock();
for constraint in local.z3_assertions.iter() { new_solver.assert(constraint); }
```

Counted as `Z3_MATERIALIZE_TIME_NS` (build_solver + the assert loop, **not**
`check()`). Evidence from `ovqja.1` attribution:

| Bench | materialize_time | forks | per-fork |
|-------|------------------|-------|----------|
| ekopartyctf2016_rev250 | 386 ms | 124 | ~3.1 ms |
| google2016_unbreakable_1 | 254 ms | 102 | ~2.5 ms |

Sibling forks share an identical prefix yet each pays a full re-assert +
Z3-internal preprocessing on that prefix. The lever 2yyao wanted: assert the
shared prefix once and only pay the local delta per fork.

## 3. Why naive push/pop sharing is infeasible (recap, do not retry)

A `z3::Solver` push/pop is **one sequential stack on one solver object**.
Sibling forks are alive **simultaneously** in the stash, each with its own local
delta and its own cached `solver` / `sat_cache` / `model_cache`. Sharing one base
via push/pop would require `pop(prev_delta)+push(this_delta)+re-assert` on EVERY
switch between live siblings, plus `sat_cache`/`model_cache` invalidation per
switch — trading the ~3 ms/fork prefix re-assert for per-QUERY churn, a net loss.
Z3 offers no cheap same-context solver clone (`translate` is cross-context only and
re-preprocesses). Full rationale: bd memory `per-fork-solver-share-infeasible`.

## 4. KEY FINDING — the pool already exists: `SharedLineageSolver`

The "solver POOL with a single active base + swap-in/swap-out + cache-invalidation
tracking" that 2yyao called for is **already implemented and wired** — it is the
`SharedLineageSolver` (`native/angr/src/symbolic/lineage.rs`, research spikes
angr-hk7k → angr-v5a5 → angr-v5ht, wired into `SymContext::fork` by angr-3ms1).

It is exactly the recommended design, point for point:

| 2yyao's recommended pool | SharedLineageSolver realization |
|--------------------------|----------------------------------|
| single active base | one `z3::Solver` per lineage; base assertions at scope 0 via `assert_base` |
| swap-in / swap-out | `SharedLineageSolver::switch_to(target_path)` — longest-common-`FrameId`-prefix diff, `pop` divergent suffix, `push` target tail, cost `O(local_diff)` not `O(total)` |
| cache-invalidation tracking | `scope_path` / `scope_savepoints` per context; the `push`/`pop` ops on `SymContext` are documented to NOT auto-invalidate `sat_cache`/`model_cache` — the query path through `with_z3_solver` owns that frame (`solving_ops.rs`) |

How it is engaged today:
- **Opt-in** via `use_shared_lineage_solver` (default OFF — see
  `proxy-gate-audit-all-default-off`).
- Materialized at fork time (`snapshot_fork_ops.rs`, angr-3ms1 step 1c): the
  child is seeded with `frozen_shared` as the lineage base, and its first query
  runs `switch_to(empty)` to land on the parent's frame.
- **Auto-enabled** by the v5ht thrash sampler in `exploration/run_loop.rs` when a
  workload crosses the hot threshold (35%, `v5ht-threshold-justification`), with a
  dismantle path (`is_lineage_dismantled`) when the cache stops paying off.

So the prefix-share win 2yyao targeted is a **strict subset** of what
`SharedLineageSolver` already delivers when enabled: the base prefix is asserted
once per lineage and each sibling pays only its `O(local_diff)` tail on switch,
instead of a full prefix re-assert per fork.

## 5. What is actually left (the real gap)

Building a *new* pool would duplicate `SharedLineageSolver` — a DRY violation and
a re-tread of three closed research spikes. The only open question is **gating
policy**: should the shared-lineage path auto-engage on rev250 / unbreakable_1?

This is governed by existing, hard-won constraints — none of which this spike may
override blind:
- `avoid-dfs-coupling-for-shared-lineage` — do NOT default-on for `strategy=dfs`;
  **workload shape**, not search strategy, predicts the win.
- `v5ht-threshold-justification-2026-05-25` — the 35% hot threshold is calibrated
  on N=4 workloads; widen the dataset before retuning.
- `avoid-full-lineage-teardown` / `invariant-v5ht-dismantle-child-none` — the
  teardown variant is a net loss; child-None on dismantle is a correctness
  invariant (baby-re `chr()` repro).

Whether rev250/unbreakable_1 actually clear the workload-shape bar (high sibling
reconvergence on a shared prefix) is an **empirical** question requiring the same
profiling evidence the deferred T3 beads (angr-11djq.10/.13) are waiting on. It is
NOT answerable from static reading, and the two benches are Z3-heavy (one,
unbreakable_1, is bimodal — memory-flagged AVOID in the loop).

## 6. Decision

**NO-GO on a new solver pool.** The recommended architecture already exists as
`SharedLineageSolver`; reimplementing it is duplication. **2yyao and ovqja.5 are
correctly DEFERRED and should stay deferred** — their win is either already
captured by the existing opt-in path (2yyao) or faithfulness-infeasible /
opt-in-only (ovqja.5, per its iter12 NOGO comment).

The productive follow-up is **not** a pool but a *gating-policy* task: measure
whether enabling `use_shared_lineage_solver` on rev250 / unbreakable_1 nets a
materialize-time win without regressing the fast-tier gate, and if so feed those
workloads into the v5ht threshold dataset. That task is **blocked on the same
profiling-evidence gate** as the T3 research beads, so it is filed-deferred, not
drainable now.

### Recommended bead actions
1. Update bd memory `per-fork-solver-share-infeasible` to record that the
   recommended pool == `SharedLineageSolver` (already built) — done this iter.
2. Close angr-mawkv with this NO-GO.
3. Leave angr-2yyao / angr-ovqja.5 DEFERRED (unchanged).
4. (Optional, human) file a gating-policy bead: "evaluate use_shared_lineage_solver
   auto-enable on rev250/unbreakable_1" — deferred behind the T3 profiling gate.

## 7. Cross-references

- Source: `native/angr/src/symbolic/context.rs` (`fn solver`),
  `native/angr/src/symbolic/lineage.rs` (`SharedLineageSolver::switch_to`,
  `assert_base`), `native/angr/src/symbolic/snapshot_fork_ops.rs` (fork-time
  lineage minting), `native/angr/src/symbolic/solving_ops.rs` (`with_z3_solver`).
- Memories: `per-fork-solver-share-infeasible`, `invariant-hk7k-design-options`,
  `avoid-dfs-coupling-for-shared-lineage`, `v5ht-threshold-justification-2026-05-25`,
  `avoid-full-lineage-teardown`, `invariant-v5ht-dismantle-child-none`,
  `invariant-bare-z3-push-depth`, `proxy-gate-audit-all-default-off`.
- Sibling decision docs: `tools/decisions/native_subcall_dispatcher_design.md`.
