Rust engine: parallel exploration threading model — design comparison
=====================================================================

This document compares threading models for parallel exploration in
the Rust engine (``angr-59jk``, ``angr-59jk.1``). The empirical
translate-cost numbers quoted below come from a companion spike
(``angr-59jk.2``) whose reproducer was deleted with the translate path (angr-mx00p.15).

Audience: contributors working on the parallel stack, or judging how a
change interacts with it. The Option A / Option B comparison is kept as
**design rationale** — it records why the shipped architecture is shaped
the way it is, not what a contributor should configure.

Status: **recommendation shipped, opt-in.** Option A (shared-nothing
states, no Z3 context crossing a thread boundary) was chosen and built.
The work-stealing pool lives in ``exploration/scheduler.rs`` and its
``scheduler_pool.rs`` / ``scheduler_worker.rs`` /
``scheduler_transport.rs`` siblings; the driver loops in
``exploration/run_loop_wave.rs``, ``run_loop_steady.rs`` and
``run_loop_worker.rs``; the cross-worker transport in
``state/migration.rs``. Both parallel modes are off by default: the wave
loop is armed by the ``RustExplorationManager(parallel_workers=N)``
kwarg or the ``RUST_PARALLEL_WORKERS`` environment variable (which
overrides the kwarg), and the steady-state loop only by the separate
``RUST_PARALLEL_STEADY`` env flag. Found-set invariance of the parallel
path across worker counts is gated by
``tests/benchmarks/run_findall_gate.py`` in
``rust-nightly.yml::findall_worker_invariance`` (see the CI gates table
in :doc:`rust_build`).

For **user-facing** documentation of those flags, and of which
SimOptions / exploration techniques are safe under them, read
:doc:`rust_engine`; this file is the rationale, not the manual. The
`Where it landed`_ section below records what shipped and what each
increment found; the Option A/B analysis predates all of it.


Why this work
-------------

The Rust engine outperforms Python on most of the angr-examples
benchmarks (``tests/benchmarks/baseline_timings.json`` is the
authoritative list), but the single-threaded
ceiling is real: the slowest benches today are dominated by Z3 wall
time (``securityfest_fairlight``, ``ekopartyctf2016_sokohashv2``,
``hackcon2016_angry-reverser``), and the engine pins exactly one core.
A workload with N independent states could in principle run with N×
the throughput, capped only by the cost of distributing work and
collecting results.

The constraints are sharp:

#. **Z3 contexts are not thread-safe.** A ``z3::Context`` cannot be
   shared across threads without a mutex; the same is true of every
   AST node that lives in it.
#. **Claripy state lives in Python**, behind the GIL. Anything that
   round-trips through claripy serializes on the GIL.
#. **States carry a lot of AST.** A typical RustSimState mid-run has
   hundreds to low-thousands of symbolic register values, plus per-page
   symbolic byte overlays, plus the solver's constraint set.

Two viable threading models for spanning multiple cores exist. This
document compares them on complexity, throughput, memory, sensitivity
to bench shape, and migration path, using the
companion spike's empirical translate-cost data (``angr-59jk.2``).

A third path — ``Z3 parallel.enable=true``, which fan-outs *inside* a
single solver — is explicitly out of scope. It was tested
(``angr-gfay``) and proved correctness-breaking on several sampled
benchmarks (bd memory `avoid-z3-parallel-enable`).
The two options below are about **parallel state stepping**, not
parallel SAT.


Empirical input: Z3_translate cost
----------------------------------

The companion spike (``angr-59jk.2``) measured ``Z3_translate`` per AST node. The
cost is linear in node count with a small constant, so a whole exported
state of a few thousand nodes moves between contexts in single-digit
milliseconds. That is negligible against a task of tens of milliseconds
of exploration and prohibitive if paid per step, which is why every
design below migrates only at task boundaries.


Threading model at design time
------------------------------

.. note::

   This section describes the engine **as it was when the comparison
   below was written**, and is what the two options were weighed
   against. It is still the shape of the *default* configuration —
   neither parallel mode arms itself — but the pieces listed here now
   sit alongside the shipped parallel stack named in the Status block
   above.

Single-threaded. The relevant pieces:

* ``native/angr/src/symbolic/context.rs:SymContext`` owns the single
  ``z3::Context`` and the ``z3::Solver``. Constructed lazily on first
  symbolic op (lazy materialization, see :doc:`rust_z3_sharing`) and
  shared by every state forked
  from the same parent through ``Arc``.
* ``native/angr/src/state/mod.rs`` ``RustSimState`` carries register values,
  memory pages, and a reference to the symcontext.
* ``native/angr/src/exploration/run_loop.rs`` drives the
  ``stash → step → categorize`` loop on the calling thread. It is a thin
  driver — it owns the loop frame and delegates the per-state body to
  ``run_loop_single.rs::step_one``, the single stepping decision point.

The Python wrapper (``angr/exploration/rust_manager.py``) is
single-threaded too — every PyO3 call into Rust runs on the
GIL-holding thread.


Option A — Shared-nothing states with Z3_translate
--------------------------------------------------

Each worker thread owns its own ``z3::Context``. States are bound to
the context they were created in. When the scheduler hands a state
from worker X to worker Y, every Z3 AST in that state (register
values, memory overlays, solver constraints) is translated into Y's
context via ``Z3_translate``.

API
~~~

The ``z3`` crate already exposes the ``z3::Translate`` trait —
its blanket ``unsafe impl<T: Ast> Translate for T`` wraps ``Z3_translate`` for
every AST type. The ``PrepareSynchronized`` submodule
(``translate/synchronization/``) further wraps it for multi-threaded
state migration. **No binding extension needed** to spike or land
Option A.

A higher-level helper would walk a ``RustBV`` tree and translate every
Z3-backed node into the destination context, producing a new
``RustBV`` with the same shape rooted in that context.

Pros
~~~~

* **Zero contention.** Worker threads make no synchronized calls
  during stepping — each owns its solver, AST cache, and context
  outright.
* **Scales linearly with cores** for any workload where migration
  is rare. The ``ekopartyctf2016_sokohashv2`` / ``securityfest_fairlight``
  /``hackcon2016_angry-reverser`` group all spend the bulk of wall
  time inside ``solver.check()``; running 4 such states in parallel
  on 4 cores approaches 4× throughput.
* **Mature primitive.** ``Z3_translate`` is the canonical Z3 idiom
  for cross-context migration and is already used by Z3 callers in
  the wild.
* **Memory locality.** A worker's solver + AST cache + per-state
  byte overlays all live in one thread's working set.

Cons
~~~~

* **Migration cost is workload-shape sensitive.** A 10K-node state
  costs ~4–7 ms to translate. A scheduler that migrates frequently
  (work-stealing per step) makes this the dominant cost. A scheduler
  that migrates per completed task amortizes it.
* **Memory overhead.** Each worker context replicates the AST trees
  it inherits. For workloads where a single fat ancestor state is
  forked into many narrow children, every worker that touches a
  descendant materializes that ancestor's AST in its own context.
  Quantified estimate: O(workers × resident_AST_node_count) memory,
  i.e. ~N× the single-threaded footprint at worst.
* **Translate is not free even at task boundaries.** For benches
  that complete a "task" in 10–50 ms (the CTF reverser shape — many
  short paths), translate at 5 ms can still be 10–50 % of runtime.
* **Cache invalidation.** The Rust engine has a translation-side
  LRU (``angr-518z``) and a thread-local AST cache (cf. ``angr-m81n``).
  Both would need per-context
  partitioning.

Implementation cost
~~~~~~~~~~~~~~~~~~~

Medium. Reusable infrastructure exists in the ``z3`` crate. The work
breaks into:

#. Move ``SymContext`` ownership from ``Arc`` to per-worker
   ``thread_local!``, with explicit migration on state handoff.
#. Implement a ``translate_into(target_ctx: &SymContext)`` helper
   over ``RustBV`` that recurses into every AST node.
#. Partition the AST caches per context.
#. Wire a work-stealing scheduler (crossbeam-deque or rayon's
   work-stealing pool) over the stash with translate on steal.
#. Audit Python-side callbacks: any SimProcedure that round-trips a
   BV through Python touches the *parent* state's context, so
   callbacks must be pinned to the worker that owns the state.

Risk surface
~~~~~~~~~~~~

* **Translate must be exhaustive.** A single un-translated AST node
  in a foreign context is undefined behavior at the Z3 layer (the
  ``unsafe impl`` is unsafe for exactly this reason). A regression
  test that round-trips a real state across contexts and re-checks
  every constraint would gate the implementation.
* **GIL still required for Python callbacks.** Workers stepping a
  state that fires a Python SimProcedure must acquire the GIL,
  serializing on it. Benches that are SimProcedure-heavy
  (``mma_howtouse``, ``flareon2015_10`` Callable flow) won't see
  linear scaling.


Option B — Single Z3 context with mutex
---------------------------------------

All worker threads share one ``z3::Context``. Access is gated by a
``Mutex`` (or ``RwLock`` with reader-mostly access for
``solver.check()``). No translation: every state references AST in
the shared context.

API
~~~

z3-rs requires ``Context: !Sync``, so the actual implementation has
to be ``Mutex<Context>`` with the lock held across every operation
that touches the context. Realistically this means **every** Rust
operation that constructs a Z3 AST or calls into the solver — i.e.
nearly the whole hot path.

Pros
~~~~

* **Zero migration cost.** Any worker can pick up any state without
  data movement.
* **Smallest memory footprint.** One AST table, one solver,
  N workers.
* **Conceptually simpler.** No per-context partitioning of caches,
  no translation correctness concerns.

Cons
~~~~

* **Contention scales poorly.** The mutex covers AST construction,
  solver constraints, ``solver.check()``, model extraction, and
  every transitive call into Z3. For workloads where Rust spends
  >50 % of wall time inside Z3 (the slow trio above), contention
  is severe past 2 workers.
* **``solver.check()`` is the longest-held lock.** A bench that
  spends seconds in a single ``check()`` (the slow mode of
  ``securityfest_fairlight``: ~21 s on slow runs) blocks every
  other worker for that duration. Workers go idle.
* **No locality gain.** A worker waiting on the lock cannot make
  forward progress on any state, even one it already holds.
* **Lock granularity is hard to reduce.** Z3's API does not lend
  itself to fine-grained locking — most calls reach into the
  context's symbol table or AST manager.

Implementation cost
~~~~~~~~~~~~~~~~~~~

Low to medium for the lock plumbing itself; medium to high for the
audit work needed to prove every Z3 access goes through the mutex.
A single unlocked ``Z3_ast_*`` call is a data race.

* Wrap the singleton context in ``Arc<Mutex<...>>``.
* Audit every call to ``ctx.*`` and ``solver.*`` in
  ``native/angr/src/symbolic/`` and ``solver.rs`` to confirm it
  acquires the lock.
* Add a regression test that exercises concurrent ``check()``
  through the lock.

Risk surface
~~~~~~~~~~~~

* **Tail-latency amplification.** A single slow ``check()`` blocks
  all workers, so total wall time approaches the worst per-state
  ``check()`` time × num_states / num_workers, not the average.
  Bimodal-variance benches (``rust_bimodal_variance``) are the
  worst case.
* **Lock-free fast paths still need the lock.** Even
  ``RustBV::Concrete`` operations that don't touch Z3 today may
  need to acquire the lock the moment they cross into a symbolic op. The hot path becomes lock-acquire-heavy.


Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Criterion
     - Option A (shared-nothing + translate)
     - Option B (single context + mutex)
   * - Implementation complexity
     - Medium. Per-worker context, per-context caches, scheduler.
       ``z3`` crate primitives exist.
     - Low-medium plumbing, high audit cost. Every Z3 access path
       must go through the lock.
   * - Throughput at 2 workers
     - Near-linear for state-heavy benches. Migration < 5 % if
       per-task work > 100 ms.
     - 1.3–1.7× for solver-bound benches; worse if a slow
       ``check()`` blocks the other worker.
   * - Throughput at 4 workers
     - Near-linear for state-heavy benches with infrequent
       migration. Capped by Python callbacks (GIL).
     - 1.5–2× max; severe contention on solver-bound benches.
   * - Throughput at 8 workers
     - 4–6× for the favorable shape; GIL becomes the cap.
     - 2× at best; lock becomes the bottleneck.
   * - Memory overhead
     - O(workers × AST footprint). Worst case N× single-threaded
       memory.
     - O(1) extra — one context, one AST table.
   * - mma_howtouse-shape (Callable-heavy)
     - Limited gain — each Callable rebuilds a state, and Python
       SimProcedure callbacks serialize on the GIL.
     - Limited gain — same GIL issue, plus mutex on the rebuild.
   * - unmapped_analysis-shape (state-heavy CFG sweep)
     - Strong gain — many independent states, infrequent migration.
     - Moderate gain — contention on the solver for parallel
       checks.
   * - sokohashv2 / fairlight (bimodal Z3)
     - Strong gain at 2-4 workers — the slow ``check()`` doesn't
       block other workers' progress.
     - Weak gain — the slow ``check()`` is the lock holder.
   * - Migration path from current engine
     - Significant refactor: ``SymContext`` ownership, scheduler,
       cache partitioning. Can be feature-gated.
     - Smaller refactor: wrap in mutex, audit access sites.
       Performance cost likely visible single-threaded too.
   * - Single-threaded fallback
     - Free — disable the scheduler, run on one worker.
     - Lock acquisition is still there; expect a small
       single-threaded regression unless lock is conditional.


Recommendation
--------------

**Pursue Option A — shared-nothing with Z3_translate at task
boundaries — when committing to parallel exploration**, contingent
on the scheduler migrating states **only at coarse boundaries** (a
completed task / exhausted budget / merge point), never per-step.

Option A was accepted on this rationale, and the shipped scheduler
enforces the coarse-boundary constraint (states move only at task
boundaries); `Where it landed`_ records what was then built.

Rationale:

#. The translate-cost spike (``angr-59jk.2``) put a typical state
   migration well below the duration of a single task on the slow
   Z3-bound benches, so migration overhead at task boundaries is a
   small fraction of wall time.
#. Option B's lock-on-every-Z3-call shape pessimizes exactly the
   workloads where parallelism would help most — long ``solver.check()``
   calls become serialization points across all workers.
#. The ``z3`` crate's ``Translate`` infrastructure is production-ready.
   The remaining work is on the angr side: per-worker ``SymContext``,
   scheduler, cache partitioning.
#. The memory cost of Option A is real (worst case N× resident set)
   but bounded by per-worker state count, and N× the engine's resident
   set on the heaviest benches still fits on commodity hardware.

**If parallel exploration is not committed to**, defer both options.
The current single-threaded engine is competitive with Python on most
benches; the unimproved gap on ``mma_howtouse`` /
``hackcon2016_angry-reverser`` / ``ekopartyctf2016_sokohashv2`` is
attributable to non-parallel causes (Callable per-init memory sync,
Z3 model nondeterminism, AST-shape mismatches — see the "Known slower
benchmarks" section of :doc:`rust_engine`). Single-threaded fixes for
those gaps have lower risk than a threading rewrite.


Where it landed
---------------

The staged rollout planned for Option A shipped, but almost every
increment taught something that redirected the next one.

* **Snapshot transport, not translate-on-steal** (``angr-1ilq``, ``angr-1ilq.1``).
  The original plan had the stealing worker call
  a ``translate_state`` (since deleted, angr-mx00p.15) to pull a victim
  state into its own context. Adversarial review showed that is a data race: it reads the
  *source* Z3 context on the *stealing* thread while the victim worker
  is still using it (hazard C — Z3 is single-threaded per context). The
  shipped transport is a ``StateMigrationPayload`` built from the same
  per-state encoding ``dump_snapshot`` uses (``state/migration.rs``);
  the receiving worker rebuilds the state and its solver locally.
* **Migration count, not transport cost, is the lever** (``angr-t3l5o``).
  The overhead gate models a level-synchronous wave
  that migrates every dispatched state; a steal-on-imbalance scheduler
  migrates only the fraction that is actually stolen. Making the
  transport several times cheaper widened the break-even steal
  fraction but did not flip the gate on its own; keeping steals rare
  did.
* **Per-wave full-frontier migration dominated** (``angr-vh834``). A
  persistent worker pool and a warm per-worker block cache removed
  thread churn and re-lifts, yet a wide, slow exhaustive workload still
  ran slower with more workers, because the wave loop drained the whole
  active stash into the shared ``Injector`` every wave and reattached
  every deep-constraint state in a new Z3 context. The fix is
  worker-local frontiers that persist across waves.
* **Anti-migration scheduler** (``angr-729vn``, ``angr-vh834``, ``angr-nkoct``).
  Each worker keeps a thread-private
  ``VecDeque<RustSimState>`` in its home Z3 context that is never
  serialized; the shared crossbeam ``Injector<StateMigrationPayload>`` is
  the only cross-thread channel; serde is paid on exactly two paths
  (surplus shed on imbalance, and a materialized found terminal
  crossing the ``thread::scope`` join); dead-path terminals come back as
  a lightweight ``TerminalSummary`` with zero serde.
  ``SchedulerStats.honest_steal_fraction`` reports how often a state
  actually crossed a thread. ``exploration/run_loop.rs::run_loop``
  dispatches between the single-threaded loop, the steady loop
  (``run_loop_steady.rs``, worker-local frontiers, ``RUST_PARALLEL_STEADY``)
  and the wave loop (``run_loop_wave.rs``, ``parallel_workers=`` /
  ``RUST_PARALLEL_WORKERS``).
* **What it did not deliver.** On the CTF benchmark corpus parallel
  exploration is not a wall-clock win: the frontiers are narrow and the
  work per state is small, so the steal fraction stays far above
  break-even. Where it does win is wide-and-slow find-all workloads;
  the synthetic ``fork_solve_*`` benches exist to exercise that shape.
  The parallel path's correctness contract is therefore set-equality of
  the found-set across worker counts, gated in ``rust-nightly.yml`` by
  ``tests/benchmarks/run_findall_gate.py`` (``make gate-findall``), not a
  timing. The steady loop stays an explicit opt-in because it
  over-collects the found stash for ``num_find > 1`` relative to the
  serial baseline.


State-selection policy seam
---------------------------

Neither end of the ordering described under *Where it landed* above is
hardcoded any more. Both the *which state do I step next* decision and the
*which state do I ship away* decision go through the ``SelectionPolicy`` trait
in ``exploration/selection_policy.rs`` (``angr-a32jl.1``, ``angr-03vl4.19``).
The trait's methods:

* ``select`` — pop the next active state (the run loop's only removal path);
* ``on_fork`` — where a freshly-forked successor lands in the active deque
  (the only insertion path);
* ``select_for_offload`` — the offload-side *mirror* of ``select``: which end
  of a worker's local deque ``scheduler_worker.rs::offload_one`` detaches for
  the shared ``Injector``. The default implementation pops the **back**, which
  is the coldest end for every built-in but ``Lifo``;
* ``on_state_removed`` — eviction notice for states that leave ``active``
  without going through ``select`` (cross-worker migration), so memoizing
  policies do not leak per-``state_id`` side-table entries (``angr-ua7fd``);
* ``name`` — a stable, human-readable policy name for logging and stats.

The shipped policies are ``Fifo``, ``Lifo``, ``RandomSelection``,
``CoverageGuided``, ``LoopHeadRoundRobin``, ``DirectedCfgDistance`` and
``FindDirected``. The production default is
``selection_policy::default_policy()`` — **``Fifo`` (BFS), not ``Lifo``**;
``RustExplorationManager``'s constructor installs it and the
``set_state_selection_*`` methods swap it. For the user-facing spelling of
those knobs (``exploration_strategy=``, the DFS technique shim) read
:doc:`rust_engine`; this section is only the rationale for the seam existing.

The parallel stack threads the manager's configured policy down rather than
naming one of its own: ``run_loop_wave.rs`` passes it to
``WaveJob::new_with_policy``, ``run_loop_steady.rs`` to
``RunSession::new_with_policy``, and it reaches each worker through
``WorkTransport``. The no-argument constructors default to ``Lifo`` for
back-compat with the pre-seam tests only — production never uses them. Getting
this wrong is not cosmetic: the offload path *did* hardcode ``pop_front``
while production ran ``Fifo``, so every default-policy offload shipped away
exactly the state ``select`` was about to hand back for free, paying a full Z3
detach/reattach round-trip for nothing (``angr-03vl4.19``). The scheduler
tests now drive ``default_policy()`` instead of naming a policy, which is what
keeps them from drifting from production again.

Single-worker exploration is *order-deterministic* under every built-in — the
module header of ``selection_policy.rs`` states that contract and the
``test_selection_trace_deterministic_across_runs`` guard in
``selection_policy_tests.rs`` enforces it. That
contract does **not** extend to the parallel path: steal order is
design-nondeterministic, so what is gated there is set-equality of the
found-set across worker counts
(``tests/benchmarks/run_findall_gate.py``, in ``rust-nightly.yml``).


Distributed (cross-machine) migration — measured KILL
-----------------------------------------------------

Migrating states over a network rather than between threads in one
process was probed and killed (``angr-op0dn.13.7``): it is strictly
worse than the in-process case, which already loses on migration
count. The reasoning is recorded in :doc:`rust_engine_decisions`. Do
not file a distributed follow-on without a human decision.


Open questions
--------------

* **GIL handling.** Python SimProcedure callbacks are the dominant
  serialization point under both options. Could state callbacks be
  routed to a dedicated "Python worker" thread that holds the GIL
  while other workers continue stepping pure-Rust states? Risk:
  callback latency adds to those states' wall times.
* **NUMA-aware stealing.** Would topology-aware stealing (steal from
  the same NUMA node first, fall back to the global ``Injector``) pay
  off on high-core-count hosts? It is an optimization, not a
  correctness concern.
* **Mid-solve cancellation.** Cancellation is cooperative at task
  boundaries (``CancelToken`` in ``exploration/scheduler.rs`` documents
  the find-versus-budget semantics). Whether to also abort a *long
  in-flight solve* via Z3's ``Context::handle().interrupt()`` is open:
  doing it safely means keeping each worker's context alive past every
  possible interrupter.


References
----------

* This design: parallel exploration with per-thread Z3 contexts (``angr-59jk``).
* Companion spike: Z3_translate cost per node (``angr-59jk.2``).
* Excluded alternative: ``Z3 parallel.enable=true`` is
  correctness-breaking in this codebase (``angr-gfay``).
* User-facing engine doc: :doc:`rust_engine` (see "Benchmarks slower
  than Python" for the workloads this work targets).
* Bimodal variance background: :doc:`rust_bimodal_variance`.
* Memory-design companion: :doc:`rust_lazy_memory_design`.
