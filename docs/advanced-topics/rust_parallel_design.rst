Rust engine: parallel exploration threading model — design comparison
=====================================================================

This document is the design-phase deliverable for parallel exploration
in the Rust engine, tracked by bead ``angr-59jk`` (deferred) and its
child ``angr-59jk.1``. It does **not** describe shipped code — the Rust
engine is single-threaded today. The empirical translate-cost numbers
quoted below come from the companion spike ``angr-59jk.2`` (commit
``2a997ec89``, ``native/angr/tests/z3_translate_spike.rs``).

Audience: contributors deciding whether to commit to parallel
exploration and, if so, which threading model to build on.

Status: **proposed, not implemented**. The recommendation below feeds
the GO/NOGO decision for the parent bead.


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

Two viable threading models for spanning multiple cores exist, both
listed in the parent bead description. This document compares them on
the criteria the parent bead calls out (complexity, throughput,
memory, sensitivity to bench shape, migration path), using the
empirical translate-cost data from ``angr-59jk.2``.

A third path — ``Z3 parallel.enable=true``, which fan-outs *inside* a
single solver — is explicitly out of scope. It was tested in
``angr-gfay`` (closed 2026-05-21) and proved correctness-breaking on
4 of 6 sampled benchmarks (see bd memory ``avoid-z3-parallel-enable``).
The two options below are about **parallel state stepping**, not
parallel SAT.


Empirical input: Z3_translate cost
----------------------------------

``angr-59jk.2`` measured Z3_translate per-AST-node cost on this
machine (release build, 5-sample medians):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Workload shape
     - Nodes
     - ``translate()`` calls
     - ns / node
   * - Deep-tree micro chain
     - 3001
     - 1
     - ~390
   * - State-export shape (256 leaves + 32 constraints)
     - 544
     - 288
     - ~677

The per-call inflation in the second row is FFI overhead: each leaf is
a separate ``Z3_translate`` invocation. The 390 ns/node steady-state
of the deep-tree variant is what dominates when migrating a whole
state — Z3 walks the AST internally with one ``Z3_translate`` call.

**Implication.** A 10K-node exported state migrates in ~4–7 ms.

* Migration at completed-task boundaries (>100 ms of exploration per
  task): translate is <5 % of runtime. **Viable.**
* Per-step migration: translate dominates. **Not viable.**

Full numbers and methodology: bd memory ``59jk2-translate-cost``,
``native/angr/tests/z3_translate_spike.rs``.


Current Rust threading model
----------------------------

Single-threaded. The relevant pieces:

* ``native/angr/src/symbolic/context.rs:SymContext`` owns the single
  ``z3::Context`` and the ``z3::Solver``. Constructed lazily on first
  symbolic op (``materialize_z3``) and shared by every state forked
  from the same parent through ``Arc``.
* ``native/angr/src/state.rs:RustSimState`` carries register values,
  memory pages, and a reference to the symcontext.
* ``native/angr/src/exploration/run_loop.rs`` drives the
  ``stash → step → categorize`` loop on the calling thread.

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

The z3-patched crate already exposes the Translate trait at
``native/z3-patched/src/translate/mod.rs:23-31`` —
``unsafe impl<T: Ast> Translate for T`` wraps ``Z3_translate`` for
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
  LRU (``angr-518z``) and a thread-local AST cache
  (``angr-m81n``-adjacent territory). Both would need per-context
  partitioning.

Implementation cost
~~~~~~~~~~~~~~~~~~~

Medium. Reusable infrastructure exists in z3-patched. The work
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
  need to acquire the lock the moment they cross into an
  ``RustBV::Expression`` op. The hot path becomes lock-acquire-heavy.


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
       z3-patched primitives exist.
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

Rationale:

#. The translate-cost spike measured 390 ns/node steady-state, putting
   a 10K-node state migration at 4–7 ms. For benches where individual
   tasks take >100 ms (any of the slow Z3-bound benches; most CTF
   benches with >1 path), migration overhead is <5 %.
#. Option B's lock-on-every-Z3-call shape pessimizes exactly the
   workloads where parallelism would help most — long ``solver.check()``
   calls become serialization points across all workers.
#. The z3-patched ``Translate`` infrastructure is production-ready.
   The remaining work is on the angr side: per-worker ``SymContext``,
   scheduler, cache partitioning.
#. The memory cost of Option A is real (worst case N× resident set)
   but bounded by per-worker state count, and the engine's resident
   sets today are 200–500 MB on the heaviest benches — N× still fits
   on commodity hardware.

**If parallel exploration is not committed to**, defer both options.
The current single-threaded engine is competitive with Python on most
benches; the unimproved gap on ``mma_howtouse`` /
``hackcon2016_angry-reverser`` / ``ekopartyctf2016_sokohashv2`` is
attributable to non-parallel causes (Callable per-init memory sync,
Z3 model nondeterminism, AST-shape mismatches — see the "Known slower
benchmarks" section of :doc:`rust_engine`). Single-threaded fixes for
those gaps have lower risk than a threading rewrite.


Migration path
--------------

If Option A is chosen, a staged rollout:

#. **Phase 0 — instrumentation.** Add per-state ``task_id`` and a
   counter that tracks state migrations across (hypothetical)
   worker boundaries. Run the bench suite to confirm migration
   cadence assumptions hold (>100 ms per task; <10 migrations per
   bench).
#. **Phase 1 — per-context AST cache.** Partition
   ``thread_local!``-driven caches in ``claripy_bridge.rs`` and the
   ``angr-518z`` Rust LRU by context identity. No threading yet.
#. **Phase 2 — translate helper.** Implement
   ``RustBV::translate_into(target_ctx: &SymContext) -> RustBV`` plus
   ``SymContext::translate_state(other: &RustSimState) -> RustSimState``.
   Round-trip every existing state through translate-into-self-context
   as a no-op regression test.
#. **Phase 3 — work-stealing pool.** Introduce a crossbeam-deque or
   rayon-backed worker pool gated by a ``RUST_PARALLEL_WORKERS`` env
   var (default 1). Single-worker remains the current path. Each
   worker owns a ``SymContext``; the scheduler translates on steal.
#. **Phase 4 — Python callback dispatch.** Workers stepping a state
   that fires a Python SimProcedure must serialize on the GIL.
   Document the throughput cap for SimProcedure-heavy benches.
#. **Phase 5 — A/B regression.** Run the full bench suite with
   ``RUST_PARALLEL_WORKERS=1, 2, 4`` and gate any regression on
   ``baseline_timings.json``. Bimodal-variance benches get
   per-mode stats; the work is GO if 2 workers gives ≥1.5×
   speedup on the slow Z3-bound trio with no regression on the
   single-threaded baseline.

If Option B is chosen instead, the migration is shorter but the
benchmark-time risk is higher: every existing bench may regress by
the lock-acquisition overhead, with no upside on
``mma_howtouse``-shaped workloads (which dominate the slower-bench
set).


Open questions
--------------

* **GIL handling.** Python SimProcedure callbacks are the dominant
  serialization point under both options. Could state callbacks be
  routed to a dedicated "Python worker" thread that holds the GIL
  while other workers continue stepping pure-Rust states? Risk:
  callback latency adds to those states' wall times.
* **Lazy memory interaction.** Under :doc:`rust_lazy_memory_design`,
  states share more structure (``Arc`` ancestors of ``MultiValues``
  pages). Cross-context translation must deep-copy those shared
  ancestors or shared-nothing's correctness depends on the page
  being immutable in both contexts. The latter is true for the
  proposed lazy-memory design but worth re-verifying at Phase 2.
* **Scheduler policy.** Work-stealing is the default candidate
  (mature, well-understood). Topology-aware variants (steal from
  the same NUMA node first, fall back to global) might matter
  past 8 cores but are an optimization, not a correctness concern.
* **Cancellation.** A worker that finds the target while others are
  still stepping should be able to signal "stop". Z3 supports
  cancellation via ``set_interrupt``; the scheduler needs to fan
  it out to each per-worker context.
* **Determinism.** Parallel exploration is non-deterministic by
  nature (state visit order depends on scheduling). Document the
  loss of determinism and gate it on an env var so reproducible
  runs remain available at ``RUST_PARALLEL_WORKERS=1``.


References
----------

* Parent bead: ``angr-59jk`` (deferred) — design of parallel
  exploration with per-thread Z3 contexts.
* Companion spike: ``angr-59jk.2`` (closed, commit ``2a997ec89``) —
  Z3_translate cost measured at ~390 ns/node steady-state, ~677 ns/node
  for state-export shape. ``native/angr/tests/z3_translate_spike.rs``.
* Excluded alternative: ``angr-gfay`` (closed) — ``Z3 parallel.enable=true``
  is correctness-breaking in this codebase; documented in bd memory
  ``avoid-z3-parallel-enable``.
* z3-patched Translate trait: ``native/z3-patched/src/translate/mod.rs:23-31``.
* User-facing engine doc: :doc:`rust_engine` (see "Known slower
  benchmarks" for the workloads this work targets).
* Bimodal variance background: :doc:`rust_bimodal_variance`.
* Memory-design companion: :doc:`rust_lazy_memory_design`.
