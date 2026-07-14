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

Full numbers and methodology: the ``z3_translate_spike`` test at
``native/angr/tests/z3_translate_spike.rs`` (bead ``angr-59jk``).


Current Rust threading model
----------------------------

Single-threaded. The relevant pieces:

* ``native/angr/src/symbolic/context.rs:SymContext`` owns the single
  ``z3::Context`` and the ``z3::Solver``. Constructed lazily on first
  symbolic op (``materialize_z3``) and shared by every state forked
  from the same parent through ``Arc``.
* ``native/angr/src/state/mod.rs`` ``RustSimState`` carries register values,
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

.. note::

   **Implementation update (angr-1ilq, 2026-06-28) — the transport
   mechanism below is superseded.** Phases 2–3 originally specced
   *translate-on-steal*: the stealing worker calls
   ``SymContext::translate_state`` to pull a victim state into its own
   context. Adversarial review (``angr-1ilq.1``) proved this **unsound** for
   work-stealing: ``translate_state`` reads the *source* Z3 context, but it
   runs on the *stealing* thread while the victim worker is still using that
   context (it was stolen *because* the victim is busy) — a data race inside
   Z3, which is single-threaded per context (hazard C).

   The shipped transport instead crosses the worker boundary with
   ``Send``-by-construction data: ``RustSimState::detach_for_migration`` →
   ``StateMigrationPayload`` (serialized, context-free bytes + live
   GIL-managed ``Py`` overlay handles) → ``reattach`` rebuilds every AST in
   the *stealer's own* context, never reading the source. Zero ``unsafe``.
   See ``native/angr/src/state/migration.rs`` and the ``angr-1ilq.1`` bead.
   ``translate_state`` survives only for same-thread context moves; it must
   **not** be reintroduced on the steal path.

   **GIL-coupling reframes Phases 3–4 (angr-1ilq.3).** Stepping is not a
   pure-Rust ``step → successors`` operation: ``step_state_with_skip`` takes
   a live ``Python<'_>`` token and yields back to Python at seven callback
   points (find/avoid predicates, SimProcedures, syscalls, symbolic
   branches, Python-VEX fallback, errors). A worker therefore cannot step a
   state without the GIL. Real parallelism requires releasing the GIL
   (``py.allow_threads``) only around the Rust-pure inner work and
   re-acquiring it for callbacks — Phase 4 is the load-bearing half, not a
   coda. The first ``angr-1ilq.3`` increment landed the work-stealing pool
   machinery (``exploration/scheduler.rs``: per-worker Z3 context,
   ``crossbeam-deque`` over ``StateMigrationPayload``, task-boundary
   cancellation) **proven in isolation**; wiring it into ``run_loop`` behind
   the GIL boundary is the deferred follow-up.

#. **Phase 0 — instrumentation.** Add per-state ``task_id`` and a
   counter that tracks state migrations across (hypothetical)
   worker boundaries. Run the bench suite to confirm migration
   cadence assumptions hold (>100 ms per task; <10 migrations per
   bench).
#. **Phase 1 — per-context AST cache.** Partition
   ``thread_local!``-driven caches in ``claripy_bridge/cache.rs`` and the
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

   .. note::

      ``RUST_PARALLEL_WORKERS`` (canonical; the number of REAL
      work-stealing worker threads, default 1) is **distinct** from
      ``ANGR_PARALLEL_WORKERS`` (the ``angr-panhl.1`` migration *model*
      worker count, default 4, which only feeds the
      ``record_migration_sample`` instrumentation and never spawns
      threads). At the default of 1 the run loop takes the verbatim
      single-threaded path; the value is surfaced as
      ``parallel_real_workers`` in ``mgr.stats`` alongside the model's
      ``parallel_num_workers``.
#. **Phase 4 — Python callback dispatch.** Workers stepping a state
   that fires a Python SimProcedure must serialize on the GIL.
   Document the throughput cap for SimProcedure-heavy benches.
#. **Phase 5 — A/B regression.** Run the full bench suite with
   ``RUST_PARALLEL_WORKERS=1, 2, 4`` and gate any regression on
   ``baseline_timings.json``. Bimodal-variance benches get
   per-mode stats; the work is GO if 2 workers gives ≥1.5×
   speedup on the slow Z3-bound trio with no regression on the
   single-threaded baseline.

   .. important:: **Overhead GO/NO-GO gate (angr-1obng, 2b′) — result: NO-GO.**

      Before wiring the live wave loop (2c), the ``2b′`` gate measured the
      *real* per-state migration tax with the ``RUST_PARALLEL_SHADOW_PROBE``
      shadow probe (``to_serialized`` + ``from_serialized`` into a foreign Z3
      context, the exact serde
      :rust:struct:`StateMigrationPayload::reattach` performs) and pitted it
      against an optimistically-biased parallelism model
      (``tests/benchmarks/run_parallel_overhead_gate.py``). At 2 workers both
      target benches come back **NO-GO**: ``cmu_binary_bomb_partial`` ≈ 0.96×
      (migration tax ≈ 0.30 s = 43 % of the 0.70 s clean serial run) and
      ``codegate_2017-angrybird`` ≈ 0.07× (migration tax ≈ 51 s = **13.6× the
      entire 3.77 s** clean serial run; ≈ 60 ms/state, ≈ 1 MB serialized/state).

      The bottleneck is **transport, not scheduling**: the
      ``to_serialized``/``from_serialized`` round-trip across Z3 contexts is
      far too expensive for level-synchronous waves. Per-state cross-context
      migration (Option A as specified) cannot pay off on these workloads
      until the transport is ≈ 10–50× cheaper. **2c live wiring is therefore
      NOT authorized.** The 2b′ scheduling-policy primitives (serial bounce
      queue, shared block cache, narrow-frontier fallback) still land as
      isolated, tested machinery ready to receive 2c *if and when* a cheaper
      migration transport closes the gap; reducing that transport cost is the
      true blocker and is tracked as a dedicated prerequisite bead
      (``angr-t3l5o``) that blocks both the 2c live wave loop
      (``angr-vh834``) and the parent ``angr-1ilq.3``.

   .. important:: **Transport reduction + pivot (angr-t3l5o, 2026-06-30) —
      transport win landed; the residual blocker is migration *count*, not
      per-state cost.**

      A measure-first attack on the transport cost split the migration
      round-trip into six phases (Phase 0, ``bench_migration_phases`` +
      env-gated ``ANGR_MIGRATE_PHASE_TIMERS`` counters) and confirmed the Z3
      **SMT-LIB2 text round-trip** dominated: on ``codegate`` the full-solver
      ``format!("{}", solver)`` emit + ``from_string`` reparse was 88.8 % of
      the round-trip (emit alone 66 %). The assertions that text dumped were
      largely the *assume class* already carried as reconstructible ``RustBV``
      IR in ``assumed_constraints`` — pure redundant work.

      **Phase 1** rebuilds the assume class on reattach by re-asserting
      ``assumed_constraints`` via ``assume_true``/``assume_false`` (no text),
      and carries only the *residual* no-``RustBV`` class
      (``add_constraint_raw`` / ``add_bv_constraint`` / ``merge`` guards) as an
      SMT-LIB2 dump that is **empty in the common case** (see
      :rust:struct:`SymContextSnapshot` ``residual_smtlib2`` +
      ``reassert_assumed``, and the ``non_bv_assertions`` two-class log on
      :rust:struct:`SymContext`). Result: the text round-trip collapsed to
      ~0, cutting ``codegate`` per-state tax **≈ 4.2×** (60 → 14.5 ms/state).
      The gate verdict improved 0.07× → 0.25× — but is **still NO-GO**.

      **Why transport alone cannot finish the job.** Post-Phase-1 attribution
      shows the remaining ~14.5 ms/state is ~40 % serde and ~60 % an
      *irreducible Z3-AST-rebuild floor* — re-asserting the constraints into
      the consumer's context builds fresh Z3 ASTs, work that **every** sound
      cross-thread transport must pay (``Z3_translate`` included; it is unsound
      on the steal path anyway). On ``codegate`` that floor (~8 ms/state)
      already exceeds the per-state *work* budget (3.5 s ÷ 844 states ≈
      4.2 ms/state). A workload this wide-and-shallow is migration-bound at
      *any* transport cost **if every state migrates**.

      **The pivot — migration count is the lever.** The shadow probe models a
      level-synchronous wave that migrates *every* dispatched state; a real
      steal-on-imbalance scheduler migrates only a fraction ``f`` of states (an
      actual cross-worker steal). The gate now reports the **break-even steal
      fraction ``f*``** (and accepts ``--steal-fraction``):
      ``codegate`` goes GO at ``f* ≈ 10.5–13.3 %`` — i.e. if work-stealing
      migrates fewer than ~1 in 9 states. Crucially, Phase 1's 4.2× cheaper
      transport **widened that budget from ~3 % (old 60 ms transport) to
      ~13 %**, making the scheduler's target ~4× easier rather than flipping
      the gate itself.

      **Consequence for 2c.** The remaining work is *not* further transport
      reduction — it is a scheduler whose deque holds **worker-local live
      states** (in their home Z3 context, never serialized) and serializes a
      state into a :rust:struct:`StateMigrationPayload` **only when it is
      offered for stealing** (surplus/imbalance), keeping steals ≤ ``f*``. Note
      the current ``exploration/scheduler.rs`` does the opposite — it
      ``detach_for_migration``s *every* child onto the deque (eager
      serialization), the level-synchronous worst case the gate measured. The
      anti-migration redesign is the live blocker for ``angr-vh834`` /
      ``angr-1ilq.3``; ``angr-t3l5o`` delivered the transport win that makes it
      feasible.

   .. important:: **Anti-migration scheduler delivered (angr-729vn,
      2026-06-30) — surplus-only serialization; gate GO with a wide-bench
      safety margin, codegate marginal.**

      The redesigned ``exploration/scheduler.rs`` now holds worker-local **live**
      states in a thread-private ``VecDeque<RustSimState>`` (home Z3 context,
      never serialized) and crosses the worker boundary as a
      :rust:struct:`StateMigrationPayload` on only two paths: an actual imbalance
      steal (surplus shed to the shared :rust:struct:`Injector`) and a
      *materialized* found terminal that must cross the ``thread::scope`` join.
      Dead-path terminals (deadended / errored) are returned as a lightweight
      ``TerminalSummary`` built in-context and dropped there — **zero serde**.
      This is a selective ``drop_terminal_states``: the full symbolic state of
      dead paths is not recoverable through the parallel path, but it removes the
      ~50 % terminal-join serde that would otherwise be paid at any steal
      fraction. Surplus selection is FIFO (coldest first) under an idle-gated
      trigger (sibling starving ``&&`` local backlog ≥ 2, mirroring the
      ``record_migration_sample`` model) plus a high-water cap; local consumption
      is LIFO (hottest child next). The crossbeam per-worker
      ``Worker``/``Stealer`` are gone (a ``Stealer`` is ``Send`` and would move
      the irreducibly ``!Send`` ``RustSimState`` across threads); the injector is
      the only cross-thread channel. ``run_instrumented`` reports
      ``SchedulerStats`` whose ``honest_steal_fraction`` =
      ``(surplus_offloaded + materialized_terminals) / dispatches``.

      **Verification (proven in isolation; not yet wired into the run loop).**
      Because the scheduler still cannot step live (GIL coupling — the downstream
      ``angr-vh834`` / ``1ilq.4`` work), the steal fraction is measured with the
      run loop's GIL-free ``f_model = parallel_migrations / parallel_tasks``
      (captured at ``ANGR_PARALLEL_WORKERS=2`` to match the gate's
      ``--workers 2``) plus the found-materialization fraction, and fed to the
      projection gate via ``--steal-fraction`` (driver:
      ``tests/benchmarks/run_steal_fraction_gate.py``). Results:

      * ``cmu_binary_bomb_partial`` — **clean GO**, speedup band 1.66×, break-even
        ``f* ≈ 100 %`` (GO even migrating *every* state). A genuinely
        wide-and-slow bench (width hist ``[3,0,1,7,11]``); the scheduler's
        ``f_model = 0 %`` clears it with an enormous margin. **This satisfies the
        bead acceptance** (GO on ≥ 1 wide-and-slow non-bimodal bench).
      * ``codegate_2017-angrybird`` — **marginal GO**. Honest
        ``f_gate ≈ 10.78 %`` (median ``f_model 10.66 %`` + found ``0.12 %``)
        clears the binding pessimistic break-even ``f*_pess ≈ 11.3 %``
        (speedup band 1.02×–1.12×), vindicating the t3l5o pivot's "keep steals
        below ~11 %" thesis — but only by ~0.5 pp, **not** the conservative 2×
        safety margin. Because ``f_model`` is a **lower-bound proxy** (it counts
        ≤ 1 steal/step over a sticky-home simulation while the real Trigger A can
        shed several per step), codegate sits right at the migration-bound edge
        and its GO is not safe. Codegate's narrow, w≈1-dominated frontier (hist
        ``[158,252,307,118,9]``) is also why real parallelism there would be low
        regardless: the deliverable here is the *mechanism* + a bounded, honest
        serialized fraction, **not** a measured wall-clock win on codegate (that
        needs the live wiring and a wide bench like cmu).

   .. important:: **Live wave loop delivered + measured (angr-vh834, 2026-06-30)
      — mechanism complete and correct; wall-clock GO does NOT materialize on the
      available bench (workload-bound, not implementation-bound).**

      The full parallel path is now wired and committed: an ``&mut self``-free
      post-step core (``run_post_step_core``/``CoreOutcome``), a GIL-free
      interpreter step (callbacks self-acquire via ``Python::attach``; the ``py``
      token removed from the step chain), and a live work-stealing wave loop
      behind ``RUST_PARALLEL_WORKERS>=2`` (``run_loop_parallel``) that keeps
      continue-states worker-local (f≈0), materializes only found terminals, and
      reports **real** ``SchedulerStats`` counters. ``workers<=1`` is byte-identical
      (untouched single-threaded path); ``workers=2`` reproduces the
      single-threaded found set (adversarial peer review confirmed the
      concurrency machinery sound; four fauxware-masked routing/accounting gaps
      were found and fixed, incl. a CRITICAL find/avoid-via-``Hook`` misroute).

      **Live measurement — ``cmu_binary_bomb_partial``, ``RUST_PARALLEL_WORKERS``
      1 vs 2 (3 reps each):** workers=2 is **≈1.57× SLOWER** (2.37 s vs 1.51 s).
      Counters at workers=2: ``parallel_tasks=28``, ``parallel_migrations=33``
      (honest steal fraction **> 100 %**), 164 GIL-serialized block lifts. The
      bench is phase-1 of the binary bomb — a **short, narrow** strcmp solve
      (~28 state-steps, frontier width ≈ 1), so there is **no exploitable parallel
      concurrency**, and per-wave overhead dominates: ``run_instrumented``
      re-spawns ``thread::scope`` + two fresh Z3 contexts **every wave**, plus
      migration serde and cold per-worker block caches that GIL-serialize re-lifts.
      The earlier ``[3,0,1,7,11]`` "wide-and-slow" label was a projection/model
      artifact, **not** this exploration.

      This vindicates the 2b′ NO-GO and the f\*≈100 % thin-margin finding: these
      CTF-partial workloads are overhead-bound. **The deliverable is the
      mechanism** (correct, determinism-gated, real-countered parallel
      exploration), **not a wall-clock win.** A real GO is **workload-bound** and
      needs, in order: (1) a genuinely wide-and-slow bench (``run_width_audit.py``
      standing NO-GO — "sequence angr-11djq first to create wide workloads");
      (2) a **persistent worker pool** (eliminate the per-wave ``thread::scope`` +
      Z3-context churn); (3) a **shared warm block cache** (eliminate the
      GIL-serialized per-worker re-lifts).

   .. important:: **Persistent pool + warm cache delivered + measured
      (angr-vh834 Work Items 1-3, 2026-07-01) — still NO wall-clock GO; the
      remaining dominant cost is per-wave full-frontier migration, and that
      is architectural.**

      All three follow-ups from the note above landed and are correct:

      * A wide-and-slow synthetic bench (commit ``3214347ac``,
        ``tests/benchmarks/synthetic_examples/fork_solve_W6_S8``): tunable fork
        width ``W`` and per-state Z3 solve cost ``S``, clearing the
        ``run_width_audit.py`` GO shape (``frac_ge3 ~ 0.80``, ``ms/task`` in the
        thousands) — the wide-AND-slow workload the corpus previously lacked.
      * A **persistent worker pool** (commit ``fa441e13a``): N long-lived
        threads each own a Z3 context for the pool's life, fed one wave at a
        time over ``mpsc`` channels (``PersistentPool`` / ``WaveJob`` /
        ``worker_thread`` in ``exploration/scheduler.rs``). Kills the per-wave
        ``thread::scope`` + ``Context::new`` churn.
      * A **warm per-worker block cache** (commit ``ec0a0b07d``): each worker's
        ``LruCache<u64, Arc<IRSB>>`` persists across dispatches and waves;
        ``block_cache_hits`` / ``block_cache_misses`` surfaced in ``mgr.stats()``.

      Correctness held throughout (workers=1 byte-identical; workers=2
      reproduces the found set — 4/4 ``test_parallel_wave``, 9/9 scheduler unit
      tests, full rust suite green modulo one pre-existing snapshot failure).

      **But there is still no wall-clock GO, and the measurement now isolates
      why.** Two findings:

      1. **num_find=1 (first-find) on a wide frontier is anti-parallel**
         regardless of overhead: single-threaded BFS short-circuits at the
         first satisfiable find (~1 ``satisfiable()``), while the parallel wave
         speculatively runs the whole frontier's find-checks before the
         cancellation lands. ``fork_solve_W6_S8`` num_find=1: workers=1 = 59.7s
         vs workers=2 timed out > 180s. The parallel win therefore requires
         **no-early-exit / exhaustive** workloads (find-all, coverage,
         bug-sweep). See bd ``parallel-numfind1-speculative-waste``.
      2. **Even exhaustive, per-wave full-frontier migration dominates.**
         ``fork_solve_trap_W5_S8_M12`` exhaustive (found=32, ``frac_ge3 ~
         0.99``, max width 45, ~119 steps): workers=1 = 32s, workers=2 and
         workers=4 both timed out > 140s (>4x SLOWER). The warm cache was
         near-perfect (15 misses / 590k hits), so block-lifting is *not* the
         cost — the per-wave Z3-AST reattach of deep-constraint states is.
         ``run_loop_parallel`` is still level-synchronous: every wave it drains
         the entire ``STASH_ACTIVE`` into a shared ``Injector`` and workers
         ``reattach`` each state into their own context. The anti-migration
         scheduler (``angr-729vn``) keeps successors worker-local *within* a
         wave, but the wave *boundary* re-migrates the full frontier. The
         persistent pool and warm cache do not touch this cost. See bd
         ``parallel-per-wave-migration-dominates``.

      **Next lever (architectural): persistent worker-LOCAL frontiers ACROSS
      waves** (``angr-nkoct``). Each persistent worker keeps its own live
      frontier in its own Z3 context for the whole run; states stay put
      wave-to-wave with zero serde, and cross-context migration happens only on
      a genuine work-stealing steal or a terminal/callback that must reach the
      coordinator. This is the real Option A (translate only at task
      boundaries) that the current Model-B level-synchronous loop only
      approximated — the persistent pool is the prerequisite that now makes it
      possible.

   .. important:: **angr-nkoct investigation + partial delivery (2026-07-01) —
      the ">4x migration-dominated" framing above was BIMODAL-INFLATED; the real
      overhead is a linear ~3x, and the full fix is deferred.**

      A focused measurement pass (bd ``parallel-cliff-was-bimodal-not-superlinear``,
      ``bounce-reduction-low-roi-corpus``) re-derived the parallel picture with
      deterministic counters + repeats instead of single-sample wall times:

      * A reproducible stress bench landed —
        ``tests/benchmarks/synthetic_examples/fork_solve_trap_W5_S8_M12`` — that
        forces a Python-SimProcedure bounce on every leaf at ``T`` sequential
        program points (identity hook in ``solve.py``), so the frontier
        re-migrates once per callback level.
      * On a trivial-solve isolation variant (``--s 2 --m 8``) the workers=4
        overhead is **LINEAR** in the callback count (T=1..4:
        2.36 / 3.26 / 4.25 / ~5.0 s, three repeats confirming ~5 s), roughly
        **~3x** vs workers=1 (~1.7 s), ``peak_mem`` flat. The earlier
        ">4x SLOWER / TIMEOUT" observations (and an interim ">26x super-linear
        cliff" claim) were **bimodal-Z3 variance** — single-run timeouts read as
        signal. See :doc:`rust_bimodal_variance`. Gate on completion +
        deterministic counters, never a single cross-worker wall time.
      * **Root cause confirmed:** the per-wave full-frontier re-migration
        (``STASH_ACTIVE.drain()`` → detach → seed → workers → materialize is
        ~1 wave per callback level, each re-seeding the ~40-wide frontier). Not
        reattach *volume* pathology (``parallel_reattaches`` ~1/bounce) and not
        GIL work (sub-ms).
      * **Bounce-reduction is a dry well.** Across 32 corpus benches, 86% of the
        4584 Rust→Python callbacks are inherent pyvex block lifts; the rest are
        user Python hooks or symbolic-arg SimProcedure declines (e.g. ``fseek``
        with a symbolic offset). Only ~20 corpus-wide are cleanly reducible.
        The engine already stays in Rust for ~everything reducible.

      **Delivered:** ``parallel_reattaches`` / ``parallel_bounce_roundtrips`` /
      ``parallel_resume_reinjects`` accounting + the duplex-protocol scaffolding
      (``WorkerCtl`` / ``WorkerUp`` / ``RunSession``, ``Send + Sync``-proven but
      unwired), and **increment 1: the persistent worker-local frontier** —
      ``local`` lifted into ``worker_thread`` so a state crosses the ctx boundary
      at most once per pool-lifetime. Behaviour-neutral today (waves quiesce, so
      ``local`` is empty at each barrier).

      **Deferred (deliberate follow-up, modest ROI):** exploiting the persistent
      frontier requires the **steady-state loop** — remove the wave barrier so
      workers step continuously and frontiers survive across the Python-callback
      boundary. Critically, the ``fork_solve_trap`` stress bench canNOT show the
      win: because every leaf bounces every level, each state must reattach in
      the *coordinator's* Z3 context to run its Python callback, so its
      per-bounce migration is ~inherent (feeding the resumed state back to a
      worker saves nothing). The fix pays off on **partial-bounce** workloads
      (most states continue worker-locally, zero serde; only a few bounce) —
      which needs a partial-bounce demonstration bench, not the stress bench. Net
      honest read: the parallel engine is in reasonable shape; the remaining win
      is real but modest and best pursued as a scoped effort.

   .. important:: **angr-nkoct steady-state loop DELIVERED (2026-07-03), opt-in
      via** ``RUST_PARALLEL_STEADY=1``\ **.**

      The deferred steady-state loop above is now implemented behind the
      ``RUST_PARALLEL_STEADY`` env flag (default off). One long-lived
      ``RunSession`` (``exploration/scheduler.rs``) spans many ``run()`` calls:
      workers keep their frontiers **resident** across the Python-callback
      boundary, streaming materialized terminals up an mpsc channel instead of
      synchronizing at a per-wave barrier, so a bounce costs one materialize +
      re-inject rather than a full-frontier detach/reattach re-seed. It engages
      only when ``RUST_PARALLEL_STEADY=1`` **and** the driver's address-based,
      no-``until``, no-technique path sets ``parallel_frontier_residency``
      **and** ``RUST_PARALLEL_WORKERS >= 2``; otherwise the wave loop runs
      unchanged (``workers <= 1`` stays byte-identical). On cancel/finalize the
      resident frontier is drained back to ``STASH_ACTIVE`` — the steady-path
      fix for wave-mode **Bug M1** (``residual_drains`` counter). New counters
      ``parallel_bounce_roundtrips`` / ``parallel_resume_reinjects`` /
      ``parallel_residual_drains`` (previously always 0) are now live.

      **Demonstrator** (``synthetic_examples/fork_solve_pbounce_W6_S8_M12_B2``,
      built to bounce only ``2^-k`` of the ``2^W`` leaves per level — the
      partial-bounce case the trap stress bench could not show):

      .. list-table::
         :header-rows: 1

         * - config
           - wall (median of 2)
           - peak_mem
         * - workers=1
           - **56 s**
           - 1088 MB
         * - workers=2 **wave**
           - **TIMEOUT (>240 s)**
           - —
         * - workers=2 **steady**
           - **~55 s**
           - 700 MB
         * - workers=4 **wave**
           - **TIMEOUT (>240 s)**
           - —
         * - workers=4 **steady**
           - **~60 s**
           - 825 MB

      The headline is **steady vs wave**: the wave loop's per-bounce
      full-frontier re-migration **times out** (>4x) on this partial-bounce
      workload, while steady **completes at workers=1 parity with lower peak
      memory**. Steady does **not** beat single-threaded here because the wall
      is bounce-service-dominated (64 serialized Python SimProcedure callbacks
      through the GIL); the parallelizable Z3-solve fraction is not the critical
      path (consistent with the low ``solver_frac`` in the GIL-strategy
      analysis). Correctness is by the **leaf-index projection**, not raw
      content fingerprints — the latter differ across worker counts because
      ``StateMigrationPayload`` does not preserve the Python-exported constraint
      log (an orthogonal migration property); the projected found-set is
      identical (all 64 leaves) across workers=1 and steady workers=2.

      **Why it stays opt-in (default NOT flipped).** Parallel exploration is
      itself opt-in (``RUST_PARALLEL_WORKERS``), and on the CTF corpus it is
      net-negative: most benches are ``num_find=1`` first-find on a wide
      frontier, which is anti-parallel by nature
      (bd ``parallel-numfind1-speculative-waste``) — steady's barrier-free
      exploration *amplifies* the speculative waste (e.g. ``xmllint_getenv``'s
      glibc-init bounce storm). Steady is the right choice **only** for a
      parallel, **exhaustive** (find-all / coverage), **partial-bounce**
      workload, where it converts the wave loop's migration timeout into
      completion. Recommend it there via the env flag; keep ``workers=1`` the
      default for general use.

If Option B is chosen instead, the migration is shorter but the
benchmark-time risk is higher: every existing bench may regress by
the lock-acquisition overhead, with no upside on
``mma_howtouse``-shaped workloads (which dominate the slower-bench
set).


Distributed (cross-machine) migration — measured KILL
----------------------------------------------------

Probed under bead ``angr-op0dn.13.7`` (M5-B17b, cost probe, no code landed).
The question was whether a state could profitably be migrated over a *network*
rather than between threads in one process. It cannot, on the find-all
workloads that motivated parallelism in the first place.

Per-state envelope measured with the shipped ``dump_snapshot`` /
``load_snapshot`` codec (the same ``StashManager`` per-state encoding a
``StateMigrationPayload`` carries) at the exhaustive frontier of each M5-P2
bench; ``work ms/state`` is the single-threaded wall time divided by the
``parallel_tasks`` dispatch count measured at ``workers=2``:

.. list-table::
   :header-rows: 1

   * - bench
     - bytes/state
     - serialize
     - deserialize
     - transfer @1/10 Gbps
     - network migration
     - work/state
   * - ``fork_solve_trap_W5_S8_M12`` (32 leaves)
     - 1.57 MB
     - 10.2 ms
     - 15.2 ms
     - 12.6 / 1.3 ms
     - 37.9 / 26.6 ms
     - 182 ms
   * - ``fork_solve_pbounce_W6_S8_M12_B2`` (74 states)
     - 0.91 MB
     - 4.6 ms
     - 10.3 ms
     - 7.3 / 0.7 ms
     - 22.1 / 15.5 ms
     - 145 ms

Read against the *break-even steal fraction* frame (bd
``migration-count-is-the-lever``: a 14.5 ms in-process transport buys
``f* ~= 10.5-13.3%``, i.e. at most ~1 state in 9 may migrate), a network
migration costing 15.5–37.9 ms shrinks that budget to **f\* ~= 4–12%**. The
steal fraction actually observed on these same benches is **76% (pbounce) /
87.6% (trap)** — 6–20x over even the most generous 10 Gbps budget. The
literal gate ("network cost < work-per-state") passes only because per-state
*work* here is large (145–182 ms); the binding constraint is the migration
*count*, and it is already violated in-process.

Two further costs are excluded from the table and both point the same way:

* The ``Py<PyAny>`` overlays a ``StateMigrationPayload`` carries
  (``symbolic_pages``, ``hook_symbolic_memory``, ``addr_to_ast``) are *not*
  in the snapshot envelope, so bytes/state above is a **lower bound** for a
  distributed transport, which would additionally have to pickle them.
* Python SimProcedure bounces are driver-local. A remote worker would have to
  round-trip to the driver on every hook — the trap benches bounce the whole
  frontier ``T=4`` times per leaf by construction.

**Verdict: KILL.** Distributed migration is strictly worse than the
in-process case that already loses. Do not file a distributed follow-on
without a human decision.


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
  still stepping should be able to signal "stop". The shipped scheduler
  (``exploration/scheduler.rs``) uses a shared ``AtomicBool``
  (``CancelToken``) checked at task boundaries — the migration grain — so a
  worker finishes its current task then stops. Z3-level mid-solve
  interruption (``Context::handle().interrupt()``; ``ContextHandle`` is
  ``Send + Sync``) is available and is the right tool to abort a *long
  in-flight solve*, but fanning it out safely requires keeping each worker's
  context alive past every possible interrupter — coupling that belongs with
  the run-loop integration, where solve durations matter. Deferred.
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
