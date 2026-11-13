Z3 contexts in the Rust engine
==============================

The Rust engine solves with its own Z3 contexts, one per thread, through
the upstream ``z3`` crate. Python never hands it a Z3 pointer: claripy is
clarirs, so a claripy AST reaching the engine is a clarirs term, and
``symbolic::z3_lower`` rebuilds it in the calling thread's Z3
context (``Context::thread_local()``), memoizing each lowered node while
the node lives. ``RustSolverContext.eval(...)`` returns plain values that
Python wraps into ``claripy.BVV``. This page documents the context
lifecycle, the push/pop discipline every consumer must respect, and the
counters exposed for inspecting solver state.

This page is companion to :doc:`rust_engine` (overview, options,
benchmarks). See :doc:`rust_engine_decisions` for why a cross-state
shared *solver* was investigated and ultimately landed as opt-in only.

Where Z3 ASTs come from
-----------------------

Z3 represents every AST as a ``Z3_ast`` pointer into a parent
``Z3_context``, valid only within that context. The engine keeps every
AST it builds in the thread-local context of the thread that built it:
``RustBV`` values are clarirs terms with a lazy per-value Z3 memo, and
a claripy constraint from Python is lowered by
``native/angr/src/claripy_bridge/lower.rs::lower_claripy_constraint``,
which checks its sort, before ``Z3_solver_assert`` sees it. clarirs' own Z3 backend (``clarirs-z3``) keeps separate contexts of
its own; the two never exchange pointers.

The solver surface is split across the files below (angr-9ke6b.205):

* ``native/angr/src/solver.rs`` — the Python-boundary error mapping and
  the claripy-AST API (``add_constraint_ast``, ``eval*``, ``min``,
  ``max``, ``push``/``pop``, ``fork``).
* ``native/angr/src/solver/z3_ast_extract.rs`` (``solver::z3_ast_extract``)
  — claripy-node evaluation (``node_to_eval_bv``,
  ``eval_claripy_node``). Unlike ``solver.rs`` it exports *no*
  ``#[pymethods]``; these are internal helpers the claripy-AST API in
  ``solver.rs`` calls. Both files ``forbid(unsafe_code)``.

How the link is wired (build.rs + venv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``native/angr/build.rs`` emits ``cargo:rustc-link-arg=-Wl,-rpath,<dir>``
for ``Z3_LIBRARY_PATH_OVERRIDE`` and nothing else: no probing of the
active Python, no system fallbacks, nothing when the variable is unset.
The extension does not depend on that runpath when imported from Python,
because ``angr/_z3.py`` opens the ``z3-solver`` wheel's libz3 before
``angr.rustylib`` loads; the runpath is for the ``cargo test`` binaries.

Which Z3 version ``z3-sys`` believes it links is a separate, build-time
concern handled by ``setup.py::configure_z3`` and
``tools/rebuild-rust.sh`` — see *Build layout* in :doc:`rust_build`.
The runtime sharing documented here applies to any working build.

Thread-local Z3 context
~~~~~~~~~~~~~~~~~~~~~~~

``z3-rs`` (from 0.19) made the Z3 context thread-local: every ``SymContext``
constructed on the same thread shares one ``Z3_context``. This is
both an opportunity (ASTs reuse across states on a thread without
translation) and a constraint:

* **One context per thread, never crossed.** Crossing a Z3 AST or
  solver between OS threads is undefined. The engine does support
  multi-worker exploration today — ``RustExplorationManager(project,
  parallel_workers=N)`` / ``RUST_PARALLEL_WORKERS``, see
  :doc:`rust_engine` — and honors the rule by giving each worker its
  own thread-local context. A worker's live successors stay in a
  thread-private queue and are never serialized; a state that *is*
  handed to another worker (stolen surplus, or a materialized
  found/matched terminal) is detached into a ``Send`` payload and
  ``reattach``-ed into the destination's own context with every AST
  minted locally, so no context is ever read cross-thread. See
  ``native/angr/src/exploration/scheduler_pool.rs`` for the mechanism and
  :doc:`rust_parallel_design` for the design rationale.
* **``Send``/``Sync`` audit.** ``RustSolverContext`` is
  ``#[pyclass(unsendable)]`` precisely so PyO3 refuses to ship it to
  another thread.
* **Pointer-equality after fork.** ``SymContext::fork()`` returns a
  child with no solver of its own yet but the same context, so AST handles
  created on the parent remain valid on the child without translation.

The fork side of this is what makes :ref:`rust-z3-sharing-fork-lazy`
correct: the child can copy the parent's ``z3_assertions_shared``
(a ``Mutex<Arc<Vec<Asserted>>>``) by ``Arc::clone`` and replay assertions lazily
without ever asking Z3 to translate them, because both contexts ARE
the same context.

Lifecycle of a ``SymContext``
-----------------------------

``SymContext`` (``native/angr-symex-core/src/symbolic/context.rs``) is the
single-state solver wrapper. Each ``RustSimState`` owns one;
``RustSolverContext`` (Python-exposed) wraps one too. Production
states travel through three phases:

1. **Construction.** ``SymContext::new()`` builds no solver. The
   context holds a ``SolverSlot``, its identity in the per-thread
   solver cache (see :ref:`rust-z3-sharing-fork-lazy`), and a
   ``timeout_ms`` that defaults to 30000 (30 s) per query.
2. **Constraint accumulation.** ``add_constraint_lowered`` / ``assume_*``
   append to two parallel structures: a frozen ``Arc<Vec<Asserted>>``
   (``z3_assertions_shared``) and a per-fork ``Vec<Asserted>``
   (``LocalConstraints::z3_assertions``). The split exists so a fork
   with no new constraints is an ``Arc::clone`` instead of a deep
   copy. An entry holds the constraint's Z3 term and its origin, the
   form it can be rebuilt from without Z3: an ``assume`` pair, a clarirs
   term, or the value and constant of an address concretization. No
   sink takes a Z3 term alone.
3. **Fork.** ``SymContext::fork()`` "freezes self": when not inside
   a ``push()`` transaction, ``local_constraints.z3_assertions`` is
   drained into ``z3_assertions_shared`` (under ``Arc::get_mut``
   when uniquely owned, otherwise a clone). The child inherits the
   resulting ``Arc`` by ``Arc::clone`` (O(1)) and starts with an
   empty ``LocalConstraints`` and a new ``SolverSlot``, so it has no
   solver until its first query — see below.

.. _rust-z3-sharing-fork-lazy:

The solver cache
~~~~~~~~~~~~~~~~

A ``SymContext`` owns no Z3 solver. Its constraints are the assertion
log (``z3_assertions_shared`` followed by
``local_constraints.z3_assertions``) and its open ``push()``
savepoints; a solver is a cache of that list, kept per thread in
``symbolic/solver_cache.rs``.

* **Adding a constraint** appends to the log and makes no Z3 call. A
  state that is forked, constrained and pruned without a query never
  builds a solver.
* **A query** (``with_z3_solver``) takes the context's solver out of the
  cache, or builds one with ``build_solver(timeout_ms)`` — a plain
  ``z3::Solver::new()`` unless ``ANGR_Z3_TACTIC`` selects a tactic
  pipeline (see :doc:`rust_engine`). ``SymContext::sync_solver`` then
  brings it up to date: it pops the frames of scopes the context has
  closed, asserts the log entries the solver has not seen, and pushes a
  frame for each savepoint opened since. A tracked constraint is
  asserted under the tracker its log entry carries: the entry holds the
  tracking literal's name, and the solver mints the literal from it, so
  every solver built from the log tracks the entry under the same one.
* **``push()`` / ``pop()``** record and remove a savepoint and truncate
  the logs. The solver's frames follow at the next query, so a scope
  closed before any query never reaches Z3.
* **Eviction.** A thread keeps at most ``ANGR_RUST_SOLVER_CACHE_SIZE``
  built solvers (``solver_cache::DEFAULT_CAPACITY`` when unset), least
  recently used first out, and a context's solver is dropped with the
  context. An evicted context rebuilds at its next query from the same
  list, open scopes included. The capacity trades memory for rebuilds:
  a frontier wider than the capacity rebuilds a solver each time a
  parked state is queried again.
* **Ownership.** A solver serves the one context whose slot names it. A
  fork does not inherit it.

``z3_materialize_count`` and ``z3_materialize_time_ns`` count solver
builds and the time to build and fill them; ``z3_solver_cache_evict``
counts evictions.

The model cache
~~~~~~~~~~~~~~~

A ``SymContext`` owns no Z3 model either. Its two witness slots
(``model_cache`` / ``alt_model_cache``) each hold a ``ModelRef``, a
token that is plain data, and the model it names is in the thread's
table in ``symbolic/model_cache.rs``.

* **A fork** clones the token, so parent and child read one model.
* **A read** goes through ``SymContext::cached_model``, which resolves
  the token in the current thread's table. A token whose model is not
  there, because it was evicted or was stored on another thread, is an
  empty slot: the query runs a check, as it does for a context that
  never had a witness. A held token is never taken as proof of a model.
* **Lifetime.** A model leaves the table when the last clone of its
  token is dropped on that thread. The table also keeps at most
  ``ANGR_RUST_MODEL_CACHE_SIZE`` models
  (``model_cache::DEFAULT_CAPACITY`` when unset), least recently read
  first out, which bounds the models whose tokens were dropped
  elsewhere.

``z3_model_cache_evict`` counts evictions. The capacity is a backstop
and not a memory control: a lost witness costs the checks it was
saving, and the states that then query build solvers they did not need
before, so a table smaller than the live witnesses raises both check
counts and peak memory.

Why not ``Z3_solver_translate``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Z3 exposes ``Z3_solver_translate`` (``Solver::clone`` in z3-rs) which
*looks* like it would skip the replay by copying the parent's
assertion list directly. It is structurally unsuitable: when source
and destination contexts are the same — the case for every fork under
our one-context-per-thread design — ``Z3_solver_translate`` returns a
solver with **zero** assertions, silently losing the entire constraint
set (``angr-59jk.2``); this engine therefore never uses
``Z3_solver_translate`` to fork a solver.

Push/pop discipline
-------------------

Scope save/restore on a ``SymContext`` is a **single** system today:
``push()`` / ``pop()`` / ``try_pop()``
(``native/angr-symex-core/src/symbolic/transaction_ops.rs``), layered over the
``scope_savepoint_push`` / ``scope_savepoint_pop`` helpers
(``native/angr-symex-core/src/symbolic/lineage_ops.rs``). ``push()`` calls
``scope_savepoint_push()`` then invalidates ``sat_cache`` and both
cached witnesses (``model_cache`` / ``alt_model_cache``); ``pop()`` is
symmetric.

.. note::

   There is no second, higher-level ``transaction_begin`` /
   ``transaction_commit`` / ``transaction_rollback`` lifecycle and no
   ``push_level`` counter (``angr-ph300.44``, ``angr-c7xno.75``). That
   lifecycle had zero callers and a latent corruption bug (commit popped
   one of the three aux stacks and never released the Z3 frame). No Rust
   or Python code calls or reads any of those names; they survive only in
   historical comments on ``SymContext`` and in ``transaction_ops.rs``'s
   module doc. A future feature needing transactional scoping should
   build on bare ``push()``/``pop()`` rather than resurrect a
   ``push_level``-style counter (bd memory `invariant-transaction-api-removed`).

Every ``push()`` records the current ``(z3_assertions, assumed,
non_bv_assertions)`` lengths on ``bare_local_savepoints`` so the
matching ``pop()`` truncates everything logged inside the scope, and
clears the dedup side-table so a re-assert after the pop is not
falsely suppressed (``angr-ph300.41`` / ``angr-ph300.42``). Beyond that
bookkeeping, the dispatch inside ``scope_savepoint_*`` keys on whether
this ``SymContext`` has a lineage (``self.lineage`` is ``Some``):

* **None branch (production today).** Records the savepoint and bumps
  ``bare_z3_push_depth`` in lockstep; the context's solver opens or
  closes the matching Z3 frame at its next query
  (``SymContext::sync_solver``). This is what
  ``RustSolverContext::push()`` / ``::pop()`` rely on.
* **Some branch (opt-in only).** Records a depth marker on
  ``scope_savepoints`` (an in-memory ``Vec<usize>``) and defers the
  Z3-side push to the lazy ``SharedLineageSolver::switch_to`` the
  next time a query fires. ``bare_z3_push_depth`` is left untouched
  along this branch.

Symmetry rules:

* Every ``push()`` MUST be balanced by exactly one ``pop()``. In the
  None branch z3-rs panics on under-pop, so the Python-facing
  ``RustSolverContext::pop()`` routes through ``try_pop()``, which
  refuses the pop (returning ``false``, which ``pop()`` turns into a
  Python exception) when
  ``bare_z3_push_depth == 0`` (``angr-ph300.48``). In the Some branch
  the ``scope_savepoints`` vector silently no-ops on an empty pop.
* ``fork()`` gives the child a *fresh* scope state — empty
  ``bare_local_savepoints`` / ``scope_savepoints`` / ``scope_path``
  and ``bare_z3_push_depth == 0`` — regardless of the parent's depth:
  inheriting a non-zero depth would let ``try_pop()`` hand a ``pop(1)``
  to a child solver that was never pushed, and permanently disqualify
  the child from lineage minting (``angr-c7xno.75``). The counter means
  "open bare push scopes on *this* context's solver" (bd memory `invariant-bare-z3-push-depth`).
* Forking *while* a bare scope is open is safe but deliberately does
  not freeze: ``fork()`` reads ``bare_local_savepoints`` before taking
  ``local_constraints`` and skips draining the local logs into the
  append-only ``*_shared`` Arcs, so a constraint added inside the
  scope cannot outlive the parent's matching ``pop()`` (``angr-c7xno.75``).

Interaction with ``SharedLineageSolver``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SharedLineageSolver`` (``native/angr-symex-core/src/symbolic/lineage.rs``)
shares **one** ``z3::Solver`` across a forked lineage of states by
tracking each state's ``scope_path`` and replaying pushes lazily on
context switch. It is **opt-in only** (the
``use_shared_lineage_solver=True`` kwarg on
``RustExplorationManager``) and has a runtime thrash detector
(``angr-v5ht``) that disables minting at fork time when the
hot-cache hit rate slips below 35%.

A bare ``RustSolverContext.push()`` followed by a Python ``fork()``
call interacts subtly with lineage minting. ``SymContext::fork``
refuses to mint a fresh lineage when ``bare_z3_push_depth > 0`` —
exposing the unbalanced Z3-stack frame to a sibling that took over
ownership of the shared solver would corrupt the parent's scope.
The counter is consulted in three places:

* The fork-time materialization gate in ``SymContext::fork``
  (``native/angr-symex-core/src/symbolic/snapshot_fork_ops.rs``) reads the parent
  counter.
* ``SymContext::try_pop`` (``transaction_ops.rs``) uses it as the
  under-pop guard described above.
* ``bare_z3_push_depth()`` is exposed publicly for telemetry (read by
  ``exploration/state_api.rs::_debug_solver_info``). A reader must know
  that the counter covers only this context's per-context solver (it
  stays 0 on the shared-lineage branch, which records on
  ``scope_savepoints``), and that it is not the "is a scope open?"
  indicator: ``bare_local_savepoints`` is, since it records on both
  branches, and that is what ``fork()``'s freeze gate reads.

``test_fork_skips_mint_when_bare_push_outstanding``
(``symbolic/context_tests/smtlib2_snapshot.rs``) exercises the
fork-time guard.

See :doc:`rust_engine_decisions` for the design verdict and why the
kwarg ships default-off.

Memory limits and bounded structures
------------------------------------

Z3 itself takes its memory limit from process resource limits; the
benchmarking harness sets ``RLIMIT_AS = 4 GB`` per subprocess in
``tests/benchmarks/run_single.py`` to keep the CI box safe.
Inside that envelope, two engine-side structures grow with
exploration and need a mental model:

* **``z3_assertions_shared`` (``Mutex<Arc<Vec<Asserted>>>``)** holds every
  constraint ever asserted on the lineage of forks leading to this
  state. Each entry is a refcounted Z3 AST handle (``Z3_inc_ref``).
  Memory growth is linear in constraint count and shared across
  forks via ``Arc::clone``, so a 1000-state BFS sees one ``Vec``
  per distinct lineage, not 1000 copies. Drop happens on the last
  ``Arc::drop``, which usually coincides with the last
  ``RustSimState`` for that lineage being pruned from a stash.
* **``bare_z3_push_depth: AtomicUsize``** is just a counter, but
  observe: a runaway value indicates an unbalanced ``push`` /
  ``pop`` pair on the context. An unmatched ``pop`` is ignored; the
  counter surfaces a missing one via the fork-time gate.
* **Built solvers** are bounded per thread by the solver cache's
  capacity, not by the number of live states.

``z3_materialize_count`` divided by the number of forked states
gives the solvers built per fork. It is low when most forks are never
queried, and above one when ``z3_solver_cache_evict`` is high: the
frontier is wider than the cache and parked states rebuild.

Inspecting solver state
-----------------------

Read the per-process counters via
``RustExplorationManager.get_solver_stats()`` — the same dict is
merged into ``mgr.stats`` for convenience:

.. code-block:: python

   import angr
   from angr.exploration import RustExplorationManager

   proj = angr.Project(binary, auto_load_libs=False)
   mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
   mgr.reset_solver_stats()
   mgr.explore(find=target_addr)

   stats = mgr.get_solver_stats()
   print(f"checks:          {stats['z3_check_count']:>8d}")
   print(f"check time (ms): {stats['z3_check_time_ns'] / 1e6:>8.1f}")
   print(f"materializes:    {stats['z3_materialize_count']:>8d}")
   print(f"assume conc:     {stats['z3_assume_concrete']:>8d}")
   print(f"assume sym:      {stats['z3_assume_symbolic']:>8d}")

Counter cheat-sheet relevant to per-context solver behavior:

* ``z3_check_count`` / ``z3_check_time_ns`` — wall time the
  process spent inside ``solver.check()``.
* ``z3_materialize_count`` / ``z3_materialize_time_ns`` — solvers
  built and the time to build and fill them from the assertion log.
* ``z3_solver_cache_evict`` — built solvers dropped to keep a thread
  within the solver cache's capacity.
* ``z3_model_cache_evict`` — models dropped to keep a thread within
  the model cache's capacity; each is a witness some context no longer
  has.
* ``z3_ast_build`` — number of *top-level* ``to_z3_ast()`` /
  ``to_z3_bool()`` entry calls, **not** the number of ``Z3_mk_*`` FFI
  calls: one entry lowers a whole clarirs term, so one increment can
  cover hundreds of Z3 node constructions on a deep expression. Do not
  read it as node-construction volume.
* ``z3_bool_build`` — the ``to_z3_bool()`` share of ``z3_ast_build``.
  Lowering goes through ``symbolic::z3_lower``, whose per-thread cache
  keyed by clarirs node is the only lowering cache, so a repeat lowering
  of a live term rebuilds no Z3 node.
* ``z3_assume_concrete`` / ``z3_assume_symbolic`` —
  ``assume_true`` / ``assume_false`` fast-path counters.
* ``z3_branch_*`` — per-branch evaluation breakdown.
* ``z3_site_<name>_count`` / ``z3_site_<name>_time_ns`` —
  per-call-site time attribution. Sites: ``satisfiable``,
  ``branch_true``, ``branch_false``, ``eval``, ``eval_upto``,
  ``min_init``, ``min_search``, ``max_init``, ``max_search``,
  ``unsat_core``.

Counters are process-wide ``AtomicU64`` statics, so
``mgr.reset_solver_stats()`` zeros them across every active
``SymContext``. They are also zero-cost when not read — the
counter design (``angr-2j5v`` / ``angr-pdq8``)
guarantees an unread counter contributes only the
``fetch_add(1, Relaxed)`` cost on its hot path.

Process-level Z3 telemetry beyond these counters (per-tactic time,
formula simplification stats) is not exposed today. Use
``Z3_LOG_*`` environment variables or attach ``perf`` /
``cargo flamegraph`` per the "Profiling Rust benches" section of
:doc:`rust_build` if you need finer attribution.
