Rust ↔ Python shared Z3 context
===============================

The Rust engine and Python's ``z3-solver`` package load the **same**
``libz3.so`` shared library and thread-local Z3 context. This is what
lets a claripy AST built on the Python side be re-asserted by Rust as
a raw ``Z3_ast`` pointer without rebuild, and conversely lets
``RustSolverContext.eval(...)`` return a value Python can wrap into a
``claripy.BVV`` directly. This page documents how the sharing is wired
in, the lifecycle and push/pop discipline that every consumer must
respect, and the counters exposed for inspecting the resulting solver
state.

This page is companion to :doc:`rust_engine` (overview, options,
benchmarks). See also the "Shared-lineage Z3 solver — rejected"
section in :doc:`rust_engine` for why a cross-state shared *solver*
(distinct from the shared *context* documented here) was investigated
and ultimately landed as opt-in only.

Why a single libz3.so is mandatory
----------------------------------

Z3 represents every AST as a ``Z3_ast`` pointer into a parent
``Z3_context``. AST pointers are **only** valid within their owning
context, and the context tracks its own reference-counted heap.
A Rust process that linked against a *different* ``libz3.so`` than the
Python ``z3-solver`` package would see the two contexts as separate
universes: claripy's ``z3.BoolRef.ast`` pointer would point into
heap memory the Rust-side ``z3::Context`` knows nothing about,
producing memory corruption or silent assertion drops the moment the
pointer is dereferenced.

The native fast paths in ``native/angr/src/solver.rs`` (e.g.
``RustSolverContext::add_constraint_ast``) extract the raw ``Z3_ast``
out of claripy's ``z3`` backend and call ``Z3_solver_assert`` on the
Rust side. This is sound only because the build pipeline guarantees
both sides resolve ``libz3.so`` to the same on-disk file at process
startup.

How the link is wired (build.rs + venv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``native/angr/build.rs::find_z3_lib_dir`` discovers the Z3 install in
this order, then emits ``cargo:rustc-link-arg=-Wl,-rpath,<lib_dir>``
so the compiled Rust ``.so`` carries the venv's library directory
baked into its ``DT_RUNPATH``:

1. ``Z3_LIBRARY_PATH_OVERRIDE`` environment variable (escape hatch
   for non-standard installs).
2. The active Python's ``z3`` package directory (``python3 -c "import
   z3, os; print(os.path.dirname(z3.__file__))"`` then ``/lib``), as
   long as ``libz3.so`` / ``libz3.dylib`` / ``z3.dll`` lives there —
   this is the production path when the build runs inside the
   project venv.
3. System fallbacks: ``/usr/lib/x86_64-linux-gnu``, ``/usr/lib64``,
   ``/usr/lib``, ``/usr/local/lib``, ``/opt/homebrew/lib``.

Because the runpath is hard-coded at link time, ``import angr`` works
from inside the venv without any ``LD_LIBRARY_PATH`` setup. Outside
the venv, point ``LD_LIBRARY_PATH`` at the venv's ``z3/lib`` or set
``Z3_LIBRARY_PATH_OVERRIDE`` to override discovery for the next build.

Header discovery (``Z3_SYS_Z3_HEADER``) is a separate concern handled
by ``setup.py::_resolve_z3_header`` and ``z3-sys``'s own pkg-config
probe — see the "Common Issues" section of ``CLAUDE.md`` for
troubleshooting. The runtime sharing documented here is independent
of header discovery and applies to any working build.

Thread-local Z3 context
~~~~~~~~~~~~~~~~~~~~~~~

``z3-rs 0.19`` made the Z3 context thread-local: every ``SymContext``
constructed on the same thread shares one ``Z3_context``. This is
both an opportunity (raw-AST passthrough, ``Z3_inc_ref``-free reuse)
and a constraint:

* **Single-threaded by design.** Crossing a Z3 AST or solver across
  OS threads is undefined. The Rust engine is single-threaded today
  (parallel exploration is tracked in :doc:`rust_parallel_design`).
* **``Send``/``Sync`` audit.** ``RustSolverContext`` is
  ``#[pyclass(unsendable)]`` precisely so PyO3 refuses to ship it to
  another thread.
* **Pointer-equality after fork.** ``SymContext::fork()`` returns a
  child with a fresh ``Solver`` but the same context, so AST handles
  created on the parent remain valid on the child without translation.

The fork side of this is what makes :ref:`rust-z3-sharing-fork-lazy`
correct: the child can copy the parent's ``z3_assertions_shared``
``Arc<Vec<Bool>>`` by ``Arc::clone`` and replay assertions lazily
without ever asking Z3 to translate them, because both contexts ARE
the same context.

Lifecycle of a ``SymContext``
-----------------------------

``SymContext`` (``native/angr/src/symbolic/context.rs``) is the
single-state solver wrapper. Each ``RustSimState`` owns one;
``RustSolverContext`` (Python-exposed) wraps one too. Production
states travel through three phases:

1. **Construction.** ``SymContext::new()`` allocates a
   ``z3::Solver`` configured by ``build_solver(timeout_ms)`` with the
   ``QF_BV``/``QF_ABV`` tactic pipeline. The solver is wrapped in
   ``Mutex<Option<Solver>>`` so it can be lazily dropped/rebuilt on
   fork. ``timeout_ms`` defaults to 30000 (30 s) per query.
2. **Constraint accumulation.** ``add_constraint_raw`` / ``assume_*``
   append to two parallel structures: a frozen ``Arc<Vec<Bool>>``
   (``z3_assertions_shared``) and a per-fork ``Vec<Bool>``
   (``LocalConstraints::z3_assertions``). The split exists so a fork
   with no new constraints is an ``Arc::clone`` instead of a deep
   copy — see the ``arc-shared-z3-cache`` bd memory for design
   history.
3. **Fork.** ``SymContext::fork()`` "freezes self": when not inside
   a ``push()`` transaction, ``local_constraints.z3_assertions`` is
   drained into ``z3_assertions_shared`` (under ``Arc::get_mut``
   when uniquely owned, otherwise a clone). The child inherits the
   resulting ``Arc`` by ``Arc::clone`` (O(1)) and starts with an
   empty ``LocalConstraints``. The child's ``solver: Mutex<None>`` is
   lazily materialized on first query — see below.

.. _rust-z3-sharing-fork-lazy:

Lazy solver materialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A forked ``SymContext`` does **not** eagerly build a Z3 solver:
``solver`` is ``Mutex<Option<Solver>>``, and remains ``None`` until
the first ``with_z3_solver`` call. This matters for benchmarks that
fork many states that are immediately pruned or avoided — those
states never pay the O(n) assertion-replay cost.

When the solver is materialized, it replays every entry of
``z3_assertions_shared`` followed by every entry of
``local_constraints.z3_assertions``. The replay cost is tracked by
the ``z3_materialize_count`` and ``z3_materialize_time_ns`` counters.

Why not ``Z3_solver_translate``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Z3 exposes ``Z3_solver_translate`` (``Solver::clone`` in z3-rs) which
*looks* like it would skip the replay by copying the parent's
assertion list directly. It is structurally unsuitable: when source
and destination contexts are the same — the case for every fork under
our shared-context design — ``Z3_solver_translate`` returns a solver
with **zero** assertions, silently losing the entire constraint set.
See the ``avoid-z3-solver-translate-same-context`` bd memory for the
reproducer and verdict.

Push/pop discipline
-------------------

Two distinct push/pop systems coexist on a ``SymContext`` and must
not be confused:

1. **Transactional pushes** (``push_level``, ``transaction_begin`` /
   ``transaction_commit`` / ``transaction_rollback``). Used by the
   Python-side ``RustStateProxy`` and the constraint-sync pipeline
   to roll back tentative changes when sync fails. The level is
   tracked separately from the Z3 stack.
2. **Scope-path savepoints** (``scope_savepoint_push`` /
   ``scope_savepoint_pop``). Used by ``push()``/``pop()`` and by the
   ``SharedLineageSolver`` integration to save and restore the
   per-state Z3 solver scope.

``push()`` calls ``scope_savepoint_push()`` then invalidates
``sat_cache`` / ``model_cache``; ``pop()`` is symmetric. The
dispatch inside ``scope_savepoint_*`` keys on whether this
``SymContext`` has a lineage (``self.lineage`` is ``Some``):

* **None branch (production today).** Directly issues
  ``solver.push()`` / ``solver.pop(1)`` against the per-context Z3
  solver, and bumps ``bare_z3_push_depth`` in lockstep. This is the
  pre-lineage behavior that ``RustSolverContext::push()`` /
  ``::pop()`` and ``transaction_begin`` rely on.
* **Some branch (opt-in only).** Records a depth marker on
  ``scope_savepoints`` (an in-memory ``Vec<usize>``) and defers the
  Z3-side push to the lazy ``SharedLineageSolver::switch_to`` the
  next time a query fires. ``bare_z3_push_depth`` is left untouched
  along this branch.

Symmetry rules:

* Every ``push()`` MUST be balanced by exactly one ``pop()``. In the
  None branch z3-rs panics on under-pop; in the Some branch the
  ``scope_savepoints`` vector silently no-ops on an empty pop.
* ``fork()`` resets ``push_level`` to 0 on the child but copies the
  parent's ``bare_z3_push_depth`` value forward. Under today's fork
  semantics a child observes ``bare_z3_push_depth == 0`` unless
  future code threads bare pushes across fork; see the
  ``invariant-bare-z3-push-depth`` bd memory.

Interaction with ``SharedLineageSolver``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SharedLineageSolver`` (``native/angr/src/symbolic/lineage.rs``)
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
ownership of the shared solver would corrupt the parent's transaction.
The counter is consulted in three places:

* The fork-time materialization gate
  (``SymContext::fork``, line ~3400) reads the parent counter.
* The ``test_fork_inside_push_isolation`` unit test exercises the
  guard.
* ``bare_z3_push_depth()`` is exposed publicly for telemetry; the
  invariants on the public reader are tracked in the
  ``invariant-bare-z3-push-depth`` bd memory.

See the "Shared-lineage Z3 solver — rejected" section of
:doc:`rust_engine` for the full design verdict, BFS-thrash measurements,
and why the kwarg ships default-off.

Memory limits and bounded structures
------------------------------------

Z3 itself takes its memory limit from process resource limits; the
benchmarking harness sets ``RLIMIT_AS = 4 GB`` per subprocess in
``tests/benchmarks/run_single.py`` to keep the 8 GB CI box safe.
Inside that envelope, two engine-side structures grow with
exploration and need a mental model:

* **``z3_assertions_shared`` ``Arc<Vec<Bool>>``** holds every
  constraint ever asserted on the lineage of forks leading to this
  state. Each entry is a refcounted Z3 AST handle (``Z3_inc_ref``).
  Memory growth is linear in constraint count and shared across
  forks via ``Arc::clone``, so a 1000-state BFS sees one ``Vec``
  per distinct lineage, not 1000 copies. Drop happens on the last
  ``Arc::drop``, which usually coincides with the last
  ``RustSimState`` for that lineage being pruned from a stash.
* **``bare_z3_push_depth: AtomicUsize``** is just a counter, but
  observe: a runaway value indicates an unbalanced ``push`` /
  ``pop`` pair on the per-context solver. Z3 itself will eventually
  refuse to ``pop`` beyond level 0 (panic in z3-rs); the counter
  surfaces the imbalance earlier via the fork-time gate.

``z3_materialize_count`` divided by the number of forked states
gives the fraction of forked solvers that actually fired a query
(materialized). Benchmarks where this ratio is low (~10%) are
the ones that benefit most from lazy materialization; benchmarks at
~100% are paying the full assertion-replay cost on every fork.

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

Counter cheat-sheet relevant to shared-context behavior:

* ``z3_check_count`` / ``z3_check_time_ns`` — wall time the
  process spent inside ``solver.check()``.
* ``z3_materialize_count`` / ``z3_materialize_time_ns`` — number
  of lazy-fork solver materializations and total replay time. A
  low ratio of materializations to forks confirms the lazy path
  is helping.
* ``z3_ast_build`` — total ``Z3_mk_*`` calls. Useful for spotting
  benchmarks that hit the raw-AST passthrough path (low value) vs.
  benchmarks that fall through to ``claripy_to_rustbv`` AST rebuild
  (high value).
* ``z3_assume_concrete`` / ``z3_assume_symbolic`` —
  ``assume_true`` / ``assume_false`` fast-path counters.
* ``z3_branch_*`` — per-branch evaluation breakdown.
* ``z3_site_<name>_count`` / ``z3_site_<name>_time_ns`` —
  per-call-site time attribution. Sites: ``satisfiable``,
  ``branch_true``, ``branch_false``, ``eval``, ``eval_upto``,
  ``min_init``, ``min_search``, ``max_init``, ``max_search``.

Counters are process-wide ``AtomicU64`` statics, so
``mgr.reset_solver_stats()`` zeros them across every active
``SymContext``. They are also zero-cost when not read — the
counter design from ``angr-2j5v`` / verified in ``angr-pdq8``
guarantees an unread counter contributes only the
``fetch_add(1, Relaxed)`` cost on its hot path.

Process-level Z3 telemetry beyond these counters (per-tactic time,
formula simplification stats) is not exposed today. Use
``Z3_LOG_*`` environment variables or attach ``perf`` /
``cargo flamegraph`` per :doc:`rust_engine` (the "Profiling Rust
Benches" section in ``CLAUDE.md``) if you need finer attribution.
