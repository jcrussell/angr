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

The native fast paths (e.g. ``RustSolverContext::add_constraint_ast``)
extract the raw ``Z3_ast`` out of claripy's ``z3`` backend via
``native/angr/src/solver/z3_ast_extract.rs::extract_z3_ast_ptr`` and call
``Z3_solver_assert`` on the Rust side. This is sound only because the
build pipeline guarantees both sides resolve ``libz3.so`` to the same
on-disk file at process startup.

The solver surface is split across three files (angr-9ke6b.205), two of
which contribute ``#[pymethods]`` to the same ``RustSolverContext``:

* ``native/angr/src/solver.rs`` — the Python-boundary error mapping and
  the claripy-AST API (``add_constraint_ast``, ``eval*``, ``min``,
  ``max``, ``push``/``pop``, ``fork``).
* ``native/angr/src/solver/handle_api.rs`` — the handle-based
  claripy-bypass API (symbol-table lifecycle plus the ``op_*``
  arithmetic wrappers), which never builds a claripy AST at all.
* ``native/angr/src/solver/z3_ast_extract.rs`` — raw ``Z3_ast``-pointer
  extraction, sort-checked wrapping and evaluation; every ``unsafe`` on
  this surface lives here. Unlike the other two it exports *no*
  ``#[pymethods]``: ``Z3AstPtr`` owns a Z3 refcount and is not
  PyO3-bridgeable, so these are internal helpers the claripy-AST API in
  ``solver.rs`` calls.

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
   copy.
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
A standalone reproducer lives at
``native/angr/tests/z3_translate_spike.rs``; this engine therefore never
uses ``Z3_solver_translate`` to fork a solver.

Push/pop discipline
-------------------

Scope save/restore on a ``SymContext`` is a **single** system today:
``push()`` / ``pop()`` / ``try_pop()``
(``native/angr/src/symbolic/transaction_ops.rs``), layered over the
``scope_savepoint_push`` / ``scope_savepoint_pop`` helpers
(``native/angr/src/symbolic/lineage_ops.rs``). ``push()`` calls
``scope_savepoint_push()`` then invalidates ``sat_cache`` /
``model_cache``; ``pop()`` is symmetric.

.. note::

   A second, higher-level ``transaction_begin`` /
   ``transaction_commit`` / ``transaction_rollback`` lifecycle used to
   coexist with this one, tracked by its own ``push_level`` counter.
   It was **removed** in ``angr-ph300.44`` — it had zero callers and a
   latent corruption bug (commit popped one of the three aux stacks and
   never released the Z3 frame) — and the ``push_level`` field itself
   was deleted in ``angr-c7xno.75``. No Rust or Python code calls or
   reads any of them; the names survive only in historical comments on
   ``SymContext`` and in ``transaction_ops.rs``'s module doc. A future
   feature needing transactional scoping should build on bare
   ``push()``/``pop()`` rather than resurrect a ``push_level``-style
   counter — see the ``invariant-transaction-api-removed`` bd memory.

Every ``push()`` records the current ``(z3_assertions, assumed,
non_bv_assertions)`` lengths on ``bare_local_savepoints`` so the
matching ``pop()`` truncates everything logged inside the scope, and
clears the dedup side-table so a re-assert after the pop is not
falsely suppressed (``angr-ph300.41`` / ``.42``). Beyond that
bookkeeping, the dispatch inside ``scope_savepoint_*`` keys on whether
this ``SymContext`` has a lineage (``self.lineage`` is ``Some``):

* **None branch (production today).** Directly issues
  ``solver.push()`` / ``solver.pop(1)`` against the per-context Z3
  solver, and bumps ``bare_z3_push_depth`` in lockstep. This is the
  pre-lineage behavior that ``RustSolverContext::push()`` /
  ``::pop()`` rely on.
* **Some branch (opt-in only).** Records a depth marker on
  ``scope_savepoints`` (an in-memory ``Vec<usize>``) and defers the
  Z3-side push to the lazy ``SharedLineageSolver::switch_to`` the
  next time a query fires. ``bare_z3_push_depth`` is left untouched
  along this branch.

Symmetry rules:

* Every ``push()`` MUST be balanced by exactly one ``pop()``. In the
  None branch z3-rs panics on under-pop, so the Python-facing
  ``RustSolverContext::pop()`` / ``::pop_to_level()`` route through
  ``try_pop()``, which refuses the pop (returning ``false``, which
  those methods turn into a Python exception) when
  ``bare_z3_push_depth == 0`` (``angr-ph300.48``). In the Some branch
  the ``scope_savepoints`` vector silently no-ops on an empty pop.
* ``fork()`` gives the child a *fresh* scope state — empty
  ``bare_local_savepoints`` / ``scope_savepoints`` / ``scope_path``
  and ``bare_z3_push_depth == 0`` — regardless of the parent's depth
  (``angr-c7xno.75``: inheriting a non-zero depth let ``try_pop()``
  hand a ``pop(1)`` to a child solver that was never pushed, and
  permanently disqualified the child from lineage minting). The
  counter means "open bare push scopes on *this* context's solver";
  see the ``invariant-bare-z3-push-depth`` bd memory.
* Forking *while* a bare scope is open is safe but deliberately does
  not freeze: ``fork()`` reads ``bare_local_savepoints`` before taking
  ``local_constraints`` and skips draining the local logs into the
  append-only ``*_shared`` Arcs, so a constraint added inside the
  scope cannot outlive the parent's matching ``pop()``
  (``angr-c7xno.75``).

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
ownership of the shared solver would corrupt the parent's scope.
The counter is consulted in three places:

* The fork-time materialization gate in ``SymContext::fork``
  (``native/angr/src/symbolic/snapshot_fork_ops.rs``) reads the parent
  counter.
* ``SymContext::try_pop`` (``transaction_ops.rs``) uses it as the
  under-pop guard described above.
* ``bare_z3_push_depth()`` is exposed publicly for telemetry (read by
  ``exploration/state_api.rs::_debug_solver_info``); the invariants on
  the public reader are tracked in the
  ``invariant-bare-z3-push-depth`` bd memory.

``test_fork_skips_mint_when_bare_push_outstanding``
(``symbolic/context_tests/smtlib2_snapshot.rs``) exercises the
fork-time guard.

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
* ``z3_ast_build`` — number of *top-level* ``to_z3_ast()`` /
  ``to_z3_bool()`` entry calls, **not** the number of ``Z3_mk_*`` FFI
  calls: a single entry recursively builds the whole ``RustBV`` tree,
  so one increment can cover hundreds of Z3 node constructions on a
  deep expression. Useful for spotting benchmarks that hit the raw-AST
  passthrough path (low value) vs. benchmarks that fall through to
  ``claripy_to_rustbv`` AST rebuild (high value) — but do not read it
  as node-construction volume.
* ``z3_ast_cache_miss`` / ``z3_ast_cache_hit`` / ``z3_ast_memo_hit`` —
  per-node accounting inside ``to_z3_ast_cached``. A miss is a unique
  ``RustBV`` pointer visited in a conversion, a hit is a subtree shared
  *within* that conversion, and a memo hit is an ``Expression`` whose
  Z3 AST survived from a *prior* top-level call. Nodes genuinely
  constructed ≈ ``z3_ast_cache_miss − z3_ast_memo_hit``; that
  difference, not ``z3_ast_build``, is the metric to use when sizing
  raw Z3 construction volume.
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
