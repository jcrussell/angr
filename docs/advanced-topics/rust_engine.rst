Rust symbolic execution engine
==============================

angr ships an experimental Rust-native symbolic-execution engine
(``RustExplorationManager``) that runs the VEX interpreter, memory
model, and Z3 solver in Rust while still cooperating with the Python
project, claripy ASTs, and SimProcedures via PyO3. It is the same
engine documented in the repository's ``CLAUDE.md`` and is enabled by
passing ``use_rust_engine=True`` to ``proj.factory.simulation_manager()``.

This page collects the limitations and performance characteristics that
users need to know before reaching for the Rust engine. It is the
single source of truth for:

* which ``angr.sim_options`` flags it actually honors,
* which ``state.inspect`` events it dispatches (and which still raise),
  and
* which benchmarks run slower than pure Python and why.

Overview
--------

* **Honored options:** ``LAZY_SOLVES``, ``ZERO_FILL_UNCONSTRAINED_MEMORY``,
  ``APPROXIMATE_MEMORY_INDICES``, ``SYMBOLIC_WRITE_ADDRESSES``,
  ``STRICT_PAGE_ACCESS``, ``ENABLE_NX``, ``NO_IP_CONCRETIZATION``,
  ``NO_SYMBOLIC_JUMP_RESOLUTION``, ``NO_SYMBOLIC_SYSCALL_RESOLUTION``,
  ``KEEP_IP_SYMBOLIC``.
  Everything else is either inherited from Python or silently ignored —
  see the matrix below.
* **Partial** ``state.inspect`` **dispatch.** The Rust engine dispatches
  ``mem_read``, ``mem_write``, ``reg_read``, ``reg_write``,
  ``instruction``, ``irsb``, and ``exit`` events to Python BPs.
  Unsupported events (``call``, ``fork``, ``return``, ``syscall``,
  ``constraints``, ``simprocedure``, ``dirty``, …) still raise
  ``NotImplementedError`` at registration time.
* **Performance:** Faster than Python on most benchmarks, with a small
  number of known slower cases driven by Python-side cache pressure or
  bimodal Z3 solver nondeterminism.

Usage
-----

Construct a ``RustExplorationManager`` directly, or pass
``use_rust_engine=True`` to ``proj.factory.simulation_manager()``:

.. code-block:: python

   import angr
   from angr.exploration import RustExplorationManager

   proj = angr.Project("/path/to/binary", auto_load_libs=False)
   state = proj.factory.entry_state()

   mgr = RustExplorationManager(proj, [state])
   mgr.explore(find=0x401234, avoid=0x401000, max_steps=10000)

   for found in mgr.found:
       print(f"Found at {hex(found.addr)}")

The ``proj.factory.simulation_manager(state, use_rust_engine=True)``
form returns a wrapper that exposes the same interface as the standard
``SimulationManager`` (``explore``, ``step``, ``found``, ``avoid``,
etc.) while running the Rust engine underneath.

Lightweight proxy accessors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mgr.found`` (and the other stash properties) materialize each Rust
state as a full angr ``SimState`` so that ``state.posix.dumps`` and
other plugins work. That materialization is the dominant cost when a
caller only needs to read the program counter, a register, or a few
constraints across many states.

For those read-mostly use cases, every stash also has a
proxy-returning counterpart that returns ``list[RustStateProxy]`` —
O(1) per state, no ``SimState`` plugin chain, no cache sync:

.. code-block:: python

   # How many of the active states are at addr 0x401234?
   target = 0x401234
   count = sum(1 for s in mgr.active_proxies() if s.addr == target)

   # Bulk-eval a tracked symbol across every found state.
   for proxy in mgr.found_proxies():
       print(hex(proxy.solver.eval(stdin_var)))

The five accessors are ``mgr.found_proxies()``,
``mgr.active_proxies()``, ``mgr.avoid_proxies()``,
``mgr.deadended_proxies()``, and ``mgr.unconstrained_proxies()``. They
are direct shortcuts to ``mgr.proxy.<stash>`` for callers who do not
need the full ``RustSimulationManagerProxy`` wrapper. Use the full
``mgr.found`` / ``mgr.active`` / etc. properties when downstream code
needs full ``SimState`` plugins (``posix.dumps(0)``, ``simgr.explore``
seeding, claripy AST round-trips).

RustStateProxy read-only contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RustStateProxy`` is a **read-only** view. Calling
``proxy.memory.store(addr, data)`` or
``proxy.regs.<name> = value`` from inside a find/avoid predicate or a
``state.inspect`` breakpoint raises ``NotImplementedError`` — the proxy
delegates reads to Rust but does not push writes back. Constraints are
the one exception: ``proxy.solver.add(...)`` writes through to a forked
Rust solver via the ``_rust_add`` interceptor in
``rust_callback_dispatch.py``.

The asymmetry is by design. Proxy-backed register/memory writes were
evaluated end-to-end and explicitly deferred in
:doc:`rust_proxy_writes_design` (Option A): the measured per-callback
diff-and-push cost is 1–14 ms, the aggregate is at most ~7 % of wall
time on the fastest bench and well under 1 % on the rest, and the
plugin-substitution refactor needed to write-through cleanly carries
the same silent-divergence risk that
``avoid-state-copy-optimization`` flagged on the diff path.

Workarounds:

* **Mutate the seed state before exploration.** Write to the
  ``SimState`` you pass to ``RustExplorationManager`` (or
  ``proj.factory.simulation_manager(use_rust_engine=True)``) before
  ``run``/``explore`` — those writes land in Rust at seed time.
* **Use a SimProcedure-style hook** (``proj.hook(addr, fn)``) when you
  need an in-exploration mutation at a specific address. SimProcedure
  callbacks receive a **full** ``SimState`` (not a proxy), and
  ``state.memory.store(...)`` / ``state.regs.<name> = ...`` are synced
  back to the Rust state via the diff-and-push path in
  ``rust_state_sync.py`` after the callback returns.
* **Drop back to the Python engine** for analyses whose find/avoid
  predicates fundamentally need to mutate state. Omit
  ``use_rust_engine=True`` and the standard angr predicate API
  applies.

Read-only paths the proxy supports — ``proxy.regs.<name>`` /
``proxy.memory.load(addr, size)`` / ``proxy.solver.eval(...)`` /
``proxy.solver.add(...)`` — are stable. The reopen condition for
proxy-write support is the same as for the other deferred
write-through items in :doc:`rust_proxy_writes_design` (callback
density 10K+/bench or an unblocking dependency from ``angr-2k64``).

Architecture
------------

The Rust engine spans a Rust core and a thin Python wrapper that
cooperate through PyO3:

* **Rust core** (``native/angr/src/``): VEX interpreter, symbolic
  memory, Z3 solver, exploration loop, and native SimProcedures.
* **Python wrapper** (``angr/exploration/rust_manager.py``): callback
  dispatch for Python SimProcedures, technique support, and state
  synchronization through ``RustStateProxy``.
* **FFI boundary:** PyO3 bindings in ``native/angr/src/lib.rs``.
* **Shared Z3 context:** Python and Rust load the same ``libz3.so`` and
  share the Z3 context so claripy ASTs round-trip without
  re-serialization.
* **Feature flag:** ``use_rust_engine=True`` on
  ``proj.factory.simulation_manager()``.

Architecture support matrix
---------------------------

Only AMD64 is exercised end-to-end. Other architectures have register
and state plumbing wired up, and (mostly) calling-convention
definitions, but limited or no binary-driven coverage. Treat anything
below "Supported" as experimental.

.. list-table::
   :header-rows: 1
   :widths: 15 12 22 12 22 17

   * - Arch
     - Unit tests
     - Integration tests
     - Benchmarks
     - Calling conv
     - Status
   * - AMD64
     - ~110+
     - ~268 (fauxware)
     - 21/22
     - SystemV, MS x64
     - Supported
   * - x86 (32-bit)
     - 3
     - 1 (Cdecl ret reg)
     - 8 (3 FAST + 5 MEDIUM)
     - Cdecl
     - Supported (selectors: CS/DS/ES/FS/GS/SS)
   * - ARM (32-bit)
     - 2
     - 3 (LE validate, LE native-proc, BE blob)
     - 1 (arm_le_branch synthetic)
     - ARMEABI
     - Supported
   * - ARM64
     - 1
     - 4 (blob branch, NEON mla, real ELF, native-proc)
     - 1 (aarch64_le_branch synthetic)
     - AArch64
     - Supported
   * - MIPS32
     - 5
     - 3 (BE blob, LE real ELF, native-proc)
     - 1 (mips32_le_branch synthetic)
     - MipsO32
     - Supported (full-ABI: GPRs, FPU F0-F31, HI/LO, FCR)
   * - MIPS64
     - 1
     - 3 (LE real ELF, BE real ELF, native-proc)
     - 2 (mips64_le_branch + mips64_be_branch synthetics)
     - MipsN64
     - Supported (full-ABI: GPRs, FPU F0-F31, HI/LO, FCR)

Status meanings:

* **Skeleton** — ``RustSimState(<arch>)`` constructs and registers
  round-trip, ``fork()`` preserves isolation, but no test runs VEX
  through the interpreter on a real binary and no calling convention
  is actually exercised. No arch is currently in this state.
* **Experimental** — at least one integration test loads a real binary,
  runs ``mgr.run(...)``, and verifies a found-state result.
* **Supported** — has at least one benchmark in
  ``tests/benchmarks/baseline_timings.json`` and stays green in the
  regression suite.

Wired-up but not fully verified:

* Register offsets for all six arches in ``native/angr/src/arch/*.rs``.
* Endianness flag (ARM BE blob test added via ``armeb`` arch, see
  ``test_armeb_explore_blob``; MIPS32 BE+LE end-to-end via ELF + blob;
  MIPS64 BE+LE end-to-end via ELF; ARM64 BE untested — blocked on
  archinfo, see ``archinfo-be-arch-alias-support`` memory).
* ARMEABI / AArch64 / MipsO32 / MipsN64 calling conventions defined in
  ``calling_conventions.rs``.

To promote an arch from Skeleton → Experimental: add at least one
integration test that loads a real binary, runs ``mgr.run(...)``, and
verifies a found state. To promote Experimental → Supported: add a
benchmark and ensure it stays green in regression runs. The Cdecl x86
return-register bug (commit ``5329d8222``) was latent for months
precisely because no end-to-end x86 test ran — assume the same risk
for any new arch added without coverage.

x86 (32-bit) single-CC note
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust engine registers only Cdecl in
``native/angr/src/arch/calling_conventions.rs::default_cc_for_arch``
for x86. The Windows i386 benchmarks listed above
(``flareon2015_2``, ``flareon2015_5``, ``flareon2015_10``,
``whitehatvn2015_re400``, ``ekopartyctf2016_sokohashv2``,
``mma_howtouse``) run successfully under that single CC because the
relevant CC boundary is **Python-side**: ``entry_state()`` and
``Callable`` place arguments through angr's Python ``SimCC`` (which
already handles ``SimCCStdcall`` / ``SimCCStdcall32`` for SimWindows
binaries), and the Rust engine executes the resulting state. No
current x86-32 bench triggers a Rust-native ``SimProcedure`` from a
stdcall caller, so cleanup-side semantics (``ret N``) never matter.

Adding stdcall / fastcall in the future requires both a
``CallingConvention`` impl in ``calling_conventions.rs`` AND a
cleanup hook on the native ``SimProcedure`` dispatcher so the callee
adjusts SP by ``num_args * 4`` after returning. The current
dispatcher only handles caller-cleans (``ret`` pops just the return
address). File a bd issue with the specific consumer when that
need arises.

Z3 solver API
-------------

``RustSolverContext`` exposes Z3-backed satisfiability and
range-bound queries directly to Python, sharing the Rust engine's
solver context:

.. code-block:: python

   from angr.rustylib.vex_engine import RustSolverContext
   import claripy

   ctx = RustSolverContext()
   print(f"Z3 available: {ctx.z3_available()}")  # True

   x = claripy.BVS("x", 32)
   ctx.add_constraint_ast(x > 10)
   ctx.add_constraint_ast(x < 20)
   print(f"satisfiable: {ctx.satisfiable()}")
   print(f"min: {ctx.min(x, signed=False)}")  # 11
   print(f"max: {ctx.max(x, signed=False)}")  # 19

This is intended for low-level solver experiments and tests; for
exploration use ``RustExplorationManager`` instead.

For the build-time wiring that makes a shared ``libz3.so`` possible,
the ``SymContext`` lifecycle across forks, and the push/pop discipline
that ``RustSolverContext`` / ``SharedLineageSolver`` both rely on,
see :doc:`rust_z3_sharing`.

Z3 solver profiling counters
----------------------------

Process-wide atomic counters in
``native/angr/src/symbolic/context.rs`` track every
``solver.check()`` call and a number of related events. Read or reset
them through the manager:

.. code-block:: python

   mgr = RustExplorationManager(proj, [state])
   mgr.reset_solver_stats()
   mgr.explore(find=...)
   stats = mgr.get_solver_stats()

The returned dict includes:

* ``z3_check_count`` — total ``solver.check()`` calls.
* ``z3_check_time_ns`` — total time spent in ``solver.check()``.
* ``z3_sat_count`` / ``z3_unsat_count`` / ``z3_timeout_count`` — by
  ``SatResult``.
* ``z3_materialize_count`` / ``z3_materialize_time_ns`` — lazy-fork
  solver materialization.
* ``z3_assume_concrete`` / ``z3_assume_symbolic`` —
  ``assume_true``/``assume_false`` fast-path counters.
* ``z3_branch_check``, ``z3_branch_concrete``, ``z3_branch_model_hit``,
  ``z3_branch_model_miss`` — branch-evaluation outcomes.
* ``z3_ast_build`` — AST construction count.
* ``z3_site_<name>_count`` / ``z3_site_<name>_time_ns`` — per-call-site
  breakdown. Sites include ``satisfiable``, ``branch_true``,
  ``branch_false``, ``eval``, ``eval_upto``, ``min_init``,
  ``min_search``, ``max_init``, ``max_search``.

Counters are global (shared across ``SymContext`` instances). Call
``mgr.reset_solver_stats()`` to zero them at the start of a measured
window. The same dict is also merged into ``mgr.stats`` for
convenience.

Determinism contract
--------------------

The Rust engine is **not** bit-for-bit deterministic across runs by
default. Two sources of nondeterminism dominate, and a third is
inherent to Z3:

* **std::HashMap iteration order.** ``std::collections::HashMap`` is
  seeded with a per-process random value; iteration produces a fresh
  order on every run. ``rustc_hash::FxHashMap`` uses a fixed seed and
  *is* deterministic for a given insertion sequence.
* **Z3 internal heuristics.** Z3's SAT engine uses pseudo-random
  variable / restart ordering whose seed is not pinned by the engine.
  Two ``solver.check()`` calls with identical assertions added in
  identical order can return different satisfying models.
* **Residual Z3 nondeterminism** (cannot be eliminated). Even with
  ``smt.random_seed`` and ``sat.random_seed`` pinned, Z3 4.13 reserves
  some heuristic latitude for simplification / preprocessing ordering.
  ``solver.check()`` *result* (sat/unsat) is reproducible at a given
  seed; ``solver.model()`` *eval answers* are not strictly guaranteed
  across rebuilds.

Iteration-order audit (HashMap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Audit of every ``std::HashMap`` in ``native/angr/src/`` against the
question *"does iteration order leak into observable output (state
ordering, branch enumeration, eval answer)?"*:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Site
     - Iteration?
     - Verdict
   * - ``stash.rs`` ``stashes`` / ``state_index`` / ``state_roots``
     - Yes (``stashes_mut().values_mut()`` in
       ``exploration/mod.rs:set_max_history`` and the slow path of
       ``find_state_by_id``)
     - **Not output-affecting.** Per-stash order is the VecDeque
       (FIFO/LIFO is honored). Cross-stash iteration only writes the
       same value to every state or finds a unique ``state_id``.
   * - ``state.rs`` ``symbolic_pages`` / ``hook_symbolic_memory`` /
       ``addr_to_ast``
     - Yes (``state_api.rs:_get_state_symbolic_pages`` etc. dump into
       ``PyDict``)
     - **Not output-affecting.** Python consumers read by key; the
       resulting ``PyDict`` is data-equivalent run-to-run. Iteration
       order leaks only into ``dict.__repr__`` for debug prints.
   * - ``state.rs`` ``simprocedures`` / ``vex_fallback_addrs`` /
       ``simprocedure_fallback_by_name``
     - Lookup-only
     - **Not output-affecting.**
   * - ``claripy_bridge.rs`` ``CLARIPY_AST_CACHE`` + ``deepcopy_memo``
     - Lookup-only (LRU caches)
     - **Not output-affecting.** Keyed by Python id / hash.
   * - ``vex/lifter.rs`` IRSB lift cache
     - Lookup-only
     - **Not output-affecting.**
   * - ``vex/dirty.rs`` ``handlers`` / ``syscalls/mod.rs`` ``handlers``
       / ``procedures/mod.rs`` ``procedures``
     - Lookup-only
     - **Not output-affecting.** Static registries.
   * - ``symbolic/registry.rs`` ``py_hash_to_rust_id`` /
       ``rust_id_to_py`` / ``name_to_info``
     - Lookup-only
     - **Not output-affecting.**
   * - ``symbolic/table.rs`` ``symbols``
     - Lookup-only
     - **Not output-affecting.**
   * - ``automaton/subset_construction.rs``
       ``inverse_mapping: HashMap<StateId, Vec<StateId>>``
     - Stored on DFA; not iterated for output during regex matching.
     - **Not output-affecting.**
   * - Stats / profiling maps (``ExecutionStats``,
       ``profiling::accumulated_stats``, ``analyze_constraint_sharing``,
       ``py_get_execution_stats``, ``get_solver_stats``,
       ``get_registers_named``)
     - Yes (returned to Python as dicts)
     - **Not exploration-affecting.** Values are summed counters or
       per-name lookups; order leaks only into stringified debug output.

No std::HashMap site currently leaks iteration order into the
exploration *result* (found/avoid stashes, evaluated bytes, or
constraint sets). The remaining nondeterminism comes from Z3.

Z3 seed pinning attempted and rejected (angr-iaol.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build_solver_params`` (``native/angr/src/symbolic/context.rs``) does
**not** pin ``smt.random_seed`` or ``sat.random_seed``. The audit
hypothesis (parent iaol) was that pinning would deliver model
stability across fresh ``RustSolverContext`` instances. iaol.1
(2026-05-25) tried all three reachable param-name forms via
``z3-0.19.7`` ``Params::set_u32`` against the
``test_model_stability_constraint_order`` and a new constraint-
identity test, and **all three failed**:

* ``smt.random_seed`` / ``sat.random_seed``: corrupt the solver.
  ``eval()`` returns models that *violate* the asserted constraints
  (eg ``x = 0`` for the constraint ``x >= 100``). Both forms have the
  same failure mode, suggesting they hit the same invalid-key error
  path in ``Z3_solver_set_params`` (Z3 4.13 expects these as
  *module-level* keys via ``Z3_global_param_set``, not solver-level).
* ``random_seed`` (no module prefix, the in-descriptor short name):
  accepted by Z3 without corruption, but produces *more* variation
  across instances than no pin at all, and breaks
  ``test_model_stability_constraint_order`` (which passes with no
  pin).

Z3 4.13 model determinism across fresh solver instances is therefore
**not reachable through ``Z3_solver_set_params``** for these keys. To
make any seed pin take effect, the call has to land via
``Z3_global_param_set`` *before* the first ``Solver::new`` — and even
then Z3 reserves variable / restart heuristic latitude that is not
pinned by these keys. ``parallel.enable=true`` remains off
(``avoid-z3-parallel-enable``).

Residual nondeterminism the engine still carries:

* The trailing-byte mismatch in ``defcamp_r100`` (unconstrained stdin
  fill values) — ``run_regression.py`` normalizes output.
* ``test_model_stability_constraint_order`` only verifies
  order-independence *within a process*, not run-to-run stability
  across Z3 rebuilds.
* Two fresh ``RustSolverContext`` instances with identical constraints
  may eval the same underconstrained variable to different witnesses.

Deterministic mode (``deterministic=True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RustExplorationManager(deterministic=True)`` (angr-iaol.2) pins
``smt.random_seed`` and ``sat.random_seed`` to ``0`` via
``Z3_global_param_set`` before any new solver is constructed. The pin
is process-global and applies to every solver built afterwards —
including solvers in other managers in the same process — but solvers
that already exist are unaffected.

Usage::

    from angr.exploration import RustExplorationManager

    state = proj.factory.entry_state()
    mgr = RustExplorationManager(proj, [state], deterministic=True)
    mgr.explore(find=target_addr, max_steps=50000)

Implementation:

* ``angr/exploration/rust_manager.py::_apply_deterministic_z3_globals``
  guards the pin with a module-level boolean so repeated manager
  constructions are no-ops after the first.
* The pin routes through the new ``set_z3_global_param`` PyO3 binding
  in ``native/angr/src/engine.rs`` (cfg-gated on ``vex-engine-z3``)
  rather than ``Z3_solver_set_params`` — the solver-level path
  empirically *corrupts* the solver for these keys (see iaol.1
  finding above).

What the flag does **not** close:

* Z3 4.13 retains heuristic latitude (variable selection ordering,
  restart timing, internal simplification passes) that is not bounded
  by the two pinned seeds. Two fresh ``RustSolverContext`` instances
  with identical asserted constraints can still produce different
  ``eval()`` witnesses for under-constrained variables.
* The ``defcamp_r100`` trailing-byte mismatch is the canonical
  residual case — ``run_regression.py`` normalizes the symbolic-fill
  padding so the bench still passes, but raw ``posix.dumps(0)`` bytes
  can differ across runs even with the flag set.

Coverage is exercised by
``tests/engines/test_rust_exploration.py::TestDeterministicMode`` —
fauxware ``explore(find=0x4006ed, avoid=0x4006fd)`` produces the same
found-stash size and the same evaluated stdin across two fresh
managers when ``deterministic=True`` is set. The std::HashMap audit
above already confirmed no HashMap site leaks iteration order into
exploration output, so this flag does **not** need to be paired with
an FxHashMap conversion.

Default is ``deterministic=False`` to preserve current behavior and
keep the call free for the common case (no global Z3 state mutation
during normal manager construction).

Pipeline instrumentation counters
---------------------------------

The same ``mgr.stats`` / ``mgr.get_solver_stats()`` dict carries
counters covering the rest of the engine: VEX op dispatch, memory
volume, address concretization, and AST construction (angr-2j5v). All
counters are process-wide ``AtomicU64`` and a single ``fetch_add`` —
zero-cost when not read. They reset alongside the Z3 counters via
``mgr.reset_solver_stats()``.

* **VEX op dispatch** (bumped at the ``IRExpr::{Unop,Binop,Triop,Qop}``
  dispatch in ``interpreter/expressions.rs``):

  * ``vex_unop_total`` / ``vex_binop_total`` / ``vex_triop_total`` /
    ``vex_qop_total`` — count per arity.
  * ``vex_op_arith`` / ``vex_op_logic`` / ``vex_op_shift`` /
    ``vex_op_cmp`` / ``vex_op_ext`` / ``vex_op_fp`` / ``vex_op_vec`` /
    ``vex_op_other`` — count per IROp family. Family classification
    lives in ``vex/ops.rs::iropclass``; ``Vec`` covers every
    ``V*``-prefixed (SIMD/NEON) variant, ``Other`` is the catch-all for
    ``Raw``/``NeonUnimplemented``. Every dispatched op bumps exactly
    one family counter, so ``sum(vex_op_*) == sum(vex_<arity>_total)``.

* **Memory volume** (bumped in ``memory/load.rs`` and
  ``memory/store.rs``):

  * ``mem_load_count`` / ``mem_store_count`` — total load/store calls
    reaching ``SymbolicMemory::{load,store}_concrete``. Catches both
    the public ``load(addr_bv)`` entry and the ``state.rs`` hot path
    that calls ``load_concrete(addr_u64)`` directly.
  * ``mem_load_bytes`` / ``mem_store_bytes`` — cumulative bytes
    accessed.
  * ``mem_load_symbolic_addr`` / ``mem_store_symbolic_addr`` — subset
    that entered via ``load(addr_bv)`` / ``store(addr_bv)`` with a
    symbolic address. (Counted only at the public entry because by
    the time the call reaches ``*_concrete`` the address has been
    evaluated to a ``u64``.)
  * ``mem_lazy_page_fault_count`` — ``UnmappedPageInRegion`` errors,
    bumped on the public ``load`` / ``store`` paths only.

* **Concretization fanout** (bumped in ``concretize.rs``):

  * ``concretize_read_count`` / ``concretize_write_count`` — number of
    ``concretize_read`` / ``concretize_write`` calls.
  * ``concretize_total_candidates`` — cumulative ``K`` (candidate
    count) across all calls. Average ``K`` = total / count.
  * ``concretize_max_candidates`` — watermark of the largest ``K``
    seen. ``TooLarge`` / ``Failed`` outcomes count as ``K = 0``.
  * ``concretize_disjunction_count`` / ``_terms_total`` /
    ``_max_terms`` — angr-62li. Number of times the engine hoisted
    ``Or(addr == a0, ..., addr == aK)`` to the top-level Rust solver
    (matches Python's ``address_concretization_mixin``), the sum of
    ``K`` across those hoists, and the largest ``K``. Hoisting is
    gated to ``2 <= K <= 8`` — beyond ``8`` the long-Or processing
    cost on every subsequent ``solver.check()`` swamps the
    propagate-values payoff (measured: 22% regression on
    flareon2015_5 with K up to 64 unbounded; neutral with the gate).
    ``Strided`` results are not hoisted (the strided abstraction is
    concretizer policy, not a tight constraint — A/B pending).

* **AST construction** (bumped at the actual
  ``RustBV::Expression { op: BVOp::X, ... }`` construction sites in
  ``symbolic/value.rs``):

  * ``bvop_reverse_count`` / ``bvop_concat_count`` /
    ``bvop_extract_count`` — node emissions of ``BVOp::Reverse``,
    ``BVOp::Concat``, ``BVOp::Extract``. Simplification short-circuits
    (e.g. ``reverse(reverse(x)) → x``, fully-concrete fold) do **not**
    bump the counter — these track real Z3-visible emissions.

* **Constraint add path** (bumped in
  ``symbolic/context.rs::add_constraint_raw``):

  * ``add_constraint_raw_total`` — total ``add_constraint_raw`` calls.
  * ``add_constraint_raw_dedup_scanned`` — calls that consulted the
    per-context HashSet side-table.
  * ``add_constraint_raw_dedup_hit`` — subset where the constraint's
    ``z3_ast`` pointer was already in the side-table (skipping the
    solver re-assert and the ``z3_assertions.push``).

  Dedup hit-rate is **bimodal** across the fast-tier corpus (2026-06-01
  audit, fauxware / defcamp_r100 / sym-write / ais3_crackme /
  defcon2016quals_baby-re / whitehatvn2015_re400 / csgames2018 /
  flareon2015_10 / google2016_unbreakable_0 / codegate_2017-angrybird /
  strcpy_find / flareon2015_2 / ekopartyctf2016_rev250 / csaw_wyvern /
  flareon2015_5):

  .. list-table::
     :header-rows: 1
     :widths: 35 15 15 15 20

     * - Bench
       - total
       - hits
       - rate
       - regime
     * - ``defcon2016quals_baby-re``
       - 13
       - 12
       - 92.3%
       - dedup pays
     * - ``csaw_wyvern``
       - 40
       - 33
       - 82.5%
       - dedup pays
     * - ``whitehatvn2015_re400``
       - 2
       - 1
       - 50.0%
       - dedup pays
     * - ``flareon2015_5``
       - 403
       - 126
       - 31.3%
       - dedup pays
     * - 11 other benches
       - 0–13
       - 0
       - 0.0%
       - no hits

  Counter + HashSet are **kept** despite the per-bench wall-clock delta
  being null in isolation (Z3 internally dedups asserts, so the saved
  ``solver.assert`` is a no-op — only the ``Bool::clone`` and
  ``z3_assertions.push`` are skipped). Rationale: 4/15 benches show
  substantial duplicate-assert traffic (31–92%) that would otherwise
  accumulate in the per-context ``z3_assertions`` vec and scale
  constraint-export / lineage-switch work for nothing. The HashSet
  overhead (one ``insert`` + ``contains`` per add) is bounded above
  by ``add_constraint_raw_total`` * O(hash), trivial against the
  ``z3_check`` time it gates. See ``angr-dtrl`` for the audit and
  ``sfp9-dedup-null-result`` bd memory for the original wall-clock A/B.

Sample output on ``defcamp_r100`` (Rust engine, 3 SAT paths):

.. code-block::

   vex op dispatch:
     vex_binop_total: 374
     vex_op_arith: 275
     vex_op_cmp: 39
     vex_op_ext: 996
     vex_op_shift: 60
     vex_unop_total: 996
   memory volume:
     mem_load_bytes: 21
     mem_load_count: 21
     mem_store_bytes: 391
     mem_store_count: 279
   ast emissions:
     bvop_extract_count: 60

``tests/benchmarks/run_single.py`` emits these blocks automatically
when the Rust engine is run with non-zero counters (see the
``vex op dispatch`` / ``memory volume`` / ``concretization fanout`` /
``ast emissions`` sections of its output).

Experimental: ``ANGR_Z3_TACTIC`` solver-strategy override
---------------------------------------------------------

Z3's solver construction strategy is selectable via the
``ANGR_Z3_TACTIC`` environment variable, read once per process on the
first ``SymContext`` materialization. **Default behavior (env unset) is
unchanged from prior releases.** This knob exists for users hitting
bimodal hash-cracker-style benchmarks where Z3's default portfolio
falls into a slow branch.

Recognized values:

* unset / empty / ``default`` / ``smt`` — use ``z3::Solver::new()``
  (Z3's default ``smt`` portfolio). **Recommended for most workloads.**
* ``qfbv`` — use Z3's quantifier-free bitvector preset
  (``Tactic::new("qfbv").solver()``). Wins big on bimodal Z3-heavy
  benches and regresses small-problem benches; see table below.
* ``qfbv_smart`` — probe-conditional tactic that dispatches on
  ``num-consts``: large goals (> ``ANGR_Z3_QFBV_THRESHOLD``, default
  20) use ``qfbv``, smaller goals use ``smt``. Logged here as an
  experimental hook — measured A/B against ``qfbv`` it picks up the
  per-check tactic-dispatch overhead, so plain ``qfbv`` wins more
  cleanly on the bimodal trio. Kept for further experimentation.
* any other string — interpreted as a colon-separated tactic pipeline
  composed with ``Tactic::and_then``, e.g.
  ``simplify:propagate-values:solve-eqs:bit-blast:sat``. Unknown
  tactic names panic loudly (no silent fallback).

Measured impact of ``ANGR_Z3_TACTIC=qfbv`` (3-sample medians,
2026-05-20):

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 14

   * - Benchmark
     - default (s)
     - qfbv (s)
     - delta
   * - securityfest_fairlight
     - 18.87
     - 3.04
     - **−84 %**
   * - ekopartyctf2016_sokohashv2
     - 9.42
     - 3.68
     - **−61 %**
   * - mma_howtouse
     - 6.04
     - 5.63
     - −7 %
   * - csgames2018
     - 1.04
     - 2.88
     - **+177 %**
   * - flareon2015_2
     - 3.51
     - 4.48
     - **+24 %**

The shape: ``qfbv`` skips Z3's portfolio dispatch and goes straight to
a bitvector-specialized pipeline. Hash-cracker problems with many
constraints over a single state-set benefit; CTF-style binaries with
many small constraint problems pay the qfbv pipeline's per-check
overhead on every solve. The acceptance criteria for the angr-ya00
spike were "≥10 % median win on at least one bench with no
regressions elsewhere (ship behind a SimOption if needed)" — the env
var is the SimOption analog for this lower-level knob.

Note: ``sat-preprocess:qfbv`` was tried and rejected — it produced
incorrect results on ``csgames2018`` and ``securityfest_fairlight``
(``IndexError: list index out of range`` from the post-solve goal
unwrap). Avoid pipelines beginning with ``sat-preprocess`` until that
is debugged.

Shared-lineage Z3 solver — rejected
-----------------------------------

The 2026-05 attempt to amortize per-fork solver construction by sharing
a single Z3 solver across a forked lineage of states was investigated,
implemented, measured, and rejected as a default. This section records
the verdict so future investigators do not re-derive the same findings.

Why it was tried
~~~~~~~~~~~~~~~~

The ``angr-hk7k`` research spike (2026-05-21) ranked four design options
for cutting the high-prefix fork path (~5 ms / materialization). Option
A — per-lineage shared Z3 solver, with each forked state tracking its
local diff as a scope path off a common base and manipulating Z3 with
``push`` / ``pop`` on context switch — was recommended. The pitch: per-
query cost becomes ``O(local_diff)``, not ``O(total_assertions)``.

The implementation spike landed under ``angr-v5a5`` (commit
``5b5689e2c``). It introduced ``SharedLineageSolver`` in
``native/angr/src/symbolic/lineage.rs`` — scope paths, push/pop
bookkeeping, a ``switch_to`` hot-cache fast path
(commit ``caa07ebf5``), and a fork-time materialization gate keyed on
``bare_z3_push_depth == 0`` to keep external bare-push frames correct.

BFS-thrash finding
~~~~~~~~~~~~~~~~~~

Fork-time lineage materialization (``angr-v5a5`` slice 4c.3, two
independent attempts on 2026-05-23) regressed
``defcon2016quals_baby-re`` ~10x: 0.46 s baseline → 3.89–5.13 s
(5-sample medians). The ``angr-3ms1`` follow-up quantified the cause:

* **Hot-cache hit rate 15.6%** (15 / 96 ``switch_to`` calls) — exactly
  the theoretical 1/N expected for BFS uniformly interleaving queries
  across N = 5–10 active states.
* **Z3 per-check time exploded 155x:** 0.88 ms baseline (29 checks) →
  137 ms with lineage (3,972 ms total). The cost is not from
  ``push`` / ``pop`` ops (only ~400 total), suggesting Z3's incremental
  solver re-derives learned clauses after each push.

The ``switch_to`` hot-cache fast path is *necessary-but-insufficient*:
lowering the fixed cost does not raise the hit rate, which is bounded
below by 1/N for the BFS access pattern.

End-to-end confirmation (``angr-awqp``, 2026-05-24): a
``--use-shared-lineage-solver`` opt-in flag on
``tests/benchmarks/run_single.py`` measured 5-sample medians:

.. list-table::
   :header-rows: 1
   :widths: 32 22 22 24

   * - Benchmark
     - default (s)
     - lineage (s)
     - delta
   * - ``defcon2016quals_baby-re``
     - 0.47
     - 3.19
     - **+579 %** (6.79x slower)
   * - ``google2016_unbreakable_0``
     - 0.94
     - 1.44
     - **+53 %**
   * - ``defcamp_r100``
     - 0.23
     - 0.23
     - neutral
   * - ``ais3_crackme``
     - 0.88
     - 0.71
     - **−19 %** (1.24x faster)

Workload shape — not exploration strategy — discriminates WIN from
LOSE: both ``ais3_crackme`` and ``baby-re`` run BFS under the harness.
Predictors: hot-cache hit rate, active-state count distribution, and
``lineage_pop_count / lineage_push_count`` ratio.

The simple variant that survived
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than ship the opt-in flag default-on (catastrophic on baby-re)
or default-off (the ais3 win never appears), ``angr-v5ht`` (commit
``971d35545``, 2026-05-24) added a runtime thrash detector. A sampler
hooks at the top of each ``run_loop`` iteration; every 10 ticks, once
≥ 20 ``lineage_switch`` events have accumulated, it computes the
hot-cache hit rate and trips a sticky ``LINEAGE_DISMANTLED``
``AtomicBool`` if the rate falls below 35%. Once tripped,
``SymContext::fork`` skips lineage minting and parent-arc inheritance;
children fall back to per-context solvers seeded from
``frozen_shared``. In-flight lineages from the pre-dismantle phase
keep running until their states die.

5-sample medians with the detector active under default-on:

* ``baby-re`` 3.17 s → 1.03 s (1.54x over the 0.67 s baseline; the
  residual gap is from in-flight lineages — see below).
* ``ais3_crackme`` 0.72 s (lineage win preserved).
* ``google2016_unbreakable_0`` 1.06 s (faster than the 1.39 s baseline
  — the detector also catches the regression on this canary).

The simple variant ships the ais3-class win without exposing the
baby-re catastrophe. New counters: ``lineage_dismantled``,
``lineage_dismantle_count``, ``lineage_sample_call_count``,
``lineage_sample_decision_count``.

Threshold justification (35 %)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 35 % hot-cache threshold (``run_loop.rs`` —
``tick_and_sample_for_thrash(10, 20, 35)``) was tuned on N = 4 workload
data points (``angr-1gfa``, 2026-05-25), running each with
``use_shared_lineage_solver=True`` and reading
``lineage_switch_hot_count / lineage_switch_count`` from
``get_solver_stats()`` at end-of-bench:

* ``defcon2016quals_baby-re`` (LOSE; 6.79 x slower under shared
  lineage): **30.2 %** — sampler would dismantle (correct).
* ``ais3_crackme`` (WIN; 1.24 x faster): **44.7 %** — stays on
  (correct).
* ``defcamp_r100`` (WIN; 4.4 x baseline): **51.0 %** — stays on
  (correct).
* ``csaw_wyvern`` (WIN; 16.9 x baseline): **84.2 %** — stays on
  (correct).

The WIN cluster spans 44.7–84.2 %; the LOSE point is at 30.2 %; 35 %
sits in the gap with 4.8 pp margin to LOSE and 9.7 pp margin to the
closest WIN (ais3). No data point in the current corpus moves the
threshold: raising it would tighten the WIN margin without value;
lowering it would shrink the LOSE margin and risk false negatives on
borderline thrash workloads.

Why alternatives (a), (b), (d) were not pursued
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``angr-3ms1`` enumerated four architectural responses to the
BFS-thrash finding. Their fates:

* **(a) Engine-level per-state query batching.** Restructures
  exploration so consecutive Z3 queries land on the same state. The
  LIFO microbench ``push_pop_lifo_chain`` (commit ``5e59ff48b``)
  confirmed the hypothesis: 10.3 ms vs 13.2 ms per-state batched,
  3.16x faster than BFS thrash. **Not pursued because the
  ``angr-v5ht`` runtime detector captures the same WIN workloads
  without an exploration-strategy rewrite.** If a future workload
  shows the detector is too coarse, (a) becomes the next step.
* **(b) Per-state solvers with shared trunk.** Defeats the lineage
  purpose by construction (separate Z3 instances erase the
  shared-trunk win). Not implemented.
* **(d) Z3 ``check`` with assumptions instead of push/pop.**
  Implemented as a spike at ``native/angr/src/symbolic/lineage_assumptions.rs``
  in commit ``561aa838b`` and microbenched in ``5e59ff48b``.
  Measured 2.1x SLOWER on ``bfs_thrash`` and 3.9x SLOWER on
  ``per_state_batched`` than ``push`` / ``pop``. The hypothesised
  learned-clause preservation never overcame the per-check overhead
  of growing assertion tables. **Negative result; spike deleted in
  ``angr-0hdq.2`` — recover from git history (commit ``561aa838b``)
  if anyone revisits the assumption strategy.**

Alternative (c) — runtime thrash-detect + lineage dismantle — is the
surviving variant (``angr-v5ht`` simple form). The premise that
``strategy='dfs'`` could be used as a default-on discriminator was
rejected independently: both the WIN canary (ais3) and LOSE canary
(baby-re) run BFS, so strategy is not the predictor. Any future "just
enable shared lineage when X" proposal must demonstrate X
discriminates WIN vs LOSE on multiple benches, not just one.

Why the full-variant teardown was reverted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``angr-0dgq`` implemented the full-variant dismantle: on trip, walk
all active ``SymContext``\ s, drop their lineage ``Arc``\ s, and
rebuild per-context ``z3::Solver`` instances from
``z3_assertions_shared + local_constraints``. Goal: close the
residual gap by killing in-flight lineages.

5-sample baby-re measurement (2026-05-24, HEAD ``971d35545``): simple
median 1.00 s / 552 MB vs full median 1.13 s / 509 MB. **Memory wins
8 %, wall time loses 13 % with higher variance (outliers up to
1.63 s).** The 13-state Z3 solver rebuild at dismantle time costs
more than the ``switch_to`` overhead it eliminates. Reverted before
commit; do not retry without an incremental rebuild that hands the
lineage's ``z3::Solver`` to the per-context path instead of dropping
and rebuilding.

Verdict
~~~~~~~

The materialize-cost win that motivated ``angr-hk7k`` is not
capturable within the current engine architecture under BFS
exploration. The shipped form (``angr-v5ht`` simple-variant runtime
dismantle plus the default-off ``use_shared_lineage_solver`` kwarg
on ``RustExplorationManager``) ships the ais3-class win where it
exists, falls back to baseline elsewhere, and adds no overhead on
the default code path — the detector hooks only when the flag is
on.

Bead trail: ``angr-v5a5`` (spike rejected), ``angr-3ms1``
(alternatives enumerated), ``angr-v5ht`` (simple-variant landed),
``angr-0dgq`` (full-variant reverted).

SimOption coverage matrix
-------------------------

This section documents which ``angr.sim_options`` flags the Rust
symbolic-execution engine actually honors. Anything not listed as
**Honored** or **Inherited** is a **silent no-op** when running under
``use_rust_engine=True`` — the option may be set on the SimState and
visible to Python introspection, but the Rust interpreter / memory
model / solver does not consult it.

Status legend:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Meaning
   * - ✅ Honored
     - Rust reads the option and changes behavior accordingly.
   * - ↪ Inherited
     - Option still has its Python effect because the relevant code path
       runs in Python (claripy AST passthrough, Python SimProcedure
       callbacks, pyvex lifting).
   * - ⚠ Ignored — divergence-risk
     - Setting the option in Python would change behavior; under Rust it
       does nothing, so two engines may produce different results.
   * - ◌ Ignored — no-op
     - The Python feature this gates isn't implemented in Rust at all
       (unicorn, abstract memory, CGC, JAVA, action tracking, etc.).
       Setting it has no effect either way.

Future-fix legend (for Ignored rows):

* **(a) implement** — Rust should grow support for this option.
* **(b) explicitly reject** — ``add()`` of this option on a state owned
  by ``RustExplorationManager`` should raise, so silent divergence
  becomes a loud error.
* **(c) accept-but-document** — option is benign, leave the silent
  ignore but note it here.

The Python-side option set lives in
``RustExplorationManager._py_state_options`` (per-state-id dict, see
``get_state_options_py``); writing to ``state.options.add(X)`` through
the proxy stores the value but does not change Rust behavior unless
this matrix shows otherwise.

Honored options
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Option
     - Where Rust reads it
     - Effect
   * - ``LAZY_SOLVES``
     - ``rust_manager.py:698``, ``:2322`` (re-checked at ``explore()`` time)
     - Calls ``set_lazy_solves(True)``; Rust skips per-block satisfiability
       checks. Also propagated through ``_clone_state_metadata`` on
       disk-cache reuse (``:1752``).
   * - ``ZERO_FILL_UNCONSTRAINED_MEMORY``
     - ``rust_manager.py:701``
     - Calls ``set_zero_fill_unconstrained(True)``; uninitialized memory
       reads return ``BVV(0, n)`` instead of fresh symbols.
   * - ``APPROXIMATE_MEMORY_INDICES``
     - ``rust_manager.py:705``
     - Passed as ``use_approx`` to ``configure_concretization_strategies``;
       Rust adds an approximation strategy ahead of full Z3 enumeration
       on symbolic loads.
   * - ``SYMBOLIC_WRITE_ADDRESSES``
     - ``rust_manager.py:706``
     - Passed as ``sym_write`` to ``configure_concretization_strategies``;
       Rust permits multi-valued symbolic write targets instead of
       forcing concretization.
   * - ``STRICT_PAGE_ACCESS``
     - ``rust_manager.py:1963`` (per-state, also propagated through
       ``_clone_state_metadata`` on disk-cache reuse)
     - Calls ``RustSimState::set_enforce_permissions(True)``; loads/stores
       violating per-page R/W bits raise ``SimSegfaultError``. Preserved
       across forks via ``SymbolicMemory::fork``.
   * - ``ENABLE_NX``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_enforce_nx(True)``; instruction fetches
       from mapped non-X pages raise ``SimSegfaultError`` at
       ``VEXInterpreter::get_or_lift_block``. Matches Python: the X
       check fires only when ``ENABLE_NX`` AND ``STRICT_PAGE_ACCESS`` are
       both set (``angr/engines/vex/heavy/heavy.py:115-124``).
   * - ``NO_IP_CONCRETIZATION``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_no_ip_concretization(True)``; at block
       boundaries with a symbolic jump target the state routes to the
       ``unconstrained`` stash silently instead of being enumerated. The
       check fires inside ``VEXInterpreter::eval_next_addr_concretized``
       (``native/angr/src/interpreter/exits.rs``). Matches Python's
       ``engines/successors.py:292-296`` behavior
       (``max_targets=0`` with ``skip_max_targets_warning=True``).
   * - ``NO_SYMBOLIC_JUMP_RESOLUTION``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_no_symbolic_jump_resolution(True)``;
       OR'd with ``no_ip_concretization`` at the
       ``eval_next_addr_concretized`` short-circuit, so a symbolic jump
       target routes to the ``unconstrained`` stash without enumeration.
       Matches Python's ``engines/successors.py:234-239`` (early elif
       branch routing symbolic targets to ``unconstrained_successors``
       before ``AddressConcretizer`` is invoked).
   * - ``NO_SYMBOLIC_SYSCALL_RESOLUTION``
     - Inherited through the Python syscall fallback
     - ``VEXInterpreter::get_syscall_num`` returns ``Option<u64>``
       (``None`` when the syscall register is symbolic), and
       ``stepping.rs::RunResult::Syscall`` skips the native syscall
       registry when ``num`` is ``None``, forcing a Python callback. The
       Python side's ``engines/successors.py:_resolve_syscall`` reads
       the option from ``state.options`` and either enumerates concrete
       syscalls or routes to the unknown-syscall stub (line 352). Before
       angr-gffd, ``get_syscall_num`` was ``unwrap_or(0)`` so a symbolic
       ``rax`` silently dispatched to ``NativeReadSyscall`` (amd64
       syscall 0).
   * - ``KEEP_IP_SYMBOLIC``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_keep_ip_symbolic(True)``; after a symbolic
       jump target is concretized to one-or-more concrete pc values, each
       successor keeps its IP register set to the original symbolic
       expression and no per-fork ``target == addr`` narrowing constraint is
       added. The concretized ``state.pc`` u64 still drives the next block
       lift. The interpreter stashes the symbolic expression in
       ``symbolic_ip_at_exit`` and the manager restores it via
       ``state.set_ip`` after ``state.set_pc(next_pc)``
       (``native/angr/src/exploration/stepping.rs``,
       ``native/angr/src/interpreter/exits.rs``). Multi-target forks are
       handled by ``handle_symbolic_jump_target``. Mirrors Python's
       ``engines/successors.py:297-307,326-331``.

Inherited (option works because the code path runs in Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Why it still works
   * - ``SYMBOLIC``
     - Rust always operates symbolically. The option is required for
       claripy/Python-side SimProcedures to behave symbolically.
   * - ``SYMBOLIC_INITIAL_VALUES``
     - claripy AST creation honors this; Rust receives ASTs from Python.
   * - ``TRACK_CONSTRAINTS``
     - Constraints are stored in claripy on the Python side via the
       constraint sync; the Rust solver mirrors them.
   * - ``COMPOSITE_SOLVER``
     - Solver flavor is on the claripy side — Rust uses its own Z3
       context but only stores constraints, not the solver topology.
   * - ``SUPPORT_FLOATING_POINT``
     - claripy gates float ops behind this. ASTs that propagate through
       Rust come back to Python for any non-VEX-native operation.
   * - ``SIMPLIFY_*`` (claripy AST simplification)
     - Simplification happens inside claripy when building the AST (e.g.
       ``state.solver.simplify``). Rust embeds those ASTs unchanged.
   * - ``USE_SYSTEM_TIMES``
     - Time-related SimProcedures (``gettimeofday``, ``time``,
       ``clock_gettime``) dispatch to Python; the Python procedure honors
       the flag. Inherited only when the procedure has not been replaced
       by a native variant in ``native/angr/src/procedures/``.
   * - ``ALLOW_SEND_FAILURES``, ``FILES_HAVE_EOF``, ``ALL_FILES_EXIST``,
       ``ANY_FILE_MIGHT_EXIST``, ``SHORT_READS``,
       ``CONCRETIZE_SYMBOLIC_FILE_READ_SIZES``
     - POSIX SimProcedures (``read``, ``write``, ``open``, ``recv``,
       ``send``) execute in Python and consult the angr SimState
       directly. Native POSIX procedures (when enabled — angr-3tek
       tracks this) would bypass these flags.
   * - ``RUN_HOOKS_AT_PLT``
     - Hook dispatch is orchestrated by the Python wrapper; PLT
       detection runs against the Python project.
   * - ``OPTIMIZE_IR``, ``NO_CROSS_INSN_OPT``
     - pyvex still produces the IRSB on the Python side before handing
       bytes to the Rust interpreter.

Ignored — divergence-risk
~~~~~~~~~~~~~~~~~~~~~~~~~

These options *would* change Python-engine behavior. Under Rust they
are silently dropped, which can produce different found/avoided sets
vs. Python.

.. note::

   **Implementation note (angr-1bqa, 2026-05-09):** The (b) classified
   options are warn-once'd via ``_REJECTED_OPTION_NAMES`` in
   ``angr/exploration/rust_manager.py``. Two entries from the (b)
   category — ``TRACK_CONSTRAINT_ACTIONS`` and ``TRACK_MEMORY_MAPPING`` —
   are intentionally *excluded* from the warn set because they ship in
   the default ``symbolic`` mode bundle (``sim_options.py:391``,
   ``:374``); warning every default-options ``entry_state()`` would
   generate noise the user did not consent to. They remain
   divergence-risk in this matrix, just not warn-on-add.

   **Followup (angr-xghv, 2026-05-16):** The user-opted
   ``TRACK_*_ACTIONS`` family (``TRACK_MEMORY_ACTIONS``,
   ``TRACK_REGISTER_ACTIONS``, ``TRACK_TMP_ACTIONS``,
   ``TRACK_JMP_ACTIONS``, ``TRACK_OP_ACTIONS``,
   ``TRACK_ACTION_HISTORY``) was promoted from warn-once to **raise
   ``NotImplementedError`` at manager construction** via
   ``_RAISE_OPTION_NAMES``. Rust never emits ``SimAction`` records, so
   any analysis driven off ``state.history.actions`` gets silently
   empty data — loud failure beats silent divergence. Note
   ``TRACK_OP_ACTIONS`` ships in the ``fastpath`` mode bundle
   (``sim_options.py:411``); fastpath users must drop to the Python
   engine.

   **Followup (angr-gmrc, 2026-05-16):** ``CONCRETIZE`` was promoted
   from warn-once to **raise ``NotImplementedError``** via the same
   ``_RAISE_OPTION_NAMES`` mechanism. The Python engine routes this
   option through ``SimSolver.BatchedConcretizationBacker`` to eagerly
   concretize every freshly-introduced symbol; the Rust engine has no
   equivalent hook, so accepting the option silently would let
   symbolic-driven analyses behave as if the option were absent — a
   semantic divergence that's hard to diagnose. Users hitting this
   should drop to the Python engine.

   **Followup (angr-csmm, 2026-05-16):** ``CONSERVATIVE_WRITE_STRATEGY``
   was promoted from warn-once to **raise ``NotImplementedError``** via
   the same mechanism. The Python engine routes this option through
   ``SimSymbolicMemory.concretize_write_addr`` to refuse concretization
   on range-check failure; the Rust ``SymbolicMemory`` always
   concretizes within strategy limits, so silent acceptance would
   defeat the user's intent to keep the analysis conservative. The read
   variant ``CONSERVATIVE_READ_STRATEGY`` remains warn-only because no
   user request to surface it loudly has come in yet — the asymmetry
   tracks ticket scope, not behavior difference.

   **Followup (angr-cf9h, 2026-05-16):** ``DO_RET_EMULATION`` and
   ``CALLLESS`` were promoted from warn-once to **raise
   ``NotImplementedError``** via the same mechanism. The Python engine
   emits an emulated ret successor when ``DO_RET_EMULATION`` is set and
   replaces every call with an unconstraining of the return register
   when ``CALLLESS`` is set; Rust does neither. Both options are
   typically set together by Callable workflows, where silently
   ignoring them would change the structure of the successor set or
   step into callees that the caller intended to skip. The paired
   guard ``TRUE_RET_EMULATION_GUARD`` remains warn-only because alone
   it has no effect — the guard only matters when ``DO_RET_EMULATION``
   is also set, and that case now raises before the guard is consulted.

   **Followup (angr-n129, 2026-05-16):** ``EFFICIENT_STATE_MERGING``
   was promoted from "Ignored — no-op" to **raise
   ``NotImplementedError``** via the same mechanism. The Python engine
   reads the option from ``SimStateHistory.set_strongref_state`` to
   retain a strong reference to each ancestor state so ``state.merge()``
   can find a common ancestor for plugin merging. The Rust engine does
   not drive ``SimStateHistory``'s strongref path, so the option was
   silently dropped — and the ``Veritesting`` exploration technique
   auto-adds the option at ``step_state`` time, meaning Veritesting
   under Rust would have silently run without ancestor refs and
   produced weak-ref merges. The paired ``SIMPLIFY_MERGED_CONSTRAINTS``
   is NOT promoted because it ships in the default ``symbolic`` mode
   bundle (``simplification`` set); it is honored implicitly through
   the Python ``state.merge()`` fallback inside
   ``RustExplorationManager.merge()``, which exports states to Python
   and calls ``state.merge()`` per group.

   **Followup (angr-apre, 2026-05-17):**
   ``SYMBOL_FILL_UNCONSTRAINED_REGISTERS`` was promoted to **raise
   ``NotImplementedError``** at manager construction. The Python filler
   (``angr/state_plugins/light_registers.py::_fill``,
   ``angr/storage/memory_mixins/default_filler_mixin.py::_default_value``)
   creates a fresh symbolic BVS on every read of an uninitialized
   register; the Rust ``RegisterFile`` (``native/angr/src/arch/mod.rs``)
   always returns concrete zero from its ``vec![0; size]`` storage with
   no "uninitialized" marker, so register reads cannot generate fresh
   symbols regardless of options. Silent acceptance means a user who
   opted into symbolic-fill would never see the divergence — paths
   driven by unconstrained initial register values would simply not be
   explored. The MEMORY variant ``SYMBOL_FILL_UNCONSTRAINED_MEMORY`` is
   NOT promoted because Rust's ``load_concrete_lazy``
   (``native/angr/src/memory/load.rs:333-339``) already falls back to a
   fresh ``unc_mem_*`` symbolic BVS when ``zero_fill_unconstrained`` is
   unset — i.e., symbolic-fill is Rust's default for memory.

   **Followup (angr-383x, 2026-05-11):** For the default-bundle options
   above, materialized Rust-owned states have their ``history`` plugin
   promoted to ``_RustOwnedSimStateHistory``
   (``angr/exploration/rust_state_export.py``), which warns once per
   process the first time a caller reads ``state.history.actions`` or
   ``state.history.events``. This catches the divergence at the moment
   the empty stream is actually consumed — users who never read
   ``.actions``/``.events`` see nothing, those who do get a single clear
   ``UserWarning`` pointing them at the Python engine for action-driven
   analyses.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Option
     - What Python does
     - Suggested fix
   * - ``KEEP_IP_SYMBOLIC``
     - Allows IP to remain symbolic across blocks.
     - **Honored** as of 2026-05-17 (angr-ph9z) — see the table above. The
       concretized ``state.pc`` drives the next block lift while each
       successor's IP register keeps the original symbolic expression.
   * - ``NO_IP_CONCRETIZATION``
     - Aborts on symbolic IP instead of concretizing.
     - **Honored** as of 2026-05-17 (angr-yl5n) — see the table above.
       Symbolic IPs route silently to the ``unconstrained`` stash.
   * - ``ENABLE_NX``
     - Raises on execution from non-X pages.
     - **Honored** as of 2026-05-17 (angr-dcva) — see the table above.
       Gated on ``STRICT_PAGE_ACCESS`` in addition, matching Python.
   * - ``NO_SYMBOLIC_JUMP_RESOLUTION``
     - Suppresses symbolic-jump enumeration.
     - **Honored** as of 2026-05-17 (angr-zmha) — see the table above.
       OR'd with ``no_ip_concretization`` at the same short-circuit;
       symbolic targets route to ``unconstrained`` without enumeration.
   * - ``NO_SYMBOLIC_SYSCALL_RESOLUTION``
     - Same, for syscalls.
     - **Honored** as of 2026-05-17 (angr-gffd). Symbolic syscall numbers
       always route to the Python callback, where
       ``engines/successors.py::_resolve_syscall`` reads the option from
       state.options and either enumerates concrete syscalls or routes to
       the unknown-syscall stub. Before the fix, a symbolic ``rax``
       silently dispatched to ``NativeReadSyscall`` (amd64 syscall 0)
       because ``get_syscall_num`` was ``unwrap_or(0)``.
   * - ``AVOID_MULTIVALUED_READS`` / ``AVOID_MULTIVALUED_WRITES``
     - Returns unconstrained instead of enumerating addresses.
     - (a) implement — Rust always enumerates within strategy limits.
   * - ``CONCRETIZE_SYMBOLIC_WRITE_SIZES``
     - Concretizes the *size* of a symbolic-sized write.
     - (a) implement — Rust concretizes addresses but not sizes the same
       way.
   * - ``CONSERVATIVE_WRITE_STRATEGY``
     - Refuses to concretize symbolic-write addresses on range-check
       failure.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). Silent ignore can mask an
       intended-conservative analysis.
   * - ``CONSERVATIVE_READ_STRATEGY``
     - Refuses to concretize symbolic-read addresses on range-check
       failure.
     - **(b) explicitly reject** — silent ignore can mask
       intended-conservative analyses.
   * - ``CONCRETIZE``
     - Eagerly concretizes every symbol introduced.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). Totally changes semantics; silent
       ignore is dangerous.
   * - ``ZERO_FILL_UNCONSTRAINED_REGISTERS``
     - Default-zero registers instead of fresh symbols.
     - (a) implement — pairs with the already-honored memory variant.
       Currently Rust always picks one or the other depending on
       init-state plumbing.
   * - ``SYMBOL_FILL_UNCONSTRAINED_REGISTERS``
     - Force symbolic fill on uninitialized register reads.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). The Rust ``RegisterFile`` always returns
       concrete zero from its ``vec![0; size]`` storage regardless of
       options — silent ignore means a user who opted into symbolic-fill
       gets concrete zeros instead.
   * - ``SYMBOL_FILL_UNCONSTRAINED_MEMORY``
     - Force symbolic fill on uninitialized memory reads.
     - **Matches by default.** Rust's ``load_concrete_lazy``
       (``native/angr/src/memory/load.rs:333-339``) returns a fresh
       ``unc_mem_*`` symbolic BVS when ``zero_fill_unconstrained`` is
       unset — i.e., symbolic-fill is already Rust's default for memory.
       The option is silently accepted but has no effect because Rust
       was already doing what it asks for.
   * - ``TRACK_MEMORY_ACTIONS``, ``TRACK_REGISTER_ACTIONS``,
       ``TRACK_TMP_ACTIONS``, ``TRACK_JMP_ACTIONS``,
       ``TRACK_OP_ACTIONS``
     - Populate ``state.history.actions`` with ``SimAction*`` records.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). The action stream is empty under Rust
       regardless, so silent acceptance misleads users who rely on
       ``state.history.actions``.
   * - ``TRACK_CONSTRAINT_ACTIONS``
     - Populate ``state.history.actions`` with ``SimAction*`` records.
     - **(b) explicitly reject** — ships in the default ``symbolic``
       mode bundle so cannot raise without spamming every
       ``entry_state()``; warn-on-read via
       ``_RustOwnedSimStateHistory`` instead.
   * - ``TRACK_ACTION_HISTORY``
     - Same, across path.
     - **(c) raise NotImplementedError**.
   * - ``TRACK_MEMORY_MAPPING``
     - Logs map/unmap into ``state.history``.
     - (b) explicitly reject.
   * - ``CONSTRAINT_TRACKING_IN_SOLVER``
     - Required for ``solver.unsat_core()``.
     - (a) implement — Rust solver tracks constraints internally but
       ``unsat_core`` is not surfaced.
   * - ``BYPASS_UNSUPPORTED_IROP``, ``BYPASS_ERRORED_IROP``,
       ``BYPASS_UNSUPPORTED_IREXPR``, ``BYPASS_UNSUPPORTED_IRSTMT``,
       ``BYPASS_UNSUPPORTED_IRDIRTY``, ``BYPASS_UNSUPPORTED_IRCCALL``,
       ``BYPASS_ERRORED_IRCCALL``, ``BYPASS_UNSUPPORTED_SYSCALL``,
       ``BYPASS_ERRORED_IRSTMT``, ``BYPASS_VERITESTING_EXCEPTIONS``,
       ``UNSUPPORTED_BYPASS_ZERO_DEFAULT``,
       ``UNSUPPORTED_FORCE_CONCRETIZE``
     - Tell the Python engine to swallow / fall back to a default on
       unsupported VEX.
     - (a) implement — Rust has its own error path; the bypass set is
       not consulted. Resilience modes that work in Python may abort
       under Rust.
   * - ``UNINITIALIZED_ACCESS_AWARENESS``, ``BEST_EFFORT_MEMORY_STORING``
     - Affect SimMemory error handling.
     - (b) explicitly reject.
   * - ``DO_RET_EMULATION``
     - Add emulated ret-site successors.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). Rust does not emulate rets, so the
       emulated successor is silently missing — typically breaks
       Callable workflows.
   * - ``TRUE_RET_EMULATION_GUARD``
     - Forces the emulated ret-site guard to ``true``.
     - **(b) explicitly reject** — only meaningful when paired with
       ``DO_RET_EMULATION``, which already raises; the guard alone
       has no effect.
   * - ``CALLLESS``
     - Replaces calls with unconstraining of return register.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). Rust has no equivalent short-circuit
       path; silent ignore steps into the callee.
   * - ``EFFICIENT_STATE_MERGING``
     - Retains strong refs on ``SimStateHistory`` ancestors so
       ``state.merge()`` can find a common ancestor for plugin
       merging.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). Rust does not drive
       ``SimStateHistory``'s strongref path; silent ignore would let
       Veritesting (which auto-adds the option) run without ancestor
       refs and produce weak-ref merges. The paired
       ``SIMPLIFY_MERGED_CONSTRAINTS`` is honored implicitly through
       the Python ``state.merge()`` fallback inside
       ``RustExplorationManager.merge()``.
   * - ``SUPER_FASTPATH``, ``FAST_MEMORY``, ``FAST_REGISTERS``,
       ``UNDER_CONSTRAINED_SYMEXEC``
     - Select alternate Python engines / memory plugins.
     - (b) explicitly reject — fundamentally incompatible with Rust state
       model.
   * - ``PRODUCE_ZERODIV_SUCCESSORS``
     - Spawns successor with ``divisor == 0``.
     - (a) implement — Rust treats div-by-zero as a single state.
   * - ``EXTENDED_IROP_SUPPORT``
     - pyvex extended ops; Rust may not handle every op.
     - (a) implement / audit per-op coverage.

Ignored — no-op
~~~~~~~~~~~~~~~

These options gate Python-only features that Rust simply doesn't have.
The silent ignore is harmless: they have no effect under either engine
in Rust mode.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Reason
   * - ``ABSTRACT_MEMORY``
     - Rust uses ``RustSimMemory`` only — there is no SimAbstractMemory
       backend.
   * - ``ABSTRACT_SOLVER``
     - Rust does not have an abstract-domain solver.
   * - ``AST_DEPS``, ``ACTION_DEPS``, ``AUTO_REFS``, ``ADD_AUTO_REFS``
     - Dependency tracking on SimActions; Rust produces no actions.
   * - ``REVERSE_MEMORY_NAME_MAP``, ``REVERSE_MEMORY_HASH_MAP``,
       ``MEMORY_SYMBOLIC_BYTES_MAP``
     - Python SimSymbolicMemory bookkeeping.
   * - ``REGION_MAPPING``
     - Python memory plugin region log.
   * - ``DO_CCALLS``, ``USE_SIMPLIFIED_CCALLS``
     - Rust executes VEX directly without ccall helpers.
   * - ``SYMBOLIC_TEMPS``
     - Rust temps are SSA-style register internals.
   * - ``SPECIAL_MEMORY_FILL``
     - Python memory fill hook; Rust uses its own fill path.
   * - ``MEMORY_CHUNK_INDIVIDUAL_READS``
     - Python memory-bp granularity tweak.
   * - ``MEMORY_FIND_STRICT_SIZE_LIMIT``
     - Argument to Python ``SimMemory.find()``.
   * - ``DOWNSIZE_Z3``
     - Python claripy Z3 downsize; Rust manages its own context.
   * - ``REPLACEMENT_SOLVER``, ``CACHELESS_SOLVER``, ``HYBRID_SOLVER``,
       ``APPROXIMATE_FIRST``
     - Alternate claripy solver flavors.
   * - ``APPROXIMATE_GUARDS``, ``APPROXIMATE_SATISFIABILITY``,
       ``APPROXIMATE_MEMORY_SIZES``, ``VALIDATE_APPROXIMATIONS``
     - claripy-side approximation; Rust has its own concretization
       strategy.
   * - ``SYMBOLIC_MEMORY_NO_SINGLEVALUE_OPTIMIZATIONS``
     - Tracing-mode SimMemory tweak.
   * - ``CPUID_SYMBOLIC``
     - Python ccall flag.
   * - ``EXCEPTION_HANDLING``
     - Python OS plugin segfault handler.
   * - ``SYNC_CLE_BACKEND_CONCRETE``
     - Python claripy concrete backend sync.
   * - ``TRACK_SOLVER_VARIABLES``
     - claripy variable tracking (``solver.all_variables``).
   * - ``COW_STATES``, ``COPY_STATES``
     - Rust always copies on fork via ``RustSimState::fork``.
       Functionally always-on.
   * - ``UNICORN``, ``UNICORN_*`` (~10 options)
     - Rust does not integrate with the unicorn engine.
   * - ``CGC_NO_SYMBOLIC_RECEIVE_LENGTH``, ``CGC_ENFORCE_FD``,
       ``CGC_NON_BLOCKING_FDS``
     - CGC-only; Rust does not run CGC binaries today.
   * - ``JAVA_IDENTIFY_GETTER_SETTER``, ``JAVA_TRACK_ATTRIBUTES``
     - Java analysis only.

Provenance
~~~~~~~~~~

Generated 2026-05-09 from ``angr/sim_options.py`` and the SimOption
read sites in ``angr/exploration/rust_manager.py`` (lines 697–706,
1751–1756, 1962–1966, 2320–2325). When new SimOptions land in
``sim_options.py``, add a row here and either wire detection into
``rust_manager.py`` (Honored) or classify the silent-ignore reason.

state.inspect dispatch (partial)
--------------------------------

The Rust symbolic-execution engine (``RustExplorationManager``)
dispatches a curated subset of ``state.inspect`` events to Python
breakpoints. Unsupported events still raise ``NotImplementedError`` at
registration time so that gaps remain loud.

Supported events
~~~~~~~~~~~~~~~~

============   ============   =====================================================
Event          Fires when     BP attributes
============   ============   =====================================================
``mem_read``    ``after``      ``mem_read_address``, ``mem_read_length``,
                              ``mem_read_expr``, ``mem_read_endness``
``mem_write``   ``after``      ``mem_write_address``, ``mem_write_length``,
                              ``mem_write_expr``, ``mem_write_endness``
``reg_read``    ``after``      ``reg_read_offset``, ``reg_read_length``,
                              ``reg_read_expr``
``reg_write``   ``after``      ``reg_write_offset``, ``reg_write_length``,
                              ``reg_write_expr``
``instruction`` ``before``     ``instruction`` (address)
``irsb``        ``before``     ``address`` (block start)
``exit``        ``before``     ``exit_target``, ``exit_guard``,
                              ``exit_jumpkind``
============   ============   =====================================================

Each event has a corresponding bit in the inspect-enabled bitmask read
by every Rust dispatch site; with no BPs registered the cost is one
``& != 0`` branch per VEX statement of that kind. Memory-event BPs
trigger on concrete addresses only — symbolic-address loads/stores are
skipped for the MVP.

Unsupported events
~~~~~~~~~~~~~~~~~~

Registering a BP for any of ``call``, ``fork``, ``return``,
``syscall``, ``constraints``, ``simprocedure``, ``dirty``,
``address_concretization``, ``expr``, ``statement``, ``tmp_read``,
``tmp_write``, ``vex_lift``, ``symbolic_variable``, ``engine_process``,
or ``memory_page_map`` raises ``NotImplementedError`` with a message
pointing at this document. ``_NoOpInspectProxy`` previously silently
accepted every registration (see angr-osuu); raising loudly prevents
users from depending on a feature the engine cannot fulfill.

Manager-wide BP storage
~~~~~~~~~~~~~~~~~~~~~~~

Inspect breakpoints are stored on ``RustExplorationManager``, not on
individual proxies. All ``RustStateProxy.inspect`` accessors return the
same manager-wide ``RustInspectProxy`` instance, and a BP registered
once applies to every state in that manager. This is a deviation from
Python's ``SimInspector`` semantics, where ``state.copy()`` forks
breakpoints with the state.

Reentrancy
~~~~~~~~~~

A BP action that triggers another inspect dispatch on the same manager
is suppressed via ``_inspect_dispatch_depth`` — outer attrs would
otherwise be clobbered by the inner action. The state owning the
firing event is currently held by the interpreter (not in any stash),
so reads through ``RustStateProxy`` from inside a BP raise a "state not
found" error that the dispatcher swallows. Use the BP attributes
(``s.inspect.mem_read_address`` etc.) rather than going through the
proxy.

Workaround for unsupported events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registering a BP for an unsupported event raises
``NotImplementedError`` from ``RustInspectProxy._check_event``
(``angr/exploration/rust_state_proxy.py:1094-1098``). A
``RustStateProxy`` constructed without a manager routes
``state.inspect.b(...)`` to ``_NoOpInspectProxy``
(``angr/exploration/rust_state_proxy.py:966-988``), which raises the
same error on any registration attempt. The error message points at
this document so callers can find these workarounds in order:

1. **Reach the same goal through a supported event.** Memory,
   register, instruction, irsb, and exit BPs all dispatch from the
   Rust engine and cover most low-level observation use cases. For
   example, "fire on every direct call" can be approximated with an
   ``exit`` BP that filters on ``jumpkind == "Ijk_Call"``:

   .. code-block:: python

      def on_call(state):
          if state.solver.eval(state.inspect.exit_jumpkind) == "Ijk_Call":
              # Inspect the call target / caller PC here.
              ...

      mgr = RustExplorationManager(proj, [state])
      state.inspect.b("exit", when=angr.BP_BEFORE, action=on_call)

   The supported set is enumerated in the *Supported events* table
   above; check it before falling back to the next workarounds.

2. **Drop back to the Python engine** for analyses that fundamentally
   depend on an unsupported event (``call``, ``fork``, ``return``,
   ``syscall``, ``constraints``, ``simprocedure``, ``dirty``, …):

   .. code-block:: python

      import angr

      proj = angr.Project("/path/to/binary", auto_load_libs=False)
      state = proj.factory.entry_state()

      # Python engine (default) — full state.inspect coverage
      mgr = proj.factory.simulation_manager(state)
      state.inspect.b("call", when=angr.BP_BEFORE, action=my_callback)
      mgr.explore(find=0x401234)

   If only part of an exploration needs the unsupported event, run
   the Rust engine first to reach an interesting region and then
   continue with the Python engine from the resulting state(s) —
   ``mgr.found[i]`` returns full ``SimState`` objects that can seed a
   Python ``SimulationManager``.

3. **Use a coarser manager-level hook** when per-statement granularity
   is not required. ``RustExplorationManager.set_progress_callback``
   (``angr/exploration/rust_manager.py:3105``) fires every
   ``interval_steps`` steps with stash counts and elapsed time, and
   the post-exploration stashes (``mgr.found`` /
   ``mgr.found_proxies()``) expose every result for inspection
   without needing a Python-side BP at all:

   .. code-block:: python

      def progress(info):
          print(f"step={info['step_count']} found={info['found_count']}")

      mgr = RustExplorationManager(proj, [state])
      mgr.set_progress_callback(progress, interval_steps=500)
      mgr.run(max_steps=10000)

      # Inspect each found state without any state.inspect BP.
      for proxy in mgr.found_proxies():
          print(hex(proxy.addr), proxy.solver.eval(stdin_var))

Single source of truth
~~~~~~~~~~~~~~~~~~~~~~

The supported-event table above is derived from
``_INSPECT_EVENT_SPECS`` in ``angr/exploration/rust_state_proxy.py`` —
the single source of truth for which inspect events the Rust engine
dispatches, what attributes each populates, and which bit they occupy
in the Rust callbacks ``inspect_enabled`` u8 bitmask. The
manager-side registry (``_inspect_breakpoints``, ``_INSPECT_EVENT_BITS``
in ``rust_manager.py``) and the registration-time rejection in
``RustInspectProxy._check_event`` both read from this table; the
``set_inspect_<event>`` callback registration in
``RustExplorationManager.set_callbacks`` iterates over it.

To wire a new event, add a row to ``_INSPECT_EVENT_SPECS`` and follow
the 5-touchpoint pattern documented next to the table (Rust callbacks
slot + setter + dispatch helper, instrumentation site, manager-side
``_cb_inspect_<name>`` method). The
``TestRustInspectAllowlistConsistency`` test class enforces that no
event ends up in the table without a working dispatch, and that every
event in ``angr.state_plugins.inspect.event_types`` either dispatches
or raises ``NotImplementedError`` on registration (no silent
pass-through).

Decision history
~~~~~~~~~~~~~~~~

* ``angr-osuu`` (2026-05-08): replaced silent no-op with
  ``NotImplementedError`` on registration.
* ``angr-mq8l`` (2026-05-10): formalized the no-dispatch limitation
  and pointed the error message at this document.
* ``angr-uq4n`` (2026-05-16): wired ``mem_read`` + ``mem_write``
  dispatch from VEX ``IRExpr::Load`` / ``IRStmt::Store``.
* ``angr-d46u`` (2026-05-22): extended dispatch to ``reg_read``,
  ``reg_write``, ``instruction``, ``irsb``, and ``exit`` events.
* ``angr-ji7h`` (2026-05-25): consolidated the allowlist to a single
  source of truth + CI tests guarding against silent pass-through.

Known slower benchmarks
-----------------------

A handful of benchmarks tracked in
``tests/benchmarks/baseline_timings.json`` run slower under the Rust
engine than under pure Python. This section documents the root cause
for each so future investigators do not re-derive the same findings,
and so the maintainer can decide whether the gap is worth closing.

Speedup column is ``python_time / rust_time`` from the live baseline
file. Numbers refresh whenever ``baseline_timings.json`` is
regenerated; treat the values here as snapshots, not invariants.

mma_howtouse — 0.59x (Py ~4.33s / Rust ~7.39s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Workload:** ``Callable`` FFI demo
(``angr-examples/mma_howtouse/solve.py``) invokes ``howtouse(i)`` 45
times in a loop. Each ``callable(i)`` constructs a fresh
``RustExplorationManager`` via the run_single.py monkey-patch on
``AngrObjectFactory.simulation_manager``. The exploration itself is
short (concrete-only, ~45 short call_state runs).

**Earlier hypothesis (now invalidated) — thread-local AST cache
lookup overhead.** Through mid-2026-05 this section claimed the
slowdown was Python-side claripy AST cache probes accumulating
entries across all 45 manager instances, with a manual
``claripy.clear_all_caches()`` cutting 6.62s → 5.07s (~23%). That
hypothesis no longer holds:

#. claripy migrated its caches to ``WeakValueDictionary`` and removed
   the public ``clear_all_caches()`` entry point. The memory
   ``avoid-trusting-stale-cache-clear-speedup`` records the 2026-05-17
   verification.
#. The Rust-side ``clear_ast_cache()`` PyO3 hook (angr-518z) targets
   only the Rust translation-side LRUs, not Python claripy state, and
   measured 0% gain on mma_howtouse (angr-gra3 validation,
   2026-05-17).
#. With ``cleanup=True`` versus ``cleanup=False`` both landing at
   ~7.3s on the post-fix tree, the AST cache is not the bottleneck.

**Current picture (HEAD 260eb4d66, 2026-05-18):** 5-sample paired
``run_single.py --both`` campaign:

* Python: median 4.33s, range 4.20–4.42s, peak_mem 226MB.
* Rust: median 7.39s, range 7.25–7.45s, peak_mem 276–277MB.
* Speedup: 0.59x (5/5 OK, no failures, very tight variance — not
  bimodal).
* Tracks the validated ``benchmark-mma-howtouse-2026-05-17-final``
  memory (HEAD 17ab6787a: 7.32s / 4.29s = 0.58x).

The ``baseline_timings.json`` rust_time of 6.513s pre-dates the post
angr-9maq / angr-gra3 / angr-8t45 wave; current Rust time sits ~13%
above it, just under the 15% PR-time regression threshold, so the
baseline is intentionally left in place (consistent with the
sokohashv2 / unbreakable_1 / fairlight precedent — see
:doc:`rust_bimodal_variance`).

**Where the gap is — per-manager memory sync (attributed 2026-05-19,
angr-i9f2).** A single-Callable ``perf_report()`` collected on HEAD
``2870a3429`` (the last manager constructed by ``run_single.py
mma_howtouse --engine rust``) breaks the per-instance cost down to:

* ``Init total``: 43.3 ms

  * Setup callbacks / load binary / register SimProcedures: ~0 ms
  * Python init: 1.2 ms
  * ``Add Rust state``: 42.0 ms

    * **Memory sync: 36.7 ms** ← dominant per-Callable cost
    * Register sync: 3.1 ms

* SimProcedure callback (``CallReturn``): 8.4 ms total
* Lift block (3 callbacks): 1.0 ms total
* Z3: 45 checks at ~7.1 ms each (320 ms cumulative)

Per-Callable Rust extras therefore sum to ~58 ms, and 45 × 58 ms ≈
2.6 s — consistent with the observed 3.0 s gap over Python.

The memory-sync slow path runs every Callable because
``_run_python_init_if_needed`` (``rust_manager.py:2317``) short-circuits
when ``state.addr`` is inside a real (non-loader) binary, so
``_mem_cache`` is never populated and ``_try_fast_memory_sync``
(``rust_state_sync.py:229``) returns ``False``. Each manager then
re-iterates ``loader.all_objects``, re-loads every page via
``loader.memory.load``, and re-issues the FFI ``map_memory_batch`` from
scratch. The loader-pages output of ``_extract_loader_pages``
(``rust_manager.py:406``) is a pure function of the loader state and is
identical for all 45 Callables.

A follow-up fix is tracked in ``angr-bzsc``: cache the loader-pages
output class-wide (keyed by ``binary_path + arch_name``) so that the
second-through-Nth Callable skip the slow loader iteration and only pay
the small per-state stack / overlay stages. Expected impact: ~36 ms ×
44 ≈ 1.6 s, which would close most of the residual 3 s gap.

**Why this was not chased earlier.** Pre-attribution, the cheap
mitigations considered (manual claripy cache clears, Rust LRU flush)
delivered ~0% on the current tree, and the gap had been left
documentation-resolved per ``angr-ed7j-doc-resolution``.

**Memory:** the original 1606MB peak was fixed independently in commit
``342df4a7f`` (2026-05-02), bringing peak to ~285MB. The current 0.59x
slowdown is purely CPU time.

**Relevant memories:** ``benchmark-mma-howtouse-2026-05-17-final``,
``avoid-trusting-stale-cache-clear-speedup``,
``mma-howtouse-leak-source``,
``benchmark-mma-howtouse-leak-fix``,
``angr-ed7j-doc-resolution``.

ekopartyctf2016_sokohashv2 — 0.36x (Py 5.83s / Rust 16.0s baseline; ~9.5s typical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Workload:** Sokoban-style CTF binary whose source uses x87
transcendentals (``fyl2x``, ``fscale``, ``f2xm1``) inside its hash
routine, but the test driver (``solve.py``) hooks every transcendental
call site with ``do_nothing`` (lines 95–107 of the example), so the
symbolically-executed code path never touches them. The baseline 16.0s
value is set at slow-mode to absorb known bimodal variance — actual
fast runs land near 9.5s (~21% improvement since commit a571d4634 from
FxHash adoption, freeze-local-assertions, and Arc-wrap fork fields).

**Root cause — Z3 solver dominates:**

A 2026-05-23 counters dump (``run_single.py --dump-counters``) shows:

* ``z3_check_time_ns`` ≈ 8.73s (out of ~10s total walltime)
* ``z3_site_eval_upto_time_ns`` ≈ 7.62s (the final ``eval_upto(buffer, 1)``)
* ``rust_run_loop_time_ns`` ≈ 144ms (Rust interpreter is negligible)
* ``vex_op_other`` = 0, ``vex_op_fp`` = 0, ``python_vex_*_fallback_count`` = 0
  — **no transcendentals (concrete or symbolic) execute under the test
  driver's hooks**

The slowdown is structurally **Z3 nondeterminism** on the final
``eval_upto`` query, not transcendentals. The solver picks one of two
(now ~three) model shapes per run, producing the historical bimodal
9.5s / 15.4s spread — now softened into a trimodal-ish ~9-17s spread
after the angr-ctct + angr-fv81 hash-divergence fixes (2026-05-14).
``rust_only=True`` in ``run_regression.py`` reflects that the same
nondeterminism causes output divergence between Rust and Python.

**Memory:** Peak memory is not an issue here; the slowdown is pure CPU
inside Z3.

**Why not chase a Rust fix?** Z3 nondeterminism is structural —
nothing in the engine controls it. The x87 transcendental code path
*is not exercised* on this benchmark (verified 2026-05-23 via counters
dump), so implementing them natively would not move sokohashv2. See bd
memory ``sokohashv2-no-transcendental-hits-2026-05-23`` and
``angr-9l1y-closed-2026-05-23``.

**2026-05-18 re-validation (post-fix):** After the angr-ctct
(commit ``c6b2824cb``) + angr-fv81 (commit ``13bb9f741``) memory-sync
fixes landed on 2026-05-14, sokohashv2 again runs to completion under
Rust. A 10-sample campaign (``bimodal_variance.py --runs 10
--benchmarks ekopartyctf2016_sokohashv2``) measured:

* min=8.63s, max=17.44s, median=11.98s, mean=12.69s, stdev=2.65s
* Histogram: 1×~8.6s, 5×~12s (11.82–12.06s), 2×~12-13s, 2×~17s (17.18s,
  17.44s)

The clean bimodal "~9.5s OR ~15.4s" picture has softened into a
trimodal-ish spread with a dominant mid-mode around 12s. Slow-mode max
17.44s stays within baseline+15% (18.4s) so the 16.0s baseline is left
unchanged. Median 11.98s gives ~0.49x; the typical 0.4x in
``CLAUDE.md`` reflects the longer-tail mean. See bd memory
``benchmark-sokohashv2-2026-05-18``.

**Relevant memories:** ``avoid-silent-zero-raw-fallback``,
``invariant-bimodal-variance-benchmarks``,
``benchmark-perf-wins-2026-05-09``,
``benchmark-sokohashv2-2026-05-18``,
``sokohashv2-no-transcendental-hits-2026-05-23``.

Other benchmarks below 1.0x
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are listed in the CLAUDE.md performance table but were not the
focus of angr-ed7j. Captured here for completeness so future drift can
be tracked.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Benchmark
     - Speedup
     - Root cause
   * - ``google2016_unbreakable_1``
     - 0.46x (slow tail)
     - **Re-bimodalized 2026-05-22 (angr-pfy4 spike resolution).**
       Was bimodal (1.6s/3.3s) pre-May; 2026-05-13 found 20/20 runs in
       3.01–3.06s (unimodal slow); 2026-05-18 (angr-hyiz.4) found 20/20
       in 2.43–2.48s and the bench was removed from
       ``BIMODAL_BENCHMARKS``. 2026-05-22 ralph iter-2 flagged 5.21s vs
       3.5s baseline, and a 15-sample re-validation on HEAD
       ``14187073d`` found a multi-modal distribution: 11/15 in
       0.91–0.99s (fast mode, well below baseline), 2/15 in 1.15s,
       1/15 in 1.86s, 1/15 in 2.65s, with the gate sample at 5.21s
       extending the slow tail. The fast mode is now ~3.6× faster than
       the May-18 median — post-May-18 perf gains (likely angr-b58a
       UltraPage memcmp + lazy-region FFI, angr-zdho z3_ast cache
       instrumentation, angr-9jly proxy fast-path) appear to have
       *widened* the gap between fast and slow Z3 modes rather than
       collapsing them. Re-added to ``BIMODAL_BENCHMARKS`` in commit
       ``e30214e88`` (run_regression.py); baseline left at 3.5s. SLA
       at fast mode would be 1.65× (1.602s Python / 0.97s Rust median)
       — a new best if not for the slow tail. Spike conclusion
       (category c, per task description): Z3 SAT heuristic
       nondeterminism dominates, not actionable in Rust. The
       ``BIMODAL_BENCHMARKS`` gate is the correct mitigation. See
       :doc:`rust_bimodal_variance` and bd memory
       ``benchmark-unbreakable_1-2026-05-22``.
   * - ``hackcon2016_angry-reverser``
     - 0.69x (Py ~10.29s / Rust ~14.84s, 5-sample median 2026-05-19)
     - The 2026-05-17 regression (11.7s → 30.6s) was bisected to
       ``fced54a07`` (angr-9maq, "skip eager allocation of all-zero
       filler pages"), which optimized ``mma_howtouse``'s memory
       blowup at the cost of a structural Z3 slowdown on hackcon2016.
       angr-8t45 added a per-state ``zero_eager_cap=200`` to
       ``_sync_extra_python_pages``: hackcon (~35 zero pages/state)
       gets the eager-map path back; mma_howtouse (~2058 zero
       pages/state) keeps the lazy path. **Reverse leaf-emission
       hypothesis invalidated (angr-tlvl, 2026-05-19):** the
       ``Reverse(x)`` leaf-case in ``build_z3_ast`` had a latent
       semantic bug (a stray ``parts.reverse()`` produced ``x``
       instead of byte-reversed x; Z3 folded it back to ``x`` so
       integration tests passed). Fixing the bug to emit
       ``Concat(extract[7:0,x], …, extract[N-1:N-8,x])`` — claripy's
       canonical shape — moved the median from ~15.22s to ~14.84s
       (within stdev). The residual hackcon cost is **not** from
       the Reverse leaf emission. Suspected next: rustbv↔claripy
       round-trip producing structurally different concat trees
       over the flag BVS, per ``hackcon-z3-ast-structure``. Python
       time unchanged (~10.3s).
   * - ``securityfest_fairlight``
     - 0.73x (slow mode)
     - Bimodal; 2026-05-13 campaign measured ~7.95s OR ~21.4s
       (15× slow, 5× fast over 20 runs). Baseline raised from 16.0s
       to 22.0s to cover slow mode. See
       :doc:`rust_bimodal_variance` and
       ``invariant-bimodal-variance-benchmarks``.
