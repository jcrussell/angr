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
* why ``state.inspect`` does not fire under it, and
* which benchmarks run slower than pure Python and why.

Overview
--------

* **Honored options:** ``LAZY_SOLVES``, ``ZERO_FILL_UNCONSTRAINED_MEMORY``,
  ``APPROXIMATE_MEMORY_INDICES``, ``SYMBOLIC_WRITE_ADDRESSES``,
  ``STRICT_PAGE_ACCESS``, ``ENABLE_NX``. Everything else is either
  inherited from Python or silently ignored — see the matrix below.
* **No** ``state.inspect`` **dispatch.** Registration raises
  ``NotImplementedError``. Use the Python engine for breakpoint-driven
  analyses.
* **Performance:** Faster than Python on most benchmarks, with a small
  number of known slower cases driven by Python-side cache pressure,
  x87 transcendental fallbacks, or bimodal Z3 solver nondeterminism.

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
   mgr.set_find_addresses([0x401234])
   mgr.set_avoid_addresses([0x401000])
   mgr.run(max_steps=10000)

   for found in mgr.found:
       print(f"Found at {hex(found.addr)}")

The ``proj.factory.simulation_manager(state, use_rust_engine=True)``
form returns a wrapper that exposes the same interface as the standard
``SimulationManager`` (``explore``, ``step``, ``found``, ``avoid``,
etc.) while running the Rust engine underneath.

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
     - 2
     - 1 (Cdecl ret reg)
     - 1 (flareon2015_2)
     - Cdecl
     - Experimental
   * - ARM (32-bit)
     - 2
     - 2 (validate, native-proc)
     - 0
     - ARMEABI
     - Experimental
   * - ARM64
     - 1
     - 4 (blob branch, NEON mla, real ELF, native-proc)
     - 1 (aarch64_le_branch synthetic)
     - AArch64
     - Supported
   * - MIPS32
     - 4
     - 3 (BE blob, LE real ELF, native-proc)
     - 1 (mips32_le_branch synthetic)
     - MipsO32
     - Supported
   * - MIPS64
     - 0
     - 2 (LE real ELF, native-proc)
     - 0
     - MipsN64
     - Experimental

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
* Endianness flag (MIPS32 BE+LE end-to-end via ELF + blob; MIPS64 LE
  end-to-end; ARM/ARM64/MIPS64 BE untested).
* ARMEABI / AArch64 / MipsO32 / MipsN64 calling conventions defined in
  ``calling_conventions.rs``.

To promote an arch from Skeleton → Experimental: add at least one
integration test that loads a real binary, runs ``mgr.run(...)``, and
verifies a found state. To promote Experimental → Supported: add a
benchmark and ensure it stays green in regression runs. The Cdecl x86
return-register bug (commit ``5329d8222``) was latent for months
precisely because no end-to-end x86 test ran — assume the same risk
for any new arch added without coverage.

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
       ``CallbackInterpreter::get_or_lift_block``. Matches Python: the X
       check fires only when ``ENABLE_NX`` AND ``STRICT_PAGE_ACCESS`` are
       both set (``angr/engines/vex/heavy/heavy.py:115-124``).

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
     - **(a) implement** — Rust always concretizes IP at block boundaries;
       symbolic-IP support is a real gap.
   * - ``NO_IP_CONCRETIZATION``
     - Aborts on symbolic IP instead of concretizing.
     - **(a) implement** — same gap as above, opposite direction.
   * - ``ENABLE_NX``
     - Raises on execution from non-X pages.
     - **Honored** as of 2026-05-17 (angr-dcva) — see the table above.
       Gated on ``STRICT_PAGE_ACCESS`` in addition, matching Python.
   * - ``NO_SYMBOLIC_JUMP_RESOLUTION``
     - Suppresses symbolic-jump enumeration.
     - (a) implement — Rust resolves symbolic jumps differently and may
       diverge on heavy-symbolic targets.
   * - ``NO_SYMBOLIC_SYSCALL_RESOLUTION``
     - Same, for syscalls.
     - (a) implement.
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
   * - ``SYMBOL_FILL_UNCONSTRAINED_MEMORY`` /
       ``SYMBOL_FILL_UNCONSTRAINED_REGISTERS``
     - Force symbolic fill (the opposite of ``ZERO_FILL_*``).
     - (a) implement — partial overlap with ``ZERO_FILL`` behavior.
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

state.inspect is not supported
------------------------------

The Rust symbolic-execution engine (``RustExplorationManager``) does
**not** dispatch ``state.inspect`` breakpoints. Any code that registers
an inspect hook on a ``RustStateProxy`` will raise
``NotImplementedError`` at registration time so that the failure is
loud.

Affected API
~~~~~~~~~~~~

All five registration methods on ``state.inspect`` raise:

* ``state.inspect.b(event, when=..., action=...)``
* ``state.inspect.make_breakpoint(event, ...)``
* ``state.inspect.add_breakpoint(event, bp)``
* ``state.inspect.remove_breakpoint(event, idx_or_bp)``
* ``state.inspect.action(event, action)``

Inspect events covered (and therefore unsupported) include
``mem_read``, ``mem_write``, ``reg_read``, ``reg_write``, ``tmp_read``,
``tmp_write``, ``address_concretization``, ``expr``, ``statement``,
``instruction``, ``irsb``, ``constraints``, ``exit``, ``fork``,
``symbolic_variable``, ``simprocedure``, ``engine_process``,
``path_step``, ``dirty``, and ``syscall``.

Why it raises instead of warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_NoOpInspectProxy`` previously silently accepted breakpoint
registration but never fired any callback (see angr-osuu). That caused
taint-tracking and analyzer techniques to appear to work but produce
wrong results. Raising ``NotImplementedError`` at registration prevents
users from unknowingly depending on a feature the engine cannot
fulfill.

Workaround
~~~~~~~~~~

Drop back to the Python engine for analyses that rely on
``state.inspect``:

.. code-block:: python

   import angr

   proj = angr.Project("/path/to/binary", auto_load_libs=False)
   state = proj.factory.entry_state()

   # Python engine (default) — state.inspect works normally
   mgr = proj.factory.simulation_manager(state)
   state.inspect.b("mem_read", when=angr.BP_BEFORE, action=my_callback)
   mgr.explore(find=0x401234)

If only part of an exploration needs inspect, run the Rust engine first
to reach an interesting region and then continue with the Python engine
from the resulting state(s) — ``mgr.found[i]`` returns full
``SimState`` objects that can seed a Python ``SimulationManager``.

Why not implement breakpoint plumbing in Rust?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust ``InspectionManager`` (in ``native/angr/src/state.rs``)
already has a ring buffer and per-event bitmask scaffold (commit
a07a999f4), but events are not auto-fired from the interpreter loop.
Adding dispatch would require:

#. Per-event call sites scattered throughout the VEX interpreter,
   memory model, solver, and exit handlers.
#. A marshalling path that turns Rust events into Python callable
   invocations on the right ``SimState``-equivalent object (today the
   callback API receives a ``SimState``, not a ``RustStateProxy``).
#. Care about reentrancy: an ``action`` callback that mutates state
   must round-trip through the Rust state without breaking interpreter
   invariants.

That is multi-session work that needs design first. Until and unless
that demand materializes, the documented limitation is the contract.

Decision history
~~~~~~~~~~~~~~~~

* ``angr-osuu`` (2026-05-08): replaced silent no-op with
  ``NotImplementedError`` on registration.
* ``angr-mq8l`` (2026-05-10): formalized the limitation (Option B),
  pointed the error message at this document, and updated CLAUDE.md.

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

mma_howtouse — 0.65x (Py 4.25s / Rust 6.51s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Workload:** ``Callable`` FFI demo
(``angr-examples/mma_howtouse/solve.py``) invokes ``howtouse(i)`` 45
times in a loop. Each ``callable(i)`` constructs a fresh
``RustExplorationManager`` via the run_single.py monkey-patch on
``AngrObjectFactory.simulation_manager``. The exploration itself is
short (concrete-only, ~45 short call_state runs).

**Root cause — thread-local AST cache lookup overhead.** angr's
claripy backend caches AST simplification results in thread-local
hashmaps. The caches accumulate entries across all 45 manager
instances (they outlive any single manager) so per-lookup cost grows
roughly with total AST count. Manually calling
``claripy.clear_all_caches()`` between ``callable()`` invocations cuts
wall time from 6.62s → 5.07s (a ~23% reduction) without affecting
correctness — but is not a default because it would penalize the
common single-long-exploration case.

**Memory:** the original 1606MB peak was fixed independently in commit
``342df4a7f`` (2026-05-02), bringing peak to ~285MB. The current 0.65x
slowdown is purely CPU time.

**Why not chase a Rust fix?** The slow path is in the Python-side
hashmap (per-``callable()`` AST equality probes), not in the Rust
engine. A real fix would need either (a) a per-manager AST cache scope
so cached entries do not leak across manager lifetimes, or (b) opt-in
``clear_all_caches()`` plumbed through Callable. Both are non-trivial
and the benchmark itself is a degenerate stress test (45 isolated
invocations) that does not reflect typical symbolic-execution
workloads.

**Relevant memories:** ``mma-howtouse-leak-source``,
``mma-howtouse-cache-clear-speedup``,
``benchmark-mma-howtouse-leak-fix``.

ekopartyctf2016_sokohashv2 — 0.36x (Py 5.83s / Rust 16.0s baseline; ~9.5s typical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Workload:** Sokoban-style CTF binary that uses x87 transcendentals
(``fyl2x``, ``fscale``, ``f2xm1``) inside its hash routine. The
baseline 16.0s value is set at slow-mode to absorb known bimodal
variance — actual fast runs land near 9.5s (~21% improvement since
commit a571d4634 from FxHash adoption, freeze-local-assertions, and
Arc-wrap fork fields).

**Root cause — two separate factors:**

#. **x87 transcendental fallback to Python.** ``fyl2x``, ``fscale``,
   and ``f2xm1`` are not implemented natively in ``vex/ops.rs``.
   Previously these silently returned 0 (see memory
   ``avoid-silent-zero-raw-fallback``, fixed in angr-n28w / commit
   ``ef020d101``). Now they correctly route to Python claripy
   operations, which means each transcendental crosses the FFI
   boundary. The benchmark hits these on every hash iteration.
#. **Bimodal Z3 nondeterminism.** The solver picks one of two model
   shapes per run, producing either ~9.5s or ~15.4s wall time.
   ``rust_only=True`` is set in ``run_regression.py`` because the same
   nondeterminism causes output divergence between Rust and Python.
   The 0.36x baseline reflects the slow mode; the typical 0.58x figure
   reflects the fast mode.

**Memory:** Peak memory is not an issue here; the slowdown is pure CPU
in the x87 fallbacks plus solver nondeterminism.

**Why not chase a Rust fix?** Implementing x87 transcendentals
natively in Rust would close part of the gap but the binary is the
only known benchmark exercising them — high implementation cost for a
niche win. Z3 nondeterminism is structural; nothing in the engine
controls it.

**Relevant memories:** ``avoid-silent-zero-raw-fallback``,
``invariant-bimodal-variance-benchmarks``,
``benchmark-perf-wins-2026-05-09``.

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
     - 0.53x
     - Was bimodal (1.6s/3.3s); 2026-05-13 campaign found 20/20 runs
       in 3.01–3.06s (stdev 0.01s) — has unimodalized at what used to
       be the slow mode. See :doc:`rust_bimodal_variance` and
       ``benchmark-google2016-unbreakable-1-regression``.
   * - ``hackcon2016_angry-reverser``
     - 0.87x
     - Close to parity, no formal investigation yet.
   * - ``securityfest_fairlight``
     - 0.73x (slow mode)
     - Bimodal; 2026-05-13 campaign measured ~7.95s OR ~21.4s
       (15× slow, 5× fast over 20 runs). Baseline raised from 16.0s
       to 22.0s to cover slow mode. See
       :doc:`rust_bimodal_variance` and
       ``invariant-bimodal-variance-benchmarks``.
