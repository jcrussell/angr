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
  ``instruction``, ``irsb``, ``exit``, ``call``, ``return``,
  ``simprocedure``, ``syscall``, ``dirty``, ``tmp_read``,
  ``tmp_write``, ``statement``, ``expr``,
  ``address_concretization``, ``symbolic_variable``, and ``fork``
  events to Python BPs. Unsupported events (``constraints``,
  ``vex_lift``, …) still raise ``NotImplementedError`` at
  registration time.
* **Performance:** Faster than Python on most benchmarks, with a small
  number of known slower cases driven by Python-side cache pressure or
  bimodal Z3 solver nondeterminism.
* **Native registry coverage** for SimProcedures and syscalls (which
  names short-circuit Python, per-arch coverage, and the bench
  fallback distribution) is indexed in
  :doc:`/extending-angr/native_coverage_matrix`.

.. _rust-engine-v1-scope:

v1.0 scope
----------

The v1.0 engine targets **single-threaded** symbolic execution against a
single in-process project. The following are explicitly out of scope and
will not be addressed before v1.0 ships; users with these needs should
keep using the Python engine or wait for a future major release.

* **Multi-threaded / parallel exploration is not supported.** Every
  ``RustExplorationManager`` entry point is gated by the GIL, and the
  load-bearing ``#[pyclass]`` types (``RustExplorationManager``,
  ``PyRustSimState``, ``RustSolverContext``) carry ``unsendable`` markers
  that prevent cross-thread sharing at compile time. This is not a
  stylistic choice — it reflects three independent architectural
  constraints: ``Rc<RefCell<SymContext>>`` in ``RustSimState`` (used at
  every fork site under a single-threaded CoW idiom), z3-rs 0.19+
  thread-local Z3 contexts (an AST handle is bound to its creating
  thread; crossing the boundary is undefined behavior), and intentional
  ``thread_local!`` AST caches in the claripy bridge. See
  :ref:`rust-engine-concurrency-send-sync-audit` below for the full
  blocker inventory, and ``docs/advanced-topics/rust_parallel_design.rst``
  for the proposed v2 path (snapshot-transport between thread-bound
  managers, leaning on the already-``Send+Sync`` ``ExplorationStateSnapshot``).

  Two related shortcuts have been ruled out and should not be revisited
  without a new audit:

  - Setting ``parallel.enable=true`` on the Z3 solver
    (``params.set_bool("parallel.enable", true)`` in
    ``build_solver_params``) is *correctness-breaking*, not merely
    performance-neutral: it produced ``IndexError``,
    ``AngrCallableError``, and ``KeyError`` on 4 of 6 sampled benchmarks
    (see ``angr-gfay``, closed 2026-05-21). Re-enabling it would require
    auditing every solver consumer for Z3-Unknown handling and pinning
    ``smt.random_seed`` for determinism.
  - Migrating ``Rc<RefCell<SymContext>>`` to ``Arc<Mutex<SymContext>>``
    does not by itself unlock parallelism. z3-rs's thread-local context
    model means AST handles still cannot migrate across threads, and the
    ``add_constraint_raw`` hot path would suffer heavy lock contention
    (≈82.5 % of csaw_wyvern calls hit the dedup fast-path; see the
    Concurrency audit section for details).

* **Multi-process exploration is also single-process.** ``RustExplorationManager``
  drives one process at a time. Distributing exploration across OS
  processes is the user's responsibility (e.g. fuzzer-driven workflows
  with separate orchestration layers, as in libafl). The
  ``RustExplorationManager`` itself is the unit of parallelism, not a
  worker inside a larger pool.

* **Real multi-process modeling is out of scope.** The engine does
  not model ``execve`` (state replacement with a new program image),
  ``clone`` (kernel-level thread/process state forking with shared
  address space), or ``wait4`` (parent blocking on child exit). This
  matches Python angr, which also does not model these — its
  ``angr/procedures/posix/fork.py`` returns a symbolic ``If(parent,
  1338, 0)`` rather than spawning a second state, and there are no
  ``execve``/``wait4`` SimProcedures wired into ``linux_kernel``. A
  binary that calls ``execve`` to swap its program image is therefore
  not analyzable end-to-end on either engine, and that gap is
  intentional for v1.0: users running multi-process targets (forking
  daemons, exec-style command runners) are expected to either hook
  the syscalls to redirect control flow, or drive each constituent
  binary as a separate ``angr.Project`` from fuzzer-style
  orchestration (libafl is the documented example). Adding real
  multi-process modeling would require a process-table abstraction
  in the kernel-syscall layer that is shared with Python angr, not a
  Rust-engine-private feature.

* **Writing symbolic output to a real OS file descriptor is out of
  scope.** ``RustPosixProxy`` (``angr/exploration/rust_state_proxy.py``)
  implements ``dumps(fd) -> bytes`` for stdin (fd=0), stdout (fd=1),
  and Rust-tracked file descriptors, mirroring
  ``SimSystemPosix.dumps(fd)`` in Python angr
  (``angr/state_plugins/posix.py:684``). There is no companion
  ``dump_fd(fd, target_fd)`` API that writes the concretized bytes
  back to a real OS file descriptor — and there is no such API in
  Python angr either, so this is a parity-preserving omission rather
  than a Rust-engine gap. Callers that need to pipe symbolic output
  to a real ``fd`` should write the bytes themselves::

      data = state.posix.dumps(0)   # works on both engines
      os.write(target_fd, data)

  Adding ``dump_fd`` would create a Rust-only convenience method
  that the Python engine does not expose, breaking the
  "behavioral parity" contract that lets ``RustExplorationManager``
  drop in for ``SimulationManager`` without surprising users.

This scope decision is informed by the Send/Sync audit
(``angr-8fo6``, 2026-06-01) and the FFI ownership audit
(``angr-t1w7``, 2026-06-03) — both audits confirmed the
single-threaded architecture is sound for v1.0 and that parallelism
requires a new epic on top of ``ExplorationStateSnapshot``, not a
typing-only refactor.

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

RustStateProxy write-through contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RustStateProxy`` is a **live, mutable view**. Writes go straight to
the Rust state — there is no Python-side shadow store. Reads after a
write observe the new value, both through the proxy and through the
underlying Rust engine when the next step executes.

Supported writes (all immediate, no queueing):

* ``proxy.regs.<name> = value`` — forwards to
  ``set_state_register_symbolic_ast``. ``value`` may be a claripy AST
  (concrete or symbolic), a Python ``int``, or ``bytes``. Ints/bytes
  are wrapped in a ``BVV`` at the register's native width.
* ``proxy.memory.store(addr, data, endness=None, size=None)`` —
  forwards to ``set_state_memory_concrete`` (concrete data) or
  ``set_state_memory_ast`` (symbolic data). ``addr`` must be concrete
  (``int`` or concrete claripy AST). Symbolic-data writes route through
  the shared symbol cache so reads return the same Z3 AST that Rust is
  tracking.
* ``proxy.solver.add(...)`` / ``proxy.add_constraints(...)`` — adds the
  constraint to a single forked Rust solver context **shared** by the
  proxy's ``solver``, ``memory``, and ``posix`` sub-proxies (angr-yodz).
  A constraint added this way is therefore honored by a subsequent
  ``proxy.posix.dumps(0)`` and by symbolic-address concretization in
  ``proxy.memory.load(...)`` on the *same* proxy. The constraint is
  **view-local**: it lands on the forked context, not the underlying
  Rust state's solver, so it does not perturb live exploration and does
  not survive into a freshly built proxy for the same state.

  This contrasts with the write-through path used during SimProcedure
  callbacks (``RustSolverProxyPlugin.add`` →
  ``add_constraints_to_state``), which persists the constraint onto the
  state's own solver (SimState semantics). To persist a constraint onto
  the state from a read-through proxy, call
  ``mgr._rust_mgr.add_constraints_to_state(state_id, [...])`` directly.

Refused writes:

* ``proxy.memory.store(symbolic_addr, ...)`` raises
  ``NotImplementedError``. The lazy Multi-cell symbolic-address path
  exists for the in-engine store but requires solver coordination the
  proxy lacks. Use a SimProcedure-style hook (``proj.hook(addr, fn)``)
  for in-exploration symbolic-address writes — SimProcedure callbacks
  receive a full ``SimState`` and writes are synced back via the lazy
  Multi-cell path.

Implementation notes:

* Register writes update the per-state ``RustRegisterProxy._cache``
  after the FFI call so a subsequent ``proxy.regs.<name>`` read returns
  the AST that was written, without an FFI round-trip.
* Memory writes do not maintain a Python cache; reads always re-query
  Rust. This keeps load semantics aligned with the Rust engine's
  page-load behavior (a write of N bytes at ``addr`` may overlap a
  later N′-byte read at ``addr ± k``).
* Symbolic register and memory writes round-trip the claripy AST
  through ``claripy_to_rustbv`` so the symbol is registered in the
  shared cache. The inverse ``get_state_register_ast`` /
  ``memory.load`` returns the same AST identity, which is what makes
  later ``proxy.solver.add(reg == K)`` constraints land on the live
  symbol Rust is tracking.

If the diff-and-push cost surfaced in earlier benchmarks becomes a
concern again, the queueing strategy in the ``angr-j28e`` design
(flush only on actually-written proxies at the next safe boundary) is
available as a follow-up optimization. The fast-tier benchmark gate
catches any regression > 15 %.

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
     - 16
     - SystemV, MS x64
     - Supported (gs_const at archinfo offset 1032; sseround at 216)
   * - x86 (32-bit)
     - 4
     - 2 (Cdecl ret reg, LE blob)
     - 9 (8 rust + CADET_00001 python-only)
     - Cdecl
     - Supported (selectors: CS/DS/ES/FS/GS/SS; bases: LDT/GDT + FS_CONST/GS_CONST placeholders)
   * - ARM (32-bit)
     - 3
     - 3 (LE validate, LE native-proc, BE blob)
     - 2 (arm_le_branch synthetic + android_arm_license_validation real ELF)
     - ARMEABI
     - Supported
   * - ARM64
     - 2
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

The per-arch *Benchmarks* counts above are a snapshot of
``tests/benchmarks/baseline_timings.json`` at the time of writing (31
entries total: 16 AMD64, 9 x86, 2 ARM, 1 ARM64, 1 MIPS32, 2 MIPS64).
That corpus grows over time (see the ``angr-vx8p`` bench-expansion epic),
so treat these as indicative — run
``jq 'keys | length' tests/benchmarks/baseline_timings.json`` for the
live total.

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

Unsupported architectures (PPC32 / PPC64 / S390X)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python angr supports several architectures the Rust engine does **not**
implement — notably PowerPC (PPC32/PPC64) and S390X. There is no
``Arch`` implementor for these in ``native/angr/src/arch/``. The
contract for them is *loud failure, then fall back to Python*:

* ``RustExplorationManager`` construction rejects an unsupported arch at
  the boundary: ``arch_from_name`` returns ``None`` → the manager raises
  rather than guessing a layout.
* The downstream ``VexArch``-keyed paths (``arch_from_vex`` and
  ``Clone for Box<dyn Arch>`` in ``arch/mod.rs``) **panic** with a
  message naming the arch and the six supported ones, instead of
  silently substituting AMD64 register offsets / word size. These paths
  are only reachable after the manager has already accepted the arch, so
  in practice the loud manager error fires first; the panics are a
  defense-in-depth backstop for snapshot-restore / interpreter-fork
  callers and the next arch port.

To run one of these architectures, use the **Python engine** (omit
``use_rust_engine=True``). Adding Rust support means writing a new
``Arch`` implementor plus a calling convention and following the
Skeleton → Experimental → Supported promotion path above.

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
precisely because no end-to-end x86 test ran — that gap is now closed
by ``test_x86_explore_blob`` (i386 ``cmp``/``je``/``jmp`` round-trip
asserting ``eax == 42`` solves ``2*eax + 16 == 100``). Assume the same
risk for any new arch added without coverage.

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

Solver incremental discipline
-----------------------------

Spike report (``angr-li83``, 2026-06-02). Companion to the determinism
audit in ``angr-iaol.2``: this section characterizes the engine's Z3
incremental-solving lifecycle — push/pop discipline, soft-assert usage,
and simplification toggles — to surface where lifecycle overhead lives
and what is *not* an optimization opportunity today.

Push/pop frame discipline
~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust engine never uses ``Z3_solver_push`` / ``Z3_solver_pop`` to
inherit a parent state's frames across a fork. ``SymContext::fork``
(``native/angr/src/symbolic/context.rs:3739``) constructs a child with
``solver: Mutex::new(None)`` and ``push_level: 0``. The Z3 solver is
re-built **lazily** on the child's first solver query
(``z3_materialize_count``); the parent's accumulated assertions
propagate via the ``z3_assertions_shared: Arc<Vec<z3::ast::Bool>>``
that is frozen on each fork and shared by structural reference. No
``Z3_solver_translate`` is invoked (that path loses every assertion on
a same-context solver, so the engine deliberately avoids it — see the
"Why not ``Z3_solver_translate``?" section of :doc:`rust_z3_sharing`).

In-flight push/pop on a single state's solver is balanced inside a
``with_z3_solver(|solver| { ... })`` closure at every call site:

* ``check_branch_feasibility`` (``context.rs:2304–2383``) — outer
  closure holds the lock; each direction does ``push → assert →
  timed_check → pop(1)``. The two pushes (one per direction) are
  serial, not nested, and dispatched from the cached-model fast path
  when available (skipping the predicted direction).
* ``eval_upto`` / ``eval_upto_wide`` (``2539``, ``2601``) — one outer
  ``push`` brackets up to *n* SAT-and-exclude iterations; one outer
  ``pop(1)`` discards every exclusion assertion in O(1).
* ``min`` / ``max`` / ``range_seeded`` (``2592``, ``2714``, ``2853``)
  — outer push brackets a binary-search loop; per-iteration
  ``push → assert(bvule/bvuge) → check → pop(1)`` keeps the search
  assertion local. The signed branch does at most one ``MinInit`` /
  ``MaxInit`` SAT call to detect a negative witness before the binary
  search.

Push/pop is O(1) in Z3 (``z3-push-pop-is-o-1-just-records``) — it
only records trail position + reinit stack position + the inconsistency
flag (~50 bytes); no clause copying. The bit-blasting cache survives
push/pop, so the binary-search variants reuse the bit-blasted
representation of every constraint across the inner loop. The
``angr-3ms1`` ``bare_z3_push_depth`` counter tracks outstanding
**unbalanced** pushes that escape the closure brackets — always 0 in
production paths today and gated by ``debug_assert`` on every fork.

Shared-lineage solver (off by default). When
``RustExplorationManager(use_shared_lineage_solver=True)`` is set, a
fresh ``SharedLineageSolver`` (``native/angr/src/symbolic/lineage.rs``)
is minted on each fork and the child writes constraints onto a
``scope_path`` instead of the per-context solver. Push/pop are
emulated: ``scope_savepoint_push`` records ``scope_path.len()`` on a
``scope_savepoints`` stack; ``scope_savepoint_pop`` truncates the
path. The shared solver itself is push/pop'd lazily by
``SharedLineageSolver::switch_to`` only when a sibling reuses the
solver — see the ``v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental``
memory for why the path is opt-in (BFS-thrash regression on
``defcon2016quals_baby-re``).

Concrete counters from ``run_single.py``:

``ais3_crackme`` (49 states, found=1):

* ``z3_check_count`` = 122 = 48 ``satisfiable`` + 48 ``branch_false``
  + 25 ``branch_true`` + 1 ``eval``.
* ``z3_materialize_count`` = 50 — every state's lazy-fork solver is
  materialized exactly once.
* ``z3_branch_model_hit`` / ``miss`` = 23 / 25 — when the parent's
  model satisfies the branch, the second direction is skipped
  (one SAT call instead of two).
* ``lineage_push_count`` / ``lineage_pop_count`` /
  ``lineage_switch_count`` = 0 / 0 / 0 — shared-lineage off.
* Per-call cost: ``z3_site_branch_false_time_ns`` / 48 ≈ 617 µs;
  ``z3_site_satisfiable_time_ns`` / 48 ≈ 5.1 ms. The push/pop-bracketed
  branch checks are ~8× cheaper than the bare incremental SAT check.

``fauxware`` (7 steps, 2 active):

* ``z3_check_count`` = 22; ``z3_materialize_count`` = 2;
  ``z3_assume_symbolic`` = 2. The 16 ``eval_upto`` checks all share
  a single outer ``push/pop`` pair, exactly as designed.

Finding 1 — no push/pop frame leak: the closure-bracketed pattern
plus the lazy-fork solver means a parent's transient push/pop frames
never reach a child. The ``bare_z3_push_depth`` invariant gate plus
the lineage's two-counter accounting are sufficient; no follow-up
work is filed.

Soft asserts and ``assert_and_track``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine has exactly **one** call site for ``assert_and_track``
(``context.rs:2073``) reached only via
``SymContext::add_constraint_tracked_indexed`` →
``RustSolverContext::add_constraint_tracked_ast`` (``solver.rs:239``).
That path is the explicit "I want ``solver.get_unsat_core()`` to name
this constraint" entry point; the Python wrapper
(``angr/exploration/rust_manager.py``) never calls it during
exploration. None of the engine's hot-path constraint sinks
(``add_constraint``, ``add_constraint_raw``, ``assume_true``,
``assume_false``) opt into tracking.

There is **no use of Z3's soft-assert API**
(``Z3_solver_assert_soft`` / weighted MaxSMT). The engine never
formulates a "prefer SAT without this constraint" query — every
assertion is hard. The path-constraint model in claripy is
fundamentally non-relaxable (an unsat path must be pruned, not
softened), so the only conceivable use case is exploration-time
hinting (e.g. "this constraint is a guess from a callback; treat it
as soft until we commit"). No such hinting layer exists today, and
introducing one would mean teaching every constraint sink to carry a
weight tag.

Finding 2 — soft asserts not actionable today: the architecture
treats constraints as one-shot hard assertions, so adding a soft tier
is a large refactor with no identified caller demand. Closed as
no-action; if a future technique (e.g. ``angr-x04s.1`` op-tree
snapshots that need to speculate over constraint sets) needs
relaxable assertions, file a fresh task with the use case attached.

Simplification toggles
~~~~~~~~~~~~~~~~~~~~~~

Z3 simplification touches the engine in three places:

1. **No call to ``Bool::simplify`` on the assertion hot path.** The
   only ``.simplify()`` invocation in ``native/angr/src/`` is the
   sampled measurement helper ``sample_simplify_skip``
   (``context.rs:989``), which runs every 64th call across
   ``assume_true`` / ``assume_false`` / ``add_constraint_raw`` and
   only records whether the simplification would have produced a
   distinct AST pointer. The simplified Bool is dropped — the
   un-simplified ``z3::ast::Bool`` is what reaches the solver. The
   sample rate keeps measurement overhead at ~1.5%.
2. **Solver-internal preprocessing.** ``build_solver_params``
   (``context.rs:1058``) sets ``bv_extract_prop=true`` and
   ``mul2concat=true``; the rest of preprocessing
   (``simplify:propagate-values:solve-eqs:bit-blast:sat``) runs on the
   first ``check()`` call after each assertion batch
   (``z3-th-rewriter-invoked-during-solver-preprocessing-on``). The
   engine never invokes ``Z3_simplify_ex`` directly.
3. **RustBV-level structural canonicalization.** Commutative ops are
   canonicalized at construction (``rustbv_commutative_canonicalize_count``
   / ``rustbv_commutative_swap_count``) so structurally-equivalent
   ASTs hit the dedup set. This is *not* Z3 simplification — it is
   pre-Z3 normalization that reduces the input AST size before
   handing to ``z3-rs``.

Concrete sample data is too thin to make a routing decision: across
the surveyed benches the sampled-population is 1–2 calls and the
reduced-count is the same number (100%). This was already flagged — at
stride 64 even ``ais3_crackme``'s 120 ``z3_assume_symbolic`` calls
yields only ~2 samples. A full-population run (stride 1) would
quantify the reduction rate per bench, but it would also tank the
benchmark gate; the measurement is a low-cost trace, not a candidate
for hot-path simplification.

Finding 3 — simplify is not on the hot path. The sampled measurement
gives a binary-pattern signal but the population is too small for
per-bench attribution. To make this actionable, the sample stride
needs to be lowered transiently under a profiling kwarg (e.g.
``rust_log=trace`` plus an env-var stride override) — but
**only inside a profiling spike**, not as a perf change.

Stride override (``ANGR_Z3_SIMPLIFY_STRIDE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sample_simplify_skip`` reads its stride from
``ANGR_Z3_SIMPLIFY_STRIDE`` via a ``OnceLock`` on first sample (same
pattern as ``ANGR_Z3_TACTIC`` / ``ANGR_Z3_QFBV_THRESHOLD``). The const
default ``SIMPLIFY_SAMPLE_STRIDE_DEFAULT = 64``
(``native/angr/src/symbolic/context.rs:360``) applies when the env var
is unset, empty, unparseable, or ``0`` — so production runs and the
benchmark gate see exactly the previous behavior. The override is
write-once per process: changing the env var after the first
``add_constraint_raw`` / ``assume_true`` / ``assume_false`` has no
effect.

Recipe for a full-population spike on a bimodal bench::

    ANGR_Z3_SIMPLIFY_STRIDE=1 \
      python tests/benchmarks/run_single.py securityfest_fairlight \
        --engine rust --counters-json \
      | jq '.branch_cond_simplify_sampled_count,
            .branch_cond_simplify_reduced_count'

A stride of 1 calls ``Bool::simplify`` on every assertion (one extra
Z3 call per assertion). On dense-assertion benches this measurably
slows the run — never enable in CI; reach for it only inside an
ad-hoc profiling session.

Follow-up beads
~~~~~~~~~~~~~~~

* **angr-ogko** (closed 2026-06-02): the env-var stride override above
  shipped per the ``li83`` follow-up. No other follow-up beads filed —
  the discipline audit is otherwise clean.

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
``tests/engines/rust/test_solver_ops.py::TestDeterministicMode`` —
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
  ``z3_check`` time it gates. See ``angr-dtrl`` for the audit and the
  original wall-clock A/B.

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

Counter prevalence across the fast-tier corpus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instrumentation is process-wide and free to read, but interpreting
a single bench in isolation can mislead. The table below summarizes
five corpus sweeps run on the fast-tier benches in
``baseline_timings.json`` (2026-06-02) and is intended as a reference
for users debugging an unexpected counter value — "is this number
normal for this counter?". Each row links the counter family to the
sweep that audited it.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Counter family
     - Zero on …
     - Top hits
   * - ``syscall_python_fallback_count``
     - 19/19 benches
     - No bench exercises an unregistered ``(arch, num)`` pair on
       AMD64. The ~60-handler native registry covers the full
       CTF-heavy corpus end-to-end. Implication: adding new syscall
       handlers does not move bench numbers until a workload hitting
       an uncovered syscall lands.
   * - ``simprocedure_python_fallback_count`` (by name)
     - 7/17 benches
     - 74 total fallbacks corpus-wide. Top names:
       ``__stack_chk_fail`` (14, ``defcon2016quals_baby-re`` only),
       ``my_scanf`` (13, same bench, app-specific),
       C++ stdlib ops (``operator<<``, ``operator new``,
       ``operator delete``, ``memmove``: 5–7 each on
       ``csaw_wyvern`` only), ``UserHook`` (6, by construction
       Python-only), ``CallReturn`` (2). The only "easy port"
       candidates concentrate in benches already at 16.9x speedup
       (``csaw_wyvern``).
   * - ``concretize_*``
     - 17/20 benches
     - Three benches drive 100% of concretize cost:
       ``csaw_wyvern`` (28 writes × 7.2 ms = 85% of Rust loop
       time, max_cands = 1), ``sym-write`` (16 calls, max_cands = 2,
       2.2 ms total), ``flareon2015_5`` (48 writes, max_cands = 64,
       5.3 ms total). ``concretize_disjunction_count = 0`` across
       the whole corpus — no bench triggers the disjunction-hoist
       path today.
   * - ``add_constraint_raw_dedup_hit``
     - 0% hit rate on 11/15
     - Bimodal: see the per-bench table in the bullet list above.
       Top four (``defcon2016quals_baby-re`` 92.3%,
       ``csaw_wyvern`` 82.5%, ``whitehatvn2015_re400`` 50.0%,
       ``flareon2015_5`` 31.3%) account for all duplicate-assert
       traffic; the rest pay only the bounded HashSet-insert cost.
   * - ``mem_ite_depth_*`` /
       ``mem_{load,store}_symbolic_addr``
     - 15/16 (ITE depth);
       16/16 (symbolic-addr)
     - Only ``sym-write`` materializes a non-trivial read-over-write
       ITE tree (max_depth = 8, total = 576). Every other fast-tier
       bench deflects symbolic addressing to concretization before
       building deep ITEs. ``mem_load_symbolic_addr`` /
       ``mem_store_symbolic_addr`` are zero everywhere — the public
       ``load(addr_bv)`` / ``store(addr_bv)`` symbolic-address path
       is never taken in the surveyed corpus.

Headline takeaway: the engine's "expensive symbolic" paths
(syscall fallback, simprocedure fallback, symbolic-address memory,
read-over-write ITEs) are either zero or pinned to a small number of
known benches across this corpus. New optimizations targeting any of
these families need a representative workload that *uses* the path
before a bench delta is plausible — counter-driven hypotheses without
a positive sample tend to optimize already-fast code.

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
     - ``rust_manager.py:1417``, ``:3896`` (re-checked at ``explore()`` time)
     - Calls ``set_lazy_solves(True)``; Rust skips per-block satisfiability
       checks. Also propagated through ``_apply_state_metadata`` on
       disk-cache reuse (``:3196``).
   * - ``ZERO_FILL_UNCONSTRAINED_MEMORY``
     - ``rust_manager.py:1420``
     - Calls ``set_zero_fill_unconstrained(True)``; uninitialized memory
       reads return ``BVV(0, n)`` instead of fresh symbols.
   * - ``APPROXIMATE_MEMORY_INDICES``
     - ``rust_manager.py:1424`` (consumed at ``:1443``)
     - Passed as ``use_approx`` to ``configure_concretization_strategies``;
       Rust adds an approximation strategy ahead of full Z3 enumeration
       on symbolic loads.
   * - ``SYMBOLIC_WRITE_ADDRESSES``
     - ``rust_manager.py:1425`` (consumed at ``:1443``)
     - Passed as ``sym_write`` to ``configure_concretization_strategies``;
       Rust permits multi-valued symbolic write targets instead of
       forcing concretization.
   * - ``STRICT_PAGE_ACCESS``
     - ``rust_manager.py:3517`` (per-state, also propagated through
       ``_apply_state_metadata`` on disk-cache reuse)
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
       both set (``angr/engines/vex/heavy/heavy.py:118-127``).
   * - ``NO_IP_CONCRETIZATION``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_no_ip_concretization(True)``; at block
       boundaries with a symbolic jump target the state routes to the
       ``unconstrained`` stash silently instead of being enumerated. The
       check fires inside ``VEXInterpreter::eval_next_addr_concretized``
       (``native/angr/src/interpreter/exits.rs``). Matches Python's
       ``engines/successors.py:290-294`` behavior
       (``max_targets=0`` with ``skip_max_targets_warning=True``).
   * - ``NO_SYMBOLIC_JUMP_RESOLUTION``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_no_symbolic_jump_resolution(True)``;
       OR'd with ``no_ip_concretization`` at the
       ``eval_next_addr_concretized`` short-circuit, so a symbolic jump
       target routes to the ``unconstrained`` stash without enumeration.
       Matches Python's ``engines/successors.py:232-237`` (early elif
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
       ``engines/successors.py:295-304,325-326``.

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
   * - ``CGC_NO_SYMBOLIC_RECEIVE_LENGTH``, ``CGC_ENFORCE_FD``
     - DECREE (CGC) syscalls have native handlers in
       ``native/angr/src/syscalls/cgc.rs`` for the concrete-fd happy
       paths (transmit to stdout/stderr clones, receive on fd 0). The
       Rust handlers fall back to Python on symbolic counts and on
       non-standard fds, so the Python SimProcedure honors the option
       on the fallback path. CGC SimProcedure coverage landed in
       commits 0e3b480de (5 of 7 syscalls) and e16ca3bf3 (allocate /
       deallocate); CADET_00001 buffer-overflow runs end-to-end under
       Rust.

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
   the default ``symbolic`` mode bundle (``sim_options.py:392``,
   ``:375``); warning every default-options ``entry_state()`` would
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
   (``sim_options.py:412``); fastpath users must drop to the Python
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
   (``native/angr/src/memory/load.rs:384-394``) already falls back to a
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
     - **Honored** as of ``angr-tfic``. Read from ``state.options`` and
       forwarded to ``configure_concretization_strategies``
       (``rust_manager.py``, ``_init_states`` ~:1362-1386), which
       switches the Rust memory plugin to return an unconstrained value
       rather than enumerate when the option is set.
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
     - **Matches by default.** Rust's ``RegisterFile``
       (``native/angr/src/arch/mod.rs``) always returns concrete zero
       from its ``vec![0; size]`` storage — i.e., zero-fill is Rust's
       default for registers (the pair to
       ``SYMBOL_FILL_UNCONSTRAINED_REGISTERS``, which raises because
       Rust cannot produce symbolic register fill). The option is
       silently accepted but has no effect because Rust was already
       doing what it asks for. Documented as matches-by-default in
       angr-rhe2 (2026-06-03).
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
       (``native/angr/src/memory/load.rs:384-394``) returns a fresh
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
     - Metadata flag toggled around preconstraint; not a
       ``TRACK_*_ACTIONS`` recording gate.
     - **Honored silently** (demoted from raise in ``angr-fkvt``). Unlike
       its ``TRACK_*_ACTIONS`` siblings it does not gate action recording;
       its only consumer is ``preconstrainer.py`` which clears/restores it
       to suppress action recording during preconstraint — a vacuous
       no-op under Rust, which never records actions. Removed from
       ``_RAISE_OPTION_NAMES`` (unblocks AEG workloads, ``angr-86c4``).
   * - ``TRACK_MEMORY_MAPPING``
     - Logs map/unmap into ``state.history``.
     - (b) explicitly reject.
   * - ``CONSTRAINT_TRACKING_IN_SOLVER``
     - Required for ``solver.unsat_core()``.
     - (a) implement — Rust solver tracks constraints internally but
       ``unsat_core`` is not surfaced.
   * - ``BYPASS_UNSUPPORTED_IROP``, ``BYPASS_UNSUPPORTED_IRDIRTY``,
       ``BYPASS_UNSUPPORTED_IRCCALL``, ``BYPASS_UNSUPPORTED_SYSCALL``
     - Tell Python's ``HeavyResilienceMixin``
       (``engines/vex/heavy/resilience.py``) to substitute a default
       symbol/zero on unsupported ops/dirty helpers/ccalls/syscalls.
     - (a) honored transparently. Rust's interpreter routes unsupported
       VEX features through ``FallbackStrategy::PythonCallback``
       (``native/angr/src/interpreter/mod.rs:277-287``, variant at
       ``:281``); Python re-runs
       the block, sees the option, and substitutes. Syscalls fall back
       to Python's syscall engine via ``_handle_syscall_callback``,
       which also honors ``BYPASS_UNSUPPORTED_SYSCALL``.
   * - ``BYPASS_UNSUPPORTED_IREXPR``, ``BYPASS_UNSUPPORTED_IRSTMT``
     - Defined in ``sim_options.py`` but never consulted anywhere in
       angr today (vestigial).
     - (a) honored vacuously — neither option has any wired effect in
       the Python engine either.
   * - ``UNSUPPORTED_BYPASS_ZERO_DEFAULT``,
       ``UNSUPPORTED_FORCE_CONCRETIZE``
     - Modify what Python substitutes when a ``BYPASS_UNSUPPORTED_*``
       fires (zero instead of symbol; or try concretizing inputs first).
     - (a) honored transparently via the same Python-fallback path as
       the ``BYPASS_UNSUPPORTED_*`` siblings.
   * - ``BYPASS_ERRORED_IROP``, ``BYPASS_ERRORED_IRCCALL``,
       ``BYPASS_ERRORED_IRSTMT``
     - Tell Python's ``HeavyResilienceMixin`` to catch
       ``SimOperationError`` / ``SimError`` raised during op / ccall /
       stmt evaluation and substitute a default.
     - **(c) raise NotImplementedError** at manager construction (see
       ``_RAISE_OPTION_NAMES``). Rust's ``Op`` / ``TypeMismatch`` /
       ``InvalidIR`` errors map to ``FallbackStrategy::Panic`` and move
       the state to the errored stash without falling back to Python —
       the bypass never fires. Silent divergence from the same state
       run under Python (angr-6rz8, 2026-06-03).
   * - ``BYPASS_VERITESTING_EXCEPTIONS``
     - Tells ``analyses/veritesting.py`` to forward ``resilience=True``
       to the nested ``SimulationManager.run`` call.
     - **(b) explicitly reject** — only consulted from inside Veritesting,
       which already raises via ``EFFICIENT_STATE_MERGING`` (Veritesting
       auto-adds that option). Outside Veritesting the option is a
       no-op. Carried by the ``angr.options.resilience`` bundle, so
       reject-with-warn rather than raise to keep bundle users alive
       (angr-6rz8, 2026-06-03).
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
   * - ``USE_SYSTEM_TIMES``
     - Return the host's real ``int(time.time())`` from the time
       SimProcedures instead of a fresh symbolic value.
     - **(b) explicitly reject** — the native syscall handlers
       (``native/angr/src/syscalls/sim_time.rs``:
       ``gettimeofday``/``time``/``clock_gettime``) always write a fresh
       symbolic ``timeval``/``timespec`` and never consult the option.
       Warn-once so a user who opted into concrete host times learns the
       native path is ignoring it (angr-0y0v, 2026-06-15).
   * - ``PRODUCE_ZERODIV_SUCCESSORS``
     - Spawns successor with ``divisor == 0``.
     - (a) implement — Rust treats div-by-zero as a single state.
   * - ``EXTENDED_IROP_SUPPORT``
     - pyvex extended ops; Rust may not handle every op.
     - (a) implement / audit per-op coverage.
   * - ``CGC_NON_BLOCKING_FDS``
     - When *not* set, Python's ``fdwait`` SimProcedure returns symbolic
       1-bit ready flags for each fd; when set, it returns concrete 1.
     - (a) honor option. Rust's ``NativeFdwaitSyscall``
       (``native/angr/src/syscalls/cgc.rs:269``) always writes concrete
       1-bits — i.e., it behaves as if the option were set. Matches
       Python when the user opts in (the typical case for CGC analyses,
       which import the option as part of the platform contract), but
       diverges when unset: a CGC binary that branches on the symbolic
       readiness flag would explore both branches under Python and only
       the all-ready branch under Rust. None of the current benches
       exercise this path.

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
   * - ``JAVA_IDENTIFY_GETTER_SETTER``, ``JAVA_TRACK_ATTRIBUTES``
     - Java analysis only.

Provenance
~~~~~~~~~~

Generated 2026-05-09 from ``angr/sim_options.py``; line refs refreshed
2026-06-06 (angr-0jzb). SimOption read sites in
``angr/exploration/rust_manager.py``: lines 1413–1449 (per-state
detection in ``__init__``),
3196–3290 (``_apply_state_metadata`` for cached/disk-loaded states),
3516–3523 (``_add_rust_state`` ENABLE_NX / NO_IP_CONCRETIZATION /
NO_SYMBOLIC_JUMP_RESOLUTION wiring), 3893–3896 (``explore()`` re-check
for late-bound ``LAZY_SOLVES``). When new SimOptions land in
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

============   ===============   ==================================================
Event          Fires when        BP attributes
============   ===============   ==================================================
``mem_read``    ``after``         ``mem_read_address``, ``mem_read_length``,
                                 ``mem_read_expr``, ``mem_read_endness``
``mem_write``   ``before``/``after`` ``mem_write_address``, ``mem_write_length``,
                                 ``mem_write_expr``, ``mem_write_endness``
``reg_read``    ``after``         ``reg_read_offset``, ``reg_read_length``,
                                 ``reg_read_expr``
``reg_write``   ``after``         ``reg_write_offset``, ``reg_write_length``,
                                 ``reg_write_expr``
``instruction`` ``before``        ``instruction`` (address)
``irsb``        ``before``        ``address`` (block start)
``exit``        ``before``        ``exit_target``, ``exit_guard``,
                                 ``exit_jumpkind``
``call``        ``before``/``after`` ``function_address`` (call target)
``return``      ``before``/``after`` ``function_address`` (popped frame's callee)
``simprocedure`` ``before``/``after`` ``simprocedure_name``, ``simprocedure_addr``,
                                  ``simprocedure``, ``simprocedure_result``
``syscall``     ``before``/``after`` ``syscall_name``, ``simprocedure``
``dirty``       ``before``/``after`` ``dirty_name``, ``dirty_handler``,
                                  ``dirty_args``, ``dirty_result``
``tmp_read``    ``after``         ``tmp_read_num``, ``tmp_read_expr``
``tmp_write``   ``after``         ``tmp_write_num``, ``tmp_write_expr``
``statement``   ``before``        ``statement`` (stmt index in irsb.statements)
``expr``        ``after``         ``expr`` (always ``None``), ``expr_result``
``address_concretization`` ``before``/``after`` ``address_concretization_action``,
                                  ``address_concretization_expr``,
                                  ``address_concretization_result``
``symbolic_variable`` ``after``   ``symbolic_name``, ``symbolic_size``,
                                  ``symbolic_expr``
``fork``         ``after``         (none — fork event takes no attrs in angr's
                                  ``inspect_attributes`` table)
============   ===============   ==================================================

Each event has a corresponding bit in the inspect-enabled bitmask read
by every Rust dispatch site; with no BPs registered the cost is one
``& != 0`` branch per VEX statement of that kind. Memory-event BPs
trigger on concrete addresses only — symbolic-address loads/stores are
skipped for the MVP.

``call`` and ``return`` mirror Python's ``callstack.py`` semantics:
fired ``before`` and ``after`` around the Rust call-stack push (for
``call``) or pop (for ``return``). ``function_address`` is the
resolved Ijk_Call target (call) or the popped frame's ``callee_addr``
(return), wrapped in a word-sized claripy BVV for parity with
``state.regs._ip``.

``simprocedure``, ``syscall``, and ``dirty`` (angr-xmfj) are dispatched
from Python — they fire from the existing callback handlers in
``rust_callback_dispatch.py`` (``_handle_simprocedure_callback`` /
``_handle_syscall_callback_inner``) and ``rust_manager._cb_dirty_call``.
``simprocedure_result`` is ``None`` on both BEFORE and AFTER
in this MVP because capturing the proc's raw return value would
require wrapping ``proc.execute`` to observe the inner
``inst.run_func`` return.

BP attribute write-back
~~~~~~~~~~~~~~~~~~~~~~~

Python angr's ``_inspect`` semantics let a breakpoint action *override*
certain attributes to inject a value or short-circuit a computation.
The Rust engine honors a curated subset of these write-backs
(angr-uy32, angr-inh0); the rest fire read-only (mutations are ignored).

**Honored (write-back applied):**

* ``mem_read_expr`` — a ``mem_read`` BP_AFTER that assigns
  ``state.inspect.mem_read_expr`` substitutes the new value for the
  loaded result. The override is converted back to a ``RustBV`` via
  ``claripy_to_rustbv`` and used in place of the load. A width mismatch
  (the override's bit-width ≠ ``mem_read_length * 8``) is rejected
  defensively and the original value stands.
* ``mem_write_expr`` — a ``mem_write`` BP_BEFORE that assigns
  ``state.inspect.mem_write_expr`` substitutes the new value for the
  stored data *before* it is committed (angr-inh0). The ``mem_write``
  event now fires twice per ``IRStmt::Store`` to mirror Python angr:
  ``when='before'`` (pre-commit, injection point) and ``when='after'``
  (post-commit, informational). The override is width-guarded against
  ``mem_write_length * 8`` exactly like ``mem_read_expr``; a mismatch is
  rejected and the original value is stored.
* ``dirty_result`` — a ``dirty`` BP_AFTER that assigns
  ``state.inspect.dirty_result`` replaces the dirty handler's return
  value before it is concretized and handed back to the interpreter.

The write-back uses an identity check: only a BP that actually swaps the
attribute object triggers the round-trip, so an untouched read or store
costs nothing extra. The Python callback returns the mutated attribute to
the Rust dispatch site (``call_inspect_mem_read`` /
``dispatch_mem_read_inspect``, ``call_inspect_mem_write`` /
``dispatch_mem_write_inspect``), which applies it.

**Read-only (mutations ignored — file a follow-up if you need these):**

* ``reg_read_expr`` / ``reg_write_expr`` — register events fire
  ``when='after'`` and are read-only.
* ``expr_result`` — the ``expr`` event is read-only (see below).
* ``simprocedure_result`` — always ``None``; not capturable in this MVP.
* All remaining attributes on every other event are read-only.

Unsupported events
~~~~~~~~~~~~~~~~~~

Registering a BP for ``constraints``, ``vex_lift``,
``engine_process``, ``memory_page_map``, ``cfg_handle_job``,
``vfg_handle_successor``, or ``vfg_widen_state`` raises
``NotImplementedError`` with a message pointing at this document.
``_NoOpInspectProxy`` previously silently accepted every registration
(see angr-osuu); raising loudly prevents users from depending on a
feature the engine cannot fulfill.

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
(``angr/exploration/rust_state_proxy.py:2702-2706``). A
``RustStateProxy`` constructed without a manager routes
``state.inspect.b(...)`` to ``_NoOpInspectProxy``
(``angr/exploration/rust_state_proxy.py:2574-2596``), which raises the
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
   depend on an unsupported event (``constraints``, ``vex_lift``, …)
   or on an unsupported write-back of a supported event (e.g.,
   overriding ``reg_write_expr`` or ``expr_result`` — see *BP attribute
   write-back* above for the honored vs. read-only split):

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
   (``angr/exploration/rust_manager.py::set_progress_callback``,
   def at ``:3815``) fires every
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
* ``angr-4ai9`` (2026-06-03): wired ``call`` + ``return`` dispatch
  from the BlockEnd path of ``interpreter/execution.rs`` (Ijk_Call /
  Ijk_Ret). Widened ``inspect_enabled`` from ``AtomicU8`` to
  ``AtomicU16`` to make room for the two new bits alongside the
  existing 8.
* ``angr-xmfj`` (2026-06-03): wired ``simprocedure`` + ``syscall`` +
  ``dirty`` dispatch from the existing Python callback handlers
  (``rust_callback_dispatch.py``, ``rust_manager._cb_dirty_call``).
  Introduced ``dispatch_origin: 'python'`` in ``_INSPECT_EVENT_SPECS``
  so the consistency tests skip the PythonCallbacks slot check for
  events the Rust engine never invokes directly.
* ``angr-64pi`` (2026-06-03): wired ``tmp_read`` + ``tmp_write``
  dispatch from the ``RdTmp`` / ``WrTmp`` arms of
  ``interpreter/expressions.rs`` and ``interpreter/statements.rs``.
  Bits 13/14 in ``inspect_enabled``. ``RdTmp`` evaluates many times
  per IRSB (every binop arg, store data, etc.), so the
  ``AtomicU16::load + AND`` short-circuit is essential — the
  no-BP-set bench-regression gate confirmed no measurable overhead.
* ``angr-t8vf`` (2026-06-03): wired ``statement`` dispatch from the
  per-statement loop in ``interpreter/execution.rs``. Bit 15 in
  ``inspect_enabled`` — the last free slot in the ``AtomicU16``
  bitmask. Fires ``when='before'`` with ``statement`` set to the
  index into ``irsb.statements``, matching Python's
  ``SimInspectMixin._handle_vex_stmt`` BP_BEFORE signature. BP_AFTER
  is not wired (same MVP gap as ``instruction``'s BP_AFTER). Wiring
  the companion ``expr`` event requires widening
  ``inspect_enabled`` from ``AtomicU16`` to ``AtomicU32`` and is
  tracked separately.
* ``angr-lge2`` (2026-06-03): widened ``inspect_enabled`` from
  ``AtomicU16`` to ``AtomicU32`` and wired ``expr`` dispatch at bit 16.
  The dispatch site is ``eval_expr_with_callbacks`` in
  ``interpreter/expressions.rs`` — the highest-frequency call in the
  engine (fires for every constant, RdTmp, register read, load, unop,
  binop, ITE, etc.), so the bitmask short-circuit is load-bearing.
  Fires ``when='after'`` with ``expr_result`` as the
  claripy-reconstructed value; ``expr`` is always passed as ``None``
  because Rust IRExpr doesn't round-trip cleanly into ``pyvex.IRExpr``.
  User mutations to ``expr_result`` are not honored — only
  ``mem_read_expr`` / ``dirty_result`` (angr-uy32) and ``mem_write_expr``
  (angr-inh0) write-back is wired; see *BP attribute write-back* above.
* ``angr-ysml`` (2026-06-03): wired ``fork`` dispatch at the
  previously-reserved bit 4. The dispatch fires from
  ``exploration/stepping.rs`` for each forked state created by the
  deferred-fork processing (``handle_block_end`` and
  ``process_deferred_forks_into`` — covers ``fork_from_snapshot``,
  ``fork_true``/``fork_false``, and the P15 conservative
  ``fork()`` fallback path). Mirrors Python
  ``engines/successors.py:201`` where ``state._inspect("fork",
  BP_AFTER)`` fires on the newly-added successor; the Rust dispatch
  passes the FORKED state's id (not the parent) and fires BEFORE the
  satisfiability check so UNSAT-pruned forks still surface. The fork
  event takes no attrs in angr's ``inspect_attributes`` table — the
  BP gets only the state via the proxy.

Exploration technique compatibility
-----------------------------------

The Rust manager accepts ``mgr.use_technique(...)`` with the standard
``angr.exploration_techniques`` classes, but dispatch is handled in
``angr/exploration/rust_techniques.py`` rather than by the stock
``SimulationManager`` machinery — the matrix below tracks which
classes are wired natively, which run as Python fallback through
``RustStateProxy``, and which are known not to work.

**Dispatch coverage.** Of the five standard ``ExplorationTechnique``
hooks (``setup``, ``filter``, ``step``, ``step_state``, ``successors``,
``complete``), the Rust manager dispatches:

* ``setup``, ``filter``, ``complete`` — invoked per the original
  contract via ``apply_technique_filters`` /
  ``check_technique_complete`` in ``rust_techniques.py``.

  **Re-filter semantics (angr-j1ue).** Rust state ids persist across
  steps for non-forking states, so ``filter()`` is *not* re-run on a
  fixed cadence. Instead each state's ``(addr, stdout_len)`` signature
  is cached and the technique's ``filter()`` is re-evaluated whenever
  that signature changes — a filter keyed on ``state.addr`` therefore
  fires the step the address moves, not just on the state's first
  appearance. Techniques in ``_MONOTONIC_FILTER_TECHNIQUES`` (currently
  ``CheckUniqueness``, whose internal seen-set must not observe a state
  twice) are exempted and evaluated at most once per id. Note that
  ``CheckUniqueness`` normally registers *natively* and never reaches
  the Python ``filter()`` path at all; the guard only covers the
  register-detection fallback.
* ``step`` — invoked once per Rust batch via
  ``dispatch_step_with_hooks`` (``angr-rqvq``). Multiple step hooks
  compose LIFO using ``HookSet`` from ``angr/misc/hookset.py``, matching
  the standard ``SimulationManager``. The base ``simgr.step()`` call
  inside the chain advances the Rust engine by exactly one batch.
* ``step_state`` and ``successors`` — **not dispatched.** These hooks
  expect a ``SimSuccessors`` object built by Python execution, which
  the Rust engine deliberately bypasses. ``RustSimulationManagerProxy``
  raises ``NotImplementedError`` from both methods so technique bugs
  surface loudly instead of silently no-op'ing. Registration is still
  permitted (no exception at ``use_technique`` time), but the
  per-state hook never fires; ``use_technique()`` emits a
  ``logging.WARNING`` flagging the overridden hook. Drop to
  ``use_rust_engine=False`` if your workflow needs them.

Caveats specific to ``step()`` dispatch:

* Techniques that mutate stash contents via
  ``simgr.stashes[name] = [...]`` are honored (``angr-wxuo``): the
  proxy's ``_StashDict`` makes the stash contain exactly the assigned
  ``RustStateProxy`` objects, in order, via Rust ``move_state`` (no
  ``SimState`` materialization). States omitted from the new list are
  dropped from that stash; assigning non-proxy or foreign-manager states
  raises. ``simgr.move(...)`` remains available for partial moves.
* The proxy does not provide
  :class:`~angr.SimSuccessors`-level introspection (``state.history``
  is the limited :class:`RustHistoryProxy`).
* Per-state mutations (``state.regs.X = ...``,
  ``state.solver.add(...)``) **write through** to the live Rust state as
  of the write-through epic (``angr-j28e`` / ``angr-qj30`` /
  ``angr-8oiw``) — see `RustStateProxy write-through contract`_ above.
  Symbolic-address ``memory.store`` is the one refused write.

.. list-table::
   :widths: 22 16 62
   :header-rows: 1

   * - Technique
     - Status
     - Notes
   * - ``DFS`` / ``DepthFirst``
     - **Native**
     - Sets ``set_state_selection_lifo()``; the active stash is popped
       in LIFO order each step.
   * - ``BFS`` / ``BreadthFirst``
     - **Native**
     - Default ordering; ``set_state_selection_fifo()`` is also a
       no-op since FIFO is the built-in policy.
   * - ``Explorer``
     - **Native (int-only)**
     - ``find`` / ``avoid`` addresses are forwarded into Rust when
       supplied as ``int``, ``list``, ``tuple``, or ``set``. Callable
       predicates fall back to a best-effort probe over
       ``_extra_stop_points`` against a mock state — there is no
       per-step Python callback path for live predicates.
   * - ``LengthLimiter``
     - **Native**
     - ``register_length_limiter(max_length, drop)`` — drop=True
       discards over-length states; drop=False routes them to the
       ``cut`` stash.
   * - ``Timeout``
     - **Native**
     - ``register_timeout(secs)`` — wall-clock; checked between steps
       in the Rust run loop.
   * - ``CheckUniqueness``
     - **Native (x86 / AMD64)**
     - ``register_uniqueness_filter(regs)`` with a built-in default
       register list for ``X86`` / ``AMD64``; other architectures
       fall through to the Python ``filter()`` path.
   * - ``LoopSeer``
     - **Native (bound-only)**
     - A ``bound`` is translated to ``register_loop_bound(bound,
       discard_stash)``: the Rust run loop moves states whose history
       repeats any single block more than ``bound`` times to the
       technique's ``discard_stash`` (default ``spinning``). This is a
       back-edge heuristic, not the CFG-derived per-loop trip counter, so
       ``use_header`` / ``limit_concrete_loops`` are not honored. A
       bound-less ``LoopSeer`` (trip recording only) registers no native
       limiter, and a ``bound_reached`` callback disables the native path
       (it cannot be invoked from Rust) with a warning.
   * - ``LocalLoopSeer``
     - **Native (bound-only)**
     - Same translation as ``LoopSeer`` above: ``bound`` →
       ``register_loop_bound(bound, discard_stash)`` via the back-edge
       heuristic. ``bound_reached`` callbacks disable the native path with
       a warning.
   * - ``MemoryWatcher``
     - **Step hook dispatched**
     - Memory check runs in ``step()`` (``psutil.virtual_memory()`` +
       ``simgr.move``). Since ``angr-rqvq``, ``step()`` is dispatched
       per Rust batch and ``simgr.move(...)`` works against the proxy,
       so the safety valve arms.
   * - ``Spiller``
     - **Step hook dispatched, copy() now supported with GC**
     - ``step()`` is dispatched (``angr-rqvq``) and
       :meth:`RustStateProxy.copy` does a Rust-side CoW deep fork
       (``angr-d1dr``) — the spiller's snapshot path runs against
       the proxy directly. Copies are parked in the ``_copies`` stash
       and reclaimed when the proxy is GC'd by Python: ``__del__``
       calls :meth:`RustExplorationManager.drop_copy` (``angr-yhe0``).
       Callers that need eager cleanup can invoke ``drop_copy`` directly.
   * - ``Veritesting``
     - **Unsupported (raises)**
     - Auto-adds ``EFFICIENT_STATE_MERGING``, which is in
       ``_RAISE_OPTION_NAMES`` (the Rust engine does not drive
       ``SimStateHistory``'s strongref path). Construction raises
       ``NotImplementedError`` listing the offending option.
       Drop to the Python engine for veritesting workflows.
   * - ``Threading``
     - **Unsupported (untested, unsafe)**
     - Wraps ``simgr.step`` in a thread pool. The Rust engine is
       single-threaded internally, and the PyO3 ``Send`` / ``Sync``
       audit on exposed classes is still open (spike ``angr-8fo6``).
       Do not enable until that audit lands.
   * - ``Oppologist``
     - **Unsupported (overrides** ``successors`` **only)**
     - The technique works by overriding ``successors()`` to wrap the
       successor call in a ``try / except (SimUnsupportedError,
       SimCCallError)`` and replay the failing instruction under
       ``unicorn`` with aggressive concretization. The Rust manager
       does not dispatch ``successors()`` to techniques — the
       method on :class:`RustSimulationManagerProxy` raises
       ``NotImplementedError`` — so the recovery hook is never
       invoked. Even if dispatch were added, the Rust engine's
       typed ``Rust*Error`` exceptions (``RustUnsupportedVexOpError``
       and siblings) do not inherit from ``SimError`` and do not
       carry ``executed_instruction_count`` / ``ins_addr``, so the
       existing catch + ``_delayed_oppology`` flow would still need
       a rewrite. Drop to ``use_rust_engine=False`` if you need
       unicorn-fallback for unsupported VEX ops.
   * - ``Tracer``
     - **Incompatible — overrides ``step_state()``**
     - Trace-following hooks run (``angr-rqvq``) and proxy writes now
       write through (the read-only-proxy blocker is gone, write-through
       epic ``angr-j28e`` landed). The remaining blocker is dispatch:
       ``Tracer`` overrides ``step_state()`` (``tracer.py:361``), and
       ``RustSimulationManagerProxy.step_state`` raises
       ``NotImplementedError`` (``rust_state_proxy.py`` ~:3051) — the
       Rust manager never routes the step through the technique, so the
       trace-replay logic never executes. Use ``use_rust_engine=False``.
   * - ``Director``
     - **Step hook dispatched, stash assignment honored**
     - Goal-prioritisation ``step()`` runs (``angr-rqvq``) and the
       technique's ``simgr.stashes[stash] = [...]`` re-ordering /
       limiting is now applied by the proxy's ``_StashDict``
       (``angr-wxuo``) via Rust ``move_state``. Assignments of foreign
       or non-proxy states raise.
   * - ``Slicecutor``
     - **Silently no-op (step_state/successors hooks)**
     - ``filter()`` runs and ``step()`` would dispatch, but the
       slice-enforcement and successor pruning happen in
       ``successors()`` / ``step_state()`` which the Rust manager
       does not dispatch (raises ``NotImplementedError`` if called).
   * - ``DrillerCore``
     - **Step hook dispatched**
     - Drilling ``step()`` runs (``angr-rqvq``). Driller handoff and
       ``setup()`` hook installation both fire; behaviour is now bounded
       by whatever ``RustStateProxy`` exposes to driller's per-state
       inspection.
   * - ``ManualMergepoint``
     - **Step hook dispatched, copy() works but history mutation raises**
     - Merge-point ``step()`` is dispatched and ``state.copy()`` now does
       a Rust-side CoW deep fork (``angr-d1dr``), but the technique also
       mutates ``SimStateHistory``, which still raises
       ``NotImplementedError`` on the proxy.
   * - ``StubStasher``
     - **Step hook dispatched**
     - ``step()`` runs (``angr-rqvq``) and ``simgr.move("active",
       "stub", ...)`` is supported by the proxy, so the ``stub`` stash
       populates as expected.
   * - ``Stochastic`` (``StochasticSearch``)
     - **Step hook dispatched, stash mutation honored**
     - ``step()`` runs (``angr-rqvq``) and the technique's
       ``simgr.stashes[stash] = [...]`` restart / re-weight assignment is
       now applied by the proxy's ``_StashDict`` (``angr-wxuo``), so the
       restart logic takes effect.
   * - ``Bucketizer``
     - **Silently no-op (successors hook)**
     - Transition tracking via ``state.globals["transition"]`` lives
       inside ``successors()``, which the Rust manager does not
       dispatch (raises ``NotImplementedError`` if called).
   * - ``Suggestions``
     - **Step hook dispatched, heavy history access**
     - ``step()`` runs (``angr-rqvq``) but reads
       ``state.history.events`` / ``state.history.lineage`` — neither
       is exposed by :class:`RustHistoryProxy` (only ``bbl_addrs`` /
       ``recent_bbl_addrs`` / ``block_count``). Use the Python engine
       for the suggestions workflow.

Anything not listed above will be accepted, tracked in
``mgr._active_techniques``, and dispatched as ``setup`` / ``filter`` /
``step`` / ``complete`` over ``RustStateProxy``; ``successors`` and
``step_state`` hooks raise ``NotImplementedError`` from
:class:`RustSimulationManagerProxy` (see "Dispatch coverage" above). The
accept-everything default keeps construction non-fatal, but techniques
that read internal ``SimState`` plugins beyond the proxy's contract
(``state.history.parent``, ``state.history.events``,
``state.solver.constraints`` mutation) will silently misbehave rather
than raise — vet each one against the proxy read-only invariant before
counting on it in CI. ``state.copy()`` is supported as of ``angr-d1dr``
— it does a Rust-side CoW deep fork and parks the copy in a dedicated
``_copies`` stash (see the Spiller / ManualMergepoint rows below).

Hook contract
-------------

User hooks installed via :meth:`angr.Project.hook`,
:meth:`angr.Project.hook_symbol`, and :meth:`angr.Project.unhook`
work the same under the Rust engine as under the Python engine. The
Rust interpreter looks up the hook on the ``Project`` at dispatch time
and runs the callback as a Python ``SimProcedure`` via the standard
PyO3 dispatch path. The four entry points covered by
``tests/engines/test_hooks_rust.py``:

* ``proj.hook(addr, simproc, replace=True)`` — install (or override)
  a :class:`~angr.SimProcedure` at a specific address. ``replace=True``
  silently overrides any existing hook (including the auto-installed
  libc bindings created at load time).
* ``proj.hook_symbol(name, simproc, replace=True)`` — resolve the
  symbol to an address (via the loader, with a CLE extern fallback
  for weak symbols) and install the SimProcedure there. Equivalent to
  ``proj.hook(loader.find_symbol(name).rebased_addr, simproc, replace=True)``
  but covers the symbol-resolution path.
* ``proj.hook(addr, hook=callback, length=0)`` — zero-length advance
  hook. The callback runs and execution resumes at the same PC; the
  dispatcher emits ``Ijk_NoHook`` so the next step lifts the real
  instruction underneath (no re-execution loop). For ``length>0``, the
  hook replaces ``length`` bytes of code and execution resumes at
  ``addr + length``.
* ``proj.unhook(addr)`` — remove the hook at ``addr``. Subsequent
  dispatches at ``addr`` execute the original (or auto-installed)
  procedure. ``is_hooked(addr)`` returns ``False`` after the call.

Hooks are heavily exercised by the benchmark suite (``flareon2015_5``,
``flareon2015_10``, ``whitehatvn2015_re400`` and others), but the explicit smoke tests pin the
contract end-to-end so a regression here fails loudly rather than as
a benchmark slowdown.

Analyses compatibility
----------------------

.. _rust-engine-inheritance-contract:

Inheritance contract
~~~~~~~~~~~~~~~~~~~~

The ``use_rust_engine=True`` kwarg on
``proj.factory.simulation_manager(state, use_rust_engine=True)`` is a
**per-call selector**. It is not stored on the project, the factory,
or any thread-local state — every call to
``proj.factory.simulation_manager(...)`` (or its ``factory.simgr``
alias) makes an independent choice based solely on the kwarg passed
in at that call site.

Consequences:

* **Default is Python.** Any call site that omits the kwarg gets a
  plain ``SimulationManager``, even if the same caller previously
  constructed a ``RustExplorationManager`` against the same project.
* **No project-level toggle.** There is no
  ``proj.use_rust_engine = True`` or equivalent. Attaching a
  ``RustExplorationManager`` to a project does not change the default
  for future ``factory.simulation_manager()`` calls.
* **``factory.successors(state, ...)`` always dispatches to the
  Python engine.** It calls ``self.default_engine.process(...)``,
  which is wired to ``UberEngine`` / ``UberEnginePcode`` at factory
  construction time and is not influenced by Rust manager
  configuration.
* **Internal analyses always use Python.** Every analysis under
  ``angr/analyses/`` that constructs an internal SimulationManager
  goes through ``proj.factory.simulation_manager(...)`` without
  threading a ``use_rust_engine`` kwarg, and every analysis that
  steps states directly does so via ``proj.factory.successors(...)``.
  Both paths land on the Python engine regardless of how the user
  invoked the surrounding analysis.

This is a **safety default**: opting an internal analysis into Rust
execution requires either an in-tree port (verdict marked
"Works (Rust)" in the table below) or an out-of-tree monkey-patch
that swaps ``AngrObjectFactory.simulation_manager`` for the duration
of the analysis call (see ``tests/benchmarks/run_single.py`` for the
precedent and ``tests/engines/test_callable_rust.py`` for a
context-manager wrapper). The contract is enforced by
``tests/engines/test_factory_rust_inheritance.py``.

The table below records each analysis' internal stepping path and
the verdict for running it against a project that has had a
``RustExplorationManager`` attached.

.. list-table::
   :header-rows: 1
   :widths: 14 14 72

   * - Analysis
     - Status
     - Notes
   * - ``CFGFast``
     - **Works (engine-agnostic)**
     - Pure static lift + heuristics; no ``SimState`` execution. Smoke
       test: ``tests/analyses/test_cfg_fast_rust.py`` compares the
       in-text function set against the Python baseline on three
       binaries.
   * - ``BackwardSlice``
     - **Works (engine-agnostic)**
     - Consumes a ``CFGEmulated`` + ``CDG`` + ``DDG`` triple and walks
       the dependency graphs; no ``factory.simulation_manager(...)``
       call site, no ``factory.successors``. Smoke test:
       ``tests/analyses/test_slicing_ddg_vfg_rust.py``.
   * - ``DDG``
     - **Works (engine-agnostic)**
     - Reads ``state.history.actions`` off the final-state set that
       ``CFGEmulated`` records on each node (requires
       ``state_add_options=angr.sim_options.refs``). No internal
       ``SimulationManager``. Smoke test:
       ``tests/analyses/test_slicing_ddg_vfg_rust.py``.
   * - ``VFG``
     - **Works (Python engine internally)**
     - Steps states directly through ``project.factory.successors``
       and maintains its own ``VFGJob`` worklist + abstract-state
       merging; no ``SimulationManager``. ``factory.successors``
       dispatches to the Python ``SimEngine`` registry, so a
       ``RustExplorationManager`` attached to the project does not
       influence the analysis. Smoke test:
       ``tests/analyses/test_slicing_ddg_vfg_rust.py``.
   * - ``CFGEmulated``
     - **Unsupported (v1.0)**
     - Builds internal ``SimulationManager`` instances at indirect-
       jump resolution (``cfg_emulated.py:2615`` / ``:2747`` /
       ``:2754``). Has no ``use_rust_engine`` plumbing — those
       internal SMs always dispatch to the Python engine, so a
       ``RustExplorationManager`` attached to the project is unused.
       Beyond the SM bypass, the analysis also reads
       ``state.scratch.ins_addr``, ``state.scratch.exit_stmt_idx``
       and ``state.scratch.exit_ins_addr`` off successor states
       (``cfg_emulated.py:1552-1554`` / ``:1760-1761``), which is
       deep ``SimState`` machinery not modelled on
       :class:`RustStateProxy`. Workaround: feed it a fresh
       :class:`SimState` from a project that has *not* had a
       ``RustExplorationManager`` attached. No Rust-parity smoke
       test exists yet.
   * - ``Identifier``
     - **Unsupported (v1.0)**
     - Both the function-identification driver
       (``identifier/identify.py:315``) and the per-call replay path
       (``identifier/runner.py:72`` / ``:82``) build internal
       ``SimulationManager`` instances via
       ``project.factory.simulation_manager(...)``; the symbolic
       pre-amble walk uses ``project.factory.successors(...)``
       repeatedly (``identifier/identify.py:482`` / ``:493`` / ``:500``
       / ``:517`` / ``:533`` / ``:619`` / ``:728``) — both dispatch to
       the Python engine. The analysis also resets and inspects
       ``state.scratch`` directly (``runner.py:93``), which is deep
       ``SimState`` machinery not modelled on
       :class:`RustStateProxy`. Workaround: feed it a project that
       has *not* had a ``RustExplorationManager`` attached.
   * - ``Jumptable`` resolver
     - **Unsupported (v1.0)**
     - The ``JumpTableResolver`` (registered as a default
       ``IndirectJumpResolver`` for x86/AMD64/ARM, used by
       ``CFGFast`` / ``CFGEmulated`` during indirect-jump
       discovery) builds an internal ``SimulationManager`` at
       ``jumptable.py:1053`` with ``resilience=True`` and steps via
       ``project.factory.successors(...)`` (``:1701`` / ``:2060``).
       Both paths dispatch to the Python engine regardless of any
       ``RustExplorationManager`` attached to the project. The
       resolver also reads ``state.scratch.temps[...]`` directly
       (``:1746`` / ``:2072`` / ``:2361`` / ``:2365`` / ``:2371``) to
       recover jump-base addresses and guard tmps, which is deep
       ``SimState`` machinery not modelled on
       :class:`RustStateProxy`. In practice the resolver runs as a
       sub-step of ``CFGFast`` / ``CFGEmulated``; the verdict here
       inherits from those parents. Defer-fix beads can be filed
       post-v1.0 if a user needs jumptable resolution against a
       Rust-attached project.
   * - ``Veritesting``
     - **Unsupported (v1.0)**
     - Constructs ``SimulationManager`` directly
       (``veritesting.py:256``) against the Python engine, so a
       ``RustExplorationManager`` attached to the project has no
       effect on the analysis itself. Two distinct failure modes:

       1. Passing a :class:`RustStateProxy` as ``input_state`` raises
          ``NotImplementedError`` at the first ``input_state.copy()``
          call inside ``veritesting.py:215``. ``RustStateProxy.copy()``
          refuses the operation rather than returning a shallow proxy
          that aliases ``_state_id`` with the source (the previous
          behaviour silently corrupted the parent on any mutation). A
          Rust-side CoW deep fork is tracked under ``angr-2zwy``.
       2. Veritesting needs ``EFFICIENT_STATE_MERGING`` for ancestor
          retention during plugin merging. The Veritesting
          *exploration technique*
          (``exploration_techniques/veritesting.py:21-22``) auto-adds
          this option at ``step_state`` time; the option lives in
          ``_RAISE_OPTION_NAMES`` so attempting to seed a Rust
          manager with a state that already has
          ``EFFICIENT_STATE_MERGING`` set raises
          ``NotImplementedError``. The Veritesting *analysis* does
          not auto-add the option, but without it the analysis falls
          back to weak-ref merging and may miss common ancestors —
          a silent correctness gap.

       Workaround: run Veritesting on a project that has *not* had a
       ``RustExplorationManager`` attached. Construct a fresh
       :class:`angr.Project` for the analysis and feed it a
       :class:`SimState` (not a proxy). See the
       "Exploration technique compatibility" section above for the
       parallel technique-side verdict.

If you need Rust-engine execution semantics inside an analysis that
constructs its own ``SimulationManager``, the only workaround today is
to monkey-patch ``AngrObjectFactory.simulation_manager`` to return a
``RustExplorationManager`` — see ``tests/benchmarks/run_single.py`` for
the precedent and ``tests/engines/test_callable_rust.py`` for a
context-manager pattern that cleans up the swap. A first-class
``use_rust_engine`` plumb-through is tracked under the v1.0 Analyses-
compatibility epic.

Memory pressure and OOM
-----------------------

The Rust engine has no built-in memory budget or watchdog. The user-
facing knobs that cap memory growth are:

``max_active_states``
  Constructor argument to ``RustExplorationManager``. When the active
  stash reaches this size, new forks are silently pruned. ``None``
  (default) disables the cap.

``max_history``
  Constructor argument (default ``1000``). Caps the per-state
  ``state.history`` / ``detailed_history`` ring buffer. Setting it to
  ``0`` disables truncation entirely and is only safe for short runs
  — long explorations will accumulate per-state history blocks
  proportional to step count and *can OOM*.

``MemoryWatcher`` technique
  ``mgr.use_technique(MemoryWatcher(min_memory=512))`` moves the
  active stash into a ``lowmem`` stash when system free memory drops
  below the threshold (MB). Runs as Python fallback; works against
  the Rust manager because it only inspects ``psutil`` and calls
  ``simgr.move``.

External resource limit
  Set ``resource.setrlimit(RLIMIT_AS, ...)`` in the host process
  before constructing the manager. ``tests/benchmarks/run_single.py``
  uses a 4 GB ``RLIMIT_AS`` to keep the benchmark loop from
  OOM-killing the orchestrator on 8 GB / no-swap machines.

Where memory accumulates
~~~~~~~~~~~~~~~~~~~~~~~~

* **Active stash** — one ``RustSimState`` per live path (memory pages,
  register file, constraint AST trees, Z3 solver clone). The Z3 solver
  is the dominant per-state cost on constraint-heavy workloads.
* **Avoided / errored stashes** — states moved to non-active stashes
  still hold their full ``SymContext`` (including the Z3 solver
  clone). On wide explorations like ``hackcon2016_angry-reverser``
  these accumulate hundreds of solver clones; the engine drops them
  at the push sites in ``exploration/run_loop.rs`` to bound the
  per-stash cost, but a user-controlled ``mgr.avoid`` workload can
  still grow O(paths).
* **claripy AST caches** — Python-side, ``WeakValueDictionary`` based;
  trims with GC, no explicit knob.
* **Rust AST translation caches** — thread-local LRUs, bounded.
  ``RustExplorationManager(..., clear_caches_on_cleanup=True)`` flushes
  them in ``__del__`` for Callable-heavy workflows.

Behavior on exhaustion
~~~~~~~~~~~~~~~~~~~~~~

* **Python ``MemoryError`` mid-step** — propagates back through the
  PyO3 boundary as a normal Python exception. The Rust ``run`` call
  unwinds, the manager remains constructed, and the caller sees
  ``MemoryError`` at the ``mgr.run(...)`` / ``mgr.explore(...)``
  call site. States already moved to the ``found`` / ``deadended``
  stashes before the failure remain accessible.
* **Kernel OOM-kill** — the process dies; all in-memory state is
  lost. There is no graceful drain to disk. Stashes are not persisted.
* **Rust ``alloc::handle_alloc_error``** — defaults to aborting the
  process (``cargo`` release profile). ``RustOomError`` exists in the
  exception hierarchy but is currently only raised by an explicit
  test hook (``engine.rs::_raise_typed_test_error`` def ``:319``,
  ``oom`` arm ``:342``); real Rust allocation failures abort rather
  than propagate.

Recovery contract
~~~~~~~~~~~~~~~~~

The :ref:`rust-engine-state-serialization` section below documents the
v1.0 ``dump_snapshot`` / ``load_snapshot`` /
:meth:`RustExplorationManager.load_from_disk` API. Snapshots are
opt-in (the caller must explicitly persist at quiescent points), so
they do not help against a sudden kernel OOM-kill mid-step — but they
do let a long-running analysis checkpoint between exploration phases
and resume in a fresh process. The caller is still responsible for
persisting per-find interesting bits (``state.posix.dumps(0)``) from
the ``find`` callback if the workload risks termination between
checkpoints.

Recommended posture for adversarial / long-running workloads:

#. Set ``max_active_states`` to a workload-appropriate cap.
#. Add ``MemoryWatcher(min_memory=...)`` so over-budget states drain
   into a stash instead of growing the active set.
#. Tune ``solver_timeout_ms`` (constructor) to bound per-query Z3
   memory growth.
#. Run inside a process with ``RLIMIT_AS`` set, and treat the
   resulting ``MemoryError`` (or process abort, when Rust hits the
   ceiling first) as a non-recoverable terminal signal — write the
   interesting bits out from ``find`` / ``avoid`` callbacks, not at
   the end.

Deep-input-loop binaries (grub-class)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binaries that read input in a tight loop and only commit to interesting
behaviour near the loop's end (grub-style: 13 backspaces + carriage
return ≈ depth 14) are a known failure mode for the default BFS state
selection. The BFS frontier expands wide at shallow depth and exhausts
the step budget before reaching the target depth, OOM-ing the host
under a typical 4 GB ``RLIMIT_AS``.

The ``angr-xel4`` characterisation (see
``tests/benchmarks/characterization/deep_loop_search/README.md``)
measured this on a synthetic grub-proxy. BFS plateaus at depth 9–12
regardless of step budget; DFS reaches the iteration cap with lower
RSS and a smaller active stash:

.. list-table:: Deep-loop proxy: BFS vs DFS depth/RSS at N=14
   :header-rows: 1
   :widths: 12 10 10 12 16 12

   * - iter cap N
     - steps
     - strategy
     - max depth
     - active states
     - peak RSS
   * - 14
     - 100
     - BFS
     - 9
     - 65
     - 429 MB
   * - 14
     - 100
     - DFS
     - 32
     - 34
     - 371 MB
   * - 14
     - 300
     - BFS
     - 11
     - 191
     - 976 MB
   * - 14
     - 300
     - DFS
     - 32
     - 31
     - 645 MB

Recommended recipe::

    from angr.exploration import RustExplorationManager
    from angr.exploration_techniques import LengthLimiter

    mgr = RustExplorationManager(
        project, [state],
        exploration_strategy="dfs",     # LIFO state selection
        max_active_states=64,           # belt-and-braces fork cap
    )
    # Drop states past the expected vulnerability depth.
    mgr.use_technique(LengthLimiter(max_length=32, drop=True))
    mgr.explore(find=vuln_addr)

Both ``exploration_strategy="dfs"`` (FFI:
``set_state_selection_lifo``) and ``LengthLimiter`` (FFI:
``register_length_limiter``) route through native Rust setters and
compose without overriding each other. See
``TestExplorationStrategy.test_deep_loop_recipe_dfs_plus_length_limiter``
in ``tests/engines/rust/`` for the regression that
pins this composition.

When to apply this recipe:

* The binary contains an input-driven loop with bound ≥ 10 basic
  blocks to the target.
* Default BFS exploration runs out of step budget or RSS before
  reaching the target.
* The active stash grows to hundreds of low-depth states while no
  state advances past the BFS plateau depth (≈ 9–12).

DFS narrows the frontier but is **not free**: workloads that
legitimately need broad coverage (CTF crackmes, multi-path inputs)
regress under DFS. Keep BFS as the default and opt into DFS per-run
when the workload matches the deep-input-loop shape.

User-facing error taxonomy
--------------------------

Failures that originate inside the Rust engine surface to Python as one
of six typed exceptions exported from ``angr.exploration`` (re-exported
from ``angr.rustylib.vex_engine``). All five concrete classes derive
from ``RustExecutionError``, which itself derives from the built-in
``Exception``, so ``except RustExecutionError:`` is the single-clause
way to catch everything the Rust core raises.

The hierarchy and per-variant trigger conditions are derived from
``native/angr/src/errors.rs`` (variant list at lines 73–97, dispatch
table at lines 99–111) — if a future commit grows or renames a
variant, this table is the first thing to update.

.. list-table:: Typed exception hierarchy
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Trigger condition
     - Where Rust raises it
   * - ``RustExecutionError``
     - Catch-all base; raised directly when an interpreter error
       collapses to ``RustExecError::Other`` (memory mismatch,
       type-mismatch, unknown temp, Python-callback failure, lift
       error, or generic ``Unsupported`` — anything not promoted to
       a specialized subclass). Useful as a single-clause catch for
       "anything the Rust engine threw".
     - ``errors.rs:99-115``; populated via the catch-all arms in
       ``engine.rs:cb_execution_error_to_typed`` / ``op_error_to_typed``.
   * - ``RustMalformedIRSBError``
     - The pyvex lifter produced an IRSB the interpreter could not
       execute (bad/missing statements, malformed exits, invalid
       block bounds).
     - ``errors.rs:79-80``; raised from
       ``CbExecutionError::InvalidIR(reason)`` mapped at
       ``engine.rs::cb_execution_error_to_typed`` (def ``:31``,
       InvalidIR arm ``:33``).
   * - ``RustUnsupportedSyscallError``
     - A syscall handler that ran in Rust hit a number / name / arch
       combination it does not implement. Reserved class — the
       syscall fast path currently routes unimplemented numbers
       through the Python fallback (``UnsupportedFeature``), so this
       class is exposed for forward-compatibility and exercised by
       the ``_raise_typed_test_error("unsupported_syscall", ...)``
       hook (``engine.rs::_raise_typed_test_error`` def ``:319``,
       ``unsupported_syscall`` arm ``:331``). No production trigger
       in the current code.
     - ``errors.rs:82-88``; production sites land here when an
       upcoming Rust syscall handler chooses to raise rather than
       fall back to Python.
   * - ``RustUnsupportedVexOpError``
     - A VEX op (NEON / vector / unmapped opcode) is not implemented
       by the Rust interpreter. The op name and arch are baked into
       the message.
     - ``errors.rs:90-91``; populated from ``OpError::UnsupportedNeon``,
       ``OpError::UnsupportedVectorOp``, ``OpError::UnsupportedVexOp``
       at ``engine.rs::op_error_to_typed`` (def ``:53``, arms ``:55-68``).
   * - ``RustZ3Error``
     - Z3 returned an error status (not ``Unknown`` — that collapses
       to UNSAT inside ``SymContext::is_sat``). Reserved class —
       solver hangs are bounded by ``solver_timeout_ms`` and the
       ``Unknown`` collapse, so this class is exposed for
       forward-compatibility and exercised only via the
       ``_raise_typed_test_error("z3", ...)`` hook
       (``engine.rs::_raise_typed_test_error`` def ``:319``, ``z3``
       arm ``:341``).
     - ``errors.rs:93-94``.
   * - ``RustOomError``
     - Rust allocator returned a failure that the engine can
       propagate (as opposed to ``alloc::handle_alloc_error`` aborting
       the process). Reserved class — see *Behavior on exhaustion*
       above: today, real Rust allocation failures abort under the
       release profile rather than raising this. Production sites
       land here when a Rust allocator hook chooses to propagate
       instead of abort. Currently only the
       ``_raise_typed_test_error("oom", ...)`` hook
       (``engine.rs::_raise_typed_test_error`` def ``:319``, ``oom``
       arm ``:342``) raises it.
     - ``errors.rs:96-97``.

The ``RustExecError`` Rust enum is ``#[non_exhaustive]``
(``errors.rs:76``) so new variants can land in minor versions
without breaking downstream code that matches on it.

NotImplementedError at manager construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separate from the ``RustExecutionError`` family, the manager
proactively raises ``NotImplementedError`` (a built-in, **not** a
subclass of ``RustExecutionError``) when a seed state requests
behavior the Rust engine cannot provide. This fires at
``RustExplorationManager.__init__`` time — long before any Rust code
runs — so a misconfigured workload fails fast at the boundary
instead of silently diverging from the Python engine.

The triggering SimOptions are listed in ``_RAISE_OPTION_NAMES``
(``angr/exploration/rust_manager.py``); the check itself lives in
``RustExplorationManager._check_raise_options``. The current set
(commit ``d05a6006f``):

* Action-stream tracking — ``TRACK_MEMORY_ACTIONS``,
  ``TRACK_REGISTER_ACTIONS``, ``TRACK_TMP_ACTIONS``,
  ``TRACK_JMP_ACTIONS``, ``TRACK_OP_ACTIONS``,
  ``TRACK_ACTION_HISTORY``.
* Eager concretization / write-strategy refusal —
  ``CONCRETIZE``, ``CONSERVATIVE_WRITE_STRATEGY``.
* Calling-convention modeling — ``DO_RET_EMULATION``, ``CALLLESS``.
* State merging — ``EFFICIENT_STATE_MERGING``.
* Symbolic register fill — ``SYMBOL_FILL_UNCONSTRAINED_REGISTERS``.
* Errored-op bypass (Panic-strategy variants only) —
  ``BYPASS_ERRORED_IROP``, ``BYPASS_ERRORED_IRCCALL``,
  ``BYPASS_ERRORED_IRSTMT``.

The full list, including the rationale for each entry, is documented
inline above ``_RAISE_OPTION_NAMES`` in ``rust_manager.py``. The
*Ignored — divergence-risk* rows of the SimOption coverage matrix
explain why these options were promoted to raise rather than warn.

Unsupported ``state.inspect`` registration also raises
``NotImplementedError`` (``RustInspectProxy._check_event``,
``rust_state_proxy.py``); see the *state.inspect support* section
for the supported/unsupported event split.

``ExplorationTechnique`` ``setup``/``step`` for the eight rejected
techniques (Veritesting, Spiller, MemoryWatcher, etc.) raises
``NotImplementedError`` from the manager's technique-dispatch path
(``rust_manager.py``); see *Exploration technique compatibility*.

Known incompatibilities
~~~~~~~~~~~~~~~~~~~~~~~

**Oppologist is structurally incompatible.** The ``Oppologist``
exploration technique single-steps around unsupported instructions
by overriding ``successors()`` and catching
``angr.errors.SimError`` to replay the failing instruction under
``unicorn``. The Rust manager does **not** dispatch ``successors()``
to techniques (the method on :class:`RustSimulationManagerProxy`
raises ``NotImplementedError``), so the recovery hook never runs —
the typed ``Rust*Error`` propagates straight out of ``mgr.run(...)``
as if no technique were installed. Resolution: use
``use_rust_engine=False`` for the unicorn-fallback workflow. See
``angr-v4qi`` and the *Exploration technique compatibility* table
above for the full chain of issues (dispatch, exception inheritance,
``executed_instruction_count`` attribute) that would each need to be
addressed before Oppologist could work under Rust.

**Standard ``except SimError`` does not catch Rust errors.** By the
same mechanism, downstream code that ``except angr.errors.SimError:``
around ``mgr.run(...)`` will not catch the typed Rust exceptions.
Migration recipe: change the clause to
``except (angr.errors.SimError, angr.exploration.RustExecutionError):``,
or catch ``Exception`` if the surrounding code already does broad
recovery.

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
   the public ``clear_all_caches()`` entry point (verified 2026-05-17);
   the earlier ~23% "speedup" was an artifact of measuring against a
   stale cache, not a real win.
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
* Validated at HEAD 17ab6787a (7.32s / 4.29s = 0.58x).

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
``_run_python_init_if_needed`` (``rust_manager.py::_run_python_init_if_needed``,
def at ``:3154``) short-circuits
when ``state.addr`` is inside a real (non-loader) binary, so
``_mem_cache`` is never populated and ``_try_fast_memory_sync``
(``rust_state_sync.py:235``) returns ``False``. Each manager then
re-iterates ``loader.all_objects``, re-loads every page via
``loader.memory.load``, and re-issues the FFI ``map_memory_batch`` from
scratch. The loader-pages output of ``_extract_loader_pages``
(``rust_manager.py::_extract_loader_pages``, def at ``:510``) is a pure
function of the loader state and is identical for all 45 Callables.

**Resolution (angr-bzsc 2026-05-19, angr-b58a 2026-05-20).** Two
follow-up fixes landed against the 36.7 ms per-Callable Memory sync:

#. ``angr-bzsc`` (commit ``f7873d9de``) added a class-wide
   ``RustExplorationManager._loader_pages_cache`` (a
   ``WeakKeyDictionary`` keyed by the ``cle.Loader`` instance, weak
   so projects auto-evict on GC). ``_get_loader_pages_cache`` in
   ``rust_state_sync.py:348`` is consumed by both
   ``_map_loader_pages`` (line 375) and ``_add_loader_lazy_regions``
   (line 466). Per-Callable ``map_loader`` dropped from several ms
   to ~32 µs and ``lazy_regions`` from N_objects-iteration to ~4 µs.
#. ``angr-b58a`` (commit ``47bf6284b``) followed up by cutting
   ``_sync_extra_python_pages`` from 35 ms to 1.84 ms per Callable
   (~19×) via UltraPage memcmp + a batched ``add_lazy_regions_batch``
   FFI. ``extra_pages`` was the dominant residual phase after the
   loader-pages cache landed.

Together these brought Rust wall from ~7.23 s to ~6.26 s
(0.59× → 0.70×) and per-Callable total memory sync from ~39 ms to
~5 ms (bead ``angr-bzsc``). The
remaining ~2 s gap is spread across phases too small individually
to be worth attribution.

A 2026-06-01 spike (``angr-eyt7``) re-verified that
``_loader_pages_cache`` is wired up and consumed, and closed without
further code change: the bead description was filed 2026-05-30 from
a stale snapshot that pre-dated the angr-bzsc commit by eleven days.

**Why this was not chased earlier.** Pre-attribution, the cheap
mitigations considered (manual claripy cache clears, Rust LRU flush)
delivered ~0% on the current tree, and the gap had been left
documentation-resolved per ``angr-ed7j-doc-resolution``.

**Memory:** the original 1606MB peak was fixed independently in commit
``342df4a7f`` (2026-05-02), bringing peak to ~285MB. The current 0.59x
slowdown is purely CPU time.

**Relevant memories:** ``benchmark-mma-howtouse-cprofile-attribution``,
``i9f2-mma-howtouse-attribution`` (bead ``angr-ed7j``).

**See also:** :doc:`rust_engine_flame_comparison` for a side-by-side
cProfile reading of mma_howtouse and sym-write under both engines —
identifies Python-side page-init as the dominant cost under both
engines on callback-heavy benches and confirms the per-callback Rust
resume overhead is a real but smaller secondary cost. The full
``angr-kvn0`` characterization suite — CoW fork scaling, fleet-level
resource profile, plus the flame comparison above — is summarized in
:doc:`rust_engine_characterization`.

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
memory ``sokohashv2-two-bugs`` (bead ``angr-9l1y``).

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
``tests/benchmarks/baseline_timings.json`` reflects the longer-tail mean.

**Relevant memories:** ``benchmark-bimodal-variance-rules``,
``sokohashv2-two-bugs``.

Other benchmarks below 1.0x
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are tracked in ``tests/benchmarks/baseline_timings.json`` but were
not the focus of angr-ed7j. Captured here for completeness so future drift can
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
     - 0.69x → ~0.45x median (Py ~10.3s / Rust ~22.5s post-fix, bimodal)
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
       (within stdev).
       **AST-structure spike (angr-rbnk, 2026-06-02):** dumped Z3
       assertion stacks from both engines via
       ``tools/dump_hackcon_smtlib.py``. With ``flag`` as a 160-bit
       BVS and 58 final assertions, pre-fix Rust dumped
       ``total_chars=740,257`` with **19,649** extract operations vs.
       Python's ``total_chars=431,752`` with **7,158** extracts
       (2.7× more extracts, 1.7× larger). Root cause: the
       ``BVOp::SignExt`` to-Z3 emission in
       ``native/angr/src/symbolic/value.rs`` was a hand-rolled
       ``concat(sign_bit, …, concat(sign_bit, inner))`` loop where
       ``sign_bit = inner.extract(w-1, w-1)``. With ``inner =
       extract(159, 152, flag)``, this emitted nested
       ``(extract 7 7 (extract 159 152 flag))`` instead of Z3's
       native ``(_ sign_extend N)`` term. Switching emission to
       ``operands[0].to_z3_ast_cached(cache).sign_ext(*bits)``
       collapsed Rust's assertion stack to ``total_chars=54,221``
       (13.7× smaller; 449 extracts, 0 concats — **8× smaller than
       Python's**). End-to-end median: ~30s → ~22.5s over an 8-run
       campaign (sorted: 8.97, 18.66, 22.39, 22.68, 26.24, 28.08,
       29.49, 34.88). The fast tail at ~9s now matches Python's
       solve time; the slow tail at ~35s indicates Z3 SAT-search
       nondeterminism dominates the residual gap (category c per
       :doc:`rust_bimodal_variance`). The simpler AST gives Z3 more
       branch-choice freedom, widening the spread. Python time
       unchanged (~10.3s).
   * - ``securityfest_fairlight``
     - 0.73x (slow mode)
     - Bimodal; 2026-05-13 campaign measured ~7.95s OR ~21.4s
       (15× slow, 5× fast over 20 runs). Baseline raised from 16.0s
       to 22.0s to cover slow mode. See
       :doc:`rust_bimodal_variance` and bd memory
       ``benchmark-bimodal-variance-rules``.
   * - ``flareon2015_2``
     - **Resolved by drift → ~1.07x (Rust faster).** Was 0.75x.
     - **angr-kvn0.1, 2026-06-15.** The 5.35s baseline rust_time was
       stale. A fresh 5-sample re-measure (``run_single.py --both``) on
       HEAD ``1512b9cb9`` found Rust median **3.94s** (3.87–4.16s) vs
       Python **4.21s** (4.17–4.22s) — Rust is now *faster*, matching
       the 2026-05-20 qfbv table's 3.51s default-mode figure (this
       bench had merely never had its baseline refreshed since the
       post-May perf gains). Counters are FFI-light
       (``ffi_crossings=1``, ``callback_count=0``, ``python_callback``
       total ≈ 4ms), so the runtime is dominated by VEX interpretation
       + Z3, not boundary tax. ``rust_time`` corrected to 3.94 in
       ``baseline_timings.json``; no longer below 1.0x.
   * - ``cmu_binary_bomb_partial``
     - **Resolved by drift → ~1.06x (Rust faster).** Was 0.84x.
     - **angr-kvn0.1, 2026-06-15.** Stale baseline. Fresh 5-sample
       re-measure: Rust median **1.31s** (1.29–1.32s) vs Python
       **1.39s**. ``rust_time`` corrected from 1.65 to 1.31; no longer
       below 1.0x.
   * - ``unmapped_analysis``
     - **Resolved by drift → ~1.06x (Rust faster).** Was 0.90x.
     - **angr-kvn0.1, 2026-06-15.** Stale baseline. Fresh 5-sample
       re-measure: Rust median **0.91s** (0.89–0.92s) vs Python
       **0.97s**. ``rust_time`` corrected from 1.07 to 0.91; no longer
       below 1.0x.
   * - ``android_arm_license_validation``
     - 0.80x (genuine, but init-tax dominated — ~50ms absolute)
     - **angr-kvn0.1, 2026-06-15.** The only one of the four that
       remains sub-1.0x. Fresh 5-sample re-measure: Rust **0.25s**
       (0.24–0.25s) vs Python **0.20s**. The ~50ms gap is one-time
       setup, not engine throughput: the Rust profiling report shows
       ``Python init`` ~1.3ms + ``Add Rust state`` ~7.8ms, and the
       per-block ``Total time`` rows sum to only a few ms — the bench
       barely steps (``steps=0`` in the baseline). This is the same
       PyO3-init/state-export tax that makes the tiny ``*_branch``
       synthetic benches show <1.0x; it is fixed-cost and amortizes
       away on any non-trivial workload. Not actionable; documented
       for completeness.

PyO3 API trust model
--------------------

The Rust engine assumes a **cooperative-Python** trust model: the
Python code that drives ``RustExplorationManager`` /
``RustSolverContext`` / ``PyRustSimState`` is treated as in-process,
non-adversarial collaborator code. The FFI surface validates inputs
enough to keep well-behaved Python from corrupting Rust state, but it
does **not** sandbox a hostile or buggy caller out of all undefined
behavior. The narrow exception is the ``import_z3_constraint_ptrs``
fast path documented below.

Spike report (``angr-9l9j``, 2026-06-01). Audit scope: every
``#[pyclass]`` / ``#[pymethods]`` / ``#[pyfunction]`` in
``native/angr/src/``, focused on the four classes most likely to be
driven directly from user Python: ``RustExplorationManager``,
``RustSimState``, ``RustSolverContext``, ``RustBVHandle``,
``PythonCallbacks``. Not in scope: fuzzer/icicle/automaton FFI (only
exercised by their own integration tests).

State ID handling
~~~~~~~~~~~~~~~~~

Every public method that accepts a ``state_id: u64`` routes through
``RustExplorationManager::find_state`` /
``find_state_mut`` (``exploration/helpers.rs:122``). Misses become one
of two surfaces depending on the call site:

* **Optional reads** — ``get_state_pc_by_id``,
  ``get_state_bbl_history_tail``, ``state_constraint_count``,
  ``state_stash``, ``get_state_root`` return ``Option<…>`` so a stale
  ID maps to Python ``None``.
* **Mutating or solver-bearing reads** —
  ``set_state_solver_timeout``, ``set_state_mmap_base``,
  ``import_symbolic_to_state``, ``add_constraints_to_state``, and the
  ``with_state`` / ``with_state_mut`` helpers raise
  ``PyValueError("<api>: state {id} not found")``.

No public ``state_id`` path panics. Negative ``state_id`` values from
Python collide with the ``u64`` extraction (PyO3 raises
``OverflowError`` at the FFI boundary) and never reach Rust.

``move_state`` and ``move_states`` return ``Ok(false)`` on miss
instead of raising — a silent no-op rather than an error. Callers
that need an assertion should check the return value
(``rust_manager.py`` does).

Callback registration
~~~~~~~~~~~~~~~~~~~~~

``PythonCallbacks`` (``callbacks.rs:398``) holds 24 ``Option<Py<PyAny>>``
slots, one per dispatch site (memory load/store, hooks, syscall,
lift_block, dirty_call, page fetch, six ``state.inspect`` slots,
etc.). ``Py<PyAny>`` is an owning Python refcount, so a callback
cannot dangle even if Python "drops" it locally — the engine's clone
keeps it alive.

* **Errors propagate**. Every dispatch site reads
  ``self.<slot>.as_ref().ok_or_else(|| PyRuntimeError::new_err("…
  callback not set"))?`` and then ``cb.call1(py, args)?``. A Python
  ``raise`` becomes a ``PyErr`` that the engine returns up through
  the exploration loop (``run_loop.rs``).
* **Type/arity mismatches propagate**. Result destructuring uses
  ``.cast_bound::<PyTuple>(py)?.get_item(N)?.extract()?`` — a 1-tuple
  returned for a 3-tuple slot raises ``IndexError``; a wrong-type
  field raises ``TypeError``. The engine surfaces these as
  ``PyErr`` rather than misinterpreting bytes.
* **GC cycle break**. ``PythonCallbacks::__traverse__`` /
  ``__clear__`` (``callbacks.rs:1184``) walk every ``Py<PyAny>`` slot
  so Python GC can collect the ``mgr → _callbacks → bound-method →
  mgr`` cycle. Without this, ``RustExplorationManager`` (and its
  ``_state_cache`` of ~4030 angr pages on ``mma_howtouse``) would
  leak permanently.

Address arguments
~~~~~~~~~~~~~~~~~

Guest-VM addresses cross the FFI as ``u64``. They reach
``state.memory_load(addr, size)`` /
``state.memory_mut().import_symbolic_value(addr, bv, …)`` etc., which
delegate to ``SymbolicMemory`` — a page-table abstraction over the
guest's address space, **not** host-process memory. Unmapped pages
return ``Err`` which the FFI maps to ``Ok(None)``. There is no raw
pointer deref reachable via an address argument.

Equally, ``addr`` values used as map keys (``set_state_symbolic_pages``,
``addr_to_ast``, ``hook_symbolic_memory``) are stored verbatim — a
malformed address just lives in the map and is ignored on lookup.

BV inputs (``RustBVHandle``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RustBVHandle`` (``symbolic/handle.rs:28``) is a pure value type:
``id: u64``, ``width: u32``, ``concrete: Option<u128>``. It carries
**no raw Rust pointer** — Python only gets back a numeric handle.
The class is ``#[derive(Clone)]``; ``__eq__`` and ``__hash__`` are
``id``-based, so a forged handle compares equal to a real one with
the same ID. The symbol-table lookup that resolves the handle to a
``RustBV`` (``RustSymbolTable``) returns ``Option<…>`` on miss — no
UB risk from forged or stale IDs.

Claripy AST inputs (``add_constraint_ast``, ``eval``,
``import_symbolic_memory``, etc.) flow through ``claripy_to_rustbv``
(``claripy_bridge.rs``). Any leaf type the bridge doesn't recognize
raises ``BridgeError`` → ``PyRuntimeError``. No malformed-AST input
reaches the Z3 layer un-converted.

Raw Z3 pointer fast path (the **only** unsafe FFI surface)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The audit found one place where a Python caller's input is treated as
a raw pointer with no validation:

* ``RustExplorationManager.import_z3_constraint_ptrs(state_id, ptrs:
  Vec<usize>)`` (``exploration/state_api.rs:100``) iterates ``ptrs``
  and calls ``unsafe { ctx.add_constraint_raw(*ptr) }`` whenever
  ``*ptr != 0``. The ``add_constraint_raw`` SAFETY contract
  (``symbolic/context.rs:1784``) requires each pointer to be a valid,
  live ``Z3_ast Bool`` in the active thread-local Z3 context. The
  function calls ``NonNull::new_unchecked(ptr as *mut _)`` followed by
  ``z3::ast::Ast::wrap`` — both UB if the pointer is anything other
  than a valid Z3 AST.

  The intended caller (``rust_manager.py::_add_rust_state`` at
  ``:3576``, def at ``:3452``) always pairs
  ``import_z3_constraint_ptrs`` with a prior
  ``export_z3_constraint_ptrs(old_state_id)`` on the same process and
  Z3 context, so the production trust model is sound. But the method
  is ``pub fn`` on a ``#[pymethods]`` impl, meaning any Python caller
  can pass arbitrary integers and trigger UB. ``add_constraint_raw``
  on ``RustSolverContext`` is **not** Python-callable directly — it
  is only reachable through ``add_constraint_ast`` /
  ``add_constraint_tracked_ast``, which extract the pointer from
  claripy's z3 backend (``solver.rs:38``) and so are safe by
  construction.

  **Hardening landed (angr-33t9, 2026-06-01).**
  ``_import_z3_constraint_ptrs`` now runs a first-pass validation
  loop before any ``add_constraint_raw`` call:

  * Each pointer is converted via ``NonNull::new`` (rejecting null
    with ``PyValueError("null pointer at index {i}")``).
  * ``Z3_get_sort(raw_ctx, raw_ast)`` is called; the Option-wrapped
    return is ``None`` for ASTs that do not belong to the active
    thread-local context, surfacing as ``PyValueError("not a valid
    Z3 AST in the active context")``.
  * ``Z3_get_sort_kind`` must return ``SortKind::Bool``; anything
    else (most commonly a BV ptr exported by mistake) raises
    ``PyValueError("sort kind {kind:?}, expected Bool")``.

  Validation runs over the full list before any constraint is added,
  so a failure leaves the state's solver unmutated (verified by
  ``test_null_after_valid_ptr_rejected_atomically``).

  **Residual UB.** Truly arbitrary integers (e.g. ``0xdeadbeef``)
  still dereference inside Z3 and can segfault before the API call
  returns. Closing that gap would require either an opaque
  ``Py<RustZ3AstHandle>`` newtype (constructible only from
  ``export_z3_constraint_ptrs``) or a thread-local side-table of
  blessed pointers. Both are tracked as longer-horizon options; the
  current sanity check covers the realistic misuse cases.

Stash name validation (footgun, not unsafe)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``StashManager::ensure_stash`` (``native/angr/src/stash.rs``) wraps
the destination-stash insertion path used by ``_create_state``,
``_add_state``, ``_merge_states``, ``_move_state``, and
``_move_states``. When the destination stash does not yet exist *and*
the name is not one of the seven standard stashes
(``active`` / ``found`` / ``avoid`` / ``deadended`` / ``errored`` /
``pruned`` / ``unconstrained``), a ``log::warn!`` line is emitted
listing the typo and the standard set:

.. code-block:: text

   [rust:WARN] angr::stash: Rust exploration: creating new stash
   'actve' (not one of the standard stashes [...]); if this is a
   typo, the state will be invisible to mgr.active / mgr.found /
   mgr.deadended etc.

The warn fires exactly once per fresh name — subsequent calls into
the now-existing stash are silent, so user techniques with their own
custom stash names (e.g. ``"timeout"`` in ``rust_techniques.py``,
``"_merge_drop"`` in the merge fallback path) pay at most one warn
per process. The default log level is ``off`` so the warn only
surfaces when the user opts in with
``set_rust_log_level("warn")`` (or higher) or ``ANGR_RUST_LOG=warn``.
The behavior is otherwise unchanged — typos are surfaced, not
rejected. Tested by ``TestStashNameValidation`` in
``tests/engines/rust/``.

PyRustSimState lifetime
~~~~~~~~~~~~~~~~~~~~~~~

``add_state(&mut self, stash, state: &PyRustSimState)``
(``exploration/state_lifecycle.rs:71``) forks the incoming state via
``state.inner().fork()``, so the engine ends up owning an isolated
copy. Subsequent Python-side mutations of the original
``PyRustSimState`` do not affect the engine's state. The ``state``
parameter is borrowed (``&PyRustSimState``) so PyO3 enforces no
re-entry while Rust holds the borrow.

Summary
~~~~~~~

For real-world angr scripts, the trust model holds: state IDs,
addresses, BV handles, callbacks, and claripy ASTs all surface
type/lookup errors as ``PyErr`` rather than panicking or UB-ing.
The ``import_z3_constraint_ptrs`` hardening landed under angr-33t9;
the stash-name footgun landed under angr-630x. The trust-model audit
itself was closed under angr-9l9j with no remaining production-blocking
gaps.

.. _rust-engine-state-serialization:

State serialization (save / restore to disk)
--------------------------------------------

Public API (angr-9o4n, v1.0). The Rust engine ships a built-in
snapshot codec that round-trips the full stash manager (every state
in every stash, plus lineage and terminal counters) through a
versioned byte envelope. Three Python entry points:

.. code-block:: python

   from angr.exploration import RustExplorationManager

   # Capture a snapshot mid-exploration.
   mgr = RustExplorationManager(project, [project.factory.entry_state()])
   mgr.step(n=100)
   mgr.dump_snapshot("/tmp/run.snap")

   # Resume in the SAME manager (replaces every stash).
   mgr.load_snapshot("/tmp/run.snap")

   # Resume in a FRESH manager (no placeholder state needed).
   resumed = RustExplorationManager.load_from_disk("/tmp/run.snap", project)
   resumed.explore(find=0x4006ed, num_find=1)

On-disk envelope is ``[STASH_SNAPSHOT_VERSION: u8] ++ serde_json(...)``.
A stale version byte fails the load fast with ``ValueError``
(``"version mismatch ..."``); an empty file fails with
``ValueError`` (``"empty snapshot envelope"``). The
``STASH_SNAPSHOT_VERSION`` constant lives at
``native/angr/src/stash.rs`` and bumps on any breaking shape change
to the per-state codec (``RustSimStateSnapshot`` at
``native/angr/src/state.rs``).

What is captured:

* Every state in every stash (active, found, deadended, avoid,
  errored, unconstrained, plus user stashes).
* Per-state: ``pc``, ``state_id``, ``parent_id``, registers (concrete
  bytes + symbolic overlay), memory (concrete pages + symbolic
  overlays), call_stack, history, posix (fd table), heap metadata,
  inspection counters, and the ``SymContext::assumed_constraints``
  log. The full Z3 solver is also captured as SMT-LIB2 so
  ``state_constraint_count`` round-trips (angr-82g6).
* Lineage: ``state_roots`` map.
* Terminal counters: avoided / pruned / deadended / errored /
  unconstrained tallies.

What is NOT captured:

* **Manager-level Python configuration** — find/avoid addresses,
  hooks, simprocedures, inspection breakpoints, exploration
  techniques, solver/memory configuration. ``load_from_disk``
  accepts ``**kwargs`` so you can re-pass them to the constructor;
  predicates and hooks must be re-registered explicitly on the
  resumed manager.
* **Bucket-D ``Py<PyAny>`` overlays** —
  ``symbolic_pages`` / ``hook_symbolic_memory`` / ``addr_to_ast``
  are restored empty. For Rust-native SimProcedure + native memory
  workloads these overlays stay empty by construction, so the gap
  doesn't affect a typical bench. Workloads that rely on Python
  hook-injected symbolic memory will lose those overlays across
  the round-trip.
* **Binary identity** — the snapshot does not store a binary path
  / sha256. Loading against a mismatched ``angr.Project`` is the
  caller's responsibility (a binary-hash field is a planned
  ``angr-x04s.2`` follow-up).

Known limitation — **model equality is NOT guaranteed across a
restore**. The snapshot preserves the constraint set, but Z3's
solving heuristics are nondeterministic (see ``BIMODAL_BENCHMARKS``
and :ref:`rust-bimodal-variance`), so a ``state.posix.dumps(0)``
captured before and after a save/load round-trip may differ in the
concrete bytes the solver picks. The ``deterministic=True``
constructor flag narrows but does not close this gap.

Round-trip test: ``tests/engines/rust/test_misc.py::
test_dump_load_fauxware_round_trip_preserves_stash_shape`` exercises
the structural contract on fauxware; the
``test_load_from_disk_*`` cases exercise the v1.0 classmethod.

State snapshot / serialization (spike report)
---------------------------------------------

Spike report (``angr-x04s``, 2026-06-01). Question: what would be
needed to serialize a ``RustSimState`` (and the surrounding
``StashManager``) to disk and reload it for replay debugging? Today
the only way to reproduce a problematic exploration is to re-run from
binary entry — slow, and (for nondeterministic Z3 paths, see
``BIMODAL_BENCHMARKS``) not always reproducible at all.

What ``RustSimState`` owns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Defined at ``native/angr/src/state.rs:948``. Roughly four buckets:

* **Concrete, plain-data fields** — ``pc``, ``state_id``, ``parent_id``,
  ``history`` (``Vec<u64>``), ``detailed_history`` (``Vec<HistoryEntry>``),
  ``max_history``, ``heap_brk``, ``posix_brk``, ``mmap_base``,
  ``stdin_symbols``, ``call_stack``, ``heap_metadata``,
  ``no_ip_concretization``, ``no_symbolic_jump_resolution``,
  ``keep_ip_symbolic``, ``vex_arch``. All trivially ``Serialize``-friendly
  with ``serde`` and small (kilobytes).

* **Concrete-with-overlay fields** — ``RegisterFile`` (``Vec<u8>``
  concrete bytes plus ``FxHashMap<u32, RustBV>`` symbolic overlays);
  ``SymbolicMemory`` (``OrdMap<u64, MemoryPage>`` of CoW pages, plus
  ``FxHashMap<Address, RustBV>`` symbolic-object sidetable, plus
  ``multi_objects`` and ``lazy_regions``). The concrete halves are
  bytes — easy. The symbolic halves are ``RustBV`` trees, see below.

* **Reference-typed fields** — ``arch: Box<dyn Arch>`` (rebuild from
  ``arch_name``); ``hooks: Arc<HashSet<u64>>``,
  ``environment: Arc<HashMap<Vec<u8>, Vec<u8>>>`` (collapse to owned
  copies on snapshot, re-share on load); ``concretizer`` (plain
  config); ``inspection`` (counts + bitmask, already exported via
  ``ExplorationStateSnapshot``); ``fs: FileSystem`` (FD table — paths,
  positions, flags; concrete content is bytes).

* **Cross-FFI handles** — ``symbolic_pages``, ``hook_symbolic_memory``,
  and ``addr_to_ast`` all map addresses to ``Py<PyAny>`` claripy ASTs.
  These cannot cross a process boundary without going through Python
  on both ends. For replay debugging, the practical answer is to
  serialize the claripy AST via ``state.solver._claripy.dumps(ast)``
  (pickle wrapping the AST tree) and re-load on the target side.
  For cross-process replay where Python is not available, these
  fields would need to be dropped or replaced with native ``RustBV``
  trees first.

The single nontrivial omission is ``solver: Rc<RefCell<SymContext>>``,
covered below.

The ``RustBV`` tree
~~~~~~~~~~~~~~~~~~~

Each symbolic register / memory / object is a ``RustBV`` enum at
``native/angr/src/symbolic/value.rs:415``:

* ``Concrete { value, width }`` — ``u128`` + ``u32``, no Z3 ref.
* ``Symbolic { id, width, name, ast }`` — Z3 AST cached in the
  variant; ``name`` and ``width`` are sufficient to **reconstruct**
  the AST in a fresh Z3 context via ``BV::new_const(name, width)``.
* ``Constrained { id, value, width }`` — pinned symbolic; same as
  Concrete plus the symbolic id, no Z3 ref.
* ``Expression { id, width, op, operands: Arc<[RustBV]> }`` — op
  tree, recursively reconstructable.

The tree is therefore **fully serializable without touching Z3 at
all**: snapshot the enum, drop the cached ``ast`` field on the
``Symbolic`` variant, and rebuild the Z3 AST lazily on first access
after load. ``RustBV`` already supports cached ``to_z3_ast()`` for
the lazy path. No custom Z3 traversal is required.

Z3 AST and constraint store
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the only piece that needs Z3-specific handling, and it is
already characterized.

``SymContext`` stores constraints as a ``Vec<RustBV>`` (the
``assumed_constraints`` field, ``angr-v5a5`` lineage rework
notwithstanding — the shared-lineage variant was rejected). Two
viable export paths:

a. **Reuse the ``RustBV`` op tree** (recommended). Serialize each
   constraint as a ``RustBV`` (same scheme as the registers /
   memory) and rebuild Z3 ASTs lazily on load. No SMT-LIB round-trip
   required. This is the natural shape of the in-memory constraint
   store.

b. **SMT-LIB2 dump via ``Solver::to_string``**. Already validated
   round-trippable both same-context and cross-context (angr-9o4n.1,
   angr-rwzi; tests at ``native/angr/src/symbolic/context.rs::tests``).
   Cost: ``~5 ms`` parser-warmup floor regardless of size, ``~65
   bytes/assertion`` linear scaling, named constants re-bind by name.
   Useful as a debugging dump or for replay into a different solver,
   but redundant for a Rust→Rust snapshot of an in-process state.

Lineage today is per-context (``SymContext::scope_path``,
``assumed_constraints``). After the ``angr-v5a5`` shared-lineage
rejection, every state's constraints are owned outright by its own
``SymContext`` — there is no cross-state structural sharing to
preserve. A snapshot of one state's constraint list is
self-contained.

What lives OUTSIDE ``RustSimState``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a "load and continue exploration" experience (vs. "load and
inspect a single state"), the snapshot would also need:

* ``StashManager`` (``native/angr/src/stash.rs:51``) — straightforward:
  ``HashMap<String, VecDeque<RustSimState>>`` plus a few counters and
  the state_index ``HashMap<u64, String>`` mirror. Reuse the same
  per-state codec.
* ``RustExplorationManager`` Python wrapper — find/avoid callbacks,
  inspection breakpoints, simoption mirrors. These are
  user-supplied closures; a useful snapshot would record their
  *intent* (find address list, avoid address list) and rely on the
  user to re-register Python callbacks on load.
* ``CallbackInterpreter`` / ``PendingCallback`` queue — best dropped
  on snapshot. Snapshot only at quiescent points (between
  ``run_until`` returns), where there are no pending callbacks.
* ``angr.Project`` (Python) — out of scope; the snapshot user supplies
  a matching Project on load. Snapshot records binary path + hash
  so the loader can detect mismatch.

Proposed on-disk format
~~~~~~~~~~~~~~~~~~~~~~~

Minimum-viable shape (one file per snapshot)::

  {
    "format_version": 1,
    "angr_engine_version": "<from cargo metadata>",
    "binary_path": "/abs/path/to/binary",
    "binary_sha256": "...",
    "arch_name": "amd64",
    "stashes": {
      "active":     [ <state>, ... ],
      "found":      [ ... ],
      "deadended":  [ ... ],
      ...
    },
    "manager_metadata": {
      "find_addrs": [ ... ],
      "avoid_addrs": [ ... ],
      "inspection_breakpoints": [ ... ],
      "step_counter": ...
    }
  }

where each ``<state>`` carries::

  {
    "state_id": ...,
    "parent_id": ...,
    "pc": ...,
    "registers": {
      "concrete_bytes": "<base64>",
      "symbolic_overlays": { "<offset>": <rust_bv>, ... }
    },
    "memory": {
      "pages": [
        { "page_num": ..., "permissions": ...,
          "data_base64": "...",
          "symbolic_bitmap_base64": "...",
          "symbolic_objects": { "<addr>": <rust_bv>, ... } }
      ],
      "lazy_regions": [[start, end], ...],
      "pending_writes": [ ... ]
    },
    "constraints": [ <rust_bv_bool>, ... ],
    "history": [ ... ],
    "call_stack": [ ... ],
    "heap": { "brk": ..., "metadata": { ... } },
    "fs": { ... },
    "claripy_pyobjs": {
      "symbolic_pages":      { "<addr>": "<pickle_b64>" },
      "hook_symbolic_memory":{ "<addr>": ["<pickle_b64>", <size>] },
      "addr_to_ast":         { "<addr>": ["<pickle_b64>", <size>] }
    },
    "flags": { "no_ip_concretization": ..., "keep_ip_symbolic": ... }
  }

The outer envelope is JSON for human inspection during debugging;
the inner ``<rust_bv>`` substructure could be bincode + base64 if
serialization speed matters (op trees with concrete-byte
``Operands`` arrays compress well). ``format_version`` is bumped
whenever any of the ``RustBV`` enum, ``MemoryPage`` layout, or
``RustSimState`` field set changes — same discipline as the disk
init cache's ``_RUST_CACHE_VERSION`` (``angr/exploration/
rust_manager.py::_RUST_CACHE_VERSION``, currently at ``:176``).

Cost estimates
~~~~~~~~~~~~~~

* **State count.** A mid-bench exploration carries 10–100 active
  states; a hard bench (e.g. ``hackcon2016_angry-reverser``) can
  push to thousands. Snapshot all states or sample a small
  representative set — the format does not distinguish.
* **Per-state size.** Empirically (from the existing
  ``ExplorationStateSnapshot`` export) registers are ``~1 KB``,
  mapped memory is ``~10–100`` pages × ``4 KB`` (concrete) plus a
  small symbolic-object table, constraints are tens to low
  thousands of nodes. Order-of-magnitude: ``~100 KB`` per state
  uncompressed for typical fast-tier benches.
* **Z3 rebuild cost.** Constraint reconstruction is one
  ``RustBV::to_z3_ast()`` traversal per constraint; same shape as a
  lineage materialize. Lineage-materialize fork cost for the
  reference benches is in the single-digit milliseconds for state
  counts < ~50 — well within "useful for debugging" budget.
* **SMT-LIB2 floor.** If route (b) is chosen instead, expect a fixed
  ``~5 ms`` ``Solver::from_string`` cost per state. For thousand-state
  snapshots, this dominates over (a).

Recommendation and follow-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Route (a) — ``RustBV`` op-tree serialization — is the right
  bet.** It re-uses primitives the engine already exercises (deep
  ``RustBV`` clone, lazy ``to_z3_ast``), avoids the SMT-LIB
  parser-warmup floor for in-process replay, and keeps the
  serialized form readable enough to diff between two snapshots.

* **Format-version discipline matters.** A single bump axis
  (``format_version``) is sufficient at the snapshot envelope.
  Internal RustBV / MemoryPage layout changes do not need a finer
  axis because the on-disk format is opaque-blob-per-bv anyway —
  any layout change forces a new ``format_version`` rev.

* **Concrete next step is bounded but not small.** A working
  prototype needs: ``serde`` derives on ``RustBV`` /
  ``MemoryPage`` / ``RegisterFile`` / ``SymContext::constraints``,
  a ``RustSimState::to_serialized() / from_serialized()`` pair,
  a ``StashManager::dump()/load()`` pair, and a Python wrapper for
  the ``.pyobj`` parts (``state.posix`` plugins, claripy AST pickles).
  Estimate: medium-size implementation task, **not** a one-iter
  session.

* **The cost/benefit is favorable for a P3 implementation task.**
  Replay debugging would unlock bisecting nondeterministic
  bimodal benches (currently impossible — each rerun rolls the Z3
  dice fresh), and would shorten the loop on path-explosion bugs
  (currently re-run from entry, ~30s+ on hard benches). Filed as
  follow-up bead (see below).

Follow-up bead filed: ``angr-x04s.1`` (task: prototype
``RustBV``-op-tree state snapshot for one fast-tier bench, gated
behind an opt-in Python flag, no production code path touches it
by default).

.. _rust-engine-api-stability:

API stability contract
----------------------

The Rust engine's public API consists of the PyO3 ``#[pyclass]`` types
re-exported from ``angr.exploration`` (and forwarded to the top-level
``angr`` package per ``angr-nncz``) plus the typed exception hierarchy
in ``native/angr/src/errors.rs``.

Public vs experimental vs private
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three tiers of stability apply to every name reachable from
``angr.exploration``:

- **Public.** No leading underscore, listed in
  ``angr/exploration/_public_api.py`` (``MODULE_EXPORTS``,
  ``TYPED_EXCEPTIONS``, or as a key in ``CLASS_PUBLIC_ATTRS``), and
  not marked experimental. Public names follow the semver contract
  below.
- **Experimental.** A no-leading-underscore name explicitly tagged
  ``Experimental:`` in its docstring or RST description (see the
  ``ANGR_Z3_TACTIC`` example earlier in this document). Experimental
  surfaces may change or be removed in any minor release without
  going through the deprecation cycle. They exist so risky/in-flight
  work can be reached from Python without committing to forever.
- **Private.** Leading underscore (``_foo``, ``_BAR``) or any name
  not in ``_public_api.py``. Internal implementation detail — may
  change or vanish in any release without notice. Do not import,
  patch, or rely on private names from downstream code.

Snapshot test ``angr-9cps .3`` (planned) consumes
``angr/exploration/_public_api.py`` as the canonical list of public
names; any unintended addition, rename, or removal trips CI before it
ships.

Reading the engine version
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust engine carries its own version, distinct from
``angr.__version__``. The engine API evolves on a different cadence
than core angr (CTF-style speedup work, write-through proxy epic, Z3
solver tuning), so consumers that gate on engine features should read
the engine version directly:

.. code-block:: python

    from angr.exploration import __rust_engine_version__

    print(__rust_engine_version__)  # e.g. '0.1.0'

The value comes from ``CARGO_PKG_VERSION`` at compile time
(``native/angr/Cargo.toml`` is the source of truth) and is re-exported
from ``angr.rustylib.vex_engine.__version__``. Same string, two
import paths — prefer ``angr.exploration.__rust_engine_version__``
since it does not require knowing the rustylib submodule name.

The attribute is intentionally **not** in ``angr.exploration.__all__``
(it follows the standard ``__version__`` convention — accessible via
explicit name, excluded from star-import). The semver rules in the
next section apply to the value of this attribute, not to
``angr.__version__``.

Semver rules (within 9.2.x and onwards)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following semver contract applies once the engine reaches a
tagged release. It governs the **public** tier only — experimental
and private names are exempt.

**Patch versions (X.Y.Z → X.Y.Z+1)**
  - No public API changes — bug fixes only.
  - No new exception classes, no new pyclass methods/fields, no new
    error-enum variants, no new event-enum variants.

**Minor versions (X.Y.Z → X.Y+1.0)**
  - May add new exception subclasses under ``RustExecutionError``.
  - May add new ``#[pyo3(get)]`` fields to event/config pyclass
    structs (e.g. new ``ExplorationEvent`` callback fields).
  - May add new variants to internal error/event enums on the Rust
    side. Public Rust types involved in this surface are marked
    ``#[non_exhaustive]`` so internal pattern matches require a
    wildcard arm and downstream Rust consumers (currently none) can
    add the new variant without it being a breaking change.
  - May add new pyclass methods. Existing method signatures stay
    stable.
  - May mark a previously-public name as deprecated. The name keeps
    working for the rest of the current minor series and is removed
    no earlier than the **next** minor release (see deprecation
    cycle below).

**Major versions (X.Y.Z → X+1.0.0)**
  - May rename or remove exception classes, pyclass methods/fields,
    or error variants. May restructure the ``RustExecutionError``
    hierarchy.
  - May remove any name that has been through one full minor's worth
    of deprecation warning.

Breaking-change rule of thumb: if a change would break code written
against ``_public_api.py``, it requires a major version bump *and*
at least one minor release where the old name still works while
emitting ``DeprecationWarning``.

Deprecation cycle
~~~~~~~~~~~~~~~~~

Removing a public name is a two-step process:

1. **Deprecate** in minor release ``X.Y.0``. The name continues to
   work; access emits ``DeprecationWarning`` (Python side) or a
   ``#[deprecated]`` attribute (Rust side, when applicable). The
   ``CLASS_PUBLIC_ATTRS`` tuple in ``_public_api.py`` keeps the
   name with a trailing ``# deprecated in X.Y, removal targeted
   X.(Y+1)`` comment.
2. **Remove** no earlier than ``X.(Y+1).0``. The name vanishes from
   ``_public_api.py``; the snapshot test enforces the removal.

One minor cycle is the floor, not the ceiling — popular names should
get more. The cycle exists so downstream code has a release with both
the new and old name working, where ``DeprecationWarning`` shows up
in test output and surfaces the rename.

Applying the ``@_deprecated`` decorator (Python side)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``angr.exploration._deprecation._deprecated`` decorator is the canonical
way to mark a Python-side public name as deprecated during step 1 of the
cycle above. It is intentionally underscore-prefixed (private) — only the
exploration package itself should import it.

Usage:

.. code-block:: python

    from angr.exploration._deprecation import _deprecated

    class RustExplorationManager:
        @_deprecated(version="9.3", removed_in="9.4", replacement="run")
        def explore(self, *args, **kwargs):
            return self.run(*args, **kwargs)

Arguments:

- ``version`` (required) — the minor release in which deprecation started.
  Matches ``X.Y`` in the cycle's step 1.
- ``removed_in`` (required) — the minor release targeted for removal.
  Matches ``X.(Y+1)`` in the cycle's step 2.
- ``replacement`` (optional) — the new name a caller should migrate to;
  appears verbatim in the warning message.

Behavior:

- Emits ``DeprecationWarning`` from the wrapper on first call **per
  decorated callable per process**. Subsequent calls are silent so test
  logs and long-running sessions stay legible.
- ``stacklevel=2`` so the warning points at the caller, not at the
  wrapper.
- ``functools.wraps`` preserves ``__name__``, ``__doc__``, and the
  original signature so the snapshot test in
  ``tests/engines/test_rust_public_api.py`` still sees the name on the
  class.

When the rest of step 1 (inventory comment, release-note entry) and step 2
(removal in the next minor release) happen, the test in
``tests/engines/test_rust_deprecation.py`` keeps the decorator's contract
itself stable: the warning fires, carries both versions, and includes the
replacement name when given.

If no public name is currently being deprecated, the decorator still lives
in the tree ready for the first cycle — the test verifies that the
mechanism works, independent of any concrete name using it today.

How to mark something experimental
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a new pyclass method, ``#[pyo3(get)]`` field, env-var hook, or
similar Python-visible knob ships in a state where its shape may
still change, mark it experimental:

- **Python side.** Add ``"Experimental:"`` (literal prefix) to the
  first line of the docstring, e.g.

  .. code-block:: python

      def some_new_knob(self, ...):
          """Experimental: tune the foo strategy.

          May change shape in any minor release until promoted.
          """

  Do **not** add the name to ``_public_api.py``. The snapshot test
  treats absence from ``_public_api.py`` as "not public" — exactly
  the contract we want for experimental.

- **Rust side.** In the RST description of any new ``#[pyo3(get)]``
  field or pyclass method, lead with ``Experimental:`` and link
  back to this section (``:ref:`rust-engine-api-stability```).
  Existing example: ``ANGR_Z3_TACTIC`` solver-strategy override
  earlier in this document.

Promoting an experimental name to public means: drop the
``Experimental:`` prefix, add it to the relevant tuple in
``_public_api.py``, and land both changes in the same minor release.
Promotion is a *minor*-version event (additive), so it does not
require a deprecation cycle.

The Python exception hierarchy intentionally **does not** offer an
exhaustive-match guarantee. Downstream code that uses
``pytest.raises(angr.RustExecutionError)`` (the base class) keeps
working when new sibling classes are added; code that catches a
specific subclass like ``pytest.raises(angr.RustUnsupportedVexOpError)``
is not affected by additions either. The variant added in a minor
release surfaces as a new subclass of ``RustExecutionError``.

On the Rust side, the types currently marked ``#[non_exhaustive]``
are:

- ``RustExecError`` (``native/angr/src/errors.rs``) — the canonical
  Rust→Python error enum.
- ``BridgeError`` (``native/angr/src/claripy_bridge.rs``)
- ``SyscallError`` (``native/angr/src/syscalls/mod.rs``)
- ``OpError`` (``native/angr/src/vex/ops.rs``)
- ``ProcedureError`` (``native/angr/src/procedures/mod.rs``)
- ``MemoryError`` (``native/angr/src/memory/mod.rs``)
- ``CallbackReason`` (``native/angr/src/exploration/mod.rs``)
- ``ExecutionEvent`` (``native/angr/src/state.rs``)
- ``ExplorationEvent`` pyclass struct
  (``native/angr/src/exploration/mod.rs``) — the Python-visible event
  envelope, expected to grow ``#[pyo3(get)]`` fields as new
  callback reasons land.
- ``ExecutionConfig`` pyclass struct (``native/angr/src/callbacks.rs``)
  — the Python-visible config struct, expected to grow knobs over
  time.

Adding a variant or field to these is **not** a breaking change. The
``#[non_exhaustive]`` attribute makes the rule machine-enforced rather
than aspirational. Other internal enums (``CbExecutionError``,
``LiftError``, ``DeserializeError``, ``NativeLiftError``,
``ConstraintSyncError``, ``ExtractionError``) are crate-local
implementation detail and are intentionally left without
``#[non_exhaustive]`` because their match sites are tightly controlled
within the same crate.

Spike report ``angr-irwe`` (2026-06-01) — see commit history for the
audit + initial application.

.. _rust-engine-concurrency-send-sync-audit:

Concurrency / Send-Sync audit
-----------------------------

Spike report ``angr-8fo6`` (2026-06-01). This section catalogs the
``Send`` / ``Sync`` posture of every ``#[pyclass]`` exposed to Python
and identifies the architectural work required before any future
parallel-exploration story (cf. ``angr-j7kn`` — speculative Z3 theory
propagator) can land.

Today the engine is **single-threaded by construction** — every entry
point through ``RustExplorationManager`` is gated by the GIL, and the
state-graph data structures are intentionally ``!Send`` so the compiler
prevents accidental cross-thread sharing. The PyO3 ``unsendable`` markers
on the load-bearing types are not stylistic — they encode a deep
architectural constraint rooted in z3-rs's thread-local Z3 context model
(see ``z3-rs 0.19+`` note at ``native/angr/src/solver.rs:99``).

Pyclass inventory
~~~~~~~~~~~~~~~~~

Six ``#[pyclass(unsendable)]`` types — cannot be shared across OS
threads even with a ``Mutex``, because they hold ``!Send`` interior
state:

- ``RustExplorationManager`` (``native/angr/src/exploration/mod.rs:386``)
  — coordinator; transitively owns the StashManager (which holds the
  per-state ``Rc<RefCell<SymContext>>``) plus thread-bound Z3 solver
  handles. The unsendable marker is correct.
- ``PyRustSimState`` (``native/angr/src/state.rs:2703``) — wraps
  ``RustSimState`` which holds ``solver: Rc<RefCell<SymContext>>``
  (``state.rs:960``). ``Rc`` is the binding constraint; replacing it with
  ``Arc<Mutex<SymContext>>`` is necessary but **not sufficient** (see
  Z3 constraints below).
- ``RustSolverContext`` (``native/angr/src/solver.rs:102``) — holds
  ``SolverCtxStorage`` which is either ``SymContext`` directly or
  ``Rc<RefCell<SymContext>>`` shared with a parent state. Same blocker
  as ``PyRustSimState``.
- ``Fuzzer`` (``native/angr/src/fuzzer.rs:59``) — libafl state machine;
  ``unsendable`` for libafl-internal reasons (Python callbacks held as
  closures). Out of scope for the symex engine's parallel story.
- ``PyOnDiskCorpus`` (``native/angr/src/fuzzer/corpus.rs:146``) — libafl
  on-disk handle; same scope as ``Fuzzer``.
- ``Icicle`` (``native/angr/src/icicle.rs:230``) — icicle VM owns
  thread-local JIT state; ``unsendable`` is correct.

The plain ``#[pyclass]`` types (without ``unsendable``) are all
``Send + Sync`` today because they hold only Plain Old Data or
``Py<PyAny>`` handles (which PyO3 declares ``Send + Sync`` since the
GIL controls actual dereference). No refactor needed for these:

- ``DeferredFork``, ``BranchPolicy``, ``ExecutionConfig``,
  ``PythonCallbacks``, ``LoopExecutionEvent``
  (``native/angr/src/callbacks.rs``) — value types, plus ``Py<PyAny>``
  callback handles (Send+Sync) and ``Arc<Atomic*>`` shared toggles.
- ``ExplorationEvent`` (``native/angr/src/exploration/mod.rs:124``) —
  value type built from primitives + ``Py<PyAny>``.
- ``RustBVHandle`` (``native/angr/src/symbolic/handle.rs:28``) — three
  POD fields (``id: u64``, ``width: u32``, ``concrete: Option<u128>``).
  Cheap to ship across threads but useless without the matching
  symbol-table entry, which lives inside the ``unsendable``
  ``RustSolverContext``.
- ``ExplorationStateSnapshot`` (``native/angr/src/state.rs:3228``) —
  the serializable snapshot type added by ``angr-zidj`` is already
  ``Send + Sync``. **This is the recommended cross-thread transport
  type** (see Recommendation below).
- ``Segment``, ``SegmentList``, ``SegmentListIter``
  (``native/angr/src/segmentlist.rs``) — pure value types, Send+Sync.
- ``VmExit``, ``ExceptionCode`` (``native/angr/src/icicle.rs:47,97``) —
  C-like enums.
- ``PyState``, ``PySymbol``, ``PyEpsilon``, ``PyEpsilonNFA``, ``PyDFA``
  (``native/angr/src/automaton/python_bindings.rs``) — automaton API;
  hold ``Py<PyAny>`` plus pure-Rust automaton state. Send+Sync today.
- ``PyHavocMutator``, ``PyInMemoryCorpus``, ``ClientStats``
  (``native/angr/src/fuzzer/{mutator,corpus,monitor}.rs``) — fuzzer
  value types; Send+Sync today.

The root blockers
~~~~~~~~~~~~~~~~~

Three architectural constraints must be relaxed before a worker thread
can step a state in parallel with the main thread. Each blocker stands
alone — fixing any one in isolation does not enable parallelism.

**1. ``Rc<RefCell<SymContext>>`` in ``RustSimState`` (state.rs:960).**
   Used at 7 mutation sites including all four fork variants
   (``fork``, ``fork_true``, ``fork_false``, plus the snapshot-restore
   fork). Pattern is "share parent solver on fork, fork-on-mutate" — a
   pure single-threaded CoW idiom. Cost to convert: large, touches
   every state-creation path. Replacing ``Rc`` with ``Arc`` is cheap;
   ``RefCell`` → ``Mutex`` introduces lock contention on the constraint
   hot path (``assume_*``, ``add_constraint_raw`` — 82.5 % of csaw_wyvern
   calls hit dedup, so the lock would be heavily contended).

**2. Z3 context thread-locality (z3-rs 0.19+).** The z3-rs crate uses
   ``thread_local!`` Z3 contexts: every thread has its own ``Z3_context``,
   and an AST handle (``z3::ast::Bool``, ``z3::ast::BV``) is bound to the
   context of the thread that created it. Crossing the boundary is
   undefined behavior. Consequence: a state's solver assertions cannot
   migrate threads. Worker threads must either (a) be pinned to a state
   from creation, or (b) re-translate the entire assertion stack across
   threads via SMT-LIB serialization. (b) erases any parallelism gain
   unless the per-solve cost is enormous.

**3. Claripy bridge thread-local AST caches**
   (``native/angr/src/claripy_bridge.rs:153, 200, 231, 269``). The four
   caches documented in the ``claripy-bridge-thread-local-caches``
   memory (``AST_CACHE``, ``CLARIPY_AST_CACHE``, ``EXPRESSION_CACHE``,
   ``EXPRESSION_BY_OPERANDS_PTR``) are intentionally ``thread_local!``
   to avoid lock contention on the hot path. They are **already correct
   for multi-threading** in the sense that each thread has its own
   instance and no cross-thread synchronization is needed. Cost:
   workers pay a cold-cache penalty on their first ~10 000 expressions
   each.

Secondary considerations
~~~~~~~~~~~~~~~~~~~~~~~~

- ``SymContext`` itself (``native/angr/src/symbolic/context.rs:1325``)
  mostly uses ``Mutex`` and ``Arc`` for its shared state (constraint
  stacks, push counters, lineage), but has two ``!Sync`` interior cells:
  ``sat_cache: Cell<Option<bool>>`` (line 1372) and
  ``model_cache: RefCell<Option<z3::Model>>`` (line 1375). These would
  need ``Mutex`` wrappers if ``SymContext`` ever needs ``Sync``. The
  ``z3::Model`` inside is also thread-bound, so the cell-conversion
  alone does not lift the Z3 thread-locality constraint.
- The Python-side proxies (``RustStateProxy`` and its sub-proxies in
  ``angr/exploration/rust_state_proxy.py``) are pure Python objects that
  hold a Python reference to the ``unsendable`` Rust manager. They are
  not directly subject to a Rust ``Send`` / ``Sync`` audit, but inherit
  the same single-thread constraint via the Rust handle they wrap.
- ``thread_local! STEPPING_STATE_ID`` (``exploration/mod.rs:63``) is
  fine — each worker would get its own.

Recommendation: ship snapshots, not states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The least-invasive path to parallel exploration is **not** to make
``RustExplorationManager`` ``Send``. Instead, lean on the
``ExplorationStateSnapshot`` type already produced by ``angr-zidj``:

1. Main thread runs a ``RustExplorationManager`` that emits snapshots
   for states queued onto a parallel-eval lane.
2. Worker threads receive ``ExplorationStateSnapshot`` (Send+Sync),
   each constructs its own thread-local ``RustExplorationManager``
   and ``RustSimState`` from the snapshot, drives Z3 in its own
   thread-local context, and returns either a result or a successor
   snapshot.
3. No ``Rc`` → ``Arc`` migration, no ``RefCell`` → ``Mutex`` migration,
   no Z3 AST cross-thread shipping. Workers re-build the per-thread
   solver from the constraint list embedded in the snapshot.

This is the same shape angr-x04s.1 (RustBV op-tree snapshot prototype)
is heading toward for OOM recovery. The two stories share the same
serialization primitive — investment is amortized.

Follow-up beads
~~~~~~~~~~~~~~~

Each candidate below is filed (or proposed) as a separate ``task``
bead so the parallel-exploration epic can be sequenced. No production
code changes from this spike.

- **No new beads filed today.** The audit's headline finding is that
  the visible ``unsendable`` markers are correct and reflect deep
  architectural constraints — not stylistic safety bumpers. The path
  forward is the snapshot-transport story above, which is already
  covered by ``angr-x04s.1`` (op-tree snapshot prototype). Parallel
  exploration on top of that prototype would be a new epic, not a
  follow-up to this spike.
- If parallel exploration becomes a near-term roadmap item, file:
  a) Worker-pool prototype consuming ``ExplorationStateSnapshot``
     (depends on ``angr-x04s.1``).
  b) ``sat_cache`` / ``model_cache`` ``Cell``/``RefCell`` → ``Mutex``
     migration in ``SymContext`` (cheap, isolated).
  c) Benchmark harness for snapshot round-trip cost
     (snapshot-emit + restore + first-solve), to validate that the
     per-state migration cost stays below the parallel-execution gain.

.. _rust-engine-ffi-ownership-audit:

PyO3 FFI ownership audit
------------------------

Audit pass ``angr-t1w7`` (2026-06-03). Companion to the Send/Sync audit
above. Rules out the FFI-ownership class of bugs (GIL discipline,
``Z3_inc_ref`` / ``Z3_dec_ref`` pairing, ``Py<PyAny>`` drop ordering,
thread-local cache teardown) on top of the cleanup infrastructure that
landed in ``angr-518z`` (commit ``af8afa9d3``, 2026-05-17) — see bd
memory ``angr-518z-infrastructure-landed`` for the baseline.

Headline: the audited surface is clean for the engine's documented
single-threaded use. One latent hazard is documented for future
multi-threaded clients; no production fix required today.

GIL discipline
~~~~~~~~~~~~~~

The Rust extension never implicitly acquires the GIL. Across the 21
modules that touch the FFI boundary (177 ``Python<'_>`` / ``Python<'py>``
signature occurrences), every entry point either receives the
``Python<'_>`` token from a ``#[pyfunction]`` / ``#[pymethods]``
signature or plumbs a ``Bound<'py, T>`` borrow forward.

- ``rg "with_gil|acquire_gil|allow_threads"`` against ``native/angr/src``
  returns no matches.
- ``Py<PyAny>::clone_ref(py)`` is used consistently (not the cheaper
  ``clone()`` which would still take a refcount but not document the
  GIL precondition).
- All ``.extract::<T>()`` and ``Bound::call_method*`` sites operate
  through a borrowed ``Python<'py>`` token, so there is no risk of a
  caller invoking PyO3 without holding the GIL.

This eliminates the most common PyO3 hazard class — implicit GIL
acquisition in a Rust callback that runs after the caller has dropped
the lock.

``clear_ast_cache`` call graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``clear_ast_cache()`` (declared at ``native/angr/src/claripy_bridge.rs:423``,
exposed to Python via ``native/angr/src/engine.rs::clear_ast_cache``
(def ``:303``, registered ``:379``)) atomically clears
all four thread-local caches in the bridge:

- ``AST_CACHE`` (LRU, ``claripy_bridge.rs:154``)
- ``CLARIPY_AST_CACHE`` (unbounded ``HashMap``, ``claripy_bridge.rs:201``)
- ``EXPRESSION_CACHE`` (LRU, ``claripy_bridge.rs:232``)
- ``EXPRESSION_BY_OPERANDS_PTR`` (LRU, ``claripy_bridge.rs:270``)

Cross-cache invariant C3 (documented in-file) makes a partial clear
incorrect by design; the single entry point enforces it.

``clear_all_caches()`` (``claripy_bridge.rs:432``) additionally clears
the process-global ``SymbolicIdentityRegistry``. It is **deliberately
not** exposed to Python because the registry is shared across all
managers in the process — clearing it from one manager would
invalidate live symbol IDs held by another. The only Python-side
flush path is ``RustExplorationManager.cleanup()`` (gated by the
``clear_caches_on_cleanup`` constructor flag, default ``False``), so
manager teardown never reaches the global registry.

Refcount discipline (Z3 ASTs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Z3AstPtr`` (``native/angr/src/symbolic/z3_ast_ptr.rs``) is the
canonical wrapper for raw ``Z3_ast`` pointers crossing the claripy
boundary. It takes an explicit ``z3::Context`` clone (line 35) so
``Z3_dec_ref`` in ``Drop`` always sees the context the ref was taken
under — independent of thread-local context state. The wrapper is
non-``Copy`` and non-``Clone`` to make double-free impossible at the
type level; duplication is explicit via ``clone_ref`` (which performs
a fresh ``Z3_inc_ref``).

``z3::ast::BV`` / ``z3::ast::Bool`` values held inside
``RustBV::Symbolic`` (``symbolic/value.rs:425``) and
``SymContext::z3_assertions_shared`` (``symbolic/context.rs:1357``)
bind to the **active thread-local Z3 context** at construction time.
While the manager and its states are alive on the thread that created
them, the TLS context outlives every AST it owns. Drop ordering inside
the manager is safe (manager → ``StashManager`` → ``RustSimState`` →
``SymContext`` → Z3 AST handles).

Cross-process exit is handled by ``atexit.register(reset_shared_z3_context)``
(registered in ``angr/exploration/rust_manager.py`` at module-import time,
currently at ``:372``), which swaps
the Rust TLS context to a fresh Rust-owned context **before** Python
frees its shared Z3 context. This prevents the manager's surviving
state from dec_ref'ing into a freed Python-owned context.

Thread-local cache teardown
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The four ``claripy_bridge`` thread-locals hold ``Py<PyAny>`` values and
(via ``RustBV::Symbolic``) ``z3::ast::BV`` values. ``Py<PyAny>::Drop``
without the GIL is safe — PyO3 0.21+ defers the decref to the next GIL
acquisition. The z3 AST drop, however, requires the **thread-local Z3
context** to still be live at TLS-destructor time.

**Latent hazard (not exercised today).** Rust runs ``thread_local!``
destructors in LIFO order of first access. If a non-main thread first
populates the four ``claripy_bridge`` caches (registering their
destructors), then calls into z3-rs (registering z3-rs's TLS destructor
later), at thread teardown z3-rs's TLS context drops **first** and the
cache destructors then run with a stale TLS context. ``RustBV::Symbolic``
drops in those cache values would dec_ref against the wrong context.

This is **not exercised today**: angr-symex is single-threaded by
construction (cf. Send/Sync audit) and the load-bearing pyclasses are
``#[pyclass(unsendable)]``. The main thread's TLS destructors are not
reliably run at process exit on either glibc or musl, so the
sub-hazard there is a memory leak that the kernel reclaims, not UAF.

If a future story spawns worker threads that touch ``rustylib``
(e.g., a parallel-eval lane built on ``ExplorationStateSnapshot``),
the worker should call ``angr.rustylib.vex_engine.clear_ast_cache()``
**before** ``join``-ing the thread, which empties the caches while the
TLS Z3 context is still guaranteed live by the runtime. Filed as
``angr-bjk8`` for the worker-pool epic to honor.

Static-lifetime registries
~~~~~~~~~~~~~~~~~~~~~~~~~~

``SymbolicIdentityRegistry`` lives in a ``OnceLock`` static
(``symbolic/registry.rs:310``) and holds ``Py<PyAny>`` values keyed by
Rust symbol IDs. Rust statics do not run ``Drop`` at process exit, so
the ``Py<PyAny>`` handles inside leak rather than running a non-GIL
decref. This is intentional: the registry is process-global by design
and any "cleanup" would race with concurrent live managers. The leak
is bounded by the leaf-symbol set per process, which grows slowly
(tens to low thousands).

Audit conclusion
~~~~~~~~~~~~~~~~

No production-code changes filed. The FFI ownership surface is clean
for the engine's documented single-threaded use, with one latent
hazard documented above (``angr-bjk8``) gated on a parallel-execution
story that does not exist yet.

.. note::

   *Last verified against commit* ``4215fe99b`` *on 2026-06-03*
   (angr-1cnv). The *Architecture support matrix*, *Z3 solver
   profiling counters*, and exploration-technique tables drift fastest
   when arches gain a benchmark or a technique flips Native /
   Unsupported — re-read those sections and bump this footer when you
   touch them.

