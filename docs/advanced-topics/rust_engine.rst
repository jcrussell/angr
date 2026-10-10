Rust symbolic execution engine
==============================

angr ships a Rust-native symbolic-execution engine
(``RustExplorationManager``) that runs the VEX interpreter, memory
model, and Z3 solver in Rust while still cooperating with the Python
project, claripy ASTs, and SimProcedures via PyO3. It is enabled by
passing ``use_rust_engine=True`` to ``proj.factory.simulation_manager()``
or by building the project with ``engine="rust"`` (see *Auto dispatch*
below).

This page is the reference for *using* the engine: which
``angr.sim_options`` flags it honors, how ``state.inspect`` dispatches,
which exploration techniques and analyses work, how errors surface, and
what the API-stability contract is. Companion pages:

* :doc:`rust_build` — building, testing, profiling and logging.
* :doc:`rust_engine_decisions` — alternatives that were tried and
  rejected, so they are not re-tried.
* :doc:`rust_z3_sharing`, :doc:`rust_parallel_design`,
  :doc:`rust_lazy_memory_design`, :doc:`rust_proxy_writes_design`,
  :doc:`rust_libvex_ffi`, :doc:`rust_wheel_distribution` — design notes.
* :doc:`/extending-angr/native_coverage_matrix`,
  :doc:`/extending-angr/rust_vex_ops`,
  :doc:`/extending-angr/simprocedures` — what is implemented natively
  and how to extend it.

Overview
--------

* **Honored options:** the *Honored options* table in the SimOption
  coverage matrix below is the authoritative list (``LAZY_SOLVES``,
  ``ZERO_FILL_UNCONSTRAINED_MEMORY``, ``STRICT_PAGE_ACCESS``,
  ``ENABLE_NX``, the symbolic-jump and syscall-resolution family, the
  concretization-strategy family, and a handful honored transparently
  because their only consumer runs in Python). Everything else is
  either inherited from Python or silently ignored — see the matrix.
* **Full** ``state.inspect`` **dispatch.** The Rust engine dispatches all
  ``state.inspect`` events to Python BPs: ``mem_read``, ``mem_write``,
  ``reg_read``, ``reg_write``, ``instruction``, ``irsb``, ``exit``,
  ``call``, ``return``, ``simprocedure``, ``syscall``, ``dirty``,
  ``tmp_read``, ``tmp_write``, ``statement``, ``expr``,
  ``address_concretization``, ``symbolic_variable``, ``fork``,
  ``constraints``, and ``vex_lift``. The last pair each fire from two
  origins: ``constraints`` from the Python proxy solver
  (``state.solver.add`` in a SimProcedure) *and* from the fork-guard
  constraints the engine adds natively; ``vex_lift`` from the Python lift
  callback (Rust block-cache miss → ``_cb_lift_block``) *and* from the
  feature-gated native in-process libVEX lift. On the native origin of
  each, a ``BP_BEFORE`` mutation of the staged attrs is not honored — the
  guard is already a Rust BV, and the block's bytes are already lifted.
* **Performance:** faster than Python on most benchmarks, with a small
  number of known slower cases driven by Python-side callback pressure
  or bimodal Z3 solver nondeterminism — see *Benchmarks slower than
  Python* below.
* **Native registry coverage** for SimProcedures and syscalls (which
  names short-circuit Python, per-arch coverage) is indexed in
  :doc:`/extending-angr/native_coverage_matrix`.

.. _rust-engine-scope:

Scope and threading model
-------------------------

One ``RustExplorationManager`` drives one in-process project. Its
parallelism is its own work-stealing worker pool, engaged with
``parallel_workers=`` (see *Parallel workers* below and
:doc:`rust_parallel_design`). Three architectural facts bound what else
is possible, and the design works *with* them rather than against them:

* ``RustSimState`` shares its solver through ``Rc<RefCell<SymContext>>``
  under a single-threaded copy-on-write idiom, so a state is owned by
  exactly one thread.
* z3-rs binds every Z3 AST to the thread-local Z3 context that created
  it; crossing threads with an AST handle is undefined behavior.
* The claripy bridge's AST caches are ``thread_local!`` by design.

Consequently the load-bearing ``#[pyclass]`` types
(``RustExplorationManager``, ``PyRustSimState``, ``RustSolverContext``)
are ``unsendable``: a manager cannot be stepped from several Python
threads, and workers never share a state. Work moves between workers
only as an ``ExplorationStateSnapshot`` (``Send + Sync``), and each
worker rebuilds its own solver from the snapshot's constraint list.
Turning Z3's own ``parallel.enable`` on, or converting the ``Rc`` to an
``Arc<Mutex<..>>``, does not change this — see
:doc:`rust_engine_decisions`.

Outside the engine's control, and matching Python angr:

* **Real multi-process modeling is not supported.** ``execve`` (program
  image replacement), ``clone`` (shared-address-space forking) and
  ``wait4`` are not modeled on either engine; Python's
  ``angr/procedures/posix/fork.py`` returns a symbolic
  ``If(parent, 1338, 0)`` rather than spawning a second state. Hook the
  syscalls to redirect control flow, or drive each constituent binary
  as its own ``angr.Project`` from an outer orchestrator.
* **No ``dump_fd``.** ``RustPosixProxy`` implements ``dumps(fd) -> bytes``
  for stdin, stdout and Rust-tracked descriptors, mirroring
  ``SimSystemPosix.dumps``. Neither engine writes concretized bytes back
  to a real OS descriptor; do it yourself::

      data = state.posix.dumps(0)   # works on both engines
      os.write(target_fd, data)

* **Bounded symbolic file content is served natively.** At state-add
  time, eligible ``state.fs._files`` entries are exported into the Rust
  ``FileSystem``'s path-keyed registry
  (``RustExplorationManager._export_fs_files_to_rust``), so a native
  guest ``open()`` attaches the per-byte content ASTs and
  ``read``/``fread``/``pread64``/``readv`` serve them without a Python
  bounce; ``fstat``/``fseek``+``ftell``/``feof`` see concrete sizes.
  BVS identity is preserved on import, so constraints added by native
  branching bind the original Python ASTs on the found state. A file is
  exported iff it is exactly ``SimFile`` with ``has_end=True``, seekable,
  ``file_exists=True``, big-endian load model, and a
  uniquely-concretizable size in (0, 64 KiB]; everything else
  (``SimPackets``, streams, symbolic sizes) keeps the Python-fallback
  path. Guest *writes* to an exported file demote it to Python
  ownership; file *position* is not synced back to ``state.fs``; a
  mid-run re-add (``merge()``, cross-manager transfer) re-registers the
  export. Tests: ``tests/engines/rust/test_symbolic_files.py``.

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

Auto dispatch and the ``engine="rust"`` project default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``use_rust_engine`` is tri-state. ``True`` demands the Rust engine and
raises when the workload is one it cannot honor; ``False`` demands the
Python engine; ``None`` means *auto* — run on Rust when the workload is
eligible and on Python when it is not, never raising and never
diverging. The routing decision is always readable afterwards on
``mgr.dispatch_reason``.

A workload is ineligible (and so routes to Python) when any of these
hold: the Rust extension is not built; the architecture is not one of
the six the Rust interpreter implements; a seeding state carries a
raise-listed SimOption (``_RAISE_OPTION_NAMES`` — the ``TRACK_*_ACTIONS``
family, the eager-solve options, …) or lacks a required one
(``_REQUIRED_OPTION_NAMES``); the caller passed a
``SimulationManager``-only kwarg the Rust manager would silently drop
(``resilience``, ``hierarchy``, ``techniques``, …); or a seeding state
already has a breakpoint on an ``inspect`` event the Rust engine does not
dispatch. The predicate is
:func:`angr.exploration.rust_manager.rust_engine_eligible`; it shares
its option sets with the manager's own construction guard, so the two
cannot drift. The decision is logged at debug level and recorded on the
returned manager as ``mgr.dispatch_reason``.

Auto mode is **opt-in**. There are three doors, in increasing scope:

.. code-block:: python

   # 1. Per-call: ask for auto on one manager (needs the global switch, below).
   angr.exploration.set_rust_auto_dispatch(True)     # or ANGR_RUST_AUTO=1
   mgr = proj.factory.simulation_manager(state, use_rust_engine=None)

   # 2. Per-project: every simulation_manager() call that does not name an
   #    engine routes through the predicate. No global switch needed.
   proj = angr.Project("/path/to/binary", engine="rust")
   mgr = proj.factory.simulation_manager(state)      # Rust if eligible

   # 3. Process-wide: same as (2), for every project built while it is set.
   #    $ ANGR_DEFAULT_ENGINE=rust python your_script.py

``engine="rust"`` swaps the **manager**, not the ``SimEngine``:
``factory.default_engine_factory`` stays ``UberEngine``, so ``block()``,
the CFG, and every other lifting/analysis path are untouched. An explicit
``use_rust_engine=True``/``False`` always overrides the project default,
in both directions — including keeping ``True``'s loud raise.

With every door shut (the default), ``simulation_manager()`` builds a
plain Python ``SimulationManager``, exactly as it always has.

.. note::

   Timing under auto dispatch is not comparable run-to-run for the five
   bimodal-Z3 benchmarks — the Rust engine inherits their solver-path
   variance, which is a permanent property of the workloads, not a
   regression. See :doc:`rust_bimodal_variance`.

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

State merge policy (the ``MergePolicy`` derive)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merging N states into one is a **per-field** decision, and
``RustSimState`` has more than forty fields. The merge body itself is
hand-written in ``state/fork.rs::RustSimState::merge``, but *which*
treatment a field gets is declared at the field, next to its
definition, with a ``#[merge_policy = "..."]`` attribute that
``#[derive(angr_macros::MergePolicy)]`` validates at compile time:

.. code-block:: rust

   #[derive(angr_macros::MergePolicy)]
   pub struct RustSimState {
       #[merge_policy = "self_wins"]
       arch: Box<dyn Arch>,
       #[merge_policy = "delegate"]
       memory: SymbolicMemory,
       #[merge_policy = "max"]
       heap_brk: u64,
       // ...
   }

Adding a field without a policy — or with a mistyped one — is a
``compile_error!``, so "the struct literal in ``merge`` silently reused
``self.x.clone()`` and nobody decided that was correct" cannot hide
among the other forty labels.

The vocabulary (see ``VALID_MERGE_POLICIES`` in
``native/angr-macros/src/lib.rs`` for the authoritative list):

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Policy
     - Generated?
     - Meaning
   * - ``self_wins``
     - yes
     - Carried from ``self`` unchanged; the field is not expected to
       diverge across branches (``arch``, ``pc``).
   * - ``in_place_self``
     - no
     - ``self_wins`` for a merge that mutates ``self`` instead of
       building a fresh struct (``SymbolicMemory::merge``), so "self
       wins" means "never assigned" and there is no line to generate.
   * - ``union``
     - yes
     - Set-like union across every branch; for the ``bool`` SimOption
       mirrors, a logical OR.
   * - ``max`` / ``min``
     - yes
     - Numeric extremum across branches (``heap_brk``, ``mmap_base``,
       ``tsc_counter``).
   * - ``warn_on_diverge``
     - yes
     - Kept from ``self``, but ``warn_config_divergence`` runs first so
       divergence is loud rather than silent (``getopt_optind``,
       ``ctype_loc``).
   * - ``computed``
     - no
     - Freshly computed for the merged state, not derived from any
       branch's prior value (``state_id``, ``parent_id``).
   * - ``delegate``
     - no
     - Handed to the field type's own ``merge`` / ``union_from``
       (``registers``, ``memory``, ``solver``, ``heap_metadata``).
   * - ``joint``
     - no
     - Computed jointly with sibling fields, or by scanning every
       branch at once, so no single-field strategy applies (``fs``'s
       longest-stdout scan; ``hooks``, which has to read its
       ``removed_hooks`` sibling).

For the five *generated* policies the derive emits a private
``fn merge_field_<name>(&self, others: &[&Self]) -> FieldTy`` whose
body **is** the policy, and ``merge`` must call it: an uncalled private
method is ``dead_code``, which CI's ``-D warnings`` turns into a build
failure. That is what stops a field labelled ``max`` from being merged
self-wins. A field whose type does not fit the generated shape (a
``union`` over a name-keyed ``Vec`` rather than a ``bool`` OR, a
``warn_on_diverge`` whose divergence test is not ``!=``) opts out
with ``#[merge_manual = "<why the generated body cannot work>"]`` — a
non-empty rationale is required, and the attribute is rejected as
redundant on a non-mechanical policy.

For the other four the derive is validation-only: the label proves a
conscious choice was made, and the merge line stays hand-written
because a per-field derive cannot express it safely.

A ``delegate`` label marks exactly where the compile-time guarantee
*ends*, so the derive is applied to every delegate target that has its
own hand-written merge:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Struct
     - Defined in
     - Behavioural half
   * - ``RustSimState``
     - ``state/mod.rs``
     - ``state/tests/merge_property_mechanical.rs`` +
       ``state/tests/merge_property_manual.rs``
   * - ``SymbolicMemory`` (the ``memory`` field)
     - ``memory/mod.rs``
     - ``memory/tests/merge_sidecars.rs``
   * - ``RegisterFile`` (the ``registers`` field)
     - ``arch/mod.rs``
     - ``arch/register_file_tests.rs``
   * - ``HeapMetadata`` (the ``heap_metadata`` field)
     - ``state/types.rs``
     - ``state/tests/merge_property_manual.rs`` (a ``delegate`` field, so
       it lands in the hand-written half)

The lower three exist because the bug family recurred one level below
the top-level derive: ``SymbolicMemory::merge`` adopted an other-only
page's ``multi_bitmap`` but never copied the matching ``multi_objects``
payloads, because that sidecar is a flat map on ``SymbolicMemory``
rather than nested in ``MemoryPage``. Any new address-keyed sidecar map
now fails to compile until it declares how ``merge`` treats it.

The remaining ``delegate`` field, ``solver``, is deliberately *not*
derived: ``SymContext`` sits behind an ``Rc<RefCell<...>>`` and is a
single shared context both merge arms already point at, so
``RustSimState::merge`` reuses it rather than combining it per-field.

Memory stores under the callback-memory proxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When Rust owns memory, the callback state's ``state.memory`` is a
``RustMemoryProxy`` wrapping the very memory the interpreter is writing
to. Re-entering it from a memory callback would double-borrow, so
``rust_manager.py``'s ``_cb_memory_store`` /
``_cb_memory_store_symbolic_value`` are deliberate **no-ops** in that
configuration; Rust announces it by setting
``PythonCallbacks::memory_is_rust_proxy`` (on by default).

The consequence for interpreter code: a store dispatched *only* to
``call_memory_store_symbolic_value`` is absorbed by nobody — not the
Python shadow (the callback returned without storing) and not Rust's own
``pending_stores`` / ``pending_symbolic_stores`` buffers. It is lost
outright, and the state reads back the page-fill value instead. This is
how flareon2015_5's base64 encoder came to read its own output as
concrete zeros.

Every such dispatch therefore goes through
``statements_store.rs::VEXInterpreter::store_symbolic_value_buffered``,
which pairs ``buffer_store_for_rust_memory`` (gated on
``use_rust_memory && memory_is_rust_proxy``, so the ungated
Python-shadow configuration pays nothing) with the callback call. New
store paths must reuse that helper rather than calling the callback
directly — the guarded-store, CAS-writeback and multi-address ITE paths
each independently grew a raw dispatch before the helper existed.

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
  ``Clone for Box<dyn Arch>`` in ``arch/registry.rs``) **panic** with a
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
* Endianness flag: every endianness flavour of every arch is driven
  end-to-end through both cle loaders by the
  ``test_explore_solves_for_42`` sweep in
  ``tests/engines/rust/test_multiarch.py`` (rows ``armel``/``armeb``,
  ``mips32be``/``mips32le``, ``mips64be``/``mips64le``, ``x86``,
  ``amd64``, ``aarch64``, each x ``blob``/``elf``). ARM64 BE untested —
  blocked on archinfo, see ``archinfo-be-arch-alias-support`` memory.
* ARMEABI / AArch64 / MipsO32 / MipsN64 calling conventions defined in
  ``calling_conventions.rs``.

To promote an arch from Skeleton → Experimental: add at least one
integration test that loads a real binary, runs ``mgr.run(...)``, and
verifies a found state. To promote Experimental → Supported: add a
benchmark and ensure it stays green in regression runs. The Cdecl x86
return-register bug was latent for months
precisely because no end-to-end x86 test ran — that gap is now closed
by the ``x86-blob`` / ``x86-elf`` rows of ``test_explore_solves_for_42``
(i386 ``cmp``/``je``/``jmp`` round-trip asserting ``eax == 42`` solves
``2*eax + 16 == 100``). Adding an architecture means adding a row to
``tests/engines/rust/arch_specs.py::ARCH_SPECS``, which buys the whole
assertion matrix; assume the same risk for any new arch added without it.

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

``satisfiable()`` is claripy-compatible and therefore lenient: a query
Z3 could not decide within the solver timeout reports ``False``, the
same answer a proven contradiction gets (a ``[rust:WARN]`` line is
logged when this happens). Code that would *drop* a state on ``False``
should call ``satisfiable_checked()`` instead, which returns ``True`` /
``False`` for a decided query and ``None`` when the query timed out.
The same distinction governs the Rust side: only a decided unsat prunes
(see ``SymContext::is_sat_checked`` and
``check_branch_feasibility``).

``eval_upto()`` is lenient in the same way: an enumeration cut short by
an undecided Z3 query returns a short list, indistinguishable from an
exhaustive one. Code whose correctness depends on the set being
*complete* — anything that concludes "only these values are possible" —
should call ``eval_upto_checked()``, which returns the list for a
decided enumeration and ``None`` when Z3 stopped deciding part-way
through. A non-``None`` result of exactly ``n`` values may still have
hit the cap; request ``n + 1`` and treat ``len > n`` as an overflow,
which is the same contract the Rust-side ``Enumeration`` carries
(``SymContext::eval_upto_checked`` /
``SymContext::eval_upto_wide_checked``).

The two constraint-import entry points on ``RustExplorationManager``
follow the same pair convention: ``add_constraints_to_state()`` and
``import_z3_constraint_ptrs()`` return the lenient bool (an undecided
post-import query reads as ``False``, with a ``[rust:WARN]`` line),
while ``add_constraints_to_state_checked()`` and
``import_z3_constraint_ptrs_checked()`` return ``None`` for that case.
The constraints are imported either way — only the reported verdict
differs — so a caller that prunes on ``False`` should use the checked
form.

For the build-time wiring that makes a shared ``libz3.so`` possible,
the ``SymContext`` lifecycle across forks, and the push/pop discipline
that ``RustSolverContext`` / ``SharedLineageSolver`` both rely on,
see :doc:`rust_z3_sharing`.

Z3 solver profiling counters
----------------------------

Process-wide atomic counters in
``native/angr/src/symbolic/stats.rs`` (``get_solver_stats``) track every
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
* ``z3_ast_build`` — top-level ``to_z3_ast()`` / ``to_z3_bool()``
  call count (entry calls, not Z3 nodes built; see
  :doc:`rust_z3_sharing`).
* ``z3_site_<name>_count`` / ``z3_site_<name>_time_ns`` — per-call-site
  breakdown. Sites include ``satisfiable``, ``branch_true``,
  ``branch_false``, ``eval``, ``eval_upto``, ``min_init``,
  ``min_search``, ``max_init``, ``max_search``.

Counters are global (shared across ``SymContext`` instances). Call
``mgr.reset_solver_stats()`` to zero them at the start of a measured
window. The same dict is also merged into ``mgr.stats`` for
convenience.

Z3 simplification
-----------------

The engine never calls ``Bool::simplify`` on the assertion hot path and
never invokes ``Z3_simplify_ex`` directly; what it does do is below.
Push/pop discipline and lineage sharing are covered in
:doc:`rust_z3_sharing`.

Simplification toggles
~~~~~~~~~~~~~~~~~~~~~~

Z3 simplification touches the engine in three places:

1. **No call to ``Bool::simplify`` on the assertion hot path.** The
   only ``.simplify()`` invocation in ``native/angr/src/`` is the
   sampled measurement helper ``sample_simplify_skip``
   (``symbolic/solver_build.rs::sample_simplify_skip``), which runs every 64th call across
   ``assume_true`` / ``assume_false`` / ``add_constraint_raw`` and
   only records whether the simplification would have produced a
   distinct AST pointer. The simplified Bool is dropped — the
   un-simplified ``z3::ast::Bool`` is what reaches the solver. The
   sample rate keeps measurement overhead at ~1.5%.
2. **Solver-internal preprocessing.** ``build_solver_params``
   (``symbolic/solver_build.rs::build_solver_params``) sets ``bv_extract_prop=true`` and
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

Stride override (``ANGR_Z3_SIMPLIFY_STRIDE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sample_simplify_skip`` reads its stride from
``ANGR_Z3_SIMPLIFY_STRIDE`` via a ``OnceLock`` on first sample (same
pattern as ``ANGR_Z3_TACTIC`` / ``ANGR_Z3_QFBV_THRESHOLD``). The const
default ``SIMPLIFY_SAMPLE_STRIDE_DEFAULT = 64``
(``symbolic/stats.rs``) applies when the env var
is unset, empty, unparseable, or ``0`` — so production runs and the
benchmark gate see exactly the previous behavior. The override is
write-once per process: changing the env var after the first
``add_constraint_raw`` / ``assume_true`` / ``assume_false`` has no
effect.

Recipe for a full-population spike on a bimodal bench::

    ANGR_Z3_SIMPLIFY_STRIDE=1 \
      python tests/benchmarks/run_single.py securityfest_fairlight \
        --engine rust --counters-json \
      | jq '.branch_cond_simplify_sampled,
            .branch_cond_simplify_reduced'

A stride of 1 calls ``Bool::simplify`` on every assertion (one extra
Z3 call per assertion). On dense-assertion benches this measurably
slows the run — never enable in CI; reach for it only inside an
ad-hoc profiling session.

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

Iteration-order rule (HashMap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No ``std::HashMap`` in ``native/angr/src/`` may leak its iteration
order into an exploration *result* — found/avoided stashes, evaluated
bytes, or constraint sets. Stash membership is kept in a ``VecDeque``
so FIFO/LIFO order is honored; the registries (dirty handlers,
syscalls, procedures, the symbol tables, the claripy-bridge caches) are
lookup-only; the stats and profiling maps are returned to Python as
dicts whose values are order-independent counters. A new ``HashMap``
that is iterated into output must either use ``rustc_hash::FxHashMap``
(fixed seed, deterministic for a given insertion sequence) or sort
before emitting. The remaining nondeterminism comes from Z3.

Residual Z3 nondeterminism
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pinning Z3's seed parameters does not make models stable across fresh
solver instances (see :doc:`rust_engine_decisions`); ``deterministic=True``
below is the supported knob. What the engine still carries:

* The trailing-byte mismatch in ``defcamp_r100`` (unconstrained stdin
  fill values) — ``run_regression.py`` normalizes output.
* ``test_model_stability_constraint_order`` only verifies
  order-independence *within a process*, not run-to-run stability
  across Z3 rebuilds.
* Two fresh ``RustSolverContext`` instances with identical constraints
  may eval the same underconstrained variable to different witnesses.

Parallel workers (``parallel_workers=``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RustExplorationManager(project, parallel_workers=N)`` is the programmatic, non-env way to engage the work-stealing scheduler
pool — the twin of the ``RUST_PARALLEL_WORKERS`` environment variable
that benches use. Default ``1`` (single-threaded, zero-regression).

Usage::

    from angr.exploration import RustExplorationManager

    state = proj.factory.entry_state()
    mgr = RustExplorationManager(proj, [state], parallel_workers=4)
    mgr.explore(find=target_addr, num_find=8)
    assert mgr.stats["parallel_real_workers"] == 4

Semantics:

* **Engages on an eligible explore only.** Parallel workers spin up when
  the explore is *address-based* — concrete ``find`` / ``avoid``
  addresses, no ``until`` predicate, no active techniques — which is
  where worker-local frontiers are correct. The kwarg is applied at ``explore()`` time, once
  eligibility is known, so ``mgr.stats["parallel_real_workers"]`` reads
  ``1`` between construction and the first eligible ``explore()``.
* **Ineligible explores fall back to single-threaded.** A callable
  ``find`` / ``avoid`` predicate has no worker-local skip-state analogue,
  so the manager logs the fallback and runs the serial loop
  (``sum(mgr.stats["parallel_worker_dispatch"]) == 0``; note
  ``parallel_tasks`` is *not* a parallelism signal — the migration
  model bumps it on the serial path too).
* **The environment variable wins.** When ``RUST_PARALLEL_WORKERS`` is
  set it is read once at construction and overrides the kwarg, so benches
  keep explicit control.
* **Incompatible with** ``deterministic=True``: passing both with
  ``parallel_workers > 1`` raises ``ValueError`` at construction — the
  steal order is nondeterministic by design (see below).
* The steady-state loop (``RUST_PARALLEL_STEADY``) is **not** auto-armed
  by this kwarg; it over-collects the found stash relative to the serial
  baseline for ``num_find > 1`` and stays an explicit env opt-in until
  that accounting is fixed. The kwarg engages the wave loop only.

Deterministic mode (``deterministic=True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RustExplorationManager(deterministic=True)`` pins
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
* Since angr-op0dn.10.3 the flag also turns on **strict-deterministic
  witness selection** in every state's ``SymContext``
  (``RustExplorationManager.set_deterministic`` on the Rust manager,
  propagated as each state enters a stash and inherited by every fork).
  The seed pin only stabilizes *which model Z3 builds*; witness
  selection makes the answer a function of the asserted constraints
  alone, which is what the seed pin could never guarantee:

  - ``eval`` returns the unsigned **minimum** of the feasible set
    rather than whatever value Z3's model happened to carry.
  - ``eval_upto(n)`` returns the **ascending prefix** of the feasible
    set, so a truncated result is a canonical prefix of the sorted
    full set instead of an arbitrary Z3-chosen subset.
  - ``eval_wide`` and the multi-part ``eval_batch`` (angr-op0dn.10.7)
    return the **lexicographic minimum** over their parts, minimized in
    order with each earlier part pinned. All parts still come off one
    satisfying assignment, and because both callers decompose
    big-endian, that minimum *is* the unsigned minimum of the
    reassembled value — the same rule ``eval`` follows. These are the
    paths a user's ``state.solver.eval(x, cast_to=bytes)`` or
    ``posix.dumps(0)`` takes on an exported found state, so an answer
    printed after exploration is now canonical too.

  This costs Z3 checks — each witness is an ``O(log width)`` binary
  search instead of one ``get_model``, and the warm model-cache seed is
  deliberately bypassed (a cached model is a history-dependent witness,
  exactly the nondeterminism the mode removes). That is why the mode is
  opt-in rather than the default.

What the flag does **not** close:

* Z3 4.13 retains heuristic latitude (variable selection ordering,
  restart timing, internal simplification passes) that is not bounded
  by the two pinned seeds. This no longer leaks into ``eval`` /
  ``eval_upto`` results, but it does still affect solve *timing* and
  any query whose answer is not canonicalized.
* Bitvectors **wider than 128 bits**: the binary search tracks its
  bounds in ``u128``, so ``eval_upto_wide`` keeps only the weaker
  enumerate-then-sort ordering (canonical only when ``n`` is at least
  the number of feasible values). Single-witness ``eval_wide`` *is*
  canonical at any width — it minimizes byte by byte, and each byte
  fits the search — but an ``eval_batch`` whose individual parts exceed
  128 bits still falls back to the arbitrary-model path.
* **Multiple scheduler workers** (``RUST_PARALLEL_WORKERS`` > 1): the
  work-stealing pool's steal order is nondeterministic by design, so
  the found *set* still matches a serial run but the order states are
  found in does not. Constructing a manager with ``deterministic=True``
  under a multi-worker pool emits a ``RuntimeWarning`` saying so; it is
  a warning rather than an error because witness choice is per-state
  and stays canonical. Set ``RUST_PARALLEL_WORKERS=1`` for a fully
  reproducible run.
* The ``defcamp_r100`` trailing-byte mismatch was the canonical
  residual case for the seed-pin-only mode —
  ``run_regression.py`` normalizes the symbolic-fill padding so the
  bench passes regardless.

Coverage is exercised by
``tests/engines/rust/test_solver_ops.py::TestDeterministicMode`` —
fauxware ``explore(find=0x4006ed, avoid=0x4006fd)`` produces the same
found-stash size and the same evaluated stdin across two fresh
managers when ``deterministic=True`` is set. The iteration-order rule
above means this flag does **not** need to be paired with an
FxHashMap conversion.

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
  dispatch in ``interpreter/expressions_arith.rs``):

  * ``vex_unop_total`` / ``vex_binop_total`` / ``vex_triop_total`` /
    ``vex_qop_total`` — count per arity.
  * ``vex_op_arith`` / ``vex_op_logic`` / ``vex_op_shift`` /
    ``vex_op_cmp`` / ``vex_op_ext`` / ``vex_op_fp`` / ``vex_op_vec`` /
    ``vex_op_other`` — count per IROp family. Family classification
    lives in ``vex/ops/classify.rs::iropclass``; ``Vec`` covers every
    ``V*``-prefixed (SIMD/NEON) variant, ``Other`` is the catch-all for
    ``Raw``/``NeonUnimplemented``. Every dispatched op bumps exactly
    one family counter, so ``sum(vex_op_*) == sum(vex_<arity>_total)``.

* **Memory volume** (bumped in ``memory/load.rs`` and
  ``memory/store.rs``):

  * ``mem_load_count`` / ``mem_store_count`` — total load/store calls
    reaching ``SymbolicMemory::{load,store}_concrete``. Catches both
    the public ``load(addr_bv)`` entry and the ``state/mod.rs`` hot path
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

  The dedup hit-rate varies widely across benches; a high rate means
  the workload keeps re-adding constraints the solver already holds.
  The side-table is kept even where it never hits: its cost is one
  ``insert`` + ``contains`` per add, trivial against the ``z3_check``
  time it gates, while a duplicate that gets through accumulates in the
  per-context ``z3_assertions`` vec and scales constraint-export and
  lineage-switch work for nothing (``angr-dtrl``).

``tests/benchmarks/run_single.py`` prints these counters grouped by
category whenever the Rust engine ran with non-zero counters;
``--dump-counters`` prints every one and ``--counters-json`` emits the
dict for ``bench_diff.py`` (see :doc:`rust_build`).

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

Measured on the bimodal hash-cracker benches, ``qfbv`` cut wall time by
more than half; on CTF-style benches with many small constraint
problems it regressed them by up to a few multiples.

The shape: ``qfbv`` skips Z3's portfolio dispatch and goes straight to
a bitvector-specialized pipeline. Hash-cracker problems with many
constraints over a single state-set benefit; CTF-style binaries with
many small constraint problems pay the qfbv pipeline's per-check
overhead on every solve.

Note: ``sat-preprocess:qfbv`` was tried and rejected — it produced
incorrect results on ``csgames2018`` and ``securityfest_fairlight``
(``IndexError: list index out of range`` from the post-solve goal
unwrap). Avoid pipelines beginning with ``sat-preprocess`` until that
is debugged.

Experimental: ``ANGR_Z3_PARAMS`` extra solver-param override
------------------------------------------------------------

``ANGR_Z3_PARAMS`` appends arbitrary Z3 solver params to every fresh
solver, *after* the baked defaults (``timeout``, ``bv_extract_prop``,
``mul2concat``), so a param survey can override or extend them without a
rebuild. The value is a comma-separated list of ``key=value`` pairs, read
once per process on first solver construction and cached in a
``OnceLock``:

.. code-block:: bash

    ANGR_Z3_PARAMS="bv.size_reduce=true,relevancy=0,sat.phase=sym:always_false" \
        python tests/benchmarks/run_single.py fauxware --engine rust

``value`` of ``true`` / ``false`` (case-insensitive) sets a bool param; a
value that parses as ``u32`` sets a uint param; a value prefixed ``sym:``
sets a symbol param (e.g. ``sat.phase=sym:always_false``). Everything else
— including a plain non-numeric value with no ``sym:`` tag — is skipped
silently rather than aborting the run. The tag is required, not inferred:
an untagged unparseable value (e.g. a typo like ``timeout=5oo``) has no way
to signal whether a bool, uint, or symbol was intended, so treating it as a
symbol would risk silently corrupting the solver instead of leaving the
mistyped override as a harmless no-op — a survey harness sweeping many
candidate specs stays robust to a typo only because unparseable entries are
dropped, never reinterpreted. Because a later ``key`` overrides an earlier
one, listing ``timeout=...`` here also overrides the per-solver timeout Z3
would otherwise receive.

**Default behavior (env unset) is unchanged** — the override list is
empty and only the baked defaults apply. This knob exists for the
A/B param surveys under ``angr-ovqja`` (z3/ffi/concretize hotpath perf):
capture ``--counters-json`` across candidate specs without recompiling
per candidate. It is *not* a supported production tuning surface — no
value is baked into a default, and unknown Z3 param names are handled by
Z3 itself (which may ignore or reject them).

Experimental: ``use_shared_lineage_solver``
-------------------------------------------

``RustExplorationManager(use_shared_lineage_solver=True)`` lets a forked
lineage of states share one Z3 solver and switch scopes with push/pop
instead of materializing a solver per fork. It is **opt-in and stays
opt-in**: the saving reaches wall time only where sibling
reconvergence is dense enough to keep the switch cache hot, and the
runtime sampler that dismantles the lineage when it is not cannot
predict the steady-state ratio early enough to make a default safe.
Expect roughly ±10 % in whichever direction
``lineage_switch_fast_path_count / lineage_switch_count`` predicts.
The investigation and the reasons the fuller variants were reverted
are recorded in :doc:`rust_engine_decisions`; the push/pop interaction
is in :doc:`rust_z3_sharing`.

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
     - ``rust_manager.py::_phase_activate`` (re-checked at ``explore()`` time)
     - Calls ``set_lazy_solves(True)``; Rust skips per-block satisfiability
       checks. Also propagated through ``_apply_state_metadata`` on
       disk-cache reuse.
   * - ``ZERO_FILL_UNCONSTRAINED_MEMORY``
     - ``rust_manager.py::_phase_activate``
     - Calls ``set_zero_fill_unconstrained(True)``; uninitialized memory
       reads return ``BVV(0, n)`` instead of fresh symbols.
   * - ``APPROXIMATE_MEMORY_INDICES``
     - ``rust_manager.py::_phase_activate`` (forwarded to
       ``configure_concretization_strategies``)
     - Passed as ``use_approx`` to ``configure_concretization_strategies``;
       Rust adds an approximation strategy ahead of full Z3 enumeration
       on symbolic loads.
   * - ``SYMBOLIC_WRITE_ADDRESSES``
     - ``rust_manager.py::_phase_activate`` (forwarded to
       ``configure_concretization_strategies``)
     - Passed as ``sym_write`` to ``configure_concretization_strategies``;
       Rust permits multi-valued symbolic write targets instead of
       forcing concretization.
   * - ``STRICT_PAGE_ACCESS``
     - ``rust_manager.py::_add_rust_state`` (per-state, also propagated through
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
       both set (``HeavyVEXMixin.process_successors`` in
       ``angr/engines/vex/heavy/heavy.py``).
   * - ``NO_IP_CONCRETIZATION``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_no_ip_concretization(True)``; at block
       boundaries with a symbolic jump target the state routes to the
       ``unconstrained`` stash silently instead of being enumerated. The
       check fires inside ``VEXInterpreter::eval_next_addr_concretized``
       (``native/angr/src/interpreter/exits.rs``). Matches Python's
       ``engines/successors.py::_categorize_successor`` behavior
       (``max_targets=0`` with ``skip_max_targets_warning=True``).
   * - ``NO_SYMBOLIC_JUMP_RESOLUTION``
     - ``rust_manager.py::_add_rust_state`` (also propagated through
       ``_apply_state_metadata`` on cache reuse)
     - Calls ``RustSimState::set_no_symbolic_jump_resolution(True)``;
       OR'd with ``no_ip_concretization`` at the
       ``eval_next_addr_concretized`` short-circuit, so a symbolic jump
       target routes to the ``unconstrained`` stash without enumeration.
       Matches Python's ``engines/successors.py::_categorize_successor`` (early elif
       branch routing symbolic targets to ``unconstrained_successors``
       before ``AddressConcretizer`` is invoked).
   * - ``NO_SYMBOLIC_SYSCALL_RESOLUTION``
     - Inherited through the Python syscall fallback
     - ``VEXInterpreter::get_syscall_num`` returns ``Option<u64>``
       (``None`` when the syscall register is symbolic), and
       ``core_outcome_handlers.rs::handle_syscall_core`` (the
       ``RunResult::Syscall`` handler) skips the native syscall
       registry when ``num`` is ``None``, forcing a Python callback. The
       Python side's ``engines/successors.py:_resolve_syscall`` reads
       the option from ``state.options`` and either enumerates concrete
       syscalls or routes to the unknown-syscall stub.
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
       handled by
       ``core_outcome_handlers.rs::handle_symbolic_jump_target_core``.
       Mirrors Python's
       ``engines/successors.py::_categorize_successor``.

   * - ``AVOID_MULTIVALUED_READS`` / ``AVOID_MULTIVALUED_WRITES``
     - ``rust_manager.py::_phase_activate`` (forwarded to
       ``configure_concretization_strategies``)
     - Switches the Rust memory plugin to return an unconstrained value
       rather than enumerate addresses when the option is set.

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
       Mirrored onto the Rust state as well (``_NATIVE_SIMOPTIONS``) so the
       native ``ReturnUnconstrained`` stub returns a concrete zero rather
       than a fresh symbol when the option is removed.
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
       directly; a native POSIX procedure that served these paths would
       bypass the flags, so none does.
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
       on the fallback path.

Ignored — divergence-risk
~~~~~~~~~~~~~~~~~~~~~~~~~

These options *would* change Python-engine behavior. Under Rust they
are silently dropped, which can produce different found/avoided sets
vs. Python.

.. note::

   The **(b)** rows are warned once per process via
   ``_REJECTED_OPTION_NAMES`` (``angr/exploration/rust_manager.py``), except
   the two that ship in the default ``symbolic`` mode bundle
   (``TRACK_CONSTRAINT_ACTIONS``, ``TRACK_MEMORY_MAPPING``) — warning on
   every ``entry_state()`` would be noise the user did not ask for. The
   **(c)** rows raise ``NotImplementedError`` at manager construction via
   ``_RAISE_OPTION_NAMES``; under the auto dispatcher that raise is a
   *routing signal* and the workload transparently gets a Python
   ``SimulationManager``. For the action-tracking family, Rust will not
   grow native ``SimAction`` recording (``SimAction``\ s wrap claripy
   objects Python-side); the read surface Rust does provide
   (``RustHistoryProxy``: ``bbl_addrs`` / ``recent_bbl_addrs`` /
   ``block_count`` / ``jumpkind``) is the parity floor. Materialized
   Rust-owned states carry a ``history`` plugin promoted to
   ``_RustOwnedSimStateHistory`` (``angr/exploration/rust_state_export.py``)
   that warns once the first time ``state.history.actions`` or
   ``.events`` is read, so the empty stream is attributed where it is
   consumed.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Option
     - What Python does
     - Suggested fix
   * - ``CONCRETIZE_SYMBOLIC_WRITE_SIZES``
     - Concretizes the *size* of a symbolic-sized write.
     - **Honored transparently** (``angr-op0dn.14.8``). The option's only
       in-tree consumer is ``SimFileBase._prep_generic``
       (``storage/file.py``), and every native write path
       (``syscalls/write.rs``, the CGC ``transmit`` in
       ``syscalls/cgc.rs``) falls back to Python on a symbolic count — so
       the option fires in Python wherever it can fire at all. The
       identically-named *memory* knob is a ``SimMemory`` constructor
       kwarg (``size_resolution_mixin.py``), not this SimOption.
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
       doing what it asks for.
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
       (``native/angr/src/memory/load.rs::load_concrete_lazy``) returns a fresh
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
     - Populate ``state.history.actions`` with ``SimActionConstraint``
       records.
     - **(b) explicitly reject, attributed at the read site** — ships in the
       default ``symbolic`` mode bundle so it can neither raise nor warn at
       manager construction without firing for every ``entry_state()``. Its
       only observable effect is the action stream, and reading
       ``state.history.actions`` on a Rust-owned state warns once per process
       via ``_RustOwnedSimStateHistory``
       (``angr/exploration/rust_state_export.py``), so the divergence is
       attributable where the user actually consumes it (``angr-op0dn.14.7``).
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
     - Nominally logs map/unmap into ``state.history``.
     - **Vacuous** (``angr-op0dn.14.7``). Vestigial in current angr: nothing
       in the tree (or in ``claripy`` / ``cle`` / ``pyvex``) ever *reads* the
       option — the sole reference adds it to an option set
       (``analyses/identifier/runner.py``). No code path branches on it, so
       Rust ignoring it cannot change an answer. Kept in this table only
       because the ``TRACK_*`` family it is named after is genuinely
       divergence-risk.
   * - ``CONSTRAINT_TRACKING_IN_SOLVER``
     - Required for ``solver.unsat_core()``.
     - **(a) honored** (``angr-op0dn.14.2``). The engine still adds
       constraints to the shared Z3 solver with no assumption literals — that
       cost belongs nowhere near the fork path — so
       ``RustSolverProxyPlugin.unsat_core()`` computes the core *on demand*
       instead: ``SymContext::unsat_core_assumed`` rebuilds a throwaway
       solver from the assumed-constraint IR with one assumption literal per
       constraint and runs ``check_assumptions``. The blamed indices line up
       1:1 with ``state.solver.constraints``, and because the IR also holds
       the constraints the *engine* added (fork guards, SimProc adds), the
       core is complete rather than silently partial. Without the option on
       the bound state, ``unsat_core()`` raises ``SimSolverOptionError``,
       matching Python.
   * - ``BYPASS_UNSUPPORTED_IROP``, ``BYPASS_UNSUPPORTED_IRDIRTY``,
       ``BYPASS_UNSUPPORTED_IRCCALL``, ``BYPASS_UNSUPPORTED_SYSCALL``
     - Tell Python's ``HeavyResilienceMixin``
       (``engines/vex/heavy/resilience.py``) to substitute a default
       symbol/zero on unsupported ops/dirty helpers/ccalls/syscalls.
     - (a) honored transparently. Rust's interpreter routes unsupported
       VEX features through ``FallbackStrategy::PythonCallback``
       (the ``FallbackStrategy`` enum in
       ``native/angr/src/interpreter/execution_error.rs``); Python re-runs
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
       run under Python (angr-6rz8).
   * - ``BYPASS_VERITESTING_EXCEPTIONS``
     - Tells ``analyses/veritesting.py`` to forward ``resilience=True``
       to the nested ``SimulationManager.run`` call.
     - **(b) explicitly reject** — only consulted from inside the
       Veritesting *analysis*, which builds its own Python
       ``SimulationManager`` and is unsupported under a Rust-attached
       project regardless. Outside Veritesting the option is a
       no-op. Carried by the ``angr.options.resilience`` bundle, so
       reject-with-warn rather than raise to keep bundle users alive
       (angr-6rz8).
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
     - **(a) honored** (demoted from raise, angr-op0dn.11.6). Merge is
       now native: ``RustExplorationManager.merge()`` runs the M3-4
       fast path (``merge_states``, no export, no ``SimStateHistory``
       walk) and the M3-5 native ``ManualMergepoint`` technique
       forks-and-merges in Rust, so the option's Python common-ancestor
       rationale is moot. It is honored by not raising — a state
       carrying it (including Veritesting's auto-added copy) explores
       under the native merge machinery. The paired
       ``SIMPLIFY_MERGED_CONSTRAINTS`` is honored implicitly through
       the Python ``state.merge()`` fallback that still backs custom
       ``merge_func``/``merge_key``.
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
       native path is ignoring it (angr-0y0v).
   * - ``PRODUCE_ZERODIV_SUCCESSORS``
     - Keeps the ``Ijk_SigFPE_IntDiv`` successor Python spawns on a
       concrete zero divisor (dropped in ``add_successor`` when unset).
     - **(c) raise NotImplementedError** at manager construction
       (``angr-op0dn.14.9``). Python's zero-division successor is produced
       during lifting — ``irop.py`` raises ``SimZeroDivisionException`` and
       the resilience mixin turns it into a SigFPE exit — whereas Rust's
       ``DivS`` / ``DivU`` lower to Z3's ``bvsdiv`` / ``bvudiv``, which are
       *total*, so the successor never exists to keep. Implementing it is a
       feature; the option ships only in the ``tracing`` bundle (reached by
       passing ``mode=``, an opt-in), so raising is routable and the
       auto-dispatcher sends those workloads to Python.
   * - ``EXTENDED_IROP_SUPPORT``
     - Widens Python's IR-op table: ``vexop_to_simop`` auto-generates a
       ``SimIROp`` from the op name instead of raising
       ``UnsupportedIROpError``.
     - **Honored when set** (the default in every mode bundle) — Rust runs
       the op natively, or defers the block to Python where the widening
       applies. **Raises when unset** (``angr-op0dn.14.7``): removing it asks
       for the *narrow* op table, and the Rust interpreter has no narrow mode,
       so it would execute an op Python would have refused. This is the
       inverse-polarity gate ``_REQUIRED_OPTION_NAMES``
       (``angr/exploration/rust_manager.py``) — the only option refused by its
       *absence*.
   * - ``CGC_NON_BLOCKING_FDS``
     - When *not* set, Python's ``fdwait`` SimProcedure returns symbolic
       1-bit ready flags for each fd; when set, it returns concrete 1.
     - **Honored** in both directions (``angr-op0dn.14.8``). Rust's
       ``NativeFdwaitSyscall`` (``native/angr/src/syscalls/cgc.rs``)
       writes concrete 1-bits — i.e., it implements exactly the
       option-is-set behavior — and returns a ``SyscallError`` when the
       option is unset, deferring to the Python ``fdwait`` proc that
       produces the unconstrained ready bits. Before that gate the stub
       ran unconditionally, so a CGC binary branching on fd readiness
       explored both arms under Python and only the all-ready arm under
       Rust.

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

.. _Automatic engine selection:

Provenance
~~~~~~~~~~

The matrix is maintained by hand from ``angr/sim_options.py``. When a
new SimOption lands, add a row here and either wire detection into
``rust_manager.py`` (Honored) or classify the silent-ignore reason. The
option ledgers in ``rust_manager.py`` (``_RAISE_OPTION_NAMES``,
``_REJECTED_OPTION_NAMES``, ``_REQUIRED_OPTION_NAMES``) and this matrix
are cross-checked by ``tests/benchmarks/parity_census.py`` in the
nightly lane, which fails on drift between them.

state.inspect dispatch
----------------------

The Rust symbolic-execution engine dispatches every *engine*
``state.inspect`` event to Python breakpoints. The handful of events
that fire from angr components outside the engine (see *Unsupported
events*) raise ``NotImplementedError`` at registration time so that
gaps remain loud.

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
``constraints`` ``before``/``after`` ``added_constraints``
``vex_lift``    ``before``/``after`` ``vex_lift_addr``, ``vex_lift_size``,
                                  ``vex_lift_buff``
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

``constraints`` and ``vex_lift`` are supported (see the table above);
the residual unsupported set is ``engine_process``, ``memory_page_map``,
``cfg_handle_job``, ``vfg_handle_successor``, and ``vfg_widen_state``.
Registering a BP for one of those raises ``NotImplementedError`` with a
message pointing at this document. ``_NoOpInspectProxy`` previously
silently accepted every registration; raising loudly
prevents users from depending on a feature the engine cannot fulfill.

None of the five is an *engine* event: they fire from angr components
that sit outside the Rust manager entirely — ``engine_process`` from
``SimEngine.process`` (the Python engine's own entry point, which the
Rust manager replaces), ``memory_page_map`` from the Python paged-memory
mixins, and the ``cfg_*`` / ``vfg_*`` events from the CFG and VFG
analyses (which always run on Python states). A workload that needs any
of them is not running under the Rust engine's step loop anyway.

Two dispatch caveats on the events that *are* supported:

* ``constraints`` fires from two origins — the Python solver
  write-through (``RustSolverProxyPlugin.add``, on SimProcedure callback
  states) and the native fork-guard add
  (``exploration/fork_materialize.rs::add_fork_guard_constraint``, when a deferred
  fork's branch guard is installed on the continuing state). A
  ``BP_BEFORE`` action may replace ``added_constraints`` on the first
  path; on the native path the guard is already lowered into a Rust BV,
  so a replacement is ignored. The inverted guard the *forked* state
  receives does not fire its own pair — the ``fork`` BP marks that
  state's creation.
* ``vex_lift`` fires from both lift paths: the Python lift callback (the
  block-cache miss path, ``_cb_lift_block``) and the feature-gated native
  in-process libVEX lift (``interpreter/execution.rs::try_native_lift``),
  which bypasses that callback. Exactly one ``BP_BEFORE``/``BP_AFTER``
  pair is fired per lift — on the native path both fire once libVEX has
  run, so a ``BP_BEFORE`` mutation of ``vex_lift_buff`` /
  ``vex_lift_addr`` / ``vex_lift_size`` is ignored there (it is ignored on
  the Python path too: the engine lifts its own bytes).

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
(``angr/exploration/rust_state_proxy.py``). A
``RustStateProxy`` constructed without a manager routes
``state.inspect.b(...)`` to ``_NoOpInspectProxy``
(same module), which raises the
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
   depend on an unsupported event (``engine_process``, ``memory_page_map``,
   …) or on an unsupported write-back of a supported event (e.g.,
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
   (``angr/exploration/rust_manager.py``) fires every
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
in the Rust callbacks ``inspect_enabled`` atomic bitmask
(``AtomicU32``), with the
bit-to-event mapping declared by the ``inspect_events!`` invocation that
defines ``InspectBit`` in ``native/angr/src/callbacks/inspect_bits.rs``. The
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
* ``step_state`` — dispatched per active state via
  ``dispatch_step_state_with_hooks`` (``angr-op0dn.11.7``), the wiring
  that makes ``Veritesting`` work under the Rust manager. Each active
  state is exported and run through the ``HookSet``-composed
  ``step_state`` chain. A technique either **applies** — driving a
  nested Python analysis (the CMU merging algorithm for Veritesting)
  and returning a successor-dict of angr ``SimState`` objects, which are
  re-imported into the Rust stashes via ``_add_rust_state`` — or
  **declines** to ``simgr.step_state(state)``, in which case the source
  is advanced by a normal native ``run()`` (which, unlike the E1 proxy
  ``step_state``, drives SimProcedure/syscall callback bounces). The
  ``veritesting_dispatches`` / ``veritesting_applied`` stats counters
  make the routing observable. Note that when a technique declines on a
  block, no Python-side merging happens; and because Veritesting's
  merges occur inside its *own* nested Python ``SimulationManager``,
  ``states_merged_native`` stays 0 (use ``veritesting_applied`` to
  confirm a merge fired).
* ``successors`` — **not dispatched.** This hook expects a
  ``SimSuccessors`` object built by Python execution, which the Rust
  engine deliberately bypasses. ``RustSimulationManagerProxy`` raises
  ``NotImplementedError`` so technique bugs surface loudly instead of
  silently no-op'ing. Registration is still permitted (no exception at
  ``use_technique`` time), but the per-successor hook never fires;
  ``use_technique()`` emits a ``logging.WARNING`` flagging the
  overridden hook. Drop to ``use_rust_engine=False`` if your workflow
  needs it.

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

``mgr.remove_technique(tech)`` mirrors
:meth:`~angr.sim_manager.SimulationManager.remove_technique`: the
technique stops being dispatched *and* the native state it armed is
undone (``angr-w9zce``). Because the Rust manager has no per-technique
unregister, removal resets the native *categories* the removed technique
touched — state-selection policy, ``find``/``avoid`` addresses,
uniqueness filter, or the native technique queue (``LoopSeer`` /
``LengthLimiter`` / ``Timeout`` / ``ManualMergepoint``) — and then
re-arms the techniques still registered. Categories the removed
technique never touched are left untouched, so a manual
``set_state_selection_*`` or ``set_find_addrs`` call made outside the
technique API is not clobbered. Removing a technique that was never
registered returns ``False`` and disarms nothing.

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
     - **step_state dispatched (angr-op0dn.11.7)**
     - The technique's ``step_state()`` is routed through
       ``dispatch_step_state_with_hooks`` (``rust_techniques.py``): each
       active state is exported, run through the ``HookSet``-composed
       chain, and the nested CMU merging analysis' successor dict is
       re-imported into the Rust stashes. ``EFFICIENT_STATE_MERGING``,
       which the technique auto-adds, is **no longer** in
       ``_RAISE_OPTION_NAMES`` (demoted in ``angr-op0dn.11.6`` — merge is
       native, so the ``SimStateHistory`` strongref rationale is moot),
       so construction no longer raises. Watch the
       ``veritesting_dispatches`` / ``veritesting_applied`` stats
       counters to confirm the hook fired; ``states_merged_native``
       stays 0 because the merges happen inside the technique's own
       nested Python ``SimulationManager``. Note the *analysis*
       (``angr.analyses.veritesting``) remains unsupported — see the
       Analyses matrix below.
   * - ``Threading``
     - **Unsupported — use** ``parallel_workers=`` **instead**
     - Wraps ``simgr.step`` in a Python thread pool. The manager and its
       states are ``unsendable`` (see *Scope and threading model*), so a
       manager cannot be stepped from several Python threads. The
       engine's own work-stealing pool, engaged with
       ``RustExplorationManager(parallel_workers=N)``, is the supported
       parallelism.
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
     - **Incompatible — silently declines (no loud failure)**
     - Trace-following hooks run (``angr-rqvq``) and proxy writes now
       write through (the read-only-proxy blocker is gone, write-through
       epic ``angr-j28e`` landed). ``Tracer.step_state()``
       (``tracer.py``) *is* dispatched by
       ``dispatch_step_state_with_hooks``, but it delegates the actual
       step to ``simgr.step_state(...)``, which the dispatcher has
       swapped for a declined-sentinel base — so the technique's
       partial side effects run (``self.predecessors``,
       ``state.globals``, ``RepHook`` installation) and then the state
       is advanced by a plain native ``run()`` with **no trace
       enforcement**. If the hook raises instead, it is caught and
       logged (``l.warning("step_state() hook raised ...; leaving it in
       place")``) and the state is likewise advanced natively. Either
       way the failure mode is *silent divergence*, not a
       ``NotImplementedError`` — do not rely on a hard failure. Use
       ``use_rust_engine=False`` for trace replay.
   * - ``Director``
     - **Step hook dispatched, stash assignment honored**
     - Goal-prioritisation ``step()`` runs (``angr-rqvq``) and the
       technique's ``simgr.stashes[stash] = [...]`` re-ordering /
       limiting is now applied by the proxy's ``_StashDict``
       (``angr-wxuo``) via Rust ``move_state``. Assignments of foreign
       or non-proxy states raise.
   * - ``Slicecutor``
     - **Silently no-op (declines / undispatched successors)**
     - ``filter()`` runs and ``step()`` would dispatch. Its
       ``step_state()`` *is* dispatched, but its first act is to call
       ``simgr.step_state(state)`` — the declined-sentinel base — so the
       hook always declines and the slice enforcement below that call
       never runs; the state is advanced natively, unpruned. Successor
       pruning in ``successors()`` is not dispatched at all (the proxy
       method raises ``NotImplementedError`` if called).
   * - ``DrillerCore``
     - **Step hook dispatched**
     - Drilling ``step()`` runs (``angr-rqvq``). Driller handoff and
       ``setup()`` hook installation both fire; behaviour is now bounded
       by whatever ``RustStateProxy`` exposes to driller's per-state
       inspection.
   * - ``ManualMergepoint``
     - **Native (angr-op0dn.11.5)**
     - ``register_technique`` forwards ``address`` /
       ``wait_counter_limit`` to ``register_merge_point(...)``
       (``rust_techniques.py``); the native MergePoint technique parks
       states reaching the address, groups waiters by callstack, and
       merges each ≥2 group entirely in Rust. Because the class is in
       ``_NATIVE_STEP_TECH_NAMES`` its Python ``step()`` is suppressed,
       so the two implementations never both run and the technique's
       ``SimStateHistory`` mutation is never reached. If native
       registration fails (non-``int`` address, or the call raises),
       there is no Python fallback — merging silently does not occur and
       a ``logging.WARNING`` is emitted.
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
``step`` / ``step_state`` / ``complete`` over ``RustStateProxy``;
``successors`` hooks are never dispatched (the method on
:class:`RustSimulationManagerProxy` raises ``NotImplementedError``), and
a ``step_state`` hook that delegates to ``simgr.step_state(...)``
*declines* — the state is then advanced by a native ``run()`` without
the technique's post-step logic (see "Dispatch coverage" above). The
accept-everything default keeps construction non-fatal, but techniques
that read internal ``SimState`` plugins beyond the proxy's contract
(``state.history.parent``, ``state.history.events``,
``state.solver.constraints`` mutation) will silently misbehave rather
than raise — vet each one against the proxy read-only invariant before
counting on it in CI. ``state.copy()`` is supported (``angr-d1dr``)
— it does a Rust-side CoW deep fork and parks the copy in a dedicated
``_copies`` stash (see the Spiller / ManualMergepoint rows above).

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
* **The project default is explicit.** ``angr.Project(engine="rust")``
  (or ``ANGR_DEFAULT_ENGINE=rust``) routes ``simulation_manager()``
  calls that do not name an engine through the eligibility predicate
  — see *Auto dispatch*. Merely attaching a ``RustExplorationManager``
  to a project changes nothing for later ``factory.simulation_manager()``
  calls.
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
       jump resolution (``cfg_emulated.py``). Has no ``use_rust_engine`` plumbing — those
       internal SMs always dispatch to the Python engine, so a
       ``RustExplorationManager`` attached to the project is unused.
       Beyond the SM bypass, the analysis also reads
       ``state.scratch.ins_addr``, ``state.scratch.exit_stmt_idx``
       and ``state.scratch.exit_ins_addr`` off successor states
       (``cfg_emulated.py``), which is
       deep ``SimState`` machinery not modelled on
       :class:`RustStateProxy`. Workaround: feed it a fresh
       :class:`SimState` from a project that has *not* had a
       ``RustExplorationManager`` attached. No Rust-parity smoke
       test exists yet.
   * - ``Identifier``
     - **Unsupported (v1.0)**
     - Both the function-identification driver
       (``identifier/identify.py``) and the per-call replay path
       (``identifier/runner.py``) build internal
       ``SimulationManager`` instances via
       ``project.factory.simulation_manager(...)``; the symbolic
       pre-amble walk uses ``project.factory.successors(...)``
       repeatedly (``identifier/identify.py``) — both dispatch to
       the Python engine. The analysis also resets and inspects
       ``state.scratch`` directly (``runner.py``), which is deep
       ``SimState`` machinery not modelled on
       :class:`RustStateProxy`. Workaround: feed it a project that
       has *not* had a ``RustExplorationManager`` attached.
   * - ``Jumptable`` resolver
     - **Unsupported (v1.0)**
     - The ``JumpTableResolver`` (registered as a default
       ``IndirectJumpResolver`` for x86/AMD64/ARM, used by
       ``CFGFast`` / ``CFGEmulated`` during indirect-jump
       discovery) builds an internal ``SimulationManager`` at
       ``jumptable.py`` with ``resilience=True`` and steps via
       ``project.factory.successors(...)``.
       Both paths dispatch to the Python engine regardless of any
       ``RustExplorationManager`` attached to the project. The
       resolver also reads ``state.scratch.temps[...]`` directly to
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
       (``veritesting.py``) against the Python engine, so a
       ``RustExplorationManager`` attached to the project has no
       effect on the analysis itself. Two distinct failure modes:

       1. Passing a :class:`RustStateProxy` as ``input_state`` raises
          ``NotImplementedError`` at the first ``input_state.copy()``
          call inside ``veritesting.py``. ``RustStateProxy.copy()``
          refuses the operation rather than returning a shallow proxy
          that aliases ``_state_id`` with the source (the previous
          behaviour silently corrupted the parent on any mutation). A
          Rust-side CoW deep fork is tracked under ``angr-2zwy``.
       2. Veritesting needs ``EFFICIENT_STATE_MERGING`` for ancestor
          retention during plugin merging. The Veritesting
          *exploration technique*
          (``exploration_techniques/veritesting.py``) auto-adds
          this option at ``step_state`` time; the *analysis* does not.
          The option was demoted out of ``_RAISE_OPTION_NAMES``
          (``angr-op0dn.11.6``) so seeding a Rust manager with a state
          carrying it no longer raises — but the analysis still runs on
          the Python engine, and without the option it falls back to
          weak-ref merging and may miss common ancestors: a silent
          correctness gap.

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
  at the push sites in ``exploration/run_loop_single.rs`` to bound the
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
  test hook (the ``oom`` arm of
  ``engine.rs::_raise_typed_test_error``); real Rust allocation
  failures abort rather
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

Most of the "real binary problem" here is *harnessing*, not engine
capability. The end-to-end recipe stacks four knobs that each suppress a
distinct source of state explosion. Full recipe::

    import claripy
    import angr.sim_options as o
    from angr.exploration import RustExplorationManager
    from angr.exploration_techniques import LengthLimiter

    MAX_INPUT = 32

    # (4) Bound the symbolic input length, and zero-fill unconstrained
    #     reads so stubbed-libc / uninitialised reads don't seed fresh
    #     symbolic branches.
    stdin = claripy.BVS("stdin", 8 * MAX_INPUT)
    state = project.factory.entry_state(
        stdin=stdin,
        add_options={
            o.ZERO_FILL_UNCONSTRAINED_MEMORY,
            o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
        },
    )

    mgr = RustExplorationManager(
        project, [state],
        exploration_strategy="dfs",     # (1) LIFO state selection
        max_active_states=64,           # belt-and-braces fork cap
    )
    # (2) Drop states past the expected vulnerability depth.
    mgr.use_technique(LengthLimiter(max_length=MAX_INPUT, drop=True))
    # (3) Loosen the uniqueness filter so a BFS-style active=0 premature
    #     termination doesn't prune deep-but-similar states. The filter is
    #     OFF by default; call this only if a prior step enabled it.
    mgr.disable_uniqueness_filter()
    mgr.explore(find=vuln_addr, avoid=avoid_addrs)

Each knob and its native FFI entry point:

#. ``exploration_strategy="dfs"`` (FFI: ``set_state_selection_lifo``) —
   depth-first frontier so a single path drives toward the target depth
   instead of the BFS frontier ballooning at shallow depth.
#. ``LengthLimiter(drop=True)`` (FFI: ``register_length_limiter``) —
   drop states once they pass the expected vulnerability depth so the
   active stash stays bounded.
#. The uniqueness filter (FFI: ``register_uniqueness_filter`` /
   ``disable_uniqueness_filter`` / ``uniqueness_filter_enabled`` /
   ``uniqueness_set_size``, exposed on the Python manager via T1a) — it
   is **off by default**; if an earlier configuration step turned it on,
   loosen or disable it here so near-duplicate deep states aren't pruned
   into a premature ``active=0`` termination.
#. ``ZERO_FILL_UNCONSTRAINED_MEMORY`` /
   ``ZERO_FILL_UNCONSTRAINED_REGISTERS`` plus a bounded symbolic
   ``stdin`` — cap the symbolic input and stop stubbed-libc return values
   from spawning extra symbolic branches.

All four compose without overriding each other. See
``TestExplorationStrategy.test_deep_loop_recipe_dfs_plus_length_limiter``
(DFS + ``LengthLimiter``) and
``TestExplorationStrategy.test_deep_loop_recipe_full_knob_stack`` (the
full four-knob stack) in ``tests/engines/rust/`` for the regressions that
pin this composition.

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
``native/angr/src/errors.rs`` (the ``RustExecError`` enum and the
``From<RustExecError> for PyErr`` dispatch impl below it) — if a future
commit grows or renames a variant, this table is the first thing to
update.

.. warning::

   **The typed taxonomy is currently a test-harness affordance, not the
   live-exploration contract** (angr-ghwsd.3). The variant→subclass
   mapping (``From<RustExecError> for PyErr``) is reached only from the
   test-only ``#[pyfunction]`` hooks ``execute_irsb_for_test`` and
   ``_raise_typed_test_error``. During a real ``mgr.run()`` / ``explore``
   the typed ``CbExecutionError`` is **stringified at the interpreter
   boundary** — ``RunResult::Error { message, addr }`` carries a
   ``String`` that ``stepping`` collapses into
   ``StepError::Error(state, String)`` and
   ``run_loop_single.rs::apply_terminal`` pushes into the
   errored stash as a ``(pc, message, state_id)`` record. The concrete
   variant is lost the moment a live step fails; only the formatted
   message survives, readable off the errored-stash record
   (``state.error``). A ``pytest.raises(RustUnsupportedVexOpError)``
   against a live run therefore will **not** match — drive
   ``execute_irsb_for_test`` to exercise the typed classes. Promoting the
   typed error into the live path is the deferred option (b) on
   angr-ghwsd.3. The "Where Rust raises it" column below names where each
   variant is *constructed*, which is the test path unless noted.

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
     - ``errors.rs::RustExecError::Other``; populated via the catch-all
       arms in ``engine.rs::cb_execution_error_to_typed`` /
       ``op_error_to_typed``.
   * - ``RustMalformedIRSBError``
     - The pyvex lifter produced an IRSB the interpreter could not
       execute (bad/missing statements, malformed exits, invalid
       block bounds).
     - ``errors.rs::RustExecError::MalformedIRSB``; raised from
       ``CbExecutionError::InvalidIR(reason)`` mapped at the
       ``InvalidIR`` arm of ``engine.rs::cb_execution_error_to_typed``.
   * - ``RustUnsupportedSyscallError``
     - A syscall handler that ran in Rust hit a number / name / arch
       combination it does not implement. Reserved class — the
       syscall fast path currently routes unimplemented numbers
       through the Python fallback (``UnsupportedFeature``), so this
       class is exposed for forward-compatibility and exercised by
       the ``_raise_typed_test_error("unsupported_syscall", ...)``
       hook (the ``unsupported_syscall`` arm of
       ``engine.rs::_raise_typed_test_error``). No production trigger
       in the current code.
     - ``errors.rs::RustExecError::UnsupportedSyscall``; production
       sites land here when an upcoming Rust syscall handler chooses
       to raise rather than fall back to Python.
   * - ``RustUnsupportedVexOpError``
     - A VEX op (NEON / vector / unmapped opcode) is not implemented
       by the Rust interpreter. The op name and arch are baked into
       the message.
     - ``errors.rs::RustExecError::UnsupportedVexOp``; populated from
       ``OpError::UnsupportedNeon``, ``OpError::UnsupportedVectorOp``,
       ``OpError::UnsupportedVexOp`` at the three matching arms of
       ``engine.rs::op_error_to_typed``.
   * - ``RustZ3Error``
     - Z3 returned an error status (not ``Unknown`` — that collapses
       to UNSAT inside ``SymContext::is_sat``). Reserved class —
       solver hangs are bounded by ``solver_timeout_ms`` and the
       ``Unknown`` collapse, so this class is exposed for
       forward-compatibility and exercised only via the
       ``_raise_typed_test_error("z3", ...)`` hook (the ``z3`` arm of
       ``engine.rs::_raise_typed_test_error``).
     - ``errors.rs::RustExecError::Z3``.
   * - ``RustOomError``
     - Rust allocator returned a failure that the engine can
       propagate (as opposed to ``alloc::handle_alloc_error`` aborting
       the process). Reserved class — see *Behavior on exhaustion*
       above: today, real Rust allocation failures abort under the
       release profile rather than raising this. Production sites
       land here when a Rust allocator hook chooses to propagate
       instead of abort. Currently only the
       ``_raise_typed_test_error("oom", ...)`` hook (the ``oom`` arm of
       ``engine.rs::_raise_typed_test_error``) raises it.
     - ``errors.rs::RustExecError::Oom``.

The ``RustExecError`` Rust enum carries ``#[non_exhaustive]``
(``native/angr/src/errors.rs``) so new variants can land in minor versions
without breaking downstream code that matches on it.

Internal error convention
~~~~~~~~~~~~~~~~~~~~~~~~~

The typed exception classes above are the *user-facing* taxonomy and,
as the warning notes, are reached only from the test-only hooks. Below
the PyO3 boundary the crate follows one convention, documented in full
in the module doc of ``native/angr/src/errors.rs``: ``PyResult`` lives
strictly at the boundary, internal functions return a subsystem-local
``thiserror`` enum (``CbExecutionError``, ``StepError``, ``MemoryError``,
``OpError``, ``ProcedureError``, ``SyscallError``, ``BridgeError``,
``LiftError``, …) composed upward with ``#[from]``, and errors collapse
to a ``String`` or a ``PyErr`` only at the two sanctioned points — the
errored-stash record and the ``From<RustExecError> for PyErr`` map.

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
``RustExplorationManager._check_raise_options``; the rationale for each
entry is inline above the set. The same module holds the two sibling
ledgers: ``_REJECTED_OPTION_NAMES`` (warn once instead of raising) and
``_REQUIRED_OPTION_NAMES`` (the inverse-polarity gate — an option
refused by its *absence*). The *Ignored — divergence-risk* rows of the
SimOption coverage matrix explain why each raise-listed option was
promoted to raise rather than warn.

Unsupported ``state.inspect`` registration also raises
``NotImplementedError`` (``RustInspectProxy._check_event``,
``rust_state_proxy.py``); see the *state.inspect dispatch* section
for the supported/unsupported event split.

``ExplorationTechnique`` registration itself never raises — every
technique is accepted, tracked in ``mgr._active_techniques``, and
dispatched over ``setup`` / ``filter`` / ``step`` / ``step_state`` /
``complete``. What raises is the undispatched ``successors`` hook
(``RustSimulationManagerProxy.successors``) plus any proxy read/write
outside the ``RustStateProxy`` contract; techniques whose logic lives
past a ``simgr.step_state(...)`` delegation instead *silently decline*.
See *Exploration technique compatibility* for the per-technique verdict.

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

**The guest environment is bridged, with two narrow gaps.** angr models
the initial environment purely in memory --- ``entry_state(env={...})``
becomes a ``KEY=VALUE`` string table on the stack plus the ``envp`` array
pointer in ``state.posix.environ``, which the Python ``getenv``
SimProcedure walks. The Rust engine instead keeps a byte-keyed map on
``RustSimState`` that the native ``getenv``/``setenv``/``putenv``/
``unsetenv`` procedures read. ``RustExplorationManager`` walks the ``envp``
array at state-add time and seeds the map from it, so a harness-supplied
var is visible to a native ``getenv``. Two differences remain: an entry
whose bytes are **symbolic** is skipped by the bridge (the native
``getenv`` returns NULL where Python would return a pointer to it), and the
map is write-only with respect to memory --- a guest that walks
``environ``/``__environ`` itself instead of calling ``getenv`` does not
observe a native ``setenv``/``putenv``/``unsetenv``.

**Standard ``except SimError`` does not catch Rust errors.** By the
same mechanism, downstream code that ``except angr.errors.SimError:``
around ``mgr.run(...)`` will not catch the typed Rust exceptions.
Migration recipe: change the clause to
``except (angr.errors.SimError, angr.exploration.RustExecutionError):``,
or catch ``Exception`` if the surrounding code already does broad
recovery.

Native procedures for library hooks (opt-in)
--------------------------------------------

By default the native SimProcedure registry only serves hooks whose address
lies **outside every loaded object** — the extern-object stubs angr synthesizes
for a statically-known symbol. On a dynamically-linked binary loaded with
``auto_load_libs=True, use_sim_procedures=True``, each libc hook instead lands
inside the loaded libc's ``.text``, so every one of them bounces to the Python
SimProcedure.

``prefer_native_library_hooks`` (**on by default**; set it to ``False``, or
export ``ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS=0``, to turn it off) lets the
native registry serve those library hooks. Hooks inside the **main object**
still go to Python, so a user ``proj.hook()`` override always wins. The
``explore(find=...)`` target is always forced to Python so the find
fires. On a concrete ``strcmp``/``strlen``/``memcpy`` loop against a
dynamically-linked libc the win is large, because every one of those calls
stays in Rust instead of bouncing across the FFI boundary.

The flag was off by default until the native string procedures matched Python's
symbolic-scan bound: they used to scan up to 4096 bytes where Python stops at
``buf_symbolic_bytes``, so on symbolic input they forked far more aggressively
and could storm a libc-heavy dynamic binary. ``MAX_SYMBOLIC_SCAN_BYTES`` in
``procedures::strings`` now caps the symbolic window at Python's. Two
residual divergences remain, both in the *conservative* direction (Python swaps
in a fresh BVS past a recursion depth of 3, and searches a narrower concrete
window), so the native procs cannot wrongly kill a state that Python would keep.
Set the env var to ``0`` if you need the old all-Python dispatch for a
bug-for-bug comparison.

Benchmarks slower than Python
-----------------------------

``tests/benchmarks/baseline_timings.json`` is the live record of every
tracked benchmark's Python and Rust timings; its ``python_time /
rust_time`` ratio is the only number worth quoting, and it is refreshed
by ``tests/benchmarks/run_regression.py --update`` (see :doc:`rust_build`).
The benches that stay below 1.0x fall into two classes:

* **Z3-bound benches.** Wall time is dominated by ``solver.check()`` /
  ``eval_upto`` rather than by the engine, and several of them are
  *bimodal*: Z3's heuristic latitude picks a fast or a slow branch per
  run. Their membership is ``BIMODAL_BENCHMARKS`` in ``run_regression.py``
  and their handling is in :doc:`rust_bimodal_variance`. No engine-side
  lever exists for this class (:doc:`rust_engine_decisions`); the
  ``ANGR_Z3_TACTIC`` knob above is the only one that moves them.
* **Callback-heavy benches.** Every Python SimProcedure, hook or syscall
  fallback crosses the FFI boundary and re-enters Python's state
  machinery. The levers are the native registries
  (:doc:`/extending-angr/native_coverage_matrix`),
  ``prefer_native_library_hooks`` above, and the per-bench counter dump
  (``run_single.py --dump-counters`` / ``bench_diff.py``) that attributes
  the bounces by name.

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

Audit scope: every
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
``find_state_mut`` (``exploration/helpers.rs``). Misses become one
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

``PythonCallbacks`` (``callbacks/mod.rs``) holds 30 ``Option<Py<PyAny>>``
slots, one per dispatch site — the ``with_callback_fields!`` macro is
the single source of truth for the list: 12 non-inspect slots (memory
load/store, batched load/store, lift_block, dirty_call, page fetch,
symbolic load/store, resolve_function) plus 18 ``state.inspect``
slots. ``Py<PyAny>`` is an owning Python refcount, so a callback
cannot dangle even if Python "drops" it locally — the engine's clone
keeps it alive.

* **Errors propagate**. Every dispatch site reads
  ``self.<slot>.as_ref().ok_or_else(|| PyRuntimeError::new_err("…
  callback not set"))?`` and then ``cb.call1(py, args)?``. A Python
  ``raise`` becomes a ``PyErr`` that the engine returns up through
  the exploration loop (``run_loop.rs``).

  **Exception** — the 18 ``call_inspect_*`` methods. They route through
  the ``with_inspect_cb`` helper rather than ``require_callback!``, and
  return ``Ok(())`` / ``Ok(None)`` when their slot is unset — an
  unregistered ``state.inspect`` breakpoint is the normal case, not an
  error. A ``raise`` from a breakpoint that *is* registered does become
  a ``PyErr``, but no dispatch site propagates it with ``?``: every one
  funnels it through ``callbacks/inspect.rs::note_inspect_error``,
  which logs it (``SILENT(cat-b)``, visible at ``debug``) and lets
  execution continue, so a buggy breakpoint cannot derail a run.
* **Type/arity mismatches propagate**. Result destructuring uses
  ``.cast_bound::<PyTuple>(py)?.get_item(N)?.extract()?`` — a 1-tuple
  returned for a 3-tuple slot raises ``IndexError``; a wrong-type
  field raises ``TypeError``. The engine surfaces these as
  ``PyErr`` rather than misinterpreting bytes.
* **GC cycle break**. ``PythonCallbacks::__traverse__`` /
  ``__clear__`` (``callbacks/mod.rs``) walk every ``Py<PyAny>`` slot
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

``RustBVHandle`` (``symbolic/handle.rs``, ``struct RustBVHandle``) is a
pure value type:
``id: u64``, ``width: u32``, ``concrete: Option<u128>``. It carries
**no raw Rust pointer** — Python only gets back a numeric handle.
The class is ``#[derive(Clone)]``; ``__eq__`` and ``__hash__`` are
``id``-based, so a forged handle compares equal to a real one with
the same ID. The symbol-table lookup that resolves the handle to a
``RustBV`` (``RustSymbolTable``) returns ``Option<…>`` on miss — no
UB risk from forged or stale IDs. Note that ids are unique only within
one table lineage: ``RustSymbolTable::fork`` continues the child's
counter from the parent's rather than partitioning the id space, so two
*sibling* tables (the tables of two states forked from the same parent)
can bind the same id to unrelated values. Comparing or dict-keying
handles drawn from sibling-forked states is therefore unsound.

Claripy AST inputs (``add_constraint_ast``, ``eval``,
``import_symbolic_memory``, etc.) flow through ``claripy_to_rustbv``
(``claripy_bridge/import.rs``). Any leaf type the bridge doesn't recognize
raises ``BridgeError`` → ``PyRuntimeError``. No malformed-AST input
reaches the Z3 layer un-converted.

Raw Z3 pointer fast path (the **only** unsafe FFI surface)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The audit found one place where a Python caller's input is treated as
a raw pointer with no validation:

* ``RustExplorationManager.import_z3_constraint_ptrs(state_id, ptrs:
  Vec<usize>)`` (``exploration/state_api.rs::_import_z3_constraint_ptrs``)
  iterates ``ptrs``
  and calls ``unsafe { ctx.add_constraint_raw(*ptr) }`` whenever
  ``*ptr != 0``. The ``add_constraint_raw`` SAFETY contract
  (``symbolic/constraint_ops.rs::add_constraint_raw``) requires each
  pointer to be a valid,
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
  claripy's z3 backend (``solver/z3_ast_extract.rs``) and so are safe by
  construction.

  **Hardening (angr-33t9).**
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
(the public pymethod in ``exploration/manager_methods_state.rs``, body in
``exploration/state_lifecycle.rs::_add_state``) forks the incoming state via
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

.. _rust-engine-init-prefix:

The Python init prefix and ``find`` / ``avoid``
-----------------------------------------------

A state constructed at the entry point (``entry_state()``,
``full_init_state()``) is stepped to ``main`` **in Python** before Rust
ever sees it — ``rust_manager.py::_run_python_init_if_needed``. The
loader stubs, ``__libc_csu_init``, ``_init``, ``frame_dummy`` and the
``.init_array`` constructors are too much for the Rust engine to execute
faithfully, so they run on the Python engine and only the resulting
state at ``main`` is handed over (usually straight out of the in-memory
or on-disk init cache, without stepping at all).

The consequence for exploration: **no address the init prefix traverses
is ever a PC that Rust's pre-step ``find_addrs`` / ``avoid_addrs`` check
sees.** Vanilla angr's ``Explorer`` matches those addresses during the
first handful of steps. Two mechanisms restore that behavior for
``find``:

* The un-advanced seed is parked, so ``explore(find=proj.entry)`` reports
  it as FOUND unstepped and drops its fast-forwarded counterpart from
  ``active`` — exactly what ``Explorer.filter()`` does (angr-lyvf2).
* A find address strictly *inside* the prefix (e.g. inside
  ``__libc_csu_init``) is handled by
  ``rust_manager.py::_replay_preinit_prefix_for_find`` (angr-bdeqa): if
  the Rust exploration finishes with an empty found stash, the prefix is
  replayed in Python from the parked seed and the real mid-init state at
  the target address is pushed to FOUND. The replay is lazy — it costs
  nothing on the (overwhelmingly common) explorations that find their
  target past ``main``, and its budget is the same 500 steps the init
  skip itself uses.

**Known divergence — ``avoid`` inside the init prefix.** An ``avoid``
address that only the init prefix executes does not kill the seed. Rust
receives a state that is already past ``main``, so the exploration
proceeds and may report a find that vanilla angr would not (vanilla
would move the state to ``avoided`` during init and end with nothing
active). ``avoid`` *is* honored inside the ``find`` replay above — a
prefix path that hits an avoid address before the find address does not
match — but it is not applied retroactively to states Rust already
explored. Checking it eagerly would mean replaying the init prefix on
every ``explore()`` call, paying the init cost twice for a case that has
never shown up in practice. If you need it, avoid on a **callable**
predicate over ``state.history`` instead, or run the Python engine.

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

On-disk envelope is ``[STASH_SNAPSHOT_VERSION: u8] ++
[process_token: u64 little-endian] ++ serde_json(...)`` — the JSON body
starts at byte 9, not byte 1. The 8-byte process token rides in the
header rather than the body so that ``load_snapshot`` knows whether the
symbol-id space is foreign *before* it deserializes anything — see the
cross-process paragraph at the end of this section, and
``StashManager::dump_snapshot`` in ``native/angr/src/stash.rs``.
A stale version byte fails the load fast with ``ValueError``
(``"version mismatch ..."``); an envelope shorter than the 9-byte header
fails with ``ValueError`` (``"truncated stash snapshot header"``); an
empty file fails with ``ValueError`` (``"empty snapshot envelope"``). The
``STASH_SNAPSHOT_VERSION`` constant lives at
``native/angr/src/stash.rs`` and bumps on any breaking shape change
to the per-state codec (``RustSimStateSnapshot`` at
``native/angr/src/state/snapshot.rs``).

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

Parallel exploration (``RUST_PARALLEL_WORKERS``) interacts with the
capture in two ways:

* **Live steady sessions are finalized, never truncated.** Under the
  steady-state loop the search frontier is resident inside the worker
  Z3 contexts and belongs to no stash, so a naive dump would silently
  omit it. ``dump_snapshot`` drains the frontier back into ``active``
  before capturing (Python ``_finalize_parallel_session``, plus the
  Rust-side ``steady_config_guard`` inside ``dump_snapshot_bytes``),
  so the snapshot always reflects the whole frontier.
* **Known gap (angr-op0dn.13.10)** — a snapshot taken *after* a
  parallel explore round-trips structurally (per-stash counts and the
  active-stash PCs match exactly) but the restored states can lose
  reachability: on the 8-leaf synthetic, resuming from such a snapshot
  reaches 5-6 leaves where resuming the same in-memory manager reaches
  all 8. Serial (``workers=1``) snapshots resume fully. Treat
  post-parallel snapshots as a checkpoint of the frontier's shape, not
  yet of its full search power.

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

Snapshots written by **another process** are supported (angr-euw28).
Symbol ids are process-global, so a foreign envelope's ids were
minted by an allocator that also started at 0 and would alias ids the
loading process already handed to its own states. The envelope header
therefore carries a process token; on a token mismatch ``load_snapshot``
shifts the whole envelope's ids above the local watermark before
deserializing. Names — and so Z3 identity — are untouched, so replayed
constraints still bind to the same variables.

When inspecting a restored state, read its constraints through
``mgr._rust_mgr.export_state_constraints(state_id)`` (or a
:class:`RustStateProxy`), **not** through the claripy list on an
exported ``SimState`` mirror. A mirror materialized via the
full-export path is built from ``project.factory.blank_state()`` and
never receives a claripy copy of the Rust constraint set: eval / min /
max / satisfiable are routed into the Rust solver by
``_attach_rust_solver_fallback`` instead, because re-importing the
constraints into claripy is expensive and identity-lossy. So an empty
``mirror.solver.constraints`` after a load is the design, not a lost
constraint set (angr-wuyo9).

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
- ``BridgeError`` (``native/angr/src/claripy_bridge/mod.rs``)
- ``SyscallError`` (``native/angr/src/syscalls/support.rs``)
- ``OpError`` (``native/angr/src/vex/ops/error.rs``)
- ``ProcedureError`` (``native/angr/src/procedures/mod.rs``)
- ``MemoryError`` (``native/angr/src/memory/mod.rs``)
- ``CallbackReason`` (``native/angr/src/exploration/callback_types.rs``)
- ``ExplorationEvent`` pyclass struct
  (``native/angr/src/exploration/event.rs``) — the Python-visible event
  envelope, expected to grow ``#[pyo3(get)]`` fields as new
  callback reasons land.
- ``ExecutionConfig`` pyclass struct (``native/angr/src/callbacks/config.rs``)
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
