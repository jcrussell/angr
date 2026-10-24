Rust engine: decisions record
=============================

Alternatives that were tried, measured and rejected while building the
Rust engine, kept so they are not re-tried without new evidence. Each
entry cites the tracker issue that holds the data; the numbers themselves are
deliberately not repeated here, because they describe one host on one
day. Design notes that are still live are in :doc:`rust_z3_sharing`,
:doc:`rust_parallel_design`, :doc:`rust_lazy_memory_design`,
:doc:`rust_proxy_writes_design` and :doc:`rust_libvex_ffi`.

Solver
------

**Shared-lineage Z3 solver stays opt-in** (angr-v5a5, angr-3ms1, angr-v5ht, angr-0dgq, angr-op0dn.9.3).
Sharing one Z3 solver across a forked lineage, switching scopes with
push/pop instead of materializing a solver per fork, was expected to recover the
materialize cost. Under BFS exploration the siblings interleave, so the
switch cache stays cold and the ``switch_to`` push/pop plus the
sat/model-cache invalidation it forces cost more than the materialize
they avoid; the win appears only on fork-dense workloads with a deep
shared constraint prefix. The fuller variants (runtime teardown of the
whole lineage, a thrash-detecting sampler as a default gate) were
reverted because the sampler's early hot-ratio estimate mispredicts
the steady state on exactly the workloads the feature helps. The simple
variant ships behind ``use_shared_lineage_solver`` and the kwarg stays
default-off. Full data in ``tools/decisions/solver_pool_design.md``.

**Z3 ``parallel.enable`` is never set** (``angr-gfay``). Turning on Z3's
own parallel mode produced wrong answers, not just no speedup: several
benches raised ``IndexError`` / ``KeyError`` / ``AngrCallableError`` from
Z3-Unknown results no consumer was prepared for. Re-enabling it would
mean auditing every solver consumer for ``Unknown`` handling first.

**Pinning Z3's random seeds does not give model stability** (``angr-iaol.1``).
Setting ``smt.random_seed`` / ``sat.random_seed``
through ``Z3_solver_set_params`` corrupts models (the solver returns
witnesses that violate asserted constraints); the unprefixed
``random_seed`` is accepted but produces *more* variation across fresh
solver instances than no pin. Z3 reserves variable- and restart-ordering
latitude that these keys do not bind. The supported determinism knob is
``RustExplorationManager(deterministic=True)``, documented in
:doc:`rust_engine`.

**``sat-preprocess`` tactic pipelines are unsafe** (``angr-ya00``). An
``ANGR_Z3_TACTIC`` pipeline beginning with ``sat-preprocess``
(``sat-preprocess:qfbv`` was the one tried) produced incorrect results on
``csgames2018`` and ``securityfest_fairlight``, surfacing as
``IndexError: list index out of range`` from the post-solve goal unwrap;
``qfbv`` alone is the only non-default tactic that has been validated.

**Z3-bound benchmarks are a solver floor, not engine work**
(``angr-9w6ad``, ``angr-34w``). The benches that stay slower than Python
spend their wall time in ``solver.check()`` / ``eval_upto``; two
independent spikes (deterministic seeding, a lazy-solving engagement
audit) found no engine-side lever. They are tracked as bimodal
(:doc:`rust_bimodal_variance`) and are not re-triaged as optimization
candidates.

Memory and allocation
---------------------

**No non-system global allocator** (``angr-kq43``). mimalloc was wired in
as ``#[global_allocator]`` and benchmarked head-to-head; every delta was
inside run-to-run noise, including on the slow callback-heavy outlier.
Wall time is dominated by Z3 ``check()`` and PyO3 state-sync FFI, not by
malloc throughput. Re-evaluate only if a profile shows allocation as a
hot path.

**Lazy memory shipped as a sidecar, not a ``Multi`` byte-cell variant**
(:doc:`rust_lazy_memory_design`). The design memo recommended a new
page-cell variant; what shipped keeps each page's byte array concrete
and stores the alternatives in a ``multi_objects`` sidecar on
``SymbolicMemory``, with only a per-page ``multi_bitmap`` (parallel to
``symbolic_bitmap`` on ``MemoryPage`` in ``memory/page.rs``) marking which
bytes are Multi. That left every concrete fast path byte-identical and
merge semantics expressible per field.

Parallelism
-----------

**Snapshots migrate, states do not** (``angr-8fo6``, Option A in
:doc:`rust_parallel_design`). Making ``RustExplorationManager``
``Send`` by converting ``Rc<RefCell<SymContext>>`` to
``Arc<Mutex<..>>`` does not unlock parallelism: z3-rs binds ASTs to the
thread-local context that created them, so a state's assertions cannot
cross threads anyway, and the constraint hot path would take a heavily
contended lock. Workers instead exchange ``ExplorationStateSnapshot``
values and rebuild thread-local solvers.

**Distributed (cross-machine) migration was killed**
(``angr-op0dn.13.7``; :doc:`rust_parallel_design`, "Distributed
(cross-machine) migration — measured KILL"). The probe was a cost
estimate only; no code landed. Migration count, not transport cost, is
the lever that decides whether parallel exploration wins at all
(:doc:`rust_parallel_design`, "Where it landed"), and every term of a
network transport moves it the wrong way:

* **The envelope is a lower bound.** The per-state cost was read from
  the shipped ``dump_snapshot`` / ``load_snapshot`` codec — the same
  ``StashManager`` per-state encoding a ``StateMigrationPayload``
  carries — at the exhaustive frontier of the synthetic find-all
  benches. Against the break-even steal-fraction frame, a network hop
  shrinks the fraction of states allowed to migrate to a sliver of the
  frontier, while the steal fraction actually observed on those benches
  is most of it.
* **The ``Py<PyAny>`` overlays are not in that envelope.** A
  ``StateMigrationPayload`` also carries ``symbolic_pages``,
  ``hook_symbolic_memory`` and ``addr_to_ast``; a distributed transport
  would additionally have to pickle them.
* **Python SimProcedure bounces are driver-local.** A remote worker
  would round-trip to the driver on every hook, and the trap benches
  bounce the whole frontier several times per leaf by construction.
* **The binding constraint already fails in-process.** The literal gate
  ("network cost < work per state") passes only because per-state work
  on those benches is large; the migration count is already over
  break-even between threads, so a slower hop cannot win it back.

State proxy
-----------

**Diff-and-push (Option C) kept at design time; write-through shipped
anyway** (:doc:`rust_proxy_writes_design`). The memo chose to keep the
read-mostly proxy with an explicit diff-and-push at callback exit
(Option C, the status quo) over proxy-backed plugins (Option A) and a
tracked-writes log (Option B). Write-through to the Rust state then
shipped for ``RustStateProxy`` register and memory writes, and the
measured cost of re-syncing whole pages per callback moved callback
memory onto the proxy by default; registers and constraints keep
diff-and-push at the callback boundary. The write-through contract is
in :doc:`rust_engine`.

Engine selection
----------------

**One knob, ``engine="rust"``** (``angr-mx00p.62``). The engine used to be
selectable four ways: a tri-state ``use_rust_engine=True/False/None`` on
``simulation_manager()``, ``Project(engine="rust")``,
``ANGR_DEFAULT_ENGINE=rust``, and an auto-dispatch switch
(``angr.exploration.set_rust_auto_dispatch()`` / ``ANGR_RUST_AUTO``) that
only gated ``use_rust_engine=None``. The last and the project setting ran
the same eligibility predicate, so they were two doors onto one heuristic.
What remains is ``engine="rust"``: per call it demands the Rust engine and
raises, on a project it routes through the predicate, and
``ANGR_DEFAULT_ENGINE=rust`` is only the process-wide default of the project
setting. A loud project setting was rejected: angr's own analyses build
managers without naming an engine (the jump-table resolver passes
``resilience=True``), so ``CFGFast`` would raise on every opted-in project.
The removed names never shipped in a released angr, so they were dropped
without shims, but none is ignored silently: ``use_rust_engine=`` is an
unknown keyword (``TypeError``), and building a project while
``ANGR_RUST_AUTO`` is set raises ``AngrError``. A warning was rejected for
the variable because a CI job that sets it would scroll past the warning and
pass on the Python engine. The routing decision is logged rather than stored
as ``mgr.dispatch_reason``, so a plain ``SimulationManager`` carries no
fork-only attribute.

Lifting
-------

**Link-time libVEX bindings replaced by runtime resolution** (``angr-z087y``, ``angr-mx00p.33``).
The native lifter first bound
libVEX at build time: a ``libvex-ffi`` cargo feature ran ``bindgen`` over a
cffi cdef vendored from pyvex, linked ``-lpyvex`` with an rpath into the
venv, and generated the enum-name tables from the same header. Every pyvex
bump meant regenerating the header, the extension carried a ``NEEDED
libpyvex`` that the wheel repair had to exclude and re-point, a source
build needed libclang, and nothing checked the generated layouts against the
pyvex actually loaded. Upstream's ``ailment::vex_ffi`` already resolves
``vex_lift`` at runtime in the libpyvex pyvex has loaded, with hand-written
struct mirrors, so the lifter moved onto it and the feature, header,
``bindgen`` dependency and link flags were deleted. The layouts and enum
tags are now checked against pyvex's cffi library by
``tests/engines/rust/test_native_lift_parity.py``; see
:doc:`rust_libvex_ffi`.

**JSON IRSB bridge replaced by reading pyvex objects** (``angr-mx00p.33``, ``angr-mx00p.44``).
The Python ``lift_block`` callback used to serialize
``block.vex`` to JSON, which ``serde`` decoded on the Rust side against a
hand-kept schema. The serializer and the schema drifted apart on every shape
libVEX itself never produces: a Python-built ``U1`` holds an int, a Spotter's
``Dirty`` leaves ``mSize`` as ``None``, and the serializer never emitted
``MBE.event``. So Spotter blocks and fence blocks failed to decode whenever
they came through the callback. ``callbacks/pyvex_reader.rs`` now reads the live
objects over PyO3 with the GIL the callback already holds, using the
native lifter's field conventions; see :doc:`rust_libvex_ffi`.

Architecture and ABI tables
---------------------------

**Register tables are not loaded from archinfo at run time** (``angr-mx00p.34``).
Passing archinfo's register table into Rust when the
manager is built, and deleting ``arch/*.rs``'s ``CANONICAL`` / ``ALIASES``,
was the original plan. The engine needs the tables before any manager
exists: the seed state is built standalone, snapshots rebuild an arch from
its name, ``Clone for Box<dyn Arch>`` relies on the arches being zero-sized
singletons, and the Rust unit tests build states with no Python at all. So
the built-in tables would stay either way, and a second, runtime copy would
add a sync problem rather than remove one. Instead the Rust tables are the
engine's source of truth. ``tests/engines/rust/test_arch_offset_parity.py``
checks them against archinfo's VEX guest layout in both directions, along
with the special-purpose and calling-convention registers. The Python
side reads them through ``arch_register_info`` instead of keeping its own
copies, and the exported-name list is derived from ``CANONICAL`` rather
than kept by hand.

**Syscall numbers are angr's, not a Rust table** (``angr-mx00p.35``). The
native syscall handlers used to sit in one hand-written
``(number, handler)`` table per arch, transcribed from angr's
``linux_kernel.py``. A transcription slip once sent i386 ``signalfd4``
into the stat writer (``angr-sqfj8.107``), and every per-arch quirk of
angr's numbering had to be re-learned for each new handler. The handlers are now registered by
the name angr gives each syscall, and each manager resolves the numbers from
its SimOS's ``syscall_number_mapping``, so no number is spelled in Rust and
DECREE needs no special key. The cost is that an upstream renumbering changes
dispatch without a Rust edit. That is intended: angr is the source, and
``tests/engines/rust/test_native_syscall_parity.py`` checks every resolved
row against angr's names.
