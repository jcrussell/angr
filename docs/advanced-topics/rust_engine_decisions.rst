Rust engine: decisions record
=============================

Alternatives that were tried, measured and rejected while building the
Rust engine, kept so they are not re-tried without new evidence. Each
entry names the bead that holds the data; the numbers themselves are
deliberately not repeated here, because they describe one host on one
day. Design notes that are still live are in :doc:`rust_z3_sharing`,
:doc:`rust_parallel_design`, :doc:`rust_lazy_memory_design`,
:doc:`rust_proxy_writes_design` and :doc:`rust_libvex_ffi`.

Solver
------

**Shared-lineage Z3 solver stays opt-in** (``angr-v5a5``, ``angr-3ms1``,
``angr-v5ht``, ``angr-0dgq``, ``angr-op0dn.9.3``). Sharing one Z3 solver
across a forked lineage, switching scopes with push/pop instead of
materializing a solver per fork, was expected to recover the
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

**Pinning Z3's random seeds does not give model stability**
(``angr-iaol.1``). Setting ``smt.random_seed`` / ``sat.random_seed``
through ``Z3_solver_set_params`` corrupts models (the solver returns
witnesses that violate asserted constraints); the unprefixed
``random_seed`` is accepted but produces *more* variation across fresh
solver instances than no pin. Z3 reserves variable- and restart-ordering
latitude that these keys do not bind. The supported determinism knob is
``RustExplorationManager(deterministic=True)``, documented in
:doc:`rust_engine`.

**``sat-preprocess`` tactic pipelines are unsafe** (``angr-ya00``). An
``ANGR_Z3_TACTIC`` pipeline beginning with ``sat-preprocess`` produced
incorrect results on two benches; ``qfbv`` alone is the only non-default
tactic that has been validated.

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
page-cell variant; what shipped keeps the page store untouched and
tracks symbolic objects in a side index, because that left every
concrete fast path byte-identical and merge semantics expressible per
field.

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
(:doc:`rust_parallel_design`, "Distributed (cross-machine) migration"). Migration count
is the lever that decides whether parallel exploration wins at all, and
on the CTF corpus the in-process scheduler already drives it to the
floor; shipping snapshots across processes adds serialization cost with
nothing left to recover.

**Z3-bound benchmarks are a solver floor, not engine work**
(``angr-9w6ad``, ``angr-34w``). The benches that stay slower than Python
spend their wall time in ``solver.check()`` / ``eval_upto``; two
independent spikes (deterministic seeding, a lazy-solving engagement
audit) found no engine-side lever. They are tracked as bimodal
(:doc:`rust_bimodal_variance`) and are not re-triaged as optimization
candidates.

State proxy
-----------

**Diff-and-push proxies (Option B) rejected; write-through shipped**
(:doc:`rust_proxy_writes_design`). The memo chose to keep the
read-mostly proxy with an explicit diff-and-push at callback exit; the
measured cost of re-syncing whole pages per callback reversed that, and
``RustStateProxy`` writes now go straight to the Rust state. The
write-through contract is in :doc:`rust_engine`.
