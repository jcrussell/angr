Rust engine: lazy symbolic memory — design
==========================================

This document is the design phase deliverable for the lazy symbolic
memory work tracked by ``angr-pogf``. It does not describe shipped
code — at the time of writing, the Rust engine still uses eager
concretization for symbolic-address stores. The implementation
children are ``angr-czph`` (lazy LOAD) and ``angr-qh5u`` (lazy STORE).

Audience: contributors implementing the children, and reviewers
deciding whether to land the design or push back on it.

Status: **proposed, not implemented**. Numbers in the rationale come
from existing benchmarks in ``tests/benchmarks/baseline_timings.json``
and from profiling memories saved against the slower Rust benches.


Why this work
-------------

The Rust engine outperforms the Python engine on most benchmarks (see
the table in ``CLAUDE.md``), but a small set of programs remain
slower. All of them share the same shape: a tight loop that issues
many symbolic-address stores, optionally interleaved with symbolic
loads. The current Rust path resolves each such operation eagerly,
producing the bottleneck described below.

Affected benchmarks, with the dominant cost line from profiling
memories:

* ``sym-write`` — 6.9× slower than Python. 275 Z3 concretization
  calls (one per symbolic store) build deeply nested ITE chains that
  later operations have to walk
  (memories ``sym-write-7s-bottleneck-is-275-z3-concretization``,
  ``symwrite-eager-vs-lazy-memory``).
* ``strcpy_find`` — was 0.21× (now 2.3× after a separate CFG fix in
  ``angr-3tek``, but the underlying eager-store cost still bites
  symbolic-heavy variants).
* ``unbreakable_1`` — high variance bimodal Z3 mode in part driven by
  ITE chain depth interacting with the model picker
  (``benchmark-google2016-unbreakable-1-regression``).
* ``mma_howtouse`` — 0.7× — separate root cause (claripy AST cache
  growth across ``Callable`` invocations), called out here only
  because it is sometimes confused with the eager-store cost.

The pattern is structural rather than incidental: every byte that has
been written through a symbolic address carries an ITE chain whose
depth grows with each subsequent symbolic write to overlapping
addresses, and every later load must walk those chains. Python avoids
this by storing the *set* of alternative values at each byte and
only collapsing it to a single AST at load time — see below.


Current Rust architecture (eager)
---------------------------------

Relevant code:

* ``native/angr/src/memory/store.rs`` —
  ``store_symbolic_unified`` (line 261), ``store_strided`` (line 216),
  ``store_conditional_multiple`` (line 363), ``store_with_concretization``
  (line 320).
* ``native/angr/src/memory/load.rs`` — ``load_symbolic_unified``
  (line 366), ``load_concrete_lazy_inner`` (line 517),
  ``apply_pending_writes_{concrete,symbolic}`` (lines 474, 489 —
  *currently disabled*).
* ``native/angr/src/memory/ite_builder.rs`` — balanced ITE tree
  builders.
* ``native/angr/src/memory/mod.rs`` — page table (``pages``),
  ``symbolic_objects`` (sparse map of base_addr → wide BV),
  ``symbolic_spans`` (reverse index byte_addr → (base_addr,
  width)), and the ``PendingWrite`` queue.
* ``native/angr/src/concretize.rs`` — ``AddressConcretizer`` with
  ``concretize_read`` (Range → Any) and ``concretize_write``
  (Range → Max), mirroring Python's strategy chain.

Store path (eager):

1. ``store_symbolic_unified`` calls ``concretize_write``.
2. ``Single`` → ``store_concrete_automap`` writes the BV at that
   page byte; if the value is symbolic the page byte is marked
   symbolic and the BV stashed in ``symbolic_objects[addr]``.
3. ``Multiple(addrs)`` or ``Strided`` → ``store_conditional_multiple``
   loads the current value at every candidate, builds
   ``ITE(addr == cand_i, new_val, current_i)`` for each, and writes
   the result back into the candidate cell. **Each subsequent
   symbolic store to overlapping candidates nests another ITE
   layer.** This is the bottleneck.
4. ``TooLarge`` / ``Failed`` → returns ``MemoryError::SymbolicAddress``
   so the interpreter falls back to a Python callback
   (``memory_store_symbolic_full``, wired at
   ``angr/exploration/rust_manager.py:1619``).

Load path (eager):

1. ``load_symbolic_unified`` concretizes the read address.
2. ``Single`` → direct page load; partial reads of wider symbolic
   objects use ``symbolic_spans`` for O(1) lookup.
3. ``Multiple`` / ``Strided`` → balanced ITE tree
   (``ite_builder.rs``). Tree depth O(log N), but every leaf reads a
   value that may itself be an ITE chain from earlier stores.
4. ``TooLarge`` / ``Failed`` → Python callback
   (``memory_load_symbolic_full``).

Pending writes (scaffolded, **not active in execution**):

The ``PendingWrite`` struct, ``add_pending_write``, ``drain_pending_writes``,
and ``flush_pending_writes`` exist but are only exercised by tests.
``apply_pending_writes_{concrete,symbolic}`` are stubs that return the
base value unchanged (``memory/load.rs:474-499``). This was an earlier
attempt at lazy memory that did not work — see memory
``lazy-memory-load-overlay-fails``: a per-load overlay forces every
load to enumerate every pending write, which costs O(n) Z3 per load
for symbolic-address pending writes, and loads outnumber stores in the
target benchmarks. The infrastructure is kept because flushing on
export is still useful, but it is not the path forward for in-flight
laziness.


Python's approach, for reference
--------------------------------

Two pieces in Python make the same workloads cheap:

1. ``AddressConcretizationMixin``
   (``angr/storage/memory_mixins/address_concretization_mixin.py``)
   still concretizes addresses upfront — Python is **not** lazy on
   address resolution. It uses the same strategy chain (Range → Any
   for reads, Range → Max for writes) that Rust already mirrors.

2. ``MultiValues`` cells in
   ``angr/storage/memory_mixins/paged_memory/pages/multi_values.py``
   plus ``MVListPage`` give each page byte the ability to store a set
   of alternative values. A symbolic-address store appends an
   alternative to each candidate cell (one ``SimMemoryObject`` per
   candidate, all coexisting in the cell's set) rather than collapsing
   to an ITE. The load builds an ITE chain at load time across only
   the bytes the load actually touches.

3. ``MultiwriteAnnotation``
   (``address_concretization_mixin.py:14``) is an opt-in annotation
   used by a few SimProcedures (``libc/strchr.py``, ``libc/gets.py``,
   ``libc/fgets.py``) to upgrade Range concretization on writes even
   when ``SYMBOLIC_WRITE_ADDRESSES`` is off. The annotation is *not*
   what makes the workload lazy — laziness comes from the page-level
   ``MultiValues``. The annotation just widens the candidate set
   beyond a single Max solution.

So the "MultiwriteAnnotation pattern" in the bead description is
really shorthand for the combined behavior: concretize to a range of
addresses, but record alternative values per byte instead of folding
into an ITE.


Options considered
------------------

Option A — Per-byte MultiValues port (Python-equivalent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Change ``MemoryPage`` so each byte cell can hold one of three things:

* ``Concrete(u8)`` (existing)
* ``Symbolic(SymbolicByteId)`` (existing — index into
  ``symbolic_objects``)
* ``Multi(Vec<(RustBV /*cond*/, ByteRef)>)`` — new

A symbolic-address store with concretization result
``Multiple(addrs)`` no longer builds ``ITE(addr==cand, new, current)``
and stores it back. Instead, for each candidate cell it appends
``(addr == cand, new_val)`` to that cell's alternatives list. Reads
collapse alternatives at load time using the existing balanced ITE
builder, scoped to only the bytes being loaded.

Pros:

* Closest to Python's well-tested approach; we already trust
  ``MVListPage`` semantics under merging, forking, and the address
  concretization mixin.
* No new Z3 theory; all operations stay in Rust BV arithmetic.
* The ``symbolic_objects`` / ``symbolic_spans`` indices already handle
  the existing "wide symbolic object spans multiple bytes" case —
  Multi cells generalize this from one BV per range to N BVs with
  guards.
* Composes with the existing balanced ITE builder
  (``ite_builder.rs``) — alternative collapse can reuse it.

Cons:

* ``MemoryPage`` byte representation gets larger. We need a careful
  measurement that the typical page (mostly concrete) does not
  regress.
* Forking semantics: ``Multi`` cells need correct CoW behavior. Since
  ``OrdMap`` already provides O(1) page-level CoW we just need to
  make sure the cell type does not internally share mutable state
  across forks.
* Merging two states needs a defined ``Multi`` ⊔ ``Multi`` rule.
  Python's merge mixin handles this; we have to mirror it.

Option B — Z3 array/lambda primitive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a new symbolic-memory backend that represents the symbolic
portion of memory as a Z3 array ``BV64 → BV8``. Stores: ``mem' =
Store(mem, addr, val)``. Loads: ``Select(mem, addr)``. Let Z3 simplify
``Select(Store(...))``.

z3-rs 0.19 (the version pinned in ``native/angr/Cargo.toml``) does
expose ``Array::store``/``select`` — confirmed in
``z3-0.19.7/src/ast/array.rs``. No version bump needed. There is no
existing Z3 array usage anywhere under
``native/angr/src/symbolic/`` (memory
``invariant-lazy-mem-deferred-2026-05-09`` confirmed by grep on
2026-05-09).

Pros:

* Conceptually cleanest — defers all address resolution to the
  solver. Models "wide-range symbolic addresses" without manual
  concretization.
* No upfront concretization means the address ↔ value relationship
  is preserved across the whole store chain — solving the
  ``read_fallback_any`` loss-of-relationship issue noted in
  ``invariant-lazy-mem-deferred-2026-05-09``.

Cons:

* Z3 array theory has its own performance cliffs. For symbolic
  execution workloads where the same address is read many times,
  array theory has historically been slower than ITE chains for
  small-to-medium concretization sets — exactly the regime our
  current concretizer handles well (Range limits 1024/128).
* Forking cost: each fork has to clone the array AST handle.
  Cheap as a reference, but solver-side state may dominate.
* Forces every page that contains a single symbolic write into the
  array model, including pages that previously held only concrete
  bytes — we lose the page-table fast path.
* Merging: combining two array-backed memories is a new operation;
  Python's merge mixin does not give us a template for this.
* Higher implementation risk: the bottleneck moves from "275 Rust
  ITE constructions" to "275 Z3 array operations plus
  Select-Store-Store nesting that the simplifier may not flatten".
  We have no data on whether Z3 wins this race.

Option C — Hybrid (Python fallback for the painful cases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect "this store will create a deep ITE" before issuing it and
fall through to Python's ``memory_store_symbolic_full`` /
``memory_load_symbolic_full`` callbacks for that operation only.
Python's ``MVListPage`` handles the lazy case; Rust handles
everything else.

Pros:

* Minimal Rust work — heuristic + plumbing.
* Avoids the Z3 array risk and the ``Multi`` cell representation
  change.

Cons:

* Inherits Python's performance ceiling exactly when we hit the
  pathological case — gains capped at Python's number for that
  workload, not better.
* Adds a permanent Python callback path that future Rust changes
  have to keep consistent. We have been steadily reducing callback
  usage (see ``angr-3tek.2``, the NativeRead/NativeWrite default
  flip that pushed fauxware from 0.9× to 1.4×); regressing on that
  direction for a few benchmarks is a poor trade.
* The heuristic to decide "would this be deep ITE?" is itself
  fragile — getting it wrong on the cheap side regresses Python,
  on the expensive side makes the work pointless.


Recommendation
--------------

**Option A**, in two phases. Option B stays on the table as a
follow-up if profiling after Phase 1 shows the in-Rust ITE collapse
itself is the bottleneck (rather than the per-store ITE depth that
Option A targets).

Phase 1 = Option A LOAD side (angr-czph): introduce ``Multi`` cells
and read-time collapse, but keep stores eager. This is a self-checking
intermediate step: existing stores produce single-value cells, so the
Multi path is exercised only by stores that someone explicitly
upgrades. We can wire a single SimProcedure (e.g. the ``strchr`` /
``gets`` family that uses ``MultiwriteAnnotation`` in Python) to the
upgraded path and verify correctness on its tests before changing
the general store path.

Phase 2 = Option A STORE side (angr-qh5u): make
``store_symbolic_unified`` emit ``Multi`` cells for
``Multiple`` / ``Strided`` concretization results by default. Once
this is on, every symbolic-address store benefits, not just
annotated ones.

Why not start with B: Option B is a much larger architecture change
with no comparable in-codebase template. Option A reuses the
existing balanced ITE builder, the existing concretization strategy
chain, the existing fork/merge plumbing — only the cell
representation changes. The implementation surface for Phase 1 is
roughly:

* ``memory/page.rs`` — extend ``MemoryPage`` byte representation.
* ``memory/load.rs`` — collapse ``Multi`` cells at load time using
  ``ite_builder.rs`` over the loaded byte range.
* ``memory/mod.rs`` — fork/clone behavior for the new cell type;
  dirty-page tracking unchanged.
* ``memory/store.rs`` — leave alone in Phase 1, gate on a new
  ``store_symbolic_multi`` API used only by the test rig and
  one annotated SimProcedure.
* ``memory/tests.rs`` — extend the existing ``pending_writes``
  tests (lines 884-1130) to cover the Multi-cell path.

Phase 2 expands ``memory/store.rs::store_symbolic_unified`` to
prefer the Multi path for ``Multiple`` / ``Strided`` results.


Implementation plan
-------------------

Each phase is one or more bd issues. The acceptance criteria below
are tight enough that a reviewer can verify them on the diff.

Phase 0 — instrumentation (prerequisite, fast)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add a counter ``ite_depth_max`` to ``SymbolicMemory`` that tracks
  the deepest ITE chain currently in any cell. Exposed via the
  existing ``get_solver_stats`` / ``RustSolverContext`` reporting
  hook so we can compare before/after.
* Add a benchmark or extend an existing one to surface
  ``sym-write`` and ``strcpy_find`` ITE-depth numbers in
  ``baseline_timings.json``.

Why a separate phase: without numbers, we cannot tell whether Phase
1 actually improved things or just moved the cost around. The
counter is cheap (single u32 update on Multi insertion) and lets us
gate Phase 2 on observed improvement.

Phase 1 — Multi cell LOAD path (angr-czph)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add ``Multi`` variant to the byte-cell enum in
   ``memory/page.rs``. Define a per-cell ``Vec<(RustBV /*cond*/, BV
   /*val*/)>`` payload, with a constraint that the conditions are
   pairwise distinct concretized address equalities (i.e. exactly
   one alternative is true under any model). Document this invariant
   in the type's doc comment.
2. Teach ``load_concrete_lazy_inner`` (``memory/load.rs:517``) to
   detect ``Multi`` cells in the range it touches, collapse them
   via the balanced ITE builder over the load's byte width, and
   merge the result into the returned BV. The existing wide-symbolic
   reconstruction path stays.
3. Add ``store_concrete_multi`` /
   ``store_symbolic_unified_multi`` helpers (not wired into the
   default store path yet) — these are the entry points the test
   rig will use.
4. Implement fork via the existing ``OrdMap`` clone path; ensure
   ``MemoryPage::clone`` is shallow for ``Multi`` payloads (the
   ``RustBV`` ASTs are Z3-managed and refcounted).
5. Tests: extend ``memory/tests.rs`` with a ``test_multi_cell_*``
   suite that mirrors the existing ``pending_writes`` tests at
   lines 884-1130. Use the new helpers directly.
6. Wire a single Python SimProcedure path (probably ``strchr``) to
   the upgraded store via the existing
   ``memory_store_symbolic_full`` callback bypass, gated on the
   ``MultiwriteAnnotation`` presence on the address. Confirm
   ``tests/engines/test_rust_exploration.py`` still passes.

Acceptance: ``test_rust_exploration.py`` 389/389 still passing;
new ``test_multi_cell_*`` cases passing; ``ite_depth_max`` for
``strcpy_find`` (or the chosen workload) measurably lower than the
eager baseline; no regression on the existing benchmark gate.

Phase 2 — Multi cell STORE path (angr-qh5u)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Switch ``store_symbolic_unified`` ``Multiple`` / ``Strided``
   branches to call the Multi-cell helpers added in Phase 1
   instead of ``store_conditional_multiple`` /
   ``store_strided``. ``Single`` and ``TooLarge`` / ``Failed``
   paths unchanged.
2. Update ``flush_pending_writes`` (already in
   ``memory/mod.rs:375``) to materialize Multi cells via the same
   collapse path Phase 1 introduced, so state export to Python
   stays correct.
3. Adjust the
   ``memory_store_symbolic_full`` callback wiring in
   ``rust_manager.py`` — only the ``TooLarge`` / ``Failed`` cases
   need Python fallback now.
4. Re-run the full ``baseline_timings.json`` bench corpus. Treat any
   regression >15 % on a non-bimodal benchmark as a blocker. Treat
   ``sym-write`` failing to improve by ≥ 2× as a soft blocker —
   keep the work behind a SimOption rather than enabling by default.

Acceptance: ``sym-write`` ≥ 2× current Rust time (currently ~7s; aim
for under 3.5s, putting it at parity-or-better with Python). No
regression on benchmarks not in the slower-than-1.0× table. Same
389 Python tests pass. New Rust unit tests cover the Multi-cell
store path under fork, merge, and export.

Phase 3 (optional, gated on measurement) — Z3 array fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only pursue if Phase 2 leaves a benchmark slower than Python and
profiling identifies the residual cost as load-time ITE collapse
rather than store path. Even then, scope Z3 arrays to a single
``SymbolicMemory`` mode (e.g. ``SYMBOLIC_MEMORY_USE_Z3_ARRAYS``) so
the default path stays Multi-cell.


Risks
-----

* **Cell representation regression on concrete-heavy workloads.**
  Phase 1 widens the byte cell type. If we use an
  ``enum { Concrete(u8), Symbolic(SymbolicByteId), Multi(...) }`` the
  size of the Multi variant inflates every cell. Mitigations:
  ``Box<MultiPayload>`` to keep the enum 16 bytes; benchmark
  ``fauxware`` (concrete-heavy) before/after; revert to the existing
  storage layout if regression >5 %.
* **Merge semantics.** Two states each carrying a Multi cell at the
  same byte must merge to a single Multi cell whose alternatives
  union the inputs with merge-condition guards. Python's merge mixin
  does this; we just have to mirror the contract. Risk: getting
  ``angr/state_plugins/inspect.py`` merge cases subtly wrong. The
  Rust engine does not currently honor ``state.inspect``
  (``angr-osuu``, ``angr-mq8l``), so we only need merge correctness
  for Rust-Rust merges.
* **Constraint pollution.** Every ``Multi`` collapse adds an
  implicit ``Or(cond_i)`` constraint to the result BV. Today, Rust's
  ``store_conditional_multiple`` adds this once and lets it sit on
  the cell; with Multi cells the constraint is rebuilt per load.
  If a load sees the same Multi cell repeatedly, we want to cache
  the collapsed BV — investigate at Phase 1 review time.
* **Disk init cache invalidation.** Multi cells must serialize
  through the disk init cache code paths in
  ``angr/exploration/rust_manager.py`` and round-trip through
  ``rust_state_export.py`` / ``rust_state_sync.py`` without losing
  alternatives. Add a regression test parallel to
  ``test_mips32_symbolic_register_survives_disk_init_cache``.
* **Memory regression from retaining alternatives.** Eager mode
  collapses, freeing the intermediate ASTs. Multi mode retains
  them. For workloads with many small symbolic stores this could
  bloat resident set size. Phase 0 instrumentation should also
  count outstanding Multi alternatives so Phase 2 can be vetoed if
  RSS jumps.


Open questions
--------------

* **When does ``Multi`` collapse back to ``Single``?** A subsequent
  concrete store with no symbolic guard can replace the cell wholly.
  Current proposal: collapse on overwrite, on merge equality, and on
  fork-quiescence (cells touched only in one branch since the most
  recent fork). Final policy to be decided at Phase 1 review.
* **Endianness for cross-cell Multi reads.** A load that spans two
  Multi cells with different alternatives needs the same byte-merge
  treatment as today's ``try_byte_merge_load``
  (``memory/load.rs:48``). The straightforward approach is to lift
  the byte-merge fallback to handle Multi cells per byte. Sketch
  before committing.
* **Annotation forwarding.** When Python wraps an address in
  ``MultiwriteAnnotation`` and exports it, the Rust side currently
  loses the annotation (we strip claripy annotations at the bridge —
  see ``claripy_bridge.rs``). Either preserve the annotation across
  the bridge or have ``rust_manager.py`` re-attach it on the
  Rust-side concretization config so the Phase 1 strchr path can
  detect the upgrade.
* **Interaction with ``LAZY_SOLVES``.** ``LAZY_SOLVES`` defers
  constraint checks; ``Multi`` defers value collapse. They are
  orthogonal but may interact when a deferred constraint check
  later trips a Multi-cell collapse. Test plan: re-run the
  full Rust test suite with ``LAZY_SOLVES`` both on and off
  during Phase 1.


References
----------

* Parent bead: ``angr-pogf`` — this design.
* Children: ``angr-czph`` (Phase 1, LOAD), ``angr-qh5u`` (Phase 2,
  STORE). Both are currently deferred to 2026-06-01 to align with
  this document.
* Existing user-facing engine doc: :doc:`rust_engine`. The slower
  benchmarks table there names the workloads this work targets.
* Python comparator: ``angr/storage/memory_mixins/address_concretization_mixin.py``
  and ``angr/storage/memory_mixins/paged_memory/pages/mv_list_page.py``.
* Z3 array bindings (z3-rs 0.19): ``z3::ast::Array::store`` /
  ``select`` (informational — Option B only).
