Rust engine: lazy symbolic memory — design
==========================================

.. note::

   **Status: implemented and shipped (angr-pogf).** This was
   originally a design-phase deliverable; lazy symbolic memory now ships
   and is the default store path. Read it as historical design rationale,
   not as a description of code-to-be-written.

   *What shipped, and where it deviates from the proposal below:*

   - **Lazy LOAD** (``angr-czph``) and **lazy STORE** (``angr-qh5u``)
     both shipped. Multi-cell stores are the *only* path for
     ``Multiple`` / ``Strided`` symbolic-address store results
     (``angr-mmdh.3``); the eager ``store_conditional_multiple`` helper
     is gone.
   - **Architecture deviation** — the shipped code is a *sidecar*, not
     the enum-per-byte ``Multi`` variant this document proposes. Symbolic
     data lives in a ``multi_objects: FxHashMap<addr, MultiPayload>``
     sidecar on ``SymbolicMemory`` (parallel to ``symbolic_objects``)
     plus a ``multi_bitmap`` on ``MemoryPage`` (parallel to
     ``symbolic_bitmap``). A byte is Multi when its page bit is set *and*
     ``multi_objects`` has an entry; ``set_multi_alternatives`` is the
     only installer.

The body below is the design as proposed (``angr-pogf``), written against
an engine that still concretized symbolic-address stores eagerly. Lazy LOAD
(``angr-czph``) and lazy STORE (``angr-qh5u``) implemented it, with the
deviations listed in the note above.

Audience: contributors changing the lazy-memory path who need the
reasoning behind the shipped shape and the alternatives it was weighed
against.


Why this work
-------------

The Rust engine outperforms the Python engine on most benchmarks (see
the live ``python_time`` / ``rust_time`` ratios in
``tests/benchmarks/baseline_timings.json``), but at design time a small
set of programs remained slower. All of them shared the same shape: a
tight loop that issues
many symbolic-address stores, optionally interleaved with symbolic
loads. The eager Rust path of the time resolved each such operation
immediately, producing the bottleneck described below.

Affected benchmarks at design time (the live ratios are in
``tests/benchmarks/baseline_timings.json``):

* ``sym-write`` — several times slower than Python. One Z3
  concretization call per symbolic store built deeply nested ITE chains
  that later operations had to walk. Resolved by the lazy-store work
  below plus a separate CFG fix; it is now faster than Python.
* ``strcpy_find`` — a separate CFG fix (``angr-3tek``) flipped it to a
  win, but the underlying eager-store cost still bites symbolic-heavy
  variants.
* ``unbreakable_1`` — high-variance bimodal Z3 mode in part driven by
  ITE chain depth interacting with the model picker (see
  :doc:`rust_bimodal_variance`).
* ``mma_howtouse`` — separate root cause (claripy AST cache growth
  across ``Callable`` invocations), called out here only because it is
  sometimes confused with the eager-store cost.

The pattern is structural rather than incidental: every byte that has
been written through a symbolic address carries an ITE chain whose
depth grows with each subsequent symbolic write to overlapping
addresses, and every later load must walk those chains. Python avoids
this by storing the *set* of alternative values at each byte and
only collapsing it to a single AST at load time — see below.


The eager architecture this replaced
------------------------------------

Before this work a symbolic-address store was resolved eagerly: the
address was enumerated through the concretization strategies and the
value stored under a guard for every candidate
(``store_conditional_multiple``, since removed), so a single wide
symbolic write cost one Z3 enumeration plus a conditional store per
candidate, and every later load of those bytes paid for the
``If``-chain it had produced. The concrete fast paths
(``load_concrete`` / ``store_concrete``) were already byte-cell based;
the question this document answers is how to keep them byte-identical
while deferring the symbolic case.


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
   (``angr/storage/memory_mixins/address_concretization_mixin.py``) is
   an opt-in annotation used by a few SimProcedures
   (``libc/strchr.py``, ``libc/memchr.py``, ``libc/gets.py``,
   ``libc/fgets.py``) to upgrade Range concretization on writes even
   when ``SYMBOLIC_WRITE_ADDRESSES`` is off. The annotation is *not*
   what makes the workload lazy — laziness comes from the page-level
   ``MultiValues``. The annotation just widens the candidate set
   beyond a single Max solution.

So the "MultiwriteAnnotation pattern" (``angr-pogf``) is
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
``native/angr/src/symbolic/``.

Pros:

* Conceptually cleanest — defers all address resolution to the
  solver. Models "wide-range symbolic addresses" without manual
  concretization.
* No upfront concretization means the address ↔ value relationship
  is preserved across the whole store chain — solving the
  ``read_fallback_any`` loss-of-relationship issue.

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
representation changes.


Open questions
--------------

.. note::

   **Resolved post-shipping (angr-1yge9.10).** The design questions
   below were open at design time; the shipped sidecar implementation
   settled each. Resolutions are inlined per
   bullet with the anchoring symbol. The ``LAZY_SOLVES`` interaction was a
   test plan, not a design gap, and is covered by the standing Rust suite.

* **When does ``Multi`` collapse back to ``Single``?** A subsequent
  concrete store with no symbolic guard can replace the cell wholly.
  Original proposal: collapse on overwrite, on merge equality, and on
  fork-quiescence.

  *Resolved:* the shipped policy is **collapse on overwrite only**. A
  concrete write drops any overwritten Multi cell from the
  ``multi_objects`` sidecar and bumps its per-byte version so the
  wider-load cache invalidates (``SymbolicMemory::store_concrete`` in
  ``memory/store.rs``, angr-1tes; the page-level ``multi_bitmap`` is
  cleared inside ``MemoryPage::store_concrete`` / ``clear_multi``). Merge
  equality and fork-quiescence collapse were **not** implemented — a
  collapsed cell is simply re-collapsed to an ITE at load time
  (``MultiPayload::collapse`` is memoized, so the cost is amortized), so
  the extra collapse-back triggers earned no measurable benefit and were
  dropped as KISS.
* **Endianness for cross-cell Multi reads.** A load that spans two
  Multi cells with different alternatives needs the same byte-merge
  treatment as today's ``try_byte_merge_load`` (``memory/load.rs``).

  *Resolved:* the wide-load path assembles the result **per byte**
  (``SymbolicMemory::assemble_load_with_multi`` in ``memory/load.rs``):
  each Multi cell collapses independently to a single-byte BV, and the
  bytes are joined by ``concat_bytes_endian`` (angr-kg58), which reverses
  for little-endian so ``byte[0]`` is the LSB. Cross-cell alternatives
  therefore need no special handling — the per-byte assembly is the
  byte-merge fallback the sketch called for.
* **Annotation forwarding.** When Python wraps an address in
  ``MultiwriteAnnotation`` and exports it, the Rust side loses the
  annotation (we strip claripy annotations at the bridge — see
  ``claripy_bridge/import.rs``). Either preserve the annotation across
  the bridge or have ``rust_manager.py`` re-attach it Rust-side.

  *Resolved:* the **second** option shipped (angr-5zw8). The
  annotation is never forwarded through the bridge; instead
  ``RustExplorationManager._try_multi_cell_store``
  (``angr/exploration/rust_manager.py``) reads ``MultiwriteAnnotation``
  off the still-Python ``addr_ast`` *before* the bridge strips it and
  routes the store to Rust's ``state_memory_store_symbolic_multi``. On a
  non-annotated address or a probe failure it falls through to the Python
  store path so no write is lost.
* **Interaction with ``LAZY_SOLVES``.** ``LAZY_SOLVES`` defers
  constraint checks; ``Multi`` defers value collapse. They are
  orthogonal but may interact when a deferred constraint check
  later trips a Multi-cell collapse. Test plan: re-run the
  full Rust test suite with ``LAZY_SOLVES`` both on and off
  once the LOAD side lands. *(This was a verification plan rather than a design
  question; the Rust suite exercises both settings and no interaction
  regression surfaced.)*


References
----------

* Provenance: ``angr-pogf`` (this design), ``angr-czph`` (Phase 1,
  LOAD), ``angr-qh5u`` (Phase 2, STORE).
* Existing user-facing engine doc: :doc:`rust_engine` ("Benchmarks
  slower than Python").
* Python comparator: ``angr/storage/memory_mixins/address_concretization_mixin.py``
  and ``angr/storage/memory_mixins/paged_memory/pages/mv_list_page.py``.
* Z3 array bindings (z3-rs 0.19): ``z3::ast::Array::store`` /
  ``select`` (informational — Option B only).
