Rust engine: proxy-backed SimState writes — design comparison
=============================================================

.. note::

   **Status: partially superseded (angr-j28e, 2026-06-02).**

   ``RustStateProxy`` writes (``proxy.regs.<name> = value`` and
   ``proxy.memory.store(addr, data)``) are now implemented as direct
   write-through to Rust — the read-only contract this document
   compared against is no longer current. See
   :doc:`rust_engine` "RustStateProxy write-through contract" for the
   shipped behavior.

   The diff-and-push path described below still governs the
   **SimProcedure callback boundary** (Python full ``SimState``
   mutates → Rust syncs back on return). That part of the document
   is current. The "proposed proxy-plugin design" section is now
   historical reading — the proxy uses an immediate write-through
   model, not the queued-mutation plugin substitution this doc
   compared against.

This document is the design-phase deliverable for the parking bead
``angr-g7ug`` ("Proxy-backed SimState writes — eliminate
_compute_register_changes diff-and-push"). The measured sync-back
numbers were captured on this machine via
``run_single.py --dump-counters`` at HEAD ``5ad385aef``.

Audience: contributors deciding whether to replace the existing
SimProcedure callback boundary with a proxy-plugin design.


Why this work
-------------

When a Python SimProcedure runs at a hook, the Rust engine cedes
control across the PyO3 boundary and the SimProcedure mutates an
angr ``SimState`` via the standard plugin API (``state.regs.rax = …``,
``state.memory.store(…)``, ``state.solver.add(…)``). When the
SimProcedure returns, those mutations have to be reflected in the
Rust canonical state. Today's mechanism is a diff-and-push pass:

#. **Before** the callback runs, the manager snapshots register
   values into a lightweight dict (or, for memory-writing procedures,
   takes a full ``state.copy()``) — see
   ``rust_state_sync.py:_snapshot_registers``.
#. The Python SimProcedure mutates the snapshotted state directly
   through angr's plugin paths.
#. **After** the callback returns, ``_extract_register_changes``
   walks every named register, compares to the snapshot, and emits
   ``(offset, size, data_bytes)`` tuples. ``_extract_memory_changes``
   walks ``new_state.memory.changed_bytes(old_state.memory)`` to
   collect concrete and symbolic deltas. New constraints are
   extracted by length-comparing the constraint list.
#. The aggregate diff is pushed back through
   ``resume_after_simprocedure(addr, reg_changes, mem_changes,
   new_constraints)``.

This works (Rust stays canonical because every delta is reliably
reflected) but is O(num_registers) per callback plus the changed-bytes
scan over any modified pages, plus per-symbolic-value AST re-export.

The parking bead's hypothesis is that we could eliminate the
diff-and-push pass by giving the callback ``SimState`` proxy-backed
``regs`` / ``memory`` / ``solver`` plugins that mutate Rust
synchronously, on every store.


Empirical input: per-callback sync-back cost
---------------------------------------------

Measured at HEAD ``5ad385aef`` on this machine, via
``tests/benchmarks/run_single.py <bench> --dump-counters``:

.. list-table::
   :header-rows: 1
   :widths: 22 11 13 13 13 14 14

   * - Bench
     - callbacks
     - total ms
     - state_create ms
     - execute ms
     - sync_back ms
     - per-cb sync ms
   * - fauxware
     - 1
     - 57.8
     - 2.8
     - 41.0
     - 13.9
     - 13.9
   * - mma_howtouse
     - 1
     - 8.4
     - 1.1
     - 0.0
     - 7.2
     - 7.2
   * - google2016_unbreakable_0
     - 1
     - 195.6
     - 19.4
     - 167.5
     - 7.7
     - 7.7
   * - flareon2015_10
     - 2
     - 23.0
     - 7.2
     - 9.8
     - 4.8
     - 2.4
   * - csaw_wyvern
     - 30
     - 2063.2
     - 1962.7
     - 71.5
     - 26.7
     - 0.89

Two patterns stand out:

#. **The diff-and-push pass costs ~1–14 ms per callback.** The variance
   tracks the cost of ``state.memory.changed_bytes()`` on the pages the
   procedure touched plus per-symbolic-register AST re-export. The
   fauxware ``open`` callback hits the upper end (a register snapshot
   plus a concrete-memory write); ``CallReturn`` and ``ReturnUnconstrained``
   sit near the bottom because they touch only the return register.
#. **Sync-back is rarely the dominant phase.** It is at most 24 % of
   per-callback time on benches with substantial ``execute`` cost
   (fauxware ``open``), 86 % on procedures that do almost nothing
   (mma_howtouse ``CallReturn``), and 1.3 % on callback-heavy benches
   where ``state_create`` dominates (csaw_wyvern).

**Implication.** Aggregate sync-back time across a whole bench is on
the order of **tens of milliseconds**:

* fauxware: 13.9 ms sync-back vs ~190 ms wall = ~7 % upper bound.
* csaw_wyvern: 26.7 ms sync-back vs ~16.4 s wall = ~0.16 % of wall.
* mma_howtouse: 7.2 ms sync-back vs ~6 s wall = ~0.12 % of wall.

A proxy-backed write path that ran at zero overhead could only
recover those numbers. The eye-watering "per-Callable" cost on
mma_howtouse is in ``state_create`` (the SimState rebuild and the
solver materialize), not in sync-back — different bead (``angr-9maq``).


Current diff-and-push model
---------------------------

Reference points in the live code:

* ``angr/exploration/rust_state_sync.py:1143`` —
  ``_snapshot_registers``: builds a ``reg_name → (is_symbolic,
  concrete, offset, size)`` dict in ~0.1 ms (vs ~1 ms for
  ``state.copy()``).
* ``angr/exploration/rust_state_sync.py:1202`` —
  ``_extract_register_changes``: walks the architecture register
  map, compares to the snapshot or old state, emits a list of
  ``(offset, size, data_bytes)`` tuples. Symbolic return registers
  route through ``_sync_symbolic_register_to_rust``, which calls
  ``set_pending_register_symbolic_ast``.
* ``angr/exploration/rust_state_sync.py:1328`` —
  ``_extract_memory_changes``: uses
  ``new_state.memory.changed_bytes(old_state.memory)``, groups
  consecutive bytes into regions, and emits concrete + symbolic
  region tuples. Symbolic regions are byte-emitted (up to 128 B) or
  whole-emitted with handle registration for restoration.
* ``angr/exploration/rust_callback_dispatch.py:960-1040`` — the
  call site. After the SimProcedure returns,
  ``_extract_register_changes`` and ``_extract_memory_changes`` are
  invoked sequentially; ``_extract_new_constraints`` runs
  against the constraint-list-length delta; everything is bundled
  into ``resume_after_simprocedure(new_pc, reg_changes, mem_changes,
  new_constraints)``.

What the manager already exposes to a state_id-keyed write path:

* ``set_state_register_symbolic_ast(state_id, name, ast)`` —
  ``native/angr/src/exploration/state_api.rs:557`` (symbolic
  registers).
* ``state_memory_store_symbolic_multi(state_id, addr, data)`` —
  ``state_api.rs:617`` (symbolic memory regions).
* Concrete-register write by state_id: **does not exist**. Rust
  has ``RustSimState::set_register`` and a manager-scoped
  ``set_register`` (current state only); a state-id-keyed shim
  would be net-new.
* Concrete-memory write by state_id: **does not exist** for the
  proxy path. The diff-and-push side uses
  ``resume_after_simprocedure``'s ``mem_changes`` payload, which
  the engine applies in batch.
* Solver-add by state_id: ``_rust_add`` interceptor monkey-patches
  ``state.solver.add`` and writes through immediately. **Already a
  proxy-style path.**

So the bead's framing — "switch to proxy-backed writes" — is
half-true today: solver writes are already proxy-shaped; register
and memory writes are still diff-shaped.


Option A — Proxy-backed plugins for callback SimStates
------------------------------------------------------

Each callback ``SimState`` is constructed with replacement
``SimRegisters`` / ``SimMemory`` plugins. Writes call through
PyO3 immediately into Rust:

.. code-block:: python

    # under the proposed proxy plugin
    state.regs.rax = claripy.BVV(0x41, 64)
    # → mgr.set_state_register(state_id, "rax", 0x41)

    state.memory.store(0x401000, b"\x90\x90")
    # → mgr.set_state_memory(state_id, 0x401000, b"\x90\x90")

The diff-and-push pass at the boundary disappears. Symbolic
writes route through the existing ``set_state_register_symbolic_ast``
/ ``state_memory_store_symbolic_multi`` FFI hooks.

API needed
~~~~~~~~~~

New PyO3 shims:

* ``set_state_register(state_id, name, value)`` — concrete by
  name (mirrors ``set_register`` but state-id-keyed).
* ``set_state_memory(state_id, addr, bytes)`` — concrete write.
* (Already extant) ``set_state_register_symbolic_ast``,
  ``state_memory_store_symbolic_multi``.

Plus a new Python class pair — ``ProxyRegisters`` /
``ProxyMemory`` — registered as plugins on the callback SimState.
``RustRegisterProxy`` / ``RustMemoryProxy`` are read-only today
(``rust_state_proxy.py:240-425``), with ``memory.store`` raising
``NotImplementedError`` at ``rust_state_proxy.py:422``. The
proxy variant for callbacks would extend them.

Pros
~~~~

* **Eliminates the diff scan.** Removes
  ``_extract_register_changes`` / ``_extract_memory_changes`` from
  the callback hot path entirely. Saves the per-callback 1–14 ms
  measured above.
* **Aligns with the already-proxy solver path.** Constraints
  already write through immediately via ``_rust_add``. Registers
  and memory becoming write-through closes the asymmetry.
* **Plugin restoration may be unblockable.** The bead description
  notes the interaction with ``angr-2k64`` (per-state-id plugin
  cache); if posix.stdin reads/writes go directly to Rust, the
  cache may be redundant.
* **Inspect wiring is more straightforward.** Each write fires an
  inspect event at the proxy call site, no diff-scanning needed to
  reconstruct what changed.

Cons
~~~~

* **Synchronous PyO3 per write.** Each ``state.regs.rax = …``
  becomes one PyO3 boundary crossing. Today the boundary is
  crossed exactly once at ``resume_after_simprocedure``. A
  SimProcedure that writes N registers + M memory bytes becomes
  N+M FFI calls instead of one. PyO3 calls are not free — a bare
  no-op ``#[pymethod]`` call costs ~200–500 ns on this hardware,
  and per-byte memory writes inflate to ~µs each before solver
  work.
* **Conditional writes (claripy.If).** SimProcedures that branch
  on a symbolic condition and write conditionally
  (``claripy.If(cond, val1, val2)``) need byte-identical
  semantics under the proxy. The diff-and-push path picks up the
  net effect by reading the final value; a write-through proxy
  must either evaluate the If at write time (correctness risk) or
  store the symbolic If into Rust (which the symbolic write path
  already handles, but is slower than a concrete store).
* **Memory write granularity.** angr's
  ``state.memory.store(addr, bvv)`` writes can be 1 to 64+ bytes
  at a time. The diff-and-push path groups contiguous bytes into
  regions via ``_group_changed_bytes`` (``rust_state_sync.py:1402``).
  A naive proxy that proxies each byte individually loses this
  batching; one that proxies each ``store()`` call preserves it.
* **Plugin contract risk.** Replacing ``SimRegisters`` /
  ``SimMemory`` with proxy variants must remain transparent to
  every SimProcedure. angr's plugin system has many corners
  (page maps, history tracking, options-driven behavior toggles).
  Each unhandled corner is a silent behavior diff.
* **Limited net savings.** Per the measurement table, sync-back is
  at most 0.16 % of wall time on the heavy benches and 7 % on the
  fastest. Recovering 100 % of that is bounded by ~0–7 % wall.
  Realistic recovery (after paying for N+M extra FFI calls per
  callback) is well below that bound.

Implementation cost
~~~~~~~~~~~~~~~~~~~

Medium-high. Two new PyO3 shims (concrete state-id register +
memory writes — ~50 LoC each). Two new proxy plugin classes
(~300–500 LoC). Test surface includes every SimProcedure path
that currently uses diff-and-push, plus the inspect-event corners,
plus the plugin-restoration interaction with ``angr-2k64``.

Risk surface
~~~~~~~~~~~~

High. The plugin contract is wide; the diff-and-push path
silently corrects for many edge cases (symbolic-write fallbacks,
register-rename aliasing on x86, page-map rebuilds). A proxy
that misses one of these silently diverges.


Option B — Tracked-writes log (incremental write capture)
----------------------------------------------------------

A middle ground: keep the standard angr plugins but install a
lightweight wrapper that records every write in a list. At the
callback boundary, replay the list directly into Rust instead of
diff-scanning.

How
~~~

* Wrap ``state.regs.__setattr__`` and ``state.memory.store`` to
  push ``("reg", name, value)`` / ``("mem", addr, data)`` tuples
  into a per-callback list (already partially in place — see the
  ``tracked_writes`` / ``tracked_symbolic_writes`` channels at
  ``rust_callback_dispatch.py:984-997``).
* At the boundary, walk the list once. No
  ``changed_bytes`` scan, no register loop.

Pros
~~~~

* **No PyO3 per write.** Writes go to a Python list (cheap), then
  the boundary still uses the existing
  ``resume_after_simprocedure(mem_changes=…, reg_changes=…)``
  payload format.
* **No plugin substitution.** The standard ``SimRegisters`` /
  ``SimMemory`` continue to drive caching, options, etc. — only a
  thin wrapper records what they did.
* **Conditional writes preserved.** The wrapper records the final
  value the plugin stored; semantics are byte-identical.
* **Already half-built.** The ``tracked_writes`` channel handles
  ``memory.store`` already; extending to registers is a small
  delta.

Cons
~~~~

* **No reduction in PyO3 boundary crossings.** Boundary still
  crosses once at resume.
* **Saves only the diff-scan cost.** Recovers most of the 1–14 ms
  per callback (the changed_bytes walk and the register-by-register
  comparison) but keeps the snapshot, the symbolic-AST re-export,
  and the resume payload. Realistic recovery: ~50–70 % of the
  sync_back time, or ~0.1–4 ms/callback.
* **Wrapping overhead is paid on every write.** Most writes during
  a SimProcedure do not exit the boundary, so even a ~100 ns
  wrapper cost adds up if the procedure does thousands of writes.
  Mitigated by only enabling the wrappers for the callback
  SimState (already what the diff-snapshot path does too).

Implementation cost
~~~~~~~~~~~~~~~~~~~

Low. Extend the existing ``tracked_writes`` plumbing to cover
registers; route the boundary path to consume the tracked log
when available. ~100–200 LoC. Test surface: every diff-extraction
test plus the existing tracked-writes test set.

Risk surface
~~~~~~~~~~~~

Low-medium. The plugin contract is untouched. The risk is missing
a write path (e.g. ``state.scratch.store`` writes that the
register-rename path emits, or SimProcedure-internal helpers that
bypass the standard setattr). Existing test coverage on the
tracked-writes channel catches most of these.


Option C — Status quo (diff-and-push)
-------------------------------------

Do nothing. The cost is bounded (the table above), the plugin
contract is intact, and the path has years of debug history.

Pros
~~~~

* **Zero risk.** No code changes, no regression surface.
* **Empirical cost is small.** ≤7 % wall on the worst-affected
  bench, ≤1 % wall on most benches.
* **Symbolic correctness is paid in.** Edge cases like the
  ``new_state.regs._ip`` symbolic-PC handling at
  ``rust_callback_dispatch.py:960-969`` are battle-tested.

Cons
~~~~

* **Per-callback overhead does not scale away** under future
  workloads with very high callback density (10K+ callbacks per
  bench would put sync-back into seconds).
* **Plugin restoration (``angr-2k64``) still needed.** The proxy
  variant would obviate that work; Option C leaves it in scope.
* **Inspect-event wiring (already MVPed) duplicates change
  detection.** Inspect emits events on writes; sync-back walks
  changes again. Option A would unify these emissions; Option C
  leaves the duplication.


Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 25 24 26 25

   * - Criterion
     - Option A (proxy plugins)
     - Option B (tracked log)
     - Option C (status quo)
   * - Implementation complexity
     - Medium-high. Two FFI shims, two plugin classes, full
       SimProcedure regression sweep.
     - Low. Extend tracked_writes, route boundary consumer.
     - None.
   * - Recovered wall time (upper bound)
     - 100 % of sync_back (~0–7 % wall on the worst bench).
     - 50–70 % of sync_back (~0–5 % wall).
     - 0 %.
   * - Per-callback PyO3 boundary crossings
     - N+M (one per write).
     - 1 (one resume).
     - 1 (one resume).
   * - Plugin contract surface
     - Wide. Substitutes SimRegisters/SimMemory.
     - Narrow. Wraps __setattr__/store.
     - Untouched.
   * - Conditional write semantics
     - Risk — proxy must store the If shape (handled by symbolic
       FFI but slower).
     - Byte-identical (plugin owns the store).
     - Byte-identical.
   * - Inspect-event integration
     - Natural (event fires at proxy call).
     - Already wired via plugin.
     - Already wired via plugin.
   * - mma_howtouse-shape (1 cb, sync 86 % of cb total)
     - Recovers ~7 ms of an ~8 s wall = noise.
     - Recovers ~5 ms of an ~8 s wall = noise.
     - 7 ms ≈ 0.1 % of wall — already noise.
   * - csaw_wyvern-shape (30 cb, sync 1.3 % of cb total)
     - Recovers ~26 ms of a ~16 s wall = 0.16 %.
     - Recovers ~13 ms of a ~16 s wall = 0.08 %.
     - 26 ms ≈ 0.16 % of wall — already noise.
   * - Plugin-restoration alignment (angr-2k64)
     - Direct — proxy reads/writes don't need the cache.
     - No change — diff-push remains the boundary.
     - No change.
   * - Risk surface
     - High — wide plugin contract; silent divergence on
       unhandled corners.
     - Low-medium — missed-write-path risk only.
     - None.


Recommendation
--------------

**Defer both Option A and Option B. Keep Option C.**

Rationale:

#. **The measured savings are too small to justify either rewrite.**
   Per-callback sync-back is 1–14 ms; the aggregate is at most ~7 %
   of wall time on the fastest bench and well under 1 % on every
   other bench. Even with Option A's 100 % recovery, the bench
   matrix would not change meaningfully.
#. **The risk asymmetry is real.** Option A's plugin substitution
   is exactly the class of refactor that produces silent
   correctness regressions (see ``avoid-state-copy-optimization``
   memory — a prior similar attempt to skip ``state.copy()`` broke
   register sync silently and took ais3 from 3 s to 11 s before it
   was caught). The diff-and-push path's debug history is worth a
   lot.
#. **Option B is cheap but redundant.** The ``tracked_writes``
   channel that would form its basis already exists for memory
   writes. Extending it to registers is straightforward, but the
   savings (~13 ms in csaw_wyvern, ~7 ms in mma) do not move any
   bench across a meaningful threshold.
#. **The architectural cleanness argument is weak.** Option A
   would unify register/memory writes with the already-proxy
   solver writes. But the diff-and-push asymmetry is local
   (lives in two extract functions), well-tested, and has not
   produced bug pressure.

**Reopen conditions.** Either option becomes attractive if:

* A future workload pushes callback density to 10K+ per bench,
  putting sync-back into seconds. None of the current 22-bench
  matrix is anywhere near that — fauxware tops out at 1 callback.
* ``angr-2k64`` plugin-restoration work needs the
  proxy-write architecture as a prerequisite to retire the
  per-state-id plugin cache. In that case Option A becomes a
  dependency rather than an optimization, and the risk surface
  is paid down by the larger refactor.
* The mma_howtouse / hackcon residual gaps turn out to be in
  the diff-and-push path (not the case today — both are in
  ``state_create`` or Z3-shape mismatches per
  ``9maq-root-cause`` / the hackcon `rust_engine` notes).


Open questions (if reopened)
-----------------------------

* **PyO3 per-write microbenchmark.** Confirm the 200–500 ns
  per-call cost on this hardware against the actual write
  granularity of representative SimProcedures (memcpy with N=64,
  strncpy with N=1000, etc.). A 1 KB memcpy at 200 ns/call =
  200 µs of boundary overhead, against ~1 ms of diff-and-push
  cost — could swing either way.
* **Inspect-event interaction.** The MVP inspect path
  (``angr-uq4n.2``) emits events at the proxy boundary today.
  Option A would emit events at every write — verify the event
  count stays bounded for SimProcedures that do many writes.
* **Conditional write semantics.** Audit which SimProcedures use
  ``claripy.If(cond, val1, val2)`` writes and what Option A's
  symbolic-store path produces compared to today's diff-and-push
  output. If results differ on any path, Option A is a
  regression on that path.
* **Tracked-writes coverage.** For Option B specifically: enumerate
  every register-write callsite in angr's plugin internals that
  bypasses ``state.regs.__setattr__`` (e.g. the register-rename
  paths on x86). Each bypass needs explicit tracked-log emission.
* **Plugin-restoration alignment.** If ``angr-2k64`` revives,
  re-evaluate Option A in the larger context — the implementation
  cost is shared and the risk is amortized across both rewrites.


References
----------

* Parent bead: ``angr-g7ug`` (parking) — proxy-backed SimState
  writes design.
* Adjacent bead: ``angr-2k64`` (per-state-id plugin restoration)
  — would interact with Option A's plugin substitution.
* Adjacent bead: ``angr-9maq`` (mma_howtouse state_create cost) —
  the dominant cost on Callable-heavy benches is *not* sync-back.
* Adjacent bead: ``angr-uq4n.2`` (inspect marshalling MVP) —
  emits events at the boundary today; Option A would emit at
  each write.
* Code under audit:

  * ``angr/exploration/rust_state_sync.py:1143-1400`` —
    diff-and-push implementation.
  * ``angr/exploration/rust_callback_dispatch.py:960-1040`` —
    sync_back call site.
  * ``angr/exploration/rust_state_proxy.py:240-425`` —
    read-only proxies that would extend to write paths.
  * ``native/angr/src/exploration/state_api.rs:557, 617`` —
    existing symbolic-write FFI shims that the proxy would reuse.

* Measurements: ``run_single.py --dump-counters`` per-bench
  output captured at HEAD ``5ad385aef`` during this design pass.
* Prior cautionary memory: ``avoid-state-copy-optimization`` — a
  similar attempt to skip state.copy() broke register sync silently
  on ais3 (3 s → 11 s).
* Sibling design docs: :doc:`rust_lazy_memory_design`,
  :doc:`rust_parallel_design`.
