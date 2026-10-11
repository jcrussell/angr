Rust engine: proxy-backed SimState writes — design comparison
=============================================================

.. note::

   **Status: partially superseded (angr-j28e).**

   ``RustStateProxy`` writes (``proxy.regs.<name> = value`` and
   ``proxy.memory.store(addr, data)``) are now implemented as direct
   write-through to Rust — the read-only contract this document
   compared against is no longer current. See
   :doc:`rust_engine` "RustStateProxy write-through contract" for the
   shipped behavior.

   **Further superseded for memory (angr-grji4).** At the **SimProcedure callback boundary**
   (Python full ``SimState`` mutates → Rust syncs back on return),
   the diff-and-push path described below now governs only
   **registers and constraints**. The **memory** half is proxy
   write-through by default: ``_use_callback_memory_proxy`` resolves
   to ``True`` in ``rust_manager.py``, so a Python SimProcedure's
   ``state.memory.store(…)`` lands in Rust immediately and
   ``rust_callback_dispatch.py`` discards the ``CallbackMemoryTracker``
   write lists instead of replaying them on return. Set
   ``ANGR_RUST_USE_CALLBACK_MEMORY_PROXY=0`` to opt back into the
   legacy tracked-write diff-and-push. See :doc:`rust_engine` for the
   shipped contract.

   The "proposed proxy-plugin design" section is now
   historical reading — the proxy uses an immediate write-through
   model, not the queued-mutation plugin substitution this doc
   compared against.

This document is the design-phase deliverable for the parking bead
``angr-g7ug`` ("Proxy-backed SimState writes — eliminate
_compute_register_changes diff-and-push").

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
   ``rust_state_sync.py::_snapshot_registers``.
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
--------------------------------------------

Measured with ``run_single.py --dump-counters`` across the benchmark
corpus (``angr-g7ug`` holds the numbers): per-callback sync-back costs
on the order of a millisecond, and even on the callback-densest bench
the aggregate is a single-digit percentage of wall time; on most
benches it is well under one percent. The dominant cost on
callback-heavy benches is state creation, not sync-back
(``angr-9maq``).


Current diff-and-push model
---------------------------

Registers and constraints still take the diff-and-push path described
under *Why this work*: ``rust_state_sync.py::_snapshot_registers`` before
the callback, ``_extract_register_changes`` after it, with symbolic
return registers routed through ``_sync_symbolic_register_to_rust``.
Memory no longer does (see the status note above). The options below
are written against the full diff-and-push model that existed at design
time.


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
At design time ``RustRegisterProxy`` / ``RustMemoryProxy``
(``rust_state_proxy.py``) were read-only, with
``RustMemoryProxy.store`` raising ``NotImplementedError``; the proxy
variant for callbacks would have extended them.

Pros
~~~~

* **Eliminates the diff scan.** Removes
  ``_extract_register_changes`` / ``_extract_memory_changes`` from
  the callback hot path entirely, recovering the whole per-callback
  sync-back cost measured above.
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
  N+M FFI calls instead of one. PyO3 calls are not free, and
  per-byte memory writes multiply that cost before any solver
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
  regions via ``rust_state_sync.py::_group_changed_bytes``.
  A naive proxy that proxies each byte individually loses this
  batching; one that proxies each ``store()`` call preserves it.
* **Plugin contract risk.** Replacing ``SimRegisters`` /
  ``SimMemory`` with proxy variants must remain transparent to
  every SimProcedure. angr's plugin system has many corners
  (page maps, history tracking, options-driven behavior toggles).
  Each unhandled corner is a silent behavior diff.
* **Limited net savings.** Sync-back is a small fraction of wall
  time on every bench (see *Empirical input*), so even full recovery
  is bounded by that fraction. Realistic recovery (after paying for
  N+M extra FFI calls per callback) is well below that bound.

Implementation cost
~~~~~~~~~~~~~~~~~~~

Medium-high. New PyO3 shims (concrete state-id register and
memory writes) and a pair of proxy plugin classes. Test surface includes every SimProcedure path
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
  ``tracked_writes`` / ``tracked_symbolic_writes`` channels that
  ``rust_identity.py::CallbackMemoryTracker`` feeds to
  ``rust_callback_dispatch.py::_handle_simprocedure_callback``).
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
* **Saves only the diff-scan cost.** Recovers the changed_bytes walk
  and the register-by-register comparison but keeps the snapshot,
  the symbolic-AST re-export, and the resume payload — a fraction of
  the sync-back time, not all of it.
* **Wrapping overhead is paid on every write.** Most writes during
  a SimProcedure do not exit the boundary, so even a cheap wrapper
  adds up if the procedure does many writes.
  Mitigated by only enabling the wrappers for the callback
  SimState (already what the diff-snapshot path does too).

Implementation cost
~~~~~~~~~~~~~~~~~~~

Low. Extend the existing ``tracked_writes`` plumbing to cover
registers; route the boundary path to consume the tracked log
when available. Test surface: every diff-extraction
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

Do nothing. The cost is bounded (see *Empirical input*), the plugin
contract is intact, and the path has a long debug history.

Pros
~~~~

* **Zero risk.** No code changes, no regression surface.
* **Empirical cost is small.** A single-digit percentage of wall on
  the worst-affected bench, under one percent on most.
* **Symbolic correctness is paid in.** Edge cases like the
  symbolic-PC handling of the successor's ``regs._ip`` in
  ``rust_callback_dispatch.py::_handle_callback_with_successors`` are
  battle-tested.

Cons
~~~~

* **Per-callback overhead does not scale away** under future
  workloads with very high callback density, where sync-back would
  grow into seconds.
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
     - All of sync_back (already a small fraction of wall).
     - The diff-scan share of sync_back.
     - None.
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
   * - Plugin-restoration alignment (angr-2k64)
     - Direct — proxy reads/writes don't need the cache.
     - No change — diff-push remains the boundary.
     - No change.
   * - Risk surface
     - High — wide plugin contract; silent divergence on
       unhandled corners.
     - Low-medium — missed-write-path risk only.
     - None.


Recommendation at design time, and what shipped
-----------------------------------------------

**At design time: defer both Option A and Option B, keep Option C.**

Rationale:

#. **The measured savings are too small to justify either rewrite.**
   Sync-back is a small fraction of wall time on every bench (see
   *Empirical input*). Even with Option A's full recovery, the bench
   matrix would not change meaningfully.
#. **The risk asymmetry is real.** Option A's plugin substitution
   is exactly the class of refactor that produces silent
   correctness regressions (see ``avoid-state-copy-optimization``
   memory — a prior similar attempt to skip ``state.copy()`` broke
   register sync silently and regressed ais3 badly before it was
   caught). The diff-and-push path's debug history is worth a
   lot.
#. **Option B is cheap but redundant.** The ``tracked_writes``
   channel that would form its basis already exists for memory
   writes. Extending it to registers is straightforward, but the
   savings are noise on every bench, including the callback-heavy
   ones.
#. **The architectural cleanness argument is weak.** Option A
   would unify register/memory writes with the already-proxy
   solver writes. But the diff-and-push asymmetry is local
   (lives in two extract functions), well-tested, and has not
   produced bug pressure.

**Reopen conditions.** Either option becomes attractive if:

* A future workload pushes callback density high enough to put
  sync-back into seconds. No bench in the regression matrix is
  anywhere near that.
* ``angr-2k64`` plugin-restoration work needs the
  proxy-write architecture as a prerequisite to retire the
  per-state-id plugin cache. In that case Option A becomes a
  dependency rather than an optimization, and the risk surface
  is paid down by the larger refactor.
* The mma_howtouse / hackcon residual gaps turn out to be in
  the diff-and-push path (not the case at design time — both were in
  ``state_create`` or Z3-shape mismatches, ``9maq-bisect-method``).

**What shipped** reversed the recommendation on the write side without
adopting either option's machinery: ``RustStateProxy`` register and
memory writes became direct write-through (``angr-j28e``), and at the
callback boundary memory moved to the proxy by default
(``angr-grji4``) once re-syncing whole pages per callback was measured
to cost more than the proxy writes do. Registers and constraints keep
diff-and-push, whose debug history is exactly the risk argument above.
The shipped contract is in :doc:`rust_engine` ("RustStateProxy
write-through contract").


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

  * ``angr/exploration/rust_state_sync.py`` (``_snapshot_registers``,
    ``_extract_register_changes``) — diff-and-push implementation.
  * ``angr/exploration/rust_callback_dispatch.py`` — sync_back call site.
  * ``angr/exploration/rust_state_proxy.py`` (``RustRegisterProxy``,
    ``RustMemoryProxy``) — the proxies that now write through.
  * ``native/angr/src/exploration/state_api.rs`` —
    ``_set_state_register_symbolic_ast`` and
    ``_state_memory_store_symbolic_multi``, the existing
    symbolic-write FFI shims that the proxy would reuse.

* Measurements: ``run_single.py --dump-counters`` per-bench output,
  recorded on ``angr-g7ug``.
* Prior cautionary memory: ``avoid-state-copy-optimization`` — a
  similar attempt to skip state.copy() broke register sync silently
  on ais3.
* Sibling design docs: :doc:`rust_lazy_memory_design`,
  :doc:`rust_parallel_design`.
