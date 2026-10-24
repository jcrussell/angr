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

   **Further superseded for memory (angr-grji4), registers and
   constraints (angr-mx00p.112).** At the **SimProcedure callback
   boundary** (Python full ``SimState`` mutates → Rust syncs back on
   return) nothing takes the diff-and-push path described below any more.
   Every callback state's ``state.memory`` is a ``RustMemoryProxy``, its
   ``state.registers`` a ``RustRegisterProxy`` and its ``state.solver`` a
   ``RustSolverProxyPlugin``, so a Python SimProcedure's
   ``state.memory.store(…)``, ``state.regs.rax = …`` or
   ``state.solver.add(…)`` lands in Rust immediately and nothing is
   replayed on return (``angr-mx00p.28``). See :doc:`rust_engine` for the shipped
   contract, and *Copy semantics* below for what ``state.copy()`` does.

   **And for exported states (angr-mx00p.61).** A ``SimState`` the manager
   hands out is built the same way: its memory, call stack and solver are
   proxies of the Rust state, and nothing is written back into Python
   plugins at export time. The user contract is :doc:`rust_engine`
   "Exported states are live views"; *Copy semantics* below covers the
   bindings behind it.

   The "proposed proxy-plugin design" section is now
   historical reading — the proxy uses an immediate write-through
   model, not the queued-mutation plugin substitution this doc
   compared against.

This document compares designs for eliminating the callback-boundary
diff-and-push (``angr-g7ug``).

Audience: contributors deciding whether to replace the existing
SimProcedure callback boundary with a proxy-plugin design.


Why this work
-------------

When a Python SimProcedure runs at a hook, the Rust engine cedes
control across the PyO3 boundary and the SimProcedure mutates an
angr ``SimState`` via the standard plugin API (``state.regs.rax = …``,
``state.memory.store(…)``, ``state.solver.add(…)``). When the
SimProcedure returns, those mutations have to be reflected in the
Rust canonical state. The design-time mechanism was a diff-and-push pass
(every helper named here has since been deleted):

#. **Before** the callback ran, the manager snapshotted register
   values into a lightweight dict (or, for memory-writing procedures,
   took a full ``state.copy()``).
#. The Python SimProcedure mutated the snapshotted state directly
   through angr's plugin paths.
#. **After** the callback returned, a register diff walked every named
   register, compared it to the snapshot, and emitted
   ``(offset, size, data_bytes)`` tuples; a memory diff walked
   ``new_state.memory.changed_bytes(old_state.memory)`` to collect
   concrete and symbolic deltas; new constraints were extracted by
   length-comparing the constraint list.
#. The aggregate diff was pushed back through
   ``resume_after_simprocedure(addr, reg_changes, mem_changes,
   new_constraints)``.

That worked (Rust stayed canonical because every delta was reflected)
but cost O(num_registers) per callback plus the changed-bytes scan over
any modified pages, plus per-symbolic-value AST re-export.

What replaced it: the callback state's ``memory``, ``registers``,
``solver`` and ``callstack`` plugins are Rust proxies on one binding
(``_create_state_for_callback`` in ``rust_callback_dispatch.py``). A
write lands in the Rust state when it is made, a ``state.copy()`` forks
its own Rust state on its first write (*Copy semantics*), and the resume
carries only the next pc plus the heap break and posix output the
proxies do not cover (``_sync_state_heap_to_rust``,
``_sync_state_posix_to_rust``).

The hypothesis under test was that we could eliminate the
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
callback-heavy benches is state creation, not sync-back (``angr-9maq``).


Current diff-and-push model
---------------------------

Nothing at the callback boundary takes it any more (see the status note
above): the register, memory and constraint diff helpers and the pre-call
snapshot they diffed against are deleted, and a resume carries only the
next pc. The options below are written against the full diff-and-push
model that existed at design time.


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

* **Eliminates the diff scan.** Removes the register and memory
  diffs from the callback hot path entirely, recovering the whole per-callback
  sync-back cost measured above.
* **Aligns with the already-proxy solver path.** Constraints
  already write through immediately via ``_rust_add``. Registers
  and memory becoming write-through closes the asymmetry.
* **Plugin restoration may be unblockable.** Option A interacts with
  the per-state-id plugin cache (``angr-2k64``): if posix.stdin
  reads/writes go directly to Rust, the cache may be redundant.
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
  at a time. The diff-and-push path grouped contiguous changed bytes
  into regions.
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
plus the interaction with per-state-id plugin restoration (``angr-2k64``).

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
  into a per-callback list (memory had such a channel, a wrapped
  ``state.memory.store``, until the memory proxy replaced it).
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
   correctness regressions. A prior similar attempt to skip
   ``state.copy()`` for extern SimProcedures left the register diff
   comparing the state against itself (the procedure mutates it in
   place), so register sync broke silently and regressed ais3 badly
   before it was caught; skipping the copy was only safe with a
   register snapshot taken before execution.
   The diff-and-push path's debug history is worth a
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
* Per-state-id plugin restoration (``angr-2k64``) needs the
  proxy-write architecture as a prerequisite to retire the
  per-state-id plugin cache. In that case Option A becomes a
  dependency rather than an optimization, and the risk surface
  is paid down by the larger refactor.
* The mma_howtouse / hackcon residual gaps turn out to be in
  the diff-and-push path (not the case at design time — both were in
  ``state_create`` or Z3-shape mismatches, per a bisect; ``angr-9maq``).

**What shipped** reversed the recommendation on the write side without
adopting either option's machinery: ``RustStateProxy`` register and
memory writes became direct write-through (``angr-j28e``), and at the
callback boundary memory moved to the proxy by default
(``angr-grji4``) once re-syncing whole pages per callback was measured
to cost more than the proxy writes do. Registers followed once a copied
callback state got its own Rust state (*Copy semantics*, ``angr-mx00p.112``)
and the register and solver proxies measured no slower on the bench gate;
the monkey-patched callback solver went with them.
The shipped contract is in :doc:`rust_engine` ("RustStateProxy
write-through contract").


Copy semantics
--------------

A SimState whose plugins are Rust proxies (``RustMemoryProxy``,
``RustRegisterProxy``, ``RustSolverProxyPlugin``,
``RustCallStackProxyPlugin``) must copy like an angr state: the copy reads
the source as it was when copied, and a write to either side reaches
neither the other nor a sibling copy. A proxy therefore holds a
``RustStateBinding`` (``angr/exploration/rust_state_binding.py``) rather than
a raw state id. ``SimState.copy()`` hands every proxy plugin of the copy the
same new binding through the copy memo, so the copy's memory, registers and
solver keep naming one Rust state (``angr-mx00p.112``).

* **Lazy.** A copied binding is *virtual*: it reads through its parent and
  costs nothing until something writes. A write through a virtual binding
  materializes it with ``RustSimState::fork`` (``fork_state_to_stash``) of
  what it reads; a write through a binding with virtual children first moves
  them onto a snapshot fork. A callback that never copies its state, or
  copies it without writing, forks nothing.
* **Owners.** Who mints and releases forks is the binding's owner. A
  ``CallbackForkScope`` owns the bindings of one SimProcedure, syscall or
  Python-VEX-fallback callback state; its forks live in the ``_cb_forks``
  stash. The lazy read-through is exact there because Rust does not step the
  pending state while Python runs the callback. Outside a callback the
  manager's ``CopyStashForkOwner`` owns them: forks go to ``_copies`` and are
  dropped when their binding is garbage collected, the lifetime
  ``RustStateProxy.copy()`` already has.
* **Successors.** When a callback returns, the dispatcher's
  ``_settle_callback_successors`` (``rust_callback_dispatch.py``) gives each
  successor its own Rust state before anything writes the pending state
  directly. The first successor resumes the pending callback: either it
  still reads the pending state unchanged and simply becomes it, or the fork
  it owns is adopted into the pending slot
  (``adopt_pending_callback_state``). Every other successor is finished from
  its own fork by ``resume_forked_successor``, which shares the resume
  payload logic with the pending state's own resume, so its constraints are
  the pending state's plus its own guard, never a sibling's
  (``angr-vmi2l.294``).
* **Release.** Every binding no successor claimed is released when the
  successors settle, and the scope drops its fork; whatever is still parked
  when the handler exits is dropped by ``CallbackForkScope.close``. A read
  or write through a released binding raises ``RustStateReleasedError``; it
  never falls through to another state's data.

Exported states take the same mechanism with a different owner, the
manager's ``CopyStashForkOwner``: ``_bind_export`` (``rust_state_export.py``)
gives an export's proxies one root binding, so ``found.copy()`` is a virtual
child that forks on its first write. What the callback scope gets for free,
a parent that does not change behind the copy's back, this owner provides
itself, because Rust steps and drops stash states without going through any
binding:

* **Before Rust steps.** ``_invalidate_state_export_cache``, which every
  Python entry into the run loop (``step``, ``run``, ``explore``) already
  calls, runs ``CopyStashForkOwner.detach_views``. A root that gained
  virtual copies since the last call shelters them on a snapshot fork, and
  a snapshot export (``found_states``, ``get_state_by_id``) moves onto a
  fork of its own. The owner learns of copies as they are made
  (``on_copy``), so a step with no new copies forks nothing. Before that,
  ``_detach_held_entries`` (``rust_state_export.py``) pops every stash entry
  (``_LazySimStateRef``) from ``_lazy_state_refs`` and probes it through a
  weak reference. The wrapper holds only the manager and an id, so one
  nobody else references is freed by the pop and costs nothing; a live one
  is pinned to its SimState, which moves onto a fork
  (``CopyStashForkOwner.detach``), while the cache gets a fresh copy bound
  to the stash state for the next read. The probe is on the wrapper because
  a SimState sits in reference cycles: whether anything besides the cache
  holds it cannot be told without a ``gc`` pass. The hook is on the Python side because the bindings are Python objects
  and Rust only ever steps between those entry points; within a run, Python
  code only sees callback states, whose copies the callback scope settles.
* **Lifetime.** An export binding holds the Python manager
  (``_keepalive``), so an export keeps its manager and Rust state alive.
  States the manager drops on request (``drop()``, a merge) first pin their
  held entries the same way and move their other exports onto forks
  (``_preserve_exports``, ``CopyStashForkOwner.preserve``). When ``step`` or
  ``explore`` returns, ``release_gone`` releases every export root whose
  state Rust dropped while it ran, so a read raises
  ``RustStateReleasedError`` instead of failing on a missing id.
* **Detaching.** A proxy's ``callstack`` push/pop and ``memory`` merge are
  stubs, so a state Python steps or merges itself gets plain plugins first
  (``_detach_export_proxies``: the snapshot's pages, a ``CallStack`` of the
  Rust frames, and a Python solver that queries Rust through
  ``RustSolverFallback``). The ``step_state`` hook path, the Python merge
  path and ``_add_rust_state`` (an export handed back to a manager) do so,
  and the Python engine does it on every copy it steps
  (``SuccessorsMixin.process`` → ``RustMemoryProxy.detach_for_python_engine``
  → ``_detach_for_python_engine``, whose solver takes the Rust path
  constraints because the successors never return to Rust). The hook skips
  a callback state: the Python-VEX fallback steps it precisely so that its
  writes reach the pending Rust state (angr-vmi2l.260).

References
----------

* Per-state-id plugin restoration (``angr-2k64``) would interact with
  Option A's plugin substitution.
* On Callable-heavy benches the dominant cost is state creation, not
  sync-back (``angr-9maq``).
* Inspect marshalling (``angr-uq4n.2``) emits events at the callback
  boundary; Option A would emit at each write.
* Code involved:

  * ``angr/exploration/rust_callback_dispatch.py`` — the resume paths
    (``_resume_with_state`` and siblings) that once carried the diff.
  * ``angr/exploration/rust_state_proxy.py`` (``RustRegisterProxy``,
    ``RustMemoryProxy``) — the proxies that now write through.
  * ``angr/exploration/rust_state_binding.py`` — the copy-on-write
    bindings behind them (*Copy semantics* above).
  * ``native/angr/src/exploration/state_api.rs`` —
    ``_set_state_register_symbolic_ast`` and
    ``_state_memory_store_symbolic_multi``, the existing
    symbolic-write FFI shims that the proxy would reuse.

* Measurements: ``run_single.py --dump-counters`` per-bench output (``angr-g7ug``).
* Prior caution: a similar attempt to skip state.copy() broke
  register sync silently on ais3 (see the risk-asymmetry item under
  *Recommendation at design time, and what shipped*).
* Sibling design docs: :doc:`rust_lazy_memory_design`,
  :doc:`rust_parallel_design`.
