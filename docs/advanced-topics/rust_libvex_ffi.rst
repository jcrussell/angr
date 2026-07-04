libVEX FFI native lifting (Stage-1 feasibility)
===============================================

.. note::

   This documents the **Stage-1 feasibility verdict** for binding libVEX
   directly from Rust (bd ``angr-z087y``, the long pole of the zero-Python
   exploration epic ``angr-gorvf``). Stages 2 (flag-gated engine use) and 3
   (AMD64 default) are tracked separately and are gated on a bench win, not on
   feasibility. The runnable probe that produced this verdict is
   ``tools/probe-libvex-ffi.py``.

Motivation
----------

Cold VEX block lifts currently go
Rust → Python callback → pyvex (libVEX) → serialize ``IRSB`` to JSON → Rust
deserialize (``native/angr/src/vex/pyvex_bridge.rs``). At ~390 µs/lift corpus
average this is *mostly* GIL-acquire + JSON round-trip; libVEX itself is tens of
µs. ``lift_block`` is 86 % of all Rust→Python callbacks by count (3921/4584
across 32 corpus benches). It is a **per-unique-block warmup cost** — the warm
per-worker block cache already amortizes repeat lifts — so the ROI is
warmup-bounded and only matters for a long/wide workload where lift warmup
dominates, and for parallel warmup (it removes a GIL serialization point).

The key correctness argument: bind the **same libVEX** pyvex already uses,
rather than reimplementing lifting in Rust from scratch (enormous and
correctness-risky — the ``NativeVEXLifter`` stub in ``vex/lifter.rs`` exists for
exactly that reason). Same libVEX, same configuration ⇒ same ``IRSB``.

Verdict: GO on feasibility
--------------------------

All three Stage-1 unknowns resolve GO against the *installed* pyvex wheel
(``tools/probe-libvex-ffi.py`` asserts each and exits non-zero on any NO-GO):

1. **Link seam.** ``pyvex/lib/libpyvex.so`` (the shared object pyvex loads via
   cffi) exports both pyvex's exact-config shim ``vex_lift`` / ``vex_init`` **and**
   the raw ``LibVEX_*`` API (``LibVEX_Translate``, ``LibVEX_Lift``, …). A Rust
   FFI backend links this ``.so`` directly — no separate libVEX build.

2. **Interface source.** The wheel ships **no C headers**, but
   ``pyvex.vex_ffi.ffi_str`` (~44 KB) carries the *authoritative* cffi cdef:
   the ``vex_lift`` signature plus ``VEXLiftResult`` / ``IRSB`` / ``VexArchInfo``
   / ``ExitInfo`` struct layouts. This is the exact interface pyvex's own cffi
   uses to marshal libVEX output, so binding against it is parity-safe. It can
   be committed as a vendored ``.h`` for ``bindgen``, or the (small, stable)
   subset can be hand-declared.

3. **Parity.** The link target is the *same object file* pyvex loads in-process,
   so a native lift through ``vex_lift`` is byte-for-byte what the Python
   callback path produces today — no divergence risk from opt-levels, arch/endness
   options, or VEX guest/host setup, because those live inside ``vex_lift``.

The exact-config shim, from ``pyvex.vex_ffi.ffi_str``:

.. code-block:: c

   typedef struct _VEXLiftResult {
       IRSB* irsb;
       Int   size;
       Bool  is_noop_block;
       Int   exit_count;
       ExitInfo exits[400];
       Int   is_default_exit_constant;
       Addr  default_exit;
       Int   insts;
       Addr  inst_addrs[200];
       Int   data_ref_count;
       DataRef data_refs[2000];
       Int   const_val_count;
       ConstVal const_vals[1000];
   } VEXLiftResult;

   VEXLiftResult *vex_lift(
       VexArch guest, VexArchInfo archinfo,
       unsigned char *insn_start, unsigned long long insn_addr,
       unsigned int max_insns, unsigned int max_bytes,
       int opt_level, int traceflags,
       int allow_arch_optimizations, int strict_block_end,
       int collect_data_refs, int load_from_ro_regions, int const_prop,
       VexRegisterUpdates px_control, unsigned int lookback_amount);

Build/link recipe
-----------------

Resolve the pyvex lib dir the same way ``build.rs`` / ``setup.py`` already
resolve Z3 (probe the active venv's ``pyvex/lib``). ``tools/probe-libvex-ffi.py``
prints the resolved recipe for the current environment; the shape is:

.. code-block:: text

   cargo:rustc-link-search=native=<venv>/site-packages/pyvex/lib
   cargo:rustc-link-lib=dylib=pyvex
   cargo:rustc-link-arg=-Wl,-rpath,<venv>/site-packages/pyvex/lib

This **must be gated behind a non-default cargo feature** (e.g.
``libvex-ffi``) so the default build — and therefore the bench-regression gate,
which builds default features — takes on no new runtime dependency on
``libpyvex.so``'s rpath. ``setup.py`` gains a ``_resolve_pyvex_libdir`` helper
mirroring ``_resolve_z3_header`` to feed the search path when the feature is on.

Implementation plan (Stage-1 remainder)
---------------------------------------

The seam is the existing ``VEXLifter`` trait (``native/angr/src/vex/lifter.rs``);
a ``NativeLibVEXLifter`` slots in beside the pyvex-callback path. Steps:

1. Add the ``libvex-ffi`` cargo feature + ``build.rs`` link logic (above).
2. Declare the FFI surface (``vex_init``, ``vex_lift``, ``VEXLiftResult``,
   ``IRSB`` and its ``IRStmt``/``IRExpr`` reachable structs) from the vendored
   ``ffi_str`` cdef — prefer ``bindgen`` over the committed ``.h``.
3. Marshal the C ``VEXLiftResult->irsb`` into Rust ``vex::ir::IRSB`` — the same
   target shape ``pyvex_bridge.rs::PyVexIRSB`` documents, but read from the C
   struct instead of JSON (no serde round-trip).
4. AMD64 only for Stage 1; call ``vex_init()`` once per process (it is
   idempotent / guarded — see ``vex_initdone``).
5. **Parity harness:** dump the corpus unique-block set from the warm LRU cache
   and replay each block through both ``NativeLibVEXLifter`` and the
   pyvex-serialized path, asserting structural ``IRSB`` equality. Gate:
   100 % structural parity on the corpus block set.

Open risks (all inherited, none new after this verdict)
-------------------------------------------------------

* **Struct ABI drift:** the vendored cdef must track the pinned pyvex
  (``==9.2.209``); a pyvex bump can change ``VEXLiftResult`` array sizes or
  ``IRStmt`` layout. Regenerate from ``pyvex.vex_ffi.ffi_str`` on any bump and
  re-run the parity harness. The pin is frozen (see ``avoid-pip-install-deps``),
  so this is dormant.
* **``vex_lift`` uses libVEX's arena allocator:** the returned ``VEXLiftResult*``
  is owned by libVEX's temporary arena and is clobbered on the next lift — marshal
  it into owned Rust types *before* the next ``vex_lift`` call. This matches how
  pyvex's cffi copies out immediately.
* **Thread-safety:** libVEX global state (``vex_control``, the arena) is not
  re-entrant; a parallel backend needs a per-worker lift lock or one lifter
  thread. Relevant to the parallel epic, not to single-threaded Stage-1.
