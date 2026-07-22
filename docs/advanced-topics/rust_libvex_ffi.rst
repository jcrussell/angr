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
correctness-risky — an earlier from-scratch ``NativeVEXLifter`` stub never got
past returning ``LiftError::Unsupported``, and was deleted in Stage-1 close-out
for exactly that reason). Same libVEX, same configuration ⇒ same ``IRSB``.

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

Running the parity gate
-----------------------

The lifter unit tests and the corpus parity harness live *inside* the
``#[cfg(feature = "libvex-ffi")]`` module, and that feature is default-**off**
in ``native/angr/Cargo.toml`` (it is default-**on** in the shipped wheel via
``setup.py::_rust_features``). A plain ``cargo test`` therefore compiles
neither. Run them explicitly::

    make test-libvex
    # == cargo test --manifest-path native/angr/Cargo.toml --release \
    #        --features libvex-ffi --lib libvex

The build needs ``libpyvex.so``; ``build.rs::find_pyvex_lib_dir`` locates it by
shelling out to ``python3 -c "import pyvex"``, or you can point it at a
directory with ``PYVEX_FFI_LIB_DIR``. In CI this runs nightly as the
``libvex_ffi_tests`` job in ``.github/workflows/nightly-ci.yml`` (nightly rather
than PR-time because of the pyvex install + link cost); PR-time
``cargo clippy --all-features`` compiles the module but does not execute it.

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

Stage-3 verdict: AMD64 default-on (measured)
--------------------------------------------

``RustExplorationManager(use_native_lift=...)`` defaults to **True** as of bd
``angr-op0dn.2.2``. The flag is still double-guarded: it only reaches
``set_native_lift_enabled(True)`` when ``libvex_ffi_enabled()`` is true (i.e.
the ``.so`` was built ``--features libvex-ffi``) *and*
``project.arch.name == "AMD64"``. As of bd ``angr-3trr7`` (human GO
2026-07-15) that feature is built **by default** on ELF/Mach-O platforms, so
on a stock AMD64 build the native lifter is now live — see "Shipping status"
below.

Evidence (feature-on cargo build, ``run_single.py --native-lift`` vs
``--no-native-lift``, wall-clock over 3 reps unless noted):

.. list-table::
   :header-rows: 1

   * - bench
     - off (s)
     - on (s)
     - native / fallback lifts
     - GIL work
   * - ``codegate_2017-angrybird``
     - 4.33 / 4.39 / 4.42
     - 3.68 / 3.90 / 3.94
     - all native, 0 fallback
     - —
   * - ``ekopartyctf2016_rev250``
     - 2.53 / 2.63 / 2.75
     - 2.40 / 2.44 / 2.47
     - 221 / 0
     - 356 ms → 198 ms
   * - ``xmllint_getenv``
     - 3.83 / 3.84 / 3.86
     - 3.79 / 3.79 / 3.81
     - 135 / 0
     - 96 ms → 65 ms
   * - ``fauxware``
     - 0.24
     - 0.23–0.24
     - 14 / 0
     - 3.0 ms → 0
   * - ``mma_howtouse`` (callback-heavy)
     - 4.81–4.84
     - 4.81–4.93
     - —
     - —
   * - ``cow_fork_scaling`` (5 reps, min)
     - 2.62
     - 2.60
     - **21 / 256**
     - 193 ms → 182 ms
   * - ``google2016_unbreakable_1`` (bimodal)
     - 2.03–10.63
     - 2.27–3.16
     - —
     - within bimodal noise

Reading: the two lift-warmup-heavy real binaries win (``angrybird`` ≈ 13 %,
``rev250`` ≈ 7 %); everything else is flat; **nothing regresses**; ``found`` is
unchanged everywhere. The GIL-work drop (30–45 %) is the durable result and the
original ROI thesis — it removes a serialization point that matters for parallel
warmup even where it does not move single-threaded wall-clock.

Where native lift gets its bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``native_lift_source_bytes`` tries the memory sidecar first
(``read_concrete_bytes_for_lift``): the sidecar reflects stores, so
self-modifying code lifts the post-store program with no dirty-bytes plumbing.
A code page only lands in the sidecar once something faults it in, though, so a
**cold** block — the first block of a page, and every block of a
``load_shellcode`` blob — reads empty there. Those are served from the load-time
binary-region store (``_load_binary_regions``) instead of dropping to the pyvex
callback, *except* on a page a store has dirtied: the load-time image is stale
there, so such pages go to the callback, which passes fresh bytes through
``byte_string=``.

Reading ``rust_native_lift_fallback_count`` as a hit-rate gap is a trap. It
counts every miss, including lift attempts at addresses that lie outside every
loaded binary region — a state returning into unmapped memory, say. No lifter can
produce a block there: the callback returns the ``"{}"`` sentinel and the state
deadends, so nothing was lost by falling back. ``cow_fork_scaling`` reads 21
native lifts against 256 fallbacks, which looks like a 7% hit rate, but all 256
are return-to-``0x0`` deadend probes — the native path is serving **21 of 21**
real blocks. ``rust_native_lift_deadend_probe_count`` splits those out; the
genuinely-lost misses are ``fallback_count - deadend_probe_count``, and that is
0 across the fast-tier corpus.

Shipping status: on by default (opt-OUT via ``ANGR_LIBVEX_FFI=0``)
------------------------------------------------------------------

As of bd ``angr-3trr7`` (human GO 2026-07-15) the ``libvex-ffi`` cargo feature
is built **by default**: a stock ``pip install angr`` / ``pip install -e .`` on
an ELF (Linux) or Mach-O (macOS) platform links the native libVEX lifter, so the
wins measured above reach every stock AMD64 build. ``use_native_lift`` defaults
to True and is double-guarded on ``libvex_ffi_enabled()`` — now satisfied on a
stock ``.so`` — so cold blocks go out through the native lifter, falling back to
the pyvex callback only on a lift error or a non-AMD64 arch.

To opt **out** (the escape hatch) and restore the pyvex-callback lift path:

.. code-block:: bash

   ANGR_LIBVEX_FFI=0 pip install -e . --no-build-isolation --no-deps

``setup.py::_rust_features`` reads that env var and appends the cargo feature to
the extension declared by ``[[tool.setuptools-rust.ext-modules]]`` unless it is
set to a falsy value (``0``/``false``/``off``/``no``). The static ext-module
table cannot express a conditional feature, so the append happens in the
``build_rust`` subclass in ``setup.py``. It also degrades gracefully: if
``_resolve_pyvex_libdir`` found no ``libpyvex.so`` next to the installed pyvex
(e.g. Windows, where pyvex ships no shared object), the feature is dropped and
the build falls back to the callback path rather than failing.

The three hazards a default-flip had to answer (bd ``angr-x5kmr``), and how each
is resolved:

1. **Wheels would double-vendor libVEX.** ``build.rs`` emits
   ``-Wl,-rpath,<venv>/site-packages/pyvex/lib``. Under ``cibuildwheel`` the
   repair step would copy ``libpyvex.so`` into the wheel — the process would then
   hold *two* copies of libVEX with independent ``vex_control`` and arena
   globals, voiding the same-``.so`` parity argument. **Resolved:**
   ``.github/workflows/wheels.yml`` now excludes ``libpyvex.so`` from the repair
   step (``auditwheel --exclude libpyvex.so`` on Linux, ``delocate --exclude
   libpyvex`` on macOS) and relativizes the runpath to ``$ORIGIN/../pyvex/lib``
   (``@loader_path/../pyvex/lib`` on macOS), exactly mirroring the ``libz3``
   handling. The wheel therefore resolves the *user's* installed pyvex at
   runtime. Windows ships no ``libpyvex.so``, so that leg auto-degrades to the
   callback path and needs no repair change.
2. **The bench baselines were measured on the callback lift path.**
   **Resolved:** ``baseline_timings.json`` is refreshed in the same commit as the
   default flip, so the fast-tier gate measures against native-lift timings.
3. **New build-time dependency.** The feature pulls ``dep:bindgen``, i.e.
   libclang, on anyone building from source. **Accepted:** libclang is now a
   documented build-time dependency for source installs (the CI Linux/macOS
   legs already provide it for ``z3-sys`` bindgen). Set ``ANGR_LIBVEX_FFI=0`` to
   build without it.
