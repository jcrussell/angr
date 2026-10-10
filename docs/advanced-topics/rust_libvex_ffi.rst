libVEX FFI native lifting
=========================

.. note::

   **Status: shipped and on by default** (feasibility ``angr-z087y``,
   default flip ``angr-3trr7``). This page keeps the feasibility argument
   (why binding pyvex's own libVEX is parity-safe), the build and link
   recipe, where the native lifter gets its bytes, how to run the parity
   gate, and the hazards a default-on build had to answer. The runnable
   probe behind the feasibility verdict is ``tools/probe-libvex-ffi.py``.

Motivation
----------

Cold VEX block lifts currently go
Rust → Python callback → pyvex (libVEX) → serialize ``IRSB`` to JSON → Rust
deserialize (``native/angr/src/vex/pyvex_bridge.rs``). Most of that time is
GIL acquisition and the JSON round-trip; libVEX itself is a small fraction.
``lift_block`` is by far the most frequent Rust→Python callback on the
benchmark corpus. It is a **per-unique-block warmup cost** — the warm
per-worker block cache already amortizes repeat lifts — so the ROI is
warmup-bounded and only matters for a long/wide workload where lift warmup
dominates, and for parallel warmup (it removes a GIL serialization point).

The key correctness argument: bind the **same libVEX** pyvex already uses,
rather than reimplementing lifting in Rust from scratch (enormous and
correctness-risky — an earlier from-scratch ``NativeVEXLifter`` stub never got
past returning ``LiftError::Unsupported``, and was deleted in Stage-1 close-out
for exactly that reason). Same libVEX, same configuration ⇒ same ``IRSB``.

Why binding pyvex's libVEX is parity-safe
-----------------------------------------

All three feasibility unknowns resolve GO against the *installed* pyvex wheel
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

The link lives behind the ``libvex-ffi`` cargo feature. It is **default-off
in** ``Cargo.toml`` so a bare ``cargo build`` / ``cargo test`` takes on no
runtime dependency on ``libpyvex.so``'s rpath (which is why ``make
test-libvex`` passes the feature explicitly), and **default-on in**
``setup.py``: ``_rust_features`` appends it to the extension unless
``ANGR_LIBVEX_FFI=0``, with ``_resolve_pyvex_libdir`` (mirroring
``_resolve_z3_header``) feeding the search path. See *Shipping status*.

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

Open risks
----------

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

Default-on on AMD64
-------------------

``RustExplorationManager(use_native_lift=...)`` defaults to **True**
(``angr-op0dn.2.2``) and is double-guarded: it only reaches
``set_native_lift_enabled(True)`` when ``libvex_ffi_enabled()`` is true
(the ``.so`` was built with the feature) *and* the project arch is one
``ffi_vex_arch`` accepts. Measured on the benchmark corpus
(``angr-op0dn.2.2``), the lift-warmup-heavy real binaries gain a
noticeable fraction of wall time, everything else is flat, nothing
regresses and ``found`` is unchanged everywhere; the durable result is
the drop in GIL-held work, which removes a serialization point that
matters for parallel warmup even where single-threaded wall time does
not move.

Where native lift gets its bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

As of bd ``angr-3trr7`` the ``libvex-ffi`` cargo feature is built **by
default** by ``setup.py``: a stock ``pip install angr`` / ``pip install -e .`` on
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
