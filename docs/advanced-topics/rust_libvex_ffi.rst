libVEX FFI native lifting
=========================

.. note::

   **Status: shipped and on by default for AMD64** (feasibility
   ``angr-z087y``, default flip ``angr-3trr7``, runtime binding
   ``angr-mx00p.33``). This page keeps the argument for why binding pyvex's
   own libVEX is parity-safe, how the binding is made, where the native lifter
   gets its bytes, how to run the parity tests, and the hazards of the
   default-on path.

Motivation
----------

Without native lifting, cold VEX block lifts go
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
past returning ``LiftError::Unsupported``, and was deleted for exactly that
reason). Same libVEX, same configuration ⇒ same ``IRSB``.

Why binding pyvex's libVEX is parity-safe
-----------------------------------------

1. **Same object.** ``pyvex/lib/libpyvex.so`` (``.dylib`` on macOS,
   ``pyvex.dll`` on Windows) is the shared object pyvex loads through cffi. It
   exports pyvex's exact-config shim ``vex_lift``. The engine calls that same
   symbol in that same loaded object, so there is one copy of libVEX and its
   globals in the process.

2. **Same interface.** pyvex's cffi cdef (``pyvex/vex_ffi.py``) is the
   authority for the ``vex_lift`` signature and the ``IRSB`` / ``VexArchInfo``
   struct layouts. Upstream's ``native/angr/src/ailment/vex_ffi.rs`` mirrors
   those layouts as ``#[repr(C)]`` structs, and ``tests/engines/rust/
   test_native_lift_parity.py`` checks the mirrors and the enum tags the
   native lifter hard-codes against pyvex's cffi ``sizeof`` / ``offsetof`` and
   library constants.

3. **Same configuration.** ``NativeLibVEXLifter::from_py_arch``
   (``native/angr/src/vex/libvex_lifter.rs``) builds the guest tag and the
   ``VexArchInfo`` from the project's ``arch.vex_arch`` and
   ``arch.vex_archinfo``, which is what ``pyvex.lifting.libvex.LibVEXLifter``
   passes, and calls ``vex_lift`` with angr's default lifter settings (opt
   level 1, ``strict_block_end`` off, ``cross_insn_opt`` on).

The shim, from pyvex's cdef:

.. code-block:: c

   VEXLiftResult *vex_lift(
       VexArch guest, VexArchInfo archinfo,
       unsigned char *insn_start, unsigned long long insn_addr,
       unsigned int max_insns, unsigned int max_bytes,
       int opt_level, int traceflags,
       int allow_arch_optimizations, int strict_block_end,
       int collect_data_refs, int load_from_ro_regions, int const_prop,
       VexRegisterUpdates px_control, unsigned int lookback_amount);

Binding at runtime
------------------

The extension has **no link-time dependency** on libpyvex. ``vex_ffi::
init_symbols`` (upstream, ``native/angr/src/ailment/vex_ffi.rs``) finds the
library next to the imported pyvex package, reopens it by path (``dlopen`` on
POSIX, ``LoadLibraryW`` on Windows — the library is already mapped, so this
only returns a handle) and resolves ``vex_lift`` against that handle.
``RustExplorationManager.set_native_lift(arch)`` triggers the resolution and
raises when the symbol is missing; there is no cargo feature, build-time
header, or rpath. pyvex runs libVEX's one-time ``vex_init`` at import, which
``from_py_arch`` guarantees by importing pyvex first. The link-time design
this replaced is in :doc:`rust_engine_decisions`.

Running the parity tests
------------------------

A lift needs the libpyvex an imported pyvex loads, which ``cargo test`` does
not have, so the native-vs-pyvex checks are pytest::

    python -m pytest tests/engines/rust/test_native_lift_parity.py

They lift every block of ``tests/engines/rust/native_lift_corpus.json``
through both paths and compare the resulting ``vex::ir`` trees, and check the
enum tags and struct layouts against pyvex's cffi library. The pure
marshalling helpers keep their unit tests in
``native/angr/src/vex/libvex_lifter_tests.rs``.

Open risks
----------

* **``vex_lift`` uses libVEX's arena allocator:** the returned ``VEXLiftResult*``
  is owned by libVEX's temporary arena and is clobbered on the next lift — the
  lifter marshals it into owned Rust types *before* releasing its lift lock.
  This matches how pyvex's cffi copies out immediately.
* **Thread-safety:** libVEX global state (``vex_control``, the arena) is not
  re-entrant. Native lifts are serialized by the Rust ``LIFT_LOCK`` and run on
  worker threads with the GIL released. pyvex serializes its own lifts with a
  Python lock, and cffi releases the GIL inside ``vex_lift``, so nothing yet
  excludes a native lift from a concurrent ``pyvex.lift`` on another thread
  (``angr-mx00p.45``).

Default-on on AMD64
-------------------

``RustExplorationManager(use_native_lift=...)`` defaults to **True**
(``angr-op0dn.2.2``). ``RustExplorationManager.__init__`` in
``rust_manager.py`` calls ``set_native_lift(project.arch)`` only when
``project.arch.name == "AMD64"``; that Python check is the gate that makes
native lift AMD64-only. The Rust lifter itself is wider:
``NativeLibVEXLifter::from_py_arch`` builds a lifter for any guest libVEX has.

It is on by default because it is parity-safe and never a loss: it binds
the same libVEX with the same configuration as pyvex (see *Why binding
pyvex's libVEX is parity-safe*), so the lifted ``IRSB`` and the
``found`` set do not change, and any block it cannot serve falls back
to the pyvex callback. What it buys is less GIL-held work per cold
block — lift-warmup-heavy binaries get faster, and parallel warmup loses
a serialization point even where single-threaded wall time does not
move. The per-bench evidence is on ``angr-op0dn.2.2``.

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
deadends, so nothing was lost by falling back. On a bench whose states
routinely return to ``0x0``, those deadend probes can dwarf the native lift
count and make the native path look like it rarely hits, while it is in fact
serving every real block. ``rust_native_lift_deadend_probe_count`` splits the
probes out; the genuinely-lost misses are
``fallback_count - deadend_probe_count``, and that is the number to watch.

Hazards of the default-on path
------------------------------

1. **Wheels must not vendor libVEX.** The extension resolves the *user's*
   installed pyvex at runtime and links nothing from it, so the wheel repair
   step has nothing of pyvex's to copy; a second libVEX in the process would
   have independent ``vex_control`` and arena globals and void the same-object
   parity argument. :doc:`rust_wheel_distribution` covers the repair recipe.
2. **Bench baselines must be measured on the native lift path.**
   ``baseline_timings.json`` is recorded with native lift on, so the fast-tier
   gate measures against native-lift timings; a baseline refreshed with
   ``--no-native-lift`` would hide a native-lift regression.
