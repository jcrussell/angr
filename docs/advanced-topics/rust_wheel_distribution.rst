Prebuilt wheel distribution for the Rust engine
================================================

.. note::

   This is a **design memo** (bd bead ``angr-ivwn``). It records the decision
   for how a prebuilt manylinux wheel of the Rust engine *must* link against
   Z3, why the obvious ``auditwheel`` default is wrong here, and what remains to
   be verified in CI. The build-from-source path documented in the repo
   ``CLAUDE.md`` ("Building the Rust Extension") stays the supported install
   method until the wheel job below is verified green.

Problem
-------

Installing angr with the Rust engine today requires, at ``pip install`` time, a
Rust toolchain, the Z3 development headers (``libz3-dev`` / ``z3-devel`` /
``brew install z3``), and ``pkg-config``. The build is a setuptools-rust source
build (``pyproject.toml`` ``build-backend = "setuptools.build_meta"``). The PyPI
``z3-solver`` wheel ships ``libz3.so`` but **not** the C headers, which
``z3-sys`` needs to generate bindings. No CI job produces wheels
(``.github/workflows`` has only ``ci``, ``coverage``, ``nightly-ci``,
``perf-dashboard``), so the "drop-in Python API" goal effectively stops at
developers who can compile a Rust + Z3 extension. A prebuilt wheel would close
that gap.

The shared-Z3-context constraint (the crux)
--------------------------------------------

The Rust engine and Python's ``claripy`` **must load the same** ``libz3.so`` at
runtime. AST passthrough hands raw Z3 ``Ast`` pointers across the FFI boundary;
a pointer minted in one ``Z3_context`` is meaningless in another. ``claripy``'s
context comes from the ``z3-solver`` wheel's ``libz3.so``; the Rust ``cdylib``
must resolve ``NEEDED libz3.so`` to that **same file**, not a private copy.

This is what makes the standard manylinux recipe wrong. ``auditwheel repair``
*by default vendors every non-system shared library into the wheel* (copied into
``angr.libs/`` with a hash-mangled SONAME). If we let it vendor ``libz3.so``,
the Rust extension would load its **own** private Z3 — a second
``Z3_context`` distinct from ``claripy``'s — and AST passthrough would silently
corrupt or crash. **Bundling libz3 is therefore forbidden**, even though it is
the normal manylinux behaviour.

Observed facts (this checkout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``z3-solver`` (4.13.0.0, pinned transitively via ``claripy==9.2.209``)
  installs ``site-packages/z3/lib/libz3.so`` with ``SONAME = libz3.so``
  (unversioned — no ``.so.4.13`` suffix).
* The built extension ``angr/rustylib.cpython-312-x86_64-linux-gnu.so`` carries
  ``NEEDED libz3.so`` and a ``RUNPATH`` pointing at the **build machine's**
  absolute venv path (``.../site-packages/z3/lib``) — set by
  ``native/angr/build.rs::find_z3_lib_dir`` (prefers the active venv's ``z3``
  package). That absolute RUNPATH is correct for an editable dev install but
  **non-portable** in a distributed wheel.
* ``pyo3 = "0.27.2"`` carries the ``abi3-py310`` feature (added 2026-06-15,
  angr-6f0i), so the distributed wheel is a single limited-ABI artifact covering
  CPython 3.10+ rather than one per minor version. The editable dev ``.so`` is
  still copied to a versioned name by ``tools/rebuild-rust.sh``, so local
  development is unaffected.

Decision: link, do not bundle
-----------------------------

The only wheel layout compatible with the shared-context constraint:

#. **Exclude** ``libz3.so`` from the wheel: ``auditwheel repair --exclude
   libz3.so`` (or ``--exclude libz3.so.*``). The wheel ships **no** copy of Z3.
#. **Relativize the RUNPATH** of ``rustylib*.so`` to the ``z3-solver`` wheel's
   install location. ``rustylib`` lives at ``site-packages/angr/`` and the Z3
   library at ``site-packages/z3/lib/``, so the portable runpath is
   ``$ORIGIN/../z3/lib``. ``auditwheel`` does not know about ``z3-solver``'s
   layout, so set this explicitly with ``patchelf --set-rpath
   '$ORIGIN/../z3/lib' angr/rustylib*.so`` **after** ``auditwheel repair``
   (or as the repair step's effect, verified with ``readelf -d``).
#. **Declare the runtime dependency** so ``site-packages/z3/lib/libz3.so``
   always exists: the wheel already depends (transitively, via the pinned
   ``archinfo/claripy/cle/pyvex==9.2.209`` ecosystem) on ``z3-solver==4.13.0.0``.
   Keep that pin; do **not** widen it. See the ``avoid-pip-install-deps`` bd
   memory — bumping past 9.2.209 risks a ``libz3.so`` SONAME/ABI change that the
   unversioned ``NEEDED libz3.so`` cannot guard against.

Why the unversioned SONAME is both convenient and dangerous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NEEDED libz3.so`` (no version) means the dynamic linker accepts **any**
``z3-solver`` version that ships a ``libz3.so`` — there is no SONAME version
check to catch an ABI break. ABI compatibility is only *actually* guaranteed for
the pinned 4.13.0.0. The wheel's pinned ``z3-solver`` dependency is the only
guard; treat it as load-bearing, not cosmetic.

abi3 vs per-CPython matrix
--------------------------

Two ways to cover supported interpreters (CPython 3.10+):

* **abi3 (CHOSEN, 2026-06-15, angr-6f0i):** the ``abi3-py310`` feature on the
  ``pyo3`` dependency. One ``rustylib.abi3.so`` then covers 3.10–3.13+, cutting
  the build matrix to a single wheel per platform. The risk — that the engine's
  ``features = ["py-clone"]`` plus its broad ``#[pymethods]`` surface might touch
  an API unavailable under the limited ABI — was **compile-verified clean**:
  ``cargo clippy --all-targets`` and a full ``--cargo-only`` rebuild both
  succeed, the limited-ABI ``.so`` imports, and ``tests/engines/rust`` passes.
  ``wheels.yml`` builds a single ``cp310-manylinux_x86_64`` (abi3) wheel.
* **Per-version matrix (fallback, NOT used):** ``cibuildwheel`` builds one wheel
  per CPython tag. More artifacts, no PyO3-surface risk. Kept documented in case
  a future PyO3-surface addition breaks the limited ABI.

Prototype CI job
----------------

``.github/workflows/wheels.yml`` (added alongside this memo) is a **prototype**
``cibuildwheel`` job encoding the decision above. It is ``workflow_dispatch``-only;
the tag trigger stays commented out until at least one leg is verified green
(blocker #1 below). Since 2026-07-14 (angr-c4xcs.3) it is a **platform matrix**
(``fail-fast: false`` — the legs are independent and none has run yet):

.. list-table::
   :header-rows: 1
   :widths: 22 14 22 42

   * - Leg
     - Runner
     - Repair tool
     - Runpath after repair
   * - ``manylinux x86_64``
     - ubuntu-latest
     - ``auditwheel`` + ``patchelf``
     - ``$ORIGIN/../z3/lib``
   * - ``manylinux aarch64``
     - ubuntu-latest + QEMU
     - ``auditwheel`` + ``patchelf``
     - ``$ORIGIN/../z3/lib``
   * - ``macos arm64``
     - macos-14
     - ``delocate`` + ``install_name_tool``
     - ``@loader_path/../z3/lib``
   * - ``windows x86_64``
     - windows-2022
     - *none (deliberate)*
     - n/a — ``os.add_dll_directory`` at import time

Common to every leg: the Rust toolchain and the Z3 **headers** are installed at
build time only (never enter the wheel); ``z3-solver==4.13.0.0`` is installed in
the build env so ``build.rs::find_z3_lib_dir`` resolves the library; libz3 is
never vendored into the wheel; and ``CIBW_TEST_COMMAND`` runs the AST-passthrough
smoke test against the finished wheel. The three ELF/Mach-O legs achieve
"never vendored" through a repair step that excludes libz3 and relativizes the
runpath; Windows, which has no runpath, achieves it by not repairing at all (see
below).

Per-platform wrinkles worth knowing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **aarch64** reuses the x86_64 recipe verbatim under QEMU emulation (slow —
  budget roughly an hour). It rests on one unverified assumption: that
  ``z3-solver==4.13.0.0`` publishes a manylinux **aarch64** wheel. If it does
  not, ``CIBW_BEFORE_BUILD_LINUX`` fails loudly, which is the right outcome — an
  aarch64 angr wheel that excludes libz3 is useless to a user who cannot
  ``pip install z3-solver`` on aarch64 either.
* **macOS** needs three changes from the ELF recipe, all encoded in
  ``CIBW_REPAIR_WHEEL_COMMAND_MACOS``: ``delocate-wheel --exclude libz3`` in
  place of ``auditwheel``; ``install_name_tool -change`` to rewrite the
  build-machine absolute libz3 path ``build.rs`` recorded as the dependent
  install name into ``@rpath/libz3.dylib``, plus ``-add_rpath
  @loader_path/../z3/lib``; and a ``codesign --force --sign -`` re-sign, because
  ``install_name_tool`` invalidates the ad-hoc arm64 signature and an unsigned
  ``.so`` will not load at all on Apple silicon. ``Z3_SYS_Z3_HEADER`` must be set
  via ``CIBW_ENVIRONMENT_MACOS``, not exported from the before-build hook —
  cibuildwheel runs that hook in its own shell, so the export never reaches cargo.

Windows: same invariant, different mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PE has **no RPATH equivalent**, so step 2 of the decision above — relativize the
runpath to the ``z3-solver`` package — has no mechanism on Windows, and the
standard repair tool (``delvewheel``) would copy ``z3.dll`` into the wheel, which
is exactly the private-second-``Z3_context`` failure this whole design exists to
prevent. Windows is therefore the one leg with **no repair step at all**
(``CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: ""``, deliberately empty). The search path
is established at *import* time instead:

.. code-block:: python

   # angr/misc/z3_dll.py, called from angr/__init__.py before anything
   # imports angr.rustylib
   os.add_dll_directory(<site-packages>/z3/lib)

**Why that resolves the same DLL claripy loads.** Since CPython 3.8, extension
modules are loaded with ``LoadLibraryEx`` under ``LOAD_LIBRARY_SEARCH_*``, whose
search set is exactly the directories registered via ``os.add_dll_directory``
(plus the DLL's own directory and the system dirs). claripy's ``z3`` bindings
``ctypes``-load their ``z3.dll`` out of ``<site-packages>/z3/lib`` — the very
directory registered above — so the extension's import-time resolution lands on
that same file. One directory, one DLL, one ``Z3_context``. The shim is a no-op
off Windows and a no-op when ``z3-solver`` is not installed (in which case the
extension's own import failure is the more informative error).

Two supporting changes fall out of this:

* ``build.rs::emit_rpath`` skips the ``-Wl,-rpath`` link-arg when
  ``CARGO_CFG_TARGET_OS == "windows"`` — MSVC's ``link.exe`` rejects the flag
  outright, so without the guard the leg would not even link.
* The Windows build env has no header or import-library source at all: the
  ``z3-solver`` wheel ships ``z3.dll`` and nothing else. ``CIBW_BEFORE_BUILD_WINDOWS``
  fetches the *matching* official z3 4.13.0 release archive for its ``include/``
  and ``libz3.lib``, both build-time only. z3-sys asks the linker for ``z3``
  (i.e. ``z3.lib``) while the archive names it ``libz3.lib``, hence the copy —
  that rename is the most likely thing to break when the leg first runs.

Every leg is committed **unverified** — this environment has no Docker/network to
run ``cibuildwheel``. Verification is filed as a blocker bead (see below).

Build-from-source fallback
--------------------------

Until the wheel job is verified, source build remains the supported path. It is
documented in ``CLAUDE.md`` → "Building the Rust Extension" (prerequisites,
bootstrap-from-clean, and the ``tools/rebuild-rust.sh`` recovery flow). No change
needed there; this memo only adds the *binary-distribution* option on top.

Open blockers (filed as beads)
-------------------------------

#. Verify the ``wheels.yml`` job actually builds, repairs (libz3 excluded +
   relativized runpath), and that the installed wheel passes the smoke test in a
   clean venv **in CI** (cannot run locally — no Docker/network). Filed as
   angr-3gjm for the x86_64 leg; the aarch64 and macOS legs added by angr-c4xcs.3
   inherit the same gate — none of the three has ever been executed, so a green
   run of *any* leg is what unlocks the tag trigger.
#. **RESOLVED (2026-06-15, angr-6f0i):** abi3 vs per-version decided in favour
   of abi3 — ``abi3-py310`` added to ``pyo3``, crate compiles clean against the
   full PyO3 surface (``py-clone`` + all ``#[pymethods]``), limited-ABI ``.so``
   imports and passes ``tests/engines/rust``. ``wheels.yml`` collapsed to a
   single abi3 wheel.
#. **RESOLVED (2026-06-15, angr-9eit):** the AST-passthrough smoke test lives at
   ``tests/smoke/wheel_ast_passthrough.py`` and is wired into ``wheels.yml``'s
   ``CIBW_TEST_COMMAND`` (``python {project}/tests/smoke/wheel_ast_passthrough.py``).
   It mints a Z3 AST in ``claripy`` and reads it back through the Rust
   ``RustSolverContext`` (constraints + ``min``/``max``/``eval`` + an UNSAT
   contradiction), proving both halves resolved the **same** ``libz3.so`` via
   ``$ORIGIN/../z3/lib``. The script runs standalone (exits nonzero on failure)
   and is also collected by pytest for local verification. Note the *execution*
   of this test inside the cibuildwheel container is still gated by the Docker/CI
   verification in blocker #1 (angr-3gjm) — the test is authored and locally
   green, but has not yet run against an actually-repaired wheel.
