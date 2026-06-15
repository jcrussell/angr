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
* ``pyo3 = "0.27.2"`` is configured **without** the ``abi3`` feature, so today's
  build is CPython-version-specific (``cpython-312``).

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

* **abi3 (preferred if it compiles):** add the ``abi3-py310`` feature to the
  ``pyo3`` dependency. One ``rustylib.abi3.so`` then covers 3.10–3.13+, cutting
  the build matrix to a single wheel per platform. Risk: the engine uses
  ``features = ["py-clone"]`` and a broad PyO3 surface; some APIs are
  unavailable under the limited ABI. This must be **compile-verified** before
  committing to abi3.
* **Per-version matrix (fallback):** ``cibuildwheel`` builds one wheel per
  CPython tag. More artifacts, no PyO3-surface risk. Use this if abi3 fails to
  build.

Prototype CI job
----------------

``.github/workflows/wheels.yml`` (added alongside this memo) is a **prototype**
``cibuildwheel`` job encoding the decision above:

* Builds in the ``manylinux`` image with ``libz3-dev`` + ``pkg-config``
  installed (headers are build-time only; they never enter the wheel).
* ``z3-solver`` is installed in the build environment so ``build.rs`` resolves
  ``libz3.so``.
* A custom ``CIBW_REPAIR_WHEEL_COMMAND_LINUX`` runs ``auditwheel repair
  --exclude libz3.so`` then ``patchelf --set-rpath '$ORIGIN/../z3/lib'``.
* A test step installs the repaired wheel into a clean venv and runs
  ``make test-quick``.

It is committed **unverified** — this environment has no Docker/network to run
``cibuildwheel``. The verification is filed as a blocker bead (see below).

Build-from-source fallback
--------------------------

Until the wheel job is verified, source build remains the supported path. It is
documented in ``CLAUDE.md`` → "Building the Rust Extension" (prerequisites,
bootstrap-from-clean, and the ``tools/rebuild-rust.sh`` recovery flow). No change
needed there; this memo only adds the *binary-distribution* option on top.

Open blockers (filed as beads)
-------------------------------

#. Verify the ``wheels.yml`` job actually builds, repairs (``--exclude
   libz3.so`` + ``$ORIGIN/../z3/lib`` runpath), and that the installed wheel
   passes ``make test-quick`` in a clean venv **in CI** (cannot run locally —
   no Docker/network).
#. Decide abi3 vs per-version: add ``abi3-py310`` to ``pyo3`` and confirm the
   crate still compiles with the current PyO3 surface (``py-clone`` + all
   ``#[pymethods]``). If it fails, keep the per-version matrix.
#. Confirm AST passthrough actually works through a repaired wheel (the whole
   point): a clean-venv smoke test that mints a Z3 AST in ``claripy`` and reads
   it back through the Rust engine, proving both resolved the **same**
   ``libz3.so`` via ``$ORIGIN/../z3/lib``.
