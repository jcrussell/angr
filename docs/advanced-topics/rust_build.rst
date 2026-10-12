Building, testing and debugging the Rust engine
===============================================

How to build the Rust extension from source, run its test and benchmark
gates, profile it, and read its logs. For what the engine *does*, see
:doc:`rust_engine`; for the design notes, see the pages it lists.

Building the Rust extension
---------------------------

The extension is built with **setuptools-rust** (not maturin) and lands
as ``angr/rustylib*.so``.

Prerequisites
~~~~~~~~~~~~~

* Python 3.12+ (the extension is built ``abi3-py312``).
* A Rust toolchain from rustup at ``~/.cargo/bin``; the version is pinned
  by the repo-root ``rust-toolchain.toml``.
* The ``z3-solver`` wheel pinned in ``pyproject.toml``. ``z3-sys`` ships
  pregenerated bindings, so no Z3 C headers or system Z3 are needed; the
  build links the wheel's ``libz3`` and must be told its version (see
  *Build layout* below). A system ``libz3-dev`` is harmless but unused.
* ``libclang``: ``bindgen`` generates the ``z3-sys`` bindings and, with
  the default-on ``libvex-ffi`` feature, the libVEX bindings.
  ``apt install libclang-dev`` / ``dnf install clang-devel`` /
  ``brew install llvm``; set ``LIBCLANG_PATH`` if it lives outside the
  default search path. Building with ``ANGR_LIBVEX_FFI=0`` still needs
  it for ``z3-sys``.
* A ``.venv/`` virtualenv holding the angr ecosystem (``claripy``,
  ``pyvex``, ``archinfo``, ``cle``). ``pyproject.toml`` pins those four
  to one version on purpose; the rationale is in the comment above the
  pins and in the ``avoid-pip-install-deps`` bd memory. Do not bump
  them casually.

Bootstrap from a clean state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt install libclang-dev rustup                        # or distro equivalent
   rustup show                                                 # picks up rust-toolchain.toml
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip setuptools setuptools-rust
   .venv/bin/pip install -e .                                  # pulls deps into venv
   .venv/bin/pip install -e . --no-build-isolation --no-deps   # rebuild the Rust .so

   # Verify
   .venv/bin/python -c "import angr; print(angr.__file__)"
   .venv/bin/python -m pytest tests/engines/rust/ --tb=short -q

If the venv exists but its ``pip``/``setuptools`` are broken, skip pip
entirely: ``make rebuild-cargo`` (``tools/rebuild-rust.sh --cargo-only``)
runs ``cargo build --release`` and copies the ``.so`` into ``angr/``. The
``venv-rebuild-cargo-direct-copy`` and ``env-venv-corruption`` bd
memories cover recovering a corrupted venv.

Incremental builds
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export PATH="$HOME/.cargo/bin:$PATH"
   source .venv/bin/activate
   make rebuild          # pip install -e . --no-build-isolation --no-deps
   make rebuild-fast     # [profile.release-fast] inner loop; NOT for bench gates
   make rebuild-cargo    # cargo build --release + copy; bypasses pip
   make check            # cargo check --release (default features only)
   make check-no-z3      # the four --no-default-features combos CI gates

``make check`` is **not** the CI gate: it compiles default features only,
so it never sees the ``fuzzer`` / ``fuzzing`` / ``libvex-ffi`` code. CI runs
``cargo clippy --all-targets --all-features -- -D warnings`` from the repo
root (no ``--manifest-path``, so the ``native/angr-macros`` test targets
are covered too). Neither of those compiles the
``#[cfg(not(feature = "vex-engine-z3"))]`` arms; ``make check-no-z3`` is the
only local check of those, which is why the ralph gate runs it first.

``native/angr/build.rs`` detects the Z3 shared library from the active
venv's ``z3-solver`` install and sets the runpath so Python and Rust load
the same ``libz3.so`` (required for AST passthrough; see
:doc:`rust_z3_sharing`). ``z3-sys`` would otherwise probe pkg-config
first and, failing that, assume its minimum-supported Z3 and hide newer
API, so ``setup.py::_configure_z3`` (pip path) and
``tools/rebuild-rust.sh`` (cargo path) export ``Z3_NO_PKG_CONFIG=1``,
``Z3_LIBRARY_PATH_OVERRIDE`` (the wheel's ``z3/lib``) and
``Z3_SYS_Z3_VERSION`` (from the installed ``z3-solver``) before cargo
runs. Export the same three before a bare ``cargo`` invocation.

Common issues
~~~~~~~~~~~~~

**"can't find Rust compiler"** — put ``~/.cargo/bin`` on ``PATH`` before
``pip install``.

**Stale .so** — ``make rebuild-clean`` (``tools/rebuild-rust.sh``) wipes
``angr/rustylib*.so`` and ``build/``, runs ``cargo clean``, and re-runs the
editable install. ``--keep-cargo-cache`` skips ``cargo clean``.

**"z3-sys: could not detect linked Z3 version"** (cargo warning) — the
build ran without ``Z3_SYS_Z3_VERSION`` and will assume the minimum
supported Z3. Export the three variables above (or go through
``make rebuild-cargo`` / ``pip install``, which set them).

**Linker can't find libz3 at runtime** — ``build.rs`` sets the runpath to
the venv's ``z3/lib``, so ``import angr`` works from inside the venv.
Outside it, set ``LD_LIBRARY_PATH`` to that directory or override with
``Z3_LIBRARY_PATH_OVERRIDE=/path/to/lib``.

Running tests
-------------

.. code-block:: bash

   make test                     # tests/engines/rust/ (the canonical fast tier)
   make test-libvex              # libVEX-FFI unit + corpus parity tests (--features libvex-ffi)
   make test-fuzzer              # icicle/libafl fuzzer unit tests (--features fuzzer)
   make test-python-baseline     # vanilla Python-engine regression subset (needs ../binaries)
   make test-full                # the whole angr suite (long)
   cargo test --profile release-checked   # Rust unit + integration suite with overflow-checks on

The live test count is ``grep -rc 'def test_' tests/engines/rust/ | awk
-F: '{s+=$2} END {print s}'`` or ``pytest --collect-only -q
tests/engines/rust/ | tail -1``; do not copy it into prose.

Benchmarks
~~~~~~~~~~

``tests/benchmarks/baseline_timings.json`` is the authoritative list of
tracked benchmarks and their Python/Rust timings (the ``*_branch`` arch
smoke benches have ``python_time=null`` by design). The corpus comes from
an ``angr/angr-examples`` checkout, found via ``ANGR_EXAMPLES_DIR`` or
``~/repos/angr-examples/examples``.

.. code-block:: bash

   make bench-regression                                    # fast tier, both engines
   python tests/benchmarks/run_regression.py --rust-only --skip-bimodal --threshold 0.15 --correctness   # the PR-time gate
   python tests/benchmarks/run_single.py sym-write --test    # one bench's own test() oracle under the Rust engine
   make bench-single EXAMPLE=fauxware ARGS="--both"         # one bench in a bounded subprocess
   python tests/benchmarks/run_single.py fauxware --dump-counters   # every mgr.stats counter, grouped
   python tests/benchmarks/run_single.py fauxware --counters-json > /tmp/base.json
   # ... change, rebuild ...
   python tests/benchmarks/run_single.py fauxware --counters-json > /tmp/curr.json
   python tests/benchmarks/bench_diff.py /tmp/base.json /tmp/curr.json   # counters by |delta|
   python tests/benchmarks/run_regression.py --full --rust-only --skip-bimodal --update-counters   # refresh baseline_counters.json

``--correctness`` runs each bench's own ``test()`` oracle under the Rust
engine before the timing pass and fails on a wrong answer; with
``--rust-only`` it is the only check of Rust's output.
``--skip-bimodal`` drops the ``BIMODAL_BENCHMARKS`` set defined in
``run_regression.py`` (see :doc:`rust_bimodal_variance`) from the timing
pass only; the correctness pass still runs them. ``--update``
rewrites ``baseline_timings.json`` except for entries in
``PINNED_RUST_TIMES``, which are deliberately pessimistic pins that a
single refresh must not demote; change a pin by editing that table so
the new value lands in a reviewable diff. The gate's decision rules
(relative threshold plus an absolute ``--regression-floor``, and
suite-median host normalization capped by ``--host-factor-cap``) are
documented in the ``run_regression.py`` module docstring, which is the
authority for them.

CI gates
~~~~~~~~

Each row names the workflow job and the script whose docstring holds
the details and tuning notes.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Gate
     - Where it runs
     - Script
   * - Bench correctness (each bench's ``test()`` under the Rust engine, bimodal benches included; ``--correctness``)
     - ``ci.yml::benchmark_regression`` (fast tier), ``nightly-ci.yml::benchmark_regression`` (``--full``)
     - ``tests/benchmarks/run_regression.py`` (``run_correctness_pass``)
   * - Benchmark regression (fast tier, ``--skip-bimodal``, 15 %)
     - ``ci.yml::benchmark_regression``
     - ``tests/benchmarks/run_regression.py``
   * - Benchmark regression (``--full``, bimodal benches tracked)
     - ``nightly-ci.yml::benchmark_regression``
     - same; auto-emits a ``bench_diff`` table against ``baseline_counters.json`` on a timing failure
   * - Find-all worker-count invariance (same found-set across ``RUST_PARALLEL_WORKERS``)
     - ``nightly-ci.yml::findall_worker_invariance`` / ``make gate-findall``
     - ``tests/benchmarks/run_findall_gate.py``
   * - RSS leak check (peak RSS ratio across N in-process runs)
     - ``nightly-ci.yml::rss_leak_check``
     - ``tests/benchmarks/run_leak_check.py``
   * - Valgrind leak check (memcheck slope on the pure-Rust ``leak_probe``)
     - ``nightly-ci.yml::valgrind_leak_check``
     - ``tests/benchmarks/run_valgrind_leak_check.py``
   * - SimOption / native-bridge parity census
     - ``nightly-ci.yml`` parity census job
     - ``tests/benchmarks/parity_census.py``
   * - Static source audits (line citations, silent fallbacks, SatResult, rounding mode, SP default, overflow, page shift)
     - ``ci.yml::rust_check``
     - ``tools/check_line_citations.py``, ``tools/audit_*.py`` — see the gates table in the repo-root ``AGENTS.md``

A gate that has never been seen to fail is indistinguishable from one
that cannot, so every static-audit script defines a ``--self-test`` mode
(in-script positive and negative fixtures fed through the real matcher)
and ``ci.yml::rust_check`` runs it ahead of the real check.
``grep -- --self-test .github/workflows/ci.yml`` lists the live set.

Profiling Rust benches
----------------------

``tests/benchmarks/profile_rust_bench.sh`` (``make profile-bench``) wraps
the criterion bench in ``native/angr/benches/vex_engine.rs`` with the best
available sampling profiler (cargo-flamegraph, then perf, then
callgrind), builds with frame pointers and full debuginfo, and writes
under ``target/profile/``. ``--list`` names the bench groups; ``--help``
has the rest, including the tool prerequisites.

Debugging Rust-side logs
------------------------

The Rust engine logs through a stderr logger backed by
``env_logger::filter::Filter`` (``native/angr/src/engine.rs``). Nothing is
printed until a level is set.

.. code-block:: python

   from angr.exploration.rust_manager import set_rust_log_level
   set_rust_log_level("debug")   # or "error" / "warn" / "info" / "trace" / "off"
   set_rust_log_level("rustylib::stash=warn,rustylib::exploration=info,off")

.. code-block:: bash

   RUST_LOG=debug python tests/benchmarks/run_single.py fauxware --engine rust
   ANGR_RUST_LOG=debug python ...          # RUST_LOG wins when both are set
   RUST_LOG="rustylib::stash=warn,off" python ...

* Rust modules appear under the ``rustylib`` crate (``rustylib::stash``,
  ``rustylib::exploration``), not ``angr::*``.
* ``RUST_LOG`` / ``ANGR_RUST_LOG`` are read once per process by
  ``_apply_rust_log_env`` in ``angr/exploration/rust_manager.py``, on the
  first manager construction; afterwards call ``set_rust_log_level``.
* Lines look like ``[rust:LEVEL] target: msg``.
