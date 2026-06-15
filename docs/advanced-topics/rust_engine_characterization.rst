Rust engine characterization suite
==================================

.. _rust-engine-characterization:

Aggregation page for the ``angr-kvn0`` characterization epic. The
suite measures *behavior under load* — fork scaling, fleet-level
memory footprint, where time goes — on workloads chosen to probe
specific architectural claims, complementary to the wall-clock
``baseline_timings.json`` benchmarks.

Tone is engineering report: the studies call out where the Rust
engine wins, where it loses, and where the hypothesis going in did
not hold. The point is measurement data future contributors can
extend, not a marketing pitch.

This page is a v1 partial aggregation. The xmllint section (``.2``,
``angr-6d3l``) is a stub pending the upstream xmllint bench
(``angr-75mc``, blocked on the optional ``fuzzer`` Cargo feature
decision). Full closure of ``angr-p65i`` waits on that section
landing.

At a glance
-----------

.. list-table:: Characterization studies summary
   :header-rows: 1
   :widths: 8 22 16 32 22

   * - Slice
     - Hypothesis
     - Verdict
     - Headline
     - Bead / writeup
   * - ``.1``
     - Python deep-copies state on fork (hockey-stick memory);
       Rust ``im::OrdMap`` linear.
     - **Partial.** Both engines linear, not hockey-stick.
       Per-state cost gap is the real story.
     - Rust ~15× faster CPU/state, ~2× lower RSS/state.
       Python OOMs at N=9 (2⁹ leaves) under 4 GB; Rust completes.
     - ``angr-uf0g`` — see
       :file:`tests/benchmarks/characterization/cow_fork_scaling/README.md`.
   * - ``.2``
     - xmllint same script across two engine flags; identical output
       expected.
     - *(pending)*
     - Blocked on ``angr-75mc`` xmllint bench landing.
     - ``angr-6d3l``.
   * - ``.3``
     - Rust's lower per-process Python+Z3 footprint → lower
       aggregate RSS at same throughput.
     - **Confirmed, modestly.**
     - Per-process RSS ~12 % lower on Rust (~206 MB vs ~231 MB
       on fauxware). 10–20 % faster per-process wall.
       Aggregate-peak 2.80 GB Rust vs 3.26 GB Python at conc=16.
     - ``angr-ayrq`` — see
       :file:`tests/benchmarks/characterization/fleet_resource_profile/README.md`.
   * - ``.4``
     - Python flame = Z3 + claripy; Rust flame = Z3.
     - **Partially confirmed.**
     - sym-write fits the model. mma_howtouse is dominated
       by Python page-init under *both* engines — driven by 45
       ``Callable`` invocations, *not* the Rust resume layer (re-measured
       2026-06-15: callback path 8.4 ms of wall).
     - ``angr-trsg`` — see :doc:`rust_engine_flame_comparison`.

.. _rust-engine-characterization-cow-fork:

.1 CoW fork scaling (angr-uf0g)
-------------------------------

A parameterised C binary with ``N`` independent symbolic ``if`` arms
on stdin produces ``2^N`` leaf states. Sweeping ``N`` and recording
wall time + peak RSS for each engine shows how Rust's
``im::OrdMap``-backed state fork scales against angr's Python
``SimState`` copy.

**Hypothesis:** Python deep-copies on fork (hockey-stick memory);
Rust uses ``im::OrdMap`` structural sharing (linear-ish).

**Result:** *Both* engines scale linearly in leaf-state count, not
exponentially. Python does not exhibit a hockey-stick curve — the
shared ``SimMemory`` page-level CoW handles the fork cost on the
Python side. The interesting delta is in **per-state cost**:

.. list-table:: CoW fork scaling: per-state steady-state
   :header-rows: 1
   :widths: 16 30 30

   * - Engine
     - Per-state CPU
     - Per-state RSS
   * - Rust
     - ~10 ms/state
     - ~3.7 MB/state
   * - Python
     - ~150 ms/state
     - ~7.2 MB/state

Rust is ~15× faster per state and uses ~2× less RAM per state. Both
grow roughly linearly with leaf-state count. Python OOMs inside the
Z3 SAT solver before exploration completes at N=9 under a 4 GB cap;
Rust completes the same workload using ~1.9 GB.

Reproduction harness and raw CSV:
:file:`tests/benchmarks/characterization/cow_fork_scaling/`. Full
N=2..9 sweep captured 2026-06-03 on the ralph host.

.. _rust-engine-characterization-xmllint:

.2 xmllint drop-in comparison (angr-6d3l — pending)
---------------------------------------------------

**Status:** stub. Blocked on ``angr-75mc`` adding xmllint to
``baseline_timings.json``. xmllint is the only real-world utility
binary in angr-examples — heavy libc, real file I/O, deep call
graphs — the closest thing to a real workload in the corpus.

The upstream ``angr-examples/examples/xmllint/solve.py`` uses the
optional ``angr.rustylib.fuzzer`` (icicle-fuzzing, libafl) pipeline
rather than symbolic execution, so a non-fuzzer alternative
``solve.py`` is required before the comparison can run. See
``angr-75mc`` for the feature-gate decision (enable the ``fuzzer``
Cargo feature in the default build vs. write an alternative
symbolic-exec ``solve.py`` with a defined goal).

When this slice lands, the section should document: wall time +
peak RSS per engine, identical-output property, and any divergence
in solver fallback (xmllint is the most promising candidate for a
nonzero ``syscall_python_fallback_count`` on the existing corpus —
the round-1 syscall beads ``angr-8j16`` / ``angr-6009`` /
``angr-aig2`` would gain a measurement surface).

.. _rust-engine-characterization-fleet:

.3 Fleet-level resource profile (angr-ayrq)
-------------------------------------------

Spawns N independent ``fauxware`` solves as separate subprocesses
across a capped concurrency, samples ``VmRSS`` across all live
workers, and aggregates wall + peak RSS. The sampler walks
``/proc/<root>/task/<root>/children`` recursively to attribute RSS
through ``multiprocessing.spawn`` grandchildren.

**Hypothesis:** Rust's lower per-process Python+Z3 footprint means
lower aggregate RSS at the same throughput.

**Result:** confirmed, modestly. Per-process steady-state RSS is
~12 % lower on Rust (~206 MB vs ~231 MB for fauxware). Per-process
wall time is 10–20 % shorter on Rust, so aggregate-wall savings
compound on top of aggregate-RSS savings as concurrency rises.

.. list-table:: Fleet aggregate at conc=16
   :header-rows: 1
   :widths: 16 22 32 30

   * - Engine
     - Per-proc peak (MB)
     - Aggregate-peak (MB)
     - Wall (s)
   * - Rust
     - 206
     - 2 804
     - 5.09
   * - Python
     - 231
     - 3 260
     - 6.05

Caveats: at fauxware-class workloads the per-process Python +
angr-import cost dominates; heavier symbolic-fork workloads
(see ``.1`` above) widen the gap because Rust's ``im::OrdMap`` CoW
state grows more slowly than Python's ``SimMemory``. Aggregate RSS
scales linearly with concurrency under both engines — no shared-page
collapse, no superlinear blowup. Past ``nproc`` per-process wall
roughly doubles; optimal scheduling point is concurrency ≈ nproc.

Reproduction harness and raw CSV:
:file:`tests/benchmarks/characterization/fleet_resource_profile/`.

.. _rust-engine-characterization-flame:

.4 Flame-graph comparison (angr-trsg)
-------------------------------------

cProfile-based reading of two contrasting benches: ``sym-write``
(Z3-heavy, 2.3× speedup) and ``mma_howtouse`` (callback-heavy,
0.65× speedup — Rust *slower*). Full per-bench reading lives in
:doc:`rust_engine_flame_comparison`; the headline is reproduced
here.

**Hypothesis:** Python flame is dominated by interpreter / claripy
overhead in the inner loop; Rust flame is dominated by Z3.
mma_howtouse should show why Rust loses on that bench.

**Result:** the hypothesis holds for sym-write but is **falsified**
for mma_howtouse. On sym-write, Python pays ~17 % of wall in Z3 C
API calls across hundreds of round trips, while Rust folds the
equivalent work into two PyO3 calls — exactly the workload the
Rust engine is built to win. On mma_howtouse, both engines pay
~24 % of wall in a single Python function
(``page_backer_mixins._initialize_page``) and ~43 % across the
full page-init stack.

**Re-measurement (2026-06-15, post-d1dr) corrects the mechanism.**
The driver is *not* SimProcedure resumes. ``mma_howtouse``'s
``solve.py`` runs a :class:`~angr.callable.Callable` **45 times**;
``angr/callable.py:88 (Callable.__call__)`` is 88 % of cProfile
wall and each invocation calls ``factory.call_state()``
(``angr/factory.py:164``, 5.4 s cumulative), which builds a *fresh*
state. Lazy page backing then re-initializes ~2,060 ultra pages
per call — **92,700** ``_initialize_page`` calls total, with 1.96 s
*tottime* concentrated in ``DictBackerMixin._initialize_page``
(``page_backer_mixins.py:240``), which linearly scans the
``_dict_memory_backer`` once per page. Both engines pay this
identically. The SimProcedure callback path is negligible here:
``callback_count=1`` and the entire callback/resume path measures
8.4 ms of a 5.94 s run — ``resume_after_simprocedure`` is no longer
a measurable layer for this bench.

**Implication (corrected).** The earlier "45 callback resumes /
cache the Python state across SimProcedure round-trips" framing was
wrong: the 45 are ``Callable`` invocations, not SimProcedure
callbacks, and the page re-init lives in ``factory.call_state()``,
not the Rust resume layer. The true lever — reusing the
``call_state`` base or memoizing CLE page backing across
``Callable`` invocations — is a general angr-core optimization
shared by *both* engines; it would lower absolute time for both
equally and so would **not** change the 0.65× ratio. It is outside
the rust-symex engine's scope. The residual Rust-vs-Python gap on
this bench comes from per-invocation execution/manager overhead
across the 45 runs, not page-init. Tracked as ``angr-dxpd`` (closed
with this finding). The ``RustStateProxy.copy()`` CoW fork bead
``angr-d1dr`` (closed, commit ``be05cf9f8``) built the Rust-side
CoW machinery but addressed proxy ``copy()`` semantics, *not* this
page-init path.

Tooling caveat:
``cargo-flamegraph`` / ``py-spy`` were unavailable in the offline
build host. The substitute is
``tests/benchmarks/profile_python_bench.py`` running each
``solve.py`` under cProfile. cProfile is a deterministic
function-call profiler and cannot see *inside* Rust — Rust-internal
hotspots show up as a single opaque PyO3 method (e.g.
``RustSolverContext.eval_upto`` for end-of-explore solve). See
:doc:`rust_engine_flame_comparison` for the workaround details and
artifact paths.

Related work outside the kvn0 epic
----------------------------------

The deep-input-loop characterization (``angr-xel4``,
``tests/benchmarks/characterization/deep_loop_search/``) followed
the same template — synthetic proxy binary, hypothesis-driven
sweep, write-up — but was filed under the grub-OOM line
(``angr-34w.12``) rather than under ``angr-kvn0``. Its finding —
DFS reaches deeper at lower memory than BFS for grub-class
binaries — is documented inline in :doc:`rust_engine`
("Deep-input-loop binaries (grub-class)" subsection) and pinned by
the ``TestExplorationStrategy.test_deep_loop_recipe_dfs_plus_length_limiter``
regression test. Listed here for navigability; future characterization
work that matches the template should consider filing under
``angr-kvn0`` for centralized discoverability.

Cross-link
----------

For the architectural picture see :doc:`rust_engine`. For
per-bench variance handling see :doc:`rust_bimodal_variance`. The
flame-graph artifacts live at
:file:`docs/advanced-topics/perf/characterization/`.
