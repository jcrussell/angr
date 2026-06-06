Rust vs Python engine: flame-graph comparison
==============================================

.. _rust-engine-flame-comparison:

This page records the angr-trsg characterization experiment: where does
time actually go on two contrasting benches?

* **sym-write** — Z3-heavy CTF: 2.3× speedup on Rust (1.0 s Python →
  0.44 s Rust per ``baseline_timings.json``).
* **mma_howtouse** — callback-heavy MFC DLL: **0.65× speedup**, i.e.
  Rust is slower (4.25 s Python → 6.51 s Rust).

Hypothesis going in (per the bead): "Python flame is dominated by
interpreter / claripy overhead in the inner loop; Rust flame is
dominated by Z3." mma_howtouse should show why we lose on that bench.

The actual measurements only partially confirm the hypothesis — page
initialization in Python's paged memory mixin turns out to dominate
mma_howtouse under *both* engines.

Tooling note
------------

The reference plan called for ``cargo-flamegraph``/``py-spy``. Neither
was available in the run environment (offline build host, no cargo
install network access, broken venv pip blocks ``pip install py-spy``).
The substitute used is ``tests/benchmarks/profile_python_bench.py``,
which runs the bench's ``solve.py`` in-process under ``cProfile`` and
dumps three artifacts per (bench, engine) pair:

* ``PREFIX.pstats`` — binary ``pstats`` blob.
* ``PREFIX.txt`` — top-50 cumulative + tottime tables. Checked in here
  under :file:`docs/advanced-topics/perf/characterization/`.
* ``PREFIX.folded`` — collapsed-stack format. **Not** checked in
  (≈1.4 MB per file). Regenerate locally to render an SVG with
  Brendan Gregg's ``flamegraph.pl``:

  .. code-block:: shell

     python tests/benchmarks/profile_python_bench.py sym-write \
         --engine rust --cprofile --out /tmp/sym-write_rust
     flamegraph.pl /tmp/sym-write_rust.folded > /tmp/sym-write_rust.svg

Caveat: cProfile is a deterministic function-call profiler, not a
stack-sampling profiler. It accurately attributes Python frames and
PyO3 boundary entry points but cannot see *inside* Rust. The "flame"
rendered from a cProfile-folded file is effectively a two-level tree
of (caller, callee) edges, not the deep call stacks a sampling
profiler produces. Rust-internal hotspots show up as a single opaque
PyO3 method (e.g. ``RustExplorationManager.run`` for the explore loop,
``RustSolverContext.eval_upto`` for end-of-explore solve). Where this
matters the text is called out below.

Reading: sym-write
------------------

**Python engine (2.69 s wall, top tottime entries):**
Z3 C API calls dominate self-time —
``Z3_solver_check_assumptions`` 0.215 s, ``Z3_solver_get_param_descrs``
0.110 s, ``Z3_solver_assert`` 0.075 s, ``Z3_solver_get_model`` 0.031 s,
``Z3_solver_dec_ref`` 0.018 s. That's ≈0.45 s in Z3 (17 % of wall),
spread across ~600 ``check`` calls. The rest is import-time machinery
(``marshal.loads`` 0.136 s), claripy AST hashing
(``claripy.ast.base.__new__`` 0.061 s self / 0.233 s cumulative), and
angr's memory bookkeeping (``refcount_mixin.acquire_shared`` /
``release_shared``, ``picklable_lock`` enter/exit). Confirms the
hypothesis: many small Z3 round trips through the Python/Z3 ctypes
boundary, sandwiched by claripy AST creation. This is the workload the
Rust engine is built to win.

**Rust engine (1.85 s wall, top tottime entries):**
Two PyO3 calls account for the work the Python side did over hundreds
of calls — ``RustSolverContext.eval_upto`` 0.203 s (the single
post-exploration solve) and ``RustExplorationManager.run`` 0.051 s
(the entire explore loop, including all per-step Z3 work inside Rust).
The rest of the profile is dominated by import-time
(``marshal.loads`` 0.132 s, ``__build_class__`` 0.047 s) and angr's
own page-init bookkeeping (``page_backer_mixins.__getitem__`` 0.020 s,
``ultra_page.store`` 0.019 s, ``isinstance`` 0.020 s). Z3 self-time is
invisible to cProfile because it executes inside the Rust binary — the
``eval_upto`` line is the only window we get. The 2.3× speedup comes
from collapsing per-call Z3 / Python boundary crossings into a single
PyO3 call, not from Z3 itself being faster.

Reading: mma_howtouse
---------------------

**Python engine (8.18 s wall, top tottime entries):**
``page_backer_mixins._initialize_page`` 1.654 s self /
1.971 s cumulative — 24 % of wall in *one* function. The full
page-initialization stack
(``_initialize_page``, ``ultra_page.__init__``,
``_initialize_default_page``, ``_map_page``,
``stack_allocation_mixin._initialize_page``) accounts for ≈3.5 s of
8.2 s ≈ 43 % of wall. The bench creates ~92,700 ultra pages. Z3 self
calls do not show in the top-50 at all. Callbacks-heavy workload means
the inner loop is constructing fresh page state, not solving — so the
hypothesis "Python flame dominated by interpreter / claripy" partially
holds (claripy ``BVV`` and ``ast.base.__new__`` are present) but the
real story is page mapping.

**Rust engine (9.65 s wall, top tottime entries):**
Same page-init pattern, slightly heavier —
``page_backer_mixins._initialize_page`` 1.976 s, ``ultra_page.store``
0.783 s, ``page_backer_mixins.__getitem__`` 0.513 s. Plus two Rust-side
costs paid on top: ``RustExplorationManager.resume_after_simprocedure``
0.353 s (45 callback resumes) and ``rust_state_sync._sync_extra_python_pages``
0.106 s. The 1.6× slowdown vs Python comes from paying the *same*
Python-side page-init cost (because the SimProcedure callback round-trip
still happens in Python) **plus** the Rust → Python → Rust resume
overhead. Hypothesis falsified for this bench: Rust's profile is not
"dominated by Z3" — it's dominated by the *Python* engine's page-init
cost, plus the new Rust resume layer.

Headline finding
----------------

The Z3-dominated Rust flame predicted by the hypothesis only emerges
when the bench does many Z3 round trips per step and few state syncs
(sym-write). For callback-heavy benches (mma_howtouse) the bottleneck
is angr's Python-side page initialization — both engines pay it,
because SimProcedures run in Python — and the Rust engine adds a
small but non-trivial sync layer on top. This matches existing
memories ``proxy-writes-design-verdict`` and the ``9maq`` bisect work:
callback-heavy benches lose because each callback re-enters Python's
page-init stack.

Follow-on signal: a future optimization that eliminates Python-side
page re-initialization on SimProcedure resume (e.g. by keeping the
Python ``state`` cached in the page registry across the round trip)
would directly attack the bottleneck on mma_howtouse and similar
benches. No bead filed — the existing ``RustStateProxy.copy()`` CoW
fork bead (angr-d1dr) and the Python-state-cache work would jointly
address it.

Cross-link
----------

For the full architectural picture see :doc:`rust_engine`. For
per-bench variance handling see :doc:`rust_bimodal_variance`. The
artifacts on disk are at
:file:`docs/advanced-topics/perf/characterization/`.
