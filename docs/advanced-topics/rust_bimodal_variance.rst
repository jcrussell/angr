.. _rust-bimodal-variance:

Rust engine bimodal Z3 variance
===============================

Five benchmarks in ``tests/benchmarks/baseline_timings.json`` have
shown two distinct timing modes per run, driven by Z3
model nondeterminism. Their baselines are pinned at the slow mode and
they are tagged ``rust_only=True`` in ``tests/benchmarks/run_regression.py``
because the same nondeterminism causes output divergence vs. the
Python engine. The PR-time gate sets ``--skip-bimodal`` to keep CI
from flapping; nightly CI runs them. The live membership is the
``BIMODAL_BENCHMARKS`` frozenset in ``run_regression.py``.

``hackcon2016_angry-reverser`` joined the bimodal set on 2026-06-02
(``angr-bl0g``) after the ``angr-rbnk`` SignExt fix (commit ``4dc7fc064``)
collapsed the asserted constraint AST by ~13x and unmasked the
underlying Z3 SAT-search nondeterminism — see the 2026-06-02 entry
below.

``CADET_00001_partial`` joined on 2026-06-15 (``angr-027h``,
commit ``f10ff9c7b``) when the convergent phases-1+2 subset of the
upstream CADET solve was recorded as a synthetic-examples bench — its
easter-egg ``explore(find=)`` over symbolic stdin hits multi-solution
unconstrained jumps, splitting into two modes — see the 2026-06-15
entry below.

This page captures the 2026-05-13 variance campaign — 20 wall-clock
samples per benchmark, captured with
``tests/benchmarks/bimodal_variance.py``. The runner spawns
``run_single.py --engine rust`` per sample so each run inherits the
4 GB ``RLIMIT_AS`` sandbox.

2026-05-13 campaign
-------------------

Tool:

.. code-block:: console

   $ python tests/benchmarks/bimodal_variance.py --runs 20 --bin-width 1.0 \
     --json /tmp/bimodal_variance.json

``google2016_unbreakable_1`` — appeared unimodal 2026-05-13 (SUPERSEDED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   **Superseded 2026-05-22.** This bench *re-bimodalized* and is back in
   ``BIMODAL_BENCHMARKS`` at HEAD. See the
   `2026-05-22 — unbreakable_1 re-bimodalized`_ section below for the
   current classification. The 2026-05-13 / 2026-05-18 unimodal
   measurements are kept here for the timeline only.

20/20 runs in **3.01–3.06s** (median 3.03s, mean 3.03s, stdev 0.01s).
The earlier characterization had a 1.4–3.3s span. The fast mode is gone
in 2026-05-13 measurements; the benchmark has stabilized at what used to
be the slow mode.

.. code-block:: text

   histogram (1.0s bins):
     3.0–4.0s | 20 ####################

The current baseline (``rust_time = 3.5s``) absorbs the post-stabilization
run-to-run noise comfortably; 15% margin = 4.0s, well above the 3.06s
worst case.

``securityfest_fairlight`` — still bimodal, slow mode is now ~21s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

20 runs split 15 slow / 5 fast.

.. code-block:: text

   histogram (1.0s bins):
      7.0– 8.0s |  4 ##########
      8.0– 9.0s |  1 ##
     21.0–22.0s | 15 ########################################

   summary: min=7.94s median=21.39s max=21.53s mean=18.05s stdev=5.98s

The slow mode has drifted from the historical ~15s (recorded in bd
memory ``benchmark-bimodal-variance-rules``) to ~21.4s. The current
baseline ``rust_time = 16.0s`` no longer absorbs it — 15× of 20 runs
exceed even the +15% threshold (16.0 × 1.15 = 18.4s). Baseline has been
raised to ``22.0s`` to cover the slow mode plus headroom.

``ekopartyctf2016_sokohashv2`` — REGRESSED 2026-05-13, FIXED 2026-05-14
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3/3 confirming runs in 2026-05-13 failed deterministically at ~0.75s
with empty ``AssertionError``:

.. code-block:: text

   FAIL rust ekopartyctf2016_sokohashv2 0.75s: AssertionError:
   FAIL rust ekopartyctf2016_sokohashv2 0.74s: AssertionError:
   FAIL rust ekopartyctf2016_sokohashv2 0.73s: AssertionError:

Python engine still passes in 4.6s. The 0.75s wall-clock was too fast
for the exploration loop to have started, so the assert fired during
state setup or the first VEX block. Tracked as **angr-7vcx**.

The root cause was not x87 fallback as initially suspected. Two
Python↔Rust memory-sync bugs were uncovered during the investigation:

* **angr-ctct** (commit ``c6b2824cb``, 2026-05-14): ``_extract_symbolic_pages``
  missed filler-materialised symbolic values from
  ``SYMBOL_FILL_UNCONSTRAINED_MEMORY``.
* **angr-fv81** (commit ``13bb9f741``, 2026-05-14):
  ``_extract_from_ultrapage`` fallback truncated multi-byte symbolic
  entries to the head byte; now walks each entry's bitmap extent
  (capped at 64 bytes).

The PR-time gate ``--skip-bimodal`` excluded this benchmark, which is
how the regression reached HEAD without tripping CI. See the 2026-05-18
re-validation section below for current distribution.

Decisions
---------

Baseline tightenings landed in this campaign:

.. list-table::
   :header-rows: 1
   :widths: 35 20 20 25

   * - Benchmark
     - Old ``rust_time``
     - New ``rust_time``
     - Reason
   * - ``google2016_unbreakable_1``
     - 3.5s
     - 3.5s (unchanged)
     - Stable at 3.03s; 3.5s already gives ~15% headroom
   * - ``securityfest_fairlight``
     - 16.0s
     - 22.0s
     - Slow mode drift; 16s no longer covers actual runs
   * - ``ekopartyctf2016_sokohashv2``
     - 16.0s
     - 16.0s (unchanged)
     - Benchmark was broken in 2026-05-13 campaign; angr-ctct +
       angr-fv81 fixes shipped 2026-05-14. 2026-05-18 re-validation
       (below) confirms 16.0s still covers the slow mode (max 17.44s
       within +15% threshold)

The bimodal classification itself:

- ``ekopartyctf2016_sokohashv2`` — re-validated 2026-05-18 post-fix.
  Distribution has softened from clean bimodal to trimodal-ish (1 fast
  ~8.6s, 7 mid ~12s, 2 slow ~17s over 10 runs). Retained in
  ``BIMODAL_BENCHMARKS`` because the slow-mode tail still exceeds the
  PR-time threshold and the structural x87+Z3 sources remain.
- ``securityfest_fairlight`` — still bimodal in 2026-05-13 *and*
  2026-05-18 campaigns; structural floor confirmed (see
  re-validation section below).
- ``google2016_unbreakable_1`` — appeared unimodal in 2026-05-13
  measurements; **removed from** ``BIMODAL_BENCHMARKS`` on 2026-05-18.
  **This removal was reverted on 2026-05-22** when the bench
  re-bimodalized — see `2026-05-22 — unbreakable_1 re-bimodalized`_. It
  is back in ``BIMODAL_BENCHMARKS`` at HEAD.

2026-05-18 re-validation
------------------------

``google2016_unbreakable_1``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Second 20-sample campaign on 2026-05-18 (HEAD ``f54bbba93``,
angr-hyiz.4):

.. code-block:: text

   summary: n=20 (fail=0) min=2.43s median=2.46s max=2.48s mean=2.46s stdev=0.01s

   histogram (1.0s bins):
     2.0–3.0s | 20 ####################

20/20 runs in **2.43–2.48s** — even tighter than the 2026-05-13
campaign (3.01–3.06s). The benchmark has further sped up at the
median, likely from intervening Rust engine work (notably the
2026-05-18 ``f54bbba93`` lazy-zero-page cap, but the speedup is
broad and not pinned to a single commit). No slow mode reappears.

Decision (2026-05-18): removed from ``BIMODAL_BENCHMARKS`` in
``run_regression.py`` so the PR-time gate (``--skip-bimodal``) now covers
it. The baseline ``rust_time = 3.5s`` is left as-is — it gives ~42%
headroom over the 2.48s worst-case sample and ~15% above the 2026-05-13
3.06s sample, preserving room for natural drift without re-tightening on
a single session's measurement. (Per the
``avoid-update-baseline-without-verification`` memory, baselines should
not be tightened on optimistic single-session data; the headroom here is
intentional.)

.. note::

   **Superseded 2026-05-22.** This removal was reverted four days later
   when ralph iter-2 caught the bench at 5.21s and a 15-sample
   re-validation re-confirmed a multi-modal distribution. The bench is
   back in ``BIMODAL_BENCHMARKS`` (commit ``e30214e88``). See
   `2026-05-22 — unbreakable_1 re-bimodalized`_.

Cached ``python_time = 1.602s`` gives a current speedup of
~0.65x (1.602 / 2.46). The PR-time SLA gate (default fail at 0.5x,
warn at 1.0x) will print a WARN line for this benchmark and not fail.

``securityfest_fairlight``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Second 20-sample campaign on 2026-05-18 (HEAD ``607eed29e``,
angr-hyiz.2):

.. code-block:: text

   summary: n=20 (fail=0) min=7.97s median=21.55s max=22.72s mean=16.96s stdev=6.68s

   histogram (1.0s bins):
      7.0– 8.0s |  1 ###
      8.0– 9.0s |  6 #####################
     21.0–22.0s | 11 ########################################
     22.0–23.0s |  2 #######

20/20 runs OK (no AssertionError). Distribution remains strongly
bimodal: **7 fast (7.97–8.33s) / 13 slow (21.48–22.72s)**. Compared
to the 2026-05-13 campaign:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * -
     - 2026-05-13
     - 2026-05-18
     - Delta
   * - Fast / slow split
     - 5 / 15
     - 7 / 13
     - +2 fast
   * - Fast-mode range
     - 7.94–8.x s
     - 7.97–8.33s
     - stable
   * - Slow-mode range
     - 21.0–21.53s
     - 21.48–22.72s
     - +1.2s ceiling drift
   * - Median
     - 21.39s
     - 21.55s
     - +0.16s
   * - Stdev
     - 5.98s
     - 6.68s
     - widened ~0.7s

Conclusion: fairlight is **structurally bimodal** under Z3 model
nondeterminism — three runs across two months (2026-05-01, 2026-05-13,
2026-05-18) all reproduce two clearly-separated modes (~8s and ~21s).
The 13:7 mix this session is close enough to the 15:5 mix from
2026-05-13 to be noise-level; mode separation (~13s) dwarfs noise
inside each mode (<1s).

The slow-mode ceiling drifted slightly upward (21.53s → 22.72s). One
run (22.72s, sample 15) is above the current ``rust_time = 22.0s``
baseline but well within the 15% threshold (25.3s) used by
``run_regression.py``. Baseline is left at **22.0s**; per
``avoid-update-baseline-without-verification`` we don't tighten on a
single session, but 22.0s is also not loosened — drift fits inside
the existing margin.

Cached ``python_time = 15.756s`` against the **median** 21.55s gives
a current speedup of ~0.73x (slow mode) or 1.97x (fast mode, 15.756 /
7.99 mean). The blended-mean speedup is 0.93x. Per the SLA gate
defaults (fail < 0.5x, warn < 1.0x), this prints a WARN line and does
not fail.

Structural floor: as recorded in the ``fairlight-bottleneck`` bd
memory, Z3 ``check()`` accounts for
~95% of wall-clock time in this benchmark; the only known further
optimization (``LAZY_SOLVES``) gave ~0.65s upper bound (25x) in a
synthetic test but requires correctness proofs that have not been
attempted. Until that path is taken — or x87/bit-blasting overhead is
shifted out of Z3 entirely — fairlight is at the floor for this
engine, and the bimodal distribution is the floor's signature.

``ekopartyctf2016_sokohashv2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First post-fix campaign on 2026-05-18 (HEAD ``d5b6b1edd``,
angr-hyiz.5), 10 runs (vs 20 for the other two — sokohashv2 takes
~12s/run so 10 runs gives the same wall-clock budget):

.. code-block:: text

   summary: n=10 (fail=0) min=8.63s median=11.98s max=17.44s mean=12.69s stdev=2.65s

   histogram (1.0s bins):
      8.0– 9.0s |  1 ########
     11.0–12.0s |  5 ########################################
     12.0–13.0s |  2 ################
     17.0–18.0s |  2 ################

10/10 runs OK (no AssertionError) — the angr-ctct + angr-fv81 fixes
hold. Distribution has shifted from the historic clean bimodal split
(~9.5s OR ~15.4s) to a **trimodal-ish spread**:

* 1× fast (8.63s)
* 7× mid (5×~12.0s tightly clustered + 2× at 12.06s/12.x)
* 2× slow (17.18s, 17.44s)

Slow-mode max 17.44s stays within the +15% threshold for the 16.0s
baseline (18.4s), so the baseline is left unchanged. Median 11.98s
gives a current speedup of ~0.49x against cached ``python_time = 5.833s``;
the mean 12.69s gives ~0.46x, with the longer-tail mean pulling the
typical figure toward ~0.4x (see the per-bench table in
:doc:`rust_engine`).

Decision: **retained in** ``BIMODAL_BENCHMARKS``. The slow-mode tail
(17.x s) still exceeds 1× python time by ~3×, and the structural
sources (x87 transcendental fallback, Z3 nondeterminism) are unchanged.
The PR-time gate ``--skip-bimodal`` continues to exclude this benchmark
so a slow-mode run does not flake a PR. Root cause: ``sokohashv2-two-bugs``
bd memory; current timings in ``tests/benchmarks/baseline_timings.json``.

2026-05-22 — unbreakable_1 re-bimodalized
-----------------------------------------

Four days after the 2026-05-18 removal, the ralph iter-2 gate flagged
``google2016_unbreakable_1`` at **5.21s** against its 3.5s baseline. A
15-sample re-validation on HEAD ``14187073d`` found the bench had **not**
stayed unimodal — it had reorganized into a multi-modal distribution
with a fast mode well below the old median and a re-emergent slow tail:

.. code-block:: text

   15-sample distribution (HEAD 14187073d):
     0.91–0.99s | 11   (fast mode)
     1.15s      |  2   (mid)
     1.86s      |  1   (slow)
     2.65s      |  1   (slow tail)
   plus the gate sample at 5.21s extending the tail further

The fast mode (~0.97s median) is ~3.6× faster than the 2026-05-18
median (2.46s). Post-May-18 perf gains — likely angr-b58a (UltraPage
memcmp + lazy-region FFI), angr-zdho (z3_ast cache instrumentation), and
angr-9jly (proxy fast-path) — appear to have *widened* the gap between
the fast and slow Z3 modes rather than collapsing them. SLA at the fast
mode would be 1.65× (1.602s Python / 0.97s Rust median) — a new best if
not for the slow tail.

Spike conclusion (category c per the taxonomy): the slow tail is Z3
SAT-heuristic nondeterminism, not an actionable Rust bug. The
``BIMODAL_BENCHMARKS`` gate is the correct mitigation.

Decisions:

- **Re-added to** ``BIMODAL_BENCHMARKS`` in commit ``e30214e88`` (spike
  angr-pfy4 / re-add bead angr-ja0i), reverting the 2026-05-18 removal so
  ``--skip-bimodal`` excludes it from the PR-time gate again; the nightly
  gate continues to track drift.
- ``rust_time`` baseline left at **3.5s** — it covers the fast mode and
  the bulk of the distribution; the rare 5.21s tail is exactly the kind
  of Z3-nondeterministic spike ``--skip-bimodal`` exists to absorb.
- See :doc:`rust_engine` ("Other benchmarks below 1.0x") for the full
  per-bench table entry.

2026-06-02 — hackcon2016_angry-reverser joins the bimodal set
-------------------------------------------------------------

``hackcon2016_angry-reverser`` had been a 0.6x outlier with tight
variance (5-sample median 14.84s, 10-sample 30.79s ±1.33s) prior to
2026-06-01. Spike ``angr-rbnk`` (commit ``4dc7fc064``) replaced the
hand-rolled ``BVOp::SignExt`` loop in
``native/angr/src/symbolic/value.rs`` with the z3-rs native
``BV::sign_ext`` call. SMT-LIB dump shrinks: 740,257 → 54,221 total
chars across the 58 final assertions (13.7x smaller; 8x smaller than
Python's). Extract nodes dropped 7,158 → 449; concat nodes dropped
9,600 → 0.

8-sample post-fix wall-clock campaign (HEAD ``4dc7fc064``):

.. code-block:: text

   sorted: 8.97, 18.66, 22.39, 22.68, 26.24, 28.08, 29.49, 34.88s
   median ~22.5s
   fast tail ~9s matches Python's solve time
   slow tail ~35s on Z3 SAT-search nondeterminism

The simpler AST gives Z3 more branch-choice freedom: distribution
widens but median improves and the fast tail finally matches Python.
Category (c) per the taxonomy — the slow tail is structurally Z3
SAT-heuristic nondeterminism, not an engine bug.

Decisions:

- ``rust_time`` baseline raised 18.0s → 35.0s to cover the observed
  slow tail (max 34.88s, threshold gives 40.25s ceiling). Per the
  ``avoid-update-baseline-without-verification`` invariant the 35s
  ceiling does not include extra headroom beyond the threshold's
  built-in margin.
- Added to ``BIMODAL_BENCHMARKS`` so ``--skip-bimodal`` excludes it
  from the PR-time gate; the nightly gate continues to track drift.
- See bd memory ``hackcon-z3-ast-structure`` for the full spike
  writeup including the construction-site analysis.

2026-06-15 — CADET_00001_partial joins the bimodal set
------------------------------------------------------

``CADET_00001_partial`` was added to ``baseline_timings.json`` as the
convergent phases-1+2 subset of the upstream CADET_00001 solve
(``angr-027h``, commit ``f10ff9c7b``). The upstream three-phase
``solve.py`` is not a viable end-to-end Rust bench — its phases need
mutually exclusive manager configs and the phase-3 step-loop is
pathologically heavy (see the ``CADET_00001`` catalog note in
``run_single.py`` and bd memory ``benchmark-cadet-phase3-not-a-bench``).
The recorded subset runs phase 1 (buffer-overflow
step-until-unconstrained) plus phase 2 (the easter-egg
``sm.explore(find=0x804833E)``) over symbolic stdin.

Phase 2 is where the bimodality comes from: the easter-egg explore
hits multi-solution unconstrained jumps, so Z3 model nondeterminism
shifts which crashing input is materialized and therefore the work
the explore does. A 4-sample campaign clustered into two modes:

.. code-block:: text

   fast mode ~6.8s / 411 MB peak RSS
   slow mode ~8.2s / 545 MB peak RSS

Category (c) per the taxonomy — structurally Z3 model
nondeterminism on a multi-solution explore, not an engine bug.

Decisions:

- ``rust_time`` baseline pinned to the slow mode (8.24s, 545 MB) so the
  PR/nightly timing gate tolerates both modes. ``python_time`` is 11.8s
  (Rust ~1.4x faster at the pinned slow mode; ~1.7x at the fast mode).
- Added to ``BIMODAL_BENCHMARKS`` so ``--skip-bimodal`` excludes it from
  the PR-time gate; the nightly gate continues to track drift. The
  ``callback_count`` / ``state_creations`` / ``steps`` count baselines
  are all 0 for this explore-based bench, so the ``--check-counts`` gate
  is a no-op on it (it is therefore not also listed in ``COUNT_EXEMPT``).
- See bd memories ``benchmark-cadet-phase3-not-a-bench`` and
  ``benchmark-cadet-two-phase-eager-retry`` for the convergence history.

Reproducing
-----------

The campaign takes ~12-13 minutes for the three pre-existing bimodal
benchmarks combined (unbreakable_1 ~3s × 20, fairlight ~13s mean × 20,
sokohashv2 ~12s mean × 10). Add 60s overhead. ``hackcon2016_angry-reverser``
adds ~5 minutes for 20 samples at the post-fix ~22s median.

.. code-block:: console

   $ python tests/benchmarks/bimodal_variance.py \
       --runs 20 \
       --benchmarks securityfest_fairlight google2016_unbreakable_1 \
       --bin-width 1.0 \
       --timeout 180 \
       --json /tmp/bimodal_variance.json

The ``--json`` option dumps the raw timing samples for later analysis.
Per-bench summary lines (``summary: n=20 min=… median=… …``) and a
1s-bin ASCII histogram are printed to stdout.

Related memories
----------------

- ``benchmark-bimodal-variance-rules`` — consolidated rules for the four
  bimodal benches and how the gate handles them (supersedes two pruned
  historical-figure memories from the 2026-06 memory prune).
- ``pr-bench-gate-jitter-risk`` — context for why ``--skip-bimodal``
  exists in the PR-time gate.
