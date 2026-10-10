.. _rust-bimodal-variance:

Rust engine bimodal Z3 variance
===============================

A handful of benchmarks in ``tests/benchmarks/baseline_timings.json``
show two (sometimes three) distinct timing modes per run. The cause is
Z3 model nondeterminism: the solver's variable- and restart-ordering
heuristics pick a witness, that witness decides which path the engine
explores next, and one path is far more expensive than the other. The
engine is not slower on these benches; a single-shot timing of them is
simply not a measurement. The live membership is the
``BIMODAL_BENCHMARKS`` frozenset in ``tests/benchmarks/run_regression.py``;
do not copy the list into prose.

How the gate handles them
-------------------------

* Their ``rust_time`` baselines are **pinned at the slow mode**, so the
  regression threshold absorbs both modes; a pin that a refresh must
  not demote lives in ``PINNED_RUST_TIMES`` (see the ``run_regression.py``
  docstring).
* They are ``rust_only=True``: the same nondeterminism makes their
  output diverge from the Python engine, so output parity is not
  checked.
* The PR-time gate passes ``--skip-bimodal`` so CI does not flap; the
  nightly run keeps them, tracking drift without expecting stability.
* ``COUNT_EXEMPT`` is a separate frozenset: the bimodal benches whose
  ``--check-counts`` baselines are non-zero are exempted from count
  comparison, while a bimodal bench with all-zero count baselines needs
  no exemption because the count check is a no-op for it.

Adding or removing a bench
--------------------------

Three places have to move together, and each has drifted alone before:

1. ``BIMODAL_BENCHMARKS`` in ``run_regression.py`` (and the slow-mode
   pin in ``baseline_timings.json`` / ``PINNED_RUST_TIMES``).
2. ``COUNT_EXEMPT`` membership, decided by whether the bench has
   non-zero count baselines — and the comment above ``COUNT_EXEMPT``,
   which must enumerate members rather than count them.
3. This page, if the bench's variance has a *new* structural cause
   worth recording; do not add a dated per-bench entry.

Known causes
------------

Every member so far traces to one of two shapes:

* **Hash-cracker constraint sets** (the ``unbreakable`` / ``fairlight`` /
  ``sokohashv2`` / ``angry-reverser`` class): one large constraint
  problem over one state set, where Z3's search either finds the
  satisfying model quickly or wanders. A SignExt fix that collapsed one
  such bench's constraint AST by an order of magnitude *unmasked* its
  bimodality rather than causing it. ``ANGR_Z3_TACTIC=qfbv`` (see
  :doc:`rust_engine`) is the only knob that moves this class.
* **Multi-solution unconstrained jumps** (the ``CADET`` class): an
  ``explore(find=)`` over symbolic stdin reaches a jump whose target has
  several witnesses, and which one Z3 returns decides the mode.

What does not fix it
--------------------

Pinning Z3's seeds narrows the spread but does not collapse it
(:doc:`rust_engine_decisions`), so ``--deterministic`` stays a
diagnostic knob for reproducing one slow path, not a gating default.
The engine-side levers were exhausted by the perf campaign that
classified the whole corpus (``angr-9w6ad``); the residual is a solver
floor.

Reproducing
-----------

``tests/benchmarks/bimodal_variance.py`` spawns
``run_single.py --engine rust`` once per sample (each run inherits the
4 GB ``RLIMIT_AS`` sandbox), prints a per-bench summary and a 1 s-bin
histogram, and classifies each bench as STABLE / BIMODAL / OUTLIER.
Sample every current member:

.. code-block:: console

   $ read -r -a B <<<"$(python3 -c "import sys;sys.path.insert(0,'tests/benchmarks');import run_regression as r;print(' '.join(sorted(r.BIMODAL_BENCHMARKS)))")"
   $ python tests/benchmarks/bimodal_variance.py --runs 20 --timeout 180 \
       --benchmarks "${B[@]}" --json /tmp/bimodal.json

``--json`` dumps the raw samples; ``--deterministic`` passes the seed pin
through to each child. Twenty samples is the floor for a bench whose
slow mode is rare; ten under-samples the trimodal ones. The
classification label is necessary but not sufficient: a bench whose
slow tail is a continuum rather than a gap can read STABLE while its
single-shot timings are still untrustworthy, so treat any bench with a
coefficient of variation well above the noise floor as
single-shot-unreliable regardless of its label. The corpus-wide sweep
(``classify()`` over every bench in ``baseline_timings.json``) is recorded
on ``angr-9w6ad.3``.

Related memories
----------------

- ``benchmark-bimodal-variance-rules`` — the three sync points above.
- ``pr-bench-gate-jitter-risk`` — why ``--skip-bimodal`` exists in the
  PR-time gate.
