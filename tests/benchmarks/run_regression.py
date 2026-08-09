#!/usr/bin/env python3
"""Benchmark regression test for the Rust symbolic execution engine.

Runs fast-tier benchmarks with both engines and checks:
1. Rust engine produces correct output (matches Python)
2. Rust engine is not slower than baseline threshold — a timing failure must
   clear both a relative bar (--threshold) and an absolute one
   (--regression-floor), so host noise on the sub-second benches does not
   fail the gate (see ``timing_regression_pct``)
3. Rust engine meets the speedup SLA vs Python (fails below 0.5x by default,
   warns below 1.0x). Uses cached python_time from baseline when this run
   skipped Python (e.g. rust_only entries).

Usage:
    python tests/benchmarks/run_regression.py              # Run all fast benchmarks
    python tests/benchmarks/run_regression.py --update      # Update baseline timings
    python tests/benchmarks/run_regression.py --threshold 0.2  # 20% regression threshold
    python tests/benchmarks/run_regression.py --regression-floor 0.1  # 100ms absolute floor
    python tests/benchmarks/run_regression.py --sla-fail-threshold 0.7  # Tighter SLA
    python tests/benchmarks/run_regression.py --no-sla       # Disable SLA check
    python tests/benchmarks/run_regression.py --memory-threshold 0.4  # Tighter peak-RSS gate
    python tests/benchmarks/run_regression.py --no-memory-check  # Disable peak-RSS gate

Exit codes:
    0 = all benchmarks pass
    1 = regression detected (wrong output, too slow, or SLA failure)
    2 = setup error
"""

from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import json
import multiprocessing
import os
import statistics
import subprocess
import sys
import time

# Reuse the subprocess runner from run_single.py
sys.path.insert(0, os.path.dirname(__file__))
from run_single import (
    DEFAULT_MEM_LIMIT_MB,
    EXAMPLES_DIR,
    _resolve_examples_dir,
    _run_in_child,
)


def libvex_ffi_enabled() -> bool | None:
    """Whether the installed rustylib .so was built with the ``libvex-ffi`` feature.

    ``baseline_timings.json`` is recorded against a stock ``pip install -e .``,
    where ``setup.py::_rust_features`` turns ``libvex-ffi`` ON. A .so built
    without it lifts every cold block through the slower pyvex-callback path,
    which reads as a uniform ~20% slowdown across the whole suite — a fake
    regression that costs a debugging session to attribute. Probe it up front
    so the run says so instead.

    Loads the extension module directly rather than via ``import angr`` — the
    package import is far heavier and this runs in the gate's parent process.
    Returns None when the .so can't be found or probed.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sos = glob.glob(os.path.join(repo_root, "angr", "rustylib*.so"))
    if not sos:
        return None
    try:
        spec = importlib.util.spec_from_file_location("rustylib", sos[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return bool(mod.vex_engine.libvex_ffi_enabled())
    except Exception:
        return None


LIBVEX_OFF_NOTE = (
    "the installed rustylib .so was built WITHOUT the libvex-ffi feature, but "
    "baseline_timings.json was recorded WITH it (default-ON in setup.py). Cold "
    "blocks lift through the slower pyvex-callback path, which inflates every "
    "bench by roughly 20%. Rebuild with `make rebuild` (or "
    "`tools/rebuild-rust.sh --cargo-only`, which now passes the feature too) "
    "before trusting these timings."
)

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline_timings.json")
# Optional per-bench counter snapshot. When present, a timing regression
# triggers a `bench_diff` table against the cached snapshot so CI logs
# show *which counter moved* without a manual re-run. Refreshed by
# ``--update`` alongside BASELINE_FILE. Soft dependency — absent file
# only suppresses the diff, the gate keeps working.
BASELINE_COUNTERS_FILE = os.path.join(os.path.dirname(__file__), "baseline_counters.json")

# Tracked metrics beyond timing. Each entry: (key_in_stats, key_in_baseline, regression_threshold_pct)
# A regression is flagged when the metric INCREASES by more than threshold_pct.
# Only fires under --check-counts (off by default historically; nightly + PR
# gate now pass it — see ci.yml / nightly-ci.yml). The baseline_val > 0 guard
# in the gate body skips any metric whose baseline is 0, so the many corpus
# entries that legitimately record 0 (the *_branch synthetic arches,
# flareon2015_2, etc.) are no-ops rather than false alarms. ``state_creations``
# in particular is 0 for essentially every fast-tier bench (only sym-write/
# csgames2018 ever recorded a nonzero value historically, and csgames2018 has
# since dropped to 0), so it almost never gates — it is kept tracked for the
# rare bench that does fork during exploration.
TRACKED_METRICS = [
    ("callback_count", "callback_count", 0.10),  # 10% more callbacks = algorithmic regression
    ("state_creations", "state_creations", 0.10),
    ("steps", "steps", 0.15),  # 15% more steps
]

# Solver fast paths that answer a query without reaching Z3's check(), plus the
# derived headline aggregate over them (angr-op0dn.9.5). Unlike TRACKED_METRICS
# — which gate on a counter going UP — these gate on a counter falling to
# exactly 0 on a bench whose baseline_counters.json snapshot recorded it
# nonzero. That is the failure mode a fast path actually has: a refactor stops
# reaching the fast-path branch, every query silently falls through to Z3, and
# nothing but wall-clock notices. Timing alone will not catch it on a bench
# where the fast path saves only a few hundred checks.
#
# Only the collapse-to-zero case gates. A partial drop is expected drift (path
# choice shifts which fast paths a bench hits) and would be a false-alarm
# machine; the timing gate covers the case where such a drop actually costs.
FASTPATH_COUNTERS = [
    "z3_saved_check_total",
    "z3_branch_concrete",
    "z3_assume_concrete",
    "zext_cmp_trivial_decide_count",
    "z3_branch_model_hit",
    "z3_extrema_model_hit",
    "z3_eval_upto_model_hit",
]

# Benches exempt from the --check-counts gate. These are the four original
# BIMODAL_BENCHMARKS: they exercise multi-solution constraints where Z3
# model nondeterminism can shift the explored path (and therefore
# step/callback counts), so their count baselines are not reliably stable. They
# still gate on timing and peak memory; only the algorithmic-count check is
# skipped.
#
# NOTE: BIMODAL_BENCHMARKS grew to five when CADET_00001_partial joined
# (angr-027h), so COUNT_EXEMPT is now a strict subset, not equal to it.
# CADET_00001_partial is intentionally omitted: its count baselines are 0/0/0,
# so the `baseline_val > 0` guard below already makes --check-counts a no-op for
# it — adding it to COUNT_EXEMPT would be redundant.
#
# The seven MEDIUM_SUITE benches (sym-write, flareon2015_5/10,
# ekopartyctf2016_rev250, csaw_wyvern, codegate_2017-angrybird, mma_howtouse)
# were soak-validated as deterministic — 0% variance in
# callback_count/state_creations/steps across 5 runs/bench (angr-lagp,
# 2026-06-15), mirroring the fast-tier soak (angr-bq9v). Their baselines were
# refreshed to the measured values and they are now gated. The soak surfaced
# the same kind of silent drift bq9v found in the fast tier: sym-write
# callback_count 22->20 / state_creations 1->0 / steps 18->20, and
# ekopartyctf2016_rev250 callback_count 5->1 / steps 3->7.
COUNT_EXEMPT = frozenset(
    {
        # bimodal (Z3 multi-solution → count nondeterminism); see
        # BIMODAL_BENCHMARKS and docs/advanced-topics/rust_bimodal_variance.rst
        "google2016_unbreakable_1",
        "securityfest_fairlight",
        "ekopartyctf2016_sokohashv2",
        "hackcon2016_angry-reverser",
    }
)

# Tiered benchmark suites. Each entry: (name, timeout_seconds, [strategy, [rust_only]])
# FAST_SUITE/MEDIUM_SUITE membership is a hand-maintained runtime *budget* split
# (fast <10s, medium 10-60s) and is deliberately NOT a mirror of the ``tier``
# field in run_single.EXAMPLE_CATALOG (fast <5s, medium 5-30s, validated by
# validate_tier.py). Six MEDIUM_SUITE members — sym-write, flareon2015_5,
# flareon2015_10, ekopartyctf2016_rev250, csaw_wyvern, codegate_2017-angrybird —
# are catalog tier='fast' after the perf waves and stay here on purpose;
# retiering the catalog must not move them.
# strategy: "bfs" (default) or "dfs"
# rust_only: True when the Python engine is unreliable under the 4GB memory
# limit (OOM/timeout) OR when Z3 model nondeterminism produces benign output
# divergence (multiple valid solutions to the same find/avoid set). In either
# case the regression check still tracks Rust timing and algorithmic metrics.
# Fast tier: < 10s, always run
# rust_only=True/False audited via property_fuzzer (angr-qz16, 2026-05-14):
# Promoted (rust_only=False, pass 10/10 trials, mixed bfs/dfs):
#   defcamp_r100 (bfs+dfs), ais3_crackme, google2016_unbreakable_0,
#   strcpy_find, flareon2015_2, defcon2016quals_baby-re
# Kept rust_only=True (consistent output divergence):
#   fauxware, google2016_unbreakable_1, unmapped_analysis, csgames2018,
#   whitehatvn2015_re400.
# MEDIUM_SUITE audit: all four non-bimodal candidates tested
# (flareon2015_5, ekopartyctf2016_rev250, csaw_wyvern, codegate_2017-angrybird)
# diverge — no MEDIUM promotions.
FAST_SUITE = [
    ("fauxware", 30, "bfs", True),
    ("defcamp_r100", 30, "bfs", False),
    ("ais3_crackme", 30, "bfs", False),
    ("google2016_unbreakable_0", 30, "bfs", False),
    ("google2016_unbreakable_1", 30, "bfs", True),
    ("strcpy_find", 30, "bfs", False),
    ("flareon2015_2", 30, "bfs", False),
    ("unmapped_analysis", 30, "bfs", True),
    ("defcon2016quals_baby-re", 30, "bfs", False),
    ("defcamp_r100", 30, "dfs", False),  # DFS variant: same example, different strategy
    ("csgames2018", 30, "bfs", True),
    ("whitehatvn2015_re400", 30, "bfs", True),
    # MIPS32 LE inline-ELF synthetic benchmark (angr-2xfz). Lives in
    # tests/benchmarks/synthetic_examples/, resolved via the fallback
    # logic in _resolve_examples_dir. Marked rust_only=True so the SLA
    # speedup gate does not flag it — the program is intentionally short
    # (~10 instructions) and the Rust PyO3 init tax (~250 ms) dominates
    # the workload's runtime, but the regression gate still validates
    # MIPS32 lift + exec stays green and timing stays within 15% of
    # baseline_timings.json's cached ``rust_time``.
    ("mips32_le_branch", 30, "bfs", True),
    # AArch64 LE inline-ELF synthetic benchmark (angr-jzn8). Same shape
    # and rationale as mips32_le_branch — promotes ARM64 from
    # Experimental to Supported in the arch matrix. Avoids NEON ops
    # (those still use the NeonUnimplemented scaffold and would panic).
    ("aarch64_le_branch", 30, "bfs", True),
    # ARM (ARMEL) LE inline-ELF synthetic benchmark (angr-duta.2). Same
    # shape and rationale as aarch64_le_branch — covers ARMEL without
    # the angr-examples Android validate binary, so the benchmark runs
    # unconditionally. Uses only base scalar ARM ops (no NEON / VFP /
    # Thumb).
    ("arm_le_branch", 30, "bfs", True),
    # First real ARM ELF in the corpus (angr-p3da). The Android NDK
    # licence-check crackme — complements arm_le_branch with a non-toy
    # ARM workload that exercises the full lift + explore loop on
    # production-shaped code. Multiple valid solutions to the find/avoid
    # set produce benign output divergence with Python, so rust_only=True.
    # rust_only=True also makes it SLA-exempt (see the SLA block): on this
    # ~0.2s ARM workload the PyO3/lift init tax makes Rust legitimately
    # slower than Python (~0.42x), an accepted arch-edge loss, not a
    # regression. The timing gate still holds it within 15% of baseline.
    ("android_arm_license_validation", 30, "bfs", True),
    # MIPS64 LE inline-ELF synthetic benchmark (angr-duta.4). Same shape
    # and rationale as mips32_le_branch — promotes MIPS64 from
    # Experimental to Supported in the arch matrix. Uses only base MIPS
    # integer ops; the ELF wrapper is ELF64 with EI_CLASS=ELF64 and
    # EF_MIPS_ARCH_64, exercising the N64 calling convention path.
    ("mips64_le_branch", 30, "bfs", True),
    # MIPS64 BE inline-ELF synthetic benchmark (angr-ig3o.3). Big-endian
    # counterpart of mips64_le_branch — same program shape, but
    # EI_DATA=MSB and BE-packed instruction words. Validates the BE
    # instruction-fetch + memory layout path on a 64-bit MIPS target
    # end-to-end through the Rust interpreter.
    ("mips64_be_branch", 30, "bfs", True),
    # CMU binary bomb (angr-w5op). Lives in tests/benchmarks/synthetic_examples/
    # as a wrapper that skips upstream's broken/path-explosion flags and
    # runs the FAST subset (1, 4, secret) of the printf/scanf-heavy
    # teaching binary. Marked rust_only=True because solve_flag_4 returns
    # a different valid model than Python (Z3 model nondeterminism on
    # multi-solution constraints — Python picks (7, 0), Rust picks (0, 0)).
    # See bd memory ``bench-cmu-binary-bomb-broken`` and the wrapper
    # docstring for the full per-flag rationale.
    ("cmu_binary_bomb_partial", 30, "bfs", True),
    # CoW fork-scaling synthetic benchmark (angr-vx8p.2, from angr-uf0g).
    # Lives in tests/benchmarks/synthetic_examples/ as a prebuilt x86_64
    # fork-tree binary (fork_tree_8) that reads 16 symbolic stdin bytes and
    # runs them through 8 independent branches, forking to 256 leaf states.
    # Unlike the rest of the corpus (28/30 entries have state_creations==0)
    # this bench deliberately forks hundreds of states, so it is the only
    # regression gate over the engine's core O(1) CoW-fork claim
    # (im::OrdMap structural sharing) — state_creations/steps carry signal
    # here. Marked rust_only=True: the Python engine OOMs past N=8 in the
    # uf0g sweep, so no Python comparison is recorded.
    #
    # TIMING is bimodal (angr-5cx4r): fast mode ~2.2s, slow mode ~2.7s from
    # Z3 SAT-search nondeterminism over the 256 fork-leaf constraints. It is
    # deliberately NOT in BIMODAL_BENCHMARKS — that set drops a bench from the
    # whole gate, and the steps=513 count signal (the O(1) CoW-fork claim) is
    # this bench's entire reason to exist and must stay gated. Instead the
    # baseline_timings rust_time is pinned to the SLOW mode (2.7), so the gate
    # tolerates both modes while keeping the count gate (same convention the
    # CADET_00001_partial BIMODAL note uses). A doc-only iter-77 commit tripped
    # the old 2.28 (fast-mode) baseline with three consecutive ~2.7 reads.
    ("cow_fork_scaling", 30, "bfs", True),
    # SharifCTF rev50 (angr-vx8p.4, promoted from the run_single catalog
    # after angr-8kmjo fixed the argv gap). A real CTF reversing binary
    # (fopen/fseek/fprintf-heavy) that both engines solve to the same flag
    # SharifCTF{b70c59275fcfa8aebf2d5911223c6589}. rust_only=False: the
    # solution is fully constrained, so Python and Rust emit identical
    # output and the ~3.9x Rust speedup (Python ~3.3s, Rust ~0.84s) is a
    # real comparison the SLA gate can track. argv ['-f','getit'] is
    # supplied by run_single.py's EXAMPLE_CATALOG (the missing entry that
    # angr-8kmjo previously mislabeled this bench a >60s TIMEOUT).
    ("sharif7_rev50", 30, "bfs", False),
    # xmllint reach-getenv (angr-11djq.3 / T1c). The first real-world,
    # libc-heavy *utility* binary in the gate (libxml2's xmllint, an x86-64
    # PIE vendored in angr-examples) — complements the CTF crackmes with a
    # de-risked bounded slice validated by the iter-11 tractability probe
    # (bd memory ``xmllint-path-b-tractability-probe``): symbolic 16-byte
    # stdin + standard CLI options stays single-state through the
    # deterministic startup and reaches getenv@libc in ~27 manager steps
    # (FAST tier, ~3.5s Rust, single found state, peak ~364 MB << 4 GB cap).
    # Wrapper lives in synthetic_examples/xmllint_getenv/; it resolves the
    # binary via ANGR_EXAMPLES_DIR (the same env CI sets) rather than
    # duplicating the 80 KB artifact. rust_only=True: this is a realism /
    # regression-coverage + syscall-fallback measurement surface, not a
    # speed-win bench (~1.1x over Python, marginal enough to flap into the
    # SLA warn band on a real binary's init-tax variance). Records
    # syscall_python_fallback_count (currently 0) so the round-1 syscall
    # beads have a real-binary measurement surface — see the bead notes.
    ("xmllint_getenv", 30, "bfs", True),
    # ctype/fprintf native-coverage micro-bench (angr-11djq.20). A tiny
    # checked-in x86-64 ELF (tests/benchmarks/synthetic_examples/
    # cli_ctype_fprintf/) that runs four symbolic stdin bytes through an
    # isdigit short-circuit chain, fanning out to five leaf states and
    # reaching a ``win`` marker on the all-digits path. It is the first gate
    # entry that exercises __ctype_b_loc at all: the rest of the corpus is
    # CTF-heavy and only touches fprintf incidentally (sharif7_rev50), so the
    # locale-ctype (angr-tx7ec.4) and fprintf (angr-884yn) native procs had
    # unit tests but no end-to-end coverage. Both fall back to Python in this
    # end-to-end harness (the __ctype_b_loc classifier table is built by
    # __libc_start_main *during* exploration, after the Rust seed is captured,
    # so the table ptr is null at native dispatch; fprintf stderr/stdout awaits
    # write-side fileno resolution, angr-csyy9) — so what the bench gates is the
    # ctype/fprintf dispatch + fallback path, the regression surface nothing
    # else in the corpus exercises. See the solve.py docstring for the detail.
    # rust_only=True: a native-coverage / fallback-measurement surface, not a
    # speed-win bench (Rust ~0.35s vs Python ~1.1s). Driver builds its manager
    # through proj.factory.simulation_manager so run_single swaps in the Rust
    # engine transparently. Inspect with --dump-counters.
    ("cli_ctype_fprintf", 30, "bfs", True),
]

# Medium tier: 10-60s, run with --full
MEDIUM_SUITE = [
    # Convergent subset (phases 1+2) of CADET_00001's upstream solve.py
    # (angr-027h). Wrapper in synthetic_examples/ skips the pathological
    # phase-3 step-loop egg hunt (see the CADET_00001 catalog note in
    # run_single.py and bd memory benchmark-cadet-phase3-not-a-bench). Both
    # engines converge — Rust ~6.9s vs Py ~11.8s — so it is NOT rust_only;
    # the explore(find=) speedup is worth tracking.
    ("CADET_00001_partial", 120),
    ("sym-write", 60, "bfs", True),
    ("flareon2015_5", 60, "bfs", True),
    ("flareon2015_10", 60),
    ("ekopartyctf2016_rev250", 60, "bfs", True),
    ("csaw_wyvern", 60, "bfs", True),
    ("securityfest_fairlight", 60, "bfs", True),
    ("codegate_2017-angrybird", 60, "bfs", True),
    ("mma_howtouse", 60),
    ("ekopartyctf2016_sokohashv2", 60, "bfs", True),
    ("hackcon2016_angry-reverser", 90, "bfs", True),
]

# Default: fast only. Use --full for fast + medium.
REGRESSION_SUITE = FAST_SUITE  # overridden in main() if --full

# Benchmarks whose Rust-side timing is bimodal under Z3 model nondeterminism
# (slow-mode runs can be ~2x the fast-mode runs even on a stable HEAD). They
# routinely exceed the 15% regression threshold without a real code change and
# should be skipped on tight PR gates. See bd memory
# `invariant-bimodal-variance-benchmarks` and
# ``docs/advanced-topics/rust_bimodal_variance.rst``.
#
# google2016_unbreakable_1 was removed on 2026-05-18 (angr-hyiz.4) after a
# 20-sample campaign showed 20/20 in 2.43-2.48s. Re-added 2026-05-22
# (angr-ja0i) after the ralph iter-2 gate flagged it at 5.21s vs 3.5s
# baseline. A 15-sample re-validation on HEAD 14187073d found a multi-modal
# distribution: 11/15 in 0.91-0.99s (fast), 2/15 in 1.15s, 1/15 in 1.86s,
# 1/15 in 2.65s. The fast mode is well below the May-18 median (post-perf-
# gains from angr-b58a/zdho/9jly), but the slow tail re-emerged. See bd
# memory `benchmark-unbreakable_1-2026-05-22`.
#
# hackcon2016_angry-reverser was added 2026-06-02 (angr-bl0g) after the
# angr-rbnk SignExt fix (commit 4dc7fc064) collapsed the bench AST by ~13x
# but unmasked a Z3 SAT-search nondeterminism floor. 8-sample post-fix
# distribution: 8.97, 18.66, 22.39, 22.68, 26.24, 28.08, 29.49, 34.88
# (median ~22.5s, fast tail ~9s matching Python's solve time, slow tail
# ~35s). Pre-fix the bench was tightly clustered at ~30s; the simpler AST
# gives Z3 more branch-choice freedom, widening the distribution.
BIMODAL_BENCHMARKS = frozenset(
    {
        "securityfest_fairlight",
        "ekopartyctf2016_sokohashv2",
        "google2016_unbreakable_1",
        "hackcon2016_angry-reverser",
        # angr-027h: the CADET easter-egg explore(find=) over symbolic stdin
        # hits multi-solution unconstrained jumps; 4 samples clustered into two
        # modes (~6.8s/411MB and ~8.2s/545MB) from Z3 model nondeterminism.
        # baseline_timings uses the slow mode so the gate tolerates both.
        "CADET_00001_partial",
    }
)


# Absolute-delta floor for the timing gate, in seconds. A relative-only
# threshold is meaningless on the sub-second benches: at sharif7_rev50's 0.16s
# baseline, 15% is ~24ms, which is inside the process-startup / OS-scheduling
# noise floor when the box is loaded. That bench failed the ralph gate twice
# (iters 165 and 021, both times including every in-gate retry) at 0.19s while
# standalone re-runs immediately afterward landed at 0.16s, and the counter
# diff showed run_wall_time_ns *down* — i.e. the extra ~30ms was never engine
# work. The gate's retries cannot help, because they run under the same load
# that caused the excursion. See bd bead angr-z8p3x.
DEFAULT_REGRESSION_FLOOR_S = 0.05


def timing_regression_pct(baseline_time, rust_time, threshold, floor_s=DEFAULT_REGRESSION_FLOOR_S):
    """Return the over-threshold timing regression percent, or ``None``.

    Pure decision helper for the rust_time gate, mirroring
    ``memory_regression_pct``. A regression must clear BOTH bars to count:

    * relative — ``rust_time > baseline_time * (1 + threshold)``
    * absolute — ``rust_time - baseline_time > floor_s``

    The absolute floor is what keeps sub-second benches out of the failure
    set for deltas that are pure host noise (see
    ``DEFAULT_REGRESSION_FLOOR_S``). It is deliberately a floor and not a
    per-bench threshold override: every bench above ``floor_s / threshold``
    (~0.33s at the defaults) keeps its previous behaviour exactly, so slow
    benches are not silently loosened.

    Returns ``None`` when there is nothing to compare (no baseline, a
    non-positive baseline, or a missing measurement) or when either bar is
    unmet. Otherwise returns the positive percent over baseline.
    """
    if baseline_time is None or baseline_time <= 0 or rust_time is None:
        return None
    if rust_time <= baseline_time * (1 + threshold):
        return None
    if rust_time - baseline_time <= floor_s:
        return None
    return ((rust_time / baseline_time) - 1) * 100


# Suite-wide host-slowdown normalization for the timing gate.
#
# The absolute floor above only covers sub-second benches. It does nothing for
# the other failure mode: the whole BOX being slow for the duration of the run.
# ralph iter24 went red on `cow_fork_scaling: 24% regression (3.09s vs 2.50s)`
# with both in-gate retries also failing, while standalone re-runs immediately
# afterward landed at 2.45-2.48s and a full gate re-run was 22/22. That run took
# 47.4s total against a normal 31-35s, and the per-bench ratios were uniformly
# shifted: median current/baseline was 1.083x, so cow_fork_scaling's 1.236x is
# only 1.14x of the suite median — under the 15% bar. At a 2.50s baseline the
# 50ms floor is irrelevant (15% is 375ms), and retries cannot help because they
# run inside the same contention window. See bd bead angr-mc8pw.
#
# So: divide out the suite median before deciding. The cap is the anti-masking
# guard — past it we refuse to explain the slowdown away, because a uniform
# shift that large is equally consistent with a genuine global regression, and
# silently normalizing it would turn the gate into a no-op exactly when it
# matters most.
DEFAULT_HOST_FACTOR_CAP = 1.25

# Minimum number of benches that must have a usable baseline before a median is
# meaningful. Below this the "suite median" is one or two benches, which is just
# the failing bench itself wearing a disguise.
MIN_HOST_FACTOR_SAMPLES = 5

# Dead-band below which a median is treated as "the host was fine". A quiet run
# still medians a percent or so off 1.0 just from run-to-run wobble, and
# normalizing by that would loosen every gate run slightly for no reason. With
# the dead-band, a quiet host decides byte-for-byte identically to the
# unnormalized gate — the same property angr-z8p3x's floor tests pin for benches
# above the floor.
HOST_FACTOR_DEAD_BAND = 1.02


def suite_median_ratio(ratios, min_samples=MIN_HOST_FACTOR_SAMPLES):
    """Return the median ``current / baseline`` timing ratio for a run.

    ``ratios`` is every bench in the run that had a usable baseline —
    passing and failing alike. Including the passing ones is the whole
    point: they are the control group that says whether the box was slow.

    This estimates the *common-mode* shift only, and deliberately so. The
    suite mixes two populations — sub-second benches, whose ratios are
    startup-noise-dominated and can sit well below 1.0 against a
    conservative baseline, and the multi-second ones that are the only
    benches able to clear the absolute floor in the first place. So the
    median under-reads a slowdown that scales with bench duration: on the
    angr-mc8pw run it explained 1.083x of cow_fork_scaling's 1.236x, not all
    of it. That is the intended bias — under-correcting leaves the gate
    strict, and a duration-weighted estimator would buy ~0.004x on that run
    (1.079x over just the >=0.33s benches) for a lot more machinery.

    Returns ``None`` when there are fewer than ``min_samples`` usable
    ratios, i.e. when no honest median exists.
    """
    usable = [r for r in ratios if r is not None and r > 0]
    if len(usable) < min_samples:
        return None
    return statistics.median(usable)


def host_scale_factor(median_ratio, cap=DEFAULT_HOST_FACTOR_CAP):
    """Return the factor to scale timing baselines by, given a suite median.

    Returns ``1.0`` (i.e. no normalization, previous behaviour exactly) when:

    * there is no median (``None`` — too few samples), or
    * the median is within ``HOST_FACTOR_DEAD_BAND`` — the host was fine (or
      faster than baseline, in which case tightening the gate would invent
      failures), or
    * the median is above ``cap`` — see ``DEFAULT_HOST_FACTOR_CAP``. A
      slowdown that big is not explained away; the caller reports the run as
      suite-wide-suspect and the failures stand.
    """
    if median_ratio is None or median_ratio < HOST_FACTOR_DEAD_BAND or median_ratio > cap:
        return 1.0
    return median_ratio


def memory_regression_pct(baseline_mem, rust_peak_mem, threshold):
    """Return the over-threshold peak-RSS regression percent, or ``None``.

    Pure decision helper for the peak_memory_mb gate. Deliberately takes
    ONLY the two memory measurements and the threshold — it has no
    solve-count / stats parameter, so the gate cannot be (re)gated behind
    "successful_solves > 0" or an empty-stats branch (angr-cudgw.2). A
    bench that runs OK but solves nothing still reports a process-wide
    ``ru_maxrss`` high-water mark, and that mark is what we compare.

    Returns ``None`` when there is nothing to compare (no baseline, a
    non-positive baseline, or a missing current measurement) or when the
    current peak is within ``threshold`` of the baseline. Otherwise returns
    the positive percent over baseline.
    """
    if baseline_mem is None or baseline_mem <= 0 or rust_peak_mem is None:
        return None
    if rust_peak_mem <= baseline_mem * (1 + threshold):
        return None
    return ((rust_peak_mem / baseline_mem) - 1) * 100


def _normalize_output(output):
    """Normalize benchmark output for comparison.

    Z3 model nondeterminism causes unconstrained stdin/stdout bytes to evaluate
    to different fill values between Python and Rust solvers (e.g. \\x00 vs \\xf5).
    Strip trailing garbage from byte-string reprs by truncating after the last
    printable ASCII character sequence.
    """
    import re

    # For byte-string reprs like b'Code_Talkers\xf5\xf5u\xf5...',
    # find the meaningful prefix: the longest prefix of printable ASCII text
    # (letters, digits, punctuation) before non-printable sequences dominate.
    # Strategy: remove any suffix that starts with a \xNN escape and contains
    # only \xNN escapes and occasional single printable chars.
    return re.sub(r"(\\x[0-9a-fA-F]{2}[^']*?)(')", r"\2", output)


def run_one(name, engine, timeout, mem_limit_mb, strategy="bfs"):
    """Run a single benchmark in a subprocess, return result dict."""
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    examples_dir = _resolve_examples_dir(name, EXAMPLES_DIR)
    try:
        async_result = pool.apply_async(_run_in_child, (name, engine, examples_dir, mem_limit_mb, strategy))
        return async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as e:
        pool.terminate()
        return {"ok": False, "error": str(e)}
    finally:
        pool.terminate()
        pool.join()


def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline(data):
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"Baseline saved to {BASELINE_FILE}")


def load_baseline_counters():
    if os.path.exists(BASELINE_COUNTERS_FILE):
        with open(BASELINE_COUNTERS_FILE) as f:
            return json.load(f)
    return {}


def save_baseline_counters(data):
    with open(BASELINE_COUNTERS_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"Counter baseline saved to {BASELINE_COUNTERS_FILE}")


def _baseline_key_for(name, strategy):
    """Map a (name, strategy) suite entry to its baseline_counters key.

    Mirrors the inline derivation in main(): bfs entries key on the bare
    name, dfs variants get a ``__dfs`` suffix so the same example can carry
    two independent counter snapshots.
    """
    return f"{name}__dfs" if strategy == "dfs" else name


def expected_counter_keys(suite, skip_bimodal=True):
    """baseline_counters keys that ``suite`` is expected to populate.

    Bimodal benches are excluded by default: their Z3-path nondeterminism
    makes a per-counter diff pure noise, so --update-counters intentionally
    never snapshots them (see BIMODAL_BENCHMARKS). Each entry is the same
    (name, timeout, [strategy, [rust_only]]) tuple the suites use, so a
    pre-normalization entry of length 2 defaults strategy to "bfs".
    """
    keys = set()
    for entry in suite:
        name = entry[0]
        if skip_bimodal and name in BIMODAL_BENCHMARKS:
            continue
        strategy = entry[2] if len(entry) > 2 else "bfs"
        keys.add(_baseline_key_for(name, strategy))
    return keys


def missing_counter_keys(suite=None, baseline_counters=None, skip_bimodal=True):
    """SUITE bench keys absent from baseline_counters (sorted).

    baseline_counters.json is refreshed manually (--update / --update-counters)
    and silently rots when a bench is added but the snapshot is not rerun,
    which quietly disables the bench_diff regression report for that bench.
    This surfaces the gap as a hard list so a gate can fail on it. Defaults
    to the full fast+medium suite and the on-disk snapshot.
    """
    if suite is None:
        suite = FAST_SUITE + MEDIUM_SUITE
    if baseline_counters is None:
        baseline_counters = load_baseline_counters()
    return sorted(expected_counter_keys(suite, skip_bimodal) - set(baseline_counters))


def _git_revparse(*args):
    try:
        out = subprocess.run(
            ["git", "rev-parse", *args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def save_history_record(path, results):
    """Emit a single timestamped record consumed by the perf dashboard.

    The schema is intentionally narrow so aggregate_bench_history.py can
    treat it as append-only time-series data. Per-bench fields are the
    same set that baseline_timings.json tracks.
    """
    commit = os.environ.get("GITHUB_SHA") or _git_revparse("HEAD") or "unknown"
    branch = os.environ.get("GITHUB_REF_NAME") or _git_revparse("--abbrev-ref", "HEAD") or "unknown"
    record = {
        "schema_version": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "commit": commit,
        "branch": branch,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"History record saved to {path} (commit={commit[:10]})")


def main():
    parser = argparse.ArgumentParser(description="Rust engine benchmark regression test")
    parser.add_argument("--update", action="store_true", help="Update baseline timings")
    parser.add_argument(
        "--update-counters",
        action="store_true",
        help="Refresh baseline_counters.json (used by the bench_diff "
        "report on a regression) without touching baseline_timings.json. "
        "Safe to run periodically — counter snapshots are not "
        "performance-gated, only diffed.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15, help="Regression threshold (default: 0.15 = 15%% slower)"
    )
    parser.add_argument(
        "--regression-floor",
        type=float,
        default=DEFAULT_REGRESSION_FLOOR_S,
        help="Absolute-delta floor for the timing gate, in seconds "
        f"(default: {DEFAULT_REGRESSION_FLOOR_S} = {DEFAULT_REGRESSION_FLOOR_S * 1000:.0f}ms). A bench must be "
        "both --threshold%% slower AND this many seconds slower than "
        "baseline to fail. Keeps sub-second benches from failing on host "
        "noise; set to 0 to restore the relative-only behaviour.",
    )
    parser.add_argument(
        "--host-factor-cap",
        type=float,
        default=DEFAULT_HOST_FACTOR_CAP,
        metavar="X",
        help="Largest suite-wide slowdown that may be normalized away, as a "
        f"ratio (default: {DEFAULT_HOST_FACTOR_CAP}). Timing failures are "
        "re-checked against a baseline scaled by the median "
        "current/baseline ratio across every bench with a baseline, so a "
        "uniformly slow host does not red the gate on its worst bench. "
        "Past this cap nothing is normalized and the run is reported as "
        "suite-wide-suspect instead.",
    )
    parser.add_argument(
        "--no-host-normalize",
        action="store_true",
        help="Disable the suite-wide host-slowdown normalization described "
        "under --host-factor-cap, restoring the raw per-bench comparison.",
    )
    parser.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT_MB)
    parser.add_argument("--rust-only", action="store_true", help="Only run Rust engine (skip Python comparison)")
    parser.add_argument("--full", action="store_true", help="Run full suite (fast + medium tier)")
    parser.add_argument(
        "--check-counter-coverage",
        action="store_true",
        help="Static check (no benches run): fail with exit 2 when "
        "baseline_counters.json is missing a key for any non-bimodal "
        "SUITE bench. Catches the manual-refresh drift where a new bench "
        "is added but --update-counters is never rerun, silently disabling "
        "its bench_diff regression report. Honors --full (fast+medium) and "
        "exits immediately.",
    )
    parser.add_argument(
        "--check-counts",
        action="store_true",
        help="Check algorithmic metrics (callback_count, state_creations, steps) for regressions",
    )
    parser.add_argument(
        "--no-memory-check",
        action="store_true",
        help="Disable the peak_memory_mb regression check (enabled by "
        "default). The check fails a bench whose Rust peak RSS exceeds "
        "its baseline peak_memory_mb by more than --memory-threshold.",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.5,
        help="Peak-RSS regression threshold (default: 0.5 = 50%% over "
        "baseline). Deliberately generous: allocator/page-cache "
        "variance on ru_maxrss is larger than timing variance, so a "
        "tighter bound risks false positives.",
    )
    parser.add_argument(
        "--memory-warn-only",
        action="store_true",
        help="Downgrade peak_memory_mb regressions from failures to "
        "warnings (for soaking the gate before promoting it to a hard "
        "fail).",
    )
    parser.add_argument(
        "--sla-warn-threshold",
        type=float,
        default=1.0,
        help="Speedup vs Python below this prints a SLA warning (default: 1.0x)",
    )
    parser.add_argument(
        "--sla-fail-threshold",
        type=float,
        default=0.5,
        help="Speedup vs Python below this fails the run (default: 0.5x)",
    )
    parser.add_argument("--no-sla", action="store_true", help="Disable SLA enforcement (skip speedup check entirely)")
    parser.add_argument(
        "--skip-bimodal",
        action="store_true",
        help="Skip benchmarks with known bimodal Z3 timing variance (intended for PR gates that need stable signal).",
    )
    parser.add_argument(
        "--retry-failures",
        type=int,
        default=0,
        metavar="N",
        help="Re-run timing/SLA/peak-memory regression failures up to N "
        "times. Each failed dimension is re-checked on the fresh "
        "measurement and cleared the first time it lands within bounds. "
        "Mitigates sub-second-bench noise and transient Z3-path / "
        "ru_maxrss variance where the same binary can flip pass/fail "
        "across runs without code changes (see bd memory "
        "benchmark-regression-noise-floor). Engine errors, output "
        "mismatches, and algorithmic metric regressions are never retried.",
    )
    parser.add_argument(
        "--history-record",
        metavar="PATH",
        help="Write a timestamped JSON record of this run to PATH "
        "(used by the perf dashboard to assemble historical "
        "time-series data).",
    )
    args = parser.parse_args()

    global REGRESSION_SUITE
    if args.full:
        REGRESSION_SUITE = FAST_SUITE + MEDIUM_SUITE

    if args.check_counter_coverage:
        suite = FAST_SUITE + MEDIUM_SUITE if args.full else FAST_SUITE
        missing = missing_counter_keys(suite)
        if missing:
            print(
                f"ERROR: baseline_counters.json missing keys for {len(missing)} SUITE bench(es): {', '.join(missing)}",
                file=sys.stderr,
            )
            print(
                "Refresh with: python tests/benchmarks/run_regression.py "
                f"{'--full ' if args.full else ''}--rust-only --skip-bimodal --update-counters",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"OK: baseline_counters.json covers all {len(expected_counter_keys(suite))} non-bimodal SUITE benches")
        sys.exit(0)

    # Normalize suite entries to (name, timeout, strategy, rust_only)
    REGRESSION_SUITE = [
        (
            e[0],
            e[1],
            e[2] if len(e) > 2 else "bfs",
            e[3] if len(e) > 3 else False,
        )
        for e in REGRESSION_SUITE
    ]

    if args.skip_bimodal:
        REGRESSION_SUITE = [e for e in REGRESSION_SUITE if e[0] not in BIMODAL_BENCHMARKS]

    # Verify examples exist (in either EXAMPLES_DIR or the in-repo synthetic dir).
    missing = []
    for name, _, _, _ in REGRESSION_SUITE:
        resolved = _resolve_examples_dir(name, EXAMPLES_DIR)
        if not os.path.exists(os.path.join(resolved, name, "solve.py")):
            missing.append(name)
    if missing:
        print(f"ERROR: Missing examples: {', '.join(missing)}", file=sys.stderr)
        print(f"Expected at: {EXAMPLES_DIR}", file=sys.stderr)
        sys.exit(2)

    libvex_on = libvex_ffi_enabled()
    if libvex_on is False:
        print(f"\nWARNING: {LIBVEX_OFF_NOTE}\n", file=sys.stderr)

    baseline = load_baseline()
    baseline_counters = load_baseline_counters()
    # Per-bench counter dict captured this run. Used to refresh
    # baseline_counters.json when --update is set, and as the
    # "current" side of the regression diff against
    # baseline_counters[baseline_key].
    current_counters: dict[str, dict] = {}
    results = {}
    failures = []
    # When --retry-failures > 0, per-bench retry records are keyed by name. A single anomalously slow/heavy
    # run can emit up to three correlated failures (timing, SLA, memory); the
    # retry pass re-checks each failed dimension and clears the ones that land
    # within bounds on a fresh measurement. Each record:
    #   {"name", "timeout", "strategy", "mem_limit", "baseline_key",
    #    "bl" (baseline rust_time), "baseline_mem", "sla_py_time",
    #    "timing_msg", "sla_msg", "mem_msg"}  — the *_msg keys are present only
    #   for the dimensions that actually failed.
    retry_info = {}
    # Suite-wide host-slowdown normalization (angr-mc8pw). `timing_ratios`
    # collects current/baseline for EVERY bench with a usable baseline —
    # the passing ones are the control group — and `timing_failures` holds
    # one record per timing failure so the normalization pass below can
    # re-decide it. Kept independent of `retry_info`, which only exists when
    # --retry-failures > 0: the PR-time CI gate runs without retries and must
    # still get the normalization.
    timing_ratios = []
    timing_failures = []
    total_start = time.perf_counter()

    def _retry_record(name, timeout, strategy, baseline_key):
        rec = retry_info.get(name)
        if rec is None:
            rec = {
                "name": name,
                "timeout": timeout,
                "strategy": strategy,
                "mem_limit": args.mem_limit,
                "baseline_key": baseline_key,
                "bl": None,
                "baseline_mem": None,
                "sla_py_time": None,
            }
            retry_info[name] = rec
        return rec

    for name, timeout, strategy, entry_rust_only in REGRESSION_SUITE:
        label = f"{name} (DFS)" if strategy == "dfs" else name
        baseline_key = f"{name}__dfs" if strategy == "dfs" else name
        print(f"\n--- {label} ---")

        skip_python = args.rust_only or entry_rust_only

        # Run Python engine (for output comparison)
        if not skip_python:
            py_result = run_one(name, "python", timeout, args.mem_limit)
            if not py_result.get("ok"):
                print(f"  Python: FAIL ({py_result.get('error', '?')})")
                failures.append(f"{name}: Python engine failed")
                continue
            py_time = py_result["elapsed"]
            py_output = py_result.get("output", "").strip()
            print(f"  Python: {py_time:.2f}s")
        else:
            py_time = None
            py_output = None

        # Run Rust engine
        rust_result = run_one(name, "rust", timeout, args.mem_limit, strategy)
        if not rust_result.get("ok"):
            print(f"  Rust:   FAIL ({rust_result.get('error', '?')})")
            failures.append(f"{name}: Rust engine failed: {rust_result.get('error', '?')}")
            continue
        rust_time = rust_result["elapsed"]
        rust_output = rust_result.get("output", "").strip()
        print(f"  Rust:   {rust_time:.2f}s", end="")

        # Compare output (normalize for Z3 model nondeterminism:
        # unconstrained stdin bytes may evaluate to different fill values
        # between Python and Rust solvers, e.g. \x00 vs \xf5)
        if py_output is not None and rust_output != py_output:
            norm_py = _normalize_output(py_output)
            norm_rust = _normalize_output(rust_output)
            # Whitespace-tolerant comparison: only a mismatch after splitting counts.
            if norm_rust != norm_py and norm_rust.split() != norm_py.split():
                print(" OUTPUT MISMATCH!")
                print(f"    Python: {py_output[:100]}")
                print(f"    Rust:   {rust_output[:100]}")
                failures.append(f"{name}: output mismatch")
                continue

        # Check speedup
        if py_time is not None and py_time > 0:
            speedup = py_time / rust_time
            print(f" ({speedup:.2f}x)")
        else:
            print()

        # Collect algorithmic metrics from Rust stats
        rust_stats = rust_result.get("stats") or {}
        rust_peak_mem = rust_result.get("peak_memory_mb")
        if rust_stats:
            current_counters[baseline_key] = rust_stats

        # Check timing regression against baseline
        if baseline_key in baseline and not args.update:
            bl = baseline[baseline_key]["rust_time"]
            if bl and bl > 0 and rust_time:
                timing_ratios.append(rust_time / bl)
            pct = timing_regression_pct(bl, rust_time, args.threshold, args.regression_floor)
            if pct is None and bl > 0 and rust_time > bl * (1 + args.threshold):
                # Over the relative bar but under the absolute floor. Never
                # suppress this silently — a quiet floor reads as "no
                # regression" when it is really "too small to trust".
                print(
                    f"  noise-floor: {rust_time:.2f}s vs baseline {bl:.2f}s "
                    f"(+{((rust_time / bl) - 1) * 100:.0f}%, "
                    f"{(rust_time - bl) * 1000:.0f}ms < {args.regression_floor * 1000:.0f}ms floor) — not failed"
                )
            if pct is not None:
                print(f"  REGRESSION: {rust_time:.2f}s vs baseline {bl:.2f}s (+{pct:.0f}%)")
                failure_msg = f"{name}: {pct:.0f}% regression ({rust_time:.2f}s vs {bl:.2f}s)"
                failures.append(failure_msg)
                # Soft-emit a per-counter diff so the CI log shows *which*
                # counter moved. Requires a cached counter snapshot in
                # baseline_counters.json — silently skipped otherwise.
                base_counters = baseline_counters.get(baseline_key)
                if base_counters and rust_stats:
                    try:
                        from bench_diff import compute_diff, format_report

                        rows = compute_diff(base_counters, rust_stats)
                        print(
                            format_report(
                                rows,
                                max_rows=20,
                                header=f"  counter diff for {baseline_key}:",
                            )
                        )
                    except Exception as exc:
                        # Diff helper is best-effort; never let it mask
                        # the underlying timing failure.
                        print(f"  (counter diff failed: {exc})")
                rec = None
                if args.retry_failures > 0:
                    rec = _retry_record(name, timeout, strategy, baseline_key)
                    rec["bl"] = bl
                    rec["timing_msg"] = failure_msg
                timing_failures.append(
                    {
                        "name": name,
                        "label": label,
                        "bl": bl,
                        "rust_time": rust_time,
                        "msg": failure_msg,
                        "retry_rec": rec,
                    }
                )

            # Check algorithmic metric regressions
            if args.check_counts and baseline_key not in COUNT_EXEMPT:
                for stat_key, bl_key, threshold_pct in TRACKED_METRICS:
                    current_val = rust_stats.get(stat_key)
                    baseline_val = baseline[baseline_key].get(bl_key)
                    if rust_stats and baseline_val is not None and baseline_val > 0 and stat_key not in rust_stats:
                        # The counter this gate compares against vanished from
                        # stats() (renamed/removed in stats_api.rs) even though
                        # stats() ran and the baseline recorded a nonzero value.
                        # Treat that as its own failure rather than a silent skip
                        # — otherwise the gate degrades to a permanent no-op for
                        # the metric on every future run. (angr-hv4lt.17)
                        print(f"  METRIC COUNTER DROPPED: {bl_key} absent from stats() (baseline {baseline_val})")
                        failures.append(f"{name}: {bl_key} dropped from stats() (baseline {baseline_val})")
                    elif (
                        current_val is not None
                        and baseline_val is not None
                        and baseline_val > 0
                        and current_val > baseline_val * (1 + threshold_pct)
                    ):
                        pct = ((current_val / baseline_val) - 1) * 100
                        print(f"  METRIC REGRESSION: {bl_key} {current_val} vs baseline {baseline_val} (+{pct:.0f}%)")
                        failures.append(f"{name}: {bl_key} regression ({current_val} vs {baseline_val}, +{pct:.0f}%)")

                # Solver fast paths: gate only on a collapse to zero.
                base_counters = baseline_counters.get(baseline_key)
                if base_counters and rust_stats:
                    for ctr in FASTPATH_COUNTERS:
                        baseline_val = base_counters.get(ctr)
                        current_val = rust_stats.get(ctr)
                        if baseline_val and ctr not in rust_stats:
                            # Full key removal is exactly the collapse-to-zero
                            # case this block exists to catch, but None != 0 so
                            # the `== 0` branch below cannot see it. Flag the
                            # dropped key explicitly. (angr-hv4lt.17)
                            print(f"  FAST-PATH COUNTER DROPPED: {ctr} absent from stats() (baseline {baseline_val})")
                            failures.append(f"{name}: {ctr} dropped from stats() (baseline {baseline_val})")
                        elif baseline_val and current_val == 0:
                            print(f"  FAST-PATH REGRESSION: {ctr} 0 vs baseline {baseline_val}")
                            failures.append(f"{name}: {ctr} collapsed to 0 (baseline {baseline_val})")

            # Check peak-RSS regression. Distinct from the nightly leak
            # check (run_leak_check.py), which gates iterative growth on a
            # single bench — this catches a one-shot allocation blowup on
            # any bench before it reaches hard OOM under the RLIMIT_AS cap.
            # ru_maxrss is a process-wide high-water mark, so the threshold
            # is intentionally generous (see --memory-threshold help).
            baseline_mem = baseline[baseline_key].get("peak_memory_mb") if not args.no_memory_check else None
            pct = memory_regression_pct(baseline_mem, rust_peak_mem, args.memory_threshold)
            if pct is not None:
                tag = "MEMORY WARN" if args.memory_warn_only else "MEMORY REGRESSION"
                print(f"  {tag}: peak_memory_mb {rust_peak_mem:.0f}MB vs baseline {baseline_mem:.0f}MB (+{pct:.0f}%)")
                if not args.memory_warn_only:
                    mem_msg = (
                        f"{name}: peak_memory_mb regression "
                        f"({rust_peak_mem:.0f}MB vs {baseline_mem:.0f}MB, +{pct:.0f}%)"
                    )
                    failures.append(mem_msg)
                    if args.retry_failures > 0:
                        rec = _retry_record(name, timeout, strategy, baseline_key)
                        rec["baseline_mem"] = baseline_mem
                        rec["mem_msg"] = mem_msg

        # SLA check: enforce minimum speedup vs Python. When the *global*
        # --rust-only flag skipped Python this run, fall back to the
        # python_time cached in baseline so the SLA still gates normal
        # benches. Per-bench `rust_only=True` entries are deliberately
        # SLA-EXEMPT (see the FAST_SUITE rust_only comments, e.g.
        # android_arm_license_validation): they are tiny / arch-edge benches
        # where Rust is legitimately slower than Python and we do not want
        # the speedup gate to flag them — so `entry_rust_only` suppresses the
        # fallback. Quietly skipped when no python_time is known.
        if not args.no_sla:
            sla_py_time = py_time
            if sla_py_time is None and not entry_rust_only and baseline_key in baseline:
                sla_py_time = baseline[baseline_key].get("python_time")
            if sla_py_time is not None and sla_py_time > 0 and rust_time > 0:
                sla_speedup = sla_py_time / rust_time
                if sla_speedup < args.sla_fail_threshold:
                    print(
                        f"  SLA FAIL: {sla_speedup:.2f}x < {args.sla_fail_threshold:.2f}x "
                        f"(Python {sla_py_time:.2f}s / Rust {rust_time:.2f}s)"
                    )
                    sla_msg = f"{name}: SLA fail ({sla_speedup:.2f}x < {args.sla_fail_threshold:.2f}x)"
                    failures.append(sla_msg)
                    if args.retry_failures > 0:
                        rec = _retry_record(name, timeout, strategy, baseline_key)
                        rec["sla_py_time"] = sla_py_time
                        rec["sla_msg"] = sla_msg
                elif sla_speedup < args.sla_warn_threshold:
                    print(
                        f"  SLA WARN: {sla_speedup:.2f}x < {args.sla_warn_threshold:.2f}x "
                        f"(Python {sla_py_time:.2f}s / Rust {rust_time:.2f}s)"
                    )

        # Print metric summary
        metric_parts = []
        for stat_key, bl_key, _ in TRACKED_METRICS:
            val = rust_stats.get(stat_key)
            if val is not None:
                metric_parts.append(f"{bl_key}={val}")
        if rust_peak_mem:
            metric_parts.append(f"peak_mem={rust_peak_mem:.0f}MB")
        if metric_parts:
            print(f"  metrics: {', '.join(metric_parts)}")

        entry = {
            "rust_time": round(rust_time, 3),
        }
        if py_time is not None:
            entry["python_time"] = round(py_time, 3)
        else:
            # Preserve cached python_time from prior runs so SLA checks
            # remain useful for rust_only entries (Python skipped this run
            # but we want to keep historical timing).
            cached_py = baseline.get(baseline_key, {}).get("python_time")
            entry["python_time"] = cached_py
        # Store algorithmic metrics in baseline
        for stat_key, bl_key, _ in TRACKED_METRICS:
            val = rust_stats.get(stat_key)
            if val is not None:
                entry[bl_key] = val
        if rust_peak_mem:
            entry["peak_memory_mb"] = round(rust_peak_mem, 1)
        results[baseline_key] = entry

    # Host-slowdown normalization pass. Runs BEFORE the retry pass on purpose:
    # a failure that the suite median already explains needs no re-run at all,
    # and re-running it inside the same contention window is precisely what
    # failed to help on ralph iter24. See `host_scale_factor` (angr-mc8pw).
    if timing_failures and not args.no_host_normalize:
        median = suite_median_ratio(timing_ratios)
        factor = host_scale_factor(median, args.host_factor_cap)
        if factor > 1.0:
            print(f"\n{'=' * 50}")
            print(
                f"Host-slowdown normalization: suite median {median:.3f}x over "
                f"{len(timing_ratios)} benches — re-checking "
                f"{len(timing_failures)} timing failure(s) against scaled baselines"
            )
            for failure in list(timing_failures):
                scaled_bl = failure["bl"] * factor
                rust_time = failure["rust_time"]
                norm_pct = ((rust_time / scaled_bl) - 1) * 100
                if timing_regression_pct(scaled_bl, rust_time, args.threshold, args.regression_floor) is None:
                    print(
                        f"  {failure['label']}: {rust_time:.2f}s vs "
                        f"{failure['bl']:.2f}s x {factor:.3f} = {scaled_bl:.2f}s "
                        f"({norm_pct:+.0f}%) — cleared, host-wide slowdown"
                    )
                    failures.remove(failure["msg"])
                    timing_failures.remove(failure)
                    rec = failure["retry_rec"]
                    if rec is not None:
                        rec.pop("timing_msg", None)
                        if not any(k in rec for k in ("sla_msg", "mem_msg")):
                            retry_info.pop(failure["name"], None)
                else:
                    print(
                        f"  {failure['label']}: {rust_time:.2f}s vs "
                        f"{failure['bl']:.2f}s x {factor:.3f} = {scaled_bl:.2f}s "
                        f"(+{norm_pct:.0f}%) — still regressed"
                    )
                    # Keep the retry pass consistent with the bar just applied:
                    # a retry judged against the raw baseline would be STRICTER
                    # than the check this bench just survived.
                    if failure["retry_rec"] is not None:
                        failure["retry_rec"]["bl_scaled"] = scaled_bl
        elif median is not None and median > args.host_factor_cap:
            # Refuse to normalize, but never let that refusal be silent — the
            # numbers below are not trustworthy either way.
            print(f"\n{'=' * 50}")
            print(
                f"SUITE-WIDE SLOWDOWN: median {median:.3f}x over "
                f"{len(timing_ratios)} benches exceeds the "
                f"{args.host_factor_cap:.2f}x cap — NOT normalized. Either the "
                "host is unusable for benchmarking or every bench regressed; "
                "the failures below stand."
            )

    # Optional retry pass for timing-regression failures. Sub-second benches are
    # noise-dominated; the same diff can flip pass/fail across consecutive runs.
    # We re-run each candidate up to N times and clear the failure on the first
    # measurement that lands within threshold.
    if args.retry_failures > 0 and retry_info:
        n_dims = sum(("timing_msg" in r) + ("sla_msg" in r) + ("mem_msg" in r) for r in retry_info.values())
        print(f"\n{'=' * 50}")
        print(
            f"Retry pass: {len(retry_info)} bench(es), {n_dims} failure(s) "
            f"(timing/SLA/memory), up to {args.retry_failures} attempt(s) each"
        )
        for rec in retry_info.values():
            name, strategy = rec["name"], rec["strategy"]
            label = f"{name} (DFS)" if strategy == "dfs" else name
            print(f"\n--- retry: {label} ---")
            # Outstanding dimensions to clear for this bench.
            pending = {k for k in ("timing_msg", "sla_msg", "mem_msg") if k in rec}
            for attempt in range(1, args.retry_failures + 1):
                if not pending:
                    break
                retry_result = run_one(name, "rust", rec["timeout"], rec["mem_limit"], strategy)
                if not retry_result.get("ok"):
                    print(f"  attempt {attempt}: Rust FAIL ({retry_result.get('error', '?')})")
                    continue
                retry_time = retry_result["elapsed"]
                retry_mem = retry_result.get("peak_memory_mb")
                # NOTE: results[baseline_key] keeps the ORIGINAL measurements on
                # purpose. The retry signals pass/fail only — overwriting with
                # the lower retry value would tighten the next baseline refresh
                # and drift the gate downward over time (see bd memory
                # `avoid-update-baseline-without-verification`).
                if "timing_msg" in pending:
                    # `bl_scaled` is present only when the host-normalization
                    # pass ran and did not clear this bench; comparing against
                    # the raw baseline here would apply a stricter bar than the
                    # check it just survived (angr-mc8pw).
                    bl = rec.get("bl_scaled") or rec["bl"]
                    pct = ((retry_time / bl) - 1) * 100
                    # Same two-bar decision as the first pass, so a retry that
                    # lands inside the absolute floor clears the failure.
                    if timing_regression_pct(bl, retry_time, args.threshold, args.regression_floor) is None:
                        print(f"  attempt {attempt}: timing {retry_time:.2f}s vs {bl:.2f}s ({pct:+.0f}%) — cleared")
                        pending.discard("timing_msg")
                    else:
                        print(
                            f"  attempt {attempt}: timing {retry_time:.2f}s vs {bl:.2f}s (+{pct:.0f}%) — still regressed"
                        )
                if "sla_msg" in pending:
                    speedup = rec["sla_py_time"] / retry_time if retry_time > 0 else 0
                    if speedup >= args.sla_fail_threshold:
                        print(f"  attempt {attempt}: SLA {speedup:.2f}x >= {args.sla_fail_threshold:.2f}x — cleared")
                        pending.discard("sla_msg")
                    else:
                        print(
                            f"  attempt {attempt}: SLA {speedup:.2f}x < {args.sla_fail_threshold:.2f}x — still failing"
                        )
                if "mem_msg" in pending:
                    bmem = rec["baseline_mem"]
                    if retry_mem is not None and retry_mem <= bmem * (1 + args.memory_threshold):
                        print(f"  attempt {attempt}: memory {retry_mem:.0f}MB vs {bmem:.0f}MB — cleared")
                        pending.discard("mem_msg")
                    elif retry_mem is not None:
                        pct = ((retry_mem / bmem) - 1) * 100
                        print(
                            f"  attempt {attempt}: memory {retry_mem:.0f}MB vs {bmem:.0f}MB (+{pct:.0f}%) — still regressed"
                        )
            # Any dimension cleared above is removed from the canonical failures
            # list; dimensions still pending stay failed.
            for key in ("timing_msg", "sla_msg", "mem_msg"):
                if key in rec and key not in pending:
                    failures.remove(rec[key])

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 50}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Benchmarks: {len(REGRESSION_SUITE)}, Passed: {len(results)}, Failed: {len(failures)}")

    if args.update and results:
        baseline.update(results)
        save_baseline(baseline)
        if current_counters:
            baseline_counters.update(current_counters)
            save_baseline_counters(baseline_counters)
    elif args.update_counters and current_counters:
        baseline_counters.update(current_counters)
        save_baseline_counters(baseline_counters)

    if args.history_record and results:
        save_history_record(args.history_record, results)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        if libvex_on is False:
            print(f"\nNOTE: {LIBVEX_OFF_NOTE}")
        sys.exit(1)
    else:
        print("\nAll benchmarks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
