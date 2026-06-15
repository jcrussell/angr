#!/usr/bin/env python3
"""Benchmark regression test for the Rust symbolic execution engine.

Runs fast-tier benchmarks with both engines and checks:
1. Rust engine produces correct output (matches Python)
2. Rust engine is not slower than baseline threshold
3. Rust engine meets the speedup SLA vs Python (fails below 0.5x by default,
   warns below 1.0x). Uses cached python_time from baseline when this run
   skipped Python (e.g. rust_only entries).

Usage:
    python tests/benchmarks/run_regression.py              # Run all fast benchmarks
    python tests/benchmarks/run_regression.py --update      # Update baseline timings
    python tests/benchmarks/run_regression.py --threshold 0.2  # 20% regression threshold
    python tests/benchmarks/run_regression.py --sla-fail-threshold 0.7  # Tighter SLA
    python tests/benchmarks/run_regression.py --no-sla       # Disable SLA check

Exit codes:
    0 = all benchmarks pass
    1 = regression detected (wrong output, too slow, or SLA failure)
    2 = setup error
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import os
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

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline_timings.json")
# Optional per-bench counter snapshot. When present, a timing regression
# triggers a `bench_diff` table against the cached snapshot so CI logs
# show *which counter moved* without a manual re-run. Refreshed by
# ``--update`` alongside BASELINE_FILE. Soft dependency — absent file
# only suppresses the diff, the gate keeps working.
BASELINE_COUNTERS_FILE = os.path.join(os.path.dirname(__file__), "baseline_counters.json")

# Tracked metrics beyond timing. Each entry: (key_in_stats, key_in_baseline, regression_threshold_pct)
# A regression is flagged when the metric INCREASES by more than threshold_pct.
TRACKED_METRICS = [
    ("callback_count", "callback_count", 0.10),  # 10% more callbacks = algorithmic regression
    ("state_creations", "state_creations", 0.10),
    ("steps", "steps", 0.15),  # 15% more steps
]

# Tiered benchmark suites. Each entry: (name, timeout_seconds, [strategy, [rust_only]])
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
    ("cow_fork_scaling", 30, "bfs", True),
]

# Medium tier: 10-60s, run with --full
MEDIUM_SUITE = [
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
    }
)


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
    output = re.sub(r"(\\x[0-9a-fA-F]{2}[^']*?)(')", r"\2", output)
    return output


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
    parser.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT_MB)
    parser.add_argument("--rust-only", action="store_true", help="Only run Rust engine (skip Python comparison)")
    parser.add_argument("--full", action="store_true", help="Run full suite (fast + medium tier)")
    parser.add_argument(
        "--check-counts",
        action="store_true",
        help="Check algorithmic metrics (callback_count, state_creations, steps) for regressions",
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
        help="Re-run timing-regression failures up to N times. A "
        "single retry passing within --threshold clears the "
        "failure. Mitigates sub-second-bench noise where the "
        "same diff can flip pass/fail across runs without code "
        "changes (see bd memory benchmark-regression-noise-floor). "
        "Engine errors, output mismatches, SLA failures, and "
        "metric regressions are never retried.",
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

    baseline = load_baseline()
    baseline_counters = load_baseline_counters()
    # Per-bench counter dict captured this run. Used to refresh
    # baseline_counters.json when --update is set, and as the
    # "current" side of the regression diff against
    # baseline_counters[baseline_key].
    current_counters: dict[str, dict] = {}
    results = {}
    failures = []
    # When --retry-failures > 0, each timing-regression failure is recorded here
    # alongside everything we need to re-run it. The retry pass below then drops
    # the matching failures[] entry if any retry comes in within threshold.
    retry_candidates = []  # list of (failure_msg, name, timeout, strategy, mem_limit, baseline_rust_time, baseline_key)
    total_start = time.perf_counter()

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
            if norm_rust != norm_py:
                # Also try whitespace-tolerant comparison
                if norm_rust.split() != norm_py.split():
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
            if rust_time > bl * (1 + args.threshold):
                pct = ((rust_time / bl) - 1) * 100
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
                if args.retry_failures > 0:
                    retry_candidates.append((failure_msg, name, timeout, strategy, args.mem_limit, bl, baseline_key))

            # Check algorithmic metric regressions
            if args.check_counts:
                for stat_key, bl_key, threshold_pct in TRACKED_METRICS:
                    current_val = rust_stats.get(stat_key)
                    baseline_val = baseline[baseline_key].get(bl_key)
                    if current_val is not None and baseline_val is not None and baseline_val > 0:
                        if current_val > baseline_val * (1 + threshold_pct):
                            pct = ((current_val / baseline_val) - 1) * 100
                            print(
                                f"  METRIC REGRESSION: {bl_key} {current_val} vs baseline {baseline_val} (+{pct:.0f}%)"
                            )
                            failures.append(
                                f"{name}: {bl_key} regression ({current_val} vs {baseline_val}, +{pct:.0f}%)"
                            )

        # SLA check: enforce minimum speedup vs Python. Falls back to the
        # python_time cached in baseline when this run skipped Python (e.g.
        # rust_only entries). Quietly skipped when no python_time is known.
        if not args.no_sla:
            sla_py_time = py_time
            if sla_py_time is None and baseline_key in baseline:
                sla_py_time = baseline[baseline_key].get("python_time")
            if sla_py_time is not None and sla_py_time > 0 and rust_time > 0:
                sla_speedup = sla_py_time / rust_time
                if sla_speedup < args.sla_fail_threshold:
                    print(
                        f"  SLA FAIL: {sla_speedup:.2f}x < {args.sla_fail_threshold:.2f}x "
                        f"(Python {sla_py_time:.2f}s / Rust {rust_time:.2f}s)"
                    )
                    failures.append(f"{name}: SLA fail ({sla_speedup:.2f}x < {args.sla_fail_threshold:.2f}x)")
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

    # Optional retry pass for timing-regression failures. Sub-second benches are
    # noise-dominated; the same diff can flip pass/fail across consecutive runs.
    # We re-run each candidate up to N times and clear the failure on the first
    # measurement that lands within threshold.
    if args.retry_failures > 0 and retry_candidates:
        print(f"\n{'=' * 50}")
        print(f"Retry pass: {len(retry_candidates)} timing regression(s), up to {args.retry_failures} attempt(s) each")
        for failure_msg, name, timeout, strategy, mem_limit, bl, baseline_key in retry_candidates:
            label = f"{name} (DFS)" if strategy == "dfs" else name
            print(f"\n--- retry: {label} ---")
            cleared = False
            for attempt in range(1, args.retry_failures + 1):
                retry_result = run_one(name, "rust", timeout, mem_limit, strategy)
                if not retry_result.get("ok"):
                    print(f"  attempt {attempt}: Rust FAIL ({retry_result.get('error', '?')})")
                    continue
                retry_time = retry_result["elapsed"]
                if retry_time <= bl * (1 + args.threshold):
                    pct = ((retry_time / bl) - 1) * 100
                    print(
                        f"  attempt {attempt}: {retry_time:.2f}s vs baseline {bl:.2f}s ({pct:+.0f}%) — within threshold, clearing failure"
                    )
                    cleared = True
                    # NOTE: results[baseline_key]["rust_time"] keeps the original
                    # measurement on purpose. The retry signals pass/fail only —
                    # overwriting with the lower retry value would tighten the
                    # next baseline refresh and drift the gate downward over
                    # time (see bd memory `avoid-update-baseline-without-verification`).
                    break
                pct = ((retry_time / bl) - 1) * 100
                print(f"  attempt {attempt}: {retry_time:.2f}s vs baseline {bl:.2f}s (+{pct:.0f}%) — still regressed")
            if cleared:
                failures.remove(failure_msg)

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
        sys.exit(1)
    else:
        print("\nAll benchmarks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
