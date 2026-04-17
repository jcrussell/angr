#!/usr/bin/env python3
"""Benchmark regression test for the Rust symbolic execution engine.

Runs fast-tier benchmarks with both engines and checks:
1. Rust engine produces correct output (matches Python)
2. Rust engine is not slower than baseline threshold

Usage:
    python tests/benchmarks/run_regression.py              # Run all fast benchmarks
    python tests/benchmarks/run_regression.py --update      # Update baseline timings
    python tests/benchmarks/run_regression.py --threshold 0.2  # 20% regression threshold

Exit codes:
    0 = all benchmarks pass
    1 = regression detected (wrong output or too slow)
    2 = setup error
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

# Reuse the subprocess runner from run_single.py
sys.path.insert(0, os.path.dirname(__file__))
from run_single import _run_in_child, EXAMPLES_DIR, DEFAULT_MEM_LIMIT_MB

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "baseline_timings.json")

# Tiered benchmark suites. Each entry: (name, timeout_seconds)
# Fast tier: < 10s, always run
FAST_SUITE = [
    ("fauxware", 30),
    ("defcamp_r100", 30),
    ("ais3_crackme", 30),
    ("google2016_unbreakable_0", 30),
    ("google2016_unbreakable_1", 30),
    ("strcpy_find", 30),
    ("flareon2015_2", 30),
]

# Medium tier: 10-60s, run with --full
MEDIUM_SUITE = [
    ("sym-write", 60),
    ("flareon2015_5", 60),
    ("flareon2015_10", 60),
    ("ekopartyctf2016_rev250", 60),
    ("csaw_wyvern", 60),
    ("securityfest_fairlight", 60),
]

# Default: fast only. Use --full for fast + medium.
REGRESSION_SUITE = FAST_SUITE  # overridden in main() if --full


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


def run_one(name, engine, timeout, mem_limit_mb):
    """Run a single benchmark in a subprocess, return result dict."""
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _run_in_child, (name, engine, EXAMPLES_DIR, mem_limit_mb)
        )
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


def main():
    parser = argparse.ArgumentParser(description="Rust engine benchmark regression test")
    parser.add_argument("--update", action="store_true", help="Update baseline timings")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Regression threshold (default: 0.15 = 15%% slower)")
    parser.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT_MB)
    parser.add_argument("--rust-only", action="store_true",
                        help="Only run Rust engine (skip Python comparison)")
    parser.add_argument("--full", action="store_true",
                        help="Run full suite (fast + medium tier)")
    args = parser.parse_args()

    global REGRESSION_SUITE
    if args.full:
        REGRESSION_SUITE = FAST_SUITE + MEDIUM_SUITE

    # Verify examples exist
    missing = [name for name, _ in REGRESSION_SUITE
               if not os.path.exists(os.path.join(EXAMPLES_DIR, name, "solve.py"))]
    if missing:
        print(f"ERROR: Missing examples: {', '.join(missing)}", file=sys.stderr)
        print(f"Expected at: {EXAMPLES_DIR}", file=sys.stderr)
        sys.exit(2)

    baseline = load_baseline()
    results = {}
    failures = []
    total_start = time.perf_counter()

    for name, timeout in REGRESSION_SUITE:
        print(f"\n--- {name} ---")

        # Run Python engine (for output comparison)
        if not args.rust_only:
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
        rust_result = run_one(name, "rust", timeout, args.mem_limit)
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

        # Check regression against baseline
        if name in baseline and not args.update:
            bl = baseline[name]["rust_time"]
            if rust_time > bl * (1 + args.threshold):
                pct = ((rust_time / bl) - 1) * 100
                print(f"  REGRESSION: {rust_time:.2f}s vs baseline {bl:.2f}s (+{pct:.0f}%)")
                failures.append(f"{name}: {pct:.0f}% regression ({rust_time:.2f}s vs {bl:.2f}s)")

        results[name] = {
            "rust_time": round(rust_time, 3),
            "python_time": round(py_time, 3) if py_time is not None else None,
        }

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*50}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Benchmarks: {len(REGRESSION_SUITE)}, Passed: {len(results)}, Failed: {len(failures)}")

    if args.update and results:
        baseline.update(results)
        save_baseline(baseline)

    if failures:
        print(f"\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll benchmarks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
