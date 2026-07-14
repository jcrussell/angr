#!/usr/bin/env python3
"""Leak-check harness: run a Callable-heavy bench N times in one process and
fail if peak RSS grows beyond ``--threshold * peak_after_iter_1``.

Codifies the bisection pattern from the ``9maq-bisect-method`` bd memory
(used to find the 286 MB -> 1888 MB Callable leak fixed in 293aa8163) into a
nightly CI gate. The idea is simple: a healthy implementation runs the same
Callable workload N times and peak RSS stays roughly flat. A regression that
leaks Z3 contexts, AST refs, Python state metadata, etc. shows up as ratio
``peak_rss(iter_N) / peak_rss(iter_1) >> 1``.

The child runs under ``RLIMIT_AS`` so a runaway leak fails fast instead of
OOM-killing the orchestrator on an 8GB/0-swap CI runner.

Usage:
    python tests/benchmarks/run_leak_check.py                       # mma_howtouse, N=10, threshold 1.5x
    python tests/benchmarks/run_leak_check.py --iters 5             # quick variant
    python tests/benchmarks/run_leak_check.py --threshold 1.3       # tighter gate
    python tests/benchmarks/run_leak_check.py --example flareon2015_10
    python tests/benchmarks/run_leak_check.py --json                # machine-readable

Exit codes:
    0 — pass
    1 — leak detected (ratio above threshold) or iteration failure
    2 — setup / harness error (solve.py missing, child crashed, etc.)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys

DEFAULT_EXAMPLE = "mma_howtouse"
DEFAULT_ITERS = 10
# Threshold rationale: pre-fix mma_howtouse went 286 MB -> 1888 MB (~6.6x) in
# the bisect that motivated this harness. Post-fix, repeated main() calls
# share Z3+claripy caches and grow only modestly. 1.5x leaves comfortable
# headroom for normal cache warmup while still flagging a >=2x regression
# with margin. Tune via --threshold when adding new Callable-heavy benches.
#
# Re-measured on the *Rust* engine once the engine swap was actually installed
# (angr-pm8ha; before that this gate silently measured Python): mma_howtouse
# N=10 grows 285 MB -> 301 MB, ratio 1.054 (~1.7 MB/iter, 450 Callable-built
# RustExplorationManagers). Same order as the Python-engine number the 1.5x was
# tuned on, so the threshold carries over unchanged.
DEFAULT_THRESHOLD = 1.5
DEFAULT_MEM_LIMIT_MB = 4096
DEFAULT_TIMEOUT_SEC = 600

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")


def _run_iters_in_child(example_name, iters, mem_limit_mb, examples_dir, engine):
    """Run ``solve.main()`` for ``example_name`` ``iters`` times in this
    process; return ``{"ok": bool, "per_iter_rss_kb": [...], "per_iter_wall_s": [...]}``.

    Called inside an mp.spawn child so the RLIMIT_AS bump is scoped.
    """
    import importlib.util
    import resource as _res
    import time

    mem_bytes = mem_limit_mb * 1024 * 1024
    _soft, _hard = _res.getrlimit(_res.RLIMIT_AS)
    try:
        _res.setrlimit(_res.RLIMIT_AS, (mem_bytes, _hard))
    except ValueError:
        pass

    # mp-spawn children inherit sys.path[0] = script dir, not the cwd
    # (cf. ``venv-script-syspath`` bd memory). Prepend repo root explicitly.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    solve_script = os.path.join(examples_dir, example_name, "solve.py")
    if not os.path.exists(solve_script):
        return {"ok": False, "error": f"solve.py not found: {solve_script}"}

    example_dir = os.path.dirname(solve_script)
    os.chdir(example_dir)
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    import angr  # noqa: F401 — make sure the editable install is reachable

    # The bench builds its managers through project.factory.simulation_manager,
    # which yields a *Python* SimulationManager. Without the engine swap this
    # gate would measure the Python engine's RSS growth, not the Rust engine's
    # (angr-pm8ha) — the Callable/cache leaks it exists to catch are Rust-side.
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)

    managers_built = 0
    if engine == "rust":
        from engine_patch import install_rust_engine_patch

        from angr.exploration import RustExplorationManager

        def _make_rust_manager(project, states):
            nonlocal managers_built
            managers_built += 1
            return RustExplorationManager(project, states)

        install_rust_engine_patch(_make_rust_manager)

    spec = importlib.util.spec_from_file_location("solve_mod", solve_script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {"ok": False, "error": f"solve.py import failed: {type(e).__name__}: {e}"}

    if not hasattr(module, "main"):
        return {"ok": False, "error": "solve.py has no main()"}

    per_iter_rss_kb = []
    per_iter_wall_s = []
    for i in range(iters):
        t0 = time.perf_counter()
        try:
            module.main()
        except MemoryError:
            return {
                "ok": False,
                "error": f"MemoryError at iter {i + 1} (hit {mem_limit_mb} MB RLIMIT_AS)",
                "per_iter_rss_kb": per_iter_rss_kb,
                "per_iter_wall_s": per_iter_wall_s,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"main() raised at iter {i + 1}: {type(e).__name__}: {e}",
                "per_iter_rss_kb": per_iter_rss_kb,
                "per_iter_wall_s": per_iter_wall_s,
            }
        per_iter_wall_s.append(time.perf_counter() - t0)
        # ru_maxrss is in KiB on Linux and is the running max for this
        # process — so it is monotonically non-decreasing across iters,
        # which is exactly the signal we want for "did peak grow?".
        per_iter_rss_kb.append(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss)

    if engine == "rust" and managers_built == 0:
        # Guards the whole point of the gate: a bench whose managers never
        # reach Rust (e.g. an exclusion in engine_patch swallows them) would
        # otherwise report a healthy Python-engine RSS series.
        return {
            "ok": False,
            "error": "no RustExplorationManager was built — the bench ran on the Python engine",
            "per_iter_rss_kb": per_iter_rss_kb,
            "per_iter_wall_s": per_iter_wall_s,
            "managers_built": 0,
        }

    return {
        "ok": True,
        "per_iter_rss_kb": per_iter_rss_kb,
        "per_iter_wall_s": per_iter_wall_s,
        "managers_built": managers_built,
    }


def run_leak_check(
    example_name: str,
    iters: int,
    threshold: float,
    mem_limit_mb: int,
    timeout_sec: int,
    examples_dir: str,
    engine: str = "rust",
):
    """Drive the leak check in an isolated subprocess. Returns a result dict."""
    solve_script = os.path.join(examples_dir, example_name, "solve.py")
    if not os.path.exists(solve_script):
        return {
            "ok": False,
            "error": f"solve.py not found: {solve_script}",
            "exit_code": 2,
        }

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _run_iters_in_child,
            (example_name, iters, mem_limit_mb, examples_dir, engine),
        )
        child_result = async_result.get(timeout=timeout_sec)
    except multiprocessing.TimeoutError:
        pool.terminate()
        return {
            "ok": False,
            "error": f"child timed out after {timeout_sec}s",
            "exit_code": 2,
        }
    except Exception as e:
        pool.terminate()
        return {
            "ok": False,
            "error": f"child crashed: {type(e).__name__}: {e}",
            "exit_code": 2,
        }
    finally:
        pool.terminate()
        pool.join()

    if not child_result.get("ok"):
        return {
            **child_result,
            "exit_code": 1,
            "example": example_name,
            "engine": engine,
            "iters": iters,
            "threshold": threshold,
        }

    rss_kb = child_result["per_iter_rss_kb"]
    wall_s = child_result["per_iter_wall_s"]
    iter1_rss = rss_kb[0]
    iterN_rss = rss_kb[-1]
    ratio = (iterN_rss / iter1_rss) if iter1_rss else float("inf")
    leaked = ratio > threshold

    return {
        "ok": not leaked,
        "exit_code": 1 if leaked else 0,
        "example": example_name,
        "engine": engine,
        "managers_built": child_result.get("managers_built"),
        "iters": iters,
        "threshold": threshold,
        "ratio": ratio,
        "iter1_rss_kb": iter1_rss,
        "iterN_rss_kb": iterN_rss,
        "per_iter_rss_kb": rss_kb,
        "per_iter_wall_s": wall_s,
    }


def _print_human(result: dict) -> None:
    if result.get("error"):
        print(f"ERROR: {result['error']}", file=sys.stderr)
        if result.get("per_iter_rss_kb"):
            for i, kb in enumerate(result["per_iter_rss_kb"], start=1):
                print(f"  iter {i:>2}: peak_rss_kb={kb}", file=sys.stderr)
        return

    print(
        f"example={result['example']} engine={result.get('engine')} "
        f"managers_built={result.get('managers_built')} "
        f"iters={result['iters']} threshold={result['threshold']:.2f}x"
    )
    for i, (kb, sec) in enumerate(zip(result["per_iter_rss_kb"], result["per_iter_wall_s"]), start=1):
        print(f"  iter {i:>2}: peak_rss_kb={kb:>10} wall={sec:.2f}s")
    print(
        f"  ratio = peak_rss(iter{result['iters']}) / peak_rss(iter1) = "
        f"{result['iterN_rss_kb']} / {result['iter1_rss_kb']} = {result['ratio']:.3f}"
    )
    if result["ok"]:
        print(f"PASS (ratio {result['ratio']:.3f} <= {result['threshold']:.2f})")
    else:
        print(
            f"FAIL: leak detected — ratio {result['ratio']:.3f} > {result['threshold']:.2f}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--example",
        default=DEFAULT_EXAMPLE,
        help=f"example name under ANGR_EXAMPLES_DIR (default: {DEFAULT_EXAMPLE})",
    )
    parser.add_argument(
        "--engine",
        default="rust",
        choices=("rust", "python"),
        help="engine to measure (default: rust — the engine this gate exists for)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"number of main() invocations (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(f"fail if peak_rss(iterN) / peak_rss(iter1) exceeds this (default: {DEFAULT_THRESHOLD})"),
    )
    parser.add_argument(
        "--mem-limit",
        type=int,
        default=DEFAULT_MEM_LIMIT_MB,
        help=f"RLIMIT_AS in MB for child (default: {DEFAULT_MEM_LIMIT_MB})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"child timeout in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--examples-dir",
        default=EXAMPLES_DIR,
        help="override ANGR_EXAMPLES_DIR for this run",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object on stdout instead of a human-readable table",
    )
    args = parser.parse_args()

    result = run_leak_check(
        example_name=args.example,
        iters=args.iters,
        threshold=args.threshold,
        mem_limit_mb=args.mem_limit,
        timeout_sec=args.timeout,
        examples_dir=args.examples_dir,
        engine=args.engine,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str, sort_keys=True))
    else:
        _print_human(result)

    return result.get("exit_code", 2)


if __name__ == "__main__":
    sys.exit(main())
