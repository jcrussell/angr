#!/usr/bin/env python3
"""Run a single angr-example with the Rust engine and print timing + callback stats.

Runs each example in an isolated subprocess with a memory limit to prevent
OOM-killing the orchestrator on an 8GB/0-swap machine.

Usage:
    python tests/benchmarks/run_single.py fauxware
    python tests/benchmarks/run_single.py ais3_crackme --engine=python
    python tests/benchmarks/run_single.py defcamp_r100 --both
    python tests/benchmarks/run_single.py grub --engine rust --mem-limit 2048
"""
import argparse
import multiprocessing
import os
import sys


EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
DEFAULT_MEM_LIMIT_MB = 4096  # 4 GB — leaves 4 GB for parent + OS on 8GB machine

# Catalog of tested examples with expected behavior
# tier: "fast" (<5s), "medium" (5-30s), "slow" (30-120s), "very_slow" (>120s)
EXAMPLE_CATALOG = {
    # === Core benchmark suite (fast, always correct) ===
    "fauxware":                {"tier": "fast",    "rust_ok": True,  "notes": "SimProcedure callbacks"},
    "defcamp_r100":            {"tier": "fast",    "rust_ok": True,  "notes": "Basic find/avoid"},
    "ais3_crackme":            {"tier": "fast",    "rust_ok": True,  "notes": "Symbolic argv, state forking"},
    "sym-write":               {"tier": "medium",  "rust_ok": True,  "notes": "Symbolic writes, callable predicates"},
    "securityfest_fairlight":  {"tier": "medium",  "rust_ok": True,  "notes": "Heavy VEX interpretation"},
    "flareon2015_5":           {"tier": "medium",   "rust_ok": True,  "notes": "Complex symbolic memory"},
    "flareon2015_10":          {"tier": "medium",  "rust_ok": True,  "notes": "Callable step_func, pruning"},
    "ekopartyctf2016_rev250":  {"tier": "medium",  "rust_ok": True,  "notes": "Deep constraint solving"},
    "csaw_wyvern":             {"tier": "medium",  "rust_ok": True,  "notes": "Linear constraints"},
    # === Extended examples ===
    "codegate_2017-angrybird": {"tier": "medium",  "rust_ok": True,  "notes": "LAZY_SOLVES, manual state init"},
    "google2016_unbreakable_0":{"tier": "fast",    "rust_ok": True,  "notes": "Basic constraint solving"},
    "google2016_unbreakable_1":{"tier": "fast",    "rust_ok": True,  "notes": "Multi-step constraints"},
    # === New benchmark candidates (untested with Rust engine) ===
    "sharif7_rev50":           {"tier": "very_slow","rust_ok": None, "notes": "Both engines timeout >60s"},
    "defcon2016quals_baby-re": {"tier": "medium",  "rust_ok": True,  "notes": "scanf hooks, Py 1.5s Rust 28s (slow)"},
    "asisctffinals2015_license":{"tier": "medium", "rust_ok": False, "notes": "Rust list index error"},
    "0ctf_momo_3":             {"tier": "very_slow","rust_ok": None, "notes": "Both engines timeout >60s"},
    "csgames2018":             {"tier": "fast",    "rust_ok": True,  "notes": "Callable predicates, stdout check"},
    "unmapped_analysis":       {"tier": "fast",    "rust_ok": True,  "notes": "STRICT_PAGE_ACCESS, DivModU128to64 fix"},
    "hitcon2017_sakura":       {"tier": "very_slow","rust_ok": True,  "notes": "Multi-stage explore, disk cache fix"},
    "strcpy_find":             {"tier": "fast",    "rust_ok": True,  "notes": "Buffer overflow / strcpy detection, Py 0.8s Rust 4.0s"},
    "flareon2015_2":           {"tier": "fast",    "rust_ok": True,  "notes": "32-bit x86, correct output"},
    "whitehatvn2015_re400":    {"tier": "fast",    "rust_ok": True,  "notes": "2.7x speedup, partial output divergence (leading zeros)"},
    "mma_howtouse":            {"tier": "medium",  "rust_ok": True,  "notes": "Callable DLL, 45 calls, Py 4.3s Rust 6.6s (0.65x)"},
    "defcamp_r200":            {"tier": "medium",  "rust_ok": None,  "notes": "BROKEN: Python fails too (ManualMergepoint)"},
    "CADET_00001":             {"tier": "slow",    "rust_ok": False, "notes": "save_unconstrained, syscall errors, Rust timeout"},
    "ekopartyctf2015_rev100":  {"tier": "medium",  "rust_ok": False, "notes": "30 sim managers, run(n=4)/step(size=), Rust timeout"},
    "whitehat_crypto400":      {"tier": "medium",  "rust_ok": False, "notes": "Multi-stage explore+unstash, Rust list index error"},
    "ekopartyctf2016_sokohashv2":{"tier": "medium", "rust_ok": True,  "notes": "save_unconstrained, Windows PE, Py 8.6s Rust 18.6s (0.46x)"},
    "insomnihack_aeg":         {"tier": "medium",  "rust_ok": None,  "notes": "AEG pattern, uses sys.argv in __main__, incompatible with harness"},
    "0ctf_trace":              {"tier": "medium",  "rust_ok": None,  "notes": "MIPS blob, uses factory.successors() not simgr, Rust engine unused, Py 26.9s"},
    # === Slow/problematic examples ===
    "hackcon2016_angry-reverser":{"tier": "slow",  "rust_ok": True,  "notes": "LAZY_SOLVES, Z3 structure mismatch"},
    "asisctffinals2015_fake":  {"tier": "very_slow","rust_ok": False, "notes": "Z3 AST structure too complex for post-exploration solve"},
    "b01lersctf2020_little_engine":{"tier": "very_slow", "rust_ok": None, "notes": "~150s Python, untested Rust"},
    "tumctf2016_zwiebel":      {"tier": "very_slow","rust_ok": None, "notes": "Self-modifying code, ~2.5h"},
}


def _run_in_child(example_name, engine, examples_dir, mem_limit_mb, strategy="bfs",
                  diff_state=False, diff_interval=1, diff_max_snapshots=200):
    """Run a single example in a subprocess. Called via multiprocessing spawn."""
    import io
    import importlib.util
    import resource
    import time

    # Set memory limit BEFORE importing angr (which is heavy)
    mem_bytes = mem_limit_mb * 1024 * 1024
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, _hard))
    except ValueError:
        pass  # Can't set limit higher than hard limit

    # multiprocessing spawn children inherit sys.path[0] = script dir, not cwd,
    # and the editable angr install ships no .pth file — so without PYTHONPATH
    # the import below fails. Prepend the repo root explicitly.
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import angr

    solve_script = os.path.join(examples_dir, example_name, "solve.py")
    if not os.path.exists(solve_script):
        return {"ok": False, "error": f"solve.py not found: {solve_script}"}

    example_dir = os.path.dirname(solve_script)

    # Track the RustExplorationManager instance for stats
    rust_mgr_instance = None
    snapshots: list = []
    original_sm = None
    original_simgr = None

    # Import shared utility from the benchmarks directory (early so diff_state is available)
    _bench_dir = os.path.dirname(os.path.abspath(__file__))
    if _bench_dir not in sys.path:
        sys.path.insert(0, _bench_dir)
    from test_utils import BufferedStringIO
    if diff_state:
        from diff_state import install_snapshotter

    if engine == "rust":
        from angr.exploration import RustExplorationManager

        original_sm = angr.factory.AngrObjectFactory.simulation_manager
        original_simgr = angr.factory.AngrObjectFactory.simgr

        def patched_simulation_manager(factory_self, thing=None, **kwargs):
            nonlocal rust_mgr_instance
            # Don't intercept calls from angr internals (CFG, analyses, etc.)
            import traceback
            caller_frames = traceback.extract_stack()
            for frame in caller_frames[:-1]:
                if '/angr/analyses/' in frame.filename or '/angr/exploration_techniques/' in frame.filename:
                    return original_sm(factory_self, thing, **kwargs)
            if thing is None:
                states = [factory_self.entry_state()]
            elif isinstance(thing, (list, tuple)):
                states = list(thing)
            else:
                states = [thing]
            rust_mgr_instance = RustExplorationManager(factory_self.project, states)
            rust_mgr_instance.enable_profiling()
            if strategy == 'dfs':
                rust_mgr_instance.set_exploration_strategy('dfs')
            if diff_state:
                install_snapshotter(rust_mgr_instance, snapshots,
                                    interval=diff_interval,
                                    max_snapshots=diff_max_snapshots)
            return rust_mgr_instance

        angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
        angr.factory.AngrObjectFactory.simgr = patched_simulation_manager
    elif diff_state:
        # For the python engine we still need to install the snapshotter on
        # the SimulationManager that solve.py builds.
        original_sm = angr.factory.AngrObjectFactory.simulation_manager
        original_simgr = angr.factory.AngrObjectFactory.simgr

        def patched_python_simulation_manager(factory_self, thing=None, **kwargs):
            mgr = original_sm(factory_self, thing, **kwargs)
            import traceback
            caller_frames = traceback.extract_stack()
            for frame in caller_frames[:-1]:
                if '/angr/analyses/' in frame.filename or '/angr/exploration_techniques/' in frame.filename:
                    return mgr
            install_snapshotter(mgr, snapshots,
                                interval=diff_interval,
                                max_snapshots=diff_max_snapshots)
            return mgr

        angr.factory.AngrObjectFactory.simulation_manager = patched_python_simulation_manager
        angr.factory.AngrObjectFactory.simgr = patched_python_simulation_manager

    try:
        os.chdir(example_dir)
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        spec = importlib.util.spec_from_file_location("__main__", solve_script)
        module = importlib.util.module_from_spec(spec)

        captured = BufferedStringIO()
        original_stdout = sys.stdout
        sys.stdout = captured

        start = time.perf_counter()
        try:
            spec.loader.exec_module(module)
        except MemoryError:
            sys.stdout = original_stdout
            elapsed = time.perf_counter() - start
            return {"ok": False, "error": "MemoryError (hit memory limit)", "elapsed": elapsed}
        except Exception as e:
            sys.stdout = original_stdout
            elapsed = time.perf_counter() - start
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed": elapsed}
        finally:
            sys.stdout = original_stdout

        elapsed = time.perf_counter() - start
        output = captured.getvalue()

        # Collect peak memory usage (includes child processes)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        peak_memory_mb = rusage.ru_maxrss / 1024  # ru_maxrss is in KB on Linux

        # Collect stats if available
        stats = None
        perf_report = None
        if engine == "rust" and rust_mgr_instance is not None:
            try:
                stats = dict(rust_mgr_instance.stats)
            except Exception:
                pass
            try:
                perf_report = rust_mgr_instance.perf_report()
            except Exception:
                pass

        return {
            "ok": True,
            "elapsed": elapsed,
            "output": output,
            "stats": stats,
            "perf_report": perf_report,
            "peak_memory_mb": round(peak_memory_mb, 1),
            "snapshots": snapshots if diff_state else None,
        }

    finally:
        if original_sm is not None:
            angr.factory.AngrObjectFactory.simulation_manager = original_sm
            angr.factory.AngrObjectFactory.simgr = original_simgr


def run_example(example_name, engine, timeout=180, mem_limit_mb=DEFAULT_MEM_LIMIT_MB, strategy="bfs",
                diff_state=False, diff_interval=1, diff_max_snapshots=200):
    """Run an example in an isolated subprocess and print results."""
    solve_script = os.path.join(EXAMPLES_DIR, example_name, "solve.py")
    if not os.path.exists(solve_script):
        print(f"ERROR: solve.py not found: {solve_script}")
        return None

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _run_in_child,
            (example_name, engine, EXAMPLES_DIR, mem_limit_mb, strategy,
             diff_state, diff_interval, diff_max_snapshots),
        )
        result = async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        print(f"TIMEOUT {engine} {example_name} after {timeout}s")
        return None
    except Exception as e:
        pool.terminate()
        print(f"FAIL {engine} {example_name}: subprocess crashed: {e}")
        return None
    finally:
        pool.terminate()
        pool.join()

    if not result.get("ok"):
        elapsed = result.get("elapsed")
        elapsed_str = f" {elapsed:.2f}s" if elapsed is not None else ""
        print(f"FAIL {engine} {example_name}{elapsed_str}: {result.get('error', 'unknown error')}")
        return result

    elapsed = result["elapsed"]
    output = result.get("output", "")
    stats = result.get("stats")
    perf_report = result.get("perf_report")
    peak_memory_mb = result.get("peak_memory_mb")

    mem_str = f" peak_mem={peak_memory_mb:.0f}MB" if peak_memory_mb else ""
    print(f"OK {engine} {example_name} {elapsed:.2f}s{mem_str}")
    if output.strip():
        lines = output.strip().split("\n")
        for line in lines[:3]:
            print(f"  > {line}")
        if len(lines) > 3:
            print(f"  > ... ({len(lines)} lines total)")

    if engine == "rust" and stats:
        parts = []
        for key in [
            "callback_count", "ffi_crossings", "state_creations",
            "cache_hits", "cache_misses", "technique_filter_calls",
            "hook_sync_calls", "time_in_callbacks",
            "z3_ptr_cache_hits", "z3_ptr_cache_misses",
            "time_in_rust_run", "time_in_predicate_eval", "time_in_active_check",
            "time_in_explore",
        ]:
            if key in stats:
                val = stats[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.3f}")
                else:
                    parts.append(f"{key}={val}")
        if parts:
            print(f"  stats: {' '.join(parts)}")

        # Print Rust-side execution profiling breakdown
        rust_keys = [k for k in stats if k.startswith("rust_")]
        if rust_keys:
            print(f"  rust profiling:")
            # Time breakdowns (convert ns to seconds)
            for key in sorted(rust_keys):
                val = stats[key]
                if val == 0:
                    continue
                if key.endswith("_time_ns"):
                    label = key[5:-8]  # strip "rust_" and "_time_ns"
                    print(f"    {label}: {val/1e9:.3f}s")
                elif key.endswith("_count"):
                    label = key[5:-6]  # strip "rust_" and "_count"
                    print(f"    {label}: {val}")
                elif key == "rust_blocks_executed":
                    print(f"    blocks_executed: {val}")
                elif key == "rust_step_count":
                    print(f"    step_count: {val}")

        # Print Z3 solver stats
        z3_keys = [k for k in stats if k.startswith("z3_")]
        if z3_keys and any(stats.get(k, 0) > 0 for k in z3_keys):
            print(f"  z3 solver stats:")
            for key in sorted(z3_keys):
                val = stats[key]
                if val == 0:
                    continue
                if key.endswith("_time_ns"):
                    label = key[:-8]  # strip "_time_ns"
                    print(f"    {label}: {val/1e6:.1f}ms ({val/1e9:.3f}s)")
                else:
                    print(f"    {key}: {val}")

    if engine == "rust" and perf_report:
        print(f"  {perf_report}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single angr-example benchmark")
    parser.add_argument("example", nargs="?", help="Example name (e.g. fauxware, ais3_crackme)")
    parser.add_argument("--engine", choices=["rust", "python"], default="rust")
    parser.add_argument("--both", action="store_true", help="Run both engines")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT_MB,
                        help=f"Memory limit in MB (default: {DEFAULT_MEM_LIMIT_MB})")
    parser.add_argument("--list", action="store_true", help="List all cataloged examples")
    parser.add_argument("--strategy", choices=["bfs", "dfs"], default="bfs",
                        help="Exploration strategy (default: bfs)")
    parser.add_argument("--suite", choices=["fast", "medium", "all"],
                        help="Run a suite of examples (fast: <5s, medium: <30s, all: everything)")
    parser.add_argument("--diff-state", action="store_true",
                        help="Run both engines and diff per-step state snapshots (regs, "
                             "constraint count, satisfiability, history depth). Implies --both.")
    parser.add_argument("--diff-interval", type=int, default=1,
                        help="Snapshot every N step() calls when --diff-state is used (default: 1)")
    parser.add_argument("--diff-max-snapshots", type=int, default=200,
                        help="Cap snapshots per engine to bound memory/time (default: 200)")
    args = parser.parse_args()

    if args.list:
        print(f"{'Example':<35} {'Tier':<10} {'Rust OK':<10} {'Notes'}")
        print("-" * 90)
        for name, info in EXAMPLE_CATALOG.items():
            rust_ok = {True: "yes", False: "NO", None: "?"}[info["rust_ok"]]
            print(f"{name:<35} {info['tier']:<10} {rust_ok:<10} {info['notes']}")
        return

    if args.diff_state:
        sys.exit(_run_diff_state(args))

    if args.suite:
        tiers = {"fast": ["fast"], "medium": ["fast", "medium"], "all": ["fast", "medium", "slow"]}
        selected = [n for n, i in EXAMPLE_CATALOG.items() if i["tier"] in tiers[args.suite]]
        for name in selected:
            print(f"\n=== {name} ===")
            if args.both:
                run_example(name, "python", args.timeout, args.mem_limit, args.strategy)
                print()
                run_example(name, "rust", args.timeout, args.mem_limit, args.strategy)
            else:
                run_example(name, args.engine, args.timeout, args.mem_limit)
        return

    if not args.example:
        parser.print_help()
        return

    if args.both:
        print(f"=== {args.example} ===")
        run_example(args.example, "python", args.timeout, args.mem_limit, args.strategy)
        print()
        run_example(args.example, "rust", args.timeout, args.mem_limit, args.strategy)
    else:
        run_example(args.example, args.engine, args.timeout, args.mem_limit, args.strategy)


def _run_diff_state(args) -> int:
    """Run --diff-state for one or more examples and report divergences.

    Returns process exit code: 0 if every example matches, 1 otherwise.
    """
    _bench_dir = os.path.dirname(os.path.abspath(__file__))
    if _bench_dir not in sys.path:
        sys.path.insert(0, _bench_dir)
    from diff_state import compare_snapshots

    if args.suite:
        tiers = {"fast": ["fast"], "medium": ["fast", "medium"], "all": ["fast", "medium", "slow"]}
        names = [n for n, i in EXAMPLE_CATALOG.items() if i["tier"] in tiers[args.suite]]
    elif args.example:
        names = [args.example]
    else:
        print("ERROR: --diff-state requires either an example name or --suite")
        return 2

    overall_ok = True
    for name in names:
        print(f"\n=== diff-state: {name} ===")
        py_res = run_example(name, "python", args.timeout, args.mem_limit, args.strategy,
                             diff_state=True, diff_interval=args.diff_interval,
                             diff_max_snapshots=args.diff_max_snapshots)
        rs_res = run_example(name, "rust", args.timeout, args.mem_limit, args.strategy,
                             diff_state=True, diff_interval=args.diff_interval,
                             diff_max_snapshots=args.diff_max_snapshots)
        if not py_res or not py_res.get("ok"):
            print(f"DIFF SKIP {name}: python run failed")
            overall_ok = False
            continue
        if not rs_res or not rs_res.get("ok"):
            print(f"DIFF SKIP {name}: rust run failed")
            overall_ok = False
            continue

        py_snaps = py_res.get("snapshots") or []
        rs_snaps = rs_res.get("snapshots") or []
        if not py_snaps or not rs_snaps:
            print(f"DIFF SKIP {name}: missing snapshots (py={len(py_snaps)} rust={len(rs_snaps)})")
            overall_ok = False
            continue

        result = compare_snapshots(py_snaps, rs_snaps)
        if result["ok"]:
            print(f"DIFF OK {name}: {result['step_count'][0]} steps match")
        else:
            overall_ok = False
            print(f"DIFF FAIL {name}: first divergence at step {result['first_divergent_step']} "
                  f"(diverged_steps={result['diverged_steps']})")
            for line in result["summary"]:
                print(line)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    main()
