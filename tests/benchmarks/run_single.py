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
# Synthetic examples shipped in-repo (e.g. inline-ELF benchmarks for arches
# without binaries in angr-examples — MIPS, ARM64, etc.). Searched as a
# fallback when EXAMPLES_DIR/<name>/solve.py does not exist.
SYNTHETIC_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_examples")
DEFAULT_MEM_LIMIT_MB = 4096  # 4 GB — leaves 4 GB for parent + OS on 8GB machine


def _resolve_examples_dir(example_name, examples_dir):
    """Return the directory containing ``example_name/solve.py``.

    Falls back to SYNTHETIC_EXAMPLES_DIR when the primary path is missing
    so in-repo synthetic benchmarks work without configuring
    ANGR_EXAMPLES_DIR to also include this repo.
    """
    if os.path.exists(os.path.join(examples_dir, example_name, "solve.py")):
        return examples_dir
    if os.path.exists(os.path.join(SYNTHETIC_EXAMPLES_DIR, example_name, "solve.py")):
        return SYNTHETIC_EXAMPLES_DIR
    return examples_dir

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
    "CADET_00001":             {"tier": "medium",  "rust_ok": False, "notes": "CGC challenge: Py 22s (buffer-overflow + 2x easter-egg explore). Rust post-angr-rdgs (e16ca3bf3) has all 7 CGC syscalls native + buffer-overflow phase passes in ~0.1s; but easter-egg phase (sm.explore find=0x804833E) does not converge inside the default step budget, so solve.py raises IndexError at sm.found[0] in ~4.3s. Tracked python-only in baseline_timings.json; see angr-zgk6 and bd memory benchmark-cadet-cgc-partial-unblock for full unblock plan."},
    "ekopartyctf2015_rev100":  {"tier": "medium",  "rust_ok": False, "notes": "30 sim managers, run(n=4)/step(size=), Rust timeout"},
    "whitehat_crypto400":      {"tier": "medium",  "rust_ok": False, "notes": "Multi-stage explore+unstash, Rust list index error"},
    "ekopartyctf2016_sokohashv2":{"tier": "medium", "rust_ok": True,  "notes": "save_unconstrained, Windows PE, Py 8.6s Rust 18.6s (0.46x)"},
    "insomnihack_aeg":         {"tier": "slow",    "rust_ok": False, "notes": "Post-fkvt: Rust OOMs at ~75s under 4GB cap (state explosion during AEG exploration); Py >180s (posix.dumps over symbolic stdin). Not viable for regression gate. See angr-86c4.", "argv": ["./demo_bin"]},
    "0ctf_trace":              {"tier": "medium",  "rust_ok": None,  "notes": "MIPS blob, uses factory.successors() not simgr, Rust engine unused, Py 26.9s"},
    "simple_heap_overflow":    {"tier": "medium",  "rust_ok": False, "notes": "Two blockers: Python needs angr/binaries CI repo for libc 2.27 (system glibc 2.39 -> 'libc too new'); Rust raises NotImplementedError on SYMBOL_FILL_UNCONSTRAINED_REGISTERS (apre-root-cause)"},
    # === Slow/problematic examples ===
    "hackcon2016_angry-reverser":{"tier": "slow",  "rust_ok": True,  "notes": "LAZY_SOLVES, Z3 structure mismatch"},
    "asisctffinals2015_fake":  {"tier": "very_slow","rust_ok": False, "notes": "Z3 AST structure too complex for post-exploration solve"},
    "b01lersctf2020_little_engine":{"tier": "very_slow", "rust_ok": None, "notes": "~150s Python, untested Rust"},
    "tumctf2016_zwiebel":      {"tier": "very_slow","rust_ok": None, "notes": "Self-modifying code, ~2.5h"},
}


def _run_in_child(example_name, engine, examples_dir, mem_limit_mb, strategy="bfs",
                  diff_state=False, diff_interval=1, diff_max_snapshots=200,
                  use_shared_lineage_solver=False):
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

    examples_dir = _resolve_examples_dir(example_name, examples_dir)
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
            rust_mgr_instance = RustExplorationManager(
                factory_self.project, states,
                use_shared_lineage_solver=use_shared_lineage_solver,
            )
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

    # Optional argv injection for solve.py scripts that read sys.argv[1] as a
    # binary path (e.g. insomnihack_aeg). Set per-example via EXAMPLE_CATALOG.
    argv_override = EXAMPLE_CATALOG.get(example_name, {}).get("argv")

    try:
        os.chdir(example_dir)
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        spec = importlib.util.spec_from_file_location("__main__", solve_script)
        module = importlib.util.module_from_spec(spec)

        captured = BufferedStringIO()
        original_stdout = sys.stdout
        sys.stdout = captured

        original_argv = None
        if argv_override is not None:
            original_argv = sys.argv
            sys.argv = [solve_script, *argv_override]

        def _collect_rust_diagnostics():
            """Pull stats / perf_report / peak_rss off the Rust manager.

            Safe to call from both the success path and the exception paths —
            angr-qcsg lets a crashed-mid-run bench (e.g. CADET_00001 raising
            IndexError after the buffer-overflow phase has already emitted
            CGC syscall counters) still surface partial characterization
            data instead of dropping it on the floor.
            """
            rusage_local = resource.getrusage(resource.RUSAGE_SELF)
            peak_mb = round(rusage_local.ru_maxrss / 1024, 1)
            stats_local = None
            perf_local = None
            if engine == "rust" and rust_mgr_instance is not None:
                try:
                    stats_local = dict(rust_mgr_instance.stats)
                except Exception:
                    pass
                try:
                    perf_local = rust_mgr_instance.perf_report()
                except Exception:
                    pass
                # angr-zdho: walk every state's assumed-constraint RustBV
                # graph and report pointer-vs-structural sharing. Negligible
                # runtime cost (single pass at end-of-bench), surfaces what
                # construction-time hash-cons would dedupe.
                if stats_local is not None:
                    try:
                        sharing = rust_mgr_instance._rust_mgr.analyze_constraint_sharing()
                        for k, v in sharing.items():
                            stats_local[f"constraint_sharing_{k}"] = v
                    except Exception:
                        pass
            return stats_local, perf_local, peak_mb

        start = time.perf_counter()
        try:
            spec.loader.exec_module(module)
        except MemoryError:
            sys.stdout = original_stdout
            elapsed = time.perf_counter() - start
            stats, perf_report, peak_memory_mb = _collect_rust_diagnostics()
            return {
                "ok": False,
                "error": "MemoryError (hit memory limit)",
                "elapsed": elapsed,
                "stats": stats,
                "perf_report": perf_report,
                "peak_memory_mb": peak_memory_mb,
            }
        except Exception as e:
            sys.stdout = original_stdout
            elapsed = time.perf_counter() - start
            stats, perf_report, peak_memory_mb = _collect_rust_diagnostics()
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed": elapsed,
                "stats": stats,
                "perf_report": perf_report,
                "peak_memory_mb": peak_memory_mb,
            }
        finally:
            sys.stdout = original_stdout
            if original_argv is not None:
                sys.argv = original_argv

        elapsed = time.perf_counter() - start
        output = captured.getvalue()

        stats, perf_report, peak_memory_mb = _collect_rust_diagnostics()

        return {
            "ok": True,
            "elapsed": elapsed,
            "output": output,
            "stats": stats,
            "perf_report": perf_report,
            "peak_memory_mb": peak_memory_mb,
            "snapshots": snapshots if diff_state else None,
        }

    finally:
        if original_sm is not None:
            angr.factory.AngrObjectFactory.simulation_manager = original_sm
            angr.factory.AngrObjectFactory.simgr = original_simgr


# Counter categorization for --dump-counters. Tried in order; first match wins.
# Keys not matched by any category fall into "misc".
_DUMP_EXPLICIT_GROUPS = {
    "exploration": {
        "active", "steps", "found", "deadended_count", "avoided_count",
        "pruned_count", "errors", "find_addrs", "avoid_addrs",
        "drop_terminal_states", "hooks", "simprocedures",
        "state_roots_size", "block_cache_size",
    },
    "python-side": {
        "callback_count", "ffi_crossings", "state_creations",
        "cache_hits", "cache_misses", "technique_filter_calls",
        "hook_sync_calls", "hook_sync_skips", "time_in_callbacks",
        "time_in_rust_run", "time_in_predicate_eval",
        "time_in_active_check", "time_in_explore",
        "z3_ptr_cache_hits", "z3_ptr_cache_misses",
    },
    "fallbacks": {
        "simprocedure_python_fallback_count",
        "simprocedure_fallback_by_name",
        "syscall_python_fallback_count",
        "syscall_python_fallback_by_num",
        "native_proc_calls", "native_proc_fallbacks",
        "native_proc_symbolic_fallbacks",
        "native_proc_symbolic_fallbacks_by_name",
        "native_proc_not_implemented_fallbacks",
        "native_proc_not_implemented_fallbacks_by_name",
        "native_proc_other_fallbacks",
        "native_proc_other_fallbacks_by_name",
        "vex_fallback_count", "vex_fallback_unique_addrs",
        "dcas_unsupported_count",
    },
}
_DUMP_PREFIX_GROUPS = [
    ("rust execution", "rust_"),
    ("z3 solver", "z3_"),
    ("vex op dispatch", "vex_"),
    ("memory volume", "mem_"),
    ("concretization fanout", "concretize_"),
    ("ast construction", "bvop_"),
    ("zext collapse", "zext_"),
]
_DUMP_GROUP_ORDER = [
    "exploration", "python-side", "fallbacks",
    "rust execution", "z3 solver", "vex op dispatch", "memory volume",
    "concretization fanout", "ast construction", "zext collapse",
    "misc",
]


def _format_counter_value(key, val):
    """Render a single counter value for the dump-counters table."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, dict):
        if not val:
            return "{}"
        return "{" + ", ".join(f"{k}={v}" for k, v in sorted(val.items())) + "}"
    if isinstance(val, int):
        if key.endswith("_time_ns"):
            return f"{val / 1e6:.2f}ms"
        return str(val)
    if isinstance(val, float):
        # time_in_* are seconds (from rust_manager.stats)
        if key.startswith("time_in_"):
            return f"{val * 1000:.2f}ms"
        return f"{val:.4f}"
    return str(val)


def _group_counters(stats):
    """Return {group_label: [(key, value), ...]} keyed by _DUMP_GROUP_ORDER."""
    sections = {}
    seen = set()

    def add(grp, key):
        sections.setdefault(grp, []).append((key, stats[key]))
        seen.add(key)

    for grp, keys in _DUMP_EXPLICIT_GROUPS.items():
        for k in sorted(keys & set(stats)):
            add(grp, k)
    for grp, prefix in _DUMP_PREFIX_GROUPS:
        for k in sorted(stats):
            if k in seen:
                continue
            if k.startswith(prefix):
                add(grp, k)
    for k in sorted(stats):
        if k not in seen:
            add("misc", k)
    return sections


def _dump_counters_table(stats):
    """Print a categorized, aligned counter table to stdout.

    All counters from ``mgr.stats()`` are surfaced, including zeros — the goal
    is to make per-bench attribution explicit rather than cherry-picked.
    Time counters (``*_time_ns`` ints, ``time_in_*`` floats in seconds) are
    rendered as milliseconds for readability.
    """
    sections = _group_counters(stats)
    key_w = max(
        (len(k) for sec in sections.values() for k, _ in sec),
        default=20,
    )
    print("  === all counters ===")
    for grp in _DUMP_GROUP_ORDER:
        if grp not in sections:
            continue
        print(f"  [{grp}]")
        for key, val in sections[grp]:
            rendered = _format_counter_value(key, val)
            print(f"    {key:<{key_w}}  {rendered:>14}")


def _dump_counters_json(stats):
    """Emit the raw stats dict as JSON to stdout (machine-consumable)."""
    import json
    # dict values (simprocedure_fallback_by_name) are fine; default=str
    # handles any unexpected non-JSON-native value without crashing.
    print(json.dumps(stats, indent=2, default=str, sort_keys=True))


def run_example(example_name, engine, timeout=180, mem_limit_mb=DEFAULT_MEM_LIMIT_MB, strategy="bfs",
                diff_state=False, diff_interval=1, diff_max_snapshots=200,
                dump_counters=False, counters_json=False,
                use_shared_lineage_solver=False):
    """Run an example in an isolated subprocess and print results."""
    examples_dir = _resolve_examples_dir(example_name, EXAMPLES_DIR)
    solve_script = os.path.join(examples_dir, example_name, "solve.py")
    if not os.path.exists(solve_script):
        print(f"ERROR: solve.py not found: {solve_script}")
        return None

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _run_in_child,
            (example_name, engine, examples_dir, mem_limit_mb, strategy,
             diff_state, diff_interval, diff_max_snapshots,
             use_shared_lineage_solver),
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
        # angr-qcsg: surface partial-run counters when a bench crashes
        # mid-flight (e.g. CADET_00001's IndexError after buffer-overflow
        # phase). Only fires for rust runs with stats actually populated.
        fail_stats = result.get("stats")
        if engine == "rust" and fail_stats:
            if counters_json:
                _dump_counters_json(fail_stats)
            elif dump_counters:
                _dump_counters_table(fail_stats)
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

    if engine == "rust" and stats and not counters_json:
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

        # Print symbolic-memory ITE-depth stats (angr-0nme Phase 0).
        # Baseline metric for lazy-memory work (angr-czph / angr-qh5u): peak
        # ITE chain depth ever stored in a memory cell, and cumulative depth.
        mem_keys = [k for k in stats if k.startswith("mem_ite_")]
        if mem_keys and any(stats.get(k, 0) > 0 for k in mem_keys):
            print(f"  symbolic memory ite-depth:")
            for key in sorted(mem_keys):
                val = stats[key]
                if val == 0:
                    continue
                print(f"    {key}: {val}")

        # angr-2j5v: VEX dispatch / memory volume / concretization / AST
        # construction counters. Surface op-family distribution and memory
        # access volume so bench attribution doesn't need a flamegraph.
        vex_keys = [k for k in stats if k.startswith("vex_")]
        if vex_keys and any(stats.get(k, 0) > 0 for k in vex_keys):
            print(f"  vex op dispatch:")
            for key in sorted(vex_keys):
                val = stats[key]
                if val == 0:
                    continue
                print(f"    {key}: {val}")

        memvol_keys = [
            k for k in stats
            if k.startswith("mem_load") or k.startswith("mem_store")
            or k == "mem_lazy_page_fault_count"
        ]
        if memvol_keys and any(stats.get(k, 0) > 0 for k in memvol_keys):
            print(f"  memory volume:")
            for key in sorted(memvol_keys):
                val = stats[key]
                if val == 0:
                    continue
                print(f"    {key}: {val}")

        conc_keys = [k for k in stats if k.startswith("concretize_")]
        if conc_keys and any(stats.get(k, 0) > 0 for k in conc_keys):
            print(f"  concretization fanout:")
            for key in sorted(conc_keys):
                val = stats[key]
                if val == 0:
                    continue
                print(f"    {key}: {val}")

        bvop_keys = [k for k in stats if k.startswith("bvop_")]
        if bvop_keys and any(stats.get(k, 0) > 0 for k in bvop_keys):
            print(f"  ast emissions:")
            for key in sorted(bvop_keys):
                val = stats[key]
                if val == 0:
                    continue
                print(f"    {key}: {val}")

    if engine == "rust" and perf_report and not counters_json:
        print(f"  {perf_report}")

    if engine == "rust" and stats:
        if counters_json:
            _dump_counters_json(stats)
        elif dump_counters:
            _dump_counters_table(stats)

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
    parser.add_argument("--dump-counters", action="store_true",
                        help="At bench end, pretty-print every counter from "
                             "mgr.stats() grouped by category (callbacks, z3, "
                             "vex, memory, concretization, etc.). Rust engine only.")
    parser.add_argument("--counters-json", action="store_true",
                        help="At bench end, emit the full mgr.stats() dict as "
                             "JSON to stdout for machine consumption. Rust "
                             "engine only. Mutually exclusive with --dump-counters "
                             "(JSON wins when both are set).")
    parser.add_argument("--use-shared-lineage-solver", action="store_true",
                        help="Rust engine only. Construct RustExplorationManager "
                             "with use_shared_lineage_solver=True so the fork-time "
                             "SharedLineageSolver materialization gate engages "
                             "(angr-v5a5 slice 4c.3 step 1c canary measurement).")
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
                run_example(name, "rust", args.timeout, args.mem_limit, args.strategy,
                            dump_counters=args.dump_counters,
                            counters_json=args.counters_json,
                            use_shared_lineage_solver=args.use_shared_lineage_solver)
            else:
                run_example(name, args.engine, args.timeout, args.mem_limit, args.strategy,
                            dump_counters=args.dump_counters,
                            counters_json=args.counters_json,
                            use_shared_lineage_solver=args.use_shared_lineage_solver)
        return

    if not args.example:
        parser.print_help()
        return

    if args.both:
        print(f"=== {args.example} ===")
        run_example(args.example, "python", args.timeout, args.mem_limit, args.strategy)
        print()
        run_example(args.example, "rust", args.timeout, args.mem_limit, args.strategy,
                    dump_counters=args.dump_counters,
                    counters_json=args.counters_json,
                    use_shared_lineage_solver=args.use_shared_lineage_solver)
    else:
        run_example(args.example, args.engine, args.timeout, args.mem_limit, args.strategy,
                    dump_counters=args.dump_counters,
                    counters_json=args.counters_json,
                    use_shared_lineage_solver=args.use_shared_lineage_solver)


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
