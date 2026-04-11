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


EXAMPLES_DIR = os.path.expanduser("~/repos/angr-examples/examples")
DEFAULT_MEM_LIMIT_MB = 4096  # 4 GB — leaves 4 GB for parent + OS on 8GB machine


def _run_in_child(example_name, engine, examples_dir, mem_limit_mb):
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

    import angr

    solve_script = os.path.join(examples_dir, example_name, "solve.py")
    if not os.path.exists(solve_script):
        return {"ok": False, "error": f"solve.py not found: {solve_script}"}

    example_dir = os.path.dirname(solve_script)

    # Track the RustExplorationManager instance for stats
    rust_mgr_instance = None
    original_sm = None
    original_simgr = None

    if engine == "rust":
        from angr.exploration import RustExplorationManager

        original_sm = angr.factory.AngrObjectFactory.simulation_manager
        original_simgr = angr.factory.AngrObjectFactory.simgr

        def patched_simulation_manager(factory_self, thing=None, **kwargs):
            nonlocal rust_mgr_instance
            if thing is None:
                states = [factory_self.entry_state()]
            elif isinstance(thing, (list, tuple)):
                states = list(thing)
            else:
                states = [thing]
            rust_mgr_instance = RustExplorationManager(factory_self.project, states)
            return rust_mgr_instance

        angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
        angr.factory.AngrObjectFactory.simgr = patched_simulation_manager

    class BufferedStringIO(io.StringIO):
        def __init__(self):
            super().__init__()
            self._buffer = io.BytesIO()

        @property
        def buffer(self):
            return self._buffer

        def getvalue(self):
            text = super().getvalue()
            binary = self._buffer.getvalue()
            if binary:
                try:
                    text += binary.decode("utf-8", errors="replace")
                except Exception:
                    pass
            return text

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
            return {"ok": False, "error": "MemoryError (hit memory limit)"}
        except Exception as e:
            sys.stdout = original_stdout
            return {"ok": False, "error": str(e)}
        finally:
            sys.stdout = original_stdout

        elapsed = time.perf_counter() - start
        output = captured.getvalue()

        # Collect stats if available
        stats = None
        if engine == "rust" and rust_mgr_instance is not None:
            try:
                stats = dict(rust_mgr_instance.stats)
            except Exception:
                pass

        return {
            "ok": True,
            "elapsed": elapsed,
            "output": output,
            "stats": stats,
        }

    finally:
        if original_sm is not None:
            angr.factory.AngrObjectFactory.simulation_manager = original_sm
            angr.factory.AngrObjectFactory.simgr = original_simgr


def run_example(example_name, engine, timeout=180, mem_limit_mb=DEFAULT_MEM_LIMIT_MB):
    """Run an example in an isolated subprocess and print results."""
    solve_script = os.path.join(EXAMPLES_DIR, example_name, "solve.py")
    if not os.path.exists(solve_script):
        print(f"ERROR: solve.py not found: {solve_script}")
        return

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _run_in_child, (example_name, engine, EXAMPLES_DIR, mem_limit_mb)
        )
        result = async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        print(f"TIMEOUT {engine} {example_name} after {timeout}s")
        return
    except Exception as e:
        pool.terminate()
        print(f"FAIL {engine} {example_name}: subprocess crashed: {e}")
        return
    finally:
        pool.terminate()
        pool.join()

    if not result.get("ok"):
        print(f"FAIL {engine} {example_name}: {result.get('error', 'unknown error')}")
        return

    elapsed = result["elapsed"]
    output = result.get("output", "")
    stats = result.get("stats")

    print(f"OK {engine} {example_name} {elapsed:.2f}s")
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
        ]:
            if key in stats:
                val = stats[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.3f}")
                else:
                    parts.append(f"{key}={val}")
        if parts:
            print(f"  stats: {' '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Run a single angr-example benchmark")
    parser.add_argument("example", help="Example name (e.g. fauxware, ais3_crackme)")
    parser.add_argument("--engine", choices=["rust", "python"], default="rust")
    parser.add_argument("--both", action="store_true", help="Run both engines")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT_MB,
                        help=f"Memory limit in MB (default: {DEFAULT_MEM_LIMIT_MB})")
    args = parser.parse_args()

    if args.both:
        print(f"=== {args.example} ===")
        run_example(args.example, "python", args.timeout, args.mem_limit)
        print()
        run_example(args.example, "rust", args.timeout, args.mem_limit)
    else:
        run_example(args.example, args.engine, args.timeout, args.mem_limit)


if __name__ == "__main__":
    main()
