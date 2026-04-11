#!/usr/bin/env python3
"""Run a single angr-example with the Rust engine and print timing + callback stats.

Usage:
    python tests/benchmarks/run_single.py fauxware
    python tests/benchmarks/run_single.py ais3_crackme --engine=python
    python tests/benchmarks/run_single.py defcamp_r100 --both
"""
import argparse
import importlib.util
import io
import os
import sys
import time


EXAMPLES_DIR = os.path.expanduser("~/repos/angr-examples/examples")


class BufferedStringIO(io.StringIO):
    """StringIO with a buffer attribute for code that uses stdout.buffer."""

    def __init__(self):
        super().__init__()
        self._buffer = io.BytesIO()

    @property
    def buffer(self):
        return self._buffer

    def getvalue(self) -> str:
        text_output = super().getvalue()
        binary_output = self._buffer.getvalue()
        if binary_output:
            try:
                text_output += binary_output.decode("utf-8", errors="replace")
            except Exception:
                pass
        return text_output


def run_example(example_name: str, engine: str, timeout: int = 180):
    """Run an example and print results with stats."""
    import angr

    solve_script = os.path.join(EXAMPLES_DIR, example_name, "solve.py")
    if not os.path.exists(solve_script):
        print(f"ERROR: solve.py not found: {solve_script}")
        return

    example_dir = os.path.dirname(solve_script)
    original_dir = os.getcwd()
    original_path = sys.path[:]

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
        except Exception as e:
            sys.stdout = original_stdout
            import traceback
            print(f"FAIL {engine} {example_name}: {e}")
            traceback.print_exc()
            return
        finally:
            sys.stdout = original_stdout

        elapsed = time.perf_counter() - start
        output = captured.getvalue()

        # Print results
        print(f"OK {engine} {example_name} {elapsed:.2f}s")
        if output.strip():
            # Show first 3 lines of output
            lines = output.strip().split("\n")
            for line in lines[:3]:
                print(f"  > {line}")
            if len(lines) > 3:
                print(f"  > ... ({len(lines)} lines total)")

        # Print stats if Rust engine was used and stats are available
        if engine == "rust" and rust_mgr_instance is not None:
            try:
                stats = rust_mgr_instance.stats
                if stats:
                    parts = []
                    for key in [
                        "callback_count",
                        "ffi_crossings",
                        "state_creations",
                        "cache_hits",
                        "cache_misses",
                        "technique_filter_calls",
                        "hook_sync_calls",
                        "time_in_callbacks",
                    ]:
                        if key in stats:
                            val = stats[key]
                            if isinstance(val, float):
                                parts.append(f"{key}={val:.3f}")
                            else:
                                parts.append(f"{key}={val}")
                    if parts:
                        print(f"  stats: {' '.join(parts)}")
            except Exception:
                pass  # Stats not yet implemented

    finally:
        os.chdir(original_dir)
        sys.path[:] = original_path
        if original_sm is not None:
            angr.factory.AngrObjectFactory.simulation_manager = original_sm
            angr.factory.AngrObjectFactory.simgr = original_simgr


def main():
    parser = argparse.ArgumentParser(description="Run a single angr-example benchmark")
    parser.add_argument("example", help="Example name (e.g. fauxware, ais3_crackme)")
    parser.add_argument("--engine", choices=["rust", "python"], default="rust")
    parser.add_argument("--both", action="store_true", help="Run both engines")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.both:
        print(f"=== {args.example} ===")
        run_example(args.example, "python", args.timeout)
        print()
        run_example(args.example, "rust", args.timeout)
    else:
        run_example(args.example, args.engine, args.timeout)


if __name__ == "__main__":
    main()
