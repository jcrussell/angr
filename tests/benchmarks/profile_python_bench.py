#!/usr/bin/env python3
"""Profile one angr-examples bench under Python or Rust engine.

Used by angr-trsg (characterize .4 flame-graph comparison). The criterion
bench wrapper `profile_rust_bench.sh` profiles the micro-bench groups in
`native/angr/benches/vex_engine.rs`, not actual bench scripts. This driver
fills the gap: it loads a bench's solve.py in-process so cProfile or perf
record can attribute time end-to-end (Python frames + Rust frames via the
PyO3 boundary).

Runs the bench in-process — no multiprocessing — so attached profilers
(cProfile, perf) see every sample. RLIMIT_AS is still installed at 4 GB
to mirror run_single.py's guard.

Usage:
    # cProfile, text top-N + pstats
    python tests/benchmarks/profile_python_bench.py sym-write \\
        --engine python --cprofile --out /tmp/profile

    # Plain run (no cProfile), suitable for `perf record -- ...`
    perf record -F 99 -g -o /tmp/perf.data -- \\
        python tests/benchmarks/profile_python_bench.py mma_howtouse \\
            --engine rust

Output (with --cprofile and --out PREFIX):
    PREFIX.pstats          binary pstats blob (load with pstats.Stats)
    PREFIX.txt             top-50 cumulative + tottime tables
    PREFIX.folded          collapsed-stack format (input for flamegraph.pl)
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import os
import pstats
import resource
import sys
import time

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
SYNTHETIC_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_examples")
DEFAULT_MEM_LIMIT_MB = 4096


def _resolve_examples_dir(example_name, examples_dir):
    if os.path.exists(os.path.join(examples_dir, example_name, "solve.py")):
        return examples_dir
    if os.path.exists(os.path.join(SYNTHETIC_EXAMPLES_DIR, example_name, "solve.py")):
        return SYNTHETIC_EXAMPLES_DIR
    return examples_dir


def _install_rust_engine_patch():
    """Re-route SimulationManager construction through RustExplorationManager."""
    import angr
    from angr.exploration import RustExplorationManager

    original_sm = angr.factory.AngrObjectFactory.simulation_manager
    original_simgr = angr.factory.AngrObjectFactory.simgr

    def patched_simulation_manager(factory_self, thing=None, **kwargs):
        import traceback

        caller_frames = traceback.extract_stack()
        for frame in caller_frames[:-1]:
            if "/angr/analyses/" in frame.filename or "/angr/exploration_techniques/" in frame.filename:
                return original_sm(factory_self, thing, **kwargs)
        if thing is None:
            states = [factory_self.entry_state()]
        elif isinstance(thing, (list, tuple)):
            states = list(thing)
        else:
            states = [thing]
        return RustExplorationManager(factory_self.project, states)

    angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
    angr.factory.AngrObjectFactory.simgr = patched_simulation_manager


def _run_solve(example_name, engine):
    examples_dir = _resolve_examples_dir(example_name, EXAMPLES_DIR)
    solve_script = os.path.join(examples_dir, example_name, "solve.py")
    if not os.path.exists(solve_script):
        raise FileNotFoundError(f"solve.py not found: {solve_script}")
    example_dir = os.path.dirname(solve_script)

    os.chdir(example_dir)
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    if engine == "rust":
        _install_rust_engine_patch()

    spec = importlib.util.spec_from_file_location("__main__", solve_script)
    module = importlib.util.module_from_spec(spec)

    captured = io.StringIO()
    saved_stdout = sys.stdout
    sys.stdout = captured
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout = saved_stdout

    return captured.getvalue()


def _write_text_report(stats_obj, out_path, top_n=50):
    """Write cumulative + tottime top-N tables to out_path."""
    with open(out_path, "w") as f:
        buf = io.StringIO()
        s = pstats.Stats(stats_obj, stream=buf)
        s.sort_stats(pstats.SortKey.CUMULATIVE)
        s.print_stats(top_n)
        f.write(f"# cProfile cumulative time, top {top_n}\n")
        f.write(buf.getvalue())
        f.write("\n\n")

        buf = io.StringIO()
        s = pstats.Stats(stats_obj, stream=buf)
        s.sort_stats(pstats.SortKey.TIME)
        s.print_stats(top_n)
        f.write(f"# cProfile tottime (self), top {top_n}\n")
        f.write(buf.getvalue())


def _write_folded(stats_obj, out_path):
    """Write collapsed-stack folded format for flamegraph.pl.

    cProfile records caller -> callee edges with cumulative time; flame
    graphs want full stack frames. The standard "fake but useful" emission
    is one line per (caller, callee) edge weighted by the callee's tottime
    in microseconds: 'caller;callee N'. Renders as a two-level flame which
    is a useful summary but NOT a true call-stack flame — this is a known
    limitation of cProfile vs sampling profilers.
    """
    s = pstats.Stats(stats_obj)
    with open(out_path, "w") as f:
        for func, (cc, nc, tt, ct, callers) in s.stats.items():
            file, line, name = func
            callee_label = f"{os.path.basename(file)}:{line}:{name}"
            # microseconds rounded; flamegraph.pl ignores <1 samples
            us = max(1, int(tt * 1e6))
            if not callers:
                f.write(f"{callee_label} {us}\n")
                continue
            for caller_func, _ in callers.items():
                cfile, cline, cname = caller_func
                caller_label = f"{os.path.basename(cfile)}:{cline}:{cname}"
                f.write(f"{caller_label};{callee_label} {us}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("example")
    ap.add_argument("--engine", choices=["python", "rust"], default="rust")
    ap.add_argument("--cprofile", action="store_true", help="Run under cProfile, dump pstats + text report")
    ap.add_argument(
        "--out", default=None, help="Output prefix for pstats / text / folded files (required with --cprofile)"
    )
    ap.add_argument(
        "--mem-limit", type=int, default=DEFAULT_MEM_LIMIT_MB, help=f"RLIMIT_AS in MB (default {DEFAULT_MEM_LIMIT_MB})"
    )
    args = ap.parse_args()

    mem_bytes = args.mem_limit * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
    except ValueError:
        pass

    if args.cprofile and not args.out:
        ap.error("--cprofile requires --out PREFIX")

    # Ensure the repo root is on sys.path so the in-process import of
    # ``angr`` resolves through the editable install. Without this the
    # subsequent ``import angr`` inside solve.py fails depending on how
    # python was invoked (``perf record -- python ...`` strips PYTHONPATH
    # in some shells).
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    start = time.perf_counter()
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            _run_solve(args.example, args.engine)
        finally:
            profiler.disable()
        elapsed = time.perf_counter() - start

        pstats_path = args.out + ".pstats"
        text_path = args.out + ".txt"
        folded_path = args.out + ".folded"
        profiler.dump_stats(pstats_path)
        _write_text_report(profiler, text_path)
        _write_folded(profiler, folded_path)
        print(f"OK {args.engine} {args.example} {elapsed:.2f}s")
        print(f"  pstats: {pstats_path}")
        print(f"  text:   {text_path}")
        print(f"  folded: {folded_path}")
    else:
        _run_solve(args.example, args.engine)
        elapsed = time.perf_counter() - start
        print(f"OK {args.engine} {args.example} {elapsed:.2f}s")


if __name__ == "__main__":
    main()
