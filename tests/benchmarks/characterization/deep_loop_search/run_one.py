#!/usr/bin/env python3
"""Run a single (engine, strategy, N, max_steps) configuration of the
deep-loop search-strategy characterisation (angr-xel4 spike).

Compiles a parameterised deep-loop binary with N iteration cap, runs it
under either Rust or Python with the chosen strategy (BFS / DFS), and
prints a single line of metrics:

    N=<n> engine=<rust|python> strategy=<bfs|dfs> wall_s=<f> rss_kb=<i>
        steps=<i> max_depth=<i> active_at_end=<i> deadended=<i> error=<str|None>

Always runs in a subprocess (via sweep.py) with RLIMIT_AS to confine OOMs.
"""
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import tempfile
import time


HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "deep_loop_template.c")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def gen_c_source(n: int) -> str:
    with open(TEMPLATE) as f:
        src = f.read()
    return src.replace("__N__", str(n))


def compile_binary(n: int, out_dir: str) -> str:
    src = gen_c_source(n)
    c_path = os.path.join(out_dir, f"deep_loop_{n}.c")
    bin_path = os.path.join(out_dir, f"deep_loop_{n}")
    with open(c_path, "w") as f:
        f.write(src)
    subprocess.check_call(
        ["gcc", "-O0", "-no-pie", "-o", bin_path, c_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    return bin_path


def _max_depth_rust(mgr) -> int:
    """Max block-history length across active+deadended Rust states.

    Uses the lightweight proxy iterator: the full SimStates returned by
    ``mgr.active`` are exported with empty history; the
    ``RustHistoryProxy`` returned by ``mgr.active_proxies()`` reads the
    real per-state lineage tail from the Rust side without paying for a
    full state export.
    """
    best = 0
    proxies = (
        list(mgr.active_proxies())
        + list(mgr.deadended_proxies())
    )
    for p in proxies:
        try:
            n = len(p.history.bbl_addrs)
        except Exception:  # noqa: BLE001
            n = 0
        if n > best:
            best = n
    return best


def _max_depth_python(sm) -> int:
    best = 0
    for s in list(sm.active) + list(getattr(sm, "deadended", [])):
        try:
            n = len(list(s.history.bbl_addrs))
        except Exception:  # noqa: BLE001
            n = 0
        if n > best:
            best = n
    return best


def run_explore(bin_path: str, engine: str, strategy: str,
                max_steps: int, max_states: int):
    """Run exploration; return dict with depth/state metrics."""
    import angr  # noqa: E402
    import claripy  # noqa: E402

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    if main_sym is None:
        return {"error": "main symbol not found"}
    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    # Pre-stock stdin with `max_steps + 8` fully-symbolic bytes so each
    # read(0, &c, 1) in the loop returns symbolic content without an extra
    # fork. The loop itself caps total iterations at N.
    nbytes = max(max_steps + 8, 32)
    sym_bytes = claripy.BVS("stdin", nbytes * 8)
    state.posix.stdin.content.append((sym_bytes, claripy.BVV(nbytes, state.arch.bits)))

    try:
        if engine == "rust":
            from angr.exploration.rust_manager import RustExplorationManager
            mgr = RustExplorationManager(proj, [state])
            if strategy == "dfs":
                from angr.exploration_techniques import DFS
                mgr.use_technique(DFS())
            # BFS is the default FIFO queue; no technique needed.
            step = 0
            while mgr.active and step < max_steps:
                mgr.step()
                step += 1
                if len(mgr.active) + len(getattr(mgr, "deadended", [])) > max_states:
                    return {
                        "steps": step,
                        "max_depth": _max_depth_rust(mgr),
                        "active_at_end": len(mgr.active),
                        "deadended": len(getattr(mgr, "deadended", [])),
                        "error": f"state-explosion>{max_states}",
                    }
            return {
                "steps": step,
                "max_depth": _max_depth_rust(mgr),
                "active_at_end": len(mgr.active),
                "deadended": len(getattr(mgr, "deadended", [])),
                "error": None,
            }
        else:
            sm = proj.factory.simulation_manager(state, save_unconstrained=True)
            if strategy == "dfs":
                sm.use_technique(angr.exploration_techniques.DFS())
            step = 0
            while sm.active and step < max_steps:
                sm.step()
                step += 1
                if len(sm.active) + len(sm.deadended) > max_states:
                    return {
                        "steps": step,
                        "max_depth": _max_depth_python(sm),
                        "active_at_end": len(sm.active),
                        "deadended": len(sm.deadended),
                        "error": f"state-explosion>{max_states}",
                    }
            return {
                "steps": step,
                "max_depth": _max_depth_python(sm),
                "active_at_end": len(sm.active),
                "deadended": len(sm.deadended),
                "error": None,
            }
    except MemoryError:
        return {"error": "MemoryError"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["rust", "python"], required=True)
    p.add_argument("--strategy", choices=["bfs", "dfs"], required=True)
    p.add_argument("--n", type=int, required=True,
                   help="iteration cap baked into the binary")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-states", type=int, default=4096)
    p.add_argument("--mem-limit-mb", type=int, default=3072)
    p.add_argument("--build-dir", default=None,
                   help="Reuse-able binary cache (default: temp dir)")
    args = p.parse_args()

    if args.build_dir:
        os.makedirs(args.build_dir, exist_ok=True)
        build_dir = args.build_dir
    else:
        build_dir = tempfile.mkdtemp(prefix="deep_loop_")

    bin_path = os.path.join(build_dir, f"deep_loop_{args.n}")
    if not os.path.exists(bin_path):
        bin_path = compile_binary(args.n, build_dir)

    mem_bytes = args.mem_limit_mb * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
    except ValueError:
        pass

    t0 = time.monotonic()
    result = run_explore(bin_path, args.engine, args.strategy,
                         args.max_steps, args.max_states)
    wall = time.monotonic() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(
        f"N={args.n} engine={args.engine} strategy={args.strategy} "
        f"wall_s={wall:.3f} rss_kb={rss_kb} "
        f"steps={result.get('steps', -1)} "
        f"max_depth={result.get('max_depth', -1)} "
        f"active_at_end={result.get('active_at_end', -1)} "
        f"deadended={result.get('deadended', -1)} "
        f"error={result.get('error')}"
    )
    return 0 if result.get("error") is None else 1


if __name__ == "__main__":
    sys.exit(main())
