#!/usr/bin/env python3
"""Run a single (engine, N) configuration of the fork-tree characterisation.

Compiles a binary with N independent symbolic branches, runs it through
either the Rust or Python engine, and prints a single line of metrics:

    N=<n> engine=<rust|python> wall_s=<f> rss_kb=<i> deadended=<i> error=<str|None>

Runs in a subprocess (forked from sweep.py) with RLIMIT_AS to avoid OOM.
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
TEMPLATE = os.path.join(HERE, "fork_tree_template.c")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def gen_c_source(n: int) -> str:
    with open(TEMPLATE) as f:
        src = f.read()
    branches = []
    for i in range(n):
        branches.append(f"    if (b[{i}] != 0) s += {1 << i};")
    return src.replace("    /* __BRANCHES__ */", "\n".join(branches))


def compile_binary(n: int, out_dir: str) -> str:
    src = gen_c_source(n)
    c_path = os.path.join(out_dir, f"fork_tree_{n}.c")
    bin_path = os.path.join(out_dir, f"fork_tree_{n}")
    with open(c_path, "w") as f:
        f.write(src)
    subprocess.check_call(
        ["gcc", "-O0", "-no-pie", "-o", bin_path, c_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    return bin_path


def run_explore(bin_path: str, engine: str, max_states: int = 65536) -> tuple[int, str | None]:
    """Run exploration; return (deadended_count, error_str_or_none)."""
    import angr  # noqa: E402
    import claripy  # noqa: E402

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    if main_sym is None:
        return 0, "main symbol not found"
    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    # Pre-stock stdin with 16 fully-symbolic bytes so the SimProc-modelled
    # `read(0, b, 16)` call returns symbolic content without an extra fork.
    sym_bytes = claripy.BVS("stdin", 16 * 8)
    state.posix.stdin.content.append((sym_bytes, claripy.BVV(16, state.arch.bits)))

    try:
        if engine == "rust":
            from angr.exploration.rust_manager import RustExplorationManager
            mgr = RustExplorationManager(proj, [state])
            step = 0
            while mgr.active and step < 4096:
                mgr.step()
                step += 1
                if len(mgr.active) + len(mgr.deadended) > max_states:
                    return 0, f"state-explosion>{max_states}"
            terminal = len(getattr(mgr, "deadended", [])) + \
                       len(getattr(mgr, "unconstrained", []))
            return terminal, None
        else:
            sm = proj.factory.simulation_manager(state, save_unconstrained=True)
            step = 0
            while sm.active and step < 4096:
                sm.step()
                step += 1
                if len(sm.active) + len(sm.deadended) > max_states:
                    return 0, f"state-explosion>{max_states}"
            terminal = len(sm.deadended) + len(sm.unconstrained)
            return terminal, None
    except MemoryError:
        return 0, "MemoryError"
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["rust", "python"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--mem-limit-mb", type=int, default=4096)
    p.add_argument("--build-dir", default=None,
                   help="Reuse-able binary cache (default: temp dir)")
    args = p.parse_args()

    if args.build_dir:
        os.makedirs(args.build_dir, exist_ok=True)
        build_dir = args.build_dir
    else:
        build_dir = tempfile.mkdtemp(prefix="fork_tree_")

    bin_path = os.path.join(build_dir, f"fork_tree_{args.n}")
    if not os.path.exists(bin_path):
        bin_path = compile_binary(args.n, build_dir)

    mem_bytes = args.mem_limit_mb * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
    except ValueError:
        pass

    t0 = time.monotonic()
    terminal, err = run_explore(bin_path, args.engine)
    wall = time.monotonic() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(
        f"N={args.n} engine={args.engine} wall_s={wall:.3f} "
        f"rss_kb={rss_kb} terminal={terminal} error={err}"
    )
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
