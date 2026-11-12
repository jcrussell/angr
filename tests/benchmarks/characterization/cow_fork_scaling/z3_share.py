#!/usr/bin/env python
"""Per-live-state memory on the fork tree, and the share Z3 allocated (angr-az6ru.1).

Runs the ``2^N``-leaf fork tree to completion on the Rust engine without
building any Python ``SimState``, then reports the RSS growth per live state
next to the growth of ``Z3_get_estimated_alloc_size``. The two together say
how much of a parked state is its materialized ``z3::Solver`` and how much is
everything else (copy-on-write deltas, history, call stack).

    python tests/benchmarks/characterization/cow_fork_scaling/z3_share.py --n 10

``run_one.py`` in this directory compiles ``fork_tree_<N>`` on demand; pass
the same ``--build-dir`` to both.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
sys.path.insert(0, HERE)


def rss_bytes() -> int:
    with open("/proc/self/statm", encoding="ascii") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def z3_alloc_bytes() -> int:
    """Z3's own allocation estimate, read from the libz3 the engine loaded."""
    with open("/proc/self/maps", encoding="ascii") as f:
        for line in f:
            if "libz3" in line:
                lib = ctypes.CDLL(line.split()[-1])
                lib.Z3_get_estimated_alloc_size.restype = ctypes.c_uint64
                return lib.Z3_get_estimated_alloc_size()
    raise RuntimeError("libz3 is not mapped: the engine has not been imported")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, required=True, help="branch count; the tree has 2^N leaves")
    p.add_argument("--build-dir", default="/tmp/fork_tree_build")
    args = p.parse_args()

    import angr
    from angr import claripy
    from angr.exploration.rust_manager import RustExplorationManager

    bin_path = os.path.join(args.build_dir, f"fork_tree_{args.n}")
    if not os.path.exists(bin_path):
        from run_one import compile_binary

        os.makedirs(args.build_dir, exist_ok=True)
        bin_path = compile_binary(args.n, args.build_dir)

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"
    state = proj.factory.blank_state(addr=main_sym.rebased_addr)
    state.posix.stdin.content.append((claripy.BVS("stdin", 16 * 8), claripy.BVV(16, state.arch.bits)))
    mgr = RustExplorationManager(proj, [state])

    gc.collect()
    rss0, z30 = rss_bytes(), z3_alloc_bytes()
    # Read the active count from stats, not the stash, so no Python state is built.
    while mgr.stats.get("active"):
        mgr.step()
    gc.collect()
    rss1, z31 = rss_bytes(), z3_alloc_bytes()

    stats = mgr.stats
    live = stats.get("state_roots_size") or 0
    if not live:
        print(f"N={args.n} no live states at the end of the run")
        return 1
    rss_per = (rss1 - rss0) / live
    z3_per = (z31 - z30) / live
    print(
        f"N={args.n} live_states={live} solvers_built={stats.get('z3_materialize_count')} "
        f"rss_per_state_kb={rss_per / 1e3:.0f} z3_per_state_kb={z3_per / 1e3:.0f} "
        f"z3_share={z3_per / rss_per:.0%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
