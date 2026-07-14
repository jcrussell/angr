"""concrete_strops: native-proc dispatch evidence for prefer_native_library_hooks (angr-gorvf.5).

Dynamically-linked concrete strcmp/strlen/memcpy+malloc loop. Every string op resolves
through the PLT into libc, so its hooks are non-main-object: they dispatch natively only
when prefer_native_library_hooks is on, and fall back to Python when it is off. Prints the
native_proc_calls / simprocedure_python_fallback_count split so the concrete win of the
flag can be re-measured on demand.

Self-caps RLIMIT_AS at 4GB — angr must never run in-process uncapped on this box.

    python tests/benchmarks/synthetic_examples/concrete_strops/measure.py
    ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS=1 python .../measure.py

Measured 2026-07-14 (iter122): flag off -> 50 Python fallbacks, 1 native, 0.66s;
flag on -> 50 native_proc_calls, 0 fallbacks, 0.14s. found=1 both ways.
"""

from __future__ import annotations

import resource

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, resource.RLIM_INFINITY))

import subprocess
import sys
import time
from pathlib import Path

import claripy

import angr

HERE = Path(__file__).parent
SRC = HERE / "concrete_strops.c"
BIN = HERE / "concrete_strops"

if not BIN.exists():
    subprocess.run(["gcc", "-O1", "-no-pie", "-o", str(BIN), str(SRC)], check=True)

proj = angr.Project(str(BIN), auto_load_libs=True)
state = proj.factory.entry_state(stdin=claripy.BVS("key", 8 * 8))

t0 = time.time()
mgr = angr.exploration.RustExplorationManager(proj, [state])
mgr.explore(find=lambda s: b"WIN" in s.posix.dumps(1), num_find=1)
wall = time.time() - t0

stats = mgr.stats
stats = stats() if callable(stats) else stats
found = len(mgr.stashes.get("found", []))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

print(f"found={found} wall={wall:.2f}s rss={rss}MB")
for key in sorted(stats):
    val = stats[key]
    if isinstance(val, int) and val and ("native_proc" in key or "fallback_count" in key):
        print(f"  {key} = {val}")
sys.stdout.flush()
