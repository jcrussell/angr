#!/usr/bin/env python3
"""Bounded convergence probe for xmllint vanilla symbolic execution (angr-75mc path-b).

Measures whether vanilla symbolic exploration of xmllint_bin (auto_load_libs,
no sim procedures) stays tractable in a tiny step budget under the Rust engine,
or explodes / OOMs. Self-caps memory via RLIMIT_AS and bails on stash blowup.

NOT a bench — a one-shot data point to resolve the brief's "cannot verify
offline: convergence" unknown. Run under a nested capped scope.
"""

from __future__ import annotations

import os
import resource
import sys
import time

# 3.5GB address-space cap so we MemoryError before the 4G nested scope OOMs.
resource.setrlimit(resource.RLIMIT_AS, (3500 * 1024 * 1024, 3500 * 1024 * 1024))

import claripy

import angr
from angr import sim_options as so

XMLLINT = os.path.expanduser("~/repos/angr-examples/examples/xmllint/xmllint_bin")
MAX_STEPS = int(os.environ.get("PROBE_MAX_STEPS", "120"))
ACTIVE_CAP = int(os.environ.get("PROBE_ACTIVE_CAP", "60"))


def main():
    args = [XMLLINT, "--noout", "--nonet", "--recover", "--noent", "-"]
    t0 = time.time()
    proj = angr.Project(XMLLINT, auto_load_libs=True, use_sim_procedures=False)
    print(f"loaded in {time.time() - t0:.1f}s; entry={hex(proj.entry)} base={hex(proj.loader.main_object.mapped_base)}")

    state = proj.factory.entry_state(
        args=args,
        add_options={so.ZERO_FILL_UNCONSTRAINED_MEMORY, so.ZERO_FILL_UNCONSTRAINED_REGISTERS},
    )
    # 64 symbolic stdin bytes
    data = claripy.BVS("stdin", 64 * 8)
    state.posix.stdin.content = [(data, claripy.BVV(64, state.arch.bits))]

    engine = os.environ.get("PROBE_ENGINE", "rust")
    if engine == "python":
        return _python_probe(proj, state, t0)

    mgr = angr.exploration.RustExplorationManager(proj, [state])
    print(f"start stash={mgr.stash_counts()}")

    def pcs(stash):
        out = []
        for sid in mgr._rust_mgr.get_state_ids(stash):
            pc = mgr._rust_mgr.get_state_pc_by_id(sid)
            out.append(hex(pc) if pc is not None else f"sid{sid}?")
        return out

    for i in range(MAX_STEPS):
        before = pcs("active")
        mgr.step(n=1)
        sc = mgr.stash_counts()
        active = sc.get("active", 0)
        if i < 8 or i % 10 == 0 or active == 0 or active > ACTIVE_CAP:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            print(
                f"step {i:3d} from={before} -> active={pcs('active')} stash={sc} rss={rss}MB t={time.time() - t0:.1f}s"
            )
        if active == 0:
            print(f"DRAINED at step {i} (no active states) — deadended/converged")
            print(f"  deadended PCs={pcs('deadended')}")
            try:
                for addr, message, sid in mgr._rust_mgr.get_errors():
                    print(f"  ERRORED @ {hex(addr)} (sid={sid}): {message}")
            except Exception as e:
                print(f"  get_errors() failed: {e!r}")
            break
        if active > ACTIVE_CAP:
            print(f"EXPLOSION: active={active} > cap {ACTIVE_CAP} at step {i} — diverges cheaply")
            break
    else:
        print(f"BUDGET EXHAUSTED at {MAX_STEPS} steps, still active={mgr.stash_counts()}")
    print(f"final stash={mgr.stash_counts()} total={time.time() - t0:.1f}s")


def _python_probe(proj, state, t0):
    """Same workload under the vanilla Python engine (no Rust manager)."""
    simgr = proj.factory.simulation_manager(state)
    print(f"start stash={ {k: len(v) for k, v in simgr.stashes.items() if v} }")
    for i in range(MAX_STEPS):
        before = [hex(s.addr) for s in simgr.active]
        simgr.step(num_inst=None)
        active = len(simgr.active)
        if i < 8 or i % 10 == 0 or active == 0 or active > ACTIVE_CAP:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            cur = [hex(s.addr) for s in simgr.active]
            print(
                f"step {i:3d} from={before} -> active={cur} dead={len(simgr.deadended)} rss={rss}MB t={time.time() - t0:.1f}s"
            )
        if active == 0:
            print(f"DRAINED at step {i} (no active states)")
            print(f"  deadended PCs={[hex(s.addr) for s in simgr.deadended]}")
            if simgr.errored:
                for er in simgr.errored:
                    print(f"  ERRORED: {er}")
            break
        if active > ACTIVE_CAP:
            print(f"EXPLOSION: active={active} > cap {ACTIVE_CAP} at step {i}")
            break
    else:
        print(f"BUDGET EXHAUSTED at {MAX_STEPS} steps")
    print(f"final total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("MEMORYERROR: hit RLIMIT_AS — vanilla symex OOMs on xmllint")
        sys.exit(42)
