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
# PROBE_TRACE_ALL=1 prints every step's PC transition (for localizing a
# cross-engine control-flow divergence; see bug angr-aca6y). Default 0 keeps
# the sparse i<8/i%10 sampling used by the convergence characterization.
TRACE_ALL = os.environ.get("PROBE_TRACE_ALL", "0") == "1"


def main():
    args = [XMLLINT, "--noout", "--nonet", "--recover", "--noent", "-"]
    t0 = time.time()
    # PROBE_SIM_PROCS=1 swaps in SimProcedures for libc (the viable path-b
    # config — the no-sim-procs path dies in uninitialized-TLS glibc startup;
    # see memory xmllint-path-b-glibc-init-wall). Default 0 preserves the
    # iter30/31 vanilla-symex characterization.
    use_sim_procs = os.environ.get("PROBE_SIM_PROCS", "0") == "1"
    proj = angr.Project(XMLLINT, auto_load_libs=True, use_sim_procedures=use_sim_procs)
    print(
        f"loaded in {time.time() - t0:.1f}s; entry={hex(proj.entry)} "
        f"base={hex(proj.loader.main_object.mapped_base)} sim_procs={use_sim_procs}"
    )

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

    # PROBE_DUMP_ARGV=1: dump argv strings + the "-maxmem"/"--maxmem" option
    # strings as Rust's memory sees them (localizing bug angr-aca6y — the
    # option-parsing strcmp that diverges). r12 holds argv at main entry.
    if os.environ.get("PROBE_DUMP_ARGV", "0") == "1":
        sids = mgr._rust_mgr.get_state_ids("active")
        if sids:
            sid = sids[0]

            def rd_str(addr, n=24):
                b = mgr._rust_mgr.get_state_memory(sid, addr, n) or b""
                z = b.find(0)
                return b[:z] if z >= 0 else b

            r12 = mgr._rust_mgr.get_state_register(sid, "r12")
            base = proj.loader.main_object.mapped_base
            print(f"[dump] r12(argv)={hex(r12) if r12 else None} base={hex(base)}")
            if r12:
                for k in range(7):
                    ptr_b = mgr._rust_mgr.get_state_memory(sid, r12 + 8 * k, 8)
                    ptr = int.from_bytes(ptr_b, "little") if ptr_b else None
                    s = rd_str(ptr) if ptr else b"<sym>"
                    print(f"[dump] argv[{k}]={hex(ptr) if ptr else None} -> {s!r}")
            print(f"[dump] opt@0xf62c(-maxmem) = {rd_str(base + 0xF62C)!r}")
            print(f"[dump] opt@0xf62b(--maxmem)= {rd_str(base + 0xF62B)!r}")
            cle_b = proj.loader.memory.load(base + 0xF62C, 24)
            print(f"[dump] cle  hex @0xf62c(24) = {cle_b.hex()}")
            for sz in (1, 2, 4, 8, 12, 16, 24):
                rb = mgr._rust_mgr.get_state_memory(sid, base + 0xF62C, sz) or b""
                ok = "OK" if rb == cle_b[:sz] else "DIFF"
                print(f"[dump] rust @0xf62c({sz:2d}) = {rb.hex()} [{ok}]")
            # byte-by-byte via 1-byte reads (what strcmp uses)
            bb = b"".join((mgr._rust_mgr.get_state_memory(sid, base + 0xF62C + j, 1) or b"\xff") for j in range(8))
            print(f"[dump] rust bytewise @0xf62c = {bb.hex()}")

    dump_regs = os.environ.get("PROBE_DUMP_REGS", "0") == "1"

    def reg_dump(tag):
        # Dump the option-loop control regs (argc=ebp, argv=r12, idx=r15,
        # cur-arg=r14, strcmp-ret=eax) plus the return addr on the stack to
        # localize bug angr-aca6y's divergence at 0x406896.
        sids = mgr._rust_mgr.get_state_ids("active")
        if not sids:
            return
        sid = sids[0]
        regs = {}
        for r in ("rbp", "r12", "r15", "r14", "rax", "rsp", "rip"):
            try:
                regs[r] = mgr._rust_mgr.get_state_register(sid, r)
            except Exception as e:
                regs[r] = f"err:{e!r}"
        rsp = regs.get("rsp")
        retb = mgr._rust_mgr.get_state_memory(sid, rsp, 8) if isinstance(rsp, int) else None
        ret = int.from_bytes(retb, "little") if retb else None

        def rd(addr, n):
            b = mgr._rust_mgr.get_state_memory(sid, addr, n) or b""
            return int.from_bytes(b, "little") if b else None

        # libxml2 init-flag globals that drive the divergent branches (base
        # 0x500000): xmlGlobalInit flag @0x5e0cc8 (test@0x561aad),
        # init-state @0x5e0008 (cmp 0xffffffff@0x561ace), ptr @0x5e11f0.
        g_cc8 = rd(0x6E0CC8, 4)
        g_008 = rd(0x6E0008, 4)
        g_1f0 = rd(0x6E11F0, 8)
        # GOT slot for pthread_mutex_init@plt (jmp *0x6dfe58); Rust jumps to a
        # libxml2 fn (0x5ee070) instead of real libc — bug angr-aca6y.
        got_pmi = rd(0x6DFE58, 8)
        print(f"[got pthread_mutex_init@0x6dfe58]={hex(got_pmi) if got_pmi is not None else None}")
        if isinstance(rsp, int):
            words = [rd(rsp + 8 * k, 8) for k in range(8)]
            print(
                f"[stack {tag}] rsp={hex(rsp)} "
                + " ".join(f"+{8 * k:#x}={hex(w) if w is not None else None}" for k, w in enumerate(words))
            )
        print(
            f"[regs {tag}] "
            + " ".join(f"{k}={hex(v) if isinstance(v, int) else v}" for k, v in regs.items())
            + f" [rsp]={hex(ret) if ret else None}"
            + f" g@0x6e0cc8={hex(g_cc8) if g_cc8 is not None else None}"
            + f" g@0x6e0008={hex(g_008) if g_008 is not None else None}"
            + f" g@0x6e11f0={hex(g_1f0) if g_1f0 is not None else None}"
        )

    for i in range(MAX_STEPS):
        before = pcs("active")
        if dump_regs and before in (
            ["0x406896"],
            ["0x561aa6"],
            ["0x561efa"],
            ["0x64db95"],
            ["0x64db78"],
            ["0x5ee070"],
            ["0x64db12"],
        ):
            reg_dump(f"pre-step{i}@{before[0]}")
        mgr.step(n=1)
        sc = mgr.stash_counts()
        active = sc.get("active", 0)
        if TRACE_ALL or i < 8 or i % 10 == 0 or active == 0 or active > ACTIVE_CAP:
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
        if TRACE_ALL or i < 8 or i % 10 == 0 or active == 0 or active > ACTIVE_CAP:
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
