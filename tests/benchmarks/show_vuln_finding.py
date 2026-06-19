#!/usr/bin/env python3
"""Real-bug / vuln-finding showcase demo for the Rust engine (angr-4n26m.7).

**Honest framing: GUIDED REACHABILITY to a known-suspicious sink.** This is
*not* an automatic crash oracle. The analyst pre-identifies the dangerous sink
(here, an unbounded ``strcpy`` whose source buffer is attacker-controlled) and
the engine solves for a concrete program input that reaches it while avoiding a
benign decoy path. The deliverable is the discovered input plus the path
constraints that the solver attached to reach the sink.

Modeled on ``angr-examples/strcpy_find/solve.py`` (Kyle Ossinger's
memory-corruption tutorial), re-implemented on top of
``RustExplorationManager`` so the search runs in the native engine. The find
condition is a Python predicate evaluated against a ``RustStateProxy`` at the
``strcpy`` PLT stub: it dereferences the source pointer (RSI) and confirms the
attacker-supplied marker is what gets copied, i.e. the analyst genuinely
controls the source of an unbounded copy -> a classic stack/heap overflow
primitive.

Why this is a fair Rust-engine showcase (not a Python-engine result relabeled):
the entire path exploration, branch forking, and constraint accumulation happen
in Rust; only the sink predicate and the final ``eval`` cross the FFI boundary,
exactly the integration pattern a real user would write.

Usage::

    # Run the demo, print the discovered input + reaching constraints.
    python tests/benchmarks/show_vuln_finding.py

    # Machine-readable; refresh the durable artifact consumed by the blog post.
    python tests/benchmarks/show_vuln_finding.py --json > \
        tests/benchmarks/vuln_finding_numbers.json

The binary is sourced from ``ANGR_EXAMPLES_DIR`` (defaults to
``~/repos/angr-examples/examples``), matching the other showcase scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
BINARY = os.path.join(EXAMPLES_DIR, "strcpy_find", "strcpy_test")

# The attacker-controlled marker we plant in argv[2]. The sink predicate
# confirms *this exact string* is what strcpy() copies, proving source control.
MARKER = b"HAHAHAHA"


def _func_addr(cfg, name, plt=None):
    """Resolve a function address from CFG symbols (plt=True for a PLT stub)."""
    found = [
        addr for addr, func in cfg.kb.functions.items() if func.name == name and (plt is None or func.is_plt == plt)
    ]
    if not found:
        raise RuntimeError(f"could not resolve address for {name!r} (plt={plt})")
    return found[0]


def find_vuln():
    """Run the guided-reachability search; return a result dict.

    Keys: ``input`` (concrete argv[1] bytes that reach the sink), ``sink_addr``,
    ``avoid_addr``, ``constraint_count`` (path constraints attached to the found
    state), ``steps`` is not tracked here, ``elapsed`` (wall-clock seconds).
    """
    import claripy

    import angr
    from angr.exploration import RustExplorationManager

    project = angr.Project(BINARY, load_options={"auto_load_libs": False})
    cfg = project.analyses.CFG(fail_fast=True)

    sink_addr = _func_addr(cfg, "strcpy", plt=True)  # the unbounded-copy sink
    avoid_addr = _func_addr(cfg, "func3")  # benign decoy path the solver must dodge

    # Symbolic argv[1] — the input we solve for. argv[2] is the concrete marker
    # whose bytes we expect to land in the strcpy source buffer.
    sym_arg = claripy.BVS("sym_arg", 8 * 40)
    argv = [project.filename, sym_arg, MARKER.decode()]
    state = project.factory.entry_state(args=argv)

    # The reaching path constraints live in the Rust solver, not the exported
    # Python SimState (constraints are intentionally not pre-pinned on export;
    # see bd memory constraint-export-no-pre-pin). Capture them at the moment
    # the sink predicate fires, straight off the matching proxy.
    captured = {}

    def reaches_controlled_strcpy(proxy):
        """Sink predicate: at the strcpy stub, is the source buffer ours?"""
        if proxy.ip.args[0] != sink_addr:
            return False
        # By disassembly, strcpy's source pointer lives in RSI. Deref it and
        # read len(MARKER) bytes; if our marker is what's being copied, the
        # analyst controls the source of an unbounded copy.
        src = proxy.memory.load(proxy.regs.rsi, len(MARKER))
        copied = proxy.solver.eval(src, cast_to=bytes)
        if MARKER not in copied:
            return False
        if not captured:  # record the first matching path's reaching constraints
            try:
                cons = proxy.solver.constraints
                captured["constraint_count"] = len(cons)
                captured["constraint_sample"] = [str(c)[:120] for c in cons[:5]]
            except Exception:
                captured["constraint_count"] = None
                captured["constraint_sample"] = []
        return True

    mgr = RustExplorationManager(project, [state])

    t0 = time.time()
    mgr.explore(find=reaches_controlled_strcpy, avoid=(avoid_addr,))
    elapsed = time.time() - t0

    found = mgr.found
    if not found:
        return {
            "found": False,
            "sink_addr": sink_addr,
            "avoid_addr": avoid_addr,
            "elapsed": elapsed,
        }

    fstate = found[0]
    raw = fstate.solver.eval(sym_arg, cast_to=bytes)
    # Trim at the first NUL — the rest is unconstrained padding.
    nul = raw.find(b"\x00")
    discovered = raw[:nul] if nul != -1 else raw

    return {
        "found": True,
        "input": discovered.decode("latin-1"),
        "input_hex": discovered.hex(),
        "sink_addr": sink_addr,
        "avoid_addr": avoid_addr,
        "constraint_count": captured.get("constraint_count"),
        "constraint_sample": captured.get("constraint_sample", []),
        "elapsed": round(elapsed, 2),
    }


def _print_report(res):
    print("=" * 70)
    print("Rust-engine guided reachability — unbounded strcpy sink")
    print("=" * 70)
    print(f"  binary       : {BINARY}")
    print(f"  sink  (find) : strcpy@plt   0x{res['sink_addr']:x}")
    print(f"  decoy (avoid): func3        0x{res['avoid_addr']:x}")
    print(f"  search time  : {res['elapsed']:.2f}s")
    print("-" * 70)
    if not res["found"]:
        print("  RESULT: no path reached the controlled-source sink.")
        print("  (Tighten the predicate or widen the symbolic input.)")
        return
    print("  RESULT: FOUND a controlled-source path to the sink.")
    print(f"  attacker input (argv[1]) : {res['input']!r}")
    print(f"  input bytes (hex)        : {res['input_hex']}")
    print(f"  reaching path constraints (Rust solver) : {res['constraint_count']}")
    for c in res.get("constraint_sample", []):
        print(f"      - {c}")
    print("-" * 70)
    print("  Feed that input as argv[1] and the program strcpy()s the")
    print(f"  attacker-controlled argv[2] ({MARKER!r}) into an undersized")
    print("  buffer — a reachable memory-corruption primitive, not a guess.")
    print("=" * 70)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if not os.path.exists(BINARY):
        sys.stderr.write(
            f"error: target binary not found at {BINARY}\n"
            "set ANGR_EXAMPLES_DIR to your angr-examples/examples checkout.\n"
        )
        return 2

    res = find_vuln()

    if args.json:
        json.dump(res, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(res)

    return 0 if res.get("found") else 1


if __name__ == "__main__":
    sys.exit(main())
