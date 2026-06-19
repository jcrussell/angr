#!/usr/bin/env python3
"""Checkpoint/resume determinism-boundary characterization (angr-4n26m.4).

Before the checkpoint/resume *demo* (angr-4n26m.8) can make an honest claim,
we have to empirically classify the chosen resume target on two axes:

1. **constraint-UNIQUE vs search-only** — is the reaching input the *only*
   solution to the path constraints (so a resume that re-solves recovers the
   *same concrete value*), or is it one of many (so a resume only reproduces
   the *search*, i.e. it reaches the same program point but may pick a
   different satisfying input)?

2. **deterministic-mode residual gap** — does ``deterministic=True`` actually
   stabilize the concrete bytes the solver picks across fresh runs and across
   a snapshot round-trip, and if not, what gap remains?

This is the grounding probe for the honesty framing in the showcase epic:
checkpoint/resume is sold as "resume the SEARCH, not perfectly restore the
run" (rust_engine.rst Known-limitation: model equality is NOT guaranteed
across a restore; ``deterministic=True`` narrows but does not close the gap).

Target: ``fauxware`` (find = 0x4006ed, the auth-success ``puts``). fauxware is
the canonical round-trip fixture (test_misc.py round-trip test) and is in-loop
memory-safe.

Method:

* **Fresh-run stability** — explore from entry to ``find`` N times in fresh
  managers (no snapshot), capture ``found[0].posix.dumps(0)`` each time. If the
  bytes are identical across all N runs the target is *value-stable*; if they
  vary the target is *search-only*. Run this for ``deterministic`` False and
  True to isolate the mode's effect.

* **Round-trip parity** — step a manager partway, ``dump_snapshot`` to disk,
  ``load_from_disk`` into a fresh manager, resume to ``find``, and compare the
  found stdin captured by the *uninterrupted* run vs the *resumed* run.

* **Uniqueness probe (best-effort)** — on the found state, ask the solver for up
  to two distinct satisfying stdin prefixes (``eval_upto``). >1 solution =
  constraint-NOT-unique. (Best-effort: the exported SimState carries no
  Python-side constraints by design — constraint-export-no-pre-pin — so this
  leans on the Rust solver fallback and is reported as advisory.)

Usage::

    python tests/benchmarks/show_resume_boundary.py            # human report
    python tests/benchmarks/show_resume_boundary.py --json \
        > tests/benchmarks/resume_boundary_numbers.json        # durable artifact
    python tests/benchmarks/show_resume_boundary.py -n 3       # fewer samples (faster)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
BINARY = os.path.join(EXAMPLES_DIR, "fauxware", "fauxware")
FIND_ADDR = 0x4006ED
SNAPSHOT_STEPS = 6  # step this many blocks before snapshotting the in-flight search


def _hexdigest(b: bytes) -> str:
    """Short, JSON-friendly rendering of a stdin capture."""
    return b.hex()


def _explore_fresh(project, deterministic: bool):
    """Fresh manager from entry, explore to FIND_ADDR, return found stdin bytes."""
    from angr.exploration import RustExplorationManager

    state = project.factory.entry_state()
    mgr = RustExplorationManager(project, [state], deterministic=deterministic)
    mgr.explore(find=FIND_ADDR, num_find=1)
    if not mgr.found:
        return None
    return bytes(mgr.found[0].posix.dumps(0))


def _round_trip(project, deterministic: bool):
    """Step partway, snapshot, reload into a fresh manager, resume to FIND_ADDR.

    Returns (uninterrupted_stdin, resumed_stdin): the found stdin from a run
    that never paused vs. the found stdin from the snapshot-resumed run.
    """
    from angr.exploration import RustExplorationManager

    # Uninterrupted reference run.
    ref = _explore_fresh(project, deterministic)

    # Snapshot-and-resume run.
    state = project.factory.entry_state()
    run_mgr = RustExplorationManager(project, [state], deterministic=deterministic)
    run_mgr.step(n=SNAPSHOT_STEPS)
    with tempfile.NamedTemporaryFile(suffix=".snap", delete=False) as fh:
        snap_path = fh.name
    try:
        run_mgr.dump_snapshot(snap_path)
        resumed = RustExplorationManager.load_from_disk(snap_path, project, deterministic=deterministic)
        resumed.explore(find=FIND_ADDR, num_find=1)
        res = bytes(resumed.found[0].posix.dumps(0)) if resumed.found else None
    finally:
        try:
            os.unlink(snap_path)
        except OSError:
            pass
    return ref, res


def _uniqueness_probe(project, deterministic: bool):
    """Best-effort: count distinct satisfying stdin prefixes at the found state.

    Returns (n_solutions, note). n_solutions is 1 (unique), 2 (>= 2, not
    unique), or None (probe could not run). Advisory only — see module docstring.
    """
    from angr.exploration import RustExplorationManager

    state = project.factory.entry_state()
    mgr = RustExplorationManager(project, [state], deterministic=deterministic)
    mgr.explore(find=FIND_ADDR, num_find=1)
    if not mgr.found:
        return None, "no found state"
    found = mgr.found[0]
    try:
        # SimPacketsStream records each read as (ast, size_ast) in .content.
        # The found-path stdin AST is what the solver must satisfy to reach
        # the find point along *this* path.
        content = getattr(found.posix.stdin, "content", None)
        if not content:
            return None, "no recorded stdin content on found state"
        stdin_ast = content[0][0]
        if not found.solver.symbolic(stdin_ast):
            # AST is concrete along this path -> the read returned a pinned
            # value; exactly one input satisfies this found path.
            return 1, "found-path stdin AST is concrete (read pinned to one value)"
        sols = found.solver.eval_upto(stdin_ast, 2, cast_to=bytes)
        return len(sols), f"eval_upto returned {len(sols)} distinct stdin value(s) for this found path"
    except Exception as exc:
        return None, f"probe error: {type(exc).__name__}: {exc}"


def characterize(samples: int):
    import angr

    project = angr.Project(BINARY, auto_load_libs=False)

    result = {
        "target": "fauxware",
        "find_addr": hex(FIND_ADDR),
        "samples": samples,
        "snapshot_steps": SNAPSHOT_STEPS,
    }

    for det in (False, True):
        key = "deterministic" if det else "nondeterministic"
        fresh = [_explore_fresh(project, det) for _ in range(samples)]
        fresh_hex = [_hexdigest(b) if b is not None else None for b in fresh]
        distinct = sorted(set(h for h in fresh_hex if h is not None))
        ref, resumed = _round_trip(project, det)
        n_sol, sol_note = _uniqueness_probe(project, det)
        result[key] = {
            "fresh_stdin_hex": fresh_hex,
            "fresh_distinct_count": len(distinct),
            "fresh_value_stable": len(distinct) == 1,
            "roundtrip_uninterrupted_hex": _hexdigest(ref) if ref is not None else None,
            "roundtrip_resumed_hex": _hexdigest(resumed) if resumed is not None else None,
            "roundtrip_value_match": (ref is not None and ref == resumed),
            "uniqueness_solutions": n_sol,
            "uniqueness_note": sol_note,
        }

    # Classification synthesis. Two distinct notions are kept separate on
    # purpose (this is the honesty crux for the .8 demo):
    #   * found-path pinning   = does the SINGLE reaching path pin the input?
    #   * program-point uniqueness = is the find ADDRESS reachable by only one
    #     input? (NOT tested here; fauxware 0x4006ed is reachable by both the
    #     SOSNEAKY backdoor and matching creds, so it is NOT point-unique.)
    # Value-reproduction across resume therefore rests on the search *landing
    # on the same path* consistently, not on global constraint-uniqueness.
    det_stable = result["deterministic"]["fresh_value_stable"]
    nondet_stable = result["nondeterministic"]["fresh_value_stable"]
    rt_match = result["deterministic"]["roundtrip_value_match"]
    found_path_pinned = result["deterministic"]["uniqueness_solutions"] == 1
    if det_stable and nondet_stable and rt_match:
        classification = "value-reproducible (found-path pinned, search lands consistently)"
        rationale = (
            "concrete input is identical across fresh runs in both modes and survives a snapshot "
            "round-trip; the reaching path pins the input"
            + (" (found-path stdin AST is concrete)" if found_path_pinned else "")
            + ". CAVEAT: the find address is reachable by other inputs, so this is value-stability "
            "of a consistently-selected path, NOT global constraint-uniqueness."
        )
    elif det_stable and not nondet_stable:
        classification = "search-only (deterministic-mode stabilizes value)"
        rationale = "value varies across fresh runs without deterministic=True; deterministic=True pins it"
    else:
        classification = "search-only"
        rationale = "concrete bytes vary across fresh runs; resume reproduces the SEARCH, not the value"
    result["classification"] = classification
    result["classification_rationale"] = rationale
    result["deterministic_mode_decision"] = _mode_decision(result)
    return result


def _mode_decision(result: dict) -> str:
    """Recommend deterministic-mode usage for the .8 demo, with residual gap."""
    det = result["deterministic"]
    nondet = result["nondeterministic"]
    parts = []
    if det["fresh_value_stable"] and not nondet["fresh_value_stable"]:
        parts.append("USE deterministic=True: it stabilizes the concrete input across fresh runs.")
    elif det["fresh_value_stable"]:
        parts.append("deterministic=True is safe; the value is already stable without it for this target.")
    else:
        parts.append("deterministic=True did NOT fully stabilize the value for this target.")
    if det["roundtrip_value_match"]:
        parts.append("Round-trip reproduced the same concrete input under deterministic=True.")
    else:
        parts.append(
            "RESIDUAL GAP: even under deterministic=True the snapshot round-trip did not reproduce "
            "the identical concrete input (Z3 heuristic latitude across re-solve) — the demo must "
            "claim search continuity, not bit-identical result restore."
        )
    return " ".join(parts)


def _print_report(res: dict) -> None:
    print("=" * 72)
    print("Checkpoint/resume determinism-boundary characterization (angr-4n26m.4)")
    print("=" * 72)
    print(f"target            : {res['target']}  (find={res['find_addr']})")
    print(f"samples / snap@   : {res['samples']} fresh runs, snapshot after {res['snapshot_steps']} steps")
    print()
    for key in ("nondeterministic", "deterministic"):
        d = res[key]
        print(f"[{key}]")
        print(f"  fresh distinct values : {d['fresh_distinct_count']}  (value_stable={d['fresh_value_stable']})")
        print(f"  fresh stdin hex       : {d['fresh_stdin_hex']}")
        print(f"  round-trip match      : {d['roundtrip_value_match']}")
        print(f"    uninterrupted={d['roundtrip_uninterrupted_hex']}")
        print(f"    resumed      ={d['roundtrip_resumed_hex']}")
        print(f"  uniqueness probe      : {d['uniqueness_solutions']}  ({d['uniqueness_note']})")
        print()
    print("-" * 72)
    print(f"CLASSIFICATION : {res['classification']}")
    print(f"  rationale    : {res['classification_rationale']}")
    print(f"MODE DECISION  : {res['deterministic_mode_decision']}")
    print("-" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("-n", "--samples", type=int, default=5, help="fresh-run samples per mode (default 5)")
    args = parser.parse_args(argv)

    # Self-cap address space at 3 GB: this runs angr in-process and the loop
    # box has no swap. Subprocess invocation inherits this guard.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    except (ValueError, OSError):
        pass

    if not os.path.exists(BINARY):
        sys.stderr.write(
            f"error: target binary not found at {BINARY}\n"
            "set ANGR_EXAMPLES_DIR to your angr-examples/examples checkout.\n"
        )
        return 2

    res = characterize(args.samples)

    if args.json:
        json.dump(res, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(res)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
