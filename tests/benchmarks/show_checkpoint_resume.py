#!/usr/bin/env python3
"""Flagship checkpoint/resume showcase demo (angr-4n26m.8).

The Rust engine can serialize an in-flight exploration to disk, let the
process die, and resume the SEARCH in a brand-new process. This script
demonstrates that across a genuine process boundary:

    process #1 (``--phase dump``)
        explore ``SNAPSHOT_STEPS`` blocks from the entry state, then
        ``mgr.dump_snapshot(path)`` and EXIT. The process is gone; only the
        on-disk snapshot survives.

    process #2 (``--phase resume``)
        ``RustExplorationManager.load_from_disk(path, project)`` into a fresh
        manager, then continue ``explore(find=...)`` to completion.

The honesty crux (established by the determinism-boundary probe,
angr-4n26m.4, bd memory ``showcase-resume-target-classification``): what is
*guaranteed* across a restore is **search-state continuity** — the resumed
manager picks up the same stash shape and the same active frontier the dump
process had. Bit-identical *result* recovery is NOT guaranteed in general
(model equality is not preserved across re-solve; see rust_engine.rst
Known-limitation and dump_snapshot's docstring). For ``fauxware`` the reaching
input happens to be value-stable empirically, but that rests on the search
landing on the same path, NOT on global constraint-uniqueness — so this demo
*claims* continuity and *reports* value-recovery as an observed bonus, clearly
labelled as not-guaranteed.

Target: ``fauxware`` (find = 0x4006ed, the auth-success ``puts``). It is a
PURE-Rust-native target — the native SimProcedures + native memory plugin keep
the Bucket-D Python overlays (``symbolic_pages`` / ``hook_symbolic_memory`` /
``addr_to_ast``) empty, which is exactly the regime ``dump_snapshot`` captures
faithfully. It is also the canonical round-trip fixture (test_misc.py) and
in-loop memory-safe.

The continuity fingerprint compared across the process boundary is::

    {stash_counts, sorted(active_state_addrs)}

both of which are deterministic structural facts about the search frontier
(independent of Z3 model latitude), so an exact match is a clean proof that
the search was restored, not merely re-run.

Usage::

    python tests/benchmarks/show_checkpoint_resume.py            # human report
    python tests/benchmarks/show_checkpoint_resume.py --json \
        > tests/benchmarks/checkpoint_resume_numbers.json         # durable artifact
    python tests/benchmarks/show_checkpoint_resume.py --steps 8   # snapshot later
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
BINARY = os.path.join(EXAMPLES_DIR, "fauxware", "fauxware")
FIND_ADDR = 0x4006ED
DEFAULT_SNAPSHOT_STEPS = 14  # mid-flight: active + deadended stashes populated, not yet found


def _self_cap_address_space() -> None:
    """Cap RLIMIT_AS at 3 GB — runs angr in-process and the loop box has no swap."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    except (ValueError, OSError):
        pass


def _fingerprint(mgr) -> dict:
    """Structural search-frontier fingerprint, stable across re-solve.

    stash_counts is the per-stash population; active_addrs is the sorted set
    of program counters in the active stash. Neither depends on Z3 model
    latitude, so an exact match across the process boundary proves the search
    state was *restored*, not re-derived.
    """
    counts = {k: v for k, v in mgr.stash_counts().items() if v}
    active_addrs = sorted(hex(s.addr) for s in mgr.active)
    return {"stash_counts": counts, "active_addrs": active_addrs}


def _phase_dump(snapshot_path: str, steps: int) -> dict:
    """Process #1: explore partway, snapshot, capture the pre-snapshot frontier."""
    import angr
    from angr.exploration import RustExplorationManager

    project = angr.Project(BINARY, auto_load_libs=False)
    state = project.factory.entry_state()
    mgr = RustExplorationManager(project, [state])
    mgr.step(n=steps)
    fp = _fingerprint(mgr)
    mgr.dump_snapshot(snapshot_path)
    return {
        "steps": steps,
        "snapshot_bytes": os.path.getsize(snapshot_path),
        "fingerprint": fp,
    }


def _phase_resume(snapshot_path: str) -> dict:
    """Process #2: load the snapshot into a fresh manager and continue to find."""
    import angr
    from angr.exploration import RustExplorationManager

    project = angr.Project(BINARY, auto_load_libs=False)
    resumed = RustExplorationManager.load_from_disk(snapshot_path, project)
    # Fingerprint taken IMMEDIATELY after restore, before any further stepping:
    # this is the frontier the dump process handed off.
    restored_fp = _fingerprint(resumed)
    resumed.explore(find=FIND_ADDR, num_find=1)
    found_stdin_hex = None
    if resumed.found:
        found_stdin_hex = bytes(resumed.found[0].posix.dumps(0)).hex()
    return {
        "restored_fingerprint": restored_fp,
        "found": bool(resumed.found),
        "found_stdin_hex": found_stdin_hex,
    }


def _run_subprocess(phase: str, snapshot_path: str, steps: int) -> dict:
    """Re-invoke this script as a child process for one phase, parse its JSON.

    The child runs angr under its own RLIMIT_AS guard and then exits — which is
    precisely the process-boundary the demo is about. We marshal the phase
    result back over stdout as a single JSON object on the last non-empty line.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--phase",
        phase,
        "--snapshot",
        snapshot_path,
        "--steps",
        str(steps),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.getcwd())
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{phase} phase failed (exit {proc.returncode})")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


def orchestrate(steps: int) -> dict:
    """Drive both phases as separate processes with a shared on-disk snapshot."""
    tmpdir = tempfile.mkdtemp(prefix="ckpt_resume_")
    snap_path = os.path.join(tmpdir, "fauxware.snap")
    try:
        dump = _run_subprocess("dump", snap_path, steps)
        # The dump process has now exited; only the file on disk remains.
        snapshot_survives = os.path.exists(snap_path)
        resume = _run_subprocess("resume", snap_path, steps)
    finally:
        try:
            os.unlink(snap_path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    dump_fp = dump["fingerprint"]
    restored_fp = resume["restored_fingerprint"]
    continuity_match = dump_fp == restored_fp

    return {
        "target": "fauxware",
        "find_addr": hex(FIND_ADDR),
        "snapshot_steps": steps,
        "snapshot_bytes": dump["snapshot_bytes"],
        "snapshot_survives_process_exit": snapshot_survives,
        "dump_fingerprint": dump_fp,
        "restored_fingerprint": restored_fp,
        # The guaranteed claim: structural search frontier restored exactly.
        "search_continuity": continuity_match,
        "resumed_found": resume["found"],
        "resumed_found_stdin_hex": resume["found_stdin_hex"],
        # Honesty framing, grounded in angr-4n26m.4 classification.
        "value_recovery_note": (
            "fauxware's reaching input is empirically value-stable, but that "
            "rests on the search landing on the same path (the find address is "
            "reachable by multiple inputs), NOT on global constraint-uniqueness. "
            "Bit-identical result recovery is therefore an observed bonus here, "
            "not a guarantee across a restore — see angr-4n26m.4 / bd memory "
            "showcase-resume-target-classification."
        ),
    }


def _print_report(res: dict) -> None:
    print("=" * 72)
    print("Checkpoint/resume across a process boundary (angr-4n26m.8)")
    print("=" * 72)
    print(f"target            : {res['target']}  (find={res['find_addr']})")
    print(f"snapshot after    : {res['snapshot_steps']} blocks  ({res['snapshot_bytes']} bytes on disk)")
    print(f"snapshot survives process exit : {res['snapshot_survives_process_exit']}")
    print()
    print("[process #1: dump]  search frontier at snapshot time")
    print(f"  stash counts : {res['dump_fingerprint']['stash_counts']}")
    print(f"  active addrs : {res['dump_fingerprint']['active_addrs']}")
    print()
    print("[process #2: resume]  frontier immediately after load_from_disk")
    print(f"  stash counts : {res['restored_fingerprint']['stash_counts']}")
    print(f"  active addrs : {res['restored_fingerprint']['active_addrs']}")
    print()
    print("-" * 72)
    verdict = "RESTORED EXACTLY" if res["search_continuity"] else "MISMATCH"
    print(f"SEARCH CONTINUITY : {verdict}  (guaranteed claim)")
    print(f"resumed reached find : {res['resumed_found']}")
    if res["resumed_found_stdin_hex"]:
        print(f"  resumed found stdin : {res['resumed_found_stdin_hex']}")
    print()
    print(f"value-recovery note : {res['value_recovery_note']}")
    print("-" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dump", "resume"),
        help="internal: run a single phase in this process (used by the orchestrator)",
    )
    parser.add_argument("--snapshot", help="snapshot path (required with --phase)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--steps", type=int, default=DEFAULT_SNAPSHOT_STEPS, help="blocks to step before snapshotting")
    args = parser.parse_args(argv)

    _self_cap_address_space()

    if not os.path.exists(BINARY):
        sys.stderr.write(
            f"error: target binary not found at {BINARY}\n"
            "set ANGR_EXAMPLES_DIR to your angr-examples/examples checkout.\n"
        )
        return 2

    # Child-process phases: do the work, dump one JSON line, exit.
    if args.phase:
        if not args.snapshot:
            parser.error("--snapshot is required with --phase")
        if args.phase == "dump":
            out = _phase_dump(args.snapshot, args.steps)
        else:
            out = _phase_resume(args.snapshot)
        print(json.dumps(out))
        return 0

    # Orchestrator: drive both phases as separate processes.
    res = orchestrate(args.steps)

    if args.json:
        json.dump(res, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(res)

    return 0 if res["search_continuity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
