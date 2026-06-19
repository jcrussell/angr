#!/usr/bin/env python3
"""Rot-proof runner for the Rust-engine showcase demos (angr-4n26m.10).

The blog post (angr-4n26m.11) and its reproducibility claim depend on every
``show_*.py`` demo still running green from a clean checkout. This single
runner invokes each demo as an isolated subprocess, classifies the result by
exit code, and prints one summary table — so the showcase cannot silently
bit-rot. Wire it into CI (or run it by hand before drafting the post).

Exit-code contract shared by every demo (see each ``main()``):

    0  -> demo passed (assertions held)
    2  -> environment not available (example binary missing); treated as SKIP
    *  -> demo FAILED (assertion broke / regression)

So this runner reports PASS / SKIP / FAIL per demo and exits non-zero iff any
demo FAILED. A SKIP (missing ``ANGR_EXAMPLES_DIR`` corpus) does not fail the
run — it is reported loudly but is an environment gap, not a regression.

Usage::

    python tests/benchmarks/show_all.py            # full run, all 6 demos
    python tests/benchmarks/show_all.py --quick     # fast subset where supported
    python tests/benchmarks/show_all.py --json      # machine-readable summary
    python tests/benchmarks/show_all.py --list      # list demos and exit

Each demo is run with ``--json`` so its own stdout stays compact; this runner
captures it but only surfaces it on failure (or with ``--verbose``). Every
demo already self-caps ``RLIMIT_AS`` (3 GB), and the binary-backed ones run
their heavy work in OOM-guarded child processes, so this runner adds only a
per-demo wall-clock timeout on top.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# angr resolves as a namespace package off the repo root (editable install is
# cwd-relative here), so demos must run with the repo root as cwd — matching how
# run_single.py is normally invoked (``python tests/benchmarks/...``).
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# Registry of showcase demos. ``quick_args`` is used in --quick mode; when a
# demo has no faster mode it reuses ``args``. ``needs_corpus`` flags the demos
# that read an example binary (and so can legitimately SKIP via exit code 2).
DEMOS = [
    {
        "name": "raw_speed",
        "script": "show_raw_speed.py",
        "args": ["--json", "-n", "5"],
        "quick_args": ["--json", "--quick", "-n", "5"],
        "needs_corpus": True,
        "blurb": "Rust-vs-Python wall-clock on identical workloads",
    },
    {
        "name": "multiarch",
        "script": "show_multiarch.py",
        "args": ["--json"],
        "quick_args": ["--json", "--quick"],
        "needs_corpus": False,
        "blurb": "same crackme solved across 6 arches (parity)",
    },
    {
        "name": "vuln_finding",
        "script": "show_vuln_finding.py",
        "args": ["--json"],
        "quick_args": ["--json"],
        "needs_corpus": True,
        "blurb": "guided reachability finds a strcpy overflow",
    },
    {
        "name": "checkpoint_resume",
        "script": "show_checkpoint_resume.py",
        "args": ["--json"],
        "quick_args": ["--json"],
        "needs_corpus": True,
        "blurb": "snapshot + resume across a process boundary",
    },
    {
        "name": "resume_boundary",
        "script": "show_resume_boundary.py",
        "args": ["--json"],
        "quick_args": ["--json", "-n", "3"],
        "needs_corpus": True,
        "blurb": "checkpoint/resume determinism-boundary probe",
    },
    {
        "name": "solver_instrumentation",
        "script": "show_solver_instrumentation.py",
        "args": ["--json"],
        "quick_args": ["--json"],
        "needs_corpus": True,
        "blurb": "per-call-site Z3 attribution + lazy-fork stats",
    },
]

# Per-demo wall-clock cap (s). raw_speed runs Python baselines via run_single
# so it is by far the slowest; the others are sub-30s in practice.
DEFAULT_TIMEOUT = 600
RAW_SPEED_TIMEOUT = 1200

PASS, SKIP, FAIL, TIMEOUT, ERROR = "PASS", "SKIP", "FAIL", "TIMEOUT", "ERROR"


def _timeout_for(demo) -> int:
    return RAW_SPEED_TIMEOUT if demo["name"] == "raw_speed" else DEFAULT_TIMEOUT


def run_demo(demo, quick: bool) -> dict:
    """Run one demo subprocess and classify the outcome by exit code."""
    script = os.path.join(HERE, demo["script"])
    argv = demo["quick_args"] if quick else demo["args"]
    cmd = [sys.executable, script, *argv]
    # angr is an editable/namespace package resolved off the repo root, not
    # installed into site-packages, so a plain ``python script.py`` cannot import
    # it. Put the repo root on PYTHONPATH (mirrors the documented
    # ``env PYTHONPATH=$PWD .venv/bin/python`` invocation).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [REPO_ROOT, env.get("PYTHONPATH", "")]))
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=_timeout_for(demo),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": demo["name"],
            "status": TIMEOUT,
            "returncode": None,
            "seconds": round(time.monotonic() - start, 2),
            "stdout": "",
            "stderr": f"timed out after {_timeout_for(demo)}s",
        }
    except OSError as exc:  # script missing / not executable
        return {
            "name": demo["name"],
            "status": ERROR,
            "returncode": None,
            "seconds": round(time.monotonic() - start, 2),
            "stdout": "",
            "stderr": str(exc),
        }

    rc = proc.returncode
    if rc == 0:
        status = PASS
    elif rc == 2 and demo["needs_corpus"]:
        status = SKIP  # example corpus not present — environment gap, not a regression
    else:
        status = FAIL
    return {
        "name": demo["name"],
        "status": status,
        "returncode": rc,
        "seconds": round(time.monotonic() - start, 2),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _print_table(results, verbose: bool) -> None:
    width = max(len(r["name"]) for r in results)
    print()
    print("Showcase demo runner — angr-4n26m.10")
    print("=" * 60)
    for r in results:
        mark = {
            PASS: "✓",
            SKIP: "○",
            FAIL: "✗",
            TIMEOUT: "✗",
            ERROR: "✗",
        }[r["status"]]
        print(f"  {mark} {r['name']:<{width}}  {r['status']:<8} {r['seconds']:>7.2f}s")
        if r["status"] in (FAIL, TIMEOUT, ERROR) or verbose:
            tail = (r["stderr"] or "").strip().splitlines()[-6:]
            for line in tail:
                print(f"        | {line}")
    print("=" * 60)
    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_skip = sum(1 for r in results if r["status"] == SKIP)
    n_fail = len(results) - n_pass - n_skip
    print(f"  {n_pass} passed, {n_skip} skipped, {n_fail} failed")
    if n_skip:
        print("  (skips = example corpus absent; set ANGR_EXAMPLES_DIR to run them)")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="run the fast subset where a demo supports one")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable summary instead of a table")
    parser.add_argument("--verbose", action="store_true", help="always print each demo's stderr tail")
    parser.add_argument("--list", action="store_true", help="list the registered demos and exit")
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="run only the named demo(s); repeatable",
    )
    args = parser.parse_args(argv)

    if args.list:
        for d in DEMOS:
            print(f"{d['name']:<24} {d['script']:<32} {d['blurb']}")
        return 0

    demos = DEMOS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {d["name"] for d in DEMOS}
        if unknown:
            parser.error(f"unknown demo(s): {', '.join(sorted(unknown))}")
        demos = [d for d in DEMOS if d["name"] in wanted]

    results = []
    for demo in demos:
        if not args.json:
            print(f"--- running {demo['name']} ({'quick' if args.quick else 'full'}) ...", file=sys.stderr, flush=True)
        results.append(run_demo(demo, args.quick))

    if args.json:
        # Drop captured stdio from the JSON payload unless a demo failed —
        # keeps the machine summary compact for CI consumption.
        payload = {
            "quick": args.quick,
            "results": [
                {k: v for k, v in r.items() if k != "stdout" or r["status"] not in (PASS, SKIP)} for r in results
            ],
        }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_table(results, args.verbose)

    # Fail the run iff a demo genuinely regressed. SKIP (missing corpus) is fine.
    return 1 if any(r["status"] in (FAIL, TIMEOUT, ERROR) for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
