#!/usr/bin/env python3
"""Valgrind memcheck leak gate for the Rust engine core (bead angr-c4xcs.4).

Complements ``run_leak_check.py``, which watches ``ru_maxrss`` across repeated
``main()`` calls of a Callable-heavy *Python* bench. RSS is a coarse signal: it
cannot see a steady small leak that the allocator satisfies out of already-owned
arenas, and it says nothing about *where* bytes went. This gate closes that gap
on the Rust side with valgrind memcheck.

Why not run valgrind over the Python bench? Because memcheck through CPython
buries real findings under interpreter false-positives (pymalloc arenas, frozen
module tables, interned strings), which is why CPython ships its own
``Misc/valgrind-python.supp`` and still warns the result is unreliable. Instead
we drive ``native/angr/examples/leak_probe.rs`` — a pure-Rust binary with no
Python in the process — over the primitives the exploration loop actually churns
(SymContext + z3 solver, RustBV, SymbolicMemory, RustSimState fork).

The gate is a **slope**, not an absolute byte count. The probe drops everything
it allocates, so a correct engine leaks a *constant* number of bytes (one-time z3
globals, Rust lazy statics) no matter how many iterations run. We therefore run
the probe twice, at N and at ``--scale``xN, and fail on the per-iteration growth::

    slope = (lost(hi) - lost(lo)) / (iters_hi - iters_lo)      [bytes/iteration]

where ``lost`` = definitely-lost + indirectly-lost. A slope near zero means the
constant baseline is whatever this machine's z3/glibc happens to allocate once
and never frees, which is exactly the part we do NOT want to gate on: it differs
across distros and would make the job flap. Unbounded growth is the real defect,
and it shows up as a positive slope.

Measured on this branch (2026-07-14, valgrind 3.22, N=20 vs N=200): definitely +
indirectly lost = **0 bytes at both sizes**, slope 0.0 B/iter. still-reachable
(171,280 B) and possibly-lost (1,160 B, TLS-shaped) are likewise byte-identical
across a 10x iteration change, so they are reported but not gated.

Usage::

    python tests/benchmarks/run_valgrind_leak_check.py                 # gate
    python tests/benchmarks/run_valgrind_leak_check.py --json
    python tests/benchmarks/run_valgrind_leak_check.py --self-test     # prove it can fail
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "native" / "angr" / "Cargo.toml"
PROBE_BIN = REPO_ROOT / "target" / "release" / "examples" / "leak_probe"

# "definitely lost: 1,160 bytes in 22 blocks" -> ("definitely lost", 1160, 22)
_SUMMARY_RE = re.compile(
    r"^==\d+==\s+(definitely lost|indirectly lost|possibly lost|still reachable):"
    r"\s+([\d,]+) bytes in ([\d,]+) blocks"
)

# Categories summed into the gated figure. still-reachable and possibly-lost are
# dominated by z3's one-time globals and thread-local bookkeeping; they are
# reported for context but never gated (see module docstring).
GATED_KINDS = ("definitely lost", "indirectly lost")


def build_probe() -> None:
    """Build the leak_probe example in release mode."""
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(MANIFEST), "--example", "leak_probe"],
        check=True,
        cwd=REPO_ROOT,
    )


def run_probe(iters: int, inject: bool) -> dict[str, int]:
    """Run the probe under memcheck; return leaked bytes keyed by loss kind."""
    env = dict(os.environ, ANGR_LEAK_ITERS=str(iters))
    if inject:
        env["ANGR_LEAK_PROBE_INJECT"] = "1"

    proc = subprocess.run(
        [
            "valgrind",
            "--tool=memcheck",
            "--leak-check=full",
            # The probe is deliberately allocation-heavy; without this valgrind
            # truncates the summary after 1000 distinct loss records.
            "--num-callers=8",
            "--error-exitcode=0",
            str(PROBE_BIN),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"leak_probe exited {proc.returncode}:\n{proc.stderr[-2000:]}")

    leaked = dict.fromkeys(("definitely lost", "indirectly lost", "possibly lost", "still reachable"), 0)
    for line in proc.stderr.splitlines():
        match = _SUMMARY_RE.match(line)
        if match:
            leaked[match.group(1)] = int(match.group(2).replace(",", ""))

    if not any(leaked.values()):
        raise SystemExit(f"could not parse a valgrind LEAK SUMMARY:\n{proc.stderr[-2000:]}")
    return leaked


def measure(iters_lo: int, iters_hi: int, inject: bool) -> dict:
    """Run the probe at both sizes and compute the per-iteration leak slope."""
    lo = run_probe(iters_lo, inject)
    hi = run_probe(iters_hi, inject)

    lost_lo = sum(lo[k] for k in GATED_KINDS)
    lost_hi = sum(hi[k] for k in GATED_KINDS)
    slope = (lost_hi - lost_lo) / (iters_hi - iters_lo)

    return {
        "iters_lo": iters_lo,
        "iters_hi": iters_hi,
        "lost_bytes_lo": lost_lo,
        "lost_bytes_hi": lost_hi,
        "bytes_per_iter": slope,
        "detail_lo": lo,
        "detail_hi": hi,
        "inject": inject,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20, help="baseline iteration count (default: 20)")
    parser.add_argument("--scale", type=int, default=10, help="high run is --iters * --scale (default: 10)")
    parser.add_argument(
        "--max-bytes-per-iter",
        type=float,
        default=1.0,
        help="fail if leaked bytes grow faster than this per iteration (default: 1.0)",
    )
    parser.add_argument("--json", action="store_true", help="emit the measurement dict as JSON")
    parser.add_argument("--no-build", action="store_true", help="use an existing leak_probe binary")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run with a deliberate 64 B/iter leak injected and assert the gate FIRES",
    )
    args = parser.parse_args()

    if shutil.which("valgrind") is None:
        raise SystemExit("valgrind not found on PATH (apt install valgrind)")
    if not args.no_build:
        build_probe()
    if not PROBE_BIN.exists():
        raise SystemExit(f"leak_probe binary missing: {PROBE_BIN}")

    iters_hi = args.iters * args.scale
    result = measure(args.iters, iters_hi, inject=args.self_test)
    slope = result["bytes_per_iter"]
    fired = slope > args.max_bytes_per_iter

    if args.json:
        print(json.dumps({**result, "threshold": args.max_bytes_per_iter, "fired": fired}, indent=2))
    else:
        for label, key in (("N", "detail_lo"), (f"{args.scale}N", "detail_hi")):
            detail = result[key]
            summary = ", ".join(f"{k}={v:,}B" for k, v in detail.items())
            print(f"  {label:>3} iters: {summary}")
        print(
            f"\nleaked (definite+indirect): {result['lost_bytes_lo']:,}B @ {args.iters} iters"
            f" -> {result['lost_bytes_hi']:,}B @ {iters_hi} iters"
            f"  => {slope:.2f} B/iter (threshold {args.max_bytes_per_iter} B/iter)"
        )

    if args.self_test:
        # The injected leak is 64 B/iter, far above any sane threshold. If the
        # gate does not trip here, the gate is broken -- not the engine.
        if not fired:
            print(f"\nSELF-TEST FAILED: injected 64 B/iter leak did not trip the gate ({slope:.2f} B/iter)")
            return 1
        print(f"\nSELF-TEST PASSED: injected leak detected at {slope:.2f} B/iter")
        return 0

    if fired:
        print(f"\nLEAK REGRESSION: {slope:.2f} B/iter exceeds {args.max_bytes_per_iter} B/iter")
        return 1
    print("\nOK: no per-iteration leak growth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
