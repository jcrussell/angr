#!/usr/bin/env python3
"""Grep gate: ban raw ``z3::SatResult::{Sat,Unsat}`` matches in Rust core code
(bd angr-qwyti.3).

Interpreting a Z3 ``check()`` outcome must go through the single
forcing-function ``SatOutcome::decided`` (in
``native/angr/src/symbolic/solver_build.rs``), which turns ``Unknown``
(timeout / incompleteness) into ``None`` so every call site is forced to decide
what a timeout means. Conflating ``Unknown`` with ``Unsat`` is the
angr-ph300.43 / angr-n0irt.1 bug class (see the ``invariant-z3-unknown-not-unsat``
memory): it fabricates extrema, permanently pins ``sat_cache=false``, and prunes
feasible branches.

This gate scans production Rust under ``native/angr/src/`` and fails when it
finds a ``SatResult::Sat`` or ``SatResult::Unsat`` token outside two allowed
contexts:

  * a line (or the two lines above it) tagged ``// satresult-exempt`` — reserved
    for the ``SatOutcome`` impl and the per-variant stats counter in
    ``timed_check``, which legitimately distinguish all three outcomes;
  * test code (files named ``*_tests.rs`` or living under a ``*_tests/`` dir),
    where asserting a raw solver verdict is the point of the test.

``SatResult::Unknown`` on its own is always allowed — naming the timeout variant
is never the bug; collapsing it into a sat/unsat boolean is.

Usage::

    tools/audit_satresult.py            # gate: exit 1 on any violation
    tools/audit_satresult.py --list     # list every SatResult::{Sat,Unsat} site

Pure stdlib; runs no symbolic execution.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "native" / "angr" / "src"

# Only the boolean-collapsing variants are banned; Unknown is fine to name.
BANNED = re.compile(r"SatResult::(Sat|Unsat)\b")
EXEMPT_TAG = "satresult-exempt"
# The rationale tag sits on the match arm's own line or just above it.
EXEMPT_LOOKBACK = 2


def _is_test_file(path: Path) -> bool:
    if path.name.endswith("_tests.rs"):
        return True
    return any(part.endswith("_tests") for part in path.parts)


def _tagged(lines: list[str], idx: int) -> bool:
    lo = max(0, idx - EXEMPT_LOOKBACK)
    return any(EXEMPT_TAG in lines[j] for j in range(lo, idx + 1))


def scan(list_all: bool) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in sorted(SRC_ROOT.rglob("*.rs")):
        if not list_all and _is_test_file(path):
            continue
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if not BANNED.search(line):
                continue
            if not list_all and _tagged(lines, idx):
                continue
            rel = path.relative_to(REPO_ROOT)
            hits.append((rel, idx + 1, line.strip()))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list",
        action="store_true",
        help="list every SatResult::{Sat,Unsat} site (including exempt/test)",
    )
    args = ap.parse_args()

    hits = scan(list_all=args.list)

    if args.list:
        for rel, ln, code in hits:
            print(f"{rel}:{ln}: {code}")
        print(f"\n{len(hits)} SatResult::{{Sat,Unsat}} site(s) total.")
        return 0

    if hits:
        print(
            "ERROR: raw SatResult::{Sat,Unsat} match outside SatOutcome::decided "
            "(angr-qwyti.3).\n"
            "Route the check through `.decided()` (Some(true)=Sat, Some(false)="
            "Unsat, None=Unknown),\n"
            "or, for a site that must distinguish all three, tag the line "
            "`// satresult-exempt: <reason>`.\n",
            file=sys.stderr,
        )
        for rel, ln, code in hits:
            print(f"  {rel}:{ln}: {code}", file=sys.stderr)
        return 1

    print("OK: no raw SatResult::{Sat,Unsat} matches outside the helper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
