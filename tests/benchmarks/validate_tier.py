#!/usr/bin/env python3
"""Tier-drift validator for ``run_single.EXAMPLE_CATALOG``.

Compares the declared ``tier`` field for each catalog entry against the
measured ``rust_time`` in ``baseline_timings.json`` and reports drift.

Tier thresholds (per the comment in ``run_single.py``):

    fast       rust_time < 5s
    medium     5s <= rust_time < 30s
    slow       30s <= rust_time < 120s
    very_slow  rust_time >= 120s

A small fuzzy-boundary tolerance is applied (``BOUNDARY_TOLERANCE``, default
10%) so a bench sitting right on a threshold (e.g. ``flareon2015_2`` at
5.35s) does not register as drift. Both the declared and the strict tier
are accepted whenever the measured time is within ``±tolerance`` of a
bucket boundary.

Catalog entries with ``rust_ok=None`` (Python-only) are classified by
Python time instead of Rust time — matching the convention documented
inline in ``EXAMPLE_CATALOG``.

Exit codes:
    0  no drift detected (or only tolerated boundary entries)
    1  drift detected — at least one entry's declared tier is wrong
       and not within the boundary tolerance

Designed to be cheap to run locally and to wire into CI as a warn-only
check (``|| true``) if desired. Reads no benchmark subprocesses — just
parses the catalog dict and the JSON baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# Allow ``import run_single`` when invoked directly from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_single import EXAMPLE_CATALOG  # noqa: E402

# (lower-inclusive, upper-exclusive) seconds for each tier.
TIER_BOUNDS = {
    "fast": (0.0, 5.0),
    "medium": (5.0, 30.0),
    "slow": (30.0, 120.0),
    "very_slow": (120.0, float("inf")),
}

DEFAULT_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "baseline_timings.json"
)
BOUNDARY_TOLERANCE = 0.10  # ±10% slack on each tier boundary


def classify(seconds: float) -> str:
    """Return the strict tier for a measured runtime in seconds."""
    for tier, (lo, hi) in TIER_BOUNDS.items():
        if lo <= seconds < hi:
            return tier
    return "very_slow"


def within_boundary(seconds: float, tier: str, tol: float) -> bool:
    """True iff ``seconds`` is within ``tol`` of a boundary of ``tier``.

    Used so that, e.g., 5.35s (declared fast, strict medium) does not
    register as drift when ``tol=0.10`` — it is 7% past the 5s boundary.
    """
    if tier not in TIER_BOUNDS:
        return False
    lo, hi = TIER_BOUNDS[tier]
    if lo > 0 and abs(seconds - lo) / lo <= tol:
        return True
    if hi != float("inf") and abs(seconds - hi) / hi <= tol:
        return True
    return False


def measured_time(entry: dict, baseline: dict, name: str) -> Optional[float]:
    """Pick rust_time or python_time per the rust_ok convention."""
    record = baseline.get(name)
    if not record:
        return None
    if entry.get("rust_ok") is None:
        return record.get("python_time")
    return record.get("rust_time")


def audit(catalog: dict, baseline: dict, tol: float = BOUNDARY_TOLERANCE):
    """Return (drift_rows, tolerated_rows, skipped_rows).

    Each row is a tuple ``(name, declared, strict, seconds, source)`` where
    ``source`` is ``"rust"`` or ``"python"`` depending on which time was
    consulted.
    """
    drift, tolerated, skipped = [], [], []
    for name, cfg in catalog.items():
        declared = cfg.get("tier", "fast")
        seconds = measured_time(cfg, baseline, name)
        source = "python" if cfg.get("rust_ok") is None else "rust"
        if seconds is None:
            skipped.append((name, declared, None, None, source))
            continue
        strict = classify(seconds)
        if declared == strict:
            continue
        row = (name, declared, strict, seconds, source)
        if within_boundary(seconds, declared, tol):
            tolerated.append(row)
        else:
            drift.append(row)
    return drift, tolerated, skipped


def _format_rows(rows):
    lines = []
    for name, declared, strict, seconds, source in sorted(rows, key=lambda r: r[3]):
        lines.append(
            f"  {name:42s} declared={declared:10s} expected={strict:10s} "
            f"{source}_time={seconds:.2f}s"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Path to baseline_timings.json (default: alongside this script).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=BOUNDARY_TOLERANCE,
        help="Fractional boundary tolerance (default: 0.10 = ±10%%).",
    )
    parser.add_argument(
        "--show-tolerated",
        action="store_true",
        help="Print rows that drifted but were within the boundary tolerance.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print catalog entries with no measurable time in the baseline.",
    )
    args = parser.parse_args()

    with open(args.baseline) as fh:
        baseline = json.load(fh)

    drift, tolerated, skipped = audit(EXAMPLE_CATALOG, baseline, tol=args.tolerance)

    if drift:
        print(f"Tier drift: {len(drift)} bench(es) out-of-tier beyond ±{args.tolerance:.0%}")
        print(_format_rows(drift))
    else:
        print(
            f"Tier drift: 0 bench(es) (tolerance ±{args.tolerance:.0%}; "
            f"{len(tolerated)} within boundary, {len(skipped)} unmeasured)"
        )

    if args.show_tolerated and tolerated:
        print(f"\nWithin tolerance ({len(tolerated)}):")
        print(_format_rows(tolerated))

    if args.show_skipped and skipped:
        print(f"\nSkipped (no measured time in baseline) ({len(skipped)}):")
        for name, declared, _strict, _seconds, source in sorted(skipped):
            print(f"  {name:42s} declared={declared:10s} ({source}_time=null)")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
