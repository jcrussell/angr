#!/usr/bin/env python3
"""Diff two ``--counters-json`` outputs from ``run_single.py``.

When ``run_regression.py`` flags a Rust-engine bench as regressed, the
question is invariably *which counter moved?* Today that requires
manually eyeballing two JSON dumps with ~80 keys each. This helper
turns the comparison into a one-shot CLI:

.. code-block:: bash

    # Capture two snapshots
    python tests/benchmarks/run_single.py fauxware --engine rust \\
        --counters-json > /tmp/baseline.json
    # ... apply candidate change, rebuild ...
    python tests/benchmarks/run_single.py fauxware --engine rust \\
        --counters-json > /tmp/current.json

    # Diff: largest absolute deltas first, hides counters that didn't
    # move enough to matter.
    python tests/benchmarks/bench_diff.py /tmp/baseline.json /tmp/current.json

Run end-to-end takes < 30 s for fast-tier benches.

The script is also imported by ``run_regression.py`` so CI failures
on a regressed bench surface the same delta inline against
``baseline_counters.json``. The integration is a soft dependency: if
the baseline counters file is missing the gate keeps working, just
without the per-counter breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable


def load_counters(path: str) -> dict:
    """Load a ``--counters-json`` payload from a file or ``-`` (stdin).

    ``run_single.py`` emits structured JSON between an ``OK rust ...``
    header line and end-of-file when ``--counters-json`` is set. To
    support piping that output directly into the diff tool, we strip
    any leading non-JSON lines before the first ``{`` so users don't
    have to hand-extract the JSON block. Reads stdin when
    ``path == "-"`` for pipeline composition.
    """
    if path == "-":
        text = sys.stdin.read()
    else:
        with open(path) as f:
            text = f.read()
    idx = text.find("{")
    if idx > 0:
        text = text[idx:]
    return json.loads(text)


def _flatten(stats: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a ``mgr.stats()`` dict into a flat ``{key: number}`` map.

    Dict-valued counters (e.g. ``simprocedure_fallback_by_name`` which
    breaks fallbacks down per procedure name) are expanded into
    ``parent.child`` keys so each sub-counter is independently diffable.
    Booleans collapse to 0/1 so a flag flip shows up as ±1.
    """
    out: dict[str, float] = {}
    for key, val in stats.items():
        full = f"{prefix}{key}"
        if isinstance(val, dict):
            for sub_k, sub_v in val.items():
                if isinstance(sub_v, bool):
                    out[f"{full}.{sub_k}"] = int(sub_v)
                elif isinstance(sub_v, (int, float)):
                    out[f"{full}.{sub_k}"] = sub_v
        elif isinstance(val, bool):
            out[full] = int(val)
        elif isinstance(val, (int, float)):
            out[full] = val
        # Strings / other types are skipped — not part of the counter
        # surface we care about for regression diagnosis.
    return out


def compute_diff(
    baseline: dict,
    current: dict,
    threshold_pct: float = 5.0,
    min_abs: float = 10.0,
) -> list[tuple[str, float, float, float, float]]:
    """Return ``[(key, baseline, current, delta, pct), ...]`` sorted by |delta|.

    Filters out counters whose change is below BOTH the absolute and
    percent thresholds — small absolute moves on large counters and
    small percent moves on small counters both qualify as noise. New
    keys are reported with ``baseline=0`` and percent ``inf``; deleted
    keys appear with ``current=0`` and percent ``-100``.
    """
    base = _flatten(baseline)
    curr = _flatten(current)
    rows: list[tuple[str, float, float, float, float]] = []
    for key in set(base) | set(curr):
        b = base.get(key, 0)
        c = curr.get(key, 0)
        delta = c - b
        if delta == 0:
            continue
        if b == 0:
            pct = float("inf") if c != 0 else 0.0
        else:
            pct = (delta / abs(b)) * 100.0
        if abs(delta) < min_abs and abs(pct) < threshold_pct:
            continue
        rows.append((key, b, c, delta, pct))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    return rows


def _fmt_val(v: float) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and not v.is_integer():
        if abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:.3f}"
    iv = int(v)
    if abs(iv) >= 1_000_000:
        return f"{iv:,}"
    return str(iv)


def _fmt_pct(p: float) -> str:
    if p == float("inf"):
        return "    +inf%"
    if p == float("-inf"):
        return "    -inf%"
    return f"{p:+8.1f}%"


def format_report(
    rows: Iterable[tuple[str, float, float, float, float]],
    max_rows: int = 40,
    header: str | None = None,
) -> str:
    """Render the diff rows as a fixed-width table for the terminal."""
    rows = list(rows)
    lines: list[str] = []
    if header:
        lines.append(header)
    if not rows:
        lines.append("no material counter changes")
        return "\n".join(lines)
    lines.append(
        f"{'counter':<50} {'baseline':>14} {'current':>14} "
        f"{'delta':>14} {'pct':>10}"
    )
    lines.append("-" * 106)
    for key, b, c, delta, pct in rows[:max_rows]:
        lines.append(
            f"{key:<50} {_fmt_val(b):>14} {_fmt_val(c):>14} "
            f"{_fmt_val(delta):>14} {_fmt_pct(pct)}"
        )
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more counters omitted)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff two --counters-json outputs from run_single.py.",
    )
    parser.add_argument(
        "baseline",
        help="Baseline JSON file path (use '-' for stdin).",
    )
    parser.add_argument(
        "current",
        help="Current JSON file path (use '-' for stdin).",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Hide counters changing by less than this percent (default: 5.0).",
    )
    parser.add_argument(
        "--min-abs",
        type=float,
        default=10.0,
        help="Hide counters whose absolute delta is below this (default: 10).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=40,
        help="Cap rows printed in the table (default: 40).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the delta as JSON instead of a table.",
    )
    args = parser.parse_args(argv)

    baseline = load_counters(args.baseline)
    current = load_counters(args.current)
    rows = compute_diff(baseline, current, args.threshold_pct, args.min_abs)

    if args.json:
        payload = [
            {
                "counter": key,
                "baseline": b,
                "current": c,
                "delta": delta,
                "pct": (None if pct == float("inf") else pct),
            }
            for key, b, c, delta, pct in rows
        ]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_report(rows, args.max_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
