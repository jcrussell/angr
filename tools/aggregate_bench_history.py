#!/usr/bin/env python3
"""Aggregate per-run benchmark history records into a single ``data.json``.

Each input file is a JSON object emitted by
``tests/benchmarks/run_regression.py --history-record``::

    {
      "schema_version": 1,
      "timestamp": "2026-05-15T00:42:00Z",
      "commit": "73e049186...",
      "branch": "master",
      "results": {
        "fauxware": { "rust_time": 0.28, "python_time": 0.40, ... },
        ...
      }
    }

The aggregator walks an input directory (recursively), de-duplicates by
commit SHA (keeping the newest timestamp), sorts chronologically, and
truncates to the most recent ``--max-points`` entries. The output is a
single time-series document consumed by ``docs/dashboard/dashboard.js``::

    {
      "generated": "...",
      "max_points": 50,
      "benchmarks": ["ais3_crackme", "fauxware", ...],
      "series": [
        {
          "timestamp": "...",
          "commit": "...",
          "commit_short": "73e04918",
          "branch": "master",
          "results": { ... }
        },
        ...
      ]
    }

The script never errors on a malformed input file — it just warns and skips.
Designed to be safe to run repeatedly under a GitHub Pages publish workflow.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any


def _iter_record_files(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".json"):
                yield os.path.join(dirpath, name)


def _load_record(path):
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"warn: skipping {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(doc, dict) or "timestamp" not in doc or "results" not in doc:
        print(f"warn: skipping {path}: missing required fields", file=sys.stderr)
        return None
    if doc.get("schema_version") not in (None, 1):
        print(
            f"warn: skipping {path}: unknown schema_version {doc.get('schema_version')!r}",
            file=sys.stderr,
        )
        return None
    return doc


def aggregate(input_dir, max_points):
    """Return the dashboard-shaped time-series document."""
    by_commit: dict[str, dict[str, Any]] = {}
    for path in _iter_record_files(input_dir):
        rec = _load_record(path)
        if rec is None:
            continue
        commit = rec.get("commit") or "unknown"
        ts = rec["timestamp"]
        existing = by_commit.get(commit)
        if existing is None or ts > existing["timestamp"]:
            by_commit[commit] = {
                "timestamp": ts,
                "commit": commit,
                "commit_short": commit[:8] if commit and commit != "unknown" else commit,
                "branch": rec.get("branch", "unknown"),
                "results": rec.get("results", {}),
            }

    # Chronological order, newest entries kept after truncation.
    series = sorted(by_commit.values(), key=lambda r: r["timestamp"])
    if max_points > 0:
        series = series[-max_points:]

    benchmarks = sorted({name for entry in series for name in entry["results"].keys()})
    return {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "max_points": max_points,
        "benchmarks": benchmarks,
        "series": series,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", help="Directory containing benchmark history records")
    parser.add_argument("output", help="Path to write the aggregated data.json")
    parser.add_argument(
        "--max-points",
        type=int,
        default=50,
        help="Maximum number of historical points to retain (default: 50)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    doc = aggregate(args.input_dir, args.max_points)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    print(f"Wrote {args.output} — {len(doc['series'])} points, {len(doc['benchmarks'])} benchmarks")


if __name__ == "__main__":
    main()
