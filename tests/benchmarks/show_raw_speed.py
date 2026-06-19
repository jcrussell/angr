#!/usr/bin/env python3
"""Raw-speed showcase demo for the Rust engine (angr-4n26m.5).

Produces a *reproducible* before/after speed table from a clean checkout,
led by the verified CTF wins and grounded with one x86-64 real-software
comparison. "Before" is the pure-Python angr engine; "after" is the Rust
engine on the identical workload. The framing is **now practical**, not
"newly possible": these explorations always finished under Python, the Rust
engine just makes them fast enough to iterate on.

Every number flows through the measurement-rigor gate
(``measure_showcase.measure`` — see angr-4n26m.3), so nothing here is quoted
from ``baseline_timings.json`` (stale by design). Each ``--both`` invocation
is an OOM-safe subprocess (run_single applies a 4 GB RLIMIT_AS).

Tiers
-----
* **headline** — heavy CTF reversing challenges where Rust wins by ~7-15x
  (fresh medians: eko ~14.7x, flareon ~7.4x). These are slow under Python (eko
  ~33 s/run, flareon ~26 s/run), so the full N>=5 run takes several minutes.
  Excluded from ``--quick``.
* **fast** — bounded CTF crackmes (sub-5 s under Python) that still show a
  solid 3-4x Rust win and re-measure quickly. This is the ``--quick`` set.
* **realsoftware** — busybox, a real static x86-64 ELF from the curate task
  (angr-4n26m.1). A short 60-step ``entry_state`` run is dominated by the
  one-time ~0.9 s PyO3/state-export init tax, so **Rust loses (~0.3x)**. It is
  in the table as an HONESTY row — breadth/realism, NOT a raw-speed win — and
  is labelled as such. Never quote it as a speedup.

Usage::

    # Full table (headline + fast + realsoftware), N=5 — several minutes.
    python tests/benchmarks/show_raw_speed.py

    # In-loop-safe fast tier only (sub-minute), still N=5.
    python tests/benchmarks/show_raw_speed.py --quick

    # Machine-readable; refresh the durable artifact.
    python tests/benchmarks/show_raw_speed.py --json > \
        tests/benchmarks/raw_speed_numbers.json
"""

from __future__ import annotations

import argparse
import json
import sys

# Reuse the number gate verbatim — measure() does the N samples + median/range.
from measure_showcase import _fmt_range, measure

# Tier definitions. Each entry: (target, kind). ``kind`` is one of
# "headline" / "fast" / "realsoftware" and drives table grouping + the
# honesty caveat on the realsoftware row.
HEADLINE = ["ekopartyctf2016_rev250", "flareon2015_5"]
FAST = ["sharif7_rev50", "defcamp_r100", "ais3_crackme"]
REALSOFTWARE = ["busybox_static"]

# busybox is a breadth/realism row, NOT a raw-speed win: its short run is
# init-tax-dominated so Rust is slower. The table marks it explicitly.
NOT_A_SPEEDUP = set(REALSOFTWARE)


def _targets(quick: bool) -> list[tuple[str, str]]:
    rows = [(t, "fast") for t in FAST]
    if not quick:
        rows = [(t, "headline") for t in HEADLINE] + rows + [(t, "realsoftware") for t in REALSOFTWARE]
    return rows


def _print_table(results: list[dict]) -> None:
    # Raw-speed rows (Rust wins) sorted by speedup desc; honesty rows last.
    speed_rows = [r for r in results if r["target"] not in NOT_A_SPEEDUP]
    honesty_rows = [r for r in results if r["target"] in NOT_A_SPEEDUP]
    speed_rows.sort(key=lambda r: r["speedup_median"] or 0, reverse=True)

    print()
    print("Raw speed: pure-Python angr (before) vs Rust engine (after)")
    print("Same binary, same exploration, identical results — just faster.")
    print()
    hdr = f"{'target':<24} {'before (python)':<26} {'after (rust)':<26} {'speedup':<9} note"
    print(hdr)
    print("-" * len(hdr))
    for r in speed_rows:
        sp = f"{r['speedup_median']:.2f}x" if r["speedup_median"] else "n/a"
        note = "BIMODAL — caveat" if r.get("bimodal") else ""
        print(f"{r['target']:<24} {_fmt_range(r['python']):<26} {_fmt_range(r['rust']):<26} {sp:<9} {note}")
    for r in honesty_rows:
        sp = f"{r['speedup_median']:.2f}x" if r["speedup_median"] else "n/a"
        print(
            f"{r['target']:<24} {_fmt_range(r['python']):<26} "
            f"{_fmt_range(r['rust']):<26} {sp:<9} "
            "breadth/init-tax — NOT a raw-speed win"
        )
    print()
    print(
        "Fresh medians over N>=5 --both runs; ranges are [min, max]. Headline "
        "CTF rows show the Rust engine turning a coffee-break run into an "
        "interactive one. The busybox row is an honesty check: a short "
        "real-software run is dominated by one-time init, so Rust loses — "
        "breadth, not raw speed. Never quote baseline_timings.json."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fast tier only (sub-minute, in-loop safe); skips the heavy CTF rows.",
    )
    ap.add_argument("-n", "--repeats", type=int, default=5)
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-run --both timeout (s); flareon needs >=120 under Python.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = ap.parse_args()

    if args.repeats < 5 and not args.json:
        print(
            f"WARNING: n={args.repeats} < 5 — the number gate requires N>=5 for any value quoted in the post.",
            file=sys.stderr,
        )

    rows = _targets(args.quick)
    results = []
    for target, kind in rows:
        print(f"=== measuring {target} ({kind}, n={args.repeats}) ===", file=sys.stderr)
        r = measure(target, args.repeats, args.timeout)
        if r is not None:
            r["tier"] = kind
            results.append(r)

    if args.json:
        json.dump(
            {
                "repeats": args.repeats,
                "quick": args.quick,
                "results": results,
                "_meta": {
                    "task": "angr-4n26m.5",
                    "harness": "tests/benchmarks/show_raw_speed.py",
                    "gate": "measure_showcase.measure (angr-4n26m.3)",
                    "framing": "now practical, not newly possible",
                    "note": (
                        "before=pure-Python angr, after=Rust engine, identical "
                        "workload. busybox_static is a breadth/realism row, NOT a "
                        "raw-speed win (init-tax-dominated; Rust ~0.3x). Never "
                        "quote baseline_timings.json."
                    ),
                },
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0 if results else 1

    _print_table(results)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
