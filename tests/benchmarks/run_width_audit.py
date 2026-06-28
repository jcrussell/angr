#!/usr/bin/env python3
"""Concurrent-width audit (bd angr-panhl.3).

The parallel-exploration GO/NO-GO turns on one variable the panhl.1 kill-gate
omitted: *sustained* concurrent stash width. panhl.1 measured per-task duration
and a work-stealing migration count, but a state-level worker pool can only
parallelize when there are >=2 independent active states *at the same time* — and
the Z3-bound trio it cleared turn out to be width-1 (single deep path). A bench
that is slow because of one deep solver.check() cannot be sped up by more workers.

This harness drives each bench through `run_single.py --counters-json` and reads
the step-weighted width histogram (`parallel_width_hist`, added in the same bead)
to report, per workload:

  * peak width          — parallel_max_active_width (what panhl.1 saw)
  * sustained width      — the step-weighted histogram [==1, ==2, 3-4, 5-8, >=9]
  * frac_ge3            — fraction of dispatched steps with >=3 concurrent states
  * ms/task             — wall_ms / parallel_tasks
  * favorability        — frac_ge3 * wall_s  (wide AND slow == the only GO shape)

GO criterion (committing to the parallel rewrite): >=1 workload with sustained
frac_ge3 above --go-frac AND ms/task above --go-ms. Otherwise the corpus has no
wide-and-slow workload and parallel is premature on it — sequence angr-11djq
(directed search) first to *create* wide workloads, then revisit.

Usage:
    python tests/benchmarks/run_width_audit.py                 # full corpus
    python tests/benchmarks/run_width_audit.py --only xmllint_getenv fauxware
    python tests/benchmarks/run_width_audit.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SINGLE = HERE / "run_single.py"
BASELINE = HERE / "baseline_timings.json"

OK_RE = re.compile(r"^OK rust \S+ ([0-9.]+)s", re.MULTILINE)


def bench_names() -> list[str]:
    data = json.loads(BASELINE.read_text())
    benches = data.get("benchmarks", data)
    return [k for k, v in benches.items() if isinstance(v, dict)]


def run_one(name: str, timeout: int, strategy: str) -> dict | None:
    """Run a single bench via run_single --counters-json; parse width data.

    Returns None on failure (timeout, crash, or no JSON payload).
    """
    env = dict(os.environ)
    env.setdefault(
        "ANGR_EXAMPLES_DIR",
        os.path.expanduser("~/repos/angr-examples/examples"),
    )
    cmd = [
        sys.executable,
        str(RUN_SINGLE),
        name,
        "--engine",
        "rust",
        "--counters-json",
        "--strategy",
        strategy,
        "--timeout",
        str(timeout),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30, env=env)
    except subprocess.TimeoutExpired:
        return None
    out = proc.stdout
    # The OK line carries wall time; the JSON payload is the trailing {...} block.
    m = OK_RE.search(out)
    elapsed = float(m.group(1)) if m else None
    brace = out.rfind("\n{")
    if brace == -1:
        if out.startswith("{"):
            brace = 0
        else:
            return None
    try:
        stats = json.loads(out[brace:])
    except json.JSONDecodeError:
        return None
    hist = stats.get("parallel_width_hist")
    tasks = stats.get("parallel_tasks", 0)
    if hist is None:
        return None
    total = sum(hist) or 1
    ge2 = sum(hist[1:])
    ge3 = sum(hist[2:])
    return {
        "name": name,
        "elapsed_s": elapsed,
        "tasks": tasks,
        "peak_width": stats.get("parallel_max_active_width", 0),
        "hist": hist,
        "frac_ge2": ge2 / total,
        "frac_ge3": ge3 / total,
        "ms_per_task": (elapsed * 1000.0 / tasks) if (elapsed and tasks) else None,
        "favorability": (ge3 / total) * elapsed if elapsed else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="Restrict to these bench names")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument(
        "--strategy",
        choices=["bfs", "dfs"],
        default="bfs",
        help="BFS maximizes concurrent width (default); use to measure potential parallel width",
    )
    ap.add_argument("--go-frac", type=float, default=0.20, help="GO needs frac_ge3 >= this on >=1 bench (default 0.20)")
    ap.add_argument("--go-ms", type=float, default=50.0, help="GO needs ms/task >= this on the same bench (default 50)")
    ap.add_argument("--json", type=str, help="Write full per-bench results here")
    args = ap.parse_args()

    names = args.only or bench_names()
    results = []
    for name in names:
        print(f"[run] {name} ...", file=sys.stderr, flush=True)
        r = run_one(name, args.timeout, args.strategy)
        if r is None:
            print(f"[skip] {name}: failed/timeout/no-json", file=sys.stderr)
            continue
        results.append(r)

    results.sort(key=lambda r: r["favorability"], reverse=True)

    hdr = f"{'bench':<32} {'wall_s':>7} {'tasks':>6} {'peak':>4} {'%>=2':>6} {'%>=3':>6} {'ms/task':>8} {'favor':>7}  hist[1,2,3-4,5-8,9+]"
    print(hdr)
    print("-" * len(hdr))
    go_hits = []
    for r in results:
        ge2 = f"{100 * r['frac_ge2']:.0f}%"
        ge3 = f"{100 * r['frac_ge3']:.0f}%"
        mspt = f"{r['ms_per_task']:.1f}" if r["ms_per_task"] is not None else "-"
        wall = f"{r['elapsed_s']:.2f}" if r["elapsed_s"] is not None else "-"
        is_go = r["frac_ge3"] >= args.go_frac and r["ms_per_task"] is not None and r["ms_per_task"] >= args.go_ms
        if is_go:
            go_hits.append(r)
        mark = " *GO" if is_go else ""
        print(
            f"{r['name']:<32} {wall:>7} {r['tasks']:>6} {r['peak_width']:>4} "
            f"{ge2:>6} {ge3:>6} {mspt:>8} {r['favorability']:>7.2f}  {r['hist']}{mark}"
        )

    print()
    if go_hits:
        print(
            f"VERDICT: GO — {len(go_hits)} wide-and-slow workload(s) clear "
            f"frac_ge3>={args.go_frac} AND ms/task>={args.go_ms}:"
        )
        for r in go_hits:
            print(
                f"  - {r['name']} (frac_ge3={100 * r['frac_ge3']:.0f}%, "
                f"ms/task={r['ms_per_task']:.0f}, wall={r['elapsed_s']:.1f}s)"
            )
        print("  -> retarget the parallel ship-gate to this class; reopen angr-1ilq.")
    else:
        print(f"VERDICT: NO-GO — no workload clears frac_ge3>={args.go_frac} AND ms/task>={args.go_ms}.")
        print("  The corpus has no wide-AND-slow workload (slow benches are narrow,")
        print("  wide benches are fast). State-level parallelism is premature here;")
        print("  sequence angr-11djq (directed search) first to create wide workloads.")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\n[wrote] {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
