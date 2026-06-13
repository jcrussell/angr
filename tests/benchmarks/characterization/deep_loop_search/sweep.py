#!/usr/bin/env python3
"""Sweep deep-loop search strategies and emit results.csv.

Usage:
    python sweep.py                            # default sweep
    python sweep.py --strategies bfs,dfs --engines rust --n-list 4,8,12
    python sweep.py --max-steps-list 100,500,1000

Each (engine, strategy, N, max_steps) cell runs `run_one.py` in a fresh
subprocess with RLIMIT_AS for OOM containment. Records depth reached,
peak RSS, wall time. Stops increasing N for a given (engine, strategy)
once the smaller N errored.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ONE = os.path.join(HERE, "run_one.py")
RESULT_CSV = os.path.join(HERE, "results.csv")

_LINE_RE = re.compile(
    r"^N=(?P<n>\d+) engine=(?P<engine>\w+) strategy=(?P<strategy>\w+) "
    r"wall_s=(?P<wall>[\d.]+) rss_kb=(?P<rss>\d+) "
    r"steps=(?P<steps>-?\d+) max_depth=(?P<depth>-?\d+) "
    r"active_at_end=(?P<active>-?\d+) deadended=(?P<dead>-?\d+) "
    r"error=(?P<err>.*)$"
)


def parse_line(line: str) -> dict | None:
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    return {
        "n": int(m.group("n")),
        "engine": m.group("engine"),
        "strategy": m.group("strategy"),
        "wall_s": float(m.group("wall")),
        "rss_kb": int(m.group("rss")),
        "steps": int(m.group("steps")),
        "max_depth": int(m.group("depth")),
        "active_at_end": int(m.group("active")),
        "deadended": int(m.group("dead")),
        "error": m.group("err"),
    }


def run_one(
    n: int, engine: str, strategy: str, max_steps: int, build_dir: str, mem_limit_mb: int, timeout_s: int
) -> dict:
    cmd = [
        sys.executable,
        RUN_ONE,
        "--engine",
        engine,
        "--strategy",
        strategy,
        "--n",
        str(n),
        "--max-steps",
        str(max_steps),
        "--build-dir",
        build_dir,
        "--mem-limit-mb",
        str(mem_limit_mb),
    ]
    base = {
        "n": n,
        "engine": engine,
        "strategy": strategy,
        "wall_s": -1.0,
        "rss_kb": -1,
        "steps": -1,
        "max_depth": -1,
        "active_at_end": -1,
        "deadended": -1,
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        base["wall_s"] = float(timeout_s)
        base["error"] = "timeout"
        return base

    for line in proc.stdout.splitlines():
        rec = parse_line(line)
        if rec is not None:
            rec["max_steps_budget"] = max_steps
            return rec
    base["error"] = f"no-output (rc={proc.returncode})"
    return base


def write_csv(records: list[dict], path: str) -> None:
    fields = [
        "n",
        "engine",
        "strategy",
        "max_steps_budget",
        "wall_s",
        "rss_kb",
        "steps",
        "max_depth",
        "active_at_end",
        "deadended",
        "error",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-list", default="4,8,14", help="comma-separated iteration caps to bake into binaries")
    p.add_argument("--max-steps-list", default="200,500", help="comma-separated step budgets")
    p.add_argument("--engines", default="rust", help="comma-separated subset of {rust,python}")
    p.add_argument("--strategies", default="bfs,dfs", help="comma-separated subset of {bfs,dfs}")
    p.add_argument("--mem-limit-mb", type=int, default=3072)
    p.add_argument("--timeout-s", type=int, default=180)
    p.add_argument("--build-dir", default="/tmp/deep_loop_build")
    args = p.parse_args()

    os.makedirs(args.build_dir, exist_ok=True)
    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    step_budgets = [int(x) for x in args.max_steps_list.split(",") if x.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(
        f"sweep: N={ns} steps={step_budgets} engines={engines} "
        f"strategies={strategies} timeout={args.timeout_s}s "
        f"mem={args.mem_limit_mb}MB"
    )

    records: list[dict] = []
    # Skip-after-error tracking is per (engine, strategy) for both N and steps.
    skip = set()  # set of (engine, strategy) tuples that errored
    for n in ns:
        for steps in step_budgets:
            for engine in engines:
                for strategy in strategies:
                    key = (engine, strategy)
                    if key in skip:
                        records.append(
                            {
                                "n": n,
                                "engine": engine,
                                "strategy": strategy,
                                "max_steps_budget": steps,
                                "wall_s": -1.0,
                                "rss_kb": -1,
                                "steps": -1,
                                "max_depth": -1,
                                "active_at_end": -1,
                                "deadended": -1,
                                "error": "skipped-after-earlier-error",
                            }
                        )
                        continue
                    t0 = time.monotonic()
                    rec = run_one(n, engine, strategy, steps, args.build_dir, args.mem_limit_mb, args.timeout_s)
                    elapsed = time.monotonic() - t0
                    print(
                        f"  N={n:>3} steps={steps:>5} {engine:<6} "
                        f"{strategy:<3} wall={rec['wall_s']:>6.2f}s "
                        f"rss={rec['rss_kb'] / 1024:>6.1f}MB "
                        f"depth={rec['max_depth']:>3} "
                        f"active={rec['active_at_end']:>4} "
                        f"dead={rec['deadended']:>4} "
                        f"err={rec['error']} ({elapsed:.1f}s)"
                    )
                    records.append(rec)
                    if rec["error"] not in ("None", None, ""):
                        skip.add(key)

    write_csv(records, RESULT_CSV)
    print(f"csv saved → {RESULT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
