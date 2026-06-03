#!/usr/bin/env python3
"""Sweep N across both engines and emit results.csv + scaling.png.

Usage:
    python sweep.py                       # default N range, both engines
    python sweep.py --n-min 2 --n-max 12
    python sweep.py --engines rust        # rust only
    python sweep.py --no-plot             # skip matplotlib
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
PLOT_PNG = os.path.join(HERE, "scaling.png")

_LINE_RE = re.compile(
    r"^N=(?P<n>\d+) engine=(?P<engine>\w+) wall_s=(?P<wall>[\d.]+) "
    r"rss_kb=(?P<rss>\d+) terminal=(?P<term>\d+) error=(?P<err>.*)$"
)


def parse_run_one_line(line: str) -> dict | None:
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    return {
        "n": int(m.group("n")),
        "engine": m.group("engine"),
        "wall_s": float(m.group("wall")),
        "rss_kb": int(m.group("rss")),
        "terminal": int(m.group("term")),
        "error": m.group("err"),
    }


def run_one(n: int, engine: str, build_dir: str, mem_limit_mb: int, timeout_s: int) -> dict:
    cmd = [
        sys.executable, RUN_ONE,
        "--engine", engine, "--n", str(n),
        "--build-dir", build_dir,
        "--mem-limit-mb", str(mem_limit_mb),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except subprocess.TimeoutExpired:
        return {"n": n, "engine": engine, "wall_s": float(timeout_s),
                "rss_kb": -1, "terminal": -1, "error": "timeout"}

    for line in proc.stdout.splitlines():
        rec = parse_run_one_line(line)
        if rec is not None:
            return rec
    return {"n": n, "engine": engine, "wall_s": -1.0,
            "rss_kb": -1, "terminal": -1, "error": f"no-output (rc={proc.returncode})"}


def write_csv(records: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "engine", "wall_s", "rss_kb", "terminal", "error"])
        w.writeheader()
        for r in records:
            w.writerow(r)


def plot(records: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return

    fig, (ax_w, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))
    for engine, marker in (("rust", "o"), ("python", "s")):
        xs, ys_w, ys_r = [], [], []
        for r in records:
            if r["engine"] != engine or r["error"] not in ("None", ""):
                continue
            xs.append(r["n"])
            ys_w.append(r["wall_s"])
            ys_r.append(r["rss_kb"] / 1024)  # MB
        if xs:
            ax_w.plot(xs, ys_w, marker=marker, label=engine)
            ax_r.plot(xs, ys_r, marker=marker, label=engine)
    for ax, ylabel, title in (
        (ax_w, "wall time (s)", "Wall time vs N"),
        (ax_r, "peak RSS (MB)", "Peak RSS vs N"),
    ):
        ax.set_xlabel("N (branches; 2^N leaf states)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("CoW fork scaling — Rust vs Python (parameterised fork-tree)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"plot saved → {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument("--engines", default="rust,python",
                   help="comma-separated subset of {rust,python}")
    p.add_argument("--mem-limit-mb", type=int, default=3072)
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--build-dir", default="/tmp/fork_tree_build")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    os.makedirs(args.build_dir, exist_ok=True)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    ns = list(range(args.n_min, args.n_max + 1))

    print(f"sweep: N={ns} engines={engines} timeout={args.timeout_s}s")
    records: list[dict] = []
    for n in ns:
        for engine in engines:
            t0 = time.monotonic()
            rec = run_one(n, engine, args.build_dir, args.mem_limit_mb, args.timeout_s)
            elapsed = time.monotonic() - t0
            print(
                f"  N={n:>2} {engine:<6} wall={rec['wall_s']:>7.2f}s "
                f"rss={rec['rss_kb']/1024:>7.1f}MB terminal={rec['terminal']:>6} "
                f"err={rec['error']} (subprocess {elapsed:.1f}s)"
            )
            records.append(rec)
            # Bail if this engine errored at a smaller N (timeout/OOM) — bigger
            # N will only be worse.
            if rec["error"] not in ("None", ""):
                print(f"  → {engine} errored at N={n}; skipping larger N")
                ns_remaining = [m for m in ns if m > n]
                for m in ns_remaining:
                    records.append({
                        "n": m, "engine": engine, "wall_s": -1.0,
                        "rss_kb": -1, "terminal": -1,
                        "error": f"skipped-after-N{n}-{rec['error']}",
                    })
                # Remove this engine from the rest of the sweep.
                engines = [e for e in engines if e != engine]
                break

    write_csv(records, RESULT_CSV)
    print(f"csv saved → {RESULT_CSV}")
    if not args.no_plot:
        plot(records, PLOT_PNG)

    return 0


if __name__ == "__main__":
    sys.exit(main())
