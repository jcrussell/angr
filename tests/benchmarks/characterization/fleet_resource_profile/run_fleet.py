#!/usr/bin/env python3
"""Fleet-level resource profile (angr-ayrq).

Runs N independent fauxware solves as separate subprocesses, capping
concurrency, and measures:

  * Total wall time (start of first child to exit of last child).
  * Aggregate peak RSS — sum of VmRSS across all live children sampled
    every --sample-ms; we report the max sum over the run.
  * Per-process peak RSS — max ru_maxrss reported by /usr/bin/time
    wrapper (avoided: too brittle) or, more reliably, the max VmRSS
    observed for each PID by the sampler.

Output is a single line of metrics; ``sweep.py`` (or a shell loop) can
drive this across (engine, count, concurrency) points.

Reuses ``tests/benchmarks/run_single.py`` as the per-process worker so
the RLIMIT_AS handling and engine selection stay consistent with the
rest of the harness.

Usage:
    python tests/benchmarks/characterization/fleet_resource_profile/run_fleet.py \\
        --engine rust --count 20 --concurrency 8

Output format:
    engine=<rust|python> count=N concurrency=K wall_s=<f> per_proc_mean_s=<f> \\
      per_proc_max_s=<f> peak_aggregate_rss_mb=<f> peak_per_proc_rss_mb=<f> \\
      sum_per_proc_peak_rss_mb=<f> failures=<i>
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUN_SINGLE = os.path.join(REPO_ROOT, "tests", "benchmarks", "run_single.py")


def _read_vmrss_kb(pid: int) -> int:
    """Read VmRSS for a PID from /proc/<pid>/status. Returns 0 if missing."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1])  # kB
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _descendants(root_pid: int) -> list[int]:
    """Return [root_pid] + all transitive children via /proc/.../task/.../children.

    ``run_single.py`` uses ``multiprocessing.spawn`` so the actual angr
    worker is a grandchild of our ``Popen`` target. We need the whole
    subtree to attribute RSS correctly.
    """
    result = [root_pid]
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        try:
            with open(f"/proc/{pid}/task/{pid}/children") as f:
                children = f.read().split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for c in children:
            try:
                cpid = int(c)
            except ValueError:
                continue
            result.append(cpid)
            stack.append(cpid)
    return result


class RssSampler(threading.Thread):
    """Background thread sampling sum-of-VmRSS across a dynamic PID set.

    Maintains ``per_pid_peak`` (max VmRSS observed per PID, kB) and
    ``aggregate_peak`` (max sum across all live PIDs at any single
    sample, kB).
    """

    def __init__(self, sample_ms: int = 100):
        super().__init__(daemon=True)
        self._period = sample_ms / 1000.0
        self._lock = threading.Lock()
        self._pids: set[int] = set()
        self._halt = threading.Event()
        self.per_pid_peak: dict[int, int] = defaultdict(int)
        self.aggregate_peak: int = 0
        self.peak_concurrency: int = 0

    def add_pid(self, pid: int) -> None:
        with self._lock:
            self._pids.add(pid)

    def halt(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.is_set():
            with self._lock:
                roots = list(self._pids)
            running_workers = 0
            agg = 0
            for root in roots:
                tree_rss = 0
                for pid in _descendants(root):
                    tree_rss += _read_vmrss_kb(pid)
                if tree_rss > 0:
                    running_workers += 1
                    agg += tree_rss
                    if tree_rss > self.per_pid_peak[root]:
                        self.per_pid_peak[root] = tree_rss
            if agg > self.aggregate_peak:
                self.aggregate_peak = agg
            if running_workers > self.peak_concurrency:
                self.peak_concurrency = running_workers
            time.sleep(self._period)


def run_fleet(
    engine: str,
    count: int,
    concurrency: int,
    example: str,
    mem_limit_mb: int,
    sample_ms: int,
) -> dict:
    """Spawn ``count`` workers (max ``concurrency`` at a time) and return metrics."""

    sampler = RssSampler(sample_ms=sample_ms)
    sampler.start()

    slot_q: queue.Queue = queue.Queue()
    for _ in range(concurrency):
        slot_q.put(None)

    results_lock = threading.Lock()
    per_proc_wall: list[float] = []
    failures = 0

    def worker(idx: int) -> None:
        nonlocal failures
        slot_q.get()
        try:
            cmd = [
                sys.executable,
                RUN_SINGLE,
                example,
                "--engine",
                engine,
                "--mem-limit",
                str(mem_limit_mb),
            ]
            t0 = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )
            sampler.add_pid(proc.pid)
            _out, _ = proc.communicate()
            elapsed = time.monotonic() - t0
            with results_lock:
                per_proc_wall.append(elapsed)
                if proc.returncode != 0:
                    failures += 1
        finally:
            slot_q.put(None)

    t_start = time.monotonic()
    threads = []
    for i in range(count):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    total_wall = time.monotonic() - t_start

    sampler.halt()
    sampler.join(timeout=1.0)

    sum_per_proc_peak_kb = sum(sampler.per_pid_peak.values())
    max_per_proc_peak_kb = max(sampler.per_pid_peak.values(), default=0)

    return {
        "engine": engine,
        "count": count,
        "concurrency": concurrency,
        "example": example,
        "wall_s": total_wall,
        "per_proc_mean_s": (
            sum(per_proc_wall) / len(per_proc_wall) if per_proc_wall else 0.0
        ),
        "per_proc_max_s": max(per_proc_wall, default=0.0),
        "peak_aggregate_rss_mb": sampler.aggregate_peak / 1024.0,
        "peak_per_proc_rss_mb": max_per_proc_peak_kb / 1024.0,
        "sum_per_proc_peak_rss_mb": sum_per_proc_peak_kb / 1024.0,
        "peak_concurrency_observed": sampler.peak_concurrency,
        "failures": failures,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["rust", "python"], required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--example", default="fauxware")
    p.add_argument(
        "--mem-limit-mb",
        type=int,
        default=512,
        help="Per-child RLIMIT_AS (MB). Default 512 because each fauxware proc "
        "is light and we want headroom for high concurrency.",
    )
    p.add_argument("--sample-ms", type=int, default=100)
    p.add_argument("--json", action="store_true", help="Emit a JSON blob")
    args = p.parse_args()

    if args.concurrency < 1 or args.count < args.concurrency:
        # Allow count == concurrency (single batch), but not concurrency > count.
        if args.count < args.concurrency:
            args.concurrency = args.count

    metrics = run_fleet(
        engine=args.engine,
        count=args.count,
        concurrency=args.concurrency,
        example=args.example,
        mem_limit_mb=args.mem_limit_mb,
        sample_ms=args.sample_ms,
    )

    if args.json:
        print(json.dumps(metrics))
    else:
        flat = " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(flat)
    return 0 if metrics["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
