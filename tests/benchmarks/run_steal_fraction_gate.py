#!/usr/bin/env python3
"""angr-729vn: drive the parallel overhead gate with the anti-migration
scheduler's *honest* steal fraction.

The anti-migration scheduler (``native/angr/src/exploration/scheduler.rs``) keeps
worker-local live states in their home Z3 context and serializes a state ONLY on
an actual imbalance steal or to materialize a found terminal. It cannot run live
yet (stepping is GIL-coupled; that is the downstream ``angr-vh834`` / ``1ilq.3``
work), so we measure the steal fraction it *would* produce with the run loop's
already-emitted steal-on-imbalance model and feed that to the projection gate.

``f_model = parallel_migrations / parallel_tasks`` (helpers.rs
``record_migration_sample``) is a live, GIL-free estimate measured on the real
bench, using the same idle+backlog>=2 trigger the scheduler implements. It is a
*lower bound* on the scheduler's true surplus fraction (it counts <=1 steal/step
over a sticky-home simulation), so we ALSO add the found-materialization fraction
(``found / steps``) — the only other full-serde site under the Path-A terminal
model — to get the honest fraction the gate's break-even ``f*`` must bound:

    f_gate = median(f_model) + found / steps

Run it with ``ANGR_PARALLEL_WORKERS=2`` so the model's worker count matches the
gate's ``--workers 2``. GO requires ``f_gate <= steal_be_pess`` (the binding
pessimistic break-even); we report the margin.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SINGLE = HERE / "run_single.py"
GATE = HERE / "run_parallel_overhead_gate.py"


def _trailing_json(out: str) -> dict:
    """Extract run_single's trailing ``--counters-json`` object."""
    brace = out.rfind("\n{")
    if brace == -1 and out.startswith("{"):
        brace = 0
    if brace == -1:
        raise SystemExit(f"no trailing JSON in run_single output:\n{out[-500:]}")
    return json.loads(out[brace:])


def capture_f_model(name: str, workers: int, timeout: int) -> tuple[float, float, dict]:
    """Run the bench once (probe-OFF) and return (f_model, found_fraction, raw)."""
    env = dict(os.environ)
    env["ANGR_PARALLEL_WORKERS"] = str(workers)
    env.setdefault("ANGR_EXAMPLES_DIR", os.path.expanduser("~/repos/angr-examples/examples"))
    env.pop("RUST_PARALLEL_SHADOW_PROBE", None)
    cmd = [
        sys.executable,
        str(RUN_SINGLE),
        name,
        "--engine",
        "rust",
        "--counters-json",
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60, env=env)
    stats = _trailing_json(proc.stdout)
    tasks = stats.get("parallel_tasks", 0)
    migr = stats.get("parallel_migrations", 0)
    steps = stats.get("steps", 0)
    found = stats.get("found", 0)
    if tasks == 0:
        raise SystemExit(f"{name}: parallel_tasks==0 (set ANGR_PARALLEL_WORKERS>=2?)")
    f_model = migr / tasks
    found_fraction = (found / steps) if steps else 0.0
    return f_model, found_fraction, stats


def run_gate(name: str, workers: int, f_gate: float, timeout: int) -> dict:
    cmd = [
        sys.executable,
        str(GATE),
        "--only",
        name,
        "--workers",
        str(workers),
        "--steal-fraction",
        f"{f_gate:.6f}",
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * 3 + 120)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-1000:])
    return {"returncode": proc.returncode, "stdout": proc.stdout}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="codegate_2017-angrybird", help="Target bench (default codegate)")
    ap.add_argument("--workers", type=int, default=2, help="Worker count (default 2)")
    ap.add_argument("--runs", type=int, default=5, help="f_model capture runs for the median (default 5)")
    ap.add_argument("--timeout", type=int, default=180, help="Per-run timeout seconds (default 180)")
    args = ap.parse_args()

    f_models: list[float] = []
    found_fractions: list[float] = []
    for i in range(args.runs):
        f_model, found_fraction, _ = capture_f_model(args.bench, args.workers, args.timeout)
        f_models.append(f_model)
        found_fractions.append(found_fraction)
        print(
            f"[capture {i + 1}/{args.runs}] {args.bench}: "
            f"f_model={f_model * 100:.3f}%  found_fraction={found_fraction * 100:.4f}%",
            file=sys.stderr,
            flush=True,
        )

    median_f_model = statistics.median(f_models)
    found_fraction = statistics.median(found_fractions)
    f_gate = median_f_model + found_fraction
    print(f"\n[steal-fraction] {args.bench} over {args.runs} runs (ANGR_PARALLEL_WORKERS={args.workers}):")
    print(f"  median f_model         = {median_f_model * 100:.3f}%  (lower-bound proxy)")
    print(f"  found materialization  = {found_fraction * 100:.4f}%")
    print(f"  honest f_gate          = {f_gate * 100:.3f}%")

    result = run_gate(args.bench, args.workers, f_gate, args.timeout)
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
