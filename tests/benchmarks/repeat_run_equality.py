#!/usr/bin/env python3
"""M2.4 (angr-op0dn.10.4): repeat-run result-equality harness.

The M2 promise is **reportable determinism**: two runs of the same bench, in
strict mode, must hand the user the same answer. Not the same wall clock — the
same *result*. This harness is the acceptance instrument for that promise, and
it is composed from parts that already exist rather than being a new framework:

* the subprocess driver is ``run_determinism_census.run_repeat`` (spawned child,
  ``RLIMIT_AS`` cap, per-run timeout) — the S3 spike built it as the measurement
  instrument, this module productizes the comparator on top;
* the reduction is ``content_fingerprint._stable_digest``, which is
  ``PYTHONHASHSEED``-immune and therefore valid across the separate processes
  each repeat runs in;
* the two compared projections come straight off ``run_single``'s child result:
  the **found set** (``stats["found_pcs"]``, the found-state pc multiset, opted
  in via ``ANGR_BENCH_FOUND_FINGERPRINT``) and the **model bytes** (the bench's
  own stdout — the flag / solution text the user would report).

  The found-set projection is deliberately *not* the vh834 AST content
  fingerprint over ``state.constraints``: on Rust found states that collapses
  every state to one value and makes set-equality pass vacuously (bd memory
  ``avoid-content-fingerprint-on-found-states``).

Equality predicate: across all N repeats, the found-fingerprint set is identical
**and** the model bytes are identical. That is the whole gate.

**There is not a single wall-clock assertion in this file, by design.** Timing
variance is out of scope for M2 and permanently so for the bimodal benches
(``benchmark-bimodal-variance-rules``). Bimodal membership is read from
``run_regression.BIMODAL_BENCHMARKS`` — never re-enumerated here — and those
benches are excluded from the gate; ``--include-bimodal`` *reports* their
result-equality without letting it decide the exit code.

Usage::

    python tests/benchmarks/repeat_run_equality.py                    # gate corpus, N=5
    python tests/benchmarks/repeat_run_equality.py --runs 20 --json /tmp/eq.json
    python tests/benchmarks/repeat_run_equality.py --medium              # + MEDIUM_SUITE
    python tests/benchmarks/repeat_run_equality.py --include-bimodal  # report-only extras
    python tests/benchmarks/repeat_run_equality.py --examples csgames2018

Exit code is 0 iff every *gated* bench produced identical results across all
repeats (a bench whose repeats all failed to run is also a failure).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from content_fingerprint import _stable_digest
from run_determinism_census import run_repeat
from run_regression import BIMODAL_BENCHMARKS, MEDIUM_SUITE
from run_single import DEFAULT_MEM_LIMIT_MB

# Fast, non-bimodal, memory-safe benches. Small enough that N=5 is a viable
# pytest gate; the 20-run corpus sweep is a close-out artifact, not a gate.
GATE_CORPUS = ["fauxware", "ais3_crackme", "defcamp_r100"]

# The heavier tier (angr-op0dn.10.6), derived from run_regression's MEDIUM_SUITE
# rather than re-listed, so a bench added there is swept here too. The bimodal
# entries drop out: their *timing* exemption does not extend to results, but
# they stay report-only via --include-bimodal, same as in the fast corpus.
MEDIUM_CORPUS = [entry[0] for entry in MEDIUM_SUITE if entry[0] not in BIMODAL_BENCHMARKS]

DEFAULT_TIMEOUT = 120

# Benches whose model bytes cannot be gated: the value the bench prints is not a
# function of the constraints alone, so repeats may legitimately differ. Their
# found set is still gated — only the model-bytes projection is report-only.
#
# EMPTY as of angr-op0dn.10.7. It held ekopartyctf2016_rev250, whose
# ``posix.dumps(0)`` over a partially-constrained stdin printed a
# different-but-valid input on most repeats. The diagnosis in the bead — "the
# witness is claripy's, chosen past the export boundary" — turned out to be
# wrong: that eval *does* reach the Rust solver (via ``RustSolverFallback``), but
# through ``eval_batch``/``eval_wide``, the two multi-part paths that 10.2 never
# canonicalized. They now take the lexicographic-minimum joint witness, and the
# bench is byte-identical across 20 strict repeats. Add an entry here only for a
# bench whose model genuinely is not constraint-determined, and say why.
PYTHON_MODEL_EVAL_BENCHES: dict[str, str] = {}

# Wall clock must never decide this gate (see the module docstring), but benches
# print it into stdout — csaw_wyvern's own ``Time elapsed: 14.577634572982788``
# would otherwise make every repeat's model fingerprint unique. Redact any float
# with >=3 decimal places: that is a timing/duration shape, while flags and
# byte-string models are printed as text or ``b'...'`` reprs.
_FLOAT_NOISE = re.compile(r"\d+\.\d{3,}")


def _normalize_output(output: str) -> str:
    """Strip wall-clock noise from a bench's stdout before fingerprinting it."""
    return _FLOAT_NOISE.sub("<float>", output)


def _digest(*parts: str) -> str:
    """Cross-process-stable short digest (hex) of canonical string parts."""
    return f"{_stable_digest(*parts):032x}"[:16]


def run_fingerprints(res: dict) -> dict:
    """Project one ``run_single`` child result onto the two compared values.

    ``found_fp`` fingerprints the found-state pc multiset; ``model_fp``
    fingerprints the bench's stdout — timing-redacted — which carries the
    evaluated model bytes (flag / solution) the user would report.
    """
    stats = res.get("stats") or {}
    found_pcs = str(stats.get("found_pcs", ""))
    output = _normalize_output(res.get("output", ""))
    return {
        "found_pcs": found_pcs,
        "found_fp": _digest(found_pcs),
        "model_fp": _digest(output),
        "output": output,
    }


def equality_verdict(runs: list[dict], *, model_gated: bool = True) -> dict:
    """Pure equality decision over the per-run projections of one bench.

    ``runs`` are ``run_fingerprints`` dicts. Equal iff every repeat agrees on
    both projections. Zero repeats is *not* equal — a bench that never ran
    proves nothing, and silently passing it would make the gate vacuous.

    ``model_gated=False`` (a ``PYTHON_MODEL_EVAL_BENCHES`` member) keeps the
    found-set projection enforced but reports the model-bytes projection instead
    of enforcing it.
    """
    found = {r["found_fp"] for r in runs}
    models = {r["model_fp"] for r in runs}
    found_identical = len(found) == 1
    model_identical = len(models) == 1
    return {
        "n": len(runs),
        "found_identical": found_identical,
        "model_identical": model_identical,
        "model_gated": model_gated,
        "equal": bool(runs) and found_identical and (model_identical or not model_gated),
        "distinct_found": sorted(found),
        "distinct_models": sorted(models),
        # Kept for triage: what the disagreeing runs actually produced.
        "found_pcs": sorted({r["found_pcs"] for r in runs}),
        "model_samples": sorted({r["output"].strip()[:200] for r in runs})[:3],
    }


def census_exit_code(census: dict) -> int:
    """0 iff every gated bench is equal. Report-only benches never decide."""
    gated = [v for v in census.values() if v.get("gated")]
    if not gated:
        return 1
    return 0 if all(v["equal"] for v in gated) else 1


def run_bench(name: str, runs: int, *, deterministic: bool, mem_limit_mb: int, timeout: int) -> list[dict]:
    projections = []
    for i in range(runs):
        res = run_repeat(name, "bfs", timeout, mem_limit_mb, deterministic)
        if not res.get("ok"):
            print(f"  FAIL {name} repeat {i + 1}/{runs}: {res.get('error')}", flush=True)
            continue
        projections.append(run_fingerprints(res))
        print(f"  {name} repeat {i + 1}/{runs}: ok", flush=True)
    return projections


def run_equality(
    names: list[str],
    runs: int,
    *,
    deterministic: bool = True,
    mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run every bench ``runs`` times and return the per-bench verdicts.

    Bimodal benches are marked ``gated=False``: their *timing* is permanently
    exempt, and while their results are expected to be stable in strict mode we
    report rather than enforce that here (see the module docstring).
    """
    # Opt the found-state pc projection into each spawned child (spawn inherits env).
    os.environ["ANGR_BENCH_FOUND_FINGERPRINT"] = "1"

    census: dict[str, dict] = {}
    for name in names:
        projections = run_bench(name, runs, deterministic=deterministic, mem_limit_mb=mem_limit_mb, timeout=timeout)
        verdict = equality_verdict(projections, model_gated=name not in PYTHON_MODEL_EVAL_BENCHES)
        verdict["gated"] = name not in BIMODAL_BENCHMARKS
        if name in PYTHON_MODEL_EVAL_BENCHES:
            verdict["model_report_only_reason"] = PYTHON_MODEL_EVAL_BENCHES[name]
        census[name] = verdict
    return census


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--runs", type=int, default=5, help="repeats per bench (default 5)")
    ap.add_argument("--examples", nargs="+", help=f"benches to check (default: {' '.join(GATE_CORPUS)})")
    ap.add_argument(
        "--medium",
        action="store_true",
        help="also check the non-bimodal MEDIUM_SUITE benches (gated, but slow: "
        "~30s/repeat for the whole tier, so N=20 is a ~20min close-out sweep, not a gate)",
    )
    ap.add_argument(
        "--include-bimodal",
        action="store_true",
        help="also run the BIMODAL_BENCHMARKS, report-only (they never decide the exit code). "
        "Slow, and includes CADET_00001_partial, which leaks states (angr-027h) — each repeat "
        "is RLIMIT_AS-capped in its own child, so it degrades to a failed repeat rather than an OOM.",
    )
    ap.add_argument(
        "--no-deterministic",
        action="store_true",
        help="run with strict mode OFF (expected to fail the gate on model-choice benches)",
    )
    ap.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEM_LIMIT_MB)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--json", dest="json_out", help="write the full census to this path")
    args = ap.parse_args()

    names = list(args.examples or GATE_CORPUS)
    if args.medium:
        names += [b for b in MEDIUM_CORPUS if b not in names]
    if args.include_bimodal:
        names += [b for b in sorted(BIMODAL_BENCHMARKS) if b not in names]

    census = run_equality(
        names,
        args.runs,
        deterministic=not args.no_deterministic,
        mem_limit_mb=args.mem_limit_mb,
        timeout=args.timeout,
    )

    print()
    print("=" * 84)
    print(f"M2 repeat-run result equality — {args.runs} runs/bench, strict={not args.no_deterministic}")
    print("=" * 84)
    print(f"{'bench':<32} {'n':>2} {'found=':>7} {'model=':>7} {'gate':>7} {'verdict':>8}")
    print("-" * 84)
    for name, v in census.items():
        model = "yes" if v["model_identical"] else "NO"
        if not v.get("model_gated", True):
            model += "*"
        print(
            f"{name:<32} {v['n']:>2} {'yes' if v['found_identical'] else 'NO':>7} "
            f"{model:>7} "
            f"{'gate' if v['gated'] else 'report':>7} {'EQUAL' if v['equal'] else 'DIFFER':>8}"
        )
    print("-" * 84)
    print("found= : found-state pc multiset identical across every repeat")
    print("model= : evaluated model bytes (bench stdout, timing redacted) identical across every repeat")
    print("     * : model evaluated in Python on partially-constrained input — reported, not gated")
    print("report : bimodal bench (BIMODAL_BENCHMARKS) — reported, never gated")

    code = census_exit_code(census)
    payload = {
        "runs": args.runs,
        "deterministic": not args.no_deterministic,
        "census": census,
        "exit_code": code,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")

    print("\nRESULT: " + ("PASS — every gated bench is result-identical" if code == 0 else "FAIL — see DIFFER rows"))
    return code


if __name__ == "__main__":
    sys.exit(main())
