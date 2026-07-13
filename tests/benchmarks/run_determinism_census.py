#!/usr/bin/env python3
"""S3 (angr-op0dn.4): attribute the residual determinism variance.

``RustExplorationManager(deterministic=True)`` pins Z3's ``smt.random_seed`` /
``sat.random_seed`` through ``Z3_global_param_set``. That narrows run-to-run
variance but does not collapse it (bd memory ``iaol1-seed-pin-empirically-broken``).
The open question this harness answers: **how much of the residual is model
CHOICE — which satisfying model Z3 happens to hand back, which then feeds
concretization and changes the work the engine does — versus Z3's internal
restart-heuristic path, which is irreducible without a Z3-version change?**

Method. For each bench x mode (``deterministic`` on/off) run N repeats, each in
its own memory-capped subprocess, and record three things per run:

* ``elapsed`` — wall clock.
* a **work fingerprint** — a whitelist of counters that are a pure function of
  the path the engine walked (steps, state creations, solver checks, ...). Two
  runs with the same work fingerprint asked Z3 the same questions in the same
  order and got answers that steered exploration identically.
* a **result fingerprint** — the found-state pc multiset plus a hash of the
  bench's own stdout (the flag / solution the user would report).

Then decompose the wall-clock variance by grouping runs on their work
fingerprint (a standard between/within one-way decomposition):

* **between-group variance** = runs that did *different work*. Different work
  under a pinned seed can only come from Z3 returning a different model, so
  this share is **model choice — boundary-fixable** by canonicalizing which
  model the SymContext boundary returns.
* **within-group variance** = runs that did *identical work* and still took
  different time. Nothing the boundary can do about that: it is Z3's internal
  search path. This share is the **irreducible floor**.

The bead's KILL criterion turns on that split, and on whether the *reportable*
result (found set + evaluated models) is already stable even when timing is not.

Usage::

    python tests/benchmarks/run_determinism_census.py                 # default corpus
    python tests/benchmarks/run_determinism_census.py -n 10 --examples google2016_unbreakable_1
    python tests/benchmarks/run_determinism_census.py --json /tmp/det.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_single import (
    DEFAULT_MEM_LIMIT_MB,
    EXAMPLES_DIR,
    _resolve_examples_dir,
    _run_in_child,
)

NUMBERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "determinism_numbers.json")

# (name, strategy, timeout). Default corpus: one bimodal bench (the thing the
# spike is actually about) plus two stable controls, so a near-zero CV on the
# controls proves the harness itself is not the noise source. The heavy bimodal
# benches (securityfest_fairlight 22s, ekopartyctf2016_sokohashv2 16s,
# hackcon2016_angry-reverser 67s) are opt-in via --examples: N repeats x 2 modes
# of those is a >10min run. CADET_00001_partial is excluded — it leaks states
# (angr-027h) and would OOM a repeat loop.
DEFAULT_CORPUS = [
    ("google2016_unbreakable_1", "bfs", 60),  # bimodal (BIMODAL_BENCHMARKS): big CV, zero work drift
    ("csgames2018", "bfs", 60),  # multi-solution keygen: tiny CV, but the model choice moves
    ("ais3_crackme", "bfs", 60),  # stable control: near-zero CV, identical work + result
]

_KNOWN_STRATEGY = {"defcamp_r100__dfs": ("defcamp_r100", "dfs")}

# Counters that are a pure function of the executed path. Deliberately excludes
# every ns-level timing counter (those vary by construction) and every cache /
# memory-high-water counter that can move with allocator luck.
WORK_COUNTERS = (
    "steps",
    "state_creations",
    "found",
    "deadended_count",
    "avoided_count",
    "pruned_count",
    "errors",
    "z3_check_count",
    "z3_sat_count",
    "z3_unsat_count",
    "z3_eval_count",
    "z3_ast_build_count",
    "callback_count",
    "ffi_crossings",
    "simprocedures",
    "blocks_executed",
)


def _fingerprint(items) -> str:
    blob = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def run_repeat(name, strategy, timeout, mem_limit_mb, deterministic):
    """One repeat, in a spawned + RLIMIT_AS-capped child."""
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    examples_dir = _resolve_examples_dir(name, EXAMPLES_DIR)
    try:
        async_result = pool.apply_async(
            _run_in_child,
            (name, "rust", examples_dir, mem_limit_mb, strategy),
            {"deterministic": deterministic},
        )
        return async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as e:  # pragma: no cover - defensive
        pool.terminate()
        return {"ok": False, "error": str(e)}
    finally:
        pool.terminate()
        pool.join()


def summarize(runs):
    """Variance decomposition over the work-fingerprint grouping.

    Returns cv, the between/within split of the wall-clock *variance*, and the
    result-stability booleans. ``model_choice_share`` is the fraction of total
    variance explained by runs having done different work — the boundary-fixable
    part. ``z3_internal_share`` is the rest: same work, different time.
    """
    times = [r["elapsed"] for r in runs]
    mean = statistics.fmean(times)
    cv = (statistics.pstdev(times) / mean) if mean else 0.0

    groups: dict[str, list[float]] = {}
    for r in runs:
        groups.setdefault(r["work_fp"], []).append(r["elapsed"])

    n = len(times)
    total_ss = sum((t - mean) ** 2 for t in times)
    between_ss = sum(len(g) * (statistics.fmean(g) - mean) ** 2 for g in groups.values())
    within_ss = total_ss - between_ss

    if total_ss > 0:
        model_choice_share = between_ss / total_ss
        z3_internal_share = within_ss / total_ss
    else:
        model_choice_share = z3_internal_share = 0.0

    # Which counters actually moved. Without this the work-fingerprint split is
    # a black box: "the runs did different work" is only actionable once you can
    # see whether the divergence is solver-shaped (z3_check_count, state_creations)
    # or analysis-shaped (block counts drifting with CFGFast's data-region
    # nondeterminism — bd memory cfgfast-data-region-nondeterminism).
    diff_keys = sorted(k for k in WORK_COUNTERS if len({str(r["work"].get(k)) for r in runs}) > 1)

    return {
        "n": n,
        "mean_s": round(mean, 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "cv": round(cv, 4),
        "work_diff_keys": diff_keys,
        "distinct_outputs": sorted({r["output"].strip()[:400] for r in runs})[:3],
        "work_groups": len(groups),
        "work_group_sizes": sorted((len(g) for g in groups.values()), reverse=True),
        "work_identical": len(groups) == 1,
        "result_identical": len({r["result_fp"] for r in runs}) == 1,
        "found_pcs_identical": len({r["found_pcs"] for r in runs}) == 1,
        "output_identical": len({r["output_fp"] for r in runs}) == 1,
        # Variance shares. Only meaningful when work_groups > 1; when every run
        # did identical work the whole residual is Z3-internal by construction.
        "model_choice_var_share": round(model_choice_share, 4),
        "z3_internal_var_share": round(z3_internal_share, 4),
        # The CV that would survive a perfect boundary canonicalization: the
        # within-group (same-work) spread alone.
        "cv_floor_same_work": round((within_ss / n) ** 0.5 / mean, 4) if mean else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--repeats", type=int, default=8, help="repeats per bench per mode (default 8)")
    ap.add_argument(
        "--examples",
        nargs="+",
        help="bench names to census (default: one bimodal + two stable controls)",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["deterministic", "baseline"],
        default=["baseline", "deterministic"],
        help="which modes to run (default both)",
    )
    ap.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEM_LIMIT_MB)
    ap.add_argument("--timeout", type=int, default=None, help="override per-run timeout (seconds)")
    ap.add_argument("--json", dest="json_out", help="write the full census to this path")
    ap.add_argument(
        "--update-numbers",
        action="store_true",
        help=f"persist the census to {os.path.basename(NUMBERS_FILE)}",
    )
    args = ap.parse_args()

    if args.examples:
        corpus = []
        for name in args.examples:
            base, strategy = _KNOWN_STRATEGY.get(name, (name, "bfs"))
            corpus.append((base, strategy, args.timeout or 180))
    else:
        corpus = [(n, s, args.timeout or t) for (n, s, t) in DEFAULT_CORPUS]

    # Opt the found-state pc projection in for every child (spawn inherits env).
    # NOT the vh834 AST content fingerprint, which is vacuous on Rust found
    # states — see bd memory avoid-content-fingerprint-on-found-states.
    os.environ["ANGR_BENCH_FOUND_FINGERPRINT"] = "1"

    census: dict[str, dict] = {}
    for name, strategy, timeout in corpus:
        label = name if strategy == "bfs" else f"{name}__{strategy}"
        census[label] = {}
        for mode in args.modes:
            deterministic = mode == "deterministic"
            runs = []
            for i in range(args.repeats):
                res = run_repeat(name, strategy, timeout, args.mem_limit_mb, deterministic)
                if not res.get("ok"):
                    print(f"FAIL {label} [{mode}] repeat {i}: {res.get('error')}", flush=True)
                    continue
                stats = res.get("stats") or {}
                work = {k: stats.get(k) for k in WORK_COUNTERS if k in stats}
                found_pcs = str(stats.get("found_pcs", ""))
                output_fp = _fingerprint(res.get("output", ""))
                runs.append(
                    {
                        "elapsed": res["elapsed"],
                        "work_fp": _fingerprint(work),
                        "found_pcs": found_pcs,
                        "output_fp": output_fp,
                        "result_fp": _fingerprint([found_pcs, output_fp]),
                        "work": work,
                        "output": res.get("output", ""),
                    }
                )
                print(f"  {label} [{mode}] {i + 1}/{args.repeats}: {res['elapsed']:.2f}s", flush=True)
            if not runs:
                census[label][mode] = {"error": "all repeats failed"}
                continue
            census[label][mode] = summarize(runs)

    print()
    print("=" * 100)
    print("S3 determinism census — wall-clock variance attribution (work-fingerprint decomposition)")
    print("=" * 100)
    hdr = f"{'bench':<28} {'mode':<14} {'n':>2} {'mean':>7} {'cv':>7} {'wgrp':>5} {'model%':>7} {'z3%':>6} {'cvfloor':>8} {'result=':>8}"
    print(hdr)
    print("-" * 100)
    for label, modes in census.items():
        for mode, s in modes.items():
            if "error" in s:
                print(f"{label:<28} {mode:<14} {s['error']}")
                continue
            print(
                f"{label:<28} {mode:<14} {s['n']:>2} {s['mean_s']:>6.2f}s {s['cv']:>7.3f} "
                f"{s['work_groups']:>5} {s['model_choice_var_share'] * 100:>6.1f}% "
                f"{s['z3_internal_var_share'] * 100:>5.1f}% {s['cv_floor_same_work']:>8.3f} "
                f"{'yes' if s['result_identical'] else 'NO':>8}"
            )
    print("-" * 100)
    print("model% = share of wall-clock VARIANCE from runs doing different work (Z3 model choice; boundary-fixable)")
    print("z3%    = share from runs doing identical work at different speed (Z3 internal path; irreducible)")
    print("cvfloor= the CV that would survive a perfect boundary canonicalization")
    print("result== found-pc multiset + bench stdout identical across every repeat")

    payload = {"repeats": args.repeats, "corpus": [c[0] for c in corpus], "census": census}
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    if args.update_numbers:
        with open(NUMBERS_FILE, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nwrote {NUMBERS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
