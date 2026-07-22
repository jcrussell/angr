#!/usr/bin/env python3
"""Property-based differential fuzzer for the Rust symbolic execution engine.

The property under test: for any (binary, exploration strategy) drawn from
the angr-examples corpus, the Rust and Python engines should produce
equivalent end-of-run output. Divergence flags a real correctness bug
(VEX op mismatch, calling-convention bug, missing simprocedure, etc.).

Unlike `tests/benchmarks/run_regression.py`, which runs a fixed list and
checks output for non-`rust_only` entries only, this fuzzer:

  * Samples random (example, strategy) tuples from the cataloged corpus.
  * Forces a side-by-side Python-vs-Rust run for every trial (never
    `rust_only`), so model nondeterminism can be diagnosed rather than
    masked.
  * Defaults to skipping examples marked `rust_ok=False` and those in
    `BIMODAL_BENCHMARKS` (their Z3-driven output drift is benign and
    swamps real divergence signals).
  * Records per-trial outcomes in a JSON report so divergences can be
    triaged independently.

The fuzzer reuses the subprocess machinery in `run_single._run_in_child`,
so every trial executes under a 4 GB `RLIMIT_AS` and cannot OOM-kill the
orchestrator on an 8 GB/no-swap host. Trials run sequentially.

================================================================
Triage workflow
================================================================

When this fuzzer reports a divergence, follow these steps in order:

1. **Reproduce single trial.** From the JSON report (`--report`), pick
   the first divergent (example, strategy, seed) entry and run:

       python tests/benchmarks/run_single.py <example> --both \\
           --strategy <strategy>

   If outputs match, the divergence is path-order-dependent and only
   reproduces under randomized exploration; re-run the fuzzer with
   `--seed <S>` to reproduce.

2. **Classify.** A divergence falls into one of three buckets:

   a) **Z3 model nondeterminism (benign).** Both engines find a
      satisfying solution, but unconstrained bytes evaluate to
      different fill values. Symptom: outputs differ only in
      `\\xNN`-escape suffix bytes. Add the example to
      `BIMODAL_BENCHMARKS` in `run_regression.py` (if not already) and
      to the fuzzer's bimodal skip-list below.

   b) **Path divergence.** One engine finds a solution, the other
      doesn't (or finds a different address). This is usually a bug —
      missing simprocedure, VEX op mismatch, or calling-convention
      drift. Drop into `run_single --diff-state` to localize:

          python tests/benchmarks/run_single.py <example> \\
              --diff-state --strategy <strategy>

      The first diverging step's PC tells you which basic block is
      the culprit.

   c) **Crash / engine error.** Either subprocess returns
      `ok: False` with an error message. Most commonly:
      `MemoryError (hit memory limit)` (raise `--mem-limit`), or a
      Rust panic surfaced as a Python exception (file a bug with
      example, strategy, seed, and the Rust error string).

3. **Fix and re-run.** After landing the fix, re-run the fuzzer with
   the same seed to confirm the divergence is gone:

       python tests/benchmarks/property_fuzzer.py --trials 200 --seed <S> --strict

   Then bump `--trials` for the regression-guard run.

================================================================
Usage
================================================================

    # Quick smoke test (~30s)
    python tests/benchmarks/property_fuzzer.py --trials 8 --seed 0

    # Full nightly sweep (~30min, 50 trials)
    python tests/benchmarks/property_fuzzer.py --trials 50 --seed 0 --report fuzz.json

    # Stress run on a single example (path-order coverage)
    python tests/benchmarks/property_fuzzer.py --trials 20 --only fauxware

    # Strict mode: every divergence is a failure (default: prints summary
    # and exits 0 only if zero divergences)
    python tests/benchmarks/property_fuzzer.py --trials 50 --strict

    # Solver-differential mode: random constraint sets, RustSolverContext
    # vs claripy (no binaries involved, ~10s for the default 100 trials)
    python tests/benchmarks/property_fuzzer.py --mode solver --trials 100 --seed 0

    # Re-run exactly one solver trial from a reported divergence
    python tests/benchmarks/property_fuzzer.py --mode solver --solver-trial-seed 12345
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field

# Reuse the subprocess machinery from run_single.py and the bimodal /
# output-normalization helpers from run_regression.py — both live in
# this directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from run_regression import (
    BIMODAL_BENCHMARKS,
    FAST_SUITE,
    MEDIUM_SUITE,
    _normalize_output,
)
from run_single import (
    DEFAULT_MEM_LIMIT_MB,
    EXAMPLE_CATALOG,
    EXAMPLES_DIR,
    _run_in_child,
)


def _known_diverge_set() -> frozenset[str]:
    """Examples where end-of-run output divergence between engines is
    expected — not a bug.

    Two sources contribute:

    * Entries in `run_regression.{FAST,MEDIUM}_SUITE` whose `rust_only`
      flag is True. Those entries are marked rust_only precisely
      because output comparison is unreliable (Z3 model
      nondeterminism, or different stash population after a Rust-side
      explore that the Python solve.py wasn't designed for). Catching
      "divergence" on them adds noise rather than signal.
    * `BIMODAL_BENCHMARKS` — same reasoning, broader scope.

    The fuzzer still RUNS these examples to exercise the engines, but
    a divergence on them is classified `expected-diverge` rather than
    `diverge` so the strict-mode gate only fails on NEW divergences.
    """
    expected: set[str] = set(BIMODAL_BENCHMARKS)
    for suite in (FAST_SUITE, MEDIUM_SUITE):
        for entry in suite:
            name = entry[0]
            rust_only = entry[3] if len(entry) > 3 else False
            if rust_only:
                expected.add(name)
    return frozenset(expected)


_KNOWN_DIVERGE = _known_diverge_set()


# Examples that the fuzzer should never sample. Reasons:
#   * Both engines time out (very_slow tier — wastes the trial budget).
#   * Rust engine known-broken (`rust_ok=False`) — angr-3hzg is about
#     finding NEW divergences, not re-confirming known ones.
#   * `rust_ok=None` cases that fall in the same broken/timeout buckets.
#
# Bimodal examples are filtered separately so `--include-bimodal` can
# opt in for those who want full coverage.
_BLOCKLIST = frozenset(
    {
        # tier=very_slow, both engines time out
        "sharif7_rev50",
        "0ctf_momo_3",
        "tumctf2016_zwiebel",
        "b01lersctf2020_little_engine",
        # rust_ok=False (known broken — out of scope for this fuzzer)
        "asisctffinals2015_license",
        "CADET_00001",
        "ekopartyctf2015_rev100",
        "whitehat_crypto400",
        "asisctffinals2015_fake",
        # Harness-incompatible (uses sys.argv / factory.successors directly,
        # so the engine swap in _run_in_child doesn't apply)
        "insomnihack_aeg",
        "0ctf_trace",
        # Python-side broken
        "defcamp_r200",
    }
)

_STRATEGIES = ("bfs", "dfs")


@dataclass
class TrialResult:
    """One fuzzer trial: a single (example, strategy, seed) run on both engines."""

    trial_idx: int
    example: str
    strategy: str
    seed: int
    bimodal: bool
    python_ok: bool
    rust_ok: bool
    python_elapsed: float | None = None
    rust_elapsed: float | None = None
    python_error: str | None = None
    rust_error: str | None = None
    output_match: bool | None = None
    output_match_normalized: bool | None = None
    python_output_snippet: str | None = None
    rust_output_snippet: str | None = None
    classification: str = "unrun"  # one of: pass | diverge | expected-diverge | py-fail | rust-fail | both-fail
    divergence_kind: str | None = None  # exact-mismatch | normalized-mismatch | crash-asymmetry

    def is_divergence(self) -> bool:
        return self.classification == "diverge"

    def is_expected_diverge(self) -> bool:
        return self.classification == "expected-diverge"


@dataclass
class FuzzerReport:
    seed: int
    trials: int
    examples_seen: list = field(default_factory=list)
    results: list = field(default_factory=list)
    total_elapsed: float = 0.0

    def summary(self) -> dict:
        counts: dict = {}
        for r in self.results:
            counts[r.classification] = counts.get(r.classification, 0) + 1
        return {
            "seed": self.seed,
            "trials": self.trials,
            "total_elapsed_sec": round(self.total_elapsed, 2),
            "examples_seen": sorted(set(self.examples_seen)),
            "counts": counts,
        }


def _eligible_examples(include_bimodal: bool = False) -> list[str]:
    """Return the catalog entries the fuzzer is allowed to sample."""
    out = []
    for name, info in EXAMPLE_CATALOG.items():
        if name in _BLOCKLIST:
            continue
        # Skip slow/very_slow tiers — the fuzzer wants throughput.
        if info["tier"] in ("slow", "very_slow"):
            continue
        # Skip known-broken Rust runs and unknown statuses for now.
        if info["rust_ok"] is not True:
            continue
        if not include_bimodal and name in BIMODAL_BENCHMARKS:
            continue
        # Make sure the binary actually exists locally — the corpus may
        # not be checked out on the host running the fuzzer.
        if not os.path.exists(os.path.join(EXAMPLES_DIR, name, "solve.py")):
            continue
        out.append(name)
    return sorted(out)


def _run_one(example: str, engine: str, strategy: str, timeout: int, mem_limit_mb: int) -> dict:
    """Run a single (example, engine, strategy) trial in a subprocess."""
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(_run_in_child, (example, engine, EXAMPLES_DIR, mem_limit_mb, strategy))
        return async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as e:
        pool.terminate()
        return {"ok": False, "error": f"subprocess crash: {e}"}
    finally:
        pool.terminate()
        pool.join()


def _truncate_output(s: str, limit: int = 200) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _outputs_match(py_out: str, rust_out: str) -> tuple[bool, bool]:
    """Return (exact_match, normalized_match). Mirrors run_regression."""
    py = py_out.strip()
    rs = rust_out.strip()
    if py == rs:
        return True, True
    npy = _normalize_output(py)
    nrs = _normalize_output(rs)
    if npy == nrs:
        return False, True
    if npy.split() == nrs.split():
        return False, True
    return False, False


def run_trial(idx: int, example: str, strategy: str, seed: int, timeout: int, mem_limit_mb: int) -> TrialResult:
    """Run both engines on (example, strategy) and classify the outcome."""
    result = TrialResult(
        trial_idx=idx,
        example=example,
        strategy=strategy,
        seed=seed,
        bimodal=example in BIMODAL_BENCHMARKS,
        python_ok=False,
        rust_ok=False,
    )

    py = _run_one(example, "python", strategy, timeout, mem_limit_mb)
    rs = _run_one(example, "rust", strategy, timeout, mem_limit_mb)

    result.python_ok = bool(py.get("ok"))
    result.rust_ok = bool(rs.get("ok"))
    result.python_elapsed = py.get("elapsed")
    result.rust_elapsed = rs.get("elapsed")
    result.python_error = None if py.get("ok") else py.get("error", "?")
    result.rust_error = None if rs.get("ok") else rs.get("error", "?")

    if not result.python_ok and not result.rust_ok:
        result.classification = "both-fail"
        return result
    if not result.python_ok:
        result.classification = "py-fail"
        return result
    if not result.rust_ok:
        result.classification = "rust-fail"
        result.divergence_kind = "crash-asymmetry"
        return result

    py_out = py.get("output", "") or ""
    rs_out = rs.get("output", "") or ""
    exact, normalized = _outputs_match(py_out, rs_out)
    result.output_match = exact
    result.output_match_normalized = normalized
    result.python_output_snippet = _truncate_output(py_out)
    result.rust_output_snippet = _truncate_output(rs_out)

    if normalized:
        result.classification = "pass"
    else:
        # Mark divergence as "expected" if the example is on the
        # known-divergent list (rust_only or bimodal). Otherwise it's a
        # new finding that should fail strict mode.
        if example in _KNOWN_DIVERGE:
            result.classification = "expected-diverge"
        else:
            result.classification = "diverge"
        result.divergence_kind = "exact-mismatch" if exact is False else "normalized-mismatch"
    return result


# ======================================================================
# Solver-differential mode (angr-ph300.4)
# ======================================================================
#
# The corpus mode above compares whole-binary exploration output, which
# makes a solver bug expensive to localize. This mode targets the solver
# layer directly: it draws a random constraint set over random-width BVs
# and compares `RustSolverContext` against a plain `claripy.Solver` on
# satisfiability, min, max, eval-set and the one-model batch path.
#
# Every trial is driven by its own `trial_seed`, so a divergence is
# reproducible standalone via `--mode solver --solver-trial-seed <S>`.

_SOLVER_WIDTHS = (8, 16, 32, 64)

# Comparison operators, as (name, callable) over two claripy BVs. Signed
# comparisons go through the operator methods claripy exposes for them.
_SOLVER_CMPS = (
    ("eq", lambda a, b: a == b),
    ("ne", lambda a, b: a != b),
    ("ult", lambda a, b: a.ULT(b)),
    ("ule", lambda a, b: a.ULE(b)),
    ("ugt", lambda a, b: a.UGT(b)),
    ("uge", lambda a, b: a.UGE(b)),
    ("slt", lambda a, b: a.SLT(b)),
    ("sle", lambda a, b: a.SLE(b)),
    ("sgt", lambda a, b: a.SGT(b)),
    ("sge", lambda a, b: a.SGE(b)),
)


def _gen_solver_case(rng, claripy):
    """Build one random (variables, constraints) pair.

    Kept deliberately shallow (depth <= 2, <= 3 variables, <= 4
    constraints): the point is to cover the Rust translation and solving
    paths broadly, not to build Z3-hard instances that dominate runtime.
    """
    width = rng.choice(_SOLVER_WIDTHS)
    nvars = rng.randint(1, 3)
    variables = [claripy.BVS(f"v{i}", width, explicit_name=True) for i in range(nvars)]

    def leaf():
        if rng.random() < 0.65:
            return rng.choice(variables)
        return claripy.BVV(rng.getrandbits(width), width)

    def term(depth=0):
        if depth >= 2 or rng.random() < 0.45:
            return leaf()
        a = term(depth + 1)
        op = rng.choice(("add", "sub", "mul", "and", "or", "xor", "lshr", "shl"))
        if op == "add":
            return a + leaf()
        if op == "sub":
            return a - leaf()
        if op == "mul":
            # Only var*const — symbolic*symbolic multiplication is the one
            # shape that reliably turns a trivial instance into a slow one.
            return a * claripy.BVV(rng.randint(1, 7), width)
        if op == "and":
            return a & leaf()
        if op == "or":
            return a | leaf()
        if op == "xor":
            return a ^ leaf()
        shift = claripy.BVV(rng.randint(1, max(1, width - 1)), width)
        return claripy.LShR(a, shift) if op == "lshr" else (a << shift)

    constraints = []
    for _ in range(rng.randint(1, 4)):
        _name, cmp_fn = rng.choice(_SOLVER_CMPS)
        constraints.append(cmp_fn(term(), term()))
    return variables, constraints


def _solver_trial_child(trial_seeds, mem_limit_mb, n_eval):
    """Run a chunk of solver-differential trials in a subprocess.

    Called via multiprocessing spawn, so it must stay top-level and take
    only picklable arguments. Returns a list of plain dicts (one per
    trial) — dataclasses are rebuilt in the parent.
    """
    import contextlib
    import resource

    mem_bytes = mem_limit_mb * 1024 * 1024
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    with contextlib.suppress(ValueError):  # Can't raise above the hard limit.
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, _hard))

    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import claripy
    from angr.rustylib.vex_engine import RustSolverContext

    from angr.exploration.rust_manager import _setup_shared_z3_context

    # Rust and claripy must share one Z3 context for AST passthrough.
    _setup_shared_z3_context()

    out = []
    for trial_seed in trial_seeds:
        try:
            out.append(_solver_trial_body(trial_seed, n_eval, claripy, RustSolverContext))
        except Exception as e:
            out.append(
                {
                    "trial_seed": trial_seed,
                    "classification": "error",
                    "detail": f"{type(e).__name__}: {e}",
                    "constraints": [],
                }
            )
    return out


def _solver_trial_body(trial_seed, n_eval, claripy, RustSolverContext):
    """The actual Rust-vs-claripy comparison for one seed.

    Returns a dict with `classification` in {pass, diverge, error} and,
    on divergence, a `detail` string plus the rendered constraint list
    (the standalone reproducer).
    """
    rng = random.Random(trial_seed)
    variables, constraints = _gen_solver_case(rng, claripy)
    rendered = [str(c) for c in constraints]

    def fail(detail):
        return {
            "trial_seed": trial_seed,
            "classification": "diverge",
            "detail": detail,
            "constraints": rendered,
        }

    ctx = RustSolverContext()
    py_solver = claripy.Solver()
    for c in constraints:
        ctx.add_constraint_ast(c)
        py_solver.add(c)

    rust_sat = bool(ctx.satisfiable())
    py_sat = bool(py_solver.satisfiable())
    if rust_sat != py_sat:
        return fail(f"satisfiable mismatch: rust={rust_sat} python={py_sat}")

    if not py_sat:
        # UNSAT: everything downstream must degrade uniformly.
        for v in variables:
            if ctx.eval(v) is not None:
                return fail(f"unsat but rust eval({v}) returned a value")
            if ctx.min(v, signed=False) is not None or ctx.max(v, signed=False) is not None:
                return fail(f"unsat but rust min/max({v}) returned a value")
        return {"trial_seed": trial_seed, "classification": "pass", "detail": "unsat", "constraints": rendered}

    for v in variables:
        rust_min = ctx.min(v, signed=False)
        rust_max = ctx.max(v, signed=False)
        py_min = py_solver.min(v)
        py_max = py_solver.max(v)
        if rust_min != py_min:
            return fail(f"min({v}) mismatch: rust={rust_min} python={py_min}")
        if rust_max != py_max:
            return fail(f"max({v}) mismatch: rust={rust_max} python={py_max}")

        # eval: compare semantically, not by model identity. Every value
        # either engine returns must be a genuine solution on BOTH.
        rust_vals = list(ctx.eval_upto(v, n_eval))
        py_vals = list(py_solver.eval(v, n_eval))
        for val in rust_vals:
            if not py_solver.solution(v, val):
                return fail(f"rust eval({v}) returned {val}, which python rejects")
        for val in py_vals:
            if not ctx.solution(v, val):
                return fail(f"python eval({v}) returned {val}, which rust rejects")
        if len(rust_vals) != len(set(rust_vals)):
            return fail(f"rust eval_upto({v}, {n_eval}) returned duplicates: {rust_vals}")
        # When BOTH sides came back short of the cap they enumerated the
        # full solution set, so the sets must agree exactly.
        if len(rust_vals) < n_eval and len(py_vals) < n_eval and set(rust_vals) != set(py_vals):
            return fail(f"exhaustive eval({v}) set mismatch: rust={sorted(rust_vals)} python={sorted(py_vals)}")

    # Batch path: one model for all variables at once. Unlike per-variable
    # eval the results must be JOINTLY consistent, so check them together.
    batch = ctx.eval_batch(list(variables))
    if batch is not None:
        if len(batch) != len(variables):
            return fail(f"eval_batch returned {len(batch)} values for {len(variables)} variables")
        joint = claripy.Solver()
        for c in constraints:
            joint.add(c)
        for v, val in zip(variables, batch):
            joint.add(v == val)
        if not joint.satisfiable():
            return fail(f"eval_batch model {list(batch)} is not jointly satisfiable under python")

    return {"trial_seed": trial_seed, "classification": "pass", "detail": "sat", "constraints": rendered}


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_solver_mode(args) -> int:
    """Entry point for `--mode solver`. Returns the process exit code."""
    seed = args.seed if args.seed is not None else int(time.time())
    if args.solver_trial_seed is not None:
        trial_seeds = [args.solver_trial_seed]
    else:
        rng = random.Random(seed)
        trial_seeds = [rng.randint(0, 2**31 - 1) for _ in range(args.trials)]

    print(
        f"property-fuzzer[solver]: seed={seed} trials={len(trial_seeds)} "
        f"chunk={args.solver_chunk} eval-n={args.solver_eval_n} mem-limit={args.mem_limit}MB"
    )

    ctx_mp = multiprocessing.get_context("spawn")
    results: list = []
    t0 = time.perf_counter()
    for chunk in _chunks(trial_seeds, args.solver_chunk):
        pool = ctx_mp.Pool(1)
        try:
            async_result = pool.apply_async(_solver_trial_child, (chunk, args.mem_limit, args.solver_eval_n))
            results.extend(async_result.get(timeout=args.timeout * max(1, len(chunk))))
        except multiprocessing.TimeoutError:
            results.append(
                {
                    "trial_seed": chunk[0],
                    "classification": "error",
                    "detail": f"chunk timeout (seeds {chunk[0]}..{chunk[-1]})",
                    "constraints": [],
                }
            )
        except Exception as e:
            results.append(
                {
                    "trial_seed": chunk[0],
                    "classification": "error",
                    "detail": f"child crash (seeds {chunk[0]}..{chunk[-1]}): {e}",
                    "constraints": [],
                }
            )
        finally:
            pool.terminate()
            pool.join()
        print(f"  {len(results)}/{len(trial_seeds)} trials done", flush=True)
    elapsed = time.perf_counter() - t0

    counts: dict = {}
    for r in results:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    bad = [r for r in results if r["classification"] != "pass"]
    for r in bad:
        print("\n" + "-" * 60)
        print(f"{r['classification'].upper()} — trial_seed={r['trial_seed']}")
        print(f"  {r['detail']}")
        for c in r["constraints"]:
            print(f"    constraint: {c}")
        print(f"  reproduce: python {os.path.relpath(__file__)} --mode solver --solver-trial-seed {r['trial_seed']}")

    print("\n" + "=" * 60)
    print("Solver-differential summary")
    print("=" * 60)
    for k, v in sorted(counts.items()):
        print(f"  {k:14s}: {v}")
    print(f"  total elapsed : {elapsed:.2f}s")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"mode": "solver", "seed": seed, "counts": counts, "results": results}, f, indent=2)
        print(f"  report written to {args.report}")

    if bad:
        print(f"\nFAIL: {len(bad)} solver divergence(s)/error(s).")
        return 1
    print("\nOK: no solver divergences.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Property-based differential fuzzer for the Rust engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n")[-1] if "Usage" in (__doc__ or "") else "",
    )
    parser.add_argument(
        "--mode",
        choices=("corpus", "solver"),
        default="corpus",
        help=(
            "corpus: differential whole-binary exploration over angr-examples (default). "
            "solver: differential random-constraint solving, RustSolverContext vs claripy "
            "(sat/min/max/eval/eval_batch); needs no example corpus."
        ),
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: time-based)")
    parser.add_argument("--timeout", type=int, default=60, help="Per-engine timeout in seconds (default: 60)")
    parser.add_argument(
        "--mem-limit",
        type=int,
        default=DEFAULT_MEM_LIMIT_MB,
        help=f"Memory limit per subprocess in MB (default: {DEFAULT_MEM_LIMIT_MB})",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="EXAMPLE",
        help="Restrict sampling to a specific example (repeatable)",
    )
    parser.add_argument(
        "--include-bimodal", action="store_true", help="Include benchmarks with known bimodal Z3 timing variance"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero on any divergence (default: only nonzero on crashes)"
    )
    parser.add_argument("--report", default=None, metavar="PATH", help="Write a JSON report of all trials to PATH")
    parser.add_argument("--list", action="store_true", help="List the eligible example set and exit")
    parser.add_argument(
        "--solver-trial-seed",
        type=int,
        default=None,
        metavar="S",
        help="[--mode solver] Re-run exactly the one trial with this seed (standalone reproducer)",
    )
    parser.add_argument(
        "--solver-chunk",
        type=int,
        default=25,
        help="[--mode solver] Trials per subprocess (default: 25). Lower localizes a crashing trial.",
    )
    parser.add_argument(
        "--solver-eval-n",
        type=int,
        default=5,
        help="[--mode solver] Number of solutions requested per eval_upto/eval call (default: 5)",
    )
    args = parser.parse_args()

    if args.mode == "solver":
        return run_solver_mode(args)

    eligible = _eligible_examples(include_bimodal=args.include_bimodal)
    if args.only:
        bad = set(args.only) - set(eligible) - set(EXAMPLE_CATALOG)
        if bad:
            print(f"ERROR: unknown example(s): {', '.join(sorted(bad))}", file=sys.stderr)
            return 2
        # With --only the user has explicitly named the example, so honor
        # it even if not in the default-eligible set.
        eligible = list(args.only)

    if args.list:
        print(f"Eligible examples ({len(eligible)}):")
        for e in eligible:
            print(f"  {e}")
        return 0

    if not eligible:
        print("ERROR: no eligible examples found. Check ANGR_EXAMPLES_DIR.", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    print(
        f"property-fuzzer: seed={seed} trials={args.trials} "
        f"timeout={args.timeout}s mem-limit={args.mem_limit}MB "
        f"eligible={len(eligible)}"
    )

    report = FuzzerReport(seed=seed, trials=args.trials)
    total_start = time.perf_counter()
    crashes = 0

    for i in range(args.trials):
        example = rng.choice(eligible)
        strategy = rng.choice(_STRATEGIES)
        trial_seed = rng.randint(0, 2**31 - 1)
        report.examples_seen.append(example)

        print(f"[{i + 1}/{args.trials}] {example} ({strategy}) seed={trial_seed}", end=" ... ", flush=True)
        t0 = time.perf_counter()
        result = run_trial(i, example, strategy, trial_seed, args.timeout, args.mem_limit)
        dt = time.perf_counter() - t0
        report.results.append(result)

        if result.classification == "pass":
            print(f"PASS ({dt:.1f}s)")
        elif result.classification == "diverge":
            print(f"DIVERGE [{result.divergence_kind}] ({dt:.1f}s)")
            print(f"    python: {result.python_output_snippet!r}")
            print(f"    rust:   {result.rust_output_snippet!r}")
        elif result.classification == "expected-diverge":
            print(f"EXPECTED-DIVERGE [{result.divergence_kind}] ({dt:.1f}s) — known rust_only/bimodal entry")
        elif result.classification == "rust-fail":
            crashes += 1
            print(f"RUST FAIL ({dt:.1f}s): {result.rust_error}")
        elif result.classification == "py-fail":
            print(f"PY FAIL ({dt:.1f}s): {result.python_error}")
        elif result.classification == "both-fail":
            print(f"BOTH FAIL ({dt:.1f}s): py={result.python_error}, rust={result.rust_error}")
        else:
            print(f"? ({dt:.1f}s) {result.classification}")

    report.total_elapsed = time.perf_counter() - total_start
    summary = report.summary()
    print("\n" + "=" * 60)
    print("Fuzzer summary")
    print("=" * 60)
    for k, v in summary["counts"].items():
        print(f"  {k:14s}: {v}")
    print(f"  total elapsed : {summary['total_elapsed_sec']}s")
    print(f"  unique examples: {len(summary['examples_seen'])}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "results": [asdict(r) for r in report.results],
                },
                f,
                indent=2,
            )
        print(f"  report written to {args.report}")

    divergences = sum(1 for r in report.results if r.is_divergence())
    if crashes > 0:
        print(f"\nFAIL: {crashes} engine crash(es) — see RUST FAIL / BOTH FAIL above.")
        return 1
    if divergences > 0 and args.strict:
        print(f"\nFAIL (strict): {divergences} divergence(s).")
        return 1
    if divergences > 0:
        print(f"\nWARN: {divergences} divergence(s) (non-strict mode).")
    else:
        print("\nOK: no divergences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
