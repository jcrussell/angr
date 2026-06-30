#!/usr/bin/env python3
"""Parallel run-loop overhead GO/NO-GO gate (SI-C, angr-1ilq.3 increment 2b').

SI-B added an opt-in "shadow probe" (``RUST_PARALLEL_SHADOW_PROBE=1``) that
round-trips every dispatched state through ``to_serialized`` +
``from_serialized`` *in a foreign Z3 context* and accumulates the REAL
migration serde tax into three counters:

  * ``parallel_shadow_migration_ns``    — total detach/serialize +
    deserialize/reattach time over all dispatched states (a LOWER bound on
    the real steal tax; channel transit and scheduler bookkeeping are not
    counted).
  * ``parallel_shadow_migration_states`` — number of states probed.
  * ``parallel_shadow_migration_bytes``  — total serialized payload bytes.

This harness drives each target bench through ``run_single.py
--counters-json`` with the probe enabled, then projects whether a
state-level worker pool at ``--workers`` would actually beat the serial
run, using a deliberately *optimistic* parallelism model plus the
*measured* migration tax.  It prints every intermediate term and a
GO / NO-GO / NO-GO-FOR-NOW verdict per bench.

Model (per bench)
-----------------
  T_s              CLEAN serial wall time, seconds (run_wall_time_ns / 1e9)
                   from a PROBE-OFF run.  The probe is SYNCHRONOUS (it blocks
                   the run loop per dispatched state on the scratch-thread
                   round-trip), so a probe-ON run's wall is inflated by
                   ~O_migrate and must NOT be used as T_s.  Each bench is run
                   TWICE: probe-OFF for T_s + exploration terms, probe-ON for
                   the migration counters only.
  O_migrate        parallel_shadow_migration_ns / 1e9 from the PROBE-ON run.
                   REAL, measured.  A LOWER bound on the steal tax.
  P_eff (modeled)  step-weighted effective parallelism, optimistic.  For
                   each of the 5 width buckets [w==1, ==2, 3-4, 5-8, >=9]
                   we credit ``min(lower_edge_width, workers)`` where the
                   lower-edge representative widths are [1,2,3,5,9] (lower
                   edge keeps the credit conservative-optimistic).
                     (a) credits all steps' width.
                     (b) charges the ``callback_count`` GIL-bound steps a
                         flat parallelism of 1 (they serialize), crediting
                         the rest the (a) average.  (b) is the gate input.
  T_serial_bounce  wall attributable to GIL-bound callback steps,
                   approximated as ``T_s * callback_count / total_steps``
                   (total_steps == sum of the width histogram == ``steps``).
  O_cache          BAND.  optimistic = 0 (a shared C2 block cache makes
                   per-wave misses ~= the serial baseline).  pessimistic =
                   extra re-lift cost assuming ZERO cross-worker cache
                   sharing: ``(workers-1) * block_cache_misses *
                   mean_lift_s`` where ``mean_lift_s`` is DERIVED from the
                   measured Python lift callback
                   (callback_lift_block_total_ns / callback_lift_block_count
                   — a real per-lift timer, not a fabricated constant).
                   If that counter is absent we do NOT invent a per-lift
                   number: we report the raw miss count and the workers x
                   misses re-lift multiplier and fold the risk into the
                   verdict qualitatively.

  T_par   = (T_s - T_serial_bounce) / P_eff_b + O_migrate + O_cache
            + T_serial_bounce
  speedup = T_s / T_par           (computed at BOTH O_cache band ends)

Verdict
-------
  GO              speedup > threshold under BOTH band ends.
  NO-GO           speedup <= threshold even under the optimistic band.
  NO-GO-FOR-NOW   optimistic clears the bar but the pessimistic band end
                  drops to/under it (band straddles the threshold).

Usage
-----
    python tests/benchmarks/run_parallel_overhead_gate.py --workers 2
    python tests/benchmarks/run_parallel_overhead_gate.py --only codegate_2017-angrybird
    python tests/benchmarks/run_parallel_overhead_gate.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SINGLE = HERE / "run_single.py"

DEFAULT_BENCHES = ["cmu_binary_bomb_partial", "codegate_2017-angrybird"]

# Lower-edge representative widths for the 5 histogram buckets
# [w==1, w==2, w in 3-4, w in 5-8, w>=9].  Lower edge keeps the parallel
# credit conservative-optimistic (we never credit more width than the bucket
# guarantees).
REP_WIDTHS = [1, 2, 3, 5, 9]

# The three SI-B shadow-probe counters that MUST be present (and states>0)
# after a probed run — a missing/zero key means the probe did not fire and any
# verdict would be silently bogus.  We raise instead of returning None.
REQUIRED_KEYS = (
    "parallel_shadow_migration_ns",
    "parallel_shadow_migration_states",
    "parallel_shadow_migration_bytes",
)


class GateError(RuntimeError):
    """Raised loudly when probe counters are missing/zero — never skip silently."""


def assert_probe_counters(name: str, stats: dict) -> None:
    """Fail loudly if the SI-B shadow-probe counters are absent or zero.

    This is the anti-silent-skip guard.  run_width_audit.py returns ``None``
    on a missing histogram key and the caller just prints ``[skip]`` — which
    would hide a probe that never fired.  Here a missing key (or
    ``parallel_shadow_migration_states == 0``) is a hard error: the projection
    is meaningless without the measured migration tax.
    """
    missing = [k for k in REQUIRED_KEYS if k not in stats]
    if missing:
        raise GateError(
            f"{name}: shadow-probe counter(s) missing from stats: {missing}. "
            f"Was RUST_PARALLEL_SHADOW_PROBE=1 set, and is this an SI-B build? "
            f"(present parallel_* keys: "
            f"{sorted(k for k in stats if k.startswith('parallel_'))})"
        )
    states = stats.get("parallel_shadow_migration_states", 0)
    if not states:
        raise GateError(
            f"{name}: parallel_shadow_migration_states == {states!r} (expected > 0). "
            f"The shadow probe never fired — no state was dispatched, or the "
            f"probe is compiled out (vex-engine-z3 feature off)."
        )


def compute_gate(
    name: str,
    stats: dict,
    workers: int,
    threshold: float,
    steal_fraction: float | None = None,
) -> dict:
    """Project the parallel speedup band for one bench from its stats dict.

    Raises GateError (via assert_probe_counters) if the probe counters are
    missing or zero.

    ``steal_fraction`` (angr-t3l5o pivot): the shadow probe measures a
    level-synchronous wave model that migrates EVERY dispatched state, so the
    raw ``O_migrate`` is the worst case.  A real steal-on-imbalance scheduler
    migrates only a fraction ``f`` of states (an actual cross-worker steal);
    the effective tax is ``f * O_migrate``.  ``None`` (default) reproduces the
    wave-model worst case (f = 1.0).  The break-even ``f*`` is always reported
    regardless of this knob.
    """
    assert_probe_counters(name, stats)

    # T_s is the CLEAN serial wall time from a PROBE-OFF run.  The shadow probe
    # is synchronous — it blocks the run loop on every dispatched state waiting
    # for the scratch-thread round-trip — so a probe-ON run's run_wall_time_ns
    # is inflated by ~O_migrate and MUST NOT be used as T_s.  ``stats`` here is
    # the merged dict: run_wall_time_ns from the probe-off run, migration
    # counters from the probe-on run.  ``_probe_on_wall_ns`` carries the
    # inflated probe-on wall for visibility (never fed into the projection).
    t_s = stats["run_wall_time_ns"] / 1e9
    probe_on_wall_ns = stats.get("_probe_on_wall_ns")
    probe_on_wall_s = probe_on_wall_ns / 1e9 if probe_on_wall_ns else None

    mig_ns = stats["parallel_shadow_migration_ns"]
    mig_states = stats["parallel_shadow_migration_states"]
    mig_bytes = stats["parallel_shadow_migration_bytes"]
    o_migrate = mig_ns / 1e9  # LOWER bound on the real steal tax
    per_state_ns = mig_ns / mig_states
    per_state_bytes = mig_bytes / mig_states

    hist = stats.get("parallel_width_hist") or [0, 0, 0, 0, 0]
    total_steps = sum(hist)
    # Prefer the histogram sum for the step total (it is exactly the count of
    # dispatched steps the width model is built on); fall back to ``steps``.
    if total_steps == 0:
        total_steps = stats.get("steps", 0)

    callback_count = stats.get("callback_count", 0)

    # P_eff (a): credit every step its bucket's lower-edge width, capped at
    # the worker count.
    if total_steps > 0 and sum(hist) > 0:
        credited = sum(c * min(rep, workers) for c, rep in zip(hist, REP_WIDTHS))
        p_eff_a = credited / sum(hist)
    else:
        p_eff_a = 1.0

    # P_eff (b): the callback_count GIL-bound steps cannot parallelize — charge
    # them a flat parallelism of 1 and credit the remaining steps the (a)
    # average.  cb is clamped to total_steps.  This is the gate input.
    cb = min(callback_count, total_steps) if total_steps > 0 else callback_count
    p_eff_b = ((total_steps - cb) * p_eff_a + cb * 1.0) / total_steps if total_steps > 0 else p_eff_a
    p_eff_b = max(p_eff_b, 1.0)  # parallelism can never drop below serial

    # T_serial_bounce: fraction of wall attributable to GIL-bound steps.
    # Approximation: assume callback steps cost the average per-step wall time,
    # so their share of wall == their share of steps.
    t_serial_bounce = t_s * (cb / total_steps) if total_steps > 0 else 0.0

    # O_cache band.  Optimistic = 0 (shared C2 cache -> per-wave misses ~=
    # serial).  Pessimistic = extra re-lift cost with ZERO cross-worker
    # sharing.  We do NOT have a direct Rust-side per-lift timer; if the
    # measured Python lift callback is available we DERIVE a mean per-lift cost
    # from it (real measurement, not a fabricated constant), else we report the
    # multiplier qualitatively.
    misses = stats.get("block_cache_misses", 0)
    lift_total_ns = stats.get("callback_lift_block_total_ns", 0)
    lift_count = stats.get("callback_lift_block_count", 0)
    relift_multiplier = workers * misses  # worst-case total lifts vs serial misses
    extra_relifts = max(workers - 1, 0) * misses  # extra lifts beyond serial baseline
    if lift_count > 0:
        mean_lift_s = (lift_total_ns / lift_count) / 1e9
        o_cache_pess = extra_relifts * mean_lift_s
        o_cache_pess_derived = True
    else:
        mean_lift_s = None
        o_cache_pess = 0.0  # cannot quantify without a per-lift timer
        o_cache_pess_derived = False

    # angr-t3l5o pivot: charge only the migrated fraction of states.  Default
    # f = 1.0 reproduces the wave-model worst case (every state migrates).
    steal_f = 1.0 if steal_fraction is None else steal_fraction
    eff_o_migrate = o_migrate * steal_f

    def project(o_cache: float) -> tuple[float, float]:
        t_par = (t_s - t_serial_bounce) / p_eff_b + eff_o_migrate + o_cache + t_serial_bounce
        speedup = t_s / t_par if t_par > 0 else 0.0
        return t_par, speedup

    t_par_opt, speedup_opt = project(0.0)
    t_par_pess, speedup_pess = project(o_cache_pess)

    # Break-even steal fraction f*: the largest fraction of states a scheduler
    # may migrate and still clear the threshold, per O_cache band.  Solve
    #   (t_s - bounce)/p_eff_b + f* * o_migrate + o_cache + bounce == t_s/threshold
    # for f*.  f* >= 1 => GO even migrating every state (the wave model already
    # passes); f* <= 0 => the non-migration terms alone miss the bar (P_eff too
    # low / cache tax too high) so no migration-count reduction can help.  This
    # is the headline pivot number: it converts "transport is too expensive"
    # into a concrete scheduler target ("keep steals below f* of states").
    def break_even(o_cache: float) -> float:
        if o_migrate <= 0:
            return float("inf")
        budget = t_s / threshold - t_serial_bounce - (t_s - t_serial_bounce) / p_eff_b - o_cache
        return budget / o_migrate

    steal_be_opt = break_even(0.0)
    steal_be_pess = break_even(o_cache_pess)

    # Verdict.  GO needs BOTH ends above threshold.  NO-GO if even the
    # optimistic end fails.  Otherwise the band straddles -> NO-GO-FOR-NOW.
    pess_quantified = o_cache_pess_derived or o_cache_pess == 0.0
    if speedup_opt <= threshold:
        verdict = "NO-GO"
    elif speedup_pess > threshold:
        verdict = "GO"
    else:
        verdict = "NO-GO-FOR-NOW (band straddles threshold)"
    # If the pessimistic end could not be quantified (no lift timer) but the
    # optimistic end clears the bar, downgrade GO -> NO-GO-FOR-NOW: we cannot
    # assert the cache tax is harmless.
    if verdict == "GO" and not o_cache_pess_derived and extra_relifts > 0:
        verdict = "NO-GO-FOR-NOW (cache tax unquantified; lift timer absent)"

    return {
        "name": name,
        "workers": workers,
        "threshold": threshold,
        "t_s": t_s,
        "probe_on_wall_s": probe_on_wall_s,
        "o_migrate": o_migrate,
        "mig_states": mig_states,
        "per_state_ns": per_state_ns,
        "per_state_bytes": per_state_bytes,
        "hist": hist,
        "total_steps": total_steps,
        "callback_count": callback_count,
        "p_eff_a": p_eff_a,
        "p_eff_b": p_eff_b,
        "t_serial_bounce": t_serial_bounce,
        "block_cache_misses": misses,
        "relift_multiplier": relift_multiplier,
        "extra_relifts": extra_relifts,
        "mean_lift_s": mean_lift_s,
        "o_cache_pess": o_cache_pess,
        "o_cache_pess_derived": o_cache_pess_derived,
        "t_par_opt": t_par_opt,
        "speedup_opt": speedup_opt,
        "t_par_pess": t_par_pess,
        "speedup_pess": speedup_pess,
        "pess_quantified": pess_quantified,
        "steal_fraction": steal_f,
        "eff_o_migrate": eff_o_migrate,
        "steal_be_opt": steal_be_opt,
        "steal_be_pess": steal_be_pess,
        "verdict": verdict,
    }


def run_one(name: str, timeout: int, probe: bool) -> dict:
    """Run a bench via run_single --counters-json once; return the stats dict.

    ``probe`` toggles ``RUST_PARALLEL_SHADOW_PROBE`` in the child env.  Raises
    GateError on a failed/empty run (no silent None — a bench that won't run is
    a hard error to surface).
    """
    env = dict(os.environ)
    env.setdefault(
        "ANGR_EXAMPLES_DIR",
        os.path.expanduser("~/repos/angr-examples/examples"),
    )
    if probe:
        env["RUST_PARALLEL_SHADOW_PROBE"] = "1"
    else:
        # Ensure an inherited probe flag from the parent shell can't pollute the
        # clean (probe-off) wall-time run.
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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60, env=env)
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"{name}: run_single timed out after {timeout + 60}s") from exc

    out = proc.stdout
    first_line = out.splitlines()[0] if out.splitlines() else ""
    # run_single emits the ``--counters-json`` blob as the TRAILING JSON object
    # (printed by ``json.dumps(indent=2)`` on its own line at end-of-stdout).
    # We locate it via the last ``\n{`` rather than the first ``{`` because the
    # solve.py output-preview lines (``  > ...``) can themselves contain braces
    # (e.g. a recovered flag) — bench_diff.load_counters' first-``{`` heuristic
    # would grab those.  Only the top-level object opens at column 0 after a
    # newline, so rfind("\n{") is unambiguous.
    brace = out.rfind("\n{")
    if brace == -1 and out.startswith("{"):
        brace = 0
    if brace == -1:
        raise GateError(
            f"{name}: no trailing JSON block in run_single output (run failed?).\n"
            f"  first stdout line: {first_line!r}\n"
            f"  stderr tail: {proc.stderr[-500:]!r}"
        )
    try:
        stats = json.loads(out[brace:])
    except json.JSONDecodeError as exc:
        raise GateError(
            f"{name}: could not parse counters JSON from run_single output.\n"
            f"  first stdout line: {first_line!r}\n"
            f"  stderr tail: {proc.stderr[-500:]!r}"
        ) from exc
    if not isinstance(stats, dict) or not stats:
        raise GateError(f"{name}: empty/invalid stats dict from run_single ({first_line!r}).")
    return stats


def gather_stats(name: str, timeout: int) -> dict:
    """Run a bench TWICE and merge into the dict compute_gate consumes.

    The shadow probe is synchronous (it blocks the run loop per dispatched
    state on the scratch-thread round-trip), so a probe-ON run's
    ``run_wall_time_ns`` is inflated by ~O_migrate.  We therefore:

      * PROBE-OFF run -> clean ``run_wall_time_ns`` (T_s) plus every
        exploration-determined term (width_hist, block_cache_misses,
        callback_count, steps, the lift timer).  The probe does not change
        exploration, but the wall time must be clean and pulling the rest from
        the same run keeps the terms mutually consistent.
      * PROBE-ON run -> ONLY the ``parallel_shadow_migration_*`` counters.

    The merged dict is the probe-off stats with the three migration counters
    overlaid from the probe-on run, plus ``_probe_on_wall_ns`` (the inflated
    probe-on wall, kept for visibility — never fed into the projection).
    """
    clean = run_one(name, timeout, probe=False)
    probe = run_one(name, timeout, probe=True)
    # Validate the PROBE run specifically — its migration counters must be
    # present and nonzero.  (A probe-off run also emits these keys but as 0, so
    # the assert has to see the probe-on dict to be meaningful.)
    assert_probe_counters(name, probe)
    merged = dict(clean)
    for k in REQUIRED_KEYS:
        merged[k] = probe[k]
    merged["_probe_on_wall_ns"] = probe.get("run_wall_time_ns", 0)
    return merged


def print_report(g: dict) -> None:
    """Render one bench's gate terms + verdict as a readable table."""
    w = g["workers"]
    print(f"\n=== {g['name']}  (workers={w}, threshold={g['threshold']:.2f}) ===")
    probe_on = f"{g['probe_on_wall_s']:.4f} s" if g.get("probe_on_wall_s") is not None else "n/a"
    rows = [
        ("T_s (CLEAN serial wall, probe-OFF run)", f"{g['t_s']:.4f} s"),
        ("probe-ON wall (INFLATED, not used for T_s)", probe_on),
        ("O_migrate (REAL total, LOWER bound)", f"{g['o_migrate']:.6f} s  over {g['mig_states']} states"),
        ("  per-state migration", f"{g['per_state_ns']:.0f} ns  /  {g['per_state_bytes']:.0f} bytes"),
        ("width hist [1,2,3-4,5-8,9+]", f"{g['hist']}"),
        ("total_steps", f"{g['total_steps']}"),
        ("callback_count (GIL-bound steps)", f"{g['callback_count']}"),
        ("P_eff (a) credit-all", f"{g['p_eff_a']:.4f}"),
        ("P_eff (b) charge-callbacks=1  [GATE]", f"{g['p_eff_b']:.4f}"),
        ("T_serial_bounce", f"{g['t_serial_bounce']:.4f} s"),
        ("block_cache_misses", f"{g['block_cache_misses']}"),
        ("  re-lift multiplier (workers x misses)", f"{g['relift_multiplier']}"),
    ]
    if g["mean_lift_s"] is not None:
        rows.append(("  mean lift (DERIVED, measured)", f"{g['mean_lift_s'] * 1e3:.4f} ms/lift"))
        rows.append(
            (
                "O_cache pessimistic (extra re-lifts)",
                f"{g['o_cache_pess']:.6f} s  = {g['extra_relifts']} x {g['mean_lift_s'] * 1e3:.4f}ms",
            )
        )
    else:
        rows.append(
            (
                "O_cache pessimistic",
                "UNQUANTIFIED — no per-lift timer; fold workers x misses qualitatively",
            )
        )
    rows.append(("O_cache optimistic", "0 s  (shared C2 cache)"))
    if g["steal_fraction"] != 1.0:
        rows.append(
            (
                f"O_migrate @ steal_fraction={g['steal_fraction']:.3f}",
                f"{g['eff_o_migrate']:.6f} s  (vs {g['o_migrate']:.6f} s wave-model)",
            )
        )
    rows.append(("T_par optimistic / pessimistic", f"{g['t_par_opt']:.4f} s  /  {g['t_par_pess']:.4f} s"))
    rows.append(
        (
            "speedup band [pess .. opt]",
            f"{g['speedup_pess']:.3f}x .. {g['speedup_opt']:.3f}x",
        )
    )

    def _fmt_fstar(f: float) -> str:
        if f == float("inf"):
            return "inf (no migration tax)"
        if f >= 1.0:
            return f"{f * 100:.1f}% (GO even at full wave-model migration)"
        if f <= 0.0:
            return f"{f * 100:.1f}% (unreachable — P_eff/cache miss the bar alone)"
        return f"{f * 100:.1f}%"

    rows.append(
        (
            "break-even steal fraction f* [pess .. opt]",
            f"{_fmt_fstar(g['steal_be_pess'])} .. {_fmt_fstar(g['steal_be_opt'])}",
        )
    )
    rows.append(
        (
            "  (max migrated states for GO)",
            "a steal-on-imbalance scheduler must migrate <= f* of dispatched states",
        )
    )
    keyw = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{keyw}}  {v}")
    print(f"  VERDICT: {g['verdict']}")


def _selftest() -> int:
    """Negative test: the loud-assert guard must raise on a renamed/missing key.

    Feeds compute_gate a stats dict with one required counter renamed and a
    second dict with states==0, and confirms GateError is raised both times.
    Also confirms a well-formed dict computes a verdict without raising.
    """
    # Mimics the merged dict gather_stats() builds: run_wall_time_ns is the
    # CLEAN probe-off wall, _probe_on_wall_ns the inflated probe-on wall (== clean
    # + O_migrate here), migration counters from the probe-on run.
    good = {
        "run_wall_time_ns": 2_000_000_000,
        "_probe_on_wall_ns": 2_005_000_000,
        "parallel_shadow_migration_ns": 5_000_000,
        "parallel_shadow_migration_states": 100,
        "parallel_shadow_migration_bytes": 200_000,
        "parallel_width_hist": [60, 20, 15, 4, 1],
        "steps": 100,
        "callback_count": 2,
        "block_cache_misses": 50,
        "callback_lift_block_total_ns": 50_000_000,
        "callback_lift_block_count": 50,
    }
    failures = []

    # 1) Renamed key -> must raise.
    renamed = dict(good)
    renamed["parallel_shadow_migration_NANOS"] = renamed.pop("parallel_shadow_migration_ns")
    try:
        compute_gate("renamed", renamed, workers=2, threshold=1.0)
        failures.append("renamed-key dict did NOT raise (silent skip bug!)")
    except GateError as e:
        print(f"[selftest] OK renamed key raised: {e}")

    # 2) Zero states -> must raise.
    zeroed = dict(good)
    zeroed["parallel_shadow_migration_states"] = 0
    try:
        compute_gate("zeroed", zeroed, workers=2, threshold=1.0)
        failures.append("states==0 dict did NOT raise")
    except GateError as e:
        print(f"[selftest] OK states==0 raised: {e}")

    # 3) Well-formed dict -> must NOT raise, must produce a verdict.
    try:
        g = compute_gate("good", good, workers=2, threshold=1.0)
        assert g["verdict"], "empty verdict"
        print(
            f"[selftest] OK well-formed dict -> verdict={g['verdict']!r} "
            f"speedup={g['speedup_pess']:.3f}..{g['speedup_opt']:.3f}x"
        )
    except Exception as e:
        failures.append(f"well-formed dict raised unexpectedly: {e}")

    if failures:
        for f in failures:
            print(f"[selftest] FAIL {f}")
        return 1
    print("[selftest] all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--workers", type=int, default=2, help="Worker count to project for (default 2, the ship-gate target)"
    )
    ap.add_argument("--only", nargs="*", help="Restrict to these bench names (default: cmu + codegate)")
    ap.add_argument("--threshold", type=float, default=1.0, help="GO requires projected speedup > this (default 1.0)")
    ap.add_argument("--timeout", type=int, default=180, help="Per-bench run_single timeout, seconds (default 180)")
    ap.add_argument(
        "--steal-fraction",
        type=float,
        default=None,
        help="angr-t3l5o pivot: fraction of dispatched states a steal-on-imbalance "
        "scheduler actually migrates (0..1). Scales O_migrate. Default (unset) = 1.0, "
        "the wave-model worst case. The break-even f* is always reported regardless.",
    )
    ap.add_argument("--selftest", action="store_true", help="Run the loud-assert negative test and exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    benches = args.only or DEFAULT_BENCHES
    results = []
    errors = []
    for name in benches:
        print(
            f"[run] {name} (probe-OFF for clean T_s + probe-ON for migration, workers={args.workers}) ...",
            file=sys.stderr,
            flush=True,
        )
        try:
            stats = gather_stats(name, args.timeout)
            g = compute_gate(name, stats, args.workers, args.threshold, args.steal_fraction)
        except GateError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            errors.append((name, str(e)))
            continue
        results.append(g)
        print_report(g)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for g in results:
        print(f"  {g['name']:<28} {g['verdict']:<45} speedup {g['speedup_pess']:.3f}x .. {g['speedup_opt']:.3f}x")
    for name, msg in errors:
        print(f"  {name:<28} ERROR: {msg.splitlines()[0]}")

    # Exit nonzero if any bench errored out (anti-silent-skip), else 0.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
