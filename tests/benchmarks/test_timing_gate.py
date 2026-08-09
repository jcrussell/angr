"""Unit tests for the rust_time regression gate decision.

Exercises the pure helpers ``run_regression.timing_regression_pct``,
``suite_median_ratio`` and ``host_scale_factor`` against synthetic and
recorded-from-CI timings. Stays in-process and angr-free for the same reason
``test_memory_gate.py`` does: importing ``run_regression`` only pulls
``run_single``'s light top-level, so these are millisecond-fast and CI-safe.

Regression guard for angr-z8p3x: a relative-only threshold made the
sub-second benches (sharif7_rev50 at a 0.16s baseline → a ~24ms bar) fail the
gate on host noise, twice, retries included. The helper adds an absolute-delta
floor; these tests pin both bars and, just as importantly, pin that benches
above the floor keep their old behaviour.

Regression guard for angr-mc8pw: the floor covers a noisy sub-second bench but
not a uniformly slow *host*, which reddened the gate on cow_fork_scaling
(2.5s baseline, so the floor was never in play) with both retries also failing.
The second block of tests pins the suite-median normalization that covers it,
and — the part that matters — pins that it does not soften a real regression,
a fast host, or a slowdown past the cap.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_regression

FLOOR = run_regression.DEFAULT_REGRESSION_FLOOR_S


def test_regression_flagged_when_both_bars_cleared():
    # 12s vs 10s baseline: +20% > 15% and +2000ms > 50ms → regression.
    pct = run_regression.timing_regression_pct(10.0, 12.0, 0.15, FLOOR)
    assert pct is not None
    assert round(pct) == 20


def test_within_threshold_no_regression():
    # +10% is within a 15% threshold, absolute delta irrelevant.
    assert run_regression.timing_regression_pct(10.0, 11.0, 0.15, FLOOR) is None


def test_sharif_style_noise_excursion_suppressed():
    """The angr-z8p3x case: 0.16s baseline, 0.19s measurement.

    +19% clears the relative bar but the delta is 30ms — under the 50ms
    floor, so it must not fail the gate.
    """
    assert run_regression.timing_regression_pct(0.16, 0.19, 0.15, FLOOR) is None


def test_sub_second_bench_still_fails_on_a_real_regression():
    # Same 0.16s baseline, but now 2.5x slower: 240ms delta clears the floor.
    pct = run_regression.timing_regression_pct(0.16, 0.40, 0.15, FLOOR)
    assert pct is not None
    assert round(pct) == 150


def test_slow_bench_behaviour_unchanged_by_the_floor():
    """Above floor/threshold (~0.33s at the defaults) the floor is inert.

    Any bench whose relative bar alone already implies a >50ms delta must
    decide identically with and without the floor — otherwise the fix would
    have quietly loosened the gate for the benches that matter.
    """
    for baseline in (0.5, 1.0, 3.5, 22.0):
        for mult in (1.0, 1.14, 1.16, 1.5, 3.0):
            current = baseline * mult
            with_floor = run_regression.timing_regression_pct(baseline, current, 0.15, FLOOR)
            relative_only = run_regression.timing_regression_pct(baseline, current, 0.15, 0.0)
            assert (with_floor is None) == (relative_only is None), (baseline, mult)


def test_zero_floor_restores_relative_only_behaviour():
    pct = run_regression.timing_regression_pct(0.16, 0.19, 0.15, 0.0)
    assert pct is not None
    assert round(pct) == 19


def test_no_baseline_no_regression():
    assert run_regression.timing_regression_pct(None, 9.0, 0.15, FLOOR) is None


def test_nonpositive_baseline_no_regression():
    assert run_regression.timing_regression_pct(0.0, 9.0, 0.15, FLOOR) is None
    assert run_regression.timing_regression_pct(-1.0, 9.0, 0.15, FLOOR) is None


def test_missing_current_measurement_no_regression():
    assert run_regression.timing_regression_pct(10.0, None, 0.15, FLOOR) is None


def test_floor_defaults_to_the_module_constant():
    # Called without the floor argument, the helper must apply the same
    # default the CLI advertises — not silently fall back to 0.
    assert run_regression.timing_regression_pct(0.16, 0.19, 0.15) is None


# --- suite-wide host-slowdown normalization (angr-mc8pw) ---------------------
#
# Second failure mode of the same gate: not one noisy sub-second bench, but the
# whole box being slow for the duration of the run. Ratios below are the real
# per-bench current/baseline values from the ralph iter24 gate log.

CAP = run_regression.DEFAULT_HOST_FACTOR_CAP

# The 21 benches with a baseline in that run, worst-last.
ITER24_RATIOS = [
    0.878, 1.016, 1.044, 1.044, 1.050, 1.050, 1.053, 1.071, 1.075, 1.075,
    1.083, 1.085, 1.092, 1.108, 1.108, 1.121, 1.130, 1.166, 1.189, 1.199,
    1.236,
]  # fmt: skip


def test_suite_median_ratio_of_the_iter24_run():
    median = run_regression.suite_median_ratio(ITER24_RATIOS)
    assert round(median, 3) == 1.083


def test_iter24_cow_fork_scaling_clears_after_normalization():
    """The angr-mc8pw case: 2.50s baseline, 3.09s measurement.

    +24% clears both the relative bar and the 50ms floor, so the raw gate
    fails it. But every other bench in that run was slow too — dividing by
    the 1.083x suite median leaves +14%, under the 15% bar.
    """
    assert run_regression.timing_regression_pct(2.50, 3.09, 0.15, FLOOR) is not None
    factor = run_regression.host_scale_factor(run_regression.suite_median_ratio(ITER24_RATIOS), CAP)
    assert run_regression.timing_regression_pct(2.50 * factor, 3.09, 0.15, FLOOR) is None


def test_real_regression_on_a_quiet_host_still_fails():
    """The counter-case: one bench doubles while the rest sit at baseline.

    The median lands inside the dead-band, so `host_scale_factor` returns
    exactly 1.0 and the decision is byte-for-byte the unnormalized one.
    """
    quiet = [1.01, 0.99, 1.00, 1.02, 0.98, 2.00]
    factor = run_regression.host_scale_factor(run_regression.suite_median_ratio(quiet), CAP)
    assert factor == 1.0
    assert run_regression.timing_regression_pct(2.50 * factor, 5.00, 0.15, FLOOR) is not None


def test_uniform_slowdown_past_the_cap_is_not_normalized():
    # A 1.40x median is equally consistent with "every bench regressed", so
    # refusing to normalize is the honest answer — the failures must stand.
    hot = [1.38, 1.39, 1.40, 1.41, 1.42, 1.60]
    median = run_regression.suite_median_ratio(hot)
    assert median > CAP
    assert run_regression.host_scale_factor(median, CAP) == 1.0
    assert run_regression.timing_regression_pct(2.50, 4.00, 0.15, FLOOR) is not None


def test_fast_host_does_not_tighten_the_gate():
    # Median below 1.0 means the box was FASTER than baseline. Scaling
    # baselines down would invent failures, so the factor clamps to 1.0.
    fast = [0.80, 0.82, 0.85, 0.88, 0.90, 0.91]
    assert run_regression.host_scale_factor(run_regression.suite_median_ratio(fast), CAP) == 1.0


def test_too_few_samples_has_no_median():
    # A median over one or two benches is just the failing bench in disguise.
    assert run_regression.suite_median_ratio([1.2, 1.3]) is None
    assert run_regression.suite_median_ratio([]) is None
    assert run_regression.host_scale_factor(None, CAP) == 1.0


def test_unusable_ratios_are_dropped_before_the_median():
    # None / non-positive entries must not shift the median or count toward
    # the sample minimum.
    ratios = [None, 0.0, -1.0, 1.10, 1.10, 1.10, 1.10]
    assert run_regression.suite_median_ratio(ratios) is None
    assert round(run_regression.suite_median_ratio([*ratios, 1.10]), 3) == 1.10


def test_host_factor_cap_defaults_to_the_module_constant():
    # Called without the cap argument, the helper must apply the same default
    # the CLI advertises — not an unbounded normalization.
    assert run_regression.host_scale_factor(1.40) == 1.0
    assert run_regression.host_scale_factor(1.10) == 1.10


def test_dead_band_keeps_a_quiet_run_byte_for_byte_unnormalized():
    # Just inside the dead-band: no normalization at all.
    assert run_regression.host_scale_factor(run_regression.HOST_FACTOR_DEAD_BAND - 0.001, CAP) == 1.0
    # Just outside it: normalization engages at exactly the median.
    edge = run_regression.HOST_FACTOR_DEAD_BAND
    assert run_regression.host_scale_factor(edge, CAP) == edge


# --- pinned baselines (angr-x6t9o) ------------------------------------------
#
# The floor and the host normalization both assume the *baseline* is honest.
# cow_fork_scaling's was not: REGRESSION_SUITE documented a deliberate
# slow-mode pin, but two blanket ``--update`` refreshes overwrote it with
# fast-mode measurements (2.559, then 2.5) and nothing warned. These pin that
# ``--update`` can no longer do that silently.


def test_update_keeps_a_pinned_rust_time_and_reports_the_measurement():
    results = {"cow_fork_scaling": {"rust_time": 2.5, "steps": 513}}
    held = run_regression.apply_pinned_baselines(results, {"cow_fork_scaling": 2.8})
    assert results["cow_fork_scaling"]["rust_time"] == 2.8
    assert held == [("cow_fork_scaling", 2.5, 2.8)]
    # Everything else in the record is still refreshed as usual.
    assert results["cow_fork_scaling"]["steps"] == 513


def test_pin_is_kept_even_when_the_measurement_is_slower():
    # A pin is a floor *and* a ceiling: a single slow read must not ratchet the
    # baseline up either, or the gate loosens itself one refresh at a time.
    results = {"cow_fork_scaling": {"rust_time": 3.4}}
    run_regression.apply_pinned_baselines(results, {"cow_fork_scaling": 2.8})
    assert results["cow_fork_scaling"]["rust_time"] == 2.8


def test_unpinned_benches_are_refreshed_untouched():
    results = {"fauxware": {"rust_time": 0.19}, "cow_fork_scaling": {"rust_time": 2.5}}
    run_regression.apply_pinned_baselines(results, {"cow_fork_scaling": 2.8})
    assert results["fauxware"]["rust_time"] == 0.19


def test_pin_matching_the_measurement_reports_nothing():
    results = {"cow_fork_scaling": {"rust_time": 2.8}}
    assert run_regression.apply_pinned_baselines(results, {"cow_fork_scaling": 2.8}) == []


def test_pin_for_a_bench_absent_from_the_run_is_skipped():
    # ``--rust-only``/``--skip-bimodal``/fast-tier runs do not measure every
    # pinned bench; a missing one must not be invented into the results dict.
    results = {"fauxware": {"rust_time": 0.19}}
    assert run_regression.apply_pinned_baselines(results, {"cow_fork_scaling": 2.8}) == []
    assert "cow_fork_scaling" not in results


def test_live_pin_matches_the_checked_in_baseline():
    # The whole point of the table is that the JSON agrees with it. If someone
    # hand-edits baseline_timings.json, this catches the drift.
    baseline = run_regression.load_baseline()
    for name, pin in run_regression.PINNED_RUST_TIMES.items():
        assert baseline[name]["rust_time"] == pin, f"{name} baseline drifted from its pin"


def test_live_pin_clears_every_in_gate_sample_recorded_on_the_bead():
    # The 12 ralph gate reads that motivated angr-x6t9o. With the old 2.5
    # baseline most of these were at or over the 15% bar; with the pin none of
    # them is a failure. Guards against a future refresh quietly re-tightening
    # the pin below the load-sensitive range.
    in_gate_reads = [2.90, 3.09, 2.89, 3.05, 2.98, 2.96, 2.94, 3.11, 2.97, 2.94, 3.14, 3.09]
    pin = run_regression.PINNED_RUST_TIMES["cow_fork_scaling"]
    for read in in_gate_reads:
        assert run_regression.timing_regression_pct(pin, read, 0.15, FLOOR) is None, read
    # ...and the pin still fails a genuine regression: fast mode plus 50%.
    assert run_regression.timing_regression_pct(pin, 3.3, 0.15, FLOOR) is not None
