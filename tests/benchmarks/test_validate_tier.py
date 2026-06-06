"""Unit tests for ``tests/benchmarks/validate_tier.py``.

These tests stay in-process — they only exercise pure-Python helpers
(``classify``, ``within_boundary``, ``audit``) against synthetic catalog
and baseline dicts, so they are millisecond-fast and CI-safe.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import validate_tier  # noqa: E402


def test_classify_buckets():
    assert validate_tier.classify(0.0) == "fast"
    assert validate_tier.classify(4.99) == "fast"
    assert validate_tier.classify(5.0) == "medium"
    assert validate_tier.classify(29.99) == "medium"
    assert validate_tier.classify(30.0) == "slow"
    assert validate_tier.classify(119.99) == "slow"
    assert validate_tier.classify(120.0) == "very_slow"
    assert validate_tier.classify(1e6) == "very_slow"


def test_within_boundary_5s_fast_medium():
    # 10% slack on the 5s boundary: 4.5 - 5.5 is tolerated either way.
    assert validate_tier.within_boundary(5.35, "fast", 0.10)
    assert validate_tier.within_boundary(5.35, "medium", 0.10)
    assert validate_tier.within_boundary(4.6, "fast", 0.10)
    # 5.6s is 12% past, beyond ±10%
    assert not validate_tier.within_boundary(5.6, "fast", 0.10)


def test_within_boundary_fast_zero_lower_bound():
    # fast's lower bound is 0 — the slack check must not divide by zero.
    # 0.5s is mid-bucket (not near 5s), so the answer is False, but the
    # important property is that it returns cleanly rather than crashing.
    assert validate_tier.within_boundary(0.5, "fast", 0.10) is False


def test_within_boundary_very_slow_unbounded():
    # very_slow has hi=inf; only the 120s lower boundary should slack.
    assert validate_tier.within_boundary(125.0, "very_slow", 0.10)
    assert not validate_tier.within_boundary(200.0, "very_slow", 0.10)


def test_audit_picks_rust_time_when_rust_ok_true():
    catalog = {"a": {"tier": "fast", "rust_ok": True}}
    baseline = {"a": {"rust_time": 0.5, "python_time": 99.0}}
    drift, tolerated, skipped = validate_tier.audit(catalog, baseline)
    assert (drift, tolerated, skipped) == ([], [], [])


def test_audit_picks_python_time_when_rust_ok_none():
    # Catalog says rust_ok=None — classifier should use python_time, ignoring rust_time.
    catalog = {"a": {"tier": "very_slow", "rust_ok": None}}
    baseline = {"a": {"rust_time": 0.1, "python_time": 200.0}}
    drift, tolerated, skipped = validate_tier.audit(catalog, baseline)
    assert (drift, tolerated, skipped) == ([], [], [])


def test_audit_skips_when_time_missing():
    catalog = {"a": {"tier": "fast", "rust_ok": True}}
    baseline = {"a": {"rust_time": None, "python_time": None}}
    drift, tolerated, skipped = validate_tier.audit(catalog, baseline)
    assert drift == []
    assert tolerated == []
    assert [r[0] for r in skipped] == ["a"]


def test_audit_flags_drift_beyond_tolerance():
    catalog = {
        "boundary": {"tier": "fast", "rust_ok": True},   # tolerated
        "clear_drift": {"tier": "fast", "rust_ok": True},  # not tolerated
    }
    baseline = {
        "boundary":    {"rust_time": 5.2, "python_time": None},
        "clear_drift": {"rust_time": 15.0, "python_time": None},
    }
    drift, tolerated, _ = validate_tier.audit(catalog, baseline, tol=0.10)
    assert [r[0] for r in drift] == ["clear_drift"]
    assert [r[0] for r in tolerated] == ["boundary"]
    # Source field should be "rust" for both since rust_ok=True
    assert all(r[4] == "rust" for r in drift + tolerated)


def test_audit_source_field_for_python_only_bench():
    catalog = {"pyonly": {"tier": "slow", "rust_ok": None}}
    baseline = {"pyonly": {"rust_time": None, "python_time": 60.0}}
    drift, tolerated, _ = validate_tier.audit(catalog, baseline)
    assert drift == [] and tolerated == []


def test_audit_real_catalog_smoke():
    """Spot-check: the real EXAMPLE_CATALOG + baseline produces zero
    out-of-tolerance drift today. Acts as a regression gate against future
    catalog edits that drift past the boundary slack."""
    import json
    from run_single import EXAMPLE_CATALOG

    baseline_path = os.path.join(os.path.dirname(__file__), "baseline_timings.json")
    with open(baseline_path) as fh:
        baseline = json.load(fh)
    drift, _, _ = validate_tier.audit(EXAMPLE_CATALOG, baseline)
    assert drift == [], f"Unexpected tier drift: {drift}"
