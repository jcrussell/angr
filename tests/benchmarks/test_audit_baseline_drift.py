"""Unit tests for ``tests/benchmarks/audit_baseline_drift.py``.

Pure-function tests only — ``parse_run``/``classify``/``apply_updates`` — so
they stay milliseconds-fast and never spawn the benchmark subprocess.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import audit_baseline_drift as abd

SAMPLE_OUTPUT = """\
--- fauxware ---
  Rust:   0.18s
--- defcamp_r100 ---
  Rust:   0.32s
--- defcamp_r100 (DFS) ---
  Rust:   0.34s
--- flareon2015_2 ---
  Rust:   3.38s
"""


def test_parse_run_maps_dfs_header_to_suffixed_key():
    timings = abd.parse_run(SAMPLE_OUTPUT)
    assert timings == {
        "fauxware": 0.18,
        "defcamp_r100": 0.32,
        "defcamp_r100__dfs": 0.34,
        "flareon2015_2": 3.38,
    }


def test_parse_run_ignores_trailing_header_without_timing():
    # A bench that times out prints its header but no "Rust:" line; the next
    # bench's timing must not be attributed to it.
    text = SAMPLE_OUTPUT + "--- timed_out_bench ---\n  TIMEOUT\n"
    timings = abd.parse_run(text)
    assert "timed_out_bench" not in timings
    assert timings["flareon2015_2"] == 3.38


def test_classify_bands_loose_ok_and_tight():
    baseline = {
        "loose_bench": {"rust_time": 1.0},
        "ok_bench": {"rust_time": 1.0},
        "tight_bench": {"rust_time": 1.0},
    }
    observations = {
        "loose_bench": [0.40, 0.50],  # max/base = 0.50
        "ok_bench": [0.90, 0.95],  # max/base = 0.95
        "tight_bench": [1.00, 1.20],  # max/base = 1.20
    }
    rows = {r["key"]: r for r in abd.classify(observations, baseline, 0.85, 1.05)}
    assert rows["loose_bench"]["verdict"] == "loose"
    assert rows["ok_bench"]["verdict"] == "ok"
    assert rows["tight_bench"]["verdict"] == "tight"
    # max-of-N, not median: the gate's headroom sits above the slowest run.
    assert rows["loose_bench"]["max"] == 0.50


def test_classify_flags_measured_bench_with_no_baseline():
    rows = abd.classify({"new_bench": [1.0]}, {}, 0.85, 1.05)
    assert rows[0]["verdict"] == "unbaselined"
    # rust_time=None (e.g. CADET_00001) is equally unbaselined, not a divide-by-zero.
    rows = abd.classify({"b": [1.0]}, {"b": {"rust_time": None}}, 0.85, 1.05)
    assert rows[0]["verdict"] == "unbaselined"


def test_apply_updates_dry_run_reports_without_writing(tmp_path, monkeypatch):
    path = tmp_path / "baseline_timings.json"
    original = {"z_bench": {"rust_time": 1.0}, "a_bench": {"rust_time": 1.0}}
    path.write_text(json.dumps(original, indent=2, sort_keys=False) + "\n")
    monkeypatch.setattr(abd, "BASELINE_PATH", path)

    rows = [{"key": "z_bench", "verdict": "loose", "max": 0.4}]
    assert abd.apply_updates(rows, dry_run=True) == ["z_bench"]
    assert json.loads(path.read_text())["z_bench"]["rust_time"] == 1.0

    assert abd.apply_updates(rows, dry_run=False) == ["z_bench"]
    written = json.loads(path.read_text())
    assert written["z_bench"]["rust_time"] == 0.4
    # Insertion order preserved -- a sort_keys=True round-trip would bury the
    # one-line change in a whole-file reorder diff.
    assert list(written) == ["z_bench", "a_bench"]
