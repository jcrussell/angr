"""Unit tests for ``tests/benchmarks/bench_diff.py``.

These tests intentionally avoid touching the benchmark subprocess
machinery — they only exercise ``compute_diff``/``_flatten``/
``format_report`` so they stay milliseconds-fast and CI-safe.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import bench_diff


def test_flatten_expands_nested_dict_counter():
    stats = {
        "callback_count": 10,
        "simprocedure_fallback_by_name": {"strcmp": 3, "memcpy": 1},
    }
    flat = bench_diff._flatten(stats)
    assert flat == {
        "callback_count": 10,
        "simprocedure_fallback_by_name.strcmp": 3,
        "simprocedure_fallback_by_name.memcpy": 1,
    }


def test_flatten_collapses_bool_to_int():
    flat = bench_diff._flatten({"a_flag": True, "b_flag": False, "nested": {"on": True}})
    assert flat == {"a_flag": 1, "b_flag": 0, "nested.on": 1}


def test_flatten_skips_strings():
    # mgr.stats() does not currently include strings, but ``default=str``
    # in the JSON dump could surface them. _flatten must not crash on them.
    flat = bench_diff._flatten({"engine_version": "rust-1.2.3", "steps": 5})
    assert flat == {"steps": 5}


def test_compute_diff_filters_noise_below_both_thresholds():
    # delta=1 on a baseline of 100 = 1% change — below default 5% and below min_abs=10
    rows = bench_diff.compute_diff({"x": 100}, {"x": 101})
    assert rows == []


def test_compute_diff_keeps_large_absolute_delta_even_at_low_percent():
    # delta=20 on a baseline of 10000 = 0.2% — below percent threshold, but
    # well above min_abs=10, so it stays.
    rows = bench_diff.compute_diff({"x": 10_000}, {"x": 10_020})
    assert len(rows) == 1
    assert rows[0][0] == "x"
    assert rows[0][3] == 20


def test_compute_diff_keeps_large_percent_delta_even_at_low_absolute():
    # delta=3 on a baseline of 4 = 75% — below min_abs, but above percent
    # threshold. Should still be reported.
    rows = bench_diff.compute_diff({"x": 4}, {"x": 7})
    assert len(rows) == 1
    assert rows[0][0] == "x"


def test_compute_diff_new_key_reports_inf_pct():
    rows = bench_diff.compute_diff({}, {"new_counter": 100})
    assert len(rows) == 1
    key, b, c, delta, pct = rows[0]
    assert key == "new_counter"
    assert b == 0
    assert c == 100
    assert pct == float("inf")


def test_compute_diff_sorts_by_absolute_delta_descending():
    base = {"a": 10, "b": 1000, "c": 100}
    curr = {"a": 50, "b": 1500, "c": 200}
    rows = bench_diff.compute_diff(base, curr)
    keys = [r[0] for r in rows]
    assert keys == ["b", "c", "a"]


def test_compute_diff_zero_baseline_zero_current_is_skipped():
    rows = bench_diff.compute_diff({"x": 0}, {"x": 0})
    assert rows == []


def test_format_report_empty_rows_shows_no_change_marker():
    out = bench_diff.format_report([])
    assert "no material counter changes" in out


def test_format_report_truncates_at_max_rows():
    base = {f"k{i}": 0 for i in range(50)}
    curr = {f"k{i}": 100 + i for i in range(50)}
    rows = bench_diff.compute_diff(base, curr)
    out = bench_diff.format_report(rows, max_rows=5)
    assert "5 lines" not in out  # sanity
    assert "more counters omitted" in out


def test_load_counters_strips_run_single_preamble(tmp_path):
    # run_single.py --counters-json emits "OK rust <name> <time>" plus a
    # printable line of program output before the JSON object. load_counters
    # must skip past those to the first '{' so users can pipe the raw output
    # in without hand-extracting the JSON block.
    raw = (
        "OK rust fauxware 0.22s peak_mem=177MB\n"
        "  >                            SOSNEAKY \n"
        '{"steps": 3, "callback_count": 1}\n'
    )
    p = tmp_path / "raw.txt"
    p.write_text(raw)
    parsed = bench_diff.load_counters(str(p))
    assert parsed == {"steps": 3, "callback_count": 1}


def test_cli_table_output_via_main(tmp_path):
    base = {"steps": 100, "callback_count": 20}
    curr = {"steps": 130, "callback_count": 25}
    base_path = tmp_path / "baseline.json"
    curr_path = tmp_path / "current.json"
    base_path.write_text(json.dumps(base))
    curr_path.write_text(json.dumps(curr))

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bench_diff.main([str(base_path), str(curr_path)])
    assert rc == 0
    text = buf.getvalue()
    assert "steps" in text
    assert "+30.0%" in text


def test_cli_json_output_via_main(tmp_path):
    base = {"steps": 100}
    curr = {"steps": 130}
    base_path = tmp_path / "b.json"
    curr_path = tmp_path / "c.json"
    base_path.write_text(json.dumps(base))
    curr_path.write_text(json.dumps(curr))

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bench_diff.main([str(base_path), str(curr_path), "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload == [
        {"counter": "steps", "baseline": 100, "current": 130, "delta": 30, "pct": 30.0},
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
