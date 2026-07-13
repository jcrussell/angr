"""Tests for the M2 repeat-run result-equality harness (angr-op0dn.10.4).

Two layers, mirroring ``test_memory_gate.py``'s split:

* the **decision** tests are pure and angr-free (``repeat_run_equality``'s
  top-level imports stay light — ``run_single``'s ``import angr`` is lazy), so
  they run in milliseconds and always execute. They include the self-test the
  bead's acceptance criteria demand: a *synthetic mismatch* must make the
  harness fail, otherwise a green gate proves nothing.
* the **real gate** actually runs N repeats of fast non-bimodal benches in
  memory-capped subprocesses with strict mode on. It costs ~a minute, so it is
  opt-in via ``ANGR_REPEAT_EQUALITY_GATE=1`` (set it in the nightly CI job);
  a plain ``pytest tests/benchmarks/`` stays fast.

No test here asserts anything about wall clock — timing variance is explicitly
out of M2's scope (bd memory ``benchmark-bimodal-variance-rules``).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import repeat_run_equality as rre
from run_regression import BIMODAL_BENCHMARKS


def _run(found_pcs: str, output: str) -> dict:
    """One synthetic child result, shaped like ``run_single``'s."""
    return rre.run_fingerprints({"stats": {"found_pcs": found_pcs}, "output": output})


def test_identical_runs_are_equal():
    runs = [_run("0x400a1b*1", "flag{abc}") for _ in range(5)]
    v = rre.equality_verdict(runs)
    assert v["equal"]
    assert v["found_identical"] and v["model_identical"]
    assert v["n"] == 5


def test_synthetic_found_set_mismatch_fails():
    # Same model bytes, one repeat found a different pc multiset.
    runs = [_run("0x400a1b*1", "flag{abc}") for _ in range(4)]
    runs.append(_run("0x400a1b*2", "flag{abc}"))
    v = rre.equality_verdict(runs)
    assert not v["equal"]
    assert not v["found_identical"]
    assert v["model_identical"]
    assert len(v["distinct_found"]) == 2


def test_synthetic_model_bytes_mismatch_fails():
    # Same found set, one repeat evaluated a different satisfying model — the
    # exact failure mode strict mode exists to rule out.
    runs = [_run("0x400a1b*1", "flag{abc}") for _ in range(4)]
    runs.append(_run("0x400a1b*1", "flag{xyz}"))
    v = rre.equality_verdict(runs)
    assert not v["equal"]
    assert v["found_identical"]
    assert not v["model_identical"]


def test_no_runs_is_not_equal():
    # A bench whose every repeat crashed proves nothing; passing it would make
    # the gate vacuous.
    assert not rre.equality_verdict([])["equal"]


def test_fingerprints_are_process_stable():
    # The comparison happens across separate processes, so the reduction must
    # not ride on PYTHONHASHSEED-randomized hash(). Same input, same digest.
    assert _run("0x1*1", "x")["found_fp"] == _run("0x1*1", "x")["found_fp"]
    assert _run("0x1*1", "x")["found_fp"] != _run("0x2*1", "x")["found_fp"]
    assert _run("0x1*1", "x")["model_fp"] != _run("0x1*1", "y")["model_fp"]


def test_exit_code_gates_on_non_bimodal_only():
    census = {
        "fauxware": {"equal": True, "gated": True},
        "securityfest_fairlight": {"equal": False, "gated": False},
    }
    assert rre.census_exit_code(census) == 0

    census["fauxware"]["equal"] = False
    assert rre.census_exit_code(census) == 1


def test_empty_census_fails():
    assert rre.census_exit_code({}) == 1
    # ...and so does one made only of report-only benches: nothing gated ran.
    assert rre.census_exit_code({"securityfest_fairlight": {"equal": True, "gated": False}}) == 1


def test_gate_corpus_excludes_bimodal_benches():
    assert not (set(rre.GATE_CORPUS) & BIMODAL_BENCHMARKS)


@pytest.mark.skipif(
    not os.environ.get("ANGR_REPEAT_EQUALITY_GATE"),
    reason="real N-repeat gate is slow (~1min); set ANGR_REPEAT_EQUALITY_GATE=1 to run",
)
def test_repeat_runs_are_result_identical():
    """The M2 acceptance gate itself: N=5 strict-mode repeats, identical results."""
    census = rre.run_equality(["fauxware", "ais3_crackme"], 5, deterministic=True)
    for name, v in census.items():
        assert v["n"] == 5, f"{name}: only {v['n']}/5 repeats ran"
        assert v["equal"], f"{name}: results differ across repeats: {v}"
    assert rre.census_exit_code(census) == 0
