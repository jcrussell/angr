"""Every gated bench must ship a correctness oracle (angr-imiwt).

``run_regression.py --correctness`` calls each FAST/MEDIUM bench's own
``test()``; a bench without one fails that pass at runtime. This static check
makes the gap visible before any Rust build, when a bench is added.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import run_regression as rr
from replay_oracle import MODEL_CHOICE_ORACLES, REPLAY_ORACLES, host_can_replay, replay
from run_single import EXAMPLES_DIR, _resolve_examples_dir

_SUITE_NAMES = sorted({e[0] for e in rr.FAST_SUITE + rr.MEDIUM_SUITE})


@pytest.mark.parametrize("name", _SUITE_NAMES)
def test_bench_defines_a_test_oracle(name):
    solve = os.path.join(_resolve_examples_dir(name, EXAMPLES_DIR), name, "solve.py")
    if not os.path.exists(solve):
        pytest.skip(f"{name}: example checkout not available")
    with open(solve) as f:
        assert re.search(r"^def test\(", f.read(), re.MULTILINE), f"{name}/solve.py has no test() oracle"


def test_replay_oracles_are_gated_benches():
    assert set(REPLAY_ORACLES) <= set(_SUITE_NAMES)


# A known-good stdin per replayed bench, and the shape a broken native proc
# would leave in posix.dumps(0) (the right answer garbled or truncated).
_REPLAY_CASES = {
    "fauxware": (b"\0" * 9 + b"SOSNEAKY\0", b"\0" * 9 + b"SOSNEAKX\0"),
    "defcamp_r100": (b"Code_Talkers", b"Code_Talker"),
}


@pytest.mark.parametrize("name", sorted(REPLAY_ORACLES))
def test_replay_oracle_tells_good_stdin_from_broken(name):
    """The replay must fail on a wrong stdin, or it gates nothing (angr-vmi2l.120)."""
    if not host_can_replay():
        pytest.skip("replay needs an x86-64 Linux host")
    example_dir = os.path.join(_resolve_examples_dir(name, EXAMPLES_DIR), name)
    spec = REPLAY_ORACLES[name]
    if not os.path.exists(os.path.join(example_dir, spec.binary)):
        pytest.skip(f"{name}: example checkout not available")
    good, broken = _REPLAY_CASES[name]
    assert replay(spec, example_dir, good) is None
    assert replay(spec, example_dir, broken) is not None
    assert replay(spec, example_dir, good.decode()) is not None


@pytest.mark.parametrize("name", sorted(MODEL_CHOICE_ORACLES))
def test_model_choice_oracle_keeps_the_pinned_answer(name):
    """The accepted set widens test(), never replaces it (angr-iqkyb)."""
    solve = os.path.join(_resolve_examples_dir(name, EXAMPLES_DIR), name, "solve.py")
    if not os.path.exists(solve):
        pytest.skip(f"{name}: example checkout not available")
    with open(solve) as f:
        pinned = re.findall(r"^def test\(\):\n\s+assert \w+\(\) == '([^']+)'", f.read(), re.MULTILINE)
    assert pinned and set(pinned) <= MODEL_CHOICE_ORACLES[name].accepted
    assert len(MODEL_CHOICE_ORACLES[name].accepted) > 1
