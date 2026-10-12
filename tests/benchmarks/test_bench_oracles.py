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

import run_regression as rr  # noqa: E402
from run_single import EXAMPLES_DIR, _resolve_examples_dir  # noqa: E402

_SUITE_NAMES = sorted({e[0] for e in rr.FAST_SUITE + rr.MEDIUM_SUITE})


@pytest.mark.parametrize("name", _SUITE_NAMES)
def test_bench_defines_a_test_oracle(name):
    solve = os.path.join(_resolve_examples_dir(name, EXAMPLES_DIR), name, "solve.py")
    if not os.path.exists(solve):
        pytest.skip(f"{name}: example checkout not available")
    with open(solve) as f:
        assert re.search(r"^def test\(", f.read(), re.M), f"{name}/solve.py has no test() oracle"
