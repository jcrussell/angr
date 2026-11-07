"""Smoke test: examples/rust_exploration_example.py solves fauxware in every example (angr-vmi2l.283)."""

from __future__ import annotations

import os
import resource
import subprocess
import sys

import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT = os.path.join(REPO_ROOT, "examples", "rust_exploration_example.py")
EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


def _cap_memory():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))


def test_every_example_finds_the_backdoor_password():
    if not os.path.isfile(os.path.join(EXAMPLES_DIR, "fauxware", "fauxware")):
        pytest.skip("angr-examples fauxware binary not available")
    pythonpath = os.pathsep.join(p for p in (REPO_ROOT, os.environ.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, SCRIPT, EXAMPLES_DIR],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": pythonpath},
        preexec_fn=_cap_memory,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for tag in ("[address]", "[predicate] Stdin:", "[techniques]", "[direct]"):
        lines = [line for line in result.stdout.splitlines() if line.startswith(tag)]
        assert lines, f"{tag} printed nothing:\n{result.stdout}"
        assert all("SOSNEAKY" in line for line in lines), f"{tag} found no solution:\n{result.stdout}"
