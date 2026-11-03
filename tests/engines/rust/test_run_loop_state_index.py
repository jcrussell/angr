# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: the serial run loop un-indexes the state it pops
(angr-mx00p.132.50).

The loop took states off ``active`` with a bare ``SelectionPolicy::select``,
which left their ``state_index`` entry behind. A state that had been indexed
(by a fork push, or by a ``sync_state_index`` between runs) and then deadended
under ``drop_terminal_states`` tripped ``push_or_drop_terminal``'s always-on I6
assert, which aborts the process. Each scenario therefore runs in a subprocess,
so a regression fails one test instead of killing pytest.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tests.engines.conftest import FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_PRELUDE = f"""
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
import logging
logging.disable(logging.WARNING)
import angr
from angr.exploration.rust_manager import RustExplorationManager
p = angr.Project({FAUXWARE_PATH!r}, auto_load_libs=False)
mgr = RustExplorationManager(p, [p.factory.entry_state()])
"""


def _run(body: str) -> str:
    if not os.path.exists(FAUXWARE_PATH):
        pytest.skip("fauxware binary not found")
    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"subprocess aborted or failed (returncode={proc.returncode}):\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr[-4000:]!r}"
    )
    return proc.stdout


class TestRunLoopUnindexesPoppedState:
    def test_address_explore_after_a_predicate_explore(self):
        out = _run(
            """
            mgr.explore(find=lambda s: False, max_steps=1)
            mgr.explore(find=0x1234)
            print("RESULT", len(mgr.found), len(mgr.active))
            """
        )
        # explore() drops terminal states, so only the exhausted frontier shows.
        assert "RESULT 0 0" in out, out

    def test_single_steps_with_drop_terminal_states(self):
        out = _run(
            """
            mgr.explore(find=0x1234, max_steps=1)
            r = mgr._rust_mgr
            r.set_drop_terminal_states(True)
            for _ in range(100):
                # Indexes every active state, including the fork that later
                # deadends at exit().
                r.sync_state_index()
                r.run(1)
                if not r.get_state_ids("active"):
                    break
            print("RESULT", len(r.get_state_ids("active")), len(r.get_state_ids("found")))
            """
        )
        assert "RESULT 0 0" in out, out
