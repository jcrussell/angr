"""A deferred fork's sibling does not inherit a native proc's writes (angr-mx00p.132.4).

The interpreter defers the unexplored side of a branch, and the run loop used
to materialize it only after running the native proc that ended the block. The
branch snapshot restores registers, memory and the solver but not the
filesystem, so the proc's stdout write reached a sibling that never made it:
fauxware's accepted path printed ``Go away!`` before its welcome line.
"""

from __future__ import annotations

import os

import pytest

import angr
from tests.engines.conftest import FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

SSCANF_PATH = os.path.join(TEST_BINARIES_DIR, "sscanf_test")


def _run_outputs(path):
    project = angr.Project(path, auto_load_libs=False, engine="rust")
    mgr = project.factory.simulation_manager()
    mgr.run()
    return [s.posix.dumps(1) for s in mgr.deadended]


@pytest.mark.skipif(not os.path.exists(FAUXWARE_PATH), reason="fauxware binary not available")
def test_fauxware_run_keeps_each_paths_stdout_its_own():
    prompt = b"Username: \nPassword: \n"
    outputs = _run_outputs(FAUXWARE_PATH)
    assert sorted(outputs) == sorted(
        [
            prompt + b"Welcome to the admin console, trusted user!\n",
            prompt + b"Welcome to the admin console, trusted user!\n",
            prompt + b"Go away!",
        ]
    )


@pytest.mark.skipif(not os.path.exists(SSCANF_PATH), reason="sscanf_test binary not available")
def test_sscanf_run_prints_one_line_per_path():
    outputs = _run_outputs(SSCANF_PATH)
    assert len(outputs) == 16
    assert all(out.count(b"\n") == 1 for out in outputs), outputs
