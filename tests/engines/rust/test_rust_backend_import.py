"""A SimState RustBackend imports keeps its plugins on every export of its lineage (angr-mx00p.136.31).

An imported state's cached mirror is the plugin template its exports copy
posix/libc/heap/fs/log from. The backend exports the import's own id, so the
template is kept in ``_lineage_templates`` rather than ``_state_cache``;
without it a mid-run state lost the stdin it had read on Python.
"""

from __future__ import annotations

import gc
import os

import pytest

import angr
from angr.exploration.rust_backend import RustBackend
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

FIND = {"x86_64": 0x4006ED, "mips": 0x4009FC}
AVOID = {"x86_64": [0x4006AA, 0x4006FD], "mips": [0x400A10, 0x400774]}
SOSNEAKY = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00SOSNEAKY\x00"
# Python-engine steps until fauxware's states have read the username and password.
PAST_READS_STEPS = 30


def _project(arch):
    return angr.Project(os.path.join(TEST_BINARIES_DIR, "..", arch, "fauxware"), auto_load_libs=False)


def _mid_run_simgr(arch):
    project = _project(arch)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()])
    simgr.run(n=PAST_READS_STEPS)
    assert any(state.posix.stdin.pos for state in simgr.active)
    return simgr


@pytest.mark.parametrize("arch", sorted(FIND))
def test_a_mid_run_python_state_keeps_its_read_stdin(arch):
    simgr = _mid_run_simgr(arch)
    backend = RustBackend(simgr._project)
    simgr._backend = backend
    simgr.explore(find=FIND[arch], avoid=AVOID[arch])
    assert backend.stats["claimed"] > 0
    assert simgr.found[0].posix.dumps(0) == SOSNEAKY


def test_an_export_after_a_non_forking_step_keeps_the_imports_posix():
    simgr = _mid_run_simgr("x86_64")
    simgr.active[:] = simgr.active[:1]
    (source,) = simgr.active
    pos, read = source.posix.stdin.pos, source.posix.dumps(0)
    simgr._backend = RustBackend(simgr._project)
    simgr.step()
    (child,) = simgr.active
    assert child.posix.stdin.pos == pos
    assert child.posix.dumps(0) == read


def test_the_template_is_dropped_with_its_lineage():
    project = _project("x86_64")
    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    simgr.run(n=PAST_READS_STEPS)
    assert backend.stats["claimed"] > 0
    templates = backend._mgr._lineage_templates
    assert templates
    assert all(template is not state for template in templates.values() for state in simgr.active)
    for stash in list(simgr.stashes):
        simgr.drop(stash=stash)
    gc.collect()
    # What the next step does first.
    backend._drop_released(backend._mgr)
    assert not templates
    assert not backend._root_of
    assert not backend._lineage_refs
