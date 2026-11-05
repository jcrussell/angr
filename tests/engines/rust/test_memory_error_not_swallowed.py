"""Out of memory fails loud instead of leaving a silently wrong state (angr-mx00p.132.65).

Near an address-space cap, a page that failed to load into an exported
SimState was logged and left blank, so the state read zeros where Rust held
data, and a ``step_state()`` hook that raised was logged and its source handed
to the native run. veritesting_b's found state then dumped NUL bytes as its
stdin. Both handlers must re-raise ``MemoryError``.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")


@pytest.fixture
def project():
    return angr.Project(FAUXWARE_PATH, auto_load_libs=False)


class _OneBlankPage:
    """A snapshot with one concrete page, as `_load_snapshot_pages` reads it."""

    def page_count(self):
        return 1

    def get_page(self, _i):
        return (0x10000, bytes(0x1000), 7, [])


def _raising(exc):
    def store(*_args, **_kwargs):
        raise exc

    return store


def test_page_load_reraises_memory_error(project, monkeypatch):
    mgr = RustExplorationManager(project, [project.factory.blank_state()])
    state = project.factory.blank_state()
    monkeypatch.setattr(state.memory, "store", _raising(MemoryError()))
    with pytest.raises(MemoryError):
        mgr._load_snapshot_pages(state, _OneBlankPage(), project.arch)


def test_page_load_still_logs_other_errors(project, monkeypatch, caplog):
    mgr = RustExplorationManager(project, [project.factory.blank_state()])
    state = project.factory.blank_state()
    monkeypatch.setattr(state.memory, "store", _raising(ValueError("bad page")))
    mgr._load_snapshot_pages(state, _OneBlankPage(), project.arch)
    assert "Failed to load page at 0x10000: bad page" in caplog.text


class _Raises(angr.ExplorationTechnique):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def step_state(self, simgr, state, **kwargs):
        raise self.exc


def test_step_state_hook_memory_error_propagates(project):
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    mgr.use_technique(_Raises(MemoryError()))
    with pytest.raises(MemoryError):
        mgr.explore(find=0x4006ED, max_steps=50)


def test_step_state_hook_other_error_leaves_the_source_to_the_native_run(project):
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    mgr.use_technique(_Raises(RuntimeError("hook failed")))
    mgr.explore(find=0x4006ED, max_steps=200)
    assert len(mgr.found) == 1
