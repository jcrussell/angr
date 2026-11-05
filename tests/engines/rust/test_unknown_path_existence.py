# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: a read-only open/stat of a path the Rust FileSystem never
saw is Python's call (angr-mx00p.132.15).

Python's ``posix.open`` decides from ``state.fs`` and ``ALL_FILES_EXIST`` /
``ANY_FILE_MIGHT_EXIST``; native stat used to answer ``-1`` and native open
used to mint an fd whatever the options. ``symbolic_fd``'s ``stat_test`` prints
``nothing`` when the stat succeeds and ``c0de`` when it fails.
"""

from __future__ import annotations

import os

import pytest

import angr
from tests.common import bin_location
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

BINARY = os.path.join(bin_location, "tests", "x86_64", "symbolic_fd")


def _stdouts(method: str, all_exist: bool, might_exist: bool = False) -> list[bytes]:
    project = angr.Project(BINARY, auto_load_libs=True, engine="rust")
    state = project.factory.blank_state(addr=project.loader.find_symbol(method).rebased_addr)
    state.options["ALL_FILES_EXIST"] = all_exist
    state.options["ANY_FILE_MIGHT_EXIST"] = might_exist
    simgr = project.factory.simgr(state)
    for _ in range(100):
        if not simgr.active:
            break
        simgr.step()
    assert not simgr.active
    return sorted(s.posix.dumps(1) for s in simgr.unconstrained + simgr.deadended)


@pytest.mark.parametrize("method", ["stat_test", "open_test", "fopen_test"])
class TestUnknownPathExistence:
    def test_all_files_exist_finds_the_file(self, method):
        assert _stdouts(method, all_exist=True) == [b"nothing\n"]

    def test_no_option_finds_no_file(self, method):
        assert _stdouts(method, all_exist=False) == [b"c0de\n"]

    def test_any_file_might_exist_forks_both_ways(self, method):
        # The open's fd is If(file_exists, fd, -1); the bounced stat's
        # eval_one must raise a SimSolverError on it, as SimSolver does.
        assert _stdouts(method, all_exist=False, might_exist=True) == [b"c0de\n", b"nothing\n"]


def test_repeat_explore_counts_only_new_finds():
    # Like SimulationManager.explore, num_find counts finds made by this call:
    # the upstream test loops explore() until active drains, and a second call
    # that stopped at once on the last call's find never advanced the other path.
    project = angr.Project(BINARY, auto_load_libs=True, engine="rust")
    state = project.factory.blank_state(addr=project.loader.find_symbol("fstat_test").rebased_addr)
    state.options["ALL_FILES_EXIST"] = False
    state.options["ANY_FILE_MIGHT_EXIST"] = True
    simgr = project.factory.simgr(state)
    for _ in range(10):
        if not simgr.active:
            break
        simgr.explore(find=lambda s: s.posix.dumps(1) == b"c0de\n", avoid=lambda s: s.posix.dumps(1) == b"nothing\n")
    assert not simgr.active
    assert [s.posix.dumps(1) for s in simgr.found] == [b"c0de\n"]
    assert [s.posix.dumps(1) for s in simgr.avoid] == [b"nothing\n"]
