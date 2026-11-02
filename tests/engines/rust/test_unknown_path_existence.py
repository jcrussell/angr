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


def _stdouts(method: str, all_exist: bool) -> list[bytes]:
    project = angr.Project(BINARY, auto_load_libs=True, engine="rust")
    state = project.factory.blank_state(addr=project.loader.find_symbol(method).rebased_addr)
    state.options["ALL_FILES_EXIST"] = all_exist
    state.options["ANY_FILE_MIGHT_EXIST"] = False
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
