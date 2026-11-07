"""``factory.simgr`` accepts a ``RustExplorationManager`` stash entry as its seed (angr-vmi2l.19)."""

from __future__ import annotations

import os

import pytest

import angr
from angr.errors import AngrError
from angr.exploration.rust_manager import RustExplorationManager

BINARIES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "binaries", "tests")
FAUXWARE = os.path.join(BINARIES, "x86_64", "fauxware")
AUTHENTICATED = 0x4006ED

pytestmark = pytest.mark.skipif(not os.path.exists(FAUXWARE), reason="binaries checkout missing")


def _stage_one():
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    s1 = RustExplorationManager(proj, [proj.factory.entry_state()])
    s1.explore(find=AUTHENTICATED)
    assert len(s1.found) == 1
    return proj, s1


@pytest.mark.parametrize("engine", [None, "rust"], ids=["python", "rust"])
@pytest.mark.parametrize("form", ["state", "list", "stash"])
def test_simgr_seeded_from_found_entry(form, engine):
    proj, s1 = _stage_one()
    counts = {name: len(states) for name, states in s1.stashes.items()}
    found = s1.found[0]
    stdin = found.posix.dumps(0)
    thing = {"state": found, "list": [found], "stash": s1.found}[form]

    s2 = proj.factory.simgr(thing, engine=engine)
    assert isinstance(s2.active[0], angr.SimState)
    assert s2.active[0].posix.dumps(0) == stdin
    s2.run()
    assert [b"trusted user" in st.posix.dumps(1) for st in s2.deadended] == [True]
    # The source manager keeps its stashes and its found state.
    assert {name: len(states) for name, states in s1.stashes.items()} == counts
    assert s1.found[0].posix.dumps(0) == stdin


def test_simgr_still_rejects_a_non_state():
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    with pytest.raises(AngrError, match="BadType"):
        proj.factory.simgr(object())
    with pytest.raises(AngrError, match="Bad type"):
        proj.factory.simgr([proj.factory.entry_state(), object()])
