"""``CongruencyCheck.compare_states`` diffs a state the Rust engine exported (angr-mx00p.132.69)."""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_state_proxy import RustMemoryProxy

BINARIES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "binaries", "tests")
FAUXWARE = os.path.join(BINARIES, "x86_64", "fauxware")
AUTHENTICATED = 0x4006ED

pytestmark = pytest.mark.skipif(not os.path.exists(FAUXWARE), reason="binaries checkout missing")


@pytest.mark.parametrize("rust_side", ["left", "right", "both"])
def test_compare_states_with_a_rust_export(rust_side):
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    simgr = proj.factory.simgr(proj.factory.entry_state(), engine="rust")
    simgr.explore(find=AUTHENTICATED)
    export = simgr.found[0]
    assert isinstance(export.memory, RustMemoryProxy)
    plain = export._plain_copy()
    assert not isinstance(plain.memory, RustMemoryProxy)
    sl, sr = {"left": (export, plain), "right": (plain, export), "both": (export, export)}[rust_side]

    # The diff walks Python memory pages, which a Rust proxy does not have.
    assert proj.analyses.CongruencyCheck(throw=True).compare_states(sl, sr)
    # Comparing reads copies: the export stays a view of its Rust state.
    assert isinstance(export.memory, RustMemoryProxy)
