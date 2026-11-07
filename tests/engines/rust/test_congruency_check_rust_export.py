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


@pytest.mark.parametrize("zero_fill", [False, True])
def test_imported_partly_written_page_reads_unset_bytes_as_python_does(zero_fill):
    """A page Python wrote one byte of, imported into Rust, reads its other bytes as fills (angr-mx00p.132.81).

    Without ``ZERO_FILL_UNCONSTRAINED_MEMORY`` Python mints a fill for an unwritten byte, so the live export and its
    detached copy must too; with it, both read zero.
    """
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    opts = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY} if zero_fill else set()
    state = proj.factory.entry_state(add_options=opts)
    state.memory.store(0x50000F30, b"\x41")
    simgr = proj.factory.simgr(state, engine="rust")
    simgr.explore(find=AUTHENTICATED)
    export = simgr.found[0]
    assert isinstance(export.memory, RustMemoryProxy)
    plain = export._plain_copy()

    for read in (export.memory.load(0x50000308, 8), plain.memory.load(0x50000310, 8)):
        assert read.symbolic == (not zero_fill)
    assert plain.memory.load(0x50000F30, 1).concrete_value == 0x41
    # A minted fill stays: the second read is the first one's value.
    assert export.memory.load(0x50000308, 8) is export.memory.load(0x50000308, 8)
