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


def test_a_fill_python_made_before_handoff_survives_the_import(caplog):
    """A byte Python default-filled before handoff reads as that fill in the export, not as 0 (angr-mx00p.132.82).

    A page whose only unstored bytes are uninit, with no symbolic offsets, also loads without a page failure.
    """
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    state = proj.factory.entry_state()
    state.memory.store(0x50000F30, b"\x41")
    # The fill lands on the page the copy shares with ``state``.
    sibling = state.copy()
    fill = sibling.memory.load(0x50000308, 8)
    assert fill.symbolic
    simgr = proj.factory.simgr(state, engine="rust")
    simgr.explore(find=AUTHENTICATED)
    export = simgr.found[0]
    with caplog.at_level("WARNING", logger="angr.exploration.rust_state_export"):
        plain = export._plain_copy()

    for read in (export.memory.load(0x50000308, 8), plain.memory.load(0x50000308, 8)):
        assert read.variables == fill.variables
    assert "Failed to load page" not in caplog.text


def test_handoff_mints_no_fills_in_the_python_state():
    """Syncing a state into Rust loads no unwritten Python memory (angr-mx00p.132.82).

    A load there mints a fill the Python engine would only make later, at another width, and CongruencyCheck then
    sees two differently shaped fills (``test_unicorn.py::test_similarity_fauxware``). The two loads this guards:
    the overlay of the non-allocated sections at vaddr 0, and the whole-page user-variable probe of a page with a gap.
    """
    proj = angr.Project(FAUXWARE, auto_load_libs=False)
    state = proj.factory.entry_state()
    state.memory.store(0x50000F30, b"\x41")
    state.memory.load(0x50000308, 8)
    page = state.memory._pages[0x50000]
    before = dict(page.symbolic_data)

    simgr = proj.factory.simgr(state, engine="rust")
    simgr.explore(find=AUTHENTICATED)
    assert simgr.found

    low = state.memory._pages.get(0)
    assert low is None or not low.symbolic_data
    assert dict(page.symbolic_data) == before
