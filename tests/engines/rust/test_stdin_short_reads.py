"""``posix.dumps(0)`` keeps only the bytes a SHORT_READS ``fgets`` returned.

Native fgets records every leaf of a ``size - 1`` read but returns a symbolic
count; Python keeps the read as a packet of that size, so the export cuts the
read there too (angr-mx00p.132.3, upstream test_fgets).
"""

from __future__ import annotations

import os

import pytest

import angr
from tests.common import bin_location
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

FGETS = os.path.join(bin_location, "tests", "x86_64", "fgets")


def _found(symbol: str):
    proj = angr.Project(FGETS, auto_load_libs=False, engine="rust")
    state = proj.factory.entry_state(add_options={angr.sim_options.SHORT_READS})
    state.options.add(angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY)
    mgr = proj.factory.simgr(state)
    assert type(mgr).__name__ == "RustExplorationManager"
    mgr.explore(find=proj.loader.find_symbol(symbol).rebased_addr)
    assert mgr.found, symbol
    return mgr.found[0]


def test_a_short_line_dumps_only_the_line():
    # fgets(buf, 10) of "normal\n" returns 7 bytes, not the 9 it minted.
    assert _found("find_normal").posix.dumps(0) == b"normal\n"


def test_each_short_read_is_cut_at_its_own_count():
    # A first read of any count, then fgets(buf, 11) of "0123456789": the
    # second read is full, so nothing may trail it.
    assert _found("find_exact").posix.dumps(0).endswith(b"0123456789")
