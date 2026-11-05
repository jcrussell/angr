"""An export under STRICT_PAGE_ACCESS loads every Rust page (angr-mx00p.132.73).

A plain store refused a page Python never mapped (the stdout buffer a native
mmap made) as unmapped and a read-only loader page as non-writable; each
failure was logged and the page left at its blank default, so test_accuracy's
write syscall segfaulted on the mmap'd buffer. The export now maps a missing
page with the Rust page's permissions, converted from Rust's bit order
(R=4, W=2, X=1) to angr's (R=1, W=2, X=4), and stores with ``priv``.
"""

from __future__ import annotations

import pytest

import angr
from angr import sim_options as o
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

MMAP_PAGE = 0xC1000000
TEXT_PAGE = 0x400000
RUST_RW = 0x6
RUST_RX = 0x5


class _Snapshot:
    """Rust pages as `_load_snapshot_pages` reads them: (addr, data, perms, symbolic_offsets)."""

    def __init__(self, pages):
        self._pages = pages

    def page_count(self):
        return len(self._pages)

    def get_page(self, i):
        return self._pages[i]


@pytest.fixture
def project():
    return angr.Project(FAUXWARE_PATH, auto_load_libs=False)


def test_strict_export_maps_a_natively_mmapped_page_and_writes_a_read_only_one(project, caplog):
    mgr = RustExplorationManager(project, [project.factory.blank_state()])
    state = project.factory.blank_state(add_options={o.STRICT_PAGE_ACCESS})
    mmap_bytes = bytes(range(256)) * 16
    text_bytes = b"\xcc" * 0x1000
    snapshot = _Snapshot([(MMAP_PAGE, mmap_bytes, RUST_RW, []), (TEXT_PAGE, text_bytes, RUST_RX, [])])

    mgr._load_snapshot_pages(state, snapshot, project.arch)

    assert "Failed to load page" not in caplog.text
    assert state.solver.eval(state.memory.load(MMAP_PAGE, 0x1000), cast_to=bytes) == mmap_bytes
    assert state.solver.eval(state.memory.load(TEXT_PAGE, 0x1000), cast_to=bytes) == text_bytes
    # angr's bits: R=1, W=2, X=4.
    assert state.solver.eval(state.memory.permissions(MMAP_PAGE)) == 0x3
    # A page the loader backs keeps its Python permissions (r-x).
    assert state.solver.eval(state.memory.permissions(TEXT_PAGE)) == 0x5
