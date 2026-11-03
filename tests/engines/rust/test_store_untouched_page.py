"""A guest store to a page nothing touched lands and reads back (angr-0mo14).

On a ``load_shellcode`` ``blank_state`` with ZERO_FILL_UNCONSTRAINED_MEMORY,
Python publishes no page-universe snapshot, so Rust could not rule out that
the host held the page. The store went to the callback ladder and the flush
dropped it as ``Unmapped``: the export read 0. Python maps the page on the
first write. A page the host does hold must keep its other bytes.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD, PAGE = 0x400000, 0x604000

CODE = (
    b"\x41\xc7\x07\x44\x33\x22\x11"  # mov dword [r15], 0x11223344
    b"\x41\x8b\x5f\x08"  # mov ebx, [r15+8]
    b"\xeb\xfe"  # jmp $
)  # fmt: skip
END = LOAD + len(CODE) - 2


def _run(engine: str, zero_fill: bool, host_word: int | None):
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    opts = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
    if zero_fill:
        opts.add(angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY)
    state = proj.factory.blank_state(addr=LOAD, add_options=opts)
    state.regs.r15 = PAGE
    if host_word is not None:
        state.memory.store(PAGE + 8, host_word, size=4, endness="Iend_LE")
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=END, n=10)
    assert len(mgr.found) == 1
    (found,) = mgr.found
    stored = found.solver.eval(found.memory.load(PAGE, 4, endness="Iend_LE"))
    ebx = found.regs.ebx
    return stored, (found.solver.eval(ebx) if ebx.concrete else None)


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize("zero_fill", [True, False], ids=["zero_fill", "default"])
@pytest.mark.parametrize("host_word", [None, 0xCAFEBABE], ids=["untouched", "host_page"])
def test_store_to_untouched_page_reads_back(engine, zero_fill, host_word):
    stored, ebx = _run(engine, zero_fill, host_word)
    assert stored == 0x11223344
    if host_word is not None:
        assert ebx == host_word
    elif zero_fill:
        assert ebx == 0
    else:
        assert ebx is None  # a filler symbol, as in Python
