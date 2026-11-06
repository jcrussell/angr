"""Seed pages reach Rust with the Python state's permissions (angr-vmi2l.273).

Each case runs the same STRICT_PAGE_ACCESS + ENABLE_NX seed on both engines
and expects the same SimSegfaultException. Pages the seed has already
touched or mapped itself used to reach Rust RWX, so the Rust engine
completed the write or the execute.
"""

from __future__ import annotations

import struct

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager, load_fauxware

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

CODE = 0x10000000
SPIN = b"\xeb\xfe"  # jmp $


def _mov_rax(addr: int) -> bytes:
    return b"\x48\xb8" + struct.pack("<Q", addr)


def _seed(project, case: str):
    main = project.loader.main_object
    text = main.sections_map[".text"].min_addr
    data = main.sections_map[".data"].min_addr
    state = project.factory.blank_state(
        addr=CODE, add_options={angr.options.STRICT_PAGE_ACCESS, angr.options.ENABLE_NX}
    )
    state.memory.map_region(0x7FFFFFFF0000, 0x10000, 3)
    if case == "write_text_touched":
        code, fault = _mov_rax(text) + b"\xc6\x00\x90" + SPIN, (text & ~0xFFF, "non-writable")
        state.memory.map_region(CODE, 0x1000, 7)
        state.memory.load(text, 4)  # the .text page is now a Python page
    elif case == "exec_data_touched":
        code, fault = _mov_rax(data) + b"\xff\xe0" + SPIN, (data, "non-executable")
        state.memory.map_region(CODE, 0x1000, 7)
        state.memory.load(data, 4)
    elif case == "write_user_rx":
        code, fault = _mov_rax(CODE + 0x800) + b"\xc6\x00\x90" + SPIN, (CODE, "non-writable")
        state.memory.map_region(CODE, 0x1000, 7)
    elif case == "exec_user_rw":
        code, fault = b"\x90\x90" + SPIN, (CODE, "non-executable")
        state.memory.map_region(CODE, 0x1000, 3)
    else:
        raise AssertionError(case)
    state.memory.store(CODE, code)
    if case == "write_user_rx":
        state.memory.permissions(CODE, 5)  # angr bits: R|X
    return state, CODE + len(code) - len(SPIN), fault


@pytest.mark.parametrize("case", ["write_text_touched", "exec_data_touched", "write_user_rx", "exec_user_rw"])
def test_seed_page_permissions_fault_like_python(case):
    # A fresh project per case: every case puts different code at CODE, and the
    # project's lifter caches the block by address.
    project = load_fauxware()
    state, end, fault = _seed(project, case)
    simgr = project.factory.simulation_manager(state.copy())
    simgr.explore(find=end, n=20)
    python_errors = [(e.error.addr, e.error.reason) for e in simgr.errored]
    assert python_errors == [fault]

    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=end)
    assert not mgr.found
    assert [(e.error.addr, e.error.reason) for e in mgr.errored] == [fault]
