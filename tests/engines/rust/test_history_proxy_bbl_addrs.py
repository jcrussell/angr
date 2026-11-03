"""RustHistoryProxy's bbl_addrs / recent_bbl_addrs match Python's history (angr-v625t).

The proxy served one memoized 256-entry tail of the plain history for both
properties. That history records each step's successor pc, so the tail was
shifted by one block (it named the state's own address), ``recent_bbl_addrs``
held far more than the last step, ``addr in bbl_addrs`` turned False once the
visit was 256+ blocks back, and a proxy read before a step kept serving the
stale list after it.
"""

from __future__ import annotations

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000

# A 2-block infinite loop: inc rax; jmp +0xb / ...; inc rbx; jmp back.
TWO_BLOCK_LOOP = bytearray(b"\x90" * 0x20)
TWO_BLOCK_LOOP[0:5] = b"\x48\xff\xc0\xeb\x0b"
TWO_BLOCK_LOOP[0x10:0x15] = b"\x48\xff\xc3\xeb\xeb"

# A straight line of 300 `inc rax` blocks (each ends in a 2-byte `jmp +0`),
# then the 2-block loop: block 0 is 300+ blocks back once the loop runs.
BLOCK = b"\x48\xff\xc0\xeb\x00"
LINE_BLOCKS = 300


def _project(code: bytes):
    return angr.load_shellcode(bytes(code), arch="amd64", load_address=LOAD)


def _state(proj):
    state = proj.factory.blank_state(addr=LOAD)
    state.regs.rax = 0
    state.regs.rbx = 0
    return state


def _python_history(code: bytes, steps: int):
    proj = _project(code)
    mgr = proj.factory.simulation_manager(_state(proj))
    mgr.run(n=steps)
    (state,) = mgr.active
    return list(state.history.bbl_addrs), list(state.history.recent_bbl_addrs)


def _rust_manager(code: bytes, **kwargs):
    proj = _project(code)
    return RustExplorationManager(proj, [_state(proj)], **kwargs)


def test_proxy_matches_python_history():
    steps = 5
    py_all, py_recent = _python_history(TWO_BLOCK_LOOP, steps)
    mgr = _rust_manager(TWO_BLOCK_LOOP)
    mgr.run(n=steps)
    (proxy,) = mgr.active_proxies()
    assert proxy.history.bbl_addrs == py_all
    assert proxy.history.recent_bbl_addrs == py_recent == [py_all[-1]]


def test_proxy_reads_are_not_memoized_across_steps():
    mgr = _rust_manager(TWO_BLOCK_LOOP)
    mgr.run(n=1)
    (proxy,) = mgr.active_proxies()
    history = proxy.history
    assert history.recent_bbl_addrs == [LOAD]
    mgr.run(n=1)
    (after,) = mgr.active_proxies()
    assert after.history.recent_bbl_addrs == [LOAD + 0x10]
    if after._state_id == proxy._state_id:
        assert history.recent_bbl_addrs == [LOAD + 0x10]
        assert history.bbl_addrs == [LOAD, LOAD + 0x10]


def test_bbl_addrs_reaches_past_256_blocks():
    code = BLOCK * LINE_BLOCKS + bytes(TWO_BLOCK_LOOP)
    steps = LINE_BLOCKS + 4
    py_all, py_recent = _python_history(code, steps)
    assert LOAD in py_all
    mgr = _rust_manager(code)
    mgr.run(n=steps)
    (proxy,) = mgr.active_proxies()
    assert LOAD in proxy.history.bbl_addrs
    assert proxy.history.bbl_addrs == py_all
    assert proxy.history.recent_bbl_addrs == py_recent


def test_bbl_addrs_is_the_max_history_window():
    mgr = _rust_manager(TWO_BLOCK_LOOP, max_history=10)
    mgr.run(n=30)
    (proxy,) = mgr.active_proxies()
    assert len(proxy.history.bbl_addrs) == 10
    assert proxy.history.block_count == 30
