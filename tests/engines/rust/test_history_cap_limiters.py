"""LengthLimiter and LoopSeer see past the ``max_history`` window (angr-vmi2l.31).

Rust keeps each state's block history in a window capped at ``max_history``.
The native LengthLimiter and LoopBound used to read that window, so a limit
the window could not hold never fired, while Python counts the whole path
(``history.block_count``). Each test caps the window well below the limit.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
MAX_HISTORY = 50

# A 2-block infinite loop: inc rax; jmp +0xb / ...; inc rbx; jmp back.
TWO_BLOCK_LOOP = bytearray(b"\x90" * 0x20)
TWO_BLOCK_LOOP[0:5] = b"\x48\xff\xc0\xeb\x0b"
TWO_BLOCK_LOOP[0x10:0x15] = b"\x48\xff\xc3\xeb\xeb"


def _long_loop(blocks: int) -> bytes:
    """An infinite loop of ``blocks`` blocks: ``inc rax; cmp rax, -1; je halt`` each."""
    code = bytearray()
    halt = 9 * blocks + 2
    for _ in range(blocks):
        base = len(code)
        code += b"\x48\xff\xc0\x48\x83\xf8\xff"
        code += bytes([0x74, (halt - (base + 9)) & 0xFF])
    code += bytes([0xEB, (0 - (len(code) + 2)) & 0xFF])
    return bytes(code + b"\xf4" * 4)


def _manager(engine: str, code: bytes):
    proj = angr.load_shellcode(bytes(code), arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    state.regs.rax = 0
    state.regs.rbx = 0
    if engine == "rust":
        return RustExplorationManager(proj, [state], max_history=MAX_HISTORY)
    return proj.factory.simulation_manager(state)


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_length_limiter_past_the_cap_cuts(engine):
    mgr = _manager(engine, TWO_BLOCK_LOOP)
    mgr.use_technique(angr.exploration_techniques.LengthLimiter(max_length=3 * MAX_HISTORY))
    mgr.run(n=10 * MAX_HISTORY)
    assert len(mgr.active) == 0
    assert len(mgr.cut) == 1


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_loop_seer_past_the_cap_spins(engine):
    # 12 blocks x bound 10 is more visits than a 50-entry window holds.
    mgr = _manager(engine, _long_loop(12))
    mgr.use_technique(angr.exploration_techniques.LocalLoopSeer(bound=10))
    mgr.run(n=10 * MAX_HISTORY)
    assert len(mgr.active) == 0
    assert len(mgr.spinning) == 1


def test_proxy_block_count_is_uncapped():
    mgr = _manager("rust", TWO_BLOCK_LOOP)
    mgr.run(n=3 * MAX_HISTORY)
    (proxy,) = mgr.active_proxies()
    assert len(proxy.history.recent_bbl_addrs) <= MAX_HISTORY
    assert proxy.history.block_count > MAX_HISTORY
