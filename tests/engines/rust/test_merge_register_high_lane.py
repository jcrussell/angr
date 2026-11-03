"""A merged xmm register's high lane reads the merged overlay, not stale bytes.

A symbolic ``movq xmm0, rbx`` leaves the concrete bytes under the 16-byte
overlay stale. ``RegisterFile::merge`` used to compare the stale high 8 bytes
against the other arm's and stage a concrete ITE at ``xmm0 + 8``, which a later
``movhps`` (``GET:I64(xmm0hq)``) read instead of the merged overlay's high lane
(angr-vmi2l.211). The Python engine is the reference.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
ARM_A = bytes.fromhex("66480f6ec3ffe1")  # movq xmm0, rbx ; jmp rcx
ARM_B = bytes.fromhex("660fefc0ffe1")  # pxor xmm0, xmm0 ; jmp rcx
CODE = bytes.fromhex("83f805") + bytes([0x77, len(ARM_B)]) + ARM_B + ARM_A  # cmp eax, 5 ; ja ARM_A
JOIN = LOAD + len(CODE)
CODE += bytes.fromhex("ebfe")  # jmp $
READ = LOAD + len(CODE)
CODE += bytes.fromhex("0f174424e0488b4424e0ebfe")  # movhps [rsp-0x20], xmm0 ; mov rax, [rsp-0x20] ; jmp $
DONE = LOAD + len(CODE) - 2


def _merged_high_lane(engine):
    project = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    state = project.factory.blank_state(addr=LOAD)
    x = claripy.BVS("x", 64)
    state.regs.rax = x
    state.regs.rbx = claripy.BVS("y", 64)
    state.regs.rcx = JOIN
    state.regs.xmm0 = claripy.BVV(0x1111111111111111_2222222222222222, 128)
    if engine == "rust":
        from angr.exploration.rust_manager import RustExplorationManager

        mgr = RustExplorationManager(project, [state])

        def active():
            return mgr.active_proxies()

    else:
        mgr = project.factory.simgr(state)

        def active():
            return mgr.active

    for _ in range(6):
        if len(active()) == 2 and all(s.addr == JOIN for s in active()):
            break
        mgr.step()
    assert [s.addr for s in active()] == [JOIN, JOIN]
    mgr.merge()
    assert len(active()) == 1
    active()[0].regs.rip = READ
    for _ in range(4):
        mgr.step()
        if all(s.addr == DONE for s in active()):
            break
    (st,) = active()
    assert st.addr == DONE
    return {
        cond: sorted(st.solver.eval_upto(st.regs.rax, 4, extra_constraints=extra))
        for cond, extra in (("any", []), ("x>5", [x[31:0] > 5]), ("x<=5", [x[31:0] <= 5]))
    }


def test_merged_xmm_high_lane_matches_python():
    expected = _merged_high_lane("python")
    assert expected == {"any": [0], "x>5": [0], "x<=5": [0]}
    assert _merged_high_lane("rust") == expected
