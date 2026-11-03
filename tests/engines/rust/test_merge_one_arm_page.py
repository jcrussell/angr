"""A page only one merge arm touched stays guarded by that arm's condition.

``SymbolicMemory::merge`` used to adopt a page only ``other`` held wholesale
and keep a page only ``self`` held as is, so every path of the merged state
read the writing arm's bytes (angr-vmi2l.187). Each arm must keep the value it
read before the merge.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
TARGETS = {
    # lea rbx, [rsp-0x3000] ; nop ; nop: a fresh stack page
    "stack": bytes.fromhex("488d9c2400d0ffff9090"),
    # lea rbx, [rsp-0x200] ; nop ; nop: the current stack page's neighbour
    "stack_near": bytes.fromhex("488d9c2400feffff9090"),
    # mov ebx, 0x401800 ; nop x5: a loader-image page with backer bytes
    "image": bytes.fromhex("bb00184000") + b"\x90" * 5,
}


def _program(where):
    writer = TARGETS[where] + bytes.fromhex("c60341")  # mov byte [rbx], 0x41
    # cmp eax, 5 ; ja writer ; jmp join ; writer ; join: jmp $
    code = bytes.fromhex("83f8057702") + bytes([0xEB, len(writer)]) + writer + bytes.fromhex("ebfe")
    return code + b"\x5a" * (0x2000 - len(code)), LOAD + len(code) - 2


@pytest.mark.parametrize("where", sorted(TARGETS))
@pytest.mark.parametrize("zero_fill", [False, True])
def test_one_arm_page_keeps_each_arms_value(where, zero_fill):
    blob, fin = _program(where)
    project = angr.load_shellcode(blob, arch="amd64", load_address=LOAD)
    opts = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY} if zero_fill else set()
    state = project.factory.blank_state(addr=LOAD, add_options=opts)
    x = claripy.BVS("x", 64)
    state.regs.rax = x
    if where == "image":
        target = 0x401800
    else:
        offset = 0x3000 if where == "stack" else 0x200
        target = state.solver.eval(state.regs.rsp) - offset

    from angr.exploration.rust_manager import RustExplorationManager

    mgr = RustExplorationManager(project, [state])
    for _ in range(6):
        active = mgr.active_proxies()
        if len(active) == 2 and all(s.addr == fin for s in active):
            break
        mgr.step()
    active = mgr.active_proxies()
    assert len(active) == 2 and all(s.addr == fin for s in active)

    writer_arm = x[31:0] > 5
    pre = {}
    for s in active:
        wrote = s.solver.satisfiable(extra_constraints=[writer_arm])
        pre[wrote] = sorted(s.solver.eval_upto(s.memory.load(target, 1), 4))
    assert pre[True] == [0x41]

    mgr.merge()
    (merged,) = mgr.active_proxies()
    value = merged.memory.load(target, 1)
    assert sorted(merged.solver.eval_upto(value, 4, extra_constraints=[writer_arm])) == [0x41]
    assert sorted(merged.solver.eval_upto(value, 4, extra_constraints=[claripy.Not(writer_arm)])) == pre[False]
