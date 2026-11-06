"""A loop that grows a register's term past depth 4096 still reaches the solver (angr-vmi2l.83).

``z3_lower::lower`` used to refuse any term deeper than 4096, so every query
on such a term failed with a warn: ``assume`` dropped the branch condition and
the state stayed under-constrained. Lowering now walks an explicit stack. The
shellcode hashes ``rax`` with ``rbx`` in a counted loop, then branches on
``rax == 5``; both engines must keep the same two states, and on the taken
side ``rax`` must be pinned to 5.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

# Each iteration adds four levels to rax's clarirs term, so this passes 4096.
ITERATIONS = 1200
BODY = b"\x48\xc1\xc0\x05\x48\x31\xd8"  # rol rax,5 ; xor rax,rbx
LOOP = BODY + b"\x48\xff\xc9"  # dec rcx
CODE = (
    b"\x48\xc7\xc1"  # mov rcx, ITERATIONS
    + ITERATIONS.to_bytes(4, "little")
    + LOOP
    + b"\x75"  # jnz loop
    + bytes([(256 - (len(LOOP) + 2)) & 0xFF])
    + b"\x48\x83\xf8\x05"  # cmp rax, 5
    + b"\x74\x02"  # je taken
    + b"\xeb\xfe"  # not taken: jmp $
    + b"\xeb\xfe"  # taken: jmp $
)
TAKEN = len(CODE) - 2
NOT_TAKEN = len(CODE) - 4


def _state():
    proj = angr.load_shellcode(CODE, arch="amd64")
    st = proj.factory.blank_state(addr=0)
    st.regs.rax = claripy.BVS("r", 64)
    st.regs.rbx = claripy.BVS("b", 64)
    return proj, st


def test_deep_loop_term_branches_like_python():
    proj, st = _state()
    simgr = proj.factory.simgr(st)
    simgr.run(n=ITERATIONS + 50)
    py_addrs = sorted(s.addr for s in simgr.active)

    proj, st = _state()
    mgr = RustExplorationManager(proj, [st])
    mgr.explore(max_steps=ITERATIONS + 50)
    rust = {s.addr: s for s in mgr.active}

    assert sorted(rust) == py_addrs == [NOT_TAKEN, TAKEN]
    taken = rust[TAKEN]
    assert taken.solver.eval_upto(taken.regs.rax, 2) == [5]
    assert 5 not in rust[NOT_TAKEN].solver.eval_upto(rust[NOT_TAKEN].regs.rax, 4)
