"""A RustStateProxy held across a step reads the stepped state (angr-vmi2l.266).

Rust keeps a state's id across a non-forking step, so a proxy taken before
``run()`` still names a live state. Its register memo and its solver fork
(``RustStateProxy._get_shared_solver_ctx``) are keyed on the manager's
``_rust_step_epoch``, so they follow the state as ``addr`` and memory do.
"""

from __future__ import annotations

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

# add rax,1 ; jmp +0 ; add rax,1 ; jmp +0 ; cmp rbx,5 ; ja +0 ; jmp $
_CODE = bytes.fromhex("4883c001eb004883c001eb004883fb057700ebfe")


def _branch(proxy, y):
    """Which side of ``cmp rbx, 5`` the proxy's solver admits: (<= 5, > 5)."""
    return (
        proxy.solver.satisfiable(extra_constraints=[claripy.ULE(y, 5)]),
        proxy.solver.satisfiable(extra_constraints=[claripy.UGT(y, 5)]),
    )


def test_a_proxy_held_across_run_reads_the_stepped_registers_and_constraints():
    proj = angr.load_shellcode(_CODE, arch="amd64", load_address=0x400000)
    state = proj.factory.blank_state(addr=0x400000)
    y = claripy.BVS("y", 64)
    state.regs.rax = 0
    state.regs.rbx = y
    mgr = RustExplorationManager(proj, [state])

    held = mgr.active_proxies()[0]
    regs, solver = held.regs, held.solver
    assert held.regs.rax.concrete_value == 0
    assert _branch(held, y) == (True, True)

    mgr.run(max_steps=3)
    fresh = next(p for p in mgr.active_proxies() if p.state_id == held.state_id)

    assert held.addr == fresh.addr
    assert regs.rax.concrete_value == fresh.regs.rax.concrete_value == 2
    side = _branch(fresh, y)
    assert side in ((True, False), (False, True)), "the kept state carries one side of the ja"
    assert _branch(held, y) == side
    assert (
        solver.satisfiable(extra_constraints=[claripy.ULE(y, 5)]),
        solver.satisfiable(extra_constraints=[claripy.UGT(y, 5)]),
    ) == side
