"""A native merge exports each arm's constraints guarded (angr-vmi2l.90).

Arms that reconverge diverged on a branch, so one holds ``c`` and the other
``!c``. The merged solver guards each arm's assertions with its merge flag,
but the exported ``state.solver.constraints`` held the bare pairs, so the
list was jointly UNSAT for a SAT state, and the wide (>128-bit) min/max
fallback that rebuilds a claripy solver from it raised UnsatError.
"""

from __future__ import annotations

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000

CODE = (
    b"\x83\xf8\x05"  # cmp eax, 5
    b"\x77\x02"  # ja +2
    b"\xeb\x02"  # jmp +2
    b"\xeb\x00"  # jmp +0
    b"\xeb\xfe"  # jmp $
)  # fmt: skip


def _merged_state():
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    x = claripy.BVS("x", 64)
    state.regs.rax = x
    mgr = RustExplorationManager(proj, [state])
    for _ in range(3):
        mgr.step()
    mgr.merge()
    assert mgr.stats.get("states_merged_native") == 2
    (merged,) = mgr.active_proxies()
    return merged, x


def test_exported_constraints_are_jointly_satisfiable():
    merged, _ = _merged_state()
    assert merged.solver.satisfiable()
    solver = claripy.Solver()
    solver.add(merged.solver.constraints)
    assert solver.satisfiable()


def test_wide_min_max_on_merged_state():
    merged, x = _merged_state()
    wide = x[31:0].zero_extend(168)
    assert merged.solver.min(wide) == 0
    assert merged.solver.max(wide) == 0xFFFFFFFF
