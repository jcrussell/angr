"""A native proc's symbol is a tracked ``("api", ...)`` variable (angr-vmi2l.126).

Python's ``rand`` and ``time`` mint their return value with
``key=("api", "rand")`` / ``key=("api", "time")``, so
``state.solver.get_variables("api", "rand")`` recovers each call's output. The
native procs mint in Rust; the found state's solver proxy must yield them
under the same temporal keys, counted in call order.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
RAND = 0x500000
TIME = 0x500100


def _call(target: int) -> bytes:
    # mov rax, target; call rax
    return b"\x48\xb8" + target.to_bytes(8, "little") + b"\xff\xd0"


# rand(); mov ebx, eax; rand(); mov r12d, eax; xor edi, edi; time(NULL); jmp $
CODE = _call(RAND) + b"\x89\xc3" + _call(RAND) + b"\x41\x89\xc4" + b"\x31\xff" + _call(TIME) + b"\xeb\xfe"
FIN = LOAD + len(CODE) - 2


def _found(engine: str):
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    lib = angr.SIM_LIBRARIES["libc.so.6"][0]
    for addr, name in ((RAND, "rand"), (TIME, "time")):
        proj.hook(addr, angr.SIM_PROCEDURES["libc"][name](prototype=lib.get_prototype(name)), replace=True)
    state = proj.factory.blank_state(addr=LOAD)
    state.regs.rsp = 0x7FFFFFFF0800
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=FIN)
    assert len(mgr.found) == 1
    if engine == "rust":
        counts = mgr._rust_mgr.native_procedure_stats().get("call_counts")
        assert counts.get("rand") == 2
        assert counts.get("time") == 1
    return mgr.found[0]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_get_variables_api_yields_each_native_call(engine):
    found = _found(engine)
    keys = dict(found.solver.get_variables("api"))
    assert sorted(keys) == [("api", "rand", 1), ("api", "rand", 2), ("api", "time", 1)]
    assert keys[("api", "rand", 1)].size() == 31
    assert keys[("api", "time", 1)].size() == 64
    # Each tracked variable is the value its call returned.
    assert not found.solver.satisfiable(extra_constraints=[found.regs.ebx != keys[("api", "rand", 1)].zero_extend(1)])
    assert not found.solver.satisfiable(extra_constraints=[found.regs.r12d != keys[("api", "rand", 2)].zero_extend(1)])
    assert not found.solver.satisfiable(extra_constraints=[found.regs.rax != keys[("api", "time", 1)]])
    assert list(found.solver.describe_variables(found.regs.ebx)) == [("api", "rand", 1)]
