"""A Rust-minted memory filler is a tracked ``("mem", addr)`` variable (angr-vmi2l.321).

Python's ``DefaultFillerMixin`` registers each filler it mints for
never-written bytes under the temporal key ``("mem", addr)``, so
``state.solver.get_variables("mem", addr)`` finds the value the program read.
Rust installs its own filler in Rust memory; the found state's solver proxy
must yield it under the same key.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
X = 0x70000020

# mov ebx, [rdi]; jmp +0; mov al, [rdi+1]; jmp $ -- the second read slices
# the first read's filler.
CODE = b"\x8b\x1f" + b"\xeb\x00" + b"\x8a\x47\x01" + b"\xeb\xfe"
FIN = LOAD + len(CODE) - 2


def _found(engine: str):
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    state.regs.rdi = X
    if engine == "rust":
        mgr = RustExplorationManager(proj, [state])
    else:
        mgr = proj.factory.simulation_manager(state)
    mgr.explore(find=FIN)
    assert len(mgr.found) == 1
    return mgr.found[0]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_get_variables_mem_yields_the_filler_read(engine):
    found = _found(engine)
    keys = {}
    for key, var in found.solver.get_variables("mem"):
        keys[key] = var
    assert list(keys) == [("mem", X, 1)]
    ebx_filler = keys[("mem", X, 1)]
    assert ebx_filler.size() == 32
    # The tracked variable is the value the program read into ebx.
    assert not found.solver.satisfiable(extra_constraints=[found.regs.ebx != ebx_filler])
    assert list(found.solver.describe_variables(found.regs.ebx)) == [("mem", X, 1)]
