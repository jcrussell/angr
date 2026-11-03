"""Native strtol's ``*endptr`` matches the Python engine (angr-vmi2l.101).

With a symbolic digit window the native proc stored one concrete end (the end
of the window), so ``end == nptr`` and ``*end != '\\0'`` checks took only the
"valid" arm. A no-conversion input stored ``nptr`` plus the skipped
whitespace and sign, where C and Python store ``nptr``. Each case calls the
libc ``strtol`` hook from amd64 shellcode and compares the set of ``*endptr``
values under both engines.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
HOOK = 0x500000
NPTR = 0x600000
ENDP = 0x602000
# mov rax, HOOK ; call rax ; jmp $
CODE = b"\x48\xb8" + HOOK.to_bytes(8, "little") + b"\xff\xd0\xeb\xfe"
RET = LOAD + 12

EXPECTED = {
    "short_run": [NPTR + 2, NPTR + 3, NPTR + 4],
    "symbolic_no_digit": [NPTR],
    "concrete_no_digit": [NPTR],
}


def _endptrs(engine: str, case: str) -> list[int]:
    proj = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    state = proj.factory.blank_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    state.regs.rsp = 0x7FFFFFFF0800
    if case == "concrete_no_digit":
        state.memory.store(NPTR, b"  -x\0")
    else:
        digits = [claripy.BVS(f"d{i}", 8) for i in range(4)]
        for i, d in enumerate(digits):
            state.memory.store(NPTR + i, d)
        state.memory.store(NPTR + 4, claripy.BVV(0, 8))
        if case == "symbolic_no_digit":
            state.add_constraints(digits[0] == ord("x"))
        else:
            for d in digits[:2]:
                state.add_constraints(d >= ord("0"), d <= ord("9"))
    state.memory.store(ENDP, claripy.BVV(0x4141, 64))
    state.regs.rdi, state.regs.rsi, state.regs.rdx = NPTR, ENDP, 10
    proto = angr.SIM_LIBRARIES["libc.so.6"][0].get_prototype("strtol")
    proj.hook(HOOK, angr.SIM_PROCEDURES["libc"]["strtol"](prototype=proto), replace=True)

    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=RET, n=10)
    if engine == "rust":
        assert mgr._rust_mgr.native_procedure_stats()["call_counts"].get("strtol", 0) >= 1
    assert len(mgr.found) == 1
    (found,) = mgr.found
    end = found.memory.load(ENDP, 8, endness="Iend_LE")
    return sorted(found.solver.eval_upto(end, 8))


@pytest.mark.parametrize("case", sorted(EXPECTED))
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_strtol_endptr_values(engine: str, case: str) -> None:
    assert _endptrs(engine, case) == EXPECTED[case]
