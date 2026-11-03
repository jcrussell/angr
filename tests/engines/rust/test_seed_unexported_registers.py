"""Seed import carries register bytes no exported engine register covers.

Seed import used to iterate the engine's ``exported`` register names only, so
a seed's write to the x87/MMX file (engine ``fpreg``, archinfo ``fpu_regs``)
or to a ymm upper lane read 0 on every Rust state (angr-vmi2l.320). Each
seeded register below is read on a SimProcedure hook state, a find predicate
and the exported found state, and compared with the Python engine.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
CODE = bytes([0x90, 0x90, 0x90, 0xEB, 0xFE])  # nop; nop; nop; jmp $
HOOK = LOAD + 1
FIND = LOAD + 2

SEEDS = {
    "AMD64": {"mm0": 0x5566, "mm3": "sym", "ymm0hx": 0x77 << 100, "zmm1hy": 0x99},
    "X86": {"mm0": 0x5566, "mm7": "sym"},
}


def _seed(state, arch):
    for name, value in SEEDS[arch].items():
        width = getattr(state.regs, name).size()
        setattr(state.regs, name, claripy.BVS(name, width) if value == "sym" else claripy.BVV(value, width))


def _probe(state, arch):
    out = {}
    for name, value in SEEDS[arch].items():
        reg = getattr(state.regs, name)
        if value == "sym":
            # The seed's symbol itself, not a fresh one or a constant.
            out[name] = sorted(state.solver.eval_upto(reg == claripy.BVS(name, reg.size()), 2))
            out[name + "_vars"] = sorted(v.split("_")[0] for v in reg.variables)
        else:
            out[name] = sorted(state.solver.eval_upto(reg, 2))
    return out


def _run(engine, arch):
    from angr.exploration.rust_manager import RustExplorationManager

    proj = angr.load_shellcode(CODE, arch=arch, load_address=LOAD)
    seen = {}

    class Probe(angr.SimProcedure):
        def run(self):
            seen["hook"] = _probe(self.state, arch)
            self.jump(FIND)

    proj.hook(HOOK, Probe(), length=1)

    def pred(s):
        if s.addr == FIND:
            seen["pred"] = _probe(s, arch)
            return True
        return False

    state = proj.factory.blank_state(addr=LOAD)
    _seed(state, arch)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=pred)
    assert len(mgr.found) == 1
    seen["found"] = _probe(mgr.found[0], arch)
    return seen


@pytest.fixture(scope="module", params=sorted(SEEDS))
def answers(request):
    return request.param, _run("python", request.param), _run("rust", request.param)


@pytest.mark.parametrize("site", ["hook", "pred", "found"])
def test_seeded_registers_match_python(answers, site):
    _, python, rust = answers
    assert rust[site] == python[site]


def test_seeded_values(answers):
    arch, _, rust = answers
    found = rust["found"]
    assert found["mm0"] == [0x5566]
    if arch == "AMD64":
        assert found["ymm0hx"] == [0x77 << 100]
        assert found["zmm1hy"] == [0x99]
        assert found["mm3"] == [False, True]
    else:
        assert found["mm7"] == [False, True]
