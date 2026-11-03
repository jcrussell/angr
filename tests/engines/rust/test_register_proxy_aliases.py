"""archinfo register names read their modelled bytes on a Rust state.

``get_state_register`` answers ``None`` both for a symbolic register and for
a name the engine's table lacks, and the proxy used to turn the second case
into a free ``BVS`` of ``arch.bits`` width (angr-vmi2l.212). Each name below
is read on a SimProcedure hook state, a find predicate and the exported found
state, and compared with the Python engine on the same input. A name archinfo
does not know raises ``AttributeError``, as ``SimState.regs`` does.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
CODE = (
    bytes.fromhex("49b83412000000000000")  # mov r8, 0x1234
    + bytes.fromhex("48c7c5bbaa0000")  # mov rbp, 0xaabb
    + bytes.fromhex("c5fd76c0")  # vpcmpeqd ymm0, ymm0, ymm0
    + bytes.fromhex("490f6ec0")  # movq mm0, r8
    + bytes([0x90, 0x90, 0xEB, 0xFE])  # nop; nop; jmp $
)
HOOK = LOAD + 25
FIND = LOAD + 26

# r8w and the xmm/ymm views are table rows; bph, dih and mm0 are slices of
# a wider table register (ebp, di, fpreg).
NAMES = ("r8w", "xmm0lq", "xmm0hq", "ymm0", "ymm0hx", "bph", "dih", "mm0")


def _probe(state):
    out = {}
    for name in NAMES:
        value = getattr(state.regs, name)
        out[name] = (value.size(), sorted(state.solver.eval_upto(value, 2)))
    # dih is bits 15..8 of the symbolic rdi, not a free symbol.
    out["dih_is_rdi"] = sorted(state.solver.eval_upto(state.regs.dih == state.regs.rdi[15:8], 2))
    out["has_typo"] = hasattr(state.regs, "raxx")
    return out


def _run(engine):
    from angr.exploration.rust_manager import RustExplorationManager

    proj = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    seen = {}

    class Probe(angr.SimProcedure):
        def run(self):
            seen["hook"] = _probe(self.state)
            self.jump(FIND)

    proj.hook(HOOK, Probe(), length=1)

    def pred(s):
        if s.addr == FIND:
            seen["pred"] = _probe(s)
            return True
        return False

    state = proj.factory.blank_state(addr=LOAD)
    state.regs.rdi = claripy.BVS("x", 64)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=pred)
    assert len(mgr.found) == 1
    seen["found"] = _probe(mgr.found[0])
    seen["found_state"] = mgr.found[0]
    return seen


@pytest.fixture(scope="module")
def python_answers():
    return _run("python")


@pytest.fixture(scope="module")
def rust_answers():
    return _run("rust")


@pytest.mark.parametrize("site", ["hook", "pred", "found"])
def test_register_aliases_match_python(python_answers, rust_answers, site):
    assert rust_answers[site] == python_answers[site]


def test_expected_values(rust_answers):
    found = rust_answers["found"]
    assert found["r8w"] == (16, [0x1234])
    assert found["bph"] == (8, [0xAA])
    assert found["xmm0hq"] == (64, [(1 << 64) - 1])
    assert found["ymm0"] == (256, [(1 << 256) - 1])
    assert found["mm0"] == (64, [0x1234])
    assert found["dih_is_rdi"] == [True]
    assert found["has_typo"] is False


def test_unknown_name_raises(rust_answers):
    regs = rust_answers["found_state"].regs
    with pytest.raises(AttributeError):
        _ = regs.raxx


def test_unmodelled_archinfo_register_raises(rust_answers):
    regs = rust_answers["found_state"].regs
    with pytest.raises(NotImplementedError):
        _ = regs.cr0
