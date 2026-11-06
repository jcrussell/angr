"""x87 ``stN`` / ``tagN`` names read and write their slot on a Rust state.

They are not archinfo registers: ``NameResolutionMixin`` resolves them
relative to ``ftop``, and ``SimLyingRegArg("st0")`` (the i386 float return
value ``Callable`` reads) loads one by name (angr-mx00p.132.76). Each name is
read on a SimProcedure hook state and on the exported found state after the
hook stores ``st1``, and compared with the Python engine on the same input.
"""

from __future__ import annotations

import os
import struct
from unittest import mock

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
CODE = bytes.fromhex("d9e8") + bytes.fromhex("d9eb") + bytes([0x90, 0x90, 0xEB, 0xFE])  # fld1; fldpi; nop; nop; jmp $
HOOK = LOAD + 4
FIND = LOAD + 5
TWO = struct.unpack("<Q", struct.pack("<d", 2.0))[0]

# st2 is left out: no instruction writes it, so its blank-state value is engine-specific.
NAMES = ("st0", "st1", "tag0", "tag1", "tag2")


def _probe(state):
    return {name: state.solver.eval_upto(state.registers.load(name), 2) for name in NAMES}


def _run(engine):
    # ANGR_DEFAULT_ENGINE=python keeps the reference run on the Python engine.
    with mock.patch.dict(os.environ, {"ANGR_DEFAULT_ENGINE": "python"}):
        proj = angr.load_shellcode(CODE, arch="X86", load_address=LOAD, engine=engine)
    seen = {}

    class Probe(angr.SimProcedure):
        def run(self):
            seen["hook"] = _probe(self.state)
            self.state.registers.store("st1", claripy.BVV(TWO, 64))
            self.jump(FIND)

    proj.hook(HOOK, Probe(), length=1)
    mgr = proj.factory.simulation_manager(proj.factory.blank_state(addr=LOAD))
    mgr.explore(find=FIND)
    assert len(mgr.found) == 1
    seen["found"] = _probe(mgr.found[0])
    seen["backend"] = type(getattr(mgr, "backend", None)).__name__
    return seen


@pytest.fixture(scope="module")
def python_answers():
    return _run(None)


@pytest.fixture(scope="module")
def rust_answers():
    return _run("rust")


def test_the_python_engine_reads_the_pushed_values(python_answers):
    pi = struct.unpack("<Q", struct.pack("<d", 3.141592653589793))[0]
    one = struct.unpack("<Q", struct.pack("<d", 1.0))[0]
    assert python_answers["hook"]["st0"] == [pi]
    assert python_answers["hook"]["st1"] == [one]
    assert python_answers["found"]["st1"] == [TWO]


def test_the_rust_backend_ran_the_states(rust_answers):
    assert rust_answers["backend"] == "RustBackend"


@pytest.mark.parametrize("where", ["hook", "found"])
def test_x87_slots_match_the_python_engine(python_answers, rust_answers, where):
    assert rust_answers[where] == python_answers[where]
