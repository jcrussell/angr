"""A ``mem_read`` / ``mem_write`` override the Rust engine cannot use warns.

A BP action that sets ``mem_read_expr`` / ``mem_write_expr`` injects a value
(angr-uy32, angr-inh0). The engine keeps its original value when the override
is the wrong width, or when the importer rejects it (a Python int, say), since
either would corrupt the ops downstream. It used to do so silently, so the user
had no sign that the value they asked for was never used. Each rejection now
logs at warn, naming the event (angr-vmi2l.205).
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD, BUF = 0x400000, 0x80000
LOAD_EAX = bytes.fromhex("8b07")  # mov eax, dword ptr [rdi]
STORE_ESI = bytes.fromhex("8937")  # mov dword ptr [rdi], esi
JMP_SELF = bytes.fromhex("ebfe")
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
OLD, NEW = 0x11112222, 0x33334444


def _run(code, event, when, inject):
    project = angr.load_shellcode(code + JMP_SELF, arch="AMD64", load_address=LOAD)
    state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
    state.regs.rdi = BUF
    state.regs.esi = NEW
    state.memory.store(BUF, claripy.BVV(OLD, 32), endness="Iend_LE")
    fired = []

    def action(st):
        fired.append(event)
        setattr(st.inspect.attrs, f"{event}_expr", inject)

    state.inspect.b(event, when=when, action=action)
    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=LOAD + len(code), n=10)
    assert fired, "the BP never fired; the test is vacuous"
    assert len(mgr.found) == 1
    found = mgr.found[0]
    return found.solver.eval(found.regs.eax), found.solver.eval(found.memory.load(BUF, 4, endness="Iend_LE"))


@pytest.mark.parametrize(
    ("inject", "why"),
    [(claripy.BVV(0x41, 8), "override is 8 bits, not the 32-bit access"), (0x41, "override 65 rejected")],
    ids=["wrong-width", "not-an-ast"],
)
def test_rejected_mem_read_override_keeps_the_load_and_warns(capfd, inject, why):
    capfd.readouterr()
    eax, _ = _run(LOAD_EAX, "mem_read", angr.BP_AFTER, inject)
    err = capfd.readouterr().err
    assert eax == OLD
    assert "state.inspect 'mem_read'" in err and why in err, err


@pytest.mark.parametrize(
    ("inject", "why"),
    [(claripy.BVV(0x41, 8), "override is 8 bits, not the 32-bit access"), (0x41, "override 65 rejected")],
    ids=["wrong-width", "not-an-ast"],
)
def test_rejected_mem_write_override_keeps_the_store_and_warns(capfd, inject, why):
    capfd.readouterr()
    _, stored = _run(STORE_ESI, "mem_write", angr.BP_BEFORE, inject)
    err = capfd.readouterr().err
    assert stored == NEW
    assert "state.inspect 'mem_write'" in err and why in err, err


def test_an_accepted_override_is_used_and_does_not_warn(capfd):
    capfd.readouterr()
    eax, _ = _run(LOAD_EAX, "mem_read", angr.BP_AFTER, claripy.BVV(0x41, 32))
    err = capfd.readouterr().err
    assert eax == 0x41
    assert "state.inspect 'mem_read'" not in err, err
