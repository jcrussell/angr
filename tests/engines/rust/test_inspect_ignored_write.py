"""A BP write to an inspect attr the Rust engine never reads back warns once.

Python angr reads back attrs such as ``reg_write_expr`` after a BP fires. The
Rust engine reads back only each event's one injectable attr (``mem_read_expr``,
``mem_write_expr``, ``dirty_result``, ``added_constraints``), so a write to any
other attr kept the original value with no sign (angr-vmi2l.206). It now warns
once per event and attribute.
"""

from __future__ import annotations

import logging

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOGGER = "angr.exploration.rust_manager"
LOAD, BUF = 0x400000, 0x80000
# mov eax, esi ; mov ecx, dword ptr [rdi] ; mov eax, esi ; jmp $
CODE = bytes.fromhex("89f08b0f89f0")
JMP_SELF = bytes.fromhex("ebfe")
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
OLD, NEW = 0x11112222, 0x33334444


def _run(event, when, attr, value):
    project = angr.load_shellcode(CODE + JMP_SELF, arch="AMD64", load_address=LOAD)
    state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
    state.regs.rdi = BUF
    state.regs.esi = NEW
    state.memory.store(BUF, claripy.BVV(OLD, 32), endness="Iend_LE")
    fired = []

    def action(st):
        fired.append(event)
        if attr is not None:
            setattr(st.inspect, attr, value)

    state.inspect.b(event, when=when, action=action)
    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=LOAD + len(CODE), n=10)
    assert fired, "the BP never fired; the test is vacuous"
    assert len(mgr.found) == 1
    found = mgr.found[0]
    return len(fired), found.solver.eval(found.regs.eax)


def _ignored_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "is ignored by the Rust engine" in r.getMessage()]


def test_reg_write_expr_write_keeps_the_value_and_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        fired, eax = _run("reg_write", angr.BP_AFTER, "reg_write_expr", claripy.BVV(0, 64))
    warnings = _ignored_warnings(caplog)
    assert fired > 1, "need several fires to show the warning is shown once"
    assert eax == NEW
    assert len(warnings) == 1, warnings
    assert "inspect reg_write" in warnings[0] and "reg_write_expr" in warnings[0], warnings


def test_non_injectable_attr_of_an_injectable_event_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run("mem_read", angr.BP_AFTER, "mem_read_address", claripy.BVV(0, 64))
    warnings = _ignored_warnings(caplog)
    assert len(warnings) == 1 and "mem_read_address" in warnings[0], warnings


@pytest.mark.parametrize(
    ("event", "attr", "value"),
    [
        ("reg_write", None, None),
        ("mem_read", "mem_read_expr", claripy.BVV(0x41, 32)),
    ],
    ids=["untouched", "injected"],
)
def test_untouched_or_injected_attr_does_not_warn(caplog, event, attr, value):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run(event, angr.BP_AFTER, attr, value)
    assert not _ignored_warnings(caplog)
