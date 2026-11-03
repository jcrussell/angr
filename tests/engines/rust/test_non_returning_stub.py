# pylint: disable=missing-class-docstring,no-self-use
"""A call to a ``non_returning`` libc stub ends its path under both engines (angr-vmi2l.122).

``SimLibrary._apply_metadata`` marks a library's ``non_returning`` names
(``longjmp``, ``pthread_exit``, ...) with ``proc.returns = False`` and leaves
``NO_RET`` False. With ``auto_load_libs=False`` ``longjmp`` has no model, so it
is a ``ReturnUnconstrained`` stub, and Python's procedure engine emits no
successor for it: the state deadends at the stub. The Rust engine must neither
serve the stub natively as a returning call nor resume past it after a
bounce into Python.

The binary also imports libc ``error``, whose ``NO_RET`` is None (it is
``DYNAMIC_RET``), so building the manager covers the registration tuple's
``no_return`` being a real bool.
"""

from __future__ import annotations

import os

import pytest

import angr
from tests.engines.conftest import BIN_LOCATION, RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

_BINARY = os.path.join(BIN_LOCATION, "tests", "x86_64", "dir_gcc_-O0")
# Just ahead of a `call longjmp@plt`.
_START = 0x407361


@pytest.mark.skipif(not os.path.exists(_BINARY), reason="dir_gcc_-O0 not present")
@pytest.mark.parametrize(("engine", "mode"), [("py", "step"), ("rust", "run"), ("rust", "step")])
def test_longjmp_stub_deadends(engine, mode):
    proj = angr.Project(_BINARY, auto_load_libs=False)
    stub = proj.loader.find_symbol("longjmp").rebased_addr
    hook = proj.hooked_by(stub)
    assert hook.returns is False and not hook.NO_RET
    state = proj.factory.blank_state(
        addr=_START,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    state.regs.rsp = 0x7FFFFFFF0800
    state.regs.rbp = 0x7FFFFFFF0900
    mgr = proj.factory.simulation_manager(state) if engine == "py" else RustExplorationManager(proj, [state])
    for _ in range(4):
        if mode == "run":
            mgr.run(max_steps=1)
        else:
            mgr.step()
    stashes = {k: len(v) for k, v in mgr.stashes.items() if v}
    assert stashes == {"deadended": 1}, stashes
    assert mgr.deadended[0].addr == stub
    if engine == "rust":
        assert "longjmp" not in mgr._rust_mgr.native_procedure_stats()["call_counts"]
