"""A hook whose ``NO_RET`` is ``None`` registers with Rust as "may return" (angr-vmi2l.133).

Upstream's libc ``error`` declares ``NO_RET = None`` (``DYNAMIC_RET``: whether
it returns depends on ``status``). The hook registration tuple handed that
``None`` to a PyO3 ``bool``, so the manager raised ``TypeError`` on every
binary importing ``error``. Each case hooks such a procedure, before or after
the manager is built, and checks that the call returns as under Python.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
HOOK = LOAD + 0x100
# call HOOK; mov rbx, rax
CODE = b"\xe8" + (HOOK - (LOAD + 5)).to_bytes(4, "little", signed=True) + b"\x48\x89\xc3"
BINARIES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "binaries", "tests", "x86_64")


class MaybeReturns(angr.SimProcedure):
    NO_RET = None

    def run(self):  # pylint:disable=arguments-differ
        return 7


def _run(engine, hook_after_construction):
    code = CODE + b"\xeb\xfe" + b"\x90" * (HOOK - LOAD - len(CODE) - 2) + b"\xc3"
    project = angr.load_shellcode(code, arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD)
    if not hook_after_construction:
        project.hook(HOOK, MaybeReturns())
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    if hook_after_construction:
        project.hook(HOOK, MaybeReturns())
    mgr.explore(find=LOAD + len(CODE), n=20)
    assert len(mgr.found) == 1, (engine, mgr.errored)
    return mgr.found[0].solver.eval(mgr.found[0].regs.rbx)


@pytest.mark.parametrize("hook_after_construction", [False, True])
def test_no_ret_none_hook_returns_like_python(hook_after_construction):
    assert _run("rust", hook_after_construction) == _run("python", hook_after_construction) == 7


def test_manager_runs_a_binary_importing_error():
    path = os.path.join(BINARIES, "true")
    if not os.path.exists(path):
        pytest.skip("binaries/tests/x86_64/true not present")
    project = angr.Project(path, auto_load_libs=False)
    assert any(getattr(p, "NO_RET", False) is None for p in project._sim_procedures.values())
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    mgr.run(n=50)
    assert not mgr.errored
    assert len(mgr.deadended) == 1
