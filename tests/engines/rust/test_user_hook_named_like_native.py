# pylint: disable=missing-class-docstring,no-self-use,invalid-name
"""A user ``hook_symbol`` override named like a native proc runs its own ``run()`` (angr-vmi2l.121).

``SimProcedure`` defaults ``display_name`` to the class name, and the native
registry is keyed on that name, so a user ``class rand`` hooked over the
``rand`` import used to be served by the native ``rand``. Only a stock angr
proc may be served natively; any other hook bounces to Python under the Rust
engine, as it runs under Python's.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_callback_dispatch import _native_dispatch_eligible, _simproc_dispatch_name
from tests.engines.conftest import BIN_LOCATION, RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = [
    pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built"),
    pytest.mark.skipif(
        not os.path.exists(os.path.join(BIN_LOCATION, "tests", "x86_64", "mod_0")), reason="mod_0 not present"
    ),
]

_BINARY = os.path.join(BIN_LOCATION, "tests", "x86_64", "mod_0")
# main's `call rand@plt` returns to main + 0x11.
_AFTER_RAND = 0x11


class rand(angr.SimProcedure):
    def run(self):
        return 7


class MyRand(angr.SimProcedure):
    def run(self):
        return 7


def _run_to_after_rand(engine, proc, disable=None):
    proj = angr.Project(_BINARY, auto_load_libs=False)
    if proc is not None:
        proj.hook_symbol("rand", proc, replace=True)
    main = proj.loader.find_symbol("main").rebased_addr
    state = proj.factory.call_state(main)
    if engine == "py":
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=main + _AFTER_RAND)
        counts = None
    else:
        mgr = RustExplorationManager(proj, [state])
        if disable is not None:
            mgr.disable_native_procedure(disable)
        mgr.explore(find=main + _AFTER_RAND)
        counts = mgr._rust_mgr.native_procedure_stats()["call_counts"]
    assert len(mgr.found) == 1
    return mgr.found[0], counts


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize("make", [rand, MyRand, lambda: MyRand(display_name="rand")], ids=["rand", "MyRand", "dn"])
def test_user_override_runs(engine, make):
    found, counts = _run_to_after_rand(engine, make())
    assert found.solver.eval_upto(found.regs.eax, 2) == [7]
    if counts is not None:
        assert "rand" not in counts


def test_stock_rand_stays_native():
    found, counts = _run_to_after_rand("rust", None)
    assert found.regs.eax.symbolic
    assert counts.get("rand") == 1


def test_disable_native_procedure_opt_out():
    found, counts = _run_to_after_rand("rust", None, disable="rand")
    assert found.regs.eax.symbolic
    assert "rand" not in counts


def test_dispatch_name():
    stock = angr.SIM_PROCEDURES["libc"]["rand"]()
    assert _native_dispatch_eligible(stock)
    assert _simproc_dispatch_name(stock) == "rand"
    assert not _native_dispatch_eligible(rand())
    assert _simproc_dispatch_name(rand()) == "python:rand"
    # Constructor kwargs reach run(), which the native proc would ignore.
    assert not _native_dispatch_eligible(angr.SIM_PROCEDURES["stubs"]["ReturnUnconstrained"](return_val=3))
