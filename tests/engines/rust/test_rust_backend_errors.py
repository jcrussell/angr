"""RustBackend errors: an errored record holds the SimError the Python engine raises (angr-mx00p.136.8).

An undecodable instruction errors with SimIRSBNoDecodeError and an NX fetch
with SimSegfaultException on both engines, the record's state is a SimState,
and resilience=False propagates the error out of step() as it does on the
Python engine. RustExplorationManager honors resilience= the same way
(angr-mx00p.132.43).
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import sim_options as o
from angr.errors import (
    SimError,
    SimIRSBNoDecodeError,
    SimMemoryError,
    SimOperationError,
    SimSegfaultException,
    SimZeroDivisionException,
)
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import RustErrorRecord, RustExplorationManager, sim_error_for_rust_message
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

LOAD = 0x400000
UD2 = b"\x0f\x0b\xeb\xfe"  # ud2; jmp $


def _ud2_project():
    return angr.load_shellcode(UD2, arch="amd64", load_address=LOAD)


def _ud2_state(project):
    return project.factory.blank_state(addr=LOAD, add_options={o.ZERO_FILL_UNCONSTRAINED_REGISTERS})


def _backend(project, rust):
    return RustBackend(project) if rust else None


@pytest.mark.parametrize("rust", [False, True], ids=["python", "rust"])
def test_undecodable_instruction_errors_as_no_decode(rust):
    project = _ud2_project()
    backend = _backend(project, rust)
    simgr = angr.SimulationManager(project, [_ud2_state(project)], backend=backend)
    # An until keeps the per-step path: without one, try_run takes the bounded run natively.
    simgr.run(n=3, until=lambda sm: False)
    assert len(simgr.errored) == 1
    (record,) = simgr.errored
    assert isinstance(record.error, SimIRSBNoDecodeError), type(record.error)
    assert isinstance(record.state, angr.SimState)
    assert record.state.addr == LOAD
    if rust:
        assert backend.stats["claimed"] == 1


@pytest.mark.parametrize("rust", [False, True], ids=["python", "rust"])
def test_resilience_false_propagates(rust):
    project = _ud2_project()
    simgr = angr.SimulationManager(project, [_ud2_state(project)], backend=_backend(project, rust), resilience=False)
    with pytest.raises(SimIRSBNoDecodeError):
        simgr.step()


@pytest.mark.parametrize("rust", [False, True], ids=["python", "rust"])
def test_nx_fetch_errors_as_segfault(rust):
    path = os.path.join(TEST_BINARIES_DIR, "fauxware")
    if not os.path.exists(path):
        pytest.skip(f"fauxware binary not found: {path}")
    project = angr.Project(path, auto_load_libs=False)
    state = project.factory.entry_state(add_options={o.STRICT_PAGE_ACCESS, o.ENABLE_NX})
    sp = state.solver.eval(state.regs.sp)
    state.regs.pc = sp  # the stack page is not executable
    backend = _backend(project, rust)
    simgr = angr.SimulationManager(project, [state], backend=backend)
    simgr.step()
    assert len(simgr.errored) == 1
    (record,) = simgr.errored
    assert isinstance(record.error, SimSegfaultException), type(record.error)
    assert (record.error.addr, record.error.reason) == (sp, "non-executable")
    assert isinstance(record.state, angr.SimState)
    if rust:
        assert backend.stats["claimed"] == 1


def test_rust_manager_record_holds_the_sim_error():
    """angr-vmi2l.252: RustExplorationManager's own errored records are typed too."""
    project = _ud2_project()
    mgr = RustExplorationManager(project, [_ud2_state(project)])
    mgr.run(n=3)
    (record,) = mgr.errored
    assert isinstance(record.error, SimIRSBNoDecodeError)
    assert record.message.startswith("IR decoding error at 0x400000")
    assert record.message in repr(record)


@pytest.mark.parametrize(
    ("resilience", "raises"),
    [(None, False), (True, False), ((SimIRSBNoDecodeError,), False), (False, True), ((SimSegfaultException,), True)],
)
@pytest.mark.parametrize("call", ["step", "run"])
def test_rust_manager_honors_resilience(resilience, raises, call):
    """The error a resilience does not cover leaves step() / run(), and the state is still in errored."""
    project = _ud2_project()
    mgr = RustExplorationManager(project, [_ud2_state(project)], resilience=resilience)
    stepped = mgr.step if call == "step" else mgr.run
    if raises:
        with pytest.raises(SimIRSBNoDecodeError):
            stepped()
    else:
        stepped()
    (record,) = mgr.errored
    assert isinstance(record.error, SimIRSBNoDecodeError)
    # Each error is raised once: the next step goes on with what is left.
    mgr.step()


def test_default_routed_resilience_manager_runs_on_rust():
    """A project-default route keeps a resilience= manager on Rust, and the manager honors the kwarg itself."""
    project = angr.load_shellcode(UD2, arch="amd64", load_address=LOAD, engine="rust")
    mgr = project.factory.simulation_manager(_ud2_state(project), resilience=False)
    assert isinstance(mgr.backend, RustBackend)
    with pytest.raises(SimIRSBNoDecodeError):
        mgr.step()
    stats = mgr.backend.stats
    assert stats["claimed"] == 1
    assert not stats["declined"]


@pytest.mark.parametrize(
    ("message", "expected", "segfault"),
    [
        ("IR decoding error at 0x400000: undecodable instruction.", SimIRSBNoDecodeError, None),
        ("memory error: unmapped memory at 0xdead0000 (size 4)", SimSegfaultException, (0xDEAD0000, "unmapped")),
        ("memory error: unmapped page at 0x5000 in mapped region", SimSegfaultException, (0x5000, "unmapped")),
        (
            "memory error: permission violation at 0x2000: required Permission { read: false, write: true, "
            "execute: false }, have Permission { read: true, write: false, execute: false }",
            SimSegfaultException,
            (0x2000, "non-writable"),
        ),
        (
            "memory error: permission violation at 0x2000: required Permission { read: false, write: true, "
            "execute: false }, have Permission { read: true, write: false, execute: true }",
            SimSegfaultException,
            (0x2000, "non-writable"),
        ),
        ("memory error: zero-size memory access at 0x10", SimMemoryError, None),
        ("operation error: shift overflow", SimOperationError, None),
        ("unexecutable VEX trap exit Ijk_SigFPE_IntDiv at 0x400683", SimZeroDivisionException, None),
        ("callback error: python raised", SimError, None),
    ],
)
def test_message_maps_to_python_error(message, expected, segfault):
    error = sim_error_for_rust_message(message, RustErrorRecord._classify(message))
    assert type(error) is expected
    if segfault is not None:
        assert (error.addr, error.reason) == segfault
