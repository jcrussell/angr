# pylint: disable=missing-class-docstring,no-self-use,invalid-name
"""A stock proc whose cc is not the arch default runs in Python under the Rust engine (angr-vmi2l.125).

Python extracts a proc's arguments through ``proc.cc``. The native dispatcher
extracts through the arch-default convention (SysV on amd64), so a native
``strlen`` hooked with ``cc=SimCCMicrosoftAMD64`` read rdi instead of rcx.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.calling_conventions import SimCCMicrosoftAMD64, SimCCSystemVAMD64
from angr.exploration.rust_callback_dispatch import _native_dispatch_eligible
from tests.engines.conftest import BIN_LOCATION, RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

_LOAD = 0x400000
_HOOK = 0x500000
_RET = _LOAD + 12
_S = 0x602000
_A = 0x600000
# mov rax, _HOOK; call rax; jmp $
_CODE = bytes([0x48, 0xB8]) + _HOOK.to_bytes(8, "little") + bytes([0xFF, 0xD0, 0xEB, 0xFE])
_WIN_EXE = os.path.join(BIN_LOCATION, "tests", "x86_64", "windows", "fauxware.exe")


def _strlen_ret(engine, cc_cls):
    proj = angr.load_shellcode(_CODE, arch="AMD64", load_address=_LOAD)
    proto = angr.SIM_LIBRARIES["libc.so.6"][0].get_prototype("strlen")
    proj.hook(_HOOK, angr.SIM_PROCEDURES["libc"]["strlen"](cc=cc_cls(proj.arch), prototype=proto), replace=True)
    state = proj.factory.blank_state(
        addr=_LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    state.memory.store(_S, b"hello\0")
    state.memory.store(_A, b"x\0")
    state.regs.rcx = _S
    state.regs.rdi = _A
    state.regs.rsp = 0x7FFFFFFF0800
    if engine == "py":
        mgr = proj.factory.simulation_manager(state)
        counts = None
    else:
        mgr = RustExplorationManager(proj, [state])
    mgr.explore(find=_RET)
    if engine == "rust":
        counts = mgr._rust_mgr.native_procedure_stats()["call_counts"]
    assert len(mgr.found) == 1
    return mgr.found[0].solver.eval(mgr.found[0].regs.rax), counts


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize(("cc_cls", "expected"), [(SimCCMicrosoftAMD64, 5), (SimCCSystemVAMD64, 1)])
def test_strlen_reads_its_own_cc(engine, cc_cls, expected):
    ret, counts = _strlen_ret(engine, cc_cls)
    assert ret == expected
    if counts is not None:
        # The arch-default cc stays native; any other defers to Python.
        assert ("strlen" in counts) == (cc_cls is SimCCSystemVAMD64)


@pytest.mark.skipif(not os.path.exists(_WIN_EXE), reason="windows fauxware.exe not present")
def test_windows_x64_imports_and_cc_less_hooks_are_not_native():
    proj = angr.Project(_WIN_EXE, auto_load_libs=False)
    procs = list(proj._sim_procedures.values())
    assert any(isinstance(p.cc, SimCCMicrosoftAMD64) for p in procs)
    assert not any(_native_dispatch_eligible(p, proj) for p in procs if isinstance(p.cc, SimCCMicrosoftAMD64))
    # A cc-less hook gets the platform default (SimCCMicrosoftAMD64) when it executes.
    assert not _native_dispatch_eligible(angr.SIM_PROCEDURES["libc"]["strlen"](), proj)
