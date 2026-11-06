"""``x86g_use_seg_selector`` honours real mode on the Rust engine (angr-vmi2l.78).

With CR0.PE clear, Python's helper linearizes ``(sel << 4) + va``. CR0 comes
from ``state.globals["x86_cr0"]``, else ``arch.vex_archinfo["x86_cr0"]``. The
native helper serves only the protected-mode ``(sel << 16) + va`` of an empty
GDT, so the manager defers the call to Python unless PE is known set.
``mov eax, es:[0x10]`` with ``es = 0x1000`` reads 0x10010 in real mode and
0x10000010 in protected mode; each address holds a distinct marker.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

CODE = bytes.fromhex("26a110000000") + b"\xeb\xfe"  # mov eax, es:[0x10] ; jmp $
BASE = 0x7C00
REAL_MODE_MARKER = 0x1234
PROTECTED_MARKER = 0x5678


def _state(archinfo_cr0: int, globals_cr0: int | None):
    proj = angr.load_shellcode(CODE, arch="x86", load_address=BASE)
    proj.arch.vex_archinfo["x86_cr0"] = archinfo_cr0
    st = proj.factory.blank_state(addr=BASE)
    st.regs.es = 0x1000
    st.regs.eax = 0
    st.memory.store(0x10010, claripy.BVV(REAL_MODE_MARKER, 16), endness="Iend_LE")
    st.memory.store(0x10000010, claripy.BVV(PROTECTED_MARKER, 16), endness="Iend_LE")
    if globals_cr0 is not None:
        st.globals["x86_cr0"] = globals_cr0
    return proj, st


def _ax(found) -> int:
    return found.solver.eval(found.regs.eax) & 0xFFFF


@pytest.mark.parametrize(
    ("archinfo_cr0", "globals_cr0", "expected"),
    [
        pytest.param(0, None, REAL_MODE_MARKER, id="archinfo_real_mode"),
        pytest.param(0xFFFFFFFF, 0, REAL_MODE_MARKER, id="globals_real_mode"),
        pytest.param(0xFFFFFFFF, None, PROTECTED_MARKER, id="protected_mode"),
    ],
)
def test_seg_selector_matches_python(archinfo_cr0, globals_cr0, expected):
    proj, st = _state(archinfo_cr0, globals_cr0)
    simgr = proj.factory.simgr(st)
    simgr.explore(find=BASE + 6)
    assert _ax(simgr.found[0]) == expected

    proj, st = _state(archinfo_cr0, globals_cr0)
    mgr = RustExplorationManager(proj, [st])
    mgr.explore(find=BASE + 6)
    assert mgr.found, "the Rust engine never reached the jmp"
    assert _ax(mgr.found[0]) == expected
