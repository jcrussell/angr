"""An ARM ``svc`` is decoded in the target's byte order on both sides (angr-vmi2l.213).

Rust read the ``svc`` word little-endian on every ARM target, so on a
big-endian one an OABI ``svc #0x90000f`` dispatched on r7 and an EABI
``svc #0`` declined to Python. The Python syscall fallback resolved the
number before it set ``ip_at_syscall``, so ``SimCCARMLinuxSyscall.syscall_num``
read address 0 and ran ``restart_syscall`` on either byte order. Each case
runs ARM-mode code from a blob and compares both engines.
"""

from __future__ import annotations

import io

import archinfo
import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

BASE = 0x10000
MOV_R4_R0 = 0xE1A04000
B_SELF = 0xEAFFFFFE


def _project(endness: str, r7: int, svc: int):
    words = [0xE3A07000 | r7, svc, MOV_R4_R0, B_SELF]  # mov r7, #r7; svc; mov r4, r0; b .
    order = "big" if endness == "Iend_BE" else "little"
    code = b"".join(w.to_bytes(4, order) for w in words)
    opts = {"backend": "blob", "arch": archinfo.ArchARM(endness), "base_addr": BASE, "entry_point": BASE}
    return angr.Project(io.BytesIO(code), main_opts=opts, simos="linux", auto_load_libs=False)


def _run(engine: str, endness: str, r7: int, svc: int):
    project = _project(endness, r7, svc)
    state = project.factory.blank_state(addr=BASE)
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=BASE + 12, n=10)
    assert len(mgr.found) == 1
    r4 = mgr.found[0].regs.r4
    # A stubbed syscall's return is named syscall_stub_<name>_...
    result = next(iter(r4.variables)).split("_")[2] if r4.symbolic else mgr.found[0].solver.eval(r4)
    return result, mgr


@pytest.mark.parametrize("endness", ["Iend_BE", "Iend_LE"])
def test_oabi_svc_runs_the_immediates_syscall(endness):
    # svc #0x90000f is OABI chmod (15), whatever r7 holds; native declines chmod.
    expected, _ = _run("python", endness, 1, 0xEF90000F)
    assert expected == "chmod"
    assert _run("rust", endness, 1, 0xEF90000F)[0] == expected


def test_big_endian_eabi_svc_dispatches_natively_on_r7():
    # svc #0 with r7 = 20 is getpid.
    expected, _ = _run("python", "Iend_BE", 20, 0xEF000000)
    result, mgr = _run("rust", "Iend_BE", 20, 0xEF000000)
    assert result == expected == 1337
    assert mgr.stats.get("syscall_native_by_num", {}).get(20) == 1
