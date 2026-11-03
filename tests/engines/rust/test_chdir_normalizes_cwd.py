"""A native chdir normalizes its path and the cwd reaches ``state.fs.cwd`` (angr-vmi2l.136).

Python's ``chdir`` goes through ``SimFilesystem.chdir``: a relative path is
joined onto the cwd and ``.``/``..`` are folded, root-clamped. The native
handler used to store the raw C-string, so ``getcwd`` returned ``sub`` and a
later relative ``open`` resolved against a relative cwd; the exported state
also kept the old ``fs.cwd``. Each case runs chdir and getcwd, and compares
with the Python engine. The relative-open resolution is pinned by the cargo
test ``chdir_normalizes_relative_and_dotted_paths`` and the py-vs-rust
``test_open_read_inserted_simfile.py``.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
DATA = 0x30000000
PATH = DATA
CWD_BUF = DATA + 0x200


def _mov_rdi(v: int) -> bytes:
    return b"\x48\xbf" + struct.pack("<Q", v)


def _sys(nr: int) -> bytes:
    return b"\xb8" + struct.pack("<I", nr) + b"\x0f\x05"


CODE = (
    _mov_rdi(PATH) + _sys(80)  # chdir(PATH)
    + _mov_rdi(CWD_BUF) + b"\xbe\x00\x01\x00\x00" + _sys(79)  # getcwd(CWD_BUF, 256)
)  # fmt: skip
END = LOAD + len(CODE)


def _run(engine: str, path: bytes):
    proj = angr.load_shellcode(CODE + b"\x90\xeb\xfe", arch="amd64", simos="linux", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    state.memory.store(DATA, b"\0" * 0x300)
    state.memory.store(PATH, path + b"\0")
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=END, n=20)
    assert len(mgr.found) == 1
    (found,) = mgr.found
    getcwd = found.solver.eval(found.memory.load(CWD_BUF, 0x40), cast_to=bytes).split(b"\0")[0]
    return getcwd, found.fs.cwd


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (b"sub", b"/home/user/sub"),
        (b"./sub/../sub/", b"/home/user/sub"),
        (b"/a/b/../c/", b"/a/c"),
        (b"../../../..", b"/"),
    ],
    ids=["relative", "dotted_relative", "dotdot_absolute", "root_clamped"],
)
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_chdir_normalizes_cwd(engine, path, expected):
    getcwd, fs_cwd = _run(engine, path)
    assert getcwd == expected
    assert fs_cwd == expected
