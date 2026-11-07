"""Regression tests for RustExplorationManager._cb_resolve_function's PLT path.

The GOT-slot load width must follow arch.bytes, not a
hardcoded 8. On a 32-bit arch a GOT entry is 4 bytes wide; loading 8 pulls in
the neighbouring entry's bytes and corrupts the resolved pointer (angr-hv4lt.1).

The GOT slot itself follows the ``jmp`` operand's base register: RIP-relative on
x86-64, an absolute address on non-PIC i386, ``%ebx``-relative on PIC i386
(angr-vmi2l.245).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from capstone import x86 as cs_x86

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.common import bin_location


class _FakeProc:
    num_args = 2
    NO_RET = False


def _make_fake_self(arch_bytes: int, extern_addr: int, recorded: dict):
    """Build a minimal duck-typed object exercising the GOT-value fallback path.

    The 64-bit ``jmp`` is RIP-relative and the 32-bit one is non-PIC i386's
    absolute ``ff 25 <abs32>``, so ``recorded["addr"]`` is the slot either way.
    """

    plt_addr = 0x8048300
    got_addr = 0x804A000
    proc = _FakeProc()

    section = SimpleNamespace(name=".plt", min_addr=plt_addr, max_addr=plt_addr + 0x100)
    # No jmprel attribute -> got_to_sym stays empty -> falls into the GOT-value branch.
    obj = SimpleNamespace(sections=[section])

    if arch_bytes == 8:
        mem = SimpleNamespace(base=cs_x86.X86_REG_RIP, index=0, disp=got_addr - (plt_addr + 6))
    else:
        mem = SimpleNamespace(base=cs_x86.X86_REG_INVALID, index=0, disp=got_addr)
    op = SimpleNamespace(type=3, mem=mem)
    insn = SimpleNamespace(mnemonic="jmp", address=plt_addr, size=6, operands=[op])
    block = SimpleNamespace(capstone=SimpleNamespace(insns=[insn]))

    def _load(addr, size, endness=None):
        recorded["size"] = size
        recorded["addr"] = addr
        return object()  # opaque BV placeholder; eval is stubbed below

    state = SimpleNamespace(
        memory=SimpleNamespace(load=_load),
        solver=SimpleNamespace(eval=lambda _bv: extern_addr),
    )

    project = SimpleNamespace(
        _sim_procedures={extern_addr: proc},
        loader=SimpleNamespace(find_object_containing=lambda _a: obj),
        factory=SimpleNamespace(block=lambda addr, num_inst=1: block),
        arch=SimpleNamespace(bytes=arch_bytes, bits=arch_bytes * 8),
    )

    return SimpleNamespace(
        _project=project,
        _get_default_state=lambda: state,
    ), plt_addr


def test_got_load_width_follows_arch_bytes_32bit():
    recorded: dict = {}
    fake_self, plt_addr = _make_fake_self(arch_bytes=4, extern_addr=0x8049000, recorded=recorded)
    result = RustExplorationManager._cb_resolve_function(fake_self, plt_addr, None)
    assert recorded["size"] == 4, f"32-bit GOT load must be 4 bytes, got {recorded.get('size')}"
    assert recorded["addr"] == 0x804A000
    assert result == ("_FakeProc", 2, False)


def test_got_load_width_follows_arch_bytes_64bit():
    recorded: dict = {}
    fake_self, plt_addr = _make_fake_self(arch_bytes=8, extern_addr=0x400500, recorded=recorded)
    result = RustExplorationManager._cb_resolve_function(fake_self, plt_addr, None)
    assert recorded["size"] == 8, f"64-bit GOT load must be 8 bytes, got {recorded.get('size')}"
    assert recorded["addr"] == 0x804A000
    assert result == ("_FakeProc", 2, False)


@pytest.mark.parametrize("binary", ["fauxware", "fauxware_pie"])
def test_i386_plt_resolves_through_its_got_slot(binary):
    """A real i386 PLT entry resolves by its ``.got.plt`` slot, with no name to fall back on.

    ``fauxware``'s entry is ``jmp *abs32`` and ``fauxware_pie``'s is ``jmp *disp(%ebx)``.
    Reading either as RIP-relative misses ``jmprel``, so the call resolved to None.
    """
    path = os.path.join(bin_location, "tests", "i386", binary)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    project = angr.Project(path, auto_load_libs=False)
    state = project.factory.blank_state()
    fake_self = SimpleNamespace(_project=project, _get_default_state=lambda: state)
    puts_plt = project.loader.main_object.plt["puts"]
    assert RustExplorationManager._cb_resolve_function(fake_self, puts_plt, None) == ("puts", 1, False)
