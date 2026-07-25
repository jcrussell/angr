"""Regression tests for RustExplorationManager._cb_resolve_function GOT-value fallback.

Covers angr-hv4lt.1: the GOT-slot load width must follow arch.bytes, not a
hardcoded 8. On a 32-bit arch a GOT entry is 4 bytes wide; loading 8 pulls in
the neighbouring entry's bytes and corrupts the resolved pointer.
"""

from __future__ import annotations

from types import SimpleNamespace

from angr.exploration.rust_manager import RustExplorationManager


class _FakeProc:
    num_args = 2
    NO_RET = False


def _make_fake_self(arch_bytes: int, extern_addr: int, recorded: dict):
    """Build a minimal duck-typed object exercising the GOT-value fallback path."""

    plt_addr = 0x8048300
    got_addr = 0x804A000
    proc = _FakeProc()

    section = SimpleNamespace(name=".plt", min_addr=plt_addr, max_addr=plt_addr + 0x100)
    # No jmprel attribute -> got_to_sym stays empty -> falls into the GOT-value branch.
    obj = SimpleNamespace(sections=[section])

    op = SimpleNamespace(type=3, mem=SimpleNamespace(disp=got_addr - (plt_addr + 6)))
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
        arch=SimpleNamespace(bytes=arch_bytes),
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
    assert result == ("_FakeProc", 2, False)


def test_got_load_width_follows_arch_bytes_64bit():
    recorded: dict = {}
    fake_self, plt_addr = _make_fake_self(arch_bytes=8, extern_addr=0x400500, recorded=recorded)
    result = RustExplorationManager._cb_resolve_function(fake_self, plt_addr, None)
    assert recorded["size"] == 8, f"64-bit GOT load must be 8 bytes, got {recorded.get('size')}"
    assert result == ("_FakeProc", 2, False)
