"""Native syscall dispatch table vs angr's own syscall numbering (angr-mx00p.35).

The Rust engine dispatches a syscall by ``(ABI, number)`` to a native handler.
A number that resolves to the wrong handler is silently wrong, so this module
pins the table two ways:

* ``test_table_matches_golden``: the table the manager reports equals the
  committed golden ``native_syscalls_golden.json``, both directions, per ABI.
  The golden is scaffolding for moving the numbers out of Rust: it was dumped
  from the hand-written per-arch tables and proves the by-name registry
  resolves to exactly the same rows.
* ``test_every_row_names_the_handlers_syscall``: every ``(ABI, number)`` row
  exists in angr's ``SimSyscallLibrary.syscall_number_mapping`` and the name
  angr gives that number is the handler's syscall (or one of the documented
  aliases that share a handler).

Regenerate the golden with ``ANGR_REGEN_SYSCALL_GOLDEN=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import angr

try:
    from angr.exploration import RustExplorationManager
except ImportError:  # pragma: no cover
    RustExplorationManager = None

pytestmark = pytest.mark.skipif(RustExplorationManager is None, reason="Rust engine not built")

GOLDEN = Path(__file__).with_name("native_syscalls_golden.json")

# angr syscall ABI -> (archinfo arch, simos) of a project angr gives that ABI.
ABI_PROJECTS = {
    "amd64": ("AMD64", "linux"),
    "i386": ("X86", "linux"),
    "arm": ("ARMEL", "linux"),
    "aarch64": ("AARCH64", "linux"),
    "mips-o32": ("MIPS32", "linux"),
    "mips-n64": ("MIPS64", "linux"),
    "cgcabi": ("X86", "cgc"),
}

# (handler name, angr syscall name) pairs where one handler serves a syscall
# angr spells differently. Anything else must match by name.
HANDLER_ALIASES = {
    ("exit", "exit_group"),
    ("exit", "_terminate"),
    ("stat", "stat64"),
    ("lstat", "lstat64"),
    ("fstat", "fstat64"),
    ("newfstatat", "fstatat64"),
    ("getrlimit", "ugetrlimit"),
    ("getuid", "getuid32"),
    ("getgid", "getgid32"),
    ("geteuid", "geteuid32"),
    ("getegid", "getegid32"),
    ("setuid", "setuid32"),
    ("setgid", "setgid32"),
    # angr's arm and mips-o32 tables call the struct-argument mmap "mmap";
    # i386 calls the same syscall "old_mmap".
    ("old_mmap", "mmap"),
}


def _project(abi):
    arch, simos = ABI_PROJECTS[abi]
    return angr.load_shellcode(b"\x00" * 16, arch=arch, simos=simos, load_address=0x1000)


def _native_table(abi):
    proj = _project(abi)
    mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
    return {int(n): name for n, name in mgr._rust_mgr.native_syscall_table().get(abi, {}).items()}


def _load_golden():
    return {abi: {int(n): name for n, name in rows.items()} for abi, rows in json.loads(GOLDEN.read_text()).items()}


def test_regen_golden_when_requested():
    if not os.environ.get("ANGR_REGEN_SYSCALL_GOLDEN"):
        pytest.skip("set ANGR_REGEN_SYSCALL_GOLDEN=1 to rewrite the golden")
    golden = {abi: {str(n): name for n, name in sorted(_native_table(abi).items())} for abi in ABI_PROJECTS}
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=False) + "\n")


@pytest.mark.parametrize("abi", sorted(ABI_PROJECTS))
def test_table_matches_golden(abi):
    expected = _load_golden()[abi]
    actual = _native_table(abi)
    missing = {n: expected[n] for n in expected.keys() - actual.keys()}
    extra = {n: actual[n] for n in actual.keys() - expected.keys()}
    changed = {n: (expected[n], actual[n]) for n in expected.keys() & actual.keys() if expected[n] != actual[n]}
    assert not (missing or extra or changed), f"{abi}: missing={missing} extra={extra} changed={changed}"


@pytest.mark.parametrize("abi", sorted(ABI_PROJECTS))
def test_every_row_names_the_handlers_syscall(abi):
    library = _project(abi).simos.syscall_library
    mapping = library.syscall_number_mapping[abi]
    wrong = {}
    for num, handler in _load_golden()[abi].items():
        angr_name = mapping.get(num)
        if angr_name != handler and (handler, angr_name) not in HANDLER_ALIASES:
            wrong[num] = (handler, angr_name)
    assert not wrong, f"{abi}: number -> (native handler, angr's name): {wrong}"


def _exit_shellcode_project(simos=None):
    # mov eax, 60 ; mov edi, 7 ; syscall  (amd64 exit(7))
    sc = bytes.fromhex("b83c000000bf070000000f05")
    kwargs = {"simos": simos} if simos else {}
    return angr.load_shellcode(sc, arch="AMD64", load_address=0x1000, **kwargs)


def test_plain_simos_has_no_native_syscalls():
    """A SimOS without a syscall library gets no native syscall handlers.

    Python's engine has no syscall procedures for a plain ``SimOS``
    (``SimOS.syscall`` returns None) and errors the state, so the Rust engine
    must neither run its Linux handlers there nor resume past the syscall.
    """
    proj = _exit_shellcode_project()
    assert getattr(proj.simos, "syscall_library", None) is None

    simgr = proj.factory.simulation_manager(proj.factory.entry_state(addr=0x1000))
    simgr.run()
    assert len(simgr.errored) == 1 and not simgr.deadended
    python_error = str(simgr.errored[0].error)

    mgr = RustExplorationManager(proj, [proj.factory.entry_state(addr=0x1000)])
    assert mgr._rust_mgr.native_syscall_table() == {}
    mgr.run()
    assert not mgr.deadended, "exit(7) must not run natively under a plain SimOS"
    assert len(mgr.errored) == 1
    assert python_error in str(mgr.errored[0])


def test_linux_simos_keeps_native_syscalls():
    proj = _exit_shellcode_project("linux")
    mgr = RustExplorationManager(proj, [proj.factory.entry_state(addr=0x1000)])
    assert mgr._rust_mgr.native_syscall_table()["amd64"][60] == "exit"
    mgr.run()
    assert len(mgr.deadended) == 1


def test_amd64_int_0x80_is_not_the_amd64_abi():
    """``int 0x80`` in 64-bit code must not run the amd64 handler for its number.

    eax=1 is i386 ``exit`` but amd64 ``write``. The Rust engine used to look
    it up in the amd64 table and keep executing (angr-mx00p.57); now it
    defers to the SimOS, which stops the state at the ``int 0x80`` just as the
    Python engine does.
    """
    # mov eax,1 ; xor ebx,ebx ; int 0x80 ; mov eax,60 ; mov edi,7 ; syscall
    sc = bytes.fromhex("b80100000031dbcd80b83c000000bf070000000f05")
    proj = angr.load_shellcode(sc, arch="AMD64", simos="linux", load_address=0x1000)
    after_int80 = 0x1009

    simgr = proj.factory.simulation_manager(proj.factory.entry_state(addr=0x1000))
    simgr.run()
    assert not simgr.deadended and not simgr.active

    mgr = RustExplorationManager(proj, [proj.factory.entry_state(addr=0x1000)])
    mgr.run()
    assert not mgr.deadended, "the int 0x80 ran as amd64 write and execution reached the exit syscall"
    assert [s.addr for s in mgr.errored] == [after_int80]
