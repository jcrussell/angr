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
