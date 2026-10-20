"""Native syscall dispatch vs angr's own syscall numbering (angr-mx00p.35).

The Rust engine registers its syscall handlers by the name angr's syscall
library gives each syscall (``BY_NAME`` in ``native/angr/src/syscalls/mod.rs``)
and takes the numbers from the project's SimOS. A number that resolves to the
wrong handler is silently wrong, so this module checks the live table against
angr's:

* ``test_every_row_names_the_handlers_syscall``: for each ABI, every number the
  manager dispatches natively is one angr's ``syscall_number_mapping`` assigns,
  and angr's name for it is the handler's syscall (or one of the documented
  aliases that share a handler).
* ``test_every_registration_names_a_syscall_angr_numbers``: every ``BY_NAME``
  row's name exists in angr's table for some ABI in its scope, so an upstream
  rename cannot silently turn a handler off.
"""

from __future__ import annotations

import pytest

import angr

try:
    from angr.exploration import RustExplorationManager
except ImportError:  # pragma: no cover
    RustExplorationManager = None

pytestmark = pytest.mark.skipif(RustExplorationManager is None, reason="Rust engine not built")

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


def _native_table(proj, abi):
    mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
    return {int(n): name for n, name in mgr._rust_mgr.native_syscall_table().get(abi, {}).items()}


@pytest.mark.parametrize("abi", sorted(ABI_PROJECTS))
def test_every_row_names_the_handlers_syscall(abi):
    proj = _project(abi)
    mapping = proj.simos.syscall_library.syscall_number_mapping[abi]
    table = _native_table(proj, abi)
    assert table, f"{abi}: no native syscall handlers registered"
    wrong = {}
    for num, handler in table.items():
        angr_name = mapping.get(num)
        if angr_name != handler and (handler, angr_name) not in HANDLER_ALIASES:
            wrong[num] = (handler, angr_name)
    assert not wrong, f"{abi}: number -> (native handler, angr's name): {wrong}"


def test_every_registration_names_a_syscall_angr_numbers():
    from angr.procedures.definitions import SIM_LIBRARIES

    mappings = {}
    for library_name in ("linux", "cgcabi"):
        library = SIM_LIBRARIES[library_name][0]
        mappings.update(library.syscall_number_mapping)

    from angr.rustylib import vex_engine

    unknown = []
    for name, abis in vex_engine._testing.native_syscall_rows_for_test():
        if not any(name in mappings[abi].values() for abi in abis):
            unknown.append((name, abis))
    assert not unknown, f"registrations angr numbers nowhere in their scope: {unknown}"


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


def test_arm_oabi_svc_takes_number_from_immediate():
    """An ARM ``svc #0x9000NN`` (OABI) is syscall NN, not whatever r7 holds.

    The shellcode sets r7=4 (EABI ``write``) and then issues the OABI
    ``exit``; the Rust engine used to dispatch on r7 and run on to the EABI
    ``exit`` after it (angr-mx00p.58). Both engines must stop at the first
    ``svc``, with r7 still 4.
    """
    # mov r7,#4 ; svc #0x900001 (OABI exit) ; mov r7,#1 ; svc #0
    sc = bytes.fromhex("0470a0e3010090ef0170a0e3000000ef")
    proj = angr.load_shellcode(sc, arch="ARMEL", simos="linux", load_address=0x1000)

    simgr = proj.factory.simulation_manager(proj.factory.entry_state(addr=0x1000))
    simgr.run()
    assert len(simgr.deadended) == 1
    assert simgr.deadended[0].regs.r7.concrete_value == 4

    mgr = RustExplorationManager(proj, [proj.factory.entry_state(addr=0x1000)])
    mgr.run()
    assert len(mgr.deadended) == 1, mgr._rust_mgr.stash_counts()
    assert mgr.deadended[0].regs.r7.concrete_value == 4, "execution ran past the OABI exit"
