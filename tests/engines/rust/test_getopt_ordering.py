"""Native getopt orders argv like the glibc port in getopt.py (angr-vmi2l.123).

The shellcode calls a stock ``getopt`` hook in a loop until it returns -1,
recording each return value. Python's getopt.py permutes argv by default,
honours a leading ``+`` (stop at the first operand) or ``-`` (return operands
as option 1), and rejects ``;``. Its scan state lives on ``state.libc``, which
must agree with the native cursor in both directions.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD, HOOK = 0x400000, 0x500000
OPTSTRING, ARGV, STRINGS, OUT = 0x602000, 0x603000, 0x603200, 0x604000

CODE = (
    b"\x4c\x89\xe7"  # mov rdi, r12 (argc)
    b"\x4c\x89\xee"  # mov rsi, r13 (argv)
    b"\x4c\x89\xf2"  # mov rdx, r14 (optstring)
    + b"\x48\xb8" + struct.pack("<Q", HOOK)  # mov rax, HOOK
    + b"\xff\xd0"  # call rax
    b"\x41\x89\x07"  # mov [r15], eax
    b"\x49\x83\xc7\x04"  # add r15, 4
    b"\x83\xf8\xff"  # cmp eax, -1
    b"\x75\xdf"  # jne LOAD
)  # fmt: skip
END = LOAD + len(CODE)


def _run(engine: str, optstring: bytes, argv: list[bytes], libc_cursor=None):
    proj = angr.load_shellcode(CODE + b"\xeb\xfe", arch="amd64", load_address=LOAD)
    proto = angr.SIM_LIBRARIES["libc.so.6"][0].get_prototype("getopt")
    proj.hook(HOOK, angr.SIM_PROCEDURES["libc"]["getopt"](prototype=proto), replace=True)
    opts = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY}
    state = proj.factory.blank_state(addr=LOAD, add_options=opts)
    state.regs.rsp = 0x7FFFFFFF0800
    state.memory.store(OUT, b"\0" * 64)  # Rust drops a store to an untouched page (angr-0mo14)
    state.memory.store(OPTSTRING, optstring + b"\0")
    ptr = STRINGS
    for i, arg in enumerate(argv):
        state.memory.store(ptr, arg + b"\0")
        state.memory.store(ARGV + 8 * i, struct.pack("<Q", ptr))
        ptr += len(arg) + 1
    state.memory.store(ARGV + 8 * len(argv), b"\0" * 8)
    state.regs.r12, state.regs.r13, state.regs.r14, state.regs.r15 = len(argv), ARGV, OPTSTRING, OUT
    if libc_cursor is not None:
        libc = state.libc
        libc.getopt_optind, libc.getopt_nextchar, libc.getopt_first_nonopt, libc.getopt_last_nonopt = libc_cursor

    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=END, n=40)
    assert len(mgr.found) == 1
    (found,) = mgr.found
    calls = (found.solver.eval(found.regs.r15) - OUT) // 4
    rets = [found.solver.eval(found.memory.load(OUT + 4 * i, 4, endness="Iend_LE")) for i in range(calls)]
    rets = [r - (1 << 32) if r >= 1 << 31 else r for r in rets]
    order = []
    for i in range(len(argv)):
        p = found.solver.eval(found.memory.load(ARGV + 8 * i, 8, endness="Iend_LE"))
        order.append(found.solver.eval(found.memory.load(p, 16), cast_to=bytes).split(b"\0")[0])
    libc = found.libc
    cursor = (libc.getopt_optind, libc.getopt_nextchar, libc.getopt_first_nonopt, libc.getopt_last_nonopt)
    return rets, order, cursor


CASES = {
    # optstring, argv, expected returns, expected final argv, expected optind
    "permute": (b"v", [b"prog", b"file", b"-v"], [ord("v"), -1], [b"prog", b"-v", b"file"], 2),
    "return_in_order": (b"-v", [b"prog", b"file", b"-v"], [1, ord("v"), -1], [b"prog", b"file", b"-v"], 3),
    "require_order": (b"+v", [b"prog", b"file", b"-v"], [-1], [b"prog", b"file", b"-v"], 1),
    "semicolon": (b"v;", [b"prog", b"-;"], [ord("?"), -1], [b"prog", b"-;"], 2),
    "double_dash": (
        b"ab:",
        [b"prog", b"x", b"-a", b"-b", b"val", b"--", b"-y", b"z"],
        [ord("a"), ord("b"), -1],
        [b"prog", b"-a", b"-b", b"val", b"--", b"x", b"-y", b"z"],
        5,
    ),
    "missing_arg": (b":a:", [b"prog", b"f", b"-a"], [ord(":"), -1], [b"prog", b"-a", b"f"], 2),
}


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_getopt_ordering_matches_glibc_port(engine, case):
    optstring, argv, rets, order, optind = CASES[case]
    got_rets, got_order, cursor = _run(engine, optstring, argv)
    assert got_rets == rets
    assert got_order == order
    assert cursor[0] == optind


@pytest.mark.parametrize("case", sorted(CASES))
def test_exported_libc_cursor_matches_python(case):
    optstring, argv, *_ = CASES[case]
    assert _run("rust", optstring, argv)[2] == _run("py", optstring, argv)[2]


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_resumes_scan_from_state_libc(engine):
    """A state whose ``state.libc`` is parked inside "-ab" resumes at 'b'."""
    rets, _, cursor = _run(engine, b"ab", [b"prog", b"-ab"], libc_cursor=(1, 2, 1, 1))
    assert rets == [ord("b"), -1]
    assert cursor == (2, 0, 2, 2)
