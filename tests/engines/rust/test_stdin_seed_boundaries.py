"""Native stdin readers honour a harness seed's newline and EOF (angr-vmi2l.110).

A seeded stdin byte used to be bound to a reader leaf by position only. fgets
forced a seeded ``'\\n'`` inside its read off ``'\\n'``, which made the path
UNSAT, and every reader ran past a ``has_end`` seed by minting fresh bytes and
returning the full count. Python's readers stop at the newline and at EOF, and
so does glibc. Now a native reader that would cross either boundary defers the
call to Python.

The libc cases use the hook-in-``.ctors`` trick from
``test_scanf_percent_s_delimiters`` so the native registry serves the call.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

HOOK_ADDR = 0x600E48  # fauxware .ctors (in-binary, non-executable)
DEAD_ADDR = 0x4008B0
FMT_ADDR = 0x601100  # past .bss, lazy-mapped
BUF_ADDR = FMT_ADDR + 0x80
STREAM_ADDR = FMT_ADDR + 0x200  # zero-filled FILE, so _fileno reads 0
BUF_LEN = 0x48


def _stdin(seed: bytes, has_end: bool = True):
    return angr.SimFileStream(name="stdin", content=seed, has_end=has_end)


def _run_proc(proj, engine, proc, args, seed):
    """One step through the hooked libc ``proc``; returns (ret, buf bytes, native call count)."""
    lib = "posix" if proc == "read" else "libc"
    proj.hook(HOOK_ADDR, angr.SIM_PROCEDURES[lib][proc](), replace=True)
    try:
        state = proj.factory.blank_state(
            addr=HOOK_ADDR,
            stdin=_stdin(seed),
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        state.memory.store(FMT_ADDR, b"%s\x00")
        state.memory.store(BUF_ADDR, b"\x7f" * BUF_LEN)
        for reg, value in zip(("rdi", "rsi", "rdx"), args):
            setattr(state.regs, reg, value)
        state.memory.store(state.regs.rsp, claripy.BVV(DEAD_ADDR, 64), endness="Iend_LE")
        if engine == "py":
            simgr = proj.factory.simulation_manager(state)
            simgr.step()
            calls = None
        else:
            simgr = RustExplorationManager(proj, [state])
            simgr.run(max_steps=1)
            calls = simgr._rust_mgr.native_procedure_stats()["call_counts"].get(proc, 0)
        if not simgr.active:
            # Python's %s over a seed shorter than its read is UNSAT. Rust
            # files that state under `pruned` (angr-vmi2l.326), so the parity
            # check only compares that no successor survives.
            return "no-active", None, calls
        (succ,) = simgr.active
        ret = succ.solver.eval(succ.regs.rax)
        buf = succ.solver.eval(succ.memory.load(BUF_ADDR, BUF_LEN), cast_to=bytes)
        return ret, buf, calls
    finally:
        proj.unhook(HOOK_ADDR)


# (proc, args, seed): each read crosses the seed's newline or its end.
BOUNDARY_CASES = {
    "fgets_multiline": ("fgets", (BUF_ADDR, 16, STREAM_ADDR), b"ab\ncd"),
    "fgets_short_seed": ("fgets", (BUF_ADDR, 16, STREAM_ADDR), b"ab"),
    "read_short_seed": ("read", (0, BUF_ADDR, 10), b"hello"),
    "getchar_at_eof": ("getchar", (), b""),
    "scanf_s_short_seed": ("scanf", (FMT_ADDR, BUF_ADDR), b"hi"),
}


@pytest.mark.parametrize("case", sorted(BOUNDARY_CASES))
def test_reader_stops_at_seed_boundary(fauxware_project, case):
    proc, args, seed = BOUNDARY_CASES[case]
    py_ret, py_buf, _ = _run_proc(fauxware_project, "py", proc, args, seed)
    rust_ret, rust_buf, _ = _run_proc(fauxware_project, "rust", proc, args, seed)
    assert (rust_ret, rust_buf) == (py_ret, py_buf)


def test_python_reference_values(fauxware_project):
    """Pin what the parity test compares against, so both engines cannot drift together."""
    ret, buf, _ = _run_proc(fauxware_project, "py", "fgets", (BUF_ADDR, 16, STREAM_ADDR), b"ab\ncd")
    # Python's fgets returns the read length, not the buffer.
    assert (ret, buf[:4]) == (3, b"ab\n\x00")
    ret, _, _ = _run_proc(fauxware_project, "py", "read", (0, BUF_ADDR, 10), b"hello")
    assert ret == 5
    ret, _, _ = _run_proc(fauxware_project, "py", "getchar", (), b"")
    assert ret & 0xFFFFFFFF == 0xFFFFFFFF
    ret, _, _ = _run_proc(fauxware_project, "py", "scanf", (FMT_ADDR, BUF_ADDR), b"hi")
    assert ret == "no-active"


def test_read_inside_seed_stays_native(fauxware_project):
    """A read the seed fully covers is still served natively."""
    ret, buf, calls = _run_proc(fauxware_project, "rust", "read", (0, BUF_ADDR, 3), b"hello")
    assert calls == 1
    assert (ret, buf[:4]) == (3, b"hel\x7f")


# read(0, BUF, 10); a jmp so the find address is a block start past the syscall.
SYSCALL_CODE = b"\x0f\x05\xeb\x00\x90\x90\x90\x90"
LOAD = 0x400000
SYS_BUF = 0x500000


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_raw_read_syscall_short_seed(engine):
    proj = angr.load_shellcode(SYSCALL_CODE, arch="amd64", simos="linux", load_address=LOAD)
    state = proj.factory.blank_state(
        addr=LOAD,
        stdin=_stdin(b"hello"),
        add_options={
            angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.regs.rax = 0
    state.regs.rdi = 0
    state.regs.rsi = SYS_BUF
    state.regs.rdx = 10
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=LOAD + 4, n=10)
    (found,) = mgr.found
    assert found.solver.eval(found.regs.rax) == 5
    assert found.solver.eval(found.memory.load(SYS_BUF, 5), cast_to=bytes) == b"hello"
    assert found.posix.dumps(0) == b"hello"


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_fgets_binary_multiline_seed(engine):
    """The bead's repro: each fgets reads one line of a two-line seed."""
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "binaries", "tests", "x86_64", "fgets")
    proj = angr.Project(path, auto_load_libs=False)
    state = proj.factory.entry_state(stdin=b"normal\n0123456789abcd")
    if engine == "py":
        mgr = proj.factory.simulation_manager(state)
        mgr.run(n=2000)
    else:
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=2000)
    outs = sorted(s.posix.dumps(1) for s in mgr.deadended)
    assert outs == [b"That's one!\nThat's two!\nThat's three!\n"]
