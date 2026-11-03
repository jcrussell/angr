"""scanf ``%s`` excludes the delimiters and stops at buf_symbolic_bytes (angr-vmi2l.111).

Python's SimPackets ``%s`` (``format_parser.py::FormatString.interpret``)
reads ``state.libc.buf_symbolic_bytes`` bytes, or the field width, constrains
each one off ``SCANF_DELIMITERS`` and puts the NUL right after them. Native
``%s`` used to mint 256 unconstrained bytes, so the engine found paths whose
input real scanf would split, and it wrote 257 bytes past the buffer.

Uses the hook-in-``.ctors`` trick from ``TestNativeScanfScansetFallback`` so
the native registry actually serves the call.
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
DELIMITERS = b"\t\n\x0b\r "


def _scanf_state(proj, fmt):
    state = proj.factory.blank_state(
        addr=HOOK_ADDR,
        add_options={
            angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.memory.store(FMT_ADDR, fmt)
    # A marker past the read, which a correct %s leaves alone.
    state.memory.store(BUF_ADDR, b"\x7f" * 0x120)
    state.regs.rdi = FMT_ADDR
    state.regs.rsi = BUF_ADDR
    state.memory.store(state.regs.rsp, claripy.BVV(DEAD_ADDR, 64), endness="Iend_LE")
    return state


def _run(proj, engine, fmt):
    """One step through the hooked scanf; returns the successor and native stats."""
    proj.hook(HOOK_ADDR, angr.SIM_PROCEDURES["libc"]["scanf"](), replace=True)
    try:
        state = _scanf_state(proj, fmt)
        if engine == "py":
            simgr = proj.factory.simulation_manager(state)
            simgr.step()
            (succ,) = simgr.active
            return succ, None
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=1)
        stats = mgr._rust_mgr.native_procedure_stats()
        (succ,) = mgr.active
        return succ, stats
    finally:
        proj.unhook(HOOK_ADDR)


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize("fmt,length", [(b"%s\x00", 60), (b"%10s\x00", 10)])
def test_percent_s_bytes_exclude_delimiters(fauxware_project, engine, fmt, length):
    succ, stats = _run(fauxware_project, engine, fmt)
    if stats is not None:
        assert stats["call_counts"].get("scanf", 0) == 1, f"scanf must be served natively, got {stats}"
    for pos in (0, 1, length - 1):
        byte = succ.memory.load(BUF_ADDR + pos, 1)
        for delim in DELIMITERS:
            assert not succ.solver.satisfiable(extra_constraints=[byte == delim]), (
                f"{engine}: buf[{pos}] == {delim:#x} must be UNSAT"
            )
        assert succ.solver.satisfiable(extra_constraints=[byte == ord("A")])


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize("fmt,length", [(b"%s\x00", 60), (b"%10s\x00", 10)])
def test_percent_s_writes_length_plus_nul_only(fauxware_project, engine, fmt, length):
    succ, _ = _run(fauxware_project, engine, fmt)
    assert succ.solver.eval_one(succ.memory.load(BUF_ADDR + length, 1)) == 0
    for off in (length + 1, 0x100):
        assert succ.solver.eval_one(succ.memory.load(BUF_ADDR + off, 1)) == 0x7F, (
            f"{engine}: buf[{off:#x}] past the read must keep its old value"
        )


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_percent_s_dumps_has_no_delimiter(fauxware_project, engine):
    succ, _ = _run(fauxware_project, engine, b"%s\x00")
    succ.add_constraints(succ.memory.load(BUF_ADDR + 1, 1) != ord("A"))
    data = succ.posix.dumps(0)
    assert len(data) == 60
    assert not any(b in DELIMITERS for b in data), data
