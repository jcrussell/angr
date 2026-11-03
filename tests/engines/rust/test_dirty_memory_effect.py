"""A dirty helper with a memory effect runs against Rust memory.

The Python ``dirty_call`` callback runs a helper on the callback/default
SimState, not the Rust state, so an x87 ``fstp m80`` (``storeF80le``) lost
its store and an ``fld m80`` (``loadF80le``) read pre-run bytes; a helper
with no handler (amd64 F80) read as zeros where the Python engine raises
(angr-vmi2l.170). Such a block now runs in Python's VEX engine, which needs
the register proxy to address the x87 tag bytes and MMX lanes by offset.
"""

from __future__ import annotations

import pytest

import angr
from angr.errors import UnsupportedDirtyError
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD, BUF, OUT = 0x400000, 0x700000, 0x700100
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
# fldpi ; fstp tbyte [edi]
STORE = b"\xd9\xeb\xdb\x3f"
# m80 3.0 at [edi] ; fld tbyte [edi] ; fstp qword [esi]
LOAD_M80 = (
    b"\xc7\x07\x00\x00\x00\x00"
    + b"\xc7\x47\x04\x00\x00\x00\xc0"
    + b"\x66\xc7\x47\x08\x00\x40"
    + b"\xdb\x2f"
    + b"\xdd\x1e"
)


def _explore(engine, code, arch):
    end = LOAD + len(code)
    proj = angr.load_shellcode(code + b"\xeb\xfe", arch=arch, load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options=OPTS)
    if arch == "amd64":
        state.regs.rdi, state.regs.rsi = BUF, OUT
    else:
        state.regs.edi, state.regs.esi = BUF, OUT
    state.memory.store(BUF, b"\x66" * 16)
    state.memory.store(OUT, b"\x66" * 8)
    if engine == "rust":
        from angr.exploration.rust_manager import RustExplorationManager

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=end, max_steps=10)
        errors = [getattr(e, "error", e) for e in mgr.errored]
    else:
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=end)
        errors = [e.error for e in mgr.errored]
    found = [
        (
            f.solver.eval(f.memory.load(BUF, 10), cast_to=bytes),
            f.solver.eval(f.memory.load(OUT, 8), cast_to=bytes),
        )
        for f in mgr.found
    ]
    return found, errors


@pytest.mark.parametrize("code", [STORE, LOAD_M80], ids=["fstp_m80", "fld_m80"])
def test_x86_m80_matches_python(code):
    py_found, py_errors = _explore("py", code, "x86")
    rust_found, rust_errors = _explore("rust", code, "x86")
    assert not py_errors
    assert not rust_errors
    assert len(py_found) == 1
    assert rust_found == py_found


def test_amd64_f80_without_a_handler_errors_like_python():
    py_found, py_errors = _explore("py", STORE, "amd64")
    rust_found, rust_errors = _explore("rust", STORE, "amd64")
    assert not py_found
    assert any(isinstance(e, UnsupportedDirtyError) for e in py_errors)
    assert not rust_found, "the store was dropped and the state found"
    assert any("storeF80le" in str(e) for e in rust_errors)


def test_dirty_callback_raises_on_a_missing_handler():
    """The callback used to warn and hand Rust a zero result."""
    from angr.exploration.rust_manager import RustExplorationManager

    proj = angr.load_shellcode(b"\x90\xeb\xfe", arch="amd64", load_address=LOAD)
    mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=LOAD)])
    with pytest.raises(UnsupportedDirtyError):
        mgr._cb_dirty_call("amd64g_dirtyhelper_NOT_A_REAL_HELPER", [], 64)
