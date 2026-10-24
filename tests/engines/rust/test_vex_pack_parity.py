"""amd64 PACK* (``Iop_{Q,}NarrowBin*``) on the Rust engine agree with the
Python engine on the same block (angr-vmi2l.55).

libVEX defines the binary narrows with the *left* operand in the high half of
the result, and pyvex lifts ``packsswb xmm0, xmm1`` as
``QNarrowBin16Sto8Sx16(GET(xmm1), GET(xmm0))``, so the destination lands in
the low half. Each test uses distinct destination and source lanes, some of
them out of the saturation range, so a swapped half or a wrong clamp shows.

The MMX forms load mm0/mm1 from rax/rbx and store the result back to rax in
the same block, because a seed state's mm registers do not reach the Rust
engine (angr-vmi2l.316).
"""

from __future__ import annotations

import pytest

import angr
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

BASE = 0x10000
JMP_SELF = bytes.fromhex("ebfe")


def _pack(lanes, bits):
    mask = (1 << bits) - 1
    return sum((v & mask) << (bits * i) for i, v in enumerate(lanes))


WORDS_DST = [1, 2, 0x180, -3, 0x7F, -0x200, 0x55, 8]
WORDS_SRC = [0x11, -0x81, 0x13, 0xFF, 0x15, 0x16, -1, 0x18]
DWORDS_DST = [1, 0x12345, -5, 0x7FFF]
DWORDS_SRC = [0x11, -0x9000, 0x22, 0x40000]

# movq mm0, rax ; movq mm1, rbx ; <pack mm0, mm1> ; movq rax, mm0
MMX_LOAD = bytes.fromhex("480f6ec0480f6ecb")
MMX_STORE = bytes.fromhex("480f7ec0")

# name -> (encoding, register class, element bits of the source lanes)
CASES = {
    "packsswb_xmm": ("660f63c1", "xmm", 16),
    "packuswb_xmm": ("660f67c1", "xmm", 16),
    "packssdw_xmm": ("660f6bc1", "xmm", 32),
    "packusdw_xmm": ("660f382bc1", "xmm", 32),
    "packsswb_mmx": ("0f63c1", "mm", 16),
    "packuswb_mmx": ("0f67c1", "mm", 16),
    "packssdw_mmx": ("0f6bc1", "mm", 32),
}


def _state(project, regclass, bits):
    state = project.factory.blank_state(addr=BASE)
    dst, src = (WORDS_DST, WORDS_SRC) if bits == 16 else (DWORDS_DST, DWORDS_SRC)
    if regclass == "xmm":
        lanes = 128 // bits
        state.regs.xmm0 = _pack(dst[:lanes], bits)
        state.regs.xmm1 = _pack(src[:lanes], bits)
    else:
        lanes = 64 // bits
        state.regs.rax = _pack(dst[:lanes], bits)
        state.regs.rbx = _pack(src[:lanes], bits)
    return state


@pytest.mark.parametrize("insn", CASES)
def test_pack_matches_python_engine(insn):
    encoding, regclass, bits = CASES[insn]
    if regclass == "xmm":
        code = bytes.fromhex(encoding) + JMP_SELF
        out = "xmm0"
    else:
        code = MMX_LOAD + bytes.fromhex(encoding) + MMX_STORE + JMP_SELF
        out = "rax"
    end = BASE + len(code) - len(JMP_SELF)

    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    python = project.factory.simulation_manager(_state(project, regclass, bits))
    python.explore(find=end)
    assert python.found, "Python engine did not reach the end of the block"
    expected = python.found[0].solver.eval(getattr(python.found[0].regs, out))

    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    mgr = RustExplorationManager(project, [_state(project, regclass, bits)])
    mgr.explore(find=end, max_steps=5)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
    assert mgr.found[0].solver.eval(getattr(mgr.found[0].regs, out)) == expected


def test_packuswb_puts_destination_in_low_half():
    """The bead's repro: destination words 1..8, source 0x11..0x18."""
    code = bytes.fromhex("660f67c1") + JMP_SELF
    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    state = project.factory.blank_state(addr=BASE)
    state.regs.xmm0 = _pack(range(1, 9), 16)
    state.regs.xmm1 = _pack(range(0x11, 0x19), 16)
    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=BASE + 4, max_steps=5)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
    xmm0 = mgr.found[0].solver.eval(mgr.found[0].regs.xmm0)
    assert xmm0 == 0x18171615141312110807060504030201
