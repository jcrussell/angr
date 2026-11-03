"""A call/jmp/ret into concrete code the state holds outside the image runs it (angr-vmi2l.158).

The out-of-image exit arms of `handle_exit` used to classify such a target as
an unmodeled external call (call/jmp) or an unconstrained return (ret), so the
code was skipped: a call resumed at the post-call address, a jmp deadended at
0. Each case is compared with the Python engine; an unmapped target keeps the
out-of-image classification.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
HEAP = 0x30000000
UNMAPPED = 0x50000000
SP = 0x7FFF0000
STACK_CODE = SP - 0x100
AFTER = LOAD + 12
FIND = LOAD + 19


def _image(target: int, op: bytes) -> bytes:
    # mov rax, target; <op>; AFTER: mov ecx, 0xdead; jmp $; FIND: nop; jmp $
    return b"\x48\xb8" + struct.pack("<Q", target) + op + b"\xb9\xad\xde\x00\x00" + b"\xeb\xfe" + b"\x90\xeb\xfe"


CALL, JMP, RET = b"\xff\xd0", b"\xff\xe0", b"\x50\xc3"  # call rax; jmp rax; push rax; ret
# mov ebx, 0x1337; mov rax, FIND; jmp rax
PAYLOAD = b"\xbb\x37\x13\x00\x00" + b"\x48\xb8" + struct.pack("<Q", FIND) + b"\xff\xe0"


def _explore(engine: str, target: int, op: bytes, zero_fill: bool, payload_at: int | None):
    proj = angr.load_shellcode(_image(target, op), arch="amd64", load_address=LOAD)
    options = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
    if zero_fill:
        options.add(angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY)
    state = proj.factory.blank_state(addr=LOAD, add_options=options)
    state.regs.rsp = SP
    if payload_at is not None:
        state.memory.store(payload_at, PAYLOAD)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=FIND, avoid=AFTER, n=20)
    return mgr


@pytest.mark.parametrize("zero_fill", [False, True], ids=["default", "zero_fill"])
@pytest.mark.parametrize(
    ("target", "op"),
    [(HEAP, CALL), (HEAP, JMP), (HEAP, RET), (STACK_CODE, CALL), (STACK_CODE, RET)],
    ids=["heap_call", "heap_jmp", "heap_ret", "stack_call", "stack_ret"],
)
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_transfer_runs_state_code(engine, target, op, zero_fill):
    mgr = _explore(engine, target, op, zero_fill, payload_at=target)
    assert len(mgr.found) == 1
    (found,) = mgr.found
    assert found.solver.eval_upto(found.regs.rbx, 2) == [0x1337]


@pytest.mark.parametrize("zero_fill", [False, True], ids=["default", "zero_fill"])
def test_call_to_unmapped_target_is_still_an_unmodeled_call(zero_fill):
    # Nothing at the target: the generic skip resumes at the return address.
    mgr = _explore("rust", UNMAPPED, CALL, zero_fill, payload_at=None)
    assert not mgr.found
    assert [s.addr for s in mgr.avoid] == [AFTER]
