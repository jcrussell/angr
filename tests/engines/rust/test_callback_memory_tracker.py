"""Unit tests for CallbackMemoryTracker (angr/exploration/rust_identity.py).

The tracker wraps ``state.memory.store`` during SimProcedure callbacks so the
resulting writes can be replayed back into the Rust core in the legacy
register-snapshot (``ANGR_RUST_USE_CALLBACK_MEMORY_PROXY=0``) path. These are
pure-Python tests that drive the tracker directly on a blank state — no Rust
manager needed.

Regression coverage for angr-zi35f.2: a store to a *symbolic* destination
address used to hit a bare ``pass`` and vanish from both ``get_writes`` and
``get_symbolic_writes`` with no log. It is now concretized to a solver witness
(the same Max/eval write-concretization the RustMemoryProxy fallback uses) and
tracked at that witness address.
"""

from __future__ import annotations

import claripy

import angr
from angr.exploration.rust_identity import CallbackMemoryTracker


def _blank_state():
    proj = angr.load_shellcode(b"\x90\x90\x90\x90", arch="AMD64", load_address=0x400000)
    return proj.factory.blank_state()


def test_concrete_addr_concrete_data_tracked():
    state = _blank_state()
    with CallbackMemoryTracker(state) as tracker:
        state.memory.store(0x1000, claripy.BVV(0x41424344, 32), endness="Iend_LE")
    writes = tracker.get_writes()
    assert (0x1000, (0x41424344).to_bytes(4, "little")) in writes
    assert tracker.get_symbolic_writes() == []


def test_symbolic_data_tracked_in_symbolic_writes():
    state = _blank_state()
    sym = claripy.BVS("v", 32)
    with CallbackMemoryTracker(state) as tracker:
        state.memory.store(0x2000, sym, endness="Iend_LE")
    sym_writes = tracker.get_symbolic_writes()
    assert len(sym_writes) == 1
    addr, ast = sym_writes[0]
    assert addr == 0x2000
    assert ast is sym
    # A concrete-witness byte record is also emitted for the same address.
    assert any(a == 0x2000 for a, _ in tracker.get_writes())


def test_symbolic_address_concretized_not_dropped():
    """A store to a symbolic destination address is concretized to a witness
    and tracked, rather than silently dropped (angr-zi35f.2)."""
    state = _blank_state()
    addr = claripy.BVS("addr", 64)
    state.solver.add(addr == 0x3000)
    with CallbackMemoryTracker(state) as tracker:
        state.memory.store(addr, claripy.BVV(0xDEAD, 32), endness="Iend_LE")
    writes = tracker.get_writes()
    assert len(writes) == 1, f"symbolic-addr store dropped: {writes}"
    concrete_addr, data = writes[0]
    assert concrete_addr == 0x3000
    assert data == (0xDEAD).to_bytes(4, "little")


def test_symbolic_address_and_symbolic_data():
    state = _blank_state()
    addr = claripy.BVS("addr", 64)
    state.solver.add(addr == 0x4000)
    sym = claripy.BVS("v", 32)
    with CallbackMemoryTracker(state) as tracker:
        state.memory.store(addr, sym, endness="Iend_LE")
    sym_writes = tracker.get_symbolic_writes()
    assert len(sym_writes) == 1
    saddr, ast = sym_writes[0]
    assert saddr == 0x4000
    assert ast is sym
    assert any(a == 0x4000 for a, _ in tracker.get_writes())


def test_store_restored_on_exit():
    state = _blank_state()
    original = state.memory.store
    with CallbackMemoryTracker(state):
        assert state.memory.store != original
    assert state.memory.store == original


def test_clear_empties_both_lists():
    state = _blank_state()
    with CallbackMemoryTracker(state) as tracker:
        state.memory.store(0x5000, claripy.BVV(1, 32))
        state.memory.store(0x5000, claripy.BVS("v", 32))
    assert tracker.get_writes()
    assert tracker.get_symbolic_writes()
    tracker.clear()
    assert tracker.get_writes() == []
    assert tracker.get_symbolic_writes() == []
