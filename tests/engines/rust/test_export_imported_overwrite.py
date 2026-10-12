# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests for angr-ccp2w: overwritten imports must reach the found state.

``_get_state_symbolic_z3_asts`` skipped every address imported from the
initial Python state "even if the binary modified them" (the flareon5 rule),
so ``found.memory.load`` over user-seeded symbolic memory the program later
overwrote returned the stale seed. The rule now skips an address only while it
still holds the imported value; an untouched seed keeps its claripy identity.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

BUF = 0x7FFF0000
# mov rax, [rdi] ; add rax, 1 ; mov [rdi], rax ; jmp .
SHELLCODE = bytes.fromhex("488b074883c001488907ebfe")
SPIN = 0x400000 + len(SHELLCODE) - 2


def _equivalent(a, b):
    return not claripy.Solver().satisfiable(extra_constraints=[a != b])


def _run(store_value):
    proj = angr.load_shellcode(SHELLCODE, arch="amd64", load_address=0x400000)
    state = proj.factory.blank_state(addr=0x400000)
    state.regs.rdi = BUF
    seed = claripy.BVS("ccp2w_seed", 64, explicit_name=True)
    neighbour = claripy.BVS("ccp2w_neighbour", 64, explicit_name=True)
    store_value(state, seed, neighbour)
    mgr = RustExplorationManager(proj, [state])
    mgr.explore(find=SPIN)
    assert mgr.found, "explore did not reach the spin loop"
    return mgr.found[0], seed, neighbour


class TestExportImportedOverwrite:
    def test_overwritten_seed_is_exported(self):
        def seed_both(state, seed, neighbour):
            state.memory.store(BUF, seed, endness="Iend_LE")
            state.memory.store(BUF + 8, neighbour, endness="Iend_LE")

        found, seed, neighbour = _run(seed_both)
        got = found.memory.load(BUF, 8, endness="Iend_LE")
        assert _equivalent(got, seed + 1), f"found state kept the stale seed: {got}"
        # The untouched neighbour is still Python's own value.
        assert _equivalent(found.memory.load(BUF + 8, 8, endness="Iend_LE"), neighbour)

    def test_overwritten_filler_seed_is_exported(self):
        """Seed created by the default filler (a bare load), as solve scripts do."""

        def seed_by_load(state, _seed, _neighbour):
            state.memory.load(BUF, 8, endness="Iend_LE")

        found, _, _ = _run(seed_by_load)
        got = found.memory.load(BUF, 8, endness="Iend_LE")
        names = got.variables
        assert names and any(n.startswith("mem_7fff0000") for n in names), got
        # The program added 1: the result must not be the bare filler symbol.
        assert got.op != "BVS" and not (got.op == "Reverse" and got.args[0].op == "BVS"), (
            f"found state kept the stale filler seed: {got}"
        )
