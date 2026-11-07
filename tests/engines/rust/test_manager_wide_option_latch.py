"""SimOptions the Rust engine applies manager-wide follow the states that run, or are refused.

LAZY_SOLVES and ZERO_FILL_UNCONSTRAINED_MEMORY are per-state SimOptions, but
Rust's lazy-solves switch and the page callbacks are manager-wide. The wrapper
used to OR them across seeds and never clear them, so an option removed after
construction stayed on, and one seed's zero fill leaked to the others
(angr-vmi2l.241).
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
# cmp rdi,5 ; jne skip ; cmp rdi,5 ; jne bad ; good: jmp $ ; bad: jmp $ ; skip: jmp $
# `bad` is reachable only on a fork that is infeasible, which LAZY_SOLVES keeps.
BRANCH = bytes.fromhex("4883ff05750a4883ff057502ebfeebfeebfe")
GOOD, BAD, SKIP = LOAD + 12, LOAD + 14, LOAD + 16
# movzx eax, byte [rdi] ; jmp $
LOAD_BYTE = bytes.fromhex("0fb607ebfe")


def _branch_state(lazy: bool):
    p = angr.load_shellcode(BRANCH, arch="amd64", load_address=LOAD)
    return p, p.factory.blank_state(addr=LOAD, add_options={angr.options.LAZY_SOLVES} if lazy else set())


def _bad_count(mgr) -> int:
    mgr.explore(find=BAD, avoid=[GOOD, SKIP], num_find=None)
    return len(mgr.found)


@pytest.mark.parametrize("lazy", [False, True])
def test_lazy_solves_keeps_the_infeasible_fork_only_when_set(lazy):
    p, s = _branch_state(lazy)
    assert _bad_count(RustExplorationManager(p, [s])) == int(lazy)


def test_lazy_solves_removed_after_construction_is_cleared():
    p, s = _branch_state(True)
    mgr = RustExplorationManager(p, [s])
    mgr.active[0].options.discard(angr.options.LAZY_SOLVES)
    assert _bad_count(mgr) == 0


def test_lazy_solves_added_after_construction_is_set():
    p, s = _branch_state(False)
    mgr = RustExplorationManager(p, [s])
    mgr.active[0].options.add(angr.options.LAZY_SOLVES)
    assert _bad_count(mgr) == 1


def test_lazy_solves_on_only_some_active_states_is_refused():
    p, s = _branch_state(True)
    other = s.copy()
    other.options.discard(angr.options.LAZY_SOLVES)
    mgr = RustExplorationManager(p, [s, other])
    with pytest.raises(NotImplementedError, match="LAZY_SOLVES"):
        mgr.explore(find=BAD, avoid=[GOOD, SKIP])
    with pytest.raises(NotImplementedError, match="LAZY_SOLVES"):
        mgr.step()


def _zero_fill_seeds(engine=None):
    p = angr.load_shellcode(LOAD_BYTE, arch="amd64", load_address=LOAD, engine=engine)
    seeds = []
    for marker, zero_fill in ((1, True), (2, False)):
        s = p.factory.blank_state(
            addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY} if zero_fill else set()
        )
        # Below the stack pointer, on a page neither seed has written.
        s.regs.rdi = s.solver.eval(s.regs.rsp) - 0x2000
        s.regs.rbx = marker
        seeds.append(s)
    return p, seeds


def test_seeds_that_differ_in_zero_fill_are_refused():
    p, seeds = _zero_fill_seeds()
    with pytest.raises(NotImplementedError, match="ZERO_FILL_UNCONSTRAINED_MEMORY"):
        RustExplorationManager(p, seeds)


def test_seeds_that_differ_in_zero_fill_route_to_python_on_a_rust_default_project():
    p, seeds = _zero_fill_seeds("rust")
    simgr = p.factory.simulation_manager(seeds)
    simgr.explore(find=LOAD + 3, num_find=2)
    reads = {f.solver.eval(f.regs.rbx): len(f.solver.eval_upto(f.regs.eax, 2)) for f in simgr.found}
    # The zero-fill seed reads 0; the other reads a fresh symbol.
    assert reads == {1: 1, 2: 2}
