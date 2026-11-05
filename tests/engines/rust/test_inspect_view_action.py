"""A BP action fired mid-step reads the stepping state (angr-mx00p.132.53).

The step holds the manager, so the action's state proxy reads the
interpreter's published mid-step view instead. Each read is compared with
what the Python engine's ``HeavyVEXMixin`` shows the same action.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import SimulationManager, claripy
from tests.common import bin_location
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

# fauxware's main: the block's `mov [rbp-0x34], edi; mov [rbp-0x40], rsi`
# end at 0x40072c, so a BP before that instruction sees same-block stores.
MAIN = 0x40071D
AFTER_STORES = 0x40072C
ARGV = 0x7FFF0000


# rbp at AFTER_STORES in the call_state frame: the stores' base, fixed so the
# load reads on its own rather than through a register read.
ARGV_SLOT = 0x7FFFFFFEFFF0 - 0x40

READS = {
    "addr": lambda s: s.addr,
    "rbp": lambda s: s.solver.eval(s.regs.rbp),
    "rsi": lambda s: s.solver.eval(s.regs.rsi),
    "stored_argv": lambda s: s.solver.eval(s.memory.load(ARGV_SLOT, 8, endness="Iend_LE")),
    "block_size": lambda s: s.block().size,
}


def _reads(state):
    """Each read's value, or the exception it raised."""
    out = {}
    for name, read in READS.items():
        try:
            out[name] = read(state)
        except Exception as e:  # pylint: disable=broad-exception-caught
            out[name] = e
    return out


def _main_state(project):
    return project.factory.call_state(MAIN, 1, ARGV)


def _at_after_stores(state):
    insn = state.inspect.attrs.instruction
    return getattr(insn, "concrete_value", insn) == AFTER_STORES


def _python_reads(project):
    seen = []

    def action(state):
        if _at_after_stores(state):
            seen.append(_reads(state))

    state = _main_state(project)
    state.inspect.b("instruction", when="before", action=action)
    project.factory.successors(state)
    return seen


def _rust_reads(project, *, view=True):
    seen, kept = [], []

    def action(state):
        if not _at_after_stores(state):
            return
        kept.append(state)
        seen.append(_reads(state))

    mgr = RustExplorationManager(project, [_main_state(project)])
    mgr._get_inspect_proxy().b("instruction", when="before", action=action)
    if not view:
        mgr._callbacks.set_inspect_wants_view(False)
    mgr.step()
    return mgr, seen, kept


def test_an_instruction_bp_action_reads_the_mid_step_state_like_python(fauxware_project):
    expected = _python_reads(fauxware_project)
    _, seen, kept = _rust_reads(fauxware_project)
    assert len(expected) == 1
    assert expected[0]["rbp"] - 0x40 == ARGV_SLOT
    assert expected[0]["stored_argv"] == ARGV
    assert seen == expected
    assert type(kept[0]).__name__ == "RustInspectViewStateProxy"


def test_each_read_fails_without_the_view(fauxware_project):
    (expected,) = _python_reads(fauxware_project)
    _, seen, kept = _rust_reads(fauxware_project, view=False)
    assert type(kept[0]).__name__ == "RustStateProxy"
    (got,) = seen
    # The step holds the manager, so each read raises, or (addr) degrades to
    # the proxy's logged 0.
    for name, value in got.items():
        assert value != expected[name], name


def test_a_kept_proxy_reads_the_post_step_state(fauxware_project):
    mgr, _, kept = _rust_reads(fauxware_project)
    (state,) = kept
    (live,) = mgr.active
    assert state.addr == live.addr != MAIN
    assert state.solver.eval(state.regs.rsp) == live.solver.eval(live.regs.rsp)


def _symbolic_store_seen(project, make_manager):
    """What a mem_write BP action reads for a store to a symbolic, constrained rdi."""
    state = project.factory.blank_state(addr=project.entry)
    rdi = claripy.BVS("rdi", 64)
    state.regs.rdi = rdi
    state.solver.add(rdi == 0x5000)
    seen = []

    def action(s):
        addr = s.inspect.attrs.mem_write_address
        seen.append((getattr(addr, "symbolic", False), s.solver.eval(addr)))

    state.inspect.b("mem_write", when="before", action=action)
    make_manager(state).step()
    return seen


def test_a_symbolic_store_fires_mem_write_and_evaluates_under_constraints_like_python():
    """The store's address reaches the action as an AST, and the view solver
    evaluates it under the path constraints (angr-mx00p.132.53.3)."""
    project = angr.load_shellcode("mov dword ptr [rdi], 1\nret", "amd64")
    expected = _symbolic_store_seen(project, lambda s: SimulationManager(project, [s]))
    assert expected == [(True, 0x5000)]
    assert _symbolic_store_seen(project, lambda s: RustExplorationManager(project, [s])) == expected


SMC_CASES = {
    "not_smc_shellcode": (
        lambda: angr.load_shellcode("mov rax, 0xdeadbeef\nret", "amd64", selfmodifying_code=True),
        None,
    ),
    "smc_shellcode": (
        lambda: angr.load_shellcode(
            "inc dword ptr [here+1]\nhere:\nmov rax, 0xdeadbeef\nret", "amd64", selfmodifying_code=True
        ),
        None,
    ),
    "smc_buffer": (
        lambda: angr.Project(
            os.path.join(bin_location, "tests", "x86_64", "smc"), selfmodifying_code=True, auto_load_libs=False
        ),
        "main",
    ),
}


@pytest.mark.parametrize("case", sorted(SMC_CASES))
def test_smc_finds_the_regions_python_does_on_an_explicit_rust_manager(case):
    """tests/analyses/test_smc.py on Rust: TraceClassifier's actions read a
    symbolic store address and ``state.block()`` of heap code mid-step."""
    make_project, subject = SMC_CASES[case]

    def regions(engine):
        project = make_project()
        if engine == "rust":
            project.factory.simgr = lambda state, **_kw: RustExplorationManager(project, [state])
        else:
            assert type(project.factory.simgr(project.factory.blank_state())) is SimulationManager
        return project.analyses.SMC(project.entry if subject is None else subject).regions

    expected = regions("python")
    assert bool(expected) == (case != "not_smc_shellcode")
    assert regions("rust") == expected
