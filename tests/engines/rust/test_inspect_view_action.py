"""A BP action fired mid-step reads the stepping state (angr-mx00p.132.53).

The step holds the manager, so the action's state proxy reads the
interpreter's published mid-step view instead. Each read is compared with
what the Python engine's ``HeavyVEXMixin`` shows the same action.
"""

from __future__ import annotations

import pytest

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
