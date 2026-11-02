"""Python hooks inside a loop on mips32 (``mips/test_loops``) under the Rust engine.

The upstream ``tests/engines/test_hook.py::TestHook::test_mips`` scenario,
split by cause (angr-mx00p.132.5):

- a callback state stored its stale symbolic-page snapshot back through the
  memory proxy, reverting frame slots ``main`` had overwritten;
- the zero-length hook just serviced stayed unhooked for the whole resumed
  run, so the inner loop ran its later iterations past it;
- a native lift ran through the hooked address, so libVEX dropped the
  ``lw $v0`` Put that the hooked ``sll $v0`` overwrites, and the stop at the
  hook left ``$v0`` stale.
"""

from __future__ import annotations

import os

import angr
from angr import claripy
from tests.common import bin_location

BINARY = os.path.join(bin_location, "tests", "mips", "test_loops")
MAIN_END = 0x4007D8
INNER_LOOP = 0x40069C
OUTER_LOOP = 0x40076C


def _explore(project):
    simgr = project.factory.simulation_manager(project.factory.entry_state()).explore(find=[MAIN_END])
    assert len(simgr.found) == 1
    return simgr.found[0]


def _printed_values_hook(values):
    def hook(state):
        num = state.solver.eval(state.regs.a1)
        values.append(num)
        state.posix.get_fd(1).write_data(claripy.BVV(b"%d " % num))

    return hook


def test_hook_after_the_frame_is_written_reads_the_frame():
    project = angr.Project(BINARY, auto_load_libs=False, engine="rust")
    values = []
    project.hook(OUTER_LOOP, _printed_values_hook(values), length=0x14)
    _explore(project)
    assert values == list(range(100))


def test_zero_length_hook_in_a_loop_fires_on_every_iteration():
    project = angr.Project(BINARY, auto_load_libs=False, engine="rust")
    seen = []

    @project.hook(INNER_LOOP)
    def count(state):
        # $v0 holds the inner counter the block loaded just before the hook.
        seen.append(state.solver.eval(state.regs.v0))

    _explore(project)
    assert seen == list(range(10)) * 10


def test_both_hooks_match_the_python_engine():
    project = angr.Project(BINARY, auto_load_libs=False, engine="rust")
    output = []
    project.hook(INNER_LOOP, lambda _: output.append(1))
    values = []
    project.hook(OUTER_LOOP, _printed_values_hook(values), length=0x14)
    found = _explore(project)
    assert found.posix.dumps(1) == b"".join(b"%d " % x for x in range(100)) + b"\n"
    assert output == [1] * 100
    assert values == list(range(100))
