#!/usr/bin/env python3
# pylint: disable=missing-class-docstring,no-self-use
from __future__ import annotations

__package__ = __package__ or "tests.exploration_techniques"  # pylint:disable=redefined-builtin

import os
import unittest

import angr
from angr.exploration_techniques import DirectedSearch
from tests.common import bin_location

test_location = os.path.join(bin_location, "tests")


class TestDirectedSearch(unittest.TestCase):
    """Unit tests for the DS-spike.2 directed-search exploration technique."""

    @classmethod
    def setUpClass(cls):
        binary = os.path.join(test_location, "x86_64", "fauxware")
        cls.proj = angr.Project(binary, load_options={"auto_load_libs": False})
        cls.cfg = cls.proj.analyses.CFGFast()
        cls.target = cls.proj.kb.functions["authenticate"].addr

    def test_rejects_zero_beam_width(self):
        with self.assertRaises(ValueError):
            DirectedSearch(self.cfg, self.target, beam_width=0)

    def test_rejects_uncovered_target(self):
        # Distance map construction must fail loudly for an address not in the CFG.
        with self.assertRaises(ValueError):
            DirectedSearch(self.cfg, 0xDEADBEEF)

    def test_setup_creates_deferred_stash(self):
        ds = DirectedSearch(self.cfg, self.target, deferred_stash="held")
        simgr = self.proj.factory.simulation_manager(self.proj.factory.entry_state())
        ds.setup(simgr)
        assert "held" in simgr.stashes

    def test_priority_orders_closer_states_first(self):
        # The entry block is strictly farther from authenticate() than the target block itself,
        # so its priority value must be larger.
        ds = DirectedSearch(self.cfg, self.target)
        entry_state = self.proj.factory.entry_state()
        target_state = self.proj.factory.blank_state(addr=self.target)
        assert ds._priority(target_state) == 0.0
        assert ds._priority(entry_state) > ds._priority(target_state)

    def test_unknown_pc_is_infinitely_far(self):
        ds = DirectedSearch(self.cfg, self.target)
        bogus = self.proj.factory.blank_state(addr=0xDEADBEEF)
        assert ds._priority(bogus) == float("inf")

    def test_reaches_target_with_beam_one(self):
        # Greedy best-first (beam_width=1) should still drive a state to authenticate().
        ds = DirectedSearch(self.cfg, self.target, beam_width=1)
        simgr = self.proj.factory.simulation_manager(self.proj.factory.entry_state())
        simgr.use_technique(ds)
        simgr.explore(find=self.target, num_find=1)
        assert len(simgr.found) >= 1
        assert simgr.found[0].addr == self.target

    def test_beam_caps_active_stash(self):
        # With beam_width=1 the active stash never holds more than one state after a step that
        # produced multiple successors; overflow lands in the deferred stash.
        ds = DirectedSearch(self.cfg, self.target, beam_width=1)
        simgr = self.proj.factory.simulation_manager(self.proj.factory.entry_state())
        simgr.use_technique(ds)
        for _ in range(40):
            if not simgr.active:
                break
            simgr.step()
            assert len(simgr.active) <= 1
            if any(s.addr == self.target for s in simgr.active):
                break


if __name__ == "__main__":
    unittest.main()
