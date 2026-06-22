#!/usr/bin/env python3
# pylint: disable=missing-class-docstring,no-self-use
from __future__ import annotations

__package__ = __package__ or "tests.exploration_techniques"  # pylint:disable=redefined-builtin

import os
import unittest

import angr
from angr.exploration_techniques import build_cfg_distance_map, cfg_distance
from tests.common import bin_location

test_location = os.path.join(bin_location, "tests")


class TestCFGDistance(unittest.TestCase):
    """Unit tests for the directed-search CFG-distance metric helper (DS-spike.1)."""

    @classmethod
    def setUpClass(cls):
        binary = os.path.join(test_location, "x86_64", "fauxware")
        cls.proj = angr.Project(binary, load_options={"auto_load_libs": False})
        cls.cfg = cls.proj.analyses.CFGFast()
        cls.target = cls.proj.kb.functions["authenticate"].addr

    def test_target_is_zero_distance(self):
        dm = build_cfg_distance_map(self.cfg, self.target)
        assert dm[self.target] == 0

    def test_direct_predecessors_are_one_hop(self):
        # Every direct CFG predecessor of the target block is exactly one edge away.
        dm = build_cfg_distance_map(self.cfg, self.target)
        target_node = self.cfg.model.get_any_node(self.target, anyaddr=True)
        preds = [p for p in self.cfg.graph.predecessors(target_node) if p.addr != self.target]
        assert preds, "expected authenticate() to have at least one non-self CFG predecessor"
        for pred in preds:
            assert dm[pred.addr] == 1

    def test_entry_distance_is_finite_and_positive(self):
        # The program entry can reach authenticate(), and is strictly farther than the target.
        dm = build_cfg_distance_map(self.cfg, self.target)
        entry_dist = cfg_distance(dm, self.cfg, self.proj.entry)
        assert entry_dist is not None
        assert entry_dist > 0

    def test_distances_decrease_monotonically_toward_target(self):
        # Along any CFG edge u -> v that lies on a shortest path to the target,
        # the distance must strictly decrease by exactly one hop.
        dm = build_cfg_distance_map(self.cfg, self.target)
        for node in self.cfg.graph.nodes():
            du = dm.get(node.addr)
            if du is None or du == 0:
                continue
            succ_dists = [dm[s.addr] for s in self.cfg.graph.successors(node) if s.addr in dm]
            assert succ_dists, f"reachable block {node.addr:#x} has no successor with a finite distance"
            # The closest successor is exactly one hop nearer the target.
            assert min(succ_dists) == du - 1

    def test_midblock_pc_resolves_to_containing_block(self):
        # A PC one byte past the target's block start still resolves to distance 0.
        dm = build_cfg_distance_map(self.cfg, self.target)
        assert cfg_distance(dm, self.cfg, self.target + 1) == 0

    def test_unknown_address_returns_none(self):
        dm = build_cfg_distance_map(self.cfg, self.target)
        assert cfg_distance(dm, self.cfg, 0xDEADBEEF) is None

    def test_uncovered_target_raises(self):
        with self.assertRaises(ValueError):
            build_cfg_distance_map(self.cfg, 0xDEADBEEF)


if __name__ == "__main__":
    unittest.main()
