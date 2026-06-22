from __future__ import annotations

from .base import ExplorationTechnique
from .cfg_distance import build_cfg_distance_map, cfg_distance


class DirectedSearch(ExplorationTechnique):
    """
    Directed (guided) best-first search toward a target address.

    Prioritizes states by their CFG control-flow distance to ``target_addr`` (computed once
    by :func:`~angr.exploration_techniques.cfg_distance.build_cfg_distance_map`). Only the
    ``beam_width`` closest states are kept active each round; the rest are stashed in
    ``deferred_stash``. When the active stash empties, the closest deferred state is pulled
    back in. With the default ``beam_width=1`` this is a greedy best-first search; larger
    widths trade breadth for robustness against CFG-distance estimation error.

    This is a pure-Python prototype (DS-spike.2) used to justify or kill a native
    implementation. It reuses the DS-spike.1 distance helper and does **not** reimplement any
    CFG recovery or distance logic.

    States whose PC is not in the CFG, or from which the target is unreachable, are treated as
    infinitely far and sink to the back of every ordering.
    """

    def __init__(self, cfg, target_addr: int, beam_width: int = 1, deferred_stash: str = "deferred"):
        """
        :param cfg:          A recovered CFG (e.g. ``project.analyses.CFGFast()``).
        :param target_addr:  The address to steer exploration toward.
        :param beam_width:   How many of the closest states to keep active each round (>= 1).
        :param deferred_stash: Stash name for states held back from the active beam.
        """
        super().__init__()
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        self._cfg = cfg
        self.target_addr = target_addr
        self.beam_width = beam_width
        self.deferred_stash = deferred_stash
        self._distance_map = build_cfg_distance_map(cfg, target_addr)

    def setup(self, simgr):
        if self.deferred_stash not in simgr.stashes:
            simgr.stashes[self.deferred_stash] = []

    def _priority(self, state) -> float:
        """Sort key: smaller is closer to the target; unreachable/unknown sink to the back."""
        dist = cfg_distance(self._distance_map, self._cfg, state.addr)
        return float("inf") if dist is None else float(dist)

    def step(self, simgr, stash="active", **kwargs):
        simgr = simgr.step(stash=stash, **kwargs)

        # Keep only the beam_width closest states active; defer the rest.
        if len(simgr.stashes[stash]) > self.beam_width:
            simgr.stashes[stash].sort(key=self._priority)
            overflow = simgr.stashes[stash][self.beam_width :]
            simgr.stashes[stash] = simgr.stashes[stash][: self.beam_width]
            simgr.stashes[self.deferred_stash].extend(overflow)

        # When the beam empties, pull the single closest deferred state back in.
        if len(simgr.stashes[stash]) == 0 and len(simgr.stashes[self.deferred_stash]) > 0:
            simgr.stashes[self.deferred_stash].sort(key=self._priority)
            simgr.stashes[stash].append(simgr.stashes[self.deferred_stash].pop(0))

        return simgr
