from __future__ import annotations

from collections import deque

__all__ = (
    "build_cfg_distance_map",
    "cfg_distance",
)


def build_cfg_distance_map(cfg, target_addr: int) -> dict[int, int]:
    """
    Compute the shortest control-flow distance (in CFG edges) from every reachable
    basic block to the block containing ``target_addr``.

    This is a thin helper over an existing CFG (typically ``project.analyses.CFGFast()``);
    it does **not** reimplement any CFG recovery. Distances are computed once by a single
    reverse breadth-first search (following CFG edges backwards from the target) and returned
    as a ``{block_start_addr: hops}`` map so that a directed-search exploration technique can
    prioritize states cheaply with a dict lookup per state.

    :param cfg:         A recovered CFG (e.g. the result of ``project.analyses.CFGFast()``).
                        Only ``cfg.graph.predecessors`` and ``cfg.model.get_any_node`` are used.
    :param target_addr: The address to measure distance toward (e.g. an ``Explorer`` find
                        target). It need not be a block start; the block containing it is used.
    :return:            A ``{block_start_addr: distance}`` map. The target block maps to ``0``.
                        Blocks from which the target is unreachable are absent from the map.
                        Callers should treat a missing key as "infinitely far".
    :raises ValueError: If no CFG node contains ``target_addr``.
    """
    target_node = cfg.model.get_any_node(target_addr, anyaddr=True)
    if target_node is None:
        raise ValueError(f"No CFG node contains target address {target_addr:#x}; is it covered by the CFG?")

    graph = cfg.graph

    # Reverse BFS from the target: walking predecessors yields the shortest number of CFG
    # edges from each reachable block *to* the target. Track visited CFGNodes (a single addr
    # may host several context nodes) and collapse to start addresses, keeping the minimum.
    distance_map: dict[int, int] = {}
    visited: set = {target_node}
    queue: deque[tuple] = deque([(target_node, 0)])
    while queue:
        node, dist = queue.popleft()
        addr = node.addr
        if addr not in distance_map or dist < distance_map[addr]:
            distance_map[addr] = dist
        for pred in graph.predecessors(node):
            if pred not in visited:
                visited.add(pred)
                queue.append((pred, dist + 1))
    return distance_map


def cfg_distance(distance_map: dict[int, int], cfg, addr: int) -> int | None:
    """
    Look up the precomputed CFG distance for ``addr`` (typically a state's PC).

    Resolves ``addr`` to the basic block that contains it, then indexes ``distance_map``
    (as built by :func:`build_cfg_distance_map`).

    :param distance_map: The map returned by :func:`build_cfg_distance_map`.
    :param cfg:          The same CFG passed to :func:`build_cfg_distance_map`.
    :param addr:         A (possibly mid-block) address, e.g. ``state.addr``.
    :return:             The shortest CFG distance to the target, or ``None`` if ``addr`` is
                         not in the CFG or the target is unreachable from it.
    """
    node = cfg.model.get_any_node(addr, anyaddr=True)
    if node is None:
        return None
    return distance_map.get(node.addr)
