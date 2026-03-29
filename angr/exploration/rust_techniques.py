"""ExplorationTechnique integration for the Rust exploration manager.

Provides use_technique(), remove_technique(), and the per-step filter/complete
callback dispatching that routes through RustStateProxy objects.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from angr.exploration.rust_manager import RustExplorationManager

l = logging.getLogger(name=__name__)


def use_technique(mgr: "RustExplorationManager", technique, **kwargs):
    """Apply an exploration technique to the Rust manager.

    Techniques are tracked and their setup methods are called.
    Common techniques like DFS, BFS, and LoopSeer have basic support.

    Args:
        mgr: The RustExplorationManager instance.
        technique: An ExplorationTechnique instance.

    Returns:
        The technique, for chaining.
    """
    tech_name = type(technique).__name__

    # Track the technique
    mgr._active_techniques.append(technique)

    # Call setup if available
    try:
        if hasattr(technique, 'setup'):
            technique.setup(mgr)
            l.debug(f"P9: Called setup() on technique {tech_name}")
    except Exception as e:
        l.warning(f"P9: Technique {tech_name} setup failed: {e}")

    # Handle specific technique types
    # DFS: Use depth-first state selection (LIFO)
    if tech_name == 'DFS' or tech_name == 'DepthFirst':
        try:
            mgr._rust_mgr.set_state_selection_lifo()
            l.debug("P9: Enabled DFS (LIFO) state selection")
        except AttributeError:
            l.debug("P9: DFS technique registered (LIFO not natively supported)")

    # BFS: Use breadth-first state selection (FIFO) - default behavior
    elif tech_name == 'BFS' or tech_name == 'BreadthFirst':
        try:
            mgr._rust_mgr.set_state_selection_fifo()
            l.debug("P9: Enabled BFS (FIFO) state selection")
        except AttributeError:
            l.debug("P9: BFS technique registered (default FIFO selection)")

    # LoopSeer: Loop detection and handling
    elif tech_name == 'LoopSeer':
        l.debug("P9: LoopSeer technique registered (basic support)")

    # Explorer: Extract find/avoid addresses
    elif tech_name == 'Explorer':
        find_addrs = []
        avoid_addrs = []
        find_func = getattr(technique, 'find', None)
        avoid_func = getattr(technique, 'avoid', None)

        # Extract addresses from _extra_stop_points by testing each
        # against the find/avoid lambdas with a mock state
        stop_points = getattr(technique, '_extra_stop_points', set())
        if stop_points and callable(find_func) and callable(avoid_func):
            class _MockState:
                def __init__(self, addr):
                    self.addr = addr
                    self._ip = addr
                    self.regs = type('regs', (), {'ip': addr})()
                def block(self, *a, **kw):
                    return type('block', (), {'size': 1})()

            for addr in stop_points:
                mock = _MockState(addr)
                try:
                    if find_func(mock):
                        find_addrs.append(addr)
                        continue
                except Exception:
                    pass
                try:
                    if avoid_func(mock):
                        avoid_addrs.append(addr)
                except Exception:
                    pass

        if find_addrs:
            mgr._rust_mgr.set_find_addrs(find_addrs)
        if avoid_addrs:
            mgr._rust_mgr.set_avoid_addrs(avoid_addrs)
        num_find = getattr(technique, 'num_find', 1)
        mgr._rust_mgr.set_num_find(num_find)
        l.debug(f"P9: Explorer technique: find={[hex(a) for a in find_addrs]}, "
                f"avoid={len(avoid_addrs)} addrs")

    # Other techniques
    else:
        l.debug(f"P9: Technique {tech_name} registered (limited support)")

    return technique


def remove_technique(mgr: "RustExplorationManager", technique) -> bool:
    """Remove an exploration technique.

    Args:
        mgr: The RustExplorationManager instance.
        technique: The technique to remove.

    Returns:
        True if removed, False if not found.
    """
    try:
        mgr._active_techniques.remove(technique)
        return True
    except ValueError:
        return False


def apply_technique_filters(mgr: "RustExplorationManager"):
    """Apply ExplorationTechnique filter() callbacks via proxy.

    After each step, iterate active states through each technique's
    filter() method. If filter() returns a stash name other than
    'active', move the state to that stash in Rust.
    """
    from angr.exploration.rust_state_proxy import RustStateProxy, RustSimulationManagerProxy

    if not mgr._active_techniques:
        return

    simgr_proxy = RustSimulationManagerProxy(
        mgr._rust_mgr,
        project=mgr._project,
        stdin_vars=getattr(mgr, '_stdin_vars', None),
        stdout_tracker=getattr(mgr, '_stdout_tracker', {}),
    )

    active_ids = list(mgr._rust_mgr.get_state_ids('active'))
    for sid in active_ids:
        state_proxy = RustStateProxy(
            mgr._rust_mgr, sid, project=mgr._project,
        )
        # Run through each technique's filter in order
        goto = None
        for tech in mgr._active_techniques:
            if hasattr(tech, 'filter'):
                try:
                    result = tech.filter(simgr_proxy, state_proxy)
                    if result is not None and result != 'active':
                        goto = result
                        break
                except Exception as e:
                    l.debug(f"Technique {type(tech).__name__}.filter() error: {e}")

        if goto is not None and goto != 'active':
            try:
                mgr._rust_mgr.move_state(sid, 'active', goto)
            except Exception as e:
                l.debug(f"Failed to move state {sid} to {goto}: {e}")


def check_technique_complete(mgr: "RustExplorationManager") -> bool:
    """Check ExplorationTechnique complete() callbacks.

    Returns True if any technique says exploration is complete.
    """
    from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

    if not mgr._active_techniques:
        return False

    simgr_proxy = RustSimulationManagerProxy(
        mgr._rust_mgr,
        project=mgr._project,
        stdin_vars=getattr(mgr, '_stdin_vars', None),
        stdout_tracker=getattr(mgr, '_stdout_tracker', {}),
    )

    for tech in mgr._active_techniques:
        if hasattr(tech, 'complete'):
            try:
                if tech.complete(simgr_proxy):
                    l.debug(f"Technique {type(tech).__name__}.complete() returned True")
                    return True
            except Exception as e:
                l.debug(f"Technique {type(tech).__name__}.complete() error: {e}")

    return False
