"""Rust-accelerated symbolic exploration engine for angr.

This package provides :class:`RustExplorationManager`, a drop-in replacement
for angr's :class:`SimulationManager` that executes VEX IR in a native Rust
interpreter with an integrated Z3 solver.  Typical speedups range from 1.5x
to 15x on CTF-style reverse-engineering challenges.

Quick start::

    import angr

    proj = angr.Project("/path/to/binary", auto_load_libs=False)
    state = proj.factory.entry_state()

    # Create the Rust-backed exploration manager
    mgr = proj.factory.simulation_manager(state, use_rust_engine=True)

    # Or directly:
    from angr.exploration import RustExplorationManager
    mgr = RustExplorationManager(proj, [state])

    mgr.explore(find=0x401234, avoid=0x401000)
    if mgr.found:
        print(mgr.found[0].posix.dumps(0))

Callable predicates, exploration techniques (DFS, LengthLimiter, etc.),
and the standard stash API (active, found, deadended, …) are supported.
"""
from __future__ import annotations

from angr.rustylib.vex_engine import (
    RustExecutionError,
    RustMalformedIRSBError,
    RustOomError,
    RustUnsupportedSyscallError,
    RustUnsupportedVexOpError,
    RustZ3Error,
)

from .rust_manager import RustErrorRecord, RustExplorationManager

__all__ = [
    'RustErrorRecord',
    'RustExecutionError',
    'RustExplorationManager',
    'RustMalformedIRSBError',
    'RustOomError',
    'RustUnsupportedSyscallError',
    'RustUnsupportedVexOpError',
    'RustZ3Error',
]
