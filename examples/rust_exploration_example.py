#!/usr/bin/env python3
"""Example: Using the Rust exploration engine for symbolic execution.

This script demonstrates the main RustExplorationManager API using the
fauxware binary from angr-examples.  It shows three usage patterns:

1. Address-based find/avoid
2. Callable predicates (find states whose stdout contains a string)
3. Using exploration techniques (DFS, LengthLimiter)

Requirements:
    pip install angr      # with Rust engine built (setuptools-rust)
    git clone https://github.com/angr/angr-examples  # for test binaries

Usage:
    python examples/rust_exploration_example.py /path/to/angr-examples/examples
"""

from __future__ import annotations

import os
import sys

import angr


def example_address_find(binary_path):
    """Find a target address while avoiding another."""
    proj = angr.Project(binary_path, auto_load_libs=False)
    state = proj.factory.entry_state()

    # Create Rust-backed simulation manager via factory
    simgr = proj.factory.simulation_manager(state, use_rust_engine=True)

    # Explore with address-based find/avoid
    simgr.explore(find=0x400844, avoid=0x40084E)  # "Welcome" vs "Go away"

    if simgr.found:
        found = simgr.found[0]
        # Extract the symbolic stdin that reaches the target
        stdin_data = found.posix.dumps(0)
        print(f"[address] Input to reach 0x400844: {stdin_data!r}")
    else:
        print("[address] No solution found")


def example_callable_predicate(binary_path):
    """Find states using a callable predicate on stdout."""
    proj = angr.Project(binary_path, auto_load_libs=False)
    state = proj.factory.entry_state()

    simgr = proj.factory.simulation_manager(state, use_rust_engine=True)

    # Callable predicates: check state properties dynamically
    simgr.explore(
        find=lambda s: b"Welcome" in s.posix.dumps(1),
        avoid=lambda s: b"Go away" in s.posix.dumps(1),
    )

    if simgr.found:
        found = simgr.found[0]
        print(f"[predicate] Stdout: {found.posix.dumps(1)!r}")
        print(f"[predicate] Stdin:  {found.posix.dumps(0)!r}")
    else:
        print("[predicate] No solution found")


def example_with_techniques(binary_path):
    """Use exploration techniques for search control."""
    proj = angr.Project(binary_path, auto_load_libs=False)
    state = proj.factory.entry_state()

    simgr = proj.factory.simulation_manager(state, use_rust_engine=True)

    # LengthLimiter: bound exploration depth
    from angr.exploration_techniques import LengthLimiter

    simgr.use_technique(LengthLimiter(max_length=200))

    simgr.explore(find=0x400844, avoid=0x40084E, timeout=30)

    if simgr.found:
        found = simgr.found[0]
        print(f"[techniques] Found with depth limit. Input: {found.posix.dumps(0)!r}")
    else:
        print("[techniques] No solution found within depth limit")


def example_direct_api(binary_path):
    """Use RustExplorationManager directly (without factory)."""
    from angr.exploration import RustExplorationManager

    proj = angr.Project(binary_path, auto_load_libs=False)
    state = proj.factory.entry_state()

    mgr = RustExplorationManager(proj, [state])
    mgr.set_find_addresses([0x400844])
    mgr.set_avoid_addresses([0x40084E])
    mgr.run(max_steps=10000)

    for s in mgr.found:
        print(f"[direct] Found at {hex(s.addr)}, stdin: {s.posix.dumps(0)!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    examples_dir = sys.argv[1]
    binary = os.path.join(examples_dir, "fauxware", "fauxware")
    if not os.path.isfile(binary):
        print(f"Binary not found: {binary}")
        print("Clone angr-examples and pass the examples/ directory as argument.")
        sys.exit(1)

    print("=== Address-based find/avoid ===")
    example_address_find(binary)

    print("\n=== Callable predicates ===")
    example_callable_predicate(binary)

    print("\n=== Exploration techniques ===")
    example_with_techniques(binary)

    print("\n=== Direct API ===")
    example_direct_api(binary)
