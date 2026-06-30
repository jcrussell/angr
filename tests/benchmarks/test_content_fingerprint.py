"""Tests for content_fingerprint -- the determinism-gate fingerprint helper.

Self-contained: builds claripy expressions directly and wraps them in a tiny
synthetic terminal record, so no live RustExplorationManager / Rust rebuild is
required. claripy is importable in the venv as-is.

Bead angr-vh834, Phase 3.
"""

from __future__ import annotations

import claripy
import pytest
from content_fingerprint import (
    compare_fingerprint_sets,
    fingerprint_state,
    fingerprint_state_shape,
    fingerprint_terminals,
)


class Terminal:
    """Synthetic stand-in for a RustStateProxy: exposes .pc and .constraints."""

    def __init__(self, pc, constraints):
        self.pc = pc
        self.constraints = list(constraints)


# --------------------------------------------------------------------------
# Strong fingerprint: identical content -> identical hash
# --------------------------------------------------------------------------


def test_identical_content_same_fingerprint():
    x = claripy.BVS("x", 64, explicit_name=True)
    cons = [x > 5, x < 100]
    fp1 = fingerprint_state(0x400000, cons)
    fp2 = fingerprint_state(0x400000, list(cons))
    assert fp1 == fp2


def test_constraint_order_does_not_matter():
    x = claripy.BVS("x", 64, explicit_name=True)
    a = x > 5
    b = x < 100
    assert fingerprint_state(0x400000, [a, b]) == fingerprint_state(0x400000, [b, a])


def test_duplicate_constraints_absorbed():
    # frozenset of structural hashes dedups repeated constraints.
    x = claripy.BVS("x", 64, explicit_name=True)
    a = x > 5
    assert fingerprint_state(0x10, [a]) == fingerprint_state(0x10, [a, a, a])


# --------------------------------------------------------------------------
# Strong fingerprint: differing content -> differing hash
# --------------------------------------------------------------------------


def test_different_pc_differs():
    x = claripy.BVS("x", 64, explicit_name=True)
    cons = [x > 5]
    assert fingerprint_state(0x1000, cons) != fingerprint_state(0x2000, cons)


def test_different_constraint_differs():
    x = claripy.BVS("x", 64, explicit_name=True)
    assert fingerprint_state(0x1000, [x > 5]) != fingerprint_state(0x1000, [x > 6])


def test_different_symbol_name_differs():
    x = claripy.BVS("x", 64, explicit_name=True)
    y = claripy.BVS("y", 64, explicit_name=True)
    # Same structure (UGT against same constant), different leaf name.
    assert fingerprint_state(0x1000, [x > 5]) != fingerprint_state(0x1000, [y > 5])


# --------------------------------------------------------------------------
# Set-level: order-insensitive, dedup-absorbing
# --------------------------------------------------------------------------


def test_terminal_set_order_insensitive():
    x = claripy.BVS("x", 64, explicit_name=True)
    t1 = Terminal(0x1000, [x > 5])
    t2 = Terminal(0x2000, [x < 100])
    set_a = fingerprint_terminals([t1, t2])
    set_b = fingerprint_terminals([t2, t1])
    assert set_a == set_b


def test_terminal_set_dedups_overcollection():
    x = claripy.BVS("x", 64, explicit_name=True)
    t = Terminal(0x1000, [x > 5])
    # Same content collected three times (cancel-token over-collection).
    s = fingerprint_terminals([t, Terminal(0x1000, [x > 5]), Terminal(0x1000, [x > 5])])
    assert len(s) == 1


# --------------------------------------------------------------------------
# Rename-invariant (shape) variant
# --------------------------------------------------------------------------


def test_shape_stable_under_pure_rename():
    x = claripy.BVS("x", 64, explicit_name=True)
    y = claripy.BVS("y", 64, explicit_name=True)
    cons_x = [x > 5, x < 100]
    cons_y = [y > 5, y < 100]
    # Pure rename: identical op/depth shape, single symbol either way.
    assert fingerprint_state_shape(0x1000, cons_x) == fingerprint_state_shape(0x1000, cons_y)
    # ... but the STRONG variant must distinguish them.
    assert fingerprint_state(0x1000, cons_x) != fingerprint_state(0x1000, cons_y)


def test_shape_still_sensitive_to_pc_and_structure():
    x = claripy.BVS("x", 64, explicit_name=True)
    cons = [x > 5]
    assert fingerprint_state_shape(0x1000, cons) != fingerprint_state_shape(0x2000, cons)
    # Different op (UGT vs ULT) -> different shape.
    assert fingerprint_state_shape(0x1000, [x > 5]) != fingerprint_state_shape(0x1000, [x < 5])


def test_shape_terminals_via_flag():
    x = claripy.BVS("x", 64, explicit_name=True)
    y = claripy.BVS("y", 64, explicit_name=True)
    t_x = Terminal(0x1000, [x > 5])
    t_y = Terminal(0x1000, [y > 5])
    strong = fingerprint_terminals([t_x, t_y])
    shape = fingerprint_terminals([t_x, t_y], shape=True)
    assert len(strong) == 2  # names distinguish
    assert len(shape) == 1  # rename-invariant collapses them


# --------------------------------------------------------------------------
# compare_fingerprint_sets
# --------------------------------------------------------------------------


def test_compare_equal_sets():
    x = claripy.BVS("x", 64, explicit_name=True)
    a = fingerprint_terminals([Terminal(0x1, [x > 5]), Terminal(0x2, [x < 9])])
    b = fingerprint_terminals([Terminal(0x2, [x < 9]), Terminal(0x1, [x > 5])])
    res = compare_fingerprint_sets(a, b)
    assert res["equal"] is True
    assert res["only_in_a"] == set()
    assert res["only_in_b"] == set()
    assert res["count_a"] == res["count_b"] == 2
    assert res["common"] == 2


def test_compare_divergent_sets():
    x = claripy.BVS("x", 64, explicit_name=True)
    shared = Terminal(0x1, [x > 5])
    extra = Terminal(0x9, [x < 3])
    a = fingerprint_terminals([shared])
    b = fingerprint_terminals([shared, extra])
    res = compare_fingerprint_sets(a, b)
    assert res["equal"] is False
    assert res["only_in_a"] == set()
    extra_fp = fingerprint_state(extra.pc, extra.constraints)
    assert res["only_in_b"] == {extra_fp}
    assert res["count_a"] == 1
    assert res["count_b"] == 2
    assert res["common"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
