"""Content fingerprinting + determinism gate for the angr Rust engine.

Bead angr-vh834, Phase 3.

A terminal (found / deadended) symbolic state is fingerprinted by its
*content* -- program counter, the set of structural constraint hashes, and
the set of symbol (leaf variable) names -- rather than by its volatile
state-id. Because parallel exploration (workers=1 vs workers=2, cancel-token
over-collection, etc.) can reorder and duplicate terminal states, the gate
compares *sets* of fingerprints: dedup absorbs over-collection and set
equality is order-insensitive but content-sensitive.

The fingerprint functions accept either a live ``RustStateProxy`` (which
exposes ``.pc`` and ``.constraints``) or any small adapter object with the
same two attributes, so the logic is testable without a live parallel run.

claripy ASTs are hashable and expose a stable, process-independent structural
hash via ``ast.hash()`` (md5-backed); we reuse that rather than re-walking the
AST ourselves. ``ast.variables`` is a frozenset of leaf variable names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any, Protocol


def _stable_digest(*parts: str) -> int:
    """Process-stable 128-bit hash of canonical string parts.

    The determinism gate compares fingerprints across SEPARATE PROCESSES
    (``run_single.py`` runs workers=1 and workers=2 as subprocesses), so the
    final reduction must not use Python's builtin ``hash()`` -- string hashing
    is randomized per process (``PYTHONHASHSEED``), which would make every
    cross-process fingerprint mismatch even for identical states. ``ast.hash()``
    is md5-backed and stable, but the symbol-name strings folded into the
    fingerprint are not, so we reduce through ``hashlib`` instead.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


class TerminalLike(Protocol):
    """Minimal interface a terminal state must expose to be fingerprinted."""

    @property
    def pc(self) -> int: ...

    @property
    def constraints(self) -> list: ...


def _ast_structural_hash(ast: Any) -> int:
    """Stable structural hash of a single claripy AST.

    Prefers ``ast.hash()`` (claripy's content-addressed, process-independent
    md5 hash). Falls back to the builtin ``hash()`` for non-claripy objects so
    the helper degrades gracefully in tests that pass plain hashables.
    """
    h = getattr(ast, "hash", None)
    if callable(h):
        return h()
    return hash(ast)


def _ast_variables(ast: Any) -> frozenset:
    """Leaf variable names of a single claripy AST (empty if unavailable)."""
    v = getattr(ast, "variables", None)
    if v is None:
        return frozenset()
    return frozenset(v)


def fingerprint_state(pc: int, constraints: list) -> int:
    """Strong content fingerprint of a terminal state.

    ``hash(( found_pc,
             frozenset(constraint_structural_hashes),
             frozenset(symbol_names) ))``

    Order-insensitive over the constraint list (frozensets) and sensitive to
    pc, constraint structure, and the names of the symbolic leaves. Two states
    collide iff they share a pc, the same *set* of structural constraint
    hashes, and the same *set* of symbol names.
    """
    constraint_hashes = {_ast_structural_hash(c) for c in constraints}
    symbol_names: frozenset = frozenset()
    for c in constraints:
        symbol_names |= _ast_variables(c)
    return _stable_digest(
        f"pc:{int(pc)}",
        "C:" + ",".join(sorted(str(h) for h in constraint_hashes)),
        "S:" + ",".join(sorted(symbol_names)),
    )


def _ast_shape(ast: Any) -> tuple:
    """Name-independent shape descriptor of a single AST: (op, depth).

    Both ``op`` and ``depth`` are structural and do not embed leaf names, so a
    pure symbol rename leaves the shape unchanged.
    """
    op = getattr(ast, "op", None)
    depth = getattr(ast, "depth", None)
    return (op, depth)


def fingerprint_state_shape(pc: int, constraints: list) -> int:
    """Rename-invariant (WEAKER) content fingerprint of a terminal state.

    Uses the *count* of distinct symbol names plus the sorted multiset of
    per-constraint (op, depth) shapes instead of the names themselves. This is
    stable under a pure symbol rename (e.g. ``x`` -> ``y`` with identical
    structure), which the strong :func:`fingerprint_state` deliberately is not.

    WEAKER -- DELIBERATELY LOSSY: because names are dropped, this CANNOT catch
    a shape-preserving symbol rebind (swapping which symbol feeds which
    constraint while keeping every op/depth identical produces an identical
    shape fingerprint). Use it only when symbol names are known to be
    nondeterministic across runs; otherwise prefer :func:`fingerprint_state`.
    """
    symbol_names: frozenset = frozenset()
    for c in constraints:
        symbol_names |= _ast_variables(c)
    symbol_count = len(symbol_names)
    shapes = sorted(f"{op}:{depth}" for op, depth in (_ast_shape(c) for c in constraints))
    return _stable_digest(
        f"pc:{int(pc)}",
        f"N:{symbol_count}",
        "SH:" + ",".join(shapes),
    )


def _fingerprint_one(state: Any, *, shape: bool = False) -> int:
    """Fingerprint a single terminal-like object (``.pc`` + ``.constraints``)."""
    pc = state.pc
    constraints = list(state.constraints)
    if shape:
        return fingerprint_state_shape(pc, constraints)
    return fingerprint_state(pc, constraints)


def fingerprint_terminals(states: Iterable[Any], *, shape: bool = False) -> set:
    """Map :func:`fingerprint_state` over a collection of terminal states.

    Each element must expose ``.pc`` (int) and ``.constraints`` (list of
    claripy ASTs). Returns a *set* of fingerprints -- duplicates (e.g. from
    cancel-token over-collection) collapse automatically. Pass ``shape=True``
    to use the rename-invariant variant.
    """
    return {_fingerprint_one(s, shape=shape) for s in states}


def compare_fingerprint_sets(set_a: set, set_b: set) -> dict:
    """Order-insensitive comparison of two fingerprint sets.

    Returns a dict with keys:
      - ``equal``        : bool, ``set_a == set_b``
      - ``only_in_a``    : fingerprints present in A but not B
      - ``only_in_b``    : fingerprints present in B but not A
      - ``count_a``      : ``len(set_a)``
      - ``count_b``      : ``len(set_b)``
      - ``common``       : count of shared fingerprints
    """
    a = set(set_a)
    b = set(set_b)
    only_in_a = a - b
    only_in_b = b - a
    return {
        "equal": a == b,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "count_a": len(a),
        "count_b": len(b),
        "common": len(a & b),
    }
