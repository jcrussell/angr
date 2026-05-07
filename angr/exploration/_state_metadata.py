"""Per-state metadata for the Rust exploration manager.

Three metadata maps are tracked per state_id, all keyed by memory address:
- ``symbolic_pages``: addr -> claripy AST for whole-page symbolic content
  preserved across Python fallback so Rust can re-establish symbolic memory.
- ``hook_symbolic_memory``: addr -> (ast, size) for symbolic writes performed
  inside Python hooks; replayed to Rust on resume.
- ``addr_to_ast``: addr -> (ast, size) recorded by handle registration so
  state export can recover the original symbol instead of a fresh BVS.

Consolidating these into a single dataclass per state replaces three parallel
``Dict[int, Dict[int, ...]]`` maps and gives a clear schema for callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class StateMetadata:
    """Per-state metadata stored by ``RustExplorationManager``."""

    symbolic_pages: Dict[int, Any] = field(default_factory=dict)
    hook_symbolic_memory: Dict[int, Tuple[Any, int]] = field(default_factory=dict)
    addr_to_ast: Dict[int, Tuple[Any, int]] = field(default_factory=dict)
