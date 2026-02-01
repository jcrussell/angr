"""
Differential testing framework for Rust vs Python VEX engines.

This module provides true differential testing that compares outputs
from both engines for the same inputs using shellcode-based testing.
"""
from __future__ import annotations

from .harness import DifferentialTestCase, DifferentialHarness
from .divergence import DivergenceTracker, DivergenceReport, DivergenceCategory

__all__ = [
    "DifferentialTestCase",
    "DifferentialHarness",
    "DivergenceTracker",
    "DivergenceReport",
    "DivergenceCategory",
]
