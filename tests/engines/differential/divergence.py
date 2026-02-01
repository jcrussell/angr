"""
Divergence tracking and reporting for differential testing.

Provides DivergenceTracker to categorize and track differences between
Rust and Python engine outputs, with JSON report generation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .harness import DifferentialTestCase, ComparisonResult

l = logging.getLogger(__name__)


class DivergenceCategory(Enum):
    """Categories for divergences between engines."""
    BUG = "bug"                      # Likely a bug in one engine
    UNSUPPORTED_OP = "unsupported_op"  # Operation not supported by Rust
    EDGE_CASE = "edge_case"          # Edge case behavior difference
    ROUNDING = "rounding"            # Floating point rounding difference
    FLAGS = "flags"                  # Flag computation difference
    UNKNOWN = "unknown"              # Needs investigation


@dataclass
class Divergence:
    """Record of a single divergence between engines."""
    test_name: str
    seed: int
    category: DivergenceCategory
    shellcode_hex: str
    register_diffs: dict[str, dict[str, int]]
    arch: str = "x86"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_name": self.test_name,
            "seed": self.seed,
            "category": self.category.value,
            "shellcode_hex": self.shellcode_hex,
            "register_diffs": self.register_diffs,
            "arch": self.arch,
            "notes": self.notes,
        }


@dataclass
class DivergenceReport:
    """Summary report of differential testing results."""
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    divergences: list[Divergence] = field(default_factory=list)
    by_category: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    @property
    def pass_rate(self) -> float:
        tested = self.passed + self.failed
        if tested == 0:
            return 0.0
        return self.passed / tested

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "total": self.total,
                "pass_rate": round(self.pass_rate, 4),
            },
            "by_category": self.by_category,
            "divergences": [d.to_dict() for d in self.divergences],
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


class DivergenceTracker:
    """
    Tracks and categorizes divergences between Rust and Python engines.

    Provides automatic categorization based on divergence patterns and
    generates comprehensive reports.
    """

    # Patterns for automatic categorization
    FLAG_REGISTERS = {"eflags", "rflags", "cf", "zf", "sf", "of", "pf", "af"}
    FLOAT_REGISTERS = {"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"}

    def __init__(self):
        self._divergences: list[Divergence] = []
        self._passed = 0
        self._failed = 0
        self._skipped = 0

    def record_pass(self) -> None:
        """Record a passing test."""
        self._passed += 1

    def record_skip(self) -> None:
        """Record a skipped test."""
        self._skipped += 1

    def record(
        self,
        test: DifferentialTestCase,
        rust_result: dict[str, int],
        python_result: dict[str, int],
        category: DivergenceCategory | None = None
    ) -> None:
        """
        Record a divergence between engines.

        Args:
            test: The test case that diverged
            rust_result: Rust engine register results
            python_result: Python engine register results
            category: Optional explicit category (auto-detected if None)
        """
        self._failed += 1

        # Build register diffs
        register_diffs = {}
        for reg in test.compare_regs:
            rust_val = rust_result.get(reg, -1)
            python_val = python_result.get(reg, -1)
            if rust_val != python_val:
                register_diffs[reg] = {"python": python_val, "rust": rust_val}

        # Auto-categorize if not provided
        if category is None:
            category = self._categorize(test, register_diffs)

        divergence = Divergence(
            test_name=test.name,
            seed=test.seed,
            category=category,
            shellcode_hex=test.shellcode_hex,
            register_diffs=register_diffs,
            arch=test.arch,
        )

        self._divergences.append(divergence)

    def record_from_comparison(
        self,
        result: ComparisonResult,
        category: DivergenceCategory | None = None
    ) -> None:
        """
        Record a divergence from a ComparisonResult.

        Args:
            result: The comparison result
            category: Optional explicit category
        """
        if result.match:
            self.record_pass()
            return

        self._failed += 1

        # Auto-categorize if not provided
        if category is None:
            category = self._categorize(result.test, result.register_diffs)

        divergence = Divergence(
            test_name=result.test.name,
            seed=result.test.seed,
            category=category,
            shellcode_hex=result.test.shellcode_hex,
            register_diffs=result.register_diffs,
            arch=result.test.arch,
        )

        self._divergences.append(divergence)

    def _categorize(
        self,
        test: DifferentialTestCase,
        register_diffs: dict[str, dict[str, int]]
    ) -> DivergenceCategory:
        """
        Automatically categorize a divergence based on patterns.

        Args:
            test: The test case
            register_diffs: Register differences

        Returns:
            Detected category
        """
        diff_regs = set(register_diffs.keys())

        # Check if only flags differ
        if diff_regs and diff_regs.issubset(self.FLAG_REGISTERS):
            return DivergenceCategory.FLAGS

        # Check if float registers differ (potential rounding)
        if diff_regs and diff_regs.issubset(self.FLOAT_REGISTERS):
            return DivergenceCategory.ROUNDING

        # Check for edge case patterns (0, -1, max values)
        for reg, vals in register_diffs.items():
            python_val = vals.get("python", 0)
            rust_val = vals.get("rust", 0)

            # Check for wrap-around or sign issues
            if (python_val == 0 and rust_val != 0) or (rust_val == 0 and python_val != 0):
                return DivergenceCategory.EDGE_CASE

            # Check for sign extension issues
            if abs(python_val - rust_val) in [0x100, 0x10000, 0x100000000]:
                return DivergenceCategory.EDGE_CASE

        return DivergenceCategory.UNKNOWN

    def report(self) -> DivergenceReport:
        """
        Generate a summary report.

        Returns:
            DivergenceReport with summary statistics
        """
        # Count by category
        by_category: dict[str, int] = {}
        for div in self._divergences:
            cat = div.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        return DivergenceReport(
            passed=self._passed,
            failed=self._failed,
            skipped=self._skipped,
            divergences=self._divergences.copy(),
            by_category=by_category,
        )

    def clear(self) -> None:
        """Clear all recorded data."""
        self._divergences = []
        self._passed = 0
        self._failed = 0
        self._skipped = 0

    @property
    def pass_rate(self) -> float:
        """Current pass rate."""
        tested = self._passed + self._failed
        if tested == 0:
            return 0.0
        return self._passed / tested

    @property
    def failure_count(self) -> int:
        """Number of failures recorded."""
        return self._failed

    def summary(self) -> str:
        """Generate a human-readable summary."""
        report = self.report()
        lines = [
            f"Differential Test Results",
            f"=" * 40,
            f"Passed: {report.passed}",
            f"Failed: {report.failed}",
            f"Skipped: {report.skipped}",
            f"Pass Rate: {report.pass_rate:.1%}",
            "",
            "Failures by Category:",
        ]

        for cat, count in sorted(report.by_category.items()):
            lines.append(f"  {cat}: {count}")

        if report.divergences:
            lines.append("")
            lines.append("First 5 Divergences:")
            for div in report.divergences[:5]:
                lines.append(f"  - {div.test_name} (seed={div.seed}): {div.category.value}")
                for reg, vals in div.register_diffs.items():
                    lines.append(f"      {reg}: Python={vals['python']}, Rust={vals['rust']}")

        return "\n".join(lines)
