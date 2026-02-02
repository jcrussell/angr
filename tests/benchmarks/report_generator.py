#!/usr/bin/env python3
"""
Report generator for Rust vs Python VEX engine benchmarks.

This module:
1. Parses Rust Criterion JSON output from target/criterion/
2. Collects Python benchmark results
3. Generates comparison tables with speedup calculations
4. Includes system info (platform, versions)
5. Outputs consolidated markdown report

Run standalone: python tests/benchmarks/report_generator.py
"""
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class CriterionResult:
    """Parsed result from Criterion benchmark."""
    name: str
    mean_ns: float
    std_dev_ns: float
    median_ns: float
    iterations: int
    unit: str = "ns"

    @property
    def mean_us(self) -> float:
        return self.mean_ns / 1000

    @property
    def mean_ms(self) -> float:
        return self.mean_ns / 1_000_000

    @property
    def mean_s(self) -> float:
        return self.mean_ns / 1_000_000_000

    @property
    def std_dev_ms(self) -> float:
        return self.std_dev_ns / 1_000_000

    @property
    def throughput(self) -> float:
        """Operations per second."""
        return 1_000_000_000 / self.mean_ns if self.mean_ns > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_ns": self.mean_ns,
            "mean_us": self.mean_us,
            "mean_ms": self.mean_ms,
            "std_dev_ns": self.std_dev_ns,
            "median_ns": self.median_ns,
            "iterations": self.iterations,
            "throughput_ops_per_s": self.throughput,
        }


class CriterionParser:
    """Parser for Criterion benchmark output."""

    def __init__(self, criterion_dir: str | Path):
        self.criterion_dir = Path(criterion_dir)

    def find_benchmark_dirs(self) -> list[Path]:
        """Find all benchmark result directories."""
        if not self.criterion_dir.exists():
            return []

        benchmark_dirs = []
        for item in self.criterion_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Check for new/estimates.json which indicates valid benchmark data
                estimates = item / "new" / "estimates.json"
                if estimates.exists():
                    benchmark_dirs.append(item)

                # Also check for benchmark groups
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        sub_estimates = subitem / "new" / "estimates.json"
                        if sub_estimates.exists():
                            benchmark_dirs.append(subitem)

        return benchmark_dirs

    def parse_estimates(self, estimates_path: Path) -> dict[str, Any] | None:
        """Parse a Criterion estimates.json file."""
        try:
            with open(estimates_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not parse {estimates_path}: {e}")
            return None

    def parse_benchmark(self, benchmark_dir: Path) -> CriterionResult | None:
        """Parse a single benchmark directory."""
        estimates_path = benchmark_dir / "new" / "estimates.json"
        if not estimates_path.exists():
            return None

        estimates = self.parse_estimates(estimates_path)
        if estimates is None:
            return None

        # Extract mean and std_dev from estimates
        mean_data = estimates.get("mean", {})
        std_dev_data = estimates.get("std_dev", {})
        median_data = estimates.get("median", {})

        mean_ns = mean_data.get("point_estimate", 0)
        std_dev_ns = std_dev_data.get("point_estimate", 0)
        median_ns = median_data.get("point_estimate", 0)

        # Try to get iteration count from sample.json
        sample_path = benchmark_dir / "new" / "sample.json"
        iterations = 0
        if sample_path.exists():
            try:
                with open(sample_path) as f:
                    sample = json.load(f)
                    iterations = len(sample.get("times", []))
            except (json.JSONDecodeError, OSError):
                pass

        # Get benchmark name from directory structure
        name = benchmark_dir.name
        if benchmark_dir.parent.name != "criterion":
            # Include parent group name
            name = f"{benchmark_dir.parent.name}/{name}"

        return CriterionResult(
            name=name,
            mean_ns=mean_ns,
            std_dev_ns=std_dev_ns,
            median_ns=median_ns,
            iterations=iterations,
        )

    def parse_all(self) -> list[CriterionResult]:
        """Parse all benchmark results."""
        results = []
        benchmark_dirs = self.find_benchmark_dirs()

        for bench_dir in benchmark_dirs:
            result = self.parse_benchmark(bench_dir)
            if result:
                results.append(result)

        return results


def get_system_info() -> dict[str, str]:
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "date": datetime.now().isoformat(),
    }

    # Try to get Rust version
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["rust_version"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["rust_version"] = "Unknown"

    # Try to get angr version
    try:
        import angr
        info["angr_version"] = angr.__version__
    except ImportError:
        info["angr_version"] = "Not installed"

    # Try to check Rust engine availability
    try:
        from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE
        info["rust_engine_available"] = str(RUST_ENGINE_AVAILABLE)
    except ImportError:
        info["rust_engine_available"] = "False"

    return info


def calculate_speedup(rust_time_ms: float, python_time_ms: float) -> float:
    """Calculate speedup ratio (how many times faster Rust is)."""
    if rust_time_ms <= 0:
        return 0.0
    return python_time_ms / rust_time_ms


def format_time(ns: float) -> str:
    """Format time in appropriate units."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f}s"
    elif ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    elif ns >= 1000:
        return f"{ns / 1000:.2f}us"
    else:
        return f"{ns:.2f}ns"


def format_throughput(ops_per_s: float) -> str:
    """Format throughput in appropriate units."""
    if ops_per_s >= 1_000_000_000:
        return f"{ops_per_s / 1_000_000_000:.1f}G ops/s"
    elif ops_per_s >= 1_000_000:
        return f"{ops_per_s / 1_000_000:.1f}M ops/s"
    elif ops_per_s >= 1000:
        return f"{ops_per_s / 1000:.1f}K ops/s"
    else:
        return f"{ops_per_s:.1f} ops/s"


class ReportGenerator:
    """Generate consolidated benchmark reports."""

    def __init__(self):
        self.rust_results: list[CriterionResult] = []
        self.python_results: list[dict[str, Any]] = []
        self.binary_results: list[dict[str, Any]] = []
        self.examples_results: list[dict[str, Any]] = []
        self.system_info: dict[str, str] = {}

    def load_criterion_results(self, criterion_dir: str | Path):
        """Load Rust Criterion benchmark results."""
        parser = CriterionParser(criterion_dir)
        self.rust_results = parser.parse_all()

    def load_python_results(self, results: list):
        """Load Python benchmark results."""
        for r in results:
            if hasattr(r, "to_dict"):
                self.python_results.append(r.to_dict())
            elif isinstance(r, dict):
                self.python_results.append(r)

    def load_binary_results(self, results: list):
        """Load binary benchmark results."""
        for r in results:
            if hasattr(r, "to_dict"):
                d = r.to_dict()
                # Mark comparison results
                if hasattr(r, "speedup"):
                    d["is_comparison"] = True
                self.binary_results.append(d)
            elif isinstance(r, dict):
                self.binary_results.append(r)

    def load_examples_results(self, comparisons: list):
        """Load angr-examples benchmark results."""
        for r in comparisons:
            if hasattr(r, "to_dict"):
                self.examples_results.append(r.to_dict())
            elif isinstance(r, dict):
                self.examples_results.append(r)

    def collect_system_info(self):
        """Collect system information."""
        self.system_info = get_system_info()

    def generate_report(self) -> str:
        """Generate the full benchmark report."""
        lines = [
            "# Rust vs Python VEX Engine Benchmark Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Executive Summary
        lines.extend(self._generate_executive_summary())

        # System Information
        lines.extend(self._generate_system_info())

        # Synthetic Benchmarks
        if self.rust_results or self.python_results:
            lines.extend(self._generate_synthetic_benchmarks())

        # Binary Benchmarks
        if self.binary_results:
            lines.extend(self._generate_binary_benchmarks())

        # angr-examples Benchmarks
        if self.examples_results:
            lines.extend(self._generate_examples_benchmarks())

        # Methodology
        lines.extend(self._generate_methodology())

        # Conclusions
        lines.extend(self._generate_conclusions())

        return "\n".join(lines)

    def _generate_executive_summary(self) -> list[str]:
        """Generate executive summary section."""
        lines = [
            "## Executive Summary",
            "",
        ]

        # Calculate overall speedups from comparison data
        speedups = []
        comparisons = self._get_benchmark_comparisons()
        for name, rust_ms, python_ms in comparisons:
            if rust_ms > 0 and python_ms > 0:
                speedup = python_ms / rust_ms
                speedups.append((name, speedup))

        if speedups:
            avg_speedup = sum(s[1] for s in speedups) / len(speedups)
            max_speedup = max(speedups, key=lambda x: x[1])
            min_speedup = min(speedups, key=lambda x: x[1])

            lines.extend([
                f"- **Benchmarks compared**: {len(speedups)}",
                f"- **Average speedup**: {avg_speedup:.1f}x",
                f"- **Maximum speedup**: {max_speedup[1]:.1f}x ({max_speedup[0]})",
                f"- **Minimum speedup**: {min_speedup[1]:.1f}x ({min_speedup[0]})",
                "",
                "### Key Findings",
                "",
            ])

            # Highlight specific findings
            fork_speedups = [(n, s) for n, s in speedups if "fork" in n.lower()]
            if fork_speedups:
                best_fork = max(fork_speedups, key=lambda x: x[1])
                lines.append(f"- Fork operations show up to **{best_fork[1]:.1f}x** speedup")

            mem_speedups = [(n, s) for n, s in speedups if "memory" in n.lower()]
            if mem_speedups:
                avg_mem = sum(s for _, s in mem_speedups) / len(mem_speedups)
                lines.append(f"- Memory operations average **{avg_mem:.1f}x** speedup")

            reg_speedups = [(n, s) for n, s in speedups if "register" in n.lower()]
            if reg_speedups:
                avg_reg = sum(s for _, s in reg_speedups) / len(reg_speedups)
                lines.append(f"- Register operations average **{avg_reg:.1f}x** speedup")

            lines.append("")

        # Include angr-examples summary if available
        if self.examples_results:
            ex_speedups = [
                r.get("speedup") for r in self.examples_results
                if r.get("speedup") is not None
            ]
            if ex_speedups:
                avg_ex_speedup = sum(ex_speedups) / len(ex_speedups)
                lines.extend([
                    "### angr-examples Performance",
                    "",
                    f"- **Examples with speedup data**: {len(ex_speedups)}",
                    f"- **Average speedup on real examples**: {avg_ex_speedup:.2f}x",
                    "",
                ])

        if not speedups and not self.examples_results:
            lines.extend([
                "No comparison data available.",
                "",
            ])

        return lines

    def _generate_system_info(self) -> list[str]:
        """Generate system information section."""
        lines = [
            "## System Information",
            "",
            "| Component | Value |",
            "|-----------|-------|",
        ]

        for key, value in self.system_info.items():
            # Format the key nicely
            display_key = key.replace("_", " ").title()
            lines.append(f"| {display_key} | {value} |")

        lines.append("")
        return lines

    def _generate_synthetic_benchmarks(self) -> list[str]:
        """Generate synthetic benchmarks section."""
        lines = [
            "## Synthetic Benchmarks",
            "",
        ]

        # Rust Engine Results
        if self.rust_results:
            lines.extend([
                "### Rust Engine (Criterion)",
                "",
                "| Benchmark | Mean | Throughput | Std Dev |",
                "|-----------|------|------------|---------|",
            ])

            for r in sorted(self.rust_results, key=lambda x: x.name):
                lines.append(
                    f"| {r.name} | {format_time(r.mean_ns)} | "
                    f"{format_throughput(r.throughput)} | {format_time(r.std_dev_ns)} |"
                )

            lines.append("")

        # Python Engine Results
        if self.python_results:
            lines.extend([
                "### Python Engine",
                "",
                "| Benchmark | Mean | Throughput | Std Dev |",
                "|-----------|------|------------|---------|",
            ])

            for r in sorted(self.python_results, key=lambda x: x.get("name", "")):
                mean_ms = r.get("mean_time_ms", 0)
                ops_per_s = r.get("ops_per_second", 0)
                std_dev_s = r.get("std_dev_s", 0)

                lines.append(
                    f"| {r.get('name', 'unknown')} | {mean_ms:.3f}ms | "
                    f"{format_throughput(ops_per_s)} | {std_dev_s*1000:.3f}ms |"
                )

            lines.append("")

        # Comparison table
        comparisons = self._get_benchmark_comparisons()
        if comparisons:
            lines.extend([
                "### Rust vs Python Comparison",
                "",
                "| Operation | Rust | Python | Speedup |",
                "|-----------|------|--------|---------|",
            ])

            for name, rust_ms, python_ms in sorted(comparisons, key=lambda x: x[0]):
                speedup = python_ms / rust_ms if rust_ms > 0 else 0
                lines.append(
                    f"| {name} | {rust_ms:.3f}ms | {python_ms:.3f}ms | {speedup:.1f}x |"
                )

            lines.append("")

        return lines

    def _generate_binary_benchmarks(self) -> list[str]:
        """Generate binary benchmarks section."""
        lines = [
            "## Real Binary Benchmarks",
            "",
        ]

        # Separate comparison results from single-engine results
        comparisons = [r for r in self.binary_results if r.get("is_comparison")]
        single_results = [r for r in self.binary_results if not r.get("is_comparison")]

        # Engine comparison table
        if comparisons:
            lines.extend([
                "### Engine Comparison (Rust vs Python)",
                "",
                "| Benchmark | Rust (ms) | Python (ms) | Speedup |",
                "|-----------|-----------|-------------|---------|",
            ])

            for r in sorted(comparisons, key=lambda x: x.get("name", "")):
                rust_ms = r.get("rust_time_ms", 0)
                python_ms = r.get("python_time_ms", 0)
                speedup = r.get("speedup", 0)
                lines.append(
                    f"| {r.get('name', 'unknown')} | {rust_ms:.2f} | {python_ms:.2f} | {speedup:.1f}x |"
                )

            # Summary stats
            if comparisons:
                avg_speedup = sum(r.get("speedup", 0) for r in comparisons) / len(comparisons)
                max_r = max(comparisons, key=lambda x: x.get("speedup", 0))
                lines.extend([
                    "",
                    f"**Average Speedup**: {avg_speedup:.1f}x",
                    f"**Maximum Speedup**: {max_r.get('speedup', 0):.1f}x ({max_r.get('name', 'unknown')})",
                    "",
                ])

        # Single engine benchmarks grouped by type
        if single_results:
            by_type: dict[str, list[dict]] = {}
            for r in single_results:
                bench_type = r.get("benchmark_type", "other")
                by_type.setdefault(bench_type, []).append(r)

            for bench_type, results in sorted(by_type.items()):
                lines.extend([
                    f"### {bench_type.title()}",
                    "",
                    "| Benchmark | Binary | Mean | Notes |",
                    "|-----------|--------|------|-------|",
                ])

                for r in sorted(results, key=lambda x: x.get("name", "")):
                    mean_ms = r.get("mean_time_ms", 0)
                    extra = r.get("extra_metrics", {})
                    notes = ", ".join(
                        f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in extra.items()
                    )
                    lines.append(
                        f"| {r.get('name', 'unknown')} | {r.get('binary', 'unknown')} | "
                        f"{mean_ms:.1f}ms | {notes} |"
                    )

                lines.append("")

        return lines

    def _generate_examples_benchmarks(self) -> list[str]:
        """Generate angr-examples benchmarks section."""
        lines = [
            "## angr-examples Benchmarks",
            "",
            "Comparison of Rust vs Python VEX engine on real-world CTF challenges",
            "from the [angr-examples](https://github.com/angr/angr-examples) repository.",
            "",
        ]

        if not self.examples_results:
            lines.extend(["No angr-examples results available.", ""])
            return lines

        # Count successes
        total = len(self.examples_results)
        python_success = sum(1 for r in self.examples_results if r.get("python_success"))
        rust_success = sum(1 for r in self.examples_results if r.get("rust_success"))
        both_success = sum(
            1 for r in self.examples_results
            if r.get("python_success") and r.get("rust_success")
        )

        lines.extend([
            "### Summary",
            "",
            f"- **Total examples**: {total}",
            f"- **Python success**: {python_success}/{total} ({100*python_success/total:.1f}%)",
            f"- **Rust success**: {rust_success}/{total} ({100*rust_success/total:.1f}%)",
            f"- **Both succeed**: {both_success}/{total} ({100*both_success/total:.1f}%)",
            "",
        ])

        # Speedup statistics
        speedups = [
            r.get("speedup") for r in self.examples_results
            if r.get("speedup") is not None
        ]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            lines.extend([
                "### Speedup Statistics",
                "",
                f"- **Average**: {avg_speedup:.2f}x",
                f"- **Minimum**: {min(speedups):.2f}x",
                f"- **Maximum**: {max(speedups):.2f}x",
                "",
            ])

        # Comparison table
        lines.extend([
            "### Comparison Table",
            "",
            "| Example | Python Time | Python Status | Rust Time | Rust Status | Speedup |",
            "|---------|-------------|---------------|-----------|-------------|---------|",
        ])

        for r in sorted(self.examples_results, key=lambda x: x.get("example_name", "")):
            name = r.get("example_name", "unknown")
            py_time = r.get("python_time_s")
            py_success = r.get("python_success", False)
            py_err = r.get("python_error")
            rust_time = r.get("rust_time_s")
            rust_success = r.get("rust_success", False)
            rust_err = r.get("rust_error")
            speedup = r.get("speedup")

            # Format values
            py_time_str = f"{py_time:.2f}s" if py_time else "N/A"
            py_status = "\u2713" if py_success else f"\u2717"
            if py_err and not py_success:
                py_status += f" ({py_err[:15]}...)" if len(py_err) > 15 else f" ({py_err})"

            rust_time_str = f"{rust_time:.2f}s" if rust_time else "N/A"
            rust_status = "\u2713" if rust_success else f"\u2717"
            if rust_err and not rust_success:
                rust_status += f" ({rust_err[:15]}...)" if len(rust_err) > 15 else f" ({rust_err})"

            speedup_str = f"{speedup:.1f}x" if speedup else "-"

            lines.append(
                f"| {name} | {py_time_str} | {py_status} | {rust_time_str} | {rust_status} | {speedup_str} |"
            )

        lines.append("")
        return lines

    def _generate_methodology(self) -> list[str]:
        """Generate methodology section."""
        lines = [
            "## Methodology",
            "",
            "### Rust Benchmarks (Criterion)",
            "",
            "- **Warmup**: Criterion automatically performs warmup iterations",
            "- **Statistical analysis**: Criterion uses statistical methods to determine",
            "  iteration count and detect outliers",
            "- **Measurement**: Each benchmark is run multiple times until confidence",
            "  intervals converge",
            "",
            "### Python Benchmarks",
            "",
            "- **Warmup**: 10 iterations before measurement",
            "- **Measurement**: Configurable iteration count (default varies by benchmark)",
            "- **Statistics**: Mean, std dev, percentiles (p50, p95, p99), 95% CI",
            "",
            "### Binary Benchmarks",
            "",
            "- **Binaries**: From angr-binaries test fixtures where available",
            "- **Measurement**: Multiple iterations with mean time reported",
            "- **Extra metrics**: CFG node/edge counts, state counts, etc.",
            "",
        ]

        if self.examples_results:
            lines.extend([
                "### angr-examples Benchmarks",
                "",
                "- **Source**: [angr-examples repository](https://github.com/angr/angr-examples)",
                "- **Isolation**: Each example runs in a separate subprocess",
                "- **Timeout**: 120 seconds per example (30 seconds in quick mode)",
                "- **Engine switching**: Python engine uses default UberEngine,",
                "  Rust engine patches simulation_manager to use UberEngineRust",
                "- **Metrics**: Success/failure status, execution time, speedup ratio",
                "",
            ])

        return lines

    def _generate_conclusions(self) -> list[str]:
        """Generate conclusions section."""
        lines = [
            "## Conclusions",
            "",
        ]

        comparisons = self._get_benchmark_comparisons()
        if comparisons:
            speedups = [(n, p/r if r > 0 else 0) for n, r, p in comparisons]
            avg_speedup = sum(s[1] for s in speedups) / len(speedups) if speedups else 0

            lines.extend([
                f"The Rust VEX engine demonstrates an average speedup of **{avg_speedup:.1f}x**",
                "compared to the Python engine across the measured benchmarks.",
                "",
            ])

            # Categorize findings
            fork_speedups = [s for n, s in speedups if "fork" in n.lower()]
            if fork_speedups:
                avg_fork = sum(fork_speedups) / len(fork_speedups)
                lines.extend([
                    "### Fork Performance",
                    "",
                    f"Fork operations show **{avg_fork:.1f}x** average improvement due to",
                    "Rust's Copy-on-Write memory implementation. This is particularly beneficial",
                    "for symbolic execution workloads with high path counts.",
                    "",
                ])

            mem_speedups = [s for n, s in speedups if "memory" in n.lower()]
            if mem_speedups:
                avg_mem = sum(mem_speedups) / len(mem_speedups)
                lines.extend([
                    "### Memory Operations",
                    "",
                    f"Memory operations show **{avg_mem:.1f}x** average improvement.",
                    "Direct memory access in Rust avoids Python object overhead.",
                    "",
                ])

            lines.extend([
                "### Recommendations",
                "",
                "1. **Use Rust engine for fork-heavy workloads**: Symbolic execution",
                "   with many paths benefits most from the CoW optimization",
                "",
                "2. **Batch operations where possible**: Minimize FFI crossing overhead",
                "   by batching register and memory operations",
                "",
                "3. **Profile Python integration**: Identify bottlenecks in state",
                "   synchronization between engines",
                "",
            ])
        else:
            lines.extend([
                "Insufficient data for conclusions. Run the full benchmark suite to",
                "generate comparison data.",
                "",
            ])

        return lines

    def _get_benchmark_comparisons(self) -> list[tuple[str, float, float]]:
        """Get matching Rust/Python benchmarks for comparison."""
        comparisons = []

        # Build lookup dictionaries from python_results (which includes rust wrapper results)
        rust_by_name: dict[str, dict] = {}
        python_by_name: dict[str, dict] = {}

        for r in self.python_results:
            name = r.get("name", "").lower()
            if name.startswith("rust_"):
                rust_by_name[name] = r
            elif name.startswith("python_"):
                python_by_name[name] = r

        # Also check Criterion results
        for r in self.rust_results:
            name = r.name.lower().replace("/", "_").replace("-", "_")
            rust_by_name[name] = {"name": r.name, "mean_time_ms": r.mean_ms}

        # Define matching pairs: (display_name, rust_key, python_key)
        match_pairs = [
            ("Fork", "rust_fork", "python_fork"),
            ("Fork + Modify", "rust_fork_modify", "python_fork_modify"),
            ("100 Forks", "rust_many_forks_100", "python_many_forks_100"),
            ("Memory Read 8B", "rust_memory_read_8bytes", "python_memory_read_8bytes"),
            ("Memory Write 8B", "rust_memory_write_8bytes", "python_memory_write_8bytes"),
            ("Memory Read 4KB", "rust_memory_read_4kb", "python_memory_read_4kb"),
            ("Memory Write 4KB", "rust_memory_write_4kb", "python_memory_write_4kb"),
            ("Register Read", "rust_register_read", "python_register_read"),
            ("Register Write", "rust_register_write", "python_register_write"),
        ]

        for display_name, rust_key, python_key in match_pairs:
            rust_r = rust_by_name.get(rust_key)
            python_r = python_by_name.get(python_key)

            if rust_r and python_r:
                rust_ms = rust_r.get("mean_time_ms", 0)
                python_ms = python_r.get("mean_time_ms", 0)
                if rust_ms > 0 and python_ms > 0:
                    comparisons.append((display_name, rust_ms, python_ms))

        return comparisons

    def save_report(self, output_path: str | Path):
        """Save the report to a file."""
        report = self.generate_report()
        with open(output_path, "w") as f:
            f.write(report)

    def save_json(self, output_path: str | Path):
        """Save raw data as JSON."""
        data = {
            "system_info": self.system_info,
            "rust_results": [r.to_dict() for r in self.rust_results],
            "python_results": self.python_results,
            "binary_results": self.binary_results,
            "examples_results": self.examples_results,
            "generated": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def main():
    """Generate report from existing benchmark data."""
    print("=" * 60)
    print("Benchmark Report Generator")
    print("=" * 60)

    generator = ReportGenerator()
    generator.collect_system_info()

    # Try to load Criterion results
    angr_dir = Path(__file__).parent.parent.parent
    native_dir = angr_dir / "native" / "angr"
    criterion_dir = native_dir / "target" / "criterion"

    if criterion_dir.exists():
        print(f"Loading Criterion results from {criterion_dir}")
        generator.load_criterion_results(criterion_dir)
        print(f"  Found {len(generator.rust_results)} Rust benchmarks")
    else:
        print(f"Criterion directory not found: {criterion_dir}")
        print("  Run 'cargo bench --features vex-engine' in native/angr to generate")

    # Run Python benchmarks to get fresh data
    print("\nRunning Python benchmarks...")
    try:
        from bench_rust_vex import RustEngineBenchmarks, PythonEngineBenchmarks

        rust_bench = RustEngineBenchmarks()
        python_bench = PythonEngineBenchmarks()

        rust_results = rust_bench.run_all()
        python_results = python_bench.run_all()

        generator.load_python_results(rust_results + python_results)
        print(f"  Collected {len(rust_results)} Rust wrapper benchmarks")
        print(f"  Collected {len(python_results)} Python benchmarks")
    except Exception as e:
        print(f"  Error running Python benchmarks: {e}")

    # Try to run binary benchmarks
    print("\nRunning binary benchmarks...")
    try:
        from bench_binaries import BinaryBenchmarks

        binary_bench = BinaryBenchmarks()
        binary_results = binary_bench.run_all()
        generator.load_binary_results(binary_results)
        print(f"  Collected {len(binary_results)} binary benchmarks")
    except Exception as e:
        print(f"  Error running binary benchmarks: {e}")

    # Generate report
    print("\nGenerating report...")
    report_path = angr_dir / "tests" / "benchmark_report.md"
    generator.save_report(report_path)
    print(f"  Saved report to {report_path}")

    # Also save JSON data
    json_path = angr_dir / "tests" / "benchmark_data.json"
    generator.save_json(json_path)
    print(f"  Saved JSON data to {json_path}")

    print("\nSystem Information:")
    for key, value in generator.system_info.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
