#!/usr/bin/env python3
"""
Exhaustive Python vs Rust Engine Comparison for angr-examples.

Tests all examples across 2 engine configurations:
1. python_only - Default UberEngine (baseline)
2. rust_engine - UberEngineRust

Usage:
    python tests/benchmarks/run_exhaustive_comparison.py \
        --examples-dir /home/vagrant/angr-examples/examples \
        --timeout 300 \
        --output tests/

Options:
    --examples-dir PATH     Path to angr-examples/examples directory
    --timeout SECONDS       Timeout per example per config (default: 300)
    --only NAME             Run only the specified example
    --skip NAME             Additional examples to skip
    --skip-pysoot           Skip PySoot examples
    --skip-android          Skip Android examples
    --skip-timeout          Skip known timeout examples
    --skip-deps             Skip examples with external dependencies
    --output PATH           Output directory for reports
    --iterations N          Number of iterations per example (default: 1)
    --verbose               Verbose output
"""
import argparse
import importlib.util
import io
import json
import math
import multiprocessing
import os
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Engine configurations to test
ENGINE_CONFIGS = [
    {
        "name": "python_only",
        "description": "Default UberEngine (Python VEX)",
        "engine": None,
        "loop": False,
        "rust_solver": False,
    },
    {
        "name": "rust_engine",
        "description": "UberEngineRust",
        "engine": "UberEngineRust",
        "loop": False,
        "rust_solver": False,
    },
]

# Skip lists by category
SKIP_PYSOOT = {
    "java_crackme1",
    "java_simple3",
    "java_simple4",
    "ictf2017_javaisnotfun",
}

SKIP_ANDROID = {
    "java_androidnative1",
    "defcon2019quals_veryandroidoso",
}

SKIP_TIMEOUT = {
    "0ctf_momo_3",
    "9447_nobranch",
    "hitcon2017_sakura",
    "tumctf2016_zwiebel",
    "b01lersctf2020_little_engine",
    "sharif7_rev50",
}

SKIP_DEPS = {
    "secconquals2016_ropsynth",  # Requires angrop
    "xmllint",  # Requires angr-targets
}

# Additional problematic examples
SKIP_MISC = {
    "defcon2016quals_baby-re",  # Interactive/missing binary
    "unmapped_analysis",  # Special analysis, not a solver
}


class BufferedStringIO(io.StringIO):
    """StringIO with a buffer attribute for compatibility with code that uses stdout.buffer."""

    def __init__(self):
        super().__init__()
        self._buffer = io.BytesIO()

    @property
    def buffer(self):
        return self._buffer

    def getvalue(self) -> str:
        """Get combined output from both text and binary writes."""
        text_output = super().getvalue()
        binary_output = self._buffer.getvalue()
        if binary_output:
            try:
                text_output += binary_output.decode('utf-8', errors='replace')
            except Exception:
                pass
        return text_output


@dataclass
class ConfigResult:
    """Result of running an example with a specific engine configuration."""
    example_name: str
    config_name: str
    success: bool
    execution_time_s: float
    peak_memory_mb: float = 0.0
    stdout_output: str = ""
    error_message: str | None = None
    error_traceback: str | None = None
    timeout: bool = False
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_name": self.example_name,
            "config_name": self.config_name,
            "success": self.success,
            "execution_time_s": self.execution_time_s,
            "peak_memory_mb": self.peak_memory_mb,
            "stdout_output": self.stdout_output[:1000] if self.stdout_output else "",  # Truncate for JSON
            "error_message": self.error_message,
            "timeout": self.timeout,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ExampleComparison:
    """Comparison results across all configurations for a single example."""
    example_name: str
    results: dict[str, ConfigResult] = field(default_factory=dict)

    @property
    def all_success(self) -> bool:
        """Check if all non-skipped configs succeeded."""
        non_skipped = [r for r in self.results.values() if not r.skipped]
        return len(non_skipped) > 0 and all(r.success for r in non_skipped)

    @property
    def python_time(self) -> float | None:
        """Get Python baseline time."""
        r = self.results.get("python_only")
        return r.execution_time_s if r and r.success else None

    def speedup(self, config_name: str) -> float | None:
        """Calculate speedup vs Python baseline."""
        py_time = self.python_time
        r = self.results.get(config_name)
        if py_time and r and r.success and r.execution_time_s > 0:
            return py_time / r.execution_time_s
        return None

    def outputs_match(self) -> tuple[bool, str]:
        """Check if all successful outputs match."""
        successful = [(k, r) for k, r in self.results.items() if r.success and r.stdout_output]
        if len(successful) < 2:
            return True, "Not enough outputs to compare"

        # Compare all outputs to the first one
        first_name, first_result = successful[0]
        first_output = _normalize_output(first_result.stdout_output)

        for name, result in successful[1:]:
            output = _normalize_output(result.stdout_output)
            if output != first_output:
                return False, f"{first_name} vs {name} differ"

        return True, "All match"

    def to_dict(self) -> dict[str, Any]:
        match, match_desc = self.outputs_match()
        return {
            "example_name": self.example_name,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "all_success": self.all_success,
            "outputs_match": match,
            "match_description": match_desc,
            "speedups": {
                cfg: self.speedup(cfg)
                for cfg in ["rust_engine"]
                if self.speedup(cfg) is not None
            },
        }


def _normalize_output(output: str) -> list[str]:
    """Normalize output for comparison by filtering timing lines."""
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    filtered = []
    for line in lines:
        lower = line.lower()
        # Skip timing information
        if any(kw in lower for kw in ['time elapsed', 'elapsed:', 'seconds', ' ms', 'timing', 'took ']):
            continue
        if lower == 'none':
            continue
        filtered.append(line)
    return filtered


def _run_example_with_config(args: tuple) -> dict:
    """
    Run an example with a specific engine configuration in a subprocess.

    Returns dict with execution metrics and output.
    """
    solve_script_path, example_name, config, timeout = args

    result = {
        "example_name": example_name,
        "config_name": config["name"],
        "success": False,
        "execution_time_s": 0.0,
        "peak_memory_mb": 0.0,
        "stdout_output": "",
        "error_message": None,
        "error_traceback": None,
        "timeout": False,
    }

    # Track memory before
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    try:
        # Change to the example directory
        example_dir = os.path.dirname(solve_script_path)
        original_dir = os.getcwd()
        os.chdir(example_dir)

        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        import angr
        from angr import sim_options as o

        # Check Rust engine availability
        rust_available = False
        UberEngineRust = None
        RustSimSolver = None
        RUST_SOLVER_AVAILABLE = False

        if config["engine"] == "UberEngineRust":
            try:
                from angr.engines import UberEngineRust, RUST_ENGINE_AVAILABLE
                rust_available = RUST_ENGINE_AVAILABLE
            except ImportError:
                pass

            if not rust_available:
                result["error_message"] = "Rust engine not available"
                return result

        if config["rust_solver"]:
            try:
                from angr.state_plugins.rust_solver import RustSimSolver, RUST_SOLVER_AVAILABLE
            except ImportError:
                result["error_message"] = "RustSimSolver not available"
                return result

            if not RUST_SOLVER_AVAILABLE:
                result["error_message"] = "RustSolverContext not available"
                return result

        # Patch the successors factory
        original_successors = None
        if config["engine"] == "UberEngineRust" and rust_available:
            original_successors = angr.factory.AngrObjectFactory.successors

            def patched_successors(self, *args, engine=None, **kwargs):
                if engine is None:
                    engine = UberEngineRust(self.project)
                return original_successors(self, *args, engine=engine, **kwargs)

            angr.factory.AngrObjectFactory.successors = patched_successors

        # Patch entry_state to add sim_options and RustSimSolver
        original_entry_state = None
        if config["loop"] or config["rust_solver"]:
            original_entry_state = angr.factory.AngrObjectFactory.entry_state

            def patched_entry_state(self, *args, add_options=None, **kwargs):
                if add_options is None:
                    add_options = set()
                else:
                    add_options = set(add_options)

                # Add RUST_VEX_LOOP if needed
                if config["loop"]:
                    add_options.add(o.RUST_VEX_LOOP)

                state = original_entry_state(self, *args, add_options=add_options, **kwargs)

                # Register RustSimSolver if needed
                if config["rust_solver"] and RustSimSolver is not None:
                    try:
                        state.register_plugin('solver', RustSimSolver())
                    except Exception as e:
                        # Log but don't fail - some examples might not need solver
                        pass

                return state

            angr.factory.AngrObjectFactory.entry_state = patched_entry_state

        try:
            # Load and run the solve script
            spec = importlib.util.spec_from_file_location("__main__", solve_script_path)
            module = importlib.util.module_from_spec(spec)

            # Capture stdout
            captured = BufferedStringIO()
            original_stdout = sys.stdout
            sys.stdout = captured

            # Save and clear sys.argv to avoid scripts misinterpreting test harness args
            original_argv = sys.argv
            sys.argv = [solve_script_path]

            start_time = time.perf_counter()

            try:
                spec.loader.exec_module(module)
            finally:
                sys.stdout = original_stdout
                sys.argv = original_argv

            end_time = time.perf_counter()

            result["success"] = True
            result["execution_time_s"] = end_time - start_time
            result["stdout_output"] = captured.getvalue()

        finally:
            # Restore patches
            if original_successors is not None:
                angr.factory.AngrObjectFactory.successors = original_successors
            if original_entry_state is not None:
                angr.factory.AngrObjectFactory.entry_state = original_entry_state

            os.chdir(original_dir)

    except Exception as e:
        result["error_message"] = str(e)
        result["error_traceback"] = traceback.format_exc()

    # Track peak memory
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_memory_mb"] = mem_after / 1024  # Convert KB to MB on Linux

    return result


class ExhaustiveComparisonRunner:
    """Runner for exhaustive engine comparison across all examples."""

    def __init__(
        self,
        examples_dir: Path,
        skip_pysoot: bool = True,
        skip_android: bool = True,
        skip_timeout: bool = True,
        skip_deps: bool = True,
        additional_skip: set[str] | None = None,
        verbose: bool = False,
    ):
        self.examples_dir = Path(examples_dir)
        self.verbose = verbose

        # Build combined skip set
        self.skip_set = set(SKIP_MISC)
        if skip_pysoot:
            self.skip_set |= SKIP_PYSOOT
        if skip_android:
            self.skip_set |= SKIP_ANDROID
        if skip_timeout:
            self.skip_set |= SKIP_TIMEOUT
        if skip_deps:
            self.skip_set |= SKIP_DEPS
        if additional_skip:
            self.skip_set |= additional_skip

        if not self.examples_dir.exists():
            raise ValueError(f"Examples directory not found: {self.examples_dir}")

    def discover_examples(self) -> list[tuple[str, Path]]:
        """Discover all examples with solve.py scripts."""
        examples = []

        for item in sorted(self.examples_dir.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue

            # Look for solve.py
            solve_script = item / "solve.py"
            if solve_script.exists():
                examples.append((item.name, solve_script))
                continue

            # Check subdirectories (like defcon2017quals_crackme2000)
            for subdir in sorted(item.iterdir()):
                if not subdir.is_dir() or subdir.name.startswith("."):
                    continue

                sub_solve = subdir / "solve.py"
                if sub_solve.exists():
                    name = f"{item.name}/{subdir.name}"
                    examples.append((name, sub_solve))
                    continue

                # Check deeper nesting (like CSCI-4968-MBE/challenges/crackme0x*)
                for subsubdir in sorted(subdir.iterdir()):
                    if not subsubdir.is_dir() or subsubdir.name.startswith("."):
                        continue
                    subsub_solve = subsubdir / "solve.py"
                    if subsub_solve.exists():
                        name = f"{item.name}/{subdir.name}/{subsubdir.name}"
                        examples.append((name, subsub_solve))

        return examples

    def run_example_config(
        self,
        example_name: str,
        solve_script: Path,
        config: dict,
        timeout: int,
    ) -> ConfigResult:
        """Run a single example with a specific configuration."""
        if example_name in self.skip_set:
            return ConfigResult(
                example_name=example_name,
                config_name=config["name"],
                success=False,
                execution_time_s=0.0,
                skipped=True,
                skip_reason="In skip list",
            )

        args = (str(solve_script), example_name, config, timeout)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            try:
                async_result = pool.apply_async(_run_example_with_config, (args,))
                result_dict = async_result.get(timeout=timeout)

                return ConfigResult(
                    example_name=result_dict["example_name"],
                    config_name=result_dict["config_name"],
                    success=result_dict["success"],
                    execution_time_s=result_dict["execution_time_s"],
                    peak_memory_mb=result_dict["peak_memory_mb"],
                    stdout_output=result_dict.get("stdout_output", ""),
                    error_message=result_dict["error_message"],
                    error_traceback=result_dict.get("error_traceback"),
                    timeout=result_dict["timeout"],
                )

            except multiprocessing.TimeoutError:
                pool.terminate()
                return ConfigResult(
                    example_name=example_name,
                    config_name=config["name"],
                    success=False,
                    execution_time_s=timeout,
                    error_message=f"Timeout after {timeout}s",
                    timeout=True,
                )
            except Exception as e:
                return ConfigResult(
                    example_name=example_name,
                    config_name=config["name"],
                    success=False,
                    execution_time_s=0.0,
                    error_message=str(e),
                    error_traceback=traceback.format_exc(),
                )

    def run_comparison(
        self,
        timeout: int = 300,
        only: str | None = None,
        configs: list[dict] | None = None,
    ) -> list[ExampleComparison]:
        """Run all examples with all configurations."""
        configs = configs or ENGINE_CONFIGS
        examples = self.discover_examples()

        if only:
            examples = [(n, p) for n, p in examples if n == only]
            if not examples:
                raise ValueError(f"Example not found: {only}")

        comparisons = []
        total = len(examples)
        num_configs = len(configs)

        for idx, (example_name, solve_script) in enumerate(examples, 1):
            print(f"\n[{idx}/{total}] {example_name}")

            if example_name in self.skip_set:
                print(f"  SKIPPED (in skip list)")
                comparison = ExampleComparison(example_name=example_name)
                for config in configs:
                    comparison.results[config["name"]] = ConfigResult(
                        example_name=example_name,
                        config_name=config["name"],
                        success=False,
                        execution_time_s=0.0,
                        skipped=True,
                        skip_reason="In skip list",
                    )
                comparisons.append(comparison)
                continue

            comparison = ExampleComparison(example_name=example_name)

            for cfg_idx, config in enumerate(configs, 1):
                cfg_name = config["name"]
                print(f"  [{cfg_idx}/{num_configs}] {cfg_name}...", end=" ", flush=True)

                result = self.run_example_config(
                    example_name, solve_script, config, timeout
                )
                comparison.results[cfg_name] = result

                if result.success:
                    print(f"OK ({result.execution_time_s:.1f}s, {result.peak_memory_mb:.0f}MB)")
                elif result.timeout:
                    print(f"TIMEOUT ({timeout}s)")
                elif result.skipped:
                    print(f"SKIPPED")
                else:
                    error_short = result.error_message[:50] if result.error_message else "Unknown"
                    print(f"FAIL: {error_short}")

            # Show summary for this example
            py_result = comparison.results.get("python_only")
            if py_result and py_result.success:
                sp = comparison.speedup("rust_engine")
                if sp is not None:
                    print(f"  Speedup: rust_engine: {sp:.2f}x")

            match, match_desc = comparison.outputs_match()
            if not match:
                print(f"  WARNING: Output mismatch - {match_desc}")

            comparisons.append(comparison)

        return comparisons


def generate_report(
    comparisons: list[ExampleComparison],
    output_dir: Path,
    configs: list[dict] | None = None,
) -> None:
    """Generate markdown and JSON reports."""
    configs = configs or ENGINE_CONFIGS
    config_names = [c["name"] for c in configs]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate summary statistics
    total = len(comparisons)
    skipped = sum(1 for c in comparisons if all(r.skipped for r in c.results.values()))
    tested = total - skipped

    # Per-config statistics
    config_stats = {}
    for cfg_name in config_names:
        results = [c.results.get(cfg_name) for c in comparisons if c.results.get(cfg_name)]
        successes = sum(1 for r in results if r and r.success)
        timeouts = sum(1 for r in results if r and r.timeout)
        failures = sum(1 for r in results if r and not r.success and not r.timeout and not r.skipped)
        times = [r.execution_time_s for r in results if r and r.success]
        memories = [r.peak_memory_mb for r in results if r and r.success]

        config_stats[cfg_name] = {
            "success": successes,
            "timeout": timeouts,
            "failure": failures,
            "avg_time": sum(times) / len(times) if times else 0,
            "avg_memory": sum(memories) / len(memories) if memories else 0,
            "total_time": sum(times),
        }

    # Calculate speedup statistics (where both Python and Rust succeeded)
    speedup_stats = {}
    speedups = []
    for c in comparisons:
        sp = c.speedup("rust_engine")
        if sp is not None:
            speedups.append(sp)

    if speedups:
        speedup_stats["rust_engine"] = {
            "count": len(speedups),
            "geo_mean": math.exp(sum(math.log(s) for s in speedups) / len(speedups)),
            "min": min(speedups),
            "max": max(speedups),
            "median": sorted(speedups)[len(speedups) // 2],
        }

    # Count output matches
    successful_pairs = [c for c in comparisons if c.all_success]
    matches = sum(1 for c in successful_pairs if c.outputs_match()[0])

    # Generate markdown report
    md_lines = [
        "# Exhaustive Python vs Rust Engine Comparison",
        "",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        "",
        f"- **Total examples**: {total}",
        f"- **Skipped**: {skipped}",
        f"- **Tested**: {tested}",
        "",
        "### Success Rate by Configuration",
        "",
        "| Config | Success | Timeout | Failed | Avg Time | Avg Memory | Total Time |",
        "|--------|---------|---------|--------|----------|------------|------------|",
    ]

    for cfg_name in config_names:
        stats = config_stats[cfg_name]
        md_lines.append(
            f"| {cfg_name} | {stats['success']}/{tested} | {stats['timeout']} | "
            f"{stats['failure']} | {stats['avg_time']:.1f}s | {stats['avg_memory']:.0f}MB | "
            f"{stats['total_time']:.0f}s |"
        )

    md_lines.extend([
        "",
        "### Speedup Analysis (vs Python baseline)",
        "",
        "| Config | Count | Geo Mean | Min | Max | Median |",
        "|--------|-------|----------|-----|-----|--------|",
    ])

    if "rust_engine" in speedup_stats:
        s = speedup_stats["rust_engine"]
        md_lines.append(
            f"| rust_engine | {s['count']} | {s['geo_mean']:.2f}x | "
            f"{s['min']:.2f}x | {s['max']:.2f}x | {s['median']:.2f}x |"
        )
    else:
        md_lines.append("| rust_engine | 0 | - | - | - | - |")

    md_lines.extend([
        "",
        f"### Output Verification",
        "",
        f"- **All configs succeed**: {len(successful_pairs)}/{tested}",
        f"- **Outputs match**: {matches}/{len(successful_pairs)}",
        "",
        "---",
        "",
        "## Per-Example Results",
        "",
    ])

    # Create per-example table
    header = "| Example | " + " | ".join(config_names) + " | Speedup | Match |"
    separator = "|---------|" + "|".join(["--------"] * len(config_names)) + "|---------|-------|"
    md_lines.extend([header, separator])

    for c in sorted(comparisons, key=lambda x: x.example_name):
        # Truncate long example names but keep enough to be identifiable
        name = c.example_name
        if len(name) > 40:
            name = name[:37] + "..."
        row = [name]

        for cfg_name in config_names:
            r = c.results.get(cfg_name)
            if r is None:
                row.append("-")
            elif r.skipped:
                row.append("SKIP")
            elif r.timeout:
                row.append("T/O")
            elif r.success:
                row.append(f"{r.execution_time_s:.1f}s")
            else:
                row.append("FAIL")

        # Speedup
        sp = c.speedup("rust_engine")
        if sp is not None:
            row.append(f"{sp:.2f}x")
        else:
            row.append("-")

        # Match
        match, _ = c.outputs_match()
        if c.all_success:
            row.append("\u2713" if match else "\u2717")
        else:
            row.append("-")

        md_lines.append("| " + " | ".join(row) + " |")

    # Add failure details section
    failures = [
        (c.example_name, cfg_name, r)
        for c in comparisons
        for cfg_name, r in c.results.items()
        if not r.success and not r.skipped and not r.timeout
    ]

    if failures:
        md_lines.extend([
            "",
            "---",
            "",
            "## Failure Details",
            "",
        ])
        for example_name, cfg_name, r in failures[:20]:  # Limit to first 20
            md_lines.extend([
                f"### {example_name} ({cfg_name})",
                "",
                f"**Error**: {r.error_message}",
                "",
            ])

    # Add output mismatches section
    mismatches = [c for c in comparisons if c.all_success and not c.outputs_match()[0]]
    if mismatches:
        md_lines.extend([
            "",
            "---",
            "",
            "## Output Mismatches",
            "",
        ])
        for c in mismatches[:10]:  # Limit to first 10
            md_lines.extend([
                f"### {c.example_name}",
                "",
            ])
            for cfg_name in config_names:
                r = c.results.get(cfg_name)
                if r and r.success and r.stdout_output:
                    output_preview = r.stdout_output[-200:].replace('\n', ' ')[:150]
                    md_lines.append(f"**{cfg_name}**: `{output_preview}`")
            md_lines.append("")

    md_lines.extend([
        "",
        "---",
        "",
        "## Legend",
        "",
        "- **Speedup**: Python time / Rust time (>1.0 means Rust is faster)",
        "- **Geo Mean**: Geometric mean of speedups (better for ratios)",
        "- **T/O**: Timeout",
        "- **Match**: \u2713 = outputs match across configs, \u2717 = outputs differ",
        "",
    ])

    # Write markdown report
    md_path = output_dir / "exhaustive_comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown report saved to: {md_path}")

    # Write JSON data
    json_data = {
        "generated": datetime.now().isoformat(),
        "summary": {
            "total_examples": total,
            "skipped": skipped,
            "tested": tested,
            "config_stats": config_stats,
            "speedup_stats": speedup_stats,
            "output_matches": matches,
            "all_success": len(successful_pairs),
        },
        "comparisons": [c.to_dict() for c in comparisons],
    }
    json_path = output_dir / "exhaustive_comparison_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data saved to: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run exhaustive Python vs Rust engine comparison"
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("/home/vagrant/angr-examples/examples"),
        help="Path to angr-examples/examples directory",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per example per config in seconds (default: 300)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the specified example",
    )
    parser.add_argument(
        "--skip",
        type=str,
        action="append",
        default=[],
        help="Additional examples to skip (can be used multiple times)",
    )
    parser.add_argument(
        "--skip-pysoot",
        action="store_true",
        default=True,
        help="Skip PySoot examples (default: True)",
    )
    parser.add_argument(
        "--no-skip-pysoot",
        action="store_false",
        dest="skip_pysoot",
        help="Don't skip PySoot examples",
    )
    parser.add_argument(
        "--skip-android",
        action="store_true",
        default=True,
        help="Skip Android examples (default: True)",
    )
    parser.add_argument(
        "--no-skip-android",
        action="store_false",
        dest="skip_android",
        help="Don't skip Android examples",
    )
    parser.add_argument(
        "--skip-timeout",
        action="store_true",
        default=True,
        help="Skip known timeout examples (default: True)",
    )
    parser.add_argument(
        "--no-skip-timeout",
        action="store_false",
        dest="skip_timeout",
        help="Don't skip known timeout examples",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        default=True,
        help="Skip examples with external dependencies (default: True)",
    )
    parser.add_argument(
        "--no-skip-deps",
        action="store_false",
        dest="skip_deps",
        help="Don't skip examples with external dependencies",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for reports (default: tests/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples and exit",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["all", "python", "rust"],
        default="all",
        help="Which configuration(s) to test (default: all)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Exhaustive Python vs Rust Engine Comparison")
    print("=" * 70)
    print(f"Examples directory: {args.examples_dir}")
    print(f"Timeout: {args.timeout}s per example per config")
    print()

    # Select configurations
    if args.config == "all":
        configs = ENGINE_CONFIGS
    elif args.config == "python":
        configs = [c for c in ENGINE_CONFIGS if c["name"] == "python_only"]
    elif args.config == "rust":
        configs = [c for c in ENGINE_CONFIGS if c["name"] == "rust_engine"]
    else:
        configs = ENGINE_CONFIGS

    print("Configurations to test:")
    for cfg in configs:
        print(f"  - {cfg['name']}: {cfg['description']}")
    print()

    # Create runner
    try:
        runner = ExhaustiveComparisonRunner(
            args.examples_dir,
            skip_pysoot=args.skip_pysoot,
            skip_android=args.skip_android,
            skip_timeout=args.skip_timeout,
            skip_deps=args.skip_deps,
            additional_skip=set(args.skip) if args.skip else None,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # List examples if requested
    if args.list:
        examples = runner.discover_examples()
        print(f"Found {len(examples)} examples:\n")
        for name, path in examples:
            skip_marker = " [SKIP]" if name in runner.skip_set else ""
            print(f"  {name}{skip_marker}")
        print(f"\nSkip set: {len(runner.skip_set)} examples")
        return 0

    print(f"Skip categories enabled:")
    print(f"  - PySoot: {args.skip_pysoot} ({len(SKIP_PYSOOT)} examples)")
    print(f"  - Android: {args.skip_android} ({len(SKIP_ANDROID)} examples)")
    print(f"  - Known timeouts: {args.skip_timeout} ({len(SKIP_TIMEOUT)} examples)")
    print(f"  - External deps: {args.skip_deps} ({len(SKIP_DEPS)} examples)")
    print()

    # Set up output directory
    output_dir = args.output or Path(__file__).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    start_time = time.perf_counter()
    comparisons = runner.run_comparison(
        timeout=args.timeout,
        only=args.only,
        configs=configs,
    )
    total_time = time.perf_counter() - start_time

    # Generate reports
    generate_report(comparisons, output_dir, configs)

    # Print summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    total = len(comparisons)
    skipped = sum(1 for c in comparisons if all(r.skipped for r in c.results.values()))
    tested = total - skipped

    for cfg in configs:
        cfg_name = cfg["name"]
        results = [c.results.get(cfg_name) for c in comparisons if c.results.get(cfg_name)]
        successes = sum(1 for r in results if r and r.success)
        print(f"{cfg_name}: {successes}/{tested} succeeded")

    print()
    print(f"Total comparison time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
