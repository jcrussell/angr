#!/usr/bin/env python3
"""
Benchmark runner for angr-examples repository.

This module runs all examples from the angr-examples repository with both
the Python and Rust VEX engines, generating a comparison table.

Run standalone:
    python tests/benchmarks/bench_angr_examples.py --examples-dir /path/to/angr-examples/examples

Options:
    --examples-dir PATH     Path to angr-examples/examples directory
    --timeout SECONDS       Timeout per example (default: 120)
    --only NAME             Run only the specified example
    --skip NAME             Skip the specified example (can be used multiple times)
    --output PATH           Output directory for reports
    --verbose               Verbose output
"""
import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class ExampleInfo:
    """Information about an angr-examples example."""
    name: str
    path: Path
    solve_script: Path
    has_binary: bool = True
    description: str = ""

    def __repr__(self) -> str:
        return f"ExampleInfo({self.name}, {self.path})"


@dataclass
class ExampleResult:
    """Result of running an example with a specific engine."""
    example_name: str
    engine: str  # 'python' or 'rust'
    success: bool
    execution_time_s: float
    error_message: str | None = None
    error_traceback: str | None = None
    timeout: bool = False
    unsupported: bool = False
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_name": self.example_name,
            "engine": self.engine,
            "success": self.success,
            "execution_time_s": self.execution_time_s,
            "error_message": self.error_message,
            "timeout": self.timeout,
            "unsupported": self.unsupported,
            "extra_info": self.extra_info,
        }


@dataclass
class ComparisonRow:
    """A row in the comparison table."""
    example_name: str
    python_time_s: float | None
    python_success: bool
    python_error: str | None
    rust_time_s: float | None
    rust_success: bool
    rust_error: str | None

    @property
    def speedup(self) -> float | None:
        """Calculate speedup (Python time / Rust time)."""
        if self.python_time_s and self.rust_time_s and self.rust_time_s > 0:
            return self.python_time_s / self.rust_time_s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_name": self.example_name,
            "python_time_s": self.python_time_s,
            "python_success": self.python_success,
            "python_error": self.python_error,
            "rust_time_s": self.rust_time_s,
            "rust_success": self.rust_success,
            "rust_error": self.rust_error,
            "speedup": self.speedup,
        }


def _run_example_subprocess(args: tuple) -> dict:
    """
    Run an example in a subprocess.

    This function is called via multiprocessing.Pool and runs the example
    in isolation to avoid state pollution between examples.
    """
    solve_script_path, example_name, engine, timeout = args

    result = {
        "example_name": example_name,
        "engine": engine,
        "success": False,
        "execution_time_s": 0.0,
        "error_message": None,
        "error_traceback": None,
        "timeout": False,
        "unsupported": False,
    }

    try:
        # Change to the example directory so relative paths work
        example_dir = os.path.dirname(solve_script_path)
        original_dir = os.getcwd()
        os.chdir(example_dir)

        # Add example directory to path
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        # Import angr and apply engine patch if using Rust
        import angr

        # Check if Rust engine is available
        rust_available = False
        UberEngineRust = None
        try:
            from angr.engines import UberEngineRust, RUST_ENGINE_AVAILABLE
            rust_available = RUST_ENGINE_AVAILABLE
        except ImportError:
            pass

        if engine == "rust" and not rust_available:
            result["unsupported"] = True
            result["error_message"] = "Rust engine not available"
            return result

        # Patch the successors factory if using Rust engine
        # The engine parameter is on factory.successors(), not simulation_manager()
        original_successors = None
        if engine == "rust" and rust_available:
            original_successors = angr.factory.AngrObjectFactory.successors

            def patched_successors(self, *args, engine=None, **kwargs):
                # Only inject engine if not already specified
                if engine is None:
                    engine = UberEngineRust(self.project)
                return original_successors(self, *args, engine=engine, **kwargs)

            angr.factory.AngrObjectFactory.successors = patched_successors

        try:
            # Load and run the solve script as __main__ so if __name__ == '__main__' works
            spec = importlib.util.spec_from_file_location("__main__", solve_script_path)
            module = importlib.util.module_from_spec(spec)

            start_time = time.perf_counter()
            spec.loader.exec_module(module)
            end_time = time.perf_counter()

            result["success"] = True
            result["execution_time_s"] = end_time - start_time

        finally:
            # Restore original successors
            if original_successors is not None:
                angr.factory.AngrObjectFactory.successors = original_successors

            # Restore working directory
            os.chdir(original_dir)

    except Exception as e:
        result["error_message"] = str(e)
        result["error_traceback"] = traceback.format_exc()

    return result


class AngrExampleRunner:
    """Runner for angr-examples benchmarks."""

    # Known examples that should be skipped (won't work in automated testing)
    SKIP_EXAMPLES = {
        # Interactive examples
        "baby-re",  # Requires interactive input
        # Missing dependencies or binaries
        "defcon2016quals_baby-re",  # Same as baby-re
    }

    def __init__(self, examples_dir: str | Path, verbose: bool = False):
        """
        Initialize the runner.

        Args:
            examples_dir: Path to the angr-examples/examples directory
            verbose: Enable verbose output
        """
        self.examples_dir = Path(examples_dir)
        self.verbose = verbose

        if not self.examples_dir.exists():
            raise ValueError(f"Examples directory not found: {self.examples_dir}")

    def discover_examples(self) -> list[ExampleInfo]:
        """
        Discover all examples in the examples directory.

        Returns a list of ExampleInfo objects for each discovered example.
        """
        examples = []

        for item in sorted(self.examples_dir.iterdir()):
            if not item.is_dir():
                continue

            # Skip hidden directories
            if item.name.startswith("."):
                continue

            # Look for solve.py
            solve_script = item / "solve.py"
            if not solve_script.exists():
                # Try solve.py in subdirectories
                for subdir in item.iterdir():
                    if subdir.is_dir():
                        sub_solve = subdir / "solve.py"
                        if sub_solve.exists():
                            examples.append(ExampleInfo(
                                name=f"{item.name}/{subdir.name}",
                                path=subdir,
                                solve_script=sub_solve,
                            ))
                continue

            examples.append(ExampleInfo(
                name=item.name,
                path=item,
                solve_script=solve_script,
            ))

        return examples

    def run_example(
        self,
        example: ExampleInfo,
        engine: str,
        timeout: int = 120,
        iterations: int = 1,
    ) -> ExampleResult:
        """
        Run a single example with the specified engine.

        Args:
            example: The example to run
            engine: 'python' or 'rust'
            timeout: Timeout in seconds

        Returns:
            ExampleResult with the execution results
        """
        if example.name in self.SKIP_EXAMPLES:
            return ExampleResult(
                example_name=example.name,
                engine=engine,
                success=False,
                execution_time_s=0.0,
                error_message="Skipped (known incompatible)",
                unsupported=True,
            )

        # Run multiple iterations and average the times
        times = []
        last_result_dict = None

        for i in range(iterations):
            # Run in subprocess with timeout
            args = (str(example.solve_script), example.name, engine, timeout)

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(1) as pool:
                try:
                    async_result = pool.apply_async(_run_example_subprocess, (args,))
                    result_dict = async_result.get(timeout=timeout)
                    last_result_dict = result_dict

                    if result_dict["success"]:
                        times.append(result_dict["execution_time_s"])
                    else:
                        # If any iteration fails, return the failure
                        return ExampleResult(
                            example_name=result_dict["example_name"],
                            engine=result_dict["engine"],
                            success=result_dict["success"],
                            execution_time_s=result_dict["execution_time_s"],
                            error_message=result_dict["error_message"],
                            error_traceback=result_dict.get("error_traceback"),
                            timeout=result_dict["timeout"],
                            unsupported=result_dict["unsupported"],
                        )

                except multiprocessing.TimeoutError:
                    pool.terminate()
                    return ExampleResult(
                        example_name=example.name,
                        engine=engine,
                        success=False,
                        execution_time_s=timeout,
                        error_message=f"Timeout after {timeout}s",
                        timeout=True,
                    )
                except Exception as e:
                    return ExampleResult(
                        example_name=example.name,
                        engine=engine,
                        success=False,
                        execution_time_s=0.0,
                        error_message=str(e),
                        error_traceback=traceback.format_exc(),
                    )

        # All iterations succeeded - return average time
        avg_time = sum(times) / len(times) if times else 0.0
        return ExampleResult(
            example_name=last_result_dict["example_name"],
            engine=last_result_dict["engine"],
            success=True,
            execution_time_s=avg_time,
            error_message=None,
            error_traceback=None,
            timeout=False,
            unsupported=False,
            extra_info={"iterations": iterations, "times": times},
        )

    def run_all_examples(
        self,
        timeout_per_example: int = 120,
        only: str | None = None,
        skip: set[str] | None = None,
        engines: list[str] | None = None,
        progress_callback: Callable[[str, str, str], None] | None = None,
        iterations: int = 1,
    ) -> tuple[list[ExampleResult], list[ComparisonRow]]:
        """
        Run all examples with both engines.

        Args:
            timeout_per_example: Timeout for each example in seconds
            only: If specified, only run this example
            skip: Set of example names to skip
            engines: List of engines to test (default: ['python', 'rust'])
            progress_callback: Called with (example_name, engine, status) for progress
            iterations: Number of iterations per example for averaging (default: 1)

        Returns:
            Tuple of (all results, comparison rows)
        """
        if engines is None:
            engines = ["python", "rust"]

        skip = skip or set()
        skip = skip | self.SKIP_EXAMPLES

        examples = self.discover_examples()

        if only:
            examples = [e for e in examples if e.name == only]
            if not examples:
                raise ValueError(f"Example not found: {only}")

        results = []
        comparisons = []

        for example in examples:
            if example.name in skip:
                if self.verbose:
                    print(f"Skipping {example.name}")
                continue

            python_result = None
            rust_result = None

            for engine in engines:
                if progress_callback:
                    progress_callback(example.name, engine, "running")

                result = self.run_example(example, engine, timeout_per_example, iterations)
                results.append(result)

                if engine == "python":
                    python_result = result
                elif engine == "rust":
                    rust_result = result

                status = "success" if result.success else "failed"
                if result.timeout:
                    status = "timeout"
                elif result.unsupported:
                    status = "unsupported"

                if progress_callback:
                    progress_callback(example.name, engine, status)

            # Create comparison row
            if python_result and rust_result:
                comparisons.append(ComparisonRow(
                    example_name=example.name,
                    python_time_s=python_result.execution_time_s if python_result.success else None,
                    python_success=python_result.success,
                    python_error=python_result.error_message,
                    rust_time_s=rust_result.execution_time_s if rust_result.success else None,
                    rust_success=rust_result.success,
                    rust_error=rust_result.error_message,
                ))

        return results, comparisons


def generate_comparison_table(comparisons: list[ComparisonRow]) -> str:
    """Generate a markdown comparison table."""
    lines = [
        "| Example | Python Time | Python Status | Rust Time | Rust Status | Speedup |",
        "|---------|-------------|---------------|-----------|-------------|---------|",
    ]

    for row in sorted(comparisons, key=lambda r: r.example_name):
        python_time = f"{row.python_time_s:.2f}s" if row.python_time_s else "N/A"
        python_status = "\u2713" if row.python_success else f"\u2717 ({row.python_error[:20]}...)" if row.python_error and len(row.python_error) > 20 else f"\u2717 ({row.python_error})" if row.python_error else "\u2717"
        rust_time = f"{row.rust_time_s:.2f}s" if row.rust_time_s else "N/A"
        rust_status = "\u2713" if row.rust_success else f"\u2717 ({row.rust_error[:20]}...)" if row.rust_error and len(row.rust_error) > 20 else f"\u2717 ({row.rust_error})" if row.rust_error else "\u2717"
        speedup = f"{row.speedup:.1f}x" if row.speedup else "-"

        lines.append(f"| {row.example_name} | {python_time} | {python_status} | {rust_time} | {rust_status} | {speedup} |")

    return "\n".join(lines)


def generate_summary(comparisons: list[ComparisonRow]) -> dict[str, Any]:
    """Generate summary statistics."""
    total = len(comparisons)
    python_success = sum(1 for r in comparisons if r.python_success)
    rust_success = sum(1 for r in comparisons if r.rust_success)
    both_success = sum(1 for r in comparisons if r.python_success and r.rust_success)

    speedups = [r.speedup for r in comparisons if r.speedup is not None]

    return {
        "total_examples": total,
        "python_success": python_success,
        "python_success_rate": python_success / total if total > 0 else 0,
        "rust_success": rust_success,
        "rust_success_rate": rust_success / total if total > 0 else 0,
        "both_success": both_success,
        "both_success_rate": both_success / total if total > 0 else 0,
        "speedups": {
            "count": len(speedups),
            "avg": sum(speedups) / len(speedups) if speedups else 0,
            "min": min(speedups) if speedups else 0,
            "max": max(speedups) if speedups else 0,
        },
    }


def generate_report(
    comparisons: list[ComparisonRow],
    results: list[ExampleResult],
    output_dir: Path,
) -> None:
    """Generate markdown and JSON reports."""
    summary = generate_summary(comparisons)

    # Generate markdown report
    md_lines = [
        "# angr-examples: Rust vs Python Engine Comparison",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total examples**: {summary['total_examples']}",
        f"- **Python success rate**: {summary['python_success']}/{summary['total_examples']} ({summary['python_success_rate']*100:.1f}%)",
        f"- **Rust success rate**: {summary['rust_success']}/{summary['total_examples']} ({summary['rust_success_rate']*100:.1f}%)",
        f"- **Both engines succeed**: {summary['both_success']}/{summary['total_examples']} ({summary['both_success_rate']*100:.1f}%)",
        "",
    ]

    if summary["speedups"]["count"] > 0:
        md_lines.extend([
            "### Speedup Statistics (where both engines succeed)",
            "",
            f"- **Average speedup**: {summary['speedups']['avg']:.2f}x",
            f"- **Minimum speedup**: {summary['speedups']['min']:.2f}x",
            f"- **Maximum speedup**: {summary['speedups']['max']:.2f}x",
            "",
        ])

    md_lines.extend([
        "## Comparison Table",
        "",
        generate_comparison_table(comparisons),
        "",
        "## Legend",
        "",
        "- \u2713 = Success",
        "- \u2717 = Failure (with error description)",
        "- N/A = Not available (failed or unsupported)",
        "- Speedup = Python time / Rust time (higher is better for Rust)",
        "",
    ])

    # Write markdown
    md_path = output_dir / "angr_examples_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report saved to: {md_path}")

    # Write JSON data
    json_data = {
        "generated": datetime.now().isoformat(),
        "summary": summary,
        "comparisons": [r.to_dict() for r in comparisons],
        "results": [r.to_dict() for r in results],
    }
    json_path = output_dir / "angr_examples_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data saved to: {json_path}")


def find_examples_dir() -> Path | None:
    """Try to find the angr-examples directory."""
    # Try common locations relative to angr repo
    angr_dir = Path(__file__).parent.parent.parent
    candidates = [
        angr_dir.parent / "angr-examples" / "examples",
        angr_dir.parent.parent / "angr-examples" / "examples",
        Path.home() / "angr-examples" / "examples",
        Path("/tmp/angr-examples/examples"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run angr-examples benchmarks with Rust and Python engines"
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Path to angr-examples/examples directory",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per example in seconds (default: 120)",
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
        help="Skip the specified example (can be used multiple times)",
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
        "--engine",
        type=str,
        choices=["python", "rust", "both"],
        default="both",
        help="Which engine(s) to test (default: both)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per example for averaging (default: 1)",
    )

    args = parser.parse_args()

    # Find examples directory
    examples_dir = args.examples_dir
    if examples_dir is None:
        examples_dir = find_examples_dir()
        if examples_dir is None:
            print("ERROR: Could not find angr-examples directory.")
            print("Please specify --examples-dir /path/to/angr-examples/examples")
            print("\nTo clone angr-examples:")
            print("  git clone https://github.com/angr/angr-examples.git")
            return 1

    print("=" * 60)
    print("angr-examples Benchmark Runner")
    print("=" * 60)
    print(f"Examples directory: {examples_dir}")

    # Create runner
    try:
        runner = AngrExampleRunner(examples_dir, verbose=args.verbose)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # List examples if requested
    if args.list:
        examples = runner.discover_examples()
        print(f"\nFound {len(examples)} examples:\n")
        for example in examples:
            skip_marker = " [SKIP]" if example.name in runner.SKIP_EXAMPLES else ""
            print(f"  {example.name}{skip_marker}")
        return 0

    # Determine engines to test
    if args.engine == "both":
        engines = ["python", "rust"]
    else:
        engines = [args.engine]

    # Set up output directory
    output_dir = args.output or Path(__file__).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Progress callback
    def progress(name: str, engine: str, status: str):
        time_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{time_str}] {name} ({engine}): {status}")

    # Run benchmarks
    print(f"\nRunning benchmarks with timeout={args.timeout}s per example...")
    print(f"Engines: {', '.join(engines)}")
    if args.iterations > 1:
        print(f"Iterations per example: {args.iterations}")
    print()

    start_time = time.perf_counter()

    results, comparisons = runner.run_all_examples(
        timeout_per_example=args.timeout,
        only=args.only,
        skip=set(args.skip),
        engines=engines,
        progress_callback=progress,
        iterations=args.iterations,
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Generate reports
    print()
    generate_report(comparisons, results, output_dir)

    # Print summary
    summary = generate_summary(comparisons)
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total examples: {summary['total_examples']}")
    print(f"Python success: {summary['python_success']} ({summary['python_success_rate']*100:.1f}%)")
    print(f"Rust success: {summary['rust_success']} ({summary['rust_success_rate']*100:.1f}%)")
    print(f"Both succeed: {summary['both_success']} ({summary['both_success_rate']*100:.1f}%)")

    if summary["speedups"]["count"] > 0:
        print(f"Average speedup: {summary['speedups']['avg']:.2f}x")

    print(f"Total time: {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
