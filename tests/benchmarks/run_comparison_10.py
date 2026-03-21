#!/usr/bin/env python3
"""
Focused 10-example comparison between Rust-VEX and Python-VEX engines.
Tracks: execution time, peak memory, result correctness.

Usage:
    python tests/benchmarks/run_comparison_10.py
"""
import importlib.util
import io
import json
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

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Selected examples for the comparison
SELECTED_EXAMPLES = [
    "flareon2015_5",
    "securityfest_fairlight",
    "fauxware",
    "ais3_crackme",
    "ekopartyctf2016_rev250",
    "csaw_wyvern",
    "flareon2015_10",
    "hackcon2016_angry-reverser",
    "sym-write",
    "grub",
]

# Default angr-examples path
DEFAULT_EXAMPLES_DIR = Path("/home/vagrant/angr-examples/examples")


@dataclass
class ExampleResult:
    """Result of running an example with a specific engine."""
    example_name: str
    engine: str
    success: bool
    execution_time_s: float
    peak_memory_mb: float = 0.0
    stdout_output: str = ""
    error_message: str | None = None
    error_traceback: str | None = None
    timeout: bool = False
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_name": self.example_name,
            "engine": self.engine,
            "success": self.success,
            "execution_time_s": self.execution_time_s,
            "peak_memory_mb": self.peak_memory_mb,
            "stdout_output": self.stdout_output,
            "error_message": self.error_message,
            "timeout": self.timeout,
            "extra_info": self.extra_info,
        }


@dataclass
class ComparisonRow:
    """A row in the comparison table."""
    example_name: str
    python_time_s: float | None
    python_success: bool
    python_error: str | None
    python_memory_mb: float
    python_output: str
    rust_time_s: float | None
    rust_success: bool
    rust_error: str | None
    rust_memory_mb: float
    rust_output: str

    @property
    def speedup(self) -> float | None:
        """Calculate speedup (Python time / Rust time)."""
        if self.python_time_s and self.rust_time_s and self.rust_time_s > 0:
            return self.python_time_s / self.rust_time_s
        return None

    @property
    def memory_ratio(self) -> float | None:
        """Calculate memory ratio (Python memory / Rust memory)."""
        if self.python_memory_mb and self.rust_memory_mb and self.rust_memory_mb > 0:
            return self.python_memory_mb / self.rust_memory_mb
        return None

    @property
    def outputs_match(self) -> tuple[bool, str]:
        """Check if outputs match between engines."""
        return verify_results_match(self.python_output, self.rust_output)

    def to_dict(self) -> dict[str, Any]:
        match, match_desc = self.outputs_match
        return {
            "example_name": self.example_name,
            "python_time_s": self.python_time_s,
            "python_success": self.python_success,
            "python_error": self.python_error,
            "python_memory_mb": self.python_memory_mb,
            "rust_time_s": self.rust_time_s,
            "rust_success": self.rust_success,
            "rust_error": self.rust_error,
            "rust_memory_mb": self.rust_memory_mb,
            "speedup": self.speedup,
            "memory_ratio": self.memory_ratio,
            "outputs_match": match,
            "match_description": match_desc,
        }


def _filter_timing_lines(output: str) -> list[str]:
    """Filter out timing and elapsed lines from output for comparison."""
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    # Filter out lines that contain timing info (these will differ between runs)
    filtered = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ['time elapsed', 'elapsed:', 'seconds', ' ms', 'timing']):
            continue
        if 'none' == lower:  # Skip bare "None" lines often printed as timing results
            continue
        filtered.append(line)
    return filtered


def verify_results_match(py_output: str, rust_output: str) -> tuple[bool, str]:
    """
    Compare outputs from both engines.
    Returns (match: bool, description: str)
    """
    if not py_output and not rust_output:
        return True, "Both empty"

    if not py_output or not rust_output:
        return False, "One output empty"

    # Filter out timing lines for comparison
    py_lines = _filter_timing_lines(py_output)
    rust_lines = _filter_timing_lines(rust_output)

    # If both are empty after filtering, consider it a match
    if not py_lines and not rust_lines:
        return True, "Both empty (after filtering timing)"

    # Exact match after filtering
    if py_lines == rust_lines:
        return True, "Exact match"

    # Check if final non-timing lines match
    if py_lines and rust_lines:
        if py_lines[-1] == rust_lines[-1]:
            return True, "Final line matches"

        # Check for common patterns in CTF solutions
        # Often the flag or result is printed with a prefix
        for line in py_lines:
            lower = line.lower()
            if any(keyword in lower for keyword in ['flag', 'password', 'key', 'solution', 'found']):
                if line in rust_lines:
                    return True, "Key result line matches"

    return False, f"Mismatch"


def _run_example_with_metrics(args: tuple) -> dict:
    """
    Run an example in a subprocess and collect metrics.

    Returns dict with:
    - execution_time_s
    - peak_memory_mb (via resource.getrusage)
    - stdout_output (captured for result comparison)
    """
    solve_script_path, example_name, engine, timeout = args

    result = {
        "example_name": example_name,
        "engine": engine,
        "success": False,
        "execution_time_s": 0.0,
        "peak_memory_mb": 0.0,
        "stdout_output": "",
        "error_message": None,
        "error_traceback": None,
        "timeout": False,
    }

    # Track memory before (to get delta)
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

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

        # Check if Rust exploration manager is available
        rust_available = False
        RustExplorationManager = None
        try:
            from angr.exploration import RustExplorationManager
            from angr.exploration.rust_manager import RUST_EXPLORATION_AVAILABLE
            rust_available = RUST_EXPLORATION_AVAILABLE
        except ImportError:
            pass

        if engine == "rust" and not rust_available:
            result["error_message"] = "RustExplorationManager not available"
            return result

        # Patch the factory to return RustExplorationManager when using Rust engine
        original_simulation_manager = None
        if engine == "rust" and rust_available:
            original_simulation_manager = angr.factory.AngrObjectFactory.simulation_manager

            def patched_simulation_manager(factory_self, thing=None, **kwargs):
                """Return RustExplorationManager instead of SimulationManager."""
                # Get states list - must create entry_state if none provided
                if thing is None:
                    states = [factory_self.entry_state()]
                elif isinstance(thing, (list, tuple)):
                    states = list(thing)
                else:
                    states = [thing]

                # Create RustExplorationManager
                rust_mgr = RustExplorationManager(factory_self.project, states)
                return rust_mgr

            angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
            # Also patch the shorthand
            angr.factory.AngrObjectFactory.simgr = patched_simulation_manager

        try:
            # Load the solve script
            spec = importlib.util.spec_from_file_location("__main__", solve_script_path)
            module = importlib.util.module_from_spec(spec)

            # Capture stdout (use BufferedStringIO to handle stdout.buffer access)
            captured = BufferedStringIO()

            # Save and replace sys.stdout
            original_stdout = sys.stdout
            sys.stdout = captured

            start_time = time.perf_counter()

            try:
                spec.loader.exec_module(module)
            finally:
                sys.stdout = original_stdout

            end_time = time.perf_counter()

            result["success"] = True
            result["execution_time_s"] = end_time - start_time
            result["stdout_output"] = captured.getvalue()

        finally:
            # Restore original factory methods
            if original_simulation_manager is not None:
                angr.factory.AngrObjectFactory.simulation_manager = original_simulation_manager
                angr.factory.AngrObjectFactory.simgr = original_simulation_manager

            # Restore working directory
            os.chdir(original_dir)

    except Exception as e:
        result["error_message"] = str(e)
        result["error_traceback"] = traceback.format_exc()

    # Track peak memory after
    # On Linux, ru_maxrss is in KB; on macOS it's in bytes
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Convert to MB (assuming Linux KB)
    result["peak_memory_mb"] = mem_after / 1024

    return result


class ComparisonRunner:
    """Runner for the 10-example comparison."""

    def __init__(self, examples_dir: Path, verbose: bool = False):
        self.examples_dir = examples_dir
        self.verbose = verbose

        if not self.examples_dir.exists():
            raise ValueError(f"Examples directory not found: {self.examples_dir}")

    def run_example(
        self,
        example_name: str,
        engine: str,
        timeout: int = 180,
        iterations: int = 3,
    ) -> ExampleResult:
        """
        Run a single example with the specified engine.

        Runs multiple iterations and returns averaged results.
        """
        solve_script = self.examples_dir / example_name / "solve.py"
        if not solve_script.exists():
            return ExampleResult(
                example_name=example_name,
                engine=engine,
                success=False,
                execution_time_s=0.0,
                error_message=f"solve.py not found at {solve_script}",
            )

        times = []
        memories = []
        last_result = None

        for i in range(iterations):
            args = (str(solve_script), example_name, engine, timeout)

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(1) as pool:
                try:
                    async_result = pool.apply_async(_run_example_with_metrics, (args,))
                    result_dict = async_result.get(timeout=timeout)
                    last_result = result_dict

                    if result_dict["success"]:
                        times.append(result_dict["execution_time_s"])
                        memories.append(result_dict["peak_memory_mb"])
                    else:
                        # If any iteration fails, return failure
                        return ExampleResult(
                            example_name=result_dict["example_name"],
                            engine=result_dict["engine"],
                            success=False,
                            execution_time_s=result_dict["execution_time_s"],
                            peak_memory_mb=result_dict["peak_memory_mb"],
                            stdout_output=result_dict.get("stdout_output", ""),
                            error_message=result_dict["error_message"],
                            error_traceback=result_dict.get("error_traceback"),
                            timeout=result_dict["timeout"],
                        )

                except multiprocessing.TimeoutError:
                    pool.terminate()
                    return ExampleResult(
                        example_name=example_name,
                        engine=engine,
                        success=False,
                        execution_time_s=timeout,
                        error_message=f"Timeout after {timeout}s",
                        timeout=True,
                    )
                except Exception as e:
                    return ExampleResult(
                        example_name=example_name,
                        engine=engine,
                        success=False,
                        execution_time_s=0.0,
                        error_message=str(e),
                        error_traceback=traceback.format_exc(),
                    )

        # All iterations succeeded - return averaged results
        avg_time = sum(times) / len(times) if times else 0.0
        avg_memory = sum(memories) / len(memories) if memories else 0.0

        return ExampleResult(
            example_name=last_result["example_name"],
            engine=last_result["engine"],
            success=True,
            execution_time_s=avg_time,
            peak_memory_mb=avg_memory,
            stdout_output=last_result.get("stdout_output", ""),
            extra_info={"iterations": iterations, "times": times, "memories": memories},
        )

    def run_comparison(
        self,
        timeout: int = 180,
        iterations: int = 3,
    ) -> list[ComparisonRow]:
        """Run all selected examples with both engines and compare."""
        comparisons = []

        total = len(SELECTED_EXAMPLES)
        for idx, example_name in enumerate(SELECTED_EXAMPLES, 1):
            print(f"\n[{idx}/{total}] Running {example_name}...")

            # Run with Python engine
            print(f"  Python engine...", end=" ", flush=True)
            start = time.perf_counter()
            python_result = self.run_example(example_name, "python", timeout, iterations)
            py_elapsed = time.perf_counter() - start
            if python_result.success:
                print(f"OK ({python_result.execution_time_s:.2f}s avg, {python_result.peak_memory_mb:.0f}MB)")
            else:
                print(f"FAILED: {python_result.error_message}")

            # Run with Rust engine
            print(f"  Rust engine...", end=" ", flush=True)
            start = time.perf_counter()
            rust_result = self.run_example(example_name, "rust", timeout, iterations)
            rust_elapsed = time.perf_counter() - start
            if rust_result.success:
                print(f"OK ({rust_result.execution_time_s:.2f}s avg, {rust_result.peak_memory_mb:.0f}MB)")
            else:
                print(f"FAILED: {rust_result.error_message}")

            # Create comparison row
            comparisons.append(ComparisonRow(
                example_name=example_name,
                python_time_s=python_result.execution_time_s if python_result.success else None,
                python_success=python_result.success,
                python_error=python_result.error_message,
                python_memory_mb=python_result.peak_memory_mb,
                python_output=python_result.stdout_output,
                rust_time_s=rust_result.execution_time_s if rust_result.success else None,
                rust_success=rust_result.success,
                rust_error=rust_result.error_message,
                rust_memory_mb=rust_result.peak_memory_mb,
                rust_output=rust_result.stdout_output,
            ))

            # Show speedup if both succeeded
            if python_result.success and rust_result.success:
                speedup = python_result.execution_time_s / rust_result.execution_time_s if rust_result.execution_time_s > 0 else 0
                match, match_desc = verify_results_match(python_result.stdout_output, rust_result.stdout_output)
                match_symbol = "\u2713" if match else "\u2717"
                print(f"  Speedup: {speedup:.2f}x | Result: {match_symbol} ({match_desc})")

        return comparisons


def generate_report(comparisons: list[ComparisonRow], output_dir: Path) -> None:
    """Generate markdown and JSON reports."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate summary statistics
    total = len(comparisons)
    both_success = [c for c in comparisons if c.python_success and c.rust_success]
    speedups = [c.speedup for c in both_success if c.speedup]
    memory_ratios = [c.memory_ratio for c in both_success if c.memory_ratio]
    matches = [c for c in both_success if c.outputs_match[0]]

    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    avg_memory_ratio = sum(memory_ratios) / len(memory_ratios) if memory_ratios else 0

    # Generate markdown report
    md_lines = [
        "=" * 60,
        "Rust-VEX vs Python-VEX: 10 Example Comparison",
        "=" * 60,
        "",
        f"Generated: {timestamp}",
        f"Iterations per example: 3 (averaged)",
        "",
        "## Results Table",
        "",
        "| Example | Py Time | Rs Time | Speedup | Py Mem | Rs Mem | Match |",
        "|---------|---------|---------|---------|--------|--------|-------|",
    ]

    for c in comparisons:
        py_time = f"{c.python_time_s:.1f}s" if c.python_time_s else "FAIL"
        rs_time = f"{c.rust_time_s:.1f}s" if c.rust_time_s else "FAIL"
        speedup = f"{c.speedup:.2f}x" if c.speedup else "-"
        py_mem = f"{c.python_memory_mb:.0f}MB" if c.python_memory_mb else "-"
        rs_mem = f"{c.rust_memory_mb:.0f}MB" if c.rust_memory_mb else "-"

        if c.python_success and c.rust_success:
            match, _ = c.outputs_match
            match_str = "\u2713" if match else "\u2717"
        else:
            match_str = "-"

        # Truncate long example names
        name = c.example_name[:26] + ".." if len(c.example_name) > 28 else c.example_name
        md_lines.append(f"| {name:<28} | {py_time:>7} | {rs_time:>7} | {speedup:>7} | {py_mem:>6} | {rs_mem:>6} | {match_str:^5} |")

    md_lines.extend([
        "",
        "## Summary",
        "",
        f"- **Examples tested**: {total}",
        f"- **Both engines succeed**: {len(both_success)}/{total}",
        f"- **Average speedup**: {avg_speedup:.2f}x",
        f"- **Average memory ratio**: {avg_memory_ratio:.2f}x (Py/Rust, <1 means Rust uses more)",
        f"- **Result verification**: {len(matches)}/{len(both_success)} outputs match",
        "",
        "## Legend",
        "",
        "- Speedup = Python time / Rust time (>1 means Rust is faster)",
        "- Memory ratio = Python memory / Rust memory",
        "- Match: \u2713 = outputs match, \u2717 = outputs differ",
        "",
    ])

    # Add details for any mismatches
    mismatches = [c for c in both_success if not c.outputs_match[0]]
    if mismatches:
        md_lines.extend([
            "## Output Mismatches",
            "",
        ])
        for c in mismatches:
            md_lines.extend([
                f"### {c.example_name}",
                "",
                "**Python output (last 200 chars):**",
                "```",
                c.python_output[-200:] if c.python_output else "(empty)",
                "```",
                "",
                "**Rust output (last 200 chars):**",
                "```",
                c.rust_output[-200:] if c.rust_output else "(empty)",
                "```",
                "",
            ])

    # Write markdown report
    md_path = output_dir / "comparison_10_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown report saved to: {md_path}")

    # Write JSON data
    json_data = {
        "generated": datetime.now().isoformat(),
        "iterations": 3,
        "summary": {
            "total_examples": total,
            "both_success": len(both_success),
            "avg_speedup": avg_speedup,
            "avg_memory_ratio": avg_memory_ratio,
            "outputs_match": len(matches),
        },
        "comparisons": [c.to_dict() for c in comparisons],
        "outputs": {
            c.example_name: {
                "python": c.python_output,
                "rust": c.rust_output,
            }
            for c in comparisons
        },
    }
    json_path = output_dir / "comparison_10_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data saved to: {json_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Rust-VEX vs Python-VEX: 10 Example Comparison")
    print("=" * 60)
    print()
    print(f"Examples: {', '.join(SELECTED_EXAMPLES)}")
    print(f"Iterations: 3 per example per engine")
    print()

    # Check examples directory
    examples_dir = DEFAULT_EXAMPLES_DIR
    if not examples_dir.exists():
        print(f"ERROR: Examples directory not found: {examples_dir}")
        return 1

    print(f"Examples directory: {examples_dir}")

    # Create runner
    runner = ComparisonRunner(examples_dir, verbose=True)

    # Run comparison
    start_time = time.perf_counter()
    comparisons = runner.run_comparison(timeout=180, iterations=3)
    total_time = time.perf_counter() - start_time

    # Generate reports
    output_dir = Path(__file__).parent.parent
    generate_report(comparisons, output_dir)

    print()
    print("=" * 60)
    print(f"Total comparison time: {total_time:.1f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
