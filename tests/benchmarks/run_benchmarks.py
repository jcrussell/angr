#!/usr/bin/env python3
"""
Unified benchmark runner for Rust vs Python VEX engine comparison.

This script orchestrates the complete benchmark suite:
1. Run Rust Criterion benchmarks (if cargo available)
2. Run Python synthetic benchmarks
3. Run Python binary benchmarks
4. Run angr-examples benchmarks (if --angr-examples specified)
5. Generate consolidated report

Usage:
    python tests/benchmarks/run_benchmarks.py [options]

Options:
    --skip-rust       Skip Rust Criterion benchmarks
    --skip-python     Skip Python synthetic benchmarks
    --skip-binaries   Skip binary benchmarks
    --angr-examples   Run angr-examples benchmarks
    --examples-dir    Path to angr-examples/examples directory
    --quick           Run with reduced iterations for quick testing
    --output PATH     Output directory for reports (default: tests/)
    --verbose         Verbose output
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def print_header(text: str):
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---\n")


def run_rust_benchmarks(native_dir: Path, verbose: bool = False) -> bool:
    """Run Rust Criterion benchmarks."""
    print_section("Running Rust Criterion Benchmarks")

    # Check if cargo is available
    try:
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            print("  [SKIP] cargo not available")
            return False
        print(f"  Using: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  [SKIP] cargo not found in PATH")
        return False

    # Check if native directory exists
    if not native_dir.exists():
        print(f"  [SKIP] Native directory not found: {native_dir}")
        return False

    # Check for Cargo.toml
    cargo_toml = native_dir / "Cargo.toml"
    if not cargo_toml.exists():
        print(f"  [SKIP] Cargo.toml not found: {cargo_toml}")
        return False

    # Run cargo bench
    print(f"  Running benchmarks in {native_dir}")
    print("  This may take several minutes...")

    cmd = ["cargo", "bench", "--features", "vex-engine"]

    try:
        if verbose:
            # Stream output in verbose mode
            process = subprocess.Popen(
                cmd,
                cwd=native_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                print(f"    {line}", end="")
            process.wait()
            success = process.returncode == 0
        else:
            result = subprocess.run(
                cmd,
                cwd=native_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            success = result.returncode == 0

            if not success:
                print(f"  [ERROR] Cargo bench failed:")
                print(result.stderr[:500] if result.stderr else "No error output")

        if success:
            print("  [OK] Rust benchmarks completed")
            return True
        else:
            return False

    except subprocess.TimeoutExpired:
        print("  [ERROR] Benchmark timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to run cargo bench: {e}")
        return False


def run_python_synthetic_benchmarks(quick: bool = False) -> list:
    """Run Python synthetic benchmarks."""
    print_section("Running Python Synthetic Benchmarks")

    try:
        from bench_rust_vex import RustEngineBenchmarks, PythonEngineBenchmarks

        results = []

        # Run Rust wrapper benchmarks
        print("  Running Rust engine wrapper benchmarks...")
        rust_bench = RustEngineBenchmarks()
        if rust_bench.available:
            rust_results = rust_bench.run_all()
            results.extend(rust_results)
            print(f"  [OK] {len(rust_results)} Rust wrapper benchmarks completed")
        else:
            print("  [SKIP] Rust engine not available")

        # Run Python engine benchmarks
        print("\n  Running Python engine benchmarks...")
        python_bench = PythonEngineBenchmarks()
        if python_bench.available:
            python_results = python_bench.run_all()
            results.extend(python_results)
            print(f"  [OK] {len(python_results)} Python benchmarks completed")
        else:
            print("  [SKIP] Python engine not available")

        return results

    except Exception as e:
        print(f"  [ERROR] Failed to run synthetic benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_binary_benchmarks(quick: bool = False) -> list:
    """Run real binary benchmarks."""
    print_section("Running Binary Benchmarks")

    try:
        from bench_binaries import BinaryBenchmarks

        bench = BinaryBenchmarks()
        results = bench.run_all()

        print(f"  [OK] {len(results)} binary benchmarks completed")
        return results

    except Exception as e:
        print(f"  [ERROR] Failed to run binary benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_angr_examples_benchmarks(
    examples_dir: Path | None,
    output_dir: Path,
    quick: bool = False,
    verbose: bool = False,
) -> tuple[list, list]:
    """Run angr-examples benchmarks."""
    print_section("Running angr-examples Benchmarks")

    try:
        from bench_angr_examples import (
            AngrExampleRunner,
            find_examples_dir,
            generate_report,
        )

        # Find examples directory
        if examples_dir is None:
            examples_dir = find_examples_dir()
            if examples_dir is None:
                print("  [SKIP] angr-examples directory not found")
                print("  Specify --examples-dir or clone angr-examples repository")
                return [], []

        print(f"  Examples directory: {examples_dir}")

        # Create runner
        runner = AngrExampleRunner(examples_dir, verbose=verbose)
        examples = runner.discover_examples()
        print(f"  Discovered {len(examples)} examples")

        # Adjust timeout for quick mode
        timeout = 30 if quick else 120

        # Progress callback
        def progress(name: str, engine: str, status: str):
            if verbose:
                print(f"    {name} ({engine}): {status}")
            elif status in ("success", "failed", "timeout", "unsupported"):
                symbol = {
                    "success": ".",
                    "failed": "F",
                    "timeout": "T",
                    "unsupported": "S",
                }.get(status, "?")
                print(symbol, end="", flush=True)

        print("  Running benchmarks", end="" if not verbose else "\n")

        results, comparisons = runner.run_all_examples(
            timeout_per_example=timeout,
            progress_callback=progress,
        )

        if not verbose:
            print()  # Newline after progress dots

        # Generate angr-examples specific report
        generate_report(comparisons, results, output_dir)

        success_count = sum(1 for r in results if r.success)
        print(f"  [OK] {len(results)} example runs completed ({success_count} successful)")

        return results, comparisons

    except Exception as e:
        print(f"  [ERROR] Failed to run angr-examples benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def generate_report(
    output_dir: Path,
    native_dir: Path,
    python_results: list,
    binary_results: list,
    examples_comparisons: list | None = None,
) -> bool:
    """Generate consolidated benchmark report."""
    print_section("Generating Report")

    try:
        from report_generator import ReportGenerator

        generator = ReportGenerator()
        generator.collect_system_info()

        # Load Criterion results if available
        criterion_dir = native_dir / "target" / "criterion"
        if criterion_dir.exists():
            generator.load_criterion_results(criterion_dir)
            print(f"  Loaded {len(generator.rust_results)} Criterion benchmarks")
        else:
            print("  [INFO] No Criterion results found")

        # Add Python results
        generator.load_python_results(python_results)
        print(f"  Loaded {len(python_results)} Python benchmark results")

        # Add binary results
        generator.load_binary_results(binary_results)
        print(f"  Loaded {len(binary_results)} binary benchmark results")

        # Add angr-examples results
        if examples_comparisons:
            generator.load_examples_results(examples_comparisons)
            print(f"  Loaded {len(examples_comparisons)} angr-examples comparisons")

        # Generate markdown report
        report_path = output_dir / "benchmark_report.md"
        generator.save_report(report_path)
        print(f"  [OK] Report saved to {report_path}")

        # Generate JSON data
        json_path = output_dir / "benchmark_data.json"
        generator.save_json(json_path)
        print(f"  [OK] JSON data saved to {json_path}")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_rust_engine() -> bool:
    """Verify that the Rust engine is available."""
    print_section("Verifying Rust Engine")

    try:
        from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE

        if RUST_ENGINE_AVAILABLE:
            print("  [OK] Rust VEX engine is available")
            return True
        else:
            print("  [WARN] Rust VEX engine is NOT available")
            print("  Compile with: maturin develop --features vex-engine")
            return False

    except ImportError as e:
        print(f"  [WARN] Could not import rust_vex module: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Rust vs Python VEX engine benchmark suite"
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip Rust Criterion benchmarks",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python synthetic benchmarks",
    )
    parser.add_argument(
        "--skip-binaries",
        action="store_true",
        help="Skip binary benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced iterations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--angr-examples",
        action="store_true",
        help="Run angr-examples benchmarks",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Path to angr-examples/examples directory",
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    tests_dir = script_dir.parent
    angr_dir = tests_dir.parent
    native_dir = angr_dir / "native" / "angr"
    output_dir = args.output or tests_dir

    print_header("Rust vs Python VEX Engine Benchmark Suite")

    print(f"Working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Native directory: {native_dir}")
    print(f"Output directory: {output_dir}")

    start_time = time.perf_counter()

    # Verify Rust engine
    rust_available = verify_rust_engine()

    # Run benchmarks
    python_results = []
    binary_results = []
    examples_results = []
    examples_comparisons = []

    # Step 1: Rust Criterion benchmarks
    if not args.skip_rust:
        run_rust_benchmarks(native_dir, args.verbose)

    # Step 2: Python synthetic benchmarks
    if not args.skip_python:
        python_results = run_python_synthetic_benchmarks(args.quick)

    # Step 3: Binary benchmarks
    if not args.skip_binaries:
        binary_results = run_binary_benchmarks(args.quick)

    # Step 4: angr-examples benchmarks (optional)
    if args.angr_examples:
        examples_results, examples_comparisons = run_angr_examples_benchmarks(
            args.examples_dir,
            output_dir,
            args.quick,
            args.verbose,
        )

    # Step 5: Generate report
    generate_report(output_dir, native_dir, python_results, binary_results, examples_comparisons)

    # Summary
    elapsed = time.perf_counter() - start_time

    print_header("Benchmark Suite Complete")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Python results: {len(python_results)}")
    print(f"Binary results: {len(binary_results)}")
    print(f"Rust engine available: {rust_available}")
    print(f"\nReport: {output_dir / 'benchmark_report.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
