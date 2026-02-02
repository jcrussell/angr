#!/usr/bin/env python3
"""
Real binary benchmarks for comparing Rust VEX engine vs Python VEX engine.

This suite measures performance on actual binaries:
1. Block execution comparison (Rust vs Python on same blocks)
2. Shellcode execution comparison
3. CFG analysis
4. Symbolic exploration

Run standalone: python tests/benchmarks/bench_binaries.py
"""
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the benchmark utilities from the main benchmark module
from bench_rust_vex import BenchmarkResult, benchmark

# Set up logging
logging.basicConfig(level=logging.WARNING)
l = logging.getLogger(__name__)


# Binary locations - from angr-binaries repo
def get_bin_location():
    """Get the path to the angr-binaries test fixtures."""
    tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    angr_dir = os.path.dirname(tests_dir)
    return os.path.join(angr_dir, "..", "binaries")


BIN_LOCATION = get_bin_location()


@dataclass
class BinaryBenchmarkResult:
    """Result of a binary benchmark with additional metadata."""
    name: str
    binary: str
    benchmark_type: str  # exploration, cfg, block_exec, loop, comparison
    iterations: int
    total_time: float
    times: list[float] = field(default_factory=list)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_time(self) -> float:
        return self.total_time / self.iterations if self.iterations > 0 else 0

    @property
    def ops_per_second(self) -> float:
        return self.iterations / self.total_time if self.total_time > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "binary": self.binary,
            "benchmark_type": self.benchmark_type,
            "iterations": self.iterations,
            "total_time_s": self.total_time,
            "mean_time_s": self.mean_time,
            "mean_time_ms": self.mean_time * 1000,
            "extra_metrics": self.extra_metrics,
        }

    def __repr__(self) -> str:
        return (
            f"{self.name} ({self.binary}): {self.mean_time*1000:.1f}ms/iter "
            f"({self.iterations} iters, {self.extra_metrics})"
        )


@dataclass
class ComparisonResult:
    """Result comparing Rust vs Python engine on same workload."""
    name: str
    rust_time_ms: float
    python_time_ms: float
    speedup: float
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rust_time_ms": self.rust_time_ms,
            "python_time_ms": self.python_time_ms,
            "speedup": self.speedup,
            "extra_metrics": self.extra_metrics,
        }

    def __repr__(self) -> str:
        return (
            f"{self.name}: Rust={self.rust_time_ms:.2f}ms, "
            f"Python={self.python_time_ms:.2f}ms, Speedup={self.speedup:.1f}x"
        )


class BinaryBenchmarks:
    """Benchmarks using real binaries from angr-binaries."""

    def __init__(self):
        self.angr = None
        self.claripy = None
        self.rust_available = False
        self.rust_wrapper = None
        self._setup()

    def _setup(self):
        """Set up angr and check availability."""
        try:
            import angr
            import claripy
            self.angr = angr
            self.claripy = claripy
        except ImportError as e:
            l.error(f"angr not available: {e}")
            return

        try:
            from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE, RustVEXEngineWrapper
            self.rust_available = RUST_ENGINE_AVAILABLE
            if RUST_ENGINE_AVAILABLE:
                self.rust_wrapper_class = RustVEXEngineWrapper
        except ImportError:
            self.rust_available = False

    def _get_binary_path(self, category: str, name: str) -> str | None:
        """Get the full path to a test binary."""
        path = os.path.join(BIN_LOCATION, category, name)
        if os.path.exists(path):
            return path

        # Try with architecture subdirectory
        for arch in ["x86_64", "i386", "amd64"]:
            arch_path = os.path.join(BIN_LOCATION, category, arch, name)
            if os.path.exists(arch_path):
                return arch_path

        return None

    def _check_binary_available(self, path: str | None, name: str) -> bool:
        """Check if a binary is available and print status."""
        if path is None or not os.path.exists(path):
            print(f"  [SKIP] {name} binary not found")
            return False
        return True

    # =========================================================================
    # Engine Comparison Benchmarks - Same code, different engines
    # =========================================================================

    def bench_block_execution_comparison(self, iterations: int = 100) -> ComparisonResult | None:
        """Compare block execution between Rust and Python engines."""
        if not self.rust_available:
            print("  [SKIP] Rust engine not available for comparison")
            return None

        # Simple arithmetic block: mov rax, 1; add rax, 2; ret
        code = bytes([
            0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  # mov rax, 1
            0x48, 0x83, 0xC0, 0x02,                      # add rax, 2
            0xC3,                                         # ret
        ])

        # Benchmark Rust engine
        rust_engine = self.rust_wrapper_class("amd64")
        rust_engine.map_memory_data(0x400000, code)
        rust_engine.pc = 0x400000

        rust_times = []
        for _ in range(iterations):
            rust_engine.pc = 0x400000
            start = time.perf_counter()
            rust_engine.execute_code(code)
            end = time.perf_counter()
            rust_times.append(end - start)

        rust_mean = sum(rust_times) / len(rust_times)

        # Benchmark Python engine
        proj = self.angr.load_shellcode(code, "amd64", start_offset=0, load_address=0x400000)

        python_times = []
        for _ in range(iterations):
            state = proj.factory.blank_state(addr=0x400000)
            simgr = proj.factory.simulation_manager(state)

            start = time.perf_counter()
            simgr.step()
            end = time.perf_counter()
            python_times.append(end - start)

        python_mean = sum(python_times) / len(python_times)

        speedup = python_mean / rust_mean if rust_mean > 0 else 0

        return ComparisonResult(
            name="block_execution",
            rust_time_ms=rust_mean * 1000,
            python_time_ms=python_mean * 1000,
            speedup=speedup,
            extra_metrics={"iterations": iterations, "code_size": len(code)},
        )

    def bench_arithmetic_comparison(self, iterations: int = 100) -> ComparisonResult | None:
        """Compare arithmetic-heavy code execution."""
        if not self.rust_available:
            return None

        # Sequence of arithmetic operations
        code = bytes([
            0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00,  # mov rax, 1
            0x48, 0xC7, 0xC3, 0x02, 0x00, 0x00, 0x00,  # mov rbx, 2
            0x48, 0x01, 0xD8,                          # add rax, rbx
            0x48, 0x0F, 0xAF, 0xC3,                    # imul rax, rbx
            0x48, 0x31, 0xD8,                          # xor rax, rbx
            0x48, 0x21, 0xD8,                          # and rax, rbx
            0x48, 0x09, 0xD8,                          # or rax, rbx
            0x48, 0xD1, 0xE0,                          # shl rax, 1
            0x48, 0xD1, 0xE8,                          # shr rax, 1
            0xC3,                                       # ret
        ])

        # Rust engine
        rust_engine = self.rust_wrapper_class("amd64")
        rust_engine.map_memory_data(0x400000, code)

        rust_times = []
        for _ in range(iterations):
            rust_engine.pc = 0x400000
            start = time.perf_counter()
            rust_engine.execute_code(code)
            end = time.perf_counter()
            rust_times.append(end - start)

        rust_mean = sum(rust_times) / len(rust_times)

        # Python engine
        proj = self.angr.load_shellcode(code, "amd64", start_offset=0, load_address=0x400000)

        python_times = []
        for _ in range(iterations):
            state = proj.factory.blank_state(addr=0x400000)
            simgr = proj.factory.simulation_manager(state)

            start = time.perf_counter()
            simgr.run()
            end = time.perf_counter()
            python_times.append(end - start)

        python_mean = sum(python_times) / len(python_times)
        speedup = python_mean / rust_mean if rust_mean > 0 else 0

        return ComparisonResult(
            name="arithmetic_sequence",
            rust_time_ms=rust_mean * 1000,
            python_time_ms=python_mean * 1000,
            speedup=speedup,
            extra_metrics={"iterations": iterations, "instructions": 10},
        )

    def bench_memory_ops_comparison(self, iterations: int = 100) -> ComparisonResult | None:
        """Compare memory-heavy code execution."""
        if not self.rust_available:
            return None

        # Memory load/store sequence
        code = bytes([
            0x48, 0xC7, 0xC0, 0x00, 0x10, 0x00, 0x00,  # mov rax, 0x1000
            0x48, 0xC7, 0x00, 0x42, 0x00, 0x00, 0x00,  # mov qword [rax], 0x42
            0x48, 0x8B, 0x18,                          # mov rbx, [rax]
            0x48, 0x89, 0x58, 0x08,                    # mov [rax+8], rbx
            0x48, 0x8B, 0x48, 0x08,                    # mov rcx, [rax+8]
            0xC3,                                       # ret
        ])

        # Rust engine
        rust_engine = self.rust_wrapper_class("amd64")
        rust_engine.map_memory_data(0x400000, code)
        rust_engine.map_memory(0x1000, 0x1000)  # Map memory for stores

        rust_times = []
        for _ in range(iterations):
            rust_engine.pc = 0x400000
            start = time.perf_counter()
            rust_engine.execute_code(code)
            end = time.perf_counter()
            rust_times.append(end - start)

        rust_mean = sum(rust_times) / len(rust_times)

        # Python engine
        proj = self.angr.load_shellcode(code, "amd64", start_offset=0, load_address=0x400000)

        python_times = []
        for _ in range(iterations):
            state = proj.factory.blank_state(addr=0x400000)
            state.memory.store(0x1000, self.claripy.BVV(0, 128))  # Pre-map memory

            simgr = proj.factory.simulation_manager(state)

            start = time.perf_counter()
            simgr.run()
            end = time.perf_counter()
            python_times.append(end - start)

        python_mean = sum(python_times) / len(python_times)
        speedup = python_mean / rust_mean if rust_mean > 0 else 0

        return ComparisonResult(
            name="memory_operations",
            rust_time_ms=rust_mean * 1000,
            python_time_ms=python_mean * 1000,
            speedup=speedup,
            extra_metrics={"iterations": iterations, "mem_ops": 4},
        )

    def bench_loop_comparison(self, iterations: int = 50) -> ComparisonResult | None:
        """Compare loop execution."""
        if not self.rust_available:
            return None

        # Simple loop: mov rcx, 10; dec rcx; jnz loop; ret
        code = bytes([
            0x48, 0xC7, 0xC1, 0x0A, 0x00, 0x00, 0x00,  # mov rcx, 10
            0x48, 0xFF, 0xC9,                          # dec rcx
            0x75, 0xFB,                                # jnz -5 (loop back)
            0xC3,                                       # ret
        ])

        # Rust engine - execute the full loop
        rust_engine = self.rust_wrapper_class("amd64")
        rust_engine.map_memory_data(0x400000, code)

        rust_times = []
        for _ in range(iterations):
            rust_engine.pc = 0x400000
            rust_engine.set_register("rcx", 0)
            start = time.perf_counter()
            # Execute multiple blocks to cover the loop
            for _ in range(15):  # 10 loop iterations + some overhead
                try:
                    rust_engine.execute_code(code)
                except:
                    break
            end = time.perf_counter()
            rust_times.append(end - start)

        rust_mean = sum(rust_times) / len(rust_times)

        # Python engine
        proj = self.angr.load_shellcode(code, "amd64", start_offset=0, load_address=0x400000)

        python_times = []
        for _ in range(iterations):
            state = proj.factory.blank_state(addr=0x400000)
            simgr = proj.factory.simulation_manager(state)

            start = time.perf_counter()
            steps = 0
            while simgr.active and steps < 15:
                simgr.step()
                steps += 1
            end = time.perf_counter()
            python_times.append(end - start)

        python_mean = sum(python_times) / len(python_times)
        speedup = python_mean / rust_mean if rust_mean > 0 else 0

        return ComparisonResult(
            name="loop_execution",
            rust_time_ms=rust_mean * 1000,
            python_time_ms=python_mean * 1000,
            speedup=speedup,
            extra_metrics={"iterations": iterations, "loop_count": 10},
        )

    def bench_block_lifting_comparison(self, iterations: int = 500) -> ComparisonResult | None:
        """Compare block lifting + execution overhead."""
        if not self.rust_available:
            return None

        # NOP sled + ret (tests lifting overhead)
        code = bytes([0x90] * 16 + [0xC3])  # 16 NOPs + ret

        # Rust engine
        rust_engine = self.rust_wrapper_class("amd64")
        rust_engine.map_memory_data(0x400000, code)

        rust_times = []
        for _ in range(iterations):
            rust_engine.pc = 0x400000
            start = time.perf_counter()
            rust_engine.execute_code(code)
            end = time.perf_counter()
            rust_times.append(end - start)

        rust_mean = sum(rust_times) / len(rust_times)

        # Python engine
        proj = self.angr.load_shellcode(code, "amd64", start_offset=0, load_address=0x400000)

        python_times = []
        for _ in range(iterations):
            state = proj.factory.blank_state(addr=0x400000)
            simgr = proj.factory.simulation_manager(state)

            start = time.perf_counter()
            simgr.step()
            end = time.perf_counter()
            python_times.append(end - start)

        python_mean = sum(python_times) / len(python_times)
        speedup = python_mean / rust_mean if rust_mean > 0 else 0

        return ComparisonResult(
            name="nop_block_lifting",
            rust_time_ms=rust_mean * 1000,
            python_time_ms=python_mean * 1000,
            speedup=speedup,
            extra_metrics={"iterations": iterations, "nops": 16},
        )

    # =========================================================================
    # Single-engine benchmarks (for reference)
    # =========================================================================

    def bench_libc_load(self, iterations: int = 5) -> BinaryBenchmarkResult | None:
        """Benchmark loading libc.so.6."""
        libc_paths = [
            "/lib/x86_64-linux-gnu/libc.so.6",
            "/lib64/libc.so.6",
            "/usr/lib/libc.so.6",
        ]

        binary_path = None
        for path in libc_paths:
            if os.path.exists(path):
                binary_path = path
                break

        if not self._check_binary_available(binary_path, "libc.so.6"):
            return None

        times = []
        start_total = time.perf_counter()

        for _ in range(iterations):
            start = time.perf_counter()
            proj = self.angr.Project(binary_path, auto_load_libs=False)
            end = time.perf_counter()
            times.append(end - start)
            del proj

        end_total = time.perf_counter()

        return BinaryBenchmarkResult(
            name="libc_load",
            binary="libc.so.6",
            benchmark_type="load",
            iterations=iterations,
            total_time=end_total - start_total,
            times=times,
            extra_metrics={},
        )

    def bench_libc_cfg_partial(self, iterations: int = 1) -> BinaryBenchmarkResult | None:
        """Benchmark partial CFG construction on libc."""
        libc_paths = [
            "/lib/x86_64-linux-gnu/libc.so.6",
            "/lib64/libc.so.6",
        ]

        binary_path = None
        for path in libc_paths:
            if os.path.exists(path):
                binary_path = path
                break

        if not self._check_binary_available(binary_path, "libc.so.6"):
            return None

        proj = self.angr.Project(binary_path, auto_load_libs=False)

        times = []
        node_counts = []
        start_total = time.perf_counter()

        for _ in range(iterations):
            start = time.perf_counter()
            cfg = proj.analyses.CFGFast(
                force_complete_scan=False,
                normalize=False,
                resolve_indirect_jumps=False,
            )
            end = time.perf_counter()
            times.append(end - start)
            node_counts.append(len(cfg.graph.nodes()))

        end_total = time.perf_counter()

        return BinaryBenchmarkResult(
            name="libc_cfg_partial",
            binary="libc.so.6",
            benchmark_type="cfg",
            iterations=iterations,
            total_time=end_total - start_total,
            times=times,
            extra_metrics={
                "avg_nodes": sum(node_counts) / len(node_counts) if node_counts else 0,
            },
        )

    def run_all(self) -> list:
        """Run all binary benchmarks."""
        if self.angr is None:
            print("angr not available - skipping binary benchmarks")
            return []

        results = []
        print("\n=== Real Binary Benchmarks ===\n")

        # Run comparison benchmarks first
        print("--- Engine Comparison Benchmarks ---\n")
        comparison_benchmarks = [
            ("block_execution", self.bench_block_execution_comparison),
            ("arithmetic_sequence", self.bench_arithmetic_comparison),
            ("memory_operations", self.bench_memory_ops_comparison),
            ("loop_execution", self.bench_loop_comparison),
            ("nop_block_lifting", self.bench_block_lifting_comparison),
        ]

        for name, bench_func in comparison_benchmarks:
            try:
                print(f"Running {name}...", end=" ", flush=True)
                result = bench_func()
                if result is not None:
                    results.append(result)
                    print(f"Rust={result.rust_time_ms:.2f}ms, Python={result.python_time_ms:.2f}ms, Speedup={result.speedup:.1f}x")
                else:
                    print("skipped")
            except Exception as e:
                print(f"error: {e}")
                l.exception(f"Error in {name}")

        # Run single-engine benchmarks
        print("\n--- Single Engine Benchmarks ---\n")
        single_benchmarks = [
            ("libc_load", self.bench_libc_load),
            ("libc_cfg_partial", self.bench_libc_cfg_partial),
        ]

        for name, bench_func in single_benchmarks:
            try:
                print(f"Running {name}...", end=" ", flush=True)
                result = bench_func()
                if result is not None:
                    results.append(result)
                    print(f"{result.mean_time*1000:.1f}ms/iter")
                else:
                    print("skipped")
            except Exception as e:
                print(f"error: {e}")
                l.exception(f"Error in {name}")

        return results


def generate_binary_report(results: list) -> str:
    """Generate a markdown report for binary benchmarks."""
    lines = [
        "# Real Binary Benchmark Report",
        "",
        "## Overview",
        "",
        "This report compares Rust vs Python VEX engine performance on real code execution.",
        "",
    ]

    # Separate comparison results from single-engine results
    comparisons = [r for r in results if isinstance(r, ComparisonResult)]
    single_results = [r for r in results if isinstance(r, BinaryBenchmarkResult)]

    if comparisons:
        lines.extend([
            "## Engine Comparison (Same Code, Different Engines)",
            "",
            "| Benchmark | Rust (ms) | Python (ms) | Speedup |",
            "|-----------|-----------|-------------|---------|",
        ])

        for r in comparisons:
            lines.append(
                f"| {r.name} | {r.rust_time_ms:.2f} | {r.python_time_ms:.2f} | {r.speedup:.1f}x |"
            )

        # Summary stats
        avg_speedup = sum(r.speedup for r in comparisons) / len(comparisons)
        max_speedup = max(comparisons, key=lambda x: x.speedup)
        lines.extend([
            "",
            f"**Average Speedup**: {avg_speedup:.1f}x",
            f"**Maximum Speedup**: {max_speedup.speedup:.1f}x ({max_speedup.name})",
            "",
        ])

    if single_results:
        # Group by benchmark type
        by_type: dict[str, list[BinaryBenchmarkResult]] = {}
        for r in single_results:
            by_type.setdefault(r.benchmark_type, []).append(r)

        lines.append("## Single Engine Benchmarks")
        lines.append("")

        for bench_type, type_results in sorted(by_type.items()):
            lines.append(f"### {bench_type.title()}")
            lines.append("")
            lines.append("| Benchmark | Binary | Mean (ms) | Notes |")
            lines.append("|-----------|--------|-----------|-------|")

            for r in type_results:
                extra_str = ", ".join(
                    f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in r.extra_metrics.items()
                )
                lines.append(
                    f"| {r.name} | {r.binary} | {r.mean_time*1000:.1f} | {extra_str} |"
                )

            lines.append("")

    return "\n".join(lines)


def main():
    """Run all binary benchmarks and generate report."""
    print("=" * 60)
    print("Real Binary Benchmark Suite")
    print("=" * 60)

    bench = BinaryBenchmarks()
    results = bench.run_all()

    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        for r in results:
            print(r)

        # Generate report
        report = generate_binary_report(results)
        report_path = os.path.join(os.path.dirname(__file__), "..", "binary_benchmark_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
