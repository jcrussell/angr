#!/usr/bin/env python3
"""
Benchmark suite for comparing Rust VEX engine vs Python VEX engine.

This suite measures:
1. State forking performance
2. Block execution speed
3. Memory operations
4. Register access patterns
5. Symbolic execution overhead

Run with: python -m pytest tests/benchmarks/bench_rust_vex.py -v --benchmark-enable
Or standalone: python tests/benchmarks/bench_rust_vex.py
"""
import os
import sys
import time
import statistics
from typing import Callable, Any
from dataclasses import dataclass, field

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    iterations: int
    total_time: float
    times: list[float] = field(default_factory=list)

    @property
    def mean_time(self) -> float:
        return self.total_time / self.iterations

    @property
    def ops_per_second(self) -> float:
        return self.iterations / self.total_time if self.total_time > 0 else 0

    @property
    def std_dev(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times)

    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.mean_time*1000:.3f}ms/op "
            f"({self.ops_per_second:.1f} ops/s, stddev={self.std_dev*1000:.3f}ms)"
        )


def benchmark(func: Callable[[], Any], name: str, iterations: int = 1000, warmup: int = 10) -> BenchmarkResult:
    """Run a benchmark function and return results."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    start_total = time.perf_counter()
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    end_total = time.perf_counter()

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=end_total - start_total,
        times=times,
    )


class RustEngineBenchmarks:
    """Benchmarks for the Rust VEX engine wrapper."""

    def __init__(self):
        try:
            from angr.engines.rust_vex import RustVEXEngineWrapper, RUST_ENGINE_AVAILABLE
            if not RUST_ENGINE_AVAILABLE:
                raise ImportError("Rust engine not available")
            self.wrapper_class = RustVEXEngineWrapper
            self.available = True
        except ImportError as e:
            print(f"Rust engine not available: {e}")
            self.available = False

    def setup_engine(self) -> "RustVEXEngineWrapper":
        """Create a fresh engine for benchmarking."""
        engine = self.wrapper_class("amd64")
        engine.map_memory(0x1000, 0x10000)
        engine.map_memory(0x400000, 0x10000)
        return engine

    def bench_fork(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark state forking - this is where Rust CoW should shine."""
        engine = self.setup_engine()
        # Set up some state
        engine.set_register("rax", 0x12345678)
        engine.set_register("rbx", 0xDEADBEEF)
        engine.pc = 0x401000
        engine.write_memory(0x1000, b"\x00" * 4096)

        def do_fork():
            forked = engine.fork()
            return forked

        return benchmark(do_fork, "rust_fork", iterations)

    def bench_register_read(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark register read operations."""
        engine = self.setup_engine()
        engine.set_register("rax", 0x12345678)

        def do_read():
            return engine.get_register("rax")

        return benchmark(do_read, "rust_register_read", iterations)

    def bench_register_write(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark register write operations."""
        engine = self.setup_engine()

        counter = [0]
        def do_write():
            engine.set_register("rax", counter[0])
            counter[0] += 1

        return benchmark(do_write, "rust_register_write", iterations)

    def bench_memory_read(self, iterations: int = 50000) -> BenchmarkResult:
        """Benchmark memory read operations."""
        engine = self.setup_engine()
        engine.write_memory(0x1000, b"\xAA" * 4096)

        def do_read():
            return engine.read_memory(0x1000, 8)

        return benchmark(do_read, "rust_memory_read", iterations)

    def bench_memory_write(self, iterations: int = 50000) -> BenchmarkResult:
        """Benchmark memory write operations."""
        engine = self.setup_engine()

        counter = [0]
        def do_write():
            data = counter[0].to_bytes(8, 'little')
            engine.write_memory(0x1000, data)
            counter[0] += 1

        return benchmark(do_write, "rust_memory_write", iterations)

    def bench_bulk_registers(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark reading all registers at once."""
        engine = self.setup_engine()
        for i, reg in enumerate(["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp"]):
            engine.set_register(reg, i * 0x1000)

        def do_bulk_read():
            return engine.get_registers()

        return benchmark(do_bulk_read, "rust_bulk_registers", iterations)

    def bench_fork_modify(self, iterations: int = 5000) -> BenchmarkResult:
        """Benchmark fork + modify pattern (common in symbolic execution)."""
        engine = self.setup_engine()
        engine.set_register("rax", 0x12345678)
        engine.pc = 0x401000

        def do_fork_modify():
            forked = engine.fork()
            forked.set_register("rax", 0xDEADBEEF)
            forked.pc = 0x401008
            return forked

        return benchmark(do_fork_modify, "rust_fork_modify", iterations)

    def bench_many_forks(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark creating many forks (path explosion scenario)."""
        engine = self.setup_engine()
        engine.set_register("rax", 0x12345678)

        def do_many_forks():
            forks = []
            current = engine
            for i in range(100):
                current = current.fork()
                current.set_register("rax", i)
                forks.append(current)
            return forks

        return benchmark(do_many_forks, "rust_many_forks_100", iterations)

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        if not self.available:
            return []

        results = []
        print("\n=== Rust VEX Engine Benchmarks ===\n")

        benchmarks = [
            self.bench_register_read,
            self.bench_register_write,
            self.bench_memory_read,
            self.bench_memory_write,
            self.bench_bulk_registers,
            self.bench_fork,
            self.bench_fork_modify,
            self.bench_many_forks,
        ]

        for bench_func in benchmarks:
            try:
                result = bench_func()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Error in {bench_func.__name__}: {e}")

        return results


class PythonEngineBenchmarks:
    """Benchmarks for the Python VEX engine (for comparison)."""

    def __init__(self):
        try:
            import angr
            self.angr = angr
            self.available = True
        except ImportError:
            print("angr not available for Python benchmarks")
            self.available = False

    def setup_state(self) -> "angr.SimState":
        """Create a simple state for benchmarking."""
        # Create a minimal project
        proj = self.angr.Project(
            "/bin/true",
            auto_load_libs=False,
            load_options={'main_opts': {'backend': 'blob', 'arch': 'amd64'}}
        )
        state = proj.factory.blank_state(addr=0x401000)
        return state

    def bench_fork(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark state forking in Python."""
        state = self.setup_state()
        state.regs.rax = 0x12345678
        state.regs.rbx = 0xDEADBEEF

        def do_fork():
            return state.copy()

        return benchmark(do_fork, "python_fork", iterations)

    def bench_register_read(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark register read in Python."""
        state = self.setup_state()
        state.regs.rax = 0x12345678

        def do_read():
            return state.regs.rax

        return benchmark(do_read, "python_register_read", iterations)

    def bench_register_write(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark register write in Python."""
        state = self.setup_state()

        counter = [0]
        def do_write():
            state.regs.rax = counter[0]
            counter[0] += 1

        return benchmark(do_write, "python_register_write", iterations)

    def bench_fork_modify(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark fork + modify pattern in Python."""
        state = self.setup_state()
        state.regs.rax = 0x12345678

        def do_fork_modify():
            forked = state.copy()
            forked.regs.rax = 0xDEADBEEF
            forked.ip = 0x401008
            return forked

        return benchmark(do_fork_modify, "python_fork_modify", iterations)

    def bench_many_forks(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark creating many forks in Python."""
        state = self.setup_state()
        state.regs.rax = 0x12345678

        def do_many_forks():
            forks = []
            current = state
            for i in range(100):
                current = current.copy()
                current.regs.rax = i
                forks.append(current)
            return forks

        return benchmark(do_many_forks, "python_many_forks_100", iterations)

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        if not self.available:
            return []

        results = []
        print("\n=== Python VEX Engine Benchmarks ===\n")

        benchmarks = [
            self.bench_register_read,
            self.bench_register_write,
            self.bench_fork,
            self.bench_fork_modify,
            self.bench_many_forks,
        ]

        for bench_func in benchmarks:
            try:
                result = bench_func()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Error in {bench_func.__name__}: {e}")

        return results


def compare_results(rust_results: list[BenchmarkResult], python_results: list[BenchmarkResult]):
    """Compare Rust and Python benchmark results."""
    print("\n=== Comparison (Rust vs Python) ===\n")

    comparisons = {
        "fork": ("rust_fork", "python_fork"),
        "register_read": ("rust_register_read", "python_register_read"),
        "register_write": ("rust_register_write", "python_register_write"),
        "fork_modify": ("rust_fork_modify", "python_fork_modify"),
        "many_forks": ("rust_many_forks_100", "python_many_forks_100"),
    }

    rust_dict = {r.name: r for r in rust_results}
    python_dict = {r.name: r for r in python_results}

    for name, (rust_name, python_name) in comparisons.items():
        rust_r = rust_dict.get(rust_name)
        python_r = python_dict.get(python_name)

        if rust_r and python_r:
            speedup = python_r.mean_time / rust_r.mean_time if rust_r.mean_time > 0 else 0
            print(f"{name}:")
            print(f"  Rust:   {rust_r.mean_time*1000:.3f}ms ({rust_r.ops_per_second:.1f} ops/s)")
            print(f"  Python: {python_r.mean_time*1000:.3f}ms ({python_r.ops_per_second:.1f} ops/s)")
            print(f"  Speedup: {speedup:.1f}x")
            print()


def generate_report(rust_results: list[BenchmarkResult], python_results: list[BenchmarkResult]) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        "# Rust VEX Engine Benchmark Report",
        "",
        "## Summary",
        "",
        "This report compares the performance of the Rust VEX engine against",
        "the Python VEX engine for common symbolic execution operations.",
        "",
        "## Results",
        "",
        "### Rust Engine",
        "",
        "| Benchmark | Time (ms) | Ops/s | Std Dev (ms) |",
        "|-----------|-----------|-------|--------------|",
    ]

    for r in rust_results:
        lines.append(f"| {r.name} | {r.mean_time*1000:.3f} | {r.ops_per_second:.1f} | {r.std_dev*1000:.3f} |")

    lines.extend([
        "",
        "### Python Engine",
        "",
        "| Benchmark | Time (ms) | Ops/s | Std Dev (ms) |",
        "|-----------|-----------|-------|--------------|",
    ])

    for r in python_results:
        lines.append(f"| {r.name} | {r.mean_time*1000:.3f} | {r.ops_per_second:.1f} | {r.std_dev*1000:.3f} |")

    lines.extend([
        "",
        "### Comparison",
        "",
        "| Operation | Rust (ms) | Python (ms) | Speedup |",
        "|-----------|-----------|-------------|---------|",
    ])

    comparisons = {
        "Fork": ("rust_fork", "python_fork"),
        "Register Read": ("rust_register_read", "python_register_read"),
        "Register Write": ("rust_register_write", "python_register_write"),
        "Fork + Modify": ("rust_fork_modify", "python_fork_modify"),
        "100 Forks": ("rust_many_forks_100", "python_many_forks_100"),
    }

    rust_dict = {r.name: r for r in rust_results}
    python_dict = {r.name: r for r in python_results}

    for name, (rust_name, python_name) in comparisons.items():
        rust_r = rust_dict.get(rust_name)
        python_r = python_dict.get(python_name)

        if rust_r and python_r:
            speedup = python_r.mean_time / rust_r.mean_time if rust_r.mean_time > 0 else 0
            lines.append(f"| {name} | {rust_r.mean_time*1000:.3f} | {python_r.mean_time*1000:.3f} | {speedup:.1f}x |")

    return "\n".join(lines)


def main():
    """Run all benchmarks and generate report."""
    print("=" * 60)
    print("Rust VEX Engine Benchmark Suite")
    print("=" * 60)

    # Run Rust benchmarks
    rust_bench = RustEngineBenchmarks()
    rust_results = rust_bench.run_all()

    # Run Python benchmarks
    python_bench = PythonEngineBenchmarks()
    python_results = python_bench.run_all()

    # Compare
    if rust_results and python_results:
        compare_results(rust_results, python_results)

        # Generate report
        report = generate_report(rust_results, python_results)
        report_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
