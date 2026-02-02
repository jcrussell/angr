#!/usr/bin/env python3
"""
Benchmark suite for comparing Rust VEX engine vs Python VEX engine.

This suite measures:
1. State forking performance
2. Block execution speed
3. Memory operations
4. Register access patterns
5. Bitvector operations
6. Symbolic execution overhead

Run with: python -m pytest tests/benchmarks/bench_rust_vex.py -v --benchmark-enable
Or standalone: python tests/benchmarks/bench_rust_vex.py
"""
import os
import sys
import time
import math
import statistics
from typing import Callable, Any
from dataclasses import dataclass, field

# Add the reference/angr directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark with comprehensive statistics."""
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

    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0.0

    @property
    def median_time(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    def percentile(self, p: float) -> float:
        """Calculate the p-th percentile (0-100)."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        k = (len(sorted_times) - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_times[int(k)]
        return sorted_times[int(f)] * (c - k) + sorted_times[int(c)] * (k - f)

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval for the mean."""
        if len(self.times) < 2:
            return (self.mean_time, self.mean_time)

        n = len(self.times)
        mean = self.mean_time
        se = self.std_dev / math.sqrt(n)

        # Use t-distribution approximation for small samples
        # For 95% confidence with large n, z ~= 1.96
        if n >= 30:
            z = 1.96 if confidence == 0.95 else 2.576
        else:
            # Simple approximation for t-distribution
            z = 2.0 if confidence == 0.95 else 2.75

        margin = z * se
        return (mean - margin, mean + margin)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        ci_low, ci_high = self.confidence_interval()
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_s": self.total_time,
            "mean_time_s": self.mean_time,
            "mean_time_ms": self.mean_time * 1000,
            "ops_per_second": self.ops_per_second,
            "std_dev_s": self.std_dev,
            "min_time_s": self.min_time,
            "max_time_s": self.max_time,
            "median_time_s": self.median_time,
            "p50_s": self.p50,
            "p95_s": self.p95,
            "p99_s": self.p99,
            "ci_95_low_s": ci_low,
            "ci_95_high_s": ci_high,
        }

    def __repr__(self) -> str:
        ci_low, ci_high = self.confidence_interval()
        return (
            f"{self.name}: {self.mean_time*1000:.3f}ms/op "
            f"({self.ops_per_second:.1f} ops/s, "
            f"p50={self.p50*1000:.3f}ms, p95={self.p95*1000:.3f}ms, "
            f"CI=[{ci_low*1000:.3f}, {ci_high*1000:.3f}]ms)"
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

    def bench_memory_read_8bytes(self, iterations: int = 50000) -> BenchmarkResult:
        """Benchmark 8-byte memory read operations."""
        engine = self.setup_engine()
        engine.write_memory(0x1000, b"\xAA" * 4096)

        def do_read():
            return engine.read_memory(0x1000, 8)

        return benchmark(do_read, "rust_memory_read_8bytes", iterations)

    def bench_memory_write_8bytes(self, iterations: int = 50000) -> BenchmarkResult:
        """Benchmark 8-byte memory write operations."""
        engine = self.setup_engine()

        counter = [0]
        def do_write():
            data = counter[0].to_bytes(8, 'little')
            engine.write_memory(0x1000, data)
            counter[0] += 1

        return benchmark(do_write, "rust_memory_write_8bytes", iterations)

    def bench_memory_read_4kb(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark bulk 4KB memory read operations."""
        engine = self.setup_engine()
        engine.write_memory(0x1000, b"\xAA" * 4096)

        def do_read():
            return engine.read_memory(0x1000, 4096)

        return benchmark(do_read, "rust_memory_read_4kb", iterations)

    def bench_memory_write_4kb(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark bulk 4KB memory write operations."""
        engine = self.setup_engine()
        data = b"\xBB" * 4096

        def do_write():
            engine.write_memory(0x1000, data)

        return benchmark(do_write, "rust_memory_write_4kb", iterations)

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

    def bench_irsb_execute(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark simple IRSB execution (mov rax, 0x1234; ret)."""
        engine = self.setup_engine()
        # mov rax, 0x1234; ret
        code = bytes([0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, 0xc3])
        engine.map_memory_data(0x401000, code)
        engine.pc = 0x401000

        def do_execute():
            return engine.execute_code(code)

        return benchmark(do_execute, "rust_irsb_execute", iterations)

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        if not self.available:
            return []

        results = []
        print("\n=== Rust VEX Engine Benchmarks ===\n")

        benchmarks = [
            self.bench_register_read,
            self.bench_register_write,
            self.bench_memory_read_8bytes,
            self.bench_memory_write_8bytes,
            self.bench_memory_read_4kb,
            self.bench_memory_write_4kb,
            self.bench_bulk_registers,
            self.bench_fork,
            self.bench_fork_modify,
            self.bench_many_forks,
            self.bench_irsb_execute,
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
            import claripy
            self.angr = angr
            self.claripy = claripy
            self.available = True
        except ImportError:
            print("angr not available for Python benchmarks")
            self.available = False

    def setup_state(self) -> "angr.SimState":
        """Create a simple state for benchmarking."""
        # Create a minimal project with blob backend
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

    def bench_memory_read_8bytes(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark 8-byte memory read in Python."""
        state = self.setup_state()
        # Write some data first
        state.memory.store(0x1000, self.claripy.BVV(0xAAAAAAAAAAAAAAAA, 64))

        def do_read():
            return state.memory.load(0x1000, 8)

        return benchmark(do_read, "python_memory_read_8bytes", iterations)

    def bench_memory_write_8bytes(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark 8-byte memory write in Python."""
        state = self.setup_state()

        counter = [0]
        def do_write():
            state.memory.store(0x1000, self.claripy.BVV(counter[0], 64))
            counter[0] += 1

        return benchmark(do_write, "python_memory_write_8bytes", iterations)

    def bench_memory_read_4kb(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark 4KB memory read in Python."""
        state = self.setup_state()
        # Write 4KB of data
        for i in range(0, 4096, 8):
            state.memory.store(0x1000 + i, self.claripy.BVV(0xAA, 8))

        def do_read():
            return state.memory.load(0x1000, 4096)

        return benchmark(do_read, "python_memory_read_4kb", iterations)

    def bench_memory_write_4kb(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark 4KB memory write in Python."""
        state = self.setup_state()
        data = self.claripy.BVV(b"\xBB" * 4096)

        def do_write():
            state.memory.store(0x1000, data)

        return benchmark(do_write, "python_memory_write_4kb", iterations)

    def bench_bv_add(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark BV addition in claripy."""
        a = self.claripy.BVV(12345, 64)
        b = self.claripy.BVV(67890, 64)

        def do_add():
            return a + b

        return benchmark(do_add, "python_bv_add", iterations)

    def bench_bv_mul(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark BV multiplication in claripy."""
        a = self.claripy.BVV(12345, 64)
        b = self.claripy.BVV(67890, 64)

        def do_mul():
            return a * b

        return benchmark(do_mul, "python_bv_mul", iterations)

    def bench_bv_and(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark BV bitwise AND in claripy."""
        a = self.claripy.BVV(0xFF00FF00FF00FF00, 64)
        b = self.claripy.BVV(0x00FF00FF00FF00FF, 64)

        def do_and():
            return a & b

        return benchmark(do_and, "python_bv_and", iterations)

    def bench_bv_shift(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark BV shift operations in claripy."""
        a = self.claripy.BVV(0x12345678DEADBEEF, 64)

        def do_shift():
            return a >> 8

        return benchmark(do_shift, "python_bv_shift", iterations)

    def bench_bv_extract(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark BV extract operations in claripy."""
        a = self.claripy.BVV(0x12345678DEADBEEF, 64)

        def do_extract():
            return a[31:0]

        return benchmark(do_extract, "python_bv_extract", iterations)

    def bench_bv_concat(self, iterations: int = 100000) -> BenchmarkResult:
        """Benchmark BV concatenation in claripy."""
        a = self.claripy.BVV(0x12345678, 32)
        b = self.claripy.BVV(0xDEADBEEF, 32)

        def do_concat():
            return self.claripy.Concat(a, b)

        return benchmark(do_concat, "python_bv_concat", iterations)

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
            self.bench_memory_read_8bytes,
            self.bench_memory_write_8bytes,
            self.bench_memory_read_4kb,
            self.bench_memory_write_4kb,
            self.bench_bv_add,
            self.bench_bv_mul,
            self.bench_bv_and,
            self.bench_bv_shift,
            self.bench_bv_extract,
            self.bench_bv_concat,
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
        "memory_read_8bytes": ("rust_memory_read_8bytes", "python_memory_read_8bytes"),
        "memory_write_8bytes": ("rust_memory_write_8bytes", "python_memory_write_8bytes"),
        "memory_read_4kb": ("rust_memory_read_4kb", "python_memory_read_4kb"),
        "memory_write_4kb": ("rust_memory_write_4kb", "python_memory_write_4kb"),
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
        "| Benchmark | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Std Dev (ms) | Ops/s |",
        "|-----------|-----------|----------|----------|----------|--------------|-------|",
    ]

    for r in rust_results:
        lines.append(
            f"| {r.name} | {r.mean_time*1000:.3f} | {r.p50*1000:.3f} | "
            f"{r.p95*1000:.3f} | {r.p99*1000:.3f} | {r.std_dev*1000:.3f} | {r.ops_per_second:.1f} |"
        )

    lines.extend([
        "",
        "### Python Engine",
        "",
        "| Benchmark | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Std Dev (ms) | Ops/s |",
        "|-----------|-----------|----------|----------|----------|--------------|-------|",
    ])

    for r in python_results:
        lines.append(
            f"| {r.name} | {r.mean_time*1000:.3f} | {r.p50*1000:.3f} | "
            f"{r.p95*1000:.3f} | {r.p99*1000:.3f} | {r.std_dev*1000:.3f} | {r.ops_per_second:.1f} |"
        )

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
        "Memory Read 8B": ("rust_memory_read_8bytes", "python_memory_read_8bytes"),
        "Memory Write 8B": ("rust_memory_write_8bytes", "python_memory_write_8bytes"),
        "Memory Read 4KB": ("rust_memory_read_4kb", "python_memory_read_4kb"),
        "Memory Write 4KB": ("rust_memory_write_4kb", "python_memory_write_4kb"),
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
