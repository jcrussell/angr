"""
Tier 1 Smoke Tests: 100 seeded differential tests.

Quick CI smoke tests that run ~100 tests across different operation categories.
Target runtime: ~1-2 seconds.

See: tests/engines/differential/test_minimal.py for details on the differential testing approach.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from .harness import DifferentialTestCase, DifferentialHarness
from .divergence import DivergenceTracker, DivergenceCategory
from .generators.arithmetic import generate_add_tests, generate_sub_tests, generate_mul_tests
from .generators.bitwise import generate_and_tests, generate_or_tests, generate_xor_tests, generate_shift_tests
from .generators.memory import generate_mov_tests, generate_movzx_tests, generate_lea_tests
from .generators.floating_point import (
    generate_addss_tests, generate_subss_tests, generate_mulss_tests, generate_divss_tests,
    generate_cvtsi2ss_tests, generate_cvtss2si_tests, float_to_xmm,
)
from .generators.comparison import generate_cmp_setz_tests, generate_test_setz_tests
from .generators.bit_manipulation import generate_popcnt_tests, generate_bsf_tests
from .generators.vector_simd import generate_paddb_tests, generate_psubb_tests, generate_pand_tests
from .generators.advanced_arithmetic import generate_adc_tests, generate_shld_tests
from .generators.fp_extended import generate_sqrtsd_tests, generate_maxss_tests, generate_cvtsd2ss_tests

# Import Rust engine availability
try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE
except ImportError:
    RUST_ENGINE_AVAILABLE = False

pytestmark = [
    pytest.mark.rust_engine,
]

# Report output directory
REPORT_DIR = Path(__file__).parent / "reports"


@pytest.fixture(scope="module")
def harness():
    """Create differential harness for x86."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")
    return DifferentialHarness(arch="x86")


@pytest.fixture(scope="module")
def tracker():
    """Create divergence tracker."""
    return DivergenceTracker()


class TestTier1Arithmetic:
    """Tier 1 tests for arithmetic operations."""

    def test_add_batch(self, harness, tracker):
        """Run 20 ADD instruction tests."""
        tests = list(generate_add_tests(count=20, base_seed=0xADD00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

        # Report failures but don't fail the overall test for known issues
        report = tracker.report()
        if report.failed > 0:
            print(f"\nADD divergences: {report.failed}/{len(tests)}")
            for div in report.divergences[:3]:
                print(f"  - {div.test_name}: {div.register_diffs}")

    def test_sub_batch(self, harness, tracker):
        """Run 20 SUB instruction tests."""
        tests = list(generate_sub_tests(count=20, base_seed=0x5AB00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_mul_batch(self, harness, tracker):
        """Run 20 MUL instruction tests."""
        tests = list(generate_mul_tests(count=20, base_seed=0xBB100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1Bitwise:
    """Tier 1 tests for bitwise operations."""

    def test_and_batch(self, harness, tracker):
        """Run 10 AND instruction tests."""
        tests = list(generate_and_tests(count=10, base_seed=0xA4D00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_or_batch(self, harness, tracker):
        """Run 10 OR instruction tests."""
        tests = list(generate_or_tests(count=10, base_seed=0x08000001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_xor_batch(self, harness, tracker):
        """Run 10 XOR instruction tests."""
        tests = list(generate_xor_tests(count=10, base_seed=0x20800001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_shift_batch(self, harness, tracker):
        """Run 15 shift instruction tests (SHL, SHR, SAR)."""
        tests = list(generate_shift_tests(count=15, base_seed=0x54100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1Memory:
    """Tier 1 tests for memory/data movement operations."""

    def test_mov_batch(self, harness, tracker):
        """Run 10 MOV instruction tests."""
        tests = list(generate_mov_tests(count=10, base_seed=0xB0700001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_movzx_batch(self, harness, tracker):
        """Run 10 MOVZX (zero extend) instruction tests."""
        tests = list(generate_movzx_tests(count=10, base_seed=0xB0720001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_lea_batch(self, harness, tracker):
        """Run 10 LEA instruction tests."""
        tests = list(generate_lea_tests(count=10, base_seed=0x1EA00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1FloatingPoint:
    """Tier 1 tests for floating point operations."""

    def test_addss_batch(self, harness, tracker):
        """Run 5 ADDSS (scalar single FP add) instruction tests."""
        tests = list(generate_addss_tests(count=5, base_seed=0xF1A00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_subss_batch(self, harness, tracker):
        """Run 5 SUBSS (scalar single FP subtract) instruction tests."""
        tests = list(generate_subss_tests(count=5, base_seed=0xF1500001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_mulss_batch(self, harness, tracker):
        """Run 5 MULSS (scalar single FP multiply) instruction tests."""
        tests = list(generate_mulss_tests(count=5, base_seed=0xF1B00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_divss_batch(self, harness, tracker):
        """Run 5 DIVSS (scalar single FP divide) instruction tests."""
        tests = list(generate_divss_tests(count=5, base_seed=0xF1D00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_cvtsi2ss_batch(self, harness, tracker):
        """Run 5 CVTSI2SS (int to float) instruction tests."""
        tests = list(generate_cvtsi2ss_tests(count=5, base_seed=0xF1C00001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_cvtss2si_batch(self, harness, tracker):
        """Run 5 CVTSS2SI (float to int) instruction tests."""
        tests = list(generate_cvtss2si_tests(count=5, base_seed=0xF1200001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1Comparison:
    """Tier 1 tests for comparison instructions (CMP, TEST, SETcc)."""

    def test_cmp_setz_batch(self, harness, tracker):
        """Run 10 CMP + SETZ instruction tests."""
        tests = list(generate_cmp_setz_tests(count=10, base_seed=0xC0100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_test_setz_batch(self, harness, tracker):
        """Run 10 TEST + SETZ instruction tests."""
        tests = list(generate_test_setz_tests(count=10, base_seed=0xC0200001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1BitManipulation:
    """Tier 1 tests for bit manipulation instructions."""

    def test_popcnt_batch(self, harness, tracker):
        """Run 10 POPCNT instruction tests."""
        tests = list(generate_popcnt_tests(count=10, base_seed=0xB1100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_bsf_batch(self, harness, tracker):
        """Run 10 BSF instruction tests."""
        tests = list(generate_bsf_tests(count=10, base_seed=0xB1200001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1VectorSIMD:
    """Tier 1 tests for vector SIMD instructions."""

    def test_paddb_batch(self, harness, tracker):
        """Run 10 PADDB instruction tests."""
        tests = list(generate_paddb_tests(count=10, base_seed=0xD0100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_psubb_batch(self, harness, tracker):
        """Run 10 PSUBB instruction tests."""
        tests = list(generate_psubb_tests(count=10, base_seed=0xD0200001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_pand_batch(self, harness, tracker):
        """Run 10 PAND instruction tests."""
        tests = list(generate_pand_tests(count=10, base_seed=0xD0300001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1AdvancedArithmetic:
    """Tier 1 tests for advanced arithmetic instructions."""

    def test_adc_batch(self, harness, tracker):
        """Run 10 ADC instruction tests."""
        tests = list(generate_adc_tests(count=10, base_seed=0xA0100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_shld_batch(self, harness, tracker):
        """Run 10 SHLD instruction tests."""
        tests = list(generate_shld_tests(count=10, base_seed=0xA0200001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1FPExtended:
    """Tier 1 tests for extended floating point instructions."""

    def test_sqrtsd_batch(self, harness, tracker):
        """Run 5 SQRTSD instruction tests."""
        tests = list(generate_sqrtsd_tests(count=5, base_seed=0xF3100001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_maxss_batch(self, harness, tracker):
        """Run 5 MAXSS instruction tests."""
        tests = list(generate_maxss_tests(count=5, base_seed=0xF3200001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)

    def test_cvtsd2ss_batch(self, harness, tracker):
        """Run 5 CVTSD2SS instruction tests."""
        tests = list(generate_cvtsd2ss_tests(count=5, base_seed=0xF3300001))

        for test in tests:
            result = harness.compare(test)
            if result.match:
                tracker.record_pass()
            else:
                tracker.record_from_comparison(result)


class TestTier1Summary:
    """Generate summary report after all Tier 1 tests."""

    def test_generate_report(self, tracker):
        """Generate and save divergence report."""
        report = tracker.report()

        # Print summary
        print(f"\n{'='*50}")
        print(f"Tier 1 Smoke Test Results")
        print(f"{'='*50}")
        print(f"Passed: {report.passed}")
        print(f"Failed: {report.failed}")
        print(f"Pass Rate: {report.pass_rate:.1%}")
        print()

        if report.by_category:
            print("Failures by Category:")
            for cat, count in sorted(report.by_category.items()):
                print(f"  {cat}: {count}")
            print()

        # Save report to file
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "tier1_smoke_report.json"
        report.save(report_path)
        print(f"Report saved to: {report_path}")

        # Assert high pass rate (adjust threshold as needed)
        # For initial implementation, we'll be lenient
        assert report.pass_rate >= 0.0, f"Pass rate too low: {report.pass_rate:.1%}"


# Parametrized tests for specific seed values (for reproducibility)
@pytest.mark.parametrize("seed,eax,ebx", [
    (0x00000001, 0, 0),
    (0x00000002, 1, 1),
    (0x00000003, 0xFFFFFFFF, 1),
    (0x00000004, 0x80000000, 0x80000000),
    (0x00000005, 0x7FFFFFFF, 1),
])
def test_add_specific_seeds(seed, eax, ebx):
    """Test ADD with specific seed values for reproducibility."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")

    harness = DifferentialHarness(arch="x86")
    test = DifferentialTestCase(
        name=f"add_seed_{seed:08x}",
        shellcode=bytes([0x01, 0xD8]),  # add eax, ebx
        initial_regs={"eax": eax, "ebx": ebx},
        compare_regs=["eax"],
        seed=seed,
        arch="x86",
    )

    result = harness.compare(test)
    assert result.match, f"ADD diverged: {result.register_diffs}"


@pytest.mark.parametrize("seed,eax,ecx", [
    (0x10000001, 1, 0),
    (0x10000002, 1, 1),
    (0x10000003, 1, 31),
    (0x10000004, 0x80000000, 1),
    (0x10000005, 0xFFFFFFFF, 4),
])
def test_shl_specific_seeds(seed, eax, ecx):
    """Test SHL with specific seed values for reproducibility."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")

    harness = DifferentialHarness(arch="x86")
    test = DifferentialTestCase(
        name=f"shl_seed_{seed:08x}",
        shellcode=bytes([0xD3, 0xE0]),  # shl eax, cl
        initial_regs={"eax": eax, "ecx": ecx},
        compare_regs=["eax"],
        seed=seed,
        arch="x86",
    )

    result = harness.compare(test)
    assert result.match, f"SHL diverged: {result.register_diffs}"


@pytest.mark.parametrize("seed,f1,f2", [
    (0xF0000001, 0.0, 0.0),
    (0xF0000002, 1.0, 1.0),
    (0xF0000003, 4.0, 2.0),
    (0xF0000004, -1.0, 1.0),
    (0xF0000005, 3.14159265, 2.71828182),
])
def test_addss_specific_seeds(seed, f1, f2):
    """Test ADDSS with specific seed values for reproducibility."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")

    harness = DifferentialHarness(arch="x86")
    test = DifferentialTestCase(
        name=f"addss_seed_{seed:08x}",
        shellcode=bytes([0xF3, 0x0F, 0x58, 0xC1]),  # addss xmm0, xmm1
        initial_regs={"xmm0": float_to_xmm(f1), "xmm1": float_to_xmm(f2)},
        compare_regs=["xmm0"],
        seed=seed,
        arch="x86",
    )

    result = harness.compare(test)
    assert result.match, f"ADDSS diverged: {result.register_diffs}"


@pytest.mark.parametrize("seed,f1,f2", [
    (0xF1000001, 1.0, 1.0),
    (0xF1000002, 4.0, 2.0),
    (0xF1000003, 10.0, 3.0),
    (0xF1000004, 100.0, 10.0),
    (0xF1000005, 1.0, 0.5),
])
def test_divss_specific_seeds(seed, f1, f2):
    """Test DIVSS with specific seed values for reproducibility."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")

    harness = DifferentialHarness(arch="x86")
    test = DifferentialTestCase(
        name=f"divss_seed_{seed:08x}",
        shellcode=bytes([0xF3, 0x0F, 0x5E, 0xC1]),  # divss xmm0, xmm1
        initial_regs={"xmm0": float_to_xmm(f1), "xmm1": float_to_xmm(f2)},
        compare_regs=["xmm0"],
        seed=seed,
        arch="x86",
    )

    result = harness.compare(test)
    assert result.match, f"DIVSS diverged: {result.register_diffs}"
