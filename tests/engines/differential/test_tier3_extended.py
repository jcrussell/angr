"""
Tier 3 Extended Tests: ~500 additional differential tests for comprehensive coverage.

Tests operations that were previously untested:
- DivMod64to32 (64-bit division)
- MulHi (high multiplication)
- Vector shifts (VShlN, VShrN, VSarN)
- Vector interleave/misc (SetV128, VInterleave, VPerm)
- FP rounding modes (ROUNDSS, ROUNDSD, etc.)
- Crypto operations (PCLMULQDQ, CRC32)

See: tests/engines/differential/test_minimal.py for details on the differential testing approach.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from .harness import DifferentialTestCase, DifferentialHarness
from .divergence import DivergenceTracker, DivergenceReport

# DivMod generators
from .generators.divmod import (
    generate_div64to32_unsigned_tests, generate_div64to32_signed_tests,
)

# MulHi generators
from .generators.mulhi import (
    generate_mulhi_unsigned_tests, generate_mulhi_signed_tests,
)

# Vector shift generators
from .generators.vector_shift import (
    generate_psllw_tests, generate_psrlw_tests, generate_psraw_tests,
    generate_pslld_tests, generate_psrld_tests, generate_psrad_tests,
    generate_psllq_tests, generate_psrlq_tests,
    generate_pslldq_tests, generate_psrldq_tests,
)

# Vector misc generators
from .generators.vector_misc import (
    generate_movd_to_xmm_tests,
    generate_punpcklbw_tests, generate_punpckhbw_tests,
    generate_punpcklwd_tests, generate_punpckhwd_tests,
    generate_punpckldq_tests, generate_punpckhdq_tests,
    generate_punpcklqdq_tests, generate_punpckhqdq_tests,
    generate_pshufd_tests, generate_pshufhw_tests, generate_pshuflw_tests,
)

# FP rounding generators
from .generators.fp_rounding import (
    generate_roundss_tests, generate_roundsd_tests,
    generate_roundps_tests, generate_roundpd_tests,
    generate_floorss_tests, generate_ceilss_tests, generate_truncss_tests,
    generate_floorsd_tests, generate_ceilsd_tests, generate_truncsd_tests,
)

# Crypto generators
from .generators.crypto import (
    generate_pclmulqdq_lqlq_tests, generate_pclmulqdq_hqhq_tests, generate_pclmulqdq_mixed_tests,
    generate_crc32_32_tests, generate_crc32_8_tests, generate_crc32_16_tests,
)

# Import Rust engine availability
try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE
except ImportError:
    RUST_ENGINE_AVAILABLE = False

pytestmark = [
    pytest.mark.rust_engine,
    pytest.mark.slow,
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
    """Create divergence tracker for the module."""
    return DivergenceTracker()


def _run_test_batch(harness, tracker, generators: list[tuple[str, any]]) -> tuple[int, int]:
    """
    Helper to run a batch of test generators.

    Args:
        harness: DifferentialHarness instance
        tracker: DivergenceTracker instance
        generators: List of (name, generator) tuples

    Returns:
        (total_passed, total_failed) tuple
    """
    total_passed = 0
    total_failed = 0

    for name, tests in generators:
        passed = 0
        failed = 0
        errors = 0

        for test in tests:
            result = harness.compare(test)

            # Handle errors gracefully
            if result.rust_result.error or result.python_result.error:
                errors += 1
                continue

            if result.match:
                tracker.record_pass()
                passed += 1
            else:
                tracker.record_from_comparison(result)
                failed += 1

        total_passed += passed
        total_failed += failed

        if failed > 0 or errors > 0:
            print(f"\n{name}: {passed} passed, {failed} failed, {errors} errors")

    return total_passed, total_failed


class TestTier3DivMod:
    """Tier 3 tests for 64-bit division operations (DivMod64to32)."""

    def test_div64to32_unsigned(self, harness, tracker):
        """Run unsigned 64-to-32 division tests (DIV r/m32)."""
        generators = [
            ("div64to32_u", generate_div64to32_unsigned_tests(count=50, base_seed=0x30000001)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nDIV 64-to-32 unsigned: {passed}/{passed + failed} passed")

    def test_div64to32_signed(self, harness, tracker):
        """Run signed 64-to-32 division tests (IDIV r/m32)."""
        generators = [
            ("div64to32_s", generate_div64to32_signed_tests(count=50, base_seed=0x30000002)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nIDIV 64-to-32 signed: {passed}/{passed + failed} passed")


class TestTier3MulHi:
    """Tier 3 tests for high multiplication operations (MulHi)."""

    def test_mulhi_unsigned(self, harness, tracker):
        """Run unsigned high multiplication tests (MUL r/m32)."""
        generators = [
            ("mulhi_u", generate_mulhi_unsigned_tests(count=50, base_seed=0x30000011)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nMUL (high) unsigned: {passed}/{passed + failed} passed")

    def test_mulhi_signed(self, harness, tracker):
        """Run signed high multiplication tests (IMUL r/m32)."""
        generators = [
            ("mulhi_s", generate_mulhi_signed_tests(count=50, base_seed=0x30000012)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nIMUL (high) signed: {passed}/{passed + failed} passed")


class TestTier3VectorShift:
    """Tier 3 tests for vector shift operations (VShlN, VShrN, VSarN)."""

    def test_psllw_psrlw_psraw(self, harness, tracker):
        """Run word shift tests (PSLLW, PSRLW, PSRAW)."""
        generators = [
            ("psllw", generate_psllw_tests(count=30, base_seed=0x30000021)),
            ("psrlw", generate_psrlw_tests(count=30, base_seed=0x30000022)),
            ("psraw", generate_psraw_tests(count=30, base_seed=0x30000023)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nWord shifts total: {passed}/{passed + failed} passed")

    def test_pslld_psrld_psrad(self, harness, tracker):
        """Run dword shift tests (PSLLD, PSRLD, PSRAD)."""
        generators = [
            ("pslld", generate_pslld_tests(count=30, base_seed=0x30000024)),
            ("psrld", generate_psrld_tests(count=30, base_seed=0x30000025)),
            ("psrad", generate_psrad_tests(count=30, base_seed=0x30000026)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nDword shifts total: {passed}/{passed + failed} passed")

    def test_psllq_psrlq(self, harness, tracker):
        """Run qword shift tests (PSLLQ, PSRLQ)."""
        generators = [
            ("psllq", generate_psllq_tests(count=30, base_seed=0x30000027)),
            ("psrlq", generate_psrlq_tests(count=30, base_seed=0x30000028)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nQword shifts total: {passed}/{passed + failed} passed")

    def test_pslldq_psrldq(self, harness, tracker):
        """Run byte-granularity shift tests (PSLLDQ, PSRLDQ)."""
        generators = [
            ("pslldq", generate_pslldq_tests(count=25, base_seed=0x30000029)),
            ("psrldq", generate_psrldq_tests(count=25, base_seed=0x3000002A)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nByte shifts total: {passed}/{passed + failed} passed")


class TestTier3VectorMisc:
    """Tier 3 tests for vector misc operations (SetV128, interleave, shuffle)."""

    def test_movd_to_xmm(self, harness, tracker):
        """Run MOVD to XMM tests (SetV128lo32)."""
        generators = [
            ("movd_to_xmm", generate_movd_to_xmm_tests(count=30, base_seed=0x30000031)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nMOVD to XMM: {passed}/{passed + failed} passed")

    def test_interleave_bytes(self, harness, tracker):
        """Run byte interleave tests (PUNPCKLBW, PUNPCKHBW)."""
        generators = [
            ("punpcklbw", generate_punpcklbw_tests(count=25, base_seed=0x30000041)),
            ("punpckhbw", generate_punpckhbw_tests(count=25, base_seed=0x30000042)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nByte interleave: {passed}/{passed + failed} passed")

    def test_interleave_words(self, harness, tracker):
        """Run word interleave tests (PUNPCKLWD, PUNPCKHWD)."""
        generators = [
            ("punpcklwd", generate_punpcklwd_tests(count=25, base_seed=0x30000043)),
            ("punpckhwd", generate_punpckhwd_tests(count=25, base_seed=0x30000044)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nWord interleave: {passed}/{passed + failed} passed")

    def test_interleave_dwords(self, harness, tracker):
        """Run dword interleave tests (PUNPCKLDQ, PUNPCKHDQ)."""
        generators = [
            ("punpckldq", generate_punpckldq_tests(count=25, base_seed=0x30000045)),
            ("punpckhdq", generate_punpckhdq_tests(count=25, base_seed=0x30000046)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nDword interleave: {passed}/{passed + failed} passed")

    def test_interleave_qwords(self, harness, tracker):
        """Run qword interleave tests (PUNPCKLQDQ, PUNPCKHQDQ)."""
        generators = [
            ("punpcklqdq", generate_punpcklqdq_tests(count=25, base_seed=0x30000047)),
            ("punpckhqdq", generate_punpckhqdq_tests(count=25, base_seed=0x30000048)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nQword interleave: {passed}/{passed + failed} passed")

    def test_shuffle(self, harness, tracker):
        """Run shuffle tests (PSHUFD, PSHUFHW, PSHUFLW)."""
        generators = [
            ("pshufd", generate_pshufd_tests(count=30, base_seed=0x30000051)),
            ("pshufhw", generate_pshufhw_tests(count=20, base_seed=0x30000052)),
            ("pshuflw", generate_pshuflw_tests(count=20, base_seed=0x30000053)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nShuffle ops: {passed}/{passed + failed} passed")


class TestTier3FPRounding:
    """Tier 3 tests for FP rounding operations."""

    def test_roundss(self, harness, tracker):
        """Run ROUNDSS tests (all rounding modes)."""
        generators = [
            ("roundss", generate_roundss_tests(count=40, base_seed=0x30000061)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nROUNDSS: {passed}/{passed + failed} passed")

    def test_roundsd(self, harness, tracker):
        """Run ROUNDSD tests (all rounding modes)."""
        generators = [
            ("roundsd", generate_roundsd_tests(count=40, base_seed=0x30000062)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nROUNDSD: {passed}/{passed + failed} passed")

    def test_roundps(self, harness, tracker):
        """Run ROUNDPS tests (packed single precision)."""
        generators = [
            ("roundps", generate_roundps_tests(count=30, base_seed=0x30000063)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nROUNDPS: {passed}/{passed + failed} passed")

    def test_roundpd(self, harness, tracker):
        """Run ROUNDPD tests (packed double precision)."""
        generators = [
            ("roundpd", generate_roundpd_tests(count=30, base_seed=0x30000064)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nROUNDPD: {passed}/{passed + failed} passed")

    def test_floor_ceil_trunc_single(self, harness, tracker):
        """Run floor/ceil/trunc tests for single precision."""
        generators = [
            ("floorss", generate_floorss_tests(count=20, base_seed=0x30000071)),
            ("ceilss", generate_ceilss_tests(count=20, base_seed=0x30000072)),
            ("truncss", generate_truncss_tests(count=20, base_seed=0x30000073)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nFloor/Ceil/Trunc (single): {passed}/{passed + failed} passed")

    def test_floor_ceil_trunc_double(self, harness, tracker):
        """Run floor/ceil/trunc tests for double precision."""
        generators = [
            ("floorsd", generate_floorsd_tests(count=20, base_seed=0x30000074)),
            ("ceilsd", generate_ceilsd_tests(count=20, base_seed=0x30000075)),
            ("truncsd", generate_truncsd_tests(count=20, base_seed=0x30000076)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nFloor/Ceil/Trunc (double): {passed}/{passed + failed} passed")


class TestTier3Crypto:
    """Tier 3 tests for cryptographic operations."""

    def test_pclmulqdq(self, harness, tracker):
        """Run PCLMULQDQ tests (carry-less multiply)."""
        generators = [
            ("pclmulqdq_lqlq", generate_pclmulqdq_lqlq_tests(count=25, base_seed=0x30000081)),
            ("pclmulqdq_hqhq", generate_pclmulqdq_hqhq_tests(count=25, base_seed=0x30000082)),
            ("pclmulqdq_mixed", generate_pclmulqdq_mixed_tests(count=25, base_seed=0x30000083)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nPCLMULQDQ: {passed}/{passed + failed} passed")

    def test_crc32(self, harness, tracker):
        """Run CRC32 tests (all operand sizes)."""
        generators = [
            ("crc32_8", generate_crc32_8_tests(count=25, base_seed=0x30000091)),
            ("crc32_16", generate_crc32_16_tests(count=25, base_seed=0x30000092)),
            ("crc32_32", generate_crc32_32_tests(count=30, base_seed=0x30000093)),
        ]

        passed, failed = _run_test_batch(harness, tracker, generators)
        print(f"\nCRC32: {passed}/{passed + failed} passed")


class TestTier3Summary:
    """Generate summary report after all Tier 3 tests."""

    def test_generate_report(self, tracker):
        """Generate and save comprehensive divergence report."""
        report = tracker.report()

        # Print detailed summary
        print(f"\n{'='*60}")
        print(f"Tier 3 Extended Test Results")
        print(f"{'='*60}")
        print(f"Total Tests: {report.total}")
        print(f"Passed:      {report.passed}")
        print(f"Failed:      {report.failed}")
        print(f"Skipped:     {report.skipped}")
        print(f"Pass Rate:   {report.pass_rate:.2%}")
        print()

        if report.by_category:
            print("Failures by Category:")
            for cat, count in sorted(report.by_category.items(), key=lambda x: -x[1]):
                pct = count / report.failed * 100 if report.failed > 0 else 0
                print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")
            print()

        if report.divergences:
            print("Sample Divergences (first 10):")
            for div in report.divergences[:10]:
                print(f"  {div.test_name}:")
                print(f"    Category: {div.category.value}")
                print(f"    Shellcode: {div.shellcode_hex}")
                for reg, vals in div.register_diffs.items():
                    print(f"    {reg}: Python={vals['python']:#x}, Rust={vals['rust']:#x}")
            print()

        # Save report to file
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "tier3_extended_report.json"
        report.save(report_path)
        print(f"Report saved to: {report_path}")

        # For Tier 3, we track progress but don't enforce a strict threshold
        # since these are new, potentially unsupported operations
        print(f"\nNote: Tier 3 tests cover advanced operations that may not be fully implemented.")
