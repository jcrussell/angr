"""
Extended floating point instruction generators for differential testing.

Generates test cases for additional FP operations:
- SQRTSD: Scalar double sqrt
- CVTSD2SS, CVTSS2SD: Double/single conversions
- CVTSI2SD, CVTSD2SI, CVTTSD2SI: Int/double conversions
- UCOMISD: Unordered compare double
- MAXSS, MINSS, MAXSD, MINSD: Min/max operations
"""
from __future__ import annotations

import random
import struct
from typing import Iterator

from ..harness import DifferentialTestCase


def _make_seed(base: int, index: int) -> int:
    """Create deterministic seed from base seed and index."""
    return (base * 1000003 + index) & 0xFFFFFFFF


def _random_value(rng: random.Random, width: int = 32) -> int:
    """Generate random value for given bit width."""
    return rng.randint(0, (1 << width) - 1)


def float_to_xmm(f: float) -> int:
    """
    Convert a Python float to XMM register value (128-bit).

    The float is stored in the low 32 bits as IEEE 754 single precision.
    Upper 96 bits are zeroed.
    """
    packed = struct.pack('<f', f)
    low32 = struct.unpack('<I', packed)[0]
    return low32


def double_to_xmm(d: float) -> int:
    """
    Convert a Python float to XMM register value as double precision.

    The double is stored in the low 64 bits as IEEE 754 double precision.
    Upper 64 bits are zeroed.
    """
    packed = struct.pack('<d', d)
    low64 = struct.unpack('<Q', packed)[0]
    return low64


def _fp_edge_values_single() -> list[float]:
    """Generate edge case float values for single precision."""
    return [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        0.25,
        0.125,
        10.0,
        100.0,
        1000.0,
        0.1,
        0.01,
        3.14159265,
        2.71828182,
        1.41421356,  # sqrt(2)
        1e10,
        -1e10,
        1e-10,
        -1e-10,
        1e38,
        -1e38,
        1e-38,
    ]


def _fp_edge_values_double() -> list[float]:
    """Generate edge case double values."""
    return [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        0.25,
        0.125,
        10.0,
        100.0,
        1000.0,
        0.1,
        0.01,
        3.14159265358979323846,
        2.71828182845904523536,
        1.41421356237309504880,  # sqrt(2)
        1e100,
        -1e100,
        1e-100,
        -1e-100,
        1e308,
        -1e308,
        1e-308,
    ]


def _fp_random_float(rng: random.Random) -> float:
    """Generate a random float value for testing."""
    choice = rng.randint(0, 4)
    if choice == 0:
        return float(rng.randint(-1000, 1000))
    elif choice == 1:
        return rng.uniform(-10.0, 10.0)
    elif choice == 2:
        return rng.uniform(-1e6, 1e6)
    elif choice == 3:
        return rng.uniform(-1e-6, 1e-6)
    else:
        exp = rng.randint(-20, 20)
        sign = rng.choice([-1, 1])
        return sign * (2.0 ** exp)


def generate_sqrtsd_tests(
    count: int = 100,
    base_seed: int = 0xF3000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SQRTSD (scalar double FP square root) instruction test cases.

    x86 SSE2: sqrtsd xmm0, xmm1 -> xmm0[0:63] = sqrt(xmm1[0:63])

    Encoding: F2 0F 51 C1 (sqrtsd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SQRTSD instructions
    """
    # f2 0f 51 c1 = sqrtsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x51, 0xC1])

    # Only non-negative values for sqrt
    edge_vals = [v for v in _fp_edge_values_double() if v >= 0.0]
    edge_vals.extend([4.0, 9.0, 16.0, 25.0, 100.0, 2.0, 0.25, 0.01])

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            d = edge_vals[i]
        else:
            d = abs(_fp_random_float(rng))
            if d == 0.0:
                d = 1.0

        xmm1 = double_to_xmm(d)

        yield DifferentialTestCase(
            name=f"sqrtsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_cvtsd2ss_tests(
    count: int = 100,
    base_seed: int = 0xF3000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTSD2SS (convert double to single) instruction test cases.

    x86 SSE2: cvtsd2ss xmm0, xmm1 -> xmm0[0:31] = (float)xmm1[0:63]

    Encoding: F2 0F 5A C1 (cvtsd2ss xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTSD2SS instructions
    """
    # f2 0f 5a c1 = cvtsd2ss xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x5A, 0xC1])

    edge_vals = _fp_edge_values_double()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            d = edge_vals[i]
        else:
            d = _fp_random_float(rng)

        xmm1 = double_to_xmm(d)

        yield DifferentialTestCase(
            name=f"cvtsd2ss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_cvtss2sd_tests(
    count: int = 100,
    base_seed: int = 0xF3000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTSS2SD (convert single to double) instruction test cases.

    x86 SSE2: cvtss2sd xmm0, xmm1 -> xmm0[0:63] = (double)xmm1[0:31]

    Encoding: F3 0F 5A C1 (cvtss2sd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTSS2SD instructions
    """
    # f3 0f 5a c1 = cvtss2sd xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x5A, 0xC1])

    edge_vals = _fp_edge_values_single()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            f = edge_vals[i]
        else:
            f = _fp_random_float(rng)

        xmm1 = float_to_xmm(f)

        yield DifferentialTestCase(
            name=f"cvtss2sd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_cvtsi2sd_tests(
    count: int = 100,
    base_seed: int = 0xF3000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTSI2SD (convert int to double) instruction test cases.

    x86 SSE2: cvtsi2sd xmm0, eax -> xmm0[0:63] = (double)eax

    Encoding: F2 0F 2A C0 (cvtsi2sd xmm0, eax)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTSI2SD instructions
    """
    # f2 0f 2a c0 = cvtsi2sd xmm0, eax
    shellcode = bytes([0xF2, 0x0F, 0x2A, 0xC0])

    int_edge_vals = [
        0, 1, -1, 127, -128, 255, 256, 32767, -32768, 65535, 65536,
        0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 1000, -1000, 123456, -123456,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(int_edge_vals):
            eax = int_edge_vals[i]
        else:
            eax = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"cvtsi2sd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "xmm0": 0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_cvtsd2si_tests(
    count: int = 100,
    base_seed: int = 0xF3000005,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTSD2SI (convert double to int) instruction test cases.

    x86 SSE2: cvtsd2si eax, xmm0 -> eax = (int)xmm0[0:63]

    Encoding: F2 0F 2D C0 (cvtsd2si eax, xmm0)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTSD2SI instructions
    """
    # f2 0f 2d c0 = cvtsd2si eax, xmm0
    shellcode = bytes([0xF2, 0x0F, 0x2D, 0xC0])

    # Double values that convert to integers
    fp_edge_vals = [
        0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0,
        127.0, -128.0, 255.0, 256.0, 32767.0, -32768.0,
        65535.0, 1000.0, -1000.0, 123456.0, -123456.0,
        1e9, -1e9,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(fp_edge_vals):
            d = fp_edge_vals[i]
        else:
            d = rng.uniform(-2e9, 2e9)

        xmm0 = double_to_xmm(d)

        yield DifferentialTestCase(
            name=f"cvtsd2si_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "eax": 0},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_cvttsd2si_tests(
    count: int = 100,
    base_seed: int = 0xF3000006,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTTSD2SI (convert double to int with truncation) test cases.

    x86 SSE2: cvttsd2si eax, xmm0 -> eax = (int)trunc(xmm0[0:63])

    Encoding: F2 0F 2C C0 (cvttsd2si eax, xmm0)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTTSD2SI instructions
    """
    # f2 0f 2c c0 = cvttsd2si eax, xmm0
    shellcode = bytes([0xF2, 0x0F, 0x2C, 0xC0])

    fp_edge_vals = [
        0.0, 1.0, -1.0, 0.5, -0.5, 0.9, -0.9, 1.1, -1.1,
        1.9, -1.9, 2.0, -2.0, 127.5, -128.5, 1000.999, -1000.999,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(fp_edge_vals):
            d = fp_edge_vals[i]
        else:
            d = rng.uniform(-2e9, 2e9)

        xmm0 = double_to_xmm(d)

        yield DifferentialTestCase(
            name=f"cvttsd2si_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "eax": 0},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_ucomisd_tests(
    count: int = 100,
    base_seed: int = 0xF3000010,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate UCOMISD (unordered compare scalar double) instruction test cases.

    x86 SSE2: ucomisd xmm0, xmm1 -> compare and set EFLAGS (ZF, PF, CF)

    Encoding: 66 0F 2E C1 (ucomisd xmm0, xmm1)

    Note: This only sets flags, so we follow with SETZ to capture results.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for UCOMISD instructions
    """
    # ucomisd xmm0, xmm1; setz al
    shellcode = bytes([0x66, 0x0F, 0x2E, 0xC1, 0x0F, 0x94, 0xC0])

    edge_vals = _fp_edge_values_double()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            d1 = edge_vals[edge_idx // len(edge_vals)]
            d2 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            d1 = _fp_random_float(rng)
            d2 = _fp_random_float(rng)

        xmm0 = double_to_xmm(d1)
        xmm1 = double_to_xmm(d2)

        yield DifferentialTestCase(
            name=f"ucomisd_setz_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1, "eax": 0},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_maxss_tests(
    count: int = 100,
    base_seed: int = 0xF3000020,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MAXSS (scalar single max) instruction test cases.

    x86 SSE: maxss xmm0, xmm1 -> xmm0[0:31] = max(xmm0[0:31], xmm1[0:31])

    Encoding: F3 0F 5F C1 (maxss xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MAXSS instructions
    """
    # f3 0f 5f c1 = maxss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x5F, 0xC1])

    edge_vals = _fp_edge_values_single()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            f1 = edge_vals[edge_idx // len(edge_vals)]
            f2 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            f1 = _fp_random_float(rng)
            f2 = _fp_random_float(rng)

        xmm0 = float_to_xmm(f1)
        xmm1 = float_to_xmm(f2)

        yield DifferentialTestCase(
            name=f"maxss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_minss_tests(
    count: int = 100,
    base_seed: int = 0xF3000021,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MINSS (scalar single min) instruction test cases.

    x86 SSE: minss xmm0, xmm1 -> xmm0[0:31] = min(xmm0[0:31], xmm1[0:31])

    Encoding: F3 0F 5D C1 (minss xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MINSS instructions
    """
    # f3 0f 5d c1 = minss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x5D, 0xC1])

    edge_vals = _fp_edge_values_single()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            f1 = edge_vals[edge_idx // len(edge_vals)]
            f2 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            f1 = _fp_random_float(rng)
            f2 = _fp_random_float(rng)

        xmm0 = float_to_xmm(f1)
        xmm1 = float_to_xmm(f2)

        yield DifferentialTestCase(
            name=f"minss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_maxsd_tests(
    count: int = 100,
    base_seed: int = 0xF3000022,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MAXSD (scalar double max) instruction test cases.

    x86 SSE2: maxsd xmm0, xmm1 -> xmm0[0:63] = max(xmm0[0:63], xmm1[0:63])

    Encoding: F2 0F 5F C1 (maxsd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MAXSD instructions
    """
    # f2 0f 5f c1 = maxsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x5F, 0xC1])

    edge_vals = _fp_edge_values_double()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            d1 = edge_vals[edge_idx // len(edge_vals)]
            d2 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            d1 = _fp_random_float(rng)
            d2 = _fp_random_float(rng)

        xmm0 = double_to_xmm(d1)
        xmm1 = double_to_xmm(d2)

        yield DifferentialTestCase(
            name=f"maxsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_minsd_tests(
    count: int = 100,
    base_seed: int = 0xF3000023,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MINSD (scalar double min) instruction test cases.

    x86 SSE2: minsd xmm0, xmm1 -> xmm0[0:63] = min(xmm0[0:63], xmm1[0:63])

    Encoding: F2 0F 5D C1 (minsd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MINSD instructions
    """
    # f2 0f 5d c1 = minsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x5D, 0xC1])

    edge_vals = _fp_edge_values_double()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            d1 = edge_vals[edge_idx // len(edge_vals)]
            d2 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            d1 = _fp_random_float(rng)
            d2 = _fp_random_float(rng)

        xmm0 = double_to_xmm(d1)
        xmm1 = double_to_xmm(d2)

        yield DifferentialTestCase(
            name=f"minsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )
