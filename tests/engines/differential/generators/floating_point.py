"""
Floating point instruction generators for differential testing.

Generates test cases for SSE scalar floating point operations:
- ADDSS, SUBSS, MULSS, DIVSS (scalar single-precision arithmetic)
- CVTSI2SS, CVTSS2SI, etc. (conversions)
- COMISS, UCOMISS (comparisons)
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

    Args:
        f: Float value to convert

    Returns:
        128-bit integer for XMM register
    """
    # Pack as single-precision float (32-bit)
    packed = struct.pack('<f', f)
    # Unpack as 32-bit unsigned int
    low32 = struct.unpack('<I', packed)[0]
    return low32


def xmm_to_float(xmm: int) -> float:
    """
    Convert XMM register value to Python float.

    Extracts the low 32 bits as IEEE 754 single precision.

    Args:
        xmm: 128-bit XMM register value

    Returns:
        Float value from low 32 bits
    """
    low32 = xmm & 0xFFFFFFFF
    packed = struct.pack('<I', low32)
    return struct.unpack('<f', packed)[0]


def double_to_xmm(d: float) -> int:
    """
    Convert a Python float to XMM register value as double precision.

    The double is stored in the low 64 bits as IEEE 754 double precision.
    Upper 64 bits are zeroed.

    Args:
        d: Double value to convert

    Returns:
        128-bit integer for XMM register
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
        # Larger values
        1e10,
        -1e10,
        1e-10,
        -1e-10,
        # Near limits (but not denormals/infinities for now)
        1e38,
        -1e38,
        1e-38,
    ]


def _fp_random_float(rng: random.Random) -> float:
    """Generate a random float value for testing."""
    # Mix of different ranges
    choice = rng.randint(0, 4)
    if choice == 0:
        # Small integers
        return float(rng.randint(-1000, 1000))
    elif choice == 1:
        # Small fractions
        return rng.uniform(-10.0, 10.0)
    elif choice == 2:
        # Larger values
        return rng.uniform(-1e6, 1e6)
    elif choice == 3:
        # Very small values
        return rng.uniform(-1e-6, 1e-6)
    else:
        # Powers of 2 (exact in binary FP)
        exp = rng.randint(-20, 20)
        sign = rng.choice([-1, 1])
        return sign * (2.0 ** exp)


def generate_addss_tests(
    count: int = 100,
    base_seed: int = 0xF1000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ADDSS (scalar single FP add) instruction test cases.

    x86 SSE: addss xmm0, xmm1 -> xmm0[0:31] = xmm0[0:31] + xmm1[0:31]

    Encoding: f3 0f 58 c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ADDSS instructions
    """
    # f3 0f 58 c1 = addss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x58, 0xC1])

    edge_vals = _fp_edge_values_single()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Use edge cases for first tests, then random
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
            name=f"addss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_subss_tests(
    count: int = 100,
    base_seed: int = 0xF1000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SUBSS (scalar single FP subtract) instruction test cases.

    x86 SSE: subss xmm0, xmm1 -> xmm0[0:31] = xmm0[0:31] - xmm1[0:31]

    Encoding: f3 0f 5c c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SUBSS instructions
    """
    # f3 0f 5c c1 = subss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x5C, 0xC1])

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
            name=f"subss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_mulss_tests(
    count: int = 100,
    base_seed: int = 0xF1000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MULSS (scalar single FP multiply) instruction test cases.

    x86 SSE: mulss xmm0, xmm1 -> xmm0[0:31] = xmm0[0:31] * xmm1[0:31]

    Encoding: f3 0f 59 c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MULSS instructions
    """
    # f3 0f 59 c1 = mulss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x59, 0xC1])

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
            name=f"mulss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_divss_tests(
    count: int = 100,
    base_seed: int = 0xF1000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate DIVSS (scalar single FP divide) instruction test cases.

    x86 SSE: divss xmm0, xmm1 -> xmm0[0:31] = xmm0[0:31] / xmm1[0:31]

    Encoding: f3 0f 5e c1

    Note: Divisor should be non-zero to avoid NaN results.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for DIVSS instructions
    """
    # f3 0f 5e c1 = divss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x5E, 0xC1])

    # Exclude zero from divisor edge values
    edge_vals = [v for v in _fp_edge_values_single() if v != 0.0]
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
            # Ensure non-zero divisor
            while f2 == 0.0:
                f2 = _fp_random_float(rng)

        xmm0 = float_to_xmm(f1)
        xmm1 = float_to_xmm(f2)

        yield DifferentialTestCase(
            name=f"divss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_cvtsi2ss_tests(
    count: int = 100,
    base_seed: int = 0xF1000010,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTSI2SS (convert int to scalar single) instruction test cases.

    x86 SSE: cvtsi2ss xmm0, eax -> xmm0[0:31] = (float)eax

    Encoding: f3 0f 2a c0

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTSI2SS instructions
    """
    # f3 0f 2a c0 = cvtsi2ss xmm0, eax
    shellcode = bytes([0xF3, 0x0F, 0x2A, 0xC0])

    # Integer edge values
    int_edge_vals = [
        0,
        1,
        -1,
        127,
        -128,
        255,
        256,
        32767,
        -32768,
        65535,
        65536,
        0x7FFFFFFF,  # INT_MAX
        0x80000000,  # INT_MIN (as unsigned, will be interpreted as signed)
        0xFFFFFFFF,  # -1 as unsigned
        1000,
        -1000,
        123456,
        -123456,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(int_edge_vals):
            eax = int_edge_vals[i]
        else:
            eax = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"cvtsi2ss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "xmm0": 0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_cvtss2si_tests(
    count: int = 100,
    base_seed: int = 0xF1000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTSS2SI (convert scalar single to int) instruction test cases.

    x86 SSE: cvtss2si eax, xmm0 -> eax = (int)xmm0[0:31]

    Encoding: f3 0f 2d c0

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTSS2SI instructions
    """
    # f3 0f 2d c0 = cvtss2si eax, xmm0
    shellcode = bytes([0xF3, 0x0F, 0x2D, 0xC0])

    # Float values that convert nicely to integers
    fp_edge_vals = [
        0.0,
        1.0,
        -1.0,
        0.5,  # Rounds to 0 or 1 depending on rounding mode
        -0.5,
        1.5,
        -1.5,
        2.0,
        -2.0,
        127.0,
        -128.0,
        255.0,
        256.0,
        32767.0,
        -32768.0,
        65535.0,
        1000.0,
        -1000.0,
        123456.0,
        -123456.0,
        1e9,  # Near int limits
        -1e9,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(fp_edge_vals):
            f = fp_edge_vals[i]
        else:
            # Random values within int32 range
            f = rng.uniform(-2e9, 2e9)

        xmm0 = float_to_xmm(f)

        yield DifferentialTestCase(
            name=f"cvtss2si_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "eax": 0},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_cvttss2si_tests(
    count: int = 100,
    base_seed: int = 0xF1000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CVTTSS2SI (convert with truncation) instruction test cases.

    x86 SSE: cvttss2si eax, xmm0 -> eax = (int)trunc(xmm0[0:31])

    Encoding: f3 0f 2c c0

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CVTTSS2SI instructions
    """
    # f3 0f 2c c0 = cvttss2si eax, xmm0
    shellcode = bytes([0xF3, 0x0F, 0x2C, 0xC0])

    fp_edge_vals = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        0.9,
        -0.9,
        1.1,
        -1.1,
        1.9,
        -1.9,
        2.0,
        -2.0,
        127.5,
        -128.5,
        1000.999,
        -1000.999,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(fp_edge_vals):
            f = fp_edge_vals[i]
        else:
            f = rng.uniform(-2e9, 2e9)

        xmm0 = float_to_xmm(f)

        yield DifferentialTestCase(
            name=f"cvttss2si_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "eax": 0},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_ucomiss_tests(
    count: int = 100,
    base_seed: int = 0xF1000020,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate UCOMISS (unordered compare scalar single) instruction test cases.

    x86 SSE: ucomiss xmm0, xmm1 -> compare and set EFLAGS (ZF, PF, CF)

    Encoding: 0f 2e c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for UCOMISS instructions
    """
    # 0f 2e c1 = ucomiss xmm0, xmm1
    shellcode = bytes([0x0F, 0x2E, 0xC1])

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

        # UCOMISS sets EFLAGS, so we should compare flags
        # But for now, just verify the instruction executes
        yield DifferentialTestCase(
            name=f"ucomiss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=[],  # UCOMISS only affects flags
            seed=seed,
            arch=arch,
        )


def generate_comiss_tests(
    count: int = 100,
    base_seed: int = 0xF1000021,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate COMISS (ordered compare scalar single) instruction test cases.

    x86 SSE: comiss xmm0, xmm1 -> compare and set EFLAGS (ZF, PF, CF)

    Encoding: 0f 2f c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for COMISS instructions
    """
    # 0f 2f c1 = comiss xmm0, xmm1
    shellcode = bytes([0x0F, 0x2F, 0xC1])

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
            name=f"comiss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=[],  # COMISS only affects flags
            seed=seed,
            arch=arch,
        )


def generate_sqrtss_tests(
    count: int = 100,
    base_seed: int = 0xF1000030,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SQRTSS (scalar single FP square root) instruction test cases.

    x86 SSE: sqrtss xmm0, xmm1 -> xmm0[0:31] = sqrt(xmm1[0:31])

    Encoding: f3 0f 51 c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SQRTSS instructions
    """
    # f3 0f 51 c1 = sqrtss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x51, 0xC1])

    # Only non-negative values for sqrt
    edge_vals = [v for v in _fp_edge_values_single() if v >= 0.0]
    edge_vals.extend([
        4.0,
        9.0,
        16.0,
        25.0,
        100.0,
        2.0,
        0.25,
        0.01,
    ])

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            f = edge_vals[i]
        else:
            # Random positive float
            f = abs(_fp_random_float(rng))
            if f == 0.0:
                f = 1.0

        xmm1 = float_to_xmm(f)

        yield DifferentialTestCase(
            name=f"sqrtss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_movss_tests(
    count: int = 100,
    base_seed: int = 0xF1000040,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MOVSS (move scalar single) instruction test cases.

    x86 SSE: movss xmm0, xmm1 -> xmm0[0:31] = xmm1[0:31], upper bits zeroed

    Encoding: f3 0f 10 c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MOVSS instructions
    """
    # f3 0f 10 c1 = movss xmm0, xmm1
    shellcode = bytes([0xF3, 0x0F, 0x10, 0xC1])

    edge_vals = _fp_edge_values_single()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        f = edge_vals[i % len(edge_vals)] if i < len(edge_vals) else _fp_random_float(rng)

        # Set xmm0 to have garbage in upper bits
        xmm0 = 0xDEADBEEFCAFEBABE << 64 | 0x12345678
        xmm1 = float_to_xmm(f)

        yield DifferentialTestCase(
            name=f"movss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Double precision variants

def generate_addsd_tests(
    count: int = 100,
    base_seed: int = 0xF2000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ADDSD (scalar double FP add) instruction test cases.

    x86 SSE2: addsd xmm0, xmm1 -> xmm0[0:63] = xmm0[0:63] + xmm1[0:63]

    Encoding: f2 0f 58 c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ADDSD instructions
    """
    # f2 0f 58 c1 = addsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x58, 0xC1])

    edge_vals = _fp_edge_values_single()  # Same edge values work for double
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
            name=f"addsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_subsd_tests(
    count: int = 100,
    base_seed: int = 0xF2000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SUBSD (scalar double FP subtract) instruction test cases.

    x86 SSE2: subsd xmm0, xmm1 -> xmm0[0:63] = xmm0[0:63] - xmm1[0:63]

    Encoding: f2 0f 5c c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SUBSD instructions
    """
    # f2 0f 5c c1 = subsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x5C, 0xC1])

    edge_vals = _fp_edge_values_single()
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
            name=f"subsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_mulsd_tests(
    count: int = 100,
    base_seed: int = 0xF2000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MULSD (scalar double FP multiply) instruction test cases.

    x86 SSE2: mulsd xmm0, xmm1 -> xmm0[0:63] = xmm0[0:63] * xmm1[0:63]

    Encoding: f2 0f 59 c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MULSD instructions
    """
    # f2 0f 59 c1 = mulsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x59, 0xC1])

    edge_vals = _fp_edge_values_single()
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
            name=f"mulsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_divsd_tests(
    count: int = 100,
    base_seed: int = 0xF2000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate DIVSD (scalar double FP divide) instruction test cases.

    x86 SSE2: divsd xmm0, xmm1 -> xmm0[0:63] = xmm0[0:63] / xmm1[0:63]

    Encoding: f2 0f 5e c1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for DIVSD instructions
    """
    # f2 0f 5e c1 = divsd xmm0, xmm1
    shellcode = bytes([0xF2, 0x0F, 0x5E, 0xC1])

    edge_vals = [v for v in _fp_edge_values_single() if v != 0.0]
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
            while d2 == 0.0:
                d2 = _fp_random_float(rng)

        xmm0 = double_to_xmm(d1)
        xmm1 = double_to_xmm(d2)

        yield DifferentialTestCase(
            name=f"divsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )
