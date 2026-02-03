"""
Floating-point rounding instruction generators for differential testing.

Generates test cases for FP operations with explicit rounding modes:
- ROUNDSS/ROUNDSD: Round scalar single/double to integer with specified rounding
- ROUNDPS/ROUNDPD: Round packed single/double

These test the FP rounding mode handling in VEX operations.
"""
from __future__ import annotations

import random
import struct
from typing import Iterator

from ..harness import DifferentialTestCase


def _make_seed(base: int, index: int) -> int:
    """Create deterministic seed from base seed and index."""
    return (base * 1000003 + index) & 0xFFFFFFFF


def float_to_xmm(f: float) -> int:
    """Convert Python float to XMM register value (single precision in low 32 bits)."""
    packed = struct.pack('<f', f)
    low32 = struct.unpack('<I', packed)[0]
    return low32


def double_to_xmm(d: float) -> int:
    """Convert Python float to XMM register value (double precision in low 64 bits)."""
    packed = struct.pack('<d', d)
    low64 = struct.unpack('<Q', packed)[0]
    return low64


def _fp_rounding_edge_values() -> list[float]:
    """
    Generate edge case float values for rounding operations.

    Focus on values near integers where rounding mode matters.
    """
    return [
        # Exact integers
        0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        100.0,
        -100.0,

        # Exactly halfway
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.5,
        -3.5,
        100.5,
        -100.5,

        # Just above/below integers
        0.1,
        -0.1,
        0.9,
        -0.9,
        1.1,
        -1.1,
        1.9,
        -1.9,
        1.01,
        -1.01,
        1.99,
        -1.99,

        # Quarter values
        0.25,
        -0.25,
        0.75,
        -0.75,
        1.25,
        -1.25,
        1.75,
        -1.75,

        # Small values
        0.001,
        -0.001,
        0.999,
        -0.999,

        # Larger values
        1000.5,
        -1000.5,
        1234.567,
        -1234.567,

        # Powers of 2 boundaries
        0.5,
        1.5,
        2.5,
        4.5,
        8.5,
        16.5,
    ]


def _random_float_for_rounding(rng: random.Random) -> float:
    """Generate random float value good for rounding tests."""
    # Generate values with fractional parts that exercise rounding
    integer_part = rng.randint(-10000, 10000)
    # Include interesting fractional values
    fractions = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99]
    frac = rng.choice(fractions)
    sign = 1 if integer_part >= 0 else -1
    return float(integer_part) + sign * frac


# Rounding modes for SSE4.1 ROUND instructions
# imm8[1:0] specifies rounding mode:
#   00 = Round to nearest even (default)
#   01 = Round toward negative infinity (floor)
#   10 = Round toward positive infinity (ceil)
#   11 = Round toward zero (truncate)
# imm8[2] = 0: use imm8[1:0], 1: use MXCSR rounding mode
# imm8[3] = precision exception disable

ROUNDING_MODES = {
    "nearest": 0x00,
    "floor": 0x01,
    "ceil": 0x02,
    "truncate": 0x03,
}


def generate_roundss_tests(
    count: int = 100,
    base_seed: int = 0xE3000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ROUNDSS (round scalar single) instruction test cases.

    SSE4.1: roundss xmm0, xmm1, imm8 -> round xmm1[31:0] per imm8

    Encoding: 66 0F 3A 0A C1 imm8 (ROUNDSS xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ROUNDSS instructions
    """
    edge_vals = _fp_rounding_edge_values()
    rounding_modes = list(ROUNDING_MODES.values())
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Cycle through edge values and rounding modes
        if test_idx < len(edge_vals) * len(rounding_modes):
            f = edge_vals[test_idx // len(rounding_modes)]
            imm8 = rounding_modes[test_idx % len(rounding_modes)]
            test_idx += 1
        else:
            f = _random_float_for_rounding(rng)
            imm8 = rng.choice(rounding_modes)

        xmm1 = float_to_xmm(f)

        # 66 0F 3A 0A C1 = ROUNDSS xmm0, xmm1, imm8
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0A, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"roundss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_roundsd_tests(
    count: int = 100,
    base_seed: int = 0xE3000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ROUNDSD (round scalar double) instruction test cases.

    SSE4.1: roundsd xmm0, xmm1, imm8 -> round xmm1[63:0] per imm8

    Encoding: 66 0F 3A 0B C1 imm8 (ROUNDSD xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ROUNDSD instructions
    """
    edge_vals = _fp_rounding_edge_values()
    rounding_modes = list(ROUNDING_MODES.values())
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(rounding_modes):
            d = edge_vals[test_idx // len(rounding_modes)]
            imm8 = rounding_modes[test_idx % len(rounding_modes)]
            test_idx += 1
        else:
            d = _random_float_for_rounding(rng)
            imm8 = rng.choice(rounding_modes)

        xmm1 = double_to_xmm(d)

        # 66 0F 3A 0B C1 = ROUNDSD xmm0, xmm1, imm8
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0B, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"roundsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_roundps_tests(
    count: int = 100,
    base_seed: int = 0xE3000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ROUNDPS (round packed single) instruction test cases.

    SSE4.1: roundps xmm0, xmm1, imm8 -> round all 4 floats in xmm1 per imm8

    Encoding: 66 0F 3A 08 C1 imm8 (ROUNDPS xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ROUNDPS instructions
    """
    edge_vals = _fp_rounding_edge_values()
    rounding_modes = list(ROUNDING_MODES.values())

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Build 128-bit value with 4 floats
        if i < len(edge_vals):
            f = edge_vals[i]
            # Same value in all 4 positions
            floats = [f, f, f, f]
        else:
            floats = [_random_float_for_rounding(rng) for _ in range(4)]

        # Pack 4 floats into 128-bit XMM value
        xmm1 = 0
        for j, f in enumerate(floats):
            packed = struct.pack('<f', f)
            val = struct.unpack('<I', packed)[0]
            xmm1 |= val << (j * 32)

        imm8 = rounding_modes[i % len(rounding_modes)]

        # 66 0F 3A 08 C1 = ROUNDPS xmm0, xmm1, imm8
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x08, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"roundps_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_roundpd_tests(
    count: int = 100,
    base_seed: int = 0xE3000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ROUNDPD (round packed double) instruction test cases.

    SSE4.1: roundpd xmm0, xmm1, imm8 -> round both doubles in xmm1 per imm8

    Encoding: 66 0F 3A 09 C1 imm8 (ROUNDPD xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ROUNDPD instructions
    """
    edge_vals = _fp_rounding_edge_values()
    rounding_modes = list(ROUNDING_MODES.values())

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Build 128-bit value with 2 doubles
        if i < len(edge_vals):
            d = edge_vals[i]
            doubles = [d, d]
        else:
            doubles = [_random_float_for_rounding(rng) for _ in range(2)]

        # Pack 2 doubles into 128-bit XMM value
        xmm1 = 0
        for j, d in enumerate(doubles):
            packed = struct.pack('<d', d)
            val = struct.unpack('<Q', packed)[0]
            xmm1 |= val << (j * 64)

        imm8 = rounding_modes[i % len(rounding_modes)]

        # 66 0F 3A 09 C1 = ROUNDPD xmm0, xmm1, imm8
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x09, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"roundpd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Floor/Ceil convenience variants (ROUNDSS/SD with specific modes)

def generate_floorss_tests(
    count: int = 50,
    base_seed: int = 0xE3000021,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate floor operation tests using ROUNDSS with floor mode.

    Uses ROUNDSS with imm8=0x01 (round toward negative infinity).

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for floor operations
    """
    edge_vals = _fp_rounding_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            f = edge_vals[i]
        else:
            f = _random_float_for_rounding(rng)

        xmm1 = float_to_xmm(f)

        # ROUNDSS xmm0, xmm1, 0x01 (floor)
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x01])

        yield DifferentialTestCase(
            name=f"floorss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_ceilss_tests(
    count: int = 50,
    base_seed: int = 0xE3000022,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ceil operation tests using ROUNDSS with ceil mode.

    Uses ROUNDSS with imm8=0x02 (round toward positive infinity).

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ceil operations
    """
    edge_vals = _fp_rounding_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            f = edge_vals[i]
        else:
            f = _random_float_for_rounding(rng)

        xmm1 = float_to_xmm(f)

        # ROUNDSS xmm0, xmm1, 0x02 (ceil)
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x02])

        yield DifferentialTestCase(
            name=f"ceilss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_truncss_tests(
    count: int = 50,
    base_seed: int = 0xE3000023,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate truncate operation tests using ROUNDSS with truncate mode.

    Uses ROUNDSS with imm8=0x03 (round toward zero).

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for truncate operations
    """
    edge_vals = _fp_rounding_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            f = edge_vals[i]
        else:
            f = _random_float_for_rounding(rng)

        xmm1 = float_to_xmm(f)

        # ROUNDSS xmm0, xmm1, 0x03 (truncate)
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x03])

        yield DifferentialTestCase(
            name=f"truncss_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_floorsd_tests(
    count: int = 50,
    base_seed: int = 0xE3000031,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate floor operation tests using ROUNDSD with floor mode.

    Uses ROUNDSD with imm8=0x01 (round toward negative infinity).

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for floor operations (double)
    """
    edge_vals = _fp_rounding_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            d = edge_vals[i]
        else:
            d = _random_float_for_rounding(rng)

        xmm1 = double_to_xmm(d)

        # ROUNDSD xmm0, xmm1, 0x01 (floor)
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x01])

        yield DifferentialTestCase(
            name=f"floorsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_ceilsd_tests(
    count: int = 50,
    base_seed: int = 0xE3000032,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ceil operation tests using ROUNDSD with ceil mode.

    Uses ROUNDSD with imm8=0x02 (round toward positive infinity).

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ceil operations (double)
    """
    edge_vals = _fp_rounding_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            d = edge_vals[i]
        else:
            d = _random_float_for_rounding(rng)

        xmm1 = double_to_xmm(d)

        # ROUNDSD xmm0, xmm1, 0x02 (ceil)
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x02])

        yield DifferentialTestCase(
            name=f"ceilsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_truncsd_tests(
    count: int = 50,
    base_seed: int = 0xE3000033,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate truncate operation tests using ROUNDSD with truncate mode.

    Uses ROUNDSD with imm8=0x03 (round toward zero).

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for truncate operations (double)
    """
    edge_vals = _fp_rounding_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            d = edge_vals[i]
        else:
            d = _random_float_for_rounding(rng)

        xmm1 = double_to_xmm(d)

        # ROUNDSD xmm0, xmm1, 0x03 (truncate)
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x03])

        yield DifferentialTestCase(
            name=f"truncsd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )
