"""
Advanced arithmetic instruction generators for differential testing.

Generates test cases for IDIV, ADC, SBB, SHLD, SHRD instructions.
"""
from __future__ import annotations

import random
from typing import Iterator

from ..harness import DifferentialTestCase


def _make_seed(base: int, index: int) -> int:
    """Create deterministic seed from base seed and index."""
    return (base * 1000003 + index) & 0xFFFFFFFF


def _random_value(rng: random.Random, width: int = 32) -> int:
    """Generate random value for given bit width."""
    return rng.randint(0, (1 << width) - 1)


def _edge_values(width: int = 32) -> list[int]:
    """Generate edge case values for given bit width."""
    mask = (1 << width) - 1
    return [
        0,                          # Zero
        1,                          # One
        mask,                       # All ones
        1 << (width - 1),           # MSB set (min signed)
        (1 << (width - 1)) - 1,     # Max signed positive
        2,
        0x7F if width >= 8 else 1,
        0x80 if width >= 8 else 1,
        0xFF if width >= 8 else mask,
        0x100 if width >= 16 else mask,
        0xFFFF if width >= 16 else mask,
        0x10000 if width >= 32 else mask,
    ]


def _shift_amounts(width: int = 32) -> list[int]:
    """Generate edge case shift amounts."""
    return [0, 1, 2, 7, 8, 15, 16, 31, width - 1]


def generate_idiv_tests(
    count: int = 100,
    base_seed: int = 0xA0000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate IDIV (signed divide) instruction test cases.

    x86 IDIV: idiv ebx -> eax = edx:eax / ebx (signed), edx = edx:eax % ebx (signed)

    Encoding: F7 FB (idiv ebx)

    Note: Divisor must be non-zero and quotient must fit in 32 bits.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for IDIV instructions
    """
    # f7 fb = idiv ebx
    shellcode = bytes([0xF7, 0xFB])

    # Edge values excluding zero
    edge_vals = [v for v in _edge_values(32) if v != 0]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Generate values that won't cause overflow
        # For signed 32-bit div, need: -2^31 <= edx:eax / ebx < 2^31
        ebx = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        if ebx == 0:
            ebx = 1  # Ensure non-zero divisor

        # Keep edx small to avoid overflow
        # For signed division, edx should sign-extend eax for safe results
        eax = _random_value(rng, 32)
        # Sign extend eax to edx (make it a valid 64-bit signed value in 32-bit range)
        if eax & 0x80000000:
            edx = 0xFFFFFFFF  # Negative: all ones
        else:
            edx = 0  # Positive: all zeros

        yield DifferentialTestCase(
            name=f"idiv_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": edx},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


def generate_adc_tests(
    count: int = 100,
    base_seed: int = 0xA0000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ADC (add with carry) instruction test cases.

    x86 ADC: adc eax, ebx -> eax = eax + ebx + CF

    We test with CF=0 and CF=1 by using CLC/STC to set up the carry flag.

    Encoding:
        F8          clc (CF=0) or F9 stc (CF=1)
        11 D8       adc eax, ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ADC instructions
    """
    # clc; adc eax, ebx (CF=0)
    shellcode_clc = bytes([0xF8, 0x11, 0xD8])
    # stc; adc eax, ebx (CF=1)
    shellcode_stc = bytes([0xF9, 0x11, 0xD8])

    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Alternate between CF=0 and CF=1
        use_carry = (i % 2) == 1
        shellcode = shellcode_stc if use_carry else shellcode_clc

        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]
            ebx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        name_suffix = "cf1" if use_carry else "cf0"
        yield DifferentialTestCase(
            name=f"adc_{name_suffix}_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_sbb_tests(
    count: int = 100,
    base_seed: int = 0xA0000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SBB (subtract with borrow) instruction test cases.

    x86 SBB: sbb eax, ebx -> eax = eax - ebx - CF

    We test with CF=0 and CF=1 by using CLC/STC to set up the carry flag.

    Encoding:
        F8          clc (CF=0) or F9 stc (CF=1)
        19 D8       sbb eax, ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SBB instructions
    """
    # clc; sbb eax, ebx (CF=0)
    shellcode_clc = bytes([0xF8, 0x19, 0xD8])
    # stc; sbb eax, ebx (CF=1)
    shellcode_stc = bytes([0xF9, 0x19, 0xD8])

    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Alternate between CF=0 and CF=1
        use_borrow = (i % 2) == 1
        shellcode = shellcode_stc if use_borrow else shellcode_clc

        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]
            ebx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        name_suffix = "cf1" if use_borrow else "cf0"
        yield DifferentialTestCase(
            name=f"sbb_{name_suffix}_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_shld_tests(
    count: int = 100,
    base_seed: int = 0xA0000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SHLD (double-precision shift left) instruction test cases.

    x86 SHLD: shld eax, ebx, cl -> shift eax left by cl bits,
              filling vacated bits with MSBs from ebx

    Encoding: 0F A5 D8 (shld eax, ebx, cl)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SHLD instructions
    """
    # 0f a5 d8 = shld eax, ebx, cl
    shellcode = bytes([0x0F, 0xA5, 0xD8])

    edge_vals = _edge_values(32)
    shift_vals = _shift_amounts(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        ebx = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        ecx = rng.choice(shift_vals) if i < len(shift_vals) else rng.randint(0, 31)

        yield DifferentialTestCase(
            name=f"shld_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "ecx": ecx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_shrd_tests(
    count: int = 100,
    base_seed: int = 0xA0000005,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SHRD (double-precision shift right) instruction test cases.

    x86 SHRD: shrd eax, ebx, cl -> shift eax right by cl bits,
              filling vacated bits with LSBs from ebx

    Encoding: 0F AD D8 (shrd eax, ebx, cl)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SHRD instructions
    """
    # 0f ad d8 = shrd eax, ebx, cl
    shellcode = bytes([0x0F, 0xAD, 0xD8])

    edge_vals = _edge_values(32)
    shift_vals = _shift_amounts(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        ebx = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        ecx = rng.choice(shift_vals) if i < len(shift_vals) else rng.randint(0, 31)

        yield DifferentialTestCase(
            name=f"shrd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "ecx": ecx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )
