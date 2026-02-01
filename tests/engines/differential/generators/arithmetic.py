"""
Arithmetic instruction generators for differential testing.

Generates test cases for ADD, SUB, MUL, DIV instructions with various
operand combinations.
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
        0x7F if width >= 8 else 1,  # Common values
        0x80 if width >= 8 else 1,
        0xFF if width >= 8 else mask,
        0x100 if width >= 16 else mask,
        0xFFFF if width >= 16 else mask,
    ]


def generate_add_tests(
    count: int = 100,
    base_seed: int = 0x12345678,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate ADD instruction test cases.

    x86 ADD: add eax, ebx -> eax = eax + ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for ADD instructions
    """
    # x86: add eax, ebx (0x01 0xD8)
    # Adds ebx to eax, stores result in eax
    shellcode = bytes([0x01, 0xD8])

    # Start with edge cases
    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Use edge cases for first tests, then random
        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]
            ebx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"add_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_sub_tests(
    count: int = 100,
    base_seed: int = 0x23456789,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate SUB instruction test cases.

    x86 SUB: sub eax, ebx -> eax = eax - ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for SUB instructions
    """
    # x86: sub eax, ebx (0x29 0xD8)
    # Subtracts ebx from eax, stores result in eax
    shellcode = bytes([0x29, 0xD8])

    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]
            ebx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"sub_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_mul_tests(
    count: int = 100,
    base_seed: int = 0x34567890,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MUL instruction test cases.

    x86 MUL: mul ebx -> edx:eax = eax * ebx (unsigned)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MUL instructions
    """
    # x86: mul ebx (0xF7 0xE3)
    # Multiplies eax by ebx, stores 64-bit result in edx:eax
    shellcode = bytes([0xF7, 0xE3])

    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]
            ebx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"mul_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": 0},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


def generate_div_tests(
    count: int = 100,
    base_seed: int = 0x45678901,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate DIV instruction test cases.

    x86 DIV: div ebx -> eax = edx:eax / ebx, edx = edx:eax % ebx (unsigned)

    Note: Divisor must be non-zero to avoid division by zero exception.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for DIV instructions
    """
    # x86: div ebx (0xF7 0xF3)
    # Divides edx:eax by ebx, quotient in eax, remainder in edx
    shellcode = bytes([0xF7, 0xF3])

    edge_vals = [v for v in _edge_values(32) if v != 0]  # Exclude zero divisors

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Generate values that won't cause overflow
        # For 32-bit div, quotient must fit in 32 bits: edx:eax / ebx < 2^32
        ebx = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        if ebx == 0:
            ebx = 1  # Ensure non-zero divisor

        # Keep edx small to avoid overflow
        edx = rng.randint(0, min(ebx - 1, 0xFFFF)) if ebx > 1 else 0
        eax = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"div_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": edx},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


def generate_imul_tests(
    count: int = 100,
    base_seed: int = 0x56789012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate IMUL (signed multiply) instruction test cases.

    x86 IMUL: imul ebx -> edx:eax = eax * ebx (signed)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for IMUL instructions
    """
    # x86: imul ebx (0xF7 0xEB)
    # Signed multiply of eax by ebx, 64-bit result in edx:eax
    shellcode = bytes([0xF7, 0xEB])

    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]
            ebx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"imul_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": 0},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


def generate_inc_dec_tests(
    count: int = 100,
    base_seed: int = 0x67890123,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate INC/DEC instruction test cases.

    x86 INC: inc eax -> eax = eax + 1
    x86 DEC: dec eax -> eax = eax - 1

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for INC/DEC instructions
    """
    # x86: inc eax (0x40) / dec eax (0x48)
    inc_shellcode = bytes([0x40])
    dec_shellcode = bytes([0x48])

    edge_vals = _edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        use_inc = (i % 2) == 0

        yield DifferentialTestCase(
            name=f"{'inc' if use_inc else 'dec'}_0x{seed:08x}",
            shellcode=inc_shellcode if use_inc else dec_shellcode,
            initial_regs={"eax": eax},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_neg_tests(
    count: int = 100,
    base_seed: int = 0x78901234,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate NEG (negate) instruction test cases.

    x86 NEG: neg eax -> eax = 0 - eax (two's complement negation)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for NEG instructions
    """
    # x86: neg eax (0xF7 0xD8)
    shellcode = bytes([0xF7, 0xD8])

    edge_vals = _edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"neg_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )
