"""
Bit manipulation instruction generators for differential testing.

Generates test cases for POPCNT, BSF, BSR, LZCNT, TZCNT instructions.
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


def _bit_manipulation_edge_values(width: int = 32) -> list[int]:
    """Generate edge case values for bit manipulation testing."""
    mask = (1 << width) - 1
    values = [
        0,                          # Zero (special case for many ops)
        1,                          # Single bit at position 0
        2,                          # Single bit at position 1
        mask,                       # All bits set
        1 << (width - 1),           # Only MSB set
        (1 << (width - 1)) - 1,     # All except MSB
        0x55555555 & mask,          # Alternating 0101
        0xAAAAAAAA & mask,          # Alternating 1010
        0x0F0F0F0F & mask,          # Alternating nibbles low
        0xF0F0F0F0 & mask,          # Alternating nibbles high
        0x00FF00FF & mask,          # Alternating bytes low
        0xFF00FF00 & mask,          # Alternating bytes high
        0x0000FFFF & mask,          # Low 16 bits
        0xFFFF0000 & mask,          # High 16 bits
    ]
    # Add single bit positions
    for i in range(0, width, 4):
        values.append(1 << i)
    return values


def generate_popcnt_tests(
    count: int = 100,
    base_seed: int = 0xB1000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate POPCNT (population count) instruction test cases.

    x86 POPCNT: popcnt eax, ebx -> eax = count of set bits in ebx

    Encoding: F3 0F B8 C3 (popcnt eax, ebx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for POPCNT instructions
    """
    # f3 0f b8 c3 = popcnt eax, ebx
    shellcode = bytes([0xF3, 0x0F, 0xB8, 0xC3])

    edge_vals = _bit_manipulation_edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            ebx = edge_vals[i]
        else:
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"popcnt_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_bsf_tests(
    count: int = 100,
    base_seed: int = 0xB1000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate BSF (bit scan forward) instruction test cases.

    x86 BSF: bsf eax, ebx -> eax = index of lowest set bit in ebx
    If ebx is 0, result is undefined (ZF is set).

    Encoding: 0F BC C3 (bsf eax, ebx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for BSF instructions
    """
    # 0f bc c3 = bsf eax, ebx
    shellcode = bytes([0x0F, 0xBC, 0xC3])

    # Exclude zero from edge values (undefined behavior)
    edge_vals = [v for v in _bit_manipulation_edge_values(32) if v != 0]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            ebx = edge_vals[i]
        else:
            ebx = _random_value(rng, 32)
            if ebx == 0:
                ebx = 1  # Avoid undefined behavior

        yield DifferentialTestCase(
            name=f"bsf_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_bsr_tests(
    count: int = 100,
    base_seed: int = 0xB1000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate BSR (bit scan reverse) instruction test cases.

    x86 BSR: bsr eax, ebx -> eax = index of highest set bit in ebx
    If ebx is 0, result is undefined (ZF is set).

    Encoding: 0F BD C3 (bsr eax, ebx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for BSR instructions
    """
    # 0f bd c3 = bsr eax, ebx
    shellcode = bytes([0x0F, 0xBD, 0xC3])

    # Exclude zero from edge values (undefined behavior)
    edge_vals = [v for v in _bit_manipulation_edge_values(32) if v != 0]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            ebx = edge_vals[i]
        else:
            ebx = _random_value(rng, 32)
            if ebx == 0:
                ebx = 1  # Avoid undefined behavior

        yield DifferentialTestCase(
            name=f"bsr_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_lzcnt_tests(
    count: int = 100,
    base_seed: int = 0xB1000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate LZCNT (count leading zeros) instruction test cases.

    x86 LZCNT: lzcnt eax, ebx -> eax = count of leading zero bits in ebx
    Unlike BSR, LZCNT has defined behavior for zero (returns operand size).

    Encoding: F3 0F BD C3 (lzcnt eax, ebx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for LZCNT instructions
    """
    # f3 0f bd c3 = lzcnt eax, ebx
    shellcode = bytes([0xF3, 0x0F, 0xBD, 0xC3])

    edge_vals = _bit_manipulation_edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            ebx = edge_vals[i]
        else:
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"lzcnt_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_tzcnt_tests(
    count: int = 100,
    base_seed: int = 0xB1000005,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate TZCNT (count trailing zeros) instruction test cases.

    x86 TZCNT: tzcnt eax, ebx -> eax = count of trailing zero bits in ebx
    Unlike BSF, TZCNT has defined behavior for zero (returns operand size).

    Encoding: F3 0F BC C3 (tzcnt eax, ebx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for TZCNT instructions
    """
    # f3 0f bc c3 = tzcnt eax, ebx
    shellcode = bytes([0xF3, 0x0F, 0xBC, 0xC3])

    edge_vals = _bit_manipulation_edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            ebx = edge_vals[i]
        else:
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"tzcnt_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )
