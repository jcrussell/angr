"""
High multiplication instruction generators for differential testing.

Generates test cases for:
- MUL r/m32: Unsigned multiply, edx:eax = eax * r/m32 (MullU32, MulHi)
- IMUL r/m32: Signed multiply (one-operand form), edx:eax = eax * r/m32 (MullS32)

These test the MulHi VEX operations which return the high half of a
widening multiply.
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


def _mulhi_edge_values(width: int = 32) -> list[int]:
    """
    Generate edge case values for multiplication testing.

    Focus on values that produce interesting high-half results.
    """
    mask = (1 << width) - 1
    half = 1 << (width // 2)

    return [
        0,                          # Zero
        1,                          # One
        2,                          # Small
        mask,                       # All ones (max unsigned)
        1 << (width - 1),           # MSB set (min signed)
        (1 << (width - 1)) - 1,     # Max signed positive
        half,                       # 2^16 for 32-bit (causes high bits)
        half - 1,                   # 2^16 - 1
        half + 1,                   # 2^16 + 1
        0x7F,                       # Byte boundary
        0x80,
        0xFF,
        0x100,
        0xFFFF,
        0x10000,
        0x12345678,                 # Arbitrary pattern
        0xDEADBEEF,                 # Another pattern
        0xCAFEBABE,
    ]


def _mulhi_unsigned_edge_cases() -> list[tuple[int, int]]:
    """
    Generate edge case pairs specifically for testing high multiplication.

    Returns (eax, ebx) pairs that produce interesting edx (high) results.
    """
    cases = []

    # Products that fit in 32 bits (edx = 0)
    cases.append((0, 0))            # 0 * 0 = 0
    cases.append((1, 1))            # 1 * 1 = 1
    cases.append((100, 100))        # 10000, fits in 32 bits
    cases.append((0xFFFF, 1))       # 0xFFFF, no high bits
    cases.append((0x10000, 0))      # anything * 0 = 0

    # Products that require high bits (edx != 0)
    cases.append((0x10000, 0x10000))    # 2^32, edx = 1, eax = 0
    cases.append((0xFFFFFFFF, 2))       # 2 * (2^32-1) = 2^33 - 2
    cases.append((0xFFFFFFFF, 0xFFFFFFFF))  # Max product
    cases.append((0x80000000, 2))       # 2^32, edx = 1
    cases.append((0x80000000, 0x80000000))  # 2^62

    # Interesting bit patterns
    cases.append((0xAAAAAAAA, 0x55555555))  # Alternating bits
    cases.append((0x12345678, 0x9ABCDEF0))
    cases.append((0xDEADBEEF, 0xCAFEBABE))

    return cases


def _mulhi_signed_edge_cases() -> list[tuple[int, int]]:
    """
    Generate edge case pairs for signed multiplication.

    Returns (eax, ebx) pairs as unsigned representations of signed values.
    """
    cases = []

    # Both positive
    cases.append((100, 100))        # 10000
    cases.append((0x7FFFFFFF, 1))   # INT_MAX * 1
    cases.append((0x7FFFFFFF, 2))   # INT_MAX * 2 (overflow into high)

    # Both negative (as unsigned representations)
    # -1 = 0xFFFFFFFF, -1 * -1 = 1
    cases.append((0xFFFFFFFF, 0xFFFFFFFF))  # -1 * -1 = 1
    # -2 = 0xFFFFFFFE, -2 * -2 = 4
    cases.append((0xFFFFFFFE, 0xFFFFFFFE))  # -2 * -2 = 4
    # -100 = 0xFFFFFF9C
    cases.append((0xFFFFFF9C, 0xFFFFFF9C))  # -100 * -100 = 10000

    # Positive * negative
    # 100 * -1 = -100
    cases.append((100, 0xFFFFFFFF))         # 100 * -1 = -100
    cases.append((0xFFFFFFFF, 100))         # -1 * 100 = -100
    cases.append((0x7FFFFFFF, 0xFFFFFFFF))  # INT_MAX * -1 = -INT_MAX

    # INT_MIN cases (0x80000000 = -2147483648)
    cases.append((0x80000000, 1))           # INT_MIN * 1 = INT_MIN
    cases.append((0x80000000, 0xFFFFFFFF))  # INT_MIN * -1 (overflow!)
    cases.append((0x80000000, 2))           # INT_MIN * 2

    # Large magnitude products
    cases.append((0x40000000, 4))           # 2^30 * 4 = 2^32

    return cases


def generate_mulhi_unsigned_tests(
    count: int = 100,
    base_seed: int = 0xAA000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MUL r/m32 (unsigned multiply) instruction test cases.

    x86 MUL ebx: edx:eax = eax * ebx (unsigned)

    This tests both the low result (eax) and high result (edx).
    The high result (edx) exercises the MulHi operation.

    Encoding: F7 E3 (MUL ebx, opcode F7 /4)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for unsigned MUL instructions
    """
    # F7 /4 = MUL r/m32, F7 E3 = MUL ebx
    shellcode = bytes([0xF7, 0xE3])

    edge_cases = _mulhi_unsigned_edge_cases()
    edge_vals = _mulhi_edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_cases):
            eax, ebx = edge_cases[edge_idx]
            edge_idx += 1
        elif edge_idx < len(edge_cases) + len(edge_vals) * len(edge_vals):
            # Cartesian product of edge values
            local_idx = edge_idx - len(edge_cases)
            eax = edge_vals[local_idx // len(edge_vals)]
            ebx = edge_vals[local_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"mulhi_u_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": 0},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


def generate_mulhi_signed_tests(
    count: int = 100,
    base_seed: int = 0xAA000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate IMUL r/m32 (signed multiply, one-operand form) test cases.

    x86 IMUL ebx: edx:eax = eax * ebx (signed)

    This tests the signed widening multiply, producing 64-bit result.

    Encoding: F7 EB (IMUL ebx, opcode F7 /5)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for signed IMUL instructions
    """
    # F7 /5 = IMUL r/m32, F7 EB = IMUL ebx
    shellcode = bytes([0xF7, 0xEB])

    edge_cases = _mulhi_signed_edge_cases()
    edge_vals = _mulhi_edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_cases):
            eax, ebx = edge_cases[edge_idx]
            edge_idx += 1
        elif edge_idx < len(edge_cases) + len(edge_vals) * len(edge_vals):
            local_idx = edge_idx - len(edge_cases)
            eax = edge_vals[local_idx // len(edge_vals)]
            ebx = edge_vals[local_idx % len(edge_vals)]
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"mulhi_s_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": 0},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


# AMD64 variants

def generate_mulhi_unsigned_64_tests(
    count: int = 50,
    base_seed: int = 0xAA000011,
    arch: str = "amd64"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MUL r/m64 (unsigned 64-bit multiply) instruction test cases.

    x86-64 MUL rbx: rdx:rax = rax * rbx (unsigned)

    Encoding: 48 F7 E3 (REX.W MUL rbx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for unsigned 64-bit MUL instructions
    """
    # REX.W + F7 /4 = MUL r/m64, 48 F7 E3 = MUL rbx
    shellcode = bytes([0x48, 0xF7, 0xE3])

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Mix of edge cases and random
        if i < 10:
            # Edge cases
            edge_64 = [
                (0, 0),
                (1, 1),
                ((1 << 64) - 1, 1),
                ((1 << 64) - 1, 2),
                ((1 << 64) - 1, (1 << 64) - 1),
                ((1 << 32), (1 << 32)),
                (0x123456789ABCDEF0, 0xFEDCBA9876543210),
                (0x8000000000000000, 2),
                (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
                (0x100000000, 0x100000000),
            ][i]
            rax, rbx = edge_64
        else:
            rax = rng.randint(0, (1 << 64) - 1)
            rbx = rng.randint(0, (1 << 64) - 1)

        yield DifferentialTestCase(
            name=f"mulhi_u64_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"rax": rax, "rbx": rbx, "rdx": 0},
            compare_regs=["rax", "rdx"],
            seed=seed,
            arch=arch,
        )


def generate_mulhi_signed_64_tests(
    count: int = 50,
    base_seed: int = 0xAA000012,
    arch: str = "amd64"
) -> Iterator[DifferentialTestCase]:
    """
    Generate IMUL r/m64 (signed 64-bit multiply, one-operand form) test cases.

    x86-64 IMUL rbx: rdx:rax = rax * rbx (signed)

    Encoding: 48 F7 EB (REX.W IMUL rbx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for signed 64-bit IMUL instructions
    """
    # REX.W + F7 /5 = IMUL r/m64, 48 F7 EB = IMUL rbx
    shellcode = bytes([0x48, 0xF7, 0xEB])

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        rax = rng.randint(0, (1 << 64) - 1)
        rbx = rng.randint(0, (1 << 64) - 1)

        yield DifferentialTestCase(
            name=f"mulhi_s64_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"rax": rax, "rbx": rbx, "rdx": 0},
            compare_regs=["rax", "rdx"],
            seed=seed,
            arch=arch,
        )
