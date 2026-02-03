"""
64-bit division instruction generators for differential testing.

Generates test cases for:
- DIV r/m32: Unsigned divide edx:eax by r/m32 (DivModU64to32)
- IDIV r/m32: Signed divide edx:eax by r/m32 (DivModS64to32)

These test the DivMod64to32 VEX operations which divide a 64-bit
dividend (edx:eax) by a 32-bit divisor, producing quotient and remainder.
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


def _div64_edge_cases() -> list[tuple[int, int, int]]:
    """
    Generate edge case test values for 64-bit division.

    Returns list of (edx, eax, divisor) tuples that won't cause overflow.
    For DIV r32, the quotient must fit in 32 bits: edx:eax / divisor < 2^32.
    """
    cases = []

    # Simple cases: small dividend, any divisor
    for divisor in [1, 2, 3, 7, 10, 100, 255, 256, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF, 0xFFFFFFFF]:
        # edx=0, so dividend fits in 32 bits - quotient always fits
        cases.append((0, 0, divisor))
        cases.append((0, 1, divisor))
        cases.append((0, 0xFFFFFFFF, divisor))
        cases.append((0, 0x12345678, divisor))

    # 64-bit dividend with small edx (quotient will still fit)
    # When edx < divisor, quotient fits in 32 bits
    cases.append((0, 0, 1))
    cases.append((1, 0, 2))  # 0x100000000 / 2 = 0x80000000
    cases.append((1, 0, 3))  # 0x100000000 / 3 = 0x55555555
    cases.append((0, 0xFFFFFFFF, 1))  # max 32-bit dividend
    cases.append((0x1234, 0x56789ABC, 0x12345678))  # divisor > edx ensures fit

    # Exact division cases (no remainder)
    cases.append((0, 100, 10))  # 100 / 10 = 10
    cases.append((0, 0x100, 0x10))  # 256 / 16 = 16
    cases.append((1, 0, 0x100))  # 0x100000000 / 0x100 = 0x1000000

    # Division with remainder
    cases.append((0, 7, 3))  # 7 / 3 = 2 r 1
    cases.append((0, 0xFF, 0x10))  # 255 / 16 = 15 r 15
    cases.append((0, 100, 7))  # 100 / 7 = 14 r 2

    return cases


def _idiv64_edge_cases() -> list[tuple[int, int, int]]:
    """
    Generate edge case test values for signed 64-bit division.

    Returns list of (edx, eax, divisor) tuples for signed division.
    Values are unsigned representations; interpreter treats them as signed.
    """
    cases = []

    # Positive dividend, positive divisor
    cases.append((0, 100, 10))  # 100 / 10 = 10
    cases.append((0, 7, 3))  # 7 / 3 = 2 r 1
    cases.append((0, 0x7FFFFFFF, 1))  # INT_MAX / 1

    # Negative dividend (sign-extended edx = 0xFFFFFFFF)
    # -1 in 64 bits = 0xFFFFFFFFFFFFFFFF = edx:eax = 0xFFFFFFFF:0xFFFFFFFF
    cases.append((0xFFFFFFFF, 0xFFFFFFFF, 1))  # -1 / 1 = -1
    cases.append((0xFFFFFFFF, 0xFFFFFF9C, 1))  # -100 / 1 = -100
    cases.append((0xFFFFFFFF, 0xFFFFFF9C, 10))  # -100 / 10 = -10

    # Negative dividend, negative divisor
    cases.append((0xFFFFFFFF, 0xFFFFFF9C, 0xFFFFFFF6))  # -100 / -10 = 10

    # Positive dividend, negative divisor
    cases.append((0, 100, 0xFFFFFFF6))  # 100 / -10 = -10

    # Small values
    cases.append((0, 1, 1))
    cases.append((0, 0, 1))
    cases.append((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))  # -1 / -1 = 1

    # Values that test sign extension
    cases.append((0, 0x80000000, 2))  # 0x80000000 / 2 (positive large)
    cases.append((0xFFFFFFFF, 0x80000000, 2))  # large negative / 2

    return cases


def generate_div64to32_unsigned_tests(
    count: int = 100,
    base_seed: int = 0xD1000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate DIV r/m32 (unsigned 64-to-32 division) instruction test cases.

    x86 DIV ebx: edx:eax / ebx -> eax = quotient, edx = remainder (unsigned)

    Encoding: F7 F3 (DIV ebx, opcode F7 /6)

    Note: The quotient must fit in 32 bits, so edx < divisor to avoid #DE.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for unsigned DIV instructions
    """
    # F7 /6 = DIV r/m32, F7 F3 = DIV ebx
    shellcode = bytes([0xF7, 0xF3])

    edge_cases = _div64_edge_cases()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_cases):
            edx, eax, ebx = edge_cases[edge_idx]
            edge_idx += 1
        else:
            # Generate random values that won't overflow
            # Ensure divisor > edx so quotient fits in 32 bits
            ebx = rng.randint(1, 0xFFFFFFFF)  # Non-zero divisor
            # edx must be < ebx for quotient to fit
            edx = rng.randint(0, min(ebx - 1, 0xFFFF))
            eax = _random_value(rng, 32)

        # Skip if divisor is 0 (would cause exception)
        if ebx == 0:
            continue

        yield DifferentialTestCase(
            name=f"div64to32_u_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": edx},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


def generate_div64to32_signed_tests(
    count: int = 100,
    base_seed: int = 0xD1000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate IDIV r/m32 (signed 64-to-32 division) instruction test cases.

    x86 IDIV ebx: edx:eax / ebx -> eax = quotient, edx = remainder (signed)

    Encoding: F7 FB (IDIV ebx, opcode F7 /7)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for signed IDIV instructions
    """
    # F7 /7 = IDIV r/m32, F7 FB = IDIV ebx
    shellcode = bytes([0xF7, 0xFB])

    edge_cases = _idiv64_edge_cases()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_cases):
            edx, eax, ebx = edge_cases[edge_idx]
            edge_idx += 1
        else:
            # Generate random signed values
            # For signed division, we need to be careful about overflow
            # Use smaller values to avoid quotient overflow
            ebx = rng.randint(1, 0x7FFFFFFF)  # Positive divisor, non-zero
            # Keep dividend small to avoid overflow
            is_neg = rng.choice([True, False])
            if is_neg:
                # Small negative: sign extend
                val = rng.randint(-0x7FFFFFFF, -1)
                eax = val & 0xFFFFFFFF
                edx = 0xFFFFFFFF  # Sign extension
            else:
                # Small positive
                eax = rng.randint(0, 0x7FFFFFFF)
                edx = 0

        # Skip if divisor is 0
        if ebx == 0:
            continue

        yield DifferentialTestCase(
            name=f"div64to32_s_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx, "edx": edx},
            compare_regs=["eax", "edx"],
            seed=seed,
            arch=arch,
        )


# AMD64 variants

def generate_div128to64_unsigned_tests(
    count: int = 50,
    base_seed: int = 0xD1000011,
    arch: str = "amd64"
) -> Iterator[DifferentialTestCase]:
    """
    Generate DIV r/m64 (unsigned 128-to-64 division) instruction test cases.

    x86-64 DIV rbx: rdx:rax / rbx -> rax = quotient, rdx = remainder (unsigned)

    Encoding: 48 F7 F3 (REX.W DIV rbx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for unsigned 64-bit DIV instructions
    """
    # REX.W (48) + F7 /6 = DIV r/m64, 48 F7 F3 = DIV rbx
    shellcode = bytes([0x48, 0xF7, 0xF3])

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Generate values that won't overflow
        rbx = rng.randint(1, (1 << 64) - 1)  # Non-zero divisor
        # rdx must be < rbx for quotient to fit in 64 bits
        max_rdx = min(rbx - 1, (1 << 32) - 1)
        rdx = rng.randint(0, max_rdx) if max_rdx > 0 else 0
        rax = rng.randint(0, (1 << 64) - 1)

        if rbx == 0:
            continue

        yield DifferentialTestCase(
            name=f"div128to64_u_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"rax": rax, "rbx": rbx, "rdx": rdx},
            compare_regs=["rax", "rdx"],
            seed=seed,
            arch=arch,
        )
