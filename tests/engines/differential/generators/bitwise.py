"""
Bitwise instruction generators for differential testing.

Generates test cases for AND, OR, XOR, NOT, SHL, SHR, SAR instructions.
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
        1 << (width - 1),           # MSB set
        (1 << (width - 1)) - 1,     # Max positive signed
        0x55555555 & mask,          # Alternating 0101
        0xAAAAAAAA & mask,          # Alternating 1010
        0x0F0F0F0F & mask,          # Alternating nibbles
        0xF0F0F0F0 & mask,          # Alternating nibbles (inverted)
    ]


def _shift_amounts(width: int = 32) -> list[int]:
    """Generate edge case shift amounts."""
    return [0, 1, 2, 7, 8, 15, 16, 31, width - 1]


def generate_and_tests(
    count: int = 100,
    base_seed: int = 0xABCD0001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate AND instruction test cases.

    x86 AND: and eax, ebx -> eax = eax & ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for AND instructions
    """
    # x86: and eax, ebx (0x21 0xD8)
    shellcode = bytes([0x21, 0xD8])

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
            name=f"and_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_or_tests(
    count: int = 100,
    base_seed: int = 0xABCD0002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate OR instruction test cases.

    x86 OR: or eax, ebx -> eax = eax | ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for OR instructions
    """
    # x86: or eax, ebx (0x09 0xD8)
    shellcode = bytes([0x09, 0xD8])

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
            name=f"or_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_xor_tests(
    count: int = 100,
    base_seed: int = 0xABCD0003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate XOR instruction test cases.

    x86 XOR: xor eax, ebx -> eax = eax ^ ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for XOR instructions
    """
    # x86: xor eax, ebx (0x31 0xD8)
    shellcode = bytes([0x31, 0xD8])

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
            name=f"xor_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_not_tests(
    count: int = 100,
    base_seed: int = 0xABCD0004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate NOT instruction test cases.

    x86 NOT: not eax -> eax = ~eax (bitwise complement)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for NOT instructions
    """
    # x86: not eax (0xF7 0xD0)
    shellcode = bytes([0xF7, 0xD0])

    edge_vals = _edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"not_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_shift_tests(
    count: int = 100,
    base_seed: int = 0xABCD0005,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate shift instruction test cases (SHL, SHR, SAR).

    x86 SHL: shl eax, cl -> eax = eax << cl
    x86 SHR: shr eax, cl -> eax = eax >> cl (logical)
    x86 SAR: sar eax, cl -> eax = eax >> cl (arithmetic)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for shift instructions
    """
    # x86 shift by cl:
    # shl eax, cl (0xD3 0xE0)
    # shr eax, cl (0xD3 0xE8)
    # sar eax, cl (0xD3 0xF8)
    shift_ops = [
        ("shl", bytes([0xD3, 0xE0])),
        ("shr", bytes([0xD3, 0xE8])),
        ("sar", bytes([0xD3, 0xF8])),
    ]

    edge_vals = _edge_values(32)
    shift_vals = _shift_amounts(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        op_name, shellcode = shift_ops[i % len(shift_ops)]

        # Select value and shift amount
        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        ecx = rng.choice(shift_vals) if i < len(shift_vals) else rng.randint(0, 31)

        yield DifferentialTestCase(
            name=f"{op_name}_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ecx": ecx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_rol_ror_tests(
    count: int = 100,
    base_seed: int = 0xABCD0006,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate rotate instruction test cases (ROL, ROR).

    x86 ROL: rol eax, cl -> rotate left by cl bits
    x86 ROR: ror eax, cl -> rotate right by cl bits

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for rotate instructions
    """
    # x86 rotate by cl:
    # rol eax, cl (0xD3 0xC0)
    # ror eax, cl (0xD3 0xC8)
    rotate_ops = [
        ("rol", bytes([0xD3, 0xC0])),
        ("ror", bytes([0xD3, 0xC8])),
    ]

    edge_vals = _edge_values(32)
    shift_vals = _shift_amounts(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        op_name, shellcode = rotate_ops[i % len(rotate_ops)]

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)
        ecx = rng.choice(shift_vals) if i < len(shift_vals) else rng.randint(0, 31)

        yield DifferentialTestCase(
            name=f"{op_name}_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ecx": ecx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_bswap_tests(
    count: int = 100,
    base_seed: int = 0xABCD0007,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate BSWAP (byte swap) instruction test cases.

    x86 BSWAP: bswap eax -> reverse byte order of eax

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for BSWAP instructions
    """
    # x86: bswap eax (0x0F 0xC8)
    shellcode = bytes([0x0F, 0xC8])

    edge_vals = _edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"bswap_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )
