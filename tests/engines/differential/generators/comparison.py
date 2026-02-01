"""
Comparison instruction generators for differential testing.

Generates test cases for CMP, TEST, and SETcc instructions.
CMP/TEST only set flags, so we test them by following with SETcc to capture results.
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
        mask,                       # All ones (-1 signed)
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


def generate_cmp_setz_tests(
    count: int = 100,
    base_seed: int = 0xC0000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMP + SETZ test cases (test for equality).

    CMP eax, ebx sets flags based on (eax - ebx)
    SETZ al sets al=1 if ZF=1 (equal), al=0 otherwise

    Encoding:
        39 D8       cmp eax, ebx
        0F 94 C0    setz al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # cmp eax, ebx; setz al
    shellcode = bytes([0x39, 0xD8, 0x0F, 0x94, 0xC0])

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
            name=f"cmp_setz_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],  # al is part of eax
            seed=seed,
            arch=arch,
        )


def generate_cmp_setnz_tests(
    count: int = 100,
    base_seed: int = 0xC0000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMP + SETNZ test cases (test for inequality).

    CMP eax, ebx sets flags based on (eax - ebx)
    SETNZ al sets al=1 if ZF=0 (not equal), al=0 otherwise

    Encoding:
        39 D8       cmp eax, ebx
        0F 95 C0    setnz al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # cmp eax, ebx; setnz al
    shellcode = bytes([0x39, 0xD8, 0x0F, 0x95, 0xC0])

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
            name=f"cmp_setnz_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_cmp_setl_tests(
    count: int = 100,
    base_seed: int = 0xC0000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMP + SETL test cases (signed less than).

    CMP eax, ebx sets flags based on (eax - ebx)
    SETL al sets al=1 if SF!=OF (signed less), al=0 otherwise

    Encoding:
        39 D8       cmp eax, ebx
        0F 9C C0    setl al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # cmp eax, ebx; setl al
    shellcode = bytes([0x39, 0xD8, 0x0F, 0x9C, 0xC0])

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
            name=f"cmp_setl_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_cmp_setg_tests(
    count: int = 100,
    base_seed: int = 0xC0000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMP + SETG test cases (signed greater than).

    CMP eax, ebx sets flags based on (eax - ebx)
    SETG al sets al=1 if ZF=0 and SF=OF (signed greater), al=0 otherwise

    Encoding:
        39 D8       cmp eax, ebx
        0F 9F C0    setg al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # cmp eax, ebx; setg al
    shellcode = bytes([0x39, 0xD8, 0x0F, 0x9F, 0xC0])

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
            name=f"cmp_setg_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_cmp_setb_tests(
    count: int = 100,
    base_seed: int = 0xC0000005,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMP + SETB test cases (unsigned below).

    CMP eax, ebx sets flags based on (eax - ebx)
    SETB al sets al=1 if CF=1 (unsigned below), al=0 otherwise

    Encoding:
        39 D8       cmp eax, ebx
        0F 92 C0    setb al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # cmp eax, ebx; setb al
    shellcode = bytes([0x39, 0xD8, 0x0F, 0x92, 0xC0])

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
            name=f"cmp_setb_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_cmp_seta_tests(
    count: int = 100,
    base_seed: int = 0xC0000006,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMP + SETA test cases (unsigned above).

    CMP eax, ebx sets flags based on (eax - ebx)
    SETA al sets al=1 if CF=0 and ZF=0 (unsigned above), al=0 otherwise

    Encoding:
        39 D8       cmp eax, ebx
        0F 97 C0    seta al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # cmp eax, ebx; seta al
    shellcode = bytes([0x39, 0xD8, 0x0F, 0x97, 0xC0])

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
            name=f"cmp_seta_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_test_setz_tests(
    count: int = 100,
    base_seed: int = 0xC0000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate TEST + SETZ test cases.

    TEST eax, ebx performs (eax AND ebx) and sets flags (SF, ZF, PF)
    SETZ al sets al=1 if ZF=1 (result was zero), al=0 otherwise

    Encoding:
        85 D8       test eax, ebx
        0F 94 C0    setz al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # test eax, ebx; setz al
    shellcode = bytes([0x85, 0xD8, 0x0F, 0x94, 0xC0])

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
            name=f"test_setz_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_test_sets_tests(
    count: int = 100,
    base_seed: int = 0xC0000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate TEST + SETS test cases.

    TEST eax, ebx performs (eax AND ebx) and sets flags
    SETS al sets al=1 if SF=1 (result was negative), al=0 otherwise

    Encoding:
        85 D8       test eax, ebx
        0F 98 C0    sets al

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects
    """
    # test eax, ebx; sets al
    shellcode = bytes([0x85, 0xD8, 0x0F, 0x98, 0xC0])

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
            name=f"test_sets_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )
