"""
Memory instruction generators for differential testing.

Generates test cases for MOV, PUSH, POP, LEA, and other memory operations.
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
        0x12345678 & mask,          # Typical pattern
        0xDEADBEEF & mask,          # Debug pattern
    ]


# Memory base address for tests (must be mapped by harness)
MEM_BASE = 0x10000


def generate_mov_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MOV register-to-register instruction test cases.

    x86 MOV: mov eax, ebx -> eax = ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MOV instructions
    """
    # x86: mov eax, ebx (0x89 0xD8)
    shellcode = bytes([0x89, 0xD8])

    edge_vals = _edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        ebx = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"mov_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_movzx_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MOVZX (zero extend) instruction test cases.

    x86 MOVZX: movzx eax, bl -> eax = zero_extend(bl)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MOVZX instructions
    """
    # x86: movzx eax, bl (0x0F 0xB6 0xC3)
    shellcode = bytes([0x0F, 0xB6, 0xC3])

    edge_vals_8 = _edge_values(8)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        ebx = rng.choice(edge_vals_8) if i < len(edge_vals_8) else _random_value(rng, 8)
        # Ensure ebx has garbage in high bits to test zero extension
        ebx = ebx | (_random_value(rng, 24) << 8)

        yield DifferentialTestCase(
            name=f"movzx_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0xFFFFFFFF, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_movsx_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MOVSX (sign extend) instruction test cases.

    x86 MOVSX: movsx eax, bl -> eax = sign_extend(bl)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MOVSX instructions
    """
    # x86: movsx eax, bl (0x0F 0xBE 0xC3)
    shellcode = bytes([0x0F, 0xBE, 0xC3])

    edge_vals_8 = _edge_values(8)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        bl = rng.choice(edge_vals_8) if i < len(edge_vals_8) else _random_value(rng, 8)
        ebx = bl | (_random_value(rng, 24) << 8)

        yield DifferentialTestCase(
            name=f"movsx_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_push_pop_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUSH/POP instruction test cases.

    x86 PUSH/POP: push eax; pop ebx -> ebx = eax (via stack)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUSH/POP instructions
    """
    # x86: push eax; pop ebx (0x50 0x5B)
    # This pushes eax to stack and pops into ebx
    shellcode = bytes([0x50, 0x5B])

    edge_vals = _edge_values(32)

    # Stack pointer must point to valid mapped memory
    stack_ptr = MEM_BASE + 0x1000

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) else _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"push_pop_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": 0, "esp": stack_ptr},
            compare_regs=["ebx", "esp"],
            seed=seed,
            initial_mem={stack_ptr - 0x100: b"\x00" * 0x200},  # Stack space
            arch=arch,
        )


def generate_lea_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0005,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate LEA (load effective address) instruction test cases.

    x86 LEA: lea eax, [ebx + ecx*4 + 0x10] -> eax = ebx + ecx*4 + 0x10

    LEA is often used for arithmetic without affecting flags.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for LEA instructions
    """
    # x86: lea eax, [ebx + ecx] (0x8D 0x04 0x0B)
    # Computes ebx + ecx and stores in eax
    shellcode = bytes([0x8D, 0x04, 0x0B])

    edge_vals = _edge_values(32)
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            ebx = edge_vals[edge_idx // len(edge_vals)]
            ecx = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            ebx = _random_value(rng, 32)
            ecx = _random_value(rng, 32)

        yield DifferentialTestCase(
            name=f"lea_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": 0, "ebx": ebx, "ecx": ecx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_xchg_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0006,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate XCHG (exchange) instruction test cases.

    x86 XCHG: xchg eax, ebx -> swap eax and ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for XCHG instructions
    """
    # x86: xchg eax, ebx (0x93)
    shellcode = bytes([0x93])

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
            name=f"xchg_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax", "ebx"],
            seed=seed,
            arch=arch,
        )


def generate_cmov_tests(
    count: int = 100,
    base_seed: int = 0xDEAD0007,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CMOVcc (conditional move) instruction test cases.

    x86 CMOVZ: cmovz eax, ebx -> if ZF then eax = ebx

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CMOVcc instructions
    """
    # x86: cmovz eax, ebx (0x0F 0x44 0xC3)
    # Moves ebx to eax if zero flag is set
    shellcode = bytes([0x0F, 0x44, 0xC3])

    edge_vals = _edge_values(32)

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        eax = rng.choice(edge_vals) if i < len(edge_vals) * 2 else _random_value(rng, 32)
        ebx = _random_value(rng, 32)

        # We'd need to set flags, which is complex. For now, test with default flags
        yield DifferentialTestCase(
            name=f"cmovz_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )
