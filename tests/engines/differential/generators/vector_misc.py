"""
Miscellaneous vector instruction generators for differential testing.

Generates test cases for:
- MOVD/MOVQ to XMM: Move 32/64-bit value to low part of XMM (SetV128lo32/64)
- PUNPCKL/H: Interleave low/high parts of vectors (VInterleave operations)
- PSHUFD: Shuffle dwords within XMM (VPerm operations)
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


def _random_xmm(rng: random.Random) -> int:
    """Generate random 128-bit XMM value."""
    return rng.randint(0, (1 << 128) - 1)


def _edge_values_32() -> list[int]:
    """Generate edge case values for 32-bit operations."""
    return [
        0,
        1,
        0xFFFFFFFF,
        0x80000000,
        0x7FFFFFFF,
        0x12345678,
        0xDEADBEEF,
        0xCAFEBABE,
        0x55555555,
        0xAAAAAAAA,
    ]


def _edge_values_64() -> list[int]:
    """Generate edge case values for 64-bit operations."""
    return [
        0,
        1,
        (1 << 64) - 1,
        1 << 63,
        (1 << 63) - 1,
        0x123456789ABCDEF0,
        0xDEADBEEFCAFEBABE,
        0x5555555555555555,
        0xAAAAAAAAAAAAAAAA,
    ]


def _xmm_interleave_edge_bytes() -> list[int]:
    """Generate edge case XMM values for byte interleave operations."""
    return [
        0,
        (1 << 128) - 1,
        0x000102030405060708090A0B0C0D0E0F,         # Sequential bytes
        0x0F0E0D0C0B0A09080706050403020100,         # Reverse sequential
        0x00000000000000000001020304050607,         # Low bytes set
        0x08090A0B0C0D0E0F0000000000000000,         # High bytes set
        0x00FF00FF00FF00FF00FF00FF00FF00FF,         # Alternating
        0xFF00FF00FF00FF00FF00FF00FF00FF00,
    ]


def _xmm_interleave_edge_words() -> list[int]:
    """Generate edge case XMM values for word interleave operations."""
    return [
        0,
        (1 << 128) - 1,
        0x00010002000300040005000600070008,         # Sequential words
        0x00000000000000000001000200030004,         # Low words set
        0x00050006000700080000000000000000,         # High words set
        0x0000FFFF0000FFFF0000FFFF0000FFFF,         # Alternating
    ]


def _xmm_interleave_edge_dwords() -> list[int]:
    """Generate edge case XMM values for dword interleave operations."""
    return [
        0,
        (1 << 128) - 1,
        0x00000001000000020000000300000004,         # Sequential dwords
        0x00000000000000000000000100000002,         # Low dwords set
        0x00000003000000040000000000000000,         # High dwords set
        0x00000000FFFFFFFF00000000FFFFFFFF,         # Alternating
    ]


def _xmm_interleave_edge_qwords() -> list[int]:
    """Generate edge case XMM values for qword interleave operations."""
    return [
        0,
        (1 << 128) - 1,
        0x00000000000000010000000000000002,         # Sequential qwords
        0x00000000000000000000000000000001,         # Low qword set
        0x00000000000000020000000000000000,         # High qword set
        0x0000000000000000FFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF0000000000000000,
    ]


# MOVD/MOVQ to XMM generators

def generate_movd_to_xmm_tests(
    count: int = 100,
    base_seed: int = 0xE2000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MOVD xmm, r32 (move 32-bit to XMM low) instruction test cases.

    SSE2: movd xmm0, eax -> xmm0[31:0] = eax, xmm0[127:32] = 0

    This tests the SetV128lo32 VEX operation.

    Encoding: 66 0F 6E C0 (MOVD xmm0, eax)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MOVD instructions
    """
    # 66 0F 6E C0 = MOVD xmm0, eax
    shellcode = bytes([0x66, 0x0F, 0x6E, 0xC0])

    edge_vals = _edge_values_32()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            eax = edge_vals[i]
        else:
            eax = _random_value(rng, 32)

        # Set xmm0 to garbage to verify upper bits are zeroed
        xmm0_initial = 0xDEADBEEFCAFEBABE123456789ABCDEF0

        yield DifferentialTestCase(
            name=f"movd_to_xmm_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "xmm0": xmm0_initial},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_movq_to_xmm_tests(
    count: int = 100,
    base_seed: int = 0xE2000002,
    arch: str = "amd64"
) -> Iterator[DifferentialTestCase]:
    """
    Generate MOVQ xmm, r64 (move 64-bit to XMM low) instruction test cases.

    x86-64: movq xmm0, rax -> xmm0[63:0] = rax, xmm0[127:64] = 0

    This tests the SetV128lo64 VEX operation.

    Encoding: 66 48 0F 6E C0 (REX.W MOVD xmm0, rax = MOVQ)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for MOVQ instructions
    """
    # 66 48 0F 6E C0 = MOVQ xmm0, rax (REX.W makes it 64-bit)
    shellcode = bytes([0x66, 0x48, 0x0F, 0x6E, 0xC0])

    edge_vals = _edge_values_64()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            rax = edge_vals[i]
        else:
            rax = _random_value(rng, 64)

        # Set xmm0 to garbage to verify upper bits are zeroed
        xmm0_initial = 0xDEADBEEFCAFEBABE123456789ABCDEF0

        yield DifferentialTestCase(
            name=f"movq_to_xmm_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"rax": rax, "xmm0": xmm0_initial},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Interleave generators - Byte granularity

def generate_punpcklbw_tests(
    count: int = 100,
    base_seed: int = 0xE2000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKLBW (unpack and interleave low bytes) test cases.

    SSE2: punpcklbw xmm0, xmm1 -> interleave low 8 bytes of xmm0 and xmm1

    Encoding: 66 0F 60 C1 (PUNPCKLBW xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKLBW instructions
    """
    # 66 0F 60 C1 = PUNPCKLBW xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x60, 0xC1])

    edge_vals = _xmm_interleave_edge_bytes()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpcklbw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_punpckhbw_tests(
    count: int = 100,
    base_seed: int = 0xE2000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKHBW (unpack and interleave high bytes) test cases.

    SSE2: punpckhbw xmm0, xmm1 -> interleave high 8 bytes of xmm0 and xmm1

    Encoding: 66 0F 68 C1 (PUNPCKHBW xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKHBW instructions
    """
    # 66 0F 68 C1 = PUNPCKHBW xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x68, 0xC1])

    edge_vals = _xmm_interleave_edge_bytes()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpckhbw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Interleave generators - Word granularity

def generate_punpcklwd_tests(
    count: int = 100,
    base_seed: int = 0xE2000021,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKLWD (unpack and interleave low words) test cases.

    SSE2: punpcklwd xmm0, xmm1 -> interleave low 4 words of xmm0 and xmm1

    Encoding: 66 0F 61 C1 (PUNPCKLWD xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKLWD instructions
    """
    # 66 0F 61 C1 = PUNPCKLWD xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x61, 0xC1])

    edge_vals = _xmm_interleave_edge_words()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpcklwd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_punpckhwd_tests(
    count: int = 100,
    base_seed: int = 0xE2000022,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKHWD (unpack and interleave high words) test cases.

    SSE2: punpckhwd xmm0, xmm1 -> interleave high 4 words of xmm0 and xmm1

    Encoding: 66 0F 69 C1 (PUNPCKHWD xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKHWD instructions
    """
    # 66 0F 69 C1 = PUNPCKHWD xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x69, 0xC1])

    edge_vals = _xmm_interleave_edge_words()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpckhwd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Interleave generators - Dword granularity

def generate_punpckldq_tests(
    count: int = 100,
    base_seed: int = 0xE2000031,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKLDQ (unpack and interleave low dwords) test cases.

    SSE2: punpckldq xmm0, xmm1 -> interleave low 2 dwords of xmm0 and xmm1

    Encoding: 66 0F 62 C1 (PUNPCKLDQ xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKLDQ instructions
    """
    # 66 0F 62 C1 = PUNPCKLDQ xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x62, 0xC1])

    edge_vals = _xmm_interleave_edge_dwords()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpckldq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_punpckhdq_tests(
    count: int = 100,
    base_seed: int = 0xE2000032,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKHDQ (unpack and interleave high dwords) test cases.

    SSE2: punpckhdq xmm0, xmm1 -> interleave high 2 dwords of xmm0 and xmm1

    Encoding: 66 0F 6A C1 (PUNPCKHDQ xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKHDQ instructions
    """
    # 66 0F 6A C1 = PUNPCKHDQ xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x6A, 0xC1])

    edge_vals = _xmm_interleave_edge_dwords()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpckhdq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Interleave generators - Qword granularity

def generate_punpcklqdq_tests(
    count: int = 100,
    base_seed: int = 0xE2000041,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKLQDQ (unpack and interleave low qwords) test cases.

    SSE2: punpcklqdq xmm0, xmm1 -> xmm0 = xmm1[63:0] : xmm0[63:0]

    Encoding: 66 0F 6C C1 (PUNPCKLQDQ xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKLQDQ instructions
    """
    # 66 0F 6C C1 = PUNPCKLQDQ xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x6C, 0xC1])

    edge_vals = _xmm_interleave_edge_qwords()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpcklqdq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_punpckhqdq_tests(
    count: int = 100,
    base_seed: int = 0xE2000042,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PUNPCKHQDQ (unpack and interleave high qwords) test cases.

    SSE2: punpckhqdq xmm0, xmm1 -> xmm0 = xmm1[127:64] : xmm0[127:64]

    Encoding: 66 0F 6D C1 (PUNPCKHQDQ xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PUNPCKHQDQ instructions
    """
    # 66 0F 6D C1 = PUNPCKHQDQ xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x6D, 0xC1])

    edge_vals = _xmm_interleave_edge_qwords()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            xmm0 = edge_vals[edge_idx // len(edge_vals)]
            xmm1 = edge_vals[edge_idx % len(edge_vals)]
            edge_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        yield DifferentialTestCase(
            name=f"punpckhqdq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Shuffle/Permute generators

def generate_pshufd_tests(
    count: int = 100,
    base_seed: int = 0xE2000051,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSHUFD (shuffle dwords) instruction test cases.

    SSE2: pshufd xmm0, xmm1, imm8 -> shuffle dwords of xmm1 into xmm0

    The imm8 byte encodes which source dword goes to each destination:
    - Bits 1:0 -> dest dword 0
    - Bits 3:2 -> dest dword 1
    - Bits 5:4 -> dest dword 2
    - Bits 7:6 -> dest dword 3

    Encoding: 66 0F 70 C1 imm8 (PSHUFD xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSHUFD instructions
    """
    edge_vals = _xmm_interleave_edge_dwords()

    # Interesting shuffle patterns
    shuffle_patterns = [
        0x00,  # Broadcast dword 0
        0x55,  # Broadcast dword 1
        0xAA,  # Broadcast dword 2
        0xFF,  # Broadcast dword 3
        0xE4,  # Identity (3,2,1,0)
        0x1B,  # Reverse (0,1,2,3)
        0xB1,  # Swap pairs
        0x4E,  # Rotate by 2
        0x39,  # Rotate by 1
        0x93,  # Rotate by 3
        0x00,  # All same
        0xD8,  # Common pattern
    ]

    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shuffle_patterns):
            xmm1 = edge_vals[test_idx // len(shuffle_patterns)]
            imm8 = shuffle_patterns[test_idx % len(shuffle_patterns)]
            test_idx += 1
        else:
            xmm1 = _random_xmm(rng)
            imm8 = rng.randint(0, 255)

        # 66 0F 70 C1 = PSHUFD xmm0, xmm1, imm8
        shellcode = bytes([0x66, 0x0F, 0x70, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"pshufd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pshufhw_tests(
    count: int = 50,
    base_seed: int = 0xE2000052,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSHUFHW (shuffle high words) instruction test cases.

    SSE2: pshufhw xmm0, xmm1, imm8 -> shuffle high 4 words of xmm1

    Low qword is copied unchanged, high qword words are shuffled.

    Encoding: F3 0F 70 C1 imm8 (PSHUFHW xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSHUFHW instructions
    """
    edge_vals = _xmm_interleave_edge_words()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            xmm1 = edge_vals[i]
        else:
            xmm1 = _random_xmm(rng)

        imm8 = rng.randint(0, 255)

        # F3 0F 70 C1 = PSHUFHW xmm0, xmm1, imm8
        shellcode = bytes([0xF3, 0x0F, 0x70, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"pshufhw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pshuflw_tests(
    count: int = 50,
    base_seed: int = 0xE2000053,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSHUFLW (shuffle low words) instruction test cases.

    SSE2: pshuflw xmm0, xmm1, imm8 -> shuffle low 4 words of xmm1

    High qword is copied unchanged, low qword words are shuffled.

    Encoding: F2 0F 70 C1 imm8 (PSHUFLW xmm0, xmm1, imm8)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSHUFLW instructions
    """
    edge_vals = _xmm_interleave_edge_words()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            xmm1 = edge_vals[i]
        else:
            xmm1 = _random_xmm(rng)

        imm8 = rng.randint(0, 255)

        # F2 0F 70 C1 = PSHUFLW xmm0, xmm1, imm8
        shellcode = bytes([0xF2, 0x0F, 0x70, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"pshuflw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": 0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )
