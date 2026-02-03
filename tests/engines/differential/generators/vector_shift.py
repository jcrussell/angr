"""
Vector shift instruction generators for differential testing.

Generates test cases for packed shift-by-immediate operations:
- PSLLW/D/Q: Packed shift left logical (word/dword/qword)
- PSRLW/D/Q: Packed shift right logical
- PSRAW/D: Packed shift right arithmetic (word/dword)

These test the VShlN, VShrN, VSarN VEX operations.
"""
from __future__ import annotations

import random
from typing import Iterator

from ..harness import DifferentialTestCase


def _make_seed(base: int, index: int) -> int:
    """Create deterministic seed from base seed and index."""
    return (base * 1000003 + index) & 0xFFFFFFFF


def _random_xmm(rng: random.Random) -> int:
    """Generate random 128-bit XMM value."""
    return rng.randint(0, (1 << 128) - 1)


def _shift_amounts(element_width: int) -> list[int]:
    """
    Generate interesting shift amounts for a given element width.

    Args:
        element_width: Width in bits of each vector element (8, 16, 32, 64)

    Returns:
        List of shift amounts to test
    """
    return [
        0,                      # No shift
        1,                      # Shift by 1
        element_width - 1,      # Max useful shift
        element_width,          # Shift by element width (zeros out for logical)
        element_width + 1,      # Shift by more than width
        element_width // 2,     # Half width
        7,                      # Common value
        8,                      # Byte boundary
        15,                     # For word shifts
        16,                     # For dword shifts
        31,                     # For dword shifts
        63,                     # For qword shifts
        127,                    # Large shift
    ]


def _xmm_shift_edge_values_words() -> list[int]:
    """Generate edge case XMM values for word shift operations."""
    return [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x00010001000100010001000100010001,         # All 1s in words
        0x7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF,         # Max positive signed
        0x80008000800080008000800080008000,         # Min signed (MSB set)
        0xFFFF0000FFFF0000FFFF0000FFFF0000,         # Alternating words
        0x0001000200040008001000200040008,          # Powers of 2
        0x8001800180018001800180018001,             # MSB and LSB set
    ]


def _xmm_shift_edge_values_dwords() -> list[int]:
    """Generate edge case XMM values for dword shift operations."""
    return [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x00000001000000010000000100000001,         # All 1s in dwords
        0x7FFFFFFF7FFFFFFF7FFFFFFF7FFFFFFF,         # Max positive signed
        0x80000000800000008000000080000000,         # Min signed (MSB set)
        0xFFFFFFFF00000000FFFFFFFF00000000,         # Alternating dwords
        0x00000001000000020000000400000008,         # Powers of 2
        0x80000001800000018000000180000001,         # MSB and LSB set
    ]


def _xmm_shift_edge_values_qwords() -> list[int]:
    """Generate edge case XMM values for qword shift operations."""
    return [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x00000000000000010000000000000001,         # All 1s in qwords
        0x7FFFFFFFFFFFFFFF7FFFFFFFFFFFFFFF,         # Max positive signed
        0x80000000000000008000000000000000,         # Min signed (MSB set)
        0xFFFFFFFFFFFFFFFF0000000000000000,         # High qword set
        0x00000000000000010000000000000002,         # Low bits set
        0x80000000000000018000000000000001,         # MSB and LSB set
    ]


# Word shift generators (16-bit elements)

def generate_psllw_tests(
    count: int = 100,
    base_seed: int = 0xE1000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSLLW (packed shift left logical words) instruction test cases.

    SSE2: psllw xmm0, imm8 -> shift each 16-bit word left by imm8

    Encoding: 66 0F 71 /6 ib (PSLLW xmm, imm8)
    For xmm0: 66 0F 71 F0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSLLW instructions
    """
    edge_vals = _xmm_shift_edge_values_words()
    shifts = _shift_amounts(16)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        # Cycle through edge values and shifts first
        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        # Clamp shift to valid imm8 range
        shift_amount = shift_amount & 0xFF

        # 66 0F 71 F0 = PSLLW xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x71, 0xF0, shift_amount])

        yield DifferentialTestCase(
            name=f"psllw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psrlw_tests(
    count: int = 100,
    base_seed: int = 0xE1000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSRLW (packed shift right logical words) instruction test cases.

    SSE2: psrlw xmm0, imm8 -> shift each 16-bit word right by imm8 (logical)

    Encoding: 66 0F 71 /2 ib (PSRLW xmm, imm8)
    For xmm0: 66 0F 71 D0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSRLW instructions
    """
    edge_vals = _xmm_shift_edge_values_words()
    shifts = _shift_amounts(16)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 71 D0 = PSRLW xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x71, 0xD0, shift_amount])

        yield DifferentialTestCase(
            name=f"psrlw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psraw_tests(
    count: int = 100,
    base_seed: int = 0xE1000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSRAW (packed shift right arithmetic words) instruction test cases.

    SSE2: psraw xmm0, imm8 -> arithmetic shift each 16-bit word right by imm8

    Encoding: 66 0F 71 /4 ib (PSRAW xmm, imm8)
    For xmm0: 66 0F 71 E0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSRAW instructions
    """
    edge_vals = _xmm_shift_edge_values_words()
    shifts = _shift_amounts(16)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 71 E0 = PSRAW xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x71, 0xE0, shift_amount])

        yield DifferentialTestCase(
            name=f"psraw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Dword shift generators (32-bit elements)

def generate_pslld_tests(
    count: int = 100,
    base_seed: int = 0xE1000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSLLD (packed shift left logical dwords) instruction test cases.

    SSE2: pslld xmm0, imm8 -> shift each 32-bit dword left by imm8

    Encoding: 66 0F 72 /6 ib (PSLLD xmm, imm8)
    For xmm0: 66 0F 72 F0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSLLD instructions
    """
    edge_vals = _xmm_shift_edge_values_dwords()
    shifts = _shift_amounts(32)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 72 F0 = PSLLD xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x72, 0xF0, shift_amount])

        yield DifferentialTestCase(
            name=f"pslld_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psrld_tests(
    count: int = 100,
    base_seed: int = 0xE1000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSRLD (packed shift right logical dwords) instruction test cases.

    SSE2: psrld xmm0, imm8 -> shift each 32-bit dword right by imm8 (logical)

    Encoding: 66 0F 72 /2 ib (PSRLD xmm, imm8)
    For xmm0: 66 0F 72 D0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSRLD instructions
    """
    edge_vals = _xmm_shift_edge_values_dwords()
    shifts = _shift_amounts(32)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 72 D0 = PSRLD xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x72, 0xD0, shift_amount])

        yield DifferentialTestCase(
            name=f"psrld_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psrad_tests(
    count: int = 100,
    base_seed: int = 0xE1000013,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSRAD (packed shift right arithmetic dwords) instruction test cases.

    SSE2: psrad xmm0, imm8 -> arithmetic shift each 32-bit dword right by imm8

    Encoding: 66 0F 72 /4 ib (PSRAD xmm, imm8)
    For xmm0: 66 0F 72 E0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSRAD instructions
    """
    edge_vals = _xmm_shift_edge_values_dwords()
    shifts = _shift_amounts(32)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 72 E0 = PSRAD xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x72, 0xE0, shift_amount])

        yield DifferentialTestCase(
            name=f"psrad_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Qword shift generators (64-bit elements)

def generate_psllq_tests(
    count: int = 100,
    base_seed: int = 0xE1000021,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSLLQ (packed shift left logical qwords) instruction test cases.

    SSE2: psllq xmm0, imm8 -> shift each 64-bit qword left by imm8

    Encoding: 66 0F 73 /6 ib (PSLLQ xmm, imm8)
    For xmm0: 66 0F 73 F0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSLLQ instructions
    """
    edge_vals = _xmm_shift_edge_values_qwords()
    shifts = _shift_amounts(64)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 73 F0 = PSLLQ xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x73, 0xF0, shift_amount])

        yield DifferentialTestCase(
            name=f"psllq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psrlq_tests(
    count: int = 100,
    base_seed: int = 0xE1000022,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSRLQ (packed shift right logical qwords) instruction test cases.

    SSE2: psrlq xmm0, imm8 -> shift each 64-bit qword right by imm8 (logical)

    Encoding: 66 0F 73 /2 ib (PSRLQ xmm, imm8)
    For xmm0: 66 0F 73 D0 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSRLQ instructions
    """
    edge_vals = _xmm_shift_edge_values_qwords()
    shifts = _shift_amounts(64)
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(shifts):
            xmm0 = edge_vals[test_idx // len(shifts)]
            shift_amount = shifts[test_idx % len(shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_amount = rng.randint(0, 127)

        shift_amount = shift_amount & 0xFF

        # 66 0F 73 D0 = PSRLQ xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x73, 0xD0, shift_amount])

        yield DifferentialTestCase(
            name=f"psrlq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# PSLLDQ/PSRLDQ - Shift entire register by bytes

def generate_pslldq_tests(
    count: int = 50,
    base_seed: int = 0xE1000031,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSLLDQ (shift double quadword left) instruction test cases.

    SSE2: pslldq xmm0, imm8 -> shift entire 128-bit register left by imm8 bytes

    Encoding: 66 0F 73 /7 ib (PSLLDQ xmm, imm8)
    For xmm0: 66 0F 73 F8 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSLLDQ instructions
    """
    edge_vals = _xmm_shift_edge_values_qwords()
    byte_shifts = [0, 1, 2, 4, 7, 8, 15, 16, 17, 32]
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(byte_shifts):
            xmm0 = edge_vals[test_idx // len(byte_shifts)]
            shift_bytes = byte_shifts[test_idx % len(byte_shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_bytes = rng.randint(0, 32)

        shift_bytes = shift_bytes & 0xFF

        # 66 0F 73 F8 = PSLLDQ xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x73, 0xF8, shift_bytes])

        yield DifferentialTestCase(
            name=f"pslldq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psrldq_tests(
    count: int = 50,
    base_seed: int = 0xE1000032,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSRLDQ (shift double quadword right) instruction test cases.

    SSE2: psrldq xmm0, imm8 -> shift entire 128-bit register right by imm8 bytes

    Encoding: 66 0F 73 /3 ib (PSRLDQ xmm, imm8)
    For xmm0: 66 0F 73 D8 imm8

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSRLDQ instructions
    """
    edge_vals = _xmm_shift_edge_values_qwords()
    byte_shifts = [0, 1, 2, 4, 7, 8, 15, 16, 17, 32]
    test_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if test_idx < len(edge_vals) * len(byte_shifts):
            xmm0 = edge_vals[test_idx // len(byte_shifts)]
            shift_bytes = byte_shifts[test_idx % len(byte_shifts)]
            test_idx += 1
        else:
            xmm0 = _random_xmm(rng)
            shift_bytes = rng.randint(0, 32)

        shift_bytes = shift_bytes & 0xFF

        # 66 0F 73 D8 = PSRLDQ xmm0, imm8
        shellcode = bytes([0x66, 0x0F, 0x73, 0xD8, shift_bytes])

        yield DifferentialTestCase(
            name=f"psrldq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )
