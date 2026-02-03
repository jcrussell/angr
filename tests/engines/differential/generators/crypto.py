"""
Cryptographic instruction generators for differential testing.

Generates test cases for:
- PCLMULQDQ: Carry-less multiplication (used in AES-GCM)
- CRC32: CRC32C checksum instruction

These test specialized crypto-related VEX operations.
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


def _pclmul_edge_values() -> list[int]:
    """
    Generate edge case XMM values for carry-less multiply.

    Focus on values that exercise the polynomial multiplication algorithm.
    """
    return [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        1,                                          # Just LSB
        1 << 63,                                    # Just MSB of low qword
        1 << 127,                                   # Just MSB of high qword
        (1 << 64) - 1,                              # Low qword all ones
        ((1 << 64) - 1) << 64,                      # High qword all ones
        0x00000000000000010000000000000001,         # 1 in both qwords
        0x80000000000000008000000000000000,         # MSB in both qwords
        0x5555555555555555AAAAAAAAAAAAAAAA,         # Alternating pattern
        0xAAAAAAAAAAAAAAAA5555555555555555,         # Inverse alternating
        0x123456789ABCDEF0FEDCBA9876543210,         # Mixed pattern
        0x0F0F0F0F0F0F0F0FF0F0F0F0F0F0F0F0,         # Nibble pattern
        # Powers of 2
        1 << 0,
        1 << 16,
        1 << 32,
        1 << 48,
        1 << 64,
        1 << 80,
        1 << 96,
        1 << 112,
    ]


# PCLMULQDQ immediate encoding:
# imm8[0] selects low (0) or high (1) qword from first operand
# imm8[4] selects low (0) or high (1) qword from second operand
# Common patterns:
#   0x00: LQLQ - Low qword * Low qword
#   0x01: HQLQ - High qword * Low qword
#   0x10: LQHQ - Low qword * High qword
#   0x11: HQHQ - High qword * High qword

PCLMUL_MODES = {
    "LQLQ": 0x00,  # Low * Low
    "HQLQ": 0x01,  # High * Low
    "LQHQ": 0x10,  # Low * High
    "HQHQ": 0x11,  # High * High
}


def generate_pclmulqdq_lqlq_tests(
    count: int = 50,
    base_seed: int = 0xE4000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCLMULQDQ tests with LQLQ mode (low qword * low qword).

    PCLMULQDQ performs carry-less multiplication of selected qwords.

    Encoding: 66 0F 3A 44 C1 00 (PCLMULQDQ xmm0, xmm1, 0x00)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCLMULQDQ LQLQ instructions
    """
    edge_vals = _pclmul_edge_values()
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

        # 66 0F 3A 44 C1 00 = PCLMULQDQ xmm0, xmm1, 0x00
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x44, 0xC1, 0x00])

        yield DifferentialTestCase(
            name=f"pclmulqdq_lqlq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pclmulqdq_hqhq_tests(
    count: int = 50,
    base_seed: int = 0xE4000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCLMULQDQ tests with HQHQ mode (high qword * high qword).

    Encoding: 66 0F 3A 44 C1 11 (PCLMULQDQ xmm0, xmm1, 0x11)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCLMULQDQ HQHQ instructions
    """
    edge_vals = _pclmul_edge_values()
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

        # 66 0F 3A 44 C1 11 = PCLMULQDQ xmm0, xmm1, 0x11
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x44, 0xC1, 0x11])

        yield DifferentialTestCase(
            name=f"pclmulqdq_hqhq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pclmulqdq_mixed_tests(
    count: int = 50,
    base_seed: int = 0xE4000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCLMULQDQ tests with mixed modes (LQHQ, HQLQ).

    Tests all combinations of qword selection.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCLMULQDQ mixed mode instructions
    """
    modes = [0x01, 0x10]  # HQLQ, LQHQ
    edge_vals = _pclmul_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(edge_vals):
            xmm0 = edge_vals[i]
            xmm1 = edge_vals[(i + 1) % len(edge_vals)]
        else:
            xmm0 = _random_xmm(rng)
            xmm1 = _random_xmm(rng)

        imm8 = modes[i % len(modes)]

        # 66 0F 3A 44 C1 imm8 = PCLMULQDQ xmm0, xmm1, imm8
        shellcode = bytes([0x66, 0x0F, 0x3A, 0x44, 0xC1, imm8])

        yield DifferentialTestCase(
            name=f"pclmulqdq_mixed_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def _crc32_edge_values() -> list[int]:
    """Generate edge case values for CRC32 testing."""
    return [
        0,
        1,
        0xFF,
        0x100,
        0xFFFF,
        0x10000,
        0xFFFFFF,
        0x1000000,
        0xFFFFFFFF,
        0x12345678,
        0xDEADBEEF,
        0xCAFEBABE,
        0x55555555,
        0xAAAAAAAA,
        # Byte patterns
        0x00000000,
        0x01010101,
        0x80808080,
        0xFF00FF00,
        0x00FF00FF,
    ]


def generate_crc32_32_tests(
    count: int = 100,
    base_seed: int = 0xE4000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CRC32 r32, r/m32 instruction test cases.

    SSE4.2: crc32 eax, ebx -> compute CRC32C of ebx, accumulating into eax

    Encoding: F2 0F 38 F1 C3 (CRC32 eax, ebx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CRC32 instructions
    """
    edge_vals = _crc32_edge_values()
    edge_idx = 0

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if edge_idx < len(edge_vals) * len(edge_vals):
            eax = edge_vals[edge_idx // len(edge_vals)]  # Initial CRC
            ebx = edge_vals[edge_idx % len(edge_vals)]   # Data to hash
            edge_idx += 1
        else:
            eax = _random_value(rng, 32)
            ebx = _random_value(rng, 32)

        # F2 0F 38 F1 C3 = CRC32 eax, ebx
        shellcode = bytes([0xF2, 0x0F, 0x38, 0xF1, 0xC3])

        yield DifferentialTestCase(
            name=f"crc32_32_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_crc32_8_tests(
    count: int = 50,
    base_seed: int = 0xE4000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CRC32 r32, r/m8 instruction test cases.

    SSE4.2: crc32 eax, bl -> compute CRC32C of byte bl, accumulating into eax

    Encoding: F2 0F 38 F0 C3 (CRC32 eax, bl)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CRC32 byte instructions
    """
    # Byte values
    byte_vals = list(range(0, 256, 16)) + [0, 1, 127, 128, 255]
    crc_vals = _crc32_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(byte_vals):
            ebx = byte_vals[i]
            eax = crc_vals[i % len(crc_vals)]
        else:
            ebx = rng.randint(0, 255)
            eax = _random_value(rng, 32)

        # F2 0F 38 F0 C3 = CRC32 eax, bl
        shellcode = bytes([0xF2, 0x0F, 0x38, 0xF0, 0xC3])

        yield DifferentialTestCase(
            name=f"crc32_8_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


def generate_crc32_16_tests(
    count: int = 50,
    base_seed: int = 0xE4000013,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CRC32 r32, r/m16 instruction test cases.

    SSE4.2: crc32 eax, bx -> compute CRC32C of word bx, accumulating into eax

    Encoding: 66 F2 0F 38 F1 C3 (CRC32 eax, bx)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CRC32 word instructions
    """
    word_vals = [0, 1, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0xABCD]
    crc_vals = _crc32_edge_values()

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(word_vals):
            ebx = word_vals[i]
            eax = crc_vals[i % len(crc_vals)]
        else:
            ebx = rng.randint(0, 0xFFFF)
            eax = _random_value(rng, 32)

        # 66 F2 0F 38 F1 C3 = CRC32 eax, bx (operand-size prefix)
        shellcode = bytes([0x66, 0xF2, 0x0F, 0x38, 0xF1, 0xC3])

        yield DifferentialTestCase(
            name=f"crc32_16_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"eax": eax, "ebx": ebx},
            compare_regs=["eax"],
            seed=seed,
            arch=arch,
        )


# AMD64 variants

def generate_crc32_64_tests(
    count: int = 50,
    base_seed: int = 0xE4000021,
    arch: str = "amd64"
) -> Iterator[DifferentialTestCase]:
    """
    Generate CRC32 r64, r/m64 instruction test cases.

    x86-64: crc32 rax, rbx -> compute CRC32C of qword rbx, accumulating into rax

    Encoding: F2 48 0F 38 F1 C3 (REX.W CRC32 rax, rbx)

    Note: The CRC accumulator is still 32 bits, stored in low 32 bits of rax.

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for CRC32 qword instructions
    """
    qword_vals = [
        0,
        1,
        (1 << 64) - 1,
        0x123456789ABCDEF0,
        0xFEDCBA9876543210,
        0x8000000000000000,
        0x7FFFFFFFFFFFFFFF,
    ]

    for i in range(count):
        seed = _make_seed(base_seed, i)
        rng = random.Random(seed)

        if i < len(qword_vals):
            rbx = qword_vals[i]
            rax = rng.randint(0, 0xFFFFFFFF)  # CRC is 32-bit
        else:
            rbx = rng.randint(0, (1 << 64) - 1)
            rax = rng.randint(0, 0xFFFFFFFF)

        # F2 48 0F 38 F1 C3 = CRC32 rax, rbx (REX.W)
        shellcode = bytes([0xF2, 0x48, 0x0F, 0x38, 0xF1, 0xC3])

        yield DifferentialTestCase(
            name=f"crc32_64_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"rax": rax, "rbx": rbx},
            compare_regs=["rax"],
            seed=seed,
            arch=arch,
        )
