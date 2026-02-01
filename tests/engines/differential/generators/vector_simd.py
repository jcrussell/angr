"""
Vector SIMD instruction generators for differential testing.

Generates test cases for packed integer operations:
- PADDB/W/D/Q: Packed add bytes/words/dwords/qwords
- PSUBB/W/D/Q: Packed subtract bytes/words/dwords/qwords
- PCMPEQB/W/D: Packed compare equal
- PCMPGTB/W/D: Packed compare greater than (signed)
- PAND/POR/PXOR: Packed bitwise operations
- PMULLW/PMULLD: Packed multiply low
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


def _xmm_edge_values_bytes() -> list[int]:
    """Generate edge case XMM values for byte operations."""
    # 16 bytes per XMM register
    values = [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x01010101010101010101010101010101,         # All 1s
        0x7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F,         # Max signed bytes
        0x80808080808080808080808080808080,         # Min signed bytes
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,         # All 0xFF
        0x0102030405060708090A0B0C0D0E0F10,         # Incrementing
        0x100F0E0D0C0B0A090807060504030201,         # Decrementing
    ]
    return values


def _xmm_edge_values_words() -> list[int]:
    """Generate edge case XMM values for word (16-bit) operations."""
    # 8 words per XMM register
    values = [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x00010001000100010001000100010001,         # All 1s
        0x7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF,         # Max signed words
        0x80008000800080008000800080008000,         # Min signed words
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,         # All 0xFFFF
        0x00010002000300040005000600070008,         # Incrementing
    ]
    return values


def _xmm_edge_values_dwords() -> list[int]:
    """Generate edge case XMM values for dword (32-bit) operations."""
    # 4 dwords per XMM register
    values = [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x00000001000000010000000100000001,         # All 1s
        0x7FFFFFFF7FFFFFFF7FFFFFFF7FFFFFFF,         # Max signed dwords
        0x80000000800000008000000080000000,         # Min signed dwords
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,         # All 0xFFFFFFFF
        0x00000001000000020000000300000004,         # Incrementing
    ]
    return values


def _xmm_edge_values_qwords() -> list[int]:
    """Generate edge case XMM values for qword (64-bit) operations."""
    # 2 qwords per XMM register
    values = [
        0,                                          # All zeros
        (1 << 128) - 1,                             # All ones
        0x00000000000000010000000000000001,         # All 1s
        0x7FFFFFFFFFFFFFFF7FFFFFFFFFFFFFFF,         # Max signed qwords
        0x80000000000000008000000000000000,         # Min signed qwords
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,         # All max
        0x00000000000000010000000000000002,         # Incrementing
    ]
    return values


# PADD - Packed Add

def generate_paddb_tests(
    count: int = 100,
    base_seed: int = 0xD0000001,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PADDB (packed add bytes) instruction test cases.

    x86 SSE2: paddb xmm0, xmm1 -> xmm0 = xmm0 + xmm1 (16x8-bit add)

    Encoding: 66 0F FC C1 (paddb xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PADDB instructions
    """
    # 66 0f fc c1 = paddb xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xFC, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"paddb_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_paddw_tests(
    count: int = 100,
    base_seed: int = 0xD0000002,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PADDW (packed add words) instruction test cases.

    x86 SSE2: paddw xmm0, xmm1 -> xmm0 = xmm0 + xmm1 (8x16-bit add)

    Encoding: 66 0F FD C1 (paddw xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PADDW instructions
    """
    # 66 0f fd c1 = paddw xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xFD, 0xC1])

    edge_vals = _xmm_edge_values_words()
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
            name=f"paddw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_paddd_tests(
    count: int = 100,
    base_seed: int = 0xD0000003,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PADDD (packed add dwords) instruction test cases.

    x86 SSE2: paddd xmm0, xmm1 -> xmm0 = xmm0 + xmm1 (4x32-bit add)

    Encoding: 66 0F FE C1 (paddd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PADDD instructions
    """
    # 66 0f fe c1 = paddd xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xFE, 0xC1])

    edge_vals = _xmm_edge_values_dwords()
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
            name=f"paddd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_paddq_tests(
    count: int = 100,
    base_seed: int = 0xD0000004,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PADDQ (packed add qwords) instruction test cases.

    x86 SSE2: paddq xmm0, xmm1 -> xmm0 = xmm0 + xmm1 (2x64-bit add)

    Encoding: 66 0F D4 C1 (paddq xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PADDQ instructions
    """
    # 66 0f d4 c1 = paddq xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xD4, 0xC1])

    edge_vals = _xmm_edge_values_qwords()
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
            name=f"paddq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# PSUB - Packed Subtract

def generate_psubb_tests(
    count: int = 100,
    base_seed: int = 0xD0000011,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSUBB (packed subtract bytes) instruction test cases.

    x86 SSE2: psubb xmm0, xmm1 -> xmm0 = xmm0 - xmm1 (16x8-bit sub)

    Encoding: 66 0F F8 C1 (psubb xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSUBB instructions
    """
    # 66 0f f8 c1 = psubb xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xF8, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"psubb_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psubw_tests(
    count: int = 100,
    base_seed: int = 0xD0000012,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSUBW (packed subtract words) instruction test cases.

    x86 SSE2: psubw xmm0, xmm1 -> xmm0 = xmm0 - xmm1 (8x16-bit sub)

    Encoding: 66 0F F9 C1 (psubw xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSUBW instructions
    """
    # 66 0f f9 c1 = psubw xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xF9, 0xC1])

    edge_vals = _xmm_edge_values_words()
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
            name=f"psubw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psubd_tests(
    count: int = 100,
    base_seed: int = 0xD0000013,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSUBD (packed subtract dwords) instruction test cases.

    x86 SSE2: psubd xmm0, xmm1 -> xmm0 = xmm0 - xmm1 (4x32-bit sub)

    Encoding: 66 0F FA C1 (psubd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSUBD instructions
    """
    # 66 0f fa c1 = psubd xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xFA, 0xC1])

    edge_vals = _xmm_edge_values_dwords()
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
            name=f"psubd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_psubq_tests(
    count: int = 100,
    base_seed: int = 0xD0000014,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PSUBQ (packed subtract qwords) instruction test cases.

    x86 SSE2: psubq xmm0, xmm1 -> xmm0 = xmm0 - xmm1 (2x64-bit sub)

    Encoding: 66 0F FB C1 (psubq xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PSUBQ instructions
    """
    # 66 0f fb c1 = psubq xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xFB, 0xC1])

    edge_vals = _xmm_edge_values_qwords()
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
            name=f"psubq_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# PCMPEQ - Packed Compare Equal

def generate_pcmpeqb_tests(
    count: int = 100,
    base_seed: int = 0xD0000021,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCMPEQB (packed compare equal bytes) instruction test cases.

    x86 SSE2: pcmpeqb xmm0, xmm1 -> xmm0[i] = (xmm0[i] == xmm1[i]) ? 0xFF : 0x00

    Encoding: 66 0F 74 C1 (pcmpeqb xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCMPEQB instructions
    """
    # 66 0f 74 c1 = pcmpeqb xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x74, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"pcmpeqb_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pcmpeqw_tests(
    count: int = 100,
    base_seed: int = 0xD0000022,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCMPEQW (packed compare equal words) instruction test cases.

    x86 SSE2: pcmpeqw xmm0, xmm1 -> xmm0[i] = (xmm0[i] == xmm1[i]) ? 0xFFFF : 0x0000

    Encoding: 66 0F 75 C1 (pcmpeqw xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCMPEQW instructions
    """
    # 66 0f 75 c1 = pcmpeqw xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x75, 0xC1])

    edge_vals = _xmm_edge_values_words()
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
            name=f"pcmpeqw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pcmpeqd_tests(
    count: int = 100,
    base_seed: int = 0xD0000023,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCMPEQD (packed compare equal dwords) instruction test cases.

    x86 SSE2: pcmpeqd xmm0, xmm1 -> xmm0[i] = (xmm0[i] == xmm1[i]) ? 0xFFFFFFFF : 0x00000000

    Encoding: 66 0F 76 C1 (pcmpeqd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCMPEQD instructions
    """
    # 66 0f 76 c1 = pcmpeqd xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x76, 0xC1])

    edge_vals = _xmm_edge_values_dwords()
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
            name=f"pcmpeqd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# PCMPGT - Packed Compare Greater Than (signed)

def generate_pcmpgtb_tests(
    count: int = 100,
    base_seed: int = 0xD0000031,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCMPGTB (packed compare greater than bytes, signed) test cases.

    x86 SSE2: pcmpgtb xmm0, xmm1 -> xmm0[i] = (xmm0[i] > xmm1[i]) ? 0xFF : 0x00 (signed)

    Encoding: 66 0F 64 C1 (pcmpgtb xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCMPGTB instructions
    """
    # 66 0f 64 c1 = pcmpgtb xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x64, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"pcmpgtb_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pcmpgtw_tests(
    count: int = 100,
    base_seed: int = 0xD0000032,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCMPGTW (packed compare greater than words, signed) test cases.

    x86 SSE2: pcmpgtw xmm0, xmm1 -> xmm0[i] = (xmm0[i] > xmm1[i]) ? 0xFFFF : 0x0000 (signed)

    Encoding: 66 0F 65 C1 (pcmpgtw xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCMPGTW instructions
    """
    # 66 0f 65 c1 = pcmpgtw xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x65, 0xC1])

    edge_vals = _xmm_edge_values_words()
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
            name=f"pcmpgtw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pcmpgtd_tests(
    count: int = 100,
    base_seed: int = 0xD0000033,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PCMPGTD (packed compare greater than dwords, signed) test cases.

    x86 SSE2: pcmpgtd xmm0, xmm1 -> xmm0[i] = (xmm0[i] > xmm1[i]) ? 0xFFFFFFFF : 0x00000000 (signed)

    Encoding: 66 0F 66 C1 (pcmpgtd xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PCMPGTD instructions
    """
    # 66 0f 66 c1 = pcmpgtd xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x66, 0xC1])

    edge_vals = _xmm_edge_values_dwords()
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
            name=f"pcmpgtd_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Packed Bitwise Operations

def generate_pand_tests(
    count: int = 100,
    base_seed: int = 0xD0000041,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PAND (packed AND) instruction test cases.

    x86 SSE2: pand xmm0, xmm1 -> xmm0 = xmm0 AND xmm1

    Encoding: 66 0F DB C1 (pand xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PAND instructions
    """
    # 66 0f db c1 = pand xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xDB, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"pand_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_por_tests(
    count: int = 100,
    base_seed: int = 0xD0000042,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate POR (packed OR) instruction test cases.

    x86 SSE2: por xmm0, xmm1 -> xmm0 = xmm0 OR xmm1

    Encoding: 66 0F EB C1 (por xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for POR instructions
    """
    # 66 0f eb c1 = por xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xEB, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"por_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pxor_tests(
    count: int = 100,
    base_seed: int = 0xD0000043,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PXOR (packed XOR) instruction test cases.

    x86 SSE2: pxor xmm0, xmm1 -> xmm0 = xmm0 XOR xmm1

    Encoding: 66 0F EF C1 (pxor xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PXOR instructions
    """
    # 66 0f ef c1 = pxor xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xEF, 0xC1])

    edge_vals = _xmm_edge_values_bytes()
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
            name=f"pxor_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


# Packed Multiply

def generate_pmullw_tests(
    count: int = 100,
    base_seed: int = 0xD0000051,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PMULLW (packed multiply low words) instruction test cases.

    x86 SSE2: pmullw xmm0, xmm1 -> xmm0[i] = (xmm0[i] * xmm1[i])[15:0]

    Encoding: 66 0F D5 C1 (pmullw xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PMULLW instructions
    """
    # 66 0f d5 c1 = pmullw xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0xD5, 0xC1])

    edge_vals = _xmm_edge_values_words()
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
            name=f"pmullw_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )


def generate_pmulld_tests(
    count: int = 100,
    base_seed: int = 0xD0000052,
    arch: str = "x86"
) -> Iterator[DifferentialTestCase]:
    """
    Generate PMULLD (packed multiply low dwords) instruction test cases.

    x86 SSE4.1: pmulld xmm0, xmm1 -> xmm0[i] = (xmm0[i] * xmm1[i])[31:0]

    Encoding: 66 0F 38 40 C1 (pmulld xmm0, xmm1)

    Args:
        count: Number of test cases to generate
        base_seed: Base seed for reproducibility
        arch: Target architecture

    Yields:
        DifferentialTestCase objects for PMULLD instructions
    """
    # 66 0f 38 40 c1 = pmulld xmm0, xmm1
    shellcode = bytes([0x66, 0x0F, 0x38, 0x40, 0xC1])

    edge_vals = _xmm_edge_values_dwords()
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
            name=f"pmulld_0x{seed:08x}",
            shellcode=shellcode,
            initial_regs={"xmm0": xmm0, "xmm1": xmm1},
            compare_regs=["xmm0"],
            seed=seed,
            arch=arch,
        )
