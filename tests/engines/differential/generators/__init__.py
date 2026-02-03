"""
Shellcode generators for differential testing.

Provides functions to generate test cases for different operation categories.
"""
from __future__ import annotations

from .arithmetic import generate_add_tests, generate_sub_tests, generate_mul_tests, generate_div_tests
from .bitwise import generate_and_tests, generate_or_tests, generate_xor_tests, generate_shift_tests
from .memory import generate_mov_tests, generate_push_pop_tests, generate_lea_tests
from .floating_point import (
    generate_addss_tests, generate_subss_tests, generate_mulss_tests, generate_divss_tests,
    generate_cvtsi2ss_tests, generate_cvtss2si_tests, generate_cvttss2si_tests,
    generate_ucomiss_tests, generate_comiss_tests,
    generate_sqrtss_tests, generate_movss_tests,
    generate_addsd_tests, generate_subsd_tests, generate_mulsd_tests, generate_divsd_tests,
    float_to_xmm, xmm_to_float, double_to_xmm,
)
from .comparison import (
    generate_cmp_setz_tests, generate_cmp_setnz_tests,
    generate_cmp_setl_tests, generate_cmp_setg_tests,
    generate_cmp_setb_tests, generate_cmp_seta_tests,
    generate_test_setz_tests, generate_test_sets_tests,
)
from .bit_manipulation import (
    generate_popcnt_tests, generate_bsf_tests, generate_bsr_tests,
    generate_lzcnt_tests, generate_tzcnt_tests,
)
from .vector_simd import (
    generate_paddb_tests, generate_paddw_tests, generate_paddd_tests, generate_paddq_tests,
    generate_psubb_tests, generate_psubw_tests, generate_psubd_tests, generate_psubq_tests,
    generate_pcmpeqb_tests, generate_pcmpeqw_tests, generate_pcmpeqd_tests,
    generate_pcmpgtb_tests, generate_pcmpgtw_tests, generate_pcmpgtd_tests,
    generate_pand_tests, generate_por_tests, generate_pxor_tests,
    generate_pmullw_tests, generate_pmulld_tests,
)
from .advanced_arithmetic import (
    generate_idiv_tests, generate_adc_tests, generate_sbb_tests,
    generate_shld_tests, generate_shrd_tests,
)
from .fp_extended import (
    generate_sqrtsd_tests,
    generate_cvtsd2ss_tests, generate_cvtss2sd_tests,
    generate_cvtsi2sd_tests, generate_cvtsd2si_tests, generate_cvttsd2si_tests,
    generate_ucomisd_tests,
    generate_maxss_tests, generate_minss_tests, generate_maxsd_tests, generate_minsd_tests,
)

# New generators for extended coverage
from .divmod import (
    generate_div64to32_unsigned_tests, generate_div64to32_signed_tests,
    generate_div128to64_unsigned_tests,
)
from .mulhi import (
    generate_mulhi_unsigned_tests, generate_mulhi_signed_tests,
    generate_mulhi_unsigned_64_tests, generate_mulhi_signed_64_tests,
)
from .vector_shift import (
    generate_psllw_tests, generate_psrlw_tests, generate_psraw_tests,
    generate_pslld_tests, generate_psrld_tests, generate_psrad_tests,
    generate_psllq_tests, generate_psrlq_tests,
    generate_pslldq_tests, generate_psrldq_tests,
)
from .vector_misc import (
    generate_movd_to_xmm_tests, generate_movq_to_xmm_tests,
    generate_punpcklbw_tests, generate_punpckhbw_tests,
    generate_punpcklwd_tests, generate_punpckhwd_tests,
    generate_punpckldq_tests, generate_punpckhdq_tests,
    generate_punpcklqdq_tests, generate_punpckhqdq_tests,
    generate_pshufd_tests, generate_pshufhw_tests, generate_pshuflw_tests,
)
from .fp_rounding import (
    generate_roundss_tests, generate_roundsd_tests,
    generate_roundps_tests, generate_roundpd_tests,
    generate_floorss_tests, generate_ceilss_tests, generate_truncss_tests,
    generate_floorsd_tests, generate_ceilsd_tests, generate_truncsd_tests,
)
from .crypto import (
    generate_pclmulqdq_lqlq_tests, generate_pclmulqdq_hqhq_tests, generate_pclmulqdq_mixed_tests,
    generate_crc32_32_tests, generate_crc32_8_tests, generate_crc32_16_tests,
    generate_crc32_64_tests,
)

__all__ = [
    # Arithmetic
    "generate_add_tests",
    "generate_sub_tests",
    "generate_mul_tests",
    "generate_div_tests",
    # Bitwise
    "generate_and_tests",
    "generate_or_tests",
    "generate_xor_tests",
    "generate_shift_tests",
    # Memory
    "generate_mov_tests",
    "generate_push_pop_tests",
    "generate_lea_tests",
    # Floating point (single precision)
    "generate_addss_tests",
    "generate_subss_tests",
    "generate_mulss_tests",
    "generate_divss_tests",
    "generate_cvtsi2ss_tests",
    "generate_cvtss2si_tests",
    "generate_cvttss2si_tests",
    "generate_ucomiss_tests",
    "generate_comiss_tests",
    "generate_sqrtss_tests",
    "generate_movss_tests",
    # Floating point (double precision)
    "generate_addsd_tests",
    "generate_subsd_tests",
    "generate_mulsd_tests",
    "generate_divsd_tests",
    # FP utilities
    "float_to_xmm",
    "xmm_to_float",
    "double_to_xmm",
    # Comparison (CMP, TEST, SETcc)
    "generate_cmp_setz_tests",
    "generate_cmp_setnz_tests",
    "generate_cmp_setl_tests",
    "generate_cmp_setg_tests",
    "generate_cmp_setb_tests",
    "generate_cmp_seta_tests",
    "generate_test_setz_tests",
    "generate_test_sets_tests",
    # Bit manipulation (POPCNT, BSF, BSR, LZCNT, TZCNT)
    "generate_popcnt_tests",
    "generate_bsf_tests",
    "generate_bsr_tests",
    "generate_lzcnt_tests",
    "generate_tzcnt_tests",
    # Vector SIMD (packed integer operations)
    "generate_paddb_tests",
    "generate_paddw_tests",
    "generate_paddd_tests",
    "generate_paddq_tests",
    "generate_psubb_tests",
    "generate_psubw_tests",
    "generate_psubd_tests",
    "generate_psubq_tests",
    "generate_pcmpeqb_tests",
    "generate_pcmpeqw_tests",
    "generate_pcmpeqd_tests",
    "generate_pcmpgtb_tests",
    "generate_pcmpgtw_tests",
    "generate_pcmpgtd_tests",
    "generate_pand_tests",
    "generate_por_tests",
    "generate_pxor_tests",
    "generate_pmullw_tests",
    "generate_pmulld_tests",
    # Advanced arithmetic (IDIV, ADC, SBB, SHLD, SHRD)
    "generate_idiv_tests",
    "generate_adc_tests",
    "generate_sbb_tests",
    "generate_shld_tests",
    "generate_shrd_tests",
    # Extended FP (SQRTSD, conversions, min/max)
    "generate_sqrtsd_tests",
    "generate_cvtsd2ss_tests",
    "generate_cvtss2sd_tests",
    "generate_cvtsi2sd_tests",
    "generate_cvtsd2si_tests",
    "generate_cvttsd2si_tests",
    "generate_ucomisd_tests",
    "generate_maxss_tests",
    "generate_minss_tests",
    "generate_maxsd_tests",
    "generate_minsd_tests",

    # --- NEW GENERATORS ---

    # DivMod64to32 (64-bit division)
    "generate_div64to32_unsigned_tests",
    "generate_div64to32_signed_tests",
    "generate_div128to64_unsigned_tests",

    # MulHi (high multiplication)
    "generate_mulhi_unsigned_tests",
    "generate_mulhi_signed_tests",
    "generate_mulhi_unsigned_64_tests",
    "generate_mulhi_signed_64_tests",

    # Vector shifts (VShlN, VShrN, VSarN)
    "generate_psllw_tests",
    "generate_psrlw_tests",
    "generate_psraw_tests",
    "generate_pslld_tests",
    "generate_psrld_tests",
    "generate_psrad_tests",
    "generate_psllq_tests",
    "generate_psrlq_tests",
    "generate_pslldq_tests",
    "generate_psrldq_tests",

    # Vector misc (SetV128, interleave, shuffle)
    "generate_movd_to_xmm_tests",
    "generate_movq_to_xmm_tests",
    "generate_punpcklbw_tests",
    "generate_punpckhbw_tests",
    "generate_punpcklwd_tests",
    "generate_punpckhwd_tests",
    "generate_punpckldq_tests",
    "generate_punpckhdq_tests",
    "generate_punpcklqdq_tests",
    "generate_punpckhqdq_tests",
    "generate_pshufd_tests",
    "generate_pshufhw_tests",
    "generate_pshuflw_tests",

    # FP rounding (ROUNDSS, ROUNDSD, etc.)
    "generate_roundss_tests",
    "generate_roundsd_tests",
    "generate_roundps_tests",
    "generate_roundpd_tests",
    "generate_floorss_tests",
    "generate_ceilss_tests",
    "generate_truncss_tests",
    "generate_floorsd_tests",
    "generate_ceilsd_tests",
    "generate_truncsd_tests",

    # Crypto (PCLMULQDQ, CRC32)
    "generate_pclmulqdq_lqlq_tests",
    "generate_pclmulqdq_hqhq_tests",
    "generate_pclmulqdq_mixed_tests",
    "generate_crc32_32_tests",
    "generate_crc32_8_tests",
    "generate_crc32_16_tests",
    "generate_crc32_64_tests",
]
