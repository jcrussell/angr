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
]
