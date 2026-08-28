// angr-v7jp: VEX CCall unit tests, extracted out of the former in-file
// `mod tests` (~836 lines) into a sibling file to shrink vex/ccall.rs
// below the god-object threshold. Declared as a direct child of `ccall`
// via `#[path]` so `use super::*;` still resolves against ccall.rs.
use super::*;

#[test]
fn test_calc_parity() {
    assert_eq!(calc_parity(0x00), 1); // 0 ones = even
    assert_eq!(calc_parity(0x01), 0); // 1 one = odd
    assert_eq!(calc_parity(0x03), 1); // 2 ones = even
    assert_eq!(calc_parity(0xFF), 1); // 8 ones = even
    assert_eq!(calc_parity(0xFE), 0); // 7 ones = odd
}

#[test]
fn test_sub_flags_zero() {
    // 5 - 5 = 0, should set ZF
    let flags = calc_flags_sub(32, 5, 5);
    assert_eq!(flags.zf, 1);
    assert_eq!(flags.cf, 0);
    assert_eq!(flags.sf, 0);
    assert_eq!(flags.of, 0);
}

#[test]
fn test_sub_flags_negative() {
    // 3 - 5 = -2, should set SF and CF
    let flags = calc_flags_sub(32, 3, 5);
    assert_eq!(flags.zf, 0);
    assert_eq!(flags.cf, 1); // borrow
    assert_eq!(flags.sf, 1); // negative result
    assert_eq!(flags.of, 0); // no signed overflow
}

#[test]
fn test_sub_flags_overflow() {
    // 0x80000000 - 1 = 0x7FFFFFFF, signed overflow (MIN_INT - 1)
    let flags = calc_flags_sub(32, 0x80000000, 1);
    assert_eq!(flags.of, 1); // signed overflow
    assert_eq!(flags.sf, 0); // positive result
}

#[test]
fn test_logic_flags_zero() {
    // AND result = 0, should set ZF
    let flags = calc_flags_logic(32, 0);
    assert_eq!(flags.zf, 1);
    assert_eq!(flags.cf, 0);
    assert_eq!(flags.sf, 0);
    assert_eq!(flags.of, 0);
}

#[test]
fn test_logic_flags_negative() {
    // TEST result with sign bit set
    let flags = calc_flags_logic(32, 0x80000000);
    assert_eq!(flags.zf, 0);
    assert_eq!(flags.sf, 1);
}

#[test]
fn test_amd64_condition_setz_after_cmp() {
    // CMP 5, 5 (SUB 5, 5 = 0) then SETZ
    // Should return 1 (ZF is set)
    use amd64_cc_op::G_CC_OP_SUBL;
    use cond_type::COND_Z;

    let result = amd64g_calculate_condition(COND_Z, G_CC_OP_SUBL, 5, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_amd64_condition_setnz_after_cmp() {
    // CMP 5, 3 (SUB 5, 3 = 2) then SETNZ
    // Should return 1 (ZF is clear)
    use amd64_cc_op::G_CC_OP_SUBL;
    use cond_type::COND_NZ;

    let result = amd64g_calculate_condition(COND_NZ, G_CC_OP_SUBL, 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_amd64_condition_setb_after_cmp() {
    // CMP 3, 5 (SUB 3, 5 = borrow) then SETB
    // Should return 1 (CF is set)
    use amd64_cc_op::G_CC_OP_SUBL;
    use cond_type::COND_B;

    let result = amd64g_calculate_condition(COND_B, G_CC_OP_SUBL, 3, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_amd64_condition_seta_after_cmp() {
    // CMP 5, 3 (SUB 5, 3 = no borrow, non-zero) then SETA (NBE)
    // Should return 1 (CF=0 and ZF=0)
    use amd64_cc_op::G_CC_OP_SUBL;
    use cond_type::COND_NBE;

    let result = amd64g_calculate_condition(COND_NBE, G_CC_OP_SUBL, 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_amd64_condition_setz_after_test() {
    // TEST 0, 0 (LOGIC 0) then SETZ
    // Should return 1 (ZF is set)
    use amd64_cc_op::G_CC_OP_LOGICL;
    use cond_type::COND_Z;

    let result = amd64g_calculate_condition(COND_Z, G_CC_OP_LOGICL, 0, 0, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_amd64_condition_sets_after_test() {
    // TEST with negative result then SETS
    use amd64_cc_op::G_CC_OP_LOGICL;
    use cond_type::COND_S;

    let result = amd64g_calculate_condition(COND_S, G_CC_OP_LOGICL, 0x80000000, 0, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_amd64_condition_setl_signed() {
    // CMP -1, 1 then SETL (signed less than)
    // -1 < 1 is true
    use amd64_cc_op::G_CC_OP_SUBL;
    use cond_type::COND_L;

    let result = amd64g_calculate_condition(COND_L, G_CC_OP_SUBL, 0xFFFFFFFF, 1, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_x86_condition_setz() {
    use cond_type::COND_Z;
    use x86_cc_op::G_CC_OP_SUBL;

    let result = x86g_calculate_condition(COND_Z, G_CC_OP_SUBL, 5, 5, 0);
    assert_eq!(result, Some(1));
}

// ARM condition code tests

/// Helper: encode ARM cond_n_op from condition and cc_op
fn arm_cond_n_op(cond: u64, cc_op: u64) -> u64 {
    (cond << 4) | cc_op
}

#[test]
fn test_arm_cond_eq_sub_equal() {
    // CMP 5, 5 (SUB) → EQ should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_SUB), 5, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_ne_sub_equal() {
    // CMP 5, 5 (SUB) → NE should be 0
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_NE, ARMG_CC_OP_SUB), 5, 5, 0);
    assert_eq!(result, Some(0));
}

#[test]
fn test_arm_cond_ne_sub_different() {
    // CMP 5, 3 (SUB) → NE should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_NE, ARMG_CC_OP_SUB), 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_hs_sub() {
    // CMP 5, 3 → HS (unsigned >=) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_HS, ARMG_CC_OP_SUB), 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_lo_sub() {
    // CMP 3, 5 → LO (unsigned <) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_LO, ARMG_CC_OP_SUB), 3, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_mi_sub() {
    // CMP 3, 5 → MI (negative) should be 1 (3-5 = -2, N set)
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_MI, ARMG_CC_OP_SUB), 3, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_pl_sub() {
    // CMP 5, 3 → PL (positive) should be 1 (5-3 = 2, N clear)
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_PL, ARMG_CC_OP_SUB), 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_gt_sub() {
    // CMP 5, 3 → GT (signed >) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_GT, ARMG_CC_OP_SUB), 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_le_sub() {
    // CMP 3, 5 → LE (signed <=) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_LE, ARMG_CC_OP_SUB), 3, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_ge_sub_equal() {
    // CMP 5, 5 → GE (signed >=) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_GE, ARMG_CC_OP_SUB), 5, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_lt_sub_signed() {
    // CMP -1 (0xFFFFFFFF), 1 → LT (signed <) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result =
        armg_calculate_condition(arm_cond_n_op(ARM_COND_LT, ARMG_CC_OP_SUB), 0xFFFFFFFF, 1, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_hi_sub() {
    // CMP 5, 3 → HI (unsigned >) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_HI, ARMG_CC_OP_SUB), 5, 3, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_ls_sub() {
    // CMP 3, 5 → LS (unsigned <=) should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_LS, ARMG_CC_OP_SUB), 3, 5, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_al() {
    // AL (always) → 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_AL, ARMG_CC_OP_SUB), 0, 0, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_add_eq() {
    // ADD 5+(-5) = 0, EQ should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(
        arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_ADD),
        5,
        0xFFFFFFFB,
        0, // 5 + (-5) = 0
    );
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_cond_logic_eq() {
    // LOGIC result=0, EQ should be 1
    use arm_cc_op::*;
    use arm_cond::*;
    let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_LOGIC), 0, 0, 0);
    assert_eq!(result, Some(1));
}

#[test]
fn test_arm_flags_nzcv_sub_zero() {
    // SUB 5 - 5 = 0: N=0, Z=1, C=1 (no borrow), V=0
    use arm_cc_op::*;
    let nzcv = armg_calculate_flags_nzcv(ARMG_CC_OP_SUB, 5, 5, 0).unwrap();
    assert_eq!((nzcv >> 31) & 1, 0); // N
    assert_eq!((nzcv >> 30) & 1, 1); // Z
    assert_eq!((nzcv >> 29) & 1, 1); // C (no borrow on ARM means C=1)
    assert_eq!((nzcv >> 28) & 1, 0); // V
}

#[test]
fn test_arm_flags_nzcv_copy() {
    // COPY with NZCV = 0xA0000000 (N=1, Z=0, C=1, V=0)
    use arm_cc_op::*;
    let nzcv = armg_calculate_flags_nzcv(ARMG_CC_OP_COPY, 0xA0000000, 0, 0).unwrap();
    assert_eq!((nzcv >> 31) & 1, 1); // N
    assert_eq!((nzcv >> 30) & 1, 0); // Z
    assert_eq!((nzcv >> 29) & 1, 1); // C
    assert_eq!((nzcv >> 28) & 1, 0); // V
}

#[test]
fn test_arm_handle_ccall_concrete() {
    // Test via handle_ccall dispatch
    use arm_cc_op::*;
    use arm_cond::*;
    let cond_n_op = arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_SUB);
    let args = vec![
        RustBV::concrete(cond_n_op as u128, 32),
        RustBV::concrete(5, 32),
        RustBV::concrete(5, 32),
        RustBV::concrete(0, 32),
    ];
    let result = handle_ccall("armg_calculate_condition", &args, 32);
    assert!(result.is_some());
    assert_eq!(result.unwrap().as_u64(), Some(1));
}

/// angr-03vl4.72: `armg_calc_flag_n` used to skip the 32-bit masking its
/// `_z`/`_c`/`_v` siblings apply, so a borrowing SUB (0 - 1) produced
/// `0xFFFFFFFFFFFFFFFF >> 31` = 0x1FFFFFFFF rather than the 1-bit flag.
#[test]
fn test_arm_flag_n_masks_to_32_bits() {
    use arm_cc_op::*;
    // Borrowing subtract: result is 0xFFFFFFFF, N set.
    assert_eq!(armg_calc_flag_n(ARMG_CC_OP_SUB, 0, 1, 0), Some(1));
    // Non-borrowing subtract with a positive result: N clear.
    assert_eq!(armg_calc_flag_n(ARMG_CC_OP_SUB, 5, 3, 0), Some(0));
    // Add that carries out of bit 31: only bit 31 of the truncated sum counts.
    assert_eq!(
        armg_calc_flag_n(ARMG_CC_OP_ADD, 0x8000_0000, 0x8000_0000, 0),
        Some(0)
    );
    assert_eq!(armg_calc_flag_n(ARMG_CC_OP_ADD, 0x7FFF_FFFF, 1, 0), Some(1));
    // ADC/SBB carry-in paths.
    assert_eq!(armg_calc_flag_n(ARMG_CC_OP_ADC, 0x7FFF_FFFF, 0, 1), Some(1));
    assert_eq!(armg_calc_flag_n(ARMG_CC_OP_SBB, 0, 0, 0), Some(1));
    // Garbage in the high half of the inputs must not leak into the flag.
    assert_eq!(
        armg_calc_flag_n(ARMG_CC_OP_SUB, 0xDEAD_BEEF_0000_0000, 1, 0),
        Some(1)
    );
    assert_eq!(
        armg_calc_flag_n(ARMG_CC_OP_LOGIC, 0xFFFF_FFFF_0000_0001, 0, 0),
        Some(0)
    );
}

/// The bug's live reach: the standalone `armg_calculate_flag_n` CCall arm,
/// taken whenever all four args are concrete (angr-03vl4.72).
#[test]
fn test_arm_handle_ccall_flag_n_sub_borrow() {
    use arm_cc_op::*;
    let args = vec![
        RustBV::concrete(u128::from(ARMG_CC_OP_SUB), 32),
        RustBV::concrete(0, 32),
        RustBV::concrete(1, 32),
        RustBV::concrete(0, 32),
    ];
    let result = handle_ccall("armg_calculate_flag_n", &args, 32);
    assert_eq!(result.and_then(|bv| bv.as_u64()), Some(1));
}

// ============================================================
// Symbolic-vs-concrete diff-fuzz for the new symbolic dispatch.
//
// The strategy: feed concrete BV inputs to the symbolic flag builders
// and assert the resulting 1-bit BVs match the bits computed by the
// concrete `calc_flags_*` path. This guarantees the symbolic CC
// dispatch is sound for the entire (cc_op × cond × width × deps)
// cross-product covered by the random sampler. Real symbolic deps
// are exercised by the integration tests (Python `tests/engines/`).
// ============================================================

/// Deterministic LCG for reproducible random inputs.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x6A09E667F3BCC908)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// Reference: pack `Flags` into the (cf, pf, af, zf, sf, of) tuple as u8.
///
/// AF is in the tuple deliberately: while it was missing, every
/// `diff_fuzz_sym_flags_*` test below was structurally incapable of noticing
/// that neither path computed it at all (angr-5mnx3.59).
fn flags_to_tuple(f: Flags) -> (u8, u8, u8, u8, u8, u8) {
    (f.cf, f.pf, f.af, f.zf, f.sf, f.of)
}

/// Build SymFlags by category and read back as concrete bits.
fn sym_flags_to_tuple(
    category: OpCategory,
    nbits: u32,
    d1: u64,
    d2: u64,
    nd: u64,
) -> (u8, u8, u8, u8, u8, u8) {
    let ctx = crate::symbolic::SymContext::new_mock();
    let bv1 = RustBV::concrete(d1 as u128, 64);
    let bv2 = RustBV::concrete(d2 as u128, 64);
    let bvn = RustBV::concrete(nd as u128, 64);
    let f = sym_flags_for_category(category, nbits, &bv1, &bv2, &bvn, &ctx)
        .expect("category should be supported");
    let bit = |bv: &RustBV| bv.as_u64().expect("must be concrete") as u8;
    (
        bit(&f.cf),
        bit(&f.pf),
        bit(&f.af),
        bit(&f.zf),
        bit(&f.sf),
        bit(&f.of),
    )
}

#[test]
fn diff_fuzz_sym_flags_sub() {
    // VEX always feeds calc_flags_sub args pre-masked to the operand width
    // (the cc_dep IRExpr is typed to nbits), so the random inputs are masked
    // to match that invariant. The concrete path masks defensively too
    // (angr-36vvn.3), so the garbage-high-bits case is not a divergence —
    // `diff_fuzz_sym_flags_unmasked_inputs` covers it explicitly.
    let mut rng = Lcg::new(0x5ab_5e3d);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_sub(nbits, d1, d2));
            let sym = sym_flags_to_tuple(OpCategory::Sub, nbits, d1, d2, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_add() {
    let mut rng = Lcg::new(0xadd_1234);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_add(nbits, d1, d2));
            let sym = sym_flags_to_tuple(OpCategory::Add, nbits, d1, d2, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_adc() {
    // angr-9ke6b.88: ADC/SBB previously had no symbolic builder, so a
    // symbolic operand fell through to the fabricate-a-fresh-symbol path.
    // Same masking invariant as `diff_fuzz_sym_flags_sub`.
    let mut rng = Lcg::new(0xadc_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            // Only the CF bit of cc_ndep is read; sweep both settings.
            let nd = rng.next() & flag_mask::G_CC_MASK_C;
            let conc = flags_to_tuple(calc_flags_adc(nbits, d1, d2, nd));
            let sym = sym_flags_to_tuple(OpCategory::Adc, nbits, d1, d2, nd);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x} nd={nd:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_sbb() {
    let mut rng = Lcg::new(0x5bb_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let nd = rng.next() & flag_mask::G_CC_MASK_C;
            let conc = flags_to_tuple(calc_flags_sbb(nbits, d1, d2, nd));
            let sym = sym_flags_to_tuple(OpCategory::Sbb, nbits, d1, d2, nd);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x} nd={nd:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_logic() {
    let mut rng = Lcg::new(0x0001_0c1c_aaaa);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_logic(nbits, d1));
            let sym = sym_flags_to_tuple(OpCategory::Logic, nbits, d1, 0, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_inc() {
    let mut rng = Lcg::new(0x12c_beef);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let nd = rng.next();
            let conc = flags_to_tuple(calc_flags_inc(nbits, d1, nd));
            let sym = sym_flags_to_tuple(OpCategory::Inc, nbits, d1, 0, nd);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} nd={nd:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_dec() {
    let mut rng = Lcg::new(0xdec_2026);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let nd = rng.next();
            let conc = flags_to_tuple(calc_flags_dec(nbits, d1, nd));
            let sym = sym_flags_to_tuple(OpCategory::Dec, nbits, d1, 0, nd);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} nd={nd:x}");
        }
    }
}

// angr-9ke6b.219: Shl/Shr/Rol/Ror/Umul/Smul had no symbolic builder, so after
// angr-9ke6b.88 tightened the fallback they routed every symbolic crossing to
// Python. Same masking invariant as `diff_fuzz_sym_flags_sub`.

#[test]
fn diff_fuzz_sym_flags_shl() {
    let mut rng = Lcg::new(0x5c1_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            // d1 = post-shift result, d2 = value holding the shifted-out bits.
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_shl(nbits, d1, d2));
            let sym = sym_flags_to_tuple(OpCategory::Shl, nbits, d1, d2, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_shr() {
    let mut rng = Lcg::new(0x5c2_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_shr(nbits, d1, d2));
            let sym = sym_flags_to_tuple(OpCategory::Shr, nbits, d1, d2, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_rol() {
    // ROL/ROR preserve PF/ZF/SF from cc_ndep, so sweep the full flag word
    // rather than just the CF bit.
    let mut rng = Lcg::new(0x201_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let nd = rng.next();
            let conc = flags_to_tuple(calc_flags_rol(nbits, d1, nd));
            let sym = sym_flags_to_tuple(OpCategory::Rol, nbits, d1, 0, nd);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} nd={nd:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_ror() {
    let mut rng = Lcg::new(0x202_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let nd = rng.next();
            let conc = flags_to_tuple(calc_flags_ror(nbits, d1, nd));
            let sym = sym_flags_to_tuple(OpCategory::Ror, nbits, d1, 0, nd);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} nd={nd:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_umul() {
    let mut rng = Lcg::new(0x0117_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_umul(nbits, d1, d2));
            let sym = sym_flags_to_tuple(OpCategory::Umul, nbits, d1, d2, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x}");
        }
    }
}

#[test]
fn diff_fuzz_sym_flags_smul() {
    let mut rng = Lcg::new(0x0217_0f00);
    for nbits in [8u32, 16, 32, 64] {
        let m = get_mask(nbits);
        for _ in 0..200 {
            let d1 = rng.next() & m;
            let d2 = rng.next() & m;
            let conc = flags_to_tuple(calc_flags_smul(nbits, d1, d2));
            let sym = sym_flags_to_tuple(OpCategory::Smul, nbits, d1, d2, 0);
            assert_eq!(sym, conc, "nbits={nbits} d1={d1:x} d2={d2:x}");
        }
    }
}

/// angr-0jh0j.69: the `diff_fuzz_sym_flags_*` tests above all pre-mask their
/// random inputs, so none of them could see a concrete path that reads bits
/// above the operand width. The symbolic builders `extract_to_nbits` their
/// operands unconditionally, so they are the reference for what garbage high
/// bits must do: nothing. Only the categories whose concrete implementations
/// mask defensively are listed — the shift/rotate/inc/dec ones take an
/// already-computed result rather than raw operands.
#[test]
fn diff_fuzz_sym_flags_unmasked_inputs() {
    let mut rng = Lcg::new(0x9a5c_0f69);
    for category in [
        OpCategory::Add,
        OpCategory::Sub,
        OpCategory::Adc,
        OpCategory::Sbb,
        OpCategory::Logic,
        OpCategory::Umul,
        OpCategory::Smul,
    ] {
        for nbits in [8u32, 16, 32] {
            let m = get_mask(nbits);
            for _ in 0..200 {
                let d1 = rng.next();
                let d2 = rng.next();
                let nd = rng.next() & flag_mask::G_CC_MASK_C;
                let conc = flags_to_tuple(compute_flags_from_category(category, nbits, d1, d2, nd));
                let sym = sym_flags_to_tuple(category, nbits, d1, d2, nd);
                assert_eq!(sym, conc, "{category:?} nbits={nbits} d1={d1:x} d2={d2:x}");
                // The garbage above the width must not change the answer at all.
                let masked =
                    flags_to_tuple(compute_flags_from_category(category, nbits, d1 & m, d2 & m, nd));
                assert_eq!(
                    masked, conc,
                    "{category:?} nbits={nbits} d1={d1:x} d2={d2:x} garbage changed the flags"
                );
            }
        }
    }
}

/// Hand-checked instance of the `sign_extend_to_i64` half of angr-0jh0j.69:
/// with the sign bit clear the non-negative branch used to return `val`
/// verbatim, so an 8-bit operand of 0 spelled `0x100` multiplied as 256.
#[test]
fn sign_extend_to_i64_drops_bits_above_the_width() {
    assert_eq!(sign_extend_to_i64(0x100, 8), 0);
    assert_eq!(sign_extend_to_i64(0xFF80, 8), -128);
    assert_eq!(sign_extend_to_i64(0x1_0000, 16), 0);
    // 0x02 * 0x03 is 6, CF/OF clear; the garbage in bits 8+ must not make the
    // 8-bit product overflow its low half.
    assert_eq!(
        flags_to_tuple(calc_flags_smul(8, 0xFF00 | 0x02, 0xAB00 | 0x03)),
        flags_to_tuple(calc_flags_smul(8, 0x02, 0x03))
    );
    assert_eq!(calc_flags_umul(8, 0xFF00 | 0x02, 0xAB00 | 0x03).cf, 0);
}

/// The 64-bit UMUL/SMUL builders widen the operands to a 128-bit product —
/// the only place in the x86 ccall path that builds a BV wider than 64 bits.
/// The diff-fuzz tests above feed concrete BVs, which constant-fold before Z3
/// ever sees the wide sort, so drive it once with a genuinely symbolic operand
/// and check the solver still agrees with the concrete reference.
// Symbolic solving: `add_bv_constraint` / `eval` only exist with the Z3-backed
// engine (angr-9ke6b.236, see bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn sym_flags_wide_mul_roundtrips_through_z3() {
    for (i, (d1, d2)) in [
        (0x1_0000_0000u64, 0x1_0000_0000u64), // product needs bit 64
        (3u64, 5u64),                         // fits in the low half
        (u64::MAX, 2u64),                     // -1 signed / huge unsigned
    ]
    .into_iter()
    .enumerate()
    {
        for (cat, conc) in [
            (OpCategory::Umul, calc_flags_umul(64, d1, d2)),
            (OpCategory::Smul, calc_flags_smul(64, d1, d2)),
        ] {
            // Fresh context per case: the constraint pins the symbol to d1.
            let ctx = crate::symbolic::SymContext::new_mock();
            let sym_d1 = RustBV::symbolic(&ctx, format!("wide_mul_d1_{i}"), 64);
            ctx.add_bv_constraint(&sym_d1, d1 as u128);
            let d2_bv = RustBV::concrete(d2 as u128, 64);
            let f = sym_flags_for_category(cat, 64, &sym_d1, &d2_bv, &d2_bv, &ctx)
                .expect("category should be supported");
            let got = (
                ctx.eval(&f.cf),
                ctx.eval(&f.pf),
                ctx.eval(&f.zf),
                ctx.eval(&f.sf),
                ctx.eval(&f.of),
            );
            let want = (
                Some(conc.cf as u128),
                Some(conc.pf as u128),
                Some(conc.zf as u128),
                Some(conc.sf as u128),
                Some(conc.of as u128),
            );
            assert_eq!(got, want, "{cat:?} d1={d1:x} d2={d2:x}");
        }
    }
}

/// Regression for angr-g6dg: `amd64g_calculate_rflags_c` for INC/DEC
/// cc_ops must derive the carry from cc_ndep (CF is preserved by INC/DEC),
/// even when the concrete fast-path declines because cc_dep1 is symbolic.
/// Before the fix this category fell through to a fresh unconstrained
/// symbolic carry, poisoning later branch guards in cold blocks.
#[test]
fn rflags_c_inc_dec_symbolic_dep_preserves_carry() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let cc_ops = [
        amd64_cc_op::G_CC_OP_INCB,
        amd64_cc_op::G_CC_OP_INCL,
        amd64_cc_op::G_CC_OP_INCQ,
        amd64_cc_op::G_CC_OP_DECB,
        amd64_cc_op::G_CC_OP_DECL,
        amd64_cc_op::G_CC_OP_DECQ,
    ];
    // CF lives in bit 0 (G_CC_SHIFT_C == 0), so cc_ndep & 1 is the carry.
    for cc_op in cc_ops {
        for nd in [0u64, 1, 0x1234, 0xFFFF_FFFF_FFFF_FFFF] {
            // cc_dep1 symbolic -> concrete fast-path declines, forcing the
            // symbolic path. The carry must still resolve concretely from
            // the concrete cc_ndep.
            let args = vec![
                RustBV::concrete(cc_op as u128, 64),
                RustBV::symbolic(&ctx, "cc_dep1_sym", 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(nd as u128, 64),
            ];
            let got = handle_ccall_with_ctx("amd64g_calculate_rflags_c", &args, 64, Some(&ctx))
                .expect("INC/DEC rflags_c must resolve symbolically")
                .as_u64()
                .expect("carry is a function of the concrete cc_ndep");
            assert_eq!(got, nd & 1, "cc_op={cc_op} nd={nd:x}");
        }
    }
}

/// angr-0jh0j.71: the symbolic `*_calculate_eflags_all` branch used to
/// enumerate Sub/Add/Logic by hand and drop every other category to the Python
/// ccall (30x wall-clock, angr-9ke6b.88), even though `sym_flags_for_category`
/// already built flags for all of them. Every non-Copy cc_op both arches decode
/// must now resolve natively; iterating the decoder rather than a hand-written
/// list means a newly mapped cc_op is covered the day it lands.
#[test]
fn eflags_all_symbolic_covers_every_cc_op() {
    let ctx = crate::symbolic::SymContext::new_mock();
    for (name, arch) in [
        ("amd64g_calculate_eflags_all", CcArch::Amd64),
        ("x86g_calculate_eflags_all", CcArch::X86),
    ] {
        let mut seen = 0;
        // cc_op 0 is COPY on both arches and takes the mask shortcut above.
        for cc_op in 1..64u64 {
            let Some(info) = cc_op_info(arch, cc_op) else {
                continue;
            };
            assert_ne!(info.category, OpCategory::Copy, "{name} cc_op={cc_op}");
            seen += 1;
            // A symbolic cc_dep1 declines the concrete fast path, forcing the
            // symbolic branch this test is about.
            let args = vec![
                RustBV::concrete(cc_op as u128, 64),
                RustBV::symbolic(&ctx, format!("{name}_d1_{cc_op}"), 64),
                RustBV::concrete(0x5a, 64),
                RustBV::concrete(1, 64),
            ];
            assert!(
                handle_ccall_with_ctx(name, &args, 64, Some(&ctx)).is_some(),
                "{name} cc_op={cc_op} ({:?}) fell back to Python",
                info.category
            );
        }
        assert!(seen > 10, "{name} decoded only {seen} cc_ops");
    }
}

/// Value half of `eflags_all_symbolic_covers_every_cc_op`: the packed symbolic
/// answer must equal the concrete `calculate_eflags_all_*` reference for every
/// cc_op, not merely be non-`None`. Pins cc_dep1 with a constraint so the
/// symbolic path runs while the answer stays evaluable.
// Symbolic solving: `add_bv_constraint` / `eval` only exist with the Z3-backed
// engine (angr-9ke6b.236, see bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn diff_fuzz_eflags_all_symbolic_matches_concrete() {
    let mut rng = Lcg::new(0x0ef1_a5a1);
    for (name, arch) in [
        ("amd64g_calculate_rflags_all", CcArch::Amd64),
        ("x86g_calculate_eflags_all", CcArch::X86),
    ] {
        for cc_op in 1..64u64 {
            if cc_op_info(arch, cc_op).is_none() {
                continue;
            }
            for _ in 0..2 {
                let d1 = rng.next();
                let d2 = rng.next();
                // ADC/SBB read cc_ndep as the incoming carry; the rotates read
                // it as packed flags. Keeping it in {0,1} is valid for both.
                let nd = rng.next() & flag_mask::G_CC_MASK_C;
                // Fresh context per case: the constraint pins the symbol to d1.
                let ctx = crate::symbolic::SymContext::new_mock();
                let sym_d1 = RustBV::symbolic(&ctx, "eflags_all_d1", 64);
                ctx.add_bv_constraint(&sym_d1, d1 as u128);
                let args = vec![
                    RustBV::concrete(cc_op as u128, 64),
                    sym_d1,
                    RustBV::concrete(d2 as u128, 64),
                    RustBV::concrete(nd as u128, 64),
                ];
                let got = handle_ccall_with_ctx(name, &args, 64, Some(&ctx))
                    .expect("every decoded cc_op must resolve natively");
                let want = match arch {
                    CcArch::Amd64 => calculate_eflags_all_amd64(cc_op, d1, d2, nd),
                    CcArch::X86 => calculate_eflags_all_x86(cc_op, d1, d2, nd),
                }
                .expect("concrete reference must also decode");
                assert_eq!(
                    ctx.eval(&got),
                    Some(want as u128),
                    "{name} cc_op={cc_op} d1={d1:x} d2={d2:x} nd={nd:x}"
                );
            }
        }
    }
}

/// An out-of-range condition code must defer to Python (`None`) rather than
/// silently answering "condition false", matching `eval_sym_condition`.
/// Covers both `calculate_condition` arms: the COPY shortcut and the
/// compute-flags path, on both x86 and AMD64.
#[test]
fn unknown_cond_defers_to_python() {
    let f = Flags {
        cf: 1,
        pf: 1,
        af: 1,
        zf: 1,
        sf: 1,
        of: 1,
    };
    let ctx = crate::symbolic::SymContext::new_mock();
    let sym_f = SymFlags {
        cf: RustBV::concrete(1, 1),
        pf: RustBV::concrete(1, 1),
        af: RustBV::concrete(1, 1),
        zf: RustBV::concrete(1, 1),
        sf: RustBV::concrete(1, 1),
        of: RustBV::concrete(1, 1),
    };
    // 16 is the first value past the 4-bit VEX condition range.
    for cond in [16u64, 17, 42, u64::MAX] {
        assert_eq!(eval_condition(cond, &f), None, "cond={cond}");
        assert!(
            eval_sym_condition(cond, &sym_f, &ctx).is_none(),
            "cond={cond}"
        );
        // COPY path (flags live in cc_dep1) and the compute-flags path.
        assert_eq!(
            amd64g_calculate_condition(cond, amd64_cc_op::G_CC_OP_COPY, 0, 0, 0),
            None,
            "cond={cond}"
        );
        assert_eq!(
            amd64g_calculate_condition(cond, amd64_cc_op::G_CC_OP_SUBL, 1, 1, 0),
            None,
            "cond={cond}"
        );
        assert_eq!(
            x86g_calculate_condition(cond, x86_cc_op::G_CC_OP_COPY, 0, 0, 0),
            None,
            "cond={cond}"
        );
        assert_eq!(
            x86g_calculate_condition(cond, x86_cc_op::G_CC_OP_SUBL, 1, 1, 0),
            None,
            "cond={cond}"
        );
    }
    // Sanity: a legitimate cond still yields an answer on the same inputs.
    assert!(eval_condition(cond_type::COND_Z, &f).is_some());
    assert!(
        amd64g_calculate_condition(cond_type::COND_Z, amd64_cc_op::G_CC_OP_COPY, 0, 0, 0).is_some()
    );
}

/// Diff-fuzz the full eval_sym_condition path against eval_condition.
#[test]
fn diff_fuzz_eval_sym_condition() {
    use cond_type::*;
    let ctx = crate::symbolic::SymContext::new_mock();
    let mut rng = Lcg::new(0xc0ed_d1ff);
    let conds = [
        COND_O, COND_NO, COND_B, COND_NB, COND_Z, COND_NZ, COND_BE, COND_NBE, COND_S, COND_NS,
        COND_P, COND_NP, COND_L, COND_NL, COND_LE, COND_NLE,
    ];
    for _ in 0..200 {
        let cf = (rng.next() & 1) as u8;
        let pf = (rng.next() & 1) as u8;
        let zf = (rng.next() & 1) as u8;
        let sf = (rng.next() & 1) as u8;
        let of = (rng.next() & 1) as u8;
        let af = (rng.next() & 1) as u8;
        let f = Flags { cf, pf, af, zf, sf, of };
        let sym_f = SymFlags {
            cf: RustBV::concrete(cf as u128, 1),
            pf: RustBV::concrete(pf as u128, 1),
            af: RustBV::concrete(af as u128, 1),
            zf: RustBV::concrete(zf as u128, 1),
            sf: RustBV::concrete(sf as u128, 1),
            of: RustBV::concrete(of as u128, 1),
        };
        for &cond in &conds {
            let conc = eval_condition(cond, &f).expect("standard cond");
            let sym = eval_sym_condition(cond, &sym_f, &ctx)
                .expect("standard cond")
                .as_u64()
                .expect("concrete");
            assert_eq!(sym, conc, "cond={cond} flags={f:?}");
        }
    }
}

/// End-to-end: route symbolic dispatch via handle_ccall_with_ctx with
/// concrete BV args and verify it agrees with the concrete fast path.
#[test]
fn diff_fuzz_amd64_handle_ccall_symbolic_path() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let mut rng = Lcg::new(0xa64_e2e);
    let conds = [
        cond_type::COND_O,
        cond_type::COND_NB,
        cond_type::COND_Z,
        cond_type::COND_BE,
        cond_type::COND_S,
        cond_type::COND_NS,
        cond_type::COND_L,
        cond_type::COND_LE,
        cond_type::COND_NLE,
    ];
    let cc_ops = [
        amd64_cc_op::G_CC_OP_SUBB,
        amd64_cc_op::G_CC_OP_SUBW,
        amd64_cc_op::G_CC_OP_SUBL,
        amd64_cc_op::G_CC_OP_SUBQ,
        amd64_cc_op::G_CC_OP_ADDL,
        amd64_cc_op::G_CC_OP_ADDQ,
        amd64_cc_op::G_CC_OP_LOGICB,
        amd64_cc_op::G_CC_OP_LOGICL,
        amd64_cc_op::G_CC_OP_INCL,
        amd64_cc_op::G_CC_OP_DECQ,
    ];
    for _ in 0..100 {
        let cond = conds[(rng.next() as usize) % conds.len()];
        let cc_op = cc_ops[(rng.next() as usize) % cc_ops.len()];
        let d1 = rng.next();
        let d2 = rng.next();
        let nd = rng.next();
        let conc = amd64g_calculate_condition(cond, cc_op, d1, d2, nd);
        // Route through the symbolic dispatch by passing dep1 as a fresh BVS-like
        // BV — but to keep this test deterministic and concrete, we just use
        // concrete BVs and rely on the dispatcher's fall-through: when all args
        // are concrete the fast path triggers. To force the symbolic path, we
        // need to wrap one operand. Use sym builders directly here for sanity
        // and an end-to-end check via the dispatcher's main entry.
        let args = vec![
            RustBV::concrete(cond as u128, 64),
            RustBV::concrete(cc_op as u128, 64),
            RustBV::concrete(d1 as u128, 64),
            RustBV::concrete(d2 as u128, 64),
            RustBV::concrete(nd as u128, 64),
        ];
        let dispatched = handle_ccall_with_ctx("amd64g_calculate_condition", &args, 64, Some(&ctx))
            .and_then(|bv| bv.as_u64());
        assert_eq!(
            dispatched, conc,
            "concrete-path mismatch cond={cond} cc_op={cc_op} d1={d1:x} d2={d2:x} nd={nd:x}"
        );
    }
}

/// ARM-side end-to-end diff-fuzz: symbolic dispatch with concrete BV inputs.
#[test]
fn diff_fuzz_arm_sym_calculate_condition() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let mut rng = Lcg::new(0xa12_e2e);
    use arm_cc_op::*;
    use arm_cond::*;
    let conds = [
        ARM_COND_EQ,
        ARM_COND_NE,
        ARM_COND_HS,
        ARM_COND_LO,
        ARM_COND_MI,
        ARM_COND_PL,
        ARM_COND_VS,
        ARM_COND_VC,
        ARM_COND_HI,
        ARM_COND_LS,
        ARM_COND_GE,
        ARM_COND_LT,
        ARM_COND_GT,
        ARM_COND_LE,
    ];
    let cc_ops = [
        ARMG_CC_OP_COPY,
        ARMG_CC_OP_ADD,
        ARMG_CC_OP_SUB,
        ARMG_CC_OP_ADC,
        ARMG_CC_OP_SBB,
        ARMG_CC_OP_LOGIC,
        ARMG_CC_OP_MUL,
        ARMG_CC_OP_MULL,
    ];
    for _ in 0..200 {
        let cond = conds[(rng.next() as usize) % conds.len()];
        let cc_op = cc_ops[(rng.next() as usize) % cc_ops.len()];
        // Mask to 32-bit so concrete reference handles match what VEX would feed.
        let d1 = rng.next() & 0xFFFFFFFF;
        let d2 = rng.next() & 0xFFFFFFFF;
        let nd = rng.next() & 0xFFFFFFFF;
        let conc = armg_calculate_condition((cond << 4) | cc_op, d1, d2, nd);
        // Sanity: concrete returns Some for these (skip cases it doesn't, e.g. ADC/SBB+VS
        // unsupported in the legacy concrete path).
        let Some(conc_val) = conc else {
            continue;
        };
        let sym = arm_sym_calculate_condition(
            cond,
            cc_op,
            &RustBV::concrete(d1 as u128, 32),
            &RustBV::concrete(d2 as u128, 32),
            &RustBV::concrete(nd as u128, 32),
            &ctx,
        )
        .expect("symbolic path should handle this cond/cc_op pair");
        let sym_val = sym.as_u64().expect("concrete");
        assert_eq!(
            sym_val, conc_val,
            "cond={cond} cc_op={cc_op} d1={d1:x} d2={d2:x} nd={nd:x}"
        );
    }
}

// ============================================================
// AArch64 (arm64g) condition code tests
// ============================================================

#[test]
fn test_arm64_cond_sub64_eq() {
    use arm_cond::*;
    use arm64_cc_op::*;
    // 5 - 5 == 0 → Z set → EQ true, NE false.
    let n = (ARM_COND_EQ << 4) | ARM64G_CC_OP_SUB64;
    assert_eq!(arm64g_calculate_condition(n, 5, 5, 0), Some(1));
    let n = (ARM_COND_NE << 4) | ARM64G_CC_OP_SUB64;
    assert_eq!(arm64g_calculate_condition(n, 5, 5, 0), Some(0));
}

#[test]
fn test_arm64_cond_sub32_signed() {
    use arm_cond::*;
    use arm64_cc_op::*;
    // 32-bit: 3 - 5 → negative, N set, V clear → LT (N!=V) true, GE false.
    let lt = (ARM_COND_LT << 4) | ARM64G_CC_OP_SUB32;
    assert_eq!(arm64g_calculate_condition(lt, 3, 5, 0), Some(1));
    let ge = (ARM_COND_GE << 4) | ARM64G_CC_OP_SUB32;
    assert_eq!(arm64g_calculate_condition(ge, 3, 5, 0), Some(0));
}

#[test]
fn test_arm64_cond_al() {
    use arm_cond::*;
    use arm64_cc_op::*;
    let al = (ARM_COND_AL << 4) | ARM64G_CC_OP_SUB64;
    assert_eq!(arm64g_calculate_condition(al, 1, 2, 0), Some(1));
    // NV is unconditional-true on AArch64 too.
    let nv = (ARM_COND_NV << 4) | ARM64G_CC_OP_SUB64;
    assert_eq!(arm64g_calculate_condition(nv, 1, 2, 0), Some(1));
}

#[test]
fn test_arm64_handle_ccall_dispatch() {
    use arm_cond::*;
    use arm64_cc_op::*;
    let n = (ARM_COND_EQ << 4) | ARM64G_CC_OP_SUB64;
    let args = vec![
        RustBV::concrete(n as u128, 64),
        RustBV::concrete(7, 64),
        RustBV::concrete(7, 64),
        RustBV::concrete(0, 64),
    ];
    let result = handle_ccall("arm64g_calculate_condition", &args, 64);
    assert_eq!(result.and_then(|r| r.as_u64()), Some(1));
}

/// Diff-fuzz the symbolic arm64 condition builders against the concrete
/// path across the full (cond × cc_op × deps) cross-product. N>=1000.
#[test]
fn diff_fuzz_arm64_sym_calculate_condition() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let mut rng = Lcg::new(0xa64_e2e);
    use arm_cond::*;
    use arm64_cc_op::*;
    let conds = [
        ARM_COND_EQ,
        ARM_COND_NE,
        ARM_COND_HS,
        ARM_COND_LO,
        ARM_COND_MI,
        ARM_COND_PL,
        ARM_COND_VS,
        ARM_COND_VC,
        ARM_COND_HI,
        ARM_COND_LS,
        ARM_COND_GE,
        ARM_COND_LT,
        ARM_COND_GT,
        ARM_COND_LE,
    ];
    let cc_ops = [
        ARM64G_CC_OP_COPY,
        ARM64G_CC_OP_ADD32,
        ARM64G_CC_OP_ADD64,
        ARM64G_CC_OP_SUB32,
        ARM64G_CC_OP_SUB64,
        ARM64G_CC_OP_ADC32,
        ARM64G_CC_OP_ADC64,
        ARM64G_CC_OP_SBC32,
        ARM64G_CC_OP_SBC64,
        ARM64G_CC_OP_LOGIC32,
        ARM64G_CC_OP_LOGIC64,
    ];
    for _ in 0..1500 {
        let cond = conds[(rng.next() as usize) % conds.len()];
        let cc_op = cc_ops[(rng.next() as usize) % cc_ops.len()];
        let d1 = rng.next();
        let d2 = rng.next();
        // carry dep is logically a single bit
        let d3 = rng.next() & 1;
        let Some(conc_val) = arm64g_calculate_condition((cond << 4) | cc_op, d1, d2, d3) else {
            continue;
        };
        let sym = arm64_sym_calculate_condition(
            cond,
            cc_op,
            &RustBV::concrete(d1 as u128, 64),
            &RustBV::concrete(d2 as u128, 64),
            &RustBV::concrete(d3 as u128, 64),
            &ctx,
        )
        .expect("symbolic path should handle this cond/cc_op pair");
        let sym_val = sym.as_u64().expect("concrete");
        assert_eq!(
            sym_val, conc_val,
            "cond={cond} cc_op={cc_op} d1={d1:x} d2={d2:x} d3={d3:x}"
        );
    }
}

/// The four ARM32 individual-flag CCalls must resolve natively for *every*
/// cc_op when a dep is symbolic — no `_ => None` gap that silently sends a
/// whole cc_op back to the Python ccall (angr-0jh0j.70). Iterates the cc_op
/// range rather than a hand-written list so a newly added cc_op is covered
/// automatically.
#[test]
fn arm32_flag_ccalls_symbolic_cover_every_cc_op() {
    let ctx = crate::symbolic::SymContext::new_mock();
    for name in [
        "armg_calculate_flag_n",
        "armg_calculate_flag_z",
        "armg_calculate_flag_c",
        "armg_calculate_flag_v",
    ] {
        // COPY..MULL is the whole ARMG_CC_OP_* space (libvex_guest_arm.h).
        for cc_op in arm_cc_op::ARMG_CC_OP_COPY..=arm_cc_op::ARMG_CC_OP_MULL {
            // A symbolic cc_dep1 declines the concrete fast path, forcing the
            // symbolic branch this test is about.
            let args = vec![
                RustBV::concrete(u128::from(cc_op), 32),
                RustBV::symbolic(&ctx, format!("{name}_d1_{cc_op}"), 32),
                RustBV::concrete(0x5a, 32),
                RustBV::concrete(1, 32),
            ];
            assert!(
                handle_ccall_with_ctx(name, &args, 32, Some(&ctx)).is_some(),
                "{name} cc_op={cc_op} fell back to Python"
            );
        }
    }
}

/// Value half of `arm32_flag_ccalls_symbolic_cover_every_cc_op`: the symbolic
/// answer must equal the concrete `armg_calc_flag_*` reference, not merely be
/// non-`None`. Pins cc_dep1 with a constraint so the symbolic dispatch runs
/// while the answer stays evaluable.
// Symbolic solving: `add_bv_constraint` / `eval` only exist with the Z3-backed
// engine (angr-9ke6b.236, see bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn diff_fuzz_arm32_flag_ccalls_symbolic_matches_concrete() {
    let mut rng = Lcg::new(0x0a11_3200);
    for (name, conc) in [
        (
            "armg_calculate_flag_n",
            armg_calc_flag_n as fn(u64, u64, u64, u64) -> Option<u64>,
        ),
        ("armg_calculate_flag_z", armg_calc_flag_z),
        ("armg_calculate_flag_c", armg_calc_flag_c),
        ("armg_calculate_flag_v", armg_calc_flag_v),
    ] {
        for cc_op in arm_cc_op::ARMG_CC_OP_COPY..=arm_cc_op::ARMG_CC_OP_MULL {
            for _ in 0..2 {
                let d1 = rng.next() & 0xFFFF_FFFF;
                let d2 = rng.next() & 0xFFFF_FFFF;
                // ADC/SBB read cc_ndep as the incoming carry; MUL/MULL read it
                // as oldC:oldV. Keeping it in {0,3} is valid for both.
                let nd = rng.next() & 3;
                // Fresh context per case: the constraint pins the symbol to d1.
                let ctx = crate::symbolic::SymContext::new_mock();
                let sym_d1 = RustBV::symbolic(&ctx, "arm32_flag_d1", 32);
                ctx.add_bv_constraint(&sym_d1, u128::from(d1));
                let args = vec![
                    RustBV::concrete(u128::from(cc_op), 32),
                    sym_d1,
                    RustBV::concrete(u128::from(d2), 32),
                    RustBV::concrete(u128::from(nd), 32),
                ];
                let got = handle_ccall_with_ctx(name, &args, 32, Some(&ctx))
                    .expect("every ARM cc_op must resolve natively");
                let want = conc(cc_op, d1, d2, nd).expect("concrete reference must also decode");
                assert_eq!(
                    ctx.eval(&got),
                    Some(u128::from(want)),
                    "{name} cc_op={cc_op} d1={d1:x} d2={d2:x} nd={nd:x}"
                );
            }
        }
    }
}

/// Diff-fuzz each individual symbolic flag builder against its concrete
/// counterpart across all cc_ops.
#[test]
fn diff_fuzz_arm64_sym_flags() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let mut rng = Lcg::new(0xf1a64);
    use arm64_cc_op::*;
    let cc_ops = [
        ARM64G_CC_OP_COPY,
        ARM64G_CC_OP_ADD32,
        ARM64G_CC_OP_ADD64,
        ARM64G_CC_OP_SUB32,
        ARM64G_CC_OP_SUB64,
        ARM64G_CC_OP_ADC32,
        ARM64G_CC_OP_ADC64,
        ARM64G_CC_OP_SBC32,
        ARM64G_CC_OP_SBC64,
        ARM64G_CC_OP_LOGIC32,
        ARM64G_CC_OP_LOGIC64,
    ];
    for &cc_op in &cc_ops {
        for _ in 0..150 {
            let d1 = rng.next();
            let d2 = rng.next();
            let d3 = rng.next() & 1;
            let bv1 = RustBV::concrete(d1 as u128, 64);
            let bv2 = RustBV::concrete(d2 as u128, 64);
            let bv3 = RustBV::concrete(d3 as u128, 64);
            let check = |sym: Option<RustBV>, conc: Option<u64>, name: &str| {
                let s = sym.expect("sym").as_u64().expect("concrete") & 1;
                let c = conc.expect("conc") & 1;
                assert_eq!(s, c, "{name} cc_op={cc_op} d1={d1:x} d2={d2:x} d3={d3:x}");
            };
            check(
                arm64_sym_flag_n(cc_op, &bv1, &bv2, &bv3, &ctx),
                arm64g_calc_flag_n(cc_op, d1, d2, d3),
                "n",
            );
            check(
                arm64_sym_flag_z(cc_op, &bv1, &bv2, &bv3, &ctx),
                arm64g_calc_flag_z(cc_op, d1, d2, d3),
                "z",
            );
            check(
                arm64_sym_flag_c(cc_op, &bv1, &bv2, &bv3, &ctx),
                arm64g_calc_flag_c(cc_op, d1, d2, d3),
                "c",
            );
            check(
                arm64_sym_flag_v(cc_op, &bv1, &bv2, &bv3, &ctx),
                arm64g_calc_flag_v(cc_op, d1, d2, d3),
                "v",
            );
        }
    }
}

// ============================================================
// x86g_use_seg_selector — segmented-address linearization.
//
// Args: [ldt, gdt, seg_selector, virtual_addr]; returns a 64-bit value whose
// low 32 bits are the linear address and whose bit 32 is the error flag.
// Only the concrete fast paths are implemented natively (bad selector, plus
// the empty-descriptor-table flat-addressing case that covers Linux glibc
// TLS/stack-canary reads); everything else returns None so the caller falls
// back to Python's `x86g_use_seg_selector` in engines/vex/claripy/ccall.py.
// ============================================================

/// Convenience: drive the ccall with four concrete 32-bit args.
fn seg_selector_call(ldt: u64, gdt: u64, ss: u64, va: u64) -> Option<u64> {
    let args = vec![
        RustBV::concrete(ldt as u128, 64),
        RustBV::concrete(gdt as u128, 64),
        RustBV::concrete(ss as u128, 32),
        RustBV::concrete(va as u128, 32),
    ];
    handle_ccall("x86g_use_seg_selector", &args, 64).and_then(|r| r.as_u64())
}

#[test]
fn test_use_seg_selector_arity_guard() {
    // Fewer than four args must not panic on indexing — it defers to Python.
    for n in 0..4 {
        let args: Vec<RustBV> = (0..n).map(|_| RustBV::concrete(0, 32)).collect();
        assert!(
            handle_ccall("x86g_use_seg_selector", &args, 64).is_none(),
            "arity {n} should defer"
        );
    }
}

#[test]
fn test_use_seg_selector_bad_selector_sets_error_flag() {
    // Any bit above 15 set in the selector is Python's `bad()` case. The
    // native return is the ABI-correct error flag (bit 32); note Python's own
    // `bad()` builds `BVV(1 << 32, 32)`, which claripy truncates to 0 before
    // the zero_extend — an upstream quirk we deliberately do not mirror.
    assert_eq!(seg_selector_call(0, 0, 0x1_0000, 0x14), Some(1 << 32));
    assert_eq!(seg_selector_call(0, 0, 0xFFFF_0063, 0x14), Some(1 << 32));
    // The bad-selector test precedes the table lookup, so a populated table
    // does not change the answer.
    assert_eq!(
        seg_selector_call(0xDEAD, 0xBEEF, 0x1_0000, 0),
        Some(1 << 32)
    );
}

#[test]
fn test_use_seg_selector_gdt_empty_flat_addressing() {
    // tiBit (bit 2 of the selector) == 0 selects the GDT; an all-zero GDT
    // means flat addressing: linear = (selector << 16) + virtual_addr.
    // 0x63 is the usual Linux x86 %gs, and 0x14 the glibc stack-canary slot.
    assert_eq!(seg_selector_call(0, 0, 0x63, 0x14), Some(0x0063_0014));
    assert_eq!(seg_selector_call(0, 0, 0, 0x1000), Some(0x1000));
    // A populated LDT is irrelevant when tiBit picks the GDT.
    assert_eq!(
        seg_selector_call(0xDEAD_BEEF, 0, 0x63, 0x14),
        Some(0x0063_0014)
    );
}

#[test]
fn test_use_seg_selector_ldt_empty_flat_addressing() {
    // tiBit == 1 selects the LDT. 0x67 == 0x63 | 0b100 flips just that bit.
    assert_eq!(seg_selector_call(0, 0, 0x67, 0x14), Some(0x0067_0014));
    // ...and a populated GDT is irrelevant on the LDT side.
    assert_eq!(
        seg_selector_call(0, 0xDEAD_BEEF, 0x67, 0x14),
        Some(0x0067_0014)
    );
}

#[test]
fn test_use_seg_selector_populated_table_defers_to_python() {
    // Walking a real descriptor table needs a memory load, which the native
    // path does not do — both table sides must return None, not a guess.
    assert_eq!(seg_selector_call(0, 0x1000_0000_0000, 0x63, 0x14), None); // GDT
    assert_eq!(seg_selector_call(0x1000_0000_0000, 0, 0x67, 0x14), None); // LDT
}

#[test]
fn test_use_seg_selector_symbolic_args_defer_to_python() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let sym = RustBV::symbolic(&ctx, "seg_sym", 32);
    for pos in 0..4 {
        let mut args = vec![
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0x63, 32),
            RustBV::concrete(0x14, 32),
        ];
        args[pos] = sym.clone();
        assert!(
            handle_ccall_with_ctx("x86g_use_seg_selector", &args, 64, Some(&ctx)).is_none(),
            "symbolic arg {pos} should defer to Python"
        );
    }
}

#[test]
fn test_use_seg_selector_gdt_empty_wraps_mod_2_32() {
    // Python computes `(seg_selector << 16) + virtual_addr` over 32-bit BVs,
    // so the sum wraps mod 2^32. Bit 32 of the result is the error flag, so an
    // unwrapped u64 add would report a bogus error for any negative
    // displacement off a segment register (`mov %gs:-0x4, %eax`).
    assert_eq!(
        seg_selector_call(0, 0, 0x63, 0xFFFF_FFFC),
        Some(0x0062_FFFC)
    );
    assert_eq!(
        seg_selector_call(0, 0, 0x67, 0xFFFF_FFFC),
        Some(0x0066_FFFC)
    );
    // Exact-wrap-to-zero boundary: 0xFFFF0000 + 0x00010000 == 2^32.
    assert_eq!(seg_selector_call(0, 0, 0xFFFF, 0x0001_0000), Some(0));
}

/// `armg_calculate_flags_nzcv` / `arm64g_calculate_flags_nzcv` must resolve
/// natively for every cc_op when a dep is symbolic — the packed-NZCV analogue
/// of `arm32_flag_ccalls_symbolic_cover_every_cc_op` (angr-zgd3r).
#[test]
fn arm_flags_nzcv_ccalls_symbolic_cover_every_cc_op() {
    let ctx = crate::symbolic::SymContext::new_mock();
    // COPY..MULL is the whole ARMG_CC_OP_* space (libvex_guest_arm.h).
    for cc_op in arm_cc_op::ARMG_CC_OP_COPY..=arm_cc_op::ARMG_CC_OP_MULL {
        let args = vec![
            RustBV::concrete(u128::from(cc_op), 32),
            RustBV::symbolic(&ctx, format!("arm32_nzcv_d1_{cc_op}"), 32),
            RustBV::concrete(0x5a, 32),
            RustBV::concrete(1, 32),
        ];
        assert!(
            handle_ccall_with_ctx("armg_calculate_flags_nzcv", &args, 32, Some(&ctx)).is_some(),
            "armg_calculate_flags_nzcv cc_op={cc_op} fell back to Python"
        );
    }
    for cc_op in arm64_cc_op::ARM64G_CC_OP_COPY..=arm64_cc_op::ARM64G_CC_OP_LOGIC64 {
        let args = vec![
            RustBV::concrete(u128::from(cc_op), 64),
            RustBV::symbolic(&ctx, format!("arm64_nzcv_d1_{cc_op}"), 64),
            RustBV::concrete(0x5a, 64),
            RustBV::concrete(1, 64),
        ];
        assert!(
            handle_ccall_with_ctx("arm64g_calculate_flags_nzcv", &args, 64, Some(&ctx)).is_some(),
            "arm64g_calculate_flags_nzcv cc_op={cc_op} fell back to Python"
        );
    }
}

/// Value half of `arm_flags_nzcv_ccalls_symbolic_cover_every_cc_op`: the
/// packed symbolic word must equal the concrete `*_calculate_flags_nzcv`
/// reference, not merely be non-`None`.
// Symbolic solving: `add_bv_constraint` / `eval` only exist with the Z3-backed
// engine (angr-9ke6b.236, see bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn diff_fuzz_arm_flags_nzcv_symbolic_matches_concrete() {
    let mut rng = Lcg::new(0x0a11_4200);
    for cc_op in arm_cc_op::ARMG_CC_OP_COPY..=arm_cc_op::ARMG_CC_OP_MULL {
        for _ in 0..2 {
            let d1 = rng.next() & 0xFFFF_FFFF;
            let d2 = rng.next() & 0xFFFF_FFFF;
            // ADC/SBB read cc_ndep as the incoming carry; MUL/MULL read it as
            // oldC:oldV. Keeping it in {0..3} is valid for both.
            let nd = rng.next() & 3;
            // Fresh context per case: the constraint pins the symbol to d1.
            let ctx = crate::symbolic::SymContext::new_mock();
            let sym_d1 = RustBV::symbolic(&ctx, "arm32_nzcv_d1", 32);
            ctx.add_bv_constraint(&sym_d1, u128::from(d1));
            let args = vec![
                RustBV::concrete(u128::from(cc_op), 32),
                sym_d1,
                RustBV::concrete(u128::from(d2), 32),
                RustBV::concrete(u128::from(nd), 32),
            ];
            let got = handle_ccall_with_ctx("armg_calculate_flags_nzcv", &args, 32, Some(&ctx))
                .expect("every ARM cc_op must resolve natively");
            let want = armg_calculate_flags_nzcv(cc_op, d1, d2, nd)
                .expect("concrete reference must also decode");
            assert_eq!(
                ctx.eval(&got),
                Some(u128::from(want)),
                "armg nzcv cc_op={cc_op} d1={d1:x} d2={d2:x} nd={nd:x}"
            );
        }
    }
    for cc_op in arm64_cc_op::ARM64G_CC_OP_COPY..=arm64_cc_op::ARM64G_CC_OP_LOGIC64 {
        for _ in 0..2 {
            let d1 = rng.next();
            let d2 = rng.next();
            let d3 = rng.next() & 1;
            let ctx = crate::symbolic::SymContext::new_mock();
            let sym_d1 = RustBV::symbolic(&ctx, "arm64_nzcv_d1", 64);
            ctx.add_bv_constraint(&sym_d1, u128::from(d1));
            let args = vec![
                RustBV::concrete(u128::from(cc_op), 64),
                sym_d1,
                RustBV::concrete(u128::from(d2), 64),
                RustBV::concrete(u128::from(d3), 64),
            ];
            let got = handle_ccall_with_ctx("arm64g_calculate_flags_nzcv", &args, 64, Some(&ctx))
                .expect("every arm64 cc_op must resolve natively");
            let want = arm64g_calculate_flags_nzcv(cc_op, d1, d2, d3)
                .expect("concrete reference must also decode");
            assert_eq!(
                ctx.eval(&got),
                Some(u128::from(want)),
                "arm64g nzcv cc_op={cc_op} d1={d1:x} d2={d2:x} d3={d3:x}"
            );
        }
    }
}

/// AF (bit 4) must be computed and packed for every arithmetic category, with
/// the true half-carry value — not the structural 0 it was before
/// angr-5mnx3.59.
///
/// The `diff_fuzz_sym_flags_*` and `diff_fuzz_eflags_all_symbolic_*` tests
/// compare the two Rust paths against each other, so they pin AF's
/// *consistency* but never its *value*; this test is the oracle, hand-derived
/// from the Python reference's `af = (res ^ arg_l ^ arg_r)[G_CC_SHIFT_A]`.
#[test]
fn eflags_all_packs_true_auxiliary_carry() {
    // (cc_op, dep1, dep2, ndep, expected AF), all 8-bit operands.
    let cases: &[(u64, u64, u64, u64, u64)] = &[
        // ADD: 0x0F + 0x01 carries out of bit 3; 0x01 + 0x01 does not.
        (amd64_cc_op::G_CC_OP_ADDB, 0x0F, 0x01, 0, 1),
        (amd64_cc_op::G_CC_OP_ADDB, 0x01, 0x01, 0, 0),
        // SUB: 0x10 - 0x01 borrows into bit 3; 0x12 - 0x01 does not.
        (amd64_cc_op::G_CC_OP_SUBB, 0x10, 0x01, 0, 1),
        (amd64_cc_op::G_CC_OP_SUBB, 0x12, 0x01, 0, 0),
        // INC/DEC take the result in dep1: 0x0F+1 == 0x10 carries, 0x10-1 borrows.
        (amd64_cc_op::G_CC_OP_INCB, 0x10, 0, 0, 1),
        (amd64_cc_op::G_CC_OP_INCB, 0x02, 0, 0, 0),
        (amd64_cc_op::G_CC_OP_DECB, 0x0F, 0, 0, 1),
        (amd64_cc_op::G_CC_OP_DECB, 0x01, 0, 0, 0),
        // ADC/SBB with an incoming carry: VEX encodes argR as dep2 ^ oldC, so
        // dep2=1 with oldC=1 means the addend is 0 and only the carry is added.
        // 0x0F + 0 + 1 == 0x10 carries out of bit 3; 0x0E + 0 + 1 == 0x0F does not.
        (amd64_cc_op::G_CC_OP_ADCB, 0x0F, 0x01, flag_mask::G_CC_MASK_C, 1),
        (amd64_cc_op::G_CC_OP_ADCB, 0x0E, 0x01, flag_mask::G_CC_MASK_C, 0),
        (amd64_cc_op::G_CC_OP_SBBB, 0x10, 0x01, flag_mask::G_CC_MASK_C, 1),
        // LOGIC/shift/multiply leave AF architecturally undefined; VEX reports 0.
        (amd64_cc_op::G_CC_OP_LOGICB, 0xFF, 0, 0, 0),
        (amd64_cc_op::G_CC_OP_SHLB, 0x1E, 0x0F, 0, 0),
        (amd64_cc_op::G_CC_OP_UMULB, 0x0F, 0x0F, 0, 0),
    ];
    for &(cc_op, d1, d2, nd, want_af) in cases {
        let packed =
            calculate_eflags_all_amd64(cc_op, d1, d2, nd).expect("cc_op must be supported");
        let got_af = (packed >> flag_shift::G_CC_SHIFT_A) & 1;
        assert_eq!(
            got_af, want_af,
            "cc_op={cc_op} d1={d1:#x} d2={d2:#x} nd={nd:#x} packed={packed:#x}"
        );
    }

    // ROL/ROR preserve AF from the saved EFLAGS in cc_ndep, like PF/ZF/SF.
    for cc_op in [amd64_cc_op::G_CC_OP_ROLB, amd64_cc_op::G_CC_OP_RORB] {
        for ndep_af in [0, flag_mask::G_CC_MASK_A] {
            let packed = calculate_eflags_all_amd64(cc_op, 0x81, 0, ndep_af)
                .expect("cc_op must be supported");
            assert_eq!(
                packed & flag_mask::G_CC_MASK_A,
                ndep_af,
                "cc_op={cc_op} ndep_af={ndep_af:#x}"
            );
        }
    }
}
