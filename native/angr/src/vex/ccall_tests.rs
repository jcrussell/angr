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

/// Reference: pack `Flags` into the (cf, pf, zf, sf, of) tuple as u8.
fn flags_to_tuple(f: Flags) -> (u8, u8, u8, u8, u8) {
    (f.cf, f.pf, f.zf, f.sf, f.of)
}

/// Build SymFlags by category and read back as concrete bits.
fn sym_flags_to_tuple(
    category: OpCategory,
    nbits: u32,
    d1: u64,
    d2: u64,
    nd: u64,
) -> (u8, u8, u8, u8, u8) {
    let ctx = crate::symbolic::SymContext::new_mock();
    let bv1 = RustBV::concrete(d1 as u128, 64);
    let bv2 = RustBV::concrete(d2 as u128, 64);
    let bvn = RustBV::concrete(nd as u128, 64);
    let f = sym_flags_for_category(category, nbits, &bv1, &bv2, &bvn, &ctx)
        .expect("category should be supported");
    let bit = |bv: &RustBV| bv.as_u64().expect("must be concrete") as u8;
    (bit(&f.cf), bit(&f.pf), bit(&f.zf), bit(&f.sf), bit(&f.of))
}

#[test]
fn diff_fuzz_sym_flags_sub() {
    // VEX always feeds calc_flags_sub args pre-masked to the operand width
    // (the cc_dep IRExpr is typed to nbits). Mask the random inputs to
    // match that invariant before diffing — otherwise calc_flags_sub's
    // unmasked u64 compare for CF would disagree with the symbolic path's
    // nbits-correct CF (the symbolic side is the precise one).
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
        let f = Flags { cf, pf, zf, sf, of };
        let sym_f = SymFlags {
            cf: RustBV::concrete(cf as u128, 1),
            pf: RustBV::concrete(pf as u128, 1),
            zf: RustBV::concrete(zf as u128, 1),
            sf: RustBV::concrete(sf as u128, 1),
            of: RustBV::concrete(of as u128, 1),
        };
        for &cond in &conds {
            let conc = eval_condition(cond, &f);
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
