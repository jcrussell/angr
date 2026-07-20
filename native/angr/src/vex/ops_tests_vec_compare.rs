// angr-ph300.59: direct coverage for the vector interleave (unpack) family
// (ops_vec_compare.rs: vec_interleave_lo / vec_interleave_hi / vec_interleave).
// Before this file, grepping Interleave across every *tests*.rs returned zero
// hits: the 14 Iop_Interleave{LO,HI}{8x8..64x2} opcodes (opcode_map.rs) and both
// the concrete-u128 and symbolic extract+concat paths were entirely untested.
// The lane order (right arg -> even output positions, left arg -> odd, so the
// MSB lane comes from the left arg) is exactly what silently flips in a refactor.
//
// Expected-value arrays are independent reference constants derived from the
// x86 PUNPCK{L,H}BW / PUNPCK{L,H}QDQ semantics documented in libvex_ir.h, never
// read back from the code under test (vacuous-test audit invariant).

use super::ops_test_helpers::*;
use super::*;

// =========================================================================
// InterleaveLO8x16 — PUNPCKLBW reference (concrete)
// =========================================================================

/// PUNPCKLBW-style low interleave over 16x i8 lanes. Reads the low 8 bytes of
/// each operand and interleaves them; the right arg lands on even output lanes,
/// the left arg on odd, so output lane `2i` = right[i], lane `2i+1` = left[i].
/// Left/right bytes are disjoint value ranges (0x0N vs 0xFN) so any lane
/// swap or wrong-half read is immediately visible.
#[test]
fn test_vec_interleave_lo_8x16_concrete() {
    let ctx = SymContext::new_mock();

    let l: [u128; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F,
    ];
    let r: [u128; 16] = [
        0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE,
        0xFF,
    ];
    // Low half of each operand interleaved: r[0],l[0],r[1],l[1],...,r[7],l[7].
    let exp: [u128; 16] = [
        0xF0, 0x00, 0xF1, 0x01, 0xF2, 0x02, 0xF3, 0x03, 0xF4, 0x04, 0xF5, 0x05, 0xF6, 0x06, 0xF7,
        0x07,
    ];

    let lv = pack_lanes_uint(&l, 8);
    let rv = pack_lanes_uint(&r, 8);
    let result = VEXOps::binop(
        IROp::VInterleaveLO { elem: IRType::I8 },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 8);
}

// =========================================================================
// InterleaveHI8x16 — PUNPCKHBW reference (concrete)
// =========================================================================

/// PUNPCKHBW-style high interleave over 16x i8 lanes. Reads the *high* 8 bytes
/// of each operand (lanes 8..15): output lane `2i` = right[8+i],
/// lane `2i+1` = left[8+i]. Distinguishes the `high` base offset from the LO
/// case above — a base-selection bug would read the low half instead.
#[test]
fn test_vec_interleave_hi_8x16_concrete() {
    let ctx = SymContext::new_mock();

    let l: [u128; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F,
    ];
    let r: [u128; 16] = [
        0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE,
        0xFF,
    ];
    // High half of each operand interleaved: r[8],l[8],...,r[15],l[15].
    let exp: [u128; 16] = [
        0xF8, 0x08, 0xF9, 0x09, 0xFA, 0x0A, 0xFB, 0x0B, 0xFC, 0x0C, 0xFD, 0x0D, 0xFE, 0x0E, 0xFF,
        0x0F,
    ];

    let lv = pack_lanes_uint(&l, 8);
    let rv = pack_lanes_uint(&r, 8);
    let result = VEXOps::binop(
        IROp::VInterleaveHI { elem: IRType::I8 },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 8);
}

// =========================================================================
// InterleaveLO64x2 — PUNPCKLQDQ reference (concrete, widest lane)
// =========================================================================

/// PUNPCKLQDQ-style low interleave over 2x i64 lanes (half_count == 1): output
/// lane 0 = right[0], lane 1 = left[0]. Exercises the single-iteration,
/// full-64-bit-lane end of the family so a lane-width miscalc can't hide.
#[test]
fn test_vec_interleave_lo_64x2_concrete() {
    let ctx = SymContext::new_mock();

    let l: [u128; 2] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    let r: [u128; 2] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    // Low lane of each: right[0] to output lane 0, left[0] to lane 1.
    let exp: [u128; 2] = [0xAAAA_AAAA_AAAA_AAAA, 0x1111_1111_1111_1111];

    let lv = pack_lanes_uint(&l, 64);
    let rv = pack_lanes_uint(&r, 64);
    let result = VEXOps::binop(
        IROp::VInterleaveLO { elem: IRType::I64 },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 64);
}

// =========================================================================
// Symbolic InterleaveLO16x8 — extract+concat path under Z3
// =========================================================================

/// Symbolic low interleave over 8x i16 lanes: the left operand is a free vector
/// constrained to a concrete packing, forcing the symbolic extract+concat path
/// (vec_interleave's `else` branch, concat_le_elements). Evaluating the model
/// pins that the symbolic path produces the same lane order as the concrete
/// path: output lane `2i` = right[i], lane `2i+1` = left[i] for i in 0..4.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_interleave_lo_16x8_symbolic() {
    let ctx = SymContext::new_mock();

    let l_lanes: [u128; 8] = [
        0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007,
    ];
    let r_lanes: [u128; 8] = [
        0xF000, 0xF001, 0xF002, 0xF003, 0xF004, 0xF005, 0xF006, 0xF007,
    ];
    // Only the low 4 lanes of each operand are consumed by InterleaveLO16x8.
    let exp: [u128; 8] = [
        0xF000, 0x0000, 0xF001, 0x0001, 0xF002, 0x0002, 0xF003, 0x0003,
    ];

    let l = RustBV::symbolic(&ctx, "vinterleave_l", 128);
    ctx.add_constraint(
        l.to_z3_ast()
            .eq(RustBV::concrete(pack_lanes_uint(&l_lanes, 16), 128).to_z3_ast()),
    );
    let r = RustBV::concrete(pack_lanes_uint(&r_lanes, 16), 128);

    let result = VEXOps::binop(IROp::VInterleaveLO { elem: IRType::I16 }, l, r, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert!(ctx.is_sat(), "expected SAT after pinning left operand");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    assert_int_lanes_eq(model, &exp, 16);
}
