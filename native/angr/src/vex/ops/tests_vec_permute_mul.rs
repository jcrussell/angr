// angr-9hleg: vector reverse / low-half packed-mul tests (mirror of ops/vec_permute_mul.rs).

use super::test_helpers::*;
use super::*;
use crate::vex::ir::IRType;

// =========================================================================
// VReverse — byte/halfword/word/bit reversal within lane (angr-tukg.4).
// =========================================================================

/// Iop_Reverse8sIn32_x2 — byte-swap within each 32-bit word (REV32
/// applied to a NEON D-register). 64-bit total.
#[test]
fn test_vec_reverse_8in32_x2_concrete() {
    let ctx = SymContext::new_mock();
    // Two 32-bit lanes: low = 0x11223344, high = 0xAABBCCDD.
    let v: u128 = 0xAABBCCDD_11223344u128;
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 8,
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(v, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    // Low lane bytes reversed: 0x11223344 → 0x44332211.
    // High lane: 0xAABBCCDD → 0xDDCCBBAA.
    let expected: u128 = 0xDDCCBBAA_44332211u128;
    assert_eq!(result.as_u128().unwrap(), expected);
}

/// Iop_Reverse32sIn64_x2 — swap the two 32-bit halves of each 64-bit
/// lane. Directly mirrors the only explicit Python reference at
/// `angr/engines/vex/claripy/irop.py:_op_Iop_Reverse32sIn64_x2`.
#[test]
fn test_vec_reverse_32in64_x2_concrete_matches_python_ref() {
    let ctx = SymContext::new_mock();
    // Python ref: Concat(arg[95:64], arg[127:96], arg[31:0], arg[63:32]).
    // Pick a 128-bit value with distinct 32-bit slices to exercise every
    // permutation slot.
    let v: u128 = 0xAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDDu128;
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 32,
            elem: IRType::I64,
            count: 2,
        },
        RustBV::concrete(v, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    // Slices of v (LSB → MSB indexing): [31:0]=DDDDDDDD, [63:32]=CCCCCCCC,
    //                                  [95:64]=BBBBBBBB, [127:96]=AAAAAAAA.
    // Python Concat(MSB→LSB): [95:64], [127:96], [31:0], [63:32]
    //   = BBBBBBBB AAAAAAAA DDDDDDDD CCCCCCCC (MSB→LSB)
    let expected: u128 = 0xBBBBBBBB_AAAAAAAA_DDDDDDDD_CCCCCCCCu128;
    assert_eq!(result.as_u128().unwrap(), expected);
}

/// Iop_Reverse1sIn8_x8 — RBIT: reverse the bit order inside each byte.
#[test]
fn test_vec_reverse_1in8_x8_concrete() {
    let ctx = SymContext::new_mock();
    // Byte 0 = 0b10110010 = 0xB2; reversed = 0b01001101 = 0x4D.
    // Byte 1 = 0xFF (palindrome); reversed = 0xFF.
    // Byte 2 = 0x01; reversed = 0x80.
    // Byte 3 = 0x80; reversed = 0x01.
    // Byte 4 = 0xA5; reversed = 0xA5 (10100101 → 10100101).
    // Byte 5 = 0x00; reversed = 0x00.
    // Byte 6 = 0x0F; reversed = 0xF0.
    // Byte 7 = 0xF0; reversed = 0x0F.
    let in_bytes: [u8; 8] = [0xB2, 0xFF, 0x01, 0x80, 0xA5, 0x00, 0x0F, 0xF0];
    let exp_bytes: [u8; 8] = [0x4D, 0xFF, 0x80, 0x01, 0xA5, 0x00, 0xF0, 0x0F];
    let mut v: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        v |= (in_bytes[i] as u128) << (i * 8);
        e |= (exp_bytes[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 1,
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(v, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Reverse16sIn64_x2 — halfword swap inside each 64-bit lane,
/// applied across two lanes (128-bit Q-register form).
#[test]
fn test_vec_reverse_16in64_x2_concrete() {
    let ctx = SymContext::new_mock();
    // Lane 0 (low 64): halfwords [0x1111, 0x2222, 0x3333, 0x4444] (LSB→MSB).
    // After reversal: [0x4444, 0x3333, 0x2222, 0x1111].
    // Lane 1 (high 64): halfwords [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD].
    // After reversal: [0xDDDD, 0xCCCC, 0xBBBB, 0xAAAA].
    let v: u128 = 0xDDDD_CCCC_BBBB_AAAA_4444_3333_2222_1111u128;
    let expected: u128 = 0xAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444u128;
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 16,
            elem: IRType::I64,
            count: 2,
        },
        RustBV::concrete(v, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), expected);
}

/// Involution: applying VReverse twice is the identity (any permutation
/// that swaps positions i ↔ n-1-i is its own inverse). Exercise on a
/// symbolic 128-bit input via a Z3 equivalence check.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_reverse_double_apply_is_identity() {
    let ctx = SymContext::new_mock();
    for (sub_width, elem, count) in [
        (8u8, IRType::I32, 4u8), // Reverse8sIn32_x4
        (16, IRType::I64, 2),    // Reverse16sIn64_x2
        (32, IRType::I64, 2),    // Reverse32sIn64_x2
        (1, IRType::I8, 16),     // Reverse1sIn8_x16
    ] {
        let width = elem.bits() * count as u32;
        let arg = RustBV::symbolic(&ctx, "vrev_arg", width);
        let op = IROp::VReverse {
            sub_width,
            elem,
            count,
        };
        let once = VEXOps::unop(op, arg.clone(), &ctx).unwrap();
        let twice = VEXOps::unop(op, once, &ctx).unwrap();
        // Assert there is no satisfying assignment where twice != arg.
        ctx.push();
        ctx.add_constraint(twice.to_z3_ast().eq(arg.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "double-apply must equal identity for sub_width={sub_width} elem={elem:?} count={count}"
        );
        ctx.pop();
    }
}

/// Symbolic parity vs claripy reference: `Iop_Reverse32sIn64_x2` must
/// produce the same bits as the explicit Python implementation
/// `Concat(arg[95:64], arg[127:96], arg[31:0], arg[63:32])` for any
/// 128-bit input. Verified through a Z3 universality check.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_reverse_32in64_x2_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "vrev_arg", 128);
    let got = VEXOps::unop(
        IROp::VReverse {
            sub_width: 32,
            elem: IRType::I64,
            count: 2,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();

    // Build the Python reference: Concat(arg[95:64], arg[127:96],
    //                                    arg[31:0],  arg[63:32]).
    // `concat_le_elements` indexes 0 → LSB, so push in LSB→MSB order:
    //   bits [31:0]  output  ← arg[63:32]
    //   bits [63:32] output  ← arg[31:0]
    //   bits [95:64] output  ← arg[127:96]
    //   bits [127:96] output ← arg[95:64]
    let py = VEXOps::concat_le_elements(
        vec![
            arg.extract(63, 32, &ctx),
            arg.extract(31, 0, &ctx),
            arg.extract(127, 96, &ctx),
            arg.extract(95, 64, &ctx),
        ],
        &ctx,
    );

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VReverse 32sIn64_x2 must match the Python reference Concat pattern"
    );
    ctx.pop();
}

// =========================================================================
// VMull — widening vector multiply (angr-ph300.78).
//   Full-lane  Iop_Mull{N}{S,U}x{M}     (I64,I64)->V128, NEON VMULL
//   Even-lane  Iop_MullEven{N}{S,U}x{M} (V128,V128)->V128, SSE PMUL[U]DQ
// Each contributing lane widens N->2N (sign/zero) before multiplying.
// =========================================================================

/// Iop_Mull8Ux8 — NEON VMULL.U8: 8 unsigned byte lanes (64-bit input) widened
/// to 8 u16 output lanes (128-bit). Exercises zero-extension incl. 255*255.
#[test]
fn test_vec_mull_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    let l: [u128; 8] = [2, 3, 255, 0, 16, 100, 7, 1];
    let r: [u128; 8] = [4, 5, 255, 9, 16, 2, 7, 255];
    let exp: [u128; 8] = [8, 15, 65025, 0, 256, 200, 49, 255];
    let lv = pack_lanes_uint(&l, 8);
    let rv = pack_lanes_uint(&r, 8);
    let result = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I8,
            count: 8,
            signed: false,
            even: false,
        },
        RustBV::concrete(lv, 64),
        RustBV::concrete(rv, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 16);
}

/// Iop_Mull16Sx4 — NEON VMULL.S16: 4 signed i16 lanes (64-bit input) widened
/// to 4 i32 output lanes (128-bit). Exercises sign-extension and the
/// -32768*-1 = 32768 case that overflows a 16-bit lane but not 32-bit.
#[test]
fn test_vec_mull_16sx4_concrete() {
    let ctx = SymContext::new_mock();
    let l: [i16; 4] = [-2, 100, -32768, 3];
    let r: [i16; 4] = [3, -4, -1, -5];
    let exp: [i32; 4] = [-6, -400, 32768, -15];
    let lv = pack_lanes_uint(&l.map(|x| x as u16 as u128), 16);
    let rv = pack_lanes_uint(&r.map(|x| x as u16 as u128), 16);
    let result = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I16,
            count: 4,
            signed: true,
            even: false,
        },
        RustBV::concrete(lv, 64),
        RustBV::concrete(rv, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u32 as u128),
        32,
    );
}

/// Iop_Mull16Ux4 — NEON VMULL.U16: 4 unsigned u16 lanes (64-bit input) widened
/// to 4 u32 output lanes (128-bit). Exercises zero-extension and the
/// 65535*65535 = 4294836225 case that overflows a 16-bit lane but not 32-bit.
/// The unsigned 16-lane family had only routing coverage before (angr-n0irt.21).
#[test]
fn test_vec_mull_16ux4_concrete() {
    let ctx = SymContext::new_mock();
    let l: [u128; 4] = [10, 65535, 0, 256];
    let r: [u128; 4] = [20, 65535, 5, 256];
    let exp: [u128; 4] = [200, 4_294_836_225, 0, 65536];
    let lv = pack_lanes_uint(&l, 16);
    let rv = pack_lanes_uint(&r, 16);
    let result = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I16,
            count: 4,
            signed: false,
            even: false,
        },
        RustBV::concrete(lv, 64),
        RustBV::concrete(rv, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 32);
}

/// Iop_Mull32Sx2 — NEON VMULL.S32: 2 signed i32 lanes (64-bit input) widened to
/// 2 i64 output lanes (128-bit). Exercises the widest full-lane signed boundary
/// i32::MIN * i32::MIN = 2^62, which overflows i32 but fits i64 — a case the
/// pre-existing tests never hit (angr-n0irt.21).
#[test]
fn test_vec_mull_32sx2_concrete() {
    let ctx = SymContext::new_mock();
    let l: [i32; 2] = [i32::MIN, -3];
    let r: [i32; 2] = [i32::MIN, 5];
    // i32::MIN^2 = 4611686018427387904 = 0x4000_0000_0000_0000; -3*5 = -15.
    let exp: [i64; 2] = [0x4000_0000_0000_0000, -15];
    let lv = pack_lanes_uint(&l.map(|x| x as u32 as u128), 32);
    let rv = pack_lanes_uint(&r.map(|x| x as u32 as u128), 32);
    let result = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I32,
            count: 2,
            signed: true,
            even: false,
        },
        RustBV::concrete(lv, 64),
        RustBV::concrete(rv, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u64 as u128),
        64,
    );
}

/// Iop_MullEven32Sx4 — SSE PMULDQ (signed): multiplies the even 32-bit lanes
/// (0, 2) of two V128 inputs with sign-extension. This is the signed+even
/// intersection that had zero coverage (angr-n0irt.21): the concrete even test
/// was unsigned-only. Even lane 0 exercises the i32::MIN^2 boundary; lane 2 a
/// mixed-sign product. Odd lanes (1, 3) are decoys that must be ignored.
#[test]
fn test_vec_mull_even_32sx4_concrete() {
    let ctx = SymContext::new_mock();
    let l: [i32; 4] = [i32::MIN, 0x7EAD_BEEF, -5, 0x1234];
    let r: [i32; 4] = [i32::MIN, 0x1CAF_EBAB, 7, 0x5678];
    // even lane 0: i32::MIN^2 = 0x4000_0000_0000_0000; even lane 2: -5*7 = -35.
    let exp: [i64; 2] = [0x4000_0000_0000_0000, -35];
    let lv = pack_lanes_uint(&l.map(|x| x as u32 as u128), 32);
    let rv = pack_lanes_uint(&r.map(|x| x as u32 as u128), 32);
    let result = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I32,
            count: 4,
            signed: true,
            even: true,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u64 as u128),
        64,
    );
}

/// Iop_MullEven32Ux4 — SSE PMULUDQ: multiplies the even 32-bit lanes (0, 2) of
/// two 128-bit inputs, producing 2 u64 output lanes. Odd lanes (1, 3) are
/// ignored. Exercises the widest widening (32->64) and full-range 0xFFFFFFFF^2.
#[test]
fn test_vec_mull_even_32ux4_concrete() {
    let ctx = SymContext::new_mock();
    // Lanes 0..3; only 0 and 2 (even) contribute. 1 and 3 are decoys.
    let l: [u128; 4] = [0xFFFF_FFFF, 0xDEAD_BEEF, 5, 0x1_0000];
    let r: [u128; 4] = [0xFFFF_FFFF, 0xCAFE_BABE, 7, 0x1_0000];
    // even lane 0: 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE_00000001
    // even lane 2: 5 * 7 = 35
    let exp: [u128; 2] = [0xFFFF_FFFE_0000_0001, 35];
    let lv = pack_lanes_uint(&l, 32);
    let rv = pack_lanes_uint(&r, 32);
    let result = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I32,
            count: 4,
            signed: false,
            even: true,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 64);
}

/// Symbolic parity: Iop_Mull16Sx4 on fully-symbolic operands must equal the
/// reference concat of per-lane sign-extended 16->32 products, for every
/// input. Verified by a Z3 universality check (no satisfying counter-example).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_mull_16sx4_symbolic_matches_reference() {
    let ctx = SymContext::new_mock();
    let left = RustBV::symbolic(&ctx, "mull_l", 64);
    let right = RustBV::symbolic(&ctx, "mull_r", 64);
    let got = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I16,
            count: 4,
            signed: true,
            even: false,
        },
        left.clone(),
        right.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: for each of the 4 lanes, sign-extend both 16-bit lanes to 32
    // bits, multiply, and concat low-to-high.
    let mut lanes = Vec::with_capacity(4);
    for i in 0..4u32 {
        let lo = i * 16;
        let hi = lo + 15;
        let la = left.extract(hi, lo, &ctx).extend_into(32, true, &ctx);
        let ra = right.extract(hi, lo, &ctx).extend_into(32, true, &ctx);
        lanes.push(la.mul(&ra, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VMull 16Sx4 must match the per-lane sign-extended product reference"
    );
    ctx.pop();
}

/// Symbolic parity for the even-lane family: Iop_MullEven32Sx4 (even=true,
/// step=2) on fully-symbolic V128 operands must equal the reference concat of
/// per-even-lane sign-extended 32->64 products. This is the intersection the
/// concrete `test_vec_mull_even_32ux4_concrete` and the symbolic
/// `test_vec_mull_16sx4_symbolic_matches_reference` each miss: the symbolic
/// branch's `lo = j * step * in_width` even-lane arithmetic runs only here.
/// If someone simplified that to `j * in_width` (dropping the step-2
/// multiplier), the symbolic MullEven path would silently multiply adjacent
/// instead of every-other lanes; the universality check below catches it since
/// the reference selects lanes 0 and 2 explicitly.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_mull_even_32sx4_symbolic_matches_reference() {
    let ctx = SymContext::new_mock();
    let left = RustBV::symbolic(&ctx, "mulleven_l", 128);
    let right = RustBV::symbolic(&ctx, "mulleven_r", 128);
    let got = VEXOps::binop(
        IROp::VMull {
            elem: IRType::I32,
            count: 4,
            signed: true,
            even: true,
        },
        left.clone(),
        right.clone(),
        &ctx,
    )
    .unwrap();
    assert_eq!(got.width(), 128);

    // Reference: only the even 32-bit lanes 0 and 2 contribute. Each is
    // sign-extended 32->64, multiplied, and the two products concat low-to-high
    // into 2 u64 output lanes. `lo` steps by 2*32=64 to skip the odd lanes.
    let mut lanes = Vec::with_capacity(2);
    for j in 0..2u32 {
        let lo = j * 2 * 32;
        let hi = lo + 31;
        let la = left.extract(hi, lo, &ctx).extend_into(64, true, &ctx);
        let ra = right.extract(hi, lo, &ctx).extend_into(64, true, &ctx);
        lanes.push(la.mul(&ra, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VMull even 32Sx4 must select only even lanes (0,2) and match the \
         per-lane sign-extended product reference"
    );
    ctx.pop();
}

// =========================================================================
// VQDMull — signed doubling saturating widening multiply (angr-1yge9.5).
//   Iop_QDMull{N}Sx{M}  (I64,I64)->V128, NEON VQDMULL. Always signed,
//   full-lane. result_lane = SignedSat(2 * sext(la) * sext(ra), 2N bits).
// =========================================================================

/// Iop_QDMull16Sx4 — VQDMULL.S16: 4 signed i16 lanes doubled-multiplied to 4
/// i32 lanes. Third lane is the saturation corner (-32768 * -32768 = 2^30,
/// doubled = 2^31, which overflows i32 max 2^31-1 and clamps).
#[test]
fn test_vec_qdmull_16sx4_concrete() {
    let ctx = SymContext::new_mock();
    let l: [i16; 4] = [-2, 100, -32768, 3];
    let r: [i16; 4] = [3, -4, -32768, -5];
    // 2*(-2*3)=-12, 2*(100*-4)=-800, 2*(2^30)=2^31 -> sat i32::MAX, 2*(3*-5)=-30
    let exp: [i32; 4] = [-12, -800, i32::MAX, -30];
    let lv = pack_lanes_uint(&l.map(|x| x as u16 as u128), 16);
    let rv = pack_lanes_uint(&r.map(|x| x as u16 as u128), 16);
    let result = VEXOps::binop(
        IROp::VQDMull {
            elem: IRType::I16,
            count: 4,
        },
        RustBV::concrete(lv, 64),
        RustBV::concrete(rv, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u32 as u128),
        32,
    );
}

/// Iop_QDMull32Sx2 — VQDMULL.S32: 2 signed i32 lanes doubled-multiplied to 2
/// i64 lanes. Second lane is the saturation corner (i32::MIN squared, doubled
/// = 2^63, overflowing i64::MAX).
#[test]
fn test_vec_qdmull_32sx2_concrete() {
    let ctx = SymContext::new_mock();
    let l: [i32; 2] = [7, i32::MIN];
    let r: [i32; 2] = [8, i32::MIN];
    // 2*(7*8)=112, 2*(2^62)=2^63 -> sat i64::MAX
    let exp: [i64; 2] = [112, i64::MAX];
    let lv = pack_lanes_uint(&l.map(|x| x as u32 as u128), 32);
    let rv = pack_lanes_uint(&r.map(|x| x as u32 as u128), 32);
    let result = VEXOps::binop(
        IROp::VQDMull {
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(lv, 32 * 2),
        RustBV::concrete(rv, 32 * 2),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u64 as u128),
        64,
    );
}

/// Symbolic parity: Iop_QDMull32Sx2 on fully-symbolic operands must equal an
/// independently-built reference (widen 32->128, multiply, double, clamp to the
/// signed 64-bit range via explicit sgt/slt), for every input. Universality
/// check — no satisfying counter-example. Independent of the impl's internal
/// `mul_width` so it catches lane-layout, doubling, and saturation-direction
/// bugs, not just a copy of the same arithmetic.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_qdmull_32sx2_symbolic_matches_reference() {
    let ctx = SymContext::new_mock();
    let left = RustBV::symbolic(&ctx, "qdmull_l", 64);
    let right = RustBV::symbolic(&ctx, "qdmull_r", 64);
    let got = VEXOps::binop(
        IROp::VQDMull {
            elem: IRType::I32,
            count: 2,
        },
        left.clone(),
        right.clone(),
        &ctx,
    )
    .unwrap();
    assert_eq!(got.width(), 128);

    let smax = RustBV::concrete(i64::MAX as u128, 128);
    let smin = RustBV::concrete((i64::MIN as i128) as u128, 128);
    let mut lanes = Vec::with_capacity(2);
    for j in 0..2u32 {
        let lo = j * 32;
        let hi = lo + 31;
        let la = left.extract(hi, lo, &ctx).extend_into(128, true, &ctx);
        let ra = right.extract(hi, lo, &ctx).extend_into(128, true, &ctx);
        let prod = la.mul(&ra, &ctx);
        let doubled = prod.add(&prod, &ctx);
        // clamp to [i64::MIN, i64::MAX] in 128-bit signed space, then truncate.
        let hi_clamped = doubled.sgt(&smax, &ctx).ite(&smax, &doubled, &ctx);
        let clamped = hi_clamped.slt(&smin, &ctx).ite(&smin, &hi_clamped, &ctx);
        lanes.push(clamped.extract(63, 0, &ctx));
    }
    let reference = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(reference.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VQDMull 32Sx2 must match the doubled sign-extended saturated product"
    );
    ctx.pop();
}

// =========================================================================
// VPwBitMtxXpose — PPC vgbbd 8x8 bit-matrix transpose (angr-sqfj8.143).
//   Iop_PwBitMtxXpose64x2, V128 -> V128, each 64-bit half independent.
// =========================================================================

/// Independent reference for PPC `vgbbd`, written directly in the ISA 2.07
/// big-endian numbering the instruction is specified in (byte 0 = most
/// significant byte of the doubleword, bit 0 = most significant bit of the
/// byte):
///
///   VRT.dword[i].byte[j].bit[k] <- VRB.dword[i].byte[k].bit[j]
///
/// `vec_bit_mtx_xpose` uses little-endian numbering instead, so this is a
/// genuinely independent replay of the spec rather than a copy of the
/// implementation — it catches a byte/bit-order confusion in either direction.
fn vgbbd_be_reference(v: u128) -> u128 {
    let mut out: u128 = 0;
    for half_shift in [0u32, 64] {
        let src = ((v >> half_shift) & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        let mut dst: u64 = 0;
        for j in 0..8u32 {
            for k in 0..8u32 {
                // BE byte b occupies bits [63-8b ..= 56-8b]; BE bit index i
                // inside it sits at 56 - 8*b + (7 - i).
                let src_bit = (src >> (56 - 8 * k + (7 - j))) & 1;
                dst |= src_bit << (56 - 8 * j + (7 - k));
            }
        }
        out |= (dst as u128) << half_shift;
    }
    out
}

/// The canonical `vgbbd` use: every byte's most significant bit is gathered
/// into a single byte of the result. Low half has all eight MSBs set (each
/// byte 0x80) and collapses to one 0xFF byte; the high half has a single set
/// bit and moves it to the opposite corner of the matrix.
#[test]
fn test_vec_bit_mtx_xpose_concrete_gathers_msbs() {
    let ctx = SymContext::new_mock();
    let v: u128 = 0x01000000_00000000_80808080_80808080u128;
    let result = VEXOps::unop(IROp::VPwBitMtxXpose, RustBV::concrete(v, 128), &ctx).unwrap();
    assert_eq!(result.width(), 128);
    // Low half: LE bit 8j+7 set for every j → transposed to LE bits 56..=63,
    // i.e. the top byte of the half becomes 0xFF.
    // High half: only LE byte 7 bit 0 set → transposed to LE byte 0 bit 7.
    let expected: u128 = 0x00000000_00000080_FF000000_00000000u128;
    assert_eq!(result.as_u128().unwrap(), expected);
    assert_eq!(expected, vgbbd_be_reference(v));
}

/// Concrete path vs the big-endian spec replay over a spread of patterns,
/// including asymmetric ones where a transposed-the-wrong-way result would
/// differ.
#[test]
fn test_vec_bit_mtx_xpose_concrete_matches_be_spec_replay() {
    let ctx = SymContext::new_mock();
    let cases: [u128; 6] = [
        0,
        u128::MAX,
        // Anti-diagonal per half: the identity matrix under BE numbering, so
        // the transpose is a fixed point (guards against a no-op passing by
        // accident — the other cases are not fixed points).
        0x80402010_08040201_80402010_08040201u128,
        // One set bit per half, off the diagonal.
        0x00000000_00000001_01000000_00000000u128,
        // Distinct bytes so every (j, k) slot is exercised.
        0x0123_4567_89AB_CDEFu128 | (0xFEDC_BA98_7654_3210u128 << 64),
        0xDEADBEEF_CAFEBABE_0BADF00D_5AA55AA5u128,
    ];
    for v in cases {
        let result = VEXOps::unop(IROp::VPwBitMtxXpose, RustBV::concrete(v, 128), &ctx).unwrap();
        assert_eq!(
            result.as_u128().unwrap(),
            vgbbd_be_reference(v),
            "vgbbd mismatch for input {v:#034x}"
        );
    }
}

/// A transpose is its own inverse, so applying the op twice must be the
/// identity — proved over a fully symbolic operand, which exercises the
/// per-bit extract/concat path rather than the u128 fold.
///
/// z3-gated (angr-c7xno.99): `SymContext::add_constraint` / `RustBV::to_z3_ast`
/// only exist under `vex-engine-z3`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_bit_mtx_xpose_symbolic_double_apply_is_identity() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "vgbbd_arg", 128);
    let once = VEXOps::unop(IROp::VPwBitMtxXpose, arg.clone(), &ctx).unwrap();
    assert!(once.as_u128().is_none(), "operand must stay symbolic");
    let twice = VEXOps::unop(IROp::VPwBitMtxXpose, once, &ctx).unwrap();
    assert_eq!(twice.width(), 128);

    ctx.push();
    ctx.add_constraint(twice.to_z3_ast().eq(arg.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwBitMtxXpose applied twice must be the identity"
    );
    ctx.pop();
}

/// The symbolic per-bit path must agree with the concrete u128 fold: pin a
/// symbolic operand to a known value and eval the result.
///
/// z3-gated (angr-c7xno.99): see the sibling double-apply test above.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_bit_mtx_xpose_symbolic_matches_concrete_path() {
    let ctx = SymContext::new_mock();
    let v: u128 = 0xDEADBEEF_CAFEBABE_0BADF00D_5AA55AA5u128;
    let arg = RustBV::symbolic(&ctx, "vgbbd_pinned", 128);
    let got = VEXOps::unop(IROp::VPwBitMtxXpose, arg.clone(), &ctx).unwrap();

    ctx.push();
    ctx.add_constraint(arg.to_z3_ast().eq(RustBV::concrete(v, 128).to_z3_ast()));
    assert!(ctx.is_sat());
    let evaluated = ctx.eval(&got).expect("pinned operand must evaluate");
    ctx.pop();
    assert_eq!(evaluated, vgbbd_be_reference(v));
}
