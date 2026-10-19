//! Tests for the native libVEX lifter's pure marshalling helpers.
//!
//! A real lift needs the `libpyvex` an imported pyvex loads, which `cargo
//! test` does not have, so lifting, native-vs-pyvex parity, and the enum-tag
//! and struct-layout checks against pyvex's cffi library live in
//! `tests/engines/rust/test_native_lift_parity.py`.

use super::*;
use crate::vex::ir::IRLoadGOp;

/// libVEX stores restricted-vector consts as one bit per byte lane; pyvex
/// expands them, so the marshaller must too. Ordinary code only ever
/// produces the all-zero and all-ones patterns, so cover a mixed one here.
#[test]
fn test_expand_v128_pattern() {
    assert_eq!(expand_v128(0x0000), 0);
    assert_eq!(expand_v128(0xffff), u128::MAX);
    assert_eq!(expand_v128(0x0001), 0xff);
    assert_eq!(expand_v128(0x8000), 0xff << 120);
    assert_eq!(expand_v128(0x0003), 0xffff);
}

#[test]
fn test_expand_v256_pattern() {
    assert_eq!(expand_v256(0x0000_0000), [0; 4]);
    assert_eq!(expand_v256(0xffff_ffff), [u64::MAX; 4]);
    assert_eq!(expand_v256(0x0000_0001), [0xff, 0, 0, 0]);
    // Bit 8 is the first lane of the second limb.
    assert_eq!(expand_v256(0x0000_0100), [0, 0xff, 0, 0]);
    assert_eq!(expand_v256(0x8000_0000), [0, 0, 0, 0xff << 56]);
}

/// Pins the `IRLoadGOp` mapping, so a renumbered tag or a dropped arm shows
/// up as a failure rather than a silent `Unknown` (which makes the
/// interpreter drop the widening and load the wrong value).
#[test]
fn test_loadg_op_handles_every_variant() {
    for ident in [
        ILGOP_IDENT_V128,
        ILGOP_IDENT64,
        ILGOP_IDENT32,
        ILGOP_IDENT16,
        ILGOP_IDENT8,
    ] {
        assert_eq!(loadg_op(ident), IRLoadGOp::Identity, "tag {ident:#x}");
    }
    assert_eq!(loadg_op(ILGOP_8UTO32), IRLoadGOp::WidenZ { src_bits: 8 });
    assert_eq!(loadg_op(ILGOP_8STO32), IRLoadGOp::WidenS { src_bits: 8 });
    assert_eq!(loadg_op(ILGOP_16UTO32), IRLoadGOp::WidenZ { src_bits: 16 });
    assert_eq!(loadg_op(ILGOP_16STO32), IRLoadGOp::WidenS { src_bits: 16 });
    // ILGop_INVALID is the one tag that legitimately has no mapping.
    assert_eq!(loadg_op(0x1D00), IRLoadGOp::Unknown);
}

/// Pins the drift fallbacks of `c_endness` / `c_jumpkind` / `c_type_parse` /
/// `marshal_const` (angr-sqfj8.119). Unlike `c_op`, none of the four has a
/// sentinel variant that can carry the raw tag, so an out-of-range tag
/// collapses to a plausible-but-possibly-wrong default; the contract is that
/// it does so *predictably* and with a `log::warn!`. A real tag that lands
/// here belongs in the name tables (or `marshal_const`) instead.
#[test]
fn test_unknown_discriminants_fall_back_predictably() {
    // 0 is not a valid tag for any of the four enums (libVEX bases them at
    // 0x1100 / 0x1200 / 0x1300 / 0x1A00-style offsets), so it exercises the
    // miss path.
    assert_eq!(c_endness(0), Endness::Little);
    assert_eq!(c_jumpkind(0), JumpKind::Boring);
    assert_eq!(c_type_parse(0), IRType::I64);
    // One past the end of each table misses too.
    assert_eq!(c_jumpkind(IJK_BASE + IJK_NAMES.len() as u32), JumpKind::Boring);
    assert_eq!(
        c_type_parse(ffi::ITY_INVALID + ITY_NAMES.len() as u32),
        IRType::I64
    );

    let unknown_const = ffi::IRConst {
        tag: 0,
        ico: ffi::IcoUnion { u64_: 0 },
    };
    assert_eq!(unsafe { marshal_const(&unknown_const) }, IRConst::U64(0));
}

/// Every `Ico_*` tag decodes as pyvex's `IRConst._from_c` does.
/// `test_native_lift_parity.py` checks this tag list is pyvex's whole set.
#[test]
fn test_marshal_const_handles_every_variant() {
    let cases: Vec<(u32, ffi::IcoUnion, IRConst)> = vec![
        (ffi::ICO_U1, ffi::IcoUnion { u1: 1 }, IRConst::U1(true)),
        (ffi::ICO_U8, ffi::IcoUnion { u8_: 0x7a }, IRConst::U8(0x7a)),
        (
            ffi::ICO_U16,
            ffi::IcoUnion { u16_: 0x7aab },
            IRConst::U16(0x7aab),
        ),
        (
            ffi::ICO_U32,
            ffi::IcoUnion { u32_: 0x7aab_cdee },
            IRConst::U32(0x7aab_cdee),
        ),
        (
            ffi::ICO_U64,
            ffi::IcoUnion {
                u64_: 0x7aab_cdee_f001_1223,
            },
            IRConst::U64(0x7aab_cdee_f001_1223),
        ),
        (
            // Restricted encoding, like V128.
            ffi::ICO_U128,
            ffi::IcoUnion { u128: 0x8001 },
            IRConst::U128((0xff << 120) | 0xff),
        ),
        (ffi::ICO_F32, ffi::IcoUnion { f32_: 1.5 }, IRConst::F32(1.5)),
        (
            // Bit pattern of 1.5f32.
            ffi::ICO_F32I,
            ffi::IcoUnion { f32i: 0x3fc0_0000 },
            IRConst::F32(1.5),
        ),
        (ffi::ICO_F64, ffi::IcoUnion { f64_: 1.5 }, IRConst::F64(1.5)),
        (
            // Bit pattern of 1.5f64.
            ffi::ICO_F64I,
            ffi::IcoUnion {
                f64i: 0x3ff8_0000_0000_0000,
            },
            IRConst::F64(1.5),
        ),
        (ffi::ICO_V128, ffi::IcoUnion { v128: 0x0001 }, IRConst::V128(0xff)),
        (
            ffi::ICO_V256,
            ffi::IcoUnion { v256: 0x0000_0001 },
            IRConst::V256([0xff, 0, 0, 0]),
        ),
        (
            // No 512-bit vector type: the documented zero stand-in.
            ffi::ICO_V512,
            ffi::IcoUnion { v512: 1 },
            IRConst::U64(0),
        ),
    ];

    for (tag, ico, expected) in cases {
        let c = ffi::IRConst { tag, ico };
        let got = unsafe { marshal_const(&c) };
        assert_eq!(got, expected, "tag {tag:#x} did not marshal as expected");
    }
}
