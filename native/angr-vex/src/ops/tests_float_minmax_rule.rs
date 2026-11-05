//! Per-guest vector float min/max ([`FloatMinMaxRule`], angr-mx00p.128):
//! x86 MINPS/MAXPS pick the second operand on +-0 and NaN, AArch64 FMIN/FMAX
//! propagate the NaN quieted and order -0 below +0, ARM32 NEON VMIN/VMAX give
//! the default NaN and order -0 below +0 (angr-mx00p.138).

use super::*;
use crate::ir::IRType;

const VMIN: IROp = IROp::VFMin {
    elem: IRType::F32,
    count: 4,
};
const VMAX: IROp = IROp::VFMax {
    elem: IRType::F32,
    count: 4,
};
const QNAN: u32 = 0x7fc0_0001;
const SNAN: u32 = 0x7f80_0001;

fn lanes(guest: VexArch, op: IROp, l: [u32; 4], r: [u32; 4]) -> [u32; 4] {
    let pack = |v: [u32; 4]| {
        v.iter()
            .enumerate()
            .fold(0u128, |acc, (i, &b)| acc | (u128::from(b) << (32 * i)))
    };
    let out = VEXOps::binop_for_guest(
        guest,
        op,
        RustBV::concrete(pack(l), 128),
        RustBV::concrete(pack(r), 128),
    )
    .unwrap()
    .as_u128()
    .unwrap();
    std::array::from_fn(|i| (out >> (32 * i)) as u32)
}

const POS0: u32 = 0;
const NEG0: u32 = 0x8000_0000;
const ONE: u32 = 0x3f80_0000;
const TWO: u32 = 0x4000_0000;

/// Lanes: (+0, -0), (-0, +0), (QNaN, 1), (1, QNaN).
const L: [u32; 4] = [POS0, NEG0, QNAN, ONE];
const R: [u32; 4] = [NEG0, POS0, ONE, QNAN];

#[test]
fn x86_minmax_pick_the_second_operand_on_zeros_and_nan() {
    for guest in [VexArch::X86, VexArch::AMD64] {
        assert_eq!(lanes(guest, VMIN, L, R), R);
        assert_eq!(lanes(guest, VMAX, L, R), R);
        assert_eq!(lanes(guest, VMIN, [ONE; 4], [TWO; 4]), [ONE; 4]);
        assert_eq!(lanes(guest, VMAX, [ONE; 4], [TWO; 4]), [TWO; 4]);
    }
}

#[test]
fn arm64_minmax_order_zeros_and_propagate_nan() {
    let g = VexArch::ARM64;
    assert_eq!(lanes(g, VMIN, L, R), [NEG0, NEG0, QNAN, QNAN]);
    assert_eq!(lanes(g, VMAX, L, R), [POS0, POS0, QNAN, QNAN]);
    assert_eq!(lanes(g, VMIN, [ONE; 4], [TWO; 4]), [ONE; 4]);
    assert_eq!(lanes(g, VMAX, [ONE; 4], [TWO; 4]), [TWO; 4]);
}

/// A signalling NaN wins over a quiet one in either position and comes back
/// quieted; between two quiet NaNs the first operand wins.
#[test]
fn arm64_minmax_nan_priority() {
    let g = VexArch::ARM64;
    let other = 0x7fc0_0002;
    let l = [QNAN, SNAN, QNAN, SNAN];
    let r = [SNAN, QNAN, other, 0x7f80_0002];
    let quiet = |b: u32| b | 0x0040_0000;
    let expected = [quiet(SNAN), quiet(SNAN), QNAN, quiet(SNAN)];
    assert_eq!(lanes(g, VMIN, l, r), expected);
    assert_eq!(lanes(g, VMAX, l, r), expected);
}

#[test]
fn arm64_minmax_f64_and_pairwise() {
    let g = VexArch::ARM64;
    let neg0 = 1u128 << 63;
    let out = VEXOps::binop_for_guest(
        g,
        IROp::VFMin {
            elem: IRType::F64,
            count: 2,
        },
        RustBV::concrete(0, 128),
        RustBV::concrete(neg0 | (neg0 << 64), 128),
    )
    .unwrap();
    assert_eq!(out.as_u128(), Some(neg0 | (neg0 << 64)));
    // FMINP over (+0, -0) pairs: both pairs give -0.
    let pair = u128::from(NEG0) << 32;
    let out = VEXOps::binop_for_guest(
        g,
        IROp::VFPwMin {
            elem: IRType::F32,
            count: 2,
        },
        RustBV::concrete(pair, 64),
        RustBV::concrete(pair, 64),
    )
    .unwrap();
    assert_eq!(out.as_u128(), Some(u128::from(NEG0) | (u128::from(NEG0) << 32)));
}

/// NEON VMIN/VMAX under the standard FPSCR: any NaN lane, signalling or
/// quiet, in either position, gives the default NaN.
#[test]
fn arm32_neon_minmax_order_zeros_and_give_the_default_nan() {
    const DEFAULT_NAN: u32 = 0x7fc0_0000;
    let g = VexArch::ARM;
    assert_eq!(FloatMinMaxRule::for_guest(g), Some(FloatMinMaxRule::Arm32Neon));
    let nans = [DEFAULT_NAN; 2];
    assert_eq!(lanes(g, VMIN, L, R), [NEG0, NEG0, nans[0], nans[1]]);
    assert_eq!(lanes(g, VMAX, L, R), [POS0, POS0, nans[0], nans[1]]);
    assert_eq!(lanes(g, VMIN, [ONE; 4], [TWO; 4]), [ONE; 4]);
    assert_eq!(lanes(g, VMAX, [ONE; 4], [TWO; 4]), [TWO; 4]);
    let l = [QNAN, SNAN, 0xffc0_0001, SNAN];
    let r = [SNAN, QNAN, ONE, 0xff80_0002];
    assert_eq!(lanes(g, VMIN, l, r), [DEFAULT_NAN; 4]);
    assert_eq!(lanes(g, VMAX, l, r), [DEFAULT_NAN; 4]);
    // VPMIN.F32 over a (+0, -0) pair and a (QNaN, 1) pair.
    let left = u128::from(NEG0) << 32;
    let right = u128::from(QNAN) | (u128::from(ONE) << 32);
    let out = VEXOps::binop_for_guest(
        g,
        IROp::VFPwMin {
            elem: IRType::F32,
            count: 2,
        },
        RustBV::concrete(left, 64),
        RustBV::concrete(right, 64),
    )
    .unwrap();
    assert_eq!(out.as_u128(), Some(u128::from(NEG0) | (u128::from(DEFAULT_NAN) << 32)));
    assert!(!FloatMinMaxRule::Arm32Neon.covers(&IROp::VFMinS { elem: IRType::F32 }));
}

/// A guest without a rule, and an op its rule does not cover, keep
/// [`VEXOps::binop`]; [`VEXOps::float_minmax`] refuses an uncovered op.
#[test]
fn rule_coverage() {
    assert_eq!(FloatMinMaxRule::for_guest(VexArch::MIPS32), None);
    assert_eq!(lanes(VexArch::MIPS32, VMIN, L, R), R);
    let scalar = IROp::VFMinS { elem: IRType::F32 };
    assert!(FloatMinMaxRule::X86.covers(&scalar));
    assert!(!FloatMinMaxRule::Arm64.covers(&scalar));
    assert!(matches!(
        VEXOps::float_minmax(
            scalar,
            FloatMinMaxRule::Arm64,
            RustBV::concrete(0, 128),
            RustBV::concrete(0, 128)
        ),
        Err(OpError::NotBinary(_))
    ));
    assert!(!FloatMinMaxRule::X86.covers(&IROp::VFPwMin {
        elem: IRType::F32,
        count: 2
    }));
}

/// The symbolic lane builds without folding a concrete operand away; its
/// value is pinned through the solver by the Python seam's differential in
/// `tests/engines/vex/test_rust_irop.py`.
#[test]
fn arm64_minmax_symbolic_builds() {
    let x = RustBV::symbol("x", 128);
    let out = VEXOps::binop_for_guest(VexArch::ARM64, VMIN, x, RustBV::concrete(0, 128)).unwrap();
    assert_eq!(out.width(), 128);
    assert!(out.is_symbolic());
    let x = RustBV::symbol("x", 128);
    let out = VEXOps::binop_for_guest(VexArch::ARM, VMAX, x, RustBV::concrete(0, 128)).unwrap();
    assert!(out.is_symbolic());
}
