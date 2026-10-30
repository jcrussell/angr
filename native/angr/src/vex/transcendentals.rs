//! Concrete-only fast paths for the x87 FPU transcendentals and ARM AArch64
//! FRECPX. The per-op coverage matrix is the "x87 transcendental ops" table
//! in `docs/extending-angr/rust_vex_ops.rst`; the `IOP_*` consts below are
//! the complete list. We expose no named `IROp` variant for these, so
//! `opcode_map_float::parse_transcendental` maps the pyvex name to
//! `IROp::Raw(<one of the IOP_* consts below>)`, and the `IROp::Raw` arms
//! of `VEXOps::binop` (in its private `binop_misc` helper) and
//! `VEXOps::binop_with_rm` dispatch that straight here.
//!
//! When all value operands are concrete f32/f64, this module computes the
//! result via Rust's libm bindings. Z3 has no native sin/cos/log/exp theory,
//! so a symbolic operand is either pinned first (below) or the op declines.
//!
//! ## Symbolic concretization fallback
//!
//! [`pins_binop`] / [`pins_triop`] name the x87 transcendentals whose
//! symbolic operands the interpreter concretizes before dispatch
//! (`interpreter::transcendental_pins`, angr-mx00p.116.6): the solver is
//! asked for one model of the input(s), the input(s) are pinned to the
//! sampled value(s) with hard constraints (so the path stays consistent),
//! and the op then runs here on the concrete value(s). `VEXOps` itself has
//! no symbolic path for these ops: a symbolic operand reaching it is
//! `OpError::RawOpcode`.
//!
//! Coverage (angr-i5lj.1, angr-i5lj.2):
//! * binop: `Iop_SinF64`, `Iop_CosF64`, `Iop_TanF64`, `Iop_2xm1F64`
//! * triop: `Iop_AtanF64`, `Iop_Yl2xF64`, `Iop_Yl2xp1F64`, `Iop_ScaleF64`
//!
//! Strategy rationale (option 1 in the bd description): cheap, no new
//! Z3 theory; loses symbolic precision — the path now sees ONE specific
//! value for the input. Alternative options not adopted:
//!
//! * Option 2 — Python callback to claripy: claripy itself has no
//!   log/exp theory, so this would just trampoline back to the same
//!   concretization with extra FFI cost.
//! * Option 3 — Pure-symbolic via a Z3 UF with monotonicity / range
//!   axioms: expensive, brittle, and no current benchmark routes a
//!   symbolic path through these ops (sokohashv2 hooks them out at the
//!   test driver). Revisit when a new benchmark or test actually invokes
//!   one of these ops with a symbolic operand (angr-9l1y).
//!
//! `Iop_RecpExpF64`/`Iop_RecpExpF32` (ARM AArch64 FRECPX) are NOT in
//! this fallback — they have a closed-form exponent-only implementation
//! and never need libm. Iop_PRem*F64 (FP remainder) is also out of
//! scope.
//!
//! ## Rounding mode
//!
//! Every entry point here takes the VEX rounding mode as its first operand
//! and discards it. The rounding-mode-threading gate
//! (`tools/audit_rounding_mode_threading.py`) requires
//! that discard to carry an `rm-ignored: <why>` comment, and this module is
//! the case that convention was written for: the results come from Rust's
//! libm bindings (`f64::sin`, `f64::log2`, `f64::atan2`, `f64::exp2`), which
//! expose round-to-nearest-even only — there is no rounding-mode-parameterized
//! form to thread `rm` into. The x87 default of rm=0 (RNE) is the common case,
//! so the discard is faithful there and off by at most one ulp otherwise.
//! `try_concrete_binop_rm` and `try_concrete_triop_rm` carry the marker.
//!
//! Opcode values: libvex_ir.h Iop_* enum, base 0x1400, validated against
//! pyvex.const.enums_to_ints. The only producer is the *string* router
//! `parse_transcendental` (angr-9ke6b.233), so these are
//! internal tags — they do not have to track libVEX renumbering, they
//! only have to stay distinct and agree with that router.

use crate::symbolic::RustBV;

pub const IOP_ATAN_F64: u32 = 0x14de;
pub const IOP_YL2X_F64: u32 = 0x14df;
pub const IOP_YL2XP1_F64: u32 = 0x14e0;
pub const IOP_SCALE_F64: u32 = 0x14e5;
pub const IOP_SIN_F64: u32 = 0x14e6;
pub const IOP_COS_F64: u32 = 0x14e7;
pub const IOP_TAN_F64: u32 = 0x14e8;
pub const IOP_2XM1_F64: u32 = 0x14e9;
pub const IOP_RECPEXP_F64: u32 = 0x14fa;
pub const IOP_RECPEXP_F32: u32 = 0x14fb;

/// Concrete Binop transcendental: (rm, x) → result.
/// Returns None if any value operand is symbolic or the opcode is unknown.
/// `rm` (rounding mode) is consumed but ignored — Rust libm always uses
/// round-to-nearest-even, which matches the typical x87 rm=0 case.
// See the module doc's "Rounding mode" section: sin/cos/tan/exp2 are
// libm-backed and Rust exposes no rounding-mode-parameterized form of them,
// while `recip_exp_f32`/`recip_exp_f64` are exact exponent arithmetic that
// never rounds.
// rm-ignored: nothing to thread the VEX rounding mode into.
pub fn try_concrete_binop_rm(opcode: u32, _rm: &RustBV, x: &RustBV) -> Option<RustBV> {
    if opcode == IOP_RECPEXP_F32 {
        let xv = x.as_u128()? as u32;
        let r = recip_exp_f32(f32::from_bits(xv));
        return Some(RustBV::concrete(r.to_bits() as u128, 32));
    }

    let xv = x.as_u128()? as u64;
    let xf = f64::from_bits(xv);
    let r = match opcode {
        IOP_SIN_F64 => xf.sin(),
        IOP_COS_F64 => xf.cos(),
        IOP_TAN_F64 => xf.tan(),
        IOP_2XM1_F64 => xf.exp2() - 1.0,
        IOP_RECPEXP_F64 => recip_exp_f64(xf),
        _ => return None,
    };
    Some(RustBV::concrete(r.to_bits() as u128, 64))
}

/// Concrete Triop transcendental: (rm, a, b) → result.
/// Returns None if any value operand is symbolic or the opcode is unknown.
/// `rm` is consumed but ignored, for the same reason as
/// `try_concrete_binop_rm`.
// See the module doc's "Rounding mode" section: log2/atan2/exp2 are
// libm-backed and Rust exposes no rounding-mode-parameterized form of them.
// rm-ignored: nothing to thread the VEX rounding mode into.
pub fn try_concrete_triop_rm(opcode: u32, _rm: &RustBV, a: &RustBV, b: &RustBV) -> Option<RustBV> {
    let av = a.as_u128()? as u64;
    let bv = b.as_u128()? as u64;
    let af = f64::from_bits(av);
    let bf = f64::from_bits(bv);
    let r = match opcode {
        // Iop_Yl2xF64(rm, y, x) = y * log2(x)
        IOP_YL2X_F64 => af * bf.log2(),
        // Iop_Yl2xp1F64(rm, y, x) = y * log2(x + 1)
        IOP_YL2XP1_F64 => af * (bf + 1.0).log2(),
        // Iop_ScaleF64(rm, x, y) = x * 2^trunc(y) (x87 fscale semantics)
        IOP_SCALE_F64 => af * bf.trunc().exp2(),
        // Iop_AtanF64(rm, y, x) = atan2(y, x) (x87 fpatan semantics)
        IOP_ATAN_F64 => af.atan2(bf),
        _ => return None,
    };
    Some(RustBV::concrete(r.to_bits() as u128, 64))
}

/// Whether a symbolic operand of binop `opcode` is concretized and pinned by
/// the interpreter (`interpreter::transcendental_pins`): `Iop_SinF64`,
/// `Iop_CosF64`, `Iop_TanF64`, `Iop_2xm1F64`.
pub fn pins_binop(opcode: u32) -> bool {
    matches!(
        opcode,
        IOP_SIN_F64 | IOP_COS_F64 | IOP_TAN_F64 | IOP_2XM1_F64
    )
}

/// Whether symbolic operands of triop `opcode` are concretized and pinned by
/// the interpreter (`interpreter::transcendental_pins`): `Iop_Yl2xF64`,
/// `Iop_Yl2xp1F64`, `Iop_ScaleF64`, `Iop_AtanF64`.
pub fn pins_triop(opcode: u32) -> bool {
    matches!(
        opcode,
        IOP_YL2X_F64 | IOP_YL2XP1_F64 | IOP_SCALE_F64 | IOP_ATAN_F64
    )
}

/// IEEE 754 reciprocal exponent (ARM AArch64 FRECPX semantics).
/// Returns 2^(-Exp(x)) preserving sign; special-cases NaN/0/Inf per the
/// architecture reference.
fn recip_exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    if x.is_infinite() {
        return 0.0_f64.copysign(x);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let unbiased = if biased == 0 {
        // Subnormal — treat as smallest normal exponent for the estimate.
        -1022
    } else {
        biased - 1023
    };
    let new_biased = ((1023 - unbiased).clamp(0, 0x7fe)) as u64;
    let sign = bits & 0x8000_0000_0000_0000;
    f64::from_bits(sign | (new_biased << 52))
}

fn recip_exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::INFINITY.copysign(x);
    }
    if x.is_infinite() {
        return 0.0_f32.copysign(x);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    let unbiased = if biased == 0 { -126 } else { biased - 127 };
    let new_biased = ((127 - unbiased).clamp(0, 0xfe)) as u32;
    let sign = bits & 0x8000_0000;
    f32::from_bits(sign | (new_biased << 23))
}

test_submod!("transcendentals_tests.rs" => tests);
