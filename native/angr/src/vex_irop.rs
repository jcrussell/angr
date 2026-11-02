//! `angr.rustylib.vex_irop`: the Rust VEX ops behind the Python engine's op
//! dispatch (`angr/engines/vex/claripy/datalayer.py`).
//!
//! The ops live in [`crate::vex::ops`], which stays PyO3-free; this module is
//! only the crossing. A claripy argument's term is `Base::ast()`, and the
//! result goes back with `Base::from_ast` after the same `simplify_ext` every
//! claripy bitvector operator applies, carrying a literal argument's
//! relocatable annotations as claripy's fold does. Anything Rust does not
//! serve (an argument of the wrong sort, a width or arity the op does not take, a
//! literal with an annotation claripy does not fold through, a
//! literal zero divisor, a symbolic value shifted by a literal or
//! multiplied by a literal power of two, a symbolic vector shifted by a
//! literal of at least its lane width, a literal lane index past the lane
//! count) returns `None`, and the caller runs `irop.py`.
//!
//! `SUPPORTED` holds the libVEX names of the scalar integer arithmetic,
//! logic, shift, compare and width-conversion ops, and of the integer-lane
//! vector ops ([`is_int_lane_vec`]) and the vector float ops
//! ([`is_vector_float_op`]) that Rust implements, less the exclusions in
//! [`SERVED`], and of the scalar float arithmetic, fused multiply-add, sign,
//! compare and float-to-float, float-to-int and round-to-integral conversion ops
//! ([`scalar_float_type`]) and the int-to-float conversions
//! ([`is_int_to_float`]).
//!
//! A scalar float op takes FP-sorted claripy ASTs, as `irop.py` does: an
//! operand enters as its IEEE bits ([`import_float`]) and a float result
//! leaves wrapped back in `bvToFP` (`Iop_CmpF*` returns its I32 condition code
//! as a bitvector, and a float-to-int conversion its integer). An
//! int-to-float conversion takes a bitvector and returns an FP-sorted result
//! the same way.

use std::sync::LazyLock;

use clarirs_core::prelude::{AstFactory, AstOp, AstRef, AstType, FSort};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rustc_hash::FxHashMap;

use crate::ailment::vexop_names::IROP_NAMES;
use crate::claripy::ast::GLOBAL_CONTEXT;
use crate::claripy::ast::base::Base;
use crate::symbolic::RustBV;
use crate::vex::ir::{IROp, IRType};
use crate::vex::opcode_map::parse_opcode;
use crate::vex::ops::{VEXOps, VexOpFamily, iropclass, is_int_lane_vec, is_vector_float_op};

/// The libVEX names [`irop`] serves, with their parsed opcodes.
static SERVED: LazyLock<FxHashMap<&'static str, IROp>> = LazyLock::new(|| {
    IROP_NAMES
        .iter()
        .filter_map(|&name| {
            let op = parse_opcode(name);
            let serve = match op {
                // FP-sorted on the Python side.
                IROp::Reinterpret { .. } => false,
                // clarirs folds fpSub of a NaN as an add of its negation, flipping the NaN's sign (angr-mx00p.127).
                IROp::FSub(_) | IROp::VFSub { .. } | IROp::VFSubS { .. } => false,
                // x86 MINPS/MAXPS and AArch64 FMIN/FMAX share these ops and differ on +-0 and NaN; irop.py follows
                // neither (angr-mx00p.128).
                IROp::VFMin { .. }
                | IROp::VFMax { .. }
                | IROp::VFMinS { .. }
                | IROp::VFMaxS { .. }
                | IROp::VFPwMin { .. }
                | IROp::VFPwMax { .. } => false,
                // libVEX leaves an out-of-range, infinite or NaN lane undefined; Rust saturates, irop.py's fold
                // gives 0 (angr-mx00p.129).
                IROp::VF32toI32 { .. } => false,
                _ if is_vector_float_op(&op)
                    || scalar_float_type(op).is_some()
                    || is_int_to_float(op) =>
                {
                    true
                }
                _ => match iropclass(&op) {
                    VexOpFamily::Arith
                    | VexOpFamily::Logic
                    | VexOpFamily::Shift
                    | VexOpFamily::Cmp
                    | VexOpFamily::Ext => true,
                    VexOpFamily::Vec => is_int_lane_vec(&op),
                    VexOpFamily::Fp | VexOpFamily::Other => false,
                },
            };
            serve.then_some((name, op))
        })
        .collect()
});

/// `irop(name, args)`: run the VEX op `name` on the claripy bitvectors
/// `args`, or return `None` when Rust does not serve this call.
#[pyfunction]
fn irop(py: Python<'_>, name: &str, args: &Bound<'_, PyTuple>) -> PyResult<Option<Py<PyAny>>> {
    let Some(&op) = SERVED.get(name) else {
        // SILENT(cat-a): not a served op; the caller runs irop.py.
        return Ok(None);
    };
    let mut imported = Vec::with_capacity(args.len());
    let mut carried = Vec::new();
    // SymBV keeps a constant wider than 128 bits as a term, so concreteness is read off claripy's AST.
    let mut symbolic = Vec::with_capacity(args.len());
    let float = scalar_float_type(op);
    // The rounding mode leads a rounded float op's operands; `Iop_{Abs,Neg,Cmp}F*` and `Iop_F32toF64` take none.
    let first_float = usize::from(!matches!(
        op,
        IROp::FAbs(_) | IROp::FNeg(_) | IROp::FComCC(_) | IROp::F32toF64
    ));
    for (pos, arg) in args.iter().enumerate() {
        let Ok(base) = arg.cast::<Base>() else {
            // SILENT(cat-a): not a claripy AST; irop.py unwraps or rejects it.
            return Ok(None);
        };
        symbolic.push(base.get().symbolic());
        let mut ast = base.get().ast();
        // A scalar float op's operands after any rounding mode are floats; irop.py refuses a bitvector there.
        if let Some(ty) = float.filter(|_| pos >= first_float) {
            let Some(bits) = import_float(&ast, ty) else {
                // SILENT(cat-a): not a float of the op's width, or one with annotations; irop.py serves or rejects it.
                return Ok(None);
            };
            ast = bits;
        } else if !matches!(ast.ast_type(), AstType::BitVec(_)) {
            // SILENT(cat-a): not a bitvector; irop.py converts it.
            return Ok(None);
        }
        if matches!(ast.op(), AstOp::BVV(_)) && !ast.annotations().is_empty() {
            if ast.annotations().iter().any(|a| !a.relocatable()) {
                // SILENT(cat-a): claripy does not fold through it; irop.py serves the call.
                return Ok(None);
            }
            // `from_clarirs` folds the literal to a value; claripy carries its annotations onto the result.
            carried.extend(ast.annotations().iter().cloned());
        }
        let Ok(value) = RustBV::from_clarirs(ast) else {
            // SILENT(cat-a): no SymBV form; irop.py serves the call.
            return Ok(None);
        };
        imported.push(value);
    }
    let result = match imported.as_slice() {
        [a] => VEXOps::unop(op, a.clone()),
        [l, r] => {
            let divides = matches!(
                op,
                IROp::DivU(_)
                    | IROp::DivS(_)
                    | IROp::DivModU64to32
                    | IROp::DivModS64to32
                    | IROp::DivModU128to64
                    | IROp::DivModS128to64
                    | IROp::DivModSameWidth { .. }
            );
            if divides && r.as_u128() == Some(0) {
                // SILENT(cat-a): irop.py raises SimZeroDivisionException (a remainder it serves).
                return Ok(None);
            }
            let shifts = matches!(op, IROp::Shl(_) | IROp::Shr(_) | IROp::Sar(_));
            if shifts && l.is_symbolic() && r.as_u128().is_some() {
                // SILENT(cat-a): Rust rewrites it to Concat/Extract (or the fill); irop.py keeps the shift, whose
                // shape the VSA domain reads more precisely (angr-mx00p.100.12).
                return Ok(None);
            }
            let muls = matches!(op, IROp::Mul(_) | IROp::MullS(_) | IROp::MullU(_));
            let pow2 = |v: &RustBV| v.as_u128().is_some_and(u128::is_power_of_two);
            if muls && ((l.is_symbolic() && pow2(r)) || (r.is_symbolic() && pow2(l))) {
                // SILENT(cat-a): Rust rewrites it to a shift and then Concat/Extract; irop.py keeps the Mul
                // (angr-mx00p.100.12).
                return Ok(None);
            }
            if symbolic[0] && literal_shift_past_lane(op, r) {
                // SILENT(cat-a): Rust folds the lane to zero; irop.py keeps the shift.
                return Ok(None);
            }
            if literal_index_past_count(op, r) {
                // SILENT(cat-a): libVEX leaves it undefined; irop.py reads lane 0.
                return Ok(None);
            }
            if literal_float_past_int_range(op, r) {
                // SILENT(cat-a): libVEX leaves it undefined; irop.py's reading of it stands (angr-mx00p.129).
                return Ok(None);
            }
            VEXOps::binop(op, l.clone(), r.clone())
        }
        // `Iop_SetElem*` (`binop_with_rm` reads its first operand as the vector) and scalar and packed float
        // arithmetic under the VEX rounding mode are the served Triops. `Iop_Sqrt*` is a Binop `binop` serves.
        [v, i, x] if matches!(op, IROp::VSetElem { .. }) => {
            if literal_index_past_count(op, i) {
                // SILENT(cat-a): libVEX leaves it undefined; irop.py leaves the vector as is.
                return Ok(None);
            }
            VEXOps::binop_with_rm(op, v.clone(), i.clone(), x.clone())
        }
        [rm, l, r]
            if matches!(
                op,
                IROp::VFAdd { .. }
                    | IROp::VFSub { .. }
                    | IROp::VFMul { .. }
                    | IROp::VFDiv { .. }
                    | IROp::FAdd(_)
                    | IROp::FMul(_)
                    | IROp::FDiv(_)
            ) =>
        {
            VEXOps::binop_with_rm(op, rm.clone(), l.clone(), r.clone())
        }
        // `Iop_M{Add,Sub}F*` round `a * b +- c` once under the VEX rounding mode.
        [rm, a, b, c] if matches!(op, IROp::FMAdd(_) | IROp::FMSub(_)) => {
            VEXOps::qop_with_rm(op, rm.clone(), a.clone(), b.clone(), c.clone())
        }
        // SILENT(cat-a): an arity the op does not take; irop.py rejects it.
        _ => return Ok(None),
    };
    let Ok(result) = result else {
        // SILENT(cat-a): a width or arity irop.py extends or rejects itself.
        return Ok(None);
    };
    let Ok(ast) = result
        .to_clarirs()
        .and_then(|ast| match op.result_type() {
            Some(IRType::F32) => GLOBAL_CONTEXT.bv_to_fp(ast, FSort::f32()),
            Some(IRType::F64) => GLOBAL_CONTEXT.bv_to_fp(ast, FSort::f64()),
            _ => Ok(ast),
        })
        .and_then(|ast| ast.simplify_ext(true, true))
        .and_then(|ast| {
            if carried.is_empty() {
                Ok(ast)
            } else {
                ast.annotate(carried)
            }
        })
    else {
        // SILENT(cat-a): irop.py serves the call.
        return Ok(None);
    };
    if matches!(op, IROp::FComCC(_)) && symbolic.contains(&true) && !ast.symbolic() {
        // SILENT(cat-a): Rust folds a NaN literal against a symbol to unordered; irop.py keeps the ite chain.
        return Ok(None);
    }
    let out = Base::from_ast(py, ast)?;
    Ok(Some(out.into_any().unbind()))
}

/// The operand float type of a served scalar float op
/// (`Iop_{Add,Mul,Div,Sqrt,Abs,Neg,Cmp,MAdd,MSub}F{32,64}`, `Iop_F32toF64`,
/// `Iop_F64toF32`, `Iop_F{32,64}toI{32,64}{S,U}`, `Iop_F64toI16S`, `Iop_RoundF{32,64}toInt`),
/// whose float operands `irop.py` takes FP-sorted.
fn scalar_float_type(op: IROp) -> Option<IRType> {
    if let Some((ty, _, _)) = float_to_int(op) {
        return Some(ty);
    }
    match op {
        IROp::F32toF64 | IROp::RoundF32toInt => Some(IRType::F32),
        IROp::F64toF32 | IROp::RoundF64toInt => Some(IRType::F64),
        IROp::FAdd(ty)
        | IROp::FMul(ty)
        | IROp::FDiv(ty)
        | IROp::FSqrt(ty)
        | IROp::FAbs(ty)
        | IROp::FNeg(ty)
        | IROp::FComCC(ty)
        | IROp::FMAdd(ty)
        | IROp::FMSub(ty)
            if matches!(ty, IRType::F32 | IRType::F64) =>
        {
            Some(ty)
        }
        _ => None,
    }
}

/// The operand float type, target width and signedness of a scalar
/// float-to-int conversion, `Iop_F{32,64}toI{32,64}{S,U}` and `Iop_F64toI16S`.
fn float_to_int(op: IROp) -> Option<(IRType, u32, bool)> {
    Some(match op {
        IROp::F32toI32S => (IRType::F32, 32, true),
        IROp::F64toI32S => (IRType::F64, 32, true),
        IROp::F64toI16S => (IRType::F64, 16, true),
        IROp::F32toI64S => (IRType::F32, 64, true),
        IROp::F64toI64S => (IRType::F64, 64, true),
        IROp::F32toI32U => (IRType::F32, 32, false),
        IROp::F64toI32U => (IRType::F64, 32, false),
        IROp::F32toI64U => (IRType::F32, 64, false),
        IROp::F64toI64U => (IRType::F64, 64, false),
        _ => return None,
    })
}

/// Whether `op` is a float-to-int conversion of a literal that some rounding
/// mode takes outside the target range, or of an infinity or NaN: libVEX
/// leaves the result undefined there. Checking the floor and the ceiling
/// covers every mode, a symbolic one included.
fn literal_float_past_int_range(op: IROp, value: &RustBV) -> bool {
    let Some((ty, bits, signed)) = float_to_int(op) else {
        return false;
    };
    let Some(raw) = value.as_u128() else {
        return false;
    };
    // An F32 widens to F64 exactly, and the range bounds are powers of two F64 holds exactly.
    let f = match ty {
        IRType::F32 => f64::from(f32::from_bits(raw as u32)),
        _ => f64::from_bits(raw as u64),
    };
    let (lo, end) = if signed {
        (-(2f64.powi(bits as i32 - 1)), 2f64.powi(bits as i32 - 1))
    } else {
        (0.0, 2f64.powi(bits as i32))
    };
    !f.is_finite() || f.floor() < lo || f.ceil() >= end
}

/// Whether `op` is a scalar int-to-float conversion,
/// `Iop_I{32,64}{S,U}toF{32,64}`: its operand is a bitvector and its result
/// FP-sorted.
fn is_int_to_float(op: IROp) -> bool {
    matches!(
        op,
        IROp::I32StoF32
            | IROp::I32StoF64
            | IROp::I64StoF32
            | IROp::I64StoF64
            | IROp::I32UtoF32
            | IROp::I32UtoF64
            | IROp::I64UtoF32
            | IROp::I64UtoF64
    )
}

/// A float operand as its IEEE bits: `bvToFP(x)` (claripy's `raw_to_fp`)
/// peels to `x` and an `FPV` becomes its bit literal, so a concrete float
/// stays concrete and a NaN keeps its payload; any other float term is
/// wrapped in `fpToIEEEBV` by `RustBV::from_clarirs`. `None` for a
/// non-float, a float other than `ty` (`irop.py` computes in the operand's
/// sort), or a float carrying annotations.
fn import_float(ast: &AstRef<'static>, ty: IRType) -> Option<AstRef<'static>> {
    let AstType::Float(sort) = ast.ast_type() else {
        return None;
    };
    if sort.size() != ty.bits() || !ast.annotations().is_empty() {
        return None;
    }
    match ast.op() {
        AstOp::BvToFp(inner, _) => Some(inner.clone()),
        AstOp::FPV(value) => GLOBAL_CONTEXT.bvv(value.to_ieee_bits()).ok(),
        _ => Some(ast.clone()),
    }
}

/// Whether `op` is a left or logical-right vector shift whose literal amount
/// reaches some lane's width (libVEX leaves that undefined). A literal too
/// wide for `u128` counts as reaching it.
fn literal_shift_past_lane(op: IROp, amount: &RustBV) -> bool {
    let (elem, lanewise) = match op {
        IROp::VShlN { elem, .. } | IROp::VShrN { elem, .. } => (elem, false),
        IROp::VShl { elem, .. } | IROp::VShr { elem, .. } => (elem, true),
        _ => return false,
    };
    if amount.is_symbolic() {
        return false;
    }
    let Some(value) = amount.as_u128() else {
        return true;
    };
    let bits = elem.bits();
    if !lanewise {
        return value >= u128::from(bits);
    }
    let mask = u128::MAX >> (128 - bits);
    (0..amount.width() / bits).any(|lane| (value >> (lane * bits)) & mask >= u128::from(bits))
}

/// Whether `op` is a lane get or set whose literal index is past its lane count.
fn literal_index_past_count(op: IROp, index: &RustBV) -> bool {
    match op {
        IROp::VGetElem { count, .. } | IROp::VSetElem { count, .. } => {
            index.as_u128().is_some_and(|i| i >= u128::from(count))
        }
        _ => false,
    }
}

pub(crate) fn vex_irop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(irop, m)?)?;
    let mut names: Vec<&str> = SERVED.keys().copied().collect();
    names.sort_unstable();
    m.add("SUPPORTED", names)?;
    Ok(())
}
