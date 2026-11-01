//! `angr.rustylib.vex_irop`: the Rust integer VEX ops behind the Python
//! engine's op dispatch (`angr/engines/vex/claripy/datalayer.py`).
//!
//! The ops live in [`crate::vex::ops`], which stays PyO3-free; this module is
//! only the crossing. A claripy argument's term is `Base::ast()`, and the
//! result goes back with `Base::from_ast` after the same `simplify_ext` every
//! claripy bitvector operator applies. Anything Rust does not serve (a
//! non-bitvector or annotated literal argument, a width or arity the op does
//! not take, a literal zero divisor, a symbolic value shifted by a literal of
//! at least its width or lane width, a literal lane index past the lane
//! count) returns `None`, and the caller runs `irop.py`.
//!
//! `SUPPORTED` holds the libVEX names of the scalar integer arithmetic,
//! logic, shift, compare and width-conversion ops, and of the integer-lane
//! vector ops ([`is_int_lane_vec`]) and the vector float ops
//! ([`is_vector_float_op`]) that Rust implements, less the exclusions in
//! [`SERVED`].

use std::collections::HashMap;
use std::sync::LazyLock;

use clarirs_core::prelude::{AstOp, AstType};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::ailment::vexop_names::IROP_NAMES;
use crate::claripy::ast::base::Base;
use crate::symbolic::{RustBV, VexOpFamily, VexRm};
use crate::vex::ir::IROp;
use crate::vex::opcode_map::parse_opcode;
use crate::vex::ops::{VEXOps, iropclass, is_int_lane_vec, is_vector_float_op};

/// The libVEX names [`irop`] serves, with their parsed opcodes.
static SERVED: LazyLock<HashMap<&'static str, IROp>> = LazyLock::new(|| {
    IROP_NAMES
        .iter()
        .filter_map(|&name| {
            let op = parse_opcode(name);
            let serve = match op {
                // FP-sorted on the Python side.
                IROp::Reinterpret { .. } => false,
                // clarirs folds fpSub of a NaN as an add of its negation, flipping the NaN's sign (angr-mx00p.127).
                IROp::VFSub { .. } | IROp::VFSubS { .. } => false,
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
                _ if is_vector_float_op(&op) => true,
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
    // SymBV keeps a constant wider than 128 bits as a term, so concreteness is read off claripy's AST.
    let mut symbolic = Vec::with_capacity(args.len());
    for arg in args.iter() {
        let Ok(base) = arg.cast::<Base>() else {
            // SILENT(cat-a): not a claripy AST; irop.py unwraps or rejects it.
            return Ok(None);
        };
        symbolic.push(base.get().symbolic());
        let ast = base.get().ast();
        if !matches!(ast.ast_type(), AstType::BitVec(_)) {
            // SILENT(cat-a): not a bitvector; irop.py converts it.
            return Ok(None);
        }
        if matches!(ast.op(), AstOp::BVV(_)) && !ast.annotations().is_empty() {
            // SILENT(cat-a): `from_clarirs` folds a literal to a value, dropping its annotations.
            return Ok(None);
        }
        let Ok(value) = RustBV::from_clarirs(ast) else {
            // SILENT(cat-a): no SymBV form; irop.py serves the call.
            return Ok(None);
        };
        imported.push(value);
    }
    // Packed float arithmetic carries the VEX rounding mode first; Rust computes it round-to-nearest-even only.
    let takes_rm = match op {
        IROp::VFAdd { .. } | IROp::VFSub { .. } | IROp::VFMul { .. } | IROp::VFDiv { .. } => {
            imported.len() == 3
        }
        IROp::VFSqrt { .. } => imported.len() == 2,
        _ => false,
    };
    if takes_rm {
        let mode = imported[0].as_u128().map(VexRm::decode);
        if !matches!(mode, Some(Ok(VexRm::NearestEven))) {
            // SILENT(cat-a): another or a symbolic rounding mode; irop.py honours it.
            return Ok(None);
        }
        imported.remove(0);
        symbolic.remove(0);
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
            if shifts
                && l.is_symbolic()
                && r.as_u128()
                    .is_some_and(|amount| amount >= u128::from(l.width()))
            {
                // SILENT(cat-a): Rust folds it to the fill; irop.py keeps the shift.
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
            VEXOps::binop(op, l.clone(), r.clone())
        }
        // `Iop_SetElem*` is the only served Triop; `binop_with_rm` reads its first operand as the vector.
        [v, i, x] if matches!(op, IROp::VSetElem { .. }) => {
            if literal_index_past_count(op, i) {
                // SILENT(cat-a): libVEX leaves it undefined; irop.py leaves the vector as is.
                return Ok(None);
            }
            VEXOps::binop_with_rm(op, v.clone(), i.clone(), x.clone())
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
        .and_then(|ast| ast.simplify_ext(true, true))
    else {
        // SILENT(cat-a): irop.py serves the call.
        return Ok(None);
    };
    let out = Base::from_ast(py, ast)?;
    Ok(Some(out.into_any().unbind()))
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
