//! `angr.rustylib.vex_irop`: the Rust integer VEX ops behind the Python
//! engine's op dispatch (`angr/engines/vex/claripy/datalayer.py`).
//!
//! The ops live in [`crate::vex::ops`], which stays PyO3-free; this module is
//! only the crossing. A claripy argument's term is `Base::ast()`, and the
//! result goes back with `Base::from_ast` after the same `simplify_ext` every
//! claripy bitvector operator applies. Anything Rust does not serve (a
//! non-bitvector or annotated literal argument, a width or arity the op does
//! not take, a literal zero divisor, a symbolic value shifted by a literal of
//! at least its width) returns `None`, and the caller runs `irop.py`.
//!
//! `SUPPORTED` holds the libVEX names of the scalar integer arithmetic,
//! logic, shift, compare and width-conversion ops Rust implements, less the
//! exclusions in [`SERVED`].

use std::collections::HashMap;
use std::sync::LazyLock;

use clarirs_core::prelude::{AstOp, AstType};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::ailment::vexop_names::IROP_NAMES;
use crate::claripy::ast::base::Base;
use crate::symbolic::{RustBV, VexOpFamily};
use crate::vex::ir::{IROp, IRType};
use crate::vex::opcode_map::parse_opcode;
use crate::vex::ops::{VEXOps, iropclass};

/// The libVEX names [`irop`] serves, with their parsed opcodes.
static SERVED: LazyLock<HashMap<&'static str, IROp>> = LazyLock::new(|| {
    IROP_NAMES
        .iter()
        .filter_map(|&name| {
            let op = parse_opcode(name);
            let serve = match op {
                // irop.py's results pass through clarirs' shift folds, which read the amount with
                // `to_u64().unwrap_or(0) as u32` (angr-mx00p.105).
                IROp::Shl(IRType::I64) | IROp::Shr(IRType::I64) | IROp::Sar(IRType::I64) => false,
                // irop.py computes it as an unsigned remainder (angr-mx00p.124).
                IROp::ModS(IRType::I128) => false,
                // FP-sorted on the Python side.
                IROp::Reinterpret { .. } => false,
                _ => matches!(
                    iropclass(&op),
                    VexOpFamily::Arith
                        | VexOpFamily::Logic
                        | VexOpFamily::Shift
                        | VexOpFamily::Cmp
                        | VexOpFamily::Ext
                ),
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
    for arg in args.iter() {
        let Ok(base) = arg.cast::<Base>() else {
            // SILENT(cat-a): not a claripy AST; irop.py unwraps or rejects it.
            return Ok(None);
        };
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
            VEXOps::binop(op, l.clone(), r.clone())
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

pub(crate) fn vex_irop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(irop, m)?)?;
    let mut names: Vec<&str> = SERVED.keys().copied().collect();
    names.sort_unstable();
    m.add("SUPPORTED", names)?;
    Ok(())
}
