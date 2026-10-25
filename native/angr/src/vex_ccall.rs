//! `angr.rustylib.vex_ccall`: the Rust VEX clean-call helpers behind the
//! Python engine's ccall dispatch (`angr/engines/vex/claripy/datalayer.py`).
//!
//! The helpers live in [`crate::vex::ccall`], which stays PyO3-free; this
//! module is only the crossing. Arguments arrive as claripy ASTs and are
//! imported with `RustBV::from_clarirs(Base::ast())`, an `Arc` clone with no
//! tree walk. A result goes back with `Base::from_ast`. Anything the Rust
//! helper does not serve (an unported name, a symbolic `cond`/`cc_op`, a
//! non-bitvector argument) returns `None`, and the caller runs `ccall.py`.
//!
//! [`SUPPORTED`] is the set of names [`ccall`] may serve. It leaves out the
//! names `handle_ccall_with_ctx` accepts but cannot answer from its arguments
//! alone: `x86g_use_seg_selector` reads the GDT/LDT from memory and the CR0
//! state global, which `ccall.py` takes from the state.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::claripy::ast::base::Base;
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ccall::handle_ccall_with_ctx;

/// The VEX helper names [`ccall`] serves, as `ccall.py` spells them.
pub(crate) const SUPPORTED: &[&str] = &[
    "amd64g_calculate_condition",
    "amd64g_calculate_rflags_all",
    "amd64g_calculate_rflags_c",
    "x86g_calculate_condition",
    "x86g_calculate_eflags_all",
    "x86g_calculate_eflags_c",
    "armg_calculate_condition",
    "armg_calculate_flags_nzcv",
    "armg_calculate_flag_n",
    "armg_calculate_flag_z",
    "armg_calculate_flag_c",
    "armg_calculate_flag_v",
    "arm64g_calculate_condition",
    "arm64g_calculate_flags_nzcv",
    "arm64g_calculate_flag_n",
    "arm64g_calculate_flag_z",
    "arm64g_calculate_flag_c",
    "arm64g_calculate_flag_v",
];

thread_local! {
    // The symbolic builders take a context for their node constructors; one
    // per thread, like the engine's own.
    static CTX: SymContext = SymContext::new();
}

/// `ccall(name, args, ret_bits)`: run the VEX helper `name` on the claripy
/// bitvectors `args` and return a `ret_bits`-wide claripy bitvector, or
/// `None` when Rust does not serve this call.
#[pyfunction]
fn ccall(
    py: Python<'_>,
    name: &str,
    args: &Bound<'_, PyTuple>,
    ret_bits: u32,
) -> PyResult<Option<Py<PyAny>>> {
    if !SUPPORTED.contains(&name) {
        // SILENT(cat-a): an unported helper; the caller runs ccall.py.
        return Ok(None);
    }
    let mut imported = Vec::with_capacity(args.len());
    for arg in args.iter() {
        let Ok(base) = arg.cast::<Base>() else {
            // SILENT(cat-a): not a claripy AST; ccall.py takes it as given.
            return Ok(None);
        };
        let Ok(bv) = RustBV::from_clarirs(base.get().ast()) else {
            // SILENT(cat-a): not a bitvector; ccall.py takes it as given.
            return Ok(None);
        };
        imported.push(bv);
    }
    let Some(result) = CTX.with(|ctx| handle_ccall_with_ctx(name, &imported, ret_bits, Some(ctx)))
    else {
        // SILENT(cat-a): Rust refuses (symbolic cond/cc_op, unported cc_op);
        // ccall.py serves the call, including the multivalued concretization.
        return Ok(None);
    };
    let term = result
        .to_clarirs()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let out = Base::from_ast(py, term)?;
    Ok(Some(out.into_any().unbind()))
}

pub(crate) fn vex_ccall(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ccall, m)?)?;
    m.add("SUPPORTED", SUPPORTED.to_vec())?;
    Ok(())
}
