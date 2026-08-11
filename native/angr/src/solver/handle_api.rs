//! Handle-based solver API — the claripy bypass.
//!
//! These `#[pymethods]` let Python create, combine and evaluate bitvectors by
//! integer handle into the context's [`RustSymbolTable`], without ever
//! building a claripy AST. That avoids the per-operation AST traversal the
//! claripy-AST API in the parent `solver` module pays, and is the path the
//! interpreter's callback returns use.
//!
//! Split out of `solver.rs` (angr-9ke6b.205): the 43 `#[pymethods]` here (33
//! `op_*` arithmetic wrappers plus the symbol-table lifecycle) are one
//! self-contained concern, and the parent module keeps the claripy-AST API
//! plus the Python-boundary error mapping they share via
//! [`invalid_handle_id`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{RustSolverContext, invalid_handle_id};
use crate::claripy_bridge::claripy_to_rustbv;
#[cfg(doc)]
use crate::symbolic::MAX_BV_WIDTH;
use crate::symbolic::{
    BinaryOpError, RustBVHandle, RustSymbolTable, SymContext, check_bv_width, check_extract_bounds,
};

/// A handle-based binary op on the symbol table, as taken by
/// [`RustSolverContext::binop`].
type SymbolTableBinOp = fn(&RustSymbolTable, u64, u64, &SymContext) -> BinOpResult;

/// What every [`SymbolTableBinOp`] returns.
type BinOpResult = Result<RustBVHandle, BinaryOpError>;
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl RustSolverContext {
    /// Create a new symbolic bitvector and return a handle.
    ///
    /// This bypasses claripy.BVS() for native Rust symbolic value creation.
    /// Widths above [`MAX_BV_WIDTH`] are rejected (angr-c7xno.94).
    pub fn create_symbolic(&self, name: &str, width: u32) -> PyResult<RustBVHandle> {
        Self::check_width("create_symbolic", width)?;
        let ctx = self.i().ctx();
        Ok(self.i().symbol_table.create_symbolic(&ctx, name, width))
    }

    /// Create a new concrete bitvector and return a handle.
    ///
    /// This bypasses claripy.BVV() for native Rust concrete value creation.
    /// Widths above [`MAX_BV_WIDTH`] are rejected (angr-c7xno.94).
    pub fn create_concrete(&self, value: u128, width: u32) -> PyResult<RustBVHandle> {
        Self::check_width("create_concrete", width)?;
        Ok(self.i().symbol_table.create_concrete(value, width))
    }

    /// Evaluate a handle to get a concrete value.
    ///
    /// Returns None if unsatisfiable or the value cannot be evaluated.
    pub fn eval_handle(&self, handle_id: u64) -> Option<u128> {
        let bv = self.i().symbol_table.get(handle_id)?;
        self.i().ctx().eval(&bv)
    }

    /// Get the minimum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn min_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.i().symbol_table.get(handle_id)?;
        self.i().ctx().min(&bv, signed)
    }

    /// Get the maximum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn max_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.i().symbol_table.get(handle_id)?;
        self.i().ctx().max(&bv, signed)
    }

    /// Evaluate a handle and return up to n solutions.
    pub fn eval_upto_handle(&self, handle_id: u64, n: usize) -> Vec<u128> {
        if let Some(bv) = self.i().symbol_table.get(handle_id) {
            self.i().ctx().eval_upto(&bv, n)
        } else {
            Vec::new()
        }
    }

    /// Add a constraint from a handle (must be 1-bit).
    // Without Z3 there is no assert path, so the resolved `bv` is unread — the
    // handle lookup still runs for its `invalid_handle_id` error (angr-sqfj8.139).
    #[cfg_attr(
        not(feature = "vex-engine-z3"),
        allow(unused_variables, reason = "Z3-only consumer")
    )]
    pub fn add_constraint_handle(&self, handle_id: u64) -> PyResult<()> {
        let bv = self
            .i()
            .symbol_table
            .get(handle_id)
            .ok_or_else(|| invalid_handle_id(&[handle_id]))?;

        #[cfg(feature = "vex-engine-z3")]
        {
            let ctx = self.i().ctx();
            ctx.assume_true(&super::bool_constraint_bv(&bv, &ctx));
        }

        Ok(())
    }

    /// Check if a specific value is a valid solution for a handle.
    pub fn solution_handle(&self, handle_id: u64, value: u128) -> bool {
        if let Some(bv) = self.i().symbol_table.get(handle_id) {
            self.i().ctx().solution(&bv, value)
        } else {
            false
        }
    }

    /// Get the number of handles in the symbol table.
    pub fn handle_count(&self) -> usize {
        self.i().symbol_table.len()
    }

    /// Convert a claripy AST to a handle.
    ///
    /// This performs a one-time conversion of a claripy AST to a RustBV,
    /// storing it in the symbol table and returning a handle. Subsequent
    /// operations can use the handle directly, avoiding repeated AST traversal.
    ///
    /// This is the key optimization for the callback return path: instead of
    /// returning a claripy AST that must be traversed on every use, we convert
    /// once and return a handle for O(1) lookups.
    pub fn claripy_ast_to_handle(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        Ok(self.i().symbol_table.insert(bv))
    }

    // =========================================================================
    // Handle-based Arithmetic Operations
    // =========================================================================

    /// Add two handles and return a new handle.
    pub fn op_add(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_add)
    }

    /// Subtract two handles and return a new handle.
    pub fn op_sub(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_sub)
    }

    /// Multiply two handles and return a new handle.
    pub fn op_mul(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_mul)
    }

    /// Unsigned division of two handles.
    pub fn op_udiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_udiv)
    }

    /// Signed division of two handles.
    pub fn op_sdiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_sdiv)
    }

    /// Unsigned remainder of two handles.
    pub fn op_urem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_urem)
    }

    /// Signed remainder of two handles.
    pub fn op_srem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_srem)
    }

    /// Negation of a handle.
    pub fn op_neg(&self, a_id: u64) -> PyResult<RustBVHandle> {
        self.opt_op(&[a_id], |table, ctx| table.op_neg(a_id, ctx))
    }

    // =========================================================================
    // Handle-based Bitwise Operations
    // =========================================================================

    /// Bitwise AND of two handles.
    pub fn op_and(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_and)
    }

    /// Bitwise OR of two handles.
    pub fn op_or(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_or)
    }

    /// Bitwise XOR of two handles.
    pub fn op_xor(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_xor)
    }

    /// Bitwise NOT of a handle.
    pub fn op_not(&self, a_id: u64) -> PyResult<RustBVHandle> {
        self.opt_op(&[a_id], |table, ctx| table.op_not(a_id, ctx))
    }

    // =========================================================================
    // Handle-based Shift Operations
    // =========================================================================

    /// Left shift.
    pub fn op_shl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_shl)
    }

    /// Logical right shift.
    pub fn op_lshr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_lshr)
    }

    /// Arithmetic right shift.
    pub fn op_ashr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_ashr)
    }

    /// Rotate left.
    pub fn op_rotl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_rotl)
    }

    /// Rotate right.
    pub fn op_rotr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_rotr)
    }

    // =========================================================================
    // Handle-based Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit handle).
    pub fn op_eq(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_eq)
    }

    /// Inequality comparison (returns 1-bit handle).
    pub fn op_ne(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_ne)
    }

    /// Unsigned less than.
    pub fn op_ult(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_ult)
    }

    /// Unsigned less than or equal.
    pub fn op_ule(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_ule)
    }

    /// Unsigned greater than.
    pub fn op_ugt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_ugt)
    }

    /// Unsigned greater than or equal.
    pub fn op_uge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_uge)
    }

    /// Signed less than.
    pub fn op_slt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_slt)
    }

    /// Signed less than or equal.
    pub fn op_sle(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_sle)
    }

    /// Signed greater than.
    pub fn op_sgt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_sgt)
    }

    /// Signed greater than or equal.
    pub fn op_sge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.binop(a_id, b_id, RustSymbolTable::op_sge)
    }

    // =========================================================================
    // Handle-based Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn op_zero_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        Self::check_width("op_zero_extend", to_width)?;
        self.reject_extend_narrowing("op_zero_extend", a_id, to_width)?;
        self.opt_op(&[a_id], |table, ctx| {
            table.op_zero_extend(a_id, to_width, ctx)
        })
    }

    /// Sign-extend to a wider width.
    pub fn op_sign_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        Self::check_width("op_sign_extend", to_width)?;
        self.reject_extend_narrowing("op_sign_extend", a_id, to_width)?;
        self.opt_op(&[a_id], |table, ctx| {
            table.op_sign_extend(a_id, to_width, ctx)
        })
    }

    /// Truncate to a narrower width.
    pub fn op_truncate(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        self.reject_truncate_widening(a_id, to_width)?;
        self.opt_op(&[a_id], |table, ctx| table.op_truncate(a_id, to_width, ctx))
    }

    /// Extract bits \[high:low\] (inclusive).
    pub fn op_extract(&self, a_id: u64, high: u32, low: u32) -> PyResult<RustBVHandle> {
        if let Some(width) = self.source_width(a_id) {
            check_extract_bounds("op_extract", high, low, width).map_err(PyValueError::new_err)?;
        }
        self.opt_op(&[a_id], |table, ctx| table.op_extract(a_id, high, low, ctx))
    }

    /// Concatenate two values (a becomes high bits).
    pub fn op_concat(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.opt_op(&[a_id, b_id], |table, ctx| table.op_concat(a_id, b_id, ctx))
    }

    /// If-then-else: if cond then then_val else else_val.
    pub fn op_ite(&self, cond_id: u64, then_id: u64, else_id: u64) -> PyResult<RustBVHandle> {
        self.opt_op(&[cond_id, then_id, else_id], |table, ctx| {
            table.op_ite(cond_id, then_id, else_id, ctx)
        })
    }
}

impl RustSolverContext {
    /// Shared body for the ~25 handle-based binary-op `#[pymethods]` wrappers.
    ///
    /// Each of them (`op_add`, `op_ult`, `op_xor`, ...) is the same three
    /// steps -- take the Z3 context, dispatch to the identically-named
    /// [`RustSymbolTable`] method, convert `BinaryOpError` into a `PyErr` --
    /// so all of that lives here once and the wrappers do nothing but name
    /// their op. They stay individually hand-written because PyO3 rejects
    /// `macro_rules!` invocations in a `#[pymethods]` impl body ("macros
    /// cannot be used as items in `#[pymethods]` impl blocks"). PyO3's
    /// `multiple-pymethods` feature is enabled (angr-9ke6b.50), so a
    /// macro-generated *second* impl block is now possible — but the macro
    /// would still have to live outside any `#[pymethods]` body, so the
    /// hand-written wrappers stay until someone shows that pays for itself.
    ///
    /// Ops whose `RustSymbolTable` method returns `Option` rather than
    /// `Result` (`op_neg`, `op_not`, `op_concat`, the width-changing
    /// conversions, `op_ite`) don't fit this signature — their operand lists
    /// differ — and go through [`opt_op`](Self::opt_op) instead.
    pub(super) fn binop(
        &self,
        a_id: u64,
        b_id: u64,
        op: SymbolTableBinOp,
    ) -> PyResult<RustBVHandle> {
        let inner = self.i();
        let ctx = inner.ctx();
        op(&inner.symbol_table, a_id, b_id, &ctx).map_err(PyErr::from)
    }

    /// Shared body for the handle-based `#[pymethods]` wrappers whose
    /// [`RustSymbolTable`] method returns `Option` instead of `Result`
    /// (`op_neg`, `op_not`, the width-changing conversions, `op_concat`,
    /// `op_ite`).
    ///
    /// Those can't use [`binop`](Self::binop) — their operand lists are 1, 2
    /// or 3 handles wide and some carry extra width arguments — but the
    /// surrounding steps are identical every time: take the Z3 context, call
    /// the symbol-table method, and turn a `None` (any operand handle absent
    /// from the table) into the shared `invalid_handle_id` `PyErr`. So the
    /// caller supplies only the two varying parts: `ids`, the operand handles
    /// to name in that error, and `op`, a closure invoking the table method.
    ///
    /// Keep `ids` in sync with the handles `op` actually looks up — it is
    /// what the Python-side error message blames, and a `None` gives no clue
    /// which operand was missing.
    pub(super) fn opt_op<F>(&self, ids: &[u64], op: F) -> PyResult<RustBVHandle>
    where
        F: FnOnce(&RustSymbolTable, &SymContext) -> Option<RustBVHandle>,
    {
        let inner = self.i();
        let ctx = inner.ctx();
        op(&inner.symbol_table, &ctx).ok_or_else(|| invalid_handle_id(ids))
    }

    /// Width of the value behind `a_id`, or `None` if the handle is unknown.
    ///
    /// A `None` is deliberately *not* an error here: the op's own
    /// [`opt_op`](Self::opt_op) dispatch reports the missing handle through
    /// [`invalid_handle_id`], and blaming the width instead would hide that.
    pub(super) fn source_width(&self, a_id: u64) -> Option<u32> {
        self.i().symbol_table.with_value(a_id, |bv| bv.width())
    }

    /// Reject a width above [`MAX_BV_WIDTH`] at the Python boundary.
    ///
    /// Nothing below this point validates a width: `RustSymbolTable`'s
    /// constructors hand it straight to `RustBV::symbolic` / `RustBV::concrete`,
    /// and an absurd width only surfaces at first Z3 materialization as an
    /// attempt to allocate a multi-gigabit bitvector sort — a hang/OOM rather
    /// than a catchable error (angr-c7xno.94).
    pub(super) fn check_width(op: &str, width: u32) -> PyResult<()> {
        check_bv_width(op, width).map_err(PyValueError::new_err)
    }

    /// Reject a widening width passed to `op_truncate` at the Python boundary.
    ///
    /// The mirror of [`reject_extend_narrowing`](Self::reject_extend_narrowing):
    /// `to_width > source_width` sends `value_ops::truncate_into` into
    /// `Extract(to_width - 1, 0)` — an extract past the end of the source — and
    /// `to_width == 0` wraps that `to_width - 1` to `u32::MAX`. Either way the
    /// bogus range reaches `Z3_mk_extract` as an invalid-argument call, which
    /// aborts the process under `panic="abort"` (angr-c7xno.93). A missing
    /// handle is left to the op's own `invalid_handle_id` path.
    pub(super) fn reject_truncate_widening(&self, a_id: u64, to_width: u32) -> PyResult<()> {
        if let Some(width) = self.source_width(a_id)
            && (to_width > width || (to_width == 0 && width > 0))
        {
            return Err(PyValueError::new_err(format!(
                "op_truncate: to_width {to_width} is not in 1..={width} (source width); \
                 use op_zero_extend/op_sign_extend to widen"
            )));
        }
        Ok(())
    }

    /// Reject a narrowing width passed to an extend op at the Python boundary.
    ///
    /// `op_zero_extend`/`op_sign_extend` with `to_width < source_width` would
    /// otherwise return a handle wider than the caller asked for, which only
    /// surfaces much later as a Z3 sort error (abort under `panic=abort`) or a
    /// silent `eq_into`-False — far from the misuse site (angr-ph300.38). A
    /// missing handle is left to the op's own `invalid_handle_id` path.
    pub(super) fn reject_extend_narrowing(
        &self,
        op: &str,
        a_id: u64,
        to_width: u32,
    ) -> PyResult<()> {
        if let Some(width) = self.source_width(a_id)
            && to_width < width
        {
            return Err(PyValueError::new_err(format!(
                "{op}: to_width {to_width} < source width {width}; \
                 use op_truncate/op_extract to narrow"
            )));
        }
        Ok(())
    }
}

test_submod!("handle_api_tests.rs" => tests);
