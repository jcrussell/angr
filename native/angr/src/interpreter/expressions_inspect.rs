//! Read-side `state.inspect` breakpoint dispatchers.
//!
//! One per event an expression evaluation can raise — `mem_read`, `reg_read`,
//! `tmp_read`, `expr`, `address_concretization`, `symbolic_variable` — each
//! gated on the `inspect_event_enabled` bitmask (via the shared `inspect_ast`
//! prelude) so the no-breakpoint case costs a single test per read. The
//! value-injection contract (a BP that overrides `state.inspect.*_expr`
//! returns `Some(bv)` for the caller to substitute) is documented per
//! function. The write-side counterparts live in `statements_inspect.rs`.

use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Shared prelude for the `dispatch_*_inspect` methods: gate on
    /// `event`, import claripy, and convert `value` into a claripy AST.
    /// Returns `None` when the breakpoint is disabled or the import/convert
    /// fails (the callers all swallow those failures — a missing claripy or a
    /// conversion error must not halt exploration). Centralizes the
    /// claripy-import-failure swallow policy that was previously open-coded at
    /// every dispatch site.
    pub(super) fn inspect_ast(
        &self,
        callbacks: &PythonCallbacks,
        event: InspectBit,
        value: &RustBV,
    ) -> Option<Py<PyAny>> {
        if !callbacks.inspect_event_enabled(event) {
            return None;
        }
        Python::attach(|py| {
            let claripy_mod = py.import("claripy").ok()?;
            crate::claripy_bridge::rustbv_to_claripy(py, value, &claripy_mod).ok()
        })
    }

    /// Fire a `mem_read` inspect callback into Python for this load.
    ///
    /// Mirrors `dispatch_mem_write_inspect` in `statements_inspect.rs`. Gated on
    /// `inspect_event_enabled(InspectBit::MemRead)` so the no-breakpoint case costs a
    /// single bitmask test per Load. Symbolic addresses are skipped for
    /// the MVP — only concrete addresses dispatch. The `when='after'`
    /// event is fired once the value has been computed; the BP receives
    /// the loaded value AST as `mem_read_expr`. Errors from the Python
    /// callback are swallowed and logged on the Python side; a user BP
    /// error must not halt exploration.
    ///
    /// Returns `Some(bv)` when the user's BP action overrode
    /// `state.inspect.mem_read_expr` (value injection — angr-uy32); the
    /// caller substitutes it for the loaded value. Returns `None` when the
    /// value is unchanged, so the original load result stands.
    pub(super) fn dispatch_mem_read_inspect(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        value: &RustBV,
        size: usize,
        endness: Endness,
    ) -> Option<RustBV> {
        let value_ast = self.inspect_ast(callbacks, InspectBit::MemRead, value)?;
        let addr_u64 = addr_val.as_u64()?;
        let endness_str = match endness {
            Endness::Little => "Iend_LE",
            Endness::Big => "Iend_BE",
        };
        let mutated = callbacks
            .call_inspect_mem_read(
                self.current_state_id,
                "after",
                addr_u64,
                size as u32,
                Some(&value_ast),
                endness_str,
            )
            .ok()??;
        // The user injected a new value via state.inspect.mem_read_expr.
        // Convert it back to a RustBV; reject a width mismatch defensively
        // so a bad override can't silently corrupt downstream ops.
        let bv = Python::attach(|py| {
            let bound = mutated.bind(py);
            crate::claripy_bridge::claripy_to_rustbv(py, bound, self.ctx).ok()
        })?;
        if bv.width() == (size * 8) as u32 {
            Some(bv)
        } else {
            None
        }
    }

    /// Fire a `reg_read` inspect callback into Python for a VEX `Get`.
    ///
    /// Gated on `inspect_event_enabled(InspectBit::RegRead)` so the no-breakpoint case
    /// is one bitmask test per `IRExpr::Get`. Dispatches `when='after'`
    /// with the loaded register value as `reg_read_expr`. Errors from the
    /// Python callback are swallowed and logged on the Python side.
    pub(super) fn dispatch_reg_read_inspect(
        &self,
        callbacks: &PythonCallbacks,
        offset: u32,
        size: u32,
        value: &RustBV,
    ) {
        let Some(value_ast) = self.inspect_ast(callbacks, InspectBit::RegRead, value) else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_reg_read(
                self.current_state_id,
                "after",
                offset,
                size,
                Some(&value_ast),
            ),
            "reg_read",
        );
    }

    /// Fire a `tmp_read` inspect callback for a VEX `RdTmp` (angr-64pi).
    ///
    /// Gated on `inspect_event_enabled(InspectBit::TmpRead)` so the no-breakpoint case is
    /// one bitmask test per `RdTmp` evaluation. Dispatches `when='after'`
    /// with the tmp's stored value as `tmp_read_expr`. RdTmp can fire many
    /// times per IRSB (every binop/load/store args go through it); the
    /// claripy AST round-trip is therefore only done when a BP is set.
    pub(super) fn dispatch_tmp_read_inspect(
        &self,
        callbacks: &PythonCallbacks,
        tmp_num: u32,
        value: &RustBV,
    ) {
        let Some(value_ast) = self.inspect_ast(callbacks, InspectBit::TmpRead, value) else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_tmp_read(
                self.current_state_id,
                "after",
                tmp_num,
                Some(&value_ast),
            ),
            "tmp_read",
        );
    }

    /// Fire an `expr` inspect callback for a VEX IRExpr eval (angr-lge2).
    ///
    /// Gated on `inspect_event_enabled(InspectBit::Expr)` so the no-BP case is one
    /// bitmask test per `eval_expr_with_callbacks` call (the most frequent
    /// dispatch site in the engine — fires for every constant, RdTmp,
    /// register read, load, unop, binop, ITE, etc.). When fired, the
    /// computed RustBV is reconstructed as a claripy AST and passed as
    /// `expr_result`. The original `IRExpr` is intentionally NOT passed
    /// (Rust IRExpr doesn't round-trip cleanly into a `pyvex.IRExpr`);
    /// the BP receives `expr=None` and only the computed value.
    pub(super) fn dispatch_expr_inspect(&self, callbacks: &PythonCallbacks, value: &RustBV) {
        let Some(value_ast) = self.inspect_ast(callbacks, InspectBit::Expr, value) else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_expr(self.current_state_id, "after", Some(&value_ast)),
            "expr",
        );
    }

    /// Fire an `address_concretization` inspect callback (angr-vfst).
    ///
    /// Gated on `inspect_event_enabled(InspectBit::AddressConcretization)`. The address AST is round-tripped
    /// into a claripy reconstruction for the BP; `result` carries the list
    /// of concrete addresses produced by the concretizer (`None` on
    /// `when="before"`). Mirrors the BEFORE/AFTER pattern in
    /// `AddressConcretizationMixin._apply_concretization_strategies`
    /// (`angr/storage/memory_mixins/address_concretization_mixin.py`); the
    /// strategy / memory / add_constraints attrs
    /// are passed as None because the Rust engine doesn't expose those
    /// objects to BPs (MVP gap, documented in `rust_engine.rst`).
    pub(super) fn dispatch_address_concretization_inspect(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        action: &str,
        when: &str,
        result: Option<Vec<u64>>,
    ) {
        let Some(addr_ast) =
            self.inspect_ast(callbacks, InspectBit::AddressConcretization, addr_val)
        else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_address_concretization(
                self.current_state_id,
                when,
                action,
                &addr_ast,
                result,
            ),
            "address_concretization",
        );
    }

    /// Fire a `symbolic_variable` inspect callback (angr-vfst).
    ///
    /// Gated on `inspect_event_enabled(InspectBit::SymbolicVariable)`. Fires `when="after"` when the
    /// Rust engine mints a fresh BVS internally — the most common dispatch
    /// site is `load_from_callback`'s fresh-symbol fallback when Python
    /// returns `is_symbolic=True` with no AST. Mirrors the BP_AFTER signature
    /// `SimSolver.Unconstrained` raises in `angr/state_plugins/solver.py`.
    pub(super) fn dispatch_symbolic_variable_inspect(
        &self,
        callbacks: &PythonCallbacks,
        name: &str,
        size_bits: u32,
        value: &RustBV,
    ) {
        let Some(expr_ast) = self.inspect_ast(callbacks, InspectBit::SymbolicVariable, value)
        else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_symbolic_variable(
                self.current_state_id,
                "after",
                name,
                size_bits,
                &expr_ast,
            ),
            "symbolic_variable",
        );
    }
}
