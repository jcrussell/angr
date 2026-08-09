//! Write-side `state.inspect` breakpoint dispatchers.
//!
//! One per event a statement can raise — `mem_write`, `reg_write`,
//! `tmp_write`, `instruction`, `exit` — each gated on the
//! `inspect_event_enabled` bitmask so the no-breakpoint case costs a single
//! test per statement. The value-injection contract (a BP that overrides
//! `state.inspect.*_expr` returns `Some(bv)` for the caller to substitute) is
//! documented per function. The read-side counterparts live in
//! `expressions.rs`.

use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Fire a `mem_write` inspect callback into Python for this store.
    ///
    /// Gated on `inspect_event_enabled(InspectBit::MemWrite)` so the common case
    /// (no breakpoints) is a single bitmask test per Store. Symbolic
    /// addresses are skipped for the MVP (uq4n.4) — only concrete
    /// addresses dispatch; symbolic-address dispatch is a follow-up.
    ///
    /// Fired twice per Store to mirror Python angr: `when='before'`
    /// (pre-commit) so a BP overriding `state.inspect.mem_write_expr`
    /// injects the stored value, then `when='after'` (post-commit, the
    /// AST is informational). Errors from the Python callback are
    /// swallowed and logged on the Python side; we do not surface them up
    /// the interpreter stack so a user BP error cannot halt exploration.
    ///
    /// Returns `Some(bv)` when the user's BP_BEFORE action overrode
    /// `state.inspect.mem_write_expr` (value injection — angr-inh0); the
    /// caller substitutes it for the stored value. Returns `None` when
    /// unchanged (and always for `when='after'`, post-commit), so the
    /// original store value stands.
    pub(super) fn dispatch_mem_write_inspect(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
        endness: Endness,
        when: &str,
    ) -> Option<RustBV> {
        let value_ast = self.inspect_ast(callbacks, InspectBit::MemWrite, data_val)?;
        let addr_u64 = addr_val.as_u64()?;
        let endness_str = match endness {
            Endness::Little => "Iend_LE",
            Endness::Big => "Iend_BE",
        };
        let mutated = callbacks
            .call_inspect_mem_write(
                self.current_state_id,
                when,
                addr_u64,
                data_size as u32,
                Some(&value_ast),
                endness_str,
            )
            .ok()??;
        // The user injected a new value via state.inspect.mem_write_expr
        // (only meaningful for when='before', pre-store — angr-inh0).
        // Convert it back to a RustBV; reject a width mismatch defensively
        // so a bad override can't silently corrupt the store.
        let bv = Python::attach(|py| {
            let bound = mutated.bind(py);
            crate::claripy_bridge::claripy_to_rustbv(py, bound, self.ctx).ok()
        })?;
        if bv.width() == (data_size * 8) as u32 {
            Some(bv)
        } else {
            None
        }
    }

    /// Fire a `reg_write` inspect callback into Python for a VEX `Put`.
    /// Gated on `inspect_event_enabled(InspectBit::RegWrite)`. Dispatches `when='after'`
    /// with the stored value as `reg_write_expr`.
    pub(super) fn dispatch_reg_write_inspect(
        &self,
        callbacks: &PythonCallbacks,
        offset: u32,
        size: u32,
        value: &RustBV,
    ) {
        let Some(value_ast) = self.inspect_ast(callbacks, InspectBit::RegWrite, value) else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_reg_write(
                self.current_state_id,
                "after",
                offset,
                size,
                Some(&value_ast),
            ),
            "reg_write",
        );
    }

    /// Fire a `tmp_write` inspect callback for a VEX `WrTmp` (angr-64pi).
    ///
    /// Gated on `inspect_event_enabled(InspectBit::TmpWrite)` so the no-breakpoint case is
    /// one bitmask test per `WrTmp`. Dispatches `when='after'` with the
    /// written value as `tmp_write_expr`. Fires before the slot mutation
    /// only when a BP is registered; the mutation itself happens in the
    /// caller after this returns so the slot is consistent post-dispatch.
    pub(super) fn dispatch_tmp_write_inspect(
        &self,
        callbacks: &PythonCallbacks,
        tmp_num: u32,
        value: &RustBV,
    ) {
        let Some(value_ast) = self.inspect_ast(callbacks, InspectBit::TmpWrite, value) else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_tmp_write(
                self.current_state_id,
                "after",
                tmp_num,
                Some(&value_ast),
            ),
            "tmp_write",
        );
    }

    /// Fire an `instruction` inspect callback into Python for a VEX `IMark`.
    /// Gated on `InspectBit::Instruction`. Bits 0..=5 mirror
    /// `crate::state::InspectEvent`; the `instruction` bit is custom (no
    /// `InspectEvent` slot — angr Python exposes it but the Rust
    /// `InspectionManager` enum doesn't track it). Dispatches `when='before'`.
    pub(super) fn dispatch_instruction_inspect(&self, callbacks: &PythonCallbacks, addr: u64) {
        if !callbacks.inspect_event_enabled(InspectBit::Instruction) {
            return;
        }
        note_inspect_error(
            callbacks.call_inspect_instruction(self.current_state_id, "before", addr),
            "instruction",
        );
    }

    /// Fire an `exit` inspect callback into Python for a VEX conditional `Exit`.
    /// Gated on `InspectBit::Exit`. Dispatches `when='before'`
    /// with the branch target, jumpkind name (`Ijk_*`), and guard AST.
    pub(super) fn dispatch_exit_inspect(
        &self,
        callbacks: &PythonCallbacks,
        target: u64,
        jk: JumpKind,
        guard: &RustBV,
    ) {
        let Some(guard_ast) = self.inspect_ast(callbacks, InspectBit::Exit, guard) else {
            return;
        };
        note_inspect_error(
            callbacks.call_inspect_exit(
                self.current_state_id,
                "before",
                target,
                jk.ijk_name(),
                Some(&guard_ast),
            ),
            "exit",
        );
    }
}
