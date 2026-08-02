//! `#[pymethods]` for [`RustExplorationManager`]: run loop and resume entry points.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl RustExplorationManager {
    /// Run the exploration loop.
    ///
    /// Returns an ExplorationEvent when:
    /// - Found enough solutions
    /// - Active stash is empty
    /// - Need Python callback (SimProcedure, syscall)
    /// - Max steps reached
    #[pyo3(signature = (n=None))]
    pub fn run(&mut self, py: Python<'_>, n: Option<u32>) -> PyResult<ExplorationEvent> {
        // Python re-entered the loop without resuming a parked callback: that
        // gap is driver overhead, not a bounce excursion. Drop the clock.
        crate::gil_profile::park_cancel();
        let event = self.run_loop(py, n)?;
        // Handing control back to Python with a callback outstanding starts a
        // park-and-bounce excursion; the matching `resume_after_*` banks it as
        // `GilClass::Bounce` (bd angr-gorvf.8).
        crate::gil_profile::park_start(
            self.profiling.profiling_enabled && !self.pending_callbacks.is_empty(),
        );
        Ok(event)
    }

    // =========================================================================
    // Resume methods (from resume.rs)
    // =========================================================================

    /// Resume after a SimProcedure callback. See `resume::_resume_after_simprocedure` for the body.
    #[pyo3(signature = (state_id, new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        self._resume_after_simprocedure(
            py,
            state_id,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Resume after a syscall callback.
    #[pyo3(signature = (state_id, new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_syscall(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure - ensures constraint sync (GAP 2)
        self._resume_after_simprocedure(
            py,
            state_id,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Fast-path: deadend the pending callback state without full apply_changes.
    /// Used for SimProcedure continuations known to just call exit().
    /// See `resume::_deadend_pending_callback` for the body.
    pub fn deadend_pending_callback(&mut self, state_id: u64) -> PyResult<()> {
        self._deadend_pending_callback(state_id)
    }

    /// Resume after an error occurred during callback execution (P17).
    /// See `resume::_resume_after_error` for the body.
    pub fn resume_after_error(&mut self, state_id: u64, error_msg: &str) -> PyResult<()> {
        self._resume_after_error(state_id, error_msg)
    }

    /// Resume after Python handles a symbolic branch.
    /// See `resume::_resume_after_symbolic_branch` for the body.
    #[pyo3(signature = (state_id, true_pc, false_pc, true_constraints=None, false_constraints=None))]
    pub fn resume_after_symbolic_branch(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        self._resume_after_symbolic_branch(
            py,
            state_id,
            true_pc,
            false_pc,
            true_constraints,
            false_constraints,
        )
    }

    /// Resume after Python evaluates a find predicate (P2).
    /// See `resume::_resume_find_predicate` for the body.
    pub fn resume_find_predicate(&mut self, state_id: u64, matched: bool) -> PyResult<()> {
        self._resume_find_predicate(state_id, matched)
    }

    /// Resume after Python evaluates an avoid predicate (P7).
    /// See `resume::_resume_avoid_predicate` for the body.
    pub fn resume_avoid_predicate(&mut self, state_id: u64, matched: bool) -> PyResult<()> {
        self._resume_avoid_predicate(state_id, matched)
    }
}
