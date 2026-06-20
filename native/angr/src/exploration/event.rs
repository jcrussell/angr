//! `ExplorationEvent` — the value returned from the Rust exploration loop to
//! Python on every step boundary / callback request.
//!
//! Split out of `exploration::mod` (angr-zel8z.3). The `#[pyclass]` and its
//! constructor helpers live here; `register_exploration` in `mod.rs` still owns
//! the `m.add_class::<ExplorationEvent>()` registration line.

use crate::stash::{STASH_DEADENDED, STASH_ERRORED, STASH_FOUND};
use pyo3::prelude::*;

/// Event returned from exploration to Python.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// `#[pyo3(get)]` fields as new callback reasons emerge. Construction
/// outside this crate must go through the `ExplorationEvent::*`
/// helpers below rather than struct-literal syntax.
#[non_exhaustive]
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExplorationEvent {
    /// Type: "found", "deadended", "need_callback", "step_complete", "errored", "active_empty"
    #[pyo3(get)]
    pub event_type: String,
    /// Number of states in found stash.
    #[pyo3(get)]
    pub found_count: usize,
    /// Number of states in active stash.
    #[pyo3(get)]
    pub active_count: usize,
    /// Number of steps taken.
    #[pyo3(get)]
    pub steps_taken: u64,
    /// State ID for callback (if need_callback).
    #[pyo3(get)]
    pub callback_state_id: Option<u64>,
    /// Callback reason string.
    #[pyo3(get)]
    pub callback_reason: Option<String>,
    /// Callback address.
    #[pyo3(get)]
    pub callback_addr: Option<u64>,
    /// Callback name (e.g., SimProcedure name).
    #[pyo3(get)]
    pub callback_name: Option<String>,
    /// Syscall number (if syscall callback).
    #[pyo3(get)]
    pub callback_syscall_num: Option<u64>,
    /// Return address for SimProcedure.
    #[pyo3(get)]
    pub callback_return_addr: Option<u64>,
    /// Number of arguments for SimProcedure.
    #[pyo3(get)]
    pub callback_num_args: Option<usize>,
    /// Symbolic branch true target (if symbolic_branch callback).
    #[pyo3(get)]
    pub branch_true_target: Option<u64>,
    /// Symbolic branch false target (if symbolic_branch callback).
    #[pyo3(get)]
    pub branch_false_target: Option<u64>,
    /// Symbolic branch condition ID (if symbolic_branch callback).
    #[pyo3(get)]
    pub branch_condition_id: Option<u64>,
}

impl ExplorationEvent {
    /// Base constructor with common fields; all Optional fields default to None.
    pub(crate) fn base(
        event_type: &str,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            event_type: event_type.to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: None,
            callback_reason: None,
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: None,
            callback_return_addr: None,
            callback_num_args: None,
            branch_true_target: None,
            branch_false_target: None,
            branch_condition_id: None,
        }
    }

    pub(crate) fn found(found_count: usize, active_count: usize, steps: u64) -> Self {
        Self::base(STASH_FOUND, found_count, active_count, steps)
    }

    #[allow(dead_code)]
    pub(crate) fn deadended(found_count: usize, active_count: usize, steps: u64) -> Self {
        Self::base(STASH_DEADENDED, found_count, active_count, steps)
    }

    pub(crate) fn active_empty(found_count: usize, steps: u64) -> Self {
        Self::base("active_empty", found_count, 0, steps)
    }

    pub(crate) fn step_complete(found_count: usize, active_count: usize, steps: u64) -> Self {
        Self::base("step_complete", found_count, active_count, steps)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn need_simprocedure(
        state_id: u64,
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("simprocedure".to_string()),
            callback_addr: Some(addr),
            callback_name: Some(name),
            callback_return_addr: Some(return_addr),
            callback_num_args: Some(num_args),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn need_syscall(
        state_id: u64,
        syscall_num: Option<u64>,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("syscall".to_string()),
            callback_syscall_num: syscall_num,
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn need_symbolic_branch(
        state_id: u64,
        condition_id: u64,
        true_target: u64,
        false_target: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("symbolic_branch".to_string()),
            branch_true_target: Some(true_target),
            branch_false_target: Some(false_target),
            branch_condition_id: Some(condition_id),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn error(
        message: String,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_reason: Some(message),
            ..Self::base(STASH_ERRORED, found_count, active_count, steps)
        }
    }

    pub(crate) fn need_python_vex(
        state_id: u64,
        addr: u64,
        reason: &str,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("python_vex_fallback".to_string()),
            callback_addr: Some(addr),
            callback_name: Some(reason.to_string()),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }
}
