//! Profiling state for `RustExplorationManager`.
//!
//! Aggregates the three independent counters that were previously inlined as
//! separate fields on the manager: an enable flag, the per-step
//! `ExecutionStats` accumulator, and `NativeProcStats` for the native
//! procedure registry. Holding them together lets us pass them through a
//! single field instead of three, and keeps all profiling-related state in
//! one place.

use std::collections::HashMap;

use crate::interpreter_cb::ExecutionStats;

/// Statistics for native procedure execution.
#[derive(Debug, Clone, Default)]
pub(crate) struct NativeProcStats {
    /// Number of native procedure executions.
    pub(crate) native_calls: u64,
    /// Number of fallbacks to Python.
    pub(crate) python_fallbacks: u64,
    /// Per-procedure call counts.
    pub(crate) call_counts: HashMap<String, u64>,
}

/// Aggregated profiling state for the exploration manager.
#[derive(Debug, Default)]
pub(crate) struct ProfilingCollector {
    /// Whether Rust-side profiling is enabled.
    pub(crate) profiling_enabled: bool,
    /// Accumulated execution statistics across all steps.
    pub(crate) accumulated_stats: ExecutionStats,
    /// Statistics for native procedure executions.
    pub(crate) native_proc_stats: NativeProcStats,
}
