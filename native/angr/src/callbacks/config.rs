//! Execution configuration and branch-policy value types.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).

use pyo3::prelude::*;

/// A branch that was taken but has an unexplored alternative.
///
/// When the Rust engine encounters a symbolic branch where both paths are
/// feasible, it picks one path to continue executing and records the other
/// as a deferred fork. Python can later create states for these unexplored
/// branches and schedule them for execution.
#[pyclass]
#[derive(Debug, Clone)]
pub struct DeferredFork {
    /// Address where the branch occurred.
    #[pyo3(get)]
    pub branch_addr: u64,
    /// The path we took (true = took true branch, false = took false branch).
    #[pyo3(get)]
    pub path_taken: bool,
    /// Address of the unexplored path.
    #[pyo3(get)]
    pub unexplored_target: u64,
    /// Condition ID for constraint tracking.
    /// Python can use this to reconstruct the branch condition.
    #[pyo3(get)]
    pub condition_id: u64,
    /// Solver push level before this branch constraint was added.
    /// Used for proper constraint handling during fork processing.
    #[pyo3(get)]
    pub push_level: u32,
    /// The branch condition as a claripy AST (if available).
    /// This is the original condition - path_taken indicates which path
    /// was explored. For the fork, we need the opposite constraint.
    #[pyo3(get)]
    pub condition_ast: Option<Py<PyAny>>,
}

#[pymethods]
impl DeferredFork {
    /// Create a new deferred fork.
    #[new]
    #[pyo3(signature = (branch_addr, path_taken, unexplored_target, condition_id, push_level=0, condition_ast=None))]
    pub fn new(
        branch_addr: u64,
        path_taken: bool,
        unexplored_target: u64,
        condition_id: u64,
        push_level: u32,
        condition_ast: Option<Py<PyAny>>,
    ) -> Self {
        DeferredFork {
            branch_addr,
            path_taken,
            unexplored_target,
            condition_id,
            push_level,
            condition_ast,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DeferredFork(branch_addr=0x{:x}, path_taken={}, unexplored=0x{:x}, push_level={})",
            self.branch_addr, self.path_taken, self.unexplored_target, self.push_level
        )
    }
}

/// Policy for choosing which branch to take when both paths are feasible.
#[pyclass]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BranchPolicy {
    /// Always take the true branch (default).
    #[default]
    TakeTrue,
    /// Always take the false branch.
    TakeFalse,
    /// Take the branch that continues to the next instruction (fall-through).
    TakeFallthrough,
    /// Alternate between true and false branches.
    Alternate,
}

#[pymethods]
impl BranchPolicy {
    /// Create the TakeTrue policy.
    #[staticmethod]
    pub fn take_true() -> Self {
        BranchPolicy::TakeTrue
    }

    /// Create the TakeFalse policy.
    #[staticmethod]
    pub fn take_false() -> Self {
        BranchPolicy::TakeFalse
    }

    /// Create the TakeFallthrough policy.
    #[staticmethod]
    pub fn take_fallthrough() -> Self {
        BranchPolicy::TakeFallthrough
    }

    /// Create the Alternate policy.
    #[staticmethod]
    pub fn alternate() -> Self {
        BranchPolicy::Alternate
    }
}

/// Configuration for the execution loop with deferred forks.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// `#[pyo3(get, set)]` knob fields. Construction outside this crate
/// must go through `ExecutionConfig::py_new` (the PyO3 `__init__`)
/// rather than struct-literal syntax.
#[non_exhaustive]
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum deferred forks before returning to Python.
    /// When this limit is reached, execution returns to Python even if
    /// max_blocks hasn't been hit.
    #[pyo3(get, set)]
    pub max_deferred_forks: u32,
    /// Branch selection policy.
    #[pyo3(get, set)]
    pub branch_policy: BranchPolicy,
    /// Whether to use deferred forks (if false, returns immediately on symbolic branch).
    #[pyo3(get, set)]
    pub use_deferred_forks: bool,
    /// Whether to enable eager region prefetch (batch prefetch entire regions).
    #[pyo3(get, set)]
    pub enable_eager_prefetch: bool,
    /// Maximum pages to prefetch in a single batch (default: 256 = 1MB).
    #[pyo3(get, set)]
    pub max_prefetch_batch: usize,
    /// Maximum concretization range for symbolic addresses (default: 65536).
    #[pyo3(get, set)]
    pub max_concretization_range: u64,
    /// Enable stride detection for array access patterns (default: true).
    #[pyo3(get, set)]
    pub enable_stride_detection: bool,
    /// Maximum symbolic IP targets before marking unconstrained (default: 257).
    /// When a symbolic jump target (e.g., ret from symbolic return address)
    /// concretizes to more than this many targets, the state is marked
    /// as unconstrained rather than forking into many states.
    #[pyo3(get, set)]
    pub max_symbolic_ip_targets: usize,
}

#[pymethods]
impl ExecutionConfig {
    /// Create a new execution config with default values.
    #[new]
    #[pyo3(signature = (max_deferred_forks=500, use_deferred_forks=false))]
    pub fn py_new(max_deferred_forks: u32, use_deferred_forks: bool) -> Self {
        ExecutionConfig {
            max_deferred_forks,
            branch_policy: BranchPolicy::TakeTrue,
            use_deferred_forks,
            enable_eager_prefetch: true,
            max_prefetch_batch: 256,
            max_concretization_range: 65536,
            enable_stride_detection: true,
            max_symbolic_ip_targets: 257,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionConfig(max_deferred_forks={}, use_deferred_forks={}, policy={:?}, eager_prefetch={}, max_prefetch_batch={})",
            self.max_deferred_forks,
            self.use_deferred_forks,
            self.branch_policy,
            self.enable_eager_prefetch,
            self.max_prefetch_batch
        )
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            max_deferred_forks: 500, // Increased from 100 for complex binaries
            branch_policy: BranchPolicy::TakeTrue,
            // Deferred forks enabled with one-per-step limit:
            // interpreter.rs limits to one deferred fork per
            // run_until_event call, then falls back to non-deferred
            // mode. This gives single-exit blocks the performance
            // benefit while multi-exit blocks are handled by Python.
            use_deferred_forks: true,
            // Eager prefetch disabled: fetching entire regions (e.g. 256 stack
            // pages) on first access is extremely slow due to Python callbacks.
            // Individual pages are fetched on demand instead.
            enable_eager_prefetch: false,
            max_prefetch_batch: 256, // 256 pages = 1MB (unused when eager disabled)
            max_concretization_range: 65536,
            enable_stride_detection: true,
            max_symbolic_ip_targets: 257, // Match Python angr default
        }
    }
}
