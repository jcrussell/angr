//! Execution configuration and branch-policy value types.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2) and
//! moved next to the interpreter that reads them (angr-mx00p.75). The
//! `ExecutionConfig` Python class is the shell's wrapper
//! (`callbacks::config::PyExecutionConfig`).

/// A branch that was taken but has an unexplored alternative.
///
/// When the Rust engine encounters a symbolic branch where both paths are
/// feasible, it picks one path to continue executing and records the other
/// as a deferred fork. The engine materializes the unexplored side itself
/// (see `exploration::fork_materialize`); the
/// struct never crosses to Python.
///
/// Rust-internal only (angr-03vl4.6). It used to carry a `#[pyclass]` with a
/// `#[new]` constructor, per-field `#[pyo3(get)]` getters and a `__repr__`,
/// registered on the module — but no pymethod anywhere returned one and
/// nothing on the Python side ever constructed one, so the whole surface was
/// unreachable. Every producer builds it with a struct literal
/// (`interpreter/statements.rs`); use that rather than reintroducing a
/// constructor, and `Debug` rather than reintroducing `__repr__`.
///
/// Dropping the getters exposed a `push_level: u32` field that only the
/// unreachable getter ever read — a third copy of the always-zero counter
/// whose siblings were also removed (angr-ph300.44, angr-c7xno.75); see
/// `symbolic::transaction_ops`'s module doc. So it went too.
#[derive(Debug, Clone)]
pub(crate) struct DeferredFork {
    /// Address where the branch occurred.
    pub(crate) branch_addr: u64,
    /// The path we took (true = took true branch, false = took false branch).
    pub(crate) path_taken: bool,
    /// Address of the unexplored path.
    pub(crate) unexplored_target: u64,
    /// Condition ID for constraint tracking, keying into the interpreter's
    /// stored-conditions map (see `apply_deferred_fork_constraints`).
    pub(crate) condition_id: u64,
}

/// Configuration for the execution loop with deferred forks.
///
/// Scope note (angr-9ke6b.17): symbolic-address concretization is *not*
/// configured here. Those bounds live in
/// [`AddressConcretizer`](crate::concretize::AddressConcretizer)'s
/// `read_range_limit` / `write_range_limit`, driven by SimOptions. A
/// `max_concretization_range` knob used to sit on this struct and was never
/// read by anything — don't re-add it; extend `AddressConcretizer` instead.
///
/// Likewise (angr-9ke6b.16) there is no branch-selection policy knob: the
/// engine always continues down the true branch and defers the other side
/// (see `use_deferred_forks`). A `BranchPolicy` enum was exposed to Python
/// here but never consulted by the fork path, so it was removed rather than
/// left as a silent no-op.
///
/// Third instance, same shape (angr-c7xno.5): an `enable_stride_detection`
/// knob lived here as a `#[pyo3(get, set)]` field wholly separate from
/// [`AddressConcretizer::enable_stride_detection`](crate::concretize::AddressConcretizer),
/// which is the field the concretizer actually reads. `AddressConcretizer` is
/// built without ever seeing an `ExecutionConfig` (see
/// `VEXInterpreter::with_config` and `RustSimState::with_solver_endian`), and
/// `VEXInterpreter::set_config` only assigns `self.config`, so setting the
/// knob from Python was a silent no-op. Removed rather than wired: nothing in
/// `angr/` ever set it, and stride detection is unconditionally on. Toggle it
/// on `AddressConcretizer` directly if a caller ever needs it off.
///
/// `#[non_exhaustive]`: minor versions may add new (angr-irwe) knob
/// fields. Construction outside this crate must go through
/// [`ExecutionConfig::default`] (or the Python `ExecutionConfig()`
/// constructor) rather than struct-literal syntax.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) struct ExecutionConfig {
    /// Maximum deferred forks before returning to Python.
    /// When this limit is reached, execution returns to Python even if
    /// max_blocks hasn't been hit.
    pub(crate) max_deferred_forks: u32,
    /// Whether to use deferred forks (if false, returns immediately on symbolic branch).
    pub(crate) use_deferred_forks: bool,
    /// Whether to enable eager region prefetch (batch prefetch entire regions).
    pub(crate) enable_eager_prefetch: bool,
    /// Maximum pages to prefetch in a single batch (default: 256 = 1MB).
    pub(crate) max_prefetch_batch: usize,
    /// Maximum symbolic IP targets before marking unconstrained (default: 257).
    /// When a symbolic jump target (e.g., ret from symbolic return address)
    /// concretizes to more than this many targets, the state is marked
    /// as unconstrained rather than forking into many states.
    pub(crate) max_symbolic_ip_targets: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            max_deferred_forks: 500, // Increased from 100 for complex binaries
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
            max_symbolic_ip_targets: 257, // Match Python angr default
        }
    }
}
