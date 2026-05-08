//! Rust-native exploration manager for symbolic execution.
//!
//! `RustExplorationManager` provides a Rust-first exploration loop that:
//! - Manages states entirely in Rust (using `RustSimState`)
//! - Processes symbolic branches with deferred forks
//! - Only returns to Python for SimProcedures and syscalls
//! - Implements find/avoid address checking in Rust
//!
//! This achieves ~3x speedup by keeping state management in Rust and
//! minimizing Python callback overhead.

use std::collections::{HashMap, HashSet, VecDeque};
use rustc_hash::FxHashMap;
use crate::stash::{StashManager, STASH_ACTIVE, STASH_FOUND, STASH_AVOID, STASH_DEADENDED, STASH_ERRORED, STASH_PRUNED, STASH_UNCONSTRAINED};
use std::sync::Arc;

use pyo3::class::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use crate::arch::{arch_from_name, default_cc_for_arch};
use crate::callbacks::{ExecutionConfig, PythonCallbacks, RunResult, DeferredFork};
use crate::claripy_bridge::{claripy_to_rustbv, rustbv_to_claripy};
use crate::interpreter_cb::{CallbackInterpreter, ExecutionStats, DCAS_UNSUPPORTED_REASON};
use crate::memory::Permission;
use crate::procedures::NativeProcedureRegistry;
use crate::syscalls::{NativeSyscallRegistry, SyscallOutcome};
use crate::solver::RustSolverContext;
use crate::state::{RustSimState, StateChanges};
use crate::symbolic::{RustBV, SymContext};

use std::cell::Cell;

mod stepping;
mod helpers;
mod profiling;
mod constraints;
mod memory_config;
mod execution_env;

use self::constraints::{ConstraintSolver, ConstraintTracker};
use self::execution_env::ExecutionEnvironment;
use self::memory_config::MemoryConfiguration;
use self::profiling::ProfilingCollector;
use self::stepping::StepError;

// Thread-local stepping state ID, accessible from callbacks without borrow conflicts.
thread_local! {
    static STEPPING_STATE_ID: Cell<Option<u64>> = Cell::new(None);
}

/// Get the current stepping state ID (safe to call from callbacks).
#[pyfunction]
pub fn get_stepping_state_id() -> Option<u64> {
    STEPPING_STATE_ID.with(|cell| cell.get())
}

/// Reason for returning to Python.
#[derive(Debug, Clone)]
pub enum CallbackReason {
    /// SimProcedure hook hit.
    SimProcedure {
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
    },
    /// Syscall instruction.
    Syscall { num: u64 },
    /// Find predicate needs Python evaluation.
    FindPredicate { addr: u64 },
    /// Avoid predicate needs Python evaluation.
    AvoidPredicate { addr: u64 },
    /// Error during execution.
    Error { message: String },
    /// Symbolic branch - both paths feasible, need Python to fork states.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Need Python VEX engine to handle block with unsupported operations.
    PythonVEXFallback { addr: u64, reason: String },
}

/// Event returned from exploration to Python.
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
    pub(crate) fn base(event_type: &str, found_count: usize, active_count: usize, steps: u64) -> Self {
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

    pub(crate) fn need_simprocedure(
        state_id: u64, addr: u64, name: String, num_args: usize, return_addr: u64,
        found_count: usize, active_count: usize, steps: u64,
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
        state_id: u64, syscall_num: u64,
        found_count: usize, active_count: usize, steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("syscall".to_string()),
            callback_syscall_num: Some(syscall_num),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn need_symbolic_branch(
        state_id: u64, condition_id: u64, true_target: u64, false_target: u64,
        found_count: usize, active_count: usize, steps: u64,
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

    pub(crate) fn error(message: String, found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            callback_reason: Some(message),
            ..Self::base(STASH_ERRORED, found_count, active_count, steps)
        }
    }

    pub(crate) fn need_python_vex(
        state_id: u64, addr: u64, reason: &str,
        found_count: usize, active_count: usize, steps: u64,
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

/// State held during a Python callback.
pub(crate) struct PendingCallback {
    pub(crate) state: RustSimState,
    /// Clean snapshot of state BEFORE any callback modifications.
    /// Used for creating deferred forks - they diverged before the callback,
    /// so they should not inherit callback constraints.
    pub(crate) pre_callback_snapshot: Option<RustSimState>,
    pub(crate) reason: CallbackReason,
    /// Jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    pub(crate) jumpkind: Option<String>,
    /// Forked solver context for Python callbacks.
    pub(crate) solver_ctx: Option<RustSolverContext>,
    /// Deferred forks accumulated before the callback.
    /// These should be processed when the callback returns.
    pub(crate) deferred_forks: Vec<DeferredFork>,
    /// Stored conditions for deferred fork handling.
    pub(crate) stored_conditions: FxHashMap<u64, RustBV>,
    /// Full state snapshots from before branch constraints were added.
    /// Keyed by condition_id, enables correct alternate-path forking.
    pub(crate) fork_snapshots: FxHashMap<u64, crate::interpreter_cb::BranchSnapshot>,
}

impl PendingCallback {
    /// Create a lightweight callback with no solver context or deferred state.
    /// Used for predicate evaluation (find/avoid predicates).
    pub(crate) fn lightweight(state: RustSimState, reason: CallbackReason) -> Self {
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason,
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: Vec::new(),
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        }
    }

    /// Create a callback with full interpreter context (deferred forks, conditions, snapshots).
    /// Used for SimProcedure, syscall, and symbolic branch callbacks after interpreter execution.
    pub(crate) fn with_context(
        state: RustSimState,
        pre_callback_snapshot: Option<RustSimState>,
        reason: CallbackReason,
        jumpkind: &str,
        solver_ctx: Option<RustSolverContext>,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, crate::interpreter_cb::BranchSnapshot>,
    ) -> Self {
        PendingCallback {
            state,
            pre_callback_snapshot,
            reason,
            jumpkind: Some(jumpkind.to_string()),
            solver_ctx,
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        }
    }
}

/// Native exploration technique variants.
///
/// These techniques run entirely in Rust during the exploration loop,
/// avoiding Python callback overhead for common technique patterns.
#[derive(Debug, Clone)]
pub(crate) enum NativeTechnique {
    /// Limits path length by block count. States exceeding `max_length` blocks
    /// are moved to "cut" (or "_DROP" if `drop` is true).
    LengthLimiter {
        max_length: usize,
        drop: bool,
    },
    /// Wall-clock timeout. Exploration stops after `timeout_secs` seconds.
    Timeout {
        timeout_secs: f64,
        start_time: Option<std::time::Instant>,
    },
    /// Basic loop bounding: limits how many times a single address can appear
    /// in a state's history. States exceeding the bound are moved to `discard_stash`.
    LoopBound {
        bound: usize,
        discard_stash: String,
    },
}

/// Rust-native exploration manager.
///
/// Manages states entirely in Rust with O(1) forking.
/// Returns to Python only for SimProcedures, syscalls, and predicates.
#[pyclass(unsendable)]
pub struct RustExplorationManager {
    /// Per-binary execution environment: arch metadata, calling convention,
    /// binary regions, block cache, endianness, history cap.
    pub(crate) environment: ExecutionEnvironment,
    /// Stash system (mirrors Python's).
    pub(crate) sm: StashManager,
    /// Find addresses.
    pub(crate) find_addrs: HashSet<u64>,
    /// Avoid addresses.
    pub(crate) avoid_addrs: HashSet<u64>,
    /// Whether find condition has callable predicates.
    pub(crate) find_needs_python: bool,
    /// Whether avoid condition has callable predicates.
    pub(crate) avoid_needs_python: bool,
    /// Execution config for deferred forks.
    pub(crate) exec_config: ExecutionConfig,
    /// Python callbacks for memory/lifting.
    pub(crate) callbacks: Option<PythonCallbacks>,
    /// Hook addresses.
    pub(crate) hooks: HashSet<u64>,
    /// SimProcedures: address -> (name, num_args, no_return).
    pub(crate) simprocedures: HashMap<u64, (String, usize, bool)>,
    /// Pending state waiting for Python callback result.
    pub(crate) pending_callback: Option<PendingCallback>,
    /// ID of the state currently being stepped (for Python callbacks to identify)
    pub(crate) current_stepping_state_id: Option<u64>,
    /// Total steps executed.
    pub(crate) steps: u64,
    /// Error log: (addr, message, state_id).
    pub(crate) errors: Vec<(u64, String, u64)>,
    /// Number of finds required before stopping.
    pub(crate) num_find: usize,
    /// Maximum steps per run iteration.
    pub(crate) max_steps_per_run: u32,
    /// Native procedure registry.
    pub(crate) native_procedures: NativeProcedureRegistry,
    /// Native syscall registry (skip Python `_handle_syscall_callback` round-trip).
    pub(crate) native_syscalls: NativeSyscallRegistry,
    /// VEX fallback tracking: count and unique addresses.
    pub(crate) vex_fallback_count: u64,
    pub(crate) vex_fallback_addrs: HashMap<u64, String>,
    /// Visibility counter for `IRStmt::CAS` double-CAS (cmpxchg16b) fallbacks.
    /// Incremented alongside `vex_fallback_count` whenever the reason carries
    /// `DCAS_UNSUPPORTED_REASON`. Surfaced via `stats()` and
    /// `get_fallback_stats()` so DCAS-driven deadends are diagnosable.
    pub(crate) dcas_unsupported_count: u64,
    /// Total count of SimProcedure invocations that were dispatched to the
    /// Python `_handle_simprocedure_callback` (rather than handled natively).
    /// This includes: native handler missing, native handler returned `Err`,
    /// and addresses inside the binary (user-placed Python hooks). Surfaced
    /// via `stats()` so a regression that flips a hot procedure off the
    /// native path is visible without rebuilding.
    pub(crate) simprocedure_python_fallback_count: u64,
    /// Total count of syscalls dispatched to the Python
    /// `_handle_syscall_callback` (rather than handled by `NativeSyscall`).
    /// Includes: no native handler registered for (arch, num) and native
    /// handler returned `Err`. Surfaced via `stats()`.
    pub(crate) syscall_python_fallback_count: u64,
    /// State IDs that have already produced a DCAS warning. We log the first
    /// DCAS hit per state to avoid spamming the log on tight DCAS loops.
    pub(crate) dcas_warned_states: HashSet<u64>,
    /// Stack of (address, expiry_step) for zero-length hook skip tracking.
    /// Each entry represents an address to skip, valid until the specified step.
    /// This prevents infinite loops when a hook with length=0 runs and
    /// returns to the same address. Stack-based to handle nested hooks.
    pub(crate) skip_hook_stack: Vec<(u64, u64)>,
    // state_roots is now in self.sm (StashManager)
    /// P9 fix: Use LIFO (stack) state selection instead of FIFO (queue).
    /// When true, states are popped from the back (DFS). Default is false (BFS).
    pub(crate) use_lifo: bool,
    /// Solver configuration: lazy_solves flag and Z3 timeout.
    pub(crate) constraint_solver: ConstraintSolver,
    /// Memory and VEX configuration: zero-fill, concretizer, vex opt levels.
    pub(crate) memory_config: MemoryConfiguration,
    /// Maximum number of states in the active stash. None = unlimited.
    pub(crate) max_active_states: Option<usize>,
    // drop_terminal_states, avoided_count, pruned_count, deadended_count
    // are now in self.sm (StashManager)
    /// Native exploration techniques that run entirely in Rust.
    pub(crate) native_techniques: Vec<NativeTechnique>,
    /// Constraint-related per-run tracking: uniqueness filter sets and
    /// the find/avoid predicate skip lists.
    pub(crate) constraint_tracker: ConstraintTracker,
    /// Profiling state: enable flag, per-step `ExecutionStats`, and
    /// `NativeProcStats` accumulator.
    pub(crate) profiling: ProfilingCollector,
    // state_index is now in self.sm (StashManager)
}

#[pymethods]
impl RustExplorationManager {
    /// Create a new exploration manager.
    #[new]
    #[pyo3(signature = (arch="amd64", little_endian=None))]
    pub fn new(arch: &str, little_endian: Option<bool>) -> PyResult<Self> {
        let arch_info = arch_from_name(arch).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported architecture: {}", arch))
        })?;

        let vex_arch = arch_info.vex_arch();

        Ok(RustExplorationManager {
            environment: ExecutionEnvironment::new(
                arch.to_string(),
                vex_arch,
                default_cc_for_arch(arch),
                little_endian,
            ),
            sm: StashManager::new(),
            find_addrs: HashSet::new(),
            avoid_addrs: HashSet::new(),
            find_needs_python: false,
            avoid_needs_python: false,
            exec_config: ExecutionConfig::default(),
            callbacks: None,
            hooks: HashSet::new(),
            simprocedures: HashMap::new(),
            pending_callback: None,
            current_stepping_state_id: None,
            steps: 0,
            errors: Vec::new(),
            num_find: 1,
            max_steps_per_run: 5000,
            native_procedures: NativeProcedureRegistry::new(),
            native_syscalls: NativeSyscallRegistry::new(),
            vex_fallback_count: 0,
            vex_fallback_addrs: HashMap::new(),
            dcas_unsupported_count: 0,
            simprocedure_python_fallback_count: 0,
            syscall_python_fallback_count: 0,
            dcas_warned_states: HashSet::new(),
            skip_hook_stack: Vec::new(),
            use_lifo: false,  // P9: Default to BFS (FIFO)
            constraint_solver: ConstraintSolver::new(),
            memory_config: MemoryConfiguration::default(),
            max_active_states: None,
            native_techniques: Vec::new(),
            constraint_tracker: ConstraintTracker::default(),
            profiling: ProfilingCollector::default(),
        })
    }


    // =========================================================================
    // PyAPI methods (from pyapi.rs)
    // =========================================================================

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.environment.arch_name
    }

    /// Get the total number of steps executed.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Get active state count.
    pub fn active_count(&self) -> usize {
        self.sm.get(STASH_ACTIVE).map(|s| s.len()).unwrap_or(0)
    }

    /// Get found state count.
    pub fn found_count(&self) -> usize {
        self.sm.stashes().get(STASH_FOUND).map(|s| s.len()).unwrap_or(0)
    }

    /// Get stash counts as a dictionary.
    pub fn stash_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, stash) in self.sm.stashes() {
            dict.set_item(name, stash.len())?;
        }
        Ok(dict)
    }

    /// Set find addresses.
    pub fn set_find_addrs(&mut self, addrs: Vec<u64>) {
        self.find_addrs = addrs.into_iter().collect();
        self.find_needs_python = false;
    }

    /// Set avoid addresses.
    pub fn set_avoid_addrs(&mut self, addrs: Vec<u64>) {
        self.avoid_addrs = addrs.into_iter().collect();
        self.avoid_needs_python = false;
    }

    /// Mark that find condition has callable predicates (needs Python).
    pub fn set_find_needs_python(&mut self, needs: bool) {
        self.find_needs_python = needs;
    }

    /// Mark that avoid condition has callable predicates (needs Python).
    pub fn set_avoid_needs_python(&mut self, needs: bool) {
        self.avoid_needs_python = needs;
    }

    /// P9 fix: Set state selection to LIFO (DFS - depth-first search).
    pub fn set_state_selection_lifo(&mut self) {
        self.use_lifo = true;
        log::debug!("State selection set to LIFO (DFS)");
    }

    /// P9 fix: Set state selection to FIFO (BFS - breadth-first search).
    pub fn set_state_selection_fifo(&mut self) {
        self.use_lifo = false;
        log::debug!("State selection set to FIFO (BFS)");
    }

    /// Set the number of solutions to find before stopping.
    pub fn set_num_find(&mut self, n: usize) {
        self.num_find = n;
    }

    /// Set maximum steps per run iteration.
    pub fn set_max_steps_per_run(&mut self, n: u32) {
        self.max_steps_per_run = n;
    }

    /// Enable lazy solves mode (skip satisfiability checks on forks).
    pub fn set_lazy_solves(&mut self, enabled: bool) {
        self.constraint_solver.lazy_solves = enabled;
    }

    /// Enable zero-fill for unconstrained memory reads.
    /// When true, unmapped memory returns zero instead of fresh symbolic values.
    pub fn set_zero_fill_unconstrained(&mut self, enabled: bool) {
        self.memory_config.zero_fill_unconstrained = enabled;
    }

    /// Set the Z3 solver timeout in milliseconds (default: 30000).
    pub fn set_solver_timeout(&mut self, timeout_ms: u32) {
        self.constraint_solver.solver_timeout_ms = timeout_ms;
    }

    /// Set the maximum number of states in the active stash.
    /// When the limit is reached, new forked states are pruned to avoid OOM.
    /// None (default) means no limit.
    #[pyo3(signature = (limit=None))]
    pub fn set_max_active_states(&mut self, limit: Option<usize>) {
        self.max_active_states = limit;
    }

    /// Get the current max_active_states limit.
    pub fn get_max_active_states(&self) -> Option<usize> {
        self.max_active_states
    }

    /// Set the global VEX optimization level (0-3).
    /// None = use pyvex default (typically 1).
    /// Level 0: no optimization. Level 1: standard. Level 2-3: aggressive.
    #[pyo3(signature = (level=None))]
    pub fn set_vex_opt_level(&mut self, level: Option<i32>) {
        self.memory_config.vex_opt_level = level;
        // Invalidate block cache since opt_level affects IR output
        self.environment.block_cache.clear();
    }

    /// Get the current VEX optimization level.
    pub fn get_vex_opt_level(&self) -> Option<i32> {
        self.memory_config.vex_opt_level
    }

    /// Set a per-address VEX optimization level override.
    /// Blocks at this address will be lifted with the specified opt_level.
    pub fn set_vex_opt_level_override(&mut self, addr: u64, level: i32) {
        self.memory_config.vex_opt_level_overrides.insert(addr, level);
        // Remove this address from block cache since opt_level changed
        self.environment.block_cache.pop(&addr);
    }

    /// Remove a per-address VEX optimization level override.
    pub fn remove_vex_opt_level_override(&mut self, addr: u64) {
        self.memory_config.vex_opt_level_overrides.remove(&addr);
        self.environment.block_cache.pop(&addr);
    }

    /// Clear all per-address VEX optimization level overrides.
    pub fn clear_vex_opt_level_overrides(&mut self) {
        let addrs: Vec<u64> = self.memory_config.vex_opt_level_overrides.keys().copied().collect();
        self.memory_config.vex_opt_level_overrides.clear();
        for addr in addrs {
            self.environment.block_cache.pop(&addr);
        }
    }

    /// Resolve the VEX optimization level for a given address.
    /// Per-address overrides take precedence over the global level.
    pub fn resolve_vex_opt_level(&self, addr: u64) -> Option<i32> {
        self.memory_config.vex_opt_level_overrides.get(&addr).copied()
            .or(self.memory_config.vex_opt_level)
    }

    /// Set whether to drop terminal states (avoid/pruned/deadended) immediately.
    /// When true (default), terminal states are dropped to save memory.
    /// Set to false when states need to be recovered (e.g., factory.callable()).
    pub fn set_drop_terminal_states(&mut self, enabled: bool) {
        self.sm.set_drop_terminal_states(enabled);
    }

    /// Configure address concretization strategies to match Python's configuration.
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `read_range_limit` - Range limit for read strategies (default: 1024)
    /// * `write_range_limit` - Range limit for write strategies (default: 128)
    /// * `symbolic_write_addresses` - Whether SYMBOLIC_WRITE_ADDRESSES is enabled
    #[pyo3(signature = (use_approximate, read_range_limit=None, write_range_limit=None, symbolic_write_addresses=false))]
    pub fn configure_concretization_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
    ) {
        self.memory_config.concretizer_config.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
        );
    }

    /// Enable or disable Rust-side profiling.
    /// When enabled, per-step timing and counters are accumulated.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling.profiling_enabled = enabled;
    }

    /// Set the maximum length of each state's `history` / `detailed_history`
    /// ring buffers. 0 means unlimited (legacy behavior — can OOM on long
    /// explorations). Default is 1000. The new value is applied to every
    /// state already in any stash, plus any future state created via this
    /// manager.
    pub fn set_max_history(&mut self, max: usize) {
        self.environment.max_history = max;
        for stash in self.sm.stashes_mut().values_mut() {
            for state in stash.iter_mut() {
                state.set_max_history(max);
            }
        }
    }

    /// Get the current per-state max_history value. 0 = unlimited.
    pub fn get_max_history(&self) -> usize {
        self.environment.max_history
    }

    /// Get accumulated execution statistics as a dict.
    pub fn get_execution_stats(&self) -> HashMap<String, u64> {
        self.profiling.accumulated_stats.to_hashmap()
    }

    /// Reset accumulated execution statistics.
    pub fn reset_execution_stats(&mut self) {
        self.profiling.accumulated_stats.reset();
    }

    /// Set Python callbacks for memory/lifting.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
    }

    /// GC traversal: visit Python callback refs held inside the cloned
    /// PythonCallbacks struct. The Python wrapper `mgr` owns
    /// `mgr._rust_mgr` (this object), and via `set_callbacks` this object
    /// holds a cloned PythonCallbacks whose Py<PyAny> bound methods point
    /// back at `mgr` — a non-trivial cycle that cycle-GC can break only if
    /// __traverse__/__clear__ are exposed.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(cbs) = &self.callbacks {
            cbs.traverse_fields(&visit)?;
        }
        Ok(())
    }

    /// GC clear: drop the bound-method refs held inside the cloned
    /// PythonCallbacks struct. After this returns, the engine can no
    /// longer call back into Python — but cycle-GC only invokes __clear__
    /// when the object is being collected, so further callbacks would not
    /// be issued.
    fn __clear__(&mut self) {
        if let Some(cbs) = &mut self.callbacks {
            cbs.clear_fields();
        }
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hooks.insert(addr);
    }

    /// Add multiple hook addresses.
    pub fn add_hooks(&mut self, addrs: Vec<u64>) {
        for addr in addrs {
            self.hooks.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    /// Register a SimProcedure.
    #[pyo3(signature = (addr, name, num_args=0, no_return=false))]
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
        self.hooks.insert(addr);
        self.simprocedures.insert(addr, (name, num_args, no_return));
    }

    /// Register multiple SimProcedures.
    pub fn register_simprocedures(&mut self, procs: Vec<(u64, String, usize, bool)>) {
        for (addr, name, num_args, no_return) in procs {
            self.hooks.insert(addr);
            self.simprocedures.insert(addr, (name, num_args, no_return));
        }
    }

    /// Load binary code regions.
    pub fn load_binary_regions(&mut self, regions: Vec<(u64, Vec<u8>)>) {
        self.environment.binary_regions = regions.into_iter()
            .map(|(base, data)| (base, Arc::new(data)))
            .collect();
    }

    /// Create a new RustSimState and add it to a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        let mut state = RustSimState::new_with_endian(&self.environment.arch_name, self.environment.little_endian)
            .map_err(|e| PyValueError::new_err(e))?;
        let state_id = state.state_id();

        // Propagate memory options
        if self.memory_config.zero_fill_unconstrained {
            state.memory_mut().set_zero_fill_unconstrained(true);
        }

        // Propagate solver timeout
        if self.constraint_solver.solver_timeout_ms != 30000 {
            state.solver().borrow().set_timeout(self.constraint_solver.solver_timeout_ms);
        }

        // Propagate per-state history cap
        state.set_max_history(self.environment.max_history);

        // Copy hooks to state
        for &_addr in &self.hooks {
            // State hooks are checked during execution
        }

        self.index_state(state_id, stash);
        self.sm.stashes_mut()
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(state);

        Ok(state_id)
    }

    /// Add an existing RustSimState to a stash.
    #[pyo3(signature = (stash, state))]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        // Fork the state to get our own copy
        let mut forked = state.inner().fork();
        let state_id = forked.state_id();

        // Propagate memory options
        if self.memory_config.zero_fill_unconstrained {
            forked.memory_mut().set_zero_fill_unconstrained(true);
        }

        // Propagate solver timeout
        if self.constraint_solver.solver_timeout_ms != 30000 {
            forked.solver().borrow().set_timeout(self.constraint_solver.solver_timeout_ms);
        }

        // Propagate per-state history cap
        forked.set_max_history(self.environment.max_history);

        // Track this state as its own root (it was added via Python)
        self.sm.set_root(state_id, state_id);

        self.index_state(state_id, stash);
        self.sm.stashes_mut()
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(forked);
    }

    /// Merge multiple states into one using symbolic merge conditions.
    ///
    /// Each state's constraints are guarded by a fresh 1-bit merge flag.
    /// Registers and memory that differ between states become ITE expressions.
    /// The merged state is placed into `dest_stash`.
    ///
    /// Returns the merged state's ID.
    #[pyo3(signature = (state_ids, dest_stash="active"))]
    pub fn merge_states(&mut self, state_ids: Vec<u64>, dest_stash: &str) -> PyResult<u64> {
        use crate::symbolic::RustBV;

        if state_ids.len() < 2 {
            return Err(PyValueError::new_err("merge_states requires at least 2 state IDs"));
        }

        // Look up all states by ID across all stashes
        let mut states: Vec<RustSimState> = Vec::new();
        for &sid in &state_ids {
            let mut found = false;
            for (_stash_name, stash) in self.sm.stashes() {
                for state in stash.iter() {
                    if state.state_id() == sid {
                        states.push(state.fork());
                        found = true;
                        break;
                    }
                }
                if found { break; }
            }
            if !found {
                return Err(PyValueError::new_err(format!("state {} not found", sid)));
            }
        }

        // Create merge conditions: one 1-bit BVS per state
        let solver = states[0].solver();
        let merge_conditions: Vec<RustBV> = (0..states.len())
            .map(|i| {
                let name = format!("merge_flag_{}", i);
                solver.borrow().new_bv(&name, 1)
            })
            .collect();

        // Perform the merge
        let others: Vec<&RustSimState> = states[1..].iter().collect();
        let merged = states[0].merge(&others, &merge_conditions);
        let merged_id = merged.state_id();

        // Track state root
        self.sm.set_root(merged_id, merged_id);
        self.index_state(merged_id, dest_stash);
        self.sm.stashes_mut()
            .entry(dest_stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(merged);

        Ok(merged_id)
    }

    /// Get the PC of a state in a stash by index.
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.sm.get(stash).and_then(|s| s.get(index)).map(|s| s.pc())
    }

    /// Get the PC of a state by its ID (O(1) via state index, no full export).
    pub fn get_state_pc_by_id(&self, state_id: u64) -> Option<u64> {
        // find_state already checks pending_callback first.
        self.find_state(state_id).map(|s| s.pc())
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.sm.get(stash)
            .map(|s| s.iter().map(|state| state.state_id()).collect())
            .unwrap_or_default()
    }

    /// Get (state_id, addr, stdout_len) tuples for states in a stash.
    /// Used by Python predicate caching to skip re-evaluation when
    /// a state's address and stdout haven't changed.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_predicate_info(&self, stash: &str) -> Vec<(u64, u64, usize)> {
        self.sm.get(stash)
            .map(|s| s.iter().map(|state| {
                (state.state_id(), state.pc(), state.stdout_buffer().len())
            }).collect())
            .unwrap_or_default()
    }

    /// Check if there are any active states (O(1), no allocation).
    pub fn has_active_states(&self) -> bool {
        self.sm.get(STASH_ACTIVE).map_or(false, |s| !s.is_empty())
    }

    /// Get the number of states in a stash (O(1), no allocation).
    #[pyo3(signature = (stash="active"))]
    pub fn stash_count(&self, stash: &str) -> usize {
        self.sm.get(stash).map_or(0, |s| s.len())
    }

    /// Get the stash name a state currently belongs to (O(1) via state index).
    /// Returns None if the state isn't found in any stash. Used by
    /// RustStateProxy.__repr__ for cheap REPL debugging output.
    pub fn state_stash(&self, state_id: u64) -> Option<String> {
        self.sm.stash_of(state_id).map(|s| s.to_string())
    }

    /// Get the number of solver constraints for a state (O(1), reads
    /// the SymContext's atomic counter — no Z3 traversal). Returns None
    /// if the state isn't found. Used by RustStateProxy.__repr__.
    pub fn state_constraint_count(&self, state_id: u64) -> Option<usize> {
        let state = self.find_state(state_id)?;
        Some(state.solver().borrow().num_constraints())
    }

    /// Rebuild the state index after run() modifies stashes internally.
    /// Call from Python after run() returns to keep index up to date.
    pub fn sync_state_index(&mut self) {
        self.rebuild_state_index();
    }

    /// Get the root state ID for any state.
    /// Returns the original (initial) state from which this state was forked.
    pub fn get_state_root(&self, state_id: u64) -> Option<u64> {
        self.sm.roots().get(&state_id).copied()
    }

    /// Set the PC of the pending callback state (for external initialization).
    pub fn set_pending_state_pc(&mut self, pc: u64) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending.state.set_pc(pc);
            Ok(())
        })
    }

    /// Map memory in the pending state.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn pending_state_map_memory(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending.state.map_memory_data(addr, data, Permission::from_bits(permissions));
            Ok(())
        })
    }

    /// Map memory in active states.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        if let Some(stash) = self.sm.get_mut(STASH_ACTIVE) {
            for state in stash.iter_mut() {
                state.map_memory_data(addr, data, Permission::from_bits(permissions));
            }
        }
    }

    /// Get the branch condition from the pending symbolic branch callback.
    ///
    /// Returns the condition as a claripy AST that Python can use for forking.
    pub fn get_pending_branch_condition(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_pending(|pending| {
            // Get the condition ID from the callback reason
            let condition_id = match &pending.reason {
                CallbackReason::SymbolicBranch { condition_id, .. } => *condition_id,
                _ => return Err(PyValueError::new_err("pending callback is not a symbolic branch")),
            };

            // Look up the condition in stored_conditions
            let condition = pending.stored_conditions.get(&condition_id)
                .ok_or_else(|| PyValueError::new_err(
                    format!("condition {} not found in stored_conditions", condition_id)
                ))?;

            // Convert to claripy AST
            let claripy = py.import("claripy")?;
            rustbv_to_claripy(py, condition, claripy.as_any())
                .map_err(|e| PyRuntimeError::new_err(format!("failed to convert condition: {}", e)))
        })
    }

    /// Get register value from pending state (concrete only).
    pub fn get_pending_register(&self, name: &str) -> PyResult<Option<u128>> {
        self.with_pending(|pending| {
            pending.state.get_register(name)
                .map(|bv| bv.as_u128())
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))
        })
    }

    /// Get register as claripy AST from pending state (handles symbolic).
    pub fn get_pending_register_ast(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self.with_pending(|pending| {
            let bv = pending.state.get_register(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let claripy = py.import("claripy")?;
            rustbv_to_claripy(py, &bv, claripy.as_any())
                .map_err(|e| PyRuntimeError::new_err(format!("register conversion: {}", e)))
        })
    }

    /// Get history (BBL addresses) from pending callback state.
    ///
    /// This is used by Python to initialize history on callback states,
    /// preventing IndexError when hooks access `state.history.recent_bbl_addrs[-1]`.
    pub fn get_pending_history(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| Ok(pending.state.history().to_vec()))
    }

    /// Get jumpkind for pending callback.
    ///
    /// Returns the jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    /// This is used by Python to properly initialize callstack management.
    pub fn get_pending_jumpkind(&self) -> PyResult<String> {
        self.with_pending(|pending| {
            Ok(pending.jumpkind.clone().unwrap_or_else(|| "Ijk_Boring".to_string()))
        })
    }

    /// Set register value in pending state.
    pub fn set_pending_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let size = pending.state.arch().register_size(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let bv = crate::symbolic::RustBV::concrete(value, size * 8);
            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        })
    }

    /// Set register to a symbolic value from a handle ID.
    ///
    /// Used for syncing symbolic return values from SimProcedures.
    /// The handle_id should reference a RustBV in the solver's symbol table.
    pub fn set_pending_register_symbolic(&mut self, name: &str, handle_id: u64) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            // Look up the RustBV from the symbol table
            let bv = if let Some(ref solver) = pending.solver_ctx {
                solver.symbol_table().get(handle_id)
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "invalid handle id: {}", handle_id
                    )))?
            } else {
                return Err(PyRuntimeError::new_err("no solver context in pending state"));
            };

            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        })
    }

    /// Set a symbolic register value in the pending state from claripy AST.
    ///
    /// This allows direct sync of symbolic register values from Python callbacks.
    /// The claripy AST is converted to RustBV and stored in the pending state.
    pub fn set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            // Convert claripy AST to RustBV
            let bv = claripy_to_rustbv(py, ast, ctx_ref)
                .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;

            drop(sym_ctx);

            if pending.state.set_register(reg_name, bv) {
                log::debug!("Set symbolic register {} from claripy AST", reg_name);
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", reg_name)))
            }
        })
    }

    /// Import symbolic memory from Python hook into Rust's symbolic_objects.
    ///
    /// Called after a hook writes symbolic memory. Converts the claripy AST
    /// to RustBV and imports it into the pending state's SymbolicMemory.
    /// Import symbolic memory into a state by ID (for init-time symbolic data).
    #[pyo3(signature = (state_id, addr, ast))]
    pub fn import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let sym_ctx = solver_ref.borrow();
            let bv = claripy_to_rustbv(py, ast, &*sym_ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion: {}", e)))?;
            drop(sym_ctx);
            state.memory_mut().import_symbolic_value(addr, bv, None);
            Ok(())
        })
    }

    pub fn import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            // Convert claripy AST to RustBV (caches original AST for round-trip)
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let bv = claripy_to_rustbv(py, ast, &*sym_ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;
            drop(sym_ctx);

            // Import into symbolic memory via existing infrastructure
            // Symbol ID is not used by import_symbolic_value, so pass None
            pending.state.memory_mut().import_symbolic_value(addr, bv, None);
            log::debug!("Imported symbolic memory at 0x{:x}", addr);
            Ok(())
        })
    }

    /// Get memory from pending state.
    pub fn get_pending_memory(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self.with_pending(|pending| {
            let bv = pending.state.memory_load(addr, size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let value = bv.to_u128();
            let bytes: Vec<u8> = (0..size as usize)
                .map(|i| (value >> (i * 8)) as u8)
                .collect();
            Ok(bytes)
        })
    }

    /// Store memory in pending state.
    pub fn set_pending_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let width = (data.len() * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in data.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = crate::symbolic::RustBV::concrete(value, width);
            pending.state.memory_store(addr, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Get dirty page addresses from pending state.
    ///
    /// This returns the list of page-aligned addresses that have been
    /// modified in the pending callback state.
    pub fn get_pending_dirty_pages(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| Ok(pending.state.get_dirty_pages()))
    }

    /// Clear dirty page tracking in pending state.
    pub fn clear_pending_dirty_tracking(&mut self) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending.state.clear_dirty_pages();
            Ok(())
        })
    }

    /// Export pending constraints as a list of claripy ASTs.
    ///
    /// Returns constraints that can be added to Python state.solver.
    /// This exports stored branch conditions accumulated during Rust execution.
    pub fn export_pending_constraints(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.with_pending(|pending| {
            let mut result = Vec::new();

            // Import claripy for AST conversion
            let claripy_mod = py.import("claripy")?;

            // Export stored branch conditions as claripy ASTs
            for (_condition_id, rustbv) in &pending.stored_conditions {
                match rustbv_to_claripy(py, rustbv, &claripy_mod) {
                    Ok(ast) => {
                        result.push(ast);
                    }
                    Err(e) => {
                        log::debug!("Could not convert stored condition to claripy: {}", e);
                    }
                }
            }

            log::debug!("Exported {} pending constraints", result.len());
            Ok(result)
        })
    }

    /// Get handle IDs that are actively referenced in the pending state.
    ///
    /// Returns handle IDs used in stored conditions and deferred forks.
    /// These should not be evicted from the AST handle cache.
    pub fn get_active_handle_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        if let Some(ref pending) = self.pending_callback {
            // Add condition IDs from stored_conditions
            for (id, _) in &pending.stored_conditions {
                ids.push(*id);
            }
            // Add condition IDs from deferred forks
            for fork in &pending.deferred_forks {
                ids.push(fork.condition_id);
            }
        }
        ids
    }

    /// Export the pending state as a full snapshot.
    ///
    /// This allows Python to get a complete snapshot of the pending state
    /// including all registers, memory pages, and metadata.
    pub fn export_pending_state(&self) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self.with_pending(|pending| Ok(pending.state.export_full()))
    }

    /// Get the root state ID for the pending callback state.
    ///
    /// When Rust forks states internally, Python only has cached data for the
    /// original state that was added via Python. This method returns the root
    /// state ID (the original state) for any forked descendant.
    ///
    /// Returns:
    ///     The root state ID if available, or None if the state has no tracked root.
    pub fn get_pending_root_state_id(&self) -> PyResult<Option<u64>> {
        self.with_pending(|pending| {
            let state_id = pending.state.state_id();
            Ok(self.sm.roots().get(&state_id).copied())
        })
    }

    /// Get the full ancestry chain for the pending callback state.
    ///
    /// Returns a list of state IDs starting with the current state and walking
    /// up the parent chain: [state_id, parent_id, grandparent_id, ...].
    ///
    /// This is used by Python to find cached state data when the current state
    /// is a multi-level fork of an original state.
    pub fn get_pending_ancestry(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| {
            let mut ancestry = vec![pending.state.state_id()];

            // Walk the parent chain
            let current_parent = pending.state.parent_id();
            while let Some(parent_id) = current_parent {
                ancestry.push(parent_id);
                // We can't traverse further without access to parent state objects,
                // but we can include the root state if known
                break;
            }

            // Add root state if not already in ancestry
            let state_id = pending.state.state_id();
            if let Some(&root_id) = self.sm.roots().get(&state_id) {
                if !ancestry.contains(&root_id) {
                    ancestry.push(root_id);
                }
            }

            Ok(ancestry)
        })
    }

    /// Fork the pending state's solver context for Python callbacks.
    ///
    /// This creates a new RustSolverContext that inherits all constraints
    /// accumulated during Rust exploration. The forked context can be
    /// attached to the Python callback state, ensuring SimProcedures
    /// see the full constraint context.
    ///
    /// This is critical for proper constraint propagation: without it,
    /// callbacks would create fresh solver contexts without parent
    /// constraints, leading to incorrect symbolic evaluation.
    /// Export a callback bundle: registers, solver context, history, jumpkind
    /// in a single FFI call. Reduces ~20 individual calls to 1.
    ///
    /// Returns a Python dict with:
    /// - "registers": dict of register_name -> concrete u128 value (None if symbolic)
    /// - "solver": forked RustSolverContext
    /// - "history": list of u64 BBL addresses
    /// - "jumpkind": string
    /// - "constraint_count": u64
    /// - "stdout": bytes (accumulated stdout buffer)
    #[pyo3(signature = (register_names, shared_solver=true))]
    pub fn export_callback_bundle<'py>(
        &self,
        py: Python<'py>,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.with_pending(|pending| {
            let dict = PyDict::new(py);

            // Registers: batch export all requested registers
            let reg_dict = PyDict::new(py);
            for name in &register_names {
                match pending.state.get_register(name) {
                    Some(bv) => {
                        if let Some(val) = bv.as_u128() {
                            reg_dict.set_item(name, val)?;
                        } else {
                            // Symbolic — set to None, Python will fetch AST if needed
                            reg_dict.set_item(name, py.None())?;
                        }
                    }
                    None => {
                        reg_dict.set_item(name, py.None())?;
                    }
                }
            }
            dict.set_item("registers", reg_dict)?;

            // Solver context: shared (O(1) Rc clone) or forked (~3ms Z3 clone)
            let solver_ref = pending.state.solver();
            if shared_solver {
                let rust_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
                let constraint_count = rust_ctx.num_constraints();
                dict.set_item("solver", Py::new(py, rust_ctx)?)?;
                dict.set_item("constraint_count", constraint_count)?;
            } else {
                let forked_ctx = solver_ref.borrow().fork();
                let rust_ctx = RustSolverContext::from_sym_context(forked_ctx);
                let constraint_count = rust_ctx.num_constraints();
                dict.set_item("solver", Py::new(py, rust_ctx)?)?;
                dict.set_item("constraint_count", constraint_count)?;
            }

            // History
            dict.set_item("history", pending.state.history().to_vec())?;

            // Jumpkind
            dict.set_item("jumpkind",
                pending.jumpkind.clone().unwrap_or_else(|| "Ijk_Boring".to_string()))?;

            // Stdout buffer
            dict.set_item("stdout", pending.state.stdout_buffer().to_vec())?;

            Ok(dict)
        })
    }

    pub fn fork_pending_solver(&self) -> PyResult<RustSolverContext> {
        self.with_pending(|pending| {
            // Fork the pending state's solver context
            let solver_ref = pending.state.solver();
            let forked_ctx = solver_ref.borrow().fork();
            // Create a new RustSolverContext wrapping the forked SymContext
            Ok(RustSolverContext::from_sym_context(forked_ctx))
        })
    }

    /// Borrow the pending state's solver context without forking.
    ///
    /// Returns a RustSolverContext that shares the same Z3 solver as the
    /// pending state via Rc reference counting. This is O(1) instead of
    /// the ~3ms Z3 solver clone in fork_pending_solver().
    ///
    /// Constraints added through this solver go directly to the pending state,
    /// so post-callback constraint sync via add_constraints_to_pending() should
    /// be skipped to avoid double-adding.
    pub fn borrow_pending_solver(&self) -> PyResult<RustSolverContext> {
        self.with_pending(|pending| {
            let solver_rc = pending.state.solver().clone();
            Ok(RustSolverContext::from_shared_sym_context(solver_rc))
        })
    }

    /// Add constraints from Python callbacks back to the pending state.
    ///
    /// This is called after a SimProcedure executes to sync any new
    /// constraints added during the callback back to the Rust solver.
    /// This ensures bidirectional constraint flow between Rust and Python.
    ///
    /// Args:
    ///     constraints: List of claripy AST constraints to add
    pub fn add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        use pyo3::types::PyListMethods;

        self.with_pending_mut(|pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            let len = constraints.len();
            for i in 0..len {
                // Use get_item with usize index
                if let Ok(constraint) = constraints.get_item(i) {
                    // Convert claripy AST to RustBV
                    if let Ok(bv) = claripy_to_rustbv(py, &constraint, ctx_ref) {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                            }
                        }
                    } else {
                        log::debug!("Could not convert constraint {} from Python", i);
                    }
                }
            }
            Ok(())
        })
    }

    /// Add constraints from Python to a state in a stash by state ID.
    /// This is used to sync initial constraints from the Python state.
    pub fn add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            // Pre-fetch Z3 backend for fast path
            #[cfg(feature = "vex-engine-z3")]
            let z3_backend = py.import("claripy")
                .and_then(|c| c.getattr("backends"))
                .and_then(|b| b.getattr("z3"))
                .ok();

            let mut added = 0u32;
            for item in constraints.iter() {
                // Fast path: extract raw Z3 AST and assert directly
                #[cfg(feature = "vex-engine-z3")]
                {
                    if let Some(ref backend) = z3_backend {
                        if let Ok(z3_obj) = backend.call_method1("convert", (&item,)) {
                            if let Ok(ast_ref) = z3_obj.call_method0("as_ast") {
                                if let Ok(ptr) = ast_ref.getattr("value").and_then(|v| v.extract::<usize>()) {
                                    if ptr != 0 {
                                        unsafe { ctx_ref.add_constraint_raw(ptr); }
                                        if let Ok(bv) = claripy_to_rustbv(py, &item, ctx_ref) {
                                            ctx_ref.assumed_constraints_push(bv, true);
                                        }
                                        added += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                // Slow path: convert via RustBV
                match claripy_to_rustbv(py, &item, ctx_ref) {
                    Ok(bv) => {
                        if bv.width() == 1 {
                            sym_ctx.assume_true(&bv);
                        } else {
                            let zero = RustBV::concrete(0, bv.width());
                            let neq = bv.ne(&zero, ctx_ref);
                            sym_ctx.assume_true(&neq);
                        }
                        added += 1;
                    }
                    Err(e) => {
                        log::debug!("Could not convert initial constraint: {}", e);
                    }
                }
            }
            log::debug!("Added {} initial constraints to state {}", added, state_id);
            Ok(state.satisfiable())
        })
    }

    /// Export raw Z3 assertion pointers from a state's solver.
    /// Lossless — captures ALL Z3 assertions, not just those tracked
    /// in assumed_constraints (which drops constraints where claripy_to_rustbv fails).
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_constraint_ptrs(&self, state_id: u64) -> PyResult<Vec<usize>> {
        self.with_state(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            Ok(ctx.export_z3_assertion_ptrs())
        })
    }

    /// Import raw Z3 assertion pointers to a state's solver.
    #[cfg(feature = "vex-engine-z3")]
    pub fn import_z3_constraint_ptrs(&mut self, state_id: u64, ptrs: Vec<usize>) -> PyResult<bool> {
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            for ptr in &ptrs {
                if *ptr != 0 {
                    unsafe { ctx.add_constraint_raw(*ptr); }
                }
            }
            log::debug!("Imported {} Z3 constraints to state {}", ptrs.len(), state_id);
            Ok(state.satisfiable())
        })
    }

    /// Debug: dump solver state for a given state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_info(&self, state_id: u64) -> PyResult<String> {
        self.with_state(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let push_level = ctx.debug_push_level();
            let n_constraints = ctx.num_constraints();
            let ptrs = ctx.export_z3_assertion_ptrs();
            Ok(format!("push_level={}, num_constraints={}, exported_ptrs={}", push_level, n_constraints, ptrs.len()))
        })
    }

    /// Export constraints from a state in any stash as claripy ASTs.
    pub fn export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let claripy = py.import("claripy")?;
        self.with_state(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let assumed = ctx.get_assumed_constraints();
            let mut results = Vec::new();
            for (bv, is_true) in &assumed {
                match rustbv_to_claripy(py, bv, claripy.as_any()) {
                    Ok(ast) => {
                        if *is_true {
                            results.push(ast);
                        } else {
                            match claripy.call_method1("Not", (ast,)) {
                                Ok(negated) => results.push(negated.unbind()),
                                Err(_) => {}
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Ok(results)
        })
    }

    /// Set the Z3 solver timeout (ms) on a specific state's solver context.
    ///
    /// Future forks of this state inherit the new timeout.  Used by
    /// RustSolverProxy to honor `state.solver.timeout = N` assignments.
    pub fn set_state_solver_timeout(&self, state_id: u64, timeout_ms: u32) -> PyResult<()> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_solver_timeout: state {} not found",
                state_id
            ))
        })?;
        state.solver().borrow().set_timeout(timeout_ms);
        Ok(())
    }

    /// Get the Z3 solver timeout (ms) on a specific state's solver context.
    pub fn get_state_solver_timeout(&self, state_id: u64) -> PyResult<u32> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "get_state_solver_timeout: state {} not found",
                state_id
            ))
        })?;
        Ok(state.solver().borrow().timeout_ms())
    }

    /// Get the per-state mmap base pointer (mirrors Python's
    /// `state.heap.mmap_base`). The native mmap syscall handler bumps this
    /// on `addr=0` calls; Python imports it on stash export to keep the two
    /// engines from handing out overlapping mmap regions.
    pub fn get_state_mmap_base(&self, state_id: u64) -> PyResult<u64> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "get_state_mmap_base: state {} not found",
                state_id
            ))
        })?;
        Ok(state.mmap_base())
    }

    /// Set the per-state mmap base pointer. Used by tests and by Python-side
    /// fallbacks that allocate from `state.heap.mmap_base` and need to push
    /// the advance back into Rust so subsequent native mmaps don't collide.
    pub fn set_state_mmap_base(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_mmap_base: state {} not found",
                state_id
            ))
        })?;
        state.set_mmap_base(addr);
        Ok(())
    }

    /// Get the per-state posix brk pointer (mirrors Python's
    /// `state.posix.brk`). The native brk syscall handler bumps this on
    /// concrete `brk(addr)` calls; Python imports it on stash export to keep
    /// a Python-side `set_brk` fallback from handing out heap addresses that
    /// overlap a Rust-allocated region.
    pub fn get_state_posix_brk(&self, state_id: u64) -> PyResult<u64> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "get_state_posix_brk: state {} not found",
                state_id
            ))
        })?;
        Ok(state.posix_brk())
    }

    /// Set the per-state posix brk pointer. Used by tests and by Python-side
    /// fallbacks (symbolic `brk` argument, collision retry) that bump
    /// `state.posix.brk` and need to push the advance back into Rust.
    pub fn set_state_posix_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_posix_brk: state {} not found",
                state_id
            ))
        })?;
        state.set_posix_brk(addr);
        Ok(())
    }

    // ----- Per-state Python-AST metadata (symbolic_pages /
    //       hook_symbolic_memory / addr_to_ast). Storage now lives in
    //       RustSimState; these methods are the FFI surface that replaces the
    //       old Python `_state_metadata: Dict[int, StateMetadata]` map.

    /// Replace the whole `symbolic_pages` map for a state. Mirrors the
    /// previous Python-side `_state_md(sid).symbolic_pages = pages` write.
    pub fn set_state_symbolic_pages<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        pages: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_symbolic_pages: state {} not found",
                state_id
            ))
        })?;
        let mut map: HashMap<u64, Py<PyAny>> = HashMap::with_capacity(pages.len());
        for (key, value) in pages.iter() {
            let addr: u64 = key.extract()?;
            map.insert(addr, value.unbind());
        }
        let _ = py;
        state.replace_symbolic_pages(map);
        Ok(())
    }

    /// Snapshot the `symbolic_pages` map for a state as a Python dict.
    /// Returns an empty dict if the state is unknown or has no entries — the
    /// truthiness check at call sites already handles both cases.
    pub fn get_state_symbolic_pages<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(state) = self.find_state(state_id) {
            for (addr, ast) in state.symbolic_pages() {
                dict.set_item(*addr, ast.clone_ref(py))?;
            }
        }
        Ok(dict)
    }

    /// Insert/replace an entry in `hook_symbolic_memory` for a state.
    pub fn set_state_hook_symbolic_memory(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_hook_symbolic_memory: state {} not found",
                state_id
            ))
        })?;
        state.set_hook_symbolic_memory(addr, ast, size);
        Ok(())
    }

    /// Snapshot the `hook_symbolic_memory` map as a `dict[int, (ast, size)]`.
    pub fn get_state_hook_symbolic_memory<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(state) = self.find_state(state_id) {
            for (addr, (ast, size)) in state.hook_symbolic_memory() {
                dict.set_item(*addr, (ast.clone_ref(py), *size))?;
            }
        }
        Ok(dict)
    }

    /// Insert/replace an entry in `addr_to_ast` for a state.
    pub fn set_state_addr_to_ast(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_addr_to_ast: state {} not found",
                state_id
            ))
        })?;
        state.set_addr_to_ast(addr, ast, size);
        Ok(())
    }

    /// Snapshot the `addr_to_ast` map as a `dict[int, (ast, size)]`.
    pub fn get_state_addr_to_ast<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(state) = self.find_state(state_id) {
            for (addr, (ast, size)) in state.addr_to_ast() {
                dict.set_item(*addr, (ast.clone_ref(py), *size))?;
            }
        }
        Ok(dict)
    }

    /// Drop all per-state metadata (`symbolic_pages`, `hook_symbolic_memory`,
    /// `addr_to_ast`) for a state. No-op if the state is unknown — matches the
    /// `_state_metadata.pop(state_id, None)` semantics it replaces.
    pub fn clear_state_metadata(&mut self, state_id: u64) -> PyResult<()> {
        if let Some(state) = self.find_state_mut(state_id) {
            state.clear_state_metadata();
        }
        Ok(())
    }

    /// Fork the solver context of an arbitrary state (by ID).
    ///
    /// Returns a new RustSolverContext with all of the state's constraints,
    /// allowing Python to evaluate/solve against any state — not just the
    /// pending callback state.  This is used by RustStateProxy.
    pub fn fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "fork_state_solver: state {} not found",
                state_id
            ))
        })?;
        let solver_ref = state.solver();
        let forked_ctx = solver_ref.borrow().fork();
        Ok(RustSolverContext::from_sym_context(forked_ctx))
    }

    /// Get the number of constraints in the pending state's solver.
    pub fn pending_constraint_count(&self) -> PyResult<usize> {
        self.with_pending(|pending| {
            let solver_ref = pending.state.solver();
            Ok(solver_ref.borrow().num_constraints())
        })
    }

    /// Get the ID of the state currently being stepped.
    pub fn get_current_stepping_state_id(&self) -> Option<u64> {
        self.current_stepping_state_id
    }

    /// Load from pending callback state's Rust memory.
    /// Used by SimProcedure callbacks to read the correct per-state memory.
    /// Get all mapped page addresses from pending callback state's memory.
    pub fn get_pending_mapped_pages(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| {
            Ok(pending.state.memory().pages().keys().map(|&pn| pn << 12).collect())
        })
    }

    /// Load an entire page (4096 bytes) from pending callback state's memory.
    pub fn pending_memory_load_page(&self, page_addr: u64) -> PyResult<Vec<u8>> {
        self.with_pending(|pending| {
            pending.state.memory().load_page_concrete(page_addr)
                .map_err(|e| PyValueError::new_err(format!("page load failed: {}", e)))
        })
    }

    pub fn pending_memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self.with_pending(|pending| {
            let solver_ref = pending.state.solver();
            let ctx = solver_ref.borrow();
            match pending.state.memory().load_concrete(addr, size, &*ctx) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u128() {
                        let byte_count = (size as usize).min(16);
                        Ok(val.to_le_bytes()[..byte_count].to_vec())
                    } else {
                        let solver = pending.state.solver();
                        let ctx = solver.borrow();
                        if let Some(val) = ctx.eval(&bv) {
                            let byte_count = (size as usize).min(16);
                            Ok(val.to_le_bytes()[..byte_count].to_vec())
                        } else {
                            Ok(vec![0u8; size as usize])
                        }
                    }
                }
                Err(_) => Ok(vec![0u8; size as usize]),
            }
        })
    }

    /// Store to pending callback state's Rust memory.
    pub fn pending_memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let mut value: u128 = 0;
            for (i, &byte) in data.iter().enumerate() {
                if i < 16 { value |= (byte as u128) << (i * 8); }
            }
            let bv = RustBV::concrete(value, (data.len() * 8) as u32);
            pending.state.memory_mut().store_concrete(addr, bv)
                .map_err(|e| PyRuntimeError::new_err(format!("memory store error: {}", e)))
        })
    }

    /// Map memory with data in pending callback state.
    pub fn pending_memory_map_data(&mut self, addr: u64, data: &[u8], perm: u8) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending.state.map_memory_data(addr, data, crate::memory::Permission::from_bits(perm));
            Ok(())
        })
    }

    /// Set address to skip hook check for on next step.
    ///
    /// This is used to prevent infinite loops with zero-length hooks.
    /// When a hook with length=0 runs, it returns to the same address.
    /// Without this skip mechanism, the hook would trigger again immediately.
    ///
    /// The skip is automatically cleared after one step or when the address is used.
    /// GAP 6: Stack-based tracking allows for nested zero-length hooks.
    pub fn set_skip_hook_addr(&mut self, addr: u64) {
        // Set expiry to current_step + 2 to account for step increment
        // This ensures the skip persists through the next step
        let expiry = self.steps + 2;
        self.skip_hook_stack.push((addr, expiry));
        log::debug!("Added skip hook 0x{:x} with expiry step {}", addr, expiry);
    }

    /// Clear all pending skip_hook entries.
    pub fn clear_skip_hook_addr(&mut self) {
        self.skip_hook_stack.clear();
    }

    /// Clear skip entry for a specific address.
    pub fn clear_skip_hook_for_addr(&mut self, addr: u64) {
        self.skip_hook_stack.retain(|&(a, _)| a != addr);
    }

    /// Get errors encountered during exploration.
    pub fn get_errors(&self) -> Vec<(u64, String, u64)> {
        self.errors.clone()
    }

    /// Clear error log.
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Move states between stashes.
    pub fn move_states(&mut self, from_stash: &str, to_stash: &str, filter_fn: Option<Py<PyAny>>) -> PyResult<usize> {
        // If no filter, move all
        if filter_fn.is_none() {
            if let Some(mut from) = self.sm.remove(from_stash) {
                let count = from.len();
                // Update index for all moved states
                for state in from.iter() {
                    self.sm.index(state.state_id(), to_stash);
                }
                let to = self.sm.stashes_mut().entry(to_stash.to_string()).or_insert_with(VecDeque::new);
                to.append(&mut from);
                self.sm.insert(from_stash, VecDeque::new());
                return Ok(count);
            }
            return Ok(0);
        }

        // With filter - evaluate Python filter_fn per state
        let filter_fn = filter_fn.expect("filter_fn checked before call");
        let from = match self.sm.stashes().get(from_stash) {
            Some(s) if !s.is_empty() => s,
            _ => return Ok(0),
        };

        // First pass: determine which states pass the filter (immutable borrow)
        let mut move_indices = Vec::new();
        Python::attach(|py| -> PyResult<()> {
            for (i, state) in from.iter().enumerate() {
                let result = filter_fn.call1(py, (state.state_id(),))?;
                if result.extract::<bool>(py).unwrap_or(false) {
                    move_indices.push(i);
                }
            }
            Ok(())
        })?;

        if move_indices.is_empty() {
            return Ok(0);
        }

        // Second pass: move matching states (mutable borrow)
        let mut moved = Vec::new();
        if let Some(from) = self.sm.get_mut(from_stash) {
            for &idx in move_indices.iter().rev() {
                if let Some(state) = from.remove(idx) {
                    moved.push(state);
                }
            }
        }
        // Update index and destination stash after releasing from-stash borrow
        for state in &moved {
            self.sm.index(state.state_id(), to_stash);
        }
        let count = moved.len();
        let to = self.sm.stashes_mut().entry(to_stash.to_string()).or_insert_with(VecDeque::new);
        for state in moved.into_iter().rev() {
            to.push_back(state);
        }
        Ok(count)
    }

    /// P8 fix: Move a single state by ID between stashes.
    pub fn move_state(&mut self, state_id: u64, from_stash: &str, to_stash: &str) -> PyResult<bool> {
        // Find and remove the state from the source stash
        let mut found_state = None;
        if let Some(stash) = self.sm.get_mut(from_stash) {
            let mut idx = None;
            for (i, state) in stash.iter().enumerate() {
                if state.state_id() == state_id {
                    idx = Some(i);
                    break;
                }
            }
            if let Some(i) = idx {
                found_state = stash.remove(i);
            }
        }

        // Add to destination stash if found
        if let Some(state) = found_state {
            self.index_state(state_id, to_stash);
            let to = self.sm.stashes_mut().entry(to_stash.to_string()).or_insert_with(VecDeque::new);
            to.push_back(state);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// P8 fix: Clear all states from a stash.
    pub fn clear_stash(&mut self, stash: &str) {
        self.sm.clear(stash);
    }

    /// Prepare for a new exploration stage: move a specific found state
    /// to active and clear all other stashes. Returns the state ID of the
    /// moved state. This avoids constraint transfer between managers.
    pub fn reset_for_stage(&mut self, found_state_id: u64) -> PyResult<u64> {
        // Move the found state from 'found' to 'active'
        let moved = self.move_state(found_state_id, "found", "active")?;
        if !moved {
            return Err(PyValueError::new_err(format!(
                "state {} not found in 'found' stash", found_state_id)));
        }

        // Clear all other stashes
        for stash in &[STASH_FOUND, STASH_AVOID, STASH_DEADENDED, STASH_ERRORED, STASH_UNCONSTRAINED] {
            self.sm.clear(stash);
        }

        // Remove all other active states (keep only the moved one)
        if let Some(active) = self.sm.get_mut("active") {
            active.retain(|s| s.state_id() == found_state_id);
        }

        Ok(found_state_id)
    }

    /// Get statistics.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("steps", self.steps)?;
        dict.set_item("active", self.active_count())?;
        dict.set_item("found", self.found_count())?;
        dict.set_item("errors", self.errors.len())?;
        dict.set_item("hooks", self.hooks.len())?;
        dict.set_item("simprocedures", self.simprocedures.len())?;
        dict.set_item("find_addrs", self.find_addrs.len())?;
        dict.set_item("avoid_addrs", self.avoid_addrs.len())?;
        dict.set_item("block_cache_size", self.environment.block_cache.len())?;
        dict.set_item("native_proc_calls", self.profiling.native_proc_stats.native_calls)?;
        dict.set_item("native_proc_fallbacks", self.profiling.native_proc_stats.python_fallbacks)?;
        dict.set_item("avoided_count", self.sm.avoided_count)?;
        dict.set_item("pruned_count", self.sm.pruned_count)?;
        dict.set_item("deadended_count", self.sm.deadended_count)?;
        dict.set_item("drop_terminal_states", self.sm.drop_terminal_states())?;
        dict.set_item("state_roots_size", self.sm.roots().len())?;
        dict.set_item("vex_fallback_count", self.vex_fallback_count)?;
        dict.set_item("vex_fallback_unique_addrs", self.vex_fallback_addrs.len())?;
        dict.set_item("dcas_unsupported_count", self.dcas_unsupported_count)?;
        dict.set_item(
            "simprocedure_python_fallback_count",
            self.simprocedure_python_fallback_count,
        )?;
        dict.set_item(
            "syscall_python_fallback_count",
            self.syscall_python_fallback_count,
        )?;
        Ok(dict)
    }

    /// Get VEX fallback statistics: total count and per-address reasons.
    ///
    /// Returns a dict with:
    ///   "count": total number of VEX fallbacks
    ///   "addresses": dict mapping hex address string -> reason string
    ///   "dcas_unsupported_count": subset of fallbacks driven by double-CAS
    ///   "simprocedure_python_fallback_count": SimProcedures dispatched to Python
    ///   "syscall_python_fallback_count": syscalls dispatched to Python
    pub fn get_fallback_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("count", self.vex_fallback_count)?;
        let addrs = PyDict::new(py);
        for (&addr, reason) in &self.vex_fallback_addrs {
            addrs.set_item(format!("0x{:x}", addr), reason)?;
        }
        dict.set_item("addresses", addrs)?;
        dict.set_item("dcas_unsupported_count", self.dcas_unsupported_count)?;
        dict.set_item(
            "simprocedure_python_fallback_count",
            self.simprocedure_python_fallback_count,
        )?;
        dict.set_item(
            "syscall_python_fallback_count",
            self.syscall_python_fallback_count,
        )?;
        Ok(dict)
    }

    // =========================================================================
    // Native Procedure Management
    // =========================================================================

    /// Disable all native procedures (always use Python).
    pub fn disable_native_procedures(&mut self) {
        self.native_procedures.disable_all();
    }

    /// Enable all native procedures.
    pub fn enable_native_procedures(&mut self) {
        self.native_procedures.enable_all();
    }

    /// Check if native procedures are enabled.
    pub fn native_procedures_enabled(&self) -> bool {
        self.native_procedures.is_enabled()
    }

    /// Disable a specific native procedure (fall back to Python).
    pub fn disable_native_procedure(&mut self, name: &str) {
        self.native_procedures.disable(name);
    }

    /// Enable a specific native procedure.
    pub fn enable_native_procedure(&mut self, name: &str) {
        self.native_procedures.enable(name);
    }

    /// Set a Python override for a procedure.
    ///
    /// When set, the native implementation is never called.
    pub fn set_python_override(&mut self, name: &str) {
        self.native_procedures.set_python_override(name);
    }

    /// Remove a Python override.
    pub fn remove_python_override(&mut self, name: &str) {
        self.native_procedures.remove_python_override(name);
    }

    /// Get list of available native procedures.
    pub fn list_native_procedures(&self) -> Vec<String> {
        self.native_procedures.procedure_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native_procedure(&self, name: &str) -> bool {
        self.native_procedures.has_native(name)
    }

    /// Get native procedure statistics.
    pub fn native_procedure_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("native_calls", self.profiling.native_proc_stats.native_calls)?;
        dict.set_item("python_fallbacks", self.profiling.native_proc_stats.python_fallbacks)?;

        let call_counts = PyDict::new(py);
        for (name, count) in &self.profiling.native_proc_stats.call_counts {
            call_counts.set_item(name, *count)?;
        }
        dict.set_item("call_counts", call_counts)?;

        Ok(dict)
    }

    /// Register a Python callable as a native procedure.
    ///
    /// The callable receives `list[int]` of concrete arg values and must
    /// return `Optional[int]` for the return value (None = no value).
    /// Symbolic arguments cause an automatic fallback to the regular
    /// Python SimProcedure path; the registered callable is only invoked
    /// when all args are concrete.
    #[pyo3(signature = (name, num_args, no_return, callable))]
    pub fn register_python_procedure(
        &mut self,
        name: String,
        num_args: usize,
        no_return: bool,
        callable: Py<PyAny>,
    ) {
        let proc = std::sync::Arc::new(
            crate::procedures::python_proc::PythonNativeProcedure::new(
                name, num_args, no_return, callable,
            ),
        );
        self.native_procedures.register(proc);
    }

    // =========================================================================
    // Native Uniqueness Filter
    // =========================================================================

    /// Enable native uniqueness filter with given register names.
    ///
    /// After each step in run(), states with duplicate register tuples
    /// are moved to 'not_unique' stash. This replaces the Python
    /// CheckUniqueness technique with zero FFI overhead.
    pub fn register_uniqueness_filter(&mut self, register_names: Vec<String>) {
        self.constraint_tracker.uniqueness_registers = register_names;
        self.constraint_tracker.uniqueness_set.clear();
        // Ensure not_unique stash exists
        self.sm.stashes_mut().entry("not_unique".to_string()).or_insert_with(VecDeque::new);
    }

    /// Disable the native uniqueness filter.
    pub fn disable_uniqueness_filter(&mut self) {
        self.constraint_tracker.uniqueness_registers.clear();
        self.constraint_tracker.uniqueness_set.clear();
    }

    /// Check if native uniqueness filter is enabled.
    pub fn uniqueness_filter_enabled(&self) -> bool {
        !self.constraint_tracker.uniqueness_registers.is_empty()
    }

    /// Get the number of unique register tuples seen.
    pub fn uniqueness_set_size(&self) -> usize {
        self.constraint_tracker.uniqueness_set.len()
    }

    // =========================================================================
    // Native Exploration Techniques
    // =========================================================================

    /// Register a native LengthLimiter technique.
    ///
    /// States whose history exceeds `max_length` blocks are moved to "cut"
    /// (or "_DROP" if `drop` is true). Runs entirely in Rust with zero FFI overhead.
    pub fn register_length_limiter(&mut self, max_length: usize, drop: bool) {
        self.native_techniques.push(NativeTechnique::LengthLimiter {
            max_length,
            drop,
        });
        if !drop {
            self.sm.stashes_mut().entry("cut".to_string()).or_insert_with(VecDeque::new);
        }
    }

    /// Register a native Timeout technique.
    ///
    /// Exploration stops after `timeout_secs` seconds. All active states are
    /// moved to "timeout" stash. Timer starts on first call to apply_native_techniques().
    pub fn register_timeout(&mut self, timeout_secs: f64) {
        self.native_techniques.push(NativeTechnique::Timeout {
            timeout_secs,
            start_time: None,
        });
        self.sm.stashes_mut().entry("timeout".to_string()).or_insert_with(VecDeque::new);
    }

    /// Register a native LoopBound technique.
    ///
    /// States where any single address appears more than `bound` times in
    /// their history are moved to `discard_stash`. This is a simplified
    /// version of LoopSeer that doesn't require CFG analysis.
    #[pyo3(signature = (bound, discard_stash="spinning"))]
    pub fn register_loop_bound(&mut self, bound: usize, discard_stash: &str) {
        self.native_techniques.push(NativeTechnique::LoopBound {
            bound,
            discard_stash: discard_stash.to_string(),
        });
        self.sm.stashes_mut().entry(discard_stash.to_string()).or_insert_with(VecDeque::new);
    }

    /// Get the number of registered native techniques.
    pub fn native_technique_count(&self) -> usize {
        self.native_techniques.len()
    }

    /// Clear all native techniques.
    pub fn clear_native_techniques(&mut self) {
        self.native_techniques.clear();
    }

    // =========================================================================
    // State Export Methods
    // =========================================================================

    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    pub fn export_state(&self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self.with_state(state_id, |state| Ok(state.export_full()))
    }

    /// Export a state by ID, flushing pending writes first.
    pub fn export_state_flushed(&mut self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        // find_state_mut only checks stashes; we still need an explicit pending fallback.
        if let Some(state) = self.find_state_mut(state_id) {
            return Ok(state.flush_and_export_full());
        }
        if let Some(ref mut pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Ok(pending.state.flush_and_export_full());
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export all states in a stash as snapshots.
    pub fn export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.sm.get(stash)
            .map(|s| s.iter().map(|state| state.export_full()).collect())
            .unwrap_or_default()
    }

    /// Export all found states as snapshots (flushing pending writes).
    pub fn export_found_states_flushed(&mut self) -> Vec<crate::state::ExplorationStateSnapshot> {
        if let Some(states) = self.sm.get_mut(STASH_FOUND) {
            states.iter_mut().map(|s| s.flush_and_export_full()).collect()
        } else {
            Vec::new()
        }
    }

    /// Export all found states as snapshots.
    pub fn export_found_states(&self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.export_stash(STASH_FOUND)
    }

    /// Evaluate a symbolic value in a state's solver context.
    ///
    /// This allows Python to get concrete values for symbolic inputs
    /// that were found during exploration.
    pub fn eval_in_state(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        self.with_state(state_id, |state| {
            match state.memory_load(addr, size) {
                Ok(bv) => {
                    if let Some(val) = state.eval(&bv) {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    Ok(None)
                }
                Err(_) => Ok(None),
            }
        })
    }

    /// Debug: Get symbolic object info for a state.
    pub fn state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        self.with_state(state_id, |state| {
            let mem = state.memory();
            let total = mem.symbolic_object_count();
            let has_at_addr = mem.get_symbolic_object(addr).map(|bv| bv.width());
            let page_num = addr >> 12;
            let offset = (addr & 0xFFF) as u16;
            let page_info = if let Some(page) = mem.pages().get(&page_num) {
                format!("page=mapped sym_at_offset={}", page.is_symbolic(offset))
            } else {
                "page=unmapped".to_string()
            };
            Ok(format!(
                "total_sym_objs={} at_0x{:x}={:?} {}",
                total, addr, has_at_addr, page_info
            ))
        })
    }

    /// Get Z3 AST pointers for all symbolic objects in a state's memory.
    ///
    /// Returns Vec<(addr, z3_ast_ptr_as_usize, width_bits)> for each symbolic
    /// object. The Z3 ASTs are built in the shared Z3 context, so Python can
    /// directly wrap them as z3.BitVecRef and convert to claripy ASTs.
    ///
    /// This is used to export Rust-computed symbolic expressions (e.g., flag
    /// computations in asisctf) to Python state memory during state export.
    #[cfg(feature = "vex-engine-z3")]
    pub fn get_state_symbolic_z3_asts(&self, state_id: u64) -> PyResult<Vec<(u64, usize, u32)>> {
        use z3::ast::Ast;
        self.with_state(state_id, |state| {
            let mem = state.memory();
            let mut result = Vec::new();
            for (&addr, bv) in mem.symbolic_objects_iter() {
                // Only export Expression values (Rust-computed).
                // Skip Symbolic values (imported from Python) — Python already
                // has those with proper claripy identity.
                // Also skip addresses that were originally imported from Python,
                // even if the binary modified them (Symbolic→Expression).
                // Python's memory has the correct original value; overwriting
                // it would break post-exploration constraint solving (flareon5).
                if mem.is_imported_addr(addr) {
                    continue;
                }
                if matches!(bv, RustBV::Expression { .. }) {
                    let z3_ast = bv.to_z3_ast();
                    let raw_ptr = z3_ast.get_z3_ast().as_ptr() as usize;
                    // Prevent z3::ast::BV destructor from decrementing the ref count.
                    // Python takes ownership of this pointer.
                    std::mem::forget(z3_ast);
                    result.push((addr, raw_ptr, bv.width()));
                }
            }
            Ok(result)
        })
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.satisfiable()))
    }

    /// Whether strict memory permission enforcement is enabled on a state.
    /// Mirrors angr's STRICT_PAGE_ACCESS option.
    pub fn state_enforce_permissions(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.enforce_permissions()))
    }

    /// Get a register value from a state.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self.with_state(state_id, |state| {
            Ok(state.get_register(name).and_then(|bv| bv.as_u128()))
        })
    }

    /// Get multiple register values from a state in one FFI call.
    /// Returns a list of Option<u128> in the same order as the input names.
    pub fn get_state_registers_batch(&self, state_id: u64, names: Vec<String>) -> PyResult<Vec<Option<u128>>> {
        self.with_state(state_id, |state| {
            Ok(names.iter()
                .map(|name| state.get_register(name).and_then(|bv| bv.as_u128()))
                .collect())
        })
    }

    /// Get memory from a state.
    pub fn get_state_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        self.with_state(state_id, |state| {
            match state.memory_load(addr, size) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u128() {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    if let Some(val) = state.eval(&bv) {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    Ok(None)
                }
                Err(_) => Ok(None),
            }
        })
    }

    /// Check if a state has stdout output (dirty flag check, no allocation).
    pub fn has_state_stdout(&self, state_id: u64) -> bool {
        // find_state already checks pending_callback first.
        self.find_state(state_id).map_or(false, |s| s.has_stdout())
    }

    /// Get the stdout buffer for a state by ID.
    ///
    /// Returns the accumulated output from native puts/printf calls.
    pub fn get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self.get_state_fd_output(state_id, 1)
    }

    /// Get the output buffer for a specific file descriptor.
    ///
    /// Returns the accumulated output from native write/puts/printf calls.
    pub fn get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self.with_state(state_id, |state| Ok(state.fd_buffer(fd).to_vec()))
    }

    /// Check if a state has recorded stdin symbols from native fgets/fgetc/getchar.
    pub fn has_state_stdin_symbols(&self, state_id: u64) -> bool {
        self.find_state(state_id).map_or(false, |s| s.has_stdin_symbols())
    }

    /// Get the stdin symbols for a state by ID.
    ///
    /// Returns list of (name, bit_width) tuples for symbolic variables
    /// created by native fgets/fgetc/getchar. Used to reconstruct stdin
    /// data in Python's posix plugin for posix.dumps(0).
    pub fn get_state_stdin_symbols(&self, state_id: u64) -> PyResult<Vec<(String, u32)>> {
        self.with_state(state_id, |state| Ok(state.stdin_symbols().to_vec()))
    }

    /// Get the call stack for a state by ID.
    ///
    /// Returns list of (call_site_addr, callee_addr, return_addr, stack_ptr) tuples.
    pub fn get_state_call_stack(&self, state_id: u64) -> PyResult<Vec<(u64, u64, u64, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.call_stack().iter()
                .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
                .collect())
        })
    }

    /// Get the call stack depth for a state by ID.
    pub fn get_state_call_stack_depth(&self, state_id: u64) -> PyResult<usize> {
        self.with_state(state_id, |state| Ok(state.call_stack_depth()))
    }

    /// Get the detailed execution history for a state by ID.
    ///
    /// Returns list of (addr, jumpkind, jump_target) tuples.
    /// jumpkind: 0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other
    pub fn get_state_detailed_history(&self, state_id: u64) -> PyResult<Vec<(u64, u8, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.detailed_history().iter()
                .map(|e| (e.addr, e.jumpkind, e.jump_target))
                .collect())
        })
    }

    /// Get heap metadata for a state by ID.
    ///
    /// Returns dict with:
    /// - allocated: list of (addr, size) tuples for active allocations
    /// - freed: list of freed addresses
    /// - alloc_count: number of active allocations
    /// - free_count: number of free calls
    pub fn get_state_heap_metadata(&self, state_id: u64) -> PyResult<(Vec<(u64, u64)>, Vec<u64>)> {
        self.with_state(state_id, |state| {
            let meta = state.heap_metadata();
            let allocated: Vec<(u64, u64)> = meta.allocated.iter()
                .map(|(&addr, &size)| (addr, size))
                .collect();
            let freed = meta.freed.clone();
            Ok((allocated, freed))
        })
    }

    /// Get the list of open file descriptors for a state.
    ///
    /// Returns list of (fd, name, position, flags, content_len, is_open) tuples.
    pub fn get_state_open_fds(&self, state_id: u64) -> PyResult<Vec<(u32, String, u64, u32, usize, bool)>> {
        self.with_state(state_id, |state| {
            Ok(state.file_system_ref().all_fds().iter().filter_map(|&fd| {
                let info = state.file_system_ref().fd_info(fd)?;
                Some((fd, info.0.to_string(), info.1, info.2, info.3, info.4))
            }).collect())
        })
    }

    /// Get the content of a file descriptor for a state.
    pub fn get_state_fd_content(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self.with_state(state_id, |state| {
            Ok(state.file_system_ref().fd_content(fd).to_vec())
        })
    }

    /// Enable inspection for an event type on a state.
    ///
    /// event_type: 0=MemRead, 1=MemWrite, 2=RegRead, 3=RegWrite, 4=Fork, 5=Exit
    pub fn enable_state_inspection(&mut self, state_id: u64, event_type: u8) -> PyResult<()> {
        let event = crate::state::InspectEvent::from_u8(event_type)
            .ok_or_else(|| PyValueError::new_err(format!("invalid event type: {}", event_type)))?;
        self.with_state_mut(state_id, |state| {
            state.inspection_mut().enable(event);
            Ok(())
        })
    }

    /// Enable all inspections on a state.
    pub fn enable_all_inspections(&mut self, state_id: u64) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            state.inspection_mut().enable_all();
            Ok(())
        })
    }

    /// Get inspection event counts for a state.
    ///
    /// Returns list of (event_name, count) tuples for events with count > 0.
    pub fn get_state_inspection_counts(&self, state_id: u64) -> PyResult<Vec<(String, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.inspection().event_counts().iter().enumerate()
                .filter(|&(_, &count)| count > 0)
                .filter_map(|(i, &count)| {
                    let event = crate::state::InspectEvent::from_u8(i as u8)?;
                    Some((event.name().to_string(), count))
                })
                .collect())
        })
    }

    /// Get inspection events for a state.
    ///
    /// Returns list of (event_type, event_name, addr, size, block_addr) tuples.
    pub fn get_state_inspection_events(&self, state_id: u64) -> PyResult<Vec<(u8, String, u64, u32, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.inspection().events().iter().map(|e| {
                (e.event as u8, e.event.name().to_string(), e.addr, e.size, e.block_addr)
            }).collect())
        })
    }

    /// Evaluate a stdin symbol by name using the state's solver.
    ///
    /// Returns the concrete value as Option<u64>, or None if the symbol
    /// cannot be found or evaluated.
    pub fn eval_stdin_symbol(&self, state_id: u64, name: &str) -> Option<u64> {
        let state = self.find_state(state_id)?;
        let ctx = state.solver().borrow();
        // Find the symbol by name in the solver context
        let sym = crate::symbolic::RustBV::symbolic(&ctx, name, 8);
        ctx.eval(&sym).map(|v| v as u64)
    }

    // =========================================================================
    // Run loop (from run_loop.rs)
    // =========================================================================

    /// Run the exploration loop.
    ///
    /// Returns an ExplorationEvent when:
    /// - Found enough solutions
    /// - Active stash is empty
    /// - Need Python callback (SimProcedure, syscall)
    /// - Max steps reached
    #[pyo3(signature = (n=None))]
    pub fn run(&mut self, py: Python<'_>, n: Option<u32>) -> PyResult<ExplorationEvent> {
        let max_steps = n.unwrap_or(self.max_steps_per_run);

        // Ensure callbacks are set and clone to avoid borrow issues
        let callbacks = self.callbacks.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("callbacks not set")
        })?.clone();

        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err("callbacks not ready"));
        }

        let run_loop_start = if self.profiling.profiling_enabled { Some(std::time::Instant::now()) } else { None };

        for _ in 0..max_steps {
            // Check if we have enough solutions
            if self.found_count() >= self.num_find {
                return Ok(ExplorationEvent::found(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }

            // Get next state from active stash
            // P9 fix: Use LIFO (pop_back) for DFS or FIFO (pop_front) for BFS
            let mut state = match self.sm.get_mut(STASH_ACTIVE).and_then(|s| {
                if self.use_lifo {
                    s.pop_back()  // DFS: LIFO (most recent state first)
                } else {
                    s.pop_front()  // BFS: FIFO (oldest state first)
                }
            }) {
                Some(s) => {
                    let sid = s.state_id();
                    self.current_stepping_state_id = Some(sid);
                    STEPPING_STATE_ID.with(|cell| cell.set(Some(sid)));
                    s
                },
                None => {
                    // No active states
                    if self.found_count() > 0 {
                        return Ok(ExplorationEvent::found(
                            self.found_count(),
                            0,
                            self.steps,
                        ));
                    } else {
                        return Ok(ExplorationEvent::active_empty(
                            self.found_count(),
                            self.steps,
                        ));
                    }
                }
            };

            // Check find/avoid before stepping
            let pc = state.pc();

            // P7 fix: Check if callable avoid predicate needs Python evaluation
            // When avoid is a callable (lambda/function), we must return to Python
            // to evaluate it for each state, not just check addresses.
            // Skip if this state was just checked (resume_avoid_predicate(false)
            // sets skip_avoid_predicate_states to prevent infinite loop).
            if self.avoid_needs_python {
                let state_id = state.state_id();
                if self.constraint_tracker.skip_avoid_predicate_states.remove(&state_id) {
                    // Fall through — predicate already checked at this PC
                } else {
                self.pending_callback = Some(PendingCallback::lightweight(
                    state,
                    CallbackReason::AvoidPredicate { addr: pc },
                ));

                return Ok(ExplorationEvent {
                    event_type: "need_callback".to_string(),
                    callback_reason: Some("avoid_predicate".to_string()),
                    callback_addr: Some(pc),
                    callback_state_id: Some(state_id),
                    found_count: self.found_count(),
                    active_count: self.active_count(),
                    steps_taken: self.steps,
                    callback_name: None,
                    callback_syscall_num: None,
                    callback_return_addr: None,
                    callback_num_args: None,
                    branch_true_target: None,
                    branch_false_target: None,
                    branch_condition_id: None,
                });
                }
            }

            // Check avoid addresses (address-based, only when NOT using callable predicate)
            if self.avoid_addrs.contains(&pc) {
                self.push_or_drop_terminal(STASH_AVOID, state);
                continue;
            }

            // P2 fix: Check if callable find predicate needs Python evaluation
            // When find is a callable (lambda/function), we must return to Python
            // to evaluate it for each state, not just check addresses.
            // Skip if this state was just checked (resume_find_predicate(false)
            // sets skip_find_predicate_state to avoid infinite loop).
            if self.find_needs_python {
                let state_id = state.state_id();
                if self.constraint_tracker.skip_find_predicate_states.remove(&state_id) {
                    // Fall through to hooks/stepping — predicate already checked
                } else {
                self.pending_callback = Some(PendingCallback::lightweight(
                    state,
                    CallbackReason::FindPredicate { addr: pc },
                ));

                return Ok(ExplorationEvent {
                    event_type: "need_callback".to_string(),
                    callback_reason: Some("find_predicate".to_string()),
                    callback_addr: Some(pc),
                    callback_state_id: Some(state_id),
                    found_count: self.found_count(),
                    active_count: self.active_count(),
                    steps_taken: self.steps,
                    callback_name: None,
                    callback_syscall_num: None,
                    callback_return_addr: None,
                    callback_num_args: None,
                    branch_true_target: None,
                    branch_false_target: None,
                    branch_condition_id: None,
                });
                } // else (not skip_find_predicate_state)
            }

            // Check find addresses (address-based, only when NOT using callable predicate)
            if self.find_addrs.contains(&pc) {
                // Only add to found if the state is satisfiable
                // (UNSAT states reached the address via infeasible paths)
                if self.constraint_solver.lazy_solves || state.satisfiable() {
                    self.sm.stashes_mut()
                        .entry(STASH_FOUND.to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                } else {
                    log::debug!("State at find address 0x{:x} is UNSAT, pruning", pc);
                    self.push_or_drop_terminal(STASH_PRUNED, state);
                }
                continue;
            }

            // Check hooks (SimProcedures)
            // GAP 6: Stack-based skip tracking for zero-length hooks
            // Clean up expired skip entries before checking
            self.skip_hook_stack.retain(|&(_, expiry)| expiry > self.steps);

            // Check if this address is in the skip stack
            let should_skip_hook = self.skip_hook_stack.iter().any(|&(addr, _)| addr == pc);
            if should_skip_hook {
                // Remove this address from the skip stack (consumed)
                self.skip_hook_stack.retain(|&(addr, _)| addr != pc);
                log::debug!("Skipping hook at 0x{:x} (zero-length hook, step {})", pc, self.steps);
            }
            if self.hooks.contains(&pc) && !should_skip_hook {
                // Check if this is a registered SimProcedure
                if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                    // Skip native for addresses inside the binary (user-placed hooks)
                    let is_in_binary = self.environment.binary_regions.iter().any(|(base, data)| {
                        pc >= *base && pc < *base + data.len() as u64
                    });
                    // Try native procedure first (only for external/library hooks)
                    if !is_in_binary {
                    if let Some(native_proc) = self.native_procedures.get(&name) {
                        // Extract arguments from state registers and stack
                        let args = self.extract_procedure_args(&state, num_args);

                        // Try to execute native procedure
                        match native_proc.call(&mut state, &args) {
                            Ok(ret_val) => {
                                // Native execution succeeded
                                self.profiling.native_proc_stats.native_calls += 1;
                                *self.profiling.native_proc_stats.call_counts
                                    .entry(name.clone())
                                    .or_insert(0) += 1;

                                // For no-return procedures (exit/abort), skip
                                // the return-address dance and deadend directly.
                                // Setting PC to a stack-derived return address
                                // can produce a spurious successor (e.g. when
                                // exit is called from rejected() in fauxware,
                                // the post-call address happens to overlap
                                // main's start, causing infinite re-entry).
                                if no_return {
                                    self.push_or_drop_terminal(STASH_DEADENDED, state);
                                    continue;
                                }

                                // Set return value if present
                                if let Some(rv) = ret_val {
                                    let ret_reg = self.environment.calling_convention.return_register();
                                    state.set_register_by_offset(ret_reg, rv);
                                }

                                // Get return address and set PC
                                let ctx = state.solver().borrow();
                                let ret_addr_opt = crate::arch::arch_from_name(&self.environment.arch_name)
                                    .and_then(|arch| {
                                        self.environment.calling_convention.get_return_addr(
                                            &crate::arch::RegisterFile::new(arch),
                                            None,
                                            &ctx,
                                        )
                                    });
                                if let Some(ret_addr) = ret_addr_opt {
                                    drop(ctx);
                                    // Pop return address from stack
                                    let sp = state.get_sp().as_u64().unwrap_or(0);
                                    let ptr_size = state.arch().bytes() as u64;
                                    state.set_sp(RustBV::concrete((sp + ptr_size) as u128, state.arch().bits()));
                                    state.set_pc(ret_addr);
                                } else {
                                    drop(ctx);
                                    // Fallback: try to get return address from stack
                                    if let Some(sp) = state.get_sp().as_u64() {
                                        if let Ok(ret_bv) = state.memory_load(sp, state.arch().bytes()) {
                                            if let Some(ret_addr) = ret_bv.as_u64() {
                                                let ptr_size = state.arch().bytes() as u64;
                                                state.set_sp(RustBV::concrete((sp + ptr_size) as u128, state.arch().bits()));
                                                state.set_pc(ret_addr);
                                            }
                                        }
                                    }
                                }

                                self.push_to_active_or_drop(state);
                                continue;
                            }
                            Err(_e) => {
                                // Native execution failed, fall back to Python
                                self.profiling.native_proc_stats.python_fallbacks += 1;
                            }
                        }
                    }
                    } // if !is_in_binary

                    // Fall back to Python for SimProcedure execution
                    self.simprocedure_python_fallback_count += 1;
                    let state_id = state.state_id();
                    let return_addr = self.get_return_addr(&state).unwrap_or(0);

                    // No deferred forks in run-loop path, so pre_callback_snapshot
                    // is unnecessary (it's only used as fork base for deferred forks).
                    // Use shared solver (O(1) Rc clone) instead of fork (~3-40ms Z3 clone).
                    let solver_ref = state.solver();
                    let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());

                    self.pending_callback = Some(PendingCallback::with_context(
                        state,
                        None,
                        CallbackReason::SimProcedure {
                            addr: pc,
                            name: name.clone(),
                            num_args,
                            return_addr,
                        },
                        "Ijk_Call",
                        Some(shared_ctx),
                        Vec::new(),
                        FxHashMap::default(),
                        FxHashMap::default(),
                    ));

                    return Ok(ExplorationEvent::need_simprocedure(
                        state_id,
                        pc,
                        name,
                        num_args,
                        return_addr,
                        self.found_count(),
                        self.active_count(),
                        self.steps,
                    ));
                }
            }

            // Step the state, passing the skip_hook_addr if we just skipped
            let skip_addr_for_step = if should_skip_hook { Some(pc) } else { None };
            match self.step_state_with_skip(py, &callbacks, state, skip_addr_for_step) {
                Ok(successors) => {
                    // Add successors to appropriate stashes, checking find/avoid
                    for successor in successors {
                        let spc = successor.pc();
                        if self.find_addrs.contains(&spc) {
                            if self.constraint_solver.lazy_solves || successor.satisfiable() {
                                self.sm.stashes_mut().entry(STASH_FOUND.to_string())
                                    .or_insert_with(VecDeque::new).push_back(successor);
                            }
                        } else if self.avoid_addrs.contains(&spc) {
                            self.push_or_drop_terminal(STASH_AVOID, successor);
                        } else {
                            self.push_to_active_or_drop(successor);
                        }
                    }
                }
                Err(StepError::NeedCallback(pending)) => {
                    // Check if the callback address is a find/avoid address
                    // (these were added as interpreter hooks to stop execution)
                    let callback_addr = match &pending.reason {
                        CallbackReason::SimProcedure { addr, .. } => Some(*addr),
                        _ => None,
                    };
                    if let Some(addr) = callback_addr {
                        if self.find_addrs.contains(&addr) || self.avoid_addrs.contains(&addr) {
                            let is_find = self.find_addrs.contains(&addr);

                            // Process deferred forks BEFORE handling the find/avoid state.
                            // These represent unexplored branches that diverged before
                            // reaching the find/avoid address and must not be dropped.
                            let fork_base = pending.pre_callback_snapshot.unwrap_or_else(|| pending.state.fork());
                            let original_state_id = pending.state.state_id();
                            let root_state_id = self.sm.roots().get(&original_state_id).copied().unwrap_or(original_state_id);

                            let mut snapshots = pending.fork_snapshots;
                            let cb_fork_start = if self.profiling.profiling_enabled { Some(std::time::Instant::now()) } else { None };
                            let cb_fork_total = pending.deferred_forks.len() as u64;
                            for fork in pending.deferred_forks {
                                let condition = pending.stored_conditions.get(&fork.condition_id);
                                let reconstructed = if condition.is_none() {
                                    if let Some(ref py_ast) = fork.condition_ast {
                                        Python::attach(|py| {
                                            let ast = py_ast.bind(py);
                                            let solver_ref = fork_base.solver();
                                            let ctx: &SymContext = &*solver_ref.borrow();
                                            claripy_to_rustbv(py, ast, ctx).ok()
                                        })
                                    } else { None }
                                } else { None };

                                if let Some(cond) = condition.or(reconstructed.as_ref()) {
                                    // Add the taken-path constraint to the main state
                                    // (mirrors the BlockEnd handling at line 3560-3564).
                                    if fork.path_taken {
                                        pending.state.solver().borrow().assume_true(cond);
                                    } else {
                                        pending.state.solver().borrow().assume_false(cond);
                                    }
                                    let fork_op_start = if self.profiling.profiling_enabled { Some(std::time::Instant::now()) } else { None };
                                    let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                                        let mut f = fork_base.fork_from_snapshot(snapshot);
                                        if fork.path_taken {
                                            f.solver().borrow().assume_false(cond);
                                        } else {
                                            f.solver().borrow().assume_true(cond);
                                        }
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    } else if fork.path_taken {
                                        let mut f = fork_base.fork_false(cond);
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    } else {
                                        let mut f = fork_base.fork_true(cond);
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    };
                                    if let Some(start) = fork_op_start {
                                        self.profiling.accumulated_stats.solver_fork_time_ns += start.elapsed().as_nanos() as u64;
                                        self.profiling.accumulated_stats.solver_fork_count += 1;
                                    }
                                    self.sm.set_root(forked.state_id(), root_state_id);
                                    let sat_start = if self.profiling.profiling_enabled { Some(std::time::Instant::now()) } else { None };
                                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                                        if let Some(start) = sat_start {
                                            self.profiling.accumulated_stats.solver_sat_time_ns += start.elapsed().as_nanos() as u64;
                                            self.profiling.accumulated_stats.solver_sat_count += 1;
                                        }
                                        self.push_to_active_or_drop(forked);
                                    } else if let Some(start) = sat_start {
                                        self.profiling.accumulated_stats.solver_sat_time_ns += start.elapsed().as_nanos() as u64;
                                        self.profiling.accumulated_stats.solver_sat_count += 1;
                                    }
                                }
                            }
                            if let Some(start) = cb_fork_start {
                                self.profiling.accumulated_stats.deferred_fork_time_ns += start.elapsed().as_nanos() as u64;
                                self.profiling.accumulated_stats.deferred_fork_count += cb_fork_total;
                            }

                            // Now handle the main state
                            if is_find {
                                if self.constraint_solver.lazy_solves || pending.state.satisfiable() {
                                    self.sm.stashes_mut().entry(STASH_FOUND.to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(pending.state);
                                } else {
                                    log::debug!("State at find address 0x{:x} is UNSAT, pruning", addr);
                                    self.push_or_drop_terminal(STASH_PRUNED, pending.state);
                                }
                            } else {
                                self.push_or_drop_terminal(STASH_AVOID, pending.state);
                            }
                            continue;
                        }
                    }

                    // Need Python callback
                    let state_id = pending.state.state_id();
                    let event = match &pending.reason {
                        CallbackReason::SimProcedure { addr, name, num_args, return_addr } => {
                            ExplorationEvent::need_simprocedure(
                                state_id,
                                *addr,
                                name.clone(),
                                *num_args,
                                *return_addr,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::Syscall { num } => {
                            ExplorationEvent::need_syscall(
                                state_id,
                                *num,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::SymbolicBranch { condition_id, true_target, false_target } => {
                            ExplorationEvent::need_symbolic_branch(
                                state_id,
                                *condition_id,
                                *true_target,
                                *false_target,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::PythonVEXFallback { addr, reason } => {
                            self.vex_fallback_count += 1;
                            self.vex_fallback_addrs.entry(*addr).or_insert_with(|| reason.clone());
                            if reason.contains(DCAS_UNSUPPORTED_REASON) {
                                self.dcas_unsupported_count += 1;
                                if self.dcas_warned_states.insert(state_id) {
                                    log::warn!(
                                        "DCAS (cmpxchg16b) unsupported in Rust interpreter at \
                                         0x{:x} (state {}); falling back to Python VEX engine",
                                        addr, state_id
                                    );
                                }
                            }
                            ExplorationEvent::need_python_vex(
                                state_id,
                                *addr,
                                reason,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::Error { message } => {
                            ExplorationEvent::error(
                                message.clone(),
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        _ => {
                            ExplorationEvent::error(
                                "unhandled callback reason".to_string(),
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                    };

                    self.pending_callback = Some(pending);
                    return Ok(event);
                }
                Err(StepError::Deadended(state)) => {
                    self.push_or_drop_terminal(STASH_DEADENDED, state);
                }
                Err(StepError::Error(state, message)) => {
                    let pc = state.pc();
                    let state_id = state.state_id();
                    self.errors.push((pc, message, state_id));
                    self.sm.stashes_mut()
                        .entry(STASH_ERRORED.to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
                Err(StepError::Unconstrained(state)) => {
                    // State has too many symbolic jump targets - move to unconstrained stash
                    log::debug!("State {} moved to unconstrained stash", state.state_id());
                    self.sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
                }
            }

            self.steps += 1;

            // Apply native uniqueness filter if enabled
            self.apply_uniqueness_filter();
            // Apply native techniques (LengthLimiter, Timeout, LoopBound)
            self.apply_native_techniques();
        }

        // Record run loop timing and active state count
        if let Some(start) = run_loop_start {
            self.profiling.accumulated_stats.run_loop_time_ns += start.elapsed().as_nanos() as u64;
            self.profiling.accumulated_stats.active_states_count = self.active_count() as u64;
        }

        // Max steps reached
        Ok(ExplorationEvent::step_complete(
            self.found_count(),
            self.active_count(),
            self.steps,
        ))
    }

    // =========================================================================
    // Resume methods (from resume.rs)
    // =========================================================================

    /// Resume after a SimProcedure callback.
    ///
    /// This is called from Python after executing a SimProcedure.
    /// The state changes (registers, memory, new PC) are applied.
    ///
    /// Args:
    ///     new_pc: The new program counter after the SimProcedure.
    ///     register_changes: List of (offset, size, data) tuples for register changes.
    ///     memory_changes: List of (addr, data) tuples for memory changes.
    ///     new_constraints: Optional list of claripy ASTs to add as constraints.
    ///         These are constraints added by the SimProcedure (e.g., strcmp results).
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        let pending = self.pending_callback.take().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state")
        })?;

        // Apply changes
        let mut changes = StateChanges::new();
        changes.new_pc = Some(new_pc);

        if let Some(reg_changes) = register_changes {
            changes.register_writes = reg_changes;
        }

        if let Some(mem_changes) = memory_changes {
            changes.memory_writes = mem_changes;
        }

        let mut state = pending.state;
        state.apply_changes(&changes);
        state.set_pc(new_pc);

        // Sync constraints from Python back to Rust
        // This ensures constraints added by SimProcedures (e.g., strcmp return conditions)
        // are properly reflected in the Rust solver state
        //
        // P12: Track if state becomes UNSAT after constraint sync
        let mut main_state_unsat = false;
        if let Some(constraints) = new_constraints {
            let is_sat = self.sync_constraints_from_python(py, &state, constraints)?;
            if !is_sat {
                log::debug!(
                    "P12: State {} became UNSAT after constraint sync in resume_after_simprocedure.",
                    state.state_id()
                );
                main_state_unsat = true;
            }
        }

        // Validate deferred forks reference valid conditions before processing
        let mut missing_conditions = 0usize;
        for fork in &pending.deferred_forks {
            if !pending.stored_conditions.contains_key(&fork.condition_id) {
                missing_conditions += 1;
                log::warn!(
                    "Deferred fork at 0x{:x} references missing condition_id={}",
                    fork.branch_addr, fork.condition_id
                );
            }
        }
        if missing_conditions > 0 {
            log::warn!(
                "{} of {} deferred forks have missing conditions - will be skipped",
                missing_conditions, pending.deferred_forks.len()
            );
        }

        // Add taken-path constraints from deferred forks to the main state.
        // Without these, the solver doesn't know which branch was taken,
        // causing incorrect results for subsequent symbolic operations.
        for fork in &pending.deferred_forks {
            if let Some(cond) = pending.stored_conditions.get(&fork.condition_id) {
                if fork.path_taken {
                    state.solver().borrow().assume_true(cond);
                } else {
                    state.solver().borrow().assume_false(cond);
                }
            }
        }

        // Process deferred forks that were stored during the step
        // These represent unexplored branches that should be added to active
        //
        // CRITICAL: Deferred forks diverged BEFORE the callback, so they should
        // NOT inherit callback constraints. Use pre_callback_snapshot as fork base.
        // Only create fork_base when there are deferred forks — state.fork() costs ~3ms
        // due to Z3 solver clone, and most callbacks have zero deferred forks.
        let has_deferred_forks = !pending.deferred_forks.is_empty();
        let fork_base = if has_deferred_forks {
            Some(pending.pre_callback_snapshot.unwrap_or_else(|| state.fork()))
        } else {
            drop(pending.pre_callback_snapshot); // explicitly drop unused snapshot
            None
        };

        // Track root state ID for lineage
        // The root is inherited from the original pending state
        let original_state_id = state.state_id();
        let root_state_id = self.sm.roots().get(&original_state_id).copied().unwrap_or(original_state_id);

        // P12: Only add main state if SAT, otherwise add to pruned list
        let (mut successors, mut pruned_states) = if main_state_unsat {
            (Vec::new(), vec![state])
        } else {
            (vec![state], Vec::new())
        };
        let mut snapshots = pending.fork_snapshots;
        for fork in pending.deferred_forks {
            // Look up the condition for this deferred fork
            let condition = pending.stored_conditions.get(&fork.condition_id);

            // P11 fix: If condition not in stored_conditions, try to reconstruct from condition_ast
            let reconstructed_condition = if condition.is_none() {
                if let Some(ref py_ast) = fork.condition_ast {
                    // Try to convert the claripy AST to RustBV
                    Python::attach(|py| {
                        let ast = py_ast.bind(py);
                        let fb = fork_base.as_ref().expect("fork_base set before deferred fork processing");
                        let solver_ref = fb.solver();
                        let ctx: &SymContext = &*solver_ref.borrow();
                        claripy_to_rustbv(py, ast, ctx).ok()
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let effective_condition = condition.or(reconstructed_condition.as_ref());

            if let Some(cond) = effective_condition {
                // Use solver snapshot (from before branch constraint) if available
                let fb = fork_base.as_ref().expect("fork_base set before deferred fork processing");
                let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                    let mut f = fb.fork_from_snapshot(snapshot);
                    if fork.path_taken {
                        f.solver().borrow().assume_false(cond);
                    } else {
                        f.solver().borrow().assume_true(cond);
                    }
                    f.set_pc(fork.unexplored_target);
                    f
                } else if fork.path_taken {
                    let mut f = fb.fork_false(cond);
                    f.set_pc(fork.unexplored_target);
                    f
                } else {
                    let mut f = fb.fork_true(cond);
                    f.set_pc(fork.unexplored_target);
                    f
                };

                // Track root state ID for this forked state
                self.sm.set_root(forked.state_id(), root_state_id);

                // DO NOT sync callback constraints to forked state!
                // These paths diverged before the callback occurred.
                // Adding callback constraints would pollute unexplored branches.

                // P13: Check satisfiability before adding to successors
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                    if reconstructed_condition.is_some() {
                        log::debug!(
                            "P11: Reconstructed condition from condition_ast for fork at 0x{:x}",
                            fork.branch_addr
                        );
                    }
                } else {
                    log::debug!(
                        "P13: Forked state at 0x{:x} is UNSAT, adding to pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            } else {
                // P15: Better handling of missing deferred fork conditions
                // Try harder to get a condition or create a fresh boolean to explore both paths
                log::warn!(
                    "P15: Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                     Creating conservative fork to explore the path.",
                    fork.branch_addr,
                    fork.condition_id
                );
                // Create a fork without additional constraints - this is conservative
                // but ensures we don't lose valid paths
                let mut forked = fork_base.as_ref().expect("fork_base set before fork").fork();
                forked.set_pc(fork.unexplored_target);
                self.sm.set_root(forked.state_id(), root_state_id);

                // P13: Still check satisfiability
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    log::debug!(
                        "P13: Unconstrained fork at 0x{:x} is UNSAT, adding to pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            }
        }

        // Add all successors (original state + forks) to stashes
        // P13: Check satisfiability for each before adding
        // Note: We split the loops to avoid double mutable borrow of self.sm
        let mut final_successors = Vec::new();
        for successor in successors {
            if self.constraint_solver.lazy_solves || successor.satisfiable() {
                final_successors.push(successor);
            } else {
                log::debug!(
                    "P13: Successor state {} is UNSAT, moving to pruned stash",
                    successor.state_id()
                );
                pruned_states.push(successor);
            }
        }

        // Add to active stash, checking find/avoid first
        for s in final_successors {
            let spc = s.pc();
            if self.find_addrs.contains(&spc) {
                self.sm.stashes_mut().entry(STASH_FOUND.to_string())
                    .or_insert_with(VecDeque::new).push_back(s);
            } else if self.avoid_addrs.contains(&spc) {
                self.push_or_drop_terminal(STASH_AVOID, s);
            } else {
                self.push_to_active_or_drop(s);
            }
        }

        // Add to pruned stash
        for s in pruned_states {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        // Apply native uniqueness filter if enabled
        self.apply_uniqueness_filter();
        // Apply native techniques (LengthLimiter, Timeout, LoopBound)
        self.apply_native_techniques();

        Ok(())
    }

    /// Resume after a syscall callback.
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_syscall(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure - ensures constraint sync (GAP 2)
        self.resume_after_simprocedure(py, new_pc, register_changes, memory_changes, new_constraints)
    }

    /// Resume after a hook callback.
    ///
    /// This is equivalent to resume_after_simprocedure but with a more explicit name
    /// for hook-specific handling. Ensures constraints added during hook execution
    /// are properly synced back to Rust (GAP 2 fix).
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_hook(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure - ensures constraint sync (GAP 2)
        self.resume_after_simprocedure(py, new_pc, register_changes, memory_changes, new_constraints)
    }

    /// Resume after an error occurred during callback execution (P17).
    ///
    /// This moves the pending state to the errored stash instead of continuing
    /// with a corrupted state. This prevents "list index out of range" errors
    /// caused by UNSAT states proliferating from callback failures.
    /// Fast-path: deadend the pending callback state without full apply_changes.
    /// Used for SimProcedure continuations known to just call exit().
    /// Still processes deferred forks to avoid losing unexplored branches.
    pub fn deadend_pending_callback(&mut self) -> PyResult<()> {
        let pending = self.pending_callback.take().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state for deadend")
        })?;

        // Process deferred forks BEFORE deadending — these represent
        // unexplored branches that diverged before the exit/abort call.
        if !pending.deferred_forks.is_empty() {
            let fork_base = pending.pre_callback_snapshot.unwrap_or_else(|| pending.state.fork());
            let original_state_id = pending.state.state_id();
            let root_state_id = self.sm.roots().get(&original_state_id).copied().unwrap_or(original_state_id);

            let mut snapshots = pending.fork_snapshots;
            for fork in pending.deferred_forks {
                let condition = pending.stored_conditions.get(&fork.condition_id);
                if let Some(cond) = condition {
                    if fork.path_taken {
                        pending.state.solver().borrow().assume_true(cond);
                    } else {
                        pending.state.solver().borrow().assume_false(cond);
                    }
                    let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                        let mut f = fork_base.fork_from_snapshot(snapshot);
                        if fork.path_taken {
                            f.solver().borrow().assume_false(cond);
                        } else {
                            f.solver().borrow().assume_true(cond);
                        }
                        f.set_pc(fork.unexplored_target);
                        f
                    } else if fork.path_taken {
                        let mut f = fork_base.fork_false(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    } else {
                        let mut f = fork_base.fork_true(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    };
                    self.sm.set_root(forked.state_id(), root_state_id);
                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                        let spc = forked.pc();
                        if self.find_addrs.contains(&spc) {
                            self.sm.stashes_mut().entry(STASH_FOUND.to_string())
                                .or_insert_with(VecDeque::new).push_back(forked);
                        } else if self.avoid_addrs.contains(&spc) {
                            self.push_or_drop_terminal(STASH_AVOID, forked);
                        } else {
                            self.push_to_active_or_drop(forked);
                        }
                    }
                }
            }
        }

        self.push_or_drop_terminal(STASH_DEADENDED, pending.state);
        Ok(())
    }

    pub fn resume_after_error(&mut self, error_msg: &str) -> PyResult<()> {
        let pending = self.pending_callback.take().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state for error handling")
        })?;

        let pc = pending.state.pc();
        let state_id = pending.state.state_id();

        log::warn!(
            "P17: Moving state {} to errored stash after callback error at 0x{:x}: {}",
            state_id, pc, error_msg
        );

        // Record the error
        self.errors.push((pc, error_msg.to_string(), state_id));

        // Move to errored stash
        self.sm.stashes_mut()
            .entry(STASH_ERRORED.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(pending.state);

        Ok(())
    }

    /// Resume after Python handles a symbolic branch.
    ///
    /// Python creates forked states with proper constraints and passes them back
    /// to be added to the active stash.
    #[pyo3(signature = (true_pc, false_pc, true_constraints=None, false_constraints=None))]
    pub fn resume_after_symbolic_branch(
        &mut self,
        _py: Python<'_>,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // true_constraints and false_constraints are accepted for API compatibility but
        // the branch condition is sourced from stored_conditions (set by interpreter).
        let _ = (true_constraints, false_constraints);
        let pending = self.pending_callback.take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending symbolic branch callback"))?;

        // Get the branch condition from stored_conditions (set by interpreter)
        let branch_condition = match &pending.reason {
            CallbackReason::SymbolicBranch { condition_id, .. } => {
                pending.stored_conditions.get(condition_id).cloned()
            }
            _ => None,
        };

        // Add taken-path constraints from deferred forks to the main state
        // BEFORE forking for the symbolic branch. Since fork() creates an
        // independent solver copy, both true_state and false_state will
        // inherit these constraints. Without this, the solver wouldn't know
        // which deferred-fork branch was taken.
        for fork in &pending.deferred_forks {
            if let Some(cond) = pending.stored_conditions.get(&fork.condition_id) {
                if fork.path_taken {
                    pending.state.solver().borrow().assume_true(cond);
                } else {
                    pending.state.solver().borrow().assume_false(cond);
                }
            }
        }

        // Track root state ID for lineage
        let original_state_id = pending.state.state_id();
        let root_state_id = self.sm.roots().get(&original_state_id).copied().unwrap_or(original_state_id);

        // Create the true state (fork of original) and add constraint
        let mut true_state = pending.state.fork();
        self.sm.set_root(true_state.state_id(), root_state_id);
        true_state.set_pc(true_pc);
        if let Some(ref cond) = branch_condition {
            // Add assume_true: guard is true → exit taken
            let solver_ref = true_state.solver();
            solver_ref.borrow().assume_true(cond);
        }

        // Create the false state (use original) and add constraint
        let mut false_state = pending.state;
        false_state.set_pc(false_pc);
        if let Some(ref cond) = branch_condition {
            // Add assume_false: guard is false → fallthrough
            let solver_ref = false_state.solver();
            solver_ref.borrow().assume_false(cond);
        }

        // Process deferred forks that were accumulated before this symbolic branch.
        // These represent unexplored branches from earlier in the step that must
        // not be silently dropped.
        let mut deferred_successors = Vec::new();
        let mut deferred_pruned = Vec::new();
        if !pending.deferred_forks.is_empty() {
            let mut snapshots = pending.fork_snapshots;
            for fork in pending.deferred_forks {
                let condition = pending.stored_conditions.get(&fork.condition_id);

                // P11 fix: reconstruct from condition_ast if not in stored_conditions
                let reconstructed_condition = if condition.is_none() {
                    if let Some(ref py_ast) = fork.condition_ast {
                        Python::attach(|py| {
                            let ast = py_ast.bind(py);
                            let solver_ref = true_state.solver();
                            let ctx: &crate::symbolic::SymContext = &*solver_ref.borrow();
                            crate::claripy_bridge::claripy_to_rustbv(py, ast, ctx).ok()
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                let effective_condition = condition.or(reconstructed_condition.as_ref());

                if let Some(cond) = effective_condition {
                    // Use solver snapshot if available (from before branch constraint)
                    let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                        let mut f = true_state.fork_from_snapshot(snapshot);
                        if fork.path_taken {
                            f.solver().borrow().assume_false(cond);
                        } else {
                            f.solver().borrow().assume_true(cond);
                        }
                        f.set_pc(fork.unexplored_target);
                        f
                    } else if fork.path_taken {
                        let mut f = true_state.fork_false(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    } else {
                        let mut f = true_state.fork_true(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    };

                    self.sm.set_root(forked.state_id(), root_state_id);

                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                        deferred_successors.push(forked);
                    } else {
                        log::debug!(
                            "Deferred fork at 0x{:x} is UNSAT, adding to pruned",
                            fork.unexplored_target
                        );
                        deferred_pruned.push(forked);
                    }
                } else {
                    // Missing condition: create conservative fork
                    log::warn!(
                        "Missing condition for deferred fork at 0x{:x} in symbolic branch handler",
                        fork.branch_addr
                    );
                    let mut forked = true_state.fork();
                    forked.set_pc(fork.unexplored_target);
                    self.sm.set_root(forked.state_id(), root_state_id);
                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                        deferred_successors.push(forked);
                    } else {
                        deferred_pruned.push(forked);
                    }
                }
            }
        }

        // Add states to stashes.
        // When branch_condition is present, the interpreter already proved
        // both paths feasible via can_be_true/can_be_false. Prime the sat
        // cache so downstream satisfiable() checks are free (cache hits)
        // instead of doing redundant Z3 check() calls.
        let mut active_states = Vec::new();
        let mut pruned_states = Vec::new();

        if branch_condition.is_some() {
            true_state.set_sat_cache(true);
            false_state.set_sat_cache(true);
            // Check find/avoid on new states before adding to active
            if self.find_addrs.contains(&true_pc) {
                self.sm.stashes_mut().entry(STASH_FOUND.to_string()).or_insert_with(VecDeque::new).push_back(true_state);
            } else if self.avoid_addrs.contains(&true_pc) {
                self.push_or_drop_terminal(STASH_AVOID, true_state);
            } else {
                active_states.push(true_state);
            }
            if self.find_addrs.contains(&false_pc) {
                self.sm.stashes_mut().entry(STASH_FOUND.to_string()).or_insert_with(VecDeque::new).push_back(false_state);
            } else if self.avoid_addrs.contains(&false_pc) {
                self.push_or_drop_terminal(STASH_AVOID, false_state);
            } else {
                active_states.push(false_state);
            }
        } else {
            // Fallback: no stored condition, need actual sat checks
            if self.constraint_solver.lazy_solves || true_state.satisfiable() {
                active_states.push(true_state);
            } else {
                pruned_states.push(true_state);
            }
            if self.constraint_solver.lazy_solves || false_state.satisfiable() {
                active_states.push(false_state);
            } else {
                pruned_states.push(false_state);
            }
        }

        // Add deferred fork states, checking find/avoid
        for s in deferred_successors {
            let spc = s.pc();
            if self.find_addrs.contains(&spc) {
                self.sm.stashes_mut().entry(STASH_FOUND.to_string())
                    .or_insert_with(VecDeque::new).push_back(s);
            } else if self.avoid_addrs.contains(&spc) {
                self.push_or_drop_terminal(STASH_AVOID, s);
            } else {
                active_states.push(s);
            }
        }

        // Add to active stash, respecting max_active_states.
        for s in active_states {
            self.push_to_active_or_drop(s);
        }

        // Add to pruned stash
        pruned_states.extend(deferred_pruned);
        for s in pruned_states {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        log::debug!("Resumed after symbolic branch: true_pc=0x{:x}, false_pc=0x{:x}",
                    true_pc, false_pc);

        // Apply native uniqueness filter if enabled
        self.apply_uniqueness_filter();
        // Apply native techniques (LengthLimiter, Timeout, LoopBound)
        self.apply_native_techniques();

        Ok(())
    }

    /// Resume after Python evaluates a find predicate.
    ///
    /// P2 fix: This is called after Python evaluates a callable find predicate.
    /// If matched=true, the state is moved to found stash; otherwise, it continues
    /// exploration in the active stash.
    pub fn resume_find_predicate(&mut self, matched: bool) -> PyResult<()> {
        let pending = self.pending_callback.take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending find predicate callback"))?;

        if matched {
            log::debug!("Find predicate matched - moving state to found stash");
            let state_id = pending.state.state_id();
            self.sm.stashes_mut()
                .entry(STASH_FOUND.to_string())
                .or_insert_with(VecDeque::new)
                .push_back(pending.state);
            self.sm.index(state_id, STASH_FOUND);
        } else {
            log::debug!("Find predicate did not match - continuing exploration");
            // Mark this state to skip the find predicate check on next pop,
            // preventing infinite loop (state was already checked at this PC).
            let state_id = pending.state.state_id();
            self.constraint_tracker.skip_find_predicate_states.insert(state_id);
            self.push_to_active_or_drop(pending.state);
        }

        Ok(())
    }

    /// Resume after Python evaluates an avoid predicate.
    ///
    /// P7 fix: This is called after Python evaluates a callable avoid predicate.
    /// If matched=true, the state is moved to avoid stash; otherwise, it continues
    /// exploration in the active stash.
    pub fn resume_avoid_predicate(&mut self, matched: bool) -> PyResult<()> {
        let pending = self.pending_callback.take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending avoid predicate callback"))?;

        if matched {
            log::debug!("Avoid predicate matched - moving state to avoid stash");
            self.push_or_drop_terminal(STASH_AVOID, pending.state);
        } else {
            log::debug!("Avoid predicate did not match - continuing exploration");
            let state_id = pending.state.state_id();
            self.constraint_tracker.skip_avoid_predicate_states.insert(state_id);
            self.push_to_active_or_drop(pending.state);
        }

        Ok(())
    }

    // =========================================================================
    // Solver Profiling Stats
    // =========================================================================

    /// Get global Z3 solver profiling stats as a dict.
    #[staticmethod]
    pub fn get_solver_stats() -> std::collections::HashMap<String, u64> {
        crate::symbolic::get_solver_stats()
    }

    /// Reset global Z3 solver profiling stats to zero.
    #[staticmethod]
    pub fn reset_solver_stats() {
        crate::symbolic::reset_solver_stats()
    }

}

/// Register the exploration module with Python.
pub fn register_exploration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustExplorationManager>()?;
    m.add_class::<ExplorationEvent>()?;
    m.add_function(pyo3::wrap_pyfunction!(get_stepping_state_id, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exploration_manager_creation() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mgr = RustExplorationManager::new("amd64", None).unwrap();
            assert_eq!(mgr.arch(), "amd64");
            assert_eq!(mgr.active_count(), 0);
            assert_eq!(mgr.found_count(), 0);
        });
    }

    #[test]
    fn test_stash_management() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

            // Create state
            let state_id = mgr.create_state("active").unwrap();
            // state_id is assigned from a global atomic counter starting at 0
            assert!(state_id < u64::MAX);
            assert_eq!(mgr.active_count(), 1);

            // Check state IDs
            let ids = mgr.get_state_ids("active");
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0], state_id);
        });
    }

    #[test]
    fn test_find_avoid_addresses() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

            mgr.set_find_addrs(vec![0x1000, 0x2000]);
            mgr.set_avoid_addrs(vec![0x3000]);

            assert!(mgr.find_addrs.contains(&0x1000));
            assert!(mgr.find_addrs.contains(&0x2000));
            assert!(mgr.avoid_addrs.contains(&0x3000));
        });
    }
}
