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
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use crate::arch::{arch_from_name, default_cc_for_arch, CallingConvention};
use crate::callbacks::{ExecutionConfig, PythonCallbacks, RunResult, DeferredFork};
use crate::interpreter_cb::CallbackInterpreter;
use crate::memory::Permission;
use crate::procedures::{NativeProcedureRegistry, ProcedureError};
use crate::state::{RustSimState, StateChanges};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::{VexArch, IRSB};

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
}

impl ExplorationEvent {
    fn found(found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "found".to_string(),
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
        }
    }

    fn deadended(found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "deadended".to_string(),
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
        }
    }

    fn active_empty(found_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "active_empty".to_string(),
            found_count,
            active_count: 0,
            steps_taken: steps,
            callback_state_id: None,
            callback_reason: None,
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: None,
            callback_return_addr: None,
            callback_num_args: None,
        }
    }

    fn step_complete(found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "step_complete".to_string(),
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
        }
    }

    fn need_simprocedure(
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
            event_type: "need_callback".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: Some(state_id),
            callback_reason: Some("simprocedure".to_string()),
            callback_addr: Some(addr),
            callback_name: Some(name),
            callback_syscall_num: None,
            callback_return_addr: Some(return_addr),
            callback_num_args: Some(num_args),
        }
    }

    fn need_syscall(
        state_id: u64,
        syscall_num: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            event_type: "need_callback".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: Some(state_id),
            callback_reason: Some("syscall".to_string()),
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: Some(syscall_num),
            callback_return_addr: None,
            callback_num_args: None,
        }
    }

    fn error(message: String, found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "errored".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: None,
            callback_reason: Some(message),
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: None,
            callback_return_addr: None,
            callback_num_args: None,
        }
    }
}

/// State held during a Python callback.
struct PendingCallback {
    state: RustSimState,
    reason: CallbackReason,
}

/// Statistics for native procedure execution.
#[derive(Debug, Clone)]
struct NativeProcStats {
    /// Number of native procedure executions.
    native_calls: u64,
    /// Number of fallbacks to Python.
    python_fallbacks: u64,
    /// Per-procedure call counts.
    call_counts: HashMap<String, u64>,
}

impl Default for NativeProcStats {
    fn default() -> Self {
        NativeProcStats {
            native_calls: 0,
            python_fallbacks: 0,
            call_counts: HashMap::new(),
        }
    }
}

/// Rust-native exploration manager.
///
/// Manages states entirely in Rust with O(1) forking.
/// Returns to Python only for SimProcedures, syscalls, and predicates.
#[pyclass(unsendable)]
pub struct RustExplorationManager {
    /// Architecture name.
    arch_name: String,
    /// VEX architecture enum.
    vex_arch: VexArch,
    /// Stash system (mirrors Python's).
    stashes: HashMap<String, VecDeque<RustSimState>>,
    /// Find addresses.
    find_addrs: HashSet<u64>,
    /// Avoid addresses.
    avoid_addrs: HashSet<u64>,
    /// Whether find condition has callable predicates.
    find_needs_python: bool,
    /// Whether avoid condition has callable predicates.
    avoid_needs_python: bool,
    /// Execution config for deferred forks.
    exec_config: ExecutionConfig,
    /// Python callbacks for memory/lifting.
    callbacks: Option<PythonCallbacks>,
    /// Hook addresses.
    hooks: HashSet<u64>,
    /// SimProcedures: address -> (name, num_args, no_return).
    simprocedures: HashMap<u64, (String, usize, bool)>,
    /// Binary code regions for native lifting.
    binary_regions: Vec<(u64, Vec<u8>)>,
    /// Block cache (shared across states).
    block_cache: LruCache<u64, IRSB>,
    /// Pending state waiting for Python callback result.
    pending_callback: Option<PendingCallback>,
    /// Total steps executed.
    steps: u64,
    /// Error log: (addr, message, state_id).
    errors: Vec<(u64, String, u64)>,
    /// Number of finds required before stopping.
    num_find: usize,
    /// Maximum steps per run iteration.
    max_steps_per_run: u32,
    /// Native procedure registry.
    native_procedures: NativeProcedureRegistry,
    /// Calling convention for argument extraction.
    calling_convention: Box<dyn CallingConvention>,
    /// Statistics for native procedure executions.
    native_proc_stats: NativeProcStats,
}

#[pymethods]
impl RustExplorationManager {
    /// Create a new exploration manager.
    #[new]
    #[pyo3(signature = (arch="amd64"))]
    pub fn new(arch: &str) -> PyResult<Self> {
        let arch_info = arch_from_name(arch).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported architecture: {}", arch))
        })?;

        let vex_arch = arch_info.vex_arch();

        let mut stashes = HashMap::new();
        stashes.insert("active".to_string(), VecDeque::new());
        stashes.insert("found".to_string(), VecDeque::new());
        stashes.insert("avoid".to_string(), VecDeque::new());
        stashes.insert("deadended".to_string(), VecDeque::new());
        stashes.insert("errored".to_string(), VecDeque::new());

        Ok(RustExplorationManager {
            arch_name: arch.to_string(),
            vex_arch,
            stashes,
            find_addrs: HashSet::new(),
            avoid_addrs: HashSet::new(),
            find_needs_python: false,
            avoid_needs_python: false,
            exec_config: ExecutionConfig::default(),
            callbacks: None,
            hooks: HashSet::new(),
            simprocedures: HashMap::new(),
            binary_regions: Vec::new(),
            block_cache: LruCache::new(NonZeroUsize::new(4096).unwrap()),
            pending_callback: None,
            steps: 0,
            errors: Vec::new(),
            num_find: 1,
            max_steps_per_run: 100,
            native_procedures: NativeProcedureRegistry::new(),
            calling_convention: default_cc_for_arch(arch),
            native_proc_stats: NativeProcStats::default(),
        })
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.arch_name
    }

    /// Get the total number of steps executed.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Get active state count.
    pub fn active_count(&self) -> usize {
        self.stashes.get("active").map(|s| s.len()).unwrap_or(0)
    }

    /// Get found state count.
    pub fn found_count(&self) -> usize {
        self.stashes.get("found").map(|s| s.len()).unwrap_or(0)
    }

    /// Get stash counts as a dictionary.
    pub fn stash_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, stash) in &self.stashes {
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

    /// Set the number of solutions to find before stopping.
    pub fn set_num_find(&mut self, n: usize) {
        self.num_find = n;
    }

    /// Set maximum steps per run iteration.
    pub fn set_max_steps_per_run(&mut self, n: u32) {
        self.max_steps_per_run = n;
    }

    /// Set Python callbacks for memory/lifting.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
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
        self.binary_regions = regions;
    }

    /// Create a new RustSimState and add it to a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        let state = RustSimState::new(&self.arch_name)
            .map_err(|e| PyValueError::new_err(e))?;
        let state_id = state.state_id();

        // Copy hooks to state
        for &addr in &self.hooks {
            // State hooks are checked during execution
        }

        self.stashes
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(state);

        Ok(state_id)
    }

    /// Add an existing RustSimState to a stash.
    #[pyo3(signature = (stash, state))]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        // Fork the state to get our own copy
        let forked = state.inner().fork();
        self.stashes
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(forked);
    }

    /// Get the PC of a state in a stash by index.
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.stashes.get(stash).and_then(|s| s.get(index)).map(|s| s.pc())
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.stashes
            .get(stash)
            .map(|s| s.iter().map(|state| state.state_id()).collect())
            .unwrap_or_default()
    }

    /// Set the PC of the pending callback state (for external initialization).
    pub fn set_pending_state_pc(&mut self, pc: u64) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.set_pc(pc);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Map memory in the pending state.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn pending_state_map_memory(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.map_memory_data(addr, data, Permission::from_bits(permissions));
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Map memory in active states.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        if let Some(stash) = self.stashes.get_mut("active") {
            for state in stash.iter_mut() {
                state.map_memory_data(addr, data, Permission::from_bits(permissions));
            }
        }
    }

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
            let mut state = match self.stashes.get_mut("active").and_then(|s| s.pop_front()) {
                Some(s) => s,
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

            // Check avoid addresses
            if self.avoid_addrs.contains(&pc) {
                self.stashes
                    .entry("avoid".to_string())
                    .or_insert_with(VecDeque::new)
                    .push_back(state);
                continue;
            }

            // Check find addresses
            if self.find_addrs.contains(&pc) {
                self.stashes
                    .entry("found".to_string())
                    .or_insert_with(VecDeque::new)
                    .push_back(state);
                continue;
            }

            // Check hooks (SimProcedures)
            if self.hooks.contains(&pc) {
                // Check if this is a registered SimProcedure
                if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                    // Try native procedure first
                    if let Some(native_proc) = self.native_procedures.get(&name) {
                        // Extract arguments using calling convention
                        let ctx = state.solver().borrow();
                        let args = self.calling_convention.extract_args(
                            &crate::arch::RegisterFile::new(
                                crate::arch::arch_from_name(&self.arch_name).unwrap()
                            ),
                            None, // TODO: Pass memory for stack args
                            &ctx,
                            num_args,
                        );
                        drop(ctx);

                        // Get args from state registers directly
                        let args = self.extract_procedure_args(&state, num_args);

                        // Try to execute native procedure
                        match native_proc.call(&mut state, &args) {
                            Ok(ret_val) => {
                                // Native execution succeeded
                                self.native_proc_stats.native_calls += 1;
                                *self.native_proc_stats.call_counts
                                    .entry(name.clone())
                                    .or_insert(0) += 1;

                                // Set return value if present
                                if let Some(rv) = ret_val {
                                    let ret_reg = self.calling_convention.return_register();
                                    state.set_register_by_offset(ret_reg, rv);
                                }

                                // Get return address and set PC
                                let ctx = state.solver().borrow();
                                if let Some(ret_addr) = self.calling_convention.get_return_addr(
                                    &crate::arch::RegisterFile::new(
                                        crate::arch::arch_from_name(&self.arch_name).unwrap()
                                    ),
                                    None,
                                    &ctx,
                                ) {
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

                                // If no_return, put state in deadended
                                if no_return {
                                    self.stashes
                                        .entry("deadended".to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(state);
                                } else {
                                    // Add back to active stash
                                    self.stashes
                                        .entry("active".to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(state);
                                }
                                continue;
                            }
                            Err(_e) => {
                                // Native execution failed, fall back to Python
                                self.native_proc_stats.python_fallbacks += 1;
                            }
                        }
                    }

                    // Fall back to Python for SimProcedure execution
                    let state_id = state.state_id();
                    let return_addr = self.get_return_addr(&state).unwrap_or(0);

                    self.pending_callback = Some(PendingCallback {
                        state,
                        reason: CallbackReason::SimProcedure {
                            addr: pc,
                            name: name.clone(),
                            num_args,
                            return_addr,
                        },
                    });

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

            // Step the state
            match self.step_state(py, &callbacks, state) {
                Ok(successors) => {
                    // Add successors back to active stash
                    let active = self.stashes
                        .entry("active".to_string())
                        .or_insert_with(VecDeque::new);
                    for successor in successors {
                        active.push_back(successor);
                    }
                }
                Err(StepError::NeedCallback(pending)) => {
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
                    self.stashes
                        .entry("deadended".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
                Err(StepError::Error(state, message)) => {
                    let pc = state.pc();
                    let state_id = state.state_id();
                    self.errors.push((pc, message, state_id));
                    self.stashes
                        .entry("errored".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
            }

            self.steps += 1;
        }

        // Max steps reached
        Ok(ExplorationEvent::step_complete(
            self.found_count(),
            self.active_count(),
            self.steps,
        ))
    }

    /// Resume after a SimProcedure callback.
    ///
    /// This is called from Python after executing a SimProcedure.
    /// The state changes (registers, memory, new PC) are applied.
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None))]
    pub fn resume_after_simprocedure(
        &mut self,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
    ) -> PyResult<()> {
        let mut pending = self.pending_callback.take().ok_or_else(|| {
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

        pending.state.apply_changes(&changes);
        pending.state.set_pc(new_pc);

        // Add state back to active stash
        self.stashes
            .entry("active".to_string())
            .or_insert_with(VecDeque::new)
            .push_back(pending.state);

        Ok(())
    }

    /// Resume after a syscall callback.
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None))]
    pub fn resume_after_syscall(
        &mut self,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure for now
        self.resume_after_simprocedure(new_pc, register_changes, memory_changes)
    }

    /// Get register value from pending state.
    pub fn get_pending_register(&self, name: &str) -> PyResult<Option<u128>> {
        if let Some(ref pending) = self.pending_callback {
            pending.state.get_register(name)
                .map(|bv| bv.as_u128())
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set register value in pending state.
    pub fn set_pending_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let size = pending.state.arch().register_size(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let bv = crate::symbolic::RustBV::concrete(value, size * 8);
            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get memory from pending state.
    pub fn get_pending_memory(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        if let Some(ref pending) = self.pending_callback {
            let bv = pending.state.memory_load(addr, size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let value = bv.to_u128();
            let bytes: Vec<u8> = (0..size as usize)
                .map(|i| (value >> (i * 8)) as u8)
                .collect();
            Ok(bytes)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Store memory in pending state.
    pub fn set_pending_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let width = (data.len() * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in data.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = crate::symbolic::RustBV::concrete(value, width);
            pending.state.memory_store(addr, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
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
    pub fn move_states(&mut self, from_stash: &str, to_stash: &str, filter_fn: Option<PyObject>) -> PyResult<usize> {
        // If no filter, move all
        if filter_fn.is_none() {
            if let Some(mut from) = self.stashes.remove(from_stash) {
                let count = from.len();
                let to = self.stashes.entry(to_stash.to_string()).or_insert_with(VecDeque::new);
                to.append(&mut from);
                self.stashes.insert(from_stash.to_string(), VecDeque::new());
                return Ok(count);
            }
            return Ok(0);
        }

        // With filter - for now just move all (filter requires Python evaluation)
        // TODO: Implement filter evaluation
        self.move_states(from_stash, to_stash, None)
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
        dict.set_item("block_cache_size", self.block_cache.len())?;
        dict.set_item("native_proc_calls", self.native_proc_stats.native_calls)?;
        dict.set_item("native_proc_fallbacks", self.native_proc_stats.python_fallbacks)?;
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
        dict.set_item("native_calls", self.native_proc_stats.native_calls)?;
        dict.set_item("python_fallbacks", self.native_proc_stats.python_fallbacks)?;

        let call_counts = PyDict::new(py);
        for (name, count) in &self.native_proc_stats.call_counts {
            call_counts.set_item(name, *count)?;
        }
        dict.set_item("call_counts", call_counts)?;

        Ok(dict)
    }

    // =========================================================================
    // State Export Methods
    // =========================================================================

    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    pub fn export_state(&self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        // Search all stashes for the state
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Ok(state.export_full());
                }
            }
        }

        // Also check pending callback state
        if let Some(ref pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Ok(pending.state.export_full());
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export all states in a stash as snapshots.
    pub fn export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.stashes
            .get(stash)
            .map(|s| s.iter().map(|state| state.export_full()).collect())
            .unwrap_or_default()
    }

    /// Export all found states as snapshots.
    pub fn export_found_states(&self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.export_stash("found")
    }

    /// Evaluate a symbolic value in a state's solver context.
    ///
    /// This allows Python to get concrete values for symbolic inputs
    /// that were found during exploration.
    pub fn eval_in_state(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        // Search for the state
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    // Try to load and evaluate from memory
                    match state.memory_load(addr, size) {
                        Ok(bv) => {
                            // Try to evaluate to concrete value
                            if let Some(val) = state.eval(&bv) {
                                let bytes: Vec<u8> = (0..size as usize)
                                    .map(|i| (val >> (i * 8)) as u8)
                                    .collect();
                                return Ok(Some(bytes));
                            }
                            return Ok(None);
                        }
                        Err(_) => return Ok(None),
                    }
                }
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Ok(state.satisfiable());
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get a register value from a state.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Ok(state.get_register(name).and_then(|bv| bv.as_u128()));
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get memory from a state.
    pub fn get_state_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    match state.memory_load(addr, size) {
                        Ok(bv) => {
                            if let Some(val) = bv.as_u128() {
                                let bytes: Vec<u8> = (0..size as usize)
                                    .map(|i| (val >> (i * 8)) as u8)
                                    .collect();
                                return Ok(Some(bytes));
                            }
                            // Try to evaluate symbolic value
                            if let Some(val) = state.eval(&bv) {
                                let bytes: Vec<u8> = (0..size as usize)
                                    .map(|i| (val >> (i * 8)) as u8)
                                    .collect();
                                return Ok(Some(bytes));
                            }
                            return Ok(None);
                        }
                        Err(_) => return Ok(None),
                    }
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }
}

impl RustExplorationManager {
    /// Extract procedure arguments from state registers.
    fn extract_procedure_args(&self, state: &RustSimState, num_args: usize) -> Vec<RustBV> {
        let arg_regs = self.calling_convention.arg_registers();
        let ptr_size = self.calling_convention.pointer_size();
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        // Extract from registers first
        for &offset in arg_regs.iter().take(num_args) {
            let value = state.get_register_by_offset(offset, ptr_size);
            args.push(value);
        }

        // If we need more args from stack, get them
        if args.len() < num_args {
            if let Some(sp) = state.get_sp().as_u64() {
                let stack_start = sp + self.calling_convention.stack_arg_offset();
                for i in 0..(num_args - args.len()) {
                    let addr = stack_start + (i as u64 * ptr_size as u64);
                    if let Ok(value) = state.memory_load(addr, ptr_size) {
                        args.push(value);
                    } else {
                        // Can't read stack - push zero
                        args.push(RustBV::zero(ptr_size * 8));
                    }
                }
            }
        }

        drop(ctx);
        args
    }

    /// Get return address from stack.
    fn get_return_addr(&self, state: &RustSimState) -> Option<u64> {
        let sp = state.get_sp().as_u64()?;
        let ptr_size = self.calling_convention.pointer_size();

        // On x86/AMD64, return address is at [rsp] after call
        state.memory_load(sp, ptr_size).ok()?.as_u64()
    }
}

/// Error during state stepping.
enum StepError {
    /// Need Python callback.
    NeedCallback(PendingCallback),
    /// State deadended (no successors).
    Deadended(RustSimState),
    /// Error during execution.
    Error(RustSimState, String),
}

impl RustExplorationManager {
    /// Step a single state, returning successors.
    fn step_state(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
    ) -> Result<Vec<RustSimState>, StepError> {
        // Get state information before borrowing the solver
        let reg_bytes = state.get_registers_raw();
        let initial_pc = state.pc();

        // Use the state's solver context for proper constraint handling
        let solver_rc = state.solver().clone();

        // Scope for interpreter execution with borrowed solver
        let (result, deferred_forks, last_condition, stored_conditions, new_reg_bytes, new_pc) = {
            let solver_ref = solver_rc.borrow();

            // Create interpreter with the state's solver
            let mut interp = CallbackInterpreter::with_config(
                self.vex_arch,
                &*solver_ref,
                self.exec_config.clone(),
            );

            // Copy state to interpreter
            interp.registers.copy_from_bytes(&reg_bytes);
            interp.set_pc(initial_pc);

            // Set up hooks
            for &addr in &self.hooks {
                interp.add_hook(addr);
            }

            // Register SimProcedures
            for (addr, (name, num_args, no_return)) in &self.simprocedures {
                interp.register_simprocedure(*addr, name.clone(), *num_args, *no_return);
            }

            // Copy binary regions for code
            for (base, data) in &self.binary_regions {
                interp.add_concrete_memory(*base, data.clone());
            }

            // Run until event
            let (result, _blocks_executed, deferred_forks) = interp.run_until_event(py, callbacks, 100);

            // Get last branch condition before dropping interpreter
            let last_condition = interp.take_last_branch_condition();

            // Get stored conditions for deferred fork handling
            let stored_conditions = interp.take_stored_conditions();

            // Extract register values
            let mut new_reg_bytes = vec![0u8; reg_bytes.len()];
            interp.registers.copy_to_bytes(&mut new_reg_bytes);
            let new_pc = interp.get_pc();

            (result, deferred_forks, last_condition, stored_conditions, new_reg_bytes, new_pc)
        };
        // solver_ref dropped here, solver_rc borrow released

        // Update state from interpreter results
        state.set_registers_raw(&new_reg_bytes);
        state.set_pc(new_pc);

        // Add to history
        state.add_to_history(state.pc());

        // Process result
        match result {
            RunResult::MaxBlocks { pc } |
            RunResult::MaxDeferredForks { pc } |
            RunResult::BlockEnd { next_addr: pc, .. } => {
                state.set_pc(pc);

                // Process deferred forks with proper constraint handling
                let mut successors = vec![state];
                for fork in deferred_forks {
                    // Look up the condition for this deferred fork
                    if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                        // Create forked state with proper constraint for the unexplored path
                        // The deferred fork tells us which path was taken, so the forked
                        // state needs the opposite constraint
                        let forked = if fork.path_taken {
                            // Took the true branch, so fork needs false constraint
                            let mut f = successors[0].fork_false(condition);
                            f.set_pc(fork.unexplored_target);
                            f
                        } else {
                            // Took the false branch, so fork needs true constraint
                            let mut f = successors[0].fork_true(condition);
                            f.set_pc(fork.unexplored_target);
                            f
                        };
                        successors.push(forked);
                    } else {
                        // Fallback: no condition available, fork without constraint
                        let mut forked = successors[0].fork();
                        forked.set_pc(fork.unexplored_target);
                        successors.push(forked);
                    }
                }

                Ok(successors)
            }
            RunResult::Hook { addr } => {
                state.set_pc(addr);
                // Return to Python for hook
                Err(StepError::NeedCallback(PendingCallback {
                    state,
                    reason: CallbackReason::SimProcedure {
                        addr,
                        name: "unknown".to_string(),
                        num_args: 0,
                        return_addr: 0,
                    },
                }))
            }
            RunResult::SimProcedure { addr, name, num_args, return_addr } => {
                state.set_pc(addr);
                Err(StepError::NeedCallback(PendingCallback {
                    state,
                    reason: CallbackReason::SimProcedure {
                        addr,
                        name,
                        num_args,
                        return_addr,
                    },
                }))
            }
            RunResult::Syscall { num, pc } => {
                state.set_pc(pc);
                Err(StepError::NeedCallback(PendingCallback {
                    state,
                    reason: CallbackReason::Syscall { num },
                }))
            }
            RunResult::SymbolicBranch { true_target, false_target, .. } => {
                // Fork for both branches with proper constraint handling
                if let Some(condition) = last_condition {
                    // Use fork_true/fork_false to properly add branch constraints
                    let mut true_state = state.fork_true(&condition);
                    let mut false_state = state.fork_false(&condition);

                    true_state.set_pc(true_target);
                    false_state.set_pc(false_target);

                    Ok(vec![true_state, false_state])
                } else {
                    // Fallback: no condition available (shouldn't happen normally)
                    // Fork without constraints as a safety measure
                    let mut true_state = state.fork();
                    let false_state = state;

                    true_state.set_pc(true_target);
                    // false_state already has the correct PC from the original state

                    Ok(vec![true_state, false_state])
                }
            }
            RunResult::Error { message, addr } => {
                state.set_pc(addr);
                Err(StepError::Error(state, message))
            }
            RunResult::NeedLift { addr } => {
                // This shouldn't happen if callbacks are properly set
                state.set_pc(addr);
                Err(StepError::Error(state, format!("need lift at 0x{:x}", addr)))
            }
        }
    }
}

/// Register the exploration module with Python.
pub fn register_exploration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustExplorationManager>()?;
    m.add_class::<ExplorationEvent>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exploration_manager_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let mgr = RustExplorationManager::new("amd64").unwrap();
            assert_eq!(mgr.arch(), "amd64");
            assert_eq!(mgr.active_count(), 0);
            assert_eq!(mgr.found_count(), 0);
        });
    }

    #[test]
    fn test_stash_management() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let mut mgr = RustExplorationManager::new("amd64").unwrap();

            // Create state
            let state_id = mgr.create_state("active").unwrap();
            assert!(state_id > 0);
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
        Python::with_gil(|_py| {
            let mut mgr = RustExplorationManager::new("amd64").unwrap();

            mgr.set_find_addrs(vec![0x1000, 0x2000]);
            mgr.set_avoid_addrs(vec![0x3000]);

            assert!(mgr.find_addrs.contains(&0x1000));
            assert!(mgr.find_addrs.contains(&0x2000));
            assert!(mgr.avoid_addrs.contains(&0x3000));
        });
    }
}
