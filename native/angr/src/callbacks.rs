//! Python callback infrastructure for Rust VEX engine.
//!
//! This module provides the callback holder that allows Rust to call back into Python
//! for operations like memory access, hook execution, and syscall handling.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

use crate::symbolic::RustBV;

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
    pub condition_ast: Option<PyObject>,
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
        condition_ast: Option<PyObject>,
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
}

#[pymethods]
impl ExecutionConfig {
    /// Create a new execution config with default values.
    #[new]
    #[pyo3(signature = (max_deferred_forks=500, use_deferred_forks=true))]
    pub fn py_new(max_deferred_forks: u32, use_deferred_forks: bool) -> Self {
        ExecutionConfig {
            max_deferred_forks,
            branch_policy: BranchPolicy::TakeTrue,
            use_deferred_forks,
            enable_eager_prefetch: true,
            max_prefetch_batch: 256,
            max_concretization_range: 65536,
            enable_stride_detection: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionConfig(max_deferred_forks={}, use_deferred_forks={}, policy={:?}, eager_prefetch={}, max_prefetch_batch={})",
            self.max_deferred_forks, self.use_deferred_forks, self.branch_policy,
            self.enable_eager_prefetch, self.max_prefetch_batch
        )
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            max_deferred_forks: 500,  // Increased from 100 for complex binaries
            branch_policy: BranchPolicy::TakeTrue,
            use_deferred_forks: true,
            enable_eager_prefetch: true,
            max_prefetch_batch: 256,        // 256 pages = 1MB
            max_concretization_range: 65536,
            enable_stride_detection: true,
        }
    }
}

/// Result of a memory load callback.
#[derive(Debug, Clone)]
pub struct MemoryLoadResult {
    /// The concrete bytes loaded.
    pub data: Vec<u8>,
    /// Whether the value is symbolic (has an associated AST).
    pub is_symbolic: bool,
    /// The symbolic AST (if symbolic). This is a Python object reference.
    pub symbolic_ast: Option<PyObject>,
    /// The RustBV representation for the engine.
    pub value: RustBV,
}

/// Result of running the execution loop.
#[derive(Debug, Clone)]
pub enum RunResult {
    /// Reached max blocks limit - continue later.
    MaxBlocks { pc: u64 },
    /// Hit a hook address - need Python to handle.
    /// This is the legacy variant without pre-extracted arguments.
    Hook { addr: u64 },
    /// SimProcedure hook hit with pre-extracted arguments.
    /// This allows Python to directly use the arguments without re-extracting.
    SimProcedure {
        /// Address where the SimProcedure is hooked.
        addr: u64,
        /// Name of the SimProcedure (e.g., "strlen", "malloc").
        name: String,
        /// Number of arguments extracted (for Python to know how many to use).
        num_args: usize,
        /// Return address (from stack for calls, or 0 if unknown).
        return_addr: u64,
    },
    /// Syscall encountered - need Python to handle.
    Syscall { num: u64, pc: u64 },
    /// Symbolic branch - need Python to fork states.
    /// This is returned when use_deferred_forks is false or max_deferred_forks is reached.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Normal block end.
    BlockEnd { next_addr: u64, jumpkind: String },
    /// Error during execution.
    Error { message: String, addr: u64 },
    /// Need to lift a block at the given address.
    NeedLift { addr: u64 },
    /// Reached max deferred forks limit - return to Python with accumulated forks.
    MaxDeferredForks { pc: u64 },
}

/// Python callback holder for the Rust VEX engine.
///
/// This struct holds references to Python callback functions that the Rust
/// engine calls during execution for memory access, hooks, syscalls, etc.
#[pyclass]
#[derive(Clone)]
pub struct PythonCallbacks {
    /// Callback for memory loads: fn(addr: u64, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub memory_load: Option<PyObject>,
    /// Callback for memory stores: fn(addr: u64, data: bytes) -> None
    pub memory_store: Option<PyObject>,
    /// Callback for batched memory stores: fn(stores: list[tuple[int, bytes]]) -> None
    /// This is more efficient than individual stores when multiple stores can be batched.
    pub memory_store_batch: Option<PyObject>,
    /// Callback for batched memory loads: fn(loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, object | None]]
    /// Each tuple in input is (address, size). Returns list of (data, is_symbolic, ast_or_none).
    /// This is more efficient than individual loads when multiple loads can be batched.
    pub memory_load_batch: Option<PyObject>,
    /// Callback for symbolic memory loads: fn(addrs: list[int], size: int, addr_ast) -> RustBV
    pub memory_load_symbolic: Option<PyObject>,
    /// Callback for symbolic memory stores: fn(addrs: list[int], data: bytes, addr_ast) -> None
    pub memory_store_symbolic: Option<PyObject>,
    /// Callback for memory load with full symbolic address AST: fn(size: int) -> (bytes, is_symbolic, symbolic_ast?)
    /// This is used when the address range is too large to concretize, delegating to angr's memory model.
    pub memory_load_ast: Option<PyObject>,
    /// Callback for memory store with full symbolic address AST: fn(data: bytes, size: int) -> None
    /// This is used when the address range is too large to concretize, delegating to angr's memory model.
    pub memory_store_ast: Option<PyObject>,
    /// Callback for hook execution: fn(addr: u64) -> new_pc
    pub on_hook: Option<PyObject>,
    /// Callback for syscall handling: fn(num: u64) -> None
    pub on_syscall: Option<PyObject>,
    /// Callback for lifting a block: fn(addr: u64) -> irsb_json
    pub lift_block: Option<PyObject>,
    /// Callback for getting register value: fn(offset: u32, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub get_register: Option<PyObject>,
    /// Callback for setting register value: fn(offset: u32, data: bytes) -> None
    pub put_register: Option<PyObject>,
    /// Callback for dirty helper calls: fn(name: str, args: list[int], ret_ty_bits: int) -> (bytes, bool, object | None)
    /// This handles VEX dirty calls to helper functions (CPUID, RDTSC, etc.)
    pub dirty_call: Option<PyObject>,
    /// Callback for fetching a single 4KB page: fn(page_addr: u64) -> (bytes, permissions: u8, is_mapped: bool)
    /// This is used for on-demand page loading when Rust memory encounters an unmapped page.
    pub fetch_page: Option<PyObject>,
    /// Callback for batched page fetching: fn(page_addrs: list[u64]) -> list[(bytes, u8, bool)]
    /// Returns list of (data, permissions, is_mapped) for each requested page.
    pub batch_fetch_pages: Option<PyObject>,
    /// Callback for syncing constraints to Python: fn(constraints: list[(str, int, int, int | None)]) -> None
    /// Each constraint is (description, width, concrete_value, handle_id) where:
    /// - description: human-readable description (e.g., "addr_concretize_0x1234")
    /// - width: bit width of the constrained expression
    /// - concrete_value: the value the expression was constrained to
    /// - handle_id: optional handle ID to look up the original claripy AST
    /// Python should add these constraints to its claripy solver.
    pub sync_constraints: Option<PyObject>,
}

#[pymethods]
impl PythonCallbacks {
    /// Create a new empty callback holder.
    #[new]
    pub fn new() -> Self {
        PythonCallbacks {
            memory_load: None,
            memory_store: None,
            memory_store_batch: None,
            memory_load_batch: None,
            memory_load_symbolic: None,
            memory_store_symbolic: None,
            memory_load_ast: None,
            memory_store_ast: None,
            on_hook: None,
            on_syscall: None,
            lift_block: None,
            get_register: None,
            put_register: None,
            dirty_call: None,
            fetch_page: None,
            batch_fetch_pages: None,
            sync_constraints: None,
        }
    }

    /// Set the memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_memory_load(&mut self, cb: PyObject) {
        self.memory_load = Some(cb);
    }

    /// Set the memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, data: bytes) -> None`
    pub fn set_memory_store(&mut self, cb: PyObject) {
        self.memory_store = Some(cb);
    }

    /// Set the batched memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(stores: list[tuple[int, bytes]]) -> None`
    ///
    /// This is called with a batch of stores for efficiency. Each element is
    /// a (address, data) tuple. If not set, falls back to individual stores.
    pub fn set_memory_store_batch(&mut self, cb: PyObject) {
        self.memory_store_batch = Some(cb);
    }

    /// Set the batched memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, object | None]]`
    ///
    /// Each tuple in input is (address, size). Returns list of (data, is_symbolic, ast_or_none).
    /// This is called with a batch of loads for efficiency, reducing FFI overhead.
    /// If not set, falls back to individual loads.
    pub fn set_memory_load_batch(&mut self, cb: PyObject) {
        self.memory_load_batch = Some(cb);
    }

    /// Set the symbolic memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(addrs: list[int], size: int, addr_ast: object) -> RustBV`
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should build an ITE chain based on the possible addresses.
    pub fn set_memory_load_symbolic(&mut self, cb: PyObject) {
        self.memory_load_symbolic = Some(cb);
    }

    /// Set the symbolic memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(addrs: list[int], data: bytes, addr_ast: object) -> None`
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should perform conditional stores to each possible address.
    pub fn set_memory_store_symbolic(&mut self, cb: PyObject) {
        self.memory_store_symbolic = Some(cb);
    }

    /// Set the symbolic address AST memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// This is called when the address is symbolic and the range is too large to
    /// concretize. The callback should use angr's full memory model to handle
    /// the symbolic address (which angr already has in the state).
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_memory_load_ast(&mut self, cb: PyObject) {
        self.memory_load_ast = Some(cb);
    }

    /// Set the symbolic address AST memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(data: bytes, size: int) -> None`
    ///
    /// This is called when the address is symbolic and the range is too large to
    /// concretize. The callback should use angr's full memory model to handle
    /// the symbolic address (which angr already has in the state).
    pub fn set_memory_store_ast(&mut self, cb: PyObject) {
        self.memory_store_ast = Some(cb);
    }

    /// Set the hook execution callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> int`
    ///
    /// Returns the new PC after hook execution.
    pub fn set_on_hook(&mut self, cb: PyObject) {
        self.on_hook = Some(cb);
    }

    /// Set the syscall handling callback.
    ///
    /// The callback should have signature:
    /// `fn(num: int) -> None`
    pub fn set_on_syscall(&mut self, cb: PyObject) {
        self.on_syscall = Some(cb);
    }

    /// Set the block lifting callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> str`
    ///
    /// Returns the IRSB as a JSON string.
    pub fn set_lift_block(&mut self, cb: PyObject) {
        self.lift_block = Some(cb);
    }

    /// Set the register get callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_get_register(&mut self, cb: PyObject) {
        self.get_register = Some(cb);
    }

    /// Set the register put callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, data: bytes) -> None`
    pub fn set_put_register(&mut self, cb: PyObject) {
        self.put_register = Some(cb);
    }

    /// Set the dirty call callback.
    ///
    /// The callback should have signature:
    /// `fn(name: str, args: list[int], ret_ty_bits: int) -> tuple[bytes, bool, object | None]`
    ///
    /// This handles VEX dirty calls to helper functions like CPUID, RDTSC, etc.
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_dirty_call(&mut self, cb: PyObject) {
        self.dirty_call = Some(cb);
    }

    /// Set the page fetch callback.
    ///
    /// The callback should have signature:
    /// `fn(page_addr: int) -> tuple[bytes, int, bool]`
    ///
    /// Returns (page_data_4kb, permissions, is_mapped).
    /// If is_mapped is False, the page doesn't exist in Python memory.
    pub fn set_fetch_page(&mut self, cb: PyObject) {
        self.fetch_page = Some(cb);
    }

    /// Set the batched page fetch callback.
    ///
    /// The callback should have signature:
    /// `fn(page_addrs: list[int]) -> list[tuple[bytes, int, bool]]`
    ///
    /// Each result is (page_data_4kb, permissions, is_mapped).
    pub fn set_batch_fetch_pages(&mut self, cb: PyObject) {
        self.batch_fetch_pages = Some(cb);
    }

    /// Set the constraint sync callback.
    ///
    /// The callback should have signature:
    /// `fn(constraints: list[tuple[str, int, int, int | None]]) -> None`
    ///
    /// Each tuple is (description, width, concrete_value, handle_id).
    /// Python should add these constraints to its claripy solver.
    /// The handle_id can be used to look up the original claripy AST.
    pub fn set_sync_constraints(&mut self, cb: PyObject) {
        self.sync_constraints = Some(cb);
    }

    /// Check if all required callbacks are set.
    pub fn is_ready(&self) -> bool {
        self.memory_load.is_some()
            && self.memory_store.is_some()
            && self.lift_block.is_some()
    }
}

impl Default for PythonCallbacks {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonCallbacks {
    /// Call the memory load callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_memory_load(
        &self,
        py: Python<'_>,
        addr: u64,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<PyObject>)> {
        let cb = self.memory_load.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("memory_load callback not set")
        })?;

        let result = cb.call1(py, (addr, size))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        // Extract (bytes, is_symbolic, symbolic_ast?)
        let data_obj = tuple.get_item(0)?;
        let data: Vec<u8> = data_obj.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Call the memory store callback.
    pub fn call_memory_store(&self, py: Python<'_>, addr: u64, data: &[u8]) -> PyResult<()> {
        let cb = self.memory_store.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("memory_store callback not set")
        })?;

        let py_bytes = PyBytes::new(py, data);
        cb.call1(py, (addr, py_bytes))?;
        Ok(())
    }

    /// Call the batched memory store callback.
    ///
    /// This sends multiple stores in a single callback for efficiency.
    /// Falls back to individual stores if batch callback is not set.
    pub fn call_memory_store_batch(
        &self,
        py: Python<'_>,
        stores: &[(u64, Vec<u8>)],
    ) -> PyResult<()> {
        if stores.is_empty() {
            return Ok(());
        }

        // Try batch callback first
        if let Some(cb) = &self.memory_store_batch {
            // Convert stores to Python list of tuples
            let py_stores: Vec<(u64, Py<PyBytes>)> = stores
                .iter()
                .map(|(addr, data)| (*addr, PyBytes::new(py, data).unbind()))
                .collect();
            cb.call1(py, (py_stores,))?;
            return Ok(());
        }

        // Fallback: call individual stores
        for (addr, data) in stores {
            self.call_memory_store(py, *addr, data)?;
        }
        Ok(())
    }

    /// Call the batched memory load callback.
    ///
    /// This sends multiple load requests in a single callback for efficiency.
    /// Falls back to individual loads if batch callback is not set.
    ///
    /// Returns a vector of (data_bytes, is_symbolic, symbolic_ast) tuples,
    /// one for each load request.
    pub fn call_memory_load_batch(
        &self,
        py: Python<'_>,
        loads: &[(u64, u32)],  // (address, size) pairs
    ) -> PyResult<Vec<(Vec<u8>, bool, Option<PyObject>)>> {
        if loads.is_empty() {
            return Ok(Vec::new());
        }

        // Try batch callback first
        if let Some(cb) = &self.memory_load_batch {
            // Convert loads to Python list of tuples
            let py_loads: Vec<(u64, u32)> = loads.to_vec();
            let result = cb.call1(py, (py_loads,))?;

            // Parse the result list
            let result_list = result.downcast_bound::<pyo3::types::PyList>(py)?;
            let mut results = Vec::with_capacity(loads.len());

            for item in result_list.iter() {
                let tuple = item.downcast::<pyo3::types::PyTuple>()?;

                // Extract (bytes, is_symbolic, symbolic_ast?)
                let data_obj = tuple.get_item(0)?;
                let data: Vec<u8> = data_obj.extract()?;
                let is_symbolic: bool = tuple.get_item(1)?.extract()?;

                let symbolic_ast = if tuple.len() > 2 {
                    let ast_obj = tuple.get_item(2)?;
                    if ast_obj.is_none() {
                        None
                    } else {
                        Some(ast_obj.unbind())
                    }
                } else {
                    None
                };

                results.push((data, is_symbolic, symbolic_ast));
            }

            return Ok(results);
        }

        // Fallback: call individual loads
        let mut results = Vec::with_capacity(loads.len());
        for &(addr, size) in loads {
            let (data, is_sym, ast) = self.call_memory_load(py, addr, size)?;
            results.push((data, is_sym, ast));
        }
        Ok(results)
    }

    /// Call the symbolic memory load callback.
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// Returns the loaded value as a RustBV.
    pub fn call_memory_load_symbolic(
        &self,
        py: Python<'_>,
        addrs: &[u64],
        size: u32,
        addr_ast: &RustBV,
    ) -> PyResult<RustBV> {
        // If symbolic callback is set, use it
        if let Some(cb) = &self.memory_load_symbolic {
            let addrs_list: Vec<u64> = addrs.to_vec();
            // Convert addr_ast to Python representation
            // For now we pass the concrete addresses and let Python handle the ITE chain
            let result = cb.call1(py, (addrs_list, size, addr_ast.width()))?;

            // The callback should return bytes
            let bytes: Vec<u8> = result.extract(py)?;
            let width = (size * 8) as u32;
            let mut value: u128 = 0;
            for (i, &byte) in bytes.iter().enumerate() {
                if (i * 8) as u32 >= width {
                    break;
                }
                value |= (byte as u128) << (i * 8);
            }
            return Ok(RustBV::concrete(value, width));
        }

        // Fallback: load from first address only
        if let Some(first_addr) = addrs.first() {
            let (data, _is_symbolic, _ast) = self.call_memory_load(py, *first_addr, size)?;
            let width = (size * 8) as u32;
            let mut value: u128 = 0;
            for (i, &byte) in data.iter().enumerate() {
                if (i * 8) as u32 >= width {
                    break;
                }
                value |= (byte as u128) << (i * 8);
            }
            Ok(RustBV::concrete(value, width))
        } else {
            // No addresses - return zero
            Ok(RustBV::zero((size * 8) as u32))
        }
    }

    /// Call the symbolic memory store callback.
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should perform conditional stores to each possible address.
    pub fn call_memory_store_symbolic(
        &self,
        py: Python<'_>,
        addrs: &[u64],
        data: &RustBV,
        addr_ast: &RustBV,
    ) -> PyResult<()> {
        // If symbolic callback is set, use it
        if let Some(cb) = &self.memory_store_symbolic {
            let addrs_list: Vec<u64> = addrs.to_vec();
            let data_bytes = bv_to_bytes(data);
            let py_bytes = PyBytes::new(py, &data_bytes);
            cb.call1(py, (addrs_list, py_bytes, addr_ast.width()))?;
            return Ok(());
        }

        // Fallback: store to first address only (not ideal but maintains progress)
        if let Some(first_addr) = addrs.first() {
            let data_bytes = bv_to_bytes(data);
            self.call_memory_store(py, *first_addr, &data_bytes)?;
        }
        Ok(())
    }

    /// Call the symbolic address AST memory load callback.
    ///
    /// This is called when the address range is too large to concretize.
    /// Python will use angr's full memory model with its address concretization strategies.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_memory_load_symbolic_ast(
        &self,
        py: Python<'_>,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<PyObject>)> {
        // If symbolic AST callback is set, use it
        if let Some(cb) = &self.memory_load_ast {
            let result = cb.call1(py, (size,))?;
            let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

            // Extract (bytes, is_symbolic, symbolic_ast?)
            let data_obj = tuple.get_item(0)?;
            let data: Vec<u8> = data_obj.extract()?;
            let is_symbolic: bool = tuple.get_item(1)?.extract()?;

            let symbolic_ast = if tuple.len() > 2 {
                let ast_obj = tuple.get_item(2)?;
                if ast_obj.is_none() {
                    None
                } else {
                    Some(ast_obj.unbind())
                }
            } else {
                None
            };

            return Ok((data, is_symbolic, symbolic_ast));
        }

        // Fallback: return zeros and symbolic marker
        // This is a placeholder when the callback is not set
        let zeros = vec![0u8; size as usize];
        Ok((zeros, true, None))
    }

    /// Call the symbolic address AST memory store callback.
    ///
    /// This is called when the address range is too large to concretize.
    /// Python will use angr's full memory model with its address concretization strategies.
    pub fn call_memory_store_symbolic_ast(
        &self,
        py: Python<'_>,
        data: &[u8],
        size: u32,
    ) -> PyResult<()> {
        // If symbolic AST callback is set, use it
        if let Some(cb) = &self.memory_store_ast {
            let py_bytes = PyBytes::new(py, data);
            cb.call1(py, (py_bytes, size))?;
            return Ok(());
        }

        // Fallback: silently ignore (data is still in Rust's view)
        // This is a placeholder when the callback is not set
        Ok(())
    }

    /// Call the hook execution callback.
    ///
    /// Returns the new PC after hook execution.
    pub fn call_on_hook(&self, py: Python<'_>, addr: u64) -> PyResult<u64> {
        let cb = self.on_hook.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("on_hook callback not set")
        })?;

        let result = cb.call1(py, (addr,))?;
        result.extract(py)
    }

    /// Call the syscall handling callback.
    pub fn call_on_syscall(&self, py: Python<'_>, num: u64) -> PyResult<()> {
        let cb = self.on_syscall.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("on_syscall callback not set")
        })?;

        cb.call1(py, (num,))?;
        Ok(())
    }

    /// Call the block lifting callback.
    ///
    /// Returns the IRSB as a JSON string.
    pub fn call_lift_block(&self, py: Python<'_>, addr: u64) -> PyResult<String> {
        let cb = self.lift_block.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("lift_block callback not set")
        })?;

        let result = cb.call1(py, (addr,))?;
        result.extract(py)
    }

    /// Call the register get callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_get_register(
        &self,
        py: Python<'_>,
        offset: u32,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<PyObject>)> {
        let cb = self.get_register.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("get_register callback not set")
        })?;

        let result = cb.call1(py, (offset, size))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        let data: Vec<u8> = tuple.get_item(0)?.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Call the register put callback.
    pub fn call_put_register(&self, py: Python<'_>, offset: u32, data: &[u8]) -> PyResult<()> {
        let cb = self.put_register.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("put_register callback not set")
        })?;

        let py_bytes = PyBytes::new(py, data);
        cb.call1(py, (offset, py_bytes))?;
        Ok(())
    }

    /// Call the dirty call callback for VEX helper functions.
    ///
    /// This handles dirty calls like CPUID, RDTSC, x87 operations, etc.
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_dirty_call(
        &self,
        py: Python<'_>,
        name: &str,
        args: &[u64],
        ret_ty_bits: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<PyObject>)> {
        let cb = self.dirty_call.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("dirty_call callback not set")
        })?;

        // Convert args to Python list
        let args_list: Vec<u64> = args.to_vec();

        let result = cb.call1(py, (name, args_list, ret_ty_bits))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        // Extract (bytes, is_symbolic, symbolic_ast?)
        let data_obj = tuple.get_item(0)?;
        let data: Vec<u8> = data_obj.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Check if dirty call callback is available.
    pub fn has_dirty_call(&self) -> bool {
        self.dirty_call.is_some()
    }

    /// Check if fetch_page callback is available.
    pub fn has_fetch_page(&self) -> bool {
        self.fetch_page.is_some()
    }

    /// Call the page fetch callback to load a single 4KB page.
    ///
    /// Returns (page_data, permissions, is_mapped).
    /// - page_data: 4096 bytes of page content
    /// - permissions: permission bits (R=4, W=2, X=1)
    /// - is_mapped: whether the page exists in Python memory
    pub fn call_fetch_page(
        &self,
        py: Python<'_>,
        page_addr: u64,
    ) -> PyResult<(Vec<u8>, u8, bool)> {
        let cb = self.fetch_page.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("fetch_page callback not set")
        })?;

        let result = cb.call1(py, (page_addr,))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        let data: Vec<u8> = tuple.get_item(0)?.extract()?;
        let permissions: u8 = tuple.get_item(1)?.extract()?;
        let is_mapped: bool = tuple.get_item(2)?.extract()?;

        Ok((data, permissions, is_mapped))
    }

    /// Call the batched page fetch callback to load multiple 4KB pages.
    ///
    /// Returns a list of (page_data, permissions, is_mapped) for each page.
    pub fn call_batch_fetch_pages(
        &self,
        py: Python<'_>,
        page_addrs: &[u64],
    ) -> PyResult<Vec<(Vec<u8>, u8, bool)>> {
        if page_addrs.is_empty() {
            return Ok(Vec::new());
        }

        // Try batch callback first
        if let Some(cb) = &self.batch_fetch_pages {
            let addrs_list: Vec<u64> = page_addrs.to_vec();
            let result = cb.call1(py, (addrs_list,))?;

            let result_list = result.downcast_bound::<pyo3::types::PyList>(py)?;
            let mut results = Vec::with_capacity(page_addrs.len());

            for item in result_list.iter() {
                let tuple = item.downcast::<pyo3::types::PyTuple>()?;
                let data: Vec<u8> = tuple.get_item(0)?.extract()?;
                let permissions: u8 = tuple.get_item(1)?.extract()?;
                let is_mapped: bool = tuple.get_item(2)?.extract()?;
                results.push((data, permissions, is_mapped));
            }

            return Ok(results);
        }

        // Fallback: call individual fetches
        let mut results = Vec::with_capacity(page_addrs.len());
        for &page_addr in page_addrs {
            let (data, perms, mapped) = self.call_fetch_page(py, page_addr)?;
            results.push((data, perms, mapped));
        }
        Ok(results)
    }

    /// Sync accumulated constraints to Python's claripy solver.
    ///
    /// This should be called before falling back to Python for operations
    /// that depend on solver state (e.g., symbolic memory operations).
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `constraints` - List of (description, width, concrete_value, handle_id) tuples
    ///
    /// # Returns
    /// Ok(()) on success, or error if callback fails.
    pub fn call_sync_constraints(
        &self,
        py: Python<'_>,
        constraints: &[(String, u32, u128, Option<u64>)],
    ) -> PyResult<()> {
        if constraints.is_empty() {
            return Ok(());
        }

        if let Some(cb) = &self.sync_constraints {
            // Convert to Python list of tuples
            let py_constraints: Vec<(String, u32, u128, Option<u64>)> = constraints.to_vec();
            cb.call1(py, (py_constraints,))?;
        }
        // If no callback is set, silently succeed - constraints will be lost
        // but this allows gradual adoption of the feature
        Ok(())
    }
}

/// Execution event returned to Python from run_loop.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LoopExecutionEvent {
    /// Type of event: "max_blocks", "hook", "simprocedure", "syscall", "symbolic_branch", "block_end", "error", "need_lift", "max_deferred_forks"
    #[pyo3(get)]
    pub event_type: String,
    /// Current/next PC address.
    #[pyo3(get)]
    pub pc: Option<u64>,
    /// Hook/target address.
    #[pyo3(get)]
    pub addr: Option<u64>,
    /// Syscall number.
    #[pyo3(get)]
    pub syscall_num: Option<u64>,
    /// True target for symbolic branch.
    #[pyo3(get)]
    pub true_target: Option<u64>,
    /// False target for symbolic branch.
    #[pyo3(get)]
    pub false_target: Option<u64>,
    /// Jump kind string.
    #[pyo3(get)]
    pub jumpkind: Option<String>,
    /// Error message.
    #[pyo3(get)]
    pub error: Option<String>,
    /// Number of blocks executed this loop.
    #[pyo3(get)]
    pub blocks_executed: u32,
    /// Deferred forks collected during execution.
    /// Each fork represents a branch where we took one path and deferred the other.
    #[pyo3(get)]
    pub deferred_forks: Vec<DeferredFork>,
    /// Current solver push level after execution.
    /// Used for proper constraint handling during fork processing.
    #[pyo3(get)]
    pub push_level: u32,
    /// SimProcedure name (for "simprocedure" events).
    #[pyo3(get)]
    pub simprocedure_name: Option<String>,
    /// Number of arguments for SimProcedure.
    #[pyo3(get)]
    pub simprocedure_num_args: Option<usize>,
    /// Return address for SimProcedure (from stack).
    #[pyo3(get)]
    pub simprocedure_return_addr: Option<u64>,
}

impl LoopExecutionEvent {
    /// Create an event from a run result with deferred forks and push level.
    pub fn from_run_result_with_forks(
        result: RunResult,
        blocks_executed: u32,
        deferred_forks: Vec<DeferredFork>,
        push_level: u32,
    ) -> Self {
        match result {
            RunResult::MaxBlocks { pc } => LoopExecutionEvent {
                event_type: "max_blocks".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::Hook { addr } => LoopExecutionEvent {
                event_type: "hook".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::SimProcedure { addr, name, num_args, return_addr } => LoopExecutionEvent {
                event_type: "simprocedure".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some("Ijk_Call".to_string()),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: Some(name),
                simprocedure_num_args: Some(num_args),
                simprocedure_return_addr: if return_addr != 0 { Some(return_addr) } else { None },
            },
            RunResult::Syscall { num, pc } => LoopExecutionEvent {
                event_type: "syscall".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: Some(num),
                true_target: None,
                false_target: None,
                jumpkind: Some("Ijk_Sys_syscall".to_string()),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => LoopExecutionEvent {
                event_type: "symbolic_branch".to_string(),
                pc: None,
                addr: None,
                syscall_num: None,
                true_target: Some(true_target),
                false_target: Some(false_target),
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::BlockEnd { next_addr, jumpkind } => LoopExecutionEvent {
                event_type: "block_end".to_string(),
                pc: Some(next_addr),
                addr: Some(next_addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some(jumpkind),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::Error { message, addr } => LoopExecutionEvent {
                event_type: "error".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: Some(message),
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::NeedLift { addr } => LoopExecutionEvent {
                event_type: "need_lift".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
            RunResult::MaxDeferredForks { pc } => LoopExecutionEvent {
                event_type: "max_deferred_forks".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
            },
        }
    }

    /// Create an event from a run result (backward compatibility, no deferred forks).
    pub fn from_run_result(result: RunResult, blocks_executed: u32) -> Self {
        Self::from_run_result_with_forks(result, blocks_executed, Vec::new(), 0)
    }
}

/// Thread-safe wrapper for Python callbacks.
///
/// This allows the callbacks to be shared across interpreter instances
/// during a single execution loop.
pub struct CallbacksRef {
    inner: Arc<PythonCallbacks>,
}

impl CallbacksRef {
    pub fn new(callbacks: PythonCallbacks) -> Self {
        CallbacksRef {
            inner: Arc::new(callbacks),
        }
    }

    pub fn get(&self) -> &PythonCallbacks {
        &self.inner
    }
}

impl Clone for CallbacksRef {
    fn clone(&self) -> Self {
        CallbacksRef {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Convert a RustBV to bytes (little-endian).
fn bv_to_bytes(bv: &RustBV) -> Vec<u8> {
    let width = bv.width();
    let num_bytes = ((width + 7) / 8) as usize;

    if let Some(value) = bv.as_u128() {
        let mut bytes = vec![0u8; num_bytes];
        for i in 0..num_bytes {
            bytes[i] = (value >> (i * 8)) as u8;
        }
        bytes
    } else {
        // For symbolic values, return zeros (the callback will handle it)
        vec![0u8; num_bytes]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callbacks_creation() {
        pyo3::prepare_freethreaded_python();
        let callbacks = PythonCallbacks::new();
        assert!(!callbacks.is_ready());
    }

    #[test]
    fn test_loop_execution_event() {
        let event = LoopExecutionEvent::from_run_result(
            RunResult::BlockEnd {
                next_addr: 0x1000,
                jumpkind: "Ijk_Boring".to_string(),
            },
            5,
        );
        assert_eq!(event.event_type, "block_end");
        assert_eq!(event.pc, Some(0x1000));
        assert_eq!(event.blocks_executed, 5);
    }
}
