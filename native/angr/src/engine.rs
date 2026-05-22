//! Python-facing VEX execution engine.
//!
//! This module provides the PyO3 bindings for the Rust VEX execution engine,
//! allowing it to be used as an alternative engine in angr.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::arch::arch_from_name;
use crate::callbacks::{
    BranchPolicy, DeferredFork, ExecutionConfig, LoopExecutionEvent, PythonCallbacks,
};
use crate::claripy_bridge::claripy_to_rustbv;
use crate::concretize::AddressConcretizer;
use crate::interpreter::{ExecutionResult, VEXInterpreter};
use crate::interpreter_cb::CallbackInterpreter;
use crate::memory::{Permission, SymbolicMemory};
use crate::solver::RustSolverContext;
use crate::symbolic::{RustBV, SymContext};
use crate::vex::{Endness, IRSB, VexArch, deserialize_irsb};

/// Execution event returned to Python.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExecutionEvent {
    /// Type of event.
    #[pyo3(get)]
    pub event_type: String,
    /// Next address (if applicable).
    #[pyo3(get)]
    pub next_addr: Option<u64>,
    /// Jump kind as string.
    #[pyo3(get)]
    pub jumpkind: Option<String>,
    /// Syscall number (for syscall events).
    #[pyo3(get)]
    pub syscall_num: Option<u64>,
    /// Error message (for error events).
    #[pyo3(get)]
    pub error: Option<String>,
    /// True target for symbolic branch.
    #[pyo3(get)]
    pub true_target: Option<u64>,
    /// False target for symbolic branch.
    #[pyo3(get)]
    pub false_target: Option<u64>,
}

impl ExecutionEvent {
    fn from_result(result: ExecutionResult) -> Self {
        match result {
            ExecutionResult::BlockEnd {
                next_addr,
                jumpkind,
            } => ExecutionEvent {
                event_type: "block_end".to_string(),
                next_addr: Some(next_addr),
                jumpkind: Some(format!("{:?}", jumpkind)),
                syscall_num: None,
                error: None,
                true_target: None,
                false_target: None,
            },
            ExecutionResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => ExecutionEvent {
                event_type: "symbolic_branch".to_string(),
                next_addr: None,
                jumpkind: None,
                syscall_num: None,
                error: None,
                true_target: Some(true_target),
                false_target: Some(false_target),
            },
            ExecutionResult::Syscall { num } => ExecutionEvent {
                event_type: "syscall".to_string(),
                next_addr: None,
                jumpkind: Some("Ijk_Sys_syscall".to_string()),
                syscall_num: Some(num),
                error: None,
                true_target: None,
                false_target: None,
            },
            ExecutionResult::Hook { addr } => ExecutionEvent {
                event_type: "hook".to_string(),
                next_addr: Some(addr),
                jumpkind: None,
                syscall_num: None,
                error: None,
                true_target: None,
                false_target: None,
            },
            ExecutionResult::Error { kind, addr } => ExecutionEvent {
                event_type: "error".to_string(),
                next_addr: Some(addr),
                jumpkind: None,
                syscall_num: None,
                error: Some(format!("{:?}", kind)),
                true_target: None,
                false_target: None,
            },
        }
    }

    fn error(msg: String) -> Self {
        ExecutionEvent {
            event_type: "error".to_string(),
            next_addr: None,
            jumpkind: None,
            syscall_num: None,
            error: Some(msg),
            true_target: None,
            false_target: None,
        }
    }
}

/// State snapshot for Python.
#[pyclass]
pub struct StateSnapshot {
    /// Register values.
    registers: HashMap<String, Vec<u8>>,
    /// PC value.
    #[pyo3(get)]
    pub pc: u64,
}

#[pymethods]
impl StateSnapshot {
    /// Get a register value as bytes.
    fn get_register(&self, name: &str) -> Option<Vec<u8>> {
        self.registers.get(name).cloned()
    }

    /// Get all register names.
    fn register_names(&self) -> Vec<String> {
        self.registers.keys().cloned().collect()
    }
}

/// Rust VEX execution engine.
///
/// This is the main Python-facing class that provides VEX-based execution.
///
/// Note: This uses `unsendable` because SymbolicMemory contains z3 AST types
/// (RustBV) which are not Send-safe due to z3's internal Rc/NonNull usage.
#[pyclass(unsendable)]
pub struct RustVEXEngine {
    /// Architecture name.
    arch_name: String,
    /// VEX architecture.
    vex_arch: VexArch,
    /// Block cache.
    block_cache: LruCache<u64, IRSB>,
    /// Hook addresses.
    hooks: std::collections::HashSet<u64>,
    /// State data (serializable part).
    registers: Vec<u8>,
    /// PC.
    pc: u64,
    /// Mapped memory regions: (addr, size, permissions, data).
    memory_regions: Vec<(u64, u64, u8, Vec<u8>)>,
    /// Dirty pages (page-aligned addresses that have been written to).
    dirty_pages: std::collections::HashSet<u64>,
    /// Symbolic objects (would need proper serialization).
    symbolic_count: u64,
    /// Cached hooks set for quick comparison (optimization).
    hooks_version: u64,
    /// Memory mapping version for detecting changes (optimization).
    memory_version: u64,
    /// Python callbacks for memory/hook/syscall handling.
    callbacks: Option<PythonCallbacks>,
    /// Execution configuration for deferred forks.
    execution_config: ExecutionConfig,
    /// Bitset tracking which register offsets have been modified.
    /// Each bit represents a 4-byte aligned offset (offset / 4).
    dirty_registers: u128,
    /// Rust-native symbolic memory model.
    /// When Some, memory operations use this instead of Python callbacks.
    symbolic_memory: Option<SymbolicMemory>,
    /// Whether to use Rust-native memory (vs Python callbacks).
    use_rust_memory: bool,
    /// Symbolic register values (offset -> RustBV).
    /// These override the concrete `registers` storage for symbolic values.
    symbolic_registers: HashMap<u32, RustBV>,
    /// Store log: individual stores (address, size) from last execution.
    /// Used for precise memory sync - only sync specific bytes, not entire pages.
    store_log: Vec<(u64, usize)>,
    /// Binary code regions for native lifting: (addr, bytes).
    /// These are read-only regions containing executable code (.text, etc.)
    /// that can be accessed directly by the native lifter without callbacks.
    binary_regions: Vec<(u64, Vec<u8>)>,
    /// Whether native VEX lifting is initialized.
    native_lift_initialized: bool,
    /// Registered SimProcedures: address -> (name, num_args, no_return).
    /// This info is passed to the interpreter for pre-extracting arguments.
    simprocedures: HashMap<u64, (String, usize, bool)>,
    /// Address concretization configuration.
    /// Used to control how symbolic addresses are concretized for memory access.
    concretizer_config: AddressConcretizer,
    /// Whether profiling is enabled.
    profiling_enabled: bool,
    /// Accumulated execution statistics from all run_loop calls.
    accumulated_stats: crate::interpreter_cb::ExecutionStats,
}

#[pymethods]
impl RustVEXEngine {
    /// Create a new Rust VEX engine for the given architecture.
    #[new]
    #[pyo3(signature = (arch="amd64"))]
    pub fn new(arch: &str) -> PyResult<Self> {
        let arch_info = arch_from_name(arch)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported architecture: {}", arch)))?;

        let vex_arch = arch_info.vex_arch();
        let state_size = arch_info.state_size();

        Ok(RustVEXEngine {
            arch_name: arch.to_string(),
            vex_arch,
            block_cache: LruCache::new(NonZeroUsize::new(1024).expect("nonzero literal")),
            hooks: std::collections::HashSet::new(),
            registers: vec![0u8; state_size],
            pc: 0,
            memory_regions: Vec::new(),
            dirty_pages: std::collections::HashSet::new(),
            symbolic_count: 0,
            hooks_version: 0,
            memory_version: 0,
            callbacks: None,
            execution_config: ExecutionConfig::default(),
            dirty_registers: 0,
            symbolic_memory: None,
            use_rust_memory: false,
            symbolic_registers: HashMap::new(),
            store_log: Vec::new(),
            binary_regions: Vec::new(),
            native_lift_initialized: false,
            simprocedures: HashMap::new(),
            concretizer_config: AddressConcretizer::default(),
            profiling_enabled: false,
            accumulated_stats: crate::interpreter_cb::ExecutionStats::default(),
        })
    }

    /// Get the execution configuration.
    #[getter]
    pub fn execution_config(&self) -> ExecutionConfig {
        self.execution_config.clone()
    }

    /// Set the execution configuration.
    #[setter]
    pub fn set_execution_config(&mut self, config: ExecutionConfig) {
        self.execution_config = config;
    }

    /// Enable or disable deferred forks.
    pub fn set_use_deferred_forks(&mut self, enabled: bool) {
        self.execution_config.use_deferred_forks = enabled;
    }

    /// Set the maximum number of deferred forks before returning to Python.
    pub fn set_max_deferred_forks(&mut self, max: u32) {
        self.execution_config.max_deferred_forks = max;
    }

    /// Set the branch policy for symbolic branches.
    pub fn set_branch_policy(&mut self, policy: BranchPolicy) {
        self.execution_config.branch_policy = policy;
    }

    /// Configure address concretization to match Python's strategy (legacy interface).
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `range_limit` - Optional custom range limit (default: 1024)
    #[pyo3(signature = (use_approximate, range_limit=None))]
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.concretizer_config
            .configure(use_approximate, range_limit);
    }

    /// Configure address concretization with full Python strategy configuration.
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
        self.concretizer_config.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
        );
    }

    /// Enable or disable profiling.
    ///
    /// When enabled, execution statistics are collected and can be retrieved
    /// via `get_execution_stats()`.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
        if enabled {
            self.accumulated_stats = crate::interpreter_cb::ExecutionStats::default();
        }
    }

    /// Check if profiling is enabled.
    pub fn is_profiling_enabled(&self) -> bool {
        self.profiling_enabled
    }

    /// Get execution statistics as a dictionary.
    ///
    /// Returns a dictionary with timing and count statistics from all
    /// `run_loop` calls since profiling was enabled or last reset.
    #[pyo3(name = "get_execution_stats")]
    pub fn py_get_execution_stats(&self) -> HashMap<String, u64> {
        self.accumulated_stats.to_hashmap()
    }

    /// Reset execution statistics.
    pub fn reset_stats(&mut self) {
        self.accumulated_stats = crate::interpreter_cb::ExecutionStats::default();
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.arch_name
    }

    /// Get the program counter.
    #[getter]
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Set the program counter.
    #[setter]
    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }

    /// Add a hook at the given address.
    pub fn add_hook(&mut self, addr: u64) {
        if self.hooks.insert(addr) {
            self.hooks_version += 1;
        }
    }

    /// Remove a hook at the given address.
    pub fn remove_hook(&mut self, addr: u64) {
        if self.hooks.remove(&addr) {
            self.hooks_version += 1;
        }
    }

    /// Add multiple hooks at once (optimization).
    pub fn add_hooks(&mut self, addrs: Vec<u64>) {
        let mut changed = false;
        for addr in addrs {
            if self.hooks.insert(addr) {
                changed = true;
            }
        }
        if changed {
            self.hooks_version += 1;
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        if !self.hooks.is_empty() {
            self.hooks.clear();
            self.hooks_version += 1;
        }
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hooks.contains(&addr)
    }

    /// Register a SimProcedure at the given address.
    ///
    /// This allows the interpreter to pre-extract arguments when the hook is hit.
    /// Arguments:
    /// - addr: The hook address
    /// - name: SimProcedure name (e.g., "strlen", "malloc")
    /// - num_args: Number of arguments to extract
    /// - no_return: Whether this procedure never returns (e.g., "exit")
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
        self.hooks_version += 1;
    }

    /// Register multiple SimProcedures at once.
    ///
    /// Each tuple is (address, name, num_args, no_return).
    pub fn register_simprocedures(&mut self, procs: Vec<(u64, String, usize, bool)>) {
        for (addr, name, num_args, no_return) in procs {
            self.hooks.insert(addr);
            self.simprocedures.insert(addr, (name, num_args, no_return));
        }
        self.hooks_version += 1;
    }

    /// Clear all SimProcedure registrations.
    pub fn clear_simprocedures(&mut self) {
        self.simprocedures.clear();
    }

    /// Map a memory region.
    #[pyo3(signature = (addr, size, permissions=7))]
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: u8) {
        self.memory_regions.push((addr, size, permissions, vec![]));
        self.memory_version += 1;
    }

    /// Map memory with initial data.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: u8) {
        self.memory_regions
            .push((addr, data.len() as u64, permissions, data.to_vec()));
        self.memory_version += 1;
    }

    /// Clear all memory mappings.
    pub fn clear_memory(&mut self) {
        if !self.memory_regions.is_empty() {
            self.memory_regions.clear();
            self.memory_version += 1;
        }
    }

    /// Read memory.
    pub fn read_memory(&self, addr: u64, size: u64) -> PyResult<Vec<u8>> {
        // Find the region containing this address
        for (base, rsize, _, data) in &self.memory_regions {
            if addr >= *base && addr + size <= *base + *rsize {
                let offset = (addr - base) as usize;
                if !data.is_empty() && offset + size as usize <= data.len() {
                    return Ok(data[offset..offset + size as usize].to_vec());
                } else {
                    return Ok(vec![0u8; size as usize]);
                }
            }
        }
        Err(PyValueError::new_err(format!(
            "address 0x{:x} not mapped",
            addr
        )))
    }

    /// Write memory.
    pub fn write_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        // Find the region containing this address
        for (base, rsize, _, region_data) in &mut self.memory_regions {
            if addr >= *base && addr + data.len() as u64 <= *base + *rsize {
                let offset = (addr - *base) as usize;

                // Ensure data vector is large enough
                if region_data.len() < offset + data.len() {
                    region_data.resize(offset + data.len(), 0);
                }

                region_data[offset..offset + data.len()].copy_from_slice(data);

                // Track dirty pages (4KB page alignment)
                let page_size: u64 = 0x1000;
                let start_page = addr & !(page_size - 1);
                let end_addr = addr + data.len() as u64;
                let mut page = start_page;
                while page < end_addr {
                    self.dirty_pages.insert(page);
                    page += page_size;
                }

                return Ok(());
            }
        }
        Err(PyValueError::new_err(format!(
            "address 0x{:x} not mapped",
            addr
        )))
    }

    /// Get the list of dirty page addresses (pages that have been written to).
    /// Returns page-aligned addresses (4KB alignment).
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.dirty_pages.iter().copied().collect()
    }

    /// Clear the dirty pages set.
    pub fn clear_dirty_pages(&mut self) {
        self.dirty_pages.clear();
    }

    /// Get the store log: list of (address, size) tuples for stores during last execution.
    /// This allows precise memory sync - only sync specific bytes, not entire pages.
    pub fn get_store_log(&self) -> Vec<(u64, usize)> {
        self.store_log.clone()
    }

    /// Clear the store log.
    pub fn clear_store_log(&mut self) {
        self.store_log.clear();
    }

    /// Get a register value.
    /// Returns u128 to handle XMM and other large registers (Python BigInt handles this).
    pub fn get_register(&self, name: &str) -> PyResult<u128> {
        let arch = arch_from_name(&self.arch_name).expect("arch validated at construction");
        let offset = arch
            .register_offset(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?
            as usize;
        let size = arch
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register size: {}", name)))?
            as usize;

        if offset + size > self.registers.len() {
            return Err(PyValueError::new_err("register out of bounds"));
        }

        let mut value: u128 = 0;
        for i in 0..size {
            value |= (self.registers[offset + i] as u128) << (i * 8);
        }
        Ok(value)
    }

    /// Set a register value.
    /// Accepts u128 to handle XMM and other large registers (Python BigInt handles this).
    pub fn set_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        let arch = arch_from_name(&self.arch_name).expect("arch validated at construction");
        let offset = arch
            .register_offset(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?
            as usize;
        let size = arch
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register size: {}", name)))?
            as usize;

        if offset + size > self.registers.len() {
            return Err(PyValueError::new_err("register out of bounds"));
        }

        for i in 0..size {
            self.registers[offset + i] = (value >> (i * 8)) as u8;
        }
        Ok(())
    }

    /// Get all register values as a dictionary.
    pub fn get_registers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let arch = arch_from_name(&self.arch_name).expect("arch validated at construction");

        for name in arch.register_names() {
            if let Ok(value) = self.get_register(name) {
                dict.set_item(name, value)?;
            }
        }

        Ok(dict)
    }

    /// Get the entire register file as bytes (bulk transfer optimization).
    pub fn get_all_registers_raw(&self) -> Vec<u8> {
        self.registers.clone()
    }

    /// Set the entire register file from bytes (bulk transfer optimization).
    pub fn set_all_registers(&mut self, data: &[u8]) -> PyResult<()> {
        if data.len() != self.registers.len() {
            return Err(PyValueError::new_err(format!(
                "register data size mismatch: expected {}, got {}",
                self.registers.len(),
                data.len()
            )));
        }
        self.registers.copy_from_slice(data);
        Ok(())
    }

    /// Get the size of the register file in bytes.
    pub fn register_file_size(&self) -> usize {
        self.registers.len()
    }

    /// Get the offset of a register by name (for bulk transfer).
    pub fn get_register_offset(&self, name: &str) -> Option<u32> {
        let arch = arch_from_name(&self.arch_name)?;
        arch.register_offset(name)
    }

    /// Get the size of a register by name.
    pub fn get_register_size(&self, name: &str) -> Option<u32> {
        let arch = arch_from_name(&self.arch_name)?;
        arch.register_size(name)
    }

    /// Get list of dirty register offsets (registers modified since last clear).
    /// Returns a list of offsets that were written to during Rust execution.
    /// Each offset is 4-byte aligned.
    pub fn get_dirty_register_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for bit in 0..128u32 {
            if (self.dirty_registers & (1u128 << bit)) != 0 {
                offsets.push(bit * 4);
            }
        }
        offsets
    }

    /// Clear dirty register tracking (called after sync to Python).
    pub fn clear_dirty_registers(&mut self) {
        self.dirty_registers = 0;
    }

    /// Get a register value by offset (not by name).
    /// This is used for syncing specific registers back to Python.
    pub fn get_register_by_offset(&self, offset: u32, size: u32) -> PyResult<u128> {
        let offset = offset as usize;
        let size = size as usize;

        if offset + size > self.registers.len() {
            return Err(PyValueError::new_err(format!(
                "register offset {} + size {} exceeds register file size {}",
                offset,
                size,
                self.registers.len()
            )));
        }

        let mut value: u128 = 0;
        for i in 0..size {
            value |= (self.registers[offset + i] as u128) << (i * 8);
        }
        Ok(value)
    }

    /// Set a symbolic register value from a claripy AST.
    ///
    /// This allows Python to pass symbolic register values to the Rust engine.
    /// The AST is converted to a RustBV and stored for use during execution.
    ///
    /// Args:
    ///     offset: Register offset in the register file.
    ///     ast: A claripy AST representing the symbolic value.
    ///
    /// Note: Requires a solver context to be available during run_loop
    /// for the symbolic value to be properly used.
    pub fn set_symbolic_register(
        &mut self,
        py: Python<'_>,
        offset: u32,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Create a temporary context for AST conversion
        // The actual execution will use the solver context passed to run_loop
        let ctx = SymContext::new_mock();

        let bv = claripy_to_rustbv(py, ast, &ctx)
            .map_err(|e| PyValueError::new_err(format!("failed to convert AST: {}", e)))?;

        self.symbolic_registers.insert(offset, bv);
        Ok(())
    }

    /// Clear all symbolic register values.
    ///
    /// This should be called when registers are synced with fresh concrete values.
    pub fn clear_symbolic_registers(&mut self) {
        self.symbolic_registers.clear();
    }

    /// Check if a register has a symbolic value.
    pub fn has_symbolic_register(&self, offset: u32) -> bool {
        self.symbolic_registers.contains_key(&offset)
    }

    /// Get the number of symbolic registers.
    pub fn symbolic_register_count(&self) -> usize {
        self.symbolic_registers.len()
    }

    /// Start building a new IRSB at the given address.
    pub fn start_block(&mut self, addr: u64) -> PyResult<()> {
        // Create a new IRSB and store it in a temporary location
        // We'll build it up and then finalize it
        let irsb = IRSB::new(addr, self.vex_arch);
        self.block_cache.put(addr, irsb);
        Ok(())
    }

    /// Add an IMark statement to the current block.
    #[pyo3(signature = (block_addr, ins_addr, len, delta=0))]
    pub fn add_imark(
        &mut self,
        block_addr: u64,
        ins_addr: u64,
        len: u32,
        delta: u8,
    ) -> PyResult<()> {
        use crate::vex::ir::IRStmt;

        let irsb = self
            .block_cache
            .get_mut(&block_addr)
            .ok_or_else(|| PyValueError::new_err(format!("no block at 0x{:x}", block_addr)))?;

        irsb.statements.push(IRStmt::IMark {
            addr: ins_addr,
            len,
            delta,
        });
        Ok(())
    }

    /// Set the default exit for a block.
    #[pyo3(signature = (block_addr, next_addr, jumpkind))]
    pub fn set_block_exit(
        &mut self,
        block_addr: u64,
        next_addr: u64,
        jumpkind: &str,
    ) -> PyResult<()> {
        use crate::vex::ir::{IRConst, IRExpr, JumpKind};

        let irsb = self
            .block_cache
            .get_mut(&block_addr)
            .ok_or_else(|| PyValueError::new_err(format!("no block at 0x{:x}", block_addr)))?;

        irsb.next = IRExpr::Const(IRConst::U64(next_addr));
        irsb.jumpkind = match jumpkind {
            "Ijk_Boring" => JumpKind::Boring,
            "Ijk_Call" => JumpKind::Call,
            "Ijk_Ret" => JumpKind::Ret,
            "Ijk_Sys_syscall" => JumpKind::Sys_syscall,
            "Ijk_Sys_int128" => JumpKind::Sys_int128,
            "Ijk_Sys_int129" => JumpKind::Sys_int129,
            "Ijk_Sys_int130" => JumpKind::Sys_int130,
            "Ijk_Sys_sysenter" => JumpKind::Sys_sysenter,
            "Ijk_NoDecode" => JumpKind::NoDecode,
            "Ijk_MapFail" => JumpKind::MapFail,
            "Ijk_ClientReq" => JumpKind::ClientReq,
            "Ijk_Yield" => JumpKind::Yield,
            "Ijk_EmWarn" => JumpKind::EmWarn,
            "Ijk_EmFail" => JumpKind::EmFail,
            _ => JumpKind::Boring,
        };
        Ok(())
    }

    /// Check if a block exists in the cache.
    pub fn has_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Clear the block cache.
    pub fn clear_blocks(&mut self) {
        self.block_cache.clear();
    }

    /// Get the number of cached blocks.
    pub fn cached_block_count(&self) -> usize {
        self.block_cache.len()
    }

    /// Execute one block and return the execution event.
    pub fn step(&mut self) -> PyResult<ExecutionEvent> {
        // Check if we have a cached block at PC
        if let Some(irsb) = self.block_cache.peek(&self.pc) {
            return self.execute_cached_block(irsb.clone());
        }

        // No cached block - return that we need lifting
        Ok(ExecutionEvent {
            event_type: "need_lift".to_string(),
            next_addr: Some(self.pc),
            jumpkind: None,
            syscall_num: None,
            error: None,
            true_target: None,
            false_target: None,
        })
    }

    /// Set the Python callbacks for memory/hook/syscall handling.
    ///
    /// This enables the callback-based execution model where Rust calls
    /// back into Python for memory access and event handling.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear the Python callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
    }

    /// Check if callbacks are set.
    pub fn has_callbacks(&self) -> bool {
        self.callbacks.is_some()
    }

    /// Load binary code regions for native VEX lifting.
    ///
    /// Each region is a tuple of (start_addr, bytes).
    /// These regions are registered with libpyvex for direct byte access
    /// during native lifting, eliminating Python callbacks for code fetch.
    ///
    /// Call this once at initialization with the .text, .rodata, and other
    /// code sections from the loaded binary.
    #[pyo3(signature = (regions))]
    pub fn load_binary_regions(&mut self, regions: Vec<(u64, Vec<u8>)>) -> PyResult<()> {
        // Store regions for direct access
        self.binary_regions = regions;

        // Initialize native lifting if available
        #[cfg(feature = "native-lift")]
        {
            use crate::vex::libpyvex_ffi;

            // Initialize VEX if needed
            if let Err(e) = libpyvex_ffi::init_vex() {
                log::warn!("Failed to initialize native VEX lifting: {}", e);
                return Ok(());
            }

            // Clear any previously registered regions
            libpyvex_ffi::clear_binary_regions();

            // Register each region with libpyvex for const propagation
            for (addr, bytes) in &self.binary_regions {
                if !libpyvex_ffi::register_binary_region(*addr, bytes) {
                    log::warn!("Failed to register binary region at 0x{:x}", addr);
                }
            }

            self.native_lift_initialized = true;
            log::info!(
                "Loaded {} binary regions for native lifting ({} total bytes)",
                self.binary_regions.len(),
                self.binary_regions
                    .iter()
                    .map(|(_, b)| b.len())
                    .sum::<usize>()
            );
        }

        Ok(())
    }

    /// Get bytes from binary regions at the given address.
    ///
    /// Returns None if the address is not in any binary region.
    pub fn get_binary_bytes(&self, addr: u64, size: usize) -> Option<Vec<u8>> {
        for (region_addr, bytes) in &self.binary_regions {
            let region_end = *region_addr + bytes.len() as u64;
            if addr >= *region_addr && addr + size as u64 <= region_end {
                let offset = (addr - *region_addr) as usize;
                return Some(bytes[offset..offset + size].to_vec());
            }
        }
        None
    }

    /// Check if native VEX lifting is available and initialized.
    #[getter]
    pub fn native_lift_available(&self) -> bool {
        #[cfg(feature = "native-lift")]
        {
            self.native_lift_initialized && crate::vex::libpyvex_ffi::is_vex_initialized()
        }
        #[cfg(not(feature = "native-lift"))]
        {
            false
        }
    }

    /// Run the execution loop until an event requires Python handling.
    ///
    /// This is the main entry point for the callback-based execution model.
    /// It runs multiple blocks in a loop, using Python callbacks for memory
    /// access, until it hits a condition that requires Python-side handling
    /// (hook, syscall, symbolic branch, max blocks, etc.).
    ///
    /// Args:
    ///     max_blocks: Maximum number of blocks to execute before returning.
    ///     solver_ctx: Optional RustSolverContext to use for constraint solving.
    ///                 When provided, the engine shares this context with the solver,
    ///                 ensuring branch constraints are properly tracked.
    ///
    /// Returns:
    ///     LoopExecutionEvent describing why execution stopped, including any
    ///     deferred forks collected during execution.
    #[pyo3(signature = (max_blocks=100, solver_ctx=None))]
    pub fn run_loop(
        &mut self,
        py: Python<'_>,
        max_blocks: u32,
        solver_ctx: Option<&RustSolverContext>,
    ) -> PyResult<LoopExecutionEvent> {
        // Ensure callbacks are set
        let callbacks = self.callbacks.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("callbacks not set - call set_callbacks() first")
        })?;

        // Check if callbacks are ready
        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err(
                "callbacks not ready - ensure memory_load, memory_store, and lift_block are set",
            ));
        }

        // Use provided solver context or create a mock one.
        // When solver_ctx is provided, constraints from branch decisions are
        // automatically added to the shared solver context.
        let default_ctx;
        let ctx: &SymContext = if let Some(solver) = solver_ctx {
            match solver.sym_context() {
                Some(ctx) => ctx,
                None => {
                    return Err(PyRuntimeError::new_err(
                        "sym_context() not available on shared solver contexts",
                    ));
                }
            }
        } else {
            default_ctx = SymContext::new_mock();
            &default_ctx
        };

        // Create the callback-aware interpreter with config
        let mut interp =
            CallbackInterpreter::with_config(self.vex_arch, ctx, self.execution_config.clone());

        // Enable profiling on interpreter if enabled on engine
        if self.profiling_enabled {
            interp.set_profiling(true);
        }

        // Set symbol table for handle-based claripy bypass
        // This allows Python to return RustBVHandle instead of claripy ASTs
        if let Some(solver) = solver_ctx {
            interp.set_symbol_table(solver.symbol_table());
        }

        // Set concretizer configuration to match Python's strategy
        interp.set_concretizer(self.concretizer_config.clone());

        // Copy registers from engine to interpreter
        interp.registers.copy_from_bytes(&self.registers);

        // Copy symbolic registers to interpreter
        for (&offset, bv) in &self.symbolic_registers {
            interp.registers.put(offset, bv.clone());
        }

        // Set PC
        interp.set_pc(self.pc);

        // Set up hooks
        for &addr in &self.hooks {
            interp.add_hook(addr);
        }

        // Register SimProcedures with their names and argument counts
        for (addr, (name, num_args, no_return)) in &self.simprocedures {
            interp.register_simprocedure(*addr, name.clone(), *num_args, *no_return);
        }

        // Copy concrete memory regions for fast local access
        for (base, _size, _perms, data) in &self.memory_regions {
            if !data.is_empty() {
                interp.add_concrete_memory(*base, data.clone());
            }
        }

        // Also copy binary regions for native VEX lifting
        // These are read-only code sections (.text, etc.) loaded via load_binary_regions()
        for (base, data) in &self.binary_regions {
            if !data.is_empty() {
                interp.add_concrete_memory(*base, data.clone());
            }
        }

        // Pass Rust memory to interpreter if enabled
        if self.use_rust_memory {
            if let Some(mem) = self.symbolic_memory.take() {
                interp.set_rust_memory(mem);
            }
        }

        // Run the execution loop
        let (result, blocks_executed, deferred_forks) =
            interp.run_until_event(py, callbacks, max_blocks);

        // angr-h0dv: The post-loop Rust→Python constraint push was removed
        // after a 20-bench soak proved it was dead code. Path A
        // (rust_solver_ctx attach in rust_callback_dispatch.py) covers every
        // live callback site. Clear any tracked constraints so the next
        // exploration window starts clean.
        if interp.has_pending_constraints() {
            interp.clear_pending_constraints();
        }

        // Update engine state from interpreter
        self.pc = interp.get_pc();
        interp.registers.copy_to_bytes(&mut self.registers);

        // Copy dirty register tracking from interpreter
        self.dirty_registers = interp.dirty_registers();

        // Recover Rust memory from interpreter
        if let Some(mem) = interp.take_rust_memory() {
            self.symbolic_memory = Some(mem);
        }

        // Get the current push level for constraint tracking
        let push_level = interp.push_level();

        // Merge profiling stats from interpreter
        if self.profiling_enabled {
            self.accumulated_stats.merge(interp.stats());
        }

        // Convert to Python event with deferred forks and push level
        Ok(LoopExecutionEvent::from_run_result_with_forks(
            result,
            blocks_executed,
            deferred_forks,
            push_level,
        ))
    }

    /// Run a single block with callbacks and return the event.
    ///
    /// This is like run_loop but only executes one block. Useful for
    /// step-by-step debugging or when you want finer control.
    pub fn step_with_callbacks(&mut self, py: Python<'_>) -> PyResult<LoopExecutionEvent> {
        self.run_loop(py, 1, None)
    }

    /// Execute an IRSB from JSON (serialized pyvex IRSB).
    ///
    /// This is the main entry point for executing lifted code from Python.
    /// The IRSB is deserialized from JSON and executed directly.
    pub fn execute_irsb_json(&mut self, irsb_json: &str) -> PyResult<ExecutionEvent> {
        // Deserialize the IRSB
        let irsb = deserialize_irsb(irsb_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to deserialize IRSB: {}", e)))?;

        // Cache it for potential re-execution
        let addr = irsb.addr;
        self.block_cache.put(addr, irsb.clone());

        // Execute the block
        self.execute_cached_block(irsb)
    }

    /// Fork the engine state.
    pub fn fork(&self) -> PyResult<Self> {
        Ok(RustVEXEngine {
            arch_name: self.arch_name.clone(),
            vex_arch: self.vex_arch,
            // New cache for forked state - they may have different access patterns
            block_cache: LruCache::new(NonZeroUsize::new(1024).expect("nonzero literal")),
            hooks: self.hooks.clone(),
            registers: self.registers.clone(),
            pc: self.pc,
            memory_regions: self.memory_regions.clone(),
            dirty_pages: self.dirty_pages.clone(),
            symbolic_count: self.symbolic_count,
            hooks_version: self.hooks_version,
            memory_version: self.memory_version,
            // Callbacks are shared (Python objects are reference-counted)
            callbacks: self.callbacks.clone(),
            // Copy execution config
            execution_config: self.execution_config.clone(),
            // Start with clean dirty tracking for fork
            dirty_registers: 0,
            // Fork symbolic memory with O(1) CoW
            symbolic_memory: self.symbolic_memory.as_ref().map(|m| m.fork()),
            use_rust_memory: self.use_rust_memory,
            // Clone symbolic registers for fork
            symbolic_registers: self.symbolic_registers.clone(),
            // Start with empty store log for fork
            store_log: Vec::new(),
            // Share binary regions (read-only, no need to clone data)
            binary_regions: self.binary_regions.clone(),
            // Native lift is already initialized if parent had it
            native_lift_initialized: self.native_lift_initialized,
            // Share SimProcedure registry (same for all forks)
            simprocedures: self.simprocedures.clone(),
            // Copy concretizer configuration
            concretizer_config: self.concretizer_config.clone(),
            // Fork inherits profiling setting but starts fresh stats
            profiling_enabled: self.profiling_enabled,
            accumulated_stats: crate::interpreter_cb::ExecutionStats::default(),
        })
    }

    /// Create a state snapshot for debugging.
    pub fn snapshot(&self) -> StateSnapshot {
        let arch = arch_from_name(&self.arch_name).expect("arch validated at construction");
        let mut registers = HashMap::new();

        for name in arch.register_names() {
            if let (Some(offset), Some(size)) =
                (arch.register_offset(name), arch.register_size(name))
            {
                let offset = offset as usize;
                let size = size as usize;
                if offset + size <= self.registers.len() {
                    registers.insert(
                        name.to_string(),
                        self.registers[offset..offset + size].to_vec(),
                    );
                }
            }
        }

        StateSnapshot {
            registers,
            pc: self.pc,
        }
    }

    /// Get engine statistics.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("cached_blocks", self.block_cache.len())?;
        dict.set_item("hooks", self.hooks.len())?;
        dict.set_item("memory_regions", self.memory_regions.len())?;
        dict.set_item("symbolic_count", self.symbolic_count)?;
        dict.set_item("register_file_size", self.registers.len())?;
        dict.set_item("hooks_version", self.hooks_version)?;
        dict.set_item("memory_version", self.memory_version)?;
        dict.set_item("rust_memory_enabled", self.use_rust_memory)?;
        dict.set_item("symbolic_registers", self.symbolic_registers.len())?;
        if let Some(ref mem) = self.symbolic_memory {
            dict.set_item("rust_memory_pages", mem.page_count())?;
            dict.set_item("rust_memory_dirty_pages", mem.get_dirty_pages().len())?;
        }
        Ok(dict)
    }

    // ======================================================================
    // Rust-native Memory Model APIs
    // ======================================================================

    /// Create a new Rust-native symbolic memory model.
    ///
    /// This creates a fresh SymbolicMemory that can be used instead of
    /// Python callbacks for memory operations, providing significant
    /// performance improvements.
    ///
    /// Args:
    ///     little_endian: If True, use little-endian byte order (default True).
    #[pyo3(signature = (little_endian=true))]
    pub fn create_rust_memory(&mut self, little_endian: bool) -> PyResult<()> {
        let endness = if little_endian {
            Endness::Little
        } else {
            Endness::Big
        };
        self.symbolic_memory = Some(SymbolicMemory::new(endness));
        Ok(())
    }

    /// Enable Rust-native memory mode.
    ///
    /// When enabled, memory operations will try to use Rust SymbolicMemory
    /// first, falling back to Python callbacks only for unmapped regions.
    pub fn enable_rust_memory(&mut self) {
        self.use_rust_memory = true;
    }

    /// Disable Rust-native memory mode.
    pub fn disable_rust_memory(&mut self) {
        self.use_rust_memory = false;
    }

    /// Check if Rust-native memory is enabled.
    pub fn is_rust_memory_enabled(&self) -> bool {
        self.use_rust_memory && self.symbolic_memory.is_some()
    }

    /// Map a memory region in Rust memory.
    ///
    /// Args:
    ///     addr: Base address to map (will be page-aligned).
    ///     size: Size of region to map.
    ///     permissions: Permission bits (R=4, W=2, X=1).
    #[pyo3(signature = (addr, size, permissions=7))]
    pub fn map_rust_memory(&mut self, addr: u64, size: u64, permissions: u8) -> PyResult<()> {
        if let Some(ref mut mem) = self.symbolic_memory {
            mem.map(addr, size, Permission::from_bits(permissions));
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Rust memory not created - call create_rust_memory() first",
            ))
        }
    }

    /// Map memory with initial data in Rust memory.
    ///
    /// Args:
    ///     addr: Base address to map.
    ///     data: Initial data bytes.
    ///     permissions: Permission bits (R=4, W=2, X=1).
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn map_rust_memory_data(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        if let Some(ref mut mem) = self.symbolic_memory {
            mem.map_data(addr, data, Permission::from_bits(permissions));
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Rust memory not created - call create_rust_memory() first",
            ))
        }
    }

    /// Store a concrete value in Rust memory.
    ///
    /// Args:
    ///     addr: Address to store at.
    ///     data: Bytes to store (little-endian).
    pub fn store_rust_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        use crate::symbolic::RustBV;

        if let Some(ref mut mem) = self.symbolic_memory {
            let bits = (data.len() * 8) as u32;
            let mut value: u128 = 0;
            for (i, &byte) in data.iter().enumerate() {
                value |= (byte as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, bits);
            mem.store_concrete(addr, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            Err(PyValueError::new_err("Rust memory not created"))
        }
    }

    /// Load a concrete value from Rust memory.
    ///
    /// Args:
    ///     addr: Address to load from.
    ///     size: Number of bytes to load.
    ///
    /// Returns:
    ///     Bytes in little-endian order.
    pub fn load_rust_memory(&self, addr: u64, size: u64) -> PyResult<Vec<u8>> {
        use crate::symbolic::SymContext;

        if let Some(ref mem) = self.symbolic_memory {
            let ctx = SymContext::new_mock();
            let bv = mem
                .load_concrete(addr, size as u32, &ctx)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            // Convert to bytes
            let value = bv.to_u128();
            let mut bytes = Vec::with_capacity(size as usize);
            for i in 0..size {
                bytes.push((value >> (i * 8)) as u8);
            }
            Ok(bytes)
        } else {
            Err(PyValueError::new_err("Rust memory not created"))
        }
    }

    /// Get list of dirty page addresses in Rust memory.
    ///
    /// Returns page-aligned addresses for pages that have been
    /// modified since the last clear.
    pub fn get_rust_memory_dirty_pages(&self) -> Vec<u64> {
        if let Some(ref mem) = self.symbolic_memory {
            mem.get_dirty_page_addrs()
        } else {
            Vec::new()
        }
    }

    /// Clear dirty page tracking in Rust memory.
    pub fn clear_rust_memory_dirty_pages(&mut self) {
        if let Some(ref mut mem) = self.symbolic_memory {
            mem.clear_dirty_pages();
        }
    }

    /// Get data for a specific page in Rust memory.
    ///
    /// Args:
    ///     page_addr: Page-aligned address.
    ///
    /// Returns:
    ///     Tuple of (data_bytes, permissions) or None if not mapped.
    pub fn get_rust_memory_page(&self, page_addr: u64) -> Option<(Vec<u8>, u8)> {
        let page_num = page_addr >> 12;
        if let Some(ref mem) = self.symbolic_memory {
            mem.get_page_data(page_num)
        } else {
            None
        }
    }

    /// Get statistics about Rust memory.
    pub fn rust_memory_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("enabled", self.use_rust_memory)?;

        if let Some(ref mem) = self.symbolic_memory {
            dict.set_item("page_count", mem.page_count())?;
            dict.set_item("mapped_bytes", mem.mapped_size())?;
            dict.set_item("dirty_pages", mem.get_dirty_pages().len())?;
            dict.set_item("lazy_regions", mem.lazy_region_count())?;
        } else {
            dict.set_item("page_count", 0)?;
            dict.set_item("mapped_bytes", 0)?;
            dict.set_item("dirty_pages", 0)?;
            dict.set_item("lazy_regions", 0)?;
        }

        Ok(dict)
    }

    /// Add a lazy region for on-demand page fetching.
    ///
    /// Pages in this region will be fetched from Python when accessed,
    /// rather than being pre-loaded. This is more efficient for large
    /// address spaces like stack and heap.
    ///
    /// Args:
    ///     start_addr: Start address of the region.
    ///     size: Size of the region in bytes.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) -> PyResult<()> {
        if let Some(ref mut mem) = self.symbolic_memory {
            mem.add_lazy_region(start_addr, size);
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Rust memory not created - call create_rust_memory() first",
            ))
        }
    }

    /// Clear all lazy regions.
    pub fn clear_lazy_regions(&mut self) {
        if let Some(ref mut mem) = self.symbolic_memory {
            mem.clear_lazy_regions();
        }
    }
}

impl RustVEXEngine {
    /// Execute a cached block.
    fn execute_cached_block(&mut self, irsb: IRSB) -> PyResult<ExecutionEvent> {
        // Create a fresh context for this execution
        let ctx = SymContext::new_mock();
        // Create interpreter
        let mut interp = VEXInterpreter::new(self.vex_arch, &ctx);

        // Copy registers from engine to interpreter
        interp.registers.copy_from_bytes(&self.registers);

        // Copy symbolic registers to interpreter (critical for branch detection)
        // Without this, symbolic values are lost and branches appear concrete
        for (&offset, bv) in &self.symbolic_registers {
            interp.registers.put(offset, bv.clone());
        }

        // Set up memory
        for (addr, size, perms, data) in &self.memory_regions {
            let perm = Permission::from_bits(*perms);
            if data.is_empty() {
                interp.memory.map(*addr, *size, perm);
            } else {
                interp.memory.map_data(*addr, data, perm);
            }
        }

        // Set up hooks
        for &addr in &self.hooks {
            interp.add_hook(addr);
        }

        // Set PC
        interp.set_pc(self.pc);

        // Execute
        match interp.execute_block(&irsb) {
            Ok(result) => {
                // Update our state from interpreter
                self.pc = interp.get_pc();

                // Sync registers back from interpreter to engine
                interp.registers.copy_to_bytes(&mut self.registers);

                // Copy dirty register tracking from interpreter
                self.dirty_registers = interp.dirty_registers();

                // Copy store log from interpreter - this allows precise memory sync
                self.store_log = interp.get_store_log().to_vec();

                // Sync dirty pages from interpreter's memory back to engine
                // This is critical: the interpreter has its own SymbolicMemory,
                // and stores during execution write to that memory. We need to
                // copy those changes back to the engine's memory_regions so that
                // Python can read the updated values via get_dirty_pages().
                let dirty_page_addrs = interp.memory.get_dirty_page_addrs();
                for page_addr in dirty_page_addrs {
                    // Mark this page as dirty in the engine
                    self.dirty_pages.insert(page_addr);

                    // Find the page data in interpreter's memory and copy to engine's memory_regions
                    let page_num = page_addr >> 12;
                    if let Some((page_data, _perms)) = interp.memory.get_page_data(page_num) {
                        // Find the region containing this page and update it
                        for (region_addr, region_size, _region_perms, region_data) in
                            &mut self.memory_regions
                        {
                            if page_addr >= *region_addr && page_addr < *region_addr + *region_size
                            {
                                let offset = (page_addr - *region_addr) as usize;
                                let copy_len =
                                    std::cmp::min(page_data.len(), region_data.len() - offset);
                                if copy_len > 0 && offset < region_data.len() {
                                    region_data[offset..offset + copy_len]
                                        .copy_from_slice(&page_data[..copy_len]);
                                }
                                break;
                            }
                        }
                    }
                }

                Ok(ExecutionEvent::from_result(result))
            }
            Err(e) => Ok(ExecutionEvent::error(format!("{}", e))),
        }
    }
}

/// Set the Rust Z3 thread-local context to share Python's Z3 context.
///
/// This enables Rust and Python to share Z3 ASTs without translation.
/// The pointer must be a valid Z3_context created by Python's z3 module.
/// Call this once at startup, before creating any Rust solver contexts.
#[cfg(feature = "vex-engine-z3")]
#[pyfunction]
fn set_shared_z3_context(py_z3_ctx_ptr: usize) -> PyResult<bool> {
    if py_z3_ctx_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Z3 context pointer is null",
        ));
    }

    // Create a z3-rs Context from the raw pointer.
    // SAFETY: The pointer comes from Python's z3.main_ctx().ctx.value which
    // is a valid Z3_context created by Z3_mk_context_rc(). Python owns this
    // context and will delete it at process exit. The Context::from_raw
    // constructor marks it as "borrowed" so ContextInternal::drop will NOT
    // call Z3_del_context — Python retains ownership.
    unsafe {
        let raw_ctx = std::ptr::NonNull::new_unchecked(py_z3_ctx_ptr as *mut _);
        let ctx = z3::Context::from_raw(raw_ctx);
        z3::Context::set_thread_local(&ctx);
        // No need to forget — from_raw marks the context as borrowed,
        // so Z3_del_context is never called from Rust's side.
    }

    Ok(true)
}

/// Reset the Rust thread-local Z3 context to a fresh Rust-owned context.
/// Call this before Python's Z3 context is freed (e.g., via atexit) to
/// prevent use-after-free during process shutdown.
#[cfg(feature = "vex-engine-z3")]
#[pyfunction]
fn reset_shared_z3_context() -> PyResult<()> {
    // Replace the thread-local with a fresh Rust-owned context.
    // The old thread-local (pointing to Python's context) has a leaked Rc
    // (refcount stays at 1 after this, ContextInternal::drop never called).
    let fresh = z3::Context::new(&z3::Config::new());
    z3::Context::set_thread_local(&fresh);
    Ok(())
}

/// Minimal stderr logger for Rust log messages.
struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!(
                "[rust:{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }
    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

/// Set the Rust log level from Python.
///
/// Valid levels: "error", "warn", "info", "debug", "trace", "off".
/// Initializes a stderr logger on first call.
#[pyfunction]
#[pyo3(signature = (level="info"))]
fn set_rust_log_level(level: &str) -> PyResult<()> {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = log::set_logger(&LOGGER);
    });

    let filter = match level.to_lowercase().as_str() {
        "error" => log::LevelFilter::Error,
        "warn" | "warning" => log::LevelFilter::Warn,
        "info" => log::LevelFilter::Info,
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        "off" => log::LevelFilter::Off,
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid log level '{}': use error/warn/info/debug/trace/off",
                level
            )));
        }
    };
    log::set_max_level(filter);
    Ok(())
}

/// Get the size in BYTES of a register on the given architecture.
///
/// Returns `None` for unknown arch names or registers not modelled by Rust.
/// Arch name follows the same case-insensitive conventions as
/// `RustSimState::new` (e.g. "AMD64", "aarch64", "armel", "mips32"). Used by
/// the Python `RustRegisterProxy` to derive register widths from the single
/// Rust source of truth instead of hardcoding prefix-based heuristics.
#[pyfunction]
fn register_size_for_arch(arch_name: &str, reg_name: &str) -> Option<u32> {
    arch_from_name(arch_name).and_then(|a| a.register_size(reg_name))
}

/// Get the canonical register name list for the given architecture.
///
/// Returns an empty vec for unknown arch names. Names are the canonical
/// Rust-side identifiers (e.g. "rax", "x0", "v0") — the same set the
/// interpreter uses. Callers that need a narrower sync subset (e.g.
/// excluding XMM/CC flags) must filter further.
#[pyfunction]
fn register_names_for_arch(arch_name: &str) -> Vec<String> {
    arch_from_name(arch_name)
        .map(|a| a.register_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

/// Clear the Rust-side thread-local claripy AST translation caches.
///
/// Drops the four LRU/HashMaps in `claripy_bridge` (AST_CACHE,
/// CLARIPY_AST_CACHE, EXPRESSION_CACHE, EXPRESSION_BY_OPERANDS_PTR).
/// Used by `RustExplorationManager.cleanup()` to bound per-process
/// growth in Callable-heavy workloads where many short-lived managers
/// share the same thread (e.g. mma_howtouse's 45 invocations).
///
/// Does NOT clear the global `SymbolicIdentityRegistry` because that
/// is shared across all managers in the process; clearing it from one
/// manager would invalidate live symbol IDs held by another.
#[pyfunction]
fn clear_ast_cache() {
    crate::claripy_bridge::clear_ast_cache();
}

/// Register the VEX engine module with Python.
pub fn vex_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustVEXEngine>()?;
    m.add_class::<ExecutionEvent>()?;
    m.add_class::<StateSnapshot>()?;
    m.add_class::<PythonCallbacks>()?;
    m.add_class::<LoopExecutionEvent>()?;
    m.add_class::<RustSolverContext>()?;
    // Handle-based API for claripy bypass
    m.add_class::<crate::symbolic::RustBVHandle>()?;
    // Deferred fork types
    m.add_class::<DeferredFork>()?;
    m.add_class::<BranchPolicy>()?;
    m.add_class::<ExecutionConfig>()?;
    // Rust-first state
    m.add_class::<crate::state::PyRustSimState>()?;
    // State snapshot for exploration export
    m.add_class::<crate::state::ExplorationStateSnapshot>()?;
    // Exploration manager
    crate::exploration::register_exploration(m)?;
    // Z3 context sharing
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(set_shared_z3_context, m)?)?;
    #[cfg(feature = "vex-engine-z3")]
    m.add_function(pyo3::wrap_pyfunction!(reset_shared_z3_context, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(set_rust_log_level, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(register_size_for_arch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(register_names_for_arch, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(clear_ast_cache, m)?)?;
    // Memory layout constants (single source of truth; Python imports these
    // rather than redeclaring 0x1000 etc.).
    m.add("PAGE_SIZE", crate::memory::PAGE_SIZE)?;
    m.add("PAGE_MASK", crate::memory::PAGE_MASK)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            let engine = RustVEXEngine::new("amd64").unwrap();
            assert_eq!(engine.arch(), "amd64");
            assert_eq!(engine.pc(), 0);
        });
    }

    #[test]
    fn test_register_access() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            let mut engine = RustVEXEngine::new("amd64").unwrap();

            engine.set_register("rax", 0x12345678).unwrap();
            assert_eq!(engine.get_register("rax").unwrap(), 0x12345678);

            engine.set_register("eax", 0xABCD).unwrap();
            // EAX is the low 32 bits of RAX
            assert_eq!(engine.get_register("eax").unwrap(), 0xABCD);
        });
    }

    #[test]
    fn test_memory_access() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            let mut engine = RustVEXEngine::new("amd64").unwrap();

            // Map memory
            engine.map_memory(0x1000, 0x1000, 7);

            // Write and read
            engine.write_memory(0x1000, &[1, 2, 3, 4]).unwrap();
            let data = engine.read_memory(0x1000, 4).unwrap();
            assert_eq!(data, vec![1, 2, 3, 4]);
        });
    }

    #[test]
    fn test_x86_addss_full_flow() {
        // This tests the full execute_irsb_json flow for x86 ADDSS
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut engine = RustVEXEngine::new("x86").unwrap();

            // Set XMM0 = 1.0f, XMM1 = 2.0f
            let f1_bits = 1.0f32.to_bits() as u128;
            let f2_bits = 2.0f32.to_bits() as u128;
            engine.set_register("xmm0", f1_bits).unwrap();
            engine.set_register("xmm1", f2_bits).unwrap();

            println!(
                "Before: xmm0 = 0x{:x}",
                engine.get_register("xmm0").unwrap()
            );
            println!(
                "Before: xmm1 = 0x{:x}",
                engine.get_register("xmm1").unwrap()
            );
            println!("Engine registers size: {}", engine.registers.len());

            // Map memory for the instruction
            engine.map_memory(0x1000, 0x1000, 7);
            engine.map_memory_data(0x1000, &[0xf3, 0x0f, 0x58, 0xc1], 7); // ADDSS xmm0, xmm1

            // Set PC
            engine.pc = 0x1000;

            // ADDSS xmm0, xmm1 VEX IR JSON
            let irsb_json = r#"{
                "addr": 4096,
                "arch": "X86",
                "statements": [
                    {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                    {"tag": "Ist_WrTmp", "tmp": 1, "data": {"tag": "Iex_Get", "offset": 176, "ty": "Ity_V128"}},
                    {"tag": "Ist_WrTmp", "tmp": 2, "data": {"tag": "Iex_Get", "offset": 160, "ty": "Ity_V128"}},
                    {"tag": "Ist_WrTmp", "tmp": 0, "data": {
                        "tag": "Iex_Binop",
                        "op": "Iop_Add32F0x4",
                        "args": [
                            {"tag": "Iex_RdTmp", "tmp": 2},
                            {"tag": "Iex_RdTmp", "tmp": 1}
                        ]
                    }},
                    {"tag": "Ist_Put", "offset": 160, "data": {"tag": "Iex_RdTmp", "tmp": 0}}
                ],
                "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U32", "value": 4100}},
                "jumpkind": "Ijk_Boring",
                "offsIP": 68,
                "tyenv": {"types": ["Ity_V128", "Ity_V128", "Ity_V128", "Ity_I32"]}
            }"#;

            // Execute
            let result = engine.execute_irsb_json(irsb_json).unwrap();
            println!(
                "Result: event_type={}, error={:?}",
                result.event_type, result.error
            );

            // Check result
            let xmm0_after = engine.get_register("xmm0").unwrap();
            println!("After: xmm0 = 0x{:x}", xmm0_after);

            let expected = 3.0f32.to_bits() as u128;
            assert_eq!(
                xmm0_after & 0xFFFFFFFF,
                expected,
                "ADDSS should produce 3.0f"
            );
        });
    }
}
