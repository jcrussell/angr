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
use crate::callbacks::{BranchPolicy, DeferredFork, ExecutionConfig, LoopExecutionEvent, PythonCallbacks, RunResult};
use crate::interpreter::{ExecutionResult, VEXInterpreter};
use crate::interpreter_cb::CallbackInterpreter;
use crate::memory::Permission;
use crate::solver::RustSolverContext;
use crate::symbolic::SymContext;
use crate::vex::{deserialize_irsb, VexArch, IRSB};

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
            ExecutionResult::BlockEnd { next_addr, jumpkind } => ExecutionEvent {
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
#[pyclass]
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
}

#[pymethods]
impl RustVEXEngine {
    /// Create a new Rust VEX engine for the given architecture.
    #[new]
    #[pyo3(signature = (arch="amd64"))]
    pub fn new(arch: &str) -> PyResult<Self> {
        let arch_info = arch_from_name(arch).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported architecture: {}", arch))
        })?;

        let vex_arch = arch_info.vex_arch();
        let state_size = arch_info.state_size();

        Ok(RustVEXEngine {
            arch_name: arch.to_string(),
            vex_arch,
            block_cache: LruCache::new(NonZeroUsize::new(1024).unwrap()),
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

    /// Get a register value.
    /// Returns u128 to handle XMM and other large registers (Python BigInt handles this).
    pub fn get_register(&self, name: &str) -> PyResult<u128> {
        let arch = arch_from_name(&self.arch_name).unwrap();
        let offset = arch.register_offset(name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown register: {}", name))
        })? as usize;
        let size = arch.register_size(name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown register size: {}", name))
        })? as usize;

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
        let arch = arch_from_name(&self.arch_name).unwrap();
        let offset = arch.register_offset(name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown register: {}", name))
        })? as usize;
        let size = arch.register_size(name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown register size: {}", name))
        })? as usize;

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
        let arch = arch_from_name(&self.arch_name).unwrap();

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
                offset, size, self.registers.len()
            )));
        }

        let mut value: u128 = 0;
        for i in 0..size {
            value |= (self.registers[offset + i] as u128) << (i * 8);
        }
        Ok(value)
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
    pub fn add_imark(&mut self, block_addr: u64, ins_addr: u64, len: u32, delta: u8) -> PyResult<()> {
        use crate::vex::ir::IRStmt;

        let irsb = self.block_cache.get_mut(&block_addr).ok_or_else(|| {
            PyValueError::new_err(format!("no block at 0x{:x}", block_addr))
        })?;

        irsb.statements.push(IRStmt::IMark {
            addr: ins_addr,
            len,
            delta,
        });
        Ok(())
    }

    /// Set the default exit for a block.
    #[pyo3(signature = (block_addr, next_addr, jumpkind))]
    pub fn set_block_exit(&mut self, block_addr: u64, next_addr: u64, jumpkind: &str) -> PyResult<()> {
        use crate::vex::ir::{IRConst, IRExpr, JumpKind};

        let irsb = self.block_cache.get_mut(&block_addr).ok_or_else(|| {
            PyValueError::new_err(format!("no block at 0x{:x}", block_addr))
        })?;

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

    /// Run the execution loop until an event requires Python handling.
    ///
    /// This is the main entry point for the callback-based execution model.
    /// It runs multiple blocks in a loop, using Python callbacks for memory
    /// access, until it hits a condition that requires Python-side handling
    /// (hook, syscall, symbolic branch, max blocks, etc.).
    ///
    /// Args:
    ///     max_blocks: Maximum number of blocks to execute before returning.
    ///
    /// Returns:
    ///     LoopExecutionEvent describing why execution stopped, including any
    ///     deferred forks collected during execution.
    #[pyo3(signature = (max_blocks=100))]
    pub fn run_loop(&mut self, py: Python<'_>, max_blocks: u32) -> PyResult<LoopExecutionEvent> {
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

        // Create the callback-aware interpreter with config
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::with_config(
            self.vex_arch,
            &ctx,
            self.execution_config.clone(),
        );

        // Copy registers from engine to interpreter
        interp.registers.copy_from_bytes(&self.registers);

        // Set PC
        interp.set_pc(self.pc);

        // Set up hooks
        for &addr in &self.hooks {
            interp.add_hook(addr);
        }

        // Copy concrete memory regions for fast local access
        for (base, _size, _perms, data) in &self.memory_regions {
            if !data.is_empty() {
                interp.add_concrete_memory(*base, data.clone());
            }
        }

        // Run the execution loop
        let (result, blocks_executed, deferred_forks) = interp.run_until_event(py, callbacks, max_blocks);

        // Update engine state from interpreter
        self.pc = interp.get_pc();
        interp.registers.copy_to_bytes(&mut self.registers);

        // Copy dirty register tracking from interpreter
        self.dirty_registers = interp.dirty_registers();

        // Convert to Python event with deferred forks
        Ok(LoopExecutionEvent::from_run_result_with_forks(result, blocks_executed, deferred_forks))
    }

    /// Run a single block with callbacks and return the event.
    ///
    /// This is like run_loop but only executes one block. Useful for
    /// step-by-step debugging or when you want finer control.
    pub fn step_with_callbacks(&mut self, py: Python<'_>) -> PyResult<LoopExecutionEvent> {
        self.run_loop(py, 1)
    }

    /// Execute an IRSB from JSON (serialized pyvex IRSB).
    ///
    /// This is the main entry point for executing lifted code from Python.
    /// The IRSB is deserialized from JSON and executed directly.
    pub fn execute_irsb_json(&mut self, irsb_json: &str) -> PyResult<ExecutionEvent> {
        // Deserialize the IRSB
        let irsb = deserialize_irsb(irsb_json).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to deserialize IRSB: {}", e))
        })?;

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
            block_cache: LruCache::new(NonZeroUsize::new(1024).unwrap()),
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
        })
    }

    /// Create a state snapshot for debugging.
    pub fn snapshot(&self) -> StateSnapshot {
        let arch = arch_from_name(&self.arch_name).unwrap();
        let mut registers = HashMap::new();

        for name in arch.register_names() {
            if let (Some(offset), Some(size)) =
                (arch.register_offset(name), arch.register_size(name))
            {
                let offset = offset as usize;
                let size = size as usize;
                if offset + size <= self.registers.len() {
                    registers.insert(name.to_string(), self.registers[offset..offset + size].to_vec());
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
        Ok(dict)
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

                Ok(ExecutionEvent::from_result(result))
            }
            Err(e) => Ok(ExecutionEvent::error(format!("{}", e))),
        }
    }
}

/// Register the VEX engine module with Python.
pub fn vex_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustVEXEngine>()?;
    m.add_class::<ExecutionEvent>()?;
    m.add_class::<StateSnapshot>()?;
    m.add_class::<PythonCallbacks>()?;
    m.add_class::<LoopExecutionEvent>()?;
    m.add_class::<RustSolverContext>()?;
    // Deferred fork types
    m.add_class::<DeferredFork>()?;
    m.add_class::<BranchPolicy>()?;
    m.add_class::<ExecutionConfig>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let engine = RustVEXEngine::new("amd64").unwrap();
            assert_eq!(engine.arch(), "amd64");
            assert_eq!(engine.pc(), 0);
        });
    }

    #[test]
    fn test_register_access() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
        Python::with_gil(|_py| {
            let mut engine = RustVEXEngine::new("x86").unwrap();

            // Set XMM0 = 1.0f, XMM1 = 2.0f
            let f1_bits = 1.0f32.to_bits() as u128;
            let f2_bits = 2.0f32.to_bits() as u128;
            engine.set_register("xmm0", f1_bits).unwrap();
            engine.set_register("xmm1", f2_bits).unwrap();

            println!("Before: xmm0 = 0x{:x}", engine.get_register("xmm0").unwrap());
            println!("Before: xmm1 = 0x{:x}", engine.get_register("xmm1").unwrap());
            println!("Engine registers size: {}", engine.registers.len());

            // Map memory for the instruction
            engine.map_memory(0x1000, 0x1000, 7);
            engine.map_memory_data(0x1000, &[0xf3, 0x0f, 0x58, 0xc1], 7);  // ADDSS xmm0, xmm1

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
            println!("Result: event_type={}, error={:?}", result.event_type, result.error);

            // Check result
            let xmm0_after = engine.get_register("xmm0").unwrap();
            println!("After: xmm0 = 0x{:x}", xmm0_after);

            let expected = 3.0f32.to_bits() as u128;
            assert_eq!(xmm0_after & 0xFFFFFFFF, expected, "ADDSS should produce 3.0f");
        });
    }
}
