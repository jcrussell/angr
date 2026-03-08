//! Rust-native simulation state for symbolic execution.
//!
//! `RustSimState` provides a Rust-first state representation that:
//! - Owns registers, memory, and solver context
//! - Supports O(1) forking via copy-on-write
//! - Minimizes Python-Rust state transfer overhead
//! - Enables Rust-native exploration loops

use std::collections::HashSet;
use std::rc::Rc;
use std::cell::RefCell;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::arch::{arch_from_name, arch_from_vex, Arch, RegisterFile};
use crate::memory::{MemoryError, Permission, SymbolicMemory, PAGE_SIZE};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::{Endness, VexArch};
use crate::concretize::AddressConcretizer;

/// Unique identifier for states.
static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_state_id() -> u64 {
    NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

/// Execution event from stepping a state.
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Reached end of a basic block, continuing to next address.
    BlockEnd { next_addr: u64 },
    /// Encountered a symbolic branch condition.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
    /// Hit a hook address (SimProcedure).
    Hook { addr: u64 },
    /// Syscall instruction.
    Syscall { num: u64 },
    /// Unmapped memory access - need Python callback.
    UnmappedMemory { addr: u64, size: u64 },
    /// Error during execution.
    Error { message: String },
}

/// Incremental state changes for efficient sync.
///
/// Instead of syncing entire state, we track only what changed.
#[derive(Debug, Clone, Default)]
pub struct StateChanges {
    /// Register changes: (offset, size, value_bytes).
    pub register_writes: Vec<(u32, u32, Vec<u8>)>,
    /// Memory writes: (addr, value_bytes).
    pub memory_writes: Vec<(u64, Vec<u8>)>,
    /// New constraints added (as Z3 AST indices or serialized form).
    pub new_constraints: Vec<u64>,
    /// Updated PC.
    pub new_pc: Option<u64>,
}

impl StateChanges {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.register_writes.is_empty()
            && self.memory_writes.is_empty()
            && self.new_constraints.is_empty()
            && self.new_pc.is_none()
    }
}

/// Rust-native simulation state.
///
/// This struct owns all state components and provides O(1) forking
/// through copy-on-write semantics. The solver context is shared
/// via Rc<RefCell<>> to allow constraint accumulation across forks.
///
/// # Design
///
/// - **Registers**: Stored in `RegisterFile` with concrete bytes and symbolic overlays
/// - **Memory**: `SymbolicMemory` with O(1) CoW via `im::OrdMap`
/// - **Solver**: Shared `Rc<RefCell<SymContext>>` for constraint accumulation
/// - **History**: Basic block trace for debugging/analysis
///
/// # Fork Semantics
///
/// Forking a state is O(1) because:
/// - Memory uses persistent data structures (im::OrdMap)
/// - Registers are cloned (small, ~700 bytes for AMD64)
/// - Solver context is cloned with constraint state preserved
/// - History is optionally shared or copied based on config
pub struct RustSimState {
    /// Architecture information.
    arch: Box<dyn Arch>,
    /// VEX architecture enum (cached for quick lookup).
    vex_arch: VexArch,
    /// Register file with concrete and symbolic values.
    registers: RegisterFile,
    /// Symbolic memory with O(1) CoW forking.
    memory: SymbolicMemory,
    /// Shared solver context for constraints.
    /// Using Rc<RefCell<>> to allow mutation during stepping
    /// while maintaining shared ownership for forking.
    solver: Rc<RefCell<SymContext>>,
    /// Program counter.
    pc: u64,
    /// Unique state identifier.
    state_id: u64,
    /// Parent state ID (for tracking fork tree).
    parent_id: Option<u64>,
    /// Basic block history (addresses visited).
    history: Vec<u64>,
    /// Maximum history length (0 = unlimited).
    max_history: usize,
    /// Hook addresses.
    hooks: HashSet<u64>,
    /// Address concretization config.
    concretizer: AddressConcretizer,
    /// Dirty register tracking (bitset, each bit = 4 bytes).
    dirty_registers: u128,
    /// Whether to track detailed history.
    track_history: bool,
}

impl RustSimState {
    /// Create a new state for the given architecture.
    ///
    /// # Arguments
    /// * `arch_name` - Architecture name (e.g., "amd64", "x86", "arm")
    ///
    /// # Returns
    /// New state with default initialization.
    pub fn new(arch_name: &str) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", arch_name))?;
        let vex_arch = arch.vex_arch();
        let endness = if arch.is_little_endian() {
            Endness::Little
        } else {
            Endness::Big
        };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            max_history: 1000,
            hooks: HashSet::new(),
            concretizer: AddressConcretizer::default(),
            dirty_registers: 0,
            track_history: true,
            arch,
        })
    }

    /// Create a state from VexArch.
    pub fn from_vex_arch(vex_arch: VexArch) -> Self {
        let arch = arch_from_vex(vex_arch);
        let endness = if arch.is_little_endian() {
            Endness::Little
        } else {
            Endness::Big
        };

        RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            max_history: 1000,
            hooks: HashSet::new(),
            concretizer: AddressConcretizer::default(),
            dirty_registers: 0,
            track_history: true,
            arch,
        }
    }

    /// Create a state with a shared solver context.
    ///
    /// This is used when forking to share constraints across states.
    pub fn with_solver(arch_name: &str, solver: Rc<RefCell<SymContext>>) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", arch_name))?;
        let vex_arch = arch.vex_arch();
        let endness = if arch.is_little_endian() {
            Endness::Little
        } else {
            Endness::Big
        };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver,
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            max_history: 1000,
            hooks: HashSet::new(),
            concretizer: AddressConcretizer::default(),
            dirty_registers: 0,
            track_history: true,
            arch,
        })
    }

    // =========================================================================
    // Basic Accessors
    // =========================================================================

    /// Get the state ID.
    pub fn state_id(&self) -> u64 {
        self.state_id
    }

    /// Get the parent state ID.
    pub fn parent_id(&self) -> Option<u64> {
        self.parent_id
    }

    /// Get the program counter.
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Set the program counter.
    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }

    /// Get the VEX architecture.
    pub fn vex_arch(&self) -> VexArch {
        self.vex_arch
    }

    /// Get the architecture.
    pub fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    /// Get the history (basic block addresses visited).
    pub fn history(&self) -> &[u64] {
        &self.history
    }

    /// Add an address to history.
    pub fn add_to_history(&mut self, addr: u64) {
        if self.track_history {
            self.history.push(addr);
            if self.max_history > 0 && self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
    }

    // =========================================================================
    // Register Access
    // =========================================================================

    /// Get a register by name.
    pub fn get_register(&self, name: &str) -> Option<RustBV> {
        let ctx = self.solver.borrow();
        self.registers.get_reg(name, &ctx)
    }

    /// Set a register by name.
    pub fn set_register(&mut self, name: &str, value: RustBV) -> bool {
        if let Some(offset) = self.arch.register_offset(name) {
            // Mark as dirty
            let bit = offset / 4;
            if bit < 128 {
                self.dirty_registers |= 1u128 << bit;
            }
        }
        self.registers.put_reg(name, value)
    }

    /// Get a register by offset.
    pub fn get_register_by_offset(&self, offset: u32, size: u32) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get(offset, size, &ctx)
    }

    /// Set a register by offset.
    pub fn set_register_by_offset(&mut self, offset: u32, value: RustBV) {
        // Mark as dirty
        let bit = offset / 4;
        if bit < 128 {
            self.dirty_registers |= 1u128 << bit;
        }
        self.registers.put(offset, value);
    }

    /// Get the instruction pointer register.
    pub fn get_ip(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_ip(&ctx)
    }

    /// Set the instruction pointer register.
    pub fn set_ip(&mut self, value: RustBV) {
        self.registers.set_ip(value);
        if let Some(v) = self.registers.get_ip(&self.solver.borrow()).as_u64() {
            self.pc = v;
        }
    }

    /// Get the stack pointer register.
    pub fn get_sp(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_sp(&ctx)
    }

    /// Set the stack pointer register.
    pub fn set_sp(&mut self, value: RustBV) {
        self.registers.set_sp(value);
    }

    /// Get all register bytes (for bulk sync to Python).
    pub fn get_registers_raw(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.arch.state_size()];
        self.registers.copy_to_bytes(&mut bytes);
        bytes
    }

    /// Set all register bytes (for bulk sync from Python).
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.registers.copy_from_bytes(bytes);
    }

    /// Get dirty register offsets.
    pub fn get_dirty_registers(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for bit in 0..128u32 {
            if (self.dirty_registers & (1u128 << bit)) != 0 {
                offsets.push(bit * 4);
            }
        }
        offsets
    }

    /// Clear dirty register tracking.
    pub fn clear_dirty_registers(&mut self) {
        self.dirty_registers = 0;
    }

    // =========================================================================
    // Memory Access
    // =========================================================================

    /// Get a reference to the memory.
    pub fn memory(&self) -> &SymbolicMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory.
    pub fn memory_mut(&mut self) -> &mut SymbolicMemory {
        &mut self.memory
    }

    /// Map a memory region.
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: Permission) {
        self.memory.map(addr, size, permissions);
    }

    /// Map memory with initial data.
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: Permission) {
        self.memory.map_data(addr, data, permissions);
    }

    /// Load from memory.
    pub fn memory_load(&self, addr: u64, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.load_concrete(addr, size, &ctx)
    }

    /// Store to memory.
    pub fn memory_store(&mut self, addr: u64, value: RustBV) -> Result<(), MemoryError> {
        self.memory.store_concrete(addr, value)
    }

    /// Load from a symbolic address.
    pub fn memory_load_symbolic(&mut self, addr: RustBV, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.load_symbolic_unified(addr, size, &ctx, &self.concretizer)
    }

    /// Store to a symbolic address.
    pub fn memory_store_symbolic(&mut self, addr: RustBV, value: RustBV) -> Result<(), MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.store_symbolic_unified(addr, value, &ctx, &self.concretizer)
    }

    /// Add a lazy region for on-demand page fetching.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.memory.add_lazy_region(start_addr, size);
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.memory.get_dirty_page_addrs()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.memory.clear_dirty_pages();
    }

    // =========================================================================
    // Solver/Constraint Access
    // =========================================================================

    /// Get a reference to the solver context.
    pub fn solver(&self) -> &Rc<RefCell<SymContext>> {
        &self.solver
    }

    /// Add a constraint.
    pub fn add_constraint(&self, constraint: RustBV) {
        let ctx = self.solver.borrow();
        ctx.assume_true(&constraint);
    }

    /// Check if current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        let ctx = self.solver.borrow();
        ctx.is_sat()
    }

    /// Evaluate an expression to a concrete value.
    pub fn eval(&self, expr: &RustBV) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.eval(expr)
    }

    /// Get minimum value of an expression.
    pub fn min(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.min(expr, signed)
    }

    /// Get maximum value of an expression.
    pub fn max(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.max(expr, signed)
    }

    // =========================================================================
    // Hooks
    // =========================================================================

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hooks.insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.hooks.remove(&addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hooks.contains(&addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the state (O(1) copy-on-write).
    ///
    /// Creates a new state that shares memory pages via CoW.
    /// The solver context is forked to preserve constraints.
    ///
    /// # Returns
    /// A new state with the same register/memory/constraint state.
    pub fn fork(&self) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork()));

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0, // Fresh dirty tracking for fork
            track_history: self.track_history,
        }
    }

    /// Fork with a constraint on the true branch.
    pub fn fork_true(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_true(condition)));

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0,
            track_history: self.track_history,
        }
    }

    /// Fork with a constraint on the false branch.
    pub fn fork_false(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_false(condition)));

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0,
            track_history: self.track_history,
        }
    }

    // =========================================================================
    // Incremental State Changes
    // =========================================================================

    /// Apply incremental changes to the state.
    ///
    /// This is used when syncing from Python after a SimProcedure runs.
    pub fn apply_changes(&mut self, changes: &StateChanges) {
        // Apply register writes
        for (offset, size, bytes) in &changes.register_writes {
            let mut value: u128 = 0;
            for (i, &b) in bytes.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, *size * 8);
            self.set_register_by_offset(*offset, bv);
        }

        // Apply memory writes
        for (addr, bytes) in &changes.memory_writes {
            let width = (bytes.len() * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in bytes.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, width);
            let _ = self.memory.store_concrete(*addr, bv);
        }

        // Apply PC change
        if let Some(new_pc) = changes.new_pc {
            self.pc = new_pc;
        }
    }

    /// Export changes since last clear for sync to Python.
    ///
    /// Returns the minimal diff needed to update Python state.
    pub fn export_changes(&self) -> StateChanges {
        let mut changes = StateChanges::new();

        // Export dirty registers
        let dirty_offsets = self.get_dirty_registers();
        for offset in dirty_offsets {
            let size = 8u32; // Most registers are 8 bytes
            let bv = self.get_register_by_offset(offset, size);
            if let Some(val) = bv.as_u128() {
                let bytes: Vec<u8> = (0..size as usize)
                    .map(|i| (val >> (i * 8)) as u8)
                    .collect();
                changes.register_writes.push((offset, size, bytes));
            }
        }

        // Export dirty pages
        for page_addr in self.memory.get_dirty_page_addrs() {
            let page_num = page_addr >> 12;
            if let Some((data, _perms)) = self.memory.get_page_data(page_num) {
                changes.memory_writes.push((page_addr, data));
            }
        }

        changes.new_pc = Some(self.pc);
        changes
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the address concretization strategy.
    pub fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
        self.concretizer = concretizer;
    }

    /// Configure address concretization.
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.concretizer.configure(use_approximate, range_limit);
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.track_history = track;
    }

    /// Set maximum history length.
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
    }
}

impl Clone for RustSimState {
    fn clone(&self) -> Self {
        self.fork()
    }
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Python-facing wrapper for RustSimState.
///
/// This provides the PyO3 interface for creating and manipulating
/// Rust-native simulation states from Python.
#[pyclass(name = "RustSimState", unsendable)]
pub struct PyRustSimState {
    inner: RustSimState,
}

#[pymethods]
impl PyRustSimState {
    /// Create a new state for the given architecture.
    #[new]
    #[pyo3(signature = (arch="amd64"))]
    pub fn new(arch: &str) -> PyResult<Self> {
        let inner = RustSimState::new(arch)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyRustSimState { inner })
    }

    /// Get the state ID.
    #[getter]
    pub fn state_id(&self) -> u64 {
        self.inner.state_id()
    }

    /// Get the parent state ID.
    #[getter]
    pub fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id()
    }

    /// Get the program counter.
    #[getter]
    pub fn pc(&self) -> u64 {
        self.inner.pc()
    }

    /// Set the program counter.
    #[setter]
    pub fn set_pc(&mut self, pc: u64) {
        self.inner.set_pc(pc);
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch_name(&self) -> &str {
        self.inner.arch().name()
    }

    /// Get the history (basic block addresses).
    pub fn history(&self) -> Vec<u64> {
        self.inner.history().to_vec()
    }

    /// Get a register value by name.
    pub fn get_register(&self, name: &str) -> PyResult<u128> {
        self.inner.get_register(name)
            .and_then(|bv| bv.as_u128())
            .ok_or_else(|| PyValueError::new_err(format!("cannot read register {}", name)))
    }

    /// Set a register value by name.
    pub fn set_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        let size = self.inner.arch().register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        let bv = RustBV::concrete(value, size * 8);
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("failed to set register: {}", name)))
        }
    }

    /// Get all register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.inner.get_registers_raw()
    }

    /// Set all register bytes.
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.inner.set_registers_raw(bytes);
    }

    /// Get dirty register offsets.
    pub fn get_dirty_registers(&self) -> Vec<u32> {
        self.inner.get_dirty_registers()
    }

    /// Clear dirty register tracking.
    pub fn clear_dirty_registers(&mut self) {
        self.inner.clear_dirty_registers();
    }

    /// Map a memory region.
    #[pyo3(signature = (addr, size, permissions=7))]
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: u8) {
        self.inner.map_memory(addr, size, Permission::from_bits(permissions));
    }

    /// Map memory with initial data.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: u8) {
        self.inner.map_memory_data(addr, data, Permission::from_bits(permissions));
    }

    /// Load from memory.
    pub fn memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        let bv = self.inner.memory_load(addr, size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let value = bv.to_u128();
        let bytes: Vec<u8> = (0..size as usize)
            .map(|i| (value >> (i * 8)) as u8)
            .collect();
        Ok(bytes)
    }

    /// Store to memory.
    pub fn memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let width = (data.len() * 8) as u32;
        let mut value: u128 = 0;
        for (i, &b) in data.iter().enumerate() {
            value |= (b as u128) << (i * 8);
        }
        let bv = RustBV::concrete(value, width);
        self.inner.memory_store(addr, bv)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Add a lazy region.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.inner.add_lazy_region(start_addr, size);
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.inner.get_dirty_pages()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.inner.clear_dirty_pages();
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.inner.add_hook(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.inner.remove_hook(addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.inner.is_hooked(addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.inner.clear_hooks();
    }

    /// Check if constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.inner.satisfiable()
    }

    /// Fork the state (O(1) CoW).
    pub fn fork(&self) -> Self {
        PyRustSimState {
            inner: self.inner.fork(),
        }
    }

    /// Configure address concretization.
    #[pyo3(signature = (use_approximate, range_limit=None))]
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.inner.configure_concretization(use_approximate, range_limit);
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.inner.set_track_history(track);
    }

    /// Set maximum history length.
    pub fn set_max_history(&mut self, max: usize) {
        self.inner.set_max_history(max);
    }
}

impl PyRustSimState {
    /// Get access to the inner state (for Rust-side use).
    pub fn inner(&self) -> &RustSimState {
        &self.inner
    }

    /// Get mutable access to the inner state.
    pub fn inner_mut(&mut self) -> &mut RustSimState {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = RustSimState::new("amd64").unwrap();
        assert_eq!(state.vex_arch(), VexArch::AMD64);
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn test_state_fork() {
        let mut state1 = RustSimState::new("amd64").unwrap();
        state1.set_pc(0x1000);
        state1.set_register("rax", RustBV::concrete(42, 64));

        let state2 = state1.fork();

        // Both should have same values
        assert_eq!(state2.pc(), 0x1000);
        assert_eq!(state2.get_register("rax").unwrap().as_u64(), Some(42));

        // Different state IDs
        assert_ne!(state1.state_id(), state2.state_id());

        // state2's parent should be state1
        assert_eq!(state2.parent_id(), Some(state1.state_id()));
    }

    #[test]
    fn test_state_memory() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map and write
        state.map_memory(0x1000, 0x1000, Permission::RWX);
        state.memory_store(0x1000, RustBV::concrete(0xDEADBEEF, 32)).unwrap();

        // Read back
        let val = state.memory_load(0x1000, 4).unwrap();
        assert_eq!(val.as_u64(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_state_fork_memory_cow() {
        let mut state1 = RustSimState::new("amd64").unwrap();
        state1.map_memory(0x1000, 0x1000, Permission::RWX);
        state1.memory_store(0x1000, RustBV::concrete(0xAAAA, 16)).unwrap();

        let mut state2 = state1.fork();

        // Modify state2
        state2.memory_store(0x1000, RustBV::concrete(0xBBBB, 16)).unwrap();

        // state1 should still have original value
        let val1 = state1.memory_load(0x1000, 2).unwrap();
        assert_eq!(val1.as_u64(), Some(0xAAAA));

        // state2 should have new value
        let val2 = state2.memory_load(0x1000, 2).unwrap();
        assert_eq!(val2.as_u64(), Some(0xBBBB));
    }
}
