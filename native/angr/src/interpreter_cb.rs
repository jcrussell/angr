//! Callback-aware VEX IR interpreter.
//!
//! This interpreter uses Python callbacks for memory operations instead of
//! local SymbolicMemory. It can run multiple blocks in a loop, returning
//! to Python only when an event requires Python handling.

use std::collections::HashSet;
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;

use crate::arch::{arch_from_vex, RegisterFile};
use crate::callbacks::{PythonCallbacks, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, is_claripy_ast};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ccall;
use crate::vex::ir::{IRConst, IRExpr, IRStmt, IRType, JumpKind, TypeEnv, VexArch, IRSB};
use crate::vex::ops::{OpError, VEXOps};
use crate::vex::deserialize_irsb;

/// Errors during callback-based VEX execution.
#[derive(Debug, Clone)]
pub enum CbExecutionError {
    /// Memory error from callback.
    Memory(String),
    /// Operation error.
    Op(OpError),
    /// Invalid VEX IR.
    InvalidIR(String),
    /// Unsupported feature.
    Unsupported(String),
    /// Type mismatch.
    TypeMismatch { expected: IRType, got: IRType },
    /// Unknown temporary variable.
    UnknownTemp(u32),
    /// Python callback error.
    Callback(String),
    /// Block lifting error.
    LiftError(String),
}

impl From<OpError> for CbExecutionError {
    fn from(e: OpError) -> Self {
        CbExecutionError::Op(e)
    }
}

impl std::fmt::Display for CbExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CbExecutionError::Memory(msg) => write!(f, "memory error: {}", msg),
            CbExecutionError::Op(e) => write!(f, "operation error: {}", e),
            CbExecutionError::InvalidIR(msg) => write!(f, "invalid VEX IR: {}", msg),
            CbExecutionError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            CbExecutionError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {:?}, got {:?}", expected, got)
            }
            CbExecutionError::UnknownTemp(tmp) => write!(f, "unknown temporary t{}", tmp),
            CbExecutionError::Callback(msg) => write!(f, "callback error: {}", msg),
            CbExecutionError::LiftError(msg) => write!(f, "lift error: {}", msg),
        }
    }
}

impl std::error::Error for CbExecutionError {}

/// Result of executing a single statement.
enum StmtResult {
    /// Continue to next statement.
    Continue,
    /// Exit the block early.
    Exit { target: u64, jumpkind: JumpKind },
    /// Symbolic branch detected - need to fork.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
}

/// Result of executing a single block.
#[derive(Debug)]
pub enum BlockResult {
    /// Continue to the next block at given address.
    Continue { next_addr: u64 },
    /// Syscall encountered.
    Syscall { num: u64 },
    /// Symbolic branch - need to fork.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Hook address hit.
    Hook { addr: u64 },
    /// Block execution error.
    Error { message: String },
    /// Normal block end with jumpkind.
    BlockEnd { next_addr: u64, jumpkind: JumpKind },
}

/// Callback-aware VEX IR interpreter.
///
/// This interpreter uses Python callbacks for memory and register access,
/// allowing it to work with angr's symbolic memory model.
pub struct CallbackInterpreter<'a> {
    /// Register file (local cache, synced via callbacks).
    pub registers: RegisterFile,
    /// Temporary variables for current block.
    temps: Vec<Option<RustBV>>,
    /// Solver context.
    ctx: &'a SymContext,
    /// Current program counter.
    pub pc: u64,
    /// Current instruction address (within block).
    current_insn_addr: u64,
    /// Hook addresses (return to Python when hit).
    hook_addrs: HashSet<u64>,
    /// VEX architecture.
    arch: VexArch,
    /// Block cache (shared across runs).
    block_cache: LruCache<u64, IRSB>,
    /// Whether to use callbacks for memory (vs local registers).
    use_memory_callbacks: bool,
}

impl<'a> CallbackInterpreter<'a> {
    /// Create a new callback-aware interpreter.
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        let arch_box = arch_from_vex(arch);

        CallbackInterpreter {
            registers: RegisterFile::new(arch_box),
            temps: Vec::new(),
            ctx,
            pc: 0,
            current_insn_addr: 0,
            hook_addrs: HashSet::new(),
            arch,
            block_cache: LruCache::new(NonZeroUsize::new(1024).unwrap()),
            use_memory_callbacks: true,
        }
    }

    /// Get the solver context.
    pub fn context(&self) -> &SymContext {
        self.ctx
    }

    /// Set the program counter.
    pub fn set_pc(&mut self, addr: u64) {
        self.pc = addr;
        let pc_bv = RustBV::concrete(addr as u128, self.registers.arch().bits());
        self.registers.set_ip(pc_bv);
    }

    /// Get the program counter.
    pub fn get_pc(&self) -> u64 {
        self.pc
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hook_addrs.insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.hook_addrs.remove(&addr);
    }

    /// Add multiple hooks at once.
    pub fn add_hooks(&mut self, addrs: &[u64]) {
        for &addr in addrs {
            self.hook_addrs.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hook_addrs.clear();
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hook_addrs.contains(&addr)
    }

    /// Check if we have a cached block at the given address.
    pub fn has_cached_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Add a block to the cache.
    pub fn cache_block(&mut self, addr: u64, irsb: IRSB) {
        self.block_cache.put(addr, irsb);
    }

    /// Get a block from the cache.
    pub fn get_cached_block(&mut self, addr: u64) -> Option<&IRSB> {
        self.block_cache.get(&addr)
    }

    /// Run the execution loop until an event requires Python handling.
    ///
    /// This is the main entry point for the callback-based execution model.
    /// It runs blocks in a loop, using Python callbacks for memory access,
    /// until it hits a condition that requires Python-side handling.
    pub fn run_until_event(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        max_blocks: u32,
    ) -> (RunResult, u32) {
        let mut blocks_executed = 0u32;

        for _ in 0..max_blocks {
            // Check for hook at current PC
            if self.is_hooked(self.pc) {
                return (RunResult::Hook { addr: self.pc }, blocks_executed);
            }

            // Try to get or lift the block
            let irsb = match self.get_or_lift_block(py, callbacks, self.pc) {
                Ok(irsb) => irsb,
                Err(e) => {
                    return (
                        RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                        blocks_executed,
                    );
                }
            };

            // Execute the block
            match self.execute_block_with_callbacks(py, callbacks, &irsb) {
                Ok(result) => {
                    blocks_executed += 1;

                    match result {
                        BlockResult::Continue { next_addr } => {
                            self.pc = next_addr;
                            // Continue to next block
                        }
                        BlockResult::BlockEnd { next_addr, jumpkind } => {
                            self.pc = next_addr;
                            // Return for jumpkinds that need Python handling
                            if jumpkind.is_syscall() {
                                let syscall_num = self.get_syscall_num();
                                return (
                                    RunResult::Syscall {
                                        num: syscall_num,
                                        pc: next_addr,
                                    },
                                    blocks_executed,
                                );
                            }
                            // For Call/Ret, we might want to return for SimProcedures
                            if self.is_hooked(next_addr) {
                                return (RunResult::Hook { addr: next_addr }, blocks_executed);
                            }
                            // Otherwise, continue execution
                        }
                        BlockResult::Syscall { num } => {
                            return (
                                RunResult::Syscall { num, pc: self.pc },
                                blocks_executed,
                            );
                        }
                        BlockResult::SymbolicBranch {
                            condition_id,
                            true_target,
                            false_target,
                        } => {
                            return (
                                RunResult::SymbolicBranch {
                                    condition_id,
                                    true_target,
                                    false_target,
                                },
                                blocks_executed,
                            );
                        }
                        BlockResult::Hook { addr } => {
                            return (RunResult::Hook { addr }, blocks_executed);
                        }
                        BlockResult::Error { message } => {
                            return (
                                RunResult::Error {
                                    message,
                                    addr: self.pc,
                                },
                                blocks_executed,
                            );
                        }
                    }
                }
                Err(e) => {
                    return (
                        RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                        blocks_executed,
                    );
                }
            }
        }

        // Reached max blocks
        (RunResult::MaxBlocks { pc: self.pc }, blocks_executed)
    }

    /// Get or lift a block at the given address.
    fn get_or_lift_block(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr: u64,
    ) -> Result<IRSB, CbExecutionError> {
        // Check cache first
        if let Some(irsb) = self.block_cache.get(&addr) {
            return Ok(irsb.clone());
        }

        // Lift via Python callback
        let irsb_json = callbacks
            .call_lift_block(py, addr)
            .map_err(|e| CbExecutionError::LiftError(format!("lift callback failed: {}", e)))?;

        let irsb = deserialize_irsb(&irsb_json)
            .map_err(|e| CbExecutionError::LiftError(format!("IRSB deserialization failed: {}", e)))?;

        // Cache it
        self.block_cache.put(addr, irsb.clone());

        Ok(irsb)
    }

    /// Execute a block using Python callbacks for memory access.
    fn execute_block_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        // Reset temps for this block
        self.temps = vec![None; irsb.tyenv.types.len()];
        self.current_insn_addr = irsb.addr;

        // Execute statements
        for stmt in &irsb.statements {
            match self.execute_stmt_with_callbacks(py, callbacks, stmt, irsb)? {
                StmtResult::Continue => continue,
                StmtResult::Exit { target, jumpkind } => {
                    return Ok(self.handle_exit(target, jumpkind));
                }
                StmtResult::SymbolicBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    return Ok(BlockResult::SymbolicBranch {
                        condition_id: 0, // TODO: proper condition tracking
                        true_target,
                        false_target,
                    });
                }
            }
        }

        // Handle default exit
        self.handle_default_exit(irsb)
    }

    /// Execute a single statement using Python callbacks.
    fn execute_stmt_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        stmt: &IRStmt,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        match stmt {
            IRStmt::NoOp => Ok(StmtResult::Continue),

            IRStmt::IMark { addr, .. } => {
                self.current_insn_addr = *addr;
                // Check for hooks at this address
                if self.is_hooked(*addr) {
                    return Ok(StmtResult::Exit {
                        target: *addr,
                        jumpkind: JumpKind::Boring,
                    });
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::AbiHint { .. } => Ok(StmtResult::Continue),

            IRStmt::Put { offset, data } => {
                let value = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                self.registers.put(*offset, value);
                Ok(StmtResult::Continue)
            }

            IRStmt::WrTmp { tmp, data } => {
                let value = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                if (*tmp as usize) < self.temps.len() {
                    self.temps[*tmp as usize] = Some(value);
                } else {
                    return Err(CbExecutionError::UnknownTemp(*tmp));
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::Store { addr, data, .. } => {
                let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;

                // Store via callback
                if let Some(addr_concrete) = addr_val.as_u64() {
                    let data_bytes = bv_to_bytes(&data_val);
                    callbacks
                        .call_memory_store(py, addr_concrete, &data_bytes)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                } else {
                    return Err(CbExecutionError::Unsupported(
                        "symbolic store address".to_string(),
                    ));
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::Exit { guard, dst, jk, .. } => {
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;

                // Check if guard is concrete
                if let Some(g) = guard_val.as_u64() {
                    if g != 0 {
                        return Ok(StmtResult::Exit {
                            target: *dst,
                            jumpkind: *jk,
                        });
                    }
                    return Ok(StmtResult::Continue);
                }

                // Guard is symbolic - check both possibilities
                let can_be_true = self.ctx.can_be_true(&guard_val);
                let can_be_false = self.ctx.can_be_false(&guard_val);

                if can_be_true && can_be_false {
                    // Need to fork
                    let fallthrough = self.eval_next_addr(py, callbacks, irsb)?;
                    return Ok(StmtResult::SymbolicBranch {
                        condition: guard_val,
                        true_target: *dst,
                        false_target: fallthrough,
                    });
                } else if can_be_true {
                    return Ok(StmtResult::Exit {
                        target: *dst,
                        jumpkind: *jk,
                    });
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::MBE(_) => Ok(StmtResult::Continue),

            IRStmt::PutI { .. } => Err(CbExecutionError::Unsupported("PutI".to_string())),
            IRStmt::StoreG { .. } => Err(CbExecutionError::Unsupported("guarded store".to_string())),
            IRStmt::LoadG { .. } => Err(CbExecutionError::Unsupported("guarded load".to_string())),
            IRStmt::CAS { .. } => Err(CbExecutionError::Unsupported("compare-and-swap".to_string())),
            IRStmt::LLSC { .. } => {
                Err(CbExecutionError::Unsupported("load-linked/store-conditional".to_string()))
            }
            IRStmt::Dirty(_) => Err(CbExecutionError::Unsupported("dirty call".to_string())),
        }
    }

    /// Evaluate an IR expression using Python callbacks for memory loads.
    fn eval_expr_with_callbacks(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),

            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }

            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }

            IRExpr::Load { addr, ty, .. } => {
                let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, tyenv)?;
                let size = ty.bytes();

                if let Some(addr_concrete) = addr_val.as_u64() {
                    // Load via callback
                    let (data, is_symbolic, symbolic_ast) = callbacks
                        .call_memory_load(py, addr_concrete, size)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

                    if is_symbolic {
                        // Try to convert claripy AST to RustBV
                        if let Some(ast_obj) = symbolic_ast {
                            let ast = ast_obj.bind(py);
                            if is_claripy_ast(&ast) {
                                match claripy_to_rustbv(py, &ast, self.ctx) {
                                    Ok(bv) => return Ok(bv),
                                    Err(_e) => {
                                        // Fall back to creating a fresh symbolic value
                                        // (claripy conversion can fail for complex/unsupported ops)
                                    }
                                }
                            }
                        }
                        // Fallback: create a fresh symbolic value
                        Ok(RustBV::symbolic(
                            self.ctx,
                            &format!("mem_{:x}_{}", addr_concrete, size),
                            size * 8,
                        ))
                    } else {
                        // Convert bytes to concrete value
                        Ok(bytes_to_bv(&data, size * 8))
                    }
                } else {
                    Err(CbExecutionError::Unsupported(
                        "symbolic load address".to_string(),
                    ))
                }
            }

            IRExpr::Unop { op, arg } => {
                let arg_val = self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?;
                VEXOps::unop(*op, arg_val, self.ctx).map_err(|e| e.into())
            }

            IRExpr::Binop { op, left, right } => {
                let left_val = self.eval_expr_with_callbacks(py, callbacks, left, tyenv)?;
                let right_val = self.eval_expr_with_callbacks(py, callbacks, right, tyenv)?;
                VEXOps::binop(*op, left_val, right_val, self.ctx).map_err(|e| e.into())
            }

            IRExpr::ITE { cond, iftrue, iffalse } => {
                let cond_val = self.eval_expr_with_callbacks(py, callbacks, cond, tyenv)?;
                let true_val = self.eval_expr_with_callbacks(py, callbacks, iftrue, tyenv)?;
                let false_val = self.eval_expr_with_callbacks(py, callbacks, iffalse, tyenv)?;
                Ok(cond_val.ite(&true_val, &false_val, self.ctx))
            }

            IRExpr::GetI { .. } => {
                Err(CbExecutionError::Unsupported("GetI (rotating registers)".to_string()))
            }

            IRExpr::Triop { .. } => Err(CbExecutionError::Unsupported("triop".to_string())),

            IRExpr::Qop { .. } => Err(CbExecutionError::Unsupported("qop".to_string())),

            IRExpr::CCall { cee, retty, args } => {
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?);
                }

                if let Some(result) = ccall::handle_ccall(&cee.name, &arg_vals, retty.bits()) {
                    return Ok(result);
                }

                Ok(RustBV::concrete(0, retty.bits()))
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                Err(CbExecutionError::Unsupported("special expr".to_string()))
            }
        }
    }

    /// Evaluate an IR constant.
    fn eval_const(&self, c: &IRConst) -> RustBV {
        match c {
            IRConst::U1(v) => RustBV::concrete(*v as u128, 1),
            IRConst::U8(v) => RustBV::concrete(*v as u128, 8),
            IRConst::U16(v) => RustBV::concrete(*v as u128, 16),
            IRConst::U32(v) => RustBV::concrete(*v as u128, 32),
            IRConst::U64(v) => RustBV::concrete(*v as u128, 64),
            IRConst::U128(v) => RustBV::concrete(*v, 128),
            IRConst::F32(v) => RustBV::concrete(v.to_bits() as u128, 32),
            IRConst::F64(v) => RustBV::concrete(v.to_bits() as u128, 64),
            IRConst::V128(v) => RustBV::concrete(*v, 128),
            IRConst::V256(v) => {
                RustBV::concrete(v[0] as u128 | ((v[1] as u128) << 64), 128)
            }
        }
    }

    /// Evaluate the next address from an IRSB.
    fn eval_next_addr(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<u64, CbExecutionError> {
        let next_val = self.eval_expr_with_callbacks(py, callbacks, &irsb.next, &irsb.tyenv)?;
        next_val.as_u64().ok_or_else(|| {
            CbExecutionError::Unsupported("symbolic next address".to_string())
        })
    }

    /// Handle the default exit (end of block).
    fn handle_default_exit(&mut self, irsb: &IRSB) -> Result<BlockResult, CbExecutionError> {
        // For default exit, we need to evaluate next without callbacks
        // since we already have temps set up
        let next_val = self.eval_expr_simple(&irsb.next, &irsb.tyenv)?;

        if let Some(addr) = next_val.as_u64() {
            Ok(self.handle_exit(addr, irsb.jumpkind))
        } else {
            Err(CbExecutionError::Unsupported(
                "symbolic next address".to_string(),
            ))
        }
    }

    /// Simple expression evaluation (no callbacks, for already-evaluated temps).
    fn eval_expr_simple(
        &self,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),
            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }
            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }
            _ => Err(CbExecutionError::Unsupported(
                "complex expr in default exit".to_string(),
            )),
        }
    }

    /// Handle an exit (update PC, return result).
    fn handle_exit(&mut self, target: u64, jumpkind: JumpKind) -> BlockResult {
        self.set_pc(target);

        if jumpkind.is_syscall() {
            let syscall_num = self.get_syscall_num();
            return BlockResult::Syscall { num: syscall_num };
        }

        if self.is_hooked(target) {
            return BlockResult::Hook { addr: target };
        }

        BlockResult::BlockEnd {
            next_addr: target,
            jumpkind,
        }
    }

    /// Get the syscall number from the appropriate register.
    fn get_syscall_num(&self) -> u64 {
        // For AMD64, syscall number is in RAX (offset 16)
        // For x86, syscall number is in EAX (offset 8)
        // TODO: make this architecture-aware
        let offset = match self.arch {
            VexArch::AMD64 => 16,  // RAX
            VexArch::X86 => 8,     // EAX
            _ => 0,  // TODO: other architectures
        };
        let syscall_bv = self.registers.get(offset, 8, self.ctx);
        syscall_bv.as_u64().unwrap_or(0)
    }

    /// Fork the interpreter state.
    pub fn fork(&self) -> CallbackInterpreter<'a> {
        CallbackInterpreter {
            registers: self.registers.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
            pc: self.pc,
            current_insn_addr: self.current_insn_addr,
            hook_addrs: self.hook_addrs.clone(),
            arch: self.arch,
            block_cache: LruCache::new(NonZeroUsize::new(1024).unwrap()), // Fresh cache for fork
            use_memory_callbacks: self.use_memory_callbacks,
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

/// Convert bytes (little-endian) to a RustBV.
fn bytes_to_bv(bytes: &[u8], width: u32) -> RustBV {
    let mut value: u128 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if i * 8 >= width as usize {
            break;
        }
        value |= (byte as u128) << (i * 8);
    }
    RustBV::concrete(value, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_bv() {
        let bytes = vec![0x78, 0x56, 0x34, 0x12];
        let bv = bytes_to_bv(&bytes, 32);
        assert_eq!(bv.as_u64(), Some(0x12345678));
    }

    #[test]
    fn test_bv_to_bytes() {
        let bv = RustBV::concrete(0x12345678, 32);
        let bytes = bv_to_bytes(&bv);
        assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_interpreter_creation() {
        let ctx = SymContext::new_mock();
        let interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        assert_eq!(interp.get_pc(), 0);
    }

    #[test]
    fn test_hook_management() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);

        interp.add_hook(0x1000);
        assert!(interp.is_hooked(0x1000));
        assert!(!interp.is_hooked(0x2000));

        interp.remove_hook(0x1000);
        assert!(!interp.is_hooked(0x1000));
    }
}
