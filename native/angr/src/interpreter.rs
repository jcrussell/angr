//! VEX IR interpreter.
//!
//! This module provides the core execution engine that interprets VEX IR
//! statements and expressions, managing registers, memory, and symbolic state.

use crate::arch::{RegisterFile, arch_from_vex};
use crate::memory::{MemoryError, SymbolicMemory};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ccall;
use crate::vex::ir::{IRConst, IRExpr, IRSB, IRStmt, IRType, JumpKind, TypeEnv, VexArch};
use crate::vex::ops::{OpError, VEXOps};

/// Result of executing a VEX block.
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// Block executed to completion, next address is concrete.
    BlockEnd { next_addr: u64, jumpkind: JumpKind },
    /// Encountered a symbolic branch condition.
    SymbolicBranch { true_target: u64, false_target: u64 },
    /// Syscall encountered.
    Syscall {
        /// Syscall number (may be symbolic).
        num: u64,
    },
    /// Hook address encountered.
    Hook { addr: u64 },
    /// Error during execution.
    Error { kind: ExecutionError, addr: u64 },
}

/// Errors during VEX execution.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutionError {
    /// Memory error.
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Operation error.
    #[error("operation error: {0}")]
    Op(#[from] OpError),
    /// Invalid VEX IR.
    #[error("invalid VEX IR: {0}")]
    InvalidIR(String),
    /// Unsupported feature.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Type mismatch.
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: IRType, got: IRType },
    /// Unknown temporary variable.
    #[error("unknown temporary t{0}")]
    UnknownTemp(u32),
}

/// Result of executing a single statement.
enum StmtResult {
    /// Continue to next statement.
    Continue,
    /// Exit the block early.
    Exit { target: u64, jumpkind: JumpKind },
    /// Symbolic branch detected.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
}

/// VEX IR interpreter.
///
/// This is the core execution engine that interprets VEX IR blocks.
pub struct VEXInterpreter<'a> {
    /// Register file.
    pub registers: RegisterFile,
    /// Memory.
    pub memory: SymbolicMemory,
    /// Temporary variables for current block.
    temps: Vec<Option<RustBV>>,
    /// Solver context.
    ctx: &'a SymContext,
    /// Current program counter.
    pub pc: u64,
    /// Current instruction address (within block).
    current_insn_addr: u64,
    /// Hook addresses (return to Python when hit).
    hook_addrs: std::collections::HashSet<u64>,
    /// Store log: tracks individual stores (address, size) for precise sync.
    /// This allows syncing only specific bytes instead of entire dirty pages.
    store_log: Vec<(u64, usize)>,
    /// Bitset tracking which register offsets have been modified.
    /// Each bit represents a 4-byte aligned offset (offset / 4).
    /// A u128 covers 512 bytes of register space (128 * 4 = 512).
    dirty_registers: u128,
}

impl<'a> VEXInterpreter<'a> {
    /// Create a new interpreter for the given architecture.
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        let arch_box = arch_from_vex(arch);
        let endness = arch.endness();

        VEXInterpreter {
            registers: RegisterFile::new(arch_box),
            memory: SymbolicMemory::new(endness),
            temps: Vec::new(),
            ctx,
            pc: 0,
            current_insn_addr: 0,
            hook_addrs: std::collections::HashSet::new(),
            store_log: Vec::new(),
            dirty_registers: 0,
        }
    }

    /// Get the solver context.
    pub fn context(&self) -> &'a SymContext {
        self.ctx
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hook_addrs.insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.hook_addrs.remove(&addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hook_addrs.contains(&addr)
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

    /// Get the store log: list of (address, size) tuples for stores during execution.
    pub fn get_store_log(&self) -> &[(u64, usize)] {
        &self.store_log
    }

    /// Clear the store log.
    pub fn clear_store_log(&mut self) {
        self.store_log.clear();
    }

    /// Get list of dirty register offsets (registers modified since last clear).
    /// Returns offsets in 4-byte granularity.
    pub fn get_dirty_register_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for bit in 0..128u32 {
            if (self.dirty_registers & (1u128 << bit)) != 0 {
                offsets.push(bit * 4);
            }
        }
        offsets
    }

    /// Get the raw dirty register bitset.
    pub fn dirty_registers(&self) -> u128 {
        self.dirty_registers
    }

    /// Clear dirty register tracking (called after sync).
    pub fn clear_dirty_registers(&mut self) {
        self.dirty_registers = 0;
    }

    /// Mark the 4-byte register slot containing `offset` as dirty.
    /// Bitset is 128 bits wide (covers offsets [0, 512)); writes beyond
    /// that fall back to the always-sync slow path.
    #[inline]
    pub(crate) fn mark_register_dirty(&mut self, offset: u32) {
        let bit_index = offset / 4;
        if bit_index < 128 {
            self.dirty_registers |= 1u128 << bit_index;
        }
    }

    /// Execute a VEX block.
    pub fn execute_block(&mut self, irsb: &IRSB) -> Result<ExecutionResult, ExecutionError> {
        // Reset temps for this block
        self.temps = vec![None; irsb.tyenv.types.len()];
        self.current_insn_addr = irsb.addr;

        // Execute statements
        for stmt in &irsb.statements {
            match self.execute_stmt(stmt, irsb)? {
                StmtResult::Continue => continue,
                StmtResult::Exit { target, jumpkind } => {
                    return Ok(self.handle_exit(target, jumpkind));
                }
                StmtResult::SymbolicBranch {
                    condition: _condition,
                    true_target,
                    false_target,
                } => {
                    return Ok(ExecutionResult::SymbolicBranch {
                        true_target,
                        false_target,
                    });
                }
            }
        }

        // Handle default exit
        self.handle_default_exit(irsb)
    }

    /// Execute a single statement.
    fn execute_stmt(&mut self, stmt: &IRStmt, irsb: &IRSB) -> Result<StmtResult, ExecutionError> {
        match stmt {
            IRStmt::NoOp => Ok(StmtResult::Continue),

            IRStmt::IMark {
                addr, len: _len, ..
            } => {
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

            IRStmt::AbiHint { .. } => {
                // ABI hints are no-ops for execution
                Ok(StmtResult::Continue)
            }

            IRStmt::Put { offset, data } => {
                let value = self.eval_expr(data, &irsb.tyenv)?;
                self.registers.put(*offset, value);
                self.mark_register_dirty(*offset);
                Ok(StmtResult::Continue)
            }

            IRStmt::WrTmp { tmp, data } => {
                let value = self.eval_expr(data, &irsb.tyenv)?;
                if (*tmp as usize) < self.temps.len() {
                    self.temps[*tmp as usize] = Some(value);
                } else {
                    return Err(ExecutionError::UnknownTemp(*tmp));
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::Store {
                addr,
                data,
                endness,
            } => {
                let addr_val = self.eval_expr(addr, &irsb.tyenv)?;
                let data_val = self.eval_expr(data, &irsb.tyenv)?;
                // Log the store for precise sync to Python
                // Get concrete address (or concretize symbolic)
                let concrete_addr = match addr_val.as_u64() {
                    Some(a) => a,
                    None => self.ctx.eval(&addr_val).unwrap_or(0) as u64,
                };
                let store_size = data_val.width() / 8;
                self.store_log.push((concrete_addr, store_size as usize));
                // If IR endness differs from memory endness, byte-reverse the value
                let store_val = if *endness != self.memory.endness() {
                    data_val.reverse(self.ctx)
                } else {
                    data_val
                };
                self.memory.store(addr_val, store_val, self.ctx)?;
                Ok(StmtResult::Continue)
            }

            IRStmt::Exit {
                guard,
                dst,
                jk,
                offsIP: _offs_ip,
            } => {
                let guard_val = self.eval_expr(guard, &irsb.tyenv)?;

                // Check if guard is concrete
                if let Some(g) = guard_val.as_u64() {
                    if g != 0 {
                        // Guard is true, take the exit
                        return Ok(StmtResult::Exit {
                            target: *dst,
                            jumpkind: *jk,
                        });
                    }
                    // Guard is false, continue
                    return Ok(StmtResult::Continue);
                }

                // Guard is symbolic - we need to check both possibilities
                let can_be_true = self.ctx.can_be_true(&guard_val);
                let can_be_false = self.ctx.can_be_false(&guard_val);

                if can_be_true && can_be_false {
                    // Need to fork - return to Python
                    let fallthrough = self.eval_next_addr(irsb)?;
                    return Ok(StmtResult::SymbolicBranch {
                        condition: guard_val,
                        true_target: *dst,
                        false_target: fallthrough,
                    });
                } else if can_be_true {
                    // Must take the exit
                    return Ok(StmtResult::Exit {
                        target: *dst,
                        jumpkind: *jk,
                    });
                }
                // Must fall through
                Ok(StmtResult::Continue)
            }

            IRStmt::MBE(_) => {
                // Memory barrier - no effect in our model
                Ok(StmtResult::Continue)
            }

            IRStmt::PutI { .. } => {
                // Rotating register put - used for x87 FPU stack
                Err(ExecutionError::Unsupported("PutI".to_string()))
            }

            IRStmt::StoreG { .. } => Err(ExecutionError::Unsupported("guarded store".to_string())),

            IRStmt::LoadG { .. } => Err(ExecutionError::Unsupported("guarded load".to_string())),

            IRStmt::CAS { .. } => Err(ExecutionError::Unsupported("compare-and-swap".to_string())),

            IRStmt::LLSC { .. } => Err(ExecutionError::Unsupported(
                "load-linked/store-conditional".to_string(),
            )),

            IRStmt::Dirty(_) => Err(ExecutionError::Unsupported("dirty call".to_string())),
        }
    }

    /// Evaluate an IR expression.
    fn eval_expr(&self, expr: &IRExpr, tyenv: &TypeEnv) -> Result<RustBV, ExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),

            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(ExecutionError::UnknownTemp(*tmp))
                }
            }

            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }

            IRExpr::Load { addr, ty, endness } => {
                let addr_val = self.eval_expr(addr, tyenv)?;
                let size = ty.bytes();
                let val = self
                    .memory
                    .load(addr_val, size, self.ctx)
                    .map_err(|e| Into::<ExecutionError>::into(e))?;
                // If IR endness differs from memory endness, byte-reverse the loaded value
                if *endness != self.memory.endness() {
                    Ok(val.reverse(self.ctx))
                } else {
                    Ok(val)
                }
            }

            IRExpr::Unop { op, arg } => {
                let arg_val = self.eval_expr(arg, tyenv)?;
                VEXOps::unop(*op, arg_val, self.ctx).map_err(|e| e.into())
            }

            IRExpr::Binop { op, left, right } => {
                let left_val = self.eval_expr(left, tyenv)?;
                let right_val = self.eval_expr(right, tyenv)?;
                VEXOps::binop(*op, left_val, right_val, self.ctx).map_err(|e| e.into())
            }

            IRExpr::ITE {
                cond,
                iftrue,
                iffalse,
            } => {
                let cond_val = self.eval_expr(cond, tyenv)?;
                let true_val = self.eval_expr(iftrue, tyenv)?;
                let false_val = self.eval_expr(iffalse, tyenv)?;
                Ok(cond_val.ite(&true_val, &false_val, self.ctx))
            }

            IRExpr::GetI { .. } => Err(ExecutionError::Unsupported(
                "GetI (rotating registers)".to_string(),
            )),

            IRExpr::Triop {
                op,
                arg1,
                arg2,
                arg3,
            } => {
                // VEX Triops are float arithmetic with a rounding mode (rm, a, b).
                // Drop the rm (arg1) and dispatch via VEXOps::binop. Concrete
                // float math is correct under default IEEE round-to-nearest;
                // symbolic floats are unsupported (mirror of binop fallback).
                let _rm = self.eval_expr(arg1, tyenv)?;
                let v2 = self.eval_expr(arg2, tyenv)?;
                let v3 = self.eval_expr(arg3, tyenv)?;
                VEXOps::binop(*op, v2, v3, self.ctx)
                    .map_err(|_| ExecutionError::Unsupported(format!("triop {:?}", op)))
            }

            IRExpr::Qop {
                op,
                arg1,
                arg2,
                arg3,
                arg4,
            } => {
                // VEX Qops are typically fused multiply-add/sub: (rm, a, b, c).
                let _rm = self.eval_expr(arg1, tyenv)?;
                let v2 = self.eval_expr(arg2, tyenv)?;
                let v3 = self.eval_expr(arg3, tyenv)?;
                let v4 = self.eval_expr(arg4, tyenv)?;
                VEXOps::qop(*op, v2, v3, v4, self.ctx)
                    .map_err(|_| ExecutionError::Unsupported(format!("qop {:?}", op)))
            }

            IRExpr::CCall { cee, retty, args } => {
                // Evaluate all arguments first
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr(arg, tyenv)?);
                }

                // Try to handle the clean call
                if let Some(result) = ccall::handle_ccall(&cee.name, &arg_vals, retty.bits()) {
                    return Ok(result);
                }

                // Surface unhandled CCalls as errors rather than silently
                // returning concrete 0 — on amd64/x86 that corrupts the
                // rflags/eflags result and miscompiles downstream branches.
                Err(ExecutionError::Unsupported(format!("CCall {}", cee.name)))
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                Err(ExecutionError::Unsupported("special expr".to_string()))
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
                // Only use lower 128 bits for now
                RustBV::concrete(v[0] as u128 | ((v[1] as u128) << 64), 128)
            }
        }
    }

    /// Evaluate the next address from an IRSB.
    fn eval_next_addr(&self, irsb: &IRSB) -> Result<u64, ExecutionError> {
        let next_val = self.eval_expr(&irsb.next, &irsb.tyenv)?;
        next_val
            .as_u64()
            .ok_or_else(|| ExecutionError::Unsupported("symbolic next address".to_string()))
    }

    /// Handle the default exit (end of block).
    fn handle_default_exit(&mut self, irsb: &IRSB) -> Result<ExecutionResult, ExecutionError> {
        let next_val = self.eval_expr(&irsb.next, &irsb.tyenv)?;

        if let Some(addr) = next_val.as_u64() {
            Ok(self.handle_exit(addr, irsb.jumpkind))
        } else {
            // Symbolic next address
            Err(ExecutionError::Unsupported(
                "symbolic next address requires address concretization".to_string(),
            ))
        }
    }

    /// Handle an exit (update PC, return result).
    fn handle_exit(&mut self, target: u64, jumpkind: JumpKind) -> ExecutionResult {
        self.set_pc(target);

        if jumpkind.is_syscall() {
            // Get syscall number from RAX (for AMD64)
            let syscall_num = self.registers.get(16, 8, self.ctx); // RAX offset = 16
            let num = syscall_num.as_u64().unwrap_or(0);
            return ExecutionResult::Syscall { num };
        }

        if self.is_hooked(target) {
            return ExecutionResult::Hook { addr: target };
        }

        ExecutionResult::BlockEnd {
            next_addr: target,
            jumpkind,
        }
    }

    /// Fork the interpreter state.
    pub fn fork(&self) -> VEXInterpreter<'a> {
        VEXInterpreter {
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
            pc: self.pc,
            current_insn_addr: self.current_insn_addr,
            hook_addrs: self.hook_addrs.clone(),
            store_log: Vec::new(), // Fresh store log for fork
            dirty_registers: 0,    // Fresh dirty tracking for fork
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vex::{Endness, IRExpr, IROp, IRSBBuilder, IRType};

    #[test]
    fn test_simple_add() {
        let ctx = SymContext::new_mock();
        let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);

        // Build a simple block: rax = rbx + 5
        let mut builder = IRSBBuilder::new(0x1000, VexArch::AMD64);
        builder.offs_ip(184);

        builder.imark(0x1000, 4);

        // t0 = GET:I64(rbx) ; offset 40
        let t0 = builder.new_tmp(IRType::I64);
        builder.wrtmp(t0, IRExpr::get(40, IRType::I64));

        // t1 = Add64(t0, 5)
        let t1 = builder.new_tmp(IRType::I64);
        builder.wrtmp(
            t1,
            IRExpr::binop(
                IROp::Add(IRType::I64),
                IRExpr::tmp(t0),
                IRExpr::const_u64(5),
            ),
        );

        // PUT(rax) = t1 ; offset 16
        builder.put(16, IRExpr::tmp(t1));

        // Next: 0x1004
        builder.next(IRExpr::const_u64(0x1004), JumpKind::Boring);

        let irsb = builder.build();

        // Set RBX = 10
        interp.registers.put(40, RustBV::concrete(10, 64));

        // Execute
        let result = interp.execute_block(&irsb).unwrap();

        // Check result
        match result {
            ExecutionResult::BlockEnd { next_addr, .. } => {
                assert_eq!(next_addr, 0x1004);
            }
            _ => panic!("unexpected result: {:?}", result),
        }

        // Check RAX = 15
        let rax = interp.registers.get(16, 8, &ctx);
        assert_eq!(rax.as_u64(), Some(15));
    }

    #[test]
    fn test_conditional_exit() {
        let ctx = SymContext::new_mock();
        let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);

        // Build: if (rax == 0) goto 0x2000 else fallthrough to 0x1004
        let mut builder = IRSBBuilder::new(0x1000, VexArch::AMD64);
        builder.offs_ip(184);

        builder.imark(0x1000, 4);

        // t0 = GET:I64(rax)
        let t0 = builder.new_tmp(IRType::I64);
        builder.wrtmp(t0, IRExpr::get(16, IRType::I64));

        // t1 = CmpEQ64(t0, 0)
        let t1 = builder.new_tmp(IRType::I1);
        builder.wrtmp(
            t1,
            IRExpr::binop(
                IROp::CmpEQ(IRType::I64),
                IRExpr::tmp(t0),
                IRExpr::const_u64(0),
            ),
        );

        // if (t1) goto 0x2000
        builder.exit(IRExpr::tmp(t1), 0x2000, JumpKind::Boring);

        // Default: goto 0x1004
        builder.next(IRExpr::const_u64(0x1004), JumpKind::Boring);

        let irsb = builder.build();

        // Test 1: RAX = 0, should take the exit
        interp.registers.put(16, RustBV::concrete(0, 64));
        let result = interp.execute_block(&irsb).unwrap();
        match result {
            ExecutionResult::BlockEnd { next_addr, .. } => {
                assert_eq!(next_addr, 0x2000);
            }
            _ => panic!("unexpected result: {:?}", result),
        }

        // Test 2: RAX != 0, should fall through
        interp.registers.put(16, RustBV::concrete(42, 64));
        let result = interp.execute_block(&irsb).unwrap();
        match result {
            ExecutionResult::BlockEnd { next_addr, .. } => {
                assert_eq!(next_addr, 0x1004);
            }
            _ => panic!("unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_symbolic_branch() {
        let ctx = SymContext::new_mock();
        let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);

        // Build: if (symbolic_rax == 0) goto 0x2000 else fallthrough to 0x1004
        let mut builder = IRSBBuilder::new(0x1000, VexArch::AMD64);
        builder.offs_ip(184);

        builder.imark(0x1000, 4);

        // t0 = GET:I64(rax) - this will be symbolic
        let t0 = builder.new_tmp(IRType::I64);
        builder.wrtmp(t0, IRExpr::get(16, IRType::I64));

        // t1 = CmpEQ64(t0, 0)
        let t1 = builder.new_tmp(IRType::I1);
        builder.wrtmp(
            t1,
            IRExpr::binop(
                IROp::CmpEQ(IRType::I64),
                IRExpr::tmp(t0),
                IRExpr::const_u64(0),
            ),
        );

        // if (t1) goto 0x2000
        builder.exit(IRExpr::tmp(t1), 0x2000, JumpKind::Boring);

        // Default: goto 0x1004
        builder.next(IRExpr::const_u64(0x1004), JumpKind::Boring);

        let irsb = builder.build();

        // Set RAX to symbolic
        let sym_rax = RustBV::symbolic(&ctx, "rax", 64);
        interp.registers.put(16, sym_rax);

        // Execute - should return SymbolicBranch since both paths are possible
        let result = interp.execute_block(&irsb).unwrap();
        match result {
            ExecutionResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => {
                assert_eq!(true_target, 0x2000);
                assert_eq!(false_target, 0x1004);
            }
            _ => panic!("expected SymbolicBranch, got {:?}", result),
        }
    }

    #[test]
    fn test_memory_load_store() {
        let ctx = SymContext::new_mock();
        let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);

        // Map some memory
        interp
            .memory
            .map(0x400000, 0x1000, crate::memory::Permission::RWX);

        // Build: store 0x12345678 at [0x400100], then load it into rax
        let mut builder = IRSBBuilder::new(0x1000, VexArch::AMD64);
        builder.offs_ip(184);

        builder.imark(0x1000, 8);

        // Store 0x12345678 at 0x400100
        let t0 = builder.new_tmp(IRType::I64);
        builder.wrtmp(t0, IRExpr::const_u64(0x400100));
        builder.store(
            IRExpr::tmp(t0),
            IRExpr::const_u32(0x12345678),
            Endness::Little,
        );

        // Load from 0x400100 into t1
        let t1 = builder.new_tmp(IRType::I32);
        builder.wrtmp(
            t1,
            IRExpr::load(IRExpr::const_u64(0x400100), IRType::I32, Endness::Little),
        );

        // Zero-extend t1 to 64 bits and put in RAX
        let t2 = builder.new_tmp(IRType::I64);
        builder.wrtmp(
            t2,
            IRExpr::unop(
                IROp::ZeroExtend {
                    from: IRType::I32,
                    to: IRType::I64,
                },
                IRExpr::tmp(t1),
            ),
        );
        builder.put(16, IRExpr::tmp(t2));

        builder.next(IRExpr::const_u64(0x1008), JumpKind::Boring);

        let irsb = builder.build();

        // Execute
        let result = interp.execute_block(&irsb).unwrap();
        assert!(matches!(result, ExecutionResult::BlockEnd { .. }));

        // Check RAX
        let rax = interp.registers.get(16, 8, &ctx);
        assert_eq!(rax.as_u64(), Some(0x12345678));
    }

    #[test]
    fn test_sse_addss() {
        // Test ADDSS: xmm0 = xmm0 + xmm1 (scalar float add)
        // XMM0: 1.0f, XMM1: 2.0f, Result: 3.0f
        let ctx = SymContext::new_mock();
        let mut interp = VEXInterpreter::new(VexArch::X86, &ctx);

        // X86 offsets: XMM0 = 160, XMM1 = 176
        let xmm0_offset = 160u32;
        let xmm1_offset = 176u32;

        // Set XMM0 = 1.0f (as 128-bit value with 1.0f in low 32 bits)
        let f1_bits = 1.0f32.to_bits() as u128;
        interp
            .registers
            .put(xmm0_offset, RustBV::concrete(f1_bits, 128));

        // Set XMM1 = 2.0f
        let f2_bits = 2.0f32.to_bits() as u128;
        interp
            .registers
            .put(xmm1_offset, RustBV::concrete(f2_bits, 128));

        // Build IRSB for ADDSS:
        // t1 = GET:V128(176)  ; xmm1
        // t2 = GET:V128(160)  ; xmm0
        // t0 = Add32F0x4(t2, t1)
        // PUT(160) = t0
        let mut builder = IRSBBuilder::new(0x1000, VexArch::X86);
        builder.offs_ip(68); // EIP offset for x86
        builder.imark(0x1000, 4);

        // t1 = GET:V128(xmm1)
        let t1 = builder.new_tmp(IRType::V128);
        builder.wrtmp(t1, IRExpr::get(xmm1_offset, IRType::V128));

        // t2 = GET:V128(xmm0)
        let t2 = builder.new_tmp(IRType::V128);
        builder.wrtmp(t2, IRExpr::get(xmm0_offset, IRType::V128));

        // t0 = VFAddS{F32}(t2, t1)
        let t0 = builder.new_tmp(IRType::V128);
        builder.wrtmp(
            t0,
            IRExpr::binop(
                IROp::VFAddS { elem: IRType::F32 },
                IRExpr::tmp(t2),
                IRExpr::tmp(t1),
            ),
        );

        // PUT(xmm0) = t0
        builder.put(xmm0_offset, IRExpr::tmp(t0));

        // Next: 0x1004
        builder.next(IRExpr::const_u32(0x1004), JumpKind::Boring);

        let irsb = builder.build();

        // Print IRSB for debugging
        println!("IRSB: {:?}", irsb);

        // Print initial register values
        let xmm0_before = interp.registers.get(xmm0_offset, 16, &ctx);
        let xmm1_before = interp.registers.get(xmm1_offset, 16, &ctx);
        println!("XMM0 before: {:?}", xmm0_before.as_u128());
        println!("XMM1 before: {:?}", xmm1_before.as_u128());

        // Execute
        let result = interp.execute_block(&irsb);
        println!("Execution result: {:?}", result);

        // Check XMM0 = 3.0f
        let xmm0_after = interp.registers.get(xmm0_offset, 16, &ctx);
        println!("XMM0 after: {:?}", xmm0_after.as_u128());

        let expected = 3.0f32.to_bits() as u128;
        let actual = xmm0_after.as_u128().unwrap() & 0xFFFFFFFF;
        println!("Expected low 32 bits: 0x{:08x}", expected);
        println!("Actual low 32 bits: 0x{:08x}", actual);

        assert_eq!(
            actual, expected,
            "ADDSS: expected 3.0f, got different value"
        );
    }
}
