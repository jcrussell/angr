//! VEX IR lifting interface.
//!
//! This module provides the interface for lifting machine code to VEX IR.
//! Lifting is done via pyvex through a Python callback.

use std::collections::HashMap;

use super::ir::{Endness, IRConst, IRExpr, IROp, IRSB, IRStmt, IRType, JumpKind, VexArch};

/// Errors from VEX lifting.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LiftError {
    /// Invalid architecture.
    #[error("invalid architecture: {0}")]
    InvalidArch(String),
    /// Failed to lift the given bytes.
    #[error("lift failed at 0x{addr:x}: {reason}")]
    LiftFailed { addr: u64, reason: String },
    /// Invalid instruction.
    #[error("invalid instruction at 0x{addr:x}")]
    InvalidInstruction { addr: u64 },
    /// Unsupported feature.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// VEX lifter trait.
///
/// This trait abstracts over different VEX lifting backends.
pub trait VEXLifter {
    /// Lift bytes at the given address to VEX IR.
    fn lift(&self, bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError>;

    /// Lift a single instruction.
    fn lift_insn(&self, bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError> {
        // Default implementation: lift with max_bytes = bytes.len()
        // but only return the first instruction
        let irsb = self.lift(bytes, addr, arch)?;

        // Find the first IMark after the initial one
        let mut found_first = false;
        let mut stmts = Vec::new();

        for stmt in &irsb.statements {
            if let IRStmt::IMark { .. } = stmt {
                if found_first {
                    break;
                }
                found_first = true;
            }
            stmts.push(stmt.clone());
        }

        let mut result = IRSB::new(addr, arch);
        result.statements = stmts;
        result.next = irsb.next.clone();
        result.jumpkind = irsb.jumpkind;
        result.offsIP = irsb.offsIP;
        result.tyenv = irsb.tyenv;

        Ok(result)
    }
}

/// Native VEX lifter (using embedded test data for now).
///
/// In a full implementation, this would use libvex-rs or similar
/// to lift machine code to VEX IR.
pub struct NativeVEXLifter {
    /// Cached lifted blocks.
    cache: parking_lot::RwLock<HashMap<(u64, VexArch), IRSB>>,
}

impl NativeVEXLifter {
    /// Create a new native VEX lifter.
    pub fn new() -> Self {
        NativeVEXLifter {
            cache: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Pre-populate the cache with a lifted block.
    pub fn add_block(&self, irsb: IRSB) {
        let key = (irsb.addr, irsb.arch);
        self.cache.write().insert(key, irsb);
    }
}

impl Default for NativeVEXLifter {
    fn default() -> Self {
        Self::new()
    }
}

impl VEXLifter for NativeVEXLifter {
    fn lift(&self, _bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError> {
        // Check cache first
        if let Some(irsb) = self.cache.read().get(&(addr, arch)) {
            return Ok(irsb.clone());
        }

        // Return error indicating fallback needed
        Err(LiftError::Unsupported(
            "native lifting not available; use pyvex callback".to_string(),
        ))
    }
}

/// Builder for creating VEX IR programmatically (for testing).
pub struct IRSBBuilder {
    irsb: IRSB,
    next_tmp: u32,
}

impl IRSBBuilder {
    /// Create a new builder for a block at the given address.
    pub fn new(addr: u64, arch: VexArch) -> Self {
        IRSBBuilder {
            irsb: IRSB::new(addr, arch),
            next_tmp: 0,
        }
    }

    /// Add an IMark statement.
    pub fn imark(&mut self, addr: u64, len: u32) -> &mut Self {
        self.irsb.statements.push(IRStmt::IMark {
            addr,
            len,
            delta: 0,
        });
        self
    }

    /// Allocate a new temporary variable.
    pub fn new_tmp(&mut self, ty: IRType) -> u32 {
        let tmp = self.next_tmp;
        self.next_tmp += 1;
        self.irsb.tyenv.types.push(ty);
        tmp
    }

    /// Add a WrTmp statement.
    pub fn wrtmp(&mut self, tmp: u32, data: IRExpr) -> &mut Self {
        self.irsb.statements.push(IRStmt::WrTmp { tmp, data });
        self
    }

    /// Add a Put statement.
    pub fn put(&mut self, offset: u32, data: IRExpr) -> &mut Self {
        self.irsb.statements.push(IRStmt::Put { offset, data });
        self
    }

    /// Add a Store statement.
    pub fn store(&mut self, addr: IRExpr, data: IRExpr, endness: Endness) -> &mut Self {
        self.irsb.statements.push(IRStmt::Store {
            addr,
            data,
            endness,
        });
        self
    }

    /// Add an Exit statement.
    pub fn exit(&mut self, guard: IRExpr, dst: u64, jk: JumpKind) -> &mut Self {
        self.irsb.statements.push(IRStmt::Exit {
            guard,
            dst,
            jk,
            offsIP: self.irsb.offsIP,
        });
        self
    }

    /// Set the default exit.
    pub fn next(&mut self, expr: IRExpr, jk: JumpKind) -> &mut Self {
        self.irsb.next = expr;
        self.irsb.jumpkind = jk;
        self
    }

    /// Set the IP offset.
    pub fn offs_ip(&mut self, offset: u32) -> &mut Self {
        self.irsb.offsIP = offset;
        self
    }

    /// Build the IRSB.
    pub fn build(self) -> IRSB {
        self.irsb
    }
}

// Helper functions for creating expressions
impl IRExpr {
    /// Create a constant U64 expression.
    pub fn const_u64(value: u64) -> Self {
        IRExpr::Const(IRConst::U64(value))
    }

    /// Create a constant U32 expression.
    pub fn const_u32(value: u32) -> Self {
        IRExpr::Const(IRConst::U32(value))
    }

    /// Create a constant U8 expression.
    pub fn const_u8(value: u8) -> Self {
        IRExpr::Const(IRConst::U8(value))
    }

    /// Create a constant U1 (boolean) expression.
    pub fn const_bool(value: bool) -> Self {
        IRExpr::Const(IRConst::U1(value))
    }

    /// Create a RdTmp expression.
    pub fn tmp(n: u32) -> Self {
        IRExpr::RdTmp(n)
    }

    /// Create a Get expression.
    pub fn get(offset: u32, ty: IRType) -> Self {
        IRExpr::Get { offset, ty }
    }

    /// Create a Load expression.
    pub fn load(addr: IRExpr, ty: IRType, endness: Endness) -> Self {
        IRExpr::Load {
            addr: Box::new(addr),
            ty,
            endness,
        }
    }

    /// Create a binary operation expression.
    pub fn binop(op: IROp, left: IRExpr, right: IRExpr) -> Self {
        IRExpr::Binop {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a unary operation expression.
    pub fn unop(op: IROp, arg: IRExpr) -> Self {
        IRExpr::Unop {
            op,
            arg: Box::new(arg),
        }
    }

    /// Create an ITE expression.
    pub fn ite(cond: IRExpr, iftrue: IRExpr, iffalse: IRExpr) -> Self {
        IRExpr::ITE {
            cond: Box::new(cond),
            iftrue: Box::new(iftrue),
            iffalse: Box::new(iffalse),
        }
    }
}

#[cfg(test)]
#[path = "lifter_tests.rs"]
mod tests;
