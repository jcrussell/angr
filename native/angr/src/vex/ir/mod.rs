//! VEX Intermediate Representation types.
//!
//! This module defines the VEX IR in Rust, matching the libVEX specification.
//! The key difference from libVEX is that operations are parameterized by width
//! rather than having separate opcodes for each width (DRY principle).

/// A VEX IR Super Block (IRSB) - a sequence of statements ending in a jump.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct IRSB {
    /// The address of this block.
    pub addr: u64,
    /// The architecture this block was lifted for.
    pub arch: VexArch,
    /// Statements in execution order.
    pub statements: Vec<IRStmt>,
    /// The default exit (fallthrough) expression.
    pub next: IRExpr,
    /// Jump kind for the default exit.
    pub jumpkind: JumpKind,
    /// Offset into the guest state for the IP register.
    pub offsIP: u32,
    /// Number of temporary variables used.
    pub tyenv: TypeEnv,
}

impl IRSB {
    /// Create a new empty IRSB.
    pub fn new(addr: u64, arch: VexArch) -> Self {
        IRSB {
            addr,
            arch,
            statements: Vec::new(),
            next: IRExpr::Const(IRConst::U64(addr)),
            jumpkind: JumpKind::Boring,
            offsIP: 0,
            tyenv: TypeEnv::new(),
        }
    }

    /// Get the number of instructions in this block.
    ///
    /// No production callers (only sibling `size()` is used, in
    /// interpreter/code_invalidation.rs + execution.rs); retained for
    /// VEX-ABI parity and future block-size logic, see angr-36vvn.9.
    pub fn num_instructions(&self) -> usize {
        self.statements
            .iter()
            .filter(|s| matches!(s, IRStmt::IMark { .. }))
            .count()
    }

    /// Get the byte size of this block.
    pub fn size(&self) -> u32 {
        self.statements
            .iter()
            .filter_map(|s| {
                if let IRStmt::IMark { len, .. } = s {
                    Some(*len)
                } else {
                    None
                }
            })
            .sum()
    }
}

/// Type environment for temporary variables.
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    /// Types of temporary variables, indexed by temp number.
    pub types: Vec<IRType>,
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv { types: Vec::new() }
    }

    pub fn new_temp(&mut self, ty: IRType) -> u32 {
        let tmp = self.types.len() as u32;
        self.types.push(ty);
        tmp
    }

    pub fn get(&self, tmp: u32) -> Option<IRType> {
        self.types.get(tmp as usize).copied()
    }
}

mod arch;
mod ast;
mod descriptors;
mod ops_def;
mod types;

pub use arch::*;
pub use ast::*;
pub use descriptors::*;
pub use ops_def::*;
pub use types::*;

test_submod!("../ir_tests.rs" => tests);
