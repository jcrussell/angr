//! The IR syntax tree proper: `IRStmt` (the statement forms a block executes
//! in order) and `IRExpr` (the pure expression forms they evaluate), plus
//! `IRExpr::get_type`, which resolves an expression's `IRType` against the
//! block's `TypeEnv`. The leaf payloads these two refer to live in sibling
//! files: constants and types in `types`, operations in `ops_def`, and the
//! `IRDirty` / `IRRegArray` / `IRLoadGOp` descriptors in `descriptors`.

use super::*;

/// IR statement types.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub enum IRStmt {
    /// No operation.
    NoOp,

    /// Instruction marker - marks the start of a guest instruction.
    IMark {
        /// Address of the instruction.
        addr: u64,
        /// Length of the instruction in bytes.
        len: u32,
        /// Delta from block start (usually 0).
        delta: u8,
    },

    /// ABI hint (no semantic effect).
    AbiHint {
        base: Box<IRExpr>,
        len: u32,
        nia: Box<IRExpr>,
    },

    /// Write to a guest register.
    Put {
        /// Offset into guest state.
        offset: u32,
        /// Data to write.
        data: IRExpr,
    },

    /// Write to a guest register with a guard.
    PutI {
        descr: IRRegArray,
        ix: Box<IRExpr>,
        bias: u32,
        data: Box<IRExpr>,
    },

    /// Write to a temporary variable.
    WrTmp {
        /// Temporary variable number.
        tmp: u32,
        /// Data to write.
        data: IRExpr,
    },

    /// Store to memory.
    Store {
        /// Target address.
        addr: IRExpr,
        /// Data to store.
        data: IRExpr,
        /// Endianness.
        endness: Endness,
    },

    /// Store with guard (conditional store).
    StoreG {
        addr: Box<IRExpr>,
        data: Box<IRExpr>,
        guard: Box<IRExpr>,
        endness: Endness,
    },

    /// Load with guard (conditional load).
    LoadG {
        dst: u32,
        addr: Box<IRExpr>,
        alt: Box<IRExpr>,
        guard: Box<IRExpr>,
        cvt: IRLoadGOp,
        endness: Endness,
    },

    /// Compare and swap.
    CAS {
        /// Temporary for old value.
        old_hi: Option<u32>,
        old_lo: u32,
        /// Address.
        addr: Box<IRExpr>,
        /// Expected values.
        expdHi: Option<Box<IRExpr>>,
        expdLo: Box<IRExpr>,
        /// New values.
        dataHi: Option<Box<IRExpr>>,
        dataLo: Box<IRExpr>,
        endness: Endness,
    },

    /// Load-linked (for LL/SC memory).
    LLSC {
        storedata: Option<Box<IRExpr>>,
        result: u32,
        addr: Box<IRExpr>,
        endness: Endness,
    },

    /// Memory barrier/fence.
    MBE(MBusEvent),

    /// Dirty call to a helper function.
    Dirty(IRDirty),

    /// Conditional exit (branch).
    Exit {
        /// Guard condition.
        guard: IRExpr,
        /// Target address (constant).
        dst: u64,
        /// Jump kind.
        jk: JumpKind,
        /// Offset of IP in guest state.
        offsIP: u32,
    },
}

/// IR expression types.
#[derive(Debug, Clone)]
pub enum IRExpr {
    /// A constant value.
    Const(IRConst),

    /// Read from a temporary variable.
    RdTmp(u32),

    /// Read from a guest register.
    Get {
        /// Offset into guest state.
        offset: u32,
        /// Type of the value.
        ty: IRType,
    },

    /// Read from a rotating guest register array.
    GetI {
        descr: IRRegArray,
        ix: Box<IRExpr>,
        bias: u32,
    },

    /// Load from memory.
    Load {
        /// Address to load from.
        addr: Box<IRExpr>,
        /// Type of the loaded value.
        ty: IRType,
        /// Endianness.
        endness: Endness,
    },

    /// Unary operation.
    Unop {
        /// The operation.
        op: IROp,
        /// The argument.
        arg: Box<IRExpr>,
    },

    /// Binary operation.
    Binop {
        /// The operation.
        op: IROp,
        /// Left argument.
        left: Box<IRExpr>,
        /// Right argument.
        right: Box<IRExpr>,
    },

    /// Ternary operation.
    Triop {
        op: IROp,
        arg1: Box<IRExpr>,
        arg2: Box<IRExpr>,
        arg3: Box<IRExpr>,
    },

    /// Quaternary operation.
    Qop {
        op: IROp,
        arg1: Box<IRExpr>,
        arg2: Box<IRExpr>,
        arg3: Box<IRExpr>,
        arg4: Box<IRExpr>,
    },

    /// If-then-else expression.
    ITE {
        /// Condition (1-bit).
        cond: Box<IRExpr>,
        /// Value if true.
        iftrue: Box<IRExpr>,
        /// Value if false.
        iffalse: Box<IRExpr>,
    },

    /// Call to a clean helper function.
    CCall {
        /// Callee information.
        cee: IRCallee,
        /// Return type.
        retty: IRType,
        /// Arguments.
        args: Vec<IRExpr>,
    },

    /// Undefined value (for modeling undefined behavior).
    VECRET,
    GSPTR,
}

impl IRExpr {
    /// Get the type of this expression (requires type environment).
    pub fn get_type(&self, tyenv: &TypeEnv) -> Option<IRType> {
        match self {
            IRExpr::Const(c) => Some(c.get_type()),
            IRExpr::RdTmp(tmp) => tyenv.get(*tmp),
            IRExpr::Get { ty, .. } => Some(*ty),
            IRExpr::GetI { descr, .. } => Some(descr.elemTy),
            IRExpr::Load { ty, .. } => Some(*ty),
            IRExpr::Unop { op, .. } => op.result_type(),
            IRExpr::Binop { op, .. } => op.result_type(),
            IRExpr::Triop { op, .. } => op.result_type(),
            IRExpr::Qop { op, .. } => op.result_type(),
            IRExpr::ITE { iftrue, .. } => iftrue.get_type(tyenv),
            IRExpr::CCall { retty, .. } => Some(*retty),
            IRExpr::VECRET | IRExpr::GSPTR => None,
        }
    }
}
