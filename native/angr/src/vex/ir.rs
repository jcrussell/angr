//! VEX Intermediate Representation types.
//!
//! This module defines the VEX IR in Rust, matching the libVEX specification.
//! The key difference from libVEX is that operations are parameterized by width
//! rather than having separate opcodes for each width (DRY principle).

use crate::symbolic::BitWidth;

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

/// VEX architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VexArch {
    X86,
    AMD64,
    ARM,
    ARM64,
    MIPS32,
    MIPS64,
    PPC32,
    PPC64,
    S390X,
}

impl VexArch {
    /// Get the pointer size for this architecture in bits.
    pub fn pointer_size(&self) -> u32 {
        match self {
            VexArch::X86 | VexArch::ARM | VexArch::MIPS32 | VexArch::PPC32 => 32,
            VexArch::AMD64 | VexArch::ARM64 | VexArch::MIPS64 | VexArch::PPC64 | VexArch::S390X => {
                64
            }
        }
    }

    /// Get the endianness.
    pub fn endness(&self) -> Endness {
        match self {
            VexArch::X86 | VexArch::AMD64 | VexArch::ARM | VexArch::ARM64 => Endness::Little,
            VexArch::MIPS32
            | VexArch::MIPS64
            | VexArch::PPC32
            | VexArch::PPC64
            | VexArch::S390X => Endness::Big,
        }
    }
}

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

/// IR constant values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IRConst {
    U1(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    F32(f32),
    F64(f64),
    V128(u128),
    V256([u64; 4]),
}

impl IRConst {
    /// Get the type of this constant.
    pub fn get_type(&self) -> IRType {
        match self {
            IRConst::U1(_) => IRType::I1,
            IRConst::U8(_) => IRType::I8,
            IRConst::U16(_) => IRType::I16,
            IRConst::U32(_) => IRType::I32,
            IRConst::U64(_) => IRType::I64,
            IRConst::U128(_) => IRType::I128,
            IRConst::F32(_) => IRType::F32,
            IRConst::F64(_) => IRType::F64,
            IRConst::V128(_) => IRType::V128,
            IRConst::V256(_) => IRType::V256,
        }
    }

    /// Get the value as u128.
    pub fn as_u128(&self) -> u128 {
        match self {
            IRConst::U1(v) => *v as u128,
            IRConst::U8(v) => *v as u128,
            IRConst::U16(v) => *v as u128,
            IRConst::U32(v) => *v as u128,
            IRConst::U64(v) => *v as u128,
            IRConst::U128(v) => *v,
            IRConst::F32(v) => v.to_bits() as u128,
            IRConst::F64(v) => v.to_bits() as u128,
            IRConst::V128(v) => *v,
            IRConst::V256(v) => {
                // Only return lower 128 bits
                v[0] as u128 | ((v[1] as u128) << 64)
            }
        }
    }
}

/// IR types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IRType {
    /// 1-bit integer (boolean).
    I1,
    /// 8-bit integer.
    I8,
    /// 16-bit integer.
    I16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 128-bit integer.
    I128,
    /// 32-bit float (IEEE 754).
    F32,
    /// 64-bit float (IEEE 754).
    F64,
    /// 80-bit float (x87).
    F80,
    /// 16-bit float (IEEE 754).
    F16,
    /// 128-bit vector.
    V128,
    /// 256-bit vector.
    V256,
}

impl IRType {
    /// Get the size in bits.
    pub fn bits(&self) -> u32 {
        match self {
            IRType::I1 => 1,
            IRType::I8 => 8,
            IRType::I16 => 16,
            IRType::I32 => 32,
            IRType::I64 => 64,
            IRType::I128 => 128,
            IRType::F16 => 16,
            IRType::F32 => 32,
            IRType::F64 => 64,
            IRType::F80 => 80,
            IRType::V128 => 128,
            IRType::V256 => 256,
        }
    }

    /// Get the size in bytes (rounded up).
    pub fn bytes(&self) -> u32 {
        (self.bits() + 7) / 8
    }

    /// Convert to BitWidth if applicable.
    pub fn to_bit_width(&self) -> Option<BitWidth> {
        match self {
            IRType::I1 => Some(BitWidth::W1),
            IRType::I8 => Some(BitWidth::W8),
            IRType::I16 => Some(BitWidth::W16),
            IRType::I32 => Some(BitWidth::W32),
            IRType::I64 => Some(BitWidth::W64),
            IRType::I128 | IRType::V128 => Some(BitWidth::W128),
            _ => None,
        }
    }

    /// Check if this is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            IRType::I1 | IRType::I8 | IRType::I16 | IRType::I32 | IRType::I64 | IRType::I128
        )
    }

    /// Check if this is a float type.
    pub fn is_float(&self) -> bool {
        matches!(self, IRType::F16 | IRType::F32 | IRType::F64 | IRType::F80)
    }

    /// Check if this is a vector type.
    pub fn is_vector(&self) -> bool {
        matches!(self, IRType::V128 | IRType::V256)
    }
}

/// Kind of FP compare used by SSE scalar-lane and packed-vector compares.
/// `Un` detects NaN (unordered). `Gt` / `Ge` are emitted by ARM NEON
/// (Iop_CmpGT/GE32Fx2) and SSE packed (Iop_CmpGT/GE32Fx4); the scalar-lane
/// SSE compares only emit Eq/Lt/Le/Un.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FCmpKind {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Un,
}

/// IR operations.
///
/// Unlike libVEX which has ~200 separate opcodes (e.g., Iop_Add8, Iop_Add16, ...),
/// we use parameterized operations to reduce code duplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IROp {
    // =========================================================================
    // Arithmetic (parameterized by width)
    // =========================================================================
    Add(IRType),
    Sub(IRType),
    Mul(IRType),
    MullS(IRType), // Signed widening multiply
    MullU(IRType), // Unsigned widening multiply
    DivS(IRType),  // Signed division
    DivU(IRType),  // Unsigned division
    ModS(IRType),  // Signed modulo
    ModU(IRType),  // Unsigned modulo
    Neg(IRType),   // Negation

    /// DivMod: 64-bit dividend / 32-bit divisor -> 64-bit (low=quotient, high=remainder)
    DivModU64to32, // Unsigned
    DivModS64to32, // Signed

    /// DivMod: 128-bit dividend / 64-bit divisor -> 128-bit (low=quotient, high=remainder)
    DivModU128to64, // Unsigned
    DivModS128to64, // Signed

    // =========================================================================
    // Bitwise (parameterized by width)
    // =========================================================================
    And(IRType),
    Or(IRType),
    Xor(IRType),
    Not(IRType),

    // =========================================================================
    // Shifts (parameterized by width)
    // =========================================================================
    Shl(IRType), // Logical left shift
    Shr(IRType), // Logical right shift
    Sar(IRType), // Arithmetic right shift

    // =========================================================================
    // Comparisons (return I1)
    // =========================================================================
    CmpEQ(IRType),  // Equal
    CmpNE(IRType),  // Not equal
    CmpLT(IRType),  // Less than (signed)
    CmpLE(IRType),  // Less or equal (signed)
    CmpLTU(IRType), // Less than (unsigned)
    CmpLEU(IRType), // Less or equal (unsigned)

    // =========================================================================
    // Conversions
    // =========================================================================
    /// Widen with sign extension.
    SignExtend {
        from: IRType,
        to: IRType,
    },
    /// Widen with zero extension.
    ZeroExtend {
        from: IRType,
        to: IRType,
    },
    /// Narrow (truncate).
    Truncate {
        from: IRType,
        to: IRType,
    },

    // =========================================================================
    // Bit manipulation
    // =========================================================================
    Clz(IRType),      // Count leading zeros
    Ctz(IRType),      // Count trailing zeros
    PopCount(IRType), // Population count

    // =========================================================================
    // Floating point operations
    // =========================================================================
    FAdd(IRType),
    FSub(IRType),
    FMul(IRType),
    FDiv(IRType),
    FNeg(IRType),
    FAbs(IRType),
    FSqrt(IRType),
    /// Fused multiply-add: a*b + c (with rounding mode)
    FMAdd(IRType),
    /// Fused multiply-sub: a*b - c (with rounding mode)
    FMSub(IRType),

    // Float comparisons
    FCmpEQ(IRType),
    FCmpLT(IRType),
    FCmpLE(IRType),

    /// SSE scalar-lane FP compare (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}).
    /// Operates on lane 0 only; result is V128 with lane 0 set to all-1s
    /// (e.g. 0xFFFFFFFF for F32, 0xFFFFFFFFFFFFFFFF for F64) on true and 0
    /// on false. Upper lanes are passed through from the left operand.
    /// `Un` is the unordered (NaN-detect) compare.
    FCmpScalarLane {
        kind: FCmpKind,
        ty: IRType,
    },

    /// Packed FP compare (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}).
    /// Each lane independently produces 0 (false) or all-ones (true).
    /// Result width = elem.bits() * count: 32Fx2 -> I64, 32Fx4 / 64Fx2 -> V128.
    /// `Un` is the unordered (NaN-detect) compare.
    FCmpVecPacked {
        kind: FCmpKind,
        elem: IRType,
        count: u8,
    },

    /// x87 FCOM-style compare (Iop_CmpF32/F64/F128).
    /// Returns I32 encoded as 0x00 = GT, 0x01 = LT, 0x40 = EQ, 0x45 = UN.
    FComCC(IRType),

    // Float conversions
    F32toF64,
    F64toF32,
    I32StoF32,
    I32StoF64,
    I64StoF32,
    I64StoF64,
    I32UtoF32,
    I32UtoF64,
    I64UtoF32,
    I64UtoF64,
    F32toI32S,
    F64toI32S,
    F32toI64S,
    F64toI64S,
    F32toI32U,
    F64toI32U,
    F32toI64U,
    F64toI64U,

    // Rounding mode operations
    RoundF32toInt,
    RoundF64toInt,

    // =========================================================================
    // Scalar-in-vector float operations (SSE scalar ops)
    // These operate on element 0 only, passing through other elements.
    // =========================================================================
    /// Scalar float add in vector (e.g., Add32F0x4 for ADDSS)
    VFAddS {
        elem: IRType,
    },
    /// Scalar float sub in vector (e.g., Sub32F0x4 for SUBSS)
    VFSubS {
        elem: IRType,
    },
    /// Scalar float mul in vector (e.g., Mul32F0x4 for MULSS)
    VFMulS {
        elem: IRType,
    },
    /// Scalar float div in vector (e.g., Div32F0x4 for DIVSS)
    VFDivS {
        elem: IRType,
    },
    /// Scalar float sqrt in vector (e.g., Sqrt32F0x4 for SQRTSS)
    VFSqrtS {
        elem: IRType,
    },
    /// Scalar float max in vector (e.g., Max32F0x4 for MAXSS)
    VFMaxS {
        elem: IRType,
    },
    /// Scalar float min in vector (e.g., Min32F0x4 for MINSS)
    VFMinS {
        elem: IRType,
    },
    /// Set low 32 bits of V128 (used by SSE scalar ops)
    SetV128lo32,
    /// Set low 64 bits of V128
    SetV128lo64,

    // =========================================================================
    // SIMD / Vector operations (parameterized)
    // =========================================================================
    /// Vector add: (element_type, num_elements)
    VAdd {
        elem: IRType,
        count: u8,
    },
    /// Vector sub
    VSub {
        elem: IRType,
        count: u8,
    },
    /// Vector mul
    VMul {
        elem: IRType,
        count: u8,
    },
    /// Vector multiply keeping low half (PMULLD)
    VMulLo {
        elem: IRType,
        count: u8,
    },
    /// Vector and
    VAnd(IRType), // V128 or V256
    /// Vector or
    VOr(IRType),
    /// Vector xor
    VXor(IRType),
    /// Vector not
    VNot(IRType),
    /// Vector shift left (by immediate)
    VShlN {
        elem: IRType,
        count: u8,
    },
    /// Vector shift right logical
    VShrN {
        elem: IRType,
        count: u8,
    },
    /// Vector shift right arithmetic
    VSarN {
        elem: IRType,
        count: u8,
    },

    /// NEON vector shift left by vector — `Iop_Shl{N}x{M}` (and `Iop_Sal{N}x{M}`,
    /// which has identical bit-level semantics: left shift on two's complement
    /// is the same operation whether labelled "logical" or "arithmetic").
    /// Both operands are the full vector width; lane `i` of the result is
    /// `lane_a[i] << lane_b[i]`, with the shift amount treated as unsigned
    /// (Z3 `bvshl` semantics — counts ≥ lane width produce zero). Maps to
    /// ARM USHL (DDI 0487 C7.2.310) when the count vector is non-negative;
    /// the negative-count branch of NEON USHL/SSHL is decomposed by libVEX
    /// into a separate `Iop_Shr`/`Iop_Sar`, so this op only sees the
    /// unsigned-count case.
    VShl {
        elem: IRType,
        count: u8,
    },
    /// NEON vector shift right logical by vector — `Iop_Shr{N}x{M}`.
    /// Same shape as `VShl`; uses Z3 `bvlshr`. Maps to ARM USHL with negative
    /// (right) count after libVEX decomposition.
    VShr {
        elem: IRType,
        count: u8,
    },
    /// NEON vector shift right arithmetic by vector — `Iop_Sar{N}x{M}`.
    /// Same shape as `VShl`; uses Z3 `bvashr` (sign-replicating). Maps to
    /// ARM SSHL with negative (right) count after libVEX decomposition.
    VSar {
        elem: IRType,
        count: u8,
    },
    /// Vector compare equal
    VCmpEQ {
        elem: IRType,
        count: u8,
    },
    /// Vector compare greater than
    VCmpGT {
        elem: IRType,
        count: u8,
    },
    /// Interleave high
    VInterleaveLO {
        elem: IRType,
    },
    VInterleaveHI {
        elem: IRType,
    },
    /// Permute/shuffle
    VPerm {
        elem: IRType,
    },
    /// NEON lane extract (Iop_GetElem{N}x{M}): (vec, idx) -> scalar lane.
    /// Binop; idx is Ity_I8. Result width = elem.bits().
    VGetElem {
        elem: IRType,
        count: u8,
    },
    /// NEON lane insert (Iop_SetElem{N}x{M}): (vec, idx, val) -> vec.
    /// Triop in VEX, but not rm-bearing — dispatched through
    /// binop_with_rm by reinterpreting (rm, left, right) as (vec, idx, val).
    VSetElem {
        elem: IRType,
        count: u8,
    },

    /// NEON broadcast scalar to vector (Iop_Dup{N}x{M}): (scalar) -> vec.
    /// Unop. Input width = elem.bits(); result width = elem.bits() * count.
    VDup {
        elem: IRType,
        count: u8,
    },

    /// NEON widen each lane (Iop_Widen{N}{S/U}to{2N}x{M}): (vec) -> vec.
    /// Unop. Input has `count` lanes of width `from`; result has `count` lanes
    /// of width `from.bits()*2`. `signed` selects sign- vs zero-extension.
    VWiden {
        from: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON unary narrow (Iop_NarrowUn{N}to{N/2}x{M}): (vec) -> vec.
    /// Unop. Input has `count` lanes of width `from`; result has `count` lanes
    /// of width `from.bits()/2` (low bits truncated).
    VNarrowUn {
        from: IRType,
        count: u8,
    },

    /// NEON binary narrow (Iop_NarrowBin{N}to{N/2}x{M}): (lo, hi) -> vec.
    /// Binop. `count` is the result lane count; each input has `count/2` lanes
    /// of width `from`. Result has `count` lanes of width `from.bits()/2`.
    VNarrowBin {
        from: IRType,
        count: u8,
    },

    /// NEON unary saturating narrow (Iop_QNarrowUn{N}{S/U}to{N/2}{S/U}x{M}).
    /// Unop. Same shape as VNarrowUn but saturates instead of truncating.
    /// `src_signed` reflects the source interpretation; `dst_signed` the
    /// saturation range (signed -> [-2^(w-1), 2^(w-1)-1], unsigned -> [0, 2^w-1]).
    VQNarrowUn {
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
    },

    /// NEON binary saturating narrow (Iop_QNarrowBin{N}{S/U}to{N/2}{S/U}x{M}).
    /// Binop variant of VQNarrowUn; `count` is total result lanes (each input
    /// contributes `count/2` lanes).
    VQNarrowBin {
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
    },

    /// NEON byte/halfword/word/bit reversal within each lane —
    /// Iop_Reverse{sub_width}sIn{elem.bits()}_x{count}. Reverse the order of
    /// `sub_width`-bit sub-units inside each `elem`-wide lane; `count` lanes
    /// total. Maps to ARM REV16/REV32/REV64 (sub_width = 8/16/32) and RBIT
    /// (sub_width = 1). Result width = elem.bits() * count.
    VReverse {
        sub_width: u8,
        elem: IRType,
        count: u8,
    },

    /// NEON saturating integer add — Iop_QAdd{N}{S/U}x{M}.
    /// Per-lane addition where positive/negative overflow clamps to the lane
    /// type's max/min. Signed lanes clamp to `[-2^(N-1), 2^(N-1)-1]`; unsigned
    /// lanes clamp to `[0, 2^N - 1]`. Maps to ARM VQADD (DDI 0487 C7.2.379)
    /// and SSE PADDS{B,W}/PADDUS{B,W}.
    VQAdd {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON saturating integer sub — Iop_QSub{N}{S/U}x{M}.
    /// Per-lane subtraction with the same clamp semantics as VQAdd. Maps to
    /// ARM VQSUB (DDI 0487 C7.2.395) and SSE PSUBS{B,W}/PSUBUS{B,W}.
    VQSub {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON saturating shift-left by vector — `Iop_QShl{N}x{M}` (unsigned,
    /// `signed=false`) / `Iop_QSal{N}x{M}` (signed, `signed=true`). Unlike
    /// `VQAdd`/`VQSub` the signedness is encoded in the *prefix* (Shl vs Sal),
    /// not a `S`/`U` infix in the suffix. Per-lane semantics:
    ///   * Cast the shift-amount lane as a signed `elem`-bit integer.
    ///   * If `amt >= 0`: shift left by `amt`; saturate the result to
    ///     `[0, 2^N-1]` (unsigned) or `[-2^(N-1), 2^(N-1)-1]` (signed).
    ///     Shifts ≥ lane width force the saturation boundary based on the
    ///     sign of the operand.
    ///   * If `amt < 0`: shift right (logical for unsigned, arithmetic for
    ///     signed) by `-amt`; OOR right shifts collapse to 0 (unsigned) or
    ///     sign-fill (signed).
    /// Maps to ARM UQSHL / SQSHL (DDI 0487 C7.2.327 / C7.2.298). Derived from
    /// libVEX `host_generic_simd64/simd128.c` h_generic_calc_QShl* helpers;
    /// no `_op_generic_QShl` exists in claripy.
    VQShlSat {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise integer add — `Iop_PwAdd{N}x{M}`. Binary; output has the
    /// same lane width and count as the inputs. Per-lane semantics:
    ///   * result[i]            = a[2i]   + a[2i+1]            for i in 0..count/2
    ///   * result[count/2 + i]  = b[2i]   + b[2i+1]            for i in 0..count/2
    /// Maps to ARM VPADD (DDI 0487 C7.2.270) — `Iop_PwAdd32Fx2` (FP variant)
    /// is NOT routed here and remains unimplemented.
    VPwAdd {
        elem: IRType,
        count: u8,
    },

    /// NEON pairwise widening integer add — `Iop_PwAddL{N}{S/U}x{M}`. Unary;
    /// output lane width is `2 * elem`, lane count is `count / 2`, total
    /// width preserved. Per-lane semantics:
    ///   * result[i] = sext_or_zext(a[2i]) + sext_or_zext(a[2i+1])
    /// Maps to ARM SADDLP / UADDLP (DDI 0487 C7.2.348 / C7.2.418).
    VPwAddL {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise integer min — `Iop_PwMin{N}{S/U}x{M}`. Binary; same shape
    /// rules as `VPwAdd` (interleave a-half then b-half). Maps to ARM VPMIN
    /// (DDI 0487 C7.2.273).
    VPwMin {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON pairwise integer max — `Iop_PwMax{N}{S/U}x{M}`. Binary; same shape
    /// rules as `VPwAdd`. Maps to ARM VPMAX (DDI 0487 C7.2.272).
    VPwMax {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON rounding halving add (a.k.a. rounding-average) —
    /// `Iop_Avg{N}{S/U}x{M}`. Binary; output has the same lane shape as the
    /// inputs. Per-lane semantics (widening to `elem+1` bits avoids overflow):
    ///   result[i] = ((a[i] + b[i] + 1) >> 1) truncated to `elem` bits.
    /// Unsigned variants map to ARM URHADD (DDI 0487 C7.2.420) and SSE
    /// PAVGB/PAVGW (which are unsigned-only). Signed variants map to ARM
    /// SRHADD (DDI 0487 C7.2.353). Distinct from the truncating halving add
    /// `(a+b) >> 1` exposed in claripy as `_op_generic_HAdd`.
    VAvg {
        elem: IRType,
        count: u8,
        signed: bool,
    },

    /// NEON per-byte population count — `Iop_Cnt8x{8,16}`. Unary; each 8-bit
    /// lane is replaced by the number of set bits in that lane (0..=8). Result
    /// width preserved (8 * count bits). Maps to ARM CNT (DDI 0487 C7.2.62).
    /// VEX only defines this op for 8-bit lanes — wider element popcounts are
    /// not part of the NEON ISA.
    VCnt {
        count: u8,
    },

    /// NEON per-lane count leading zeros — `Iop_Clz{N}x{M}`. Unary; each
    /// `elem`-wide lane is replaced by its leading-zero count (0..=N). Result
    /// width preserved. Maps to ARM CLZ (DDI 0487 C7.2.57); lanes are 8/16/32
    /// bits across D-reg (total=64) and Q-reg (total=128) shapes.
    VClz {
        elem: IRType,
        count: u8,
    },

    /// NEON per-lane count leading sign bits — `Iop_Cls{N}x{M}`. Unary; each
    /// `elem`-wide lane is replaced by the number of consecutive bits below
    /// the most significant bit that equal the MSB (range 0..=N-1). All-sign
    /// lanes yield `N-1`. Maps to ARM CLS (DDI 0487 C7.2.56); lanes are 8/16/32
    /// bits across D-reg (total=64) and Q-reg (total=128) shapes.
    VCls {
        elem: IRType,
        count: u8,
    },

    /// NEON GF(2) polynomial multiply — `Iop_PolynomialMul8x{8,16}` (non-
    /// widening, `widen=false`) and `Iop_PolynomialMull8x8` (widening,
    /// `widen=true`). Per-lane carry-less multiply over GF(2): for `a*b` with
    /// `a,b` being 8-bit polynomials, the result is XOR of shifted copies of
    /// `b` selected by the bits of `a`. Non-widening returns the low 8 bits
    /// per lane (width preserved). Widening returns 16 bits per lane (output
    /// total = 2 * input total). Maps to ARM PMUL / PMULL (DDI 0487 C7.2.281).
    /// Claripy has no generic equivalent — universality tests use the spec-
    /// replay template against the same XOR-shift primitive.
    VPolynomialMul {
        count: u8,
        widen: bool,
    },

    // =========================================================================
    // Packed integer min/max/abs
    // =========================================================================
    /// Packed integer min (PMINSB/PMINSW/PMINSD/PMINUB/PMINUW/PMINUD)
    VMin {
        elem: IRType,
        count: u8,
        signed: bool,
    },
    /// Packed integer max (PMAXSB/PMAXSW/PMAXSD/PMAXUB/PMAXUW/PMAXUD)
    VMax {
        elem: IRType,
        count: u8,
        signed: bool,
    },
    /// Packed integer absolute value (PABSB/PABSW/PABSD/PABSQ)
    VAbs {
        elem: IRType,
        count: u8,
    },

    // =========================================================================
    // Packed FP arithmetic (whole-vector — *not* the scalar-lane VF*S variants)
    // =========================================================================
    /// Packed float add (ADDPS/ADDPD)
    VFAdd {
        elem: IRType,
        count: u8,
    },
    /// Packed float sub (SUBPS/SUBPD)
    VFSub {
        elem: IRType,
        count: u8,
    },
    /// Packed float mul (MULPS/MULPD)
    VFMul {
        elem: IRType,
        count: u8,
    },
    /// Packed float div (DIVPS/DIVPD)
    VFDiv {
        elem: IRType,
        count: u8,
    },
    /// Packed float sqrt (SQRTPS/SQRTPD)
    VFSqrt {
        elem: IRType,
        count: u8,
    },
    /// Packed float abs (Iop_Abs32Fx4/Iop_Abs64Fx2)
    VFAbs {
        elem: IRType,
        count: u8,
    },
    /// Packed float min (MINPS/MINPD)
    VFMin {
        elem: IRType,
        count: u8,
    },
    /// Packed float max (MAXPS/MAXPD)
    VFMax {
        elem: IRType,
        count: u8,
    },
    /// Packed FP reciprocal estimate (1/x approximation) — RCPPS / NEON FRECPE.
    /// Returns a fresh symbolic per lane: VEX leaves precision implementation-
    /// defined, so binaries that use this typically follow with one or two
    /// Newton-Raphson refinement steps (RecipStep) which converge to the exact
    /// 1/x irrespective of the seed.
    VFRecipEst {
        elem: IRType,
        count: u8,
    },
    /// Packed FP Newton-Raphson reciprocal step — NEON FRECPS.
    /// Mathematically 2.0 - x*y per lane; treated as fresh-symbolic per lane to
    /// match angr Python's conservative handling (no `_op_fgeneric_RecipStep`).
    VFRecipStep {
        elem: IRType,
        count: u8,
    },
    /// Packed FP reciprocal-sqrt estimate (1/sqrt(x) approximation) —
    /// RSQRTPS / NEON FRSQRTE. Fresh-symbolic per lane, see VFRecipEst.
    VFRSqrtEst {
        elem: IRType,
        count: u8,
    },
    /// Packed FP Newton-Raphson reciprocal-sqrt step — NEON FRSQRTS.
    /// Mathematically (3.0 - x*y*y) / 2.0 per lane; treated as fresh-symbolic
    /// per lane (mirrors angr Python, which has no `_op_fgeneric_RSqrtStep`).
    VFRSqrtStep {
        elem: IRType,
        count: u8,
    },
    /// SSE scalar-in-vector reciprocal estimate (RCPSS, Iop_RecipEst32F0x4).
    /// Lane 0 fresh-symbolic, upper lanes pass through from arg.
    VFRecipEstS {
        elem: IRType,
    },
    /// SSE scalar-in-vector reciprocal-sqrt estimate (RSQRTSS,
    /// Iop_RSqrtEst32F0x4). Lane 0 fresh-symbolic, upper lanes pass through.
    VFRSqrtEstS {
        elem: IRType,
    },

    // =========================================================================
    // Special operations
    // =========================================================================
    /// Reinterpret bits as different type.
    Reinterpret {
        from: IRType,
        to: IRType,
    },

    /// High half of multiplication result.
    MulHi {
        ty: IRType,
        signed: bool,
    },

    /// Concatenate two values.
    Concat {
        ty: IRType,
    },

    /// Extract bits.
    Extract {
        from: IRType,
        to: IRType,
        low_bit: u8,
    },

    // =========================================================================
    // x86-specific operations
    // =========================================================================
    /// x86 PCLMUL (carry-less multiply)
    PclmulLQLQ,
    PclmulHQHQ,
    PclmulLQHQ,
    PclmulHQLQ,

    /// x86 CRC32
    Crc32C,

    // =========================================================================
    // ARM/AArch64 NEON SIMD ops (scaffolded — panic on dispatch)
    // =========================================================================
    /// Placeholder for a NEON SIMD opcode that has been mapped from pyvex
    /// (so it does NOT silently fall back to a fresh-symbolic result), but
    /// whose semantics have not been implemented yet. Dispatch (`VEXOps::unop`,
    /// `VEXOps::binop`, etc.) panics with the captured opcode name so missing
    /// NEON coverage is visible immediately. Implementations land one-by-one
    /// in angr-bkcs.2 by replacing the matching `opcode_map` entry with a
    /// real `IROp::V*` variant.
    NeonUnimplemented(&'static str),

    // =========================================================================
    // Unmapped opcode (no entry in `parse_opcode`)
    // =========================================================================
    /// Opcode string that `parse_opcode` could not match to any known
    /// `IROp` variant. Holds an interned `&'static str` of the original
    /// pyvex opcode name (e.g. `"Iop_FakeNotARealOp"`). Dispatch in
    /// `VEXOps::unop`/`binop`/`ternop`/`qop` surfaces this as
    /// `OpError::UnsupportedVexOp { op_name }`, which the engine maps to
    /// `RustUnsupportedVexOpError(op_name, arch)` (angr-tkbr.2). Replaces
    /// the previous silent `IROp::Raw(0)` fallback that lost the name
    /// and produced fresh-symbolic results.
    Unmapped(&'static str),

    // =========================================================================
    // Raw VEX opcode (for unhandled operations)
    // =========================================================================
    /// Fallback for operations not yet implemented.
    Raw(u32),
}

impl IROp {
    /// Get the result type of this operation.
    pub fn result_type(&self) -> Option<IRType> {
        match self {
            // Arithmetic ops return same type as input
            IROp::Add(t)
            | IROp::Sub(t)
            | IROp::Mul(t)
            | IROp::DivS(t)
            | IROp::DivU(t)
            | IROp::ModS(t)
            | IROp::ModU(t)
            | IROp::Neg(t) => Some(*t),

            // Widening multiply
            IROp::MullS(t) | IROp::MullU(t) => match t {
                IRType::I8 => Some(IRType::I16),
                IRType::I16 => Some(IRType::I32),
                IRType::I32 => Some(IRType::I64),
                IRType::I64 => Some(IRType::I128),
                _ => None,
            },

            // DivMod: 64-bit / 32-bit -> 64-bit
            IROp::DivModU64to32 | IROp::DivModS64to32 => Some(IRType::I64),

            // DivMod: 128-bit / 64-bit -> 128-bit
            IROp::DivModU128to64 | IROp::DivModS128to64 => Some(IRType::I128),

            // Bitwise ops return same type
            IROp::And(t)
            | IROp::Or(t)
            | IROp::Xor(t)
            | IROp::Not(t)
            | IROp::Shl(t)
            | IROp::Shr(t)
            | IROp::Sar(t) => Some(*t),

            // Comparisons return I1
            IROp::CmpEQ(_)
            | IROp::CmpNE(_)
            | IROp::CmpLT(_)
            | IROp::CmpLE(_)
            | IROp::CmpLTU(_)
            | IROp::CmpLEU(_) => Some(IRType::I1),

            // Conversions
            IROp::SignExtend { to, .. }
            | IROp::ZeroExtend { to, .. }
            | IROp::Truncate { to, .. } => Some(*to),

            // Bit manipulation returns same type
            IROp::Clz(t) | IROp::Ctz(t) | IROp::PopCount(t) => Some(*t),

            // Float ops
            IROp::FAdd(t)
            | IROp::FSub(t)
            | IROp::FMul(t)
            | IROp::FDiv(t)
            | IROp::FNeg(t)
            | IROp::FAbs(t)
            | IROp::FSqrt(t)
            | IROp::FMAdd(t)
            | IROp::FMSub(t) => Some(*t),

            IROp::FCmpEQ(_) | IROp::FCmpLT(_) | IROp::FCmpLE(_) => Some(IRType::I1),

            // Scalar-lane SSE compares write into a V128 register (lane 0 mask
            // + upper lanes from `left`).
            IROp::FCmpScalarLane { .. } => Some(IRType::V128),

            // Packed FP compare: total = elem.bits() * count.
            // 32Fx2 -> I64, 32Fx4 / 64Fx2 -> V128.
            IROp::FCmpVecPacked { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            // x87 FCOM-style compare encodes the result as a 32-bit value.
            IROp::FComCC(_) => Some(IRType::I32),

            // Scalar-in-vector float ops return V128
            IROp::VFAddS { elem: _elem }
            | IROp::VFSubS { elem: _elem }
            | IROp::VFMulS { elem: _elem }
            | IROp::VFDivS { elem: _elem }
            | IROp::VFSqrtS { elem: _elem }
            | IROp::VFMaxS { elem: _elem }
            | IROp::VFMinS { elem: _elem }
            | IROp::VFRecipEstS { elem: _elem }
            | IROp::VFRSqrtEstS { elem: _elem } => Some(IRType::V128),

            // SetV128lo ops return V128
            IROp::SetV128lo32 | IROp::SetV128lo64 => Some(IRType::V128),

            // Float conversions
            IROp::F32toF64 => Some(IRType::F64),
            IROp::F64toF32 => Some(IRType::F32),
            IROp::I32StoF32 | IROp::I32UtoF32 | IROp::I64StoF32 | IROp::I64UtoF32 => {
                Some(IRType::F32)
            }
            IROp::I32StoF64 | IROp::I32UtoF64 | IROp::I64StoF64 | IROp::I64UtoF64 => {
                Some(IRType::F64)
            }
            IROp::F32toI32S | IROp::F64toI32S | IROp::F32toI32U | IROp::F64toI32U => {
                Some(IRType::I32)
            }
            IROp::F32toI64S | IROp::F64toI64S | IROp::F32toI64U | IROp::F64toI64U => {
                Some(IRType::I64)
            }
            IROp::RoundF32toInt => Some(IRType::F32),
            IROp::RoundF64toInt => Some(IRType::F64),

            // Vector ops
            IROp::VAnd(t) | IROp::VOr(t) | IROp::VXor(t) | IROp::VNot(t) => Some(*t),
            IROp::VAdd { .. }
            | IROp::VSub { .. }
            | IROp::VMul { .. }
            | IROp::VMulLo { .. }
            | IROp::VShlN { .. }
            | IROp::VShrN { .. }
            | IROp::VSarN { .. }
            | IROp::VCmpEQ { .. }
            | IROp::VCmpGT { .. } => Some(IRType::V128),

            // Vector shift by vector (Iop_Shl/Shr/Sar/Sal{N}x{M}): width
            // preserved — total = elem * count, either 64 or 128 bits.
            IROp::VShl { elem, count }
            | IROp::VShr { elem, count }
            | IROp::VSar { elem, count } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }
            IROp::VInterleaveLO { .. } | IROp::VInterleaveHI { .. } | IROp::VPerm { .. } => {
                Some(IRType::V128)
            }

            // GetElem returns one lane.
            IROp::VGetElem { elem, .. } => Some(*elem),

            // SetElem returns the full vector — width = elem * count.
            IROp::VSetElem { elem, count } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            // Dup: total width = elem * count.
            IROp::VDup { elem, count } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            // Widen: each lane doubles in width; total = (from.bits()*2) * count.
            IROp::VWiden { from, count, .. } => {
                let total = from.bits() * 2 * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            // Narrow (unary or binary, saturating or not): each lane halves;
            // total = (from.bits()/2) * count.
            IROp::VNarrowUn { from, count }
            | IROp::VNarrowBin { from, count }
            | IROp::VQNarrowUn { from, count, .. }
            | IROp::VQNarrowBin { from, count, .. } => {
                let total = (from.bits() / 2) * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            // Reverse: width preserved (sub-units permuted within each lane).
            IROp::VReverse { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Saturating add/sub: width preserved.
            IROp::VQAdd { elem, count, .. } | IROp::VQSub { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Saturating shift by vector: width preserved.
            IROp::VQShlSat { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Pairwise add/min/max (non-widening): output width = elem * count.
            IROp::VPwAdd { elem, count }
            | IROp::VPwMin { elem, count, .. }
            | IROp::VPwMax { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Pairwise widening add: output total = input total = elem * count
            // (lane width doubles, lane count halves).
            IROp::VPwAddL { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Rounding halving add (Iop_Avg*): width preserved.
            IROp::VAvg { elem, count, .. } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Per-byte popcount (Iop_Cnt8x{8,16}): width preserved.
            IROp::VCnt { count } => match *count {
                8 => Some(IRType::I64),
                16 => Some(IRType::V128),
                _ => None,
            },

            // Per-lane Clz/Cls: width preserved (lane width = elem bits).
            IROp::VClz { elem, count } | IROp::VCls { elem, count } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    _ => None,
                }
            }

            // Polynomial multiply (Iop_PolynomialMul / Mull): non-widening
            // preserves width; widening doubles per-lane width (8 → 16) so
            // 8x8 → V128 and the (theoretical) 8x16 widening shape (not
            // emitted by libVEX) would be V256.
            IROp::VPolynomialMul { count, widen } => {
                let elem_out = if *widen { 16 } else { 8 };
                let total = elem_out * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            // Packed integer min/max/abs and packed FP arith all return V128 (or V256
            // for AVX variants — we pick V128 to match the rest of the family for now)
            IROp::VMin { .. }
            | IROp::VMax { .. }
            | IROp::VAbs { .. }
            | IROp::VFAdd { .. }
            | IROp::VFSub { .. }
            | IROp::VFMul { .. }
            | IROp::VFDiv { .. }
            | IROp::VFSqrt { .. }
            | IROp::VFAbs { .. }
            | IROp::VFMin { .. }
            | IROp::VFMax { .. } => Some(IRType::V128),

            // Newton-Raphson reciprocal/rsqrt families: result width = elem * count
            // (NEON D-reg variants are I64, Q-reg variants are V128).
            IROp::VFRecipEst { elem, count }
            | IROp::VFRecipStep { elem, count }
            | IROp::VFRSqrtEst { elem, count }
            | IROp::VFRSqrtStep { elem, count } => {
                let total = elem.bits() * (*count as u32);
                match total {
                    64 => Some(IRType::I64),
                    128 => Some(IRType::V128),
                    256 => Some(IRType::V256),
                    _ => None,
                }
            }

            IROp::Reinterpret { to, .. } => Some(*to),
            IROp::MulHi { ty, .. } => Some(*ty),
            IROp::Concat { ty } => Some(*ty),
            IROp::Extract { to, .. } => Some(*to),

            IROp::PclmulLQLQ
            | IROp::PclmulHQHQ
            | IROp::PclmulLQHQ
            | IROp::PclmulHQLQ
            | IROp::Crc32C => Some(IRType::I64),

            // NEON ops are scaffolded — dispatch panics before result_type
            // is consulted in a hot path. Returning None here means callers
            // that *do* peek at the result type (e.g. fallback width guess
            // in expressions.rs) won't crash, but in practice the dispatch
            // panic fires first.
            IROp::NeonUnimplemented(_) => None,

            // Unmapped opcode — dispatch surfaces UnsupportedVexOp before
            // result_type is consulted. None matches the Raw(_) convention.
            IROp::Unmapped(_) => None,

            IROp::Raw(_) => None,
        }
    }

    /// Check if this operation is commutative.
    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            IROp::Add(_)
                | IROp::Mul(_)
                | IROp::MullS(_)
                | IROp::MullU(_)
                | IROp::And(_)
                | IROp::Or(_)
                | IROp::Xor(_)
                | IROp::CmpEQ(_)
                | IROp::CmpNE(_)
        )
    }
}

/// Jump kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum JumpKind {
    /// Normal jump (fallthrough or unconditional).
    Boring,
    /// Function call.
    Call,
    /// Function return.
    Ret,
    /// System call.
    Sys_syscall,
    Sys_int128,
    Sys_int129,
    Sys_int130,
    Sys_int145,
    Sys_int210,
    Sys_sysenter,
    /// Client request (Valgrind).
    ClientReq,
    /// Yield (threading).
    Yield,
    /// Emit warning.
    EmWarn,
    /// Emit fail.
    EmFail,
    /// No redirect.
    NoDecode,
    /// Map fail.
    MapFail,
    /// Invalid instruction.
    InvalICache,
    /// Flush dcache.
    FlushDCache,
    /// Flush dcache line.
    FlushDCacheLine,
    /// Vectorized exit.
    ExtV128,
    /// Extended exit.
    Extension,
}

impl JumpKind {
    /// Check if this is a syscall.
    pub fn is_syscall(&self) -> bool {
        matches!(
            self,
            JumpKind::Sys_syscall
                | JumpKind::Sys_int128
                | JumpKind::Sys_int129
                | JumpKind::Sys_int130
                | JumpKind::Sys_int145
                | JumpKind::Sys_int210
                | JumpKind::Sys_sysenter
        )
    }

    /// Check if this is a function call.
    pub fn is_call(&self) -> bool {
        matches!(self, JumpKind::Call)
    }

    /// Check if this is a function return.
    pub fn is_ret(&self) -> bool {
        matches!(self, JumpKind::Ret)
    }

    /// Return the VEX `Ijk_*` tag name for this jumpkind. Used by the
    /// state.inspect exit dispatcher to match angr Python's exit_jumpkind
    /// attribute convention.
    pub fn ijk_name(&self) -> &'static str {
        match self {
            JumpKind::Boring => "Ijk_Boring",
            JumpKind::Call => "Ijk_Call",
            JumpKind::Ret => "Ijk_Ret",
            JumpKind::Sys_syscall => "Ijk_Sys_syscall",
            JumpKind::Sys_int128 => "Ijk_Sys_int128",
            JumpKind::Sys_int129 => "Ijk_Sys_int129",
            JumpKind::Sys_int130 => "Ijk_Sys_int130",
            JumpKind::Sys_int145 => "Ijk_Sys_int145",
            JumpKind::Sys_int210 => "Ijk_Sys_int210",
            JumpKind::Sys_sysenter => "Ijk_Sys_sysenter",
            JumpKind::ClientReq => "Ijk_ClientReq",
            JumpKind::Yield => "Ijk_Yield",
            JumpKind::EmWarn => "Ijk_EmWarn",
            JumpKind::EmFail => "Ijk_EmFail",
            JumpKind::NoDecode => "Ijk_NoDecode",
            JumpKind::MapFail => "Ijk_MapFail",
            JumpKind::InvalICache => "Ijk_InvalICache",
            JumpKind::FlushDCache => "Ijk_FlushDCache",
            JumpKind::FlushDCacheLine => "Ijk_FlushDCacheLine",
            JumpKind::ExtV128 => "Ijk_ExtV128",
            JumpKind::Extension => "Ijk_Extension",
        }
    }
}

/// Endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endness {
    Little,
    Big,
}

/// Memory bus event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MBusEvent {
    Fence,
    SFence,
    LFence,
    MFence,
}

/// Guarded load operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IRLoadGOp {
    WidenS,
    WidenZ,
    Identity,
}

/// Register array descriptor.
#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct IRRegArray {
    pub base: u32,
    pub elemTy: IRType,
    pub nElems: u32,
}

/// Clean helper callee info.
#[derive(Debug, Clone)]
pub struct IRCallee {
    pub name: String,
    pub addr: u64,
    pub mcx_mask: u32,
}

/// Dirty call info.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct IRDirty {
    pub cee: IRCallee,
    pub guard: Option<Box<IRExpr>>,
    pub tmp: Option<u32>,
    pub mFx: DirtyFx,
    pub mAddr: Option<Box<IRExpr>>,
    pub mSize: u32,
    pub nFxState: u32,
    pub args: Vec<IRExpr>,
}

/// Dirty call side effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyFx {
    None,
    Read,
    Write,
    Modify,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_type_sizes() {
        assert_eq!(IRType::I1.bits(), 1);
        assert_eq!(IRType::I8.bits(), 8);
        assert_eq!(IRType::I32.bits(), 32);
        assert_eq!(IRType::I64.bits(), 64);
        assert_eq!(IRType::V128.bits(), 128);
    }

    #[test]
    fn test_ir_const_types() {
        assert_eq!(IRConst::U8(42).get_type(), IRType::I8);
        assert_eq!(IRConst::U32(42).get_type(), IRType::I32);
        assert_eq!(IRConst::U64(42).get_type(), IRType::I64);
    }

    #[test]
    fn test_irsb_creation() {
        let irsb = IRSB::new(0x1000, VexArch::AMD64);
        assert_eq!(irsb.addr, 0x1000);
        assert_eq!(irsb.num_instructions(), 0);
    }
}
