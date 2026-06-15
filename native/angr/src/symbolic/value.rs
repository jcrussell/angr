//! Symbolic bitvector values for VEX execution.
//!
//! The `RustBV` type represents a bitvector that can be either:
//! - Concrete: A known fixed value
//! - Symbolic: Represents an unknown value (with optional Z3 backing)
//! - Expression: A compound expression with operation tree for reconstruction

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::SymContext;
use super::stats::{
    record_bvop_concat, record_bvop_extract, record_bvop_reverse, record_commutative_canonicalize,
    record_zext_cmp_collapse, record_zext_cmp_trivial_decide,
};

/// Bitvector operation type for expression tree reconstruction.
///
/// This enum represents all operations that can be performed on bitvectors,
/// enabling reconstruction of claripy ASTs from Rust expression trees.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BVOp {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Neg,

    // Bitwise operations
    And,
    Or,
    Xor,
    Not,

    // Shift operations
    Shl,
    Lshr,
    Ashr,
    RotL,
    RotR,

    // Conversion operations
    ZeroExt(u32),      // Number of bits to extend
    SignExt(u32),      // Number of bits to extend
    Extract(u32, u32), // (high, low) bit indices
    Concat,

    // Comparison operations (return 1-bit result)
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,

    // Conditional
    Ite,

    // Utility
    Reverse,
    Clz,
    Ctz,
    Popcount,

    // Floating-point operations (Z3 FP theory).
    // Operands are RustBVs holding the IEEE-754 bit pattern at the given
    // precision; the op is applied via Z3 FP and the result is converted
    // back to the IEEE bit-vector. Round-to-nearest-even is used for
    // arith; conversions and comparison results follow IEEE-754.
    Float { kind: FloatOpKind, prec: FloatPrec },
}

/// IEEE-754 precision for symbolic float operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatPrec {
    /// 32-bit single precision (8 ebits, 24 sbits).
    F32,
    /// 64-bit double precision (11 ebits, 53 sbits).
    F64,
}

impl FloatPrec {
    /// Width of the IEEE bit-vector encoding for this precision.
    #[inline]
    pub fn bits(&self) -> u32 {
        match self {
            FloatPrec::F32 => 32,
            FloatPrec::F64 => 64,
        }
    }
}

/// Kinds of symbolic float operations expressible via Z3 FP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatOpKind {
    // Binary arithmetic (round to nearest, ties to even).
    Add,
    Sub,
    Mul,
    Div,
    // Unary.
    Sqrt,
    Neg,
    Abs,
    // Ternary fused multiply-add / multiply-sub: a*b + c, a*b - c.
    Fma,
    Fms,
    // Comparisons: produce 1-bit BV (1 = true).
    CmpEq,
    CmpLt,
    CmpLe,
    /// Unary IEEE-754 isNaN predicate; 1-bit result. Lets NaN detection
    /// avoid building `CmpEq(v, v)` with the operand cloned twice.
    IsNaN,
    /// Round to integer with rounding mode.
    /// operand[0] = rm BV (32-bit, VEX rounding mode 0..3),
    /// operand[1] = value BV (prec.bits()).
    RoundToInt,
    /// Convert int (signed/unsigned 2's complement) BV to FP at `prec`.
    /// operand[0] = src BV at src_bits. RNE rounding implicit.
    ConvertItoF {
        src_bits: u8,
        signed: bool,
    },
    /// Convert FP at `prec` to int BV (signed/unsigned 2's complement).
    /// operand[0] = src FP BV at prec.bits(). RNE rounding implicit.
    ConvertFtoI {
        dst_bits: u8,
        signed: bool,
    },
    /// Convert FP at `prec` to int BV (signed/unsigned 2's complement)
    /// with explicit rounding mode.
    /// operand[0] = rm BV (32-bit), operand[1] = src FP BV at prec.bits().
    ConvertFtoIRm {
        dst_bits: u8,
        signed: bool,
    },
    /// Convert FP at `src_prec` to FP at `prec`. RNE rounding implicit.
    /// operand[0] = src FP BV at src_prec.bits().
    ConvertFtoF {
        src_prec: FloatPrec,
    },
    /// Convert FP at `src_prec` to FP at `prec` with explicit rounding mode.
    /// operand[0] = rm BV (32-bit), operand[1] = src FP BV at src_prec.bits().
    ConvertFtoFRm {
        src_prec: FloatPrec,
    },
    /// Binary FP arithmetic with explicit rounding mode.
    /// operand[0] = rm BV (32-bit, VEX rm low-2-bits 0..3),
    /// operand[1] = a BV at prec.bits(), operand[2] = b BV at prec.bits().
    AddRm,
    SubRm,
    MulRm,
    DivRm,
    /// Unary FP sqrt with explicit rounding mode.
    /// operand[0] = rm BV (32-bit), operand[1] = a BV at prec.bits().
    SqrtRm,
}

impl FloatOpKind {
    /// Number of operands this op consumes.
    #[inline]
    pub fn arity(&self) -> usize {
        match self {
            FloatOpKind::Sqrt
            | FloatOpKind::Neg
            | FloatOpKind::Abs
            | FloatOpKind::IsNaN
            | FloatOpKind::ConvertItoF { .. }
            | FloatOpKind::ConvertFtoI { .. }
            | FloatOpKind::ConvertFtoF { .. } => 1,
            FloatOpKind::Add
            | FloatOpKind::Sub
            | FloatOpKind::Mul
            | FloatOpKind::Div
            | FloatOpKind::CmpEq
            | FloatOpKind::CmpLt
            | FloatOpKind::CmpLe
            | FloatOpKind::RoundToInt
            | FloatOpKind::ConvertFtoIRm { .. }
            | FloatOpKind::ConvertFtoFRm { .. }
            | FloatOpKind::SqrtRm => 2,
            FloatOpKind::Fma
            | FloatOpKind::Fms
            | FloatOpKind::AddRm
            | FloatOpKind::SubRm
            | FloatOpKind::MulRm
            | FloatOpKind::DivRm => 3,
        }
    }

    /// Whether this op produces a 1-bit (boolean) result instead of a BV.
    #[inline]
    pub fn is_compare(&self) -> bool {
        matches!(
            self,
            FloatOpKind::CmpEq | FloatOpKind::CmpLt | FloatOpKind::CmpLe | FloatOpKind::IsNaN
        )
    }

    /// Width (in bits) of the BV result for this op, given the `prec` field
    /// of the enclosing `BVOp::Float`. Compares are 1-bit; FtoI conversions
    /// take their result width from `dst_bits`; everything else returns the
    /// IEEE encoding width of `prec`.
    #[inline]
    pub fn result_bits(&self, prec: FloatPrec) -> u32 {
        match self {
            FloatOpKind::CmpEq | FloatOpKind::CmpLt | FloatOpKind::CmpLe | FloatOpKind::IsNaN => 1,
            FloatOpKind::ConvertFtoI { dst_bits, .. }
            | FloatOpKind::ConvertFtoIRm { dst_bits, .. } => *dst_bits as u32,
            _ => prec.bits(),
        }
    }
}

impl BVOp {
    /// Get the claripy method name for this operation.
    pub fn claripy_method(&self) -> &'static str {
        match self {
            BVOp::Add => "__add__",
            BVOp::Sub => "__sub__",
            BVOp::Mul => "__mul__",
            BVOp::UDiv => "UDiv",
            BVOp::SDiv => "SDiv",
            BVOp::URem => "URem",
            BVOp::SRem => "SMod",
            BVOp::Neg => "__neg__",
            BVOp::And => "__and__",
            BVOp::Or => "__or__",
            BVOp::Xor => "__xor__",
            BVOp::Not => "__invert__",
            BVOp::Shl => "__lshift__",
            BVOp::Lshr => "LShR",
            BVOp::Ashr => "__rshift__",
            BVOp::RotL => "RotateLeft",
            BVOp::RotR => "RotateRight",
            BVOp::ZeroExt(_) => "ZeroExt",
            BVOp::SignExt(_) => "SignExt",
            BVOp::Extract(_, _) => "Extract",
            BVOp::Concat => "Concat",
            BVOp::Eq => "__eq__",
            BVOp::Ne => "__ne__",
            BVOp::Ult => "ULT",
            BVOp::Ule => "ULE",
            BVOp::Ugt => "UGT",
            BVOp::Uge => "UGE",
            BVOp::Slt => "SLT",
            BVOp::Sle => "SLE",
            BVOp::Sgt => "SGT",
            BVOp::Sge => "SGE",
            BVOp::Ite => "If",
            BVOp::Reverse => "Reverse",
            BVOp::Clz => "clz",
            BVOp::Ctz => "ctz",
            BVOp::Popcount => "popcount",
            // No clean claripy mapping — float ops returning to Python fall
            // back to fresh symbolic in rustbv_to_claripy (constraint info
            // stays in Z3 within the Rust engine).
            BVOp::Float { .. } => "fpOp",
        }
    }

    /// Check if this is a unary operation.
    pub fn is_unary(&self) -> bool {
        if let BVOp::Float { kind, .. } = self {
            return kind.arity() == 1;
        }
        matches!(
            self,
            BVOp::Neg
                | BVOp::Not
                | BVOp::ZeroExt(_)
                | BVOp::SignExt(_)
                | BVOp::Extract(_, _)
                | BVOp::Reverse
                | BVOp::Clz
                | BVOp::Ctz
                | BVOp::Popcount
        )
    }

    /// Check if this is a binary operation.
    pub fn is_binary(&self) -> bool {
        if let BVOp::Float { kind, .. } = self {
            return kind.arity() == 2;
        }
        matches!(
            self,
            BVOp::Add
                | BVOp::Sub
                | BVOp::Mul
                | BVOp::UDiv
                | BVOp::SDiv
                | BVOp::URem
                | BVOp::SRem
                | BVOp::And
                | BVOp::Or
                | BVOp::Xor
                | BVOp::Shl
                | BVOp::Lshr
                | BVOp::Ashr
                | BVOp::RotL
                | BVOp::RotR
                | BVOp::Eq
                | BVOp::Ne
                | BVOp::Ult
                | BVOp::Ule
                | BVOp::Ugt
                | BVOp::Uge
                | BVOp::Slt
                | BVOp::Sle
                | BVOp::Sgt
                | BVOp::Sge
        )
    }

    /// Check if this is a ternary operation (ITE or FP fused MAdd/MSub).
    pub fn is_ternary(&self) -> bool {
        if let BVOp::Float { kind, .. } = self {
            return kind.arity() == 3;
        }
        matches!(self, BVOp::Ite)
    }
}

/// Bit width for parameterized operations.
/// This reduces ~200 VEX ops to ~30 parameterized variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitWidth {
    W1 = 1,
    W8 = 8,
    W16 = 16,
    W32 = 32,
    W64 = 64,
    W128 = 128,
}

impl BitWidth {
    #[inline]
    pub fn bits(&self) -> u32 {
        *self as u32
    }

    #[inline]
    pub fn bytes(&self) -> u32 {
        self.bits() / 8
    }

    pub fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            1 => Some(BitWidth::W1),
            8 => Some(BitWidth::W8),
            16 => Some(BitWidth::W16),
            32 => Some(BitWidth::W32),
            64 => Some(BitWidth::W64),
            128 => Some(BitWidth::W128),
            _ => None,
        }
    }

    #[inline]
    pub fn mask(&self) -> u128 {
        match self {
            BitWidth::W1 => 0x1,
            BitWidth::W8 => 0xFF,
            BitWidth::W16 => 0xFFFF,
            BitWidth::W32 => 0xFFFF_FFFF,
            BitWidth::W64 => 0xFFFF_FFFF_FFFF_FFFF,
            BitWidth::W128 => u128::MAX,
        }
    }
}

/// Signedness for comparison and extension operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// A bitvector value that can be concrete or symbolic.
///
/// This is the core value type for the VEX execution engine. All register
/// and memory values are represented as `RustBV`.
///
/// With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
/// lifetime parameters on the Z3 AST.
///
/// ## Serialization
///
/// `RustBV` implements `Serialize`/`Deserialize` via the [`RustBVData`]
/// shadow type — the Z3 AST cache on `Symbolic` is skipped at serialize
/// time and rebuilt on first `to_z3_ast()` call after load
/// (a fresh `BV::new_const(name, width)` in the active Z3 thread-local
/// context). Deserialization must happen with a Z3 context active when
/// the `vex-engine-z3` feature is enabled.
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "RustBVData", into = "RustBVData")]
pub enum RustBV {
    /// A known concrete value.
    Concrete {
        /// The value, masked to `width` bits.
        value: u128,
        /// Width in bits (1, 8, 16, 32, 64, or 128).
        width: u32,
    },
    /// A symbolic value (backed by Z3 when available, otherwise just a name).
    /// This represents a leaf symbolic variable (e.g., BVS("x", 32)).
    Symbolic {
        /// Unique identifier for this symbolic value.
        id: u64,
        /// Width in bits.
        width: u32,
        /// Name (for debugging). `Arc<str>` so cloning a Symbolic is alloc-free —
        /// hot on the RdTmp path where a temp slot is read multiple times per block.
        name: Arc<str>,
        /// Z3 AST when z3 feature is enabled.
        #[cfg(feature = "vex-engine-z3")]
        ast: z3::ast::BV,
    },
    /// Optimization: a symbolic value with a known concrete value.
    Constrained {
        /// The symbolic identifier.
        id: u64,
        /// The known concrete value.
        value: u128,
        /// Width in bits.
        width: u32,
    },
    /// A compound expression with operation tree for claripy reconstruction.
    ///
    /// This variant stores the operation and operands that created this value,
    /// enabling accurate conversion back to claripy ASTs without creating
    /// fresh symbolic variables. Z3 ASTs are computed lazily on demand
    /// via `to_z3_ast()` rather than eagerly at construction time.
    Expression {
        /// Identity discriminant — **NOT** a content hash or interner id.
        ///
        /// In practice this field is always set to [`RustBV::EXPRESSION_ID`]
        /// (`u64::MAX`) by every `Expression` constructor in this module
        /// (see the `define_*_op` macro families and direct builders such as
        /// `add_into`, `mul_into`, `and_into`, etc.). The sentinel
        /// distinguishes compound expressions from leaf symbols (`Symbolic`
        /// / `Constrained`), which carry real allocated ids from
        /// [`crate::symbolic::SymbolicIdentityRegistry`].
        ///
        /// **Why a sentinel, not a per-node hash:** allocating a unique id
        /// for every intermediate operation was rejected for solver
        /// compatibility (only real leaf symbols need tracking by the
        /// claripy-side registry) and to avoid per-construction overhead.
        ///
        /// **Identity for caching purposes** is keyed by *other* fields,
        /// not this one:
        /// - The structural pointer `Arc::as_ptr(operands)` is the cache
        ///   key for `EXPRESSION_BY_OPERANDS_PTR` in `claripy_bridge`
        ///   (stable across `RustBV::clone`, which just bumps the Arc).
        /// - A caller-computed content hash of (op, operands) keys
        ///   `EXPRESSION_CACHE` in `claripy_bridge`.
        ///
        /// **Collision potential:** none — there is no content hash here
        /// to collide on. Two structurally distinct `Expression` values
        /// share the same `id` sentinel.
        id: u64,
        /// Width in bits.
        width: u32,
        /// The operation that created this expression.
        op: BVOp,
        /// The operands to this operation. `Arc<[T]>` lets cloning the outer
        /// Expression bump a single refcount (alloc-free; matters on the RdTmp
        /// path). Operands are stored inline rather than wrapped in `Arc<RustBV>`
        /// so each Expression construction is one allocation instead of N+1.
        operands: Arc<[RustBV]>,
    },
}

/// Serde shadow form for [`RustBV`].
///
/// The wire format mirrors `RustBV` field-for-field except:
///
/// - The `Symbolic` variant's `ast: z3::ast::BV` cache is dropped (Z3
///   ASTs cannot cross process boundaries and are reconstructable from
///   `(name, width)`). On deserialize, the AST is recreated lazily via
///   `BV::new_const(&name, width)` inside the active thread-local Z3
///   context — callers must run within `with_z3_context` (or equivalent)
///   when the `vex-engine-z3` feature is enabled.
/// - `Arc<str>` / `Arc<[RustBV]>` collapse to owned `String` / `Vec`
///   on the wire and rebuild Arc handles on load.
///
/// This is the snapshot/serialization format from the angr-x04s spike
/// (see `snapshot-serialization-design` bd memory). Intermediate cache
/// fields (`EXPRESSION_BY_OPERANDS_PTR`, content hashes) are not part of
/// the wire format; they rewarm naturally as the loaded ops are touched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RustBVData {
    Concrete {
        value: u128,
        width: u32,
    },
    Symbolic {
        id: u64,
        width: u32,
        name: String,
    },
    Constrained {
        id: u64,
        value: u128,
        width: u32,
    },
    Expression {
        id: u64,
        width: u32,
        op: BVOp,
        operands: Vec<RustBVData>,
    },
}

impl From<RustBV> for RustBVData {
    fn from(bv: RustBV) -> Self {
        match bv {
            RustBV::Concrete { value, width } => RustBVData::Concrete { value, width },
            RustBV::Symbolic {
                id, width, name, ..
            } => RustBVData::Symbolic {
                id,
                width,
                name: name.to_string(),
            },
            RustBV::Constrained { id, value, width } => {
                RustBVData::Constrained { id, value, width }
            }
            RustBV::Expression {
                id,
                width,
                op,
                operands,
            } => RustBVData::Expression {
                id,
                width,
                op,
                operands: operands.iter().cloned().map(RustBVData::from).collect(),
            },
        }
    }
}

impl From<RustBVData> for RustBV {
    fn from(data: RustBVData) -> Self {
        match data {
            RustBVData::Concrete { value, width } => RustBV::Concrete { value, width },
            RustBVData::Symbolic { id, width, name } => {
                #[cfg(feature = "vex-engine-z3")]
                {
                    // Rebuild the Z3 AST in the active thread-local context.
                    // Callers must be inside with_z3_context when this runs.
                    let ast = z3::ast::BV::new_const(name.as_str(), width);
                    RustBV::Symbolic {
                        id,
                        width,
                        name: Arc::<str>::from(name),
                        ast,
                    }
                }
                #[cfg(not(feature = "vex-engine-z3"))]
                {
                    RustBV::Symbolic {
                        id,
                        width,
                        name: Arc::<str>::from(name),
                    }
                }
            }
            RustBVData::Constrained { id, value, width } => {
                RustBV::Constrained { id, value, width }
            }
            RustBVData::Expression {
                id,
                width,
                op,
                operands,
            } => RustBV::Expression {
                id,
                width,
                op,
                operands: Arc::<[RustBV]>::from(
                    operands.into_iter().map(RustBV::from).collect::<Vec<_>>(),
                ),
            },
        }
    }
}

/// Generate a borrow + consuming comparison pair for an unsigned ordering
/// operator (Ult/Ule/Ugt/Uge).
///
/// Entry shape:
///   `name: into_name, BVOp::Variant, concrete_op, doc;
///    zext_fwd: ZExtCmp::Variant, zext_swap: ZExtCmp::Variant`
///
/// The concrete branch evaluates `if a $cmp b { 1 } else { 0 }` on the raw
/// `u128` representations. The symbolic branch tries the bidirectional
/// `try_zext_const_cmp_fold` shortcut (angr-g7nq pattern (b)) before falling
/// through to a generic `Expression` node.
macro_rules! define_unsigned_cmp_pair {
    (
        $name:ident, $into_name:ident, $op:expr, $cmp:tt,
        $doc_borrow:literal, $doc_consume:literal,
        $zext_fwd:expr, $zext_swap:expr
    ) => {
        #[doc = $doc_borrow]
        #[inline]
        pub fn $name(&self, other: &Self, ctx: &SymContext) -> Self {
            self.clone().$into_name(other.clone(), ctx)
        }

        #[doc = $doc_consume]
        #[inline]
        pub fn $into_name(self, other: Self, ctx: &SymContext) -> Self {
            debug_assert_eq!(self.width(), other.width());
            match (self.as_u128(), other.as_u128()) {
                (Some(a), Some(b)) => Self::concrete(if a $cmp b { 1 } else { 0 }, 1),
                _ => {
                    if let Some(folded) =
                        try_zext_const_cmp_fold(&self, &other, $zext_fwd, ctx)
                    {
                        return folded;
                    }
                    if let Some(folded) =
                        try_zext_const_cmp_fold(&other, &self, $zext_swap, ctx)
                    {
                        return folded;
                    }
                    RustBV::Expression {
                        id: Self::EXPRESSION_ID,
                        width: 1,
                        op: $op,
                        operands: Arc::<[RustBV]>::from([self, other]),
                    }
                }
            }
        }
    };
}

/// Generate a borrow + consuming comparison pair for a signed ordering
/// operator (Slt/Sle/Sgt/Sge).
///
/// Entry shape: `name: into_name, BVOp::Variant, signed_op, doc`.
///
/// The concrete branch sign-extends both operands to `i128` (via the module-
/// level `sign_extend` helper) before applying the operator. No zext-fold
/// shortcut — that path is unsigned-only.
macro_rules! define_signed_cmp_pair {
    (
        $name:ident, $into_name:ident, $op:expr, $cmp:tt,
        $doc_borrow:literal, $doc_consume:literal
    ) => {
        #[doc = $doc_borrow]
        #[inline]
        pub fn $name(&self, other: &Self, ctx: &SymContext) -> Self {
            self.clone().$into_name(other.clone(), ctx)
        }

        #[doc = $doc_consume]
        #[inline]
        pub fn $into_name(self, other: Self, _ctx: &SymContext) -> Self {
            debug_assert_eq!(self.width(), other.width());
            match (self.as_u128(), other.as_u128()) {
                (Some(a), Some(b)) => {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete(if a_signed $cmp b_signed { 1 } else { 0 }, 1)
                }
                _ => RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: 1,
                    op: $op,
                    operands: Arc::<[RustBV]>::from([self, other]),
                },
            }
        }
    };
}

impl RustBV {
    /// Sentinel ID for expression results (not real symbolic variables).
    /// Using u64::MAX avoids allocating new IDs for every intermediate operation.
    /// This improves solver compatibility since only true symbolic variables
    /// (inputs like `x`, `y`) get unique IDs that need tracking.
    pub const EXPRESSION_ID: u64 = u64::MAX;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a concrete bitvector from a value.
    #[inline]
    pub fn concrete(value: u128, width: u32) -> Self {
        let mask = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        RustBV::Concrete {
            value: value & mask,
            width,
        }
    }

    /// Create a zero bitvector.
    #[inline]
    pub fn zero(width: u32) -> Self {
        RustBV::Concrete { value: 0, width }
    }

    /// Create a bitvector with all bits set to 1.
    #[inline]
    pub fn ones(width: u32) -> Self {
        RustBV::Concrete {
            value: Self::all_ones_mask(width),
            width,
        }
    }

    /// Return the all-ones mask for a given bit width.
    #[inline]
    fn all_ones_mask(width: u32) -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
    }

    /// Create a symbolic bitvector variable.
    ///
    /// Accepts anything String-like via `AsRef<str>`. The internal name is
    /// stored as `Arc<str>` so cloning a Symbolic is alloc-free, which matters
    /// on the RdTmp path where a temp slot is read multiple times per block.
    pub fn symbolic(ctx: &SymContext, name: impl AsRef<str>, width: u32) -> Self {
        let id = ctx.next_id();
        let name: Arc<str> = Arc::from(name.as_ref());
        #[cfg(feature = "vex-engine-z3")]
        {
            let ast = z3::ast::BV::new_const(&*name, width);
            RustBV::Symbolic {
                id,
                width,
                name,
                ast,
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic { id, width, name }
        }
    }

    /// Create a symbolic bitvector variable with a specific ID.
    ///
    /// This is used for identity preservation when the same symbol
    /// was previously imported from Python. By reusing the same ID,
    /// we ensure that constraints on the original symbol apply correctly.
    pub fn symbolic_with_id(id: u64, name: impl AsRef<str>, width: u32) -> Self {
        let name: Arc<str> = Arc::from(name.as_ref());
        #[cfg(feature = "vex-engine-z3")]
        {
            let ast = z3::ast::BV::new_const(&*name, width);
            RustBV::Symbolic {
                id,
                width,
                name,
                ast,
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            RustBV::Symbolic { id, width, name }
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the width in bits.
    #[inline]
    pub fn width(&self) -> u32 {
        match self {
            RustBV::Concrete { width, .. } => *width,
            RustBV::Symbolic { width, .. } => *width,
            RustBV::Constrained { width, .. } => *width,
            RustBV::Expression { width, .. } => *width,
        }
    }

    /// Check if this value is concrete.
    #[inline]
    pub fn is_concrete(&self) -> bool {
        matches!(self, RustBV::Concrete { .. })
    }

    /// Check if this value is symbolic.
    #[inline]
    pub fn is_symbolic(&self) -> bool {
        matches!(
            self,
            RustBV::Symbolic { .. } | RustBV::Constrained { .. } | RustBV::Expression { .. }
        )
    }

    /// Check if this value is an expression (compound symbolic).
    #[inline]
    pub fn is_expression(&self) -> bool {
        matches!(self, RustBV::Expression { .. })
    }

    /// Try to get the concrete value.
    #[inline]
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            RustBV::Concrete { value, .. } => Some(*value),
            RustBV::Constrained { value, .. } => Some(*value),
            RustBV::Symbolic { .. } => None,
            RustBV::Expression { .. } => None,
        }
    }

    /// Get the operation if this is an Expression variant.
    #[inline]
    pub fn op(&self) -> Option<&BVOp> {
        match self {
            RustBV::Expression { op, .. } => Some(op),
            _ => None,
        }
    }

    /// Get the operands if this is an Expression variant.
    #[inline]
    pub fn operands(&self) -> Option<&[RustBV]> {
        match self {
            RustBV::Expression { operands, .. } => Some(operands),
            _ => None,
        }
    }

    /// Get the concrete value, panicking if symbolic.
    #[inline]
    pub fn to_u128(&self) -> u128 {
        self.as_u128().expect("value is symbolic")
    }

    /// Try to get the concrete value as u64.
    #[inline]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_u128().map(|v| v as u64)
    }

    /// Get the concrete value as u64, panicking if symbolic or too wide.
    #[inline]
    pub fn to_u64(&self) -> u64 {
        self.as_u64().expect("value is symbolic or too wide")
    }

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    /// Canonical sort key for commutative-operand ordering (angr-kkpr).
    ///
    /// Z3's mk_bv* hash-cons by AST identity, but `mk_bvadd(x, y)` and
    /// `mk_bvadd(y, x)` produce DIFFERENT Z3 AST nodes — Z3's preprocessing
    /// tactics canonicalize commutative arg order only on the formulas given
    /// to the solver, not on intermediate ASTs used as branch conditions,
    /// register values, etc. Sorting operands by this key at construction
    /// time so two callers that build the same commutative op in opposite
    /// orders produce the same RustBV (and therefore the same Z3 AST).
    ///
    /// Ordering (low → high):
    ///   0: Symbolic    (variables — most "primary")
    ///   1: Constrained
    ///   2: Expression  (subtrees with sub-key from operand-Arc pointer)
    ///   3: Concrete    (constants always end up on the right)
    ///
    /// The Expression sub-key uses the operand-Arc pointer for cheap
    /// within-run determinism. Two structurally-equal Expressions with
    /// distinct Arc allocations sort differently, but Z3 still merges
    /// their ASTs via its own hash-cons at to_z3_ast time, so this does
    /// not weaken Z3-side dedup. The key is not stable across process
    /// runs — that is intentional: RustBVs don't survive across runs.
    #[inline]
    fn canonical_sort_key(&self) -> (u8, u128) {
        match self {
            RustBV::Symbolic { id, .. } => (0, *id as u128),
            RustBV::Constrained { id, .. } => (1, *id as u128),
            RustBV::Expression { operands, .. } => {
                (2, Arc::as_ptr(operands) as *const () as usize as u128)
            }
            RustBV::Concrete { value, .. } => (3, *value),
        }
    }

    /// Sort a commutative operand pair into canonical order.
    ///
    /// Call only after constant-folding short-circuits (e.g. `x + 0 → x`,
    /// `x & all_ones → x`); those rely on the original `(self, other)`
    /// argument order to fire.
    #[inline]
    fn canonicalize_commutative(self, other: Self) -> (Self, Self) {
        if self.canonical_sort_key() <= other.canonical_sort_key() {
            record_commutative_canonicalize(false);
            (self, other)
        } else {
            record_commutative_canonicalize(true);
            (other, self)
        }
    }

    /// Add two bitvectors.
    #[inline]
    pub fn add(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().add_into(other.clone(), ctx)
    }

    /// Add two bitvectors, consuming both arguments.
    ///
    /// Avoids `self.clone()`/`other.clone()` for the Expression branch and
    /// identity simplifications. Hot-path callers (e.g. `VEXOps::binop`) that
    /// already own the operands should prefer this.
    #[inline]
    pub fn add_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_add(b), self.width()),
            // x + 0 → x
            (None, Some(0)) => self,
            // 0 + x → x
            (Some(0), None) => other,
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Add,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Subtract two bitvectors.
    #[inline]
    pub fn sub(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sub_into(other.clone(), ctx)
    }

    /// Subtract two bitvectors, consuming both arguments.
    #[inline]
    pub fn sub_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_sub(b), self.width()),
            // x - 0 → x
            (None, Some(0)) => self,
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Sub,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Multiply two bitvectors.
    #[inline]
    pub fn mul(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().mul_into(other.clone(), ctx)
    }

    /// Multiply two bitvectors, consuming both arguments.
    #[inline]
    pub fn mul_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        let width = self.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a.wrapping_mul(b), width),
            // x * 0 → 0
            (_, Some(0)) | (Some(0), _) => Self::zero(width),
            // x * 1 → x
            (None, Some(1)) => self,
            // 1 * x → x
            (Some(1), None) => other,
            // sym * 2^k → sym << k (avoids Z3's O(N^2) Dadda bit-blast)
            (None, Some(b)) if b.is_power_of_two() => {
                let k = b.trailing_zeros();
                let amt = Self::concrete(k as u128, width);
                self.shl_into(amt, _ctx)
            }
            // 2^k * sym → sym << k
            (Some(a), None) if a.is_power_of_two() => {
                let k = a.trailing_zeros();
                let amt = Self::concrete(k as u128, width);
                other.shl_into(amt, _ctx)
            }
            _ => {
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Mul,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Unsigned division.
    #[inline]
    pub fn udiv(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().udiv_into(other.clone(), ctx)
    }

    /// Unsigned division, consuming both arguments.
    #[inline]
    pub fn udiv_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    Self::ones(self.width())
                } else {
                    Self::concrete(a / b, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::UDiv,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Signed division.
    #[inline]
    pub fn sdiv(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().sdiv_into(other.clone(), ctx)
    }

    /// Signed division, consuming both arguments.
    #[inline]
    pub fn sdiv_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    Self::ones(self.width())
                } else {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete((a_signed / b_signed) as u128, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::SDiv,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Unsigned remainder.
    #[inline]
    pub fn urem(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().urem_into(other.clone(), ctx)
    }

    /// Unsigned remainder, consuming both arguments.
    #[inline]
    pub fn urem_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self
                } else {
                    Self::concrete(a % b, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::URem,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Signed remainder.
    #[inline]
    pub fn srem(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().srem_into(other.clone(), ctx)
    }

    /// Signed remainder, consuming both arguments.
    #[inline]
    pub fn srem_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => {
                if b == 0 {
                    self
                } else {
                    let a_signed = sign_extend(a, self.width());
                    let b_signed = sign_extend(b, self.width());
                    Self::concrete((a_signed % b_signed) as u128, self.width())
                }
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::SRem,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Negate (two's complement).
    #[inline]
    pub fn neg(&self, ctx: &SymContext) -> Self {
        self.clone().neg_into(ctx)
    }

    /// Negate (two's complement), consuming the argument.
    #[inline]
    pub fn neg_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete((!v).wrapping_add(1), self.width()),
            None => {
                // neg(neg(x)) → x
                if let RustBV::Expression {
                    op: BVOp::Neg,
                    operands,
                    ..
                } = &self
                {
                    return operands[0].clone();
                }
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Neg,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    // =========================================================================
    // Bitwise Operations
    // =========================================================================

    /// Bitwise AND.
    #[inline]
    pub fn and(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().and_into(other.clone(), ctx)
    }

    /// Bitwise AND, consuming both arguments.
    #[inline]
    pub fn and_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        let all_ones = Self::all_ones_mask(self.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a & b, self.width()),
            // x & 0 → 0
            (_, Some(0)) | (Some(0), _) => Self::zero(self.width()),
            // x & all_ones → x
            (None, Some(v)) if v == all_ones => self,
            (Some(v), None) if v == all_ones => other,
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::And,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Bitwise OR.
    #[inline]
    pub fn or(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().or_into(other.clone(), ctx)
    }

    /// Bitwise OR, consuming both arguments.
    #[inline]
    pub fn or_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        let all_ones = Self::all_ones_mask(self.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a | b, self.width()),
            // x | 0 → x
            (None, Some(0)) => self,
            (Some(0), None) => other,
            // x | all_ones → all_ones
            (_, Some(v)) if v == all_ones => Self::ones(self.width()),
            (Some(v), _) if v == all_ones => Self::ones(self.width()),
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Or,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Bitwise XOR.
    #[inline]
    pub fn xor(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().xor_into(other.clone(), ctx)
    }

    /// Bitwise XOR, consuming both arguments.
    #[inline]
    pub fn xor_into(self, other: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(a ^ b, self.width()),
            // x ^ 0 → x
            (None, Some(0)) => self,
            (Some(0), None) => other,
            _ => {
                let width = self.width();
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Xor,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Bitwise NOT.
    #[inline]
    pub fn not(&self, ctx: &SymContext) -> Self {
        self.clone().not_into(ctx)
    }

    /// Bitwise NOT, consuming the argument.
    #[inline]
    pub fn not_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(!v, self.width()),
            None => {
                // not(not(x)) → x
                if let RustBV::Expression {
                    op: BVOp::Not,
                    operands,
                    ..
                } = &self
                {
                    return operands[0].clone();
                }
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Not,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Byte-reverse a bitvector (endianness swap).
    #[inline]
    pub fn reverse(&self, ctx: &SymContext) -> Self {
        self.clone().reverse_into(ctx)
    }

    /// Byte-reverse a bitvector, consuming the argument.
    #[inline]
    pub fn reverse_into(self, _ctx: &SymContext) -> Self {
        let w = self.width();
        debug_assert!(w.is_multiple_of(8), "reverse requires byte-aligned width");
        if w <= 8 {
            return self; // Single byte, no-op
        }
        match self.as_u128() {
            Some(v) => {
                let num_bytes = (w / 8) as usize;
                let mut result: u128 = 0;
                for i in 0..num_bytes {
                    let byte = (v >> (i * 8)) & 0xff;
                    result |= byte << ((num_bytes - 1 - i) * 8);
                }
                Self::concrete(result, w)
            }
            None => {
                // reverse(reverse(x)) → x
                if let RustBV::Expression {
                    op: BVOp::Reverse,
                    operands,
                    ..
                } = &self
                {
                    return operands[0].clone();
                }
                record_bvop_reverse();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: w,
                    op: BVOp::Reverse,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    // =========================================================================
    // Shift Operations
    // =========================================================================

    /// Logical shift left.
    #[inline]
    pub fn shl(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().shl_into(amount.clone(), ctx)
    }

    /// Logical shift left, consuming both arguments.
    #[inline]
    pub fn shl_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        let width = self.width();
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(width);
                Self::concrete(v.wrapping_shl(amt), width)
            }
            // x << 0 → x
            (None, Some(0)) => self,
            // 0 << x → 0
            (Some(0), None) => Self::zero(width),
            // sym << c for 0 < c < w → Concat(Extract(w-c-1, 0, sym), 0^c).
            // Avoids Z3 bit-blasting an O(N^2) mux tree for the symbolic shift.
            (None, Some(c)) if c < width as u128 => {
                let c = c as u32;
                let lo = self.extract_into(width - c - 1, 0, _ctx);
                lo.concat_into(Self::zero(c), _ctx)
            }
            // sym << c with c >= w → 0
            (None, Some(_)) => Self::zero(width),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width,
                op: BVOp::Shl,
                operands: Arc::<[RustBV]>::from([self, amount]),
            },
        }
    }

    /// Logical shift right.
    #[inline]
    pub fn lshr(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().lshr_into(amount.clone(), ctx)
    }

    /// Logical shift right, consuming both arguments.
    #[inline]
    pub fn lshr_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        let width = self.width();
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(width);
                Self::concrete(v.wrapping_shr(amt), width)
            }
            // x >> 0 → x
            (None, Some(0)) => self,
            // 0 >> x → 0
            (Some(0), None) => Self::zero(width),
            // sym >> c for 0 < c < w → Concat(0^c, Extract(w-1, c, sym)).
            (None, Some(c)) if c < width as u128 => {
                let c = c as u32;
                let hi = self.extract_into(width - 1, c, _ctx);
                Self::zero(c).concat_into(hi, _ctx)
            }
            // sym >> c with c >= w → 0
            (None, Some(_)) => Self::zero(width),
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width,
                op: BVOp::Lshr,
                operands: Arc::<[RustBV]>::from([self, amount]),
            },
        }
    }

    /// Arithmetic shift right (sign-extending).
    #[inline]
    pub fn ashr(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().ashr_into(amount.clone(), ctx)
    }

    /// Arithmetic shift right, consuming both arguments.
    #[inline]
    pub fn ashr_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        let width = self.width();
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let amt = (a as u32).min(width);
                let signed = sign_extend(v, width);
                Self::concrete((signed >> amt) as u128, width)
            }
            // x >>> 0 → x
            (None, Some(0)) => self,
            // sym >>> c for 0 < c < w → SignExt(Extract(w-1, c, sym), w).
            (None, Some(c)) if c < width as u128 => {
                let c = c as u32;
                let hi = self.extract_into(width - 1, c, _ctx);
                hi.sign_extend_into(width, _ctx)
            }
            // sym >>> c with c >= w → SignExt(MSB, w) (saturates to all sign bits).
            (None, Some(_)) => {
                let msb = self.extract_into(width - 1, width - 1, _ctx);
                msb.sign_extend_into(width, _ctx)
            }
            _ => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width,
                op: BVOp::Ashr,
                operands: Arc::<[RustBV]>::from([self, amount]),
            },
        }
    }

    /// Rotate left.
    #[inline]
    pub fn rotl(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().rotl_into(amount.clone(), ctx)
    }

    /// Rotate left, consuming both arguments.
    #[inline]
    pub fn rotl_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v << amt) | (v >> (w - amt));
                Self::concrete(rotated, w)
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::RotL,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
        }
    }

    /// Rotate right.
    #[inline]
    pub fn rotr(&self, amount: &Self, ctx: &SymContext) -> Self {
        self.clone().rotr_into(amount.clone(), ctx)
    }

    /// Rotate right, consuming both arguments.
    #[inline]
    pub fn rotr_into(self, amount: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), amount.width());
        match (self.as_u128(), amount.as_u128()) {
            (Some(v), Some(a)) => {
                let w = self.width();
                let amt = (a as u32) % w;
                let rotated = (v >> amt) | (v << (w - amt));
                Self::concrete(rotated, w)
            }
            _ => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::RotR,
                    operands: Arc::<[RustBV]>::from([self, amount]),
                }
            }
        }
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit result).
    #[inline]
    pub fn eq(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().eq_into(other.clone(), ctx)
    }

    /// Equality comparison, consuming both arguments.
    #[inline]
    pub fn eq_into(self, other: Self, ctx: &SymContext) -> Self {
        // Width mismatch guard — return concrete 0 instead of panicking
        if self.width() != other.width() {
            return Self::concrete(0, 1);
        }
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a == b { 1 } else { 0 }, 1),
            _ => {
                // angr-g7nq pattern (b): `Eq(ZeroExt(k, x), BVV(c, W))`.
                // ZeroExt is zero in the top k bits, so:
                //   * if `c >> (W-k) != 0` the equation is unsatisfiable → false
                //   * if `c >> (W-k) == 0` rewrite to `Eq(x, BVV(c, W-k))` so a
                //     smaller Z3 AST is built (and downstream rules — e.g.
                //     `eq_into`'s concrete branch — can fold further).
                // Handles either operand order; signed ext is intentionally not
                // touched (the sign-bit handling is more delicate, see
                // `try_zext_const_cmp_fold`).
                if let Some(folded) = try_zext_const_cmp_fold(&self, &other, ZExtCmp::Eq, ctx) {
                    return folded;
                }
                if let Some(folded) = try_zext_const_cmp_fold(&other, &self, ZExtCmp::Eq, ctx) {
                    return folded;
                }
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: 1,
                    op: BVOp::Eq,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    /// Inequality comparison (returns 1-bit result).
    #[inline]
    pub fn ne(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().ne_into(other.clone(), ctx)
    }

    /// Inequality comparison, consuming both arguments.
    #[inline]
    pub fn ne_into(self, other: Self, ctx: &SymContext) -> Self {
        debug_assert_eq!(self.width(), other.width());
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => Self::concrete(if a != b { 1 } else { 0 }, 1),
            _ => {
                // angr-g7nq pattern (b), see `eq_into`. For Ne the trivial-decide
                // direction inverts (high-bit-nonzero const → always not-equal).
                if let Some(folded) = try_zext_const_cmp_fold(&self, &other, ZExtCmp::Ne, ctx) {
                    return folded;
                }
                if let Some(folded) = try_zext_const_cmp_fold(&other, &self, ZExtCmp::Ne, ctx) {
                    return folded;
                }
                let (lhs, rhs) = self.canonicalize_commutative(other);
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: 1,
                    op: BVOp::Ne,
                    operands: Arc::<[RustBV]>::from([lhs, rhs]),
                }
            }
        }
    }

    // Unsigned ordering comparisons. `Ult` direction matters for the
    // `try_zext_const_cmp_fold` shortcut: high-bit-nonzero const → different
    // trivial answers for `Ult(zext, const)` vs `Ult(const, zext)`. `Ugt(a,b)
    // == Ult(b,a)`, so the forward zext arg is the swapped variant. Symmetric
    // story for Ule/Uge.
    define_unsigned_cmp_pair!(
        ult, ult_into, BVOp::Ult, <,
        "Unsigned less-than comparison.",
        "Unsigned less-than, consuming both arguments.",
        ZExtCmp::Ult, ZExtCmp::UltSwapped
    );
    define_unsigned_cmp_pair!(
        ule, ule_into, BVOp::Ule, <=,
        "Unsigned less-than-or-equal comparison.",
        "Unsigned less-than-or-equal, consuming both arguments.",
        ZExtCmp::Ule, ZExtCmp::UleSwapped
    );
    define_unsigned_cmp_pair!(
        ugt, ugt_into, BVOp::Ugt, >,
        "Unsigned greater-than comparison.",
        "Unsigned greater-than, consuming both arguments.",
        ZExtCmp::UltSwapped, ZExtCmp::Ult
    );
    define_unsigned_cmp_pair!(
        uge, uge_into, BVOp::Uge, >=,
        "Unsigned greater-than-or-equal comparison.",
        "Unsigned greater-than-or-equal, consuming both arguments.",
        ZExtCmp::UleSwapped, ZExtCmp::Ule
    );

    // Signed ordering comparisons. Sign-extend both operands to i128 before
    // applying the operator. No zext-const-cmp-fold (unsigned-only shortcut).
    define_signed_cmp_pair!(
        slt, slt_into, BVOp::Slt, <,
        "Signed less-than comparison.",
        "Signed less-than, consuming both arguments."
    );
    define_signed_cmp_pair!(
        sle, sle_into, BVOp::Sle, <=,
        "Signed less-than-or-equal comparison.",
        "Signed less-than-or-equal, consuming both arguments."
    );
    define_signed_cmp_pair!(
        sgt, sgt_into, BVOp::Sgt, >,
        "Signed greater-than comparison.",
        "Signed greater-than, consuming both arguments."
    );
    define_signed_cmp_pair!(
        sge, sge_into, BVOp::Sge, >=,
        "Signed greater-than-or-equal comparison.",
        "Signed greater-than-or-equal, consuming both arguments."
    );

    // =========================================================================
    // Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    #[inline]
    pub fn zero_extend(&self, to_width: u32, ctx: &SymContext) -> Self {
        self.clone().zero_extend_into(to_width, ctx)
    }

    /// Zero-extend to a wider width, consuming the argument.
    #[inline]
    pub fn zero_extend_into(self, to_width: u32, _ctx: &SymContext) -> Self {
        if to_width <= self.width() {
            // No extension needed (or truncation — just return self)
            return self;
        }
        let extend_bits = to_width - self.width();
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                op: BVOp::ZeroExt(extend_bits),
                operands: Arc::<[RustBV]>::from([self]),
            },
        }
    }

    /// Sign-extend to a wider width.
    #[inline]
    pub fn sign_extend(&self, to_width: u32, ctx: &SymContext) -> Self {
        self.clone().sign_extend_into(to_width, ctx)
    }

    /// Sign-extend to a wider width, consuming the argument.
    #[inline]
    pub fn sign_extend_into(self, to_width: u32, _ctx: &SymContext) -> Self {
        debug_assert!(to_width >= self.width());
        // No extension needed
        if to_width == self.width() {
            return self;
        }
        let extend_bits = to_width - self.width();
        match self.as_u128() {
            Some(v) => {
                let extended = sign_extend_to(v, self.width(), to_width);
                Self::concrete(extended, to_width)
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: to_width,
                op: BVOp::SignExt(extend_bits),
                operands: Arc::<[RustBV]>::from([self]),
            },
        }
    }

    /// Truncate to a narrower width.
    #[inline]
    pub fn truncate(&self, to_width: u32, ctx: &SymContext) -> Self {
        self.clone().truncate_into(to_width, ctx)
    }

    /// Truncate to a narrower width, consuming the argument.
    #[inline]
    pub fn truncate_into(self, to_width: u32, _ctx: &SymContext) -> Self {
        debug_assert!(to_width <= self.width());
        match self.as_u128() {
            Some(v) => Self::concrete(v, to_width),
            None => {
                record_bvop_extract();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: to_width,
                    op: BVOp::Extract(to_width - 1, 0),
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Extract bits [high:low] (inclusive).
    #[inline]
    pub fn extract(&self, high: u32, low: u32, ctx: &SymContext) -> Self {
        self.clone().extract_into(high, low, ctx)
    }

    /// Extract bits [high:low] (inclusive), consuming the argument.
    #[inline]
    pub fn extract_into(self, high: u32, low: u32, _ctx: &SymContext) -> Self {
        debug_assert!(high >= low);
        debug_assert!(high < self.width());
        let result_width = high - low + 1;

        // Fast path for concrete values
        if let Some(v) = self.as_u128() {
            let extracted = (v >> low) & ((1u128 << result_width) - 1);
            return Self::concrete(extracted, result_width);
        }

        // Identity extraction: Extract(width-1, 0, x) → x
        if high == self.width() - 1 && low == 0 {
            return self;
        }

        // Canonicalization rules for Expression nodes — borrow self to inspect,
        // then either delegate (returning early) or fall through to construct.
        if let RustBV::Expression { op, operands, .. } = &self {
            match op {
                // Rule 1: Extract(Extract(x)) → fused single Extract
                // Extract(h2, l2, Extract(h1, l1, x)) → Extract(l1+h2, l1+l2, x)
                BVOp::Extract(_, inner_low) => {
                    return operands[0].extract(inner_low + high, inner_low + low, _ctx);
                }

                // Rule 2: Extract(Concat(a, b)) → distribute to relevant part(s)
                BVOp::Concat if operands.len() == 2 => {
                    let b_width = operands[1].width();
                    if high < b_width {
                        // Entirely within the low part (b)
                        return operands[1].extract(high, low, _ctx);
                    } else if low >= b_width {
                        // Entirely within the high part (a)
                        return operands[0].extract(high - b_width, low - b_width, _ctx);
                    }
                    // Crosses boundary — extract from each part and concat
                    let lo_part = operands[1].extract(b_width - 1, low, _ctx);
                    let hi_part = operands[0].extract(high - b_width, 0, _ctx);
                    return hi_part.concat_into(lo_part, _ctx);
                }

                // Rule 3: Extract(Reverse(x)) with byte-aligned bounds.
                //
                // The byte at position k of Reverse(x) (with B = x.width/8 bytes)
                // is x's byte at position B-1-k. Extracting bytes [l/8..h/8] from
                // Reverse(x) is therefore the byte sequence of x at indices
                // [B-1-h/8 .. B-1-l/8], delivered in reversed order.
                //
                // → Single-byte (h == l + 7): plain Extract(w-1-low, w-1-high, x).
                // → Multi-byte:                Reverse(Extract(w-1-low, w-1-high, x)).
                //
                // The previous form dropped the byte-shuffle for the multi-byte
                // case, which is silently wrong on a Z3 round-trip.
                BVOp::Reverse
                    if operands[0].width() % 8 == 0 && high % 8 == 7 && low.is_multiple_of(8) =>
                {
                    let w = operands[0].width();
                    let inner = operands[0].extract(w - 1 - low, w - 1 - high, _ctx);
                    if high - low + 1 == 8 {
                        return inner;
                    }
                    return inner.reverse(_ctx);
                }

                // Rule 4: Extract(ZeroExt(x)) — if entirely within original width,
                // extract from x directly; if entirely in extended bits, result is 0
                BVOp::ZeroExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return operands[0].extract(high, low, _ctx);
                    } else if low >= inner_width {
                        return Self::zero(result_width);
                    }
                }

                // Rule 5: Extract(SignExt(x)) — if entirely within original width,
                // extract from x directly
                BVOp::SignExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return operands[0].extract(high, low, _ctx);
                    }
                }

                _ => {}
            }
        }

        record_bvop_extract();
        RustBV::Expression {
            id: Self::EXPRESSION_ID,
            width: result_width,
            op: BVOp::Extract(high, low),
            operands: Arc::<[RustBV]>::from([self]),
        }
    }

    /// Concatenate two bitvectors (self becomes high bits).
    #[inline]
    pub fn concat(&self, other: &Self, ctx: &SymContext) -> Self {
        self.clone().concat_into(other.clone(), ctx)
    }

    /// Concatenate two bitvectors, consuming both arguments.
    #[inline]
    pub fn concat_into(self, other: Self, _ctx: &SymContext) -> Self {
        let result_width = self.width() + other.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(hi), Some(lo)) => {
                let combined = (hi << other.width()) | lo;
                Self::concrete(combined, result_width)
            }
            _ => {
                record_bvop_concat();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: result_width,
                    op: BVOp::Concat,
                    operands: Arc::<[RustBV]>::from([self, other]),
                }
            }
        }
    }

    /// Extract bits without requiring a SymContext (same logic, ctx unused).
    #[inline]
    pub fn extract_no_ctx(&self, high: u32, low: u32) -> Self {
        debug_assert!(high >= low);
        debug_assert!(high < self.width());
        let result_width = high - low + 1;

        if let Some(v) = self.as_u128() {
            let extracted = (v >> low) & ((1u128 << result_width) - 1);
            return Self::concrete(extracted, result_width);
        }

        if high == self.width() - 1 && low == 0 {
            return self.clone();
        }

        record_bvop_extract();
        RustBV::Expression {
            id: Self::EXPRESSION_ID,
            width: result_width,
            op: BVOp::Extract(high, low),
            operands: Arc::<[RustBV]>::from([self.clone()]),
        }
    }

    /// Build a balanced Concat tree from `parts`, ordered HIGH bits first
    /// and LOW bits last (i.e. `parts[0]` becomes the high bits of the
    /// result, `parts[parts.len()-1]` becomes the low bits).
    ///
    /// A linear left-fold (`acc = concat(acc, next)`) produces a skewed
    /// AST of depth N-1; pre-Z3 analysis passes walking this DAG do extra
    /// work. This helper recursively splits the slice in half, producing
    /// a tree of depth `ceil(log2(N))`. Z3's flat=true rewriter still
    /// flattens at solve time, but starting from a balanced shape exposes
    /// structural sharing to max-bv-sharing (which matches by AST node
    /// identity) and saves rewrite cycles.
    ///
    /// Panics if `parts` is empty.
    #[inline]
    pub fn concat_balanced(parts: &[RustBV], ctx: &SymContext) -> Self {
        assert!(
            !parts.is_empty(),
            "concat_balanced requires at least one element"
        );
        if parts.len() == 1 {
            return parts[0].clone();
        }
        let mid = parts.len() / 2;
        let high = Self::concat_balanced(&parts[..mid], ctx);
        let low = Self::concat_balanced(&parts[mid..], ctx);
        high.concat_into(low, ctx)
    }

    /// Concatenate without requiring a SymContext (same logic, ctx unused).
    #[inline]
    pub fn concat_no_ctx(&self, other: &Self) -> Self {
        let result_width = self.width() + other.width();
        match (self.as_u128(), other.as_u128()) {
            (Some(hi), Some(lo)) => {
                let combined = (hi << other.width()) | lo;
                Self::concrete(combined, result_width)
            }
            _ => {
                record_bvop_concat();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width: result_width,
                    op: BVOp::Concat,
                    operands: Arc::<[RustBV]>::from([self.clone(), other.clone()]),
                }
            }
        }
    }

    // =========================================================================
    // Utility Operations
    // =========================================================================

    /// If-then-else: returns `then_val` if `self` is non-zero, else `else_val`.
    #[inline]
    pub fn ite(&self, then_val: &Self, else_val: &Self, ctx: &SymContext) -> Self {
        self.clone()
            .ite_into(then_val.clone(), else_val.clone(), ctx)
    }

    /// If-then-else, consuming all three arguments.
    #[inline]
    pub fn ite_into(self, then_val: Self, else_val: Self, _ctx: &SymContext) -> Self {
        debug_assert_eq!(then_val.width(), else_val.width());
        match self.as_u128() {
            Some(v) => {
                if v != 0 {
                    then_val
                } else {
                    else_val
                }
            }
            None => RustBV::Expression {
                id: Self::EXPRESSION_ID,
                width: then_val.width(),
                op: BVOp::Ite,
                operands: Arc::<[RustBV]>::from([self, then_val, else_val]),
            },
        }
    }

    /// Count leading zeros.
    #[inline]
    pub fn clz(&self, ctx: &SymContext) -> Self {
        self.clone().clz_into(ctx)
    }

    /// Count leading zeros, consuming the argument.
    #[inline]
    pub fn clz_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => {
                let leading = if v == 0 {
                    self.width()
                } else {
                    self.width() - (128 - v.leading_zeros())
                };
                Self::concrete(leading as u128, self.width())
            }
            None => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Clz,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Count trailing zeros.
    #[inline]
    pub fn ctz(&self, ctx: &SymContext) -> Self {
        self.clone().ctz_into(ctx)
    }

    /// Count trailing zeros, consuming the argument.
    #[inline]
    pub fn ctz_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => {
                let trailing = if v == 0 {
                    self.width()
                } else {
                    v.trailing_zeros().min(self.width())
                };
                Self::concrete(trailing as u128, self.width())
            }
            None => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Ctz,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    /// Population count (number of set bits).
    #[inline]
    pub fn popcount(&self, ctx: &SymContext) -> Self {
        self.clone().popcount_into(ctx)
    }

    /// Population count, consuming the argument.
    #[inline]
    pub fn popcount_into(self, _ctx: &SymContext) -> Self {
        match self.as_u128() {
            Some(v) => Self::concrete(v.count_ones() as u128, self.width()),
            None => {
                let width = self.width();
                RustBV::Expression {
                    id: Self::EXPRESSION_ID,
                    width,
                    op: BVOp::Popcount,
                    operands: Arc::<[RustBV]>::from([self]),
                }
            }
        }
    }

    // =========================================================================
    // Z3 Integration (when feature is enabled)
    // =========================================================================

    /// Convert this RustBV to a Z3 AST.
    ///
    /// For Expression nodes, the Z3 AST is computed lazily by recursively
    /// building from the operation tree. This avoids creating Z3 ASTs for
    /// intermediate results that never become constraints (matching claripy's
    /// lazy evaluation approach).
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_ast(&self) -> z3::ast::BV {
        super::stats::record_z3_ast_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_ast_cached(&mut cache)
    }

    /// Build Z3 AST with caching to avoid exponential blowup on DAG expressions.
    ///
    /// Expression trees built from symbolic memory stores (ITE chains) often share
    /// sub-expressions via Arc. Without caching, to_z3_ast() traverses the DAG as
    /// a tree, rebuilding shared subtrees exponentially. This version caches by
    /// Arc pointer identity, ensuring each unique sub-expression is built once.
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_ast_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        // For Expression nodes, use pointer identity as cache key
        // (Arc-shared sub-expressions will have the same pointer)
        let cache_key = self as *const RustBV as usize;
        if let Some(cached) = cache.get(&cache_key) {
            super::stats::record_z3_ast_cache_hit();
            return cached.clone();
        }
        super::stats::record_z3_ast_cache_miss();
        let result = match self {
            RustBV::Concrete { value, width } => {
                if *width <= 64 {
                    z3::ast::BV::from_u64(*value as u64, *width)
                } else {
                    let lo = z3::ast::BV::from_u64(*value as u64, 64);
                    let hi = z3::ast::BV::from_u64((*value >> 64) as u64, *width - 64);
                    hi.concat(&lo)
                }
            }
            RustBV::Symbolic { ast, .. } => ast.clone(),
            RustBV::Constrained { value, width, .. } => {
                if *width <= 64 {
                    z3::ast::BV::from_u64(*value as u64, *width)
                } else {
                    let lo = z3::ast::BV::from_u64(*value as u64, 64);
                    let hi = z3::ast::BV::from_u64((*value >> 64) as u64, *width - 64);
                    hi.concat(&lo)
                }
            }
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => Self::build_z3_ast_cached(op, operands, *width, cache),
        };
        cache.insert(cache_key, result.clone());
        result
    }

    /// Convert a 1-bit RustBV to a native Z3 Bool, avoiding ITE wrapping.
    ///
    /// For comparison ops (Eq, Ne, Ult, etc.), produces the native Z3 Bool
    /// directly instead of going through ITE(cmp, BV(1,1), BV(0,1)) then
    /// `._eq(BV(1,1))`. Saves 3 Z3 AST nodes per comparison constraint.
    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_bool(&self) -> z3::ast::Bool {
        super::stats::record_z3_ast_build();
        let mut cache = std::collections::HashMap::new();
        self.to_z3_bool_cached(&mut cache)
    }

    #[cfg(feature = "vex-engine-z3")]
    pub fn to_z3_bool_cached(
        &self,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::Bool {
        match self {
            RustBV::Concrete { value, .. } => z3::ast::Bool::from_bool(*value != 0),
            RustBV::Constrained { value, .. } => z3::ast::Bool::from_bool(*value != 0),
            RustBV::Expression { op, operands, .. } => {
                match op {
                    BVOp::Eq => operands[0]
                        .to_z3_ast_cached(cache)
                        .eq(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ne => operands[0]
                        .to_z3_ast_cached(cache)
                        .eq(operands[1].to_z3_ast_cached(cache))
                        .not(),
                    BVOp::Ult => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvult(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ule => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvule(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Ugt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvugt(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Uge => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvuge(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Slt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvslt(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sle => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsle(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sgt => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsgt(operands[1].to_z3_ast_cached(cache)),
                    BVOp::Sge => operands[0]
                        .to_z3_ast_cached(cache)
                        .bvsge(operands[1].to_z3_ast_cached(cache)),
                    // Not() on a 1-bit comparison: negate the inner bool
                    BVOp::Not if operands.len() == 1 => operands[0].to_z3_bool_cached(cache).not(),
                    // Fallback: convert BV to Bool via _eq(1)
                    _ => {
                        let ast = self.to_z3_ast_cached(cache);
                        let one = z3::ast::BV::from_u64(1, 1);
                        ast.eq(&one)
                    }
                }
            }
            // Symbolic BV — use _eq(1)
            _ => {
                let ast = self.to_z3_ast_cached(cache);
                let one = z3::ast::BV::from_u64(1, 1);
                ast.eq(&one)
            }
        }
    }

    /// Emit a Z3 AST for `Extract(high, low, inner)` while re-applying the
    /// canonicalization rules from `extract_into` at Z3-emission time.
    ///
    /// `extract_into` only fires at construction time. Extract nodes built via
    /// `truncate_into` or `extract_no_ctx` bypass those rules, and so do Extract
    /// nodes whose inner shape was rewritten *after* the Extract was created.
    /// This walks the inner operand and distributes the Extract through
    /// Concat/Reverse/ZeroExt/SignExt/Extract patterns before handing anything
    /// to Z3, which avoids emitting intermediate Z3 ASTs that Z3's bv_rewriter
    /// would have to simplify (and, in the Reverse case, often can't — Z3 has
    /// no native Reverse, so the Concat-of-Extracts encoding survives).
    #[cfg(feature = "vex-engine-z3")]
    fn emit_extract_z3_cached(
        inner: &RustBV,
        high: u32,
        low: u32,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        debug_assert!(high >= low);
        debug_assert!(high < inner.width());
        let result_width = high - low + 1;

        // Identity: Extract(width-1, 0, x) → x
        if high == inner.width() - 1 && low == 0 {
            return inner.to_z3_ast_cached(cache);
        }

        // Concrete fast path — fold the extraction at the Rust level so Z3
        // never sees Extract over a literal.
        if let Some(v) = inner.as_u128() {
            let extracted = (v >> low) & ((1u128 << result_width) - 1);
            if result_width <= 64 {
                return z3::ast::BV::from_u64(extracted as u64, result_width);
            }
            let lo = z3::ast::BV::from_u64(extracted as u64, 64);
            let hi = z3::ast::BV::from_u64((extracted >> 64) as u64, result_width - 64);
            return hi.concat(&lo);
        }

        if let RustBV::Expression { op, operands, .. } = inner {
            match op {
                // Rule 1: Extract(h2, l2, Extract(h1, l1, x)) → Extract(l1+h2, l1+l2, x)
                BVOp::Extract(_, inner_low) => {
                    return Self::emit_extract_z3_cached(
                        &operands[0],
                        inner_low + high,
                        inner_low + low,
                        cache,
                    );
                }

                // Rule 2: Extract(Concat(a, b)) → distribute to the relevant part(s)
                BVOp::Concat if operands.len() == 2 => {
                    let b_width = operands[1].width();
                    if high < b_width {
                        return Self::emit_extract_z3_cached(&operands[1], high, low, cache);
                    } else if low >= b_width {
                        return Self::emit_extract_z3_cached(
                            &operands[0],
                            high - b_width,
                            low - b_width,
                            cache,
                        );
                    }
                    // Crosses boundary — extract from each part and concat.
                    let lo_part =
                        Self::emit_extract_z3_cached(&operands[1], b_width - 1, low, cache);
                    let hi_part =
                        Self::emit_extract_z3_cached(&operands[0], high - b_width, 0, cache);
                    return hi_part.concat(&lo_part);
                }

                // Rule 3: Extract(Reverse(x)) with byte-aligned bounds.
                // Single-byte: Reverse is a no-op, just extract the flipped byte from x.
                // Multi-byte: extract the matching byte range from x, then byte-reverse
                // (emitted as the canonical Concat-of-Extracts Z3 shape, matching
                // build_z3_ast_cached's BVOp::Reverse arm).
                BVOp::Reverse
                    if operands[0].width() % 8 == 0 && high % 8 == 7 && low.is_multiple_of(8) =>
                {
                    let w = operands[0].width();
                    let inner_ast = Self::emit_extract_z3_cached(
                        &operands[0],
                        w - 1 - low,
                        w - 1 - high,
                        cache,
                    );
                    if high - low + 1 == 8 {
                        return inner_ast;
                    }
                    let inner_w = high - low + 1;
                    let byte_count = inner_w / 8;
                    let parts: Vec<z3::ast::BV> = (0..byte_count)
                        .map(|i| inner_ast.extract(i * 8 + 7, i * 8))
                        .collect();
                    let mut result = parts[0].clone();
                    for part in &parts[1..] {
                        result = result.concat(part);
                    }
                    return result;
                }

                // Rule 4: Extract(ZeroExt(x)) — collapse to original or zero
                BVOp::ZeroExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return Self::emit_extract_z3_cached(&operands[0], high, low, cache);
                    } else if low >= inner_width {
                        return z3::ast::BV::from_u64(0, result_width);
                    }
                    // Straddles the extension boundary — fall through to the
                    // generic path. Don't try to split here: the existing
                    // build_z3_ast_cached emits ZeroExt as a concat of zero
                    // bits, so Z3's bv_rewriter already collapses
                    // Extract(zero_ext) cleanly.
                }

                // Rule 5: Extract(SignExt(x)) — collapse if entirely within original width
                BVOp::SignExt(_) => {
                    let inner_width = operands[0].width();
                    if high < inner_width {
                        return Self::emit_extract_z3_cached(&operands[0], high, low, cache);
                    }
                    // Otherwise fall through — the SignExt encoding handles
                    // sign-bit propagation; bv_rewriter folds the Extract.
                }

                _ => {}
            }
        }

        // Default: build the inner Z3 AST and apply Extract.
        inner.to_z3_ast_cached(cache).extract(high, low)
    }

    /// Build Z3 AST with caching to avoid exponential blowup on DAG expressions.
    /// See `to_z3_ast_cached()` for rationale.
    #[cfg(feature = "vex-engine-z3")]
    fn build_z3_ast_cached(
        op: &BVOp,
        operands: &[RustBV],
        _width: u32,
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        match op {
            // Arithmetic
            BVOp::Add => operands[0]
                .to_z3_ast_cached(cache)
                .bvadd(operands[1].to_z3_ast_cached(cache)),
            BVOp::Sub => operands[0]
                .to_z3_ast_cached(cache)
                .bvsub(operands[1].to_z3_ast_cached(cache)),
            BVOp::Mul => operands[0]
                .to_z3_ast_cached(cache)
                .bvmul(operands[1].to_z3_ast_cached(cache)),
            BVOp::UDiv => operands[0]
                .to_z3_ast_cached(cache)
                .bvudiv(operands[1].to_z3_ast_cached(cache)),
            BVOp::SDiv => operands[0]
                .to_z3_ast_cached(cache)
                .bvsdiv(operands[1].to_z3_ast_cached(cache)),
            BVOp::URem => operands[0]
                .to_z3_ast_cached(cache)
                .bvurem(operands[1].to_z3_ast_cached(cache)),
            BVOp::SRem => operands[0]
                .to_z3_ast_cached(cache)
                .bvsrem(operands[1].to_z3_ast_cached(cache)),
            BVOp::Neg => operands[0].to_z3_ast_cached(cache).bvneg(),

            // Bitwise
            BVOp::And => operands[0]
                .to_z3_ast_cached(cache)
                .bvand(operands[1].to_z3_ast_cached(cache)),
            BVOp::Or => operands[0]
                .to_z3_ast_cached(cache)
                .bvor(operands[1].to_z3_ast_cached(cache)),
            BVOp::Xor => operands[0]
                .to_z3_ast_cached(cache)
                .bvxor(operands[1].to_z3_ast_cached(cache)),
            BVOp::Not => operands[0].to_z3_ast_cached(cache).bvnot(),

            // Shifts
            BVOp::Shl => operands[0]
                .to_z3_ast_cached(cache)
                .bvshl(operands[1].to_z3_ast_cached(cache)),
            BVOp::Lshr => operands[0]
                .to_z3_ast_cached(cache)
                .bvlshr(operands[1].to_z3_ast_cached(cache)),
            BVOp::Ashr => operands[0]
                .to_z3_ast_cached(cache)
                .bvashr(operands[1].to_z3_ast_cached(cache)),
            BVOp::RotL => operands[0]
                .to_z3_ast_cached(cache)
                .bvrotl(operands[1].to_z3_ast_cached(cache)),
            BVOp::RotR => operands[0]
                .to_z3_ast_cached(cache)
                .bvrotr(operands[1].to_z3_ast_cached(cache)),

            // Comparisons (return 1-bit BV: If(cmp, BV(1,1), BV(0,1)))
            BVOp::Eq => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ne => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(operands[1].to_z3_ast_cached(cache))
                    .not();
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ult => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvult(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ule => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvule(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Ugt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvugt(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Uge => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvuge(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Slt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvslt(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sle => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsle(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sgt => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsgt(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }
            BVOp::Sge => {
                let cmp = operands[0]
                    .to_z3_ast_cached(cache)
                    .bvsge(operands[1].to_z3_ast_cached(cache));
                cmp.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1))
            }

            // Conversions
            BVOp::ZeroExt(bits) => operands[0].to_z3_ast_cached(cache).zero_ext(*bits),
            BVOp::SignExt(bits) => operands[0].to_z3_ast_cached(cache).sign_ext(*bits),
            BVOp::Extract(high, low) => {
                Self::emit_extract_z3_cached(&operands[0], *high, *low, cache)
            }
            BVOp::Concat => {
                fn collect_concat_leaves_cached(
                    bv: &RustBV,
                    leaves: &mut Vec<z3::ast::BV>,
                    cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
                ) {
                    if let RustBV::Expression {
                        op: BVOp::Concat,
                        operands,
                        ..
                    } = bv
                    {
                        collect_concat_leaves_cached(&operands[0], leaves, cache);
                        collect_concat_leaves_cached(&operands[1], leaves, cache);
                    } else {
                        leaves.push(bv.to_z3_ast_cached(cache));
                    }
                }
                let mut leaves = Vec::new();
                collect_concat_leaves_cached(&operands[0], &mut leaves, cache);
                collect_concat_leaves_cached(&operands[1], &mut leaves, cache);
                let mut result = leaves
                    .pop()
                    .expect("Concat operands always produce at least one leaf");
                while let Some(part) = leaves.pop() {
                    result = part.concat(&result);
                }
                result
            }

            // Conditional
            BVOp::Ite => {
                let cond = operands[0]
                    .to_z3_ast_cached(cache)
                    .eq(z3::ast::BV::from_u64(0, operands[0].width()))
                    .not();
                cond.ite(
                    &operands[1].to_z3_ast_cached(cache),
                    &operands[2].to_z3_ast_cached(cache),
                )
            }

            // Byte reverse
            BVOp::Reverse => {
                if let RustBV::Expression {
                    op: BVOp::Concat, ..
                } = &operands[0]
                {
                    fn collect_concat_parts_cached(bv: &RustBV, parts: &mut Vec<RustBV>) {
                        if let RustBV::Expression {
                            op: BVOp::Concat,
                            operands,
                            ..
                        } = bv
                        {
                            collect_concat_parts_cached(&operands[0], parts);
                            collect_concat_parts_cached(&operands[1], parts);
                        } else {
                            parts.push(bv.clone());
                        }
                    }
                    let mut parts = Vec::new();
                    collect_concat_parts_cached(&operands[0], &mut parts);
                    parts.reverse();
                    let reversed_asts: Vec<z3::ast::BV> = parts
                        .iter()
                        .map(|p| {
                            let w = p.width();
                            if w == 8 {
                                p.to_z3_ast_cached(cache)
                            } else {
                                Self::build_z3_ast_cached(
                                    &BVOp::Reverse,
                                    std::slice::from_ref(p),
                                    w,
                                    cache,
                                )
                            }
                        })
                        .collect();
                    let mut result = reversed_asts[0].clone();
                    for ast in &reversed_asts[1..] {
                        result = result.concat(ast);
                    }
                    return result;
                }

                let ast = operands[0].to_z3_ast_cached(cache);
                let w = operands[0].width();
                if w.is_multiple_of(8) && w >= 16 {
                    // Same canonical shape as the non-cached path above.
                    let bytes = w / 8;
                    let parts: Vec<z3::ast::BV> =
                        (0..bytes).map(|i| ast.extract(i * 8 + 7, i * 8)).collect();
                    let mut result = parts[0].clone();
                    for part in &parts[1..] {
                        result = result.concat(part);
                    }
                    result
                } else {
                    ast
                }
            }

            // Bit counting ops
            BVOp::Clz => z3::ast::BV::new_const("clz", _width),
            BVOp::Ctz => z3::ast::BV::new_const("ctz", _width),
            BVOp::Popcount => z3::ast::BV::new_const("popcount", _width),

            // Floating-point operations via Z3 FP theory.
            BVOp::Float { kind, prec } => {
                Self::build_fp_z3_ast_cached(*kind, *prec, operands, cache)
            }
        }
    }

    /// Build a Z3 AST for a symbolic float operation, returning the result
    /// encoded as an IEEE-754 bit-vector (or 1-bit BV for compares).
    ///
    /// Operands are RustBVs holding IEEE-754 bit patterns; we reinterpret
    /// them as Z3 Float values (`Z3_mk_fpa_to_fp_bv`), apply the FP op,
    /// and convert results back to IEEE bits via `to_ieee_bv`.
    ///
    /// Each intermediate Z3 ast is wrapped via `Ast::wrap` so its refcount
    /// is properly tracked — passing raw `Z3_ast` pointers to multiple FFI
    /// calls is unsafe because Z3's ref-counted contexts may GC the
    /// intermediate ASTs between calls.
    ///
    /// # SAFETY invariants for all `unsafe { z3_sys::… }` calls in this and
    /// the sibling `build_fp_*_cached` helpers
    ///
    /// 1. **Context validity**: `raw_ctx` is `z3::Context::thread_local()
    ///    .get_z3_context()`. The thread-local context lives for the
    ///    duration of the thread, so the handle is valid for the call.
    /// 2. **Pointer validity**: All `Z3_ast` operands are obtained from
    ///    `.get_z3_ast()` on live wrappers (`RustBV`, `Float`, `Bool`,
    ///    `RoundingMode`, `Sort`) bound in the same scope, so Z3 holds at
    ///    least one refcount on each for the duration of the FFI call.
    /// 3. **Null handling**: Every `Z3_mk_fpa_*` is followed by
    ///    `.expect(…)`, converting the only failure mode (NULL) into a
    ///    panic — so any pointer that escapes the `unsafe` block is
    ///    non-null and points at a fresh Z3 AST with one refcount.
    /// 4. **Refcount discipline**: Each fresh raw `Z3_ast` is immediately
    ///    handed to `Float::wrap` / `BV::wrap` / `Bool::wrap`, which takes
    ///    over the refcount Z3 added at construction. The wrapper is then
    ///    held in a local for the rest of its use. The wrap functions are
    ///    `unsafe` only because they require this caller-supplied refcount
    ///    discipline; no other invariant is needed.
    /// 5. **Thread-safety**: All ASTs in this helper live in the
    ///    thread-local context; nothing escapes the calling thread, so
    ///    Z3's per-context single-thread requirement is upheld.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_z3_ast_cached(
        kind: FloatOpKind,
        prec: FloatPrec,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Bool, Float, RoundingMode};
        use z3_sys::{
            Z3_mk_fpa_abs, Z3_mk_fpa_add, Z3_mk_fpa_div, Z3_mk_fpa_eq, Z3_mk_fpa_fma,
            Z3_mk_fpa_is_nan, Z3_mk_fpa_leq, Z3_mk_fpa_lt, Z3_mk_fpa_mul, Z3_mk_fpa_neg,
            Z3_mk_fpa_sqrt, Z3_mk_fpa_sub, Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv,
        };

        // RoundToInt has a non-Float operand (the rm BV) and needs its own path.
        if let FloatOpKind::RoundToInt = kind {
            return Self::build_fp_round_to_int_cached(prec, operands, cache);
        }
        // FP conversions (FtoI, ItoF, FtoF) have non-uniform operand types
        // (BV vs FP, varying widths/sorts). Each routes to a dedicated helper
        // per the split-helper invariant for FloatOpKind metadata operands.
        match kind {
            FloatOpKind::ConvertItoF { src_bits, signed } => {
                return Self::build_fp_i_to_f_cached(prec, src_bits, signed, operands, cache);
            }
            FloatOpKind::ConvertFtoI { dst_bits, signed } => {
                return Self::build_fp_f_to_i_cached(
                    prec, dst_bits, signed, /*rm*/ None, operands, cache,
                );
            }
            FloatOpKind::ConvertFtoIRm { dst_bits, signed } => {
                return Self::build_fp_f_to_i_cached(
                    prec,
                    dst_bits,
                    signed,
                    Some(()),
                    operands,
                    cache,
                );
            }
            FloatOpKind::ConvertFtoF { src_prec } => {
                return Self::build_fp_f_to_f_cached(
                    prec, src_prec, /*has_rm*/ false, operands, cache,
                );
            }
            FloatOpKind::ConvertFtoFRm { src_prec } => {
                return Self::build_fp_f_to_f_cached(
                    prec, src_prec, /*has_rm*/ true, operands, cache,
                );
            }
            FloatOpKind::AddRm
            | FloatOpKind::SubRm
            | FloatOpKind::MulRm
            | FloatOpKind::DivRm
            | FloatOpKind::SqrtRm => {
                return Self::build_fp_arith_rm_cached(kind, prec, operands, cache);
            }
            _ => {}
        }

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert each BV operand to a Z3 Float wrapper. Holding the wrapper
        // (not just the raw Z3_ast) ensures the intermediate AST has its
        // refcount incremented and is not freed before we use it.
        let fp_args: Vec<Float> = operands
            .iter()
            .map(|bv| {
                let bv_ast = bv.to_z3_ast_cached(cache);
                // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
                // `bv_ast` is a live `BV` wrapper; `raw_sort` is a live
                // `Sort` handle. `Z3_mk_fpa_to_fp_bv` returns a fresh AST
                // (or NULL → panic via `.expect`).
                let raw = unsafe {
                    Z3_mk_fpa_to_fp_bv(raw_ctx, bv_ast.get_z3_ast(), raw_sort)
                        .expect("Z3_mk_fpa_to_fp_bv returned NULL")
                };
                // SAFETY: `raw` is a fresh non-null Z3_ast in `z3_ctx`
                // with one refcount held by Z3; `Float::wrap` takes it.
                unsafe { Float::wrap(&z3_ctx, raw) }
            })
            .collect();

        // Round-to-nearest-ties-to-even (IEEE-754 default).
        let rm = RoundingMode::round_nearest_ties_to_even();
        let rm_raw = rm.get_z3_ast();

        // Apply the operation. Wrap the result as Float (or Bool) so its
        // refcount is held until we convert it.
        let raw_a = fp_args[0].get_z3_ast();
        // Helper to safely get raw_b / raw_c only when needed (some ops are unary).
        let raw_b = || fp_args[1].get_z3_ast();
        let raw_c = || fp_args[2].get_z3_ast();

        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `rm_raw` is held by the `rm: RoundingMode` wrapper; `raw_a`,
        // `raw_b()`, `raw_c()` come from the `fp_args` Float wrappers and
        // remain live for the duration of this match. Every `Z3_mk_fpa_*`
        // returns a fresh AST (or NULL → panic via `.expect`).
        let result_raw = unsafe {
            match kind {
                FloatOpKind::Add => Z3_mk_fpa_add(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Sub => Z3_mk_fpa_sub(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Mul => Z3_mk_fpa_mul(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Div => Z3_mk_fpa_div(raw_ctx, rm_raw, raw_a, raw_b()),
                FloatOpKind::Sqrt => Z3_mk_fpa_sqrt(raw_ctx, rm_raw, raw_a),
                FloatOpKind::Neg => Z3_mk_fpa_neg(raw_ctx, raw_a),
                FloatOpKind::Abs => Z3_mk_fpa_abs(raw_ctx, raw_a),
                FloatOpKind::Fma => Z3_mk_fpa_fma(raw_ctx, rm_raw, raw_a, raw_b(), raw_c()),
                FloatOpKind::Fms => {
                    // a*b - c == a*b + (-c). Wrap neg_c in a Float so the
                    // intermediate AST is held while we build the FMA.
                    let neg_c_raw =
                        Z3_mk_fpa_neg(raw_ctx, raw_c()).expect("Z3_mk_fpa_neg returned NULL");
                    let neg_c = Float::wrap(&z3_ctx, neg_c_raw);
                    Z3_mk_fpa_fma(raw_ctx, rm_raw, raw_a, raw_b(), neg_c.get_z3_ast())
                }
                FloatOpKind::CmpEq => Z3_mk_fpa_eq(raw_ctx, raw_a, raw_b()),
                FloatOpKind::CmpLt => Z3_mk_fpa_lt(raw_ctx, raw_a, raw_b()),
                FloatOpKind::CmpLe => Z3_mk_fpa_leq(raw_ctx, raw_a, raw_b()),
                FloatOpKind::IsNaN => Z3_mk_fpa_is_nan(raw_ctx, raw_a),
                FloatOpKind::RoundToInt
                | FloatOpKind::ConvertItoF { .. }
                | FloatOpKind::ConvertFtoI { .. }
                | FloatOpKind::ConvertFtoIRm { .. }
                | FloatOpKind::ConvertFtoF { .. }
                | FloatOpKind::ConvertFtoFRm { .. }
                | FloatOpKind::AddRm
                | FloatOpKind::SubRm
                | FloatOpKind::MulRm
                | FloatOpKind::DivRm
                | FloatOpKind::SqrtRm => unreachable!("handled above"),
            }
            .expect("Z3 FPA op returned NULL")
        };

        if kind.is_compare() {
            // SAFETY: `result_raw` is the fresh Z3 Bool produced by the
            // comparison op above (one refcount held by Z3); `Bool::wrap`
            // takes that refcount. Same context as the wrapper.
            let cmp_bool = unsafe { Bool::wrap(&z3_ctx, result_raw) };
            cmp_bool.ite(&BV::from_u64(1, 1), &BV::from_u64(0, 1))
        } else {
            // SAFETY: `result_raw` is the fresh Z3 Float produced above;
            // `Float::wrap` takes its refcount.
            let result_fp = unsafe { Float::wrap(&z3_ctx, result_raw) };
            // SAFETY: `result_fp` is a live Float in `z3_ctx`;
            // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL →
            // panic via `.expect`).
            let ieee_bv_raw = unsafe {
                Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                    .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
            };
            // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call
            // above; `BV::wrap` takes its refcount.
            unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
        }
    }

    /// Build the Z3 AST for a `FloatOpKind::RoundToInt` operation.
    ///
    /// VEX rounding modes (low 2 bits of operand[0]):
    ///   0 = nearest (ties to even), 1 = -inf, 2 = +inf, 3 = zero (truncate).
    ///
    /// Concrete rm: pick the matching Z3 RoundingMode and call
    /// `Z3_mk_fpa_round_to_integral` once. Symbolic rm: build all four
    /// variants and ITE on the rm low-bits — Z3 simplifies away dead arms
    /// at solve time.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_round_to_int_cached(
        prec: FloatPrec,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_round_to_integral, Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv};

        debug_assert_eq!(operands.len(), 2);
        let rm_bv = &operands[0];
        let value_bv = &operands[1];

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert the value BV to a Z3 Float; keep the wrapper alive.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `value_z3` is a live BV in `z3_ctx`; `raw_sort` is a live Sort
        // handle; `Z3_mk_fpa_to_fp_bv` returns a fresh AST whose refcount
        // `Float::wrap` immediately takes.
        let value_fp = unsafe {
            let raw = Z3_mk_fpa_to_fp_bv(raw_ctx, value_z3.get_z3_ast(), raw_sort)
                .expect("Z3_mk_fpa_to_fp_bv returned NULL");
            Float::wrap(&z3_ctx, raw)
        };
        let value_raw = value_fp.get_z3_ast();

        // Helper: round value with one concrete VEX rounding mode (0..3).
        let round_with = |vex_rm: u8| -> Float {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by the `value_fp` wrapper above.
            // `Z3_mk_fpa_round_to_integral` returns a fresh AST (or NULL →
            // panic via `.expect`).
            let raw = unsafe {
                Z3_mk_fpa_round_to_integral(raw_ctx, rm.get_z3_ast(), value_raw)
                    .expect("Z3_mk_fpa_round_to_integral returned NULL")
            };
            // SAFETY: `raw` is the fresh Float AST from the call above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = if let Some(m) = rm_bv.as_u128() {
            round_with((m & 0x3) as u8)
        } else {
            // Symbolic rm: build all 4 results and ITE on rm[1:0].
            let r0 = round_with(0);
            let r1 = round_with(1);
            let r2 = round_with(2);
            let r3 = round_with(3);
            let rm_z3 = rm_bv.to_z3_ast_cached(cache);
            let rm_low2 = rm_z3.extract(1, 0);
            let zero = BV::from_u64(0, 2);
            let one = BV::from_u64(1, 2);
            let two = BV::from_u64(2, 2);
            // Chain: rm==0 ? r0 : rm==1 ? r1 : rm==2 ? r2 : r3
            let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
            let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
            rm_low2.eq(&zero).ite(&r0, &pick123)
        };

        // SAFETY: `result_fp` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for an FP arithmetic op with explicit rounding mode
    /// (`AddRm`/`SubRm`/`MulRm`/`DivRm`/`SqrtRm`). operand[0] is the rm BV;
    /// remaining operands are the FP operands. Concrete rm picks one Z3
    /// `RoundingMode`; symbolic rm builds all four variants and ITEs on the
    /// rm low-2-bits — Z3 simplifies away dead arms at solve time. Mirrors
    /// the pattern in `build_fp_round_to_int_cached`.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_arith_rm_cached(
        kind: FloatOpKind,
        prec: FloatPrec,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{
            Z3_mk_fpa_add, Z3_mk_fpa_div, Z3_mk_fpa_mul, Z3_mk_fpa_sqrt, Z3_mk_fpa_sub,
            Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_ieee_bv,
        };

        let is_unary = matches!(kind, FloatOpKind::SqrtRm);
        debug_assert_eq!(operands.len(), if is_unary { 2 } else { 3 });
        let rm_bv = &operands[0];

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert FP operand BVs to Z3 Float wrappers; keep them alive across
        // the helper closure since intermediate ASTs may be GC'd otherwise.
        let to_fp =
            |bv: &RustBV, cache: &mut std::collections::HashMap<usize, z3::ast::BV>| -> Float {
                let z3 = bv.to_z3_ast_cached(cache);
                // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
                // `z3` is a live BV in `z3_ctx`; `raw_sort` is a live Sort.
                // `Z3_mk_fpa_to_fp_bv` returns a fresh AST (or NULL →
                // panic via `.expect`).
                let raw = unsafe {
                    Z3_mk_fpa_to_fp_bv(raw_ctx, z3.get_z3_ast(), raw_sort)
                        .expect("Z3_mk_fpa_to_fp_bv returned NULL")
                };
                // SAFETY: `raw` is the fresh Float AST from above;
                // `Float::wrap` takes its refcount.
                unsafe { Float::wrap(&z3_ctx, raw) }
            };
        let a_fp = to_fp(&operands[1], cache);
        let b_fp = if is_unary {
            None
        } else {
            Some(to_fp(&operands[2], cache))
        };

        let apply = |vex_rm: u8| -> Float {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            let rm_raw = rm.get_z3_ast();
            let raw_a = a_fp.get_z3_ast();
            // SAFETY: `rm` (RoundingMode) and `a_fp` / `b_fp` (Float) are
            // live wrappers in `z3_ctx` for the duration of this closure
            // body, so `rm_raw`, `raw_a` and the `get_z3_ast()` calls on
            // `b_fp` all yield valid Z3_ast pointers. Each `Z3_mk_fpa_*`
            // returns a fresh AST (or NULL → panic via `.expect`).
            let raw = unsafe {
                match kind {
                    FloatOpKind::AddRm => {
                        Z3_mk_fpa_add(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::SubRm => {
                        Z3_mk_fpa_sub(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::MulRm => {
                        Z3_mk_fpa_mul(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::DivRm => {
                        Z3_mk_fpa_div(raw_ctx, rm_raw, raw_a, b_fp.as_ref().unwrap().get_z3_ast())
                    }
                    FloatOpKind::SqrtRm => Z3_mk_fpa_sqrt(raw_ctx, rm_raw, raw_a),
                    _ => unreachable!("non-Rm FP arith kind in build_fp_arith_rm_cached"),
                }
                .expect("Z3 FPA arith op returned NULL")
            };
            // SAFETY: `raw` is the fresh Float AST from the call above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = if let Some(m) = rm_bv.as_u128() {
            apply((m & 0x3) as u8)
        } else {
            // Symbolic rm: build all 4 results and ITE on rm[1:0]. Z3 folds
            // dead arms during simplification.
            let r0 = apply(0);
            let r1 = apply(1);
            let r2 = apply(2);
            let r3 = apply(3);
            let rm_z3 = rm_bv.to_z3_ast_cached(cache);
            let rm_low2 = rm_z3.extract(1, 0);
            let zero = BV::from_u64(0, 2);
            let one = BV::from_u64(1, 2);
            let two = BV::from_u64(2, 2);
            let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
            let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
            rm_low2.eq(&zero).ite(&r0, &pick123)
        };

        // SAFETY: `result_fp` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertItoF`. operand[0] is a BV
    /// of width `src_bits` interpreted as signed/unsigned per `signed`.
    /// Result is the IEEE bits of the FP at `prec`. RNE rounding.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_i_to_f_cached(
        prec: FloatPrec,
        src_bits: u8,
        signed: bool,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_signed, Z3_mk_fpa_to_fp_unsigned, Z3_mk_fpa_to_ieee_bv};

        debug_assert_eq!(operands.len(), 1);
        let src_bv = &operands[0];
        debug_assert_eq!(src_bv.width(), src_bits as u32);

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        let src_z3 = src_bv.to_z3_ast_cached(cache);
        let rm = RoundingMode::round_nearest_ties_to_even();
        let rm_raw = rm.get_z3_ast();

        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `rm_raw` is held by `rm: RoundingMode`; `src_z3` is a live BV
        // wrapper; `raw_sort` is a live Sort. The signed/unsigned
        // `Z3_mk_fpa_to_fp_*` calls return a fresh Float AST (or NULL →
        // panic via `.expect`).
        let fp_raw = unsafe {
            if signed {
                Z3_mk_fpa_to_fp_signed(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .expect("Z3_mk_fpa_to_fp_signed returned NULL")
            } else {
                Z3_mk_fpa_to_fp_unsigned(raw_ctx, rm_raw, src_z3.get_z3_ast(), raw_sort)
                    .expect("Z3_mk_fpa_to_fp_unsigned returned NULL")
            }
        };
        // SAFETY: `fp_raw` is the fresh Float AST from the call above;
        // `Float::wrap` takes its refcount.
        let fp_wrap = unsafe { Float::wrap(&z3_ctx, fp_raw) };
        // SAFETY: `fp_wrap` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, fp_wrap.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertFtoI` (no rm operand,
    /// implicit RNE) or `FloatOpKind::ConvertFtoIRm` (operand[0] = rm BV,
    /// operand[1] = FP value). The result is a BV of width `dst_bits`,
    /// signed or unsigned 2's complement per `signed`.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_f_to_i_cached(
        prec: FloatPrec,
        dst_bits: u8,
        signed: bool,
        rm_marker: Option<()>,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_sbv, Z3_mk_fpa_to_ubv};

        let (rm_bv_opt, value_bv) = if rm_marker.is_some() {
            debug_assert_eq!(operands.len(), 2);
            (Some(&operands[0]), &operands[1])
        } else {
            debug_assert_eq!(operands.len(), 1);
            (None, &operands[0])
        };

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let raw_sort = sort.get_z3_sort();

        // Convert the operand BV to a Z3 Float; keep the wrapper alive.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `value_z3` is a live BV in `z3_ctx`; `raw_sort` is a live Sort.
        // `Z3_mk_fpa_to_fp_bv` returns a fresh AST whose refcount
        // `Float::wrap` immediately takes.
        let value_fp = unsafe {
            let raw = Z3_mk_fpa_to_fp_bv(raw_ctx, value_z3.get_z3_ast(), raw_sort)
                .expect("Z3_mk_fpa_to_fp_bv returned NULL");
            Float::wrap(&z3_ctx, raw)
        };
        let value_raw = value_fp.get_z3_ast();

        // Helper: convert the value to BV using one concrete VEX rounding mode (0..3).
        let convert_with = |vex_rm: u8| -> BV {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by `value_fp` above. The signed/unsigned
            // `Z3_mk_fpa_to_*bv` calls return a fresh BV AST (or NULL →
            // panic via `.expect`).
            let raw = unsafe {
                if signed {
                    Z3_mk_fpa_to_sbv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .expect("Z3_mk_fpa_to_sbv returned NULL")
                } else {
                    Z3_mk_fpa_to_ubv(raw_ctx, rm.get_z3_ast(), value_raw, dst_bits as u32)
                        .expect("Z3_mk_fpa_to_ubv returned NULL")
                }
            };
            // SAFETY: `raw` is the fresh BV AST from the call above;
            // `BV::wrap` takes its refcount.
            unsafe { BV::wrap(&z3_ctx, raw) }
        };

        match rm_bv_opt {
            // Implicit RNE.
            None => convert_with(0),
            // Explicit rm; if symbolic, ITE over the four cases.
            Some(rm_bv) => {
                if let Some(m) = rm_bv.as_u128() {
                    convert_with((m & 0x3) as u8)
                } else {
                    let r0 = convert_with(0);
                    let r1 = convert_with(1);
                    let r2 = convert_with(2);
                    let r3 = convert_with(3);
                    let rm_z3 = rm_bv.to_z3_ast_cached(cache);
                    let rm_low2 = rm_z3.extract(1, 0);
                    let zero = BV::from_u64(0, 2);
                    let one = BV::from_u64(1, 2);
                    let two = BV::from_u64(2, 2);
                    let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
                    let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
                    rm_low2.eq(&zero).ite(&r0, &pick123)
                }
            }
        }
    }

    /// Build the Z3 AST for `FloatOpKind::ConvertFtoF` (no rm) or
    /// `FloatOpKind::ConvertFtoFRm` (operand[0] = rm BV, operand[1] = FP).
    /// Source FP is at `src_prec`, destination FP is at `prec`.
    #[cfg(feature = "vex-engine-z3")]
    fn build_fp_f_to_f_cached(
        prec: FloatPrec,
        src_prec: FloatPrec,
        has_rm: bool,
        operands: &[RustBV],
        cache: &mut std::collections::HashMap<usize, z3::ast::BV>,
    ) -> z3::ast::BV {
        use z3::ast::{Ast, BV, Float, RoundingMode};
        use z3_sys::{Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_fp_float, Z3_mk_fpa_to_ieee_bv};

        let (rm_bv_opt, value_bv) = if has_rm {
            debug_assert_eq!(operands.len(), 2);
            (Some(&operands[0]), &operands[1])
        } else {
            debug_assert_eq!(operands.len(), 1);
            (None, &operands[0])
        };

        let z3_ctx = z3::Context::thread_local();
        let raw_ctx = z3_ctx.get_z3_context();
        let src_sort = match src_prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let dst_sort = match prec {
            FloatPrec::F32 => z3::Sort::float32(),
            FloatPrec::F64 => z3::Sort::double(),
        };
        let src_raw_sort = src_sort.get_z3_sort();
        let dst_raw_sort = dst_sort.get_z3_sort();

        // Convert the source operand BV to a Z3 Float at src_prec.
        let value_z3 = value_bv.to_z3_ast_cached(cache);
        // SAFETY: see invariants block on `build_fp_z3_ast_cached`.
        // `value_z3` is a live BV in `z3_ctx`; `src_raw_sort` is a live
        // Sort handle. `Z3_mk_fpa_to_fp_bv` returns a fresh AST whose
        // refcount `Float::wrap` immediately takes.
        let value_fp = unsafe {
            let raw = Z3_mk_fpa_to_fp_bv(raw_ctx, value_z3.get_z3_ast(), src_raw_sort)
                .expect("Z3_mk_fpa_to_fp_bv returned NULL");
            Float::wrap(&z3_ctx, raw)
        };
        let value_raw = value_fp.get_z3_ast();

        let convert_with = |vex_rm: u8| -> Float {
            let rm = match vex_rm & 0x3 {
                0 => RoundingMode::round_nearest_ties_to_even(),
                1 => RoundingMode::round_towards_negative(),
                2 => RoundingMode::round_towards_positive(),
                3 => RoundingMode::round_towards_zero(),
                _ => unreachable!(),
            };
            // SAFETY: `rm` is a live RoundingMode in `z3_ctx`; `value_raw`
            // is held alive by `value_fp` above; `dst_raw_sort` is a live
            // Sort handle. `Z3_mk_fpa_to_fp_float` returns a fresh Float
            // AST (or NULL → panic via `.expect`).
            let raw = unsafe {
                Z3_mk_fpa_to_fp_float(raw_ctx, rm.get_z3_ast(), value_raw, dst_raw_sort)
                    .expect("Z3_mk_fpa_to_fp_float returned NULL")
            };
            // SAFETY: `raw` is the fresh Float AST from above;
            // `Float::wrap` takes its refcount.
            unsafe { Float::wrap(&z3_ctx, raw) }
        };

        let result_fp = match rm_bv_opt {
            None => convert_with(0),
            Some(rm_bv) => {
                if let Some(m) = rm_bv.as_u128() {
                    convert_with((m & 0x3) as u8)
                } else {
                    let r0 = convert_with(0);
                    let r1 = convert_with(1);
                    let r2 = convert_with(2);
                    let r3 = convert_with(3);
                    let rm_z3 = rm_bv.to_z3_ast_cached(cache);
                    let rm_low2 = rm_z3.extract(1, 0);
                    let zero = BV::from_u64(0, 2);
                    let one = BV::from_u64(1, 2);
                    let two = BV::from_u64(2, 2);
                    let pick23 = rm_low2.eq(&two).ite(&r2, &r3);
                    let pick123 = rm_low2.eq(&one).ite(&r1, &pick23);
                    rm_low2.eq(&zero).ite(&r0, &pick123)
                }
            }
        };

        // SAFETY: `result_fp` is a live Float in `z3_ctx`;
        // `Z3_mk_fpa_to_ieee_bv` returns a fresh BV AST (or NULL → panic
        // via `.expect`).
        let ieee_bv_raw = unsafe {
            Z3_mk_fpa_to_ieee_bv(raw_ctx, result_fp.get_z3_ast())
                .expect("Z3_mk_fpa_to_ieee_bv returned NULL")
        };
        // SAFETY: `ieee_bv_raw` is the fresh BV AST from the call above;
        // `BV::wrap` takes its refcount.
        unsafe { BV::wrap(&z3_ctx, ieee_bv_raw) }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// angr-g7nq: comparison-op tag for `try_zext_const_cmp_fold`. The `Swapped`
/// variants describe the case where the ZeroExt operand sits on the right of
/// the original operator (e.g. `Ult(const, ZeroExt(x))` is `UltSwapped` from
/// the helper's perspective, since the helper always takes `(zext_side,
/// const_side)`).
#[derive(Copy, Clone)]
enum ZExtCmp {
    Eq,
    Ne,
    Ult,
    UltSwapped,
    Ule,
    UleSwapped,
}

/// Fold `Cmp(ZeroExt(k, x), BVV(c, W))` (and commuted variants) using the
/// fact that `ZeroExt(k, x)` is always in the range `[0, 2^(W-k))`.
///
/// Two outcomes:
/// * Trivial decide — the comparison is structurally `true` or `false`
///   regardless of `x`. Returns `Self::concrete(0|1, 1)`. Bumps
///   `zext_cmp_trivial_decide_count`.
/// * Collapse — the high `k` bits of `c` are zero (or the comparison is
///   insensitive to them), so the comparison is rewritten on the `W-k`-bit
///   operands by recursing into the same op on the narrowed sides. Bumps
///   `zext_cmp_collapse_count`.
///
/// Returns `None` if `zext_side` is not a `BVOp::ZeroExt(_)` expression, or
/// `const_side` is not concrete, or the widths don't match the expected shape.
fn try_zext_const_cmp_fold(
    zext_side: &RustBV,
    const_side: &RustBV,
    op: ZExtCmp,
    ctx: &SymContext,
) -> Option<RustBV> {
    let (extend_bits, inner) = match zext_side {
        RustBV::Expression {
            op: BVOp::ZeroExt(k),
            operands,
            ..
        } => (*k, &operands[0]),
        _ => return None,
    };
    // ZeroExt(0, x) → x; the wrapping caller hands us a normal width
    // comparison and we shouldn't bother. `extend_bits == 0` is unusual but
    // safe to skip — falling through builds the same Cmp expression as before.
    if extend_bits == 0 {
        return None;
    }
    let c = const_side.as_u128()?;
    let total_width = zext_side.width();
    let inner_width = inner.width();
    debug_assert_eq!(inner_width + extend_bits, total_width);
    // Compute `c_high = c >> inner_width` (the bits that ZeroExt forces to
    // zero) and `c_low` (the low W-k bits) without overflowing the shift on
    // total_width == 128.
    let (c_high, c_low) = if inner_width >= 128 {
        // Defensive; in practice all BVVs we touch fit in <= 128 bits and the
        // ZeroExt source can't be wider than 128 either.
        (0u128, c)
    } else {
        let low_mask = (1u128 << inner_width) - 1;
        (c >> inner_width, c & low_mask)
    };

    match op {
        ZExtCmp::Eq | ZExtCmp::Ne => {
            if c_high != 0 {
                // ZeroExt produces high bits zero; const_side disagrees → eq
                // is unsat, ne is tautological.
                record_zext_cmp_trivial_decide();
                let val = match op {
                    ZExtCmp::Eq => 0,
                    ZExtCmp::Ne => 1,
                    _ => unreachable!(),
                };
                return Some(RustBV::concrete(val, 1));
            }
            // High bits agree (both zero) → narrow.
            record_zext_cmp_collapse();
            let narrowed_const = RustBV::concrete(c_low, inner_width);
            let narrowed_inner = inner.clone();
            let result = match op {
                ZExtCmp::Eq => narrowed_inner.eq_into(narrowed_const, ctx),
                ZExtCmp::Ne => narrowed_inner.ne_into(narrowed_const, ctx),
                _ => unreachable!(),
            };
            Some(result)
        }
        ZExtCmp::Ult => {
            // `ZeroExt(k, x) < c`. ZeroExt ∈ [0, 2^(W-k)).
            if c_high != 0 {
                // c >= 2^(W-k) ⇒ ZeroExt(x) < c is always true.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(1, 1));
            }
            if c_low == 0 {
                // ZeroExt(x) < 0 is unsat.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(0, 1));
            }
            record_zext_cmp_collapse();
            Some(
                inner
                    .clone()
                    .ult_into(RustBV::concrete(c_low, inner_width), ctx),
            )
        }
        ZExtCmp::UltSwapped => {
            // `c < ZeroExt(k, x)`. ZeroExt ∈ [0, 2^(W-k)).
            if c_high != 0 {
                // c >= 2^(W-k) > ZeroExt(x); never c < ZeroExt(x).
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(0, 1));
            }
            record_zext_cmp_collapse();
            Some(RustBV::concrete(c_low, inner_width).ult_into(inner.clone(), ctx))
        }
        ZExtCmp::Ule => {
            // `ZeroExt(k, x) <= c`.
            if c_high != 0 {
                // c >= 2^(W-k) > all ZeroExt values; always true.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(1, 1));
            }
            record_zext_cmp_collapse();
            Some(
                inner
                    .clone()
                    .ule_into(RustBV::concrete(c_low, inner_width), ctx),
            )
        }
        ZExtCmp::UleSwapped => {
            // `c <= ZeroExt(k, x)`.
            if c_high != 0 {
                // c >= 2^(W-k) > ZeroExt(x); never c <= ZeroExt(x).
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(0, 1));
            }
            if c_low == 0 {
                // 0 <= ZeroExt(x) is always true.
                record_zext_cmp_trivial_decide();
                return Some(RustBV::concrete(1, 1));
            }
            record_zext_cmp_collapse();
            Some(RustBV::concrete(c_low, inner_width).ule_into(inner.clone(), ctx))
        }
    }
}

/// Sign-extend a value from `width` bits to i128.
fn sign_extend(value: u128, width: u32) -> i128 {
    if width >= 128 {
        value as i128
    } else {
        let sign_bit = 1u128 << (width - 1);
        if value & sign_bit != 0 {
            (value | !((1u128 << width) - 1)) as i128
        } else {
            value as i128
        }
    }
}

/// Sign-extend a value from one width to another (staying in u128).
fn sign_extend_to(value: u128, from_width: u32, to_width: u32) -> u128 {
    if from_width >= to_width {
        value
    } else {
        let sign_bit = 1u128 << (from_width - 1);
        if value & sign_bit != 0 {
            let extension_mask = ((1u128 << to_width) - 1) & !((1u128 << from_width) - 1);
            value | extension_mask
        } else {
            value
        }
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl fmt::Debug for RustBV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustBV::Concrete { value, width } => {
                write!(f, "Concrete(0x{:x}, {})", value, width)
            }
            RustBV::Symbolic { width, name, .. } => {
                write!(f, "Symbolic({}, {})", name, width)
            }
            RustBV::Constrained { value, width, .. } => {
                write!(f, "Constrained(0x{:x}, {})", value, width)
            }
            RustBV::Expression {
                width,
                op,
                operands,
                ..
            } => {
                write!(
                    f,
                    "Expression({:?}, {}, {} operands)",
                    op,
                    width,
                    operands.len()
                )
            }
        }
    }
}

impl fmt::Display for RustBV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustBV::Concrete { value, width } => {
                write!(f, "<BV{} 0x{:x}>", width, value)
            }
            RustBV::Symbolic { width, name, .. } => {
                write!(f, "<BV{} {}>", width, name)
            }
            RustBV::Constrained { value, width, .. } => {
                write!(f, "<BV{} 0x{:x} (constrained)>", width, value)
            }
            RustBV::Expression { width, op, .. } => {
                write!(f, "<BV{} {:?}>", width, op)
            }
        }
    }
}

impl PartialEq for RustBV {
    fn eq(&self, other: &Self) -> bool {
        match (self.as_u128(), other.as_u128()) {
            (Some(a), Some(b)) => a == b && self.width() == other.width(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concrete_add() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(5, 32);
        let b = RustBV::concrete(3, 32);
        let result = a.add(&b, &ctx);
        assert_eq!(result.as_u64(), Some(8));
    }

    #[test]
    fn test_concrete_sub() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(10, 32);
        let b = RustBV::concrete(3, 32);
        let result = a.sub(&b, &ctx);
        assert_eq!(result.as_u64(), Some(7));
    }

    #[test]
    fn test_concrete_overflow() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8);
        let b = RustBV::concrete(1, 8);
        let result = a.add(&b, &ctx);
        assert_eq!(result.as_u64(), Some(0)); // 8-bit overflow wraps to 0
    }

    #[test]
    fn test_sign_extend() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits
        let result = a.sign_extend(32, &ctx);
        assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
    }

    #[test]
    fn test_zero_extend() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8);
        let result = a.zero_extend(32, &ctx);
        assert_eq!(result.as_u64(), Some(0xFF));
    }

    #[test]
    fn test_truncate() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0x12345678, 32);
        let result = a.truncate(8, &ctx);
        assert_eq!(result.as_u64(), Some(0x78));
    }

    #[test]
    fn test_extract() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xABCD, 16);
        let result = a.extract(11, 4, &ctx);
        assert_eq!(result.width(), 8);
        assert_eq!(result.as_u64(), Some(0xBC));
    }

    #[test]
    fn test_concat() {
        let ctx = SymContext::new_mock();
        let hi = RustBV::concrete(0xAB, 8);
        let lo = RustBV::concrete(0xCD, 8);
        let result = hi.concat(&lo, &ctx);
        assert_eq!(result.width(), 16);
        assert_eq!(result.as_u64(), Some(0xABCD));
    }

    /// Helper: recursive depth of an expression AST. Concrete/symbolic
    /// leaves have depth 0; every Expression node adds one.
    fn ast_depth(bv: &RustBV) -> u32 {
        match bv {
            RustBV::Expression { operands, .. } => {
                1 + operands.iter().map(ast_depth).max().unwrap_or(0)
            }
            _ => 0,
        }
    }

    #[test]
    fn test_concat_balanced_single() {
        let ctx = SymContext::new_mock();
        let parts = [RustBV::symbolic(&ctx, "a", 8)];
        let result = RustBV::concat_balanced(&parts, &ctx);
        assert_eq!(result.width(), 8);
    }

    #[test]
    fn test_concat_balanced_pair() {
        let ctx = SymContext::new_mock();
        let hi = RustBV::concrete(0xAB, 8);
        let lo = RustBV::concrete(0xCD, 8);
        let result = RustBV::concat_balanced(&[hi, lo], &ctx);
        assert_eq!(result.width(), 16);
        assert_eq!(result.as_u64(), Some(0xABCD));
    }

    #[test]
    fn test_concat_balanced_concrete_value() {
        // Build 0xDEADBEEF byte-by-byte (high to low) and check the value.
        let ctx = SymContext::new_mock();
        let parts: Vec<RustBV> = [0xDE, 0xAD, 0xBE, 0xEF]
            .iter()
            .map(|&b| RustBV::concrete(b, 8))
            .collect();
        let result = RustBV::concat_balanced(&parts, &ctx);
        assert_eq!(result.width(), 32);
        assert_eq!(result.as_u64(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_concat_balanced_depth_is_log() {
        // 8 symbolic bytes → linear chain would have depth 7; balanced
        // tree should be 3 (log2(8)).
        let ctx = SymContext::new_mock();
        let parts: Vec<RustBV> = (0..8)
            .map(|i| RustBV::symbolic(&ctx, format!("b{}", i), 8))
            .collect();
        let balanced = RustBV::concat_balanced(&parts, &ctx);
        assert_eq!(balanced.width(), 64);
        assert_eq!(ast_depth(&balanced), 3);

        // Sanity: the left-fold reference is depth 7.
        let mut linear = parts[0].clone();
        for p in &parts[1..] {
            linear = linear.concat(p, &ctx);
        }
        assert_eq!(ast_depth(&linear), 7);
    }

    #[test]
    fn test_concat_balanced_odd_length() {
        // Odd length (5) should still produce ceil(log2(5))=3-deep tree.
        let ctx = SymContext::new_mock();
        let parts: Vec<RustBV> = (0..5)
            .map(|i| RustBV::symbolic(&ctx, format!("o{}", i), 8))
            .collect();
        let balanced = RustBV::concat_balanced(&parts, &ctx);
        assert_eq!(balanced.width(), 40);
        assert!(ast_depth(&balanced) <= 3);
    }

    #[test]
    fn test_concat_balanced_matches_linear_value() {
        // For concrete inputs, both balanced and linear concat must
        // produce the same numeric value.
        let ctx = SymContext::new_mock();
        let bytes = [0x12u128, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let parts: Vec<RustBV> = bytes.iter().map(|&b| RustBV::concrete(b, 8)).collect();
        let balanced = RustBV::concat_balanced(&parts, &ctx);
        let mut linear = parts[0].clone();
        for p in &parts[1..] {
            linear = linear.concat(p, &ctx);
        }
        assert_eq!(balanced.width(), linear.width());
        assert_eq!(balanced.as_u64(), linear.as_u64());
        assert_eq!(balanced.as_u64(), Some(0x123456789ABC));
    }

    #[test]
    fn test_comparison_unsigned() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(5, 32);
        let b = RustBV::concrete(10, 32);

        assert_eq!(a.ult(&b, &ctx).as_u64(), Some(1));
        assert_eq!(a.ule(&b, &ctx).as_u64(), Some(1));
        assert_eq!(a.ugt(&b, &ctx).as_u64(), Some(0));
        assert_eq!(a.uge(&b, &ctx).as_u64(), Some(0));
    }

    #[test]
    fn test_comparison_signed() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0xFF, 8); // -1 signed
        let b = RustBV::concrete(1, 8);

        assert_eq!(a.slt(&b, &ctx).as_u64(), Some(1)); // -1 < 1
        assert_eq!(a.ult(&b, &ctx).as_u64(), Some(0)); // 255 > 1 unsigned
    }

    // =========================================================================
    // Expression simplification tests
    // =========================================================================

    #[test]
    fn test_add_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x + 0 → x (should return symbolic, not expression)
        let r1 = x.add(&zero, &ctx);
        assert!(matches!(r1, RustBV::Symbolic { .. }));
        // 0 + x → x
        let r2 = zero.add(&x, &ctx);
        assert!(matches!(r2, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_sub_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x - 0 → x
        let r = x.sub(&zero, &ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_mul_identity_and_annihilator() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        let one = RustBV::concrete(1, 32);
        // x * 0 → 0
        assert_eq!(x.mul(&zero, &ctx).as_u128(), Some(0));
        // 0 * x → 0
        assert_eq!(zero.mul(&x, &ctx).as_u128(), Some(0));
        // x * 1 → x
        assert!(matches!(x.mul(&one, &ctx), RustBV::Symbolic { .. }));
        // 1 * x → x
        assert!(matches!(one.mul(&x, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_and_identity_and_annihilator() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zero = RustBV::zero(8);
        let ones = RustBV::ones(8);
        // x & 0 → 0
        assert_eq!(x.and(&zero, &ctx).as_u128(), Some(0));
        // x & 0xFF → x
        assert!(matches!(x.and(&ones, &ctx), RustBV::Symbolic { .. }));
        // 0xFF & x → x
        assert!(matches!(ones.and(&x, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_or_identity_and_annihilator() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zero = RustBV::zero(8);
        let ones = RustBV::ones(8);
        // x | 0 → x
        assert!(matches!(x.or(&zero, &ctx), RustBV::Symbolic { .. }));
        // 0 | x → x
        assert!(matches!(zero.or(&x, &ctx), RustBV::Symbolic { .. }));
        // x | 0xFF → 0xFF
        assert_eq!(x.or(&ones, &ctx).as_u128(), Some(0xFF));
    }

    #[test]
    fn test_xor_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x ^ 0 → x
        assert!(matches!(x.xor(&zero, &ctx), RustBV::Symbolic { .. }));
        // 0 ^ x → x
        assert!(matches!(zero.xor(&x, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_not_double_negation() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // not(not(x)) → x
        let r = x.not(&ctx).not(&ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_neg_double_negation() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // neg(neg(x)) → x
        let r = x.neg(&ctx).neg(&ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_reverse_double() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // reverse(reverse(x)) → x
        let r = x.reverse(&ctx).reverse(&ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_shift_by_zero() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // x << 0 → x
        assert!(matches!(x.shl(&zero, &ctx), RustBV::Symbolic { .. }));
        // x >> 0 → x
        assert!(matches!(x.lshr(&zero, &ctx), RustBV::Symbolic { .. }));
        // x >>> 0 → x
        assert!(matches!(x.ashr(&zero, &ctx), RustBV::Symbolic { .. }));
    }

    #[test]
    fn test_shift_zero_value() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let zero = RustBV::zero(32);
        // 0 << x → 0
        assert_eq!(zero.shl(&x, &ctx).as_u128(), Some(0));
        // 0 >> x → 0
        assert_eq!(zero.lshr(&x, &ctx).as_u128(), Some(0));
    }

    #[test]
    fn test_shl_concrete_amount_rewrites_to_concat() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let four = RustBV::concrete(4, 32);
        // sym << 4 → Concat(Extract(27, 0, sym), 0^4)
        let r = x.shl(&four, &ctx);
        assert_eq!(r.width(), 32);
        match r {
            RustBV::Expression {
                op: BVOp::Concat,
                operands,
                ..
            } => {
                assert_eq!(operands.len(), 2);
                assert_eq!(operands[1].as_u128(), Some(0)); // low bits are zero
                assert_eq!(operands[1].width(), 4);
                assert_eq!(operands[0].width(), 28); // top bits extracted from x
            }
            other => panic!("expected Concat, got {:?}", other),
        }
        // Behavior preserved when LHS happens to be concrete (still constant-folds).
        let v = RustBV::concrete(0x1234, 32);
        assert_eq!(v.shl(&four, &ctx).as_u128(), Some(0x12340));
    }

    #[test]
    fn test_shl_concrete_amount_at_or_above_width_yields_zero() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // sym << 32 → 0
        let r = x.shl(&RustBV::concrete(32, 32), &ctx);
        assert_eq!(r.as_u128(), Some(0));
        // sym << 999 (oversized amount) → 0
        let r = x.shl(&RustBV::concrete(999, 32), &ctx);
        assert_eq!(r.as_u128(), Some(0));
    }

    #[test]
    fn test_lshr_concrete_amount_rewrites_to_concat() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let eight = RustBV::concrete(8, 32);
        // sym >> 8 → Concat(0^8, Extract(31, 8, sym))
        let r = x.lshr(&eight, &ctx);
        assert_eq!(r.width(), 32);
        match r {
            RustBV::Expression {
                op: BVOp::Concat,
                operands,
                ..
            } => {
                assert_eq!(operands.len(), 2);
                assert_eq!(operands[0].as_u128(), Some(0));
                assert_eq!(operands[0].width(), 8);
                assert_eq!(operands[1].width(), 24);
            }
            other => panic!("expected Concat, got {:?}", other),
        }
        let r = x.lshr(&RustBV::concrete(32, 32), &ctx);
        assert_eq!(r.as_u128(), Some(0));
    }

    #[test]
    fn test_ashr_concrete_amount_rewrites_to_sign_extend() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let four = RustBV::concrete(4, 32);
        // sym >>> 4 → SignExt(4, Extract(31, 4, sym))  [extends 28-bit slice by 4 bits]
        let r = x.ashr(&four, &ctx);
        assert_eq!(r.width(), 32);
        match r {
            RustBV::Expression {
                op: BVOp::SignExt(4),
                operands,
                ..
            } => {
                assert_eq!(operands[0].width(), 28);
            }
            other => panic!("expected SignExt(4), got {:?}", other),
        }
        // Beyond width: SignExt of MSB (1-bit slice extended by 31 bits).
        let r = x.ashr(&RustBV::concrete(64, 32), &ctx);
        assert_eq!(r.width(), 32);
        match r {
            RustBV::Expression {
                op: BVOp::SignExt(31),
                operands,
                ..
            } => {
                assert_eq!(operands[0].width(), 1);
            }
            other => panic!("expected SignExt(31), got {:?}", other),
        }
    }

    #[test]
    fn test_mul_by_power_of_two_rewrites_to_shl_then_concat() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let eight = RustBV::concrete(8, 32);
        // sym * 8 → sym << 3 → Concat(Extract(28, 0, sym), 0^3)
        let r = x.mul(&eight, &ctx);
        assert_eq!(r.width(), 32);
        match r {
            RustBV::Expression {
                op: BVOp::Concat,
                operands,
                ..
            } => {
                assert_eq!(operands[1].as_u128(), Some(0));
                assert_eq!(operands[1].width(), 3);
                assert_eq!(operands[0].width(), 29);
            }
            other => panic!("expected Concat (via Shl), got {:?}", other),
        }
        // Commutative case: 8 * sym → same shape.
        let r = eight.mul(&x, &ctx);
        match r {
            RustBV::Expression {
                op: BVOp::Concat,
                operands,
                ..
            } => {
                assert_eq!(operands[1].as_u128(), Some(0));
                assert_eq!(operands[1].width(), 3);
            }
            other => panic!("expected Concat (commutative), got {:?}", other),
        }
    }

    #[test]
    fn test_mul_by_non_power_of_two_stays_as_mul() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // sym * 3 → still Mul (no rewrite for non-pow2 constants)
        let r = x.mul(&RustBV::concrete(3, 32), &ctx);
        match r {
            RustBV::Expression { op: BVOp::Mul, .. } => {}
            other => panic!("expected Mul, got {:?}", other),
        }
    }

    #[test]
    fn test_sign_extend_identity() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        // sign_extend to same width → x
        let r = x.sign_extend(32, &ctx);
        assert!(matches!(r, RustBV::Symbolic { .. }));
        assert_eq!(r.width(), 32);
    }

    #[test]
    fn test_extract_zero_ext() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let ext = x.zero_extend(32, &ctx); // 8-bit → 32-bit
        // Extract low 8 bits → original x
        let lo = ext.extract(7, 0, &ctx);
        assert!(matches!(lo, RustBV::Symbolic { .. }));
        assert_eq!(lo.width(), 8);
        // Extract high 8 bits → 0
        let hi = ext.extract(31, 24, &ctx);
        assert_eq!(hi.as_u128(), Some(0));
    }

    #[test]
    fn test_extract_sign_ext_low_bits() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let ext = x.sign_extend(32, &ctx);
        // Extract low 8 bits → original x
        let lo = ext.extract(7, 0, &ctx);
        assert!(matches!(lo, RustBV::Symbolic { .. }));
        assert_eq!(lo.width(), 8);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_32bit_leaf() {
        // Reverse(x) over a symbolic leaf should produce byte-reversed value
        // through the Z3 emission path.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let rev = x.reverse(&ctx); // Expression { Reverse, [x] }
        // Pin x = 0x11223344; expect Reverse(x) = 0x44332211
        let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&rev), Some(0x44332211));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_64bit_leaf() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 64);
        let rev = x.reverse(&ctx);
        let pinned = x.eq(&RustBV::concrete(0x0123456789ABCDEF, 64), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&rev), Some(0xEFCDAB8967452301));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_16bit_leaf() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 16);
        let rev = x.reverse(&ctx);
        let pinned = x.eq(&RustBV::concrete(0xAABB, 16), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&rev), Some(0xBBAA));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_reverse_z3_emission_concat_of_bytes() {
        // Reverse(Concat(b0, b1, ..., b7)) with independent byte BVSes.
        // Memory loads in angr typically produce this shape; the result should
        // be Concat(b7, b6, ..., b0) — the byte-reversed value.
        let ctx = SymContext::new_mock();
        let bytes: Vec<RustBV> = (0..8u32)
            .map(|i| RustBV::symbolic(&ctx, format!("b{}", i), 8))
            .collect();
        // Build claripy-style Concat(b0, b1, ..., b7) with b0 as high.
        let mut concat = bytes[0].clone();
        for b in &bytes[1..] {
            concat = concat.concat(b, &ctx);
        }
        let rev = concat.reverse(&ctx);
        // Pin each byte to a distinct value and verify byte-reversed result.
        let vals: [u64; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        for (b, v) in bytes.iter().zip(vals.iter()) {
            let pin = b.eq(&RustBV::concrete(*v as u128, 8), &ctx);
            ctx.add_constraint(pin.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        }
        // concat = 0x1122334455667788; reverse → 0x8877665544332211
        assert_eq!(ctx.eval(&rev), Some(0x8877665544332211));
    }

    // --- Pre-Z3 Extract rewrite pass (angr-p8cz) ---

    /// Helper: build a raw Extract Expression node without going through
    /// `extract_into`. This simulates Extract nodes that bypass the
    /// construction-time rewrite (e.g., via `truncate_into` or
    /// `extract_no_ctx`), so we can verify the Z3-emission pass picks them up.
    #[cfg(feature = "vex-engine-z3")]
    fn raw_extract_node(inner: RustBV, high: u32, low: u32) -> RustBV {
        let result_width = high - low + 1;
        RustBV::Expression {
            id: RustBV::EXPRESSION_ID,
            width: result_width,
            op: BVOp::Extract(high, low),
            operands: std::sync::Arc::<[RustBV]>::from([inner]),
        }
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_reverse_byte_aligned() {
        // Extract a single byte from Reverse(x). With the rewrite pass the
        // Reverse should be eliminated entirely from the Z3 AST.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let rev = x.reverse(&ctx);
        // Bypass extract_into via the raw-node helper: pretend this Extract was
        // built by truncate_into or extract_no_ctx after the Reverse existed.
        // Reverse(x) byte 0 ([7:0]) is byte 3 ([31:24]) of x.
        let lo_byte = raw_extract_node(rev, 7, 0);
        let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&lo_byte), Some(0x11));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_reverse_crossing_byte() {
        // Two-byte extract across a byte boundary on Reverse — still
        // byte-aligned (high % 8 == 7, low % 8 == 0), should be rewritten.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let rev = x.reverse(&ctx);
        // Reverse(x)[15:0] = bytes 0,1 of reverse = bytes 3,2 of x = top half reversed.
        let lower16 = raw_extract_node(rev, 15, 0);
        let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        // Reverse(0x11223344) = 0x44332211; low 16 bits = 0x2211
        assert_eq!(ctx.eval(&lower16), Some(0x2211));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_concat_within_low() {
        // Extract entirely within the low (right) part of a Concat — should
        // delegate to that operand alone.
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "a", 16);
        let b = RustBV::symbolic(&ctx, "b", 16);
        let cat = a.concat(&b, &ctx); // a:high, b:low
        // Extract [15:0] of cat = entirely within b.
        let lo = raw_extract_node(cat, 15, 0);
        ctx.add_constraint(
            a.eq(&RustBV::concrete(0xAAAA, 16), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        ctx.add_constraint(
            b.eq(&RustBV::concrete(0xBBBB, 16), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        assert_eq!(ctx.eval(&lo), Some(0xBBBB));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_concat_within_high() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "a", 16);
        let b = RustBV::symbolic(&ctx, "b", 16);
        let cat = a.concat(&b, &ctx);
        // Extract [31:16] of cat = entirely within a.
        let hi = raw_extract_node(cat, 31, 16);
        ctx.add_constraint(
            a.eq(&RustBV::concrete(0xAAAA, 16), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        ctx.add_constraint(
            b.eq(&RustBV::concrete(0xBBBB, 16), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        assert_eq!(ctx.eval(&hi), Some(0xAAAA));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_concat_crossing() {
        // Crosses the a/b boundary in the middle — distribute Extract to both.
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "a", 16);
        let b = RustBV::symbolic(&ctx, "b", 16);
        let cat = a.concat(&b, &ctx);
        // Extract [23:8] = high 8 bits of b ([15:8]) concat with low 8 bits of a ([7:0]).
        let mid = raw_extract_node(cat, 23, 8);
        ctx.add_constraint(
            a.eq(&RustBV::concrete(0x1234, 16), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        ctx.add_constraint(
            b.eq(&RustBV::concrete(0x5678, 16), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        // cat = 0x12345678; extract [23:8] = 0x3456
        assert_eq!(ctx.eval(&mid), Some(0x3456));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_extract_fused() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 64);
        let mid = x.extract(47, 16, &ctx); // width 32, = bits 16..=47 of x
        // Build outer Extract WITHOUT going through extract_into.
        let outer = raw_extract_node(mid, 23, 8); // mid[23:8] = bits [39:24] of x
        ctx.add_constraint(
            x.eq(&RustBV::concrete(0x0011_2233_4455_6677, 64), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        // x bytes (LSB→MSB): 0x77 0x66 0x55 0x44 0x33 0x22 0x11 0x00.
        // bits [39:24] of x = byte indices 3..=4 = 0x33:0x44 (high:low) = 0x3344.
        assert_eq!(ctx.eval(&outer), Some(0x3344));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_zero_ext_low_bits() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(32, &ctx); // width 32
        // Extract [7:0] entirely within original x.
        let lo = raw_extract_node(zx, 7, 0);
        ctx.add_constraint(
            x.eq(&RustBV::concrete(0xAB, 8), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        assert_eq!(ctx.eval(&lo), Some(0xAB));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_zero_ext_extended_bits() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(32, &ctx);
        // Extract [31:24] is entirely in the zero-extended region.
        let top = raw_extract_node(zx, 31, 24);
        ctx.add_constraint(
            x.eq(&RustBV::concrete(0xFF, 8), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        assert_eq!(ctx.eval(&top), Some(0));
    }

    /// Regression for the latent bug in extract_into's Rule 3: multi-byte
    /// byte-aligned Extract of Reverse(x) used to drop the byte shuffle
    /// (returned plain Extract from x), giving the wrong byte order in the
    /// constraint tree. Now it returns Reverse(Extract(...)), preserving
    /// semantics through every consumer (Z3 round-trip, downstream rewrites).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_extract_over_reverse_multibyte_construction() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let rev = x.reverse(&ctx);
        // Use the normal extract (which now goes through the fixed Rule 3).
        let lower16 = rev.extract(15, 0, &ctx);
        let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        // Reverse(0x11223344) = 0x44332211; low 16 = 0x2211.
        assert_eq!(ctx.eval(&lower16), Some(0x2211));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_extract_over_reverse_single_byte_construction() {
        // Single-byte case: rule reduces to plain Extract from x.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let rev = x.reverse(&ctx);
        // Byte 0 of reverse = byte 3 of x.
        let b0 = rev.extract(7, 0, &ctx);
        ctx.add_constraint(
            x.eq(&RustBV::concrete(0x11223344, 32), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        assert_eq!(ctx.eval(&b0), Some(0x11));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_pre_z3_extract_over_sign_ext_low_bits() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let sx = x.sign_extend(32, &ctx);
        let lo = raw_extract_node(sx, 7, 0);
        ctx.add_constraint(
            x.eq(&RustBV::concrete(0x80, 8), &ctx)
                .to_z3_ast()
                .eq(z3::ast::BV::from_u64(1, 1)),
        );
        assert_eq!(ctx.eval(&lo), Some(0x80));
    }

    // =========================================================================
    // angr-g7nq: Cmp(ZeroExt(k, x), BVV) trivial-constraint fast path
    // =========================================================================

    #[test]
    fn test_zext_eq_high_bits_nonzero_is_false() {
        // ZeroExt(8, x:8) == 0x100 — high byte nonzero → folds to concrete 0.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let c = RustBV::concrete(0x100, 16);
        let r = zx.eq(&c, &ctx);
        assert_eq!(r.as_u64(), Some(0));
        // Commuted form folds the same way.
        let r_rev = c.eq(&zx, &ctx);
        assert_eq!(r_rev.as_u64(), Some(0));
    }

    #[test]
    fn test_zext_ne_high_bits_nonzero_is_true() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let c = RustBV::concrete(0x100, 16);
        let r = zx.ne(&c, &ctx);
        assert_eq!(r.as_u64(), Some(1));
    }

    #[test]
    fn test_zext_eq_high_bits_zero_collapses() {
        // ZeroExt(8, x:8) == 0x42 — high byte zero → narrows to Eq(x, 0x42),
        // which is still symbolic but should be a width-8 Expression not the
        // width-16 form, evidenced by the operand width.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = zx.eq(&RustBV::concrete(0x42, 16), &ctx);
        // Result is a width-1 Eq expression over width-8 operands.
        match &r {
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                assert_eq!(*op, BVOp::Eq);
                assert_eq!(*width, 1);
                assert_eq!(operands.len(), 2);
                assert_eq!(operands[0].width(), 8);
                assert_eq!(operands[1].width(), 8);
                assert_eq!(operands[1].as_u64(), Some(0x42));
            }
            _ => panic!("expected narrowed Eq expression, got {:?}", r),
        }
    }

    #[test]
    fn test_zext_eq_collapse_solver_consistency() {
        // After narrowing, asserting Eq(zext(x), 0x42) must still pin x to 0x42.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let eq = zx.eq(&RustBV::concrete(0x42, 16), &ctx);
        ctx.add_constraint(eq.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        assert_eq!(ctx.eval(&x), Some(0x42));
        // Also verify the wider zext expression evaluates to the constant.
        assert_eq!(ctx.eval(&zx), Some(0x42));
    }

    #[test]
    fn test_zext_ult_high_bits_nonzero_is_true() {
        // ZeroExt(8, x:8) < 0x200 — all zext values < 256, so always < 0x200.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = zx.ult(&RustBV::concrete(0x200, 16), &ctx);
        assert_eq!(r.as_u64(), Some(1));
    }

    #[test]
    fn test_zext_ult_const_zero_is_false() {
        // ZeroExt(8, x:8) < 0 — never true.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = zx.ult(&RustBV::concrete(0, 16), &ctx);
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_zext_ult_swapped_high_bits_nonzero_is_false() {
        // 0x200 < ZeroExt(8, x:8) — never true (RHS < 256).
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = RustBV::concrete(0x200, 16).ult(&zx, &ctx);
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_zext_ule_high_bits_nonzero_is_true() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = zx.ule(&RustBV::concrete(0x200, 16), &ctx);
        assert_eq!(r.as_u64(), Some(1));
    }

    #[test]
    fn test_zext_ule_swapped_const_zero_is_true() {
        // 0 <= ZeroExt(x) — always true.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = RustBV::concrete(0, 16).ule(&zx, &ctx);
        assert_eq!(r.as_u64(), Some(1));
    }

    #[test]
    fn test_zext_ugt_high_bits_nonzero_is_false() {
        // ZeroExt(8, x:8) > 0x200 — never true (LHS < 256 <= 0x200).
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = zx.ugt(&RustBV::concrete(0x200, 16), &ctx);
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_zext_uge_high_bits_nonzero_is_false() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx = x.zero_extend(16, &ctx);
        let r = zx.uge(&RustBV::concrete(0x200, 16), &ctx);
        assert_eq!(r.as_u64(), Some(0));
    }

    #[test]
    fn test_zext_cmp_no_fold_when_both_symbolic() {
        // Cmp(ZeroExt(x), ZeroExt(y)) — no const side, no fold.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let y = RustBV::symbolic(&ctx, "y", 8);
        let zx = x.zero_extend(16, &ctx);
        let zy = y.zero_extend(16, &ctx);
        let r = zx.eq(&zy, &ctx);
        // Should be a regular Eq expression at width 16 — no fold.
        match &r {
            RustBV::Expression { op, operands, .. } => {
                assert_eq!(*op, BVOp::Eq);
                assert_eq!(operands[0].width(), 16);
                assert_eq!(operands[1].width(), 16);
            }
            _ => panic!("expected Eq expression, got {:?}", r),
        }
    }

    #[test]
    fn test_zext_cmp_no_fold_for_signed_ops() {
        // SignExt(k, x) is not handled — the comparison should pass through.
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let sx = x.sign_extend(16, &ctx); // SignExt, not ZeroExt
        let r = sx.eq(&RustBV::concrete(0x100, 16), &ctx);
        // Should be Eq expression (no fold).
        match &r {
            RustBV::Expression { op, operands, .. } => {
                assert_eq!(*op, BVOp::Eq);
                assert_eq!(operands[0].width(), 16);
            }
            _ => panic!("expected Eq expression, got {:?}", r),
        }
    }

    #[test]
    fn test_zext_cmp_chain_collapse_to_narrowest() {
        // ZeroExt(16, ZeroExt(8, x:8)) == 0xFF — both extends collapse cleanly.
        // The inner zero_extend collapses via constant fold; the outer is what
        // we're testing. After the first call, the inner zx is a ZeroExt(8, x);
        // wrapping in zero_extend(32, ...) produces ZeroExt(16, ZeroExt(8, x))
        // which by the operand structure is treated as ZeroExt(16, <inner>),
        // where <inner> has width 16. The const 0xFF has zero high 16 bits, so
        // we narrow to Eq(<inner-as-zext>, 0xFF:16); that then narrows again
        // recursively. Final: Eq(x, 0xFF:8).
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let zx16 = x.zero_extend(16, &ctx);
        let zx32 = zx16.zero_extend(32, &ctx);
        let r = zx32.eq(&RustBV::concrete(0xFF, 32), &ctx);
        match &r {
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                assert_eq!(*op, BVOp::Eq);
                assert_eq!(*width, 1);
                // Should have narrowed to width-8 operands.
                assert_eq!(operands[0].width(), 8);
                assert_eq!(operands[1].as_u64(), Some(0xFF));
            }
            _ => panic!("expected narrowed Eq, got {:?}", r),
        }
    }

    /// Regression guard for the angr-behq finding (2026-05-21):
    /// Z3's AST hash-cons already de-dupes structurally-equal RustBV trees
    /// at to_z3_ast time. Two structurally-equal Expression trees produce
    /// the SAME Z3_ast pointer — so RustBV-level construction hash-cons
    /// would NOT reduce Z3 AST node count for the to_z3_ast() output.
    ///
    /// If this test ever fails, the assumption that drove closing
    /// angr-behq is broken and that bead should be re-opened.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn z3_already_dedupes_structurally_equal_rustbv_trees() {
        use z3::ast::Ast as Z3AstTrait;
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic_with_id(2001, "x_behq", 32);
        let y = RustBV::symbolic_with_id(2002, "y_behq", 32);

        // Two structurally-identical add(x, 5) trees built independently.
        let p1 = x.clone().add_into(RustBV::concrete(5, 32), &ctx);
        let p2 = x.clone().add_into(RustBV::concrete(5, 32), &ctx);
        let p1_ptr = p1.to_z3_ast().get_z3_ast().as_ptr();
        let p2_ptr = p2.to_z3_ast().get_z3_ast().as_ptr();
        assert_eq!(p1_ptr, p2_ptr, "Z3 should canonicalize add(x,5)");

        // Two structurally-identical mul(add(x,5), y) trees, depth 2.
        let q1 = p1.mul(&y, &ctx);
        let q2 = p2.mul(&y, &ctx);
        let q1_ptr = q1.to_z3_ast().get_z3_ast().as_ptr();
        let q2_ptr = q2.to_z3_ast().get_z3_ast().as_ptr();
        assert_eq!(q1_ptr, q2_ptr, "Z3 should canonicalize mul(add(x,5),y)");
    }

    /// Regression guard for angr-kkpr (2026-05-21):
    /// RustBV constructors for commutative ops (add/mul/and/or/xor/eq/ne)
    /// canonicalize operand order via `canonical_sort_key`. After this fix,
    /// `add(x, y)` and `add(y, x)` produce the SAME RustBV (and therefore
    /// the same Z3 AST). Z3 itself does NOT normalize commutative arg order
    /// at mk_bv* time — preprocessing tactics do, but only on assertions —
    /// so the canonicalization happens on the Rust side at construction.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn commutative_ops_canonicalize_operand_order() {
        use z3::ast::Ast as Z3AstTrait;
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic_with_id(3001, "x_kkpr", 32);
        let y = RustBV::symbolic_with_id(3002, "y_kkpr", 32);
        let c = RustBV::concrete(7, 32);

        // add(x, y) and add(y, x) → same RustBV → same Z3 AST.
        let r1 = x.clone().add_into(y.clone(), &ctx);
        let r2 = y.clone().add_into(x.clone(), &ctx);
        assert_eq!(
            r1.to_z3_ast().get_z3_ast().as_ptr(),
            r2.to_z3_ast().get_z3_ast().as_ptr(),
            "add(x,y) and add(y,x) should canonicalize to the same Z3 AST"
        );

        // add(x, c) and add(c, x) → same RustBV → same Z3 AST.
        // (concrete should sort to the right, so both become add(x, c).)
        let r3 = x.clone().add_into(c.clone(), &ctx);
        let r4 = c.clone().add_into(x.clone(), &ctx);
        assert_eq!(
            r3.to_z3_ast().get_z3_ast().as_ptr(),
            r4.to_z3_ast().get_z3_ast().as_ptr(),
            "add(x,c) and add(c,x) should canonicalize to the same Z3 AST"
        );
        // Verify concrete is in operand[1] after canonicalization.
        match &r3 {
            RustBV::Expression { operands, .. } => {
                assert!(
                    matches!(operands[1], RustBV::Concrete { .. }),
                    "concrete should sort to the right of symbolic"
                );
            }
            _ => panic!("expected Expression for add(x, c)"),
        }

        // Same property for mul / and / or / xor / eq / ne.
        type BinOp = fn(RustBV, RustBV, &SymContext) -> RustBV;
        let cases: &[(&str, BinOp)] = &[
            ("mul", |a, b, c| a.mul_into(b, c)),
            ("and", |a, b, c| a.and_into(b, c)),
            ("or", |a, b, c| a.or_into(b, c)),
            ("xor", |a, b, c| a.xor_into(b, c)),
            ("eq", |a, b, c| a.eq_into(b, c)),
            ("ne", |a, b, c| a.ne_into(b, c)),
        ];
        for (name, build) in cases {
            let lhs = build(x.clone(), y.clone(), &ctx);
            let rhs = build(y.clone(), x.clone(), &ctx);
            assert_eq!(
                lhs.to_z3_ast().get_z3_ast().as_ptr(),
                rhs.to_z3_ast().get_z3_ast().as_ptr(),
                "{}(x,y) and {}(y,x) should canonicalize to the same Z3 AST",
                name,
                name
            );
        }
    }

    // -----------------------------------------------------------------
    // Serde round-trip tests for the op-tree primitives (angr-x04s.1.1).
    // Each variant is constructed inside a fresh SymContext, serialized
    // to JSON, deserialized back, and structurally compared to the
    // original. The Symbolic variant's Z3 AST is reconstructed lazily
    // — we verify that round-tripped Symbolic values still produce a
    // valid Z3 AST via to_z3_ast().
    // -----------------------------------------------------------------

    #[test]
    fn serde_roundtrip_concrete() {
        let bv = RustBV::concrete(0xdead_beef, 32);
        let json = serde_json::to_string(&bv).expect("serialize");
        let back: RustBV = serde_json::from_str(&json).expect("deserialize");
        match (&bv, &back) {
            (
                RustBV::Concrete {
                    value: v1,
                    width: w1,
                },
                RustBV::Concrete {
                    value: v2,
                    width: w2,
                },
            ) => {
                assert_eq!(v1, v2);
                assert_eq!(w1, w2);
            }
            _ => panic!("variant changed across round-trip"),
        }
    }

    #[test]
    fn serde_roundtrip_constrained() {
        let bv = RustBV::Constrained {
            id: 42,
            value: 7,
            width: 64,
        };
        let json = serde_json::to_string(&bv).expect("serialize");
        let back: RustBV = serde_json::from_str(&json).expect("deserialize");
        match back {
            RustBV::Constrained { id, value, width } => {
                assert_eq!(id, 42);
                assert_eq!(value, 7);
                assert_eq!(width, 64);
            }
            _ => panic!("variant changed across round-trip"),
        }
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn serde_roundtrip_symbolic() {
        use z3::ast::Ast as Z3AstTrait;
        let ctx = SymContext::new_mock();
        let bv = RustBV::symbolic(&ctx, "x_serde", 32);
        let (orig_id, orig_width, orig_name) = match &bv {
            RustBV::Symbolic {
                id, width, name, ..
            } => (*id, *width, name.to_string()),
            _ => panic!("expected Symbolic"),
        };
        let json = serde_json::to_string(&bv).expect("serialize");
        // Deserialize under the same context (thread-local is still active).
        let back: RustBV = serde_json::from_str(&json).expect("deserialize");
        match &back {
            RustBV::Symbolic {
                id, width, name, ..
            } => {
                assert_eq!(*id, orig_id);
                assert_eq!(*width, orig_width);
                assert_eq!(name.as_ref(), orig_name.as_str());
            }
            _ => panic!("variant changed across round-trip"),
        }
        // The lazily-rebuilt AST must be usable: it should produce a Z3 AST
        // pointer (sanity check that BV::new_const succeeded under the
        // active thread-local context).
        let _ptr = back.to_z3_ast().get_z3_ast().as_ptr();
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn serde_roundtrip_expression_tree() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x_expr_serde", 32);
        let c = RustBV::concrete(7, 32);
        let expr = x.add(&c, &ctx);
        // expr is Expression(Add, [Symbolic(x), Concrete(7)]) after
        // commutative canonicalization (concrete sorts to the right).
        let json = serde_json::to_string(&expr).expect("serialize");
        let back: RustBV = serde_json::from_str(&json).expect("deserialize");
        match &back {
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                assert_eq!(*op, BVOp::Add);
                assert_eq!(*width, 32);
                assert_eq!(operands.len(), 2);
                assert!(matches!(operands[0], RustBV::Symbolic { .. }));
                assert!(matches!(
                    operands[1],
                    RustBV::Concrete {
                        value: 7,
                        width: 32
                    }
                ));
            }
            _ => panic!("variant changed across round-trip"),
        }
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn serde_roundtrip_float_op() {
        // BVOp::Float carries kind + prec; verify it survives JSON
        // round-trip without losing fields (covers FloatOpKind +
        // FloatPrec derive correctness).
        let op = BVOp::Float {
            kind: FloatOpKind::ConvertItoF {
                src_bits: 32,
                signed: true,
            },
            prec: FloatPrec::F64,
        };
        let json = serde_json::to_string(&op).expect("serialize");
        let back: BVOp = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(op, back);
    }
}
